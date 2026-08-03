// Package plugin bundles the SDK's worker-side registration into a single
// [go.temporal.io/sdk/worker.Plugin], so a worker is wired with one entry in
// worker.Options rather than a Register call per piece.
//
// A [Config] declares the pieces — the model activity, any MCP activities, the
// sub-agent workflow, and the conversation workflow — and the plugin registers
// them at worker start:
//
//	p, err := plugin.New(plugin.Config{
//		Providers: []model.Provider{provider},
//		Agents:    registry,
//	})
//	if err != nil {
//		log.Fatal(err)
//	}
//	w := worker.New(c, taskQueue, worker.Options{Plugins: []worker.Plugin{p}})
//
// The plugin registers its items at worker start, so it composes with other
// entries in worker.Options.Plugins (e.g. an interceptor plugin) and leaves the
// worker free for the consumer's own workflows and activities. Registering the
// SDK activities directly is still supported and is what tests use, since the
// Temporal test environments do not run plugins.
package plugin

import (
	"context"
	"fmt"

	"go.temporal.io/sdk/worker"

	"github.com/wleev/temporal-agent-sdk/agent"
	"github.com/wleev/temporal-agent-sdk/conversation"
	"github.com/wleev/temporal-agent-sdk/mcp"
	"github.com/wleev/temporal-agent-sdk/model"
)

// Config declares what a worker should serve. Only Providers is required; the
// rest are wired if set. Build the MCP and agent values with their own
// constructors (mcp.NewActivities, agent.NewRegistry, ...) so factory and agent
// registration errors surface before the worker starts.
type Config struct {
	// Providers back the model activity; at least one is required.
	Providers []model.Provider

	// StreamSink, if set, enables streaming on the model activity.
	StreamSink model.SinkFactory

	// BlobResolver, if set, resolves URI media blocks to inline bytes.
	BlobResolver model.BlobResolver

	// MCP, if set, registers the stateless MCP tool activities.
	MCP *mcp.Activities

	// StatefulMCP, if set, registers the stateful MCP session activity.
	StatefulMCP *mcp.StatefulActivities

	// Agents, if set, registers the sub-agent child workflow.
	Agents *agent.Registry

	// Conversation, if set, registers the conversation workflow and its
	// compaction activity.
	Conversation *conversation.Conversation
}

// plugin implements worker.Plugin. It embeds worker.PluginBase, whose defaults
// cover every method except the two this plugin overrides.
type plugin struct {
	worker.PluginBase

	model        *model.Activities
	mcp          *mcp.Activities
	statefulMCP  *mcp.StatefulActivities
	agents       *agent.Registry
	conversation *conversation.Conversation
}

// New builds the plugin from cfg. It constructs the model activity up front, so
// a missing or duplicate provider is reported here rather than at worker start.
func New(cfg Config) (worker.Plugin, error) {
	acts, err := model.NewActivities(cfg.Providers...)
	if err != nil {
		return nil, fmt.Errorf("plugin: building model activities: %w", err)
	}
	if cfg.StreamSink != nil {
		acts.SetStreamSink(cfg.StreamSink)
	}
	if cfg.BlobResolver != nil {
		acts.SetBlobResolver(cfg.BlobResolver)
	}
	return &plugin{
		model:        acts,
		mcp:          cfg.MCP,
		statefulMCP:  cfg.StatefulMCP,
		agents:       cfg.Agents,
		conversation: cfg.Conversation,
	}, nil
}

// Name identifies the plugin in worker plugin telemetry.
func (p *plugin) Name() string { return "temporal-agent-sdk" }

// StartWorker registers the configured items on the worker, then hands off to
// the next plugin. The registry it receives satisfies each package's own
// registration interface, so the ordinary Register helpers do the work.
func (p *plugin) StartWorker(
	ctx context.Context,
	options worker.PluginStartWorkerOptions,
	next func(context.Context, worker.PluginStartWorkerOptions) error,
) error {
	reg := options.WorkerRegistry
	p.model.Register(reg)
	if p.mcp != nil {
		p.mcp.RegisterWith(reg)
	}
	if p.statefulMCP != nil {
		p.statefulMCP.RegisterWith(reg)
	}
	if p.agents != nil {
		agent.RegisterWorkflows(reg, p.agents)
	}
	if p.conversation != nil {
		p.conversation.Register(reg)
	}
	return next(ctx, options)
}
