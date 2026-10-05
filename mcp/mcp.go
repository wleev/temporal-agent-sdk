// Package mcp exposes Model Context Protocol server tools to an agent.
//
// [Client] speaks the MCP SDK's own types, so any server's tools and results
// reach the agent exactly as advertised, including fields this SDK has no
// opinion about.
//
// Operations are stateless: each is its own activity that connects, does one
// thing, and disconnects. No MCP session spans a workflow task, so a run may
// park for days, migrate to another worker, or replay from history without a
// live session to lose. Servers with expensive startup or genuine per-session
// state instead want [OpenStatefulSession], which pins a session to a worker.
//
// Register a factory per server, then name the server on an agent:
//
//	acts, err := mcp.NewActivities(mcp.WithServer("filesystem",
//		mcpsdk.CommandFactory("npx", "-y", "@modelcontextprotocol/server-filesystem", "/data")))
//	// ... handle err ...
//	acts.RegisterWith(w)
//
//	a, err := agent.NewAgent("assistant", "gpt-5.2", agent.WithMCPServers("filesystem"))
//
// The mcp/mcpsdk subpackage adapts the official SDK's client. Implement [Client]
// directly to use a different one.
package mcp

import (
	"context"
	"encoding/json"
	"fmt"
	"sort"

	"go.temporal.io/sdk/activity"
	"go.temporal.io/sdk/temporal"

	"github.com/wleev/temporal-agent-sdk/internal/heartbeat"
	"github.com/wleev/temporal-agent-sdk/model"
)

// Activity names registered by [Activities.RegisterWith].
const (
	ListToolsActivity = "agentsdk_mcp_list_tools"
	CallToolActivity  = "agentsdk_mcp_call_tool"
)

// ErrorTypeUnknownServer marks a reference to an unregistered MCP server. It is
// non-retryable: the factory set is fixed at worker startup.
const ErrorTypeUnknownServer = "AgentSDKUnknownMCPServer"

// Client is a connected MCP server.
//
// Implementations are used inside activities and may block on I/O. A Client is
// used for exactly one operation and then closed.
type Client interface {
	// ListTools returns the server's advertised tools.
	ListTools(ctx context.Context) ([]*model.Tool, error)

	// CallTool invokes a tool.
	//
	// A returned error means the call itself failed — the transport broke, the
	// server was unreachable — and is worth retrying. A result with IsError set
	// means the tool ran and reported a failure, which is not.
	CallTool(ctx context.Context, name string, args json.RawMessage) (*model.CallToolResult, error)

	// Close releases the connection. It is always called, including on error.
	Close() error
}

// Factory connects to an MCP server. It is called once per activity attempt, so
// it must be safe to call repeatedly and concurrently.
type Factory func(ctx context.Context) (Client, error)

// Activities is the activity-side half of the MCP integration.
type Activities struct {
	servers

	// pulse is the base configuration of each call's heartbeats.
	pulse heartbeat.Config
}

// Option configures an [Activities] or a [StatefulActivities].
type Option func(*servers) error

// WithServer adds a server factory under name, the name an agent lists in
// MCPServers. The name must not be empty or already added, and f must not be
// nil.
func WithServer(name string, f Factory) Option {
	return func(s *servers) error {
		if name == "" {
			return fmt.Errorf("mcp: server name must not be empty")
		}
		if f == nil {
			return fmt.Errorf("mcp: factory for %q must not be nil", name)
		}
		if _, dup := s.factories[name]; dup {
			return fmt.Errorf("mcp: server %q is already registered", name)
		}
		s.factories[name] = f
		s.names = append(s.names, name)
		return nil
	}
}

// servers is a set of server factories keyed by name.
type servers struct {
	factories map[string]Factory
	// names holds the keys of factories, sorted.
	names []string
}

// newServers builds the server set opts describe.
func newServers(opts []Option) (servers, error) {
	s := servers{factories: make(map[string]Factory)}
	for _, opt := range opts {
		if err := opt(&s); err != nil {
			return servers{}, err
		}
	}
	sort.Strings(s.names)
	return s, nil
}

// NewActivities creates the MCP activity set for the servers of opts.
func NewActivities(opts ...Option) (*Activities, error) {
	s, err := newServers(opts)
	if err != nil {
		return nil, err
	}
	return &Activities{servers: s}, nil
}

// ActivityRegistry is the part of a worker needed to register activities. Test
// environments implementing only this much can be passed directly.
type ActivityRegistry interface {
	RegisterActivityWithOptions(a any, options activity.RegisterOptions)
}

// RegisterWith wires the MCP activities into a worker.
func (a *Activities) RegisterWith(r ActivityRegistry) {
	r.RegisterActivityWithOptions(a.ListTools, activity.RegisterOptions{Name: ListToolsActivity})
	r.RegisterActivityWithOptions(a.CallTool, activity.RegisterOptions{Name: CallToolActivity})
}

// ListToolsInput is the argument to the list-tools activity.
type ListToolsInput struct {
	Server string `json:"server"`
}

// CallToolInput is the argument to the call-tool activity.
type CallToolInput struct {
	Server    string          `json:"server"`
	Tool      string          `json:"tool"`
	Arguments json.RawMessage `json:"arguments,omitempty"`
}

// ListTools connects to a server and returns its tools, exactly as advertised.
func (a *Activities) ListTools(ctx context.Context, in ListToolsInput) ([]*model.Tool, error) {
	c, err := a.connect(ctx, in.Server)
	if err != nil {
		return nil, err
	}
	defer closeQuietly(ctx, c, in.Server)

	tools, err := c.ListTools(ctx)
	if err != nil {
		return nil, fmt.Errorf("mcp: listing tools on server %q: %w", in.Server, err)
	}
	return tools, nil
}

// CallTool connects to a server and invokes one tool.
//
// A tool that reports IsError is not an activity failure: the result is returned
// unchanged. Only a call that fails (the error return) is retried.
//
// It queues an event when it sends the call and for each progress notification
// from the server. It records a heartbeat carrying a [CallProgress] when it
// starts connecting, when events are queued (at most one per second), and on a
// timer while the call reports nothing.
func (a *Activities) CallTool(ctx context.Context, in CallToolInput) (*model.CallToolResult, error) {
	state := &callState{progress: CallProgress{Server: in.Server, Tool: in.Tool, Phase: CallPhaseConnecting}}
	callCtx, pulse := heartbeat.Start(ctx, state, a.pulse)
	defer pulse.Stop()

	c, err := a.connect(callCtx, in.Server)
	if err != nil {
		return nil, err
	}
	defer closeQuietly(ctx, c, in.Server)

	pulse.Progress(callEvent{phase: CallPhaseCalling})
	callCtx = WithProgressFunc(callCtx, func(u ProgressUpdate) {
		pulse.Progress(callEvent{phase: CallPhaseProgress, update: u})
	})
	res, err := c.CallTool(callCtx, in.Tool, in.Arguments)
	if err != nil {
		// A call failure (transport, protocol, timeout) is retryable, unlike a
		// tool-reported IsError.
		return nil, fmt.Errorf("mcp: calling tool %q on server %q: %w", in.Tool, in.Server, err)
	}
	if res == nil {
		return nil, fmt.Errorf("mcp: server %q returned no result for tool %q", in.Server, in.Tool)
	}
	return res, nil
}

// callEvent is one step of a tool call: a change of phase or a progress
// notification from the server.
type callEvent struct {
	phase CallPhase
	// update is the server's notification. It is set when phase is
	// [CallPhaseProgress].
	update ProgressUpdate
}

// callState is the progress of one tool call. It implements
// [heartbeat.Detailer].
type callState struct {
	progress CallProgress
}

// Detail adds events to the call's progress and returns the [CallProgress] for
// b.
func (s *callState) Detail(b heartbeat.Beat, events []callEvent) any {
	for _, e := range events {
		s.progress.Phase = e.phase
		if e.phase == CallPhaseProgress {
			s.progress.Notifications++
			s.progress.Progress, s.progress.Total = e.update.Progress, e.update.Total
		}
	}

	detail := s.progress
	detail.Keepalive = b.Keepalive
	detail.IdleSeconds = b.Idle.Seconds()
	return detail
}

func (a *Activities) connect(ctx context.Context, server string) (Client, error) {
	f, ok := a.factories[server]
	if !ok {
		return nil, temporal.NewNonRetryableApplicationError(
			fmt.Sprintf("no MCP server named %q is registered (have %v)", server, a.names),
			ErrorTypeUnknownServer, nil)
	}

	c, err := f(ctx)
	if err != nil {
		// Connection failures are usually transient, so this stays retryable.
		return nil, fmt.Errorf("mcp: connecting to server %q: %w", server, err)
	}
	return c, nil
}

// closeQuietly logs a close error rather than returning it, so a teardown
// failure does not fail the activity after its result is already produced.
func closeQuietly(ctx context.Context, c Client, server string) {
	if err := c.Close(); err != nil {
		activity.GetLogger(ctx).Warn("mcp: closing server connection",
			"server", server, "error", err)
	}
}
