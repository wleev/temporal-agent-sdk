package plugin_test

import (
	"context"
	"testing"
	"time"

	"github.com/nexus-rpc/sdk-go/nexus"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.temporal.io/sdk/activity"
	"go.temporal.io/sdk/client"
	"go.temporal.io/sdk/testsuite"
	"go.temporal.io/sdk/worker"
	"go.temporal.io/sdk/workflow"

	"github.com/wleev/temporal-agent-sdk/agent"
	"github.com/wleev/temporal-agent-sdk/agenttest"
	"github.com/wleev/temporal-agent-sdk/conversation"
	"github.com/wleev/temporal-agent-sdk/mcp"
	"github.com/wleev/temporal-agent-sdk/model"
	"github.com/wleev/temporal-agent-sdk/plugin"
)

// recordingRegistry implements the plugin StartWorker registry, capturing the
// names of everything the plugin registers so a test can assert the full set.
type recordingRegistry struct {
	activities []string
	workflows  []string
}

func (r *recordingRegistry) RegisterActivityWithOptions(_ any, o activity.RegisterOptions) {
	r.activities = append(r.activities, o.Name)
}

func (r *recordingRegistry) RegisterWorkflowWithOptions(_ any, o workflow.RegisterOptions) {
	r.workflows = append(r.workflows, o.Name)
}

func (r *recordingRegistry) RegisterDynamicActivity(any, activity.DynamicRegisterOptions) {}
func (r *recordingRegistry) RegisterDynamicWorkflow(any, workflow.DynamicRegisterOptions) {}
func (r *recordingRegistry) RegisterNexusService(*nexus.Service)                          {}

func TestNew_RequiresProvider(t *testing.T) {
	_, err := plugin.New(plugin.Config{})
	require.Error(t, err, "a plugin with no providers must not build")
}

// StartWorker must register every item its Config declares, so a worker holding
// the plugin serves the model, both MCP shapes, sub-agents, and conversations.
func TestPlugin_RegistersConfiguredItems(t *testing.T) {
	fake := agenttest.NewFakeProvider()

	mcpActs := mcp.NewActivities()
	require.NoError(t, mcpActs.Register("srv", func(context.Context) (mcp.Client, error) {
		return nil, nil
	}))
	statefulActs := mcp.NewStatefulActivities()
	require.NoError(t, statefulActs.Register("session", func(context.Context) (mcp.Client, error) {
		return nil, nil
	}))

	child, err := agent.NewAgent("child", "test-model")
	require.NoError(t, err)
	reg := agent.NewRegistry()
	require.NoError(t, reg.Add(child))

	p, err := plugin.New(plugin.Config{
		Providers:    []model.Provider{fake},
		MCP:          mcpActs,
		StatefulMCP:  statefulActs,
		Agents:       reg,
		Conversation: conversation.New(reg),
	})
	require.NoError(t, err)

	rec := &recordingRegistry{}
	nextCalled := false
	err = p.StartWorker(context.Background(),
		worker.PluginStartWorkerOptions{WorkerRegistry: rec},
		func(context.Context, worker.PluginStartWorkerOptions) error {
			nextCalled = true
			return nil
		})
	require.NoError(t, err)
	assert.True(t, nextCalled, "StartWorker must hand off to the next plugin")

	assert.Subset(t, rec.activities, []string{
		model.InvokeModelActivity,
		mcp.ListToolsActivity,
		mcp.CallToolActivity,
		mcp.RunSessionActivity,
	}, "model and MCP activities must be registered")
	assert.Subset(t, rec.workflows, []string{
		agent.WorkflowName,
		conversation.WorkflowName,
	}, "sub-agent and conversation workflows must be registered")
}

// A worker built only from the plugin runs an agent that delegates to a
// sub-agent: the plugin's StartWorker is the sole thing that registered the
// model activity and the sub-agent child workflow.
func TestPlugin_EndToEnd(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping plugin e2e: needs a dev server binary (-short)")
	}

	fake := agenttest.NewFakeProvider(
		agenttest.CallsTool("research", `{"input":"the capital of Belgium"}`),
		agenttest.Says("Brussels."),
		agenttest.Says("The capital of Belgium is Brussels."),
	)

	researcher, err := agent.NewAgent("researcher", "test-model",
		agent.WithInstructions("You research things."))
	require.NoError(t, err)
	sub, err := agent.AsSubAgent(researcher, "research", "Research a topic")
	require.NoError(t, err)
	parent, err := agent.NewAgent("assistant", "test-model", agent.WithTools(sub))
	require.NoError(t, err)

	reg := agent.NewRegistry()
	require.NoError(t, reg.Add(researcher, parent))

	p, err := plugin.New(plugin.Config{
		Providers: []model.Provider{fake},
		Agents:    reg,
	})
	require.NoError(t, err)

	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Minute)
	defer cancel()
	srv, err := testsuite.StartDevServer(ctx, testsuite.DevServerOptions{LogLevel: "error"})
	if err != nil {
		t.Skipf("skipping: could not start dev server: %v", err)
	}
	t.Cleanup(func() { _ = srv.Stop() })
	c := srv.Client()

	const tq = "agentsdk-plugin-test"
	w := worker.New(c, tq, worker.Options{Plugins: []worker.Plugin{p}})
	// The entry workflow is the consumer's own; only the SDK wiring comes from
	// the plugin.
	w.RegisterWorkflowWithOptions(func(wctx workflow.Context) (*agent.Result, error) {
		return agent.Run(wctx, parent, "What is the capital of Belgium?")
	}, workflow.RegisterOptions{Name: "entry"})
	require.NoError(t, w.Start())
	t.Cleanup(w.Stop)

	run, err := c.ExecuteWorkflow(ctx, client.StartWorkflowOptions{TaskQueue: tq}, "entry")
	require.NoError(t, err)

	var res agent.Result
	require.NoError(t, run.Get(ctx, &res))
	assert.Equal(t, "The capital of Belgium is Brussels.", res.Output)
}
