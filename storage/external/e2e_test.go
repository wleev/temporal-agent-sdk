package external_test

import (
	"context"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	enumspb "go.temporal.io/api/enums/v1"
	historypb "go.temporal.io/api/history/v1"
	"go.temporal.io/sdk/client"
	"go.temporal.io/sdk/testsuite"
	"go.temporal.io/sdk/worker"
	"go.temporal.io/sdk/workflow"
	"google.golang.org/protobuf/proto"

	"github.com/wleev/temporal-agent-sdk/agent"
	"github.com/wleev/temporal-agent-sdk/agenttest"
	"github.com/wleev/temporal-agent-sdk/model"
	"github.com/wleev/temporal-agent-sdk/plugin"
	"github.com/wleev/temporal-agent-sdk/storage"
	"github.com/wleev/temporal-agent-sdk/storage/external"
	"github.com/wleev/temporal-agent-sdk/storage/storagetest"
)

const (
	e2eWorkflow  = "storage-e2e"
	e2eThreshold = 4 << 10
)

func e2eAgentWorkflow(ctx workflow.Context) (*agent.Result, error) {
	a, err := agent.NewAgent("assistant", "test-model")
	if err != nil {
		return nil, err
	}
	return agent.Run(ctx, a, "Write a long report.")
}

// TestExternalStorage_AgentRunEndToEnd runs an agent whose model reply exceeds
// the threshold and checks that the run returns the full reply, that no history
// event reaches the threshold, and that a replayer configured through
// [external.NewPlugin] replays the history.
func TestExternalStorage_AgentRunEndToEnd(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping storage e2e: needs a dev server binary (-short)")
	}

	report := strings.Repeat("A quarterly figure worth reporting. ", 4000)
	fake := agenttest.NewFakeProvider(agenttest.Says(report))

	mem := storagetest.NewMemoryStore()
	ext, err := external.New(mem, storage.WithThreshold(e2eThreshold))
	require.NoError(t, err)
	storagePlugin, err := external.NewPlugin(ext)
	require.NoError(t, err)
	sdkPlugin, err := plugin.New(plugin.Config{Providers: []model.Provider{fake}})
	require.NoError(t, err)

	ctx, cancel := context.WithTimeout(t.Context(), 3*time.Minute)
	defer cancel()
	srv, err := testsuite.StartDevServer(ctx, testsuite.DevServerOptions{LogLevel: "error"})
	if err != nil {
		t.Skipf("skipping: could not start dev server: %v", err)
	}
	t.Cleanup(func() { _ = srv.Stop() })

	c, err := client.Dial(client.Options{
		HostPort: srv.FrontendHostPort(),
		Plugins:  []client.Plugin{storagePlugin},
	})
	require.NoError(t, err)
	t.Cleanup(c.Close)

	const tq = "agentsdk-storage-test"
	w := worker.New(c, tq, worker.Options{Plugins: []worker.Plugin{sdkPlugin}})
	w.RegisterWorkflowWithOptions(e2eAgentWorkflow, workflow.RegisterOptions{Name: e2eWorkflow})
	require.NoError(t, w.Start())
	t.Cleanup(w.Stop)

	run, err := c.ExecuteWorkflow(ctx, client.StartWorkflowOptions{TaskQueue: tq}, e2eWorkflow)
	require.NoError(t, err)
	var res agent.Result
	require.NoError(t, run.Get(ctx, &res))
	assert.Equal(t, report, res.Output)
	assert.NotEmpty(t, mem.Keys(), "the oversized reply must be written to the store")

	// A client without external storage reads the history as recorded.
	history := &historypb.History{}
	iter := srv.Client().GetWorkflowHistory(ctx, run.GetID(), run.GetRunID(), false,
		enumspb.HISTORY_EVENT_FILTER_TYPE_ALL_EVENT)
	for iter.HasNext() {
		ev, err := iter.Next()
		require.NoError(t, err)
		history.Events = append(history.Events, ev)
	}
	for _, ev := range history.Events {
		size := proto.Size(ev)
		assert.Less(t, size, e2eThreshold, "event %d (%s) is %d bytes; oversized payloads must be references",
			ev.GetEventId(), ev.GetEventType(), size)
	}

	replayer, err := worker.NewWorkflowReplayerWithOptions(worker.WorkflowReplayerOptions{
		Plugins: []worker.Plugin{storagePlugin},
	})
	require.NoError(t, err)
	replayer.RegisterWorkflowWithOptions(e2eAgentWorkflow, workflow.RegisterOptions{Name: e2eWorkflow})
	assert.NoError(t, replayer.ReplayWorkflowHistory(nil, history))
}
