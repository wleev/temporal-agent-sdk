package conversation_test

import (
	"context"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	enumspb "go.temporal.io/api/enums/v1"
	historypb "go.temporal.io/api/history/v1"
	"go.temporal.io/sdk/activity"
	"go.temporal.io/sdk/client"
	"go.temporal.io/sdk/testsuite"
	"go.temporal.io/sdk/worker"
	"go.temporal.io/sdk/workflow"

	"github.com/wleev/temporal-agent-sdk/agent"
	"github.com/wleev/temporal-agent-sdk/agenttest"
	"github.com/wleev/temporal-agent-sdk/conversation"
	"github.com/wleev/temporal-agent-sdk/model"
)

// A conversation with Update-driven turns, a Query, and continue-as-new is a
// long-lived workflow whose history must replay deterministically. Run it on a
// real dev server (Update handlers and CAN are exercised for real), then replay
// the recorded history against the same code.
func TestReplay_ConversationSession(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping replay test: needs a dev server binary (-short)")
	}

	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Minute)
	defer cancel()
	srv, err := testsuite.StartDevServer(ctx, testsuite.DevServerOptions{
		ClientOptions: &client.Options{}, LogLevel: "error",
	})
	if err != nil {
		t.Skipf("skipping: could not start dev server: %v", err)
	}
	t.Cleanup(func() { _ = srv.Stop() })
	c := srv.Client()

	a, err := agent.NewAgent("assistant", "test-model", agent.WithInstructions("Be brief."))
	require.NoError(t, err)
	reg := agent.NewRegistry()
	require.NoError(t, reg.Add(a))
	sum := &fixedSummarizer{}
	cv := conversation.New(reg, conversation.WithCompactor(
		conversation.NewSummarizingCompactor(sum, conversation.WithKeepLast(1))))

	newWorker := func() worker.Worker {
		w := worker.New(c, "conv-replay", worker.Options{})
		fake := agenttest.NewFakeProvider(
			agenttest.Says("r1"), agenttest.Says("r2"), agenttest.Says("r3"))
		acts, err := model.NewActivities(fake)
		require.NoError(t, err)
		w.RegisterActivityWithOptions(acts.InvokeModel,
			activity.RegisterOptions{Name: model.InvokeModelActivity})
		cv.Register(w)
		agent.RegisterWorkflows(w, reg)
		return w
	}

	w := newWorker()
	require.NoError(t, w.Start())
	t.Cleanup(w.Stop)

	wfID := "conv-replay-session"
	run, err := c.ExecuteWorkflow(ctx, client.StartWorkflowOptions{ID: wfID, TaskQueue: "conv-replay"},
		conversation.WorkflowName, conversation.Input{Agent: "assistant"})
	require.NoError(t, err)

	// Three turns.
	for _, text := range []string{"one", "two", "three"} {
		h, err := c.UpdateWorkflow(ctx, client.UpdateWorkflowOptions{
			WorkflowID: wfID, UpdateName: conversation.SendMessageUpdate,
			WaitForStage: client.WorkflowUpdateStageCompleted,
			Args:         []any{conversation.Message{Text: text}},
		})
		require.NoError(t, err)
		require.NoError(t, h.Get(ctx, &conversation.Reply{}))
	}

	// Terminate so the (otherwise perpetual) workflow closes; the history up to
	// here is what we replay.
	require.NoError(t, c.TerminateWorkflow(ctx, wfID, run.GetRunID(), "test done"))

	hist := history(t, ctx, c, wfID, run.GetRunID())

	r := worker.NewWorkflowReplayer()
	cv.Register(replayRegistry{r})
	agent.RegisterWorkflows(r, reg)
	require.NoError(t, r.ReplayWorkflowHistory(nil, hist),
		"a multi-turn conversation must replay deterministically")
}

func history(t *testing.T, ctx context.Context, c client.Client, wfID, runID string) *historypb.History {
	t.Helper()
	iter := c.GetWorkflowHistory(ctx, wfID, runID, false, enumspb.HISTORY_EVENT_FILTER_TYPE_ALL_EVENT)
	var h historypb.History
	for iter.HasNext() {
		ev, err := iter.Next()
		require.NoError(t, err)
		h.Events = append(h.Events, ev)
	}
	require.NotEmpty(t, h.Events)
	return &h
}

// replayRegistry adapts a WorkflowReplayer to conversation.WorkflowRegistry.
type replayRegistry struct{ r worker.WorkflowReplayer }

func (a replayRegistry) RegisterWorkflowWithOptions(w any, o workflow.RegisterOptions) {
	a.r.RegisterWorkflowWithOptions(w, o)
}
func (a replayRegistry) RegisterActivityWithOptions(any, activity.RegisterOptions) {
	// The replayer does not run activities; their results come from history.
}
