package agent_test

import (
	"context"
	"encoding/json"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.temporal.io/sdk/activity"
	"go.temporal.io/sdk/client"
	"go.temporal.io/sdk/testsuite"
	"go.temporal.io/sdk/worker"
	"go.temporal.io/sdk/workflow"

	"github.com/wleev/temporal-agent-sdk/agent"
	"github.com/wleev/temporal-agent-sdk/model"
	"github.com/wleev/temporal-agent-sdk/tool"
)

// gateProvider drives a run that calls one gated tool: it requests the tool until
// a tool result comes back, then answers.
type gateProvider struct{}

func (gateProvider) Name() string { return "gate" }

func (gateProvider) Invoke(_ context.Context, req model.Request) (model.Response, error) {
	last := req.Messages[len(req.Messages)-1]
	if last.Role == model.RoleTool {
		return model.Response{Message: model.AssistantMessage("done"), FinishReason: model.FinishStop}, nil
	}
	return model.Response{
		Message: model.Message{Role: model.RoleAssistant, Blocks: []model.Block{
			model.ToolCallBlock("call-1", "gated_tool", json.RawMessage(`{}`)),
		}},
		FinishReason: model.FinishToolCalls,
	}, nil
}

func toolResultText(msgs []model.Message) string {
	for _, m := range msgs {
		if m.Role == model.RoleTool {
			return m.ToolResultText()
		}
	}
	return ""
}

// TestApprovalClient_ApproveDenyPending verifies Pending, Approve, and Deny
// against a live workflow. It needs a dev server; skipped under -short.
func TestApprovalClient_ApproveDenyPending(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping approval-client e2e: needs a dev server binary (-short)")
	}

	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Minute)
	defer cancel()
	srv, err := testsuite.StartDevServer(ctx, testsuite.DevServerOptions{LogLevel: "error"})
	if err != nil {
		t.Skipf("skipping: could not start dev server: %v", err)
	}
	t.Cleanup(func() { _ = srv.Stop() })
	c := srv.Client()

	const tq = "agentsdk-approval-test"
	w := worker.New(c, tq, worker.Options{})

	acts, err := model.NewActivities(gateProvider{})
	require.NoError(t, err)
	w.RegisterActivityWithOptions(acts.InvokeModel, activity.RegisterOptions{Name: model.InvokeModelActivity})

	gatedTool, err := tool.New("gated_tool", "A gated tool",
		func(_ workflow.Context, _ struct{}) (string, error) { return "tool ran", nil },
		tool.RequiresApproval())
	require.NoError(t, err)
	gatedAgent, err := agent.NewAgent("assistant", "gate-model", agent.WithTools(gatedTool))
	require.NoError(t, err)

	w.RegisterWorkflowWithOptions(func(wctx workflow.Context) (*agent.Result, error) {
		return agent.Run(wctx, gatedAgent, "do it")
	}, workflow.RegisterOptions{Name: "gated_agent"})
	require.NoError(t, w.Start())
	t.Cleanup(w.Stop)

	ac := agent.NewApprovalClient(c)

	awaitPending := func(wfID string) agent.PendingApproval {
		var pending []agent.PendingApproval
		require.Eventually(t, func() bool {
			pending, _ = ac.Pending(ctx, wfID)
			return len(pending) == 1
		}, 30*time.Second, 100*time.Millisecond, "the gated call should be pending")
		return pending[0]
	}

	t.Run("approve", func(t *testing.T) {
		run, err := c.ExecuteWorkflow(ctx, client.StartWorkflowOptions{TaskQueue: tq}, "gated_agent")
		require.NoError(t, err)

		p := awaitPending(run.GetID())
		assert.Equal(t, "gated_tool", p.Tool)
		assert.Equal(t, "assistant", p.Agent)

		// An unknown call ID is rejected by the update validator.
		assert.Error(t, ac.Approve(ctx, run.GetID(), "bogus"), "a stale call ID is rejected")

		require.NoError(t, ac.Approve(ctx, run.GetID(), p.CallID))

		var res agent.Result
		require.NoError(t, run.Get(ctx, &res))
		assert.Equal(t, "done", res.Output)
		assert.Contains(t, toolResultText(res.Messages), "tool ran", "the approved tool executed")
	})

	t.Run("deny", func(t *testing.T) {
		run, err := c.ExecuteWorkflow(ctx, client.StartWorkflowOptions{TaskQueue: tq}, "gated_agent")
		require.NoError(t, err)

		p := awaitPending(run.GetID())
		require.NoError(t, ac.Deny(ctx, run.GetID(), p.CallID, "not allowed"))

		var res agent.Result
		require.NoError(t, run.Get(ctx, &res))
		assert.Contains(t, toolResultText(res.Messages), "denied", "the model is told of the denial")
		assert.Contains(t, toolResultText(res.Messages), "not allowed", "the denial reason reaches the model")
	})
}
