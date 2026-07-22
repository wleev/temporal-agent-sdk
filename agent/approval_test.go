package agent_test

import (
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	"go.temporal.io/sdk/workflow"

	"github.com/wleev/temporal-agent-sdk/agent"
	"github.com/wleev/temporal-agent-sdk/agenttest"
	"github.com/wleev/temporal-agent-sdk/model"
	"github.com/wleev/temporal-agent-sdk/tool"
)

type refundIn struct {
	Amount float64 `json:"amount" jsonschema_description:"Amount to refund"`
}

// refundTool records whether it actually ran, which is how the tests tell an
// approval from a denial.
func refundTool(t *testing.T, ran *bool, opts ...tool.Option) tool.Tool {
	t.Helper()
	tl, err := tool.New("refund", "Issue a refund",
		func(_ workflow.Context, in refundIn) (string, error) {
			*ran = true
			return "refunded", nil
		}, opts...)
	require.NoError(t, err)
	return tl
}

func approvalWorkflow(a *agent.Agent, input string) func(workflow.Context) (*agent.Result, error) {
	return func(ctx workflow.Context) (*agent.Result, error) {
		return agent.Run(ctx, a, input)
	}
}

func TestApproval_Approved(t *testing.T) {
	fake := agenttest.NewFakeProvider(
		agenttest.CallsTool("refund", `{"amount":42}`),
		agenttest.Says("Refund issued."),
	)
	env := newEnv(t, fake)

	var ran bool
	a := mustAgent(agent.NewAgent("support", "test-model",
		agent.WithTools(refundTool(t, &ran, tool.RequiresApproval()))))

	// The loop is parked when this fires; the update releases it.
	env.RegisterDelayedCallback(func() {
		env.UpdateWorkflow(agent.ApproveUpdate, "req-1", &testUpdateCallbacks{t: t},
			agent.ApprovalRequest{CallID: "call-1", Approved: true})
	}, time.Second)

	env.ExecuteWorkflow(approvalWorkflow(a, "refund 42"))

	require.True(t, env.IsWorkflowCompleted())
	require.NoError(t, env.GetWorkflowError())
	require.True(t, ran, "an approved tool must actually run")

	var res agent.Result
	require.NoError(t, env.GetWorkflowResult(&res))
	require.Equal(t, "Refund issued.", res.Output)
}

func TestApproval_Denied(t *testing.T) {
	fake := agenttest.NewFakeProvider(
		agenttest.CallsTool("refund", `{"amount":42}`),
		agenttest.Says("I could not issue the refund."),
	)
	env := newEnv(t, fake)

	var ran bool
	a := mustAgent(agent.NewAgent("support", "test-model",
		agent.WithTools(refundTool(t, &ran, tool.RequiresApproval()))))

	env.RegisterDelayedCallback(func() {
		env.UpdateWorkflow(agent.ApproveUpdate, "req-1", &testUpdateCallbacks{t: t},
			agent.ApprovalRequest{CallID: "call-1", Approved: false, Reason: "over the limit"})
	}, time.Second)

	env.ExecuteWorkflow(approvalWorkflow(a, "refund 42"))

	require.True(t, env.IsWorkflowCompleted())
	require.NoError(t, env.GetWorkflowError(), "a denial must not fail the workflow")
	require.False(t, ran, "a denied tool must not run")

	// The model is told why, so it can respond sensibly rather than retry.
	second := fake.Calls()[1]
	last := second.Messages[len(second.Messages)-1]
	require.Equal(t, model.RoleTool, last.Role)
	require.Contains(t, last.ToolResultText(), "denied")
	require.Contains(t, last.ToolResultText(), "over the limit")
}

// A slow reviewer must not destroy the conversation: the timeout is reported to
// the model as a denial.
func TestApproval_TimesOut(t *testing.T) {
	fake := agenttest.NewFakeProvider(
		agenttest.CallsTool("refund", `{"amount":42}`),
		agenttest.Says("No one approved, so I stopped."),
	)
	env := newEnv(t, fake)

	var ran bool
	a := mustAgent(agent.NewAgent("support", "test-model",
		agent.WithTools(refundTool(t, &ran, tool.RequiresApproval())),
		agent.WithApprovalTimeout(time.Hour)))

	env.ExecuteWorkflow(approvalWorkflow(a, "refund 42"))

	require.True(t, env.IsWorkflowCompleted())
	require.NoError(t, env.GetWorkflowError(), "an approval timeout must not fail the workflow")
	require.False(t, ran, "a tool must not run after its approval timed out")

	second := fake.Calls()[1]
	last := second.Messages[len(second.Messages)-1]
	require.Contains(t, last.ToolResultText(), "timed out")
	require.Contains(t, last.ToolResultText(), "Do not retry", "the model should be steered away from retrying")
}

// Ungated tools must not park.
func TestApproval_NotRequiredRunsImmediately(t *testing.T) {
	fake := agenttest.NewFakeProvider(
		agenttest.CallsTool("refund", `{"amount":42}`),
		agenttest.Says("done"),
	)
	env := newEnv(t, fake)

	var ran bool
	a := mustAgent(agent.NewAgent("support", "test-model",
		agent.WithTools(refundTool(t, &ran)))) // no RequiresApproval

	env.ExecuteWorkflow(approvalWorkflow(a, "refund 42"))

	require.True(t, env.IsWorkflowCompleted())
	require.NoError(t, env.GetWorkflowError())
	require.True(t, ran)
}

// The pending query is how a UI discovers what needs review.
func TestApproval_PendingQuery(t *testing.T) {
	fake := agenttest.NewFakeProvider(
		agenttest.CallsTool("refund", `{"amount":42}`),
		agenttest.Says("done"),
	)
	env := newEnv(t, fake)

	var ran bool
	a := mustAgent(agent.NewAgent("support", "test-model",
		agent.WithTools(refundTool(t, &ran, tool.WithApprovalPrompt("Refunds over €25 need review.")))))

	env.RegisterDelayedCallback(func() {
		res, err := env.QueryWorkflow(agent.PendingApprovalsQuery)
		require.NoError(t, err)

		var pending []agent.PendingApproval
		require.NoError(t, res.Get(&pending))
		require.Len(t, pending, 1)
		require.Equal(t, "call-1", pending[0].CallID)
		require.Equal(t, "refund", pending[0].Tool)
		require.Equal(t, "support", pending[0].Agent)
		require.JSONEq(t, `{"amount":42}`, pending[0].Arguments,
			"the reviewer must see what the tool would be called with")
		require.Equal(t, "Refunds over €25 need review.", pending[0].Prompt)

		env.UpdateWorkflow(agent.ApproveUpdate, "req-1", &testUpdateCallbacks{t: t},
			agent.ApprovalRequest{CallID: "call-1", Approved: true})
	}, time.Second)

	env.ExecuteWorkflow(approvalWorkflow(a, "refund 42"))
	require.True(t, env.IsWorkflowCompleted())
	require.NoError(t, env.GetWorkflowError())

	// The decided approval must leave the pending list.
	res, err := env.QueryWorkflow(agent.PendingApprovalsQuery)
	require.NoError(t, err)
	var pending []agent.PendingApproval
	require.NoError(t, res.Get(&pending))
	require.Empty(t, pending, "a decided approval must no longer be pending")
}

// The validator rejects bad decisions synchronously and keeps them out of
// history. A signal could not do this.
func TestApproval_ValidatorRejectsUnknownCallID(t *testing.T) {
	fake := agenttest.NewFakeProvider(
		agenttest.CallsTool("refund", `{"amount":42}`),
		agenttest.Says("done"),
	)
	env := newEnv(t, fake)

	var ran bool
	a := mustAgent(agent.NewAgent("support", "test-model",
		agent.WithTools(refundTool(t, &ran, tool.RequiresApproval()))))

	var rejected error
	env.RegisterDelayedCallback(func() {
		env.UpdateWorkflow(agent.ApproveUpdate, "bad", &testUpdateCallbacks{
			t:          t,
			onValidate: func(err error) { rejected = err },
			expectFail: true,
		}, agent.ApprovalRequest{CallID: "no-such-call", Approved: true})
	}, time.Second)

	// Release the real approval so the workflow can finish.
	env.RegisterDelayedCallback(func() {
		env.UpdateWorkflow(agent.ApproveUpdate, "req-1", &testUpdateCallbacks{t: t},
			agent.ApprovalRequest{CallID: "call-1", Approved: true})
	}, 2*time.Second)

	env.ExecuteWorkflow(approvalWorkflow(a, "refund 42"))

	require.True(t, env.IsWorkflowCompleted())
	require.NoError(t, env.GetWorkflowError())
	require.Error(t, rejected, "an unknown call ID must be rejected by the validator")
	require.Contains(t, rejected.Error(), "no pending approval")
}

func TestApproval_ValidatorRejectsEmptyCallID(t *testing.T) {
	fake := agenttest.NewFakeProvider(
		agenttest.CallsTool("refund", `{"amount":42}`),
		agenttest.Says("done"),
	)
	env := newEnv(t, fake)

	var ran bool
	a := mustAgent(agent.NewAgent("support", "test-model",
		agent.WithTools(refundTool(t, &ran, tool.RequiresApproval()))))

	var rejected error
	env.RegisterDelayedCallback(func() {
		env.UpdateWorkflow(agent.ApproveUpdate, "bad", &testUpdateCallbacks{
			t:          t,
			onValidate: func(err error) { rejected = err },
			expectFail: true,
		}, agent.ApprovalRequest{CallID: "", Approved: true})
	}, time.Second)

	env.RegisterDelayedCallback(func() {
		env.UpdateWorkflow(agent.ApproveUpdate, "req-1", &testUpdateCallbacks{t: t},
			agent.ApprovalRequest{CallID: "call-1", Approved: true})
	}, 2*time.Second)

	env.ExecuteWorkflow(approvalWorkflow(a, "refund 42"))

	require.True(t, env.IsWorkflowCompleted())
	require.ErrorContains(t, rejected, "call_id must not be empty")
}

// Two gated tools in one turn must each get their own decision.
func TestApproval_ParallelGatedTools(t *testing.T) {
	fake := agenttest.NewFakeProvider(
		agenttest.CallsTools(
			agenttest.ToolCall{ID: "x", Name: "refund", Args: `{"amount":10}`},
			agenttest.ToolCall{ID: "y", Name: "refund", Args: `{"amount":20}`},
		),
		agenttest.Says("done"),
	)
	env := newEnv(t, fake)

	var count int
	tl, err := tool.New("refund", "Issue a refund",
		func(_ workflow.Context, in refundIn) (string, error) {
			count++
			return "refunded", nil
		}, tool.RequiresApproval())
	require.NoError(t, err)

	a := mustAgent(agent.NewAgent("support", "test-model", agent.WithTools(tl)))

	env.RegisterDelayedCallback(func() {
		res, err := env.QueryWorkflow(agent.PendingApprovalsQuery)
		require.NoError(t, err)
		var pending []agent.PendingApproval
		require.NoError(t, res.Get(&pending))
		require.Len(t, pending, 2, "both gated calls should be pending at once")

		// Approve one, deny the other.
		env.UpdateWorkflow(agent.ApproveUpdate, "u1", &testUpdateCallbacks{t: t},
			agent.ApprovalRequest{CallID: "x", Approved: true})
		env.UpdateWorkflow(agent.ApproveUpdate, "u2", &testUpdateCallbacks{t: t},
			agent.ApprovalRequest{CallID: "y", Approved: false, Reason: "too much"})
	}, time.Second)

	env.ExecuteWorkflow(approvalWorkflow(a, "refund"))

	require.True(t, env.IsWorkflowCompleted())
	require.NoError(t, env.GetWorkflowError())
	require.Equal(t, 1, count, "only the approved call should have run")

	toolMsgs := filterRole(fake.Calls()[1].Messages, model.RoleTool)
	require.Len(t, toolMsgs, 2)
	require.Equal(t, "refunded", toolMsgs[0].ToolResultText())
	require.Contains(t, toolMsgs[1].ToolResultText(), "too much")
}

// testUpdateCallbacks adapts the test env's update callbacks to assertions.
type testUpdateCallbacks struct {
	t          *testing.T
	onValidate func(error)
	expectFail bool
}

func (c *testUpdateCallbacks) Accept() {}

func (c *testUpdateCallbacks) Reject(err error) {
	if c.onValidate != nil {
		c.onValidate(err)
		return
	}
	if !c.expectFail {
		c.t.Errorf("update unexpectedly rejected: %v", err)
	}
}

func (c *testUpdateCallbacks) Complete(success any, err error) {
	if err != nil && !c.expectFail {
		c.t.Errorf("update unexpectedly failed: %v", err)
	}
	_ = success
}
