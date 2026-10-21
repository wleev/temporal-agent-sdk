package agent_test

import (
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.temporal.io/sdk/temporal"
	"go.temporal.io/sdk/workflow"

	"github.com/wleev/temporal-agent-sdk/agent"
	"github.com/wleev/temporal-agent-sdk/agenttest"
	"github.com/wleev/temporal-agent-sdk/tool"
)

// TestNewSession_WithoutApprovalHandlersRegistersNeither verifies that a
// session created with WithoutApprovalHandlers registers neither the approve
// update nor the pending-approvals query.
func TestNewSession_WithoutApprovalHandlersRegistersNeither(t *testing.T) {
	env := newEnv(t, agenttest.NewFakeProvider())

	var queryErr, updateErr error
	env.RegisterDelayedCallback(func() {
		_, queryErr = env.QueryWorkflow(agent.PendingApprovalsQuery)
		env.UpdateWorkflow(agent.ApproveUpdate, "u1", &testUpdateCallbacks{
			t:          t,
			onValidate: func(err error) { updateErr = err },
			expectFail: true,
		}, agent.ApprovalRequest{CallID: "call-1", Approved: true})
	}, time.Second)

	env.ExecuteWorkflow(func(ctx workflow.Context) error {
		if _, err := agent.NewSession(ctx, agent.WithoutApprovalHandlers()); err != nil {
			return err
		}
		return workflow.Sleep(ctx, time.Hour)
	})
	require.True(t, env.IsWorkflowCompleted())
	assert.NoError(t, env.GetWorkflowError())

	if assert.Error(t, queryErr) {
		assert.Contains(t, queryErr.Error(), agent.PendingApprovalsQuery)
	}
	if assert.Error(t, updateErr) {
		assert.Contains(t, updateErr.Error(), agent.ApproveUpdate)
	}
}

// TestNewSession_WithoutApprovalHandlersLeavesOtherSessionsHandlers verifies
// that sessions created with WithoutApprovalHandlers after a default session
// leave the default session's approval handlers in place.
func TestNewSession_WithoutApprovalHandlersLeavesOtherSessionsHandlers(t *testing.T) {
	fake := agenttest.NewFakeProvider(
		agenttest.CallsTool("refund", `{"amount":42}`),
		agenttest.Says("Refund issued."),
	)
	env := newEnv(t, fake)

	var ran bool
	a, err := agent.NewAgent("support", "test-model",
		agent.WithTools(refundTool(t, &ran, tool.RequiresApproval())))
	require.NoError(t, err)

	env.RegisterDelayedCallback(func() {
		env.UpdateWorkflow(agent.ApproveUpdate, "req-1", &testUpdateCallbacks{t: t},
			agent.ApprovalRequest{CallID: "call-1", Approved: true})
	}, time.Second)

	env.ExecuteWorkflow(func(ctx workflow.Context) (*agent.Result, error) {
		s, err := agent.NewSession(ctx)
		if err != nil {
			return nil, err
		}
		for range 2 {
			if _, err := agent.NewSession(ctx, agent.WithoutApprovalHandlers()); err != nil {
				return nil, err
			}
		}
		return s.Run(ctx, a, "refund 42")
	})
	require.True(t, env.IsWorkflowCompleted())
	assert.NoError(t, env.GetWorkflowError())
	assert.True(t, ran, "the default session's approval released the tool")
}

// TestSessionWithoutApprovalHandlers_UngatedToolRuns verifies that a session
// without approval handlers runs ungated tools.
func TestSessionWithoutApprovalHandlers_UngatedToolRuns(t *testing.T) {
	fake := agenttest.NewFakeProvider(
		agenttest.CallsTool("get_weather", `{"city":"Ghent"}`),
		agenttest.Says("It is 18°C in Ghent."),
	)
	env := newEnv(t, fake)

	a, err := agent.NewAgent("assistant", "test-model", agent.WithTools(weatherTool(t)))
	require.NoError(t, err)

	var res *agent.Result
	env.ExecuteWorkflow(func(ctx workflow.Context) error {
		s, err := agent.NewSession(ctx, agent.WithoutApprovalHandlers())
		if err != nil {
			return err
		}
		res, err = s.Run(ctx, a, "Weather in Ghent?")
		return err
	})
	require.True(t, env.IsWorkflowCompleted())
	assert.NoError(t, env.GetWorkflowError())
	if assert.NotNil(t, res) {
		assert.Equal(t, "It is 18°C in Ghent.", res.Output)
	}
}

// TestSessionWithoutApprovalHandlers_GatedToolFailsRun verifies that a call to
// a gated tool in a session without approval handlers fails the run with a
// non-retryable ErrorTypeApproval error, and the tool does not run.
func TestSessionWithoutApprovalHandlers_GatedToolFailsRun(t *testing.T) {
	fake := agenttest.NewFakeProvider(
		agenttest.CallsTool("refund", `{"amount":42}`),
		agenttest.Says("unreached"),
	)
	env := newEnv(t, fake)

	var ran bool
	a, err := agent.NewAgent("support", "test-model",
		agent.WithTools(refundTool(t, &ran, tool.RequiresApproval())))
	require.NoError(t, err)

	var runErr error
	env.ExecuteWorkflow(func(ctx workflow.Context) error {
		s, err := agent.NewSession(ctx, agent.WithoutApprovalHandlers())
		if err != nil {
			return err
		}
		_, runErr = s.Run(ctx, a, "refund 42")
		return runErr
	})
	require.True(t, env.IsWorkflowCompleted())

	var appErr *temporal.ApplicationError
	if assert.ErrorAs(t, runErr, &appErr) {
		assert.Equal(t, agent.ErrorTypeApproval, appErr.Type())
		assert.True(t, appErr.NonRetryable())
		assert.Contains(t, appErr.Error(), `"refund"`)
	}
	assert.Error(t, env.GetWorkflowError())
	assert.False(t, ran)
	assert.Equal(t, 1, fake.CallCount())
}

// TestSessionWithoutApprovalHandlers_MixedReplyFailsRun verifies that a reply
// calling a gated and an ungated tool fails the run in a session without
// approval handlers.
func TestSessionWithoutApprovalHandlers_MixedReplyFailsRun(t *testing.T) {
	fake := agenttest.NewFakeProvider(
		agenttest.CallsTools(
			agenttest.ToolCall{ID: "x", Name: "refund", Args: `{"amount":10}`},
			agenttest.ToolCall{ID: "y", Name: "get_weather", Args: `{"city":"Ghent"}`},
		),
		agenttest.Says("unreached"),
	)
	env := newEnv(t, fake)

	var ran, weatherRan bool
	weather, err := tool.New("get_weather", "Look up the weather",
		func(_ workflow.Context, in weatherIn) (string, error) {
			weatherRan = true
			return "18C in " + in.City, nil
		})
	require.NoError(t, err)
	a, err := agent.NewAgent("support", "test-model",
		agent.WithTools(refundTool(t, &ran, tool.RequiresApproval()), weather))
	require.NoError(t, err)

	var runErr error
	env.ExecuteWorkflow(func(ctx workflow.Context) error {
		s, err := agent.NewSession(ctx, agent.WithoutApprovalHandlers())
		if err != nil {
			return err
		}
		_, runErr = s.Run(ctx, a, "refund and weather")
		return runErr
	})
	require.True(t, env.IsWorkflowCompleted())

	var appErr *temporal.ApplicationError
	if assert.ErrorAs(t, runErr, &appErr) {
		assert.Equal(t, agent.ErrorTypeApproval, appErr.Type())
		assert.True(t, appErr.NonRetryable())
	}
	if assert.Error(t, runErr) {
		assert.Contains(t, runErr.Error(), `tool "refund"`)
	}
	assert.False(t, ran)
	assert.True(t, weatherRan, "the ungated sibling ran")
	assert.Equal(t, 1, fake.CallCount())
}
