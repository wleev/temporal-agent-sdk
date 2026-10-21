package agent_test

import (
	"errors"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.temporal.io/sdk/workflow"

	"github.com/wleev/temporal-agent-sdk/agent"
	"github.com/wleev/temporal-agent-sdk/agenttest"
	"github.com/wleev/temporal-agent-sdk/tool"
)

// probeTool returns a tool named "probe" that stores the tool.CallInfo of each
// call in seen, keyed by call ID, and fails a call whose context carries none.
func probeTool(t *testing.T, seen map[string]tool.CallInfo, opts ...tool.Option) tool.Tool {
	t.Helper()
	tl, err := tool.New("probe", "Record the call information",
		func(ctx workflow.Context, _ echoIn) (string, error) {
			info, ok := tool.CallInfoFrom(ctx)
			if !ok {
				return "", errors.New("no call information on the context")
			}
			seen[info.ID] = info
			return "ok", nil
		}, opts...)
	require.NoError(t, err)
	return tl
}

// TestRun_ToolReadsCallInfo verifies each tool call's context carries the
// call's ID, its turn, and its index among all tool calls of the reply,
// counting calls to unknown tools.
func TestRun_ToolReadsCallInfo(t *testing.T) {
	fake := agenttest.NewFakeProvider(
		agenttest.CallsTools(
			agenttest.ToolCall{ID: "a", Name: "probe", Args: `{"text":"1"}`},
			agenttest.ToolCall{ID: "b", Name: "no_such_tool"},
			agenttest.ToolCall{ID: "c", Name: "probe", Args: `{"text":"3"}`},
		),
		agenttest.CallsTools(agenttest.ToolCall{ID: "d", Name: "probe", Args: `{"text":"4"}`}),
		agenttest.Says("done"),
	)
	env := newEnv(t, fake)

	seen := map[string]tool.CallInfo{}
	a, err := agent.NewAgent("assistant", "test-model", agent.WithTools(probeTool(t, seen)))
	require.NoError(t, err)

	env.ExecuteWorkflow(func(ctx workflow.Context) (*agent.Result, error) {
		return agent.Run(ctx, a, "go")
	})
	require.True(t, env.IsWorkflowCompleted())
	assert.NoError(t, env.GetWorkflowError())

	assert.Equal(t, map[string]tool.CallInfo{
		"a": {ID: "a", Turn: 1, Index: 0},
		"c": {ID: "c", Turn: 1, Index: 2},
		"d": {ID: "d", Turn: 2, Index: 0},
	}, seen)
}

// TestSession_CallInfoTurnRestartsPerRun verifies CallInfo.Turn counts from 1 in each run of a session.
func TestSession_CallInfoTurnRestartsPerRun(t *testing.T) {
	fake := agenttest.NewFakeProvider(
		agenttest.CallsTools(agenttest.ToolCall{ID: "r1", Name: "probe", Args: `{"text":"1"}`}),
		agenttest.Says("one"),
		agenttest.CallsTools(agenttest.ToolCall{ID: "r2", Name: "probe", Args: `{"text":"2"}`}),
		agenttest.Says("two"),
	)
	env := newEnv(t, fake)

	seen := map[string]tool.CallInfo{}
	a, err := agent.NewAgent("assistant", "test-model", agent.WithTools(probeTool(t, seen)))
	require.NoError(t, err)

	env.ExecuteWorkflow(func(ctx workflow.Context) error {
		s, err := agent.NewSession(ctx)
		if err != nil {
			return err
		}
		if _, err := s.Run(ctx, a, "first"); err != nil {
			return err
		}
		_, err = s.Run(ctx, a, "second")
		return err
	})
	require.True(t, env.IsWorkflowCompleted())
	assert.NoError(t, env.GetWorkflowError())

	assert.Equal(t, map[string]tool.CallInfo{
		"r1": {ID: "r1", Turn: 1, Index: 0},
		"r2": {ID: "r2", Turn: 1, Index: 0},
	}, seen)
}

// TestApproval_ApprovedToolReadsCallInfo verifies a gated tool reads its CallInfo after its approval.
func TestApproval_ApprovedToolReadsCallInfo(t *testing.T) {
	fake := agenttest.NewFakeProvider(
		agenttest.CallsTool("probe", `{"text":"1"}`),
		agenttest.Says("done"),
	)
	env := newEnv(t, fake)

	seen := map[string]tool.CallInfo{}
	a, err := agent.NewAgent("support", "test-model",
		agent.WithTools(probeTool(t, seen, tool.RequiresApproval())))
	require.NoError(t, err)

	env.RegisterDelayedCallback(func() {
		env.UpdateWorkflow(agent.ApproveUpdate, "req-1", &testUpdateCallbacks{t: t},
			agent.ApprovalRequest{CallID: "call-1", Approved: true})
	}, time.Second)

	env.ExecuteWorkflow(approvalWorkflow(a, "go"))
	require.True(t, env.IsWorkflowCompleted())
	assert.NoError(t, env.GetWorkflowError())

	assert.Equal(t, map[string]tool.CallInfo{
		"call-1": {ID: "call-1", Turn: 1, Index: 0},
	}, seen)
}
