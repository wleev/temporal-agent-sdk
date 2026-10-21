package tool_test

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.temporal.io/sdk/testsuite"
	"go.temporal.io/sdk/workflow"

	"github.com/wleev/temporal-agent-sdk/tool"
)

// TestCallInfo_RoundTrip verifies CallInfoFrom returns the CallInfo that WithCallInfo put on the context.
func TestCallInfo_RoundTrip(t *testing.T) {
	var s testsuite.WorkflowTestSuite
	env := s.NewTestWorkflowEnvironment()

	want := tool.CallInfo{ID: "call-7", Turn: 2, Index: 1}
	var got tool.CallInfo
	var found bool
	env.ExecuteWorkflow(func(ctx workflow.Context) error {
		got, found = tool.CallInfoFrom(tool.WithCallInfo(ctx, want))
		return nil
	})
	require.True(t, env.IsWorkflowCompleted())
	assert.NoError(t, env.GetWorkflowError())

	assert.True(t, found)
	assert.Equal(t, want, got)
}

// TestCallInfo_AbsentReportsFalse verifies CallInfoFrom reports false for a context that carries no CallInfo.
func TestCallInfo_AbsentReportsFalse(t *testing.T) {
	var s testsuite.WorkflowTestSuite
	env := s.NewTestWorkflowEnvironment()

	found := true
	var got tool.CallInfo
	env.ExecuteWorkflow(func(ctx workflow.Context) error {
		got, found = tool.CallInfoFrom(ctx)
		return nil
	})
	require.True(t, env.IsWorkflowCompleted())
	assert.NoError(t, env.GetWorkflowError())

	assert.False(t, found)
	assert.Equal(t, tool.CallInfo{}, got)
}
