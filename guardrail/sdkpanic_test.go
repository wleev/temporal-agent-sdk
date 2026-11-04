package guardrail_test

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.temporal.io/sdk/temporal"
	"go.temporal.io/sdk/testsuite"
	"go.temporal.io/sdk/workflow"

	"github.com/wleev/temporal-agent-sdk/guardrail"
)

func TestScreen_AnSDKIllegalStatePanicIsRaisedAgain(t *testing.T) {
	g, err := guardrail.Func("versioned", stagesOf(guardrail.StageInput), func(ctx workflow.Context, _ guardrail.CheckInput) (guardrail.Verdict, error) {
		workflow.GetVersion(ctx, "guardrail-change", 2, 2)
		return guardrail.Allow(), nil
	})
	require.NoError(t, err)

	var s testsuite.WorkflowTestSuite
	env := s.NewTestWorkflowEnvironment()
	returned := false
	env.ExecuteWorkflow(func(ctx workflow.Context) error {
		workflow.GetVersion(ctx, "guardrail-change", 1, 1)
		_, _ = guardrail.Screen(ctx, []guardrail.Guardrail{g}, guardrail.CheckInput{Stage: guardrail.StageInput, Text: "Weather in Ghent?"})
		returned = true
		return nil
	})

	require.True(t, env.IsWorkflowCompleted())
	assert.False(t, returned, "Screen does not return after the SDK's illegal-state panic")
	var panicErr *temporal.PanicError
	if assert.ErrorAs(t, env.GetWorkflowError(), &panicErr) {
		assert.Contains(t, panicErr.Error(), "removed support of version")
	}
}
