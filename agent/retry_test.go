package agent_test

import (
	"context"
	"errors"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.temporal.io/sdk/activity"
	"go.temporal.io/sdk/temporal"
	"go.temporal.io/sdk/testsuite"
	"go.temporal.io/sdk/workflow"

	"github.com/wleev/temporal-agent-sdk/agent"
	"github.com/wleev/temporal-agent-sdk/model"
)

// alwaysFailingProvider mimics a misconfigured endpoint: every call is a
// retryable transport error (status 0), as a bad base URL would produce.
type alwaysFailingProvider struct{ calls int }

func (p *alwaysFailingProvider) Name() string { return "broken" }
func (p *alwaysFailingProvider) Invoke(context.Context, model.Request) (model.Response, error) {
	p.calls++
	return model.Response{}, &model.APIError{StatusCode: 0, Err: errors.New("connection refused")}
}

// A persistently failing model must make the workflow FAIL after a bounded
// number of attempts, not retry forever. Before the default retry cap this hung
// the workflow indefinitely.
func TestRun_BrokenModelFailsInsteadOfHanging(t *testing.T) {
	prov := &alwaysFailingProvider{}
	acts, err := model.NewActivities(prov)
	require.NoError(t, err)

	var s testsuite.WorkflowTestSuite
	env := s.NewTestWorkflowEnvironment()
	env.RegisterActivityWithOptions(acts.InvokeModel, activity.RegisterOptions{Name: model.InvokeModelActivity})

	a, err := agent.NewAgent("assistant", "m")
	require.NoError(t, err)
	env.ExecuteWorkflow(func(ctx workflow.Context) (*agent.Result, error) {
		return agent.Run(ctx, a, "hi")
	})

	require.True(t, env.IsWorkflowCompleted(), "the workflow must finish, not hang")
	assert.Error(t, env.GetWorkflowError(), "a persistently broken model must fail the workflow")

	// Bounded: the model activity retried up to the default cap, then gave up —
	// it did not retry forever.
	assert.Equal(t, agent.DefaultModelMaxAttempts, prov.calls,
		"the model call must be retried a bounded number of times")
}

// A caller-supplied retry policy is respected (not overridden by the default).
func TestRun_RespectsCustomRetryPolicy(t *testing.T) {
	prov := &alwaysFailingProvider{}
	acts, err := model.NewActivities(prov)
	require.NoError(t, err)

	var s testsuite.WorkflowTestSuite
	env := s.NewTestWorkflowEnvironment()
	env.RegisterActivityWithOptions(acts.InvokeModel, activity.RegisterOptions{Name: model.InvokeModelActivity})

	a, err := agent.NewAgent("assistant", "m",
		agent.WithModelActivityOptions(workflow.ActivityOptions{
			StartToCloseTimeout: agent.DefaultModelTimeout,
			RetryPolicy:         &temporal.RetryPolicy{MaximumAttempts: 2},
		}))
	require.NoError(t, err)
	env.ExecuteWorkflow(func(ctx workflow.Context) (*agent.Result, error) {
		return agent.Run(ctx, a, "hi")
	})

	require.True(t, env.IsWorkflowCompleted())
	assert.Error(t, env.GetWorkflowError())
	assert.Equal(t, 2, prov.calls, "a caller-set retry policy must not be overridden")
}
