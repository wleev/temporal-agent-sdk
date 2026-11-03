package model_test

import (
	"context"
	"errors"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.temporal.io/sdk/activity"
	"go.temporal.io/sdk/temporal"
	"go.temporal.io/sdk/testsuite"
	"go.temporal.io/sdk/workflow"

	"github.com/wleev/temporal-agent-sdk/model"
)

// failOnceProvider fails its first call with a retryable error, then answers "ok".
type failOnceProvider struct {
	calls atomic.Int32
}

func (p *failOnceProvider) Name() string { return "fail-once" }
func (p *failOnceProvider) Invoke(context.Context, model.Request) (model.Response, error) {
	if p.calls.Add(1) == 1 {
		return model.Response{}, &model.APIError{StatusCode: 503, Err: errors.New("unavailable")}
	}
	return model.Response{Message: model.AssistantMessage("ok")}, nil
}

func TestInvokeModel_ReportsAttempt(t *testing.T) {
	acts, err := model.NewActivities([]model.Provider{&failOnceProvider{}})
	require.NoError(t, err)

	var s testsuite.WorkflowTestSuite
	env := s.NewTestWorkflowEnvironment()
	env.RegisterActivityWithOptions(acts.InvokeModel, activity.RegisterOptions{Name: model.InvokeModelActivity})

	env.ExecuteWorkflow(func(ctx workflow.Context) (*model.Response, error) {
		ctx = workflow.WithActivityOptions(ctx, workflow.ActivityOptions{
			StartToCloseTimeout: time.Minute,
			RetryPolicy: &temporal.RetryPolicy{
				InitialInterval: time.Millisecond,
				MaximumAttempts: 2,
			},
		})
		var resp model.Response
		err := workflow.ExecuteActivity(ctx, model.InvokeModelActivity, model.Request{Provider: "fail-once"}).Get(ctx, &resp)
		return &resp, err
	})
	require.True(t, env.IsWorkflowCompleted())
	require.NoError(t, env.GetWorkflowError())

	var resp model.Response
	require.NoError(t, env.GetWorkflowResult(&resp))
	assert.Equal(t, int32(2), resp.Attempt)
	assert.Equal(t, "ok", resp.Message.Text())
}
