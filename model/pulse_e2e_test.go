package model_test

import (
	"context"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	enumspb "go.temporal.io/api/enums/v1"
	"go.temporal.io/sdk/client"
	"go.temporal.io/sdk/temporal"
	"go.temporal.io/sdk/testsuite"
	"go.temporal.io/sdk/worker"
	"go.temporal.io/sdk/workflow"

	"github.com/wleev/temporal-agent-sdk/model"
)

const (
	// pulseHeartbeatTimeout is the HeartbeatTimeout of every model call in these
	// tests.
	pulseHeartbeatTimeout = 3 * time.Second
	// pulseStartToClose is the StartToCloseTimeout of every model call in these
	// tests.
	pulseStartToClose = 90 * time.Second
	// pulsePrompt bounds how long a failure that should come from inside the
	// activity may take.
	pulsePrompt = 20 * time.Second
)

// pulseProvider is a scripted provider for the heartbeat tests.
type pulseProvider struct {
	// invoke serves a call that is not streamed.
	invoke func(ctx context.Context) error
	// stream serves a streamed call.
	stream func(ctx context.Context, sink model.StreamSink) error
}

func (pulseProvider) Name() string { return "pulse" }

func (p pulseProvider) Invoke(ctx context.Context, _ model.Request) (model.Response, error) {
	if err := p.invoke(ctx); err != nil {
		return model.Response{}, err
	}
	return model.Response{Message: model.AssistantMessage("done"), FinishReason: model.FinishStop}, nil
}

func (p pulseProvider) InvokeStream(ctx context.Context, _ model.Request, sink model.StreamSink) (model.Response, error) {
	if err := p.stream(ctx, sink); err != nil {
		return model.Response{}, err
	}
	return model.Response{Message: model.AssistantMessage("done"), FinishReason: model.FinishStop}, nil
}

// pulseWorkflow runs one model call under the tests' activity options.
func pulseWorkflow(ctx workflow.Context, req model.Request) (*model.Response, error) {
	ctx = workflow.WithActivityOptions(ctx, workflow.ActivityOptions{
		StartToCloseTimeout: pulseStartToClose,
		HeartbeatTimeout:    pulseHeartbeatTimeout,
		RetryPolicy:         &temporal.RetryPolicy{MaximumAttempts: 1},
		WaitForCancellation: true,
	})
	var resp model.Response
	if err := workflow.ExecuteActivity(ctx, model.InvokeModelActivity, req).Get(ctx, &resp); err != nil {
		return nil, err
	}
	return &resp, nil
}

// startPulseRun starts a worker serving provider with the limit options limits
// on its own task queue and starts pulseWorkflow on it.
func startPulseRun(
	t *testing.T,
	c client.Client,
	name string,
	provider pulseProvider,
	limits []model.Option,
	stream bool,
) client.WorkflowRun {
	t.Helper()

	acts, err := model.NewActivities([]model.Provider{provider}, limits...)
	require.NoError(t, err)

	taskQueue := "agentsdk-pulse-" + name
	w := worker.New(c, taskQueue, worker.Options{})
	w.RegisterWorkflow(pulseWorkflow)
	acts.Register(w)
	require.NoError(t, w.Start())
	t.Cleanup(w.Stop)

	run, err := c.ExecuteWorkflow(t.Context(), client.StartWorkflowOptions{TaskQueue: taskQueue},
		pulseWorkflow, model.Request{Provider: "pulse", Model: "m", Stream: stream})
	require.NoError(t, err)
	return run
}

// waitWithin returns run's error, failing the test when the run takes longer
// than limit.
func waitWithin(t *testing.T, run client.WorkflowRun, limit time.Duration) error {
	t.Helper()
	ctx, cancel := context.WithTimeout(t.Context(), limit)
	defer cancel()
	err := run.Get(ctx, nil)
	require.NoError(t, ctx.Err(), "the run did not finish within %s", limit)
	return err
}

// TestInvokeModel_HeartbeatsAgainstAServer runs the model activity on a Temporal
// dev server and checks how it behaves when a provider call stalls, hangs, runs
// long, is cancelled, or ignores cancellation.
func TestInvokeModel_HeartbeatsAgainstAServer(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping: needs a Temporal dev server binary (-short)")
	}
	startCtx, cancelStart := context.WithTimeout(t.Context(), 3*time.Minute)
	defer cancelStart()
	server, err := testsuite.StartDevServer(startCtx, testsuite.DevServerOptions{LogLevel: "error"})
	if err != nil {
		t.Skipf("skipping: could not start dev server: %v", err)
	}
	t.Cleanup(func() { _ = server.Stop() })
	c := server.Client()

	untilCancelled := func(ctx context.Context) error {
		<-ctx.Done()
		return ctx.Err()
	}

	t.Run("a stream that stalls fails as stalled", func(t *testing.T) {
		provider := pulseProvider{stream: func(ctx context.Context, sink model.StreamSink) error {
			if err := sink.OnDelta(ctx, model.StreamDelta{Text: "partial", ToolCallIndex: -1}); err != nil {
				return err
			}
			return untilCancelled(ctx)
		}}
		run := startPulseRun(t, c, "stall", provider, []model.Option{model.WithStreamIdle(2 * time.Second)}, true)

		err := waitWithin(t, run, pulsePrompt)

		var appErr *temporal.ApplicationError
		require.ErrorAs(t, err, &appErr)
		assert.Equal(t, model.ErrorTypeStalled, appErr.Type())
	})

	t.Run("a call that is not streamed and hangs fails as stalled", func(t *testing.T) {
		provider := pulseProvider{invoke: untilCancelled}
		run := startPulseRun(t, c, "hang", provider, []model.Option{model.WithUnstreamedLimit(2 * time.Second)}, false)

		err := waitWithin(t, run, pulsePrompt)

		var appErr *temporal.ApplicationError
		require.ErrorAs(t, err, &appErr)
		assert.Equal(t, model.ErrorTypeStalled, appErr.Type())
	})

	t.Run("a healthy call longer than the heartbeat timeout succeeds", func(t *testing.T) {
		provider := pulseProvider{invoke: func(ctx context.Context) error {
			select {
			case <-time.After(2 * pulseHeartbeatTimeout):
				return nil
			case <-ctx.Done():
				return ctx.Err()
			}
		}}
		run := startPulseRun(t, c, "long", provider, nil, false)

		assert.NoError(t, waitWithin(t, run, pulsePrompt))
	})

	t.Run("cancelling the workflow cancels the call in flight", func(t *testing.T) {
		started := make(chan struct{})
		cancelled := make(chan struct{})
		provider := pulseProvider{invoke: func(ctx context.Context) error {
			close(started)
			<-ctx.Done()
			close(cancelled)
			return ctx.Err()
		}}
		run := startPulseRun(t, c, "cancel", provider, nil, false)

		select {
		case <-started:
		case <-time.After(pulsePrompt):
			t.Fatal("the provider call did not start")
		}
		require.NoError(t, c.CancelWorkflow(t.Context(), run.GetID(), run.GetRunID()))

		select {
		case <-cancelled:
		case <-time.After(pulsePrompt):
			t.Fatal("the provider call was not cancelled")
		}
		err := waitWithin(t, run, pulsePrompt)
		var cancelErr *temporal.CanceledError
		assert.ErrorAs(t, err, &cancelErr)
	})

	t.Run("a stalled call that ignores cancellation fails by heartbeat timeout", func(t *testing.T) {
		release := make(chan struct{})
		t.Cleanup(func() { close(release) })
		provider := pulseProvider{stream: func(ctx context.Context, sink model.StreamSink) error {
			if err := sink.OnDelta(ctx, model.StreamDelta{Text: "partial", ToolCallIndex: -1}); err != nil {
				return err
			}
			<-release
			return nil
		}}
		run := startPulseRun(t, c, "deaf", provider, []model.Option{model.WithStreamIdle(2 * time.Second)}, true)

		err := waitWithin(t, run, pulsePrompt)

		var timeoutErr *temporal.TimeoutError
		require.ErrorAs(t, err, &timeoutErr)
		assert.Equal(t, enumspb.TIMEOUT_TYPE_HEARTBEAT, timeoutErr.TimeoutType())
	})
}
