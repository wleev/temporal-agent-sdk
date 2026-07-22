package model_test

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	"go.temporal.io/sdk/temporal"
	"go.temporal.io/sdk/testsuite"

	"github.com/wleev/temporal-agent-sdk/model"
)

// stubProvider returns a fixed response or error.
type stubProvider struct {
	name string
	resp model.Response
	err  error
}

func (p *stubProvider) Name() string { return p.name }
func (p *stubProvider) Invoke(context.Context, model.Request) (model.Response, error) {
	if p.err != nil {
		return model.Response{}, p.err
	}
	return p.resp, nil
}

func ptr[T any](v T) *T { return &v }

// Retry posture is the whole reason the provider returns a neutral error rather
// than calling the model itself: getting this table wrong means either hammering
// a dead endpoint or giving up on a transient blip.
func TestAPIError_Retryable(t *testing.T) {
	for _, tc := range []struct {
		name string
		err  model.APIError
		want bool
	}{
		{"transport failure (no response)", model.APIError{StatusCode: 0}, true},
		{"408 request timeout", model.APIError{StatusCode: 408}, true},
		{"409 conflict", model.APIError{StatusCode: 409}, true},
		{"429 rate limited", model.APIError{StatusCode: 429}, true},
		{"500 server error", model.APIError{StatusCode: 500}, true},
		{"503 unavailable", model.APIError{StatusCode: 503}, true},

		{"400 bad request", model.APIError{StatusCode: 400}, false},
		{"401 unauthorized", model.APIError{StatusCode: 401}, false},
		{"403 forbidden", model.APIError{StatusCode: 403}, false},
		{"404 not found", model.APIError{StatusCode: 404}, false},
		{"422 unprocessable", model.APIError{StatusCode: 422}, false},

		// An explicit instruction from the backend beats our inference.
		{"x-should-retry overrides 400", model.APIError{StatusCode: 400, ShouldRetry: ptr(true)}, true},
		{"x-should-retry overrides 500", model.APIError{StatusCode: 500, ShouldRetry: ptr(false)}, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			require.Equal(t, tc.want, tc.err.Retryable())
		})
	}
}

func TestInvokeModel_NonRetryableMapping(t *testing.T) {
	var s testsuite.WorkflowTestSuite
	env := s.NewTestActivityEnvironment()

	acts, err := model.NewActivities(&stubProvider{
		name: "stub",
		err:  &model.APIError{StatusCode: 401, Err: errors.New("bad key")},
	})
	require.NoError(t, err)
	acts.Register(env)

	_, err = env.ExecuteActivity(model.InvokeModelActivity, model.Request{Model: "m"})
	require.Error(t, err)

	var appErr *temporal.ApplicationError
	require.ErrorAs(t, err, &appErr)
	require.True(t, appErr.NonRetryable(), "a 401 must not be retried: the key is wrong on every attempt")
	require.Equal(t, "ModelHTTP401", appErr.Type())
}

func TestInvokeModel_RetryableMapping(t *testing.T) {
	var s testsuite.WorkflowTestSuite
	env := s.NewTestActivityEnvironment()

	acts, err := model.NewActivities(&stubProvider{
		name: "stub",
		err:  &model.APIError{StatusCode: 503, Err: errors.New("unavailable")},
	})
	require.NoError(t, err)
	acts.Register(env)

	_, err = env.ExecuteActivity(model.InvokeModelActivity, model.Request{Model: "m"})
	require.Error(t, err)

	var appErr *temporal.ApplicationError
	require.ErrorAs(t, err, &appErr)
	require.False(t, appErr.NonRetryable(), "a 503 should be retried")
}

// A backend-supplied delay should reach Temporal, so a 429 waits out the real
// rate-limit window instead of following our backoff curve.
func TestInvokeModel_RetryAfterBecomesNextRetryDelay(t *testing.T) {
	var s testsuite.WorkflowTestSuite
	env := s.NewTestActivityEnvironment()

	acts, err := model.NewActivities(&stubProvider{
		name: "stub",
		err: &model.APIError{
			StatusCode: 429,
			RetryAfter: 3 * time.Second,
			Err:        errors.New("slow down"),
		},
	})
	require.NoError(t, err)
	acts.Register(env)

	_, err = env.ExecuteActivity(model.InvokeModelActivity, model.Request{Model: "m"})
	require.Error(t, err)

	var appErr *temporal.ApplicationError
	require.ErrorAs(t, err, &appErr)
	require.Equal(t, 3*time.Second, appErr.NextRetryDelay())
}

func TestInvokeModel_Success(t *testing.T) {
	var s testsuite.WorkflowTestSuite
	env := s.NewTestActivityEnvironment()

	acts, err := model.NewActivities(&stubProvider{
		name: "stub",
		resp: model.Response{
			Message: model.AssistantMessage("hi"),
			Usage:   model.Usage{TotalTokens: 7},
		},
	})
	require.NoError(t, err)
	acts.Register(env)

	val, err := env.ExecuteActivity(model.InvokeModelActivity, model.Request{Model: "m"})
	require.NoError(t, err)

	var resp model.Response
	require.NoError(t, val.Get(&resp))
	require.Equal(t, "hi", resp.Message.Text())
	require.Equal(t, int64(7), resp.Usage.TotalTokens)
}

// An empty provider name is the common single-provider case and must resolve.
func TestInvokeModel_EmptyProviderNameSelectsSole(t *testing.T) {
	var s testsuite.WorkflowTestSuite
	env := s.NewTestActivityEnvironment()

	acts, err := model.NewActivities(&stubProvider{
		name: "only",
		resp: model.Response{Message: model.AssistantMessage("ok")},
	})
	require.NoError(t, err)
	acts.Register(env)

	_, err = env.ExecuteActivity(model.InvokeModelActivity, model.Request{Model: "m"})
	require.NoError(t, err)
}

// With several providers, an unnamed request is ambiguous and must fail loudly
// rather than pick one arbitrarily.
func TestInvokeModel_AmbiguousProviderFails(t *testing.T) {
	var s testsuite.WorkflowTestSuite
	env := s.NewTestActivityEnvironment()

	acts, err := model.NewActivities(
		&stubProvider{name: "a"},
		&stubProvider{name: "b"},
	)
	require.NoError(t, err)
	acts.Register(env)

	_, err = env.ExecuteActivity(model.InvokeModelActivity, model.Request{Model: "m"})
	require.Error(t, err)

	var appErr *temporal.ApplicationError
	require.ErrorAs(t, err, &appErr)
	require.True(t, appErr.NonRetryable())
	require.Equal(t, model.ErrorTypeNoProvider, appErr.Type())
	require.Contains(t, appErr.Error(), "does not name a provider")
}

func TestInvokeModel_UnknownProviderFails(t *testing.T) {
	var s testsuite.WorkflowTestSuite
	env := s.NewTestActivityEnvironment()

	acts, err := model.NewActivities(&stubProvider{name: "real"})
	require.NoError(t, err)
	acts.Register(env)

	_, err = env.ExecuteActivity(model.InvokeModelActivity,
		model.Request{Model: "m", Provider: "ghost"})
	require.Error(t, err)

	var appErr *temporal.ApplicationError
	require.ErrorAs(t, err, &appErr)
	require.True(t, appErr.NonRetryable(), "the provider set is fixed at startup")
	require.Contains(t, appErr.Error(), `no provider named "ghost"`)
}

// streamingStub is a provider that can stream; it records whether InvokeStream
// (vs Invoke) was used.
type streamingStub struct {
	stubProvider
	streamed bool
}

func (p *streamingStub) InvokeStream(ctx context.Context, req model.Request, sink model.StreamSink) (model.Response, error) {
	p.streamed = true
	_ = sink.OnDelta(ctx, model.StreamDelta{Text: p.resp.Message.Text(), ToolCallIndex: -1})
	return p.resp, p.err
}

func TestInvokeModel_StreamsWhenEnabled(t *testing.T) {
	var s testsuite.WorkflowTestSuite
	env := s.NewTestActivityEnvironment()

	prov := &streamingStub{stubProvider: stubProvider{
		name: "s",
		resp: model.Response{Message: model.AssistantMessage("hi")},
	}}
	acts, err := model.NewActivities(prov)
	require.NoError(t, err)

	var got []model.StreamDelta
	acts.SetStreamSink(func(context.Context) (model.StreamSink, error) {
		return sinkFunc(func(_ context.Context, d model.StreamDelta) error {
			got = append(got, d)
			return nil
		}), nil
	})
	acts.Register(env)

	_, err = env.ExecuteActivity(model.InvokeModelActivity, model.Request{Model: "m", Stream: true})
	require.NoError(t, err)
	require.True(t, prov.streamed, "streaming provider should have been used")
	require.Len(t, got, 1)
	require.Equal(t, "hi", got[0].Text)
}

// Without a sink, a streaming request falls back to Invoke — same result.
func TestInvokeModel_NoSinkFallsBackToInvoke(t *testing.T) {
	var s testsuite.WorkflowTestSuite
	env := s.NewTestActivityEnvironment()

	prov := &streamingStub{stubProvider: stubProvider{
		name: "s",
		resp: model.Response{Message: model.AssistantMessage("hi")},
	}}
	acts, err := model.NewActivities(prov) // no SetStreamSink
	require.NoError(t, err)
	acts.Register(env)

	_, err = env.ExecuteActivity(model.InvokeModelActivity, model.Request{Model: "m", Stream: true})
	require.NoError(t, err)
	require.False(t, prov.streamed, "with no sink, streaming must fall back to Invoke")
}

// A non-streaming provider ignores the Stream flag gracefully.
func TestInvokeModel_NonStreamingProviderFallsBack(t *testing.T) {
	var s testsuite.WorkflowTestSuite
	env := s.NewTestActivityEnvironment()

	acts, err := model.NewActivities(&stubProvider{name: "s", resp: model.Response{Message: model.AssistantMessage("hi")}})
	require.NoError(t, err)
	acts.SetStreamSink(func(context.Context) (model.StreamSink, error) {
		return sinkFunc(func(context.Context, model.StreamDelta) error { return nil }), nil
	})
	acts.Register(env)

	_, err = env.ExecuteActivity(model.InvokeModelActivity, model.Request{Model: "m", Stream: true})
	require.NoError(t, err, "a provider that cannot stream must still serve a streaming request")
}

type sinkFunc func(context.Context, model.StreamDelta) error

func (f sinkFunc) OnDelta(ctx context.Context, d model.StreamDelta) error { return f(ctx, d) }

func TestNewActivities_Validation(t *testing.T) {
	_, err := model.NewActivities()
	require.ErrorContains(t, err, "at least one provider")

	_, err = model.NewActivities(&stubProvider{name: ""})
	require.ErrorContains(t, err, "empty name")

	_, err = model.NewActivities(&stubProvider{name: "dup"}, &stubProvider{name: "dup"})
	require.ErrorContains(t, err, "duplicate provider name")
}
