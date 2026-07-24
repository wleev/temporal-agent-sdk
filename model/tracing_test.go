package model_test

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
	oteltrace "go.opentelemetry.io/otel/trace"

	"github.com/wleev/temporal-agent-sdk/model"
)

// tracedProvider is a stub whose Invoke runs under a real span so the activity's
// enrichment has something to annotate.
type tracedProvider struct {
	resp model.Response
	err  error
}

func (p *tracedProvider) Name() string { return "openai" }
func (p *tracedProvider) Invoke(context.Context, model.Request) (model.Response, error) {
	return p.resp, p.err
}

// recorderTracer wires an in-memory span recorder and returns a tracer plus the
// finished-span accessor.
func recorderTracer(t *testing.T) (oteltrace.Tracer, func() []trace.ReadOnlySpan) {
	t.Helper()
	rec := tracetest.NewSpanRecorder()
	tp := trace.NewTracerProvider(trace.WithSpanProcessor(rec))
	t.Cleanup(func() { _ = tp.Shutdown(context.Background()) })
	return tp.Tracer("test"), rec.Ended
}

func TestModelSpan_GenAIAttributes(t *testing.T) {
	tracer, ended := recorderTracer(t)

	acts, err := model.NewActivities(&tracedProvider{
		resp: model.Response{
			Message:      model.AssistantMessage("hi"),
			FinishReason: "stop",
			Usage:        model.Usage{PromptTokens: 11, CompletionTokens: 3, TotalTokens: 14},
		},
	})
	require.NoError(t, err)

	// Run InvokeModel inside a span, as the Temporal interceptor would.
	ctx, span := tracer.Start(context.Background(), "RunActivity")
	_, err = acts.InvokeModel(ctx, model.Request{
		Model:    "gpt-test",
		Messages: []model.Message{model.UserMessage("hi")},
		Settings: model.Settings{Temperature: model.Ptr(0.5), MaxTokens: model.Ptr[int64](256)},
	})
	require.NoError(t, err)
	span.End()

	spans := ended()
	require.Len(t, spans, 1)
	s := spans[0]

	assert.Equal(t, "chat gpt-test", s.Name(), "span renamed per the GenAI convention")

	attrs := map[attribute.Key]attribute.Value{}
	for _, kv := range s.Attributes() {
		attrs[kv.Key] = kv.Value
	}
	assert.Equal(t, "chat", attrs["gen_ai.operation.name"].AsString())
	assert.Equal(t, "openai", attrs["gen_ai.provider.name"].AsString())
	assert.Equal(t, "gpt-test", attrs["gen_ai.request.model"].AsString())
	assert.Equal(t, 0.5, attrs["gen_ai.request.temperature"].AsFloat64())
	assert.Equal(t, int64(256), attrs["gen_ai.request.max_tokens"].AsInt64())
	assert.Equal(t, int64(11), attrs["gen_ai.usage.input_tokens"].AsInt64())
	assert.Equal(t, int64(3), attrs["gen_ai.usage.output_tokens"].AsInt64())
	assert.Equal(t, []string{"stop"}, attrs["gen_ai.response.finish_reasons"].AsStringSlice())
}

func TestModelSpan_ErrorType(t *testing.T) {
	tracer, ended := recorderTracer(t)

	acts, err := model.NewActivities(&tracedProvider{
		err: &model.APIError{StatusCode: 429, Err: context.DeadlineExceeded},
	})
	require.NoError(t, err)

	ctx, span := tracer.Start(context.Background(), "RunActivity")
	_, _ = acts.InvokeModel(ctx, model.Request{Model: "m", Messages: []model.Message{model.UserMessage("hi")}})
	span.End()

	s := ended()[0]
	attrs := map[attribute.Key]attribute.Value{}
	for _, kv := range s.Attributes() {
		attrs[kv.Key] = kv.Value
	}
	assert.Equal(t, "429", attrs["error.type"].AsString())
}

// With no tracer configured, enrichment must be a harmless no-op.
func TestModelSpan_NoTracerIsNoop(t *testing.T) {
	acts, err := model.NewActivities(&tracedProvider{
		resp: model.Response{Message: model.AssistantMessage("hi")},
	})
	require.NoError(t, err)

	// No span in the context: SpanFromContext returns a non-recording span.
	_, err = acts.InvokeModel(context.Background(), model.Request{
		Model: "m", Messages: []model.Message{model.UserMessage("hi")},
	})
	assert.NoError(t, err)
}
