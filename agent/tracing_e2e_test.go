package agent_test

import (
	"context"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	"go.opentelemetry.io/otel"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
	"go.temporal.io/sdk/activity"
	"go.temporal.io/sdk/client"
	"go.temporal.io/sdk/interceptor"
	"go.temporal.io/sdk/testsuite"
	"go.temporal.io/sdk/worker"
	"go.temporal.io/sdk/workflow"

	"github.com/wleev/temporal-agent-sdk/agent"
	"github.com/wleev/temporal-agent-sdk/agenttest"
	"github.com/wleev/temporal-agent-sdk/model"
	"github.com/wleev/temporal-agent-sdk/observability"
)

// End-to-end tracing needs a real server: the OTel interceptor propagates trace
// context across the workflow→activity boundary via the Temporal header, which
// the in-memory env does not fully model. Gated by -short like the other
// dev-server tests.
func TestTracing_ModelSpanUnderWorkflow(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping tracing e2e: needs a dev server binary (-short)")
	}

	// Global span recorder; the interceptor defaults to the global provider.
	rec := tracetest.NewSpanRecorder()
	tp := sdktrace.NewTracerProvider(sdktrace.WithSpanProcessor(rec))
	prev := otel.GetTracerProvider()
	otel.SetTracerProvider(tp)
	t.Cleanup(func() { otel.SetTracerProvider(prev); _ = tp.Shutdown(context.Background()) })

	ti, err := observability.TracingInterceptor()
	require.NoError(t, err)

	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Minute)
	defer cancel()
	srv, err := testsuite.StartDevServer(ctx, testsuite.DevServerOptions{
		ClientOptions: &client.Options{Interceptors: []interceptor.ClientInterceptor{ti}},
		LogLevel:      "error",
	})
	if err != nil {
		t.Skipf("skipping: could not start dev server: %v", err)
	}
	t.Cleanup(func() { _ = srv.Stop() })
	c := srv.Client()

	const tq = "agentsdk-tracing-test"
	w := worker.New(c, tq, worker.Options{Interceptors: []interceptor.WorkerInterceptor{ti}})

	fake := agenttest.NewFakeProvider(agenttest.Says("Hello."))
	w.RegisterActivityWithOptions(agenttest.MustActivities(fake).InvokeModel,
		activity.RegisterOptions{Name: model.InvokeModelActivity})
	w.RegisterWorkflowWithOptions(func(wctx workflow.Context) (*agent.Result, error) {
		return agent.Run(wctx, mustAgent(agent.NewAgent("assistant", "gpt-test")), "hi")
	}, workflow.RegisterOptions{Name: "traced_agent"})
	require.NoError(t, w.Start())
	t.Cleanup(w.Stop)

	run, err := c.ExecuteWorkflow(ctx, client.StartWorkflowOptions{TaskQueue: tq}, "traced_agent")
	require.NoError(t, err)
	require.NoError(t, run.Get(ctx, &agent.Result{}))

	// The model call produced a "chat gpt-test" span carrying GenAI attributes,
	// parented (via header propagation) under a workflow-level span.
	spans := rec.Ended()
	var chat sdktrace.ReadOnlySpan
	for _, s := range spans {
		if s.Name() == "chat gpt-test" {
			chat = s
		}
	}
	require.NotNil(t, chat, "expected a 'chat gpt-test' model span; got %d spans", len(spans))

	attrs := map[string]string{}
	for _, kv := range chat.Attributes() {
		attrs[string(kv.Key)] = kv.Value.Emit()
	}
	require.Equal(t, "chat", attrs["gen_ai.operation.name"])
	require.Equal(t, "gpt-test", attrs["gen_ai.request.model"])
	require.Contains(t, attrs, "gen_ai.usage.input_tokens")

	require.True(t, chat.Parent().IsValid(), "the model span must be parented, not orphaned")
}
