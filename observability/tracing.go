// Package observability wires OpenTelemetry tracing into an agent worker.
//
// The Temporal OTel contrib interceptor produces the structural trace — a span
// for the workflow (the whole agent run), one per activity (each model call,
// each activity-backed tool, each MCP operation), and one per sub-agent child
// workflow — with context propagated across every boundary so the spans parent
// correctly. It suppresses spans during replay, so no duplicates. Use
// [TracingInterceptor] and set it on BOTH client and worker.
//
// The model activity separately enriches its own span with GenAI
// semantic-convention attributes (see the model package); nothing here is needed
// for that.
//
// The interceptor uses the globally-registered TracerProvider by default, so the
// application owns exporter setup (OTLP, stdout, etc.).
package observability

import (
	"go.temporal.io/sdk/contrib/opentelemetry"
	"go.temporal.io/sdk/interceptor"
)

// TracingInterceptor builds the Temporal OpenTelemetry interceptor using the
// global TracerProvider. Set the returned value on both
// client.Options.Interceptors and worker.Options.Interceptors — it implements
// both. Use [TracingInterceptorWith] to override the tracer or disable
// signal/query/update tracing.
func TracingInterceptor() (interceptor.Interceptor, error) {
	return TracingInterceptorWith(opentelemetry.TracerOptions{})
}

// TracingInterceptorWith is [TracingInterceptor] with explicit tracer options.
func TracingInterceptorWith(o opentelemetry.TracerOptions) (interceptor.Interceptor, error) {
	return opentelemetry.NewTracingInterceptor(o)
}
