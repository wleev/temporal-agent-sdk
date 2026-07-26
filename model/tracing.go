package model

import (
	"context"
	"errors"
	"strconv"

	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/trace"
)

// OTel GenAI semantic-convention attribute keys the model activity sets on its
// span. Any OpenTelemetry backend that understands gen_ai.* reads model, token
// usage, and finish reason from them.
//
// Reference: OpenTelemetry semantic conventions for generative AI.
const (
	attrOperationName = attribute.Key("gen_ai.operation.name")
	attrProviderName  = attribute.Key("gen_ai.provider.name")
	attrSystem        = attribute.Key("gen_ai.system") // legacy alias; emitted for older backends
	attrRequestModel  = attribute.Key("gen_ai.request.model")
	attrTemperature   = attribute.Key("gen_ai.request.temperature")
	attrMaxTokens     = attribute.Key("gen_ai.request.max_tokens")
	attrInputTokens   = attribute.Key("gen_ai.usage.input_tokens")
	attrOutputTokens  = attribute.Key("gen_ai.usage.output_tokens")
	attrFinishReasons = attribute.Key("gen_ai.response.finish_reasons")
	attrErrorType     = attribute.Key("error.type")

	operationChat = "chat"
)

// startModelSpan annotates the current activity span with request-side GenAI
// attributes and names it per the convention ("chat {model}").
//
// It reads the span the Temporal OTel interceptor already started for this
// activity. When no tracer is configured that span is non-recording and every
// call below is a no-op, so tracing stays opt-in.
func startModelSpan(ctx context.Context, provider string, req Request) trace.Span {
	span := trace.SpanFromContext(ctx)
	if !span.IsRecording() {
		return span
	}

	span.SetName(operationChat + " " + req.Model)
	span.SetAttributes(
		attrOperationName.String(operationChat),
		attrProviderName.String(provider),
		attrSystem.String(provider),
		attrRequestModel.String(req.Model),
	)
	if t := req.Settings.Temperature; t != nil {
		span.SetAttributes(attrTemperature.Float64(*t))
	}
	if m := req.Settings.MaxTokens; m != nil {
		span.SetAttributes(attrMaxTokens.Int64(*m))
	}
	return span
}

// recordModelResponse adds response-side usage and finish reason.
func recordModelResponse(span trace.Span, resp *Response) {
	if !span.IsRecording() {
		return
	}
	span.SetAttributes(
		attrInputTokens.Int64(resp.Usage.PromptTokens),
		attrOutputTokens.Int64(resp.Usage.CompletionTokens),
	)
	if resp.FinishReason != "" {
		span.SetAttributes(attrFinishReasons.StringSlice([]string{string(resp.FinishReason)}))
	}
}

// recordModelError marks the span failed with a typed error, per the stable
// error.type convention.
func recordModelError(span trace.Span, err error) {
	if !span.IsRecording() {
		return
	}
	span.SetStatus(codes.Error, err.Error())
	span.SetAttributes(attrErrorType.String(errorType(err)))
}

// errorType classifies the failure for error.type: the HTTP status for a
// provider APIError, otherwise the concrete Go type.
func errorType(err error) string {
	var apiErr *APIError
	if errors.As(err, &apiErr) && apiErr.StatusCode != 0 {
		return strconv.Itoa(apiErr.StatusCode)
	}
	return "_OTHER"
}
