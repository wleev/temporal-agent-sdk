package agent

import (
	"encoding/json"
	"fmt"

	"go.temporal.io/sdk/temporal"
	"go.temporal.io/sdk/workflow"

	"github.com/wleev/temporal-agent-sdk/internal/schema"
	"github.com/wleev/temporal-agent-sdk/model"
)

// TypedResult is the outcome of a [RunTyped], the structured value plus the same
// bookkeeping a plain [Result] carries.
type TypedResult[Out any] struct {
	// Value is the model's final answer, decoded and validated against Out.
	Value Out

	// Result is the underlying run: full transcript, usage, turn count. Its
	// Output holds the raw JSON the value was decoded from.
	Result
}

// RunTyped runs the agent and returns a validated value of type Out.
//
// The JSON Schema is reflected from Out and set as the run's output schema, so
// the model's final answer is constrained to match. Tools still work in
// intermediate turns; only the terminal answer is structured.
//
// Like [Run], it creates a [Session] internally. Use struct types for Out; the
// reflector rejects recursive types and requires the OpenAI-strict shape the
// providers send.
func RunTyped[Out any](ctx workflow.Context, a *Agent, input string) (*TypedResult[Out], error) {
	return RunTypedWith[Out](ctx, a, input, RunOptions{})
}

// RunTypedWith is [RunTyped] with [RunOptions].
//
//workflowcheck:ignore schema reflection and json.Unmarshal are deterministic for a fixed type
func RunTypedWith[Out any](ctx workflow.Context, a *Agent, input string, ro RunOptions) (*TypedResult[Out], error) {
	s, err := NewSession(ctx)
	if err != nil {
		return nil, err
	}

	// schema.For is deterministic (byte-stable), so reflecting here in workflow
	// code is replay-safe.
	raw, err := schema.For[Out]()
	if err != nil {
		return nil, temporal.NewNonRetryableApplicationError(
			fmt.Sprintf("agent %q: output type is not a valid schema: %v", a.name, err),
			ErrorTypeConfig, err)
	}

	ro.OutputSchema = &model.OutputSchema{Name: "output", Schema: raw, Strict: true}

	res, err := s.RunWith(ctx, a, input, ro)
	if err != nil {
		return nil, err
	}

	if len(res.StructuredOutput) == 0 {
		// With a schema set, providers surface the terminal answer as structured
		// JSON. An empty result means the model ended on plain text, a permanent
		// mismatch for this run.
		return nil, temporal.NewNonRetryableApplicationError(
			fmt.Sprintf("agent %q produced no structured output for the requested schema", a.name),
			ErrorTypeConfig, nil)
	}

	var value Out
	if err := json.Unmarshal(res.StructuredOutput, &value); err != nil {
		return nil, temporal.NewNonRetryableApplicationError(
			fmt.Sprintf("agent %q: decoding structured output: %v", a.name, err),
			ErrorTypeConfig, err)
	}

	return &TypedResult[Out]{Value: value, Result: *res}, nil
}
