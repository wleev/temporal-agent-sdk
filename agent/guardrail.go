package agent

import (
	"errors"
	"fmt"
	"slices"

	"go.temporal.io/sdk/temporal"
	"go.temporal.io/sdk/workflow"

	"github.com/wleev/temporal-agent-sdk/guardrail"
	"github.com/wleev/temporal-agent-sdk/model"
)

// TripwireError reports that a guardrail blocked the run. [AsTripwire]
// recovers it from an error, [NewTripwireError] wraps it in a non-retryable
// application error of type [ErrorTypeTripwire], and [TripwireFromRecord]
// builds it from a block's [guardrail.Record].
type TripwireError struct {
	// Stage is the stage of the blocked content.
	Stage guardrail.Stage `json:"stage"`

	// Guardrail is the name of the guardrail that blocked it.
	Guardrail string `json:"guardrail"`

	// Reason is the guardrail's reason, or the failure of a failed check.
	Reason string `json:"reason"`

	// Notice is the text of the block's user notice, if any.
	Notice string `json:"notice,omitempty"`

	// Turn is the model turn of the content; 0 for the input stage.
	Turn int `json:"turn"`

	// Index is the tool call's index within its reply for the tool stages; 0
	// otherwise.
	Index int `json:"index"`

	// Failed reports that the check itself failed and the content was blocked
	// because of that failure.
	Failed bool `json:"failed,omitempty"`
}

// ErrTripwire is a sentinel target for errors.Is to match any [TripwireError].
var ErrTripwire = &TripwireError{}

func (e *TripwireError) Error() string {
	return fmt.Sprintf("%s guardrail %q tripped: %s", e.Stage, e.Guardrail, e.Reason)
}

// Is reports whether target matches this error. It matches any *TripwireError
// (including [ErrTripwire]) whose non-empty Stage, Guardrail and Reason equal
// this error's. Notice, Turn, Index and Failed take no part in the match; use
// [AsTripwire] to check them.
func (e *TripwireError) Is(target error) bool {
	t, ok := target.(*TripwireError)
	if !ok {
		return false
	}
	return (t.Stage == "" || t.Stage == e.Stage) &&
		(t.Guardrail == "" || t.Guardrail == e.Guardrail) &&
		(t.Reason == "" || t.Reason == e.Reason)
}

// NewTripwireError returns te as a non-retryable application error of type
// [ErrorTypeTripwire] with te as its cause and details. A tool whose Invoke
// returns it fails the call fatally; the run ends with the round's first fatal
// error in call order once every call of the round has finished.
func NewTripwireError(te TripwireError) error {
	return temporal.NewNonRetryableApplicationError(te.Error(), ErrorTypeTripwire, &te, te)
}

// TripwireFromRecord returns the [TripwireError] of r, a block or failed
// check, such as the [guardrail.Outcome.Blocked] of a [guardrail.Screen]: its
// stage, guardrail, reason, notice text, turn, index and Failed.
func TripwireFromRecord(r guardrail.Record) TripwireError {
	te := TripwireError{
		Stage:     r.Stage,
		Guardrail: r.Guardrail,
		Reason:    r.Reason,
		Turn:      r.Turn,
		Index:     r.Index,
		Failed:    r.Failed,
	}
	if r.Notice != nil {
		te.Notice = r.Notice.Text
	}
	return te
}

// AsTripwire reports whether err is a guardrail tripwire and returns it.
func AsTripwire(err error) (*TripwireError, bool) {
	var te *TripwireError
	if errors.As(err, &te) {
		return te, true
	}
	var appErr *temporal.ApplicationError
	if errors.As(err, &appErr) && appErr.Type() == ErrorTypeTripwire {
		var detail TripwireError
		if appErr.HasDetails() && appErr.Details(&detail) == nil {
			return &detail, true
		}
		return &TripwireError{Reason: appErr.Message()}, true
	}
	return nil, false
}

const (
	// toolResultRefusal is the message for a guardrail that declares the
	// tool_result stage.
	toolResultRefusal = "the agent loop does not run the tool_result stage; hosts screen tool results in their tools with guardrail.Screen"

	// continuationRefusal is the message for output guardrails on an agent
	// built with WithContinueOnLength.
	continuationRefusal = "output guardrails cannot be combined with WithContinueOnLength"
)

// checkGuardrails returns a non-retryable [ErrorTypeConfig] error when gs, the
// guardrails of kind ("guardrail" or "run guardrail") of agent a, holds a nil
// guardrail; a guardrail with an empty name, no stages or a stage that is not
// one of the five; a guardrail that declares the tool_result stage; or, for an
// agent built with [WithContinueOnLength], a guardrail that declares the
// output stage.
func checkGuardrails(a *Agent, kind string, gs []guardrail.Guardrail) error {
	for i, g := range gs {
		if g == nil {
			return configError("agent %q: %s %d is nil", a.name, kind, i)
		}
		name := g.Name()
		if name == "" {
			return configError("agent %q: %s %d has no name", a.name, kind, i)
		}
		stages := g.Stages()
		if len(stages) == 0 {
			return configError("agent %q: %s %q declares no stages", a.name, kind, name)
		}
		for _, stage := range stages {
			if !guardrail.ValidStage(stage) {
				return configError("agent %q: %s %q declares unknown stage %q", a.name, kind, name, stage)
			}
		}
		if slices.Contains(stages, guardrail.StageToolResult) {
			return configError("agent %q: %s %q: %s", a.name, kind, name, toolResultRefusal)
		}
		if a.maxContinuations > 0 && slices.Contains(stages, guardrail.StageOutput) {
			return configError("agent %q: %s %q: %s", a.name, kind, name, continuationRefusal)
		}
	}
	return nil
}

// screen runs [runState.screenOutcome] on in with attempt. When the outcome is blocked
// it returns the outcome and the [NewTripwireError] of the blocking record.
func (r *runState) screen(ctx workflow.Context, in guardrail.CheckInput, attempt int32) (guardrail.Outcome, error) {
	out, err := r.screenOutcome(ctx, in, attempt)
	if err != nil {
		return out, err
	}
	if b := out.Blocked; b != nil {
		return out, NewTripwireError(TripwireFromRecord(*b))
	}
	return out, nil
}

// screenOutcome runs [guardrail.Screen] on in over the agent's and the run's
// guardrails and invokes the OnGuardrail hook with attempt for each record. A
// blocked outcome is returned without an error.
func (r *runState) screenOutcome(ctx workflow.Context, in guardrail.CheckInput, attempt int32) (guardrail.Outcome, error) {
	out, err := guardrail.Screen(ctx, r.guards, in)
	if err != nil {
		return out, err
	}
	for _, rec := range out.Records {
		if err := r.ro.emitGuardrail(ctx, rec, attempt); err != nil {
			return out, err
		}
	}
	return out, nil
}

// blockedCallResult returns the error result of a call to the tool name that
// the tool_call stage blocked.
func blockedCallResult(name string) *model.CallToolResult {
	return model.ErrorResult(fmt.Sprintf("Error: a guardrail blocked this call to %q.", name))
}
