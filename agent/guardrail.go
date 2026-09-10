package agent

import (
	"errors"
	"fmt"

	"go.temporal.io/sdk/temporal"
	"go.temporal.io/sdk/workflow"

	"github.com/wleev/temporal-agent-sdk/guardrail"
)

// Stage identifies which side of the agent a guardrail screens.
type Stage string

const (
	// StageInput guardrails screen the user input before the first model call.
	StageInput Stage = "input"
	// StageOutput guardrails screen the agent's final answer before it is returned.
	StageOutput Stage = "output"
)

// TripwireError reports that a guardrail blocked the run.
//
// It is a permanent outcome for the given text: the same input trips the same
// way, so retrying does not help. Recover it with [AsTripwire] to answer a
// blocked run with a safe reply rather than failing the workflow. If instead you
// let it propagate out of a workflow that has a retry policy, wrap it with
// temporal.NewNonRetryableApplicationError so the workflow is not retried.
type TripwireError struct {
	Stage     Stage
	Guardrail string
	Reason    string
}

// ErrTripwire is a sentinel target for errors.Is to match any [TripwireError].
var ErrTripwire = &TripwireError{}

func (e *TripwireError) Error() string {
	return fmt.Sprintf("%s guardrail %q tripped: %s", e.Stage, e.Guardrail, e.Reason)
}

// Is reports whether target matches this error. It matches any *TripwireError
// (including [ErrTripwire]), and if target specifies Stage, Guardrail, or Reason,
// those non-empty fields must also match.
func (e *TripwireError) Is(target error) bool {
	t, ok := target.(*TripwireError)
	if !ok {
		return false
	}
	return (t.Stage == "" || t.Stage == e.Stage) &&
		(t.Guardrail == "" || t.Guardrail == e.Guardrail) &&
		(t.Reason == "" || t.Reason == e.Reason)
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

// runGuardrails runs every guardrail at a stage concurrently, then evaluates the
// verdicts in declared order and returns the first tripwire as a [TripwireError].
//
// Verdicts are written by index and evaluated in declared order, so the result
// does not depend on which check finished first. A guardrail's own error (a
// failed check, not a tripwire) is fatal and aborts the run.
func (s *Session) runGuardrails(ctx workflow.Context, stage Stage, guards []guardrail.Guardrail, text string) error {
	if len(guards) == 0 {
		return nil
	}

	type outcome struct {
		result guardrail.Result
		err    error
	}
	outcomes := make([]outcome, len(guards))
	wg := workflow.NewWaitGroup(ctx)
	for i, g := range guards {
		wg.Add(1)
		workflow.Go(ctx, func(gctx workflow.Context) {
			defer wg.Done()
			r, err := g.Check(gctx, text)
			outcomes[i] = outcome{result: r, err: err}
		})
	}
	wg.Wait(ctx)

	for i, o := range outcomes {
		if o.err != nil {
			return o.err
		}
		if o.result.Tripwire {
			te := &TripwireError{Stage: stage, Guardrail: guards[i].Name(), Reason: o.result.Reason}
			return temporal.NewNonRetryableApplicationError(te.Error(), ErrorTypeTripwire, te, te)
		}
	}
	return nil
}
