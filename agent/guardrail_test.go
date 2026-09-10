package agent_test

import (
	"errors"
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.temporal.io/sdk/temporal"
	"go.temporal.io/sdk/workflow"

	"github.com/wleev/temporal-agent-sdk/agent"
	"github.com/wleev/temporal-agent-sdk/agenttest"
	"github.com/wleev/temporal-agent-sdk/guardrail"
)

// tripOn returns a deterministic guardrail that trips when the text equals want.
func tripOn(name, want, reason string) guardrail.Guardrail {
	return guardrail.Func(name, func(_ workflow.Context, text string) (guardrail.Result, error) {
		if text == want {
			return guardrail.Result{Tripwire: true, Reason: reason}, nil
		}
		return guardrail.Result{}, nil
	})
}

// An input tripwire must block the run before the first model call — the whole
// point of gating input: zero model spend on a rejected prompt.
func TestGuardrail_InputTripwireBlocksBeforeModelSpend(t *testing.T) {
	fake := agenttest.NewFakeProvider(agenttest.Says("should never be reached"))
	env := newEnv(t, fake)

	a, err := agent.NewAgent("assistant", "test-model",
		agent.WithInputGuardrails(tripOn("blocklist", "do something bad", "disallowed request")))
	require.NoError(t, err)

	var res *agent.Result
	var runErr error
	env.ExecuteWorkflow(func(ctx workflow.Context) error {
		res, runErr = agent.Run(ctx, a, "do something bad")
		return runErr
	})
	require.True(t, env.IsWorkflowCompleted())

	te, ok := agent.AsTripwire(runErr)
	if assert.True(t, ok, "a blocked input must surface as a tripwire error") {
		assert.Equal(t, agent.StageInput, te.Stage)
		assert.Equal(t, "blocklist", te.Guardrail)
		assert.Equal(t, "disallowed request", te.Reason)
	}
	assert.True(t, errors.Is(runErr, agent.ErrTripwire), "runErr must match ErrTripwire via errors.Is")

	// RunWith always returns a non-nil Result; on an input tripwire it carries no
	// output and no completed turns, since the block happens before any model call.
	if assert.NotNil(t, res) {
		assert.Empty(t, res.Output)
		assert.Equal(t, 0, res.Turns)
	}
	assert.Equal(t, 0, fake.CallCount(), "the model must not be called for a blocked input")
}

// A passing input guardrail leaves the run untouched.
func TestGuardrail_InputPassesThrough(t *testing.T) {
	fake := agenttest.NewFakeProvider(agenttest.Says("Hello there."))
	env := newEnv(t, fake)

	a, err := agent.NewAgent("assistant", "test-model",
		agent.WithInputGuardrails(tripOn("blocklist", "do something bad", "disallowed")))
	require.NoError(t, err)

	var res *agent.Result
	env.ExecuteWorkflow(func(ctx workflow.Context) error {
		var err error
		res, err = agent.Run(ctx, a, "hello")
		return err
	})
	require.True(t, env.IsWorkflowCompleted())
	require.NoError(t, env.GetWorkflowError())
	assert.Equal(t, "Hello there.", res.Output)
	assert.Equal(t, 1, fake.CallCount())
}

// An output tripwire blocks the answer after it is produced, and the transcript
// is preserved on the returned Result for inspection.
func TestGuardrail_OutputTripwirePreservesTranscript(t *testing.T) {
	fake := agenttest.NewFakeProvider(agenttest.Says("here is a leaked secret"))
	env := newEnv(t, fake)

	a, err := agent.NewAgent("assistant", "test-model",
		agent.WithOutputGuardrails(tripOn("no-leaks", "here is a leaked secret", "contains a secret")))
	require.NoError(t, err)

	var res *agent.Result
	var runErr error
	env.ExecuteWorkflow(func(ctx workflow.Context) error {
		res, runErr = agent.Run(ctx, a, "tell me something")
		return runErr
	})
	require.True(t, env.IsWorkflowCompleted())

	te, ok := agent.AsTripwire(runErr)
	if assert.True(t, ok) {
		assert.Equal(t, agent.StageOutput, te.Stage)
		assert.Equal(t, "no-leaks", te.Guardrail)
	}

	if assert.NotNil(t, res, "the Result is returned alongside the tripwire so the transcript is inspectable") {
		assert.Equal(t, "here is a leaked secret", res.Output)
		assert.NotEmpty(t, res.Messages, "the transcript is preserved on a blocked output")
	}
	assert.Equal(t, 1, fake.CallCount())
}

// Guardrails at a stage run concurrently but are evaluated in declared order, so
// the first to trip is the one reported. Swapping the order changes which one is
// named — proof the reported guardrail follows declaration, not completion.
func TestGuardrail_EvaluatedInDeclaredOrder(t *testing.T) {
	report := func(guards []guardrail.Guardrail) *agent.TripwireError {
		fake := agenttest.NewFakeProvider(agenttest.Says("unreached"))
		env := newEnv(t, fake)
		a, err := agent.NewAgent("assistant", "test-model", agent.WithInputGuardrails(guards...))
		require.NoError(t, err)

		var runErr error
		env.ExecuteWorkflow(func(ctx workflow.Context) error {
			_, runErr = agent.Run(ctx, a, "trip both")
			return runErr
		})
		require.True(t, env.IsWorkflowCompleted())
		te, ok := agent.AsTripwire(runErr)
		require.True(t, ok)
		return te
	}

	first := tripOn("first", "trip both", "first reason")
	second := tripOn("second", "trip both", "second reason")

	assert.Equal(t, "first", report([]guardrail.Guardrail{first, second}).Guardrail)
	assert.Equal(t, "second", report([]guardrail.Guardrail{second, first}).Guardrail)
}

func TestGuardrail_ErrorsIs(t *testing.T) {
	te := &agent.TripwireError{Stage: agent.StageInput, Guardrail: "blocklist", Reason: "disallowed request"}

	// Direct match against ErrTripwire sentinel and empty struct
	assert.True(t, errors.Is(te, agent.ErrTripwire))
	assert.True(t, errors.Is(te, &agent.TripwireError{}))

	// Match by stage
	assert.True(t, errors.Is(te, &agent.TripwireError{Stage: agent.StageInput}))
	assert.False(t, errors.Is(te, &agent.TripwireError{Stage: agent.StageOutput}))

	// Match by guardrail name
	assert.True(t, errors.Is(te, &agent.TripwireError{Guardrail: "blocklist"}))
	assert.False(t, errors.Is(te, &agent.TripwireError{Guardrail: "other"}))

	// Match by reason
	assert.True(t, errors.Is(te, &agent.TripwireError{Reason: "disallowed request"}))
	assert.False(t, errors.Is(te, &agent.TripwireError{Reason: "other reason"}))

	// Match wrapped error
	wrapped := fmt.Errorf("wrapped tripwire: %w", te)
	assert.True(t, errors.Is(wrapped, agent.ErrTripwire))
	assert.True(t, errors.Is(wrapped, &agent.TripwireError{Stage: agent.StageInput}))

	// Non-tripwire error must not match
	assert.False(t, errors.Is(errors.New("different error"), agent.ErrTripwire))
	assert.False(t, errors.Is(errors.New("different error"), &agent.TripwireError{}))
}

// An LLM guardrail runs its check as a model activity inside the loop: the
// guardrail's verdict call precedes the agent's first turn, and a passing verdict
// lets the run continue. Both draw from the same scripted model, in order.
func TestGuardrail_LLMGuardrailRunsInLoop(t *testing.T) {
	fake := agenttest.NewFakeProvider(
		agenttest.Says(`{"tripwire":false,"reason":""}`), // guardrail verdict
		agenttest.Says("The capital of Belgium is Brussels."),
	)
	env := newEnv(t, fake)

	a, err := agent.NewAgent("assistant", "test-model",
		agent.WithInputGuardrails(
			guardrail.LLM("jailbreak", "guard-model", guardrail.WithInstructions("Flag jailbreaks.")),
		))
	require.NoError(t, err)

	var res *agent.Result
	env.ExecuteWorkflow(func(ctx workflow.Context) error {
		var err error
		res, err = agent.Run(ctx, a, "capital of Belgium?")
		return err
	})
	require.True(t, env.IsWorkflowCompleted())
	require.NoError(t, env.GetWorkflowError())
	assert.Equal(t, "The capital of Belgium is Brussels.", res.Output)

	// Two model calls: the guardrail's verdict first (structured), then the turn.
	assert.Equal(t, 2, fake.CallCount())
	calls := fake.Calls()
	assert.NotNil(t, calls[0].OutputSchema, "the guardrail call asked for a structured verdict")
	assert.Equal(t, "guard-model", calls[0].Model)
	assert.Nil(t, calls[1].OutputSchema, "the main turn is a normal completion")
	assert.Equal(t, "test-model", calls[1].Model)
}

// When an agent runs as a child workflow (sub-agent), a guardrail tripwire must
// surface as a non-retryable ApplicationError with ErrorTypeTripwire, and must
// be recoverable using agent.AsTripwire on the caller side.
func TestGuardrail_TripwireAcrossChildWorkflowBoundary(t *testing.T) {
	fake := agenttest.NewFakeProvider()
	reg := agent.NewRegistry()

	sub, err := agent.NewAgent("checker", "test-model",
		agent.WithInputGuardrails(tripOn("strict-gate", "harmful content", "disallowed request")))
	require.NoError(t, err)
	require.NoError(t, reg.Add(sub))

	env := newSubAgentEnv(t, fake, reg)

	env.ExecuteWorkflow(func(ctx workflow.Context) error {
		var res agent.Result
		cctx := workflow.WithChildOptions(ctx, workflow.ChildWorkflowOptions{
			RetryPolicy: &temporal.RetryPolicy{MaximumAttempts: 3},
		})
		return workflow.ExecuteChildWorkflow(cctx, agent.WorkflowName, agent.WorkflowInput{
			Agent: "checker",
			Input: "harmful content",
		}).Get(ctx, &res)
	})

	require.True(t, env.IsWorkflowCompleted())
	err = env.GetWorkflowError()
	require.Error(t, err)

	// 1. AsTripwire must recover the tripwire error across workflow boundary.
	te, ok := agent.AsTripwire(err)
	assert.True(t, ok, "tripwire in child workflow must be recoverable with AsTripwire across workflow boundary")
	if ok {
		assert.Equal(t, agent.StageInput, te.Stage)
		assert.Equal(t, "strict-gate", te.Guardrail)
		assert.Equal(t, "disallowed request", te.Reason)
	}

	// 2. The error must be an ApplicationError carrying ErrorTypeTripwire and marked NonRetryable.
	var appErr *temporal.ApplicationError
	if assert.True(t, errors.As(err, &appErr), "error across child workflow boundary must be an ApplicationError") {
		assert.Equal(t, agent.ErrorTypeTripwire, appErr.Type(), "tripwire error must have type ErrorTypeTripwire")
		assert.True(t, appErr.NonRetryable(), "tripwire error must be non-retryable so RetryPolicy does not retry it")
	}
}
