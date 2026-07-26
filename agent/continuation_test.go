package agent_test

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.temporal.io/sdk/workflow"

	"github.com/wleev/temporal-agent-sdk/agent"
	"github.com/wleev/temporal-agent-sdk/agenttest"
	"github.com/wleev/temporal-agent-sdk/model"
)

// A truncated answer is continued up to the budget, and Output is the fragments
// concatenated in order. Each continuation is a turn and fires OnTurn.
func TestRun_ContinuesOnLength(t *testing.T) {
	fake := agenttest.NewFakeProvider(
		agenttest.Truncated("The answer "), // turn 1: length
		agenttest.Truncated("is made of "), // turn 2: length
		agenttest.Says("three parts."),     // turn 3: stop
	)
	env := newEnv(t, fake)

	a, err := agent.NewAgent("assistant", "test-model", agent.WithContinueOnLength(5))
	require.NoError(t, err)

	var events []agent.TurnEvent
	var res *agent.Result
	env.ExecuteWorkflow(func(ctx workflow.Context) error {
		var runErr error
		res, runErr = agent.RunWith(ctx, a, "Tell me.", agent.RunOptions{
			OnTurn: func(_ workflow.Context, e agent.TurnEvent) error {
				events = append(events, e)
				return nil
			},
		})
		return runErr
	})
	require.True(t, env.IsWorkflowCompleted())
	require.NoError(t, env.GetWorkflowError())

	require.NotNil(t, res)
	assert.Equal(t, "The answer is made of three parts.", res.Output, "fragments concatenate in order")
	assert.Equal(t, 3, res.Turns, "each continuation consumes a turn")
	assert.Equal(t, model.FinishStop, res.FinishReason, "the run concluded normally")
	assert.Equal(t, 3, fake.CallCount())

	// Every turn — the two continuations and the final answer — fired OnTurn.
	if assert.Len(t, events, 3, "one hook call per turn, continuations included") {
		assert.Equal(t, 1, events[0].Turn)
		assert.Equal(t, "The answer ", events[0].Assistant.Text())
		assert.Equal(t, 2, events[1].Turn)
		assert.Equal(t, "is made of ", events[1].Assistant.Text())
		assert.Equal(t, 3, events[2].Turn)
		assert.Equal(t, "three parts.", events[2].Assistant.Text())
	}

	// Each continuation re-sends the growing transcript, so the model sees its own
	// partial answer as history.
	last := fake.LastCall()
	assert.Equal(t, model.RoleAssistant, last.Messages[len(last.Messages)-1].Role,
		"the final call's history ends with the prior partial answer")
}

// Exhausting the continuation budget is not an error: the accumulated output is
// returned with FinishReason length.
func TestRun_ContinuationBudgetExhausted(t *testing.T) {
	fake := agenttest.NewFakeProvider(
		agenttest.Truncated("part one "), // turn 1: length
		agenttest.Truncated("part two "), // turn 2 (continuation 1): still length, budget spent
	)
	env := newEnv(t, fake)

	a, err := agent.NewAgent("assistant", "test-model", agent.WithContinueOnLength(1))
	require.NoError(t, err)

	var res *agent.Result
	var runErr error
	env.ExecuteWorkflow(func(ctx workflow.Context) error {
		res, runErr = agent.Run(ctx, a, "Tell me.")
		return runErr
	})
	require.True(t, env.IsWorkflowCompleted())

	require.NoError(t, env.GetWorkflowError(), "a spent budget is not an error")
	require.NotNil(t, res)
	assert.Equal(t, "part one part two ", res.Output, "what was accumulated is returned")
	assert.Equal(t, model.FinishLength, res.FinishReason, "the run is still truncated")
	assert.Equal(t, 2, res.Turns)
	assert.Equal(t, 2, fake.CallCount(), "one initial call plus one continuation")
}

// Without opting in, a length response ends the run as-is: no continuation, no
// error.
func TestRun_NoContinuationByDefault(t *testing.T) {
	fake := agenttest.NewFakeProvider(agenttest.Truncated("partial answer"))
	env := newEnv(t, fake)

	a, err := agent.NewAgent("assistant", "test-model")
	require.NoError(t, err)

	var res *agent.Result
	env.ExecuteWorkflow(func(ctx workflow.Context) error {
		var runErr error
		res, runErr = agent.Run(ctx, a, "Tell me.")
		return runErr
	})
	require.True(t, env.IsWorkflowCompleted())
	require.NoError(t, env.GetWorkflowError())

	require.NotNil(t, res)
	assert.Equal(t, "partial answer", res.Output)
	assert.Equal(t, 1, res.Turns, "no continuation without opting in")
	assert.Equal(t, model.FinishLength, res.FinishReason)
	assert.Equal(t, 1, fake.CallCount())
}

// A tool round between fragments ends the current answer: only consecutive
// continuations concatenate, so the pre-tool fragment does not bleed into the
// final answer.
func TestRun_ContinuationResetsAcrossToolRound(t *testing.T) {
	fake := agenttest.NewFakeProvider(
		agenttest.Truncated("stale fragment "),                 // turn 1: length, continues
		agenttest.CallsTool("get_weather", `{"city":"Ghent"}`), // turn 2: tool call resets the accumulator
		agenttest.Says("It is 18°C in Ghent."),                 // turn 3: the real answer
	)
	env := newEnv(t, fake)

	a, err := agent.NewAgent("assistant", "test-model",
		agent.WithTools(weatherTool(t)), agent.WithContinueOnLength(3))
	require.NoError(t, err)

	var res *agent.Result
	env.ExecuteWorkflow(func(ctx workflow.Context) error {
		var runErr error
		res, runErr = agent.Run(ctx, a, "Weather?")
		return runErr
	})
	require.True(t, env.IsWorkflowCompleted())
	require.NoError(t, env.GetWorkflowError())

	require.NotNil(t, res)
	assert.Equal(t, "It is 18°C in Ghent.", res.Output, "the stale fragment is dropped at the tool round")
	assert.Equal(t, 3, res.Turns)
}

// When continuation exhausts the turn budget, the MaxTurns error still carries
// the text accumulated so far.
func TestRun_ContinuationHitsMaxTurns(t *testing.T) {
	fake := agenttest.NewFakeProvider(
		agenttest.Truncated("part one "), // turn 1: length
		agenttest.Truncated("part two "), // turn 2: length, then maxTurns=2 is hit
	)
	env := newEnv(t, fake)

	a, err := agent.NewAgent("assistant", "test-model",
		agent.WithContinueOnLength(5), agent.WithMaxTurns(2))
	require.NoError(t, err)

	var res *agent.Result
	var runErr error
	env.ExecuteWorkflow(func(ctx workflow.Context) error {
		res, runErr = agent.Run(ctx, a, "Tell me.")
		return runErr
	})
	require.True(t, env.IsWorkflowCompleted())

	require.Error(t, runErr, "running out of turns is an error")
	if assert.NotNil(t, res) {
		assert.Equal(t, "part one part two ", res.Output, "the accumulated answer survives the turn-budget error")
		assert.Equal(t, 2, res.Turns)
	}
}

// FinishReason is surfaced on Result for an ordinary, non-truncated run.
func TestRun_SurfacesFinishReason(t *testing.T) {
	fake := agenttest.NewFakeProvider(agenttest.Says("done."))
	env := newEnv(t, fake)

	a, err := agent.NewAgent("assistant", "test-model")
	require.NoError(t, err)

	var res *agent.Result
	env.ExecuteWorkflow(func(ctx workflow.Context) error {
		var runErr error
		res, runErr = agent.Run(ctx, a, "Hi")
		return runErr
	})
	require.True(t, env.IsWorkflowCompleted())
	require.NoError(t, env.GetWorkflowError())
	require.NotNil(t, res)
	assert.Equal(t, model.FinishStop, res.FinishReason)
}
