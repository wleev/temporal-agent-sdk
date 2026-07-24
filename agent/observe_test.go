package agent_test

import (
	"errors"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.temporal.io/sdk/temporal"
	"go.temporal.io/sdk/workflow"

	"github.com/wleev/temporal-agent-sdk/agent"
	"github.com/wleev/temporal-agent-sdk/agenttest"
)

// OnTurn fires once per turn, in order, with turn numbers matching Result.Turns,
// including the final tool-less turn.
func TestRunWith_OnTurnFiresPerTurn(t *testing.T) {
	fake := agenttest.NewFakeProvider(
		agenttest.CallsTool("get_weather", `{"city":"Ghent"}`), // turn 1: a tool call
		agenttest.Says("It is 18°C in Ghent."),                 // turn 2: the final answer
	)
	env := newEnv(t, fake)

	a, err := agent.NewAgent("assistant", "test-model", agent.WithTools(weatherTool(t)))
	require.NoError(t, err)

	var events []agent.TurnEvent
	var res *agent.Result
	env.ExecuteWorkflow(func(ctx workflow.Context) error {
		var runErr error
		res, runErr = agent.RunWith(ctx, a, "Weather in Ghent?", agent.RunOptions{
			OnTurn: func(_ workflow.Context, e agent.TurnEvent) error {
				events = append(events, e)
				return nil
			},
		})
		return runErr
	})
	require.True(t, env.IsWorkflowCompleted())
	require.NoError(t, env.GetWorkflowError())

	if assert.Len(t, events, 2, "one hook call per turn") {
		assert.Equal(t, 1, events[0].Turn)
		assert.Len(t, events[0].Assistant.ToolCalls(), 1, "turn 1 called a tool")
		assert.Len(t, events[0].ToolResults, 1, "turn 1 carries its tool result")

		assert.Equal(t, 2, events[1].Turn)
		assert.Equal(t, "It is 18°C in Ghent.", events[1].Assistant.Text())
		assert.Empty(t, events[1].ToolResults, "the final turn has no tool results")
	}
	require.NotNil(t, res)
	assert.Equal(t, res.Turns, len(events), "hook count matches the run's turn count")
}

// ProviderMetadata reaches the provider on every model request unchanged, so a
// routing provider can resolve on it without overloading the model name.
func TestRunWith_ProviderMetadataReachesProvider(t *testing.T) {
	fake := agenttest.NewFakeProvider(
		agenttest.CallsTool("get_weather", `{"city":"Ghent"}`),
		agenttest.Says("done"),
	)
	env := newEnv(t, fake)

	a, err := agent.NewAgent("assistant", "test-model", agent.WithTools(weatherTool(t)))
	require.NoError(t, err)

	md := map[string]string{"tenant": "acme", "feature": "scheduled_task"}
	env.ExecuteWorkflow(func(ctx workflow.Context) error {
		_, runErr := agent.RunWith(ctx, a, "Weather?", agent.RunOptions{ProviderMetadata: md})
		return runErr
	})
	require.True(t, env.IsWorkflowCompleted())
	require.NoError(t, env.GetWorkflowError())

	// Both turns carry the same metadata.
	calls := fake.Calls()
	if assert.Len(t, calls, 2) {
		assert.Equal(t, md, calls[0].Metadata)
		assert.Equal(t, md, calls[1].Metadata)
	}
}

// An error from OnTurn aborts the run and surfaces with the partial Result.
func TestRunWith_OnTurnErrorAbortsWithPartialResult(t *testing.T) {
	fake := agenttest.NewFakeProvider(
		agenttest.CallsTool("get_weather", `{"city":"Ghent"}`),
		agenttest.Says("unreached"),
	)
	env := newEnv(t, fake)

	a, err := agent.NewAgent("assistant", "test-model", agent.WithTools(weatherTool(t)))
	require.NoError(t, err)

	boom := errors.New("persisting the turn failed")
	var res *agent.Result
	var runErr error
	env.ExecuteWorkflow(func(ctx workflow.Context) error {
		res, runErr = agent.RunWith(ctx, a, "Weather?", agent.RunOptions{
			OnTurn: func(_ workflow.Context, _ agent.TurnEvent) error { return boom },
		})
		return runErr
	})
	require.True(t, env.IsWorkflowCompleted())

	require.Error(t, runErr)
	assert.ErrorContains(t, runErr, "persisting the turn failed")
	if assert.NotNil(t, res) {
		assert.Equal(t, 1, res.Turns, "the first turn ran before the hook aborted")
		assert.NotEmpty(t, res.Messages, "the aborted run's transcript is preserved")
	}
	assert.Equal(t, 1, fake.CallCount(), "the run aborted after the first turn")
}

// A model-call failure returns the partial transcript accumulated so far rather
// than a nil Result.
func TestRunWith_PartialResultOnModelFailure(t *testing.T) {
	fake := agenttest.NewFakeProvider(
		agenttest.CallsTool("get_weather", `{"city":"Ghent"}`), // turn 1 succeeds
		agenttest.Fails(errors.New("model endpoint exploded")), // turn 2 fails
	)
	env := newEnv(t, fake)

	// Cap model retries at 1 so the second turn fails immediately.
	a, err := agent.NewAgent("assistant", "test-model",
		agent.WithTools(weatherTool(t)),
		agent.WithModelActivityOptions(workflow.ActivityOptions{
			StartToCloseTimeout: time.Minute,
			RetryPolicy:         &temporal.RetryPolicy{MaximumAttempts: 1},
		}))
	require.NoError(t, err)

	var res *agent.Result
	var runErr error
	env.ExecuteWorkflow(func(ctx workflow.Context) error {
		res, runErr = agent.Run(ctx, a, "Weather?")
		return runErr
	})
	require.True(t, env.IsWorkflowCompleted())

	require.Error(t, runErr)
	if assert.NotNil(t, res, "a model failure still returns the partial transcript") {
		assert.Equal(t, 1, res.Turns, "turn 1 completed before turn 2 failed")
		assert.NotEmpty(t, res.Messages, "turn 1's assistant call and tool result are preserved")
		assert.Empty(t, res.Output, "there is no final answer on a failed run")
	}
}
