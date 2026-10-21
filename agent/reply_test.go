package agent_test

import (
	"context"
	"errors"
	"fmt"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.temporal.io/sdk/activity"
	"go.temporal.io/sdk/client"
	"go.temporal.io/sdk/worker"
	"go.temporal.io/sdk/workflow"

	"github.com/wleev/temporal-agent-sdk/agent"
	"github.com/wleev/temporal-agent-sdk/agenttest"
	"github.com/wleev/temporal-agent-sdk/model"
	"github.com/wleev/temporal-agent-sdk/tool"
)

// markerWeatherTool returns a get_weather tool that appends "tool" to seq each
// time it runs.
func markerWeatherTool(t *testing.T, seq *[]string) tool.Tool {
	t.Helper()
	tl, err := tool.New("get_weather", "Look up the weather",
		func(_ workflow.Context, in weatherIn) (string, error) {
			*seq = append(*seq, "tool")
			return "18°C in " + in.City, nil
		})
	require.NoError(t, err)
	return tl
}

// markerOptions returns RunOptions whose OnReply appends "reply:<turn>" and
// whose OnTurn appends "turn:<turn>" to seq.
func markerOptions(seq *[]string) agent.RunOptions {
	return agent.RunOptions{
		OnReply: func(_ workflow.Context, e agent.ReplyEvent) error {
			*seq = append(*seq, fmt.Sprintf("reply:%d", e.Turn))
			return nil
		},
		OnTurn: func(_ workflow.Context, e agent.TurnEvent) error {
			*seq = append(*seq, fmt.Sprintf("turn:%d", e.Turn))
			return nil
		},
	}
}

// TestRunWith_OnReplyFiresBeforeDispatchAndOnTurn verifies that OnReply fires
// for each reply before that reply's tools run and before the turn's OnTurn.
func TestRunWith_OnReplyFiresBeforeDispatchAndOnTurn(t *testing.T) {
	fake := agenttest.NewFakeProvider(
		agenttest.CallsTools(
			agenttest.ToolCall{ID: "x", Name: "get_weather", Args: `{"city":"Ghent"}`},
			agenttest.ToolCall{ID: "y", Name: "get_weather", Args: `{"city":"Bruges"}`},
		),
		agenttest.Says("Mild in both."),
	)
	env := newEnv(t, fake)

	var seq []string
	a, err := agent.NewAgent("assistant", "test-model", agent.WithTools(markerWeatherTool(t, &seq)))
	require.NoError(t, err)

	env.ExecuteWorkflow(func(ctx workflow.Context) error {
		_, runErr := agent.RunWith(ctx, a, "Weather?", markerOptions(&seq))
		return runErr
	})
	require.True(t, env.IsWorkflowCompleted())
	assert.NoError(t, env.GetWorkflowError())

	assert.Equal(t, []string{"reply:1", "tool", "tool", "turn:1", "reply:2", "turn:2"}, seq)
}

// TestRunWith_OnReplyCarriesReply verifies that OnReply receives the turn, the
// reply, and the reply's usage, for a tool call and for the final answer.
func TestRunWith_OnReplyCarriesReply(t *testing.T) {
	fake := agenttest.NewFakeProvider(
		agenttest.CallsTool("get_weather", `{"city":"Ghent"}`),
		agenttest.Says("It is 18°C in Ghent."),
	)
	env := newEnv(t, fake)

	a, err := agent.NewAgent("assistant", "test-model", agent.WithTools(weatherTool(t)))
	require.NoError(t, err)

	var events []agent.ReplyEvent
	env.ExecuteWorkflow(func(ctx workflow.Context) error {
		_, runErr := agent.RunWith(ctx, a, "Weather in Ghent?", agent.RunOptions{
			OnReply: func(_ workflow.Context, e agent.ReplyEvent) error {
				events = append(events, e)
				return nil
			},
		})
		return runErr
	})
	require.True(t, env.IsWorkflowCompleted())
	assert.NoError(t, env.GetWorkflowError())

	usage := model.Usage{PromptTokens: 10, CompletionTokens: 5, TotalTokens: 15}
	if assert.Len(t, events, 2) {
		assert.Equal(t, 1, events[0].Turn)
		if assert.Len(t, events[0].Assistant.ToolCalls(), 1) {
			assert.Equal(t, "get_weather", events[0].Assistant.ToolCalls()[0].Name)
		}
		assert.Equal(t, usage, events[0].Usage)

		assert.Equal(t, 2, events[1].Turn)
		assert.Equal(t, "It is 18°C in Ghent.", events[1].Assistant.Text())
		assert.Empty(t, events[1].Assistant.ToolCalls())
		assert.Equal(t, usage, events[1].Usage)
	}
}

// TestRunWith_OnReplyFiresForContinuations verifies that OnReply fires for
// each length-continuation fragment.
func TestRunWith_OnReplyFiresForContinuations(t *testing.T) {
	fake := agenttest.NewFakeProvider(
		agenttest.Truncated("Hel"),
		agenttest.Says("lo"),
	)
	env := newEnv(t, fake)

	a, err := agent.NewAgent("assistant", "test-model", agent.WithContinueOnLength(1))
	require.NoError(t, err)

	var seq []string
	var res *agent.Result
	env.ExecuteWorkflow(func(ctx workflow.Context) error {
		var runErr error
		res, runErr = agent.RunWith(ctx, a, "Greet.", markerOptions(&seq))
		return runErr
	})
	require.True(t, env.IsWorkflowCompleted())
	assert.NoError(t, env.GetWorkflowError())

	assert.Equal(t, []string{"reply:1", "turn:1", "reply:2", "turn:2"}, seq)
	if assert.NotNil(t, res) {
		assert.Equal(t, "Hello", res.Output)
	}
}

// TestRunWith_OnReplyErrorAbortsBeforeDispatch verifies that an OnReply error
// ends the run before the reply's tools run and that the partial Result ends
// with the reply.
func TestRunWith_OnReplyErrorAbortsBeforeDispatch(t *testing.T) {
	fake := agenttest.NewFakeProvider(
		agenttest.CallsTool("get_weather", `{"city":"Ghent"}`),
		agenttest.Says("unreached"),
	)
	env := newEnv(t, fake)

	var seq []string
	a, err := agent.NewAgent("assistant", "test-model", agent.WithTools(markerWeatherTool(t, &seq)))
	require.NoError(t, err)

	boom := errors.New("saving the reply failed")
	ro := markerOptions(&seq)
	ro.OnReply = func(_ workflow.Context, _ agent.ReplyEvent) error {
		seq = append(seq, "reply")
		return boom
	}

	var res *agent.Result
	var runErr error
	env.ExecuteWorkflow(func(ctx workflow.Context) error {
		res, runErr = agent.RunWith(ctx, a, "Weather?", ro)
		return runErr
	})
	require.True(t, env.IsWorkflowCompleted())

	assert.ErrorIs(t, runErr, boom)
	assert.Equal(t, []string{"reply"}, seq, "no tool ran and OnTurn did not fire")
	assert.Equal(t, 1, fake.CallCount())
	if assert.NotNil(t, res) {
		assert.Equal(t, 1, res.Turns)
		assert.Empty(t, res.Output)
		assert.Empty(t, filterRole(res.Messages, model.RoleTool))
		if assert.NotEmpty(t, res.Messages) {
			last := res.Messages[len(res.Messages)-1]
			assert.Equal(t, model.RoleAssistant, last.Role)
			assert.Len(t, last.ToolCalls(), 1)
		}
	}
}

// TestRunWith_OnReplyErrorOnFinalReply verifies that an OnReply error on the
// final answer ends the run with an empty Output and that the partial Result
// ends with the answer.
func TestRunWith_OnReplyErrorOnFinalReply(t *testing.T) {
	fake := agenttest.NewFakeProvider(agenttest.Says("Hello there."))
	env := newEnv(t, fake)

	a, err := agent.NewAgent("assistant", "test-model")
	require.NoError(t, err)

	var seq []string
	boom := errors.New("saving the reply failed")
	ro := markerOptions(&seq)
	ro.OnReply = func(_ workflow.Context, _ agent.ReplyEvent) error { return boom }

	var res *agent.Result
	var runErr error
	env.ExecuteWorkflow(func(ctx workflow.Context) error {
		res, runErr = agent.RunWith(ctx, a, "Hi", ro)
		return runErr
	})
	require.True(t, env.IsWorkflowCompleted())

	assert.ErrorIs(t, runErr, boom)
	assert.Empty(t, seq, "OnTurn did not fire")
	if assert.NotNil(t, res) {
		assert.Equal(t, 1, res.Turns)
		assert.Empty(t, res.Output)
		if assert.NotEmpty(t, res.Messages) {
			assert.Equal(t, "Hello there.", res.Messages[len(res.Messages)-1].Text())
		}
	}
}

// TestReplay_OnReplyHook verifies that an OnReply hook that schedules an
// activity per reply replays deterministically.
func TestReplay_OnReplyHook(t *testing.T) {
	env := startReplayEnv(t)

	a, err := agent.NewAgent("assistant", "test-model", agent.WithTools(weatherTool(t)))
	require.NoError(t, err)

	const name = "replay_onreply"
	const persistName = "agentsdk_test_persist_reply"

	wf := func(ctx workflow.Context) (*agent.Result, error) {
		return agent.RunWith(ctx, a, "Weather in Ghent?", agent.RunOptions{
			OnReply: func(ctx workflow.Context, e agent.ReplyEvent) error {
				actx := workflow.WithActivityOptions(ctx, workflow.ActivityOptions{
					StartToCloseTimeout: 10 * time.Second,
				})
				return workflow.ExecuteActivity(actx, persistName, e.Turn).Get(actx, nil)
			},
		})
	}

	fake := agenttest.NewFakeProvider(
		agenttest.CallsTool("get_weather", `{"city":"Ghent"}`),
		agenttest.Says("It is 18°C in Ghent."),
	)

	w := worker.New(env.client, replayTaskQueue, worker.Options{})
	w.RegisterWorkflowWithOptions(wf, workflow.RegisterOptions{Name: name})
	acts, err := model.NewActivities([]model.Provider{fake})
	require.NoError(t, err)
	w.RegisterActivityWithOptions(acts.InvokeModel, activity.RegisterOptions{Name: model.InvokeModelActivity})
	w.RegisterActivityWithOptions(
		func(context.Context, int) error { return nil },
		activity.RegisterOptions{Name: persistName})
	require.NoError(t, w.Start())
	t.Cleanup(w.Stop)

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	run, err := env.client.ExecuteWorkflow(ctx, client.StartWorkflowOptions{TaskQueue: replayTaskQueue}, name)
	require.NoError(t, err)
	var res agent.Result
	require.NoError(t, run.Get(ctx, &res))
	assert.Equal(t, 2, res.Turns)

	hist := env.history(t, run.GetID(), run.GetRunID())
	assertReplays(t, hist, nil, wf, name)
}
