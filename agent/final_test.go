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
	"github.com/wleev/temporal-agent-sdk/model"
)

// recordFinal returns an OnFinal hook that appends each event to events and
// returns no messages.
func recordFinal(events *[]agent.FinalEvent) func(workflow.Context, agent.FinalEvent) ([]model.Message, error) {
	return func(_ workflow.Context, e agent.FinalEvent) ([]model.Message, error) {
		*events = append(*events, e)
		return nil, nil
	}
}

// nudgeOnce returns an OnFinal hook that returns a user message with text on
// its first call and no messages after.
func nudgeOnce(text string) func(workflow.Context, agent.FinalEvent) ([]model.Message, error) {
	nudged := false
	return func(workflow.Context, agent.FinalEvent) ([]model.Message, error) {
		if nudged {
			return nil, nil
		}
		nudged = true
		return []model.Message{model.UserMessage(text)}, nil
	}
}

func TestOnFinal_ReceivesFinalReply(t *testing.T) {
	fake := agenttest.NewFakeProvider(
		agenttest.CallsTool("get_weather", `{"city":"Ghent"}`),
		agenttest.Says("done"),
	)
	a, err := agent.NewAgent("assistant", "test-model", agent.WithTools(weatherTool(t)))
	require.NoError(t, err)

	var events []agent.FinalEvent
	res, err := runPrepared(t, fake, a, "Weather in Ghent?", agent.RunOptions{OnFinal: recordFinal(&events)})
	require.NoError(t, err)

	if assert.Len(t, events, 1) {
		assert.Equal(t, 2, events[0].Turn)
		assert.Equal(t, "done", events[0].Assistant.Text())
		assert.Equal(t, model.FinishStop, events[0].FinishReason)
		assert.Equal(t, int64(15), events[0].Usage.TotalTokens)
	}
	assert.Equal(t, "done", res.Output)
}

func TestOnFinal_SkipsContinuationFragments(t *testing.T) {
	fake := agenttest.NewFakeProvider(agenttest.Truncated("par"), agenttest.Says("tial"))
	a, err := agent.NewAgent("assistant", "test-model", agent.WithContinueOnLength(1))
	require.NoError(t, err)

	var events []agent.FinalEvent
	res, err := runPrepared(t, fake, a, "Tell me.", agent.RunOptions{OnFinal: recordFinal(&events)})
	require.NoError(t, err)

	if assert.Len(t, events, 1) {
		assert.Equal(t, 2, events[0].Turn)
		assert.Equal(t, "tial", events[0].Assistant.Text())
	}
	assert.Equal(t, "partial", res.Output)
}

func TestOnFinal_CarriesLengthFinishReason(t *testing.T) {
	fake := agenttest.NewFakeProvider(agenttest.Truncated("cut"))
	a, err := agent.NewAgent("assistant", "test-model")
	require.NoError(t, err)

	var events []agent.FinalEvent
	_, err = runPrepared(t, fake, a, "Tell me.", agent.RunOptions{OnFinal: recordFinal(&events)})
	require.NoError(t, err)

	if assert.Len(t, events, 1) {
		assert.Equal(t, model.FinishLength, events[0].FinishReason)
	}
}

func TestOnFinal_ContinuesWithMessages(t *testing.T) {
	fake := agenttest.NewFakeProvider(
		agenttest.Response{Message: model.Message{Role: model.RoleAssistant}},
		agenttest.Says("answer"),
	)
	a, err := agent.NewAgent("assistant", "test-model")
	require.NoError(t, err)

	var events []agent.FinalEvent
	nudge := nudgeOnce("nudge")
	res, err := runPrepared(t, fake, a, "Question?", agent.RunOptions{
		OnFinal: func(ctx workflow.Context, e agent.FinalEvent) ([]model.Message, error) {
			events = append(events, e)
			return nudge(ctx, e)
		},
	})
	require.NoError(t, err)

	if assert.Equal(t, 2, fake.CallCount()) {
		msgs := fake.Calls()[1].Messages
		last := msgs[len(msgs)-1]
		assert.Equal(t, model.RoleUser, last.Role)
		assert.Equal(t, "nudge", last.Text())
	}
	assert.Equal(t, "answer", res.Output)
	assert.Equal(t, 2, res.Turns)
	assert.Len(t, events, 2)
}

func TestOnFinal_DropsContentlessReplyBeforeContinuing(t *testing.T) {
	fake := agenttest.NewFakeProvider(
		agenttest.Response{Message: model.Message{Role: model.RoleAssistant}},
		agenttest.Says("answer"),
	)
	a, err := agent.NewAgent("assistant", "test-model")
	require.NoError(t, err)

	res, err := runPrepared(t, fake, a, "Question?", agent.RunOptions{
		OnFinal: nudgeOnce("nudge"),
	})
	require.NoError(t, err)

	if assert.Equal(t, 2, fake.CallCount()) {
		msgs := fake.Calls()[1].Messages
		last := msgs[len(msgs)-1]
		assert.Equal(t, model.RoleUser, last.Role)
		assert.Equal(t, "nudge", last.Text())
		for _, m := range msgs {
			if m.Role == model.RoleAssistant {
				assert.NotEmpty(t, m.Blocks, "empty assistant message re-sent")
			}
		}
	}
	for _, m := range res.Messages {
		if m.Role == model.RoleAssistant {
			assert.NotEmpty(t, m.Blocks, "empty assistant message kept in Result.Messages")
		}
	}
}

func TestOnFinal_DropsWhitespaceReplyBeforeContinuing(t *testing.T) {
	fake := agenttest.NewFakeProvider(agenttest.Says(" \n\t"), agenttest.Says("answer"))
	a, err := agent.NewAgent("assistant", "test-model")
	require.NoError(t, err)

	res, err := runPrepared(t, fake, a, "Question?", agent.RunOptions{OnFinal: nudgeOnce("nudge")})
	require.NoError(t, err)

	if assert.Equal(t, 2, fake.CallCount()) {
		msgs := fake.Calls()[1].Messages
		if assert.Len(t, msgs, 2) {
			assert.Equal(t, "Question?", msgs[0].Text())
			assert.Equal(t, "nudge", msgs[1].Text())
		}
	}
	assert.Equal(t, "answer", res.Output)
	if assert.Len(t, res.Messages, 3) {
		assert.Equal(t, "answer", res.Messages[2].Text())
	}
}

func TestOnFinal_KeepsReplyWithContentBeforeContinuing(t *testing.T) {
	fake := agenttest.NewFakeProvider(agenttest.Says("first"), agenttest.Says("second"))
	a, err := agent.NewAgent("assistant", "test-model")
	require.NoError(t, err)

	_, err = runPrepared(t, fake, a, "Question?", agent.RunOptions{
		OnFinal: nudgeOnce("nudge"),
	})
	require.NoError(t, err)

	if assert.Equal(t, 2, fake.CallCount()) {
		msgs := fake.Calls()[1].Messages
		if assert.GreaterOrEqual(t, len(msgs), 2) {
			assert.Equal(t, model.RoleAssistant, msgs[len(msgs)-2].Role)
			assert.Equal(t, "first", msgs[len(msgs)-2].Text())
			assert.Equal(t, "nudge", msgs[len(msgs)-1].Text())
		}
	}
}

func TestOnFinal_WithPrepareModelCall(t *testing.T) {
	fake := agenttest.NewFakeProvider(agenttest.Says("first"), agenttest.Says("second"))
	a, err := agent.NewAgent("assistant", "test-model")
	require.NoError(t, err)

	var turns []int
	res, err := runPrepared(t, fake, a, "Question?", agent.RunOptions{
		PrepareModelCall: func(_ workflow.Context, p agent.ModelCallPlan) (agent.ModelCallPlan, error) {
			turns = append(turns, p.Turn)
			if p.Turn == 2 {
				p.Append = []model.Message{model.UserMessage("extra")}
			}
			return p, nil
		},
		OnFinal: nudgeOnce("nudge"),
	})
	require.NoError(t, err)

	assert.Equal(t, []int{1, 2}, turns)
	if assert.Equal(t, 2, fake.CallCount()) {
		msgs := fake.Calls()[1].Messages
		if assert.GreaterOrEqual(t, len(msgs), 2) {
			assert.Equal(t, "nudge", msgs[len(msgs)-2].Text())
			assert.Equal(t, "extra", msgs[len(msgs)-1].Text())
		}
	}
	assert.Equal(t, "second", res.Output)
}

func TestOnFinal_RunsAfterOnTurn(t *testing.T) {
	fake := agenttest.NewFakeProvider(agenttest.Says("done"))
	a, err := agent.NewAgent("assistant", "test-model")
	require.NoError(t, err)

	var seq []string
	ro := markerOptions(&seq)
	ro.OnFinal = func(_ workflow.Context, e agent.FinalEvent) ([]model.Message, error) {
		seq = append(seq, fmt.Sprintf("final:%d", e.Turn))
		return nil, nil
	}
	_, err = runPrepared(t, fake, a, "Hi", ro)
	require.NoError(t, err)

	assert.Equal(t, []string{"reply:1", "turn:1", "final:1"}, seq)
}

func TestOnFinal_ContinueOnLastTurnIsMaxTurns(t *testing.T) {
	fake := agenttest.NewFakeProvider(agenttest.Says("first"))
	a, err := agent.NewAgent("assistant", "test-model", agent.WithMaxTurns(1))
	require.NoError(t, err)

	res, err := runPrepared(t, fake, a, "Hi", agent.RunOptions{
		OnFinal: func(_ workflow.Context, _ agent.FinalEvent) ([]model.Message, error) {
			return []model.Message{model.UserMessage("nudge")}, nil
		},
	})
	var appErr *temporal.ApplicationError
	if assert.ErrorAs(t, err, &appErr) {
		assert.Equal(t, agent.ErrorTypeMaxTurns, appErr.Type())
	}
	assert.Equal(t, 1, fake.CallCount())
	require.NotNil(t, res)
	if assert.NotEmpty(t, res.Messages) {
		last := res.Messages[len(res.Messages)-1]
		assert.Equal(t, model.RoleUser, last.Role)
		assert.Equal(t, "nudge", last.Text())
	}
	assert.Equal(t, "", res.Output)
}

func TestOnFinal_ErrorAbortsRun(t *testing.T) {
	fake := agenttest.NewFakeProvider(agenttest.Says("done"))
	a, err := agent.NewAgent("assistant", "test-model")
	require.NoError(t, err)

	_, err = runPrepared(t, fake, a, "Hi", agent.RunOptions{
		OnFinal: func(_ workflow.Context, _ agent.FinalEvent) ([]model.Message, error) {
			return nil, errors.New("final failed")
		},
	})
	if assert.Error(t, err) {
		assert.Contains(t, err.Error(), "final failed")
	}
	assert.Equal(t, 1, fake.CallCount())
}

func TestOnFinal_OutputGuardrailSeesOnlyTheEndingAnswer(t *testing.T) {
	fake := agenttest.NewFakeProvider(agenttest.Says("first"), agenttest.Says("second"))
	var checked []string
	a, err := agent.NewAgent("assistant", "test-model",
		agent.WithOutputGuardrails(guardrail.Func("record", func(_ workflow.Context, text string) (guardrail.Result, error) {
			checked = append(checked, text)
			return guardrail.Result{}, nil
		})))
	require.NoError(t, err)

	_, err = runPrepared(t, fake, a, "Hi", agent.RunOptions{
		OnFinal: nudgeOnce("nudge"),
	})
	require.NoError(t, err)

	assert.Equal(t, []string{"second"}, checked)
}
