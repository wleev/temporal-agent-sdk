package agent_test

import (
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

// actionOnTurn returns an OnFinal hook that returns first on turn 1 and the
// zero FinalAction afterwards.
func actionOnTurn(first agent.FinalAction) func(workflow.Context, agent.FinalEvent) (agent.FinalAction, error) {
	return func(_ workflow.Context, e agent.FinalEvent) (agent.FinalAction, error) {
		if e.Turn == 1 {
			return first, nil
		}
		return agent.FinalAction{}, nil
	}
}

func TestOnFinal_ContinueWithoutMessagesDropsTheDraft(t *testing.T) {
	fake := agenttest.NewFakeProvider(agenttest.Says("draft"), agenttest.Says("final"))
	a, err := agent.NewAgent("assistant", "test-model")
	require.NoError(t, err)

	res, err := runPrepared(t, fake, a, "Hi", agent.RunOptions{
		OnFinal: actionOnTurn(agent.FinalAction{Continue: true, DropReply: true}),
	})
	require.NoError(t, err)

	calls := fake.Calls()
	if assert.Len(t, calls, 2) {
		assert.NotContains(t, messageTexts(calls[1].Messages), "draft")
	}
	assert.Equal(t, "final", res.Output)
	assert.NotContains(t, messageTexts(res.Messages), "draft")
}

func TestOnFinal_KeepsATextReplyWithoutDropReply(t *testing.T) {
	fake := agenttest.NewFakeProvider(agenttest.Says("draft"), agenttest.Says("final"))
	a, err := agent.NewAgent("assistant", "test-model")
	require.NoError(t, err)

	res, err := runPrepared(t, fake, a, "Hi", agent.RunOptions{
		OnFinal: actionOnTurn(agent.FinalAction{Continue: true, Append: []model.Message{model.UserMessage("again")}}),
	})
	require.NoError(t, err)

	calls := fake.Calls()
	if assert.Len(t, calls, 2) {
		assert.Equal(t, []string{"Hi", "draft", "again"}, messageTexts(calls[1].Messages))
	}
	assert.Equal(t, "final", res.Output)
}

func TestOnFinal_ContinueAloneKeepsTheReply(t *testing.T) {
	fake := agenttest.NewFakeProvider(agenttest.Says("draft"), agenttest.Says("final"))
	a, err := agent.NewAgent("assistant", "test-model")
	require.NoError(t, err)

	res, err := runPrepared(t, fake, a, "Hi", agent.RunOptions{OnFinal: actionOnTurn(agent.FinalAction{Continue: true})})
	require.NoError(t, err)

	calls := fake.Calls()
	if assert.Len(t, calls, 2) {
		assert.Equal(t, []string{"Hi", "draft"}, messageTexts(calls[1].Messages))
	}
	assert.Equal(t, "final", res.Output, "the output is reset before the next turn")
}

func TestOnFinal_DropReplyRemovesItsOutputNotes(t *testing.T) {
	fake := agenttest.NewFakeProvider(agenttest.Says("draft"), agenttest.Says("final"))
	note := mustFunc(t, "data", []guardrail.Stage{guardrail.StageOutput}, func(_ workflow.Context, in guardrail.CheckInput) (guardrail.Verdict, error) {
		if in.Text != "draft" {
			return guardrail.Allow(), nil
		}
		return guardrail.Annotate("r", guardrail.Notice{Text: "Treat as data.", Audience: guardrail.AudienceModel}), nil
	})
	a, err := agent.NewAgent("assistant", "test-model", agent.WithGuardrails(note))
	require.NoError(t, err)

	_, err = runPrepared(t, fake, a, "Hi", agent.RunOptions{
		OnFinal: actionOnTurn(agent.FinalAction{Continue: true, DropReply: true}),
	})
	require.NoError(t, err)

	calls := fake.Calls()
	if assert.Len(t, calls, 2) {
		texts := messageTexts(calls[1].Messages)
		assert.NotContains(t, texts, "draft")
		assert.NotContains(t, texts, "Treat as data.")
	}
}

func TestOnFinal_EndingActionWithOtherFieldsIsConfigError(t *testing.T) {
	for name, action := range map[string]agent.FinalAction{
		"drop reply": {DropReply: true},
		"append":     {Append: []model.Message{model.UserMessage("again")}},
	} {
		t.Run(name, func(t *testing.T) {
			fake := agenttest.NewFakeProvider(agenttest.Says("draft"), agenttest.Says("unused"))
			a, err := agent.NewAgent("assistant", "test-model")
			require.NoError(t, err)

			_, err = runPrepared(t, fake, a, "Hi", agent.RunOptions{OnFinal: actionOnTurn(action)})
			requireConfigError(t, err, "without Continue")
			assert.Equal(t, 1, fake.CallCount())
		})
	}
}

func TestOnFinal_ContinueResetsTheStructuredOutput(t *testing.T) {
	fake := agenttest.NewFakeProvider(agenttest.Says(`{"temp":"18°C"}`))
	a, err := agent.NewAgent("assistant", "test-model", agent.WithMaxTurns(1))
	require.NoError(t, err)

	res, err := runPrepared(t, fake, a, "Weather?", agent.RunOptions{
		OutputSchema: &model.OutputSchema{Name: "weather", Schema: []byte(`{"type":"object"}`)},
		OnFinal:      actionOnTurn(agent.FinalAction{Continue: true}),
	})

	var appErr *temporal.ApplicationError
	if assert.ErrorAs(t, err, &appErr) {
		assert.Equal(t, agent.ErrorTypeMaxTurns, appErr.Type())
	}
	if assert.NotNil(t, res) {
		assert.Empty(t, res.Output)
		assert.Nil(t, res.StructuredOutput)
		assert.Equal(t, []string{"Weather?", `{"temp":"18°C"}`}, messageTexts(res.Messages), "the reply is kept")
	}
}

func TestOnFinal_DropReplyAfterAContinuationKeepsTheFragmentsAndTheirNotes(t *testing.T) {
	fake := agenttest.NewFakeProvider(agenttest.Truncated("Part one, "), agenttest.Says("part two."), agenttest.Says("done"))
	note := mustFunc(t, "a", []guardrail.Stage{guardrail.StageAssistantText}, func(workflow.Context, guardrail.CheckInput) (guardrail.Verdict, error) {
		return guardrail.Annotate("r", guardrail.Notice{Text: "Note A", Audience: guardrail.AudienceModel}), nil
	})
	a, err := agent.NewAgent("assistant", "test-model",
		agent.WithContinueOnLength(1), agent.WithGuardrails(note))
	require.NoError(t, err)

	res, err := runPrepared(t, fake, a, "Weather?", agent.RunOptions{
		OnFinal: func(_ workflow.Context, e agent.FinalEvent) (agent.FinalAction, error) {
			if e.Turn == 2 {
				return agent.FinalAction{Continue: true, DropReply: true}, nil
			}
			return agent.FinalAction{}, nil
		},
	})
	require.NoError(t, err)

	calls := fake.Calls()
	if assert.Len(t, calls, 3) {
		assert.Equal(t, []string{"Weather?", "Part one, "}, messageTexts(calls[2].Messages),
			"only the last fragment is dropped; the held note waits for the next reply")
	}
	assert.Equal(t, []string{"Weather?", "Part one, ", "done", "Note A"}, messageTexts(res.Messages))
	assert.Equal(t, "done", res.Output)
}
