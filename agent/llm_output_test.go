package agent_test

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.temporal.io/sdk/workflow"

	"github.com/wleev/temporal-agent-sdk/agent"
	"github.com/wleev/temporal-agent-sdk/agenttest"
	"github.com/wleev/temporal-agent-sdk/guardrail"
	"github.com/wleev/temporal-agent-sdk/model"
)

func TestRunWith_AnLLMOutputGuardrailLetsOnFinalNudgeAnEmptyReply(t *testing.T) {
	fake := agenttest.NewFakeProvider(
		agenttest.Says(""),
		agenttest.Says("Ghent is mild."),
		agenttest.Says(`{"tripwire":false,"reason":"fine"}`),
	)
	moderation, err := guardrail.LLM("moderation", "guard-model", []guardrail.Stage{guardrail.StageOutput})
	require.NoError(t, err)
	a, err := agent.NewAgent("assistant", "test-model", agent.WithGuardrails(moderation))
	require.NoError(t, err)

	res, err := runPrepared(t, fake, a, "Weather in Ghent?", agent.RunOptions{
		OnFinal: func(_ workflow.Context, e agent.FinalEvent) (agent.FinalAction, error) {
			if e.Assistant.Text() == "" {
				return agent.FinalAction{Continue: true, Append: []model.Message{model.UserMessage("Answer in text.")}}, nil
			}
			return agent.FinalAction{}, nil
		},
	})
	require.NoError(t, err)

	assert.Equal(t, "Ghent is mild.", res.Output)
	if calls := fake.Calls(); assert.Len(t, calls, 3) {
		assert.Equal(t, "guard-model", calls[2].Model, "only the text answer is sent to the guardrail")
	}
}
