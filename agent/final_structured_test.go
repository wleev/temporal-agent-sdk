package agent_test

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.temporal.io/sdk/workflow"

	"github.com/wleev/temporal-agent-sdk/agent"
	"github.com/wleev/temporal-agent-sdk/agenttest"
	"github.com/wleev/temporal-agent-sdk/guardrail"
)

func TestOnFinal_ReceivesTheStructuredOutputAfterItsVerdicts(t *testing.T) {
	fake := agenttest.NewFakeProvider(agenttest.Says(`{"city":"Ghent","celsius":18}`))
	rewrite := mustFunc(t, "round", []guardrail.Stage{guardrail.StageOutput},
		func(workflow.Context, guardrail.CheckInput) (guardrail.Verdict, error) {
			return guardrail.RewriteText(`{"city":"Ghent","celsius":20}`, "round", nil), nil
		})
	a, err := agent.NewAgent("assistant", "test-model", agent.WithGuardrails(rewrite))
	require.NoError(t, err)

	var events []agent.FinalEvent
	res, err := runPrepared(t, fake, a, "Weather in Ghent?", agent.RunOptions{
		OutputSchema: answerSchema(),
		OnFinal:      recordFinal(&events),
	})
	require.NoError(t, err)

	if assert.Len(t, events, 1) {
		assert.JSONEq(t, `{"city":"Ghent","celsius":20}`, string(events[0].StructuredOutput))
	}
	assert.JSONEq(t, `{"city":"Ghent","celsius":20}`, string(res.StructuredOutput))

	events = nil
	_, err = runPrepared(t, agenttest.NewFakeProvider(agenttest.Says("Ghent is mild.")), a, "Weather in Ghent?",
		agent.RunOptions{OnFinal: recordFinal(&events)})
	require.NoError(t, err)
	if assert.Len(t, events, 1) {
		assert.Equal(t, json.RawMessage(nil), events[0].StructuredOutput, "a call without an output schema has none")
	}
}
