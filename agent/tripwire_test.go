package agent_test

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/wleev/temporal-agent-sdk/agent"
	"github.com/wleev/temporal-agent-sdk/guardrail"
)

func TestTripwireFromRecord_CopiesTheRecord(t *testing.T) {
	r := guardrail.Record{
		Stage: guardrail.StageToolResult, Guardrail: "mask", Action: guardrail.ActionBlock, Reason: "contains a secret",
		Notice: &guardrail.Notice{Text: "Blocked.", Audience: guardrail.AudienceUser}, Failed: true, Turn: 2, Index: 1,
	}
	assert.Equal(t, agent.TripwireError{
		Stage: guardrail.StageToolResult, Guardrail: "mask", Reason: "contains a secret",
		Notice: "Blocked.", Turn: 2, Index: 1, Failed: true,
	}, agent.TripwireFromRecord(r))

	r.Notice = nil
	assert.Empty(t, agent.TripwireFromRecord(r).Notice)
}

func TestTripwireError_JSONFieldNames(t *testing.T) {
	te := agent.TripwireError{Stage: guardrail.StageToolCall, Guardrail: "no-deletes", Reason: "account deletion is not allowed",
		Notice: "I can't delete accounts.", Turn: 2, Index: 1, Failed: true}
	b, err := json.Marshal(te)
	require.NoError(t, err)
	assert.JSONEq(t, `{"stage":"tool_call","guardrail":"no-deletes","reason":"account deletion is not allowed",
		"notice":"I can't delete accounts.","turn":2,"index":1,"failed":true}`, string(b))

	b, err = json.Marshal(agent.TripwireError{Stage: guardrail.StageInput, Guardrail: "blocklist", Reason: "disallowed"})
	require.NoError(t, err)
	assert.JSONEq(t, `{"stage":"input","guardrail":"blocklist","reason":"disallowed","turn":0,"index":0}`, string(b))
}
