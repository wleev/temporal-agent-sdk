package guardrail_test

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/wleev/temporal-agent-sdk/agenttest"
	"github.com/wleev/temporal-agent-sdk/guardrail"
)

func TestLLM_EmptyTextIsAllowedWithoutAModelCall(t *testing.T) {
	fake := agenttest.NewFakeProvider(agenttest.Says(`{"tripwire":true,"reason":"empty answer"}`))
	g, err := guardrail.LLM("moderation", "guard-model", []guardrail.Stage{guardrail.StageOutput})
	require.NoError(t, err)

	for _, text := range []string{"", " \n\t"} {
		v, err := runCheck(t, g, guardrail.CheckInput{Stage: guardrail.StageOutput, Text: text}, withModel(t, fake))
		require.NoError(t, err)
		assert.Equal(t, guardrail.Allow(), v, "text %q", text)
	}
	assert.Equal(t, 0, fake.CallCount())
}
