package guardrail_test

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/wleev/temporal-agent-sdk/guardrail"
)

func TestScreen_ABlockNoticeForAnotherAudienceIsAFailedBlock(t *testing.T) {
	for _, audience := range []guardrail.Audience{guardrail.AudienceModel, guardrail.AudienceBoth, ""} {
		t.Run(string(audience), func(t *testing.T) {
			v := guardrail.Verdict{Action: guardrail.ActionBlock, Reason: "policy",
				Notice: &guardrail.Notice{Text: "Do not retry.", Audience: audience}}
			out, err := screen(t, []guardrail.Guardrail{verdictGuard(t, "judge", stagesOf(guardrail.StageInput), v)},
				guardrail.CheckInput{Stage: guardrail.StageInput, Text: "Weather in Ghent?"})
			require.NoError(t, err)
			if assert.NotNil(t, out.Blocked) {
				assert.True(t, out.Blocked.Failed)
				assert.Nil(t, out.Blocked.Notice, "the notice of a failed check is dropped")
				assert.Contains(t, out.Blocked.Reason, "block notice audience")
			}
		})
	}
}
