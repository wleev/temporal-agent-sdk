package anthropic

import (
	"testing"

	anth "github.com/anthropics/anthropic-sdk-go"
	"github.com/stretchr/testify/assert"

	"github.com/wleev/temporal-agent-sdk/model"
)

func TestFinishReason(t *testing.T) {
	for sr, want := range map[anth.StopReason]model.FinishReason{
		"":                          model.FinishStop,
		anth.StopReasonEndTurn:      model.FinishStop,
		anth.StopReasonStopSequence: model.FinishStop,
		anth.StopReasonMaxTokens:    model.FinishLength,
		anth.StopReasonToolUse:      model.FinishToolCalls,
		anth.StopReasonRefusal:      model.FinishContentFilter,
		anth.StopReasonPauseTurn:    model.FinishOther,
	} {
		assert.Equal(t, want, finishReason(sr), "stop_reason %q", sr)
	}
}
