package vertex

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"google.golang.org/genai"

	"github.com/wleev/temporal-agent-sdk/model"
)

func TestFinishReason(t *testing.T) {
	for fr, want := range map[genai.FinishReason]model.FinishReason{
		"":                                      model.FinishStop,
		genai.FinishReasonStop:                  model.FinishStop,
		genai.FinishReasonMaxTokens:             model.FinishLength,
		genai.FinishReasonSafety:                model.FinishContentFilter,
		genai.FinishReasonProhibitedContent:     model.FinishContentFilter,
		genai.FinishReasonMalformedFunctionCall: model.FinishOther,
	} {
		assert.Equal(t, want, finishReason(fr), "finish_reason %q", fr)
	}
}
