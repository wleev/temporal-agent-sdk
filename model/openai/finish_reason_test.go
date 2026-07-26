package openai

import (
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/wleev/temporal-agent-sdk/model"
)

func TestFinishReason(t *testing.T) {
	for raw, want := range map[string]model.FinishReason{
		"":               model.FinishStop,
		"stop":           model.FinishStop,
		"length":         model.FinishLength,
		"tool_calls":     model.FinishToolCalls,
		"function_call":  model.FinishToolCalls,
		"content_filter": model.FinishContentFilter,
		"something_new":  model.FinishOther,
	} {
		assert.Equal(t, want, finishReason(raw), "finish_reason %q", raw)
	}
}
