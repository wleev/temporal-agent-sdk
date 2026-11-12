package model_test

import (
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/wleev/temporal-agent-sdk/model"
)

func TestUsageAdd(t *testing.T) {
	n := func(v int64) *int64 { return &v }

	got := model.Usage{PromptTokens: 10, CompletionTokens: 2, TotalTokens: 12, CacheReadTokens: n(4)}.
		Add(model.Usage{PromptTokens: 5, CompletionTokens: 1, TotalTokens: 6, CacheReadTokens: n(3), ReasoningTokens: n(1)})

	assert.Equal(t, model.Usage{
		PromptTokens:     15,
		CompletionTokens: 3,
		TotalTokens:      18,
		CacheReadTokens:  n(7),
		ReasoningTokens:  n(1),
	}, got)
	assert.Nil(t, got.CacheWriteTokens, "a count unreported by both stays nil")
}
