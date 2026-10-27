package model_test

import (
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/wleev/temporal-agent-sdk/model"
)

func TestMessagePredicates(t *testing.T) {
	call := model.ToolCallBlock("call-1", "search", []byte(`{}`))
	for _, tc := range []struct {
		name         string
		message      model.Message
		assistant    bool
		hasText      bool
		hasToolCalls bool
	}{
		{name: "assistant text", message: model.AssistantMessage("Done."), assistant: true, hasText: true},
		{name: "user text", message: model.UserMessage("Hi."), hasText: true},
		{name: "whitespace text", message: model.AssistantMessage(" \n\t"), assistant: true},
		{name: "text split over blocks", message: model.Message{
			Role: model.RoleAssistant, Blocks: []model.Block{model.TextBlock(" "), model.TextBlock("ok")},
		}, assistant: true, hasText: true},
		{name: "thinking only", message: model.Message{
			Role: model.RoleAssistant, Blocks: []model.Block{model.ThinkingBlock("pondering", "sig")},
		}, assistant: true},
		{name: "tool call only", message: model.Message{
			Role: model.RoleAssistant, Blocks: []model.Block{call},
		}, assistant: true, hasToolCalls: true},
		{name: "text and tool call", message: model.Message{
			Role: model.RoleAssistant, Blocks: []model.Block{model.TextBlock("Looking."), call},
		}, assistant: true, hasText: true, hasToolCalls: true},
		{name: "empty", message: model.Message{Role: model.RoleAssistant}, assistant: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			assert.Equal(t, tc.assistant, tc.message.IsAssistant())
			assert.Equal(t, tc.hasText, tc.message.HasText())
			assert.Equal(t, tc.hasToolCalls, tc.message.HasToolCalls())
		})
	}
}
