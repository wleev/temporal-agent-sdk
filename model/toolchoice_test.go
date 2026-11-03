package model

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"go.temporal.io/sdk/temporal"
)

func TestToTemporalError_UnsupportedSetting(t *testing.T) {
	err := toTemporalError(&UnsupportedSettingError{Provider: "openai", Setting: "tool_choice", Reason: "unknown mode"})
	var appErr *temporal.ApplicationError
	if assert.ErrorAs(t, err, &appErr) {
		assert.Equal(t, ErrorTypeUnsupportedSetting, appErr.Type())
		assert.True(t, appErr.NonRetryable())
	}
}

func TestUnsupportedSettingError_Error(t *testing.T) {
	err := &UnsupportedSettingError{Provider: "openai", Setting: "tool_choice", Reason: "unknown mode"}
	assert.Equal(t, `model: provider "openai" cannot honor tool_choice: unknown mode`, err.Error())
}

func TestToolChoiceMode_Valid(t *testing.T) {
	for _, m := range []ToolChoiceMode{ToolChoiceAuto, ToolChoiceNone, ToolChoiceRequired, ToolChoiceTool} {
		assert.True(t, m.Valid(), m)
	}
	for _, m := range []ToolChoiceMode{"", "sometimes", "AUTO"} {
		assert.False(t, m.Valid(), m)
	}
}

func TestResolveToolChoice(t *testing.T) {
	lookup := NewTool("lookup", "Look up", []byte(`{"type":"object"}`))
	withTools := []*Tool{nil, lookup}
	cases := []struct {
		name       string
		choice     *ToolChoice
		tools      []*Tool
		want       *ToolChoice
		wantReason string
	}{
		{name: "nil", choice: nil, tools: withTools, want: nil},
		{name: "auto", choice: &ToolChoice{Mode: ToolChoiceAuto}, tools: withTools, want: &ToolChoice{Mode: ToolChoiceAuto}},
		{name: "none", choice: &ToolChoice{Mode: ToolChoiceNone}, tools: withTools, want: &ToolChoice{Mode: ToolChoiceNone}},
		{name: "required", choice: &ToolChoice{Mode: ToolChoiceRequired}, tools: withTools, want: &ToolChoice{Mode: ToolChoiceRequired}},
		{name: "tool", choice: &ToolChoice{Mode: ToolChoiceTool, Name: "lookup"}, tools: withTools, want: &ToolChoice{Mode: ToolChoiceTool, Name: "lookup"}},
		{name: "auto without tools", choice: &ToolChoice{Mode: ToolChoiceAuto}, tools: []*Tool{nil}, want: nil},
		{name: "none without tools", choice: &ToolChoice{Mode: ToolChoiceNone}, tools: nil, want: nil},
		{name: "required without tools", choice: &ToolChoice{Mode: ToolChoiceRequired}, tools: nil, wantReason: `mode "required" needs tools and the request has none`},
		{name: "tool without tools", choice: &ToolChoice{Mode: ToolChoiceTool, Name: "lookup"}, tools: nil, wantReason: `mode "tool" needs tools and the request has none`},
		{name: "empty mode", choice: &ToolChoice{}, tools: withTools, wantReason: `unknown mode ""`},
		{name: "unknown mode", choice: &ToolChoice{Mode: "sometimes"}, tools: withTools, wantReason: `unknown mode "sometimes"`},
		{name: "empty name", choice: &ToolChoice{Mode: ToolChoiceTool}, tools: withTools, wantReason: "mode tool needs a tool name"},
		{name: "unadvertised name", choice: &ToolChoice{Mode: ToolChoiceTool, Name: "other"}, tools: withTools, wantReason: `tool "other" is not advertised`},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := ResolveToolChoice("openai", tc.choice, tc.tools)
			if tc.wantReason == "" {
				assert.NoError(t, err)
				assert.Equal(t, tc.want, got)
				return
			}
			assert.Nil(t, got)
			var unsupported *UnsupportedSettingError
			if assert.ErrorAs(t, err, &unsupported) {
				want := UnsupportedSettingError{Provider: "openai", Setting: "tool_choice", Reason: tc.wantReason}
				assert.Equal(t, want, *unsupported)
			}
		})
	}
}
