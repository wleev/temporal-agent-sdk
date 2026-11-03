package vertex

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"google.golang.org/genai"

	"github.com/wleev/temporal-agent-sdk/model"
)

func TestBuild_ToolChoice(t *testing.T) {
	lookup := model.NewTool("lookup", "Look up", []byte(`{"type":"object"}`))
	config := func(mode genai.FunctionCallingConfigMode, names ...string) *genai.ToolConfig {
		return &genai.ToolConfig{FunctionCallingConfig: &genai.FunctionCallingConfig{Mode: mode, AllowedFunctionNames: names}}
	}
	cases := []struct {
		name   string
		choice *model.ToolChoice
		want   *genai.ToolConfig
	}{
		{name: "nil", want: nil},
		{name: "auto", choice: &model.ToolChoice{Mode: model.ToolChoiceAuto}, want: config(genai.FunctionCallingConfigModeAuto)},
		{name: "none", choice: &model.ToolChoice{Mode: model.ToolChoiceNone}, want: config(genai.FunctionCallingConfigModeNone)},
		{name: "required", choice: &model.ToolChoice{Mode: model.ToolChoiceRequired}, want: config(genai.FunctionCallingConfigModeAny)},
		{name: "tool", choice: &model.ToolChoice{Mode: model.ToolChoiceTool, Name: "lookup"}, want: config(genai.FunctionCallingConfigModeAny, "lookup")},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, cfg, err := (&Provider{}).build(model.Request{Model: "m", Tools: []*model.Tool{lookup},
				Settings: model.Settings{ToolChoice: tc.choice}})
			require.NoError(t, err)
			assert.Equal(t, tc.want, cfg.ToolConfig)
		})
	}
}

func TestBuild_ToolChoiceWithoutTools(t *testing.T) {
	for _, mode := range []model.ToolChoiceMode{model.ToolChoiceAuto, model.ToolChoiceNone} {
		t.Run(string(mode), func(t *testing.T) {
			_, cfg, err := (&Provider{}).build(model.Request{Model: "m",
				Settings: model.Settings{ToolChoice: &model.ToolChoice{Mode: mode}}})
			require.NoError(t, err)
			assert.Nil(t, cfg.ToolConfig)
		})
	}
}

func TestBuild_ToolChoiceUnsupported(t *testing.T) {
	_, _, err := (&Provider{}).build(model.Request{Model: "m",
		Settings: model.Settings{ToolChoice: &model.ToolChoice{Mode: model.ToolChoiceRequired}}})
	var unsupported *model.UnsupportedSettingError
	if assert.ErrorAs(t, err, &unsupported) {
		assert.Equal(t, "vertex", unsupported.Provider)
		assert.Equal(t, "tool_choice", unsupported.Setting)
	}
}

func TestBuild_IgnoresParallelAndStrict(t *testing.T) {
	lookup := model.NewTool("lookup", "Look up", []byte(`{"type":"object"}`))
	_, base, err := (&Provider{}).build(model.Request{Model: "m", Tools: []*model.Tool{lookup}})
	require.NoError(t, err)
	_, got, err := (&Provider{}).build(model.Request{Model: "m", Tools: []*model.Tool{model.StrictTool(lookup)},
		Settings: model.Settings{ParallelToolCalls: model.Ptr(false)}})
	require.NoError(t, err)
	assert.Equal(t, base, got)
}
