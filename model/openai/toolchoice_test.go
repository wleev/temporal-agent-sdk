package openai

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/wleev/temporal-agent-sdk/internal/jsontest"
	"github.com/wleev/temporal-agent-sdk/model"
)

// marshalParams returns the JSON encoding of the params toParams builds for
// req.
func marshalParams(t *testing.T, req model.Request) []byte {
	t.Helper()
	params, err := toParams(req)
	require.NoError(t, err)
	raw, err := json.Marshal(params)
	require.NoError(t, err)
	return raw
}

func TestToParams_ToolChoice(t *testing.T) {
	lookup := model.NewTool("lookup", "Look up", []byte(`{"type":"object"}`))
	cases := []struct {
		name   string
		choice *model.ToolChoice
		want   string
	}{
		{name: "nil", want: ``},
		{name: "auto", choice: &model.ToolChoice{Mode: model.ToolChoiceAuto}, want: `"auto"`},
		{name: "none", choice: &model.ToolChoice{Mode: model.ToolChoiceNone}, want: `"none"`},
		{name: "required", choice: &model.ToolChoice{Mode: model.ToolChoiceRequired}, want: `"required"`},
		{name: "tool", choice: &model.ToolChoice{Mode: model.ToolChoiceTool, Name: "lookup"}, want: `{"function":{"name":"lookup"},"type":"function"}`},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			raw := marshalParams(t, model.Request{Model: "m", Tools: []*model.Tool{lookup},
				Settings: model.Settings{ToolChoice: tc.choice}})
			got := jsontest.Path(t, raw, "tool_choice")
			if tc.want == "" {
				assert.Empty(t, got)
				return
			}
			assert.JSONEq(t, tc.want, got)
		})
	}
}

func TestToParams_ToolControlsWithoutTools(t *testing.T) {
	cases := []struct {
		name   string
		choice *model.ToolChoice
	}{
		{name: "nil"},
		{name: "auto", choice: &model.ToolChoice{Mode: model.ToolChoiceAuto}},
		{name: "none", choice: &model.ToolChoice{Mode: model.ToolChoiceNone}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			raw := marshalParams(t, model.Request{Model: "m",
				Settings: model.Settings{ToolChoice: tc.choice, ParallelToolCalls: model.Ptr(false)}})
			assert.Empty(t, jsontest.Path(t, raw, "tool_choice"))
			assert.Empty(t, jsontest.Path(t, raw, "parallel_tool_calls"))
		})
	}
}

func TestToParams_ToolChoiceUnsupported(t *testing.T) {
	_, err := toParams(model.Request{Model: "m",
		Settings: model.Settings{ToolChoice: &model.ToolChoice{Mode: model.ToolChoiceRequired}}})
	var unsupported *model.UnsupportedSettingError
	if assert.ErrorAs(t, err, &unsupported) {
		assert.Equal(t, "openai", unsupported.Provider)
		assert.Equal(t, "tool_choice", unsupported.Setting)
	}
}

func TestToParams_ParallelToolCalls(t *testing.T) {
	lookup := model.NewTool("lookup", "Look up", []byte(`{"type":"object"}`))
	cases := []struct {
		name     string
		parallel *bool
		want     string
	}{
		{name: "nil", want: ``},
		{name: "true", parallel: model.Ptr(true), want: `true`},
		{name: "false", parallel: model.Ptr(false), want: `false`},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			raw := marshalParams(t, model.Request{Model: "m", Tools: []*model.Tool{lookup},
				Settings: model.Settings{ParallelToolCalls: tc.parallel}})
			assert.Equal(t, tc.want, jsontest.Path(t, raw, "parallel_tool_calls"))
		})
	}
}

func TestToParams_StrictTool(t *testing.T) {
	plain := model.NewTool("lookup", "Look up", []byte(`{"type":"object"}`))
	strict := model.StrictTool(model.NewTool("submit", "Submit", []byte(`{"type":"object","additionalProperties":false}`)))
	raw := marshalParams(t, model.Request{Model: "m", Tools: []*model.Tool{plain, strict}})
	assert.Empty(t, jsontest.Path(t, raw, "tools.0.function.strict"))
	assert.Equal(t, "true", jsontest.Path(t, raw, "tools.1.function.strict"))
}
