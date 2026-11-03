package anthropic

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
	params, err := (&Provider{defaultMaxTokens: 100}).toParams(req)
	require.NoError(t, err)
	raw, err := json.Marshal(params)
	require.NoError(t, err)
	return raw
}

func TestToParams_ToolChoice(t *testing.T) {
	lookup := model.NewTool("lookup", "Look up", []byte(`{"type":"object"}`))
	serial := model.Ptr(false)
	cases := []struct {
		name     string
		choice   *model.ToolChoice
		parallel *bool
		want     string
	}{
		{name: "nil", want: ``},
		{name: "auto", choice: &model.ToolChoice{Mode: model.ToolChoiceAuto}, want: `{"type":"auto"}`},
		{name: "none", choice: &model.ToolChoice{Mode: model.ToolChoiceNone}, want: `{"type":"none"}`},
		{name: "required", choice: &model.ToolChoice{Mode: model.ToolChoiceRequired}, want: `{"type":"any"}`},
		{name: "tool", choice: &model.ToolChoice{Mode: model.ToolChoiceTool, Name: "lookup"}, want: `{"type":"tool","name":"lookup"}`},
		{name: "parallel true, nil", parallel: model.Ptr(true), want: ``},
		{name: "parallel false, nil", parallel: serial, want: `{"type":"auto","disable_parallel_tool_use":true}`},
		{name: "parallel false, auto", choice: &model.ToolChoice{Mode: model.ToolChoiceAuto}, parallel: serial, want: `{"type":"auto","disable_parallel_tool_use":true}`},
		{name: "parallel false, required", choice: &model.ToolChoice{Mode: model.ToolChoiceRequired}, parallel: serial, want: `{"type":"any","disable_parallel_tool_use":true}`},
		{name: "parallel false, tool", choice: &model.ToolChoice{Mode: model.ToolChoiceTool, Name: "lookup"}, parallel: serial, want: `{"type":"tool","name":"lookup","disable_parallel_tool_use":true}`},
		{name: "parallel false, none", choice: &model.ToolChoice{Mode: model.ToolChoiceNone}, parallel: serial, want: `{"type":"none"}`},
		{name: "parallel true, required", choice: &model.ToolChoice{Mode: model.ToolChoiceRequired}, parallel: model.Ptr(true), want: `{"type":"any"}`},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			raw := marshalParams(t, model.Request{Model: "m", Tools: []*model.Tool{lookup},
				Settings: model.Settings{ToolChoice: tc.choice, ParallelToolCalls: tc.parallel}})
			got := jsontest.Path(t, raw, "tool_choice")
			if tc.want == "" {
				assert.Empty(t, got)
				return
			}
			assert.JSONEq(t, tc.want, got)
		})
	}
}

func TestToParams_ToolChoiceWithoutTools(t *testing.T) {
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
		})
	}
}

func TestToParams_ToolChoiceUnsupported(t *testing.T) {
	_, err := (&Provider{defaultMaxTokens: 100}).toParams(model.Request{Model: "m",
		Settings: model.Settings{ToolChoice: &model.ToolChoice{Mode: model.ToolChoiceRequired}}})
	var unsupported *model.UnsupportedSettingError
	if assert.ErrorAs(t, err, &unsupported) {
		assert.Equal(t, "anthropic", unsupported.Provider)
		assert.Equal(t, "tool_choice", unsupported.Setting)
	}
}

func TestToParams_StrictToolHasNoStrictField(t *testing.T) {
	strict := model.StrictTool(model.NewTool("submit", "Submit", []byte(`{"type":"object","additionalProperties":false}`)))
	raw := marshalParams(t, model.Request{Model: "m", Tools: []*model.Tool{strict}})
	assert.Equal(t, `"submit"`, jsontest.Path(t, raw, "tools.0.name"))
	assert.Empty(t, jsontest.Path(t, raw, "tools.0.strict"))
}
