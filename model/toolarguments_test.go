package model_test

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/wleev/temporal-agent-sdk/model"
)

func TestToolArguments(t *testing.T) {
	tests := []struct {
		name string
		raw  string
		want string
	}{
		{name: "object", raw: `{"a":1}`, want: `{"a":1}`},
		{name: "spaced object", raw: ` { "a" : 1 } `, want: ` { "a" : 1 } `},
		{name: "json string", raw: `"x"`, want: `"x"`},
		{name: "empty", raw: ``, want: ``},
		{name: "truncated", raw: `{"a":`, want: `"{\"a\":"`},
		{name: "not json", raw: `hello`, want: `"hello"`},
		{name: "whitespace", raw: ` `, want: `" "`},
		{name: "invalid utf-8", raw: "{\xff", want: "\"{\\ufffd\""},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := model.ToolArguments([]byte(tt.raw))
			assert.Equal(t, tt.want, string(got))
			if tt.raw != "" {
				assert.True(t, json.Valid(got))
			}
		})
	}
}

func TestToolArguments_ResponseRoundTrips(t *testing.T) {
	resp := model.Response{Message: model.Message{
		Role:   model.RoleAssistant,
		Blocks: []model.Block{model.ToolCallBlock("c1", "t", model.ToolArguments([]byte(`{"a":`)))},
	}}
	raw, err := json.Marshal(resp)
	require.NoError(t, err)
	var back model.Response
	require.NoError(t, json.Unmarshal(raw, &back))
	if assert.Len(t, back.Message.ToolCalls(), 1) {
		assert.Equal(t, `"{\"a\":"`, string(back.Message.ToolCalls()[0].Arguments))
	}
}
