package model_test

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/wleev/temporal-agent-sdk/model"
)

// InputSchema is typed any and reaches the accessor in several shapes: raw JSON
// from a local tool's reflector, or a value decoded from JSON once it has
// crossed an activity boundary. Every shape normalizes to the same bytes.
func TestToolInputSchemaJSON(t *testing.T) {
	const schema = `{"type":"object","properties":{"q":{"type":"string"}}}`

	for _, tc := range []struct {
		name string
		in   any
	}{
		{"raw JSON from the reflector", json.RawMessage(schema)},
		{"string", schema},
		{"bytes", []byte(schema)},
		{"decoded map after an activity boundary", map[string]any{
			"type":       "object",
			"properties": map[string]any{"q": map[string]any{"type": "string"}},
		}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			raw, err := model.ToolInputSchemaJSON(&model.Tool{Name: "search", InputSchema: tc.in})
			require.NoError(t, err)
			assert.JSONEq(t, schema, string(raw))
		})
	}
}

func TestToolInputSchemaJSON_AbsentIsNil(t *testing.T) {
	raw, err := model.ToolInputSchemaJSON(nil)
	require.NoError(t, err)
	assert.Nil(t, raw)

	raw, err = model.ToolInputSchemaJSON(&model.Tool{Name: "x"})
	require.NoError(t, err)
	assert.Nil(t, raw)
}

func TestToolInputSchemaJSON_UnmarshalableErrors(t *testing.T) {
	_, err := model.ToolInputSchemaJSON(&model.Tool{Name: "bad", InputSchema: make(chan int)})
	assert.ErrorContains(t, err, "unmarshalable input schema")
}
