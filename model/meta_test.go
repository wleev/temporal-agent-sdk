package model_test

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/wleev/temporal-agent-sdk/model"
)

func TestStrictTool(t *testing.T) {
	plain := model.NewTool("submit", "Submit", []byte(`{"type":"object"}`))
	strict := model.StrictTool(plain)
	assert.True(t, model.IsStrictTool(strict))
	assert.False(t, model.IsStrictTool(plain), "the argument is unchanged")
	assert.Nil(t, model.StrictTool(nil))
	assert.False(t, model.IsStrictTool(nil))
	raw, err := json.Marshal(strict)
	require.NoError(t, err)
	assert.Contains(t, string(raw), `"_meta":{"temporal-agent-sdk/strict":true}`)
}

func TestStrictTool_KeepsOtherMeta(t *testing.T) {
	plain := model.NewTool("submit", "Submit", []byte(`{"type":"object"}`))
	plain.Meta = map[string]any{"other": 1}
	strict := model.StrictTool(plain)
	assert.Equal(t, 1, strict.Meta["other"])
	assert.NotContains(t, plain.Meta, model.StrictToolMetaKey)
}
