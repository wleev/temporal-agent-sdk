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

func TestFinalResult(t *testing.T) {
	final := model.FinalResult(model.TextResult("accepted"), "The answer.")
	out, ok := model.FinalOutput(final)
	assert.True(t, ok)
	assert.Equal(t, "The answer.", out)
	_, ok = model.FinalOutput(model.TextResult("plain"))
	assert.False(t, ok)
	assert.Equal(t, "accepted", model.ResultText(final))
}

func TestFinalResult_CopiesAndKeepsOtherMeta(t *testing.T) {
	plain := model.TextResult("accepted")
	plain.Meta = map[string]any{"other": 1}
	final := model.FinalResult(plain, "The answer.")
	assert.Equal(t, 1, final.Meta["other"])
	assert.NotContains(t, plain.Meta, model.FinalOutputMetaKey, "the argument is unchanged")

	empty := model.FinalResult(nil, "The answer.")
	if assert.NotNil(t, empty) {
		assert.Equal(t, "", model.ResultText(empty))
		out, ok := model.FinalOutput(empty)
		assert.True(t, ok)
		assert.Equal(t, "The answer.", out)
	}
	_, ok := model.FinalOutput(nil)
	assert.False(t, ok)

	raw, err := json.Marshal(final)
	require.NoError(t, err)
	assert.Contains(t, string(raw), `"temporal-agent-sdk/final_output":"The answer."`)
}

func TestWithoutFinalOutput(t *testing.T) {
	plain := model.TextResult("accepted")
	assert.Same(t, plain, model.WithoutFinalOutput(plain), "a result without the key is returned as is")
	assert.Nil(t, model.WithoutFinalOutput(nil))

	only := model.FinalResult(plain, "The answer.")
	stripped := model.WithoutFinalOutput(only)
	assert.Nil(t, stripped.Meta)
	assert.Contains(t, only.Meta, model.FinalOutputMetaKey, "the argument is unchanged")

	withOther := model.FinalResult(&model.CallToolResult{Meta: map[string]any{"trace": "abc"}}, "The answer.")
	assert.Equal(t, map[string]any{"trace": "abc"}, map[string]any(model.WithoutFinalOutput(withOther).Meta))
}

func TestCloneResult(t *testing.T) {
	assert.Nil(t, model.CloneResult(nil))

	orig := model.TextResult("accepted")
	orig.Meta = map[string]any{"trace": "abc"}
	c := model.CloneResult(orig)
	require.NotSame(t, orig, c)
	c.Content = append(c.Content, model.TextResult("extra").Content...)
	c.Meta["trace"] = "changed"
	c.IsError = true

	assert.Len(t, orig.Content, 1)
	assert.Equal(t, "abc", orig.Meta["trace"])
	assert.False(t, orig.IsError)
}
