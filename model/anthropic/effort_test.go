package anthropic_test

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/wleev/temporal-agent-sdk/model"
)

// invokeWithRequest invokes the provider on a request changed by mutate and returns the sent body.
func invokeWithRequest(t *testing.T, mutate func(*model.Request)) map[string]any {
	t.Helper()
	s := newStub(t)
	s.resp = textReply
	p := newProvider(t, s)
	req := model.Request{
		Model:    "claude-test",
		Messages: []model.Message{model.UserMessage("hi")},
	}
	mutate(&req)
	_, err := p.Invoke(context.Background(), req)
	require.NoError(t, err)
	return s.body
}

func TestInvoke_MapsReasoningEffort(t *testing.T) {
	body := invokeWithRequest(t, func(r *model.Request) {
		r.Settings = model.Settings{
			ReasoningEffort: model.Ptr("high"),
			Temperature:     model.Ptr(0.2),
			TopP:            model.Ptr(0.9),
		}
	})
	oc, ok := body["output_config"].(map[string]any)
	if assert.True(t, ok) {
		assert.Equal(t, "high", oc["effort"])
	}
	assert.NotContains(t, body, "thinking")
	assert.NotContains(t, body, "temperature")
	assert.InDelta(t, 0.9, body["top_p"], 1e-9)
}

func TestInvoke_ReasoningEffortWithStructuredOutput(t *testing.T) {
	body := invokeWithRequest(t, func(r *model.Request) {
		r.Settings = model.Settings{ReasoningEffort: model.Ptr("low")}
		r.OutputSchema = &model.OutputSchema{
			Name:   "weather",
			Schema: json.RawMessage(`{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}`),
		}
	})
	oc, ok := body["output_config"].(map[string]any)
	if assert.True(t, ok) {
		assert.Equal(t, "low", oc["effort"])
		format, ok := oc["format"].(map[string]any)
		if assert.True(t, ok) {
			assert.Equal(t, "json_schema", format["type"])
		}
	}
}

func TestInvoke_ReasoningEffortPassesThrough(t *testing.T) {
	for _, effort := range []string{"xhigh", "max"} {
		t.Run(effort, func(t *testing.T) {
			body := invokeWithRequest(t, func(r *model.Request) {
				r.Settings = model.Settings{ReasoningEffort: model.Ptr(effort)}
			})
			oc, ok := body["output_config"].(map[string]any)
			if assert.True(t, ok) {
				assert.Equal(t, effort, oc["effort"])
			}
		})
	}
}

func TestInvoke_ReasoningEffortOmitted(t *testing.T) {
	tests := []struct {
		name   string
		effort *string
	}{
		{name: "none", effort: model.Ptr("none")},
		{name: "empty", effort: model.Ptr("")},
		{name: "unset"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			body := invokeWithRequest(t, func(r *model.Request) {
				r.Settings = model.Settings{ReasoningEffort: tt.effort, Temperature: model.Ptr(0.2)}
			})
			assert.NotContains(t, body, "output_config")
			assert.InDelta(t, 0.2, body["temperature"], 1e-9)
		})
	}
}
