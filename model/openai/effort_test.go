package openai_test

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/wleev/temporal-agent-sdk/model"
)

// invokeWithSettings invokes the provider with settings and returns the sent body.
func invokeWithSettings(t *testing.T, settings model.Settings) map[string]any {
	t.Helper()
	s := newStub(t)
	s.resp = plainReply
	p := newProvider(t, s)
	_, err := p.Invoke(context.Background(), model.Request{
		Model:    "gpt-test",
		Messages: []model.Message{model.UserMessage("hi")},
		Settings: settings,
	})
	require.NoError(t, err)
	return s.body
}

func TestInvoke_MapsReasoningEffort(t *testing.T) {
	body := invokeWithSettings(t, model.Settings{
		ReasoningEffort: model.Ptr("high"),
		Temperature:     model.Ptr(0.0),
		TopP:            model.Ptr(0.9),
	})
	assert.Equal(t, "high", body["reasoning_effort"])
	assert.NotContains(t, body, "temperature")
	assert.InDelta(t, 0.9, body["top_p"], 1e-9)
}

func TestInvoke_ReasoningEffortNoneKeepsTemperature(t *testing.T) {
	body := invokeWithSettings(t, model.Settings{
		ReasoningEffort: model.Ptr("none"),
		Temperature:     model.Ptr(0.3),
	})
	assert.Equal(t, "none", body["reasoning_effort"])
	assert.InDelta(t, 0.3, body["temperature"], 1e-9)
}

func TestInvoke_ReasoningEffortPassesThroughVerbatim(t *testing.T) {
	body := invokeWithSettings(t, model.Settings{ReasoningEffort: model.Ptr("xhigh")})
	assert.Equal(t, "xhigh", body["reasoning_effort"])
}

func TestInvoke_ReasoningEffortOmitted(t *testing.T) {
	tests := []struct {
		name   string
		effort *string
	}{
		{name: "empty", effort: model.Ptr("")},
		{name: "unset"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			body := invokeWithSettings(t, model.Settings{ReasoningEffort: tt.effort, Temperature: model.Ptr(0.4)})
			assert.NotContains(t, body, "reasoning_effort")
			assert.InDelta(t, 0.4, body["temperature"], 1e-9)
		})
	}
}
