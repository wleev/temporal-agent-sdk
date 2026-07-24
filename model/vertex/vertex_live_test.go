package vertex_test

import (
	"context"
	"os"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/wleev/temporal-agent-sdk/model"
	vertexprovider "github.com/wleev/temporal-agent-sdk/model/vertex"
)

// TestLive_GeminiRoundTrip hits the real Gemini Developer API. It is opt-in — set
// VERTEX_E2E=1 and GEMINI_API_KEY — so the default suite stays offline, mirroring
// the opt-in real-MCP test. It proves the mapping works against a live backend,
// not just the stub: a text turn and a tool call both round-trip.
func TestLive_GeminiRoundTrip(t *testing.T) {
	if os.Getenv("VERTEX_E2E") != "1" {
		t.Skip("set VERTEX_E2E=1 and GEMINI_API_KEY to run the live Gemini test")
	}
	key := os.Getenv("GEMINI_API_KEY")
	if key == "" {
		t.Skip("GEMINI_API_KEY not set")
	}

	model_ := os.Getenv("VERTEX_E2E_MODEL")
	if model_ == "" {
		model_ = "gemini-2.5-flash"
	}

	p, err := vertexprovider.New(vertexprovider.WithAPIKey(key))
	require.NoError(t, err)

	// Plain text.
	resp, err := p.Invoke(context.Background(), model.Request{
		Model: model_,
		Messages: []model.Message{
			model.SystemMessage("Answer in one word."),
			model.UserMessage("What is the capital of Belgium?"),
		},
	})
	require.NoError(t, err)
	assert.Contains(t, resp.Message.Text(), "Brussels")
	assert.Greater(t, resp.Usage.TotalTokens, int64(0))

	// A tool call: the model should ask for the weather tool.
	resp, err = p.Invoke(context.Background(), model.Request{
		Model:    model_,
		Messages: []model.Message{model.UserMessage("Use the tool to get the weather in Ghent.")},
		Tools: []*model.Tool{model.NewTool("get_weather", "Look up the weather for a city",
			[]byte(`{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}`))},
	})
	require.NoError(t, err)
	calls := resp.Message.ToolCalls()
	if assert.NotEmpty(t, calls, "the model should call get_weather") {
		assert.Equal(t, "get_weather", calls[0].Name)
		assert.NotEmpty(t, calls[0].ID)
	}
}
