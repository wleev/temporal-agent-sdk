package vertex_test

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"google.golang.org/genai"

	"github.com/wleev/temporal-agent-sdk/model"
	vertexprovider "github.com/wleev/temporal-agent-sdk/model/vertex"
)

// A grounded turn: Gemini executes the built-in search server-side and returns
// text plus grounding metadata, never a function call.
const groundedReply = `{
  "candidates":[{
    "content":{"role":"model","parts":[{"text":"Brussels is the capital of Belgium."}]},
    "finishReason":"STOP",
    "groundingMetadata":{
      "webSearchQueries":["capital of Belgium"],
      "groundingChunks":[{"web":{"uri":"https://example.com","title":"Belgium"}}]
    }
  }],
  "usageMetadata":{"promptTokenCount":8,"candidatesTokenCount":7,"totalTokenCount":15}
}`

// A provider-native tool runs server-side, so its result comes back as ordinary
// text with grounding metadata attached. The provider extracts the text and
// ignores the metadata, and the agent loop ends the turn as it would for any
// other text answer — there is no tool call to dispatch.
func TestInvoke_ToleratesGroundedResponse(t *testing.T) {
	s := newStub(t)
	s.resp = groundedReply
	p := newProvider(t, s)

	resp, err := p.Invoke(context.Background(), model.Request{
		Model:    "gemini-test",
		Messages: []model.Message{model.UserMessage("What is the capital of Belgium?")},
	})
	require.NoError(t, err)
	assert.Equal(t, "Brussels is the capital of Belgium.", resp.Message.Text())
	assert.Empty(t, resp.Message.ToolCalls(), "a server-executed built-in yields no dispatchable tool call")
	assert.Equal(t, model.FinishStop, resp.FinishReason)
}

// WithParams is applied after the SDK fills the request config, so a consumer
// can add a provider-native tool (here Google Search grounding) and it reaches
// the wire request. This is the supported path for provider built-in tools.
func TestWithParams_InjectsGoogleSearchTool(t *testing.T) {
	s := newStub(t)
	s.resp = groundedReply
	p := newProvider(t, s, vertexprovider.WithParams(func(c *genai.GenerateContentConfig) {
		c.Tools = append(c.Tools, &genai.Tool{GoogleSearch: &genai.GoogleSearch{}})
	}))

	_, err := p.Invoke(context.Background(), model.Request{
		Model:    "gemini-test",
		Messages: []model.Message{model.UserMessage("What is the capital of Belgium?")},
	})
	require.NoError(t, err)

	tools, ok := s.body["tools"].([]any)
	require.True(t, ok, "request carries no tools array: %v", s.body)
	found := false
	for _, tl := range tools {
		if _, has := tl.(map[string]any)["googleSearch"]; has {
			found = true
		}
	}
	assert.True(t, found, "the google search tool must reach the wire request: %v", tools)
}
