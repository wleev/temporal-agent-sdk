package vertex_test

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/wleev/temporal-agent-sdk/model"
	vertexprovider "github.com/wleev/temporal-agent-sdk/model/vertex"
)

// stubServer stands in for the Gemini generateContent endpoint and captures the
// request body, so tests assert the exact contents/parts wire shape the provider
// produces. The structural transforms — system hoisting, tool-result coalescing,
// image inline_data — compile fine when wrong and only show up in the body.
type stubServer struct {
	*httptest.Server
	body   map[string]any
	status int
	resp   string
}

func newStub(t *testing.T) *stubServer {
	t.Helper()
	s := &stubServer{status: 200}
	s.Server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, err := io.ReadAll(r.Body)
		require.NoError(t, err)
		if len(raw) > 0 {
			require.NoError(t, json.Unmarshal(raw, &s.body))
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(s.status)
		_, _ = io.WriteString(w, s.resp)
	}))
	t.Cleanup(s.Close)
	return s
}

// newProvider points a Gemini-API-backend client at the stub: an API key needs no
// GCP credentials, so the whole suite runs offline while producing the same
// generateContent body a Vertex call would.
func newProvider(t *testing.T, s *stubServer, opts ...vertexprovider.Option) *vertexprovider.Provider {
	t.Helper()
	opts = append([]vertexprovider.Option{
		vertexprovider.WithAPIKey("test-key"),
		vertexprovider.WithBaseURL(s.URL),
	}, opts...)
	p, err := vertexprovider.New(opts...)
	require.NoError(t, err)
	return p
}

const textReply = `{
  "candidates":[{"content":{"role":"model","parts":[{"text":"Hello."}]},"finishReason":"STOP"}],
  "usageMetadata":{"promptTokenCount":11,"candidatesTokenCount":3,"totalTokenCount":14}
}`

func contents(t *testing.T, s *stubServer) []any {
	t.Helper()
	c, ok := s.body["contents"].([]any)
	require.True(t, ok, "request has no contents array: %v", s.body)
	return c
}

func TestInvoke_PlainCompletion(t *testing.T) {
	s := newStub(t)
	s.resp = textReply
	p := newProvider(t, s)

	resp, err := p.Invoke(context.Background(), model.Request{
		Model: "gemini-test",
		Messages: []model.Message{
			model.SystemMessage("Be brief."),
			model.UserMessage("Hi"),
		},
	})
	require.NoError(t, err)
	assert.Equal(t, "Hello.", resp.Message.Text())
	assert.Equal(t, model.RoleAssistant, resp.Message.Role)
	assert.Equal(t, "STOP", resp.FinishReason)
	assert.Equal(t, int64(14), resp.Usage.TotalTokens)
	assert.Equal(t, int64(11), resp.Usage.PromptTokens)

	// System is hoisted out of the messages into systemInstruction.
	sys := s.body["systemInstruction"].(map[string]any)
	sysParts := sys["parts"].([]any)
	assert.Equal(t, "Be brief.", sysParts[0].(map[string]any)["text"])

	c := contents(t, s)
	if assert.Len(t, c, 1, "the system message must not remain in contents") {
		assert.Equal(t, "user", c[0].(map[string]any)["role"])
	}
}

func TestInvoke_SendsToolSchema(t *testing.T) {
	s := newStub(t)
	s.resp = textReply
	p := newProvider(t, s)

	_, err := p.Invoke(context.Background(), model.Request{
		Model:    "gemini-test",
		Messages: []model.Message{model.UserMessage("Hi")},
		Tools: []*model.Tool{model.NewTool("get_weather", "Look up the weather",
			[]byte(`{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}`))},
	})
	require.NoError(t, err)

	tools := s.body["tools"].([]any)
	if assert.Len(t, tools, 1) {
		decls := tools[0].(map[string]any)["functionDeclarations"].([]any)
		fn := decls[0].(map[string]any)
		assert.Equal(t, "get_weather", fn["name"])
		assert.Equal(t, "Look up the weather", fn["description"])
		// The raw JSON schema passed through parametersJsonSchema.
		schema := fn["parametersJsonSchema"].(map[string]any)
		assert.Contains(t, schema["properties"].(map[string]any), "city")
	}
}

// The load-bearing structural transform: a tool call and its result must map to
// a model functionCall turn and a user functionResponse turn, with the function
// name resolved onto the response (Gemini needs it; the result block lacks it).
func TestInvoke_ToolCallAndResult(t *testing.T) {
	s := newStub(t)
	s.resp = textReply
	p := newProvider(t, s)

	_, err := p.Invoke(context.Background(), model.Request{
		Model: "gemini-test",
		Messages: []model.Message{
			model.UserMessage("weather?"),
			{Role: model.RoleAssistant, Blocks: []model.Block{
				model.ToolCallBlock("call_1", "get_weather", json.RawMessage(`{"city":"Ghent"}`)),
			}},
			model.ToolResultMessage("call_1", model.TextResult("18C")),
		},
	})
	require.NoError(t, err)

	c := contents(t, s)
	if assert.Len(t, c, 3) { // user, model(functionCall), user(functionResponse)
		modelTurn := c[1].(map[string]any)
		assert.Equal(t, "model", modelTurn["role"])
		fc := modelTurn["parts"].([]any)[0].(map[string]any)["functionCall"].(map[string]any)
		assert.Equal(t, "get_weather", fc["name"])

		respTurn := c[2].(map[string]any)
		assert.Equal(t, "user", respTurn["role"], "function responses live in a user turn")
		fr := respTurn["parts"].([]any)[0].(map[string]any)["functionResponse"].(map[string]any)
		assert.Equal(t, "get_weather", fr["name"], "the function name is resolved from the matching call")
		assert.Equal(t, "18C", fr["response"].(map[string]any)["result"])
	}
}

// A parsed tool call comes back as a tool-call block; a missing Gemini ID is
// synthesized so the loop can key the result.
func TestInvoke_ParsesFunctionCall(t *testing.T) {
	s := newStub(t)
	s.resp = `{"candidates":[{"content":{"role":"model","parts":[
	  {"text":"let me check"},
	  {"functionCall":{"name":"get_weather","args":{"city":"Ghent"}}}
	]},"finishReason":"STOP"}],
	"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":8,"totalTokenCount":18}}`
	p := newProvider(t, s)

	resp, err := p.Invoke(context.Background(), model.Request{
		Model:    "gemini-test",
		Messages: []model.Message{model.UserMessage("weather?")},
	})
	require.NoError(t, err)
	assert.Equal(t, "let me check", resp.Message.Text())
	calls := resp.Message.ToolCalls()
	if assert.Len(t, calls, 1) {
		assert.Equal(t, "get_weather", calls[0].Name)
		assert.NotEmpty(t, calls[0].ID, "a synthesized ID lets the loop route the result")
		assert.JSONEq(t, `{"city":"Ghent"}`, string(calls[0].Arguments))
	}
}

// Multimodal input: a user image block must ride through as an inline_data part
// with its MIME type and base64 bytes — the whole reason Gemini is here.
func TestInvoke_UserImageInlineData(t *testing.T) {
	s := newStub(t)
	s.resp = textReply
	p := newProvider(t, s)

	pngBytes := []byte{0x89, 0x50, 0x4e, 0x47} // not a real PNG; the bytes just round-trip
	_, err := p.Invoke(context.Background(), model.Request{
		Model: "gemini-test",
		Messages: []model.Message{
			model.UserContent(model.TextBlock("what is this?"), model.ImageBlock("image/png", pngBytes)),
		},
	})
	require.NoError(t, err)

	c := contents(t, s)
	parts := c[0].(map[string]any)["parts"].([]any)
	if assert.Len(t, parts, 2) {
		assert.Equal(t, "what is this?", parts[0].(map[string]any)["text"])

		inline := parts[1].(map[string]any)["inlineData"].(map[string]any)
		assert.Equal(t, "image/png", inline["mimeType"])
		assert.Equal(t, base64.StdEncoding.EncodeToString(pngBytes), inline["data"])
	}
}

// An OutputSchema becomes Gemini's controlled generation: a JSON MIME type plus a
// JSON Schema, both under generationConfig; the terminal text is StructuredOutput.
func TestInvoke_StructuredOutput(t *testing.T) {
	s := newStub(t)
	s.resp = `{"candidates":[{"content":{"role":"model","parts":[{"text":"{\"city\":\"Ghent\",\"temp\":18}"}]},"finishReason":"STOP"}],
	  "usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":4,"totalTokenCount":9}}`
	p := newProvider(t, s)

	resp, err := p.Invoke(context.Background(), model.Request{
		Model:    "gemini-test",
		Messages: []model.Message{model.UserMessage("weather?")},
		OutputSchema: &model.OutputSchema{
			Name:   "weather",
			Schema: json.RawMessage(`{"type":"object","properties":{"city":{"type":"string"},"temp":{"type":"integer"}},"required":["city","temp"]}`),
		},
	})
	require.NoError(t, err)

	gc := s.body["generationConfig"].(map[string]any)
	assert.Equal(t, "application/json", gc["responseMimeType"])
	assert.Contains(t, gc["responseJsonSchema"].(map[string]any)["properties"], "city")
	assert.JSONEq(t, `{"city":"Ghent","temp":18}`, string(resp.StructuredOutput))
}

// Extended thinking round-trips: a thought part (with signature) comes back as a
// thinking block, and on the next turn it is echoed back — thought flag and
// signature intact — ahead of the tool call, mirroring the Anthropic guarantee.
func TestThinking_RoundTrips(t *testing.T) {
	sig := base64.StdEncoding.EncodeToString([]byte("opaque-sig"))
	s := newStub(t)
	s.resp = `{"candidates":[{"content":{"role":"model","parts":[
	  {"thought":true,"text":"I should check the weather.","thoughtSignature":"` + sig + `"},
	  {"functionCall":{"name":"get_weather","args":{"city":"Ghent"}}}
	]},"finishReason":"STOP"}],
	"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":8,"totalTokenCount":18}}`
	p := newProvider(t, s)

	resp, err := p.Invoke(context.Background(), model.Request{
		Model:    "gemini-test",
		Messages: []model.Message{model.UserMessage("weather in Ghent?")},
	})
	require.NoError(t, err)

	// Direction 1: order preserved, signature captured.
	assert.Equal(t, []model.BlockKind{model.BlockThinking, model.BlockToolCall},
		[]model.BlockKind{resp.Message.Blocks[0].Kind, resp.Message.Blocks[1].Kind})
	assert.Equal(t, "I should check the weather.", resp.Message.Blocks[0].Text)
	assert.Equal(t, sig, resp.Message.Blocks[0].Signature)

	// Direction 2: echo the assistant turn back and assert the thought part carries
	// the signature ahead of the functionCall.
	s2 := newStub(t)
	s2.resp = textReply
	p2 := newProvider(t, s2)
	_, err = p2.Invoke(context.Background(), model.Request{
		Model: "gemini-test",
		Messages: []model.Message{
			model.UserMessage("weather in Ghent?"),
			resp.Message,
			model.ToolResultMessage(resp.Message.ToolCalls()[0].ID, model.TextResult("18C")),
		},
	})
	require.NoError(t, err)

	modelTurn := contents(t, s2)[1].(map[string]any)
	parts := modelTurn["parts"].([]any)
	first := parts[0].(map[string]any)
	assert.Equal(t, true, first["thought"], "the thought part is echoed back")
	assert.Equal(t, sig, first["thoughtSignature"], "the signature survives verbatim")
	assert.Contains(t, parts[1].(map[string]any), "functionCall", "tool call follows the thought")
}

type capturingSink struct{ deltas []model.StreamDelta }

func (c *capturingSink) OnDelta(_ context.Context, d model.StreamDelta) error {
	c.deltas = append(c.deltas, d)
	return nil
}

func sseServer(t *testing.T, chunks ...string) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(200)
		fl, _ := w.(http.Flusher)
		for _, c := range chunks {
			_, _ = io.WriteString(w, "data: "+c+"\n\n")
			if fl != nil {
				fl.Flush()
			}
		}
	}))
	t.Cleanup(srv.Close)
	return srv
}

// Streamed text deltas reach the sink, and the aggregated response carries the
// full text and usage — the invariant that makes streaming safe for replay.
func TestInvokeStream_TextDeltasAndAggregate(t *testing.T) {
	srv := sseServer(t,
		`{"candidates":[{"content":{"role":"model","parts":[{"text":"Hel"}]}}]}`,
		`{"candidates":[{"content":{"role":"model","parts":[{"text":"lo."}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":2,"totalTokenCount":7}}`,
	)
	t.Cleanup(srv.Close)
	p, err := vertexprovider.New(vertexprovider.WithAPIKey("test-key"), vertexprovider.WithBaseURL(srv.URL))
	require.NoError(t, err)

	sink := &capturingSink{}
	resp, err := p.InvokeStream(context.Background(), model.Request{
		Model:    "gemini-test",
		Messages: []model.Message{model.UserMessage("hi")},
	}, sink)
	require.NoError(t, err)

	var streamed string
	for _, d := range sink.deltas {
		streamed += d.Text
	}
	assert.Equal(t, "Hello.", streamed)
	assert.Equal(t, "Hello.", resp.Message.Text())
	assert.Equal(t, int64(7), resp.Usage.TotalTokens)
}

func TestInvoke_ErrorMapping(t *testing.T) {
	for _, tc := range []struct {
		name      string
		status    int
		retryable bool
	}{
		{"rate limit", 429, true},
		{"server error", 503, true},
		{"bad request", 400, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			s := newStub(t)
			s.status = tc.status
			s.resp = `{"error":{"code":` + itoa(tc.status) + `,"message":"boom","status":"ERR"}}`
			p := newProvider(t, s)

			_, err := p.Invoke(context.Background(), model.Request{
				Model:    "gemini-test",
				Messages: []model.Message{model.UserMessage("hi")},
			})
			require.Error(t, err)
			var apiErr *model.APIError
			require.ErrorAs(t, err, &apiErr)
			assert.Equal(t, tc.status, apiErr.StatusCode)
			assert.Equal(t, tc.retryable, apiErr.Retryable())
		})
	}
}

func itoa(n int) string {
	return map[int]string{400: "400", 429: "429", 503: "503"}[n]
}
