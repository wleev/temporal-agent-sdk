package anthropic_test

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	sdk "github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
	"github.com/stretchr/testify/require"

	"github.com/wleev/temporal-agent-sdk/model"
	anthropicprovider "github.com/wleev/temporal-agent-sdk/model/anthropic"
)

// stubServer stands in for the Anthropic Messages endpoint and captures the
// request body, so the tests can assert the exact wire format the provider
// produces from neutral input. The structural transforms — system hoisting, tool
// result coalescing — compile fine when wrong and only show up in the body.
type stubServer struct {
	*httptest.Server
	body   map[string]any
	status int
	resp   string
	header map[string]string
}

func newStub(t *testing.T) *stubServer {
	t.Helper()
	s := &stubServer{status: 200, header: map[string]string{}}
	s.Server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, err := io.ReadAll(r.Body)
		require.NoError(t, err)
		if len(raw) > 0 {
			require.NoError(t, json.Unmarshal(raw, &s.body))
		}
		w.Header().Set("Content-Type", "application/json")
		for k, v := range s.header {
			w.Header().Set(k, v)
		}
		w.WriteHeader(s.status)
		_, _ = io.WriteString(w, s.resp)
	}))
	t.Cleanup(s.Close)
	return s
}

func newProvider(t *testing.T, s *stubServer, opts ...anthropicprovider.Option) *anthropicprovider.Provider {
	t.Helper()
	opts = append([]anthropicprovider.Option{
		anthropicprovider.WithBaseURL(s.URL),
		anthropicprovider.WithAPIKey("test-key"),
	}, opts...)
	p, err := anthropicprovider.New(opts...)
	require.NoError(t, err)
	return p
}

const textReply = `{
  "id":"msg_1","type":"message","role":"assistant","model":"claude-test",
  "content":[{"type":"text","text":"Hello."}],
  "stop_reason":"end_turn",
  "usage":{"input_tokens":11,"output_tokens":3}
}`

// A system message must be hoisted to the top-level system parameter — Anthropic
// has no system role — and max_tokens must be present because Anthropic requires
// it even though the neutral seam leaves it optional.
func TestInvoke_HoistsSystemAndDefaultsMaxTokens(t *testing.T) {
	s := newStub(t)
	s.resp = textReply
	p := newProvider(t, s)

	resp, err := p.Invoke(context.Background(), model.Request{
		Model: "claude-test",
		Messages: []model.Message{
			model.SystemMessage("Be brief."),
			model.UserMessage("Hi"),
		},
	})
	require.NoError(t, err)
	require.Equal(t, "Hello.", resp.Message.Text())
	require.Equal(t, "end_turn", resp.FinishReason)
	require.Equal(t, int64(14), resp.Usage.TotalTokens)
	require.Equal(t, int64(11), resp.Usage.PromptTokens)

	// System hoisted out of messages into the top-level array.
	sys := s.body["system"].([]any)
	require.Len(t, sys, 1)
	require.Equal(t, "Be brief.", sys[0].(map[string]any)["text"])

	msgs := s.body["messages"].([]any)
	require.Len(t, msgs, 1, "the system message must not remain in the message list")
	require.Equal(t, "user", msgs[0].(map[string]any)["role"])

	// max_tokens defaulted (required by Anthropic).
	require.EqualValues(t, anthropicprovider.DefaultMaxTokens, s.body["max_tokens"])
}

func TestInvoke_ExplicitMaxTokensUsed(t *testing.T) {
	s := newStub(t)
	s.resp = textReply
	p := newProvider(t, s)

	max := int64(256)
	_, err := p.Invoke(context.Background(), model.Request{
		Model:    "claude-test",
		Messages: []model.Message{model.UserMessage("Hi")},
		Settings: model.Settings{MaxTokens: &max},
	})
	require.NoError(t, err)
	require.EqualValues(t, 256, s.body["max_tokens"])
}

// The tool schema must be decomposed into Anthropic's input_schema shape:
// properties and required split out, and other keywords (additionalProperties)
// carried through.
func TestInvoke_DecomposesToolSchema(t *testing.T) {
	s := newStub(t)
	s.resp = textReply
	p := newProvider(t, s)

	_, err := p.Invoke(context.Background(), model.Request{
		Model:    "claude-test",
		Messages: []model.Message{model.UserMessage("Hi")},
		Tools: []*model.Tool{model.NewTool("get_weather", "Look up the weather",
			[]byte(`{"type":"object","properties":{"city":{"type":"string"}},`+
				`"required":["city"],"additionalProperties":false}`))},
	})
	require.NoError(t, err)

	tools := s.body["tools"].([]any)
	require.Len(t, tools, 1)
	tl := tools[0].(map[string]any)
	require.Equal(t, "get_weather", tl["name"])
	require.Equal(t, "Look up the weather", tl["description"])

	schema := tl["input_schema"].(map[string]any)
	require.Equal(t, "object", schema["type"])
	require.Contains(t, schema["properties"].(map[string]any), "city")
	require.Equal(t, []any{"city"}, schema["required"])
	// additionalProperties is not a field Anthropic models, so it must ride
	// through as an extra key.
	require.Equal(t, false, schema["additionalProperties"])
}

// An OutputSchema must become Anthropic's output_config, and the terminal text
// content must come back as StructuredOutput.
func TestInvoke_StructuredOutput(t *testing.T) {
	s := newStub(t)
	s.resp = `{"id":"msg_1","type":"message","role":"assistant","model":"claude-test",
	  "content":[{"type":"text","text":"{\"city\":\"Ghent\",\"temp\":18}"}],
	  "stop_reason":"end_turn","usage":{"input_tokens":5,"output_tokens":4}}`
	p := newProvider(t, s)

	resp, err := p.Invoke(context.Background(), model.Request{
		Model:    "claude-test",
		Messages: []model.Message{model.UserMessage("weather?")},
		OutputSchema: &model.OutputSchema{
			Name:   "weather",
			Schema: json.RawMessage(`{"type":"object","properties":{"city":{"type":"string"},"temp":{"type":"integer"}},"required":["city","temp"]}`),
		},
	})
	require.NoError(t, err)

	// Request carried output_config.format.schema (Anthropic's native mechanism).
	oc := s.body["output_config"].(map[string]any)
	format := oc["format"].(map[string]any)
	require.Equal(t, "json_schema", format["type"])
	require.Contains(t, format["schema"].(map[string]any)["properties"], "city")
	require.NotContains(t, s.body, "response_format", "Anthropic uses output_config, not response_format")

	require.JSONEq(t, `{"city":"Ghent","temp":18}`, string(resp.StructuredOutput))
}

type capturingSink struct{ deltas []model.StreamDelta }

func (c *capturingSink) OnDelta(_ context.Context, d model.StreamDelta) error {
	c.deltas = append(c.deltas, d)
	return nil
}

// sseServer serves named SSE events (Anthropic streams use `event:` + `data:`).
func sseServer(t *testing.T, events ...[2]string) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(200)
		fl, _ := w.(http.Flusher)
		for _, e := range events {
			_, _ = io.WriteString(w, "event: "+e[0]+"\ndata: "+e[1]+"\n\n")
			if fl != nil {
				fl.Flush()
			}
		}
	}))
	t.Cleanup(srv.Close)
	return srv
}

// Streamed text deltas reach the sink, and the aggregated Message carries the
// full content and usage.
func TestInvokeStream_TextDeltasAndAggregate(t *testing.T) {
	srv := sseServer(t,
		[2]string{"message_start", `{"type":"message_start","message":{"id":"m","type":"message","role":"assistant","model":"claude-test","content":[],"stop_reason":null,"usage":{"input_tokens":5,"output_tokens":0}}}`},
		[2]string{"content_block_start", `{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`},
		[2]string{"content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hel"}}`},
		[2]string{"content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"lo."}}`},
		[2]string{"content_block_stop", `{"type":"content_block_stop","index":0}`},
		[2]string{"message_delta", `{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":2}}`},
		[2]string{"message_stop", `{"type":"message_stop"}`},
	)
	p := newProvider(t, &stubServer{Server: srv})

	sink := &capturingSink{}
	resp, err := p.InvokeStream(context.Background(), model.Request{
		Model:    "claude-test",
		Messages: []model.Message{model.UserMessage("hi")},
	}, sink)
	require.NoError(t, err)

	var streamed string
	for _, d := range sink.deltas {
		streamed += d.Text
	}
	require.Equal(t, "Hello.", streamed)
	require.Equal(t, "Hello.", resp.Message.Text())
	require.Equal(t, int64(5), resp.Usage.PromptTokens)
	require.Equal(t, int64(2), resp.Usage.CompletionTokens)
}

func TestInvokeStream_ForwardsToolInputDeltas(t *testing.T) {
	srv := sseServer(t,
		[2]string{"message_start", `{"type":"message_start","message":{"id":"m","type":"message","role":"assistant","model":"claude-test","content":[],"stop_reason":null,"usage":{"input_tokens":5,"output_tokens":0}}}`},
		[2]string{"content_block_start", `{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_1","name":"get_weather","input":{}}}`},
		[2]string{"content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"ci"}}`},
		[2]string{"content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"ty\":\"Ghent\"}"}}`},
		[2]string{"content_block_stop", `{"type":"content_block_stop","index":0}`},
		[2]string{"message_delta", `{"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":4}}`},
		[2]string{"message_stop", `{"type":"message_stop"}`},
	)
	p := newProvider(t, &stubServer{Server: srv})

	sink := &capturingSink{}
	resp, err := p.InvokeStream(context.Background(), model.Request{
		Model:    "claude-test",
		Messages: []model.Message{model.UserMessage("weather?")},
	}, sink)
	require.NoError(t, err)

	var args string
	for _, d := range sink.deltas {
		if d.ToolCallIndex == 0 {
			args += d.ArgsFragment
		}
	}
	require.JSONEq(t, `{"city":"Ghent"}`, args)

	require.Len(t, resp.Message.ToolCalls(), 1)
	require.Equal(t, "get_weather", resp.Message.ToolCalls()[0].Name)
	require.JSONEq(t, `{"city":"Ghent"}`, string(resp.Message.ToolCalls()[0].Arguments))
}

func TestInvoke_ParsesToolUse(t *testing.T) {
	s := newStub(t)
	s.resp = `{
	  "id":"msg_1","type":"message","role":"assistant","model":"claude-test",
	  "content":[
	    {"type":"text","text":"let me check"},
	    {"type":"tool_use","id":"toolu_1","name":"get_weather","input":{"city":"Ghent"}}
	  ],
	  "stop_reason":"tool_use",
	  "usage":{"input_tokens":10,"output_tokens":8}
	}`
	p := newProvider(t, s)

	resp, err := p.Invoke(context.Background(), model.Request{
		Model:    "claude-test",
		Messages: []model.Message{model.UserMessage("Weather?")},
	})
	require.NoError(t, err)
	require.Equal(t, "tool_use", resp.FinishReason)
	require.Equal(t, "let me check", resp.Message.Text())
	require.Len(t, resp.Message.ToolCalls(), 1)

	tc := resp.Message.ToolCalls()[0]
	require.Equal(t, "toolu_1", tc.ID)
	require.Equal(t, "get_weather", tc.Name)
	require.JSONEq(t, `{"city":"Ghent"}`, string(tc.Arguments))
}

// Extended thinking plus tools is the reason the block model exists. Anthropic
// emits a thinking block (with a signature) before the tool_use, and REQUIRES
// that same thinking block — signature intact, ahead of the tool_use — echoed
// back on the following turn or it rejects the request. The flat message model
// had nowhere to hold thinking, so this round trip was impossible. Verify both
// directions: the response captures thinking in order, and the next request
// sends it back before the tool_use.
func TestThinking_RoundTripsBeforeToolUse(t *testing.T) {
	s := newStub(t)
	s.resp = `{
	  "id":"msg_1","type":"message","role":"assistant","model":"claude-test",
	  "content":[
	    {"type":"thinking","thinking":"The user wants Ghent weather; I'll call the tool.","signature":"sig-abc123"},
	    {"type":"text","text":"Let me check."},
	    {"type":"tool_use","id":"toolu_1","name":"get_weather","input":{"city":"Ghent"}}
	  ],
	  "stop_reason":"tool_use",
	  "usage":{"input_tokens":10,"output_tokens":8}
	}`
	p := newProvider(t, s)

	resp, err := p.Invoke(context.Background(), model.Request{
		Model:    "claude-test",
		Messages: []model.Message{model.UserMessage("Weather in Ghent?")},
	})
	require.NoError(t, err)

	// Direction 1: the response preserves block order — thinking, text, tool_call —
	// and captures the thinking signature (opaque, but load-bearing on echo).
	require.Equal(t,
		[]model.BlockKind{model.BlockThinking, model.BlockText, model.BlockToolCall},
		[]model.BlockKind{resp.Message.Blocks[0].Kind, resp.Message.Blocks[1].Kind, resp.Message.Blocks[2].Kind},
	)
	require.Equal(t, "The user wants Ghent weather; I'll call the tool.", resp.Message.Blocks[0].Text)
	require.Equal(t, "sig-abc123", resp.Message.Blocks[0].Signature)

	// Direction 2: feed that assistant turn plus its tool result back, and the
	// request must send the thinking block — signature intact — BEFORE the tool_use.
	s2 := newStub(t)
	s2.resp = textReply
	p2 := newProvider(t, s2)
	_, err = p2.Invoke(context.Background(), model.Request{
		Model: "claude-test",
		Messages: []model.Message{
			model.UserMessage("Weather in Ghent?"),
			resp.Message,
			model.ToolResultMessage("toolu_1", model.TextResult("18C")),
		},
	})
	require.NoError(t, err)

	assistant := s2.body["messages"].([]any)[1].(map[string]any)
	content := assistant["content"].([]any)
	require.Equal(t, "thinking", content[0].(map[string]any)["type"],
		"the thinking block must be echoed back first")
	require.Equal(t, "sig-abc123", content[0].(map[string]any)["signature"],
		"the signature must survive the round trip verbatim")
	require.Equal(t, "tool_use", content[2].(map[string]any)["type"],
		"the tool_use must follow the thinking block, not precede it")
}

// The load-bearing structural transform: the neutral loop emits one message per
// tool result (OpenAI's shape), but Anthropic requires all tool results for an
// assistant's parallel tool_use turn to sit in a single user message as
// tool_result blocks. Verify the coalescing.
func TestInvoke_CoalescesToolResultsIntoOneUserMessage(t *testing.T) {
	s := newStub(t)
	s.resp = textReply
	p := newProvider(t, s)

	_, err := p.Invoke(context.Background(), model.Request{
		Model: "claude-test",
		Messages: []model.Message{
			model.UserMessage("weather in two cities?"),
			{Role: model.RoleAssistant, Blocks: []model.Block{
				model.ToolCallBlock("a", "get_weather", json.RawMessage(`{"city":"Ghent"}`)),
				model.ToolCallBlock("b", "get_weather", json.RawMessage(`{"city":"Bruges"}`)),
			}},
			model.ToolResultMessage("a", model.TextResult("18C")),
			model.ToolResultMessage("b", model.TextResult("17C")),
		},
	})
	require.NoError(t, err)

	msgs := s.body["messages"].([]any)
	// user, assistant(2 tool_use), user(2 tool_result) — the two tool results
	// must NOT be two separate messages.
	require.Len(t, msgs, 3, "the two tool results must coalesce into one user message")

	assistant := msgs[1].(map[string]any)
	require.Equal(t, "assistant", assistant["role"])
	aContent := assistant["content"].([]any)
	require.Len(t, aContent, 2, "both tool calls in the assistant turn")
	require.Equal(t, "tool_use", aContent[0].(map[string]any)["type"])
	require.Equal(t, "get_weather", aContent[0].(map[string]any)["name"])

	results := msgs[2].(map[string]any)
	require.Equal(t, "user", results["role"], "tool results live in a user message")
	rContent := results["content"].([]any)
	require.Len(t, rContent, 2, "both tool results in a single user message")
	require.Equal(t, "tool_result", rContent[0].(map[string]any)["type"])
	require.Equal(t, "a", rContent[0].(map[string]any)["tool_use_id"])
	require.Equal(t, "b", rContent[1].(map[string]any)["tool_use_id"])
}

// A user image block becomes a base64 image content block; text-only user
// messages keep their plain-text form.
func TestInvoke_UserImageBecomesImageBlock(t *testing.T) {
	s := newStub(t)
	s.resp = textReply
	p := newProvider(t, s)

	png := []byte{0x89, 0x50, 0x4e, 0x47}
	_, err := p.Invoke(context.Background(), model.Request{
		Model: "claude-test",
		Messages: []model.Message{
			model.UserContent(model.TextBlock("what is this?"), model.ImageBlock("image/png", png)),
		},
	})
	require.NoError(t, err)

	content := s.body["messages"].([]any)[0].(map[string]any)["content"].([]any)
	require.Len(t, content, 2)
	require.Equal(t, "text", content[0].(map[string]any)["type"])

	img := content[1].(map[string]any)
	require.Equal(t, "image", img["type"])
	src := img["source"].(map[string]any)
	require.Equal(t, "base64", src["type"])
	require.Equal(t, "image/png", src["media_type"])
	require.Equal(t, base64.StdEncoding.EncodeToString(png), src["data"])
}

// Audio has no Anthropic representation, so it degrades to a named placeholder.
func TestInvoke_UserAudioBecomesPlaceholder(t *testing.T) {
	s := newStub(t)
	s.resp = textReply
	p := newProvider(t, s)

	_, err := p.Invoke(context.Background(), model.Request{
		Model:    "claude-test",
		Messages: []model.Message{model.UserContent(model.AudioBlock("audio/wav", []byte{1, 2}))},
	})
	require.NoError(t, err)

	content := s.body["messages"].([]any)[0].(map[string]any)["content"].([]any)
	require.Equal(t, "text", content[0].(map[string]any)["type"])
	require.Contains(t, content[0].(map[string]any)["text"], "audio omitted")
}

// WithParams is the construction-time home for Anthropic-specific request fields
// the neutral Settings does not model — top_k, stop sequences, thinking.
func TestWithParams_CustomizesRequest(t *testing.T) {
	s := newStub(t)
	s.resp = textReply
	p := newProvider(t, s, anthropicprovider.WithParams(func(pr *sdk.MessageNewParams) {
		pr.TopK = sdk.Int(40)
		pr.StopSequences = []string{"STOP"}
	}))

	_, err := p.Invoke(context.Background(), model.Request{
		Model:    "claude-test",
		Messages: []model.Message{model.UserMessage("Hi")},
		Settings: model.Settings{Temperature: model.Ptr(0.5)},
	})
	require.NoError(t, err)

	require.EqualValues(t, 40, s.body["top_k"])
	require.Equal(t, []any{"STOP"}, s.body["stop_sequences"])
	require.InDelta(t, 0.5, s.body["temperature"], 0.001)
}

// NewWithClient is the full-control injection point.
func TestNewWithClient_UsesInjectedClient(t *testing.T) {
	s := newStub(t)
	s.resp = textReply

	client := sdk.NewClient(
		option.WithBaseURL(s.URL),
		option.WithAPIKey("injected"),
		option.WithMaxRetries(0),
	)
	p, err := anthropicprovider.NewWithClient(client, anthropicprovider.WithName("custom"))
	require.NoError(t, err)
	require.Equal(t, "custom", p.Name())

	resp, err := p.Invoke(context.Background(), model.Request{
		Model:    "claude-test",
		Messages: []model.Message{model.UserMessage("Hi")},
	})
	require.NoError(t, err)
	require.Equal(t, "Hello.", resp.Message.Text())
}

func TestInvoke_OmitsUnsetSettings(t *testing.T) {
	s := newStub(t)
	s.resp = textReply
	p := newProvider(t, s)

	_, err := p.Invoke(context.Background(), model.Request{
		Model:    "claude-test",
		Messages: []model.Message{model.UserMessage("Hi")},
	})
	require.NoError(t, err)
	require.NotContains(t, s.body, "temperature", "unset temperature must be omitted")
	require.NotContains(t, s.body, "top_p")
}

func TestInvoke_ErrorMapping(t *testing.T) {
	for _, tc := range []struct {
		name          string
		status        int
		headers       map[string]string
		wantRetryable bool
		wantRetryFor  time.Duration
	}{
		{name: "400 permanent", status: 400, wantRetryable: false},
		{name: "401 permanent", status: 401, wantRetryable: false},
		{name: "404 permanent", status: 404, wantRetryable: false},
		{name: "429 rate limit retryable", status: 429, wantRetryable: true},
		{name: "500 retryable", status: 500, wantRetryable: true},
		{name: "529 overloaded retryable", status: 529, wantRetryable: true},
		{
			name:          "retry-after honored",
			status:        429,
			headers:       map[string]string{"retry-after": "3"},
			wantRetryable: true,
			wantRetryFor:  3 * time.Second,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			s := newStub(t)
			s.status = tc.status
			s.resp = `{"type":"error","error":{"type":"invalid_request_error","message":"boom"}}`
			s.header = tc.headers
			p := newProvider(t, s)

			_, err := p.Invoke(context.Background(), model.Request{
				Model:    "claude-test",
				Messages: []model.Message{model.UserMessage("Hi")},
			})
			require.Error(t, err)

			var apiErr *model.APIError
			require.ErrorAs(t, err, &apiErr)
			require.Equal(t, tc.status, apiErr.StatusCode)
			require.Equal(t, tc.wantRetryable, apiErr.Retryable(),
				"529 overloaded and 429 must be retryable; 4xx must not")
			if tc.wantRetryFor > 0 {
				require.Equal(t, tc.wantRetryFor, apiErr.RetryAfter)
			}
		})
	}
}

func TestInvoke_TransportFailureIsRetryable(t *testing.T) {
	p, err := anthropicprovider.New(
		anthropicprovider.WithBaseURL("http://127.0.0.1:1"),
		anthropicprovider.WithAPIKey("k"),
	)
	require.NoError(t, err)

	_, err = p.Invoke(context.Background(), model.Request{
		Model:    "claude-test",
		Messages: []model.Message{model.UserMessage("Hi")},
	})
	require.Error(t, err)

	var apiErr *model.APIError
	require.ErrorAs(t, err, &apiErr)
	require.Equal(t, 0, apiErr.StatusCode)
	require.True(t, apiErr.Retryable())
}

// Temporal owns retry, so the client must make one HTTP call per attempt.
func TestInvoke_DoesNotRetryClientSide(t *testing.T) {
	var calls int
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls++
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(500)
		_, _ = io.WriteString(w, `{"type":"error","error":{"type":"api_error","message":"boom"}}`)
	}))
	defer srv.Close()

	p, err := anthropicprovider.New(anthropicprovider.WithBaseURL(srv.URL), anthropicprovider.WithAPIKey("k"))
	require.NoError(t, err)

	_, _ = p.Invoke(context.Background(), model.Request{
		Model:    "claude-test",
		Messages: []model.Message{model.UserMessage("Hi")},
	})
	require.Equal(t, 1, calls, "the provider must make exactly one call per attempt; Temporal owns retry")
}

func TestInvoke_RejectsUnknownRole(t *testing.T) {
	s := newStub(t)
	s.resp = textReply
	p := newProvider(t, s)

	_, err := p.Invoke(context.Background(), model.Request{
		Model:    "claude-test",
		Messages: []model.Message{{Role: "wizard", Blocks: []model.Block{model.TextBlock("Hi")}}},
	})
	require.ErrorContains(t, err, "unknown role")
}

func TestNew_NameDefaultAndOverride(t *testing.T) {
	p, err := anthropicprovider.New(anthropicprovider.WithAPIKey("k"))
	require.NoError(t, err)
	require.Equal(t, "anthropic", p.Name())

	p2, err := anthropicprovider.New(anthropicprovider.WithName("claude-eu"), anthropicprovider.WithAPIKey("k"))
	require.NoError(t, err)
	require.Equal(t, "claude-eu", p2.Name())
}
