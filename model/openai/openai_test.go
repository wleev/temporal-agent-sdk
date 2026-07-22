package openai_test

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	oai "github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
	"github.com/stretchr/testify/require"

	"github.com/wleev/temporal-agent-sdk/model"
	oaiprovider "github.com/wleev/temporal-agent-sdk/model/openai"
)

// stubServer stands in for an OpenAI-compatible endpoint and captures the
// request body, so the tests can assert the exact wire format. Several of these
// mappings — the tool-message argument order especially — compile fine when
// wrong and fail only against a real API.
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

func newProvider(t *testing.T, s *stubServer, opts ...oaiprovider.Option) *oaiprovider.Provider {
	t.Helper()
	opts = append([]oaiprovider.Option{
		oaiprovider.WithBaseURL(s.URL),
		oaiprovider.WithAPIKey("test-key"),
	}, opts...)
	p, err := oaiprovider.New(opts...)
	require.NoError(t, err)
	return p
}

const plainReply = `{
  "id":"c1","object":"chat.completion","model":"m",
  "choices":[{"index":0,"finish_reason":"stop","message":{"role":"assistant","content":"Hello."}}],
  "usage":{"prompt_tokens":11,"completion_tokens":3,"total_tokens":14}
}`

func TestInvoke_PlainCompletion(t *testing.T) {
	s := newStub(t)
	s.resp = plainReply
	p := newProvider(t, s)

	resp, err := p.Invoke(context.Background(), model.Request{
		Model: "gpt-test",
		Messages: []model.Message{
			model.SystemMessage("Be brief."),
			model.UserMessage("Hi"),
		},
	})
	require.NoError(t, err)
	require.Equal(t, "Hello.", resp.Message.Text())
	require.Equal(t, model.RoleAssistant, resp.Message.Role)
	require.Equal(t, "stop", resp.FinishReason)
	require.Equal(t, int64(14), resp.Usage.TotalTokens)
	require.Equal(t, int64(11), resp.Usage.PromptTokens)

	require.Equal(t, "gpt-test", s.body["model"])
	msgs := s.body["messages"].([]any)
	require.Len(t, msgs, 2)
	require.Equal(t, "system", msgs[0].(map[string]any)["role"])
	require.Equal(t, "user", msgs[1].(map[string]any)["role"])
}

func TestInvoke_SendsToolSchema(t *testing.T) {
	s := newStub(t)
	s.resp = plainReply
	p := newProvider(t, s)

	_, err := p.Invoke(context.Background(), model.Request{
		Model:    "gpt-test",
		Messages: []model.Message{model.UserMessage("Hi")},
		Tools: []*model.Tool{model.NewTool("get_weather", "Look up the weather",
			[]byte(`{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}`))},
	})
	require.NoError(t, err)

	tools := s.body["tools"].([]any)
	require.Len(t, tools, 1)
	tl := tools[0].(map[string]any)
	require.Equal(t, "function", tl["type"])

	fn := tl["function"].(map[string]any)
	require.Equal(t, "get_weather", fn["name"])
	require.Equal(t, "Look up the weather", fn["description"])

	// The schema must survive the round trip through map[string]any intact.
	params := fn["parameters"].(map[string]any)
	require.Equal(t, "object", params["type"])
	require.Contains(t, params["properties"].(map[string]any), "city")
}

// An OutputSchema on the request must become a response_format json_schema, and
// the terminal message content must come back as StructuredOutput.
func TestInvoke_StructuredOutput(t *testing.T) {
	s := newStub(t)
	s.resp = `{"id":"c1","object":"chat.completion","model":"m",
	  "choices":[{"index":0,"finish_reason":"stop","message":{"role":"assistant",
	    "content":"{\"city\":\"Ghent\",\"temp\":18}"}}],
	  "usage":{"prompt_tokens":5,"completion_tokens":4,"total_tokens":9}}`
	p := newProvider(t, s)

	resp, err := p.Invoke(context.Background(), model.Request{
		Model:    "gpt-test",
		Messages: []model.Message{model.UserMessage("weather?")},
		OutputSchema: &model.OutputSchema{
			Name:   "weather",
			Strict: true,
			Schema: json.RawMessage(`{"type":"object","properties":{"city":{"type":"string"},"temp":{"type":"integer"}},"required":["city","temp"],"additionalProperties":false}`),
		},
	})
	require.NoError(t, err)

	// Request carried response_format json_schema with the schema and strict.
	rf := s.body["response_format"].(map[string]any)
	require.Equal(t, "json_schema", rf["type"])
	js := rf["json_schema"].(map[string]any)
	require.Equal(t, "weather", js["name"])
	require.Equal(t, true, js["strict"])
	require.Contains(t, js["schema"].(map[string]any)["properties"], "city")

	// Terminal content came back as StructuredOutput.
	require.JSONEq(t, `{"city":"Ghent","temp":18}`, string(resp.StructuredOutput))
}

// With tools present, a tool-call turn must NOT be treated as structured output.
func TestInvoke_StructuredOutputSkippedOnToolCall(t *testing.T) {
	s := newStub(t)
	s.resp = `{"id":"c1","object":"chat.completion","model":"m",
	  "choices":[{"index":0,"finish_reason":"tool_calls","message":{"role":"assistant","content":null,
	    "tool_calls":[{"id":"call_1","type":"function","function":{"name":"get_weather","arguments":"{}"}}]}}],
	  "usage":{"prompt_tokens":5,"completion_tokens":2,"total_tokens":7}}`
	p := newProvider(t, s)

	resp, err := p.Invoke(context.Background(), model.Request{
		Model:        "gpt-test",
		Messages:     []model.Message{model.UserMessage("weather?")},
		OutputSchema: &model.OutputSchema{Name: "w", Schema: json.RawMessage(`{"type":"object"}`)},
	})
	require.NoError(t, err)
	require.Empty(t, resp.StructuredOutput, "a tool-call turn is not the structured final answer")
	require.Len(t, resp.Message.ToolCalls(), 1)
}

// capturingSink records deltas for assertions.
type capturingSink struct{ deltas []model.StreamDelta }

func (c *capturingSink) OnDelta(_ context.Context, d model.StreamDelta) error {
	c.deltas = append(c.deltas, d)
	return nil
}

// sseServer serves a fixed sequence of SSE data lines.
func sseServer(t *testing.T, events ...string) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(200)
		fl, _ := w.(http.Flusher)
		for _, e := range events {
			_, _ = io.WriteString(w, "data: "+e+"\n\n")
			if fl != nil {
				fl.Flush()
			}
		}
		_, _ = io.WriteString(w, "data: [DONE]\n\n")
	}))
	t.Cleanup(srv.Close)
	return srv
}

// Streamed text deltas reach the sink, and the aggregated Response carries the
// full content and usage — the invariant that makes streaming safe for replay.
func TestInvokeStream_TextDeltasAndAggregate(t *testing.T) {
	srv := sseServer(t,
		`{"id":"c","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"role":"assistant","content":"Hel"}}]}`,
		`{"id":"c","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"lo."}}]}`,
		`{"id":"c","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":2,"total_tokens":7}}`,
	)
	p, err := oaiprovider.New(oaiprovider.WithBaseURL(srv.URL), oaiprovider.WithAPIKey("k"))
	require.NoError(t, err)

	sink := &capturingSink{}
	resp, err := p.InvokeStream(context.Background(), model.Request{
		Model:    "gpt-test",
		Messages: []model.Message{model.UserMessage("hi")},
	}, sink)
	require.NoError(t, err)

	// Live text deltas.
	var streamed string
	for _, d := range sink.deltas {
		streamed += d.Text
	}
	require.Equal(t, "Hello.", streamed)

	// Aggregated response == what a non-streamed call would produce.
	require.Equal(t, "Hello.", resp.Message.Text())
	require.Equal(t, int64(7), resp.Usage.TotalTokens)
}

func TestInvokeStream_ForwardsToolCallDeltas(t *testing.T) {
	srv := sseServer(t,
		`{"id":"c","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"role":"assistant","tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"get_weather","arguments":"{\"ci"}}]}}]}`,
		`{"id":"c","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"ty\":\"Ghent\"}"}}]}}]}`,
		`{"id":"c","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":5,"completion_tokens":3,"total_tokens":8}}`,
	)
	p, err := oaiprovider.New(oaiprovider.WithBaseURL(srv.URL), oaiprovider.WithAPIKey("k"))
	require.NoError(t, err)

	sink := &capturingSink{}
	resp, err := p.InvokeStream(context.Background(), model.Request{
		Model:    "gpt-test",
		Messages: []model.Message{model.UserMessage("weather?")},
	}, sink)
	require.NoError(t, err)

	// Argument fragments were forwarded, keyed by tool-call index.
	var args string
	for _, d := range sink.deltas {
		if d.ToolCallIndex == 0 {
			args += d.ArgsFragment
		}
	}
	require.JSONEq(t, `{"city":"Ghent"}`, args)

	// Aggregated response has the complete tool call.
	require.Len(t, resp.Message.ToolCalls(), 1)
	require.Equal(t, "get_weather", resp.Message.ToolCalls()[0].Name)
	require.JSONEq(t, `{"city":"Ghent"}`, string(resp.Message.ToolCalls()[0].Arguments))
}

func TestInvoke_ParsesToolCalls(t *testing.T) {
	s := newStub(t)
	s.resp = `{
	  "id":"c1","object":"chat.completion","model":"m",
	  "choices":[{"index":0,"finish_reason":"tool_calls","message":{
	    "role":"assistant","content":null,
	    "tool_calls":[{"id":"call_abc","type":"function",
	      "function":{"name":"get_weather","arguments":"{\"city\":\"Ghent\"}"}}]}}],
	  "usage":{"prompt_tokens":5,"completion_tokens":2,"total_tokens":7}
	}`
	p := newProvider(t, s)

	resp, err := p.Invoke(context.Background(), model.Request{
		Model:    "gpt-test",
		Messages: []model.Message{model.UserMessage("Weather?")},
	})
	require.NoError(t, err)
	require.Equal(t, "tool_calls", resp.FinishReason)
	require.Len(t, resp.Message.ToolCalls(), 1)

	tc := resp.Message.ToolCalls()[0]
	require.Equal(t, "call_abc", tc.ID)
	require.Equal(t, "get_weather", tc.Name)
	require.JSONEq(t, `{"city":"Ghent"}`, string(tc.Arguments))
}

// The tool result must carry its call ID, and the assistant turn that requested
// it must be replayed alongside — the API rejects an orphaned tool result.
//
// openai.ToolMessage takes (content, toolCallID). Both are strings, so swapping
// them compiles and fails only here.
func TestInvoke_RoundTripsToolConversation(t *testing.T) {
	s := newStub(t)
	s.resp = plainReply
	p := newProvider(t, s)

	_, err := p.Invoke(context.Background(), model.Request{
		Model: "gpt-test",
		Messages: []model.Message{
			model.UserMessage("Weather?"),
			{Role: model.RoleAssistant, Blocks: []model.Block{
				model.ToolCallBlock("call_abc", "get_weather", json.RawMessage(`{"city":"Ghent"}`)),
			}},
			model.ToolResultMessage("call_abc", model.TextResult("18°C")),
		},
	})
	require.NoError(t, err)

	msgs := s.body["messages"].([]any)
	require.Len(t, msgs, 3)

	assistant := msgs[1].(map[string]any)
	require.Equal(t, "assistant", assistant["role"])
	calls := assistant["tool_calls"].([]any)
	require.Len(t, calls, 1)
	call := calls[0].(map[string]any)
	require.Equal(t, "call_abc", call["id"])
	require.Equal(t, "get_weather", call["function"].(map[string]any)["name"])

	toolMsg := msgs[2].(map[string]any)
	require.Equal(t, "tool", toolMsg["role"])
	require.Equal(t, "18°C", toolMsg["content"], "content must be the result, not the call ID")
	require.Equal(t, "call_abc", toolMsg["tool_call_id"], "tool_call_id must be the ID, not the result")
}

// A user image block becomes a multi-part message with an image_url data URL;
// text-only user messages keep their plain-string form.
func TestInvoke_UserImageBecomesImageURL(t *testing.T) {
	s := newStub(t)
	s.resp = plainReply
	p := newProvider(t, s)

	png := []byte{0x89, 0x50, 0x4e, 0x47}
	_, err := p.Invoke(context.Background(), model.Request{
		Model: "gpt-test",
		Messages: []model.Message{
			model.UserContent(model.TextBlock("what is this?"), model.ImageBlock("image/png", png)),
		},
	})
	require.NoError(t, err)

	parts := s.body["messages"].([]any)[0].(map[string]any)["content"].([]any)
	require.Len(t, parts, 2)
	require.Equal(t, "text", parts[0].(map[string]any)["type"])

	img := parts[1].(map[string]any)
	require.Equal(t, "image_url", img["type"])
	url := img["image_url"].(map[string]any)["url"].(string)
	require.Equal(t, "data:image/png;base64,"+base64.StdEncoding.EncodeToString(png), url)
}

// Audio is not something Chat Completions can show, so it degrades to a named
// placeholder rather than being dropped.
func TestInvoke_UserAudioBecomesPlaceholder(t *testing.T) {
	s := newStub(t)
	s.resp = plainReply
	p := newProvider(t, s)

	_, err := p.Invoke(context.Background(), model.Request{
		Model:    "gpt-test",
		Messages: []model.Message{model.UserContent(model.AudioBlock("audio/wav", []byte{1, 2}))},
	})
	require.NoError(t, err)

	parts := s.body["messages"].([]any)[0].(map[string]any)["content"].([]any)
	require.Equal(t, "text", parts[0].(map[string]any)["type"])
	require.Contains(t, parts[0].(map[string]any)["text"], "audio omitted")
}

func TestInvoke_MapsSettings(t *testing.T) {
	s := newStub(t)
	s.resp = plainReply
	p := newProvider(t, s)

	settings := model.Settings{
		Temperature: model.Ptr(0.25),
		TopP:        model.Ptr(0.9),
		MaxTokens:   model.Ptr[int64](256),
	}

	_, err := p.Invoke(context.Background(), model.Request{
		Model:    "gpt-test",
		Messages: []model.Message{model.UserMessage("Hi")},
		Settings: settings,
	})
	require.NoError(t, err)

	require.InDelta(t, 0.25, s.body["temperature"], 0.001)
	require.InDelta(t, 0.9, s.body["top_p"], 0.001)

	// max_tokens is deprecated and rejected by newer models.
	require.EqualValues(t, 256, s.body["max_completion_tokens"])
	require.NotContains(t, s.body, "max_tokens")
}

// WithParams is the construction-time home for OpenAI-specific request fields the
// neutral Settings does not model. The hook runs on every request.
func TestWithParams_CustomizesRequest(t *testing.T) {
	s := newStub(t)
	s.resp = plainReply
	p := newProvider(t, s, oaiprovider.WithParams(func(pr *oai.ChatCompletionNewParams) {
		pr.Seed = oai.Int(42)
		pr.ParallelToolCalls = oai.Bool(false)
		pr.FrequencyPenalty = oai.Float(0.5)
	}))

	_, err := p.Invoke(context.Background(), model.Request{
		Model:    "gpt-test",
		Messages: []model.Message{model.UserMessage("Hi")},
	})
	require.NoError(t, err)

	require.EqualValues(t, 42, s.body["seed"])
	require.Equal(t, false, s.body["parallel_tool_calls"])
	require.InDelta(t, 0.5, s.body["frequency_penalty"], 0.001)
}

// The customizer runs after the neutral mapping, so it can override a neutral
// setting.
func TestWithParams_OverridesNeutralSettings(t *testing.T) {
	s := newStub(t)
	s.resp = plainReply
	p := newProvider(t, s, oaiprovider.WithParams(func(pr *oai.ChatCompletionNewParams) {
		pr.Temperature = oai.Float(0.1)
	}))

	_, err := p.Invoke(context.Background(), model.Request{
		Model:    "gpt-test",
		Messages: []model.Message{model.UserMessage("Hi")},
		Settings: model.Settings{Temperature: model.Ptr(0.9)},
	})
	require.NoError(t, err)
	require.InDelta(t, 0.1, s.body["temperature"], 0.001, "WithParams runs last and wins")
}

// NewWithClient is the full-control injection point: hand in a client you built
// (and wrapped) yourself.
func TestNewWithClient_UsesInjectedClient(t *testing.T) {
	s := newStub(t)
	s.resp = plainReply

	client := oai.NewClient(
		option.WithBaseURL(s.URL),
		option.WithAPIKey("injected"),
		option.WithMaxRetries(0),
	)
	p, err := oaiprovider.NewWithClient(client, oaiprovider.WithName("custom"))
	require.NoError(t, err)
	require.Equal(t, "custom", p.Name())

	resp, err := p.Invoke(context.Background(), model.Request{
		Model:    "gpt-test",
		Messages: []model.Message{model.UserMessage("Hi")},
	})
	require.NoError(t, err)
	require.Equal(t, "Hello.", resp.Message.Text())
}

// An unset setting must be absent, not sent as a zero. Temperature 0 is a
// meaningful value, so it must be distinguishable from "unspecified".
func TestInvoke_OmitsUnsetSettings(t *testing.T) {
	s := newStub(t)
	s.resp = plainReply
	p := newProvider(t, s)

	_, err := p.Invoke(context.Background(), model.Request{
		Model:    "gpt-test",
		Messages: []model.Message{model.UserMessage("Hi")},
	})
	require.NoError(t, err)

	for _, k := range []string{"temperature", "top_p", "seed", "max_completion_tokens", "parallel_tool_calls"} {
		require.NotContains(t, s.body, k, "%s should be omitted when unset", k)
	}
}

func TestInvoke_ExplicitZeroTemperatureIsSent(t *testing.T) {
	s := newStub(t)
	s.resp = plainReply
	p := newProvider(t, s)

	zero := 0.0
	_, err := p.Invoke(context.Background(), model.Request{
		Model:    "gpt-test",
		Messages: []model.Message{model.UserMessage("Hi")},
		Settings: model.Settings{Temperature: &zero},
	})
	require.NoError(t, err)
	require.Contains(t, s.body, "temperature", "an explicit temperature of 0 must be sent")
	require.InDelta(t, 0.0, s.body["temperature"], 0.001)
}

func TestInvoke_ErrorMapping(t *testing.T) {
	for _, tc := range []struct {
		name          string
		status        int
		headers       map[string]string
		wantRetryable bool
		wantRetryFor  time.Duration
	}{
		{name: "401 is permanent", status: 401, wantRetryable: false},
		{name: "400 is permanent", status: 400, wantRetryable: false},
		{name: "404 is permanent", status: 404, wantRetryable: false},
		{name: "429 is transient", status: 429, wantRetryable: true},
		{name: "500 is transient", status: 500, wantRetryable: true},
		{name: "503 is transient", status: 503, wantRetryable: true},
		{
			name:   "retry-after-ms is honored",
			status: 429,
			// The millisecond header wins over the second-granularity one.
			headers:       map[string]string{"retry-after-ms": "1500", "retry-after": "9"},
			wantRetryable: true,
			wantRetryFor:  1500 * time.Millisecond,
		},
		{
			name:          "retry-after seconds is honored",
			status:        429,
			headers:       map[string]string{"retry-after": "2"},
			wantRetryable: true,
			wantRetryFor:  2 * time.Second,
		},
		{
			name:          "x-should-retry overrides a permanent status",
			status:        400,
			headers:       map[string]string{"x-should-retry": "true"},
			wantRetryable: true,
		},
		{
			name:          "x-should-retry overrides a transient status",
			status:        500,
			headers:       map[string]string{"x-should-retry": "false"},
			wantRetryable: false,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			s := newStub(t)
			s.status = tc.status
			s.resp = `{"error":{"message":"boom","type":"invalid_request_error"}}`
			s.header = tc.headers
			p := newProvider(t, s)

			_, err := p.Invoke(context.Background(), model.Request{
				Model:    "gpt-test",
				Messages: []model.Message{model.UserMessage("Hi")},
			})
			require.Error(t, err)

			var apiErr *model.APIError
			require.ErrorAs(t, err, &apiErr)
			require.Equal(t, tc.status, apiErr.StatusCode)
			require.Equal(t, tc.wantRetryable, apiErr.Retryable())
			if tc.wantRetryFor > 0 {
				require.Equal(t, tc.wantRetryFor, apiErr.RetryAfter)
			}
		})
	}
}

// A dial failure never gets a response, so it must still be classified — and as
// retryable, since the endpoint may just be starting up.
func TestInvoke_TransportFailureIsRetryable(t *testing.T) {
	p, err := oaiprovider.New(
		oaiprovider.WithBaseURL("http://127.0.0.1:1"), // nothing listening
		oaiprovider.WithAPIKey("k"),
	)
	require.NoError(t, err)

	_, err = p.Invoke(context.Background(), model.Request{
		Model:    "m",
		Messages: []model.Message{model.UserMessage("Hi")},
	})
	require.Error(t, err)

	var apiErr *model.APIError
	require.ErrorAs(t, err, &apiErr)
	require.Equal(t, 0, apiErr.StatusCode, "a transport failure has no HTTP status")
	require.True(t, apiErr.Retryable())
}

// Temporal owns retry, so the client must make exactly one HTTP call per
// attempt. Client-side retries would nest under the activity retry policy and
// multiply the real call count.
func TestInvoke_DoesNotRetryClientSide(t *testing.T) {
	var calls int
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls++
		w.WriteHeader(500)
		_, _ = io.WriteString(w, `{"error":{"message":"boom"}}`)
	}))
	defer srv.Close()

	p, err := oaiprovider.New(oaiprovider.WithBaseURL(srv.URL), oaiprovider.WithAPIKey("k"))
	require.NoError(t, err)

	_, _ = p.Invoke(context.Background(), model.Request{
		Model:    "m",
		Messages: []model.Message{model.UserMessage("Hi")},
	})
	require.Equal(t, 1, calls,
		"the provider must make exactly one call per attempt; Temporal owns retry")
}

func TestInvoke_NoChoicesIsRetryable(t *testing.T) {
	s := newStub(t)
	s.resp = `{"id":"c1","object":"chat.completion","model":"m","choices":[]}`
	p := newProvider(t, s)

	_, err := p.Invoke(context.Background(), model.Request{
		Model:    "m",
		Messages: []model.Message{model.UserMessage("Hi")},
	})
	require.Error(t, err)

	var apiErr *model.APIError
	require.ErrorAs(t, err, &apiErr)
	require.True(t, apiErr.Retryable())
}

func TestInvoke_RejectsUnknownRole(t *testing.T) {
	s := newStub(t)
	s.resp = plainReply
	p := newProvider(t, s)

	_, err := p.Invoke(context.Background(), model.Request{
		Model:    "m",
		Messages: []model.Message{{Role: "wizard", Blocks: []model.Block{model.TextBlock("Hi")}}},
	})
	require.ErrorContains(t, err, "unknown role")
}

func TestNew_NameDefaultsAndOverride(t *testing.T) {
	p, err := oaiprovider.New()
	require.NoError(t, err)
	require.Equal(t, "openai", p.Name())

	// A second name lets an OpenAI provider and a vLLM one coexist.
	p2, err := oaiprovider.New(oaiprovider.WithName("vllm"))
	require.NoError(t, err)
	require.Equal(t, "vllm", p2.Name())

	_, err = oaiprovider.New(oaiprovider.WithName(""))
	require.ErrorContains(t, err, "must not be empty")
}
