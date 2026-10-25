package openai_test

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/wleev/temporal-agent-sdk/model"
	oaiprovider "github.com/wleev/temporal-agent-sdk/model/openai"
)

func toolCallReply(arguments string) string {
	quoted, _ := json.Marshal(arguments)
	return `{"id":"c1","object":"chat.completion","model":"m",
	  "choices":[{"index":0,"finish_reason":"tool_calls","message":{"role":"assistant","content":null,
	    "tool_calls":[{"id":"call_abc","type":"function","function":{"name":"t","arguments":` + string(quoted) + `}}]}}],
	  "usage":{"prompt_tokens":5,"completion_tokens":2,"total_tokens":7}}`
}

func assertRoundTrips(t *testing.T, resp model.Response) {
	t.Helper()
	raw, err := json.Marshal(resp)
	require.NoError(t, err)
	var back model.Response
	assert.NoError(t, json.Unmarshal(raw, &back))
}

func TestInvoke_ToolArguments(t *testing.T) {
	tests := []struct {
		name string
		args string
		want string
	}{
		{name: "valid", args: `{"a":1}`, want: `{"a":1}`},
		{name: "empty", args: ``, want: `{}`},
		{name: "truncated", args: `{"a":`, want: `"{\"a\":"`},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			s := newStub(t)
			s.resp = toolCallReply(tt.args)
			p := newProvider(t, s)
			resp, err := p.Invoke(context.Background(), model.Request{
				Model:    "gpt-test",
				Messages: []model.Message{model.UserMessage("hi")},
			})
			require.NoError(t, err)
			if assert.Len(t, resp.Message.ToolCalls(), 1) {
				assert.Equal(t, tt.want, string(resp.Message.ToolCalls()[0].Arguments))
			}
			assertRoundTrips(t, resp)
		})
	}
}

func TestInvokeStream_ToolArguments(t *testing.T) {
	tests := []struct {
		name      string
		fragments []string
		want      string
	}{
		{name: "valid", fragments: []string{`{\"a\"`, `:1}`}, want: `{"a":1}`},
		{name: "truncated", fragments: []string{`{\"a\"`, `:`}, want: `"{\"a\":"`},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			events := []string{
				`{"id":"c","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"role":"assistant","tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"t","arguments":"` + tt.fragments[0] + `"}}]}}]}`,
				`{"id":"c","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"` + tt.fragments[1] + `"}}]}}]}`,
				`{"id":"c","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":5,"completion_tokens":3,"total_tokens":8}}`,
			}
			srv := sseServer(t, events...)
			p, err := oaiprovider.New(oaiprovider.WithBaseURL(srv.URL), oaiprovider.WithAPIKey("k"))
			require.NoError(t, err)
			resp, err := p.InvokeStream(context.Background(), model.Request{
				Model:    "gpt-test",
				Messages: []model.Message{model.UserMessage("hi")},
			}, &capturingSink{})
			require.NoError(t, err)
			if assert.Len(t, resp.Message.ToolCalls(), 1) {
				assert.Equal(t, tt.want, string(resp.Message.ToolCalls()[0].Arguments))
			}
			assertRoundTrips(t, resp)
		})
	}
}
