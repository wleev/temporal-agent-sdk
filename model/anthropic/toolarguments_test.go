package anthropic_test

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/wleev/temporal-agent-sdk/model"
)

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
			srv := sseServer(t,
				[2]string{"message_start", `{"type":"message_start","message":{"id":"m","type":"message","role":"assistant","model":"claude-test","content":[],"stop_reason":null,"usage":{"input_tokens":5,"output_tokens":0}}}`},
				[2]string{"content_block_start", `{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_1","name":"t","input":{}}}`},
				[2]string{"content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"` + tt.fragments[0] + `"}}`},
				[2]string{"content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"` + tt.fragments[1] + `"}}`},
				[2]string{"content_block_stop", `{"type":"content_block_stop","index":0}`},
				[2]string{"message_delta", `{"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":4}}`},
				[2]string{"message_stop", `{"type":"message_stop"}`},
			)
			p := newProvider(t, &stubServer{Server: srv})
			resp, err := p.InvokeStream(context.Background(), model.Request{
				Model:    "claude-test",
				Messages: []model.Message{model.UserMessage("hi")},
			}, &capturingSink{})
			require.NoError(t, err)
			if assert.Len(t, resp.Message.ToolCalls(), 1) {
				assert.Equal(t, tt.want, string(resp.Message.ToolCalls()[0].Arguments))
			}
			raw, err := json.Marshal(resp)
			require.NoError(t, err)
			var back model.Response
			assert.NoError(t, json.Unmarshal(raw, &back))
		})
	}
}

func TestInvoke_ToolArgumentsEmptyInput(t *testing.T) {
	s := newStub(t)
	s.resp = `{"id":"m","type":"message","role":"assistant","model":"claude-test",
	  "content":[{"type":"tool_use","id":"toolu_1","name":"t","input":{}}],
	  "stop_reason":"tool_use","usage":{"input_tokens":1,"output_tokens":1}}`
	p := newProvider(t, s)
	resp, err := p.Invoke(context.Background(), model.Request{
		Model:    "claude-test",
		Messages: []model.Message{model.UserMessage("hi")},
	})
	require.NoError(t, err)
	if assert.Len(t, resp.Message.ToolCalls(), 1) {
		assert.JSONEq(t, `{}`, string(resp.Message.ToolCalls()[0].Arguments))
	}
}

func TestInvokeStream_ToolArgumentsAcrossBlocks(t *testing.T) {
	srv := sseServer(t,
		[2]string{"message_start", `{"type":"message_start","message":{"id":"m","type":"message","role":"assistant","model":"claude-test","content":[],"stop_reason":null,"usage":{"input_tokens":5,"output_tokens":0}}}`},
		[2]string{"content_block_start", `{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`},
		[2]string{"content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hi"}}`},
		[2]string{"content_block_stop", `{"type":"content_block_stop","index":0}`},
		[2]string{"content_block_start", `{"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"a","name":"ta","input":{}}}`},
		[2]string{"content_block_delta", `{"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"k\""}}`},
		[2]string{"content_block_delta", `{"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":":2}"}}`},
		[2]string{"content_block_stop", `{"type":"content_block_stop","index":1}`},
		[2]string{"content_block_start", `{"type":"content_block_start","index":2,"content_block":{"type":"tool_use","id":"b","name":"tb","input":{}}}`},
		[2]string{"content_block_stop", `{"type":"content_block_stop","index":2}`},
		[2]string{"content_block_start", `{"type":"content_block_start","index":3,"content_block":{"type":"tool_use","id":"c","name":"tc","input":{}}}`},
		[2]string{"content_block_delta", `{"type":"content_block_delta","index":3,"delta":{"type":"input_json_delta","partial_json":""}}`},
		[2]string{"content_block_stop", `{"type":"content_block_stop","index":3}`},
		[2]string{"message_delta", `{"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":4}}`},
		[2]string{"message_stop", `{"type":"message_stop"}`},
	)
	p := newProvider(t, &stubServer{Server: srv})
	resp, err := p.InvokeStream(context.Background(), model.Request{
		Model:    "claude-test",
		Messages: []model.Message{model.UserMessage("hi")},
	}, &capturingSink{})
	require.NoError(t, err)
	assert.Equal(t, "Hi", resp.Message.Text())
	if calls := resp.Message.ToolCalls(); assert.Len(t, calls, 3) {
		assert.JSONEq(t, `{"k":2}`, string(calls[0].Arguments))
		assert.JSONEq(t, `{}`, string(calls[1].Arguments))
		assert.JSONEq(t, `{}`, string(calls[2].Arguments))
	}
}

func TestInvoke_HistoryToolInput(t *testing.T) {
	tests := []struct {
		name string
		args string
		want string
	}{
		{name: "object", args: `{"a":1}`, want: `{"a":1}`},
		{name: "empty", args: ``, want: `{}`},
		{name: "string from malformed reply", args: `"{\"a\":"`, want: `{}`},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			s := newStub(t)
			s.resp = textReply
			p := newProvider(t, s)
			_, err := p.Invoke(context.Background(), model.Request{
				Model: "claude-test",
				Messages: []model.Message{
					model.UserMessage("hi"),
					{Role: model.RoleAssistant, Blocks: []model.Block{model.ToolCallBlock("c1", "t", json.RawMessage(tt.args))}},
					model.ToolResultMessage("c1", model.TextResult("ok")),
				},
			})
			require.NoError(t, err)
			msgs, _ := s.body["messages"].([]any)
			if assert.Len(t, msgs, 3) {
				content, _ := msgs[1].(map[string]any)["content"].([]any)
				if assert.Len(t, content, 1) {
					input, _ := json.Marshal(content[0].(map[string]any)["input"])
					assert.JSONEq(t, tt.want, string(input))
				}
			}
		})
	}
}
