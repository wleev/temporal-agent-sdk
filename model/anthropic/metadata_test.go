package anthropic_test

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/wleev/temporal-agent-sdk/model"
)

func ptr(v int64) *int64 { return &v }

func TestInvoke_ResponseMetadataAndUsageDetails(t *testing.T) {
	s := newStub(t)
	s.resp = `{
	  "id":"msg_42","type":"message","role":"assistant","model":"claude-test-20260101",
	  "content":[{"type":"text","text":"Hello."}],
	  "stop_reason":"end_turn",
	  "usage":{"input_tokens":11,"output_tokens":30,"cache_read_input_tokens":100,
	    "cache_creation_input_tokens":20,"output_tokens_details":{"thinking_tokens":12}}
	}`
	p := newProvider(t, s)

	resp, err := p.Invoke(context.Background(), model.Request{
		Model:    "claude-test",
		Messages: []model.Message{model.UserMessage("Hi")},
	})
	require.NoError(t, err)

	assert.Equal(t, "msg_42", resp.ResponseID)
	assert.Equal(t, "claude-test-20260101", resp.Model)
	assert.Equal(t, ptr(100), resp.Usage.CacheReadTokens)
	assert.Equal(t, ptr(20), resp.Usage.CacheWriteTokens)
	assert.Equal(t, ptr(12), resp.Usage.ReasoningTokens)
}

func TestInvoke_UsageDetailsAbsent(t *testing.T) {
	s := newStub(t)
	s.resp = textReply
	p := newProvider(t, s)

	resp, err := p.Invoke(context.Background(), model.Request{
		Model:    "claude-test",
		Messages: []model.Message{model.UserMessage("Hi")},
	})
	require.NoError(t, err)

	assert.Nil(t, resp.Usage.CacheReadTokens)
	assert.Nil(t, resp.Usage.CacheWriteTokens)
	assert.Nil(t, resp.Usage.ReasoningTokens)
}

func TestInvokeStream_ResponseMetadataAndUsageDetails(t *testing.T) {
	srv := sseServer(t,
		[2]string{"message_start", `{"type":"message_start","message":{"id":"msg_7","type":"message","role":"assistant","model":"claude-test","content":[],"stop_reason":null,"usage":{"input_tokens":5,"output_tokens":0,"cache_read_input_tokens":64,"cache_creation_input_tokens":0}}}`},
		[2]string{"content_block_start", `{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`},
		[2]string{"content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hi."}}`},
		[2]string{"content_block_stop", `{"type":"content_block_stop","index":0}`},
		[2]string{"message_delta", `{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":9,"output_tokens_details":{"thinking_tokens":6}}}`},
		[2]string{"message_stop", `{"type":"message_stop"}`},
	)
	p := newProvider(t, &stubServer{Server: srv})

	resp, err := p.InvokeStream(context.Background(), model.Request{
		Model:    "claude-test",
		Messages: []model.Message{model.UserMessage("hi")},
	}, &capturingSink{})
	require.NoError(t, err)

	assert.Equal(t, "msg_7", resp.ResponseID)
	assert.Equal(t, "claude-test", resp.Model)
	assert.Equal(t, ptr(64), resp.Usage.CacheReadTokens)
	assert.Equal(t, ptr(0), resp.Usage.CacheWriteTokens)
	assert.Equal(t, ptr(6), resp.Usage.ReasoningTokens)
}

func TestInvokeStream_CacheCountsInDelta(t *testing.T) {
	srv := sseServer(t,
		[2]string{"message_start", `{"type":"message_start","message":{"id":"m","type":"message","role":"assistant","model":"claude-test","content":[],"stop_reason":null,"usage":{"input_tokens":5,"output_tokens":0}}}`},
		[2]string{"content_block_start", `{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`},
		[2]string{"content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hi."}}`},
		[2]string{"content_block_stop", `{"type":"content_block_stop","index":0}`},
		[2]string{"message_delta", `{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":3,"cache_read_input_tokens":40,"cache_creation_input_tokens":8}}`},
		[2]string{"message_stop", `{"type":"message_stop"}`},
	)
	p := newProvider(t, &stubServer{Server: srv})

	resp, err := p.InvokeStream(context.Background(), model.Request{
		Model:    "claude-test",
		Messages: []model.Message{model.UserMessage("hi")},
	}, &capturingSink{})
	require.NoError(t, err)

	assert.Equal(t, ptr(40), resp.Usage.CacheReadTokens)
	assert.Equal(t, ptr(8), resp.Usage.CacheWriteTokens)
}
