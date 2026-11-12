package openai_test

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/wleev/temporal-agent-sdk/model"
	oaiprovider "github.com/wleev/temporal-agent-sdk/model/openai"
)

func int64Ptr(v int64) *int64 { return &v }

func TestInvoke_ResponseMetadataAndUsageDetails(t *testing.T) {
	s := newStub(t)
	s.resp = `{
	  "id":"chatcmpl-9","object":"chat.completion","model":"gpt-test-2026",
	  "choices":[{"index":0,"message":{"role":"assistant","content":"Hello."},"finish_reason":"stop"}],
	  "usage":{"prompt_tokens":50,"completion_tokens":20,"total_tokens":70,
	    "prompt_tokens_details":{"cached_tokens":32},
	    "completion_tokens_details":{"reasoning_tokens":8}}
	}`
	p := newProvider(t, s)

	resp, err := p.Invoke(context.Background(), model.Request{
		Model:    "gpt-test",
		Messages: []model.Message{model.UserMessage("Hi")},
	})
	require.NoError(t, err)

	assert.Equal(t, "chatcmpl-9", resp.ResponseID)
	assert.Equal(t, "gpt-test-2026", resp.Model)
	assert.Equal(t, int64Ptr(32), resp.Usage.CacheReadTokens)
	assert.Nil(t, resp.Usage.CacheWriteTokens)
	assert.Equal(t, int64Ptr(8), resp.Usage.ReasoningTokens)
}

func TestInvokeStream_ResponseMetadataAndUsageDetails(t *testing.T) {
	srv := sseServer(t,
		`{"id":"c1","object":"chat.completion.chunk","model":"gpt-test-2026","choices":[{"index":0,"delta":{"role":"assistant","content":"Hi."}}]}`,
		`{"id":"c1","object":"chat.completion.chunk","model":"gpt-test-2026","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`,
		`{"id":"c1","object":"chat.completion.chunk","model":"gpt-test-2026","choices":[],"usage":{"prompt_tokens":5,"completion_tokens":2,"total_tokens":7,"prompt_tokens_details":{"cached_tokens":4},"completion_tokens_details":{"reasoning_tokens":1}}}`,
	)
	p, err := oaiprovider.New(oaiprovider.WithBaseURL(srv.URL), oaiprovider.WithAPIKey("k"))
	require.NoError(t, err)

	resp, err := p.InvokeStream(context.Background(), model.Request{
		Model:    "gpt-test",
		Messages: []model.Message{model.UserMessage("hi")},
	}, &capturingSink{})
	require.NoError(t, err)

	assert.Equal(t, "c1", resp.ResponseID)
	assert.Equal(t, "gpt-test-2026", resp.Model)
	assert.Equal(t, int64(7), resp.Usage.TotalTokens)
	assert.Equal(t, int64Ptr(4), resp.Usage.CacheReadTokens)
	assert.Equal(t, int64Ptr(1), resp.Usage.ReasoningTokens)
}
