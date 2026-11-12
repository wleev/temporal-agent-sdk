package vertex_test

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/wleev/temporal-agent-sdk/model"
	vertexprovider "github.com/wleev/temporal-agent-sdk/model/vertex"
)

func int64Ptr(v int64) *int64 { return &v }

func TestInvoke_ResponseMetadataAndUsageDetails(t *testing.T) {
	s := newStub(t)
	s.resp = `{"candidates":[{"content":{"role":"model","parts":[{"text":"Hello."}]},"finishReason":"STOP"}],
	  "responseId":"resp-3","modelVersion":"gemini-test-001",
	  "usageMetadata":{"promptTokenCount":40,"candidatesTokenCount":5,"totalTokenCount":55,
	    "cachedContentTokenCount":16,"thoughtsTokenCount":10}}`
	p := newProvider(t, s)

	resp, err := p.Invoke(context.Background(), model.Request{
		Model:    "gemini-test",
		Messages: []model.Message{model.UserMessage("Hi")},
	})
	require.NoError(t, err)

	assert.Equal(t, "resp-3", resp.ResponseID)
	assert.Equal(t, "gemini-test-001", resp.Model)
	assert.Equal(t, int64Ptr(16), resp.Usage.CacheReadTokens)
	assert.Nil(t, resp.Usage.CacheWriteTokens)
	assert.Equal(t, int64Ptr(10), resp.Usage.ReasoningTokens)
}

func TestInvoke_UsageDetailsAbsent(t *testing.T) {
	s := newStub(t)
	s.resp = `{"candidates":[{"content":{"role":"model","parts":[{"text":"Hello."}]},"finishReason":"STOP"}],
	  "usageMetadata":{"promptTokenCount":11,"candidatesTokenCount":3,"totalTokenCount":14}}`
	p := newProvider(t, s)

	resp, err := p.Invoke(context.Background(), model.Request{
		Model:    "gemini-test",
		Messages: []model.Message{model.UserMessage("Hi")},
	})
	require.NoError(t, err)

	assert.Empty(t, resp.ResponseID)
	assert.Nil(t, resp.Usage.CacheReadTokens)
	assert.Nil(t, resp.Usage.ReasoningTokens)
}

func TestInvokeStream_ResponseMetadata(t *testing.T) {
	srv := sseServer(t,
		`{"candidates":[{"content":{"role":"model","parts":[{"text":"Hi"}]}}],"responseId":"resp-s","modelVersion":"gemini-test-001"}`,
		`{"candidates":[{"content":{"role":"model","parts":[{"text":"."}]},"finishReason":"STOP"}],"responseId":"resp-s","usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":2,"totalTokenCount":9,"thoughtsTokenCount":2}}`,
	)
	p, err := vertexprovider.New(vertexprovider.WithAPIKey("test-key"), vertexprovider.WithBaseURL(srv.URL))
	require.NoError(t, err)

	resp, err := p.InvokeStream(context.Background(), model.Request{
		Model:    "gemini-test",
		Messages: []model.Message{model.UserMessage("hi")},
	}, &capturingSink{})
	require.NoError(t, err)

	assert.Equal(t, "resp-s", resp.ResponseID)
	assert.Equal(t, "gemini-test-001", resp.Model)
	assert.Equal(t, int64Ptr(2), resp.Usage.ReasoningTokens)
}
