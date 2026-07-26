package model_test

import (
	"context"
	"errors"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.temporal.io/sdk/temporal"
	"go.temporal.io/sdk/testsuite"

	"github.com/wleev/temporal-agent-sdk/model"
)

// capturingProvider records the request it received, so tests assert what the
// activity handed the provider after blob resolution.
type capturingProvider struct {
	name string
	got  model.Request
}

func (p *capturingProvider) Name() string { return p.name }

func (p *capturingProvider) Invoke(_ context.Context, req model.Request) (model.Response, error) {
	p.got = req
	return model.Response{Message: model.AssistantMessage("ok"), FinishReason: model.FinishStop}, nil
}

func uriRequest(uri string) model.Request {
	return model.Request{Model: "m", Messages: []model.Message{
		model.UserContent(
			model.TextBlock("summarize"),
			model.MediaURIBlock("application/pdf", uri),
		),
	}}
}

// A URI media block is resolved to inline bytes before the provider call, with
// the URI dropped so nothing downstream re-resolves it.
func TestInvokeModel_ResolvesURIMedia(t *testing.T) {
	var s testsuite.WorkflowTestSuite
	env := s.NewTestActivityEnvironment()

	prov := &capturingProvider{name: "cap"}
	acts, err := model.NewActivities(prov)
	require.NoError(t, err)

	pdf := []byte{0x25, 0x50, 0x44, 0x46} // %PDF
	var gotURI string
	acts.SetBlobResolver(func(_ context.Context, uri string) ([]byte, string, error) {
		gotURI = uri
		return pdf, "application/pdf", nil
	})
	acts.Register(env)

	_, err = env.ExecuteActivity(model.InvokeModelActivity, uriRequest("s3://bucket/doc.pdf"))
	require.NoError(t, err)

	assert.Equal(t, "s3://bucket/doc.pdf", gotURI, "resolver is called with the block's URI")

	require.Len(t, prov.got.Messages, 1)
	blocks := prov.got.Messages[0].Blocks
	require.Len(t, blocks, 2)
	media := blocks[1]
	assert.Equal(t, pdf, media.Data, "the provider sees the fetched bytes inline")
	assert.Equal(t, "application/pdf", media.MIMEType)
	assert.Empty(t, media.URI, "the URI is dropped once bytes are inline")
}

// A resolver may correct the MIME type from what the caller declared.
func TestInvokeModel_ResolverMIMETypeWins(t *testing.T) {
	var s testsuite.WorkflowTestSuite
	env := s.NewTestActivityEnvironment()

	prov := &capturingProvider{name: "cap"}
	acts, err := model.NewActivities(prov)
	require.NoError(t, err)
	acts.SetBlobResolver(func(context.Context, string) ([]byte, string, error) {
		return []byte{1, 2}, "image/jpeg", nil
	})
	acts.Register(env)

	_, err = env.ExecuteActivity(model.InvokeModelActivity, uriRequest("s3://bucket/photo"))
	require.NoError(t, err)
	assert.Equal(t, "image/jpeg", prov.got.Messages[0].Blocks[1].MIMEType)
}

// A URI block with no resolver registered fails with a typed, non-retryable
// error.
func TestInvokeModel_URIMediaWithoutResolverErrors(t *testing.T) {
	var s testsuite.WorkflowTestSuite
	env := s.NewTestActivityEnvironment()

	acts, err := model.NewActivities(&capturingProvider{name: "cap"})
	require.NoError(t, err)
	acts.Register(env)

	_, err = env.ExecuteActivity(model.InvokeModelActivity, uriRequest("s3://bucket/doc.pdf"))
	require.Error(t, err)

	var appErr *temporal.ApplicationError
	require.ErrorAs(t, err, &appErr)
	assert.True(t, appErr.NonRetryable(), "the resolver set is fixed at startup")
	assert.Equal(t, model.ErrorTypeNoBlobResolver, appErr.Type())
}

// A gs:// URI passes through unresolved for the provider to fetch natively, so it
// needs no resolver.
func TestInvokeModel_GSURIPassesThrough(t *testing.T) {
	var s testsuite.WorkflowTestSuite
	env := s.NewTestActivityEnvironment()

	prov := &capturingProvider{name: "cap"}
	acts, err := model.NewActivities(prov)
	require.NoError(t, err)
	acts.Register(env)

	_, err = env.ExecuteActivity(model.InvokeModelActivity, uriRequest("gs://bucket/doc.pdf"))
	require.NoError(t, err, "gs:// is passed through, not resolved, so no resolver is required")

	media := prov.got.Messages[0].Blocks[1]
	assert.Equal(t, "gs://bucket/doc.pdf", media.URI, "the gs:// URI reaches the provider unchanged")
	assert.Empty(t, media.Data)
}

// A resolver failure fails the call.
func TestInvokeModel_ResolverErrorFailsCall(t *testing.T) {
	var s testsuite.WorkflowTestSuite
	env := s.NewTestActivityEnvironment()

	acts, err := model.NewActivities(&capturingProvider{name: "cap"})
	require.NoError(t, err)
	acts.SetBlobResolver(func(context.Context, string) ([]byte, string, error) {
		return nil, "", errors.New("s3 unavailable")
	})
	acts.Register(env)

	_, err = env.ExecuteActivity(model.InvokeModelActivity, uriRequest("s3://bucket/doc.pdf"))
	require.Error(t, err)
	assert.ErrorContains(t, err, "s3 unavailable")
}

// A request with no URI blocks never touches the resolver.
func TestInvokeModel_NoURIBlocksSkipsResolver(t *testing.T) {
	var s testsuite.WorkflowTestSuite
	env := s.NewTestActivityEnvironment()

	prov := &capturingProvider{name: "cap"}
	acts, err := model.NewActivities(prov)
	require.NoError(t, err)
	called := false
	acts.SetBlobResolver(func(context.Context, string) ([]byte, string, error) {
		called = true
		return nil, "", nil
	})
	acts.Register(env)

	req := model.Request{Model: "m", Messages: []model.Message{model.UserMessage("hi")}}
	_, err = env.ExecuteActivity(model.InvokeModelActivity, req)
	require.NoError(t, err)
	assert.False(t, called, "a request with no URI media never calls the resolver")
}
