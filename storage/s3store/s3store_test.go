package s3store_test

import (
	"bytes"
	"context"
	"errors"
	"io"
	"net/http"
	"sync"
	"testing"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/s3"
	"github.com/aws/aws-sdk-go-v2/service/s3/types"
	"github.com/aws/smithy-go"
	smithyhttp "github.com/aws/smithy-go/transport/http"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	commonpb "go.temporal.io/api/common/v1"
	"go.temporal.io/sdk/converter"
	"google.golang.org/protobuf/proto"

	"github.com/wleev/temporal-agent-sdk/storage"
	"github.com/wleev/temporal-agent-sdk/storage/external"
	"github.com/wleev/temporal-agent-sdk/storage/s3store"
)

// fakeS3 is an in-memory [s3store.API] that records the last PutObject request,
// returns putErr and getErr from PutObject and GetObject when set, and counts
// body Close calls in closed.
type fakeS3 struct {
	mu      sync.Mutex
	objects map[string][]byte
	lastPut *s3.PutObjectInput
	putErr  error
	getErr  error
	closed  int
}

func newFakeS3() *fakeS3 { return &fakeS3{objects: map[string][]byte{}} }

func (f *fakeS3) PutObject(_ context.Context, in *s3.PutObjectInput, _ ...func(*s3.Options)) (*s3.PutObjectOutput, error) {
	data, err := io.ReadAll(in.Body)
	if err != nil {
		return nil, err
	}
	f.mu.Lock()
	defer f.mu.Unlock()
	f.lastPut = in
	if f.putErr != nil {
		return nil, f.putErr
	}
	f.objects[aws.ToString(in.Bucket)+"/"+aws.ToString(in.Key)] = data
	return &s3.PutObjectOutput{}, nil
}

func (f *fakeS3) GetObject(_ context.Context, in *s3.GetObjectInput, _ ...func(*s3.Options)) (*s3.GetObjectOutput, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.getErr != nil {
		return nil, f.getErr
	}
	data, ok := f.objects[aws.ToString(in.Bucket)+"/"+aws.ToString(in.Key)]
	if !ok {
		return nil, &types.NoSuchKey{}
	}
	return &s3.GetObjectOutput{
		Body:          &closeCounter{Reader: bytes.NewReader(data), f: f},
		ContentLength: aws.Int64(int64(len(data))),
	}, nil
}

func (f *fakeS3) snapshot() (lastPut *s3.PutObjectInput, closed int) {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.lastPut, f.closed
}

// closeCounter is a GetObject body that counts Close calls on its fakeS3.
type closeCounter struct {
	io.Reader
	f *fakeS3
}

func (c *closeCounter) Close() error {
	c.f.mu.Lock()
	defer c.f.mu.Unlock()
	c.f.closed++
	return nil
}

// statusError returns the error the AWS SDK reports for an HTTP response with
// the given status code and no parsed error code.
func statusError(code int) error {
	return &smithyhttp.ResponseError{
		Response: &smithyhttp.Response{Response: &http.Response{StatusCode: code}},
		Err:      errors.New(http.StatusText(code)),
	}
}

func TestNew_RejectsMissingArguments(t *testing.T) {
	_, err := s3store.New(nil, "bucket")
	assert.Error(t, err)
	_, err = s3store.New(newFakeS3(), "")
	assert.Error(t, err)
}

func TestPut_SendsObjectAndAppliesOptions(t *testing.T) {
	api := newFakeS3()
	s, err := s3store.New(api, "payloads",
		s3store.WithPutObjectInput(func(in *s3.PutObjectInput) {
			in.ServerSideEncryption = types.ServerSideEncryptionAwsKms
			in.Key = aws.String("overridden")
		}))
	require.NoError(t, err)

	require.NoError(t, s.Put(t.Context(), "p/abc", []byte("hello")))

	in, _ := api.snapshot()
	require.NotNil(t, in)
	assert.Equal(t, "payloads", aws.ToString(in.Bucket))
	assert.Equal(t, "p/abc", aws.ToString(in.Key), "the store's key must win over an option")
	assert.Equal(t, int64(5), aws.ToInt64(in.ContentLength))
	assert.Equal(t, "application/x-protobuf", aws.ToString(in.ContentType))
	assert.Equal(t, types.ServerSideEncryptionAwsKms, in.ServerSideEncryption)
}

func TestPut_ReturnsAPIErrors(t *testing.T) {
	api := newFakeS3()
	api.putErr = statusError(http.StatusForbidden)
	s, err := s3store.New(api, "payloads")
	require.NoError(t, err)

	err = s.Put(t.Context(), "k", []byte("data"))
	assert.ErrorIs(t, err, api.putErr)
}

func TestGet_ReturnsObjectAndClosesBody(t *testing.T) {
	api := newFakeS3()
	s, err := s3store.New(api, "payloads")
	require.NoError(t, err)
	require.NoError(t, s.Put(t.Context(), "k", []byte("data")))

	got, err := s.Get(t.Context(), "k")
	require.NoError(t, err)
	assert.Equal(t, []byte("data"), got)
	_, closed := api.snapshot()
	assert.Equal(t, 1, closed)
}

// apiError is a smithy API error with a fixed code.
type apiError struct{ code string }

func (e apiError) Error() string                 { return e.code }
func (e apiError) ErrorCode() string             { return e.code }
func (e apiError) ErrorMessage() string          { return e.code }
func (e apiError) ErrorFault() smithy.ErrorFault { return smithy.FaultClient }

func TestGet_ClassifiesErrors(t *testing.T) {
	tests := []struct {
		name         string
		getErr       error
		wantNotFound bool
	}{
		{name: "NoSuchKey", getErr: &types.NoSuchKey{}, wantNotFound: true},
		{name: "NotFound code", getErr: apiError{code: "NotFound"}, wantNotFound: true},
		{name: "404 without a code", getErr: statusError(http.StatusNotFound), wantNotFound: true},
		{name: "NoSuchBucket", getErr: apiError{code: "NoSuchBucket"}},
		{name: "403", getErr: statusError(http.StatusForbidden)},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			api := newFakeS3()
			api.getErr = tt.getErr
			s, err := s3store.New(api, "payloads")
			require.NoError(t, err)

			_, err = s.Get(t.Context(), "k")
			require.Error(t, err)
			assert.ErrorIs(t, err, tt.getErr)
			assert.Equal(t, tt.wantNotFound, errors.Is(err, storage.ErrNotFound))
		})
	}
}

func TestStore_BacksAStorageDriver(t *testing.T) {
	s, err := s3store.New(newFakeS3(), "payloads")
	require.NoError(t, err)
	d, err := external.NewDriver(s, storage.WithDriverName("s3"))
	require.NoError(t, err)

	in := &commonpb.Payload{
		Metadata: map[string][]byte{"encoding": []byte("json/plain")},
		Data:     []byte(`"a large model request"`),
	}
	claims, err := d.Store(converter.StorageDriverStoreContext{Context: t.Context()},
		[]*commonpb.Payload{in})
	require.NoError(t, err)

	out, err := d.Retrieve(converter.StorageDriverRetrieveContext{Context: t.Context()}, claims)
	require.NoError(t, err)
	if assert.Len(t, out, 1) {
		assert.True(t, proto.Equal(in, out[0]))
	}
}
