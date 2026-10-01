package storage_test

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	commonpb "go.temporal.io/api/common/v1"
	"go.temporal.io/sdk/client"
	"go.temporal.io/sdk/converter"
	"go.temporal.io/sdk/worker"
	"google.golang.org/protobuf/proto"

	"github.com/wleev/temporal-agent-sdk/storage"
	"github.com/wleev/temporal-agent-sdk/storage/storagetest"
)

func storeCtx(t *testing.T) converter.StorageDriverStoreContext {
	return converter.StorageDriverStoreContext{Context: t.Context()}
}

func retrieveCtx(t *testing.T) converter.StorageDriverRetrieveContext {
	return converter.StorageDriverRetrieveContext{Context: t.Context()}
}

func payload(data string) *commonpb.Payload {
	return &commonpb.Payload{
		Metadata: map[string][]byte{
			"encoding": []byte("json/plain"),
			"codec":    []byte("none"),
		},
		Data: []byte(data),
	}
}

func TestDriver_RoundTripKeepsClaimOrder(t *testing.T) {
	tests := []struct {
		name string
		opts []storage.Option
	}{
		{name: "sequential", opts: []storage.Option{storage.WithConcurrency(1)}},
		{name: "default concurrency"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			mem := storagetest.NewMemoryStore()
			d, err := storage.NewDriver(mem, append(tt.opts, storage.WithKeyPrefix("tenant-a/"))...)
			require.NoError(t, err)

			in := make([]*commonpb.Payload, 40)
			for i := range in {
				in[i] = payload(fmt.Sprintf(`"payload %d"`, i))
			}
			claims, err := d.Store(storeCtx(t), in)
			require.NoError(t, err)
			for _, c := range claims {
				assert.True(t, strings.HasPrefix(c.ClaimData["key"], "tenant-a/"), "key %q lacks the prefix", c.ClaimData["key"])
			}

			out, err := d.Retrieve(retrieveCtx(t), claims)
			require.NoError(t, err)
			if assert.Len(t, out, len(in)) {
				for i := range in {
					assert.True(t, proto.Equal(in[i], out[i]), "payload %d differs after the round trip", i)
				}
			}
		})
	}
}

func TestDriver_EmptyBatch(t *testing.T) {
	d, err := storage.NewDriver(storagetest.NewMemoryStore())
	require.NoError(t, err)

	claims, err := d.Store(storeCtx(t), nil)
	require.NoError(t, err)
	assert.Empty(t, claims)

	payloads, err := d.Retrieve(retrieveCtx(t), nil)
	require.NoError(t, err)
	assert.Empty(t, payloads)
}

// TestDriver_EqualPayloadsShareAKey checks that separately built but equal
// payloads are stored under one key.
func TestDriver_EqualPayloadsShareAKey(t *testing.T) {
	mem := storagetest.NewMemoryStore()
	d, err := storage.NewDriver(mem)
	require.NoError(t, err)

	in := make([]*commonpb.Payload, 10)
	for i := range in {
		in[i] = payload(`"same"`)
	}
	claims, err := d.Store(storeCtx(t), in)
	require.NoError(t, err)
	if assert.Len(t, claims, len(in)) {
		for _, c := range claims[1:] {
			assert.Equal(t, claims[0].ClaimData, c.ClaimData)
		}
	}
	assert.Len(t, mem.Keys(), 1)
}

func TestDriver_RetrieveMissingObjectIsNotFound(t *testing.T) {
	mem := storagetest.NewMemoryStore()
	d, err := storage.NewDriver(mem)
	require.NoError(t, err)

	claims, err := d.Store(storeCtx(t), []*commonpb.Payload{payload(`"gone"`)})
	require.NoError(t, err)
	if assert.Len(t, claims, 1) {
		mem.Delete(claims[0].ClaimData["key"])

		_, err = d.Retrieve(retrieveCtx(t), claims)
		assert.ErrorIs(t, err, storage.ErrNotFound)
	}
}

func TestDriver_RetrieveRejectsAlteredObject(t *testing.T) {
	mem := storagetest.NewMemoryStore()
	d, err := storage.NewDriver(mem)
	require.NoError(t, err)

	claims, err := d.Store(storeCtx(t), []*commonpb.Payload{payload(`"original"`)})
	require.NoError(t, err)
	if assert.Len(t, claims, 1) {
		require.NoError(t, mem.Put(t.Context(), claims[0].ClaimData["key"], []byte("altered")))

		_, err = d.Retrieve(retrieveCtx(t), claims)
		assert.ErrorIs(t, err, storage.ErrDigestMismatch)
	}
}

func TestDriver_RetrieveRejectsIncompleteClaim(t *testing.T) {
	tests := []struct {
		name  string
		claim converter.StorageDriverClaim
	}{
		{name: "nil claim data"},
		{name: "missing digest", claim: converter.StorageDriverClaim{
			ClaimData: map[string]string{"key": "temporal-payloads/abc"},
		}},
		{name: "missing key", claim: converter.StorageDriverClaim{
			ClaimData: map[string]string{"sha256": "abc"},
		}},
	}
	d, err := storage.NewDriver(storagetest.NewMemoryStore())
	require.NoError(t, err)
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := d.Retrieve(retrieveCtx(t), []converter.StorageDriverClaim{tt.claim})
			assert.Error(t, err)
		})
	}
}

// failingStore is a [storage.Store] whose methods return err.
type failingStore struct{ err error }

func (f failingStore) Put(context.Context, string, []byte) error { return f.err }

func (f failingStore) Get(context.Context, string) ([]byte, error) { return nil, f.err }

func TestDriver_ReturnsStoreErrors(t *testing.T) {
	boom := errors.New("bucket unavailable")
	d, err := storage.NewDriver(failingStore{err: boom})
	require.NoError(t, err)

	_, err = d.Store(storeCtx(t), []*commonpb.Payload{payload(`"a"`), payload(`"b"`)})
	assert.ErrorIs(t, err, boom)

	_, err = d.Retrieve(retrieveCtx(t), []converter.StorageDriverClaim{
		{ClaimData: map[string]string{"key": "temporal-payloads/abc", "sha256": "abc"}},
	})
	assert.ErrorIs(t, err, boom)
}

// panickingStore is a [storage.Store] whose methods panic.
type panickingStore struct{}

func (panickingStore) Put(context.Context, string, []byte) error { panic("put exploded") }

func (panickingStore) Get(context.Context, string) ([]byte, error) { panic("get exploded") }

func TestDriver_ReportsStorePanicsAsErrors(t *testing.T) {
	d, err := storage.NewDriver(panickingStore{})
	require.NoError(t, err)

	_, err = d.Store(storeCtx(t), []*commonpb.Payload{payload(`"a"`)})
	assert.ErrorContains(t, err, "put exploded")

	_, err = d.Retrieve(retrieveCtx(t), []converter.StorageDriverClaim{
		{ClaimData: map[string]string{"key": "temporal-payloads/abc", "sha256": "abc"}},
	})
	assert.ErrorContains(t, err, "get exploded")
}

// deadlineStore is a [storage.Store] that blocks until its context is done and
// returns the context's error.
type deadlineStore struct{}

func (deadlineStore) Put(ctx context.Context, _ string, _ []byte) error {
	<-ctx.Done()
	return ctx.Err()
}

func (deadlineStore) Get(ctx context.Context, _ string) ([]byte, error) {
	<-ctx.Done()
	return nil, ctx.Err()
}

func TestDriver_BoundsEachOperationByTimeout(t *testing.T) {
	d, err := storage.NewDriver(deadlineStore{}, storage.WithTimeout(10*time.Millisecond))
	require.NoError(t, err)

	_, err = d.Store(storeCtx(t), []*commonpb.Payload{payload(`"a"`)})
	assert.ErrorIs(t, err, context.DeadlineExceeded)

	_, err = d.Retrieve(retrieveCtx(t), []converter.StorageDriverClaim{
		{ClaimData: map[string]string{"key": "temporal-payloads/abc", "sha256": "abc"}},
	})
	assert.ErrorIs(t, err, context.DeadlineExceeded)
}

func TestDriver_PassesCancellationToTheStore(t *testing.T) {
	d, err := storage.NewDriver(storagetest.NewMemoryStore())
	require.NoError(t, err)

	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	_, err = d.Store(converter.StorageDriverStoreContext{Context: ctx}, []*commonpb.Payload{payload(`"a"`)})
	assert.ErrorIs(t, err, context.Canceled)
}

func TestDriver_NameAndType(t *testing.T) {
	d, err := storage.NewDriver(storagetest.NewMemoryStore())
	require.NoError(t, err)
	assert.Equal(t, storage.DefaultDriverName, d.Name())

	named, err := storage.NewDriver(storagetest.NewMemoryStore(), storage.WithDriverName("s3-primary"))
	require.NoError(t, err)
	assert.Equal(t, "s3-primary", named.Name())
	assert.Equal(t, d.Type(), named.Type(), "every driver of the package shares one Type")
}

func TestNew_SetsThresholdAndDriver(t *testing.T) {
	ext, err := storage.New(storagetest.NewMemoryStore(), storage.WithThreshold(64<<10))
	require.NoError(t, err)
	assert.Equal(t, 64<<10, ext.PayloadSizeThreshold)
	if assert.Len(t, ext.Drivers, 1) {
		assert.Equal(t, storage.DefaultDriverName, ext.Drivers[0].Name())
	}
}

func TestNew_RejectsInvalidConfiguration(t *testing.T) {
	mem := storagetest.NewMemoryStore()
	tests := []struct {
		name  string
		store storage.Store
		opts  []storage.Option
	}{
		{name: "nil store", store: nil},
		{name: "empty driver name", store: mem, opts: []storage.Option{storage.WithDriverName("")}},
		{name: "negative threshold", store: mem, opts: []storage.Option{storage.WithThreshold(-1)}},
		{name: "zero concurrency", store: mem, opts: []storage.Option{storage.WithConcurrency(0)}},
		{name: "zero timeout", store: mem, opts: []storage.Option{storage.WithTimeout(0)}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := storage.New(tt.store, tt.opts...)
			assert.Error(t, err)
		})
	}
}

func TestNewDriver_RejectsInvalidConfiguration(t *testing.T) {
	mem := storagetest.NewMemoryStore()
	tests := []struct {
		name  string
		store storage.Store
		opts  []storage.Option
	}{
		{name: "nil store", store: nil},
		{name: "empty driver name", store: mem, opts: []storage.Option{storage.WithDriverName("")}},
		{name: "zero concurrency", store: mem, opts: []storage.Option{storage.WithConcurrency(0)}},
		{name: "zero timeout", store: mem, opts: []storage.Option{storage.WithTimeout(0)}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := storage.NewDriver(tt.store, tt.opts...)
			assert.Error(t, err)
		})
	}
}

func TestNewDriver_IgnoresThreshold(t *testing.T) {
	_, err := storage.NewDriver(storagetest.NewMemoryStore(), storage.WithThreshold(-1))
	assert.NoError(t, err)
}

func TestNewPlugin_ConfiguresClientAndReplayer(t *testing.T) {
	ext, err := storage.New(storagetest.NewMemoryStore())
	require.NoError(t, err)
	p, err := storage.NewPlugin(ext)
	require.NoError(t, err)

	var copts client.Options
	require.NoError(t, p.ConfigureClient(t.Context(),
		client.PluginConfigureClientOptions{ClientOptions: &copts}))
	if assert.Len(t, copts.ExternalStorage.Drivers, 1) {
		assert.Equal(t, storage.DefaultDriverName, copts.ExternalStorage.Drivers[0].Name())
	}

	var ropts worker.WorkflowReplayerOptions
	require.NoError(t, p.ConfigureWorkflowReplayer(t.Context(),
		worker.PluginConfigureWorkflowReplayerOptions{WorkflowReplayerOptions: &ropts}))
	assert.Len(t, ropts.ExternalStorage.Drivers, 1)
}

func TestNewPlugin_RefusesToReplaceExistingStorage(t *testing.T) {
	ext, err := storage.New(storagetest.NewMemoryStore())
	require.NoError(t, err)
	p, err := storage.NewPlugin(ext)
	require.NoError(t, err)

	copts := client.Options{ExternalStorage: ext}
	assert.Error(t, p.ConfigureClient(t.Context(),
		client.PluginConfigureClientOptions{ClientOptions: &copts}))

	ropts := worker.WorkflowReplayerOptions{ExternalStorage: ext}
	assert.Error(t, p.ConfigureWorkflowReplayer(t.Context(),
		worker.PluginConfigureWorkflowReplayerOptions{WorkflowReplayerOptions: &ropts}))
}

func TestNewPlugin_RequiresADriver(t *testing.T) {
	_, err := storage.NewPlugin(converter.ExternalStorage{})
	assert.Error(t, err)
}
