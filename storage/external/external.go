// Package external offloads large Temporal payloads to a [storage.Store] and
// leaves a small reference in workflow history in their place.
//
// It adapts a store to Temporal's external payload storage
// ([converter.ExternalStorage]): payloads at or above a size threshold are
// written to the store, and the SDK resolves each reference back to the original
// payload before workflow or activity code sees it. The options are those of
// package [github.com/wleev/temporal-agent-sdk/storage].
//
// # Wiring
//
// Build the external storage once and set it on the client and on any workflow
// replayer. Workers inherit it from their client.
//
//	store, err := s3store.New(s3.NewFromConfig(awsCfg), "agent-payloads")
//	if err != nil {
//		log.Fatal(err)
//	}
//	ext, err := external.New(store, storage.WithThreshold(128<<10))
//	if err != nil {
//		log.Fatal(err)
//	}
//	c, err := client.Dial(client.Options{ExternalStorage: ext})
//
// [NewPlugin] wraps the same value as a Temporal plugin for
// [client.Options].Plugins and [worker.WorkflowReplayerOptions].Plugins.
//
// # Objects
//
// Each payload is serialized as a Temporal Payload protobuf and stored under a
// key derived from the SHA-256 digest of those bytes, so equal payloads encoded
// by the same build share one object. The digest is recorded in the reference
// and checked on read. The package never deletes objects: keep them for at least
// the namespace retention period plus the longest workflow run, or replaying and
// resetting workflows that reference them fails.
//
// Payload codecs run before external storage, so a client whose DataConverter
// encrypts payloads writes ciphertext to the store.
//
// External payload storage is experimental in the Temporal Go SDK, and this
// package follows its API.
package external

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"time"

	commonpb "go.temporal.io/api/common/v1"
	"go.temporal.io/sdk/client"
	"go.temporal.io/sdk/converter"
	"go.temporal.io/sdk/temporal"
	"go.temporal.io/sdk/worker"
	"golang.org/x/sync/errgroup"
	"google.golang.org/protobuf/proto"

	"github.com/wleev/temporal-agent-sdk/storage"
)

const (
	// driverType is the Type of every driver this package builds.
	driverType = "temporal-agent-sdk.store"

	claimKey    = "key"
	claimDigest = "sha256"
)

// New returns external storage with one driver backed by store that offloads
// every payload at or above the [storage.WithThreshold] size.
func New(store storage.Store, opts ...storage.Option) (converter.ExternalStorage, error) {
	c, err := storage.NewConfig(opts...)
	if err != nil {
		return converter.ExternalStorage{}, err
	}
	d, err := newDriver(store, c)
	if err != nil {
		return converter.ExternalStorage{}, err
	}
	return converter.ExternalStorage{
		Drivers:              []converter.StorageDriver{d},
		PayloadSizeThreshold: c.Threshold,
	}, nil
}

// NewDriver returns a Temporal storage driver backed by store, for use in a
// [converter.ExternalStorage] with several drivers. It ignores
// [storage.WithThreshold].
func NewDriver(store storage.Store, opts ...storage.Option) (converter.StorageDriver, error) {
	c, err := storage.NewConfig(append(opts, storage.WithThreshold(0))...)
	if err != nil {
		return nil, err
	}
	return newDriver(store, c)
}

func newDriver(store storage.Store, c storage.Config) (*driver, error) {
	if store == nil {
		return nil, errors.New("storage: store must not be nil")
	}
	return &driver{
		store:       store,
		name:        c.DriverName,
		keyPrefix:   c.KeyPrefix,
		concurrency: c.Concurrency,
		timeout:     c.Timeout,
	}, nil
}

// driver implements [converter.StorageDriver] over a [storage.Store].
type driver struct {
	store       storage.Store
	name        string
	keyPrefix   string
	concurrency int
	timeout     time.Duration
}

// Name returns the driver name recorded with each reference.
func (d *driver) Name() string { return d.name }

// Type returns the implementation type shared by all drivers of this package.
func (d *driver) Type() string { return driverType }

// Store writes each payload to the store and returns one claim per payload, in
// order.
func (d *driver) Store(
	ctx converter.StorageDriverStoreContext,
	payloads []*commonpb.Payload,
) ([]converter.StorageDriverClaim, error) {
	claims := make([]converter.StorageDriverClaim, len(payloads))
	err := d.forEach(ctx.Context, len(payloads), func(ctx context.Context, i int) error {
		claim, err := d.storeOne(ctx, payloads[i])
		claims[i] = claim
		return err
	})
	if err != nil {
		return nil, err
	}
	return claims, nil
}

// Retrieve reads the payload behind each claim, verifies it against the digest
// the claim records, and returns the payloads in claim order.
func (d *driver) Retrieve(
	ctx converter.StorageDriverRetrieveContext,
	claims []converter.StorageDriverClaim,
) ([]*commonpb.Payload, error) {
	payloads := make([]*commonpb.Payload, len(claims))
	err := d.forEach(ctx.Context, len(claims), func(ctx context.Context, i int) error {
		p, err := d.retrieveOne(ctx, claims[i])
		payloads[i] = p
		return err
	})
	if err != nil {
		return nil, err
	}
	return payloads, nil
}

// forEach calls fn for each index in [0, n), at most d.concurrency at a time,
// each under its own d.timeout. It returns the first error, and reports a panic
// in fn as an error.
func (d *driver) forEach(ctx context.Context, n int, fn func(context.Context, int) error) error {
	g, gctx := errgroup.WithContext(ctx)
	g.SetLimit(d.concurrency)
	for i := range n {
		g.Go(func() (err error) {
			defer func() {
				if r := recover(); r != nil {
					err = fmt.Errorf("storage: store panicked: %v", r)
				}
			}()
			opCtx, cancel := context.WithTimeout(gctx, d.timeout)
			defer cancel()
			return fn(opCtx, i)
		})
	}
	return g.Wait()
}

func (d *driver) storeOne(ctx context.Context, p *commonpb.Payload) (converter.StorageDriverClaim, error) {
	data, err := proto.MarshalOptions{Deterministic: true}.Marshal(p)
	if err != nil {
		return converter.StorageDriverClaim{}, fmt.Errorf("storage: encoding payload: %w", err)
	}
	sum := sha256.Sum256(data)
	digest := hex.EncodeToString(sum[:])
	key := d.keyPrefix + digest
	if err := d.store.Put(ctx, key, data); err != nil {
		return converter.StorageDriverClaim{}, fmt.Errorf("storage: writing %s: %w", key, err)
	}
	return converter.StorageDriverClaim{
		ClaimData: map[string]string{claimKey: key, claimDigest: digest},
	}, nil
}

func (d *driver) retrieveOne(ctx context.Context, claim converter.StorageDriverClaim) (*commonpb.Payload, error) {
	key := claim.ClaimData[claimKey]
	want := claim.ClaimData[claimDigest]
	if key == "" || want == "" {
		return nil, fmt.Errorf("storage: claim is missing %q or %q", claimKey, claimDigest)
	}
	data, err := d.store.Get(ctx, key)
	if err != nil {
		return nil, fmt.Errorf("storage: reading %s: %w", key, err)
	}
	sum := sha256.Sum256(data)
	if got := hex.EncodeToString(sum[:]); got != want {
		return nil, fmt.Errorf("%w: %s has %s, reference records %s", storage.ErrDigestMismatch, key, got, want)
	}
	var p commonpb.Payload
	if err := proto.Unmarshal(data, &p); err != nil {
		return nil, fmt.Errorf("storage: decoding %s: %w", key, err)
	}
	return &p, nil
}

// PluginName is the name of the plugin [NewPlugin] returns.
const PluginName = "temporal-agent-sdk.storage"

// NewPlugin returns a Temporal plugin that sets ext as the external storage of
// each client and workflow replayer it is registered on. Configuration fails if
// the client or replayer already has external storage. NewPlugin returns an
// error if ext has no drivers.
func NewPlugin(ext converter.ExternalStorage) (*temporal.SimplePlugin, error) {
	if len(ext.Drivers) == 0 {
		return nil, errors.New("storage: external storage has no drivers")
	}
	return temporal.NewSimplePlugin(temporal.SimplePluginOptions{
		Name: PluginName,
		ConfigureClient: func(_ context.Context, o client.PluginConfigureClientOptions) error {
			if len(o.ClientOptions.ExternalStorage.Drivers) > 0 {
				return errors.New("storage: client already has external storage")
			}
			o.ClientOptions.ExternalStorage = ext
			return nil
		},
		ConfigureWorkflowReplayer: func(_ context.Context, o worker.PluginConfigureWorkflowReplayerOptions) error {
			if len(o.WorkflowReplayerOptions.ExternalStorage.Drivers) > 0 {
				return errors.New("storage: workflow replayer already has external storage")
			}
			o.WorkflowReplayerOptions.ExternalStorage = ext
			return nil
		},
	})
}
