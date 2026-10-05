// Package storage defines the blob store that large Temporal payloads are
// offloaded to and the options that configure the offloading.
//
// [Store] is a two-method key-value interface. Package
// [github.com/wleev/temporal-agent-sdk/storage/s3store] implements it for Amazon
// S3 and S3-compatible services, and package
// [github.com/wleev/temporal-agent-sdk/storage/external] adapts a Store to
// Temporal's external payload storage. This package imports only the standard
// library.
package storage

import (
	"context"
	"errors"
	"fmt"
	"time"
)

var (
	// ErrNotFound is the error a [Store] wraps when a key holds no object.
	ErrNotFound = errors.New("storage: object not found")

	// ErrDigestMismatch is the error a driver wraps when an object's bytes do not
	// match the digest its reference records.
	ErrDigestMismatch = errors.New("storage: object digest mismatch")
)

// Store holds offloaded payloads as opaque bytes under string keys.
//
// Put writes data under key, replacing any existing object. Get returns the
// bytes stored under key, or an error wrapping [ErrNotFound] when there are
// none. Implementations must be safe for concurrent use.
type Store interface {
	Put(ctx context.Context, key string, data []byte) error
	Get(ctx context.Context, key string) ([]byte, error)
}

const (
	// DefaultDriverName is the driver name used when [WithDriverName] is not
	// given.
	DefaultDriverName = "temporal-agent-sdk"

	// DefaultKeyPrefix is the object key prefix used when [WithKeyPrefix] is not
	// given.
	DefaultKeyPrefix = "temporal-payloads/"

	// DefaultConcurrency is the number of objects read or written at once when
	// [WithConcurrency] is not given.
	DefaultConcurrency = 8

	// DefaultTimeout is the time limit for one object read or write when
	// [WithTimeout] is not given.
	DefaultTimeout = time.Minute
)

// Option configures the offloading of payloads to a [Store].
type Option func(*Config)

// Config is the configuration a set of [Option] values describes.
type Config struct {
	// DriverName is the name Temporal records with every reference.
	DriverName string
	// KeyPrefix is prepended to every object key.
	KeyPrefix string
	// Threshold is the serialized payload size, in bytes, at or above which a
	// payload is offloaded. Zero selects the Temporal SDK default of 256 KiB.
	Threshold int
	// Concurrency is the number of objects one batch reads or writes at once.
	Concurrency int
	// Timeout is the time limit for each object read or write.
	Timeout time.Duration
}

// WithDriverName sets the driver name, which Temporal records with every
// reference. Only a driver with the same name can read those references. Drivers
// in one external storage need distinct names.
func WithDriverName(name string) Option {
	return func(c *Config) { c.DriverName = name }
}

// WithKeyPrefix sets the prefix prepended to every object key.
func WithKeyPrefix(prefix string) Option {
	return func(c *Config) { c.KeyPrefix = prefix }
}

// WithThreshold sets the serialized payload size, in bytes, at or above which a
// payload is offloaded. Zero selects the Temporal SDK default of 256 KiB. It
// applies to external storage, not to a single driver.
func WithThreshold(bytes int) Option {
	return func(c *Config) { c.Threshold = bytes }
}

// WithConcurrency sets how many objects one batch reads or writes at once.
func WithConcurrency(n int) Option {
	return func(c *Config) { c.Concurrency = n }
}

// WithTimeout sets the time limit for each object read or write.
func WithTimeout(d time.Duration) Option {
	return func(c *Config) { c.Timeout = d }
}

// NewConfig returns the configuration opts describe, starting from the defaults.
// It returns an error for an empty driver name, a negative threshold, a
// concurrency below one, or a timeout that is not positive.
func NewConfig(opts ...Option) (Config, error) {
	c := Config{
		DriverName:  DefaultDriverName,
		KeyPrefix:   DefaultKeyPrefix,
		Concurrency: DefaultConcurrency,
		Timeout:     DefaultTimeout,
	}
	for _, opt := range opts {
		opt(&c)
	}
	switch {
	case c.DriverName == "":
		return Config{}, errors.New("storage: driver name must not be empty")
	case c.Threshold < 0:
		return Config{}, fmt.Errorf("storage: threshold must not be negative, got %d", c.Threshold)
	case c.Concurrency < 1:
		return Config{}, fmt.Errorf("storage: concurrency must be at least 1, got %d", c.Concurrency)
	case c.Timeout <= 0:
		return Config{}, fmt.Errorf("storage: timeout must be positive, got %s", c.Timeout)
	}
	return c, nil
}
