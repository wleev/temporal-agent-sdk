// Package s3store implements [storage.Store] on an Amazon S3 bucket or an
// S3-compatible object store.
//
// The caller builds and owns the S3 client, including its region, credentials,
// and endpoint:
//
//	awsCfg, err := config.LoadDefaultConfig(ctx)
//	if err != nil {
//		log.Fatal(err)
//	}
//	store, err := s3store.New(s3.NewFromConfig(awsCfg), "agent-payloads",
//		s3store.WithPutObjectInput(func(in *s3.PutObjectInput) {
//			in.ServerSideEncryption = types.ServerSideEncryptionAwsKms
//		}))
//
// The AWS SDK adds request checksums and validates response checksums by
// default, which some S3-compatible services reject or do not return. For those
// services, set both to WhenRequired; the storage driver verifies each object
// against its own SHA-256 digest:
//
//	client := s3.NewFromConfig(awsCfg, func(o *s3.Options) {
//		o.BaseEndpoint = aws.String("http://minio:9000")
//		o.UsePathStyle = true
//		o.RequestChecksumCalculation = aws.RequestChecksumCalculationWhenRequired
//		o.ResponseChecksumValidation = aws.ResponseChecksumValidationWhenRequired
//	})
//
// Objects are never deleted by the store; expire them with a bucket lifecycle
// rule on the key prefix whose expiration exceeds the namespace retention period
// plus the longest workflow run.
package s3store

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"net/http"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/s3"
	"github.com/aws/aws-sdk-go-v2/service/s3/types"
	"github.com/aws/smithy-go"

	"github.com/wleev/temporal-agent-sdk/storage"
)

// contentType is the Content-Type set on every object.
const contentType = "application/x-protobuf"

// API is the subset of [s3.Client] the store calls.
type API interface {
	PutObject(ctx context.Context, in *s3.PutObjectInput, opts ...func(*s3.Options)) (*s3.PutObjectOutput, error)
	GetObject(ctx context.Context, in *s3.GetObjectInput, opts ...func(*s3.Options)) (*s3.GetObjectOutput, error)
}

// Store is a [storage.Store] that keeps objects in a single S3 bucket.
type Store struct {
	api    API
	bucket string
	putFns []func(*s3.PutObjectInput)
}

var _ storage.Store = (*Store)(nil)

// Option configures [New].
type Option func(*Store)

// WithPutObjectInput adjusts every PutObject request before it is sent, for
// example to set server-side encryption, a storage class, or tags. The store
// sets Bucket, Key, Body, ContentLength, and ContentType after fn runs.
func WithPutObjectInput(fn func(*s3.PutObjectInput)) Option {
	return func(s *Store) { s.putFns = append(s.putFns, fn) }
}

// New returns a Store that reads and writes objects in bucket through api. It
// returns an error if api is nil or bucket is empty.
func New(api API, bucket string, opts ...Option) (*Store, error) {
	if api == nil {
		return nil, errors.New("s3store: api must not be nil")
	}
	if bucket == "" {
		return nil, errors.New("s3store: bucket must not be empty")
	}
	s := &Store{api: api, bucket: bucket}
	for _, opt := range opts {
		opt(s)
	}
	return s, nil
}

// Put writes data to the object at key.
func (s *Store) Put(ctx context.Context, key string, data []byte) error {
	in := &s3.PutObjectInput{}
	for _, fn := range s.putFns {
		fn(in)
	}
	in.Bucket = aws.String(s.bucket)
	in.Key = aws.String(key)
	in.Body = bytes.NewReader(data)
	in.ContentLength = aws.Int64(int64(len(data)))
	in.ContentType = aws.String(contentType)
	if _, err := s.api.PutObject(ctx, in); err != nil {
		return fmt.Errorf("s3store: put s3://%s/%s: %w", s.bucket, key, err)
	}
	return nil
}

// Get reads the object at key. A missing object yields an error wrapping
// [storage.ErrNotFound].
func (s *Store) Get(ctx context.Context, key string) ([]byte, error) {
	out, err := s.api.GetObject(ctx, &s3.GetObjectInput{
		Bucket: aws.String(s.bucket),
		Key:    aws.String(key),
	})
	if err != nil {
		if isMissingObject(err) {
			return nil, fmt.Errorf("s3store: get s3://%s/%s: %w: %w", s.bucket, key, storage.ErrNotFound, err)
		}
		return nil, fmt.Errorf("s3store: get s3://%s/%s: %w", s.bucket, key, err)
	}
	defer func() { _ = out.Body.Close() }()
	var buf bytes.Buffer
	if n := aws.ToInt64(out.ContentLength); n > 0 {
		buf.Grow(int(n))
	}
	if _, err := buf.ReadFrom(out.Body); err != nil {
		return nil, fmt.Errorf("s3store: read s3://%s/%s: %w", s.bucket, key, err)
	}
	return buf.Bytes(), nil
}

// isMissingObject reports whether err means the object does not exist. It
// matches the NoSuchKey and NotFound error codes, and an HTTP 404 that carries
// no error code.
func isMissingObject(err error) bool {
	var noSuchKey *types.NoSuchKey
	if errors.As(err, &noSuchKey) {
		return true
	}
	var apiErr smithy.APIError
	if errors.As(err, &apiErr) && apiErr.ErrorCode() != "" {
		code := apiErr.ErrorCode()
		return code == "NoSuchKey" || code == "NotFound"
	}
	var status interface{ HTTPStatusCode() int }
	return errors.As(err, &status) && status.HTTPStatusCode() == http.StatusNotFound
}
