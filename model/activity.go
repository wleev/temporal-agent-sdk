package model

import (
	"context"
	"errors"
	"fmt"
	"sort"
	"strings"
	"sync"

	"go.temporal.io/sdk/activity"
	"go.temporal.io/sdk/temporal"

	"github.com/wleev/temporal-agent-sdk/internal/heartbeat"
)

// InvokeModelActivity is the registered name of the model activity. The agent
// loop schedules it by name, so the workflow never links against a provider.
const InvokeModelActivity = "agentsdk_invoke_model"

// ErrorTypeNoProvider is the application error type used when a request names a
// provider that was not registered. It is non-retryable: the worker's provider
// set is fixed at startup, so a later attempt resolves the same way.
const ErrorTypeNoProvider = "AgentSDKNoSuchProvider"

// ErrorTypeNoBlobResolver is the application error type used when a request
// carries a URI media block but no blob resolver is registered. It is
// non-retryable: the resolver is fixed at worker startup.
const ErrorTypeNoBlobResolver = "AgentSDKNoBlobResolver"

// Activities is the activity-side half of the model seam. It owns the providers
// and the network calls; the workflow side only sends a [Request].
type Activities struct {
	providers    map[string]Provider
	names        []string // sorted; for deterministic error messages
	sinkFactory  SinkFactory
	blobResolver BlobResolver
}

// BlobResolver fetches the bytes of a URI media block, returning the data and
// its MIME type (empty to keep the block's own). It is called activity-side, once
// per URI block per model call, so it may do I/O. The URI is passed
// uninterpreted, so the resolver may fetch from any store (S3, GCS, a signed HTTP
// URL). A returned error fails the call and is retried under the model activity's
// retry policy; wrap a permanent failure as non-retryable to stop that.
type BlobResolver func(ctx context.Context, uri string) (data []byte, mimeType string, err error)

// SinkFactory builds a [StreamSink] for one streamed model call, given the
// activity context (which carries the workflow ID via activity.GetInfo). Return
// a nil sink to skip streaming for this call.
type SinkFactory func(ctx context.Context) (StreamSink, error)

// SetStreamSink enables streaming: when a request sets Stream and its provider
// implements [StreamingProvider], the activity builds a sink from f and forwards
// live deltas to it. Without a sink, streaming requests fall back to a normal
// (non-streamed) call; the durable result is identical either way. Call it once
// at construction, before registering.
func (a *Activities) SetStreamSink(f SinkFactory) *Activities {
	a.sinkFactory = f
	return a
}

// SetBlobResolver registers the resolver that turns URI media blocks
// ([MediaURIBlock]) into inline bytes for the provider call. Without one, a
// request carrying a URI block fails with [ErrorTypeNoBlobResolver]. Call it once
// at construction, before registering. Returns the receiver for chaining.
func (a *Activities) SetBlobResolver(r BlobResolver) *Activities {
	a.blobResolver = r
	return a
}

// NewActivities builds the model activity set. At least one provider is
// required, and provider names must be unique.
func NewActivities(providers ...Provider) (*Activities, error) {
	if len(providers) == 0 {
		return nil, errors.New("model: at least one provider is required")
	}
	a := &Activities{providers: make(map[string]Provider, len(providers))}
	for _, p := range providers {
		name := p.Name()
		if name == "" {
			return nil, fmt.Errorf("model: provider %T has an empty name", p)
		}
		if _, dup := a.providers[name]; dup {
			return nil, fmt.Errorf("model: duplicate provider name %q", name)
		}
		a.providers[name] = p
		a.names = append(a.names, name)
	}
	sort.Strings(a.names)
	return a, nil
}

// ActivityRegistry is the part of a worker needed to register activities.
//
// It is an interface rather than worker.Registry so that test environments,
// which implement only this much, can be passed directly.
type ActivityRegistry interface {
	RegisterActivityWithOptions(a any, options activity.RegisterOptions)
}

// Register wires the model activity into a worker under [InvokeModelActivity].
func (a *Activities) Register(r ActivityRegistry) {
	r.RegisterActivityWithOptions(a.InvokeModel, activity.RegisterOptions{
		Name: InvokeModelActivity,
	})
}

// InvokeModel calls the selected provider and normalizes its failure into a
// Temporal error carrying the right retry posture.
func (a *Activities) InvokeModel(ctx context.Context, req Request) (*Response, error) {
	p, err := a.provider(req.Provider)
	if err != nil {
		return nil, err
	}

	// Annotate the interceptor's activity span with GenAI attributes. Activities
	// run once and never replay, so normal OTel from a context.Context is fine. If
	// no tracing is configured this is a no-op span.
	span := startModelSpan(ctx, p.Name(), req)

	resp, err := a.invoke(ctx, p, req)
	if err != nil {
		recordModelError(span, err)
		return nil, toTemporalError(err)
	}
	recordModelResponse(span, &resp)
	return &resp, nil
}

// invoke calls the provider, streaming when the request asks for it, a sink is
// configured, and the provider can stream. The returned Response is always the
// fully aggregated result, so the workflow and its replay see the same value
// whether or not tokens were streamed. On replay the activity does not re-run.
//
// It heartbeats for the duration of the call: a dead worker is detected within
// the activity's HeartbeatTimeout, cancellation reaches the provider, and the
// heartbeat carries a [Progress] snapshot that advances per delta when
// streaming.
func (a *Activities) invoke(ctx context.Context, p Provider, req Request) (Response, error) {
	// Resolve URI media blocks to inline bytes for this call only.
	req, err := a.resolveBlobs(ctx, req)
	if err != nil {
		return Response{}, err
	}

	var prog progressTracker
	beater := heartbeat.Start(ctx, prog.snapshot)
	defer beater.Stop()

	if !req.Stream || a.sinkFactory == nil {
		return p.Invoke(ctx, req)
	}
	sp, ok := p.(StreamingProvider)
	if !ok {
		return p.Invoke(ctx, req)
	}
	sink, err := a.sinkFactory(ctx)
	if err != nil {
		// A sink that cannot be built is an observability problem, not a reason to
		// fail the model call; fall back to a normal invocation.
		activity.GetLogger(ctx).Warn("model: stream sink factory failed; falling back to non-streamed", "error", err)
		return p.Invoke(ctx, req)
	}
	if sink == nil {
		return p.Invoke(ctx, req)
	}
	return sp.InvokeStream(ctx, req, &progressSink{inner: sink, prog: &prog})
}

// progressTracker accumulates a [Progress] snapshot from streamed deltas
// for the heartbeat goroutine to read. Its methods are safe for concurrent use.
type progressTracker struct {
	mu   sync.Mutex
	prog Progress
}

func (t *progressTracker) snapshot() any {
	t.mu.Lock()
	defer t.mu.Unlock()
	return t.prog
}

func (t *progressTracker) observe(d StreamDelta) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.prog.Streaming = true
	t.prog.TextChars += len(d.Text)
	if d.ToolCallIndex >= 0 && d.ToolCallIndex+1 > t.prog.ToolCalls {
		t.prog.ToolCalls = d.ToolCallIndex + 1
	}
}

// progressSink updates a progressTracker from each delta, then forwards it to
// the configured sink unchanged.
type progressSink struct {
	inner StreamSink
	prog  *progressTracker
}

func (s *progressSink) OnDelta(ctx context.Context, d StreamDelta) error {
	s.prog.observe(d)
	return s.inner.OnDelta(ctx, d)
}

// resolveBlobs replaces each URI media block with its fetched bytes, returning a
// request the provider can send. It copies only the messages it changes, so the
// caller's request is untouched. A block whose URI a provider fetches natively
// (see [passthroughURI]) is left for the provider. A URI block that needs
// resolution with no resolver registered fails with [ErrorTypeNoBlobResolver].
func (a *Activities) resolveBlobs(ctx context.Context, req Request) (Request, error) {
	if !needsBlobResolution(req.Messages) {
		return req, nil
	}
	if a.blobResolver == nil {
		return Request{}, temporal.NewNonRetryableApplicationError(
			"request carries a URI media block but no blob resolver is registered; call Activities.SetBlobResolver",
			ErrorTypeNoBlobResolver, nil)
	}

	msgs := make([]Message, len(req.Messages))
	copy(msgs, req.Messages)
	for i, m := range msgs {
		var blocks []Block // allocated only if this message has a block to resolve
		for j, b := range m.Blocks {
			if !resolvableMedia(b) {
				continue
			}
			data, mimeType, err := a.blobResolver(ctx, b.URI)
			if err != nil {
				return Request{}, fmt.Errorf("model: resolving media %q: %w", b.URI, err)
			}
			if blocks == nil {
				blocks = append([]Block(nil), m.Blocks...)
			}
			resolved := b
			resolved.Data = data
			if mimeType != "" {
				resolved.MIMEType = mimeType
			}
			resolved.URI = "" // the bytes are inline now; drop the reference
			blocks[j] = resolved
		}
		if blocks != nil {
			msgs[i].Blocks = blocks
		}
	}
	req.Messages = msgs
	return req, nil
}

func needsBlobResolution(msgs []Message) bool {
	for _, m := range msgs {
		for _, b := range m.Blocks {
			if resolvableMedia(b) {
				return true
			}
		}
	}
	return false
}

// resolvableMedia reports whether a block is a URI media block the resolver must
// fetch — one with a URI, no inline bytes yet, and not a provider-native scheme.
func resolvableMedia(b Block) bool {
	return b.Kind == BlockMedia && b.URI != "" && len(b.Data) == 0 && !passthroughURI(b.URI)
}

// passthroughURI reports whether a URI is left for the provider to fetch rather
// than resolved to bytes. gs:// objects are passed through: the Vertex provider
// maps them to Gemini file data natively.
func passthroughURI(uri string) bool {
	return strings.HasPrefix(uri, "gs://")
}

// provider resolves a request's provider. An empty name selects the only
// registered provider, so single-provider setups need not name it.
func (a *Activities) provider(name string) (Provider, error) {
	if name == "" {
		if len(a.names) == 1 {
			return a.providers[a.names[0]], nil
		}
		return nil, temporal.NewNonRetryableApplicationError(
			fmt.Sprintf("request does not name a provider and %d are registered (%v); set Agent.Provider",
				len(a.names), a.names),
			ErrorTypeNoProvider, nil)
	}
	p, ok := a.providers[name]
	if !ok {
		return nil, temporal.NewNonRetryableApplicationError(
			fmt.Sprintf("no provider named %q is registered (have %v)", name, a.names),
			ErrorTypeNoProvider, nil)
	}
	return p, nil
}

// toTemporalError translates a provider failure into Temporal's retry model.
//
// Anything that is not an [APIError] is left alone: Temporal retries unknown
// errors by default, which is the right posture for an unrecognized fault.
func toTemporalError(err error) error {
	var apiErr *APIError
	if !errors.As(err, &apiErr) {
		return err
	}

	errType := fmt.Sprintf("ModelHTTP%d", apiErr.StatusCode)

	if !apiErr.Retryable() {
		return temporal.NewNonRetryableApplicationError(apiErr.Error(), errType, err)
	}

	// A backend-supplied delay reflects the real rate-limit window, so it is
	// passed to Temporal as the explicit next-retry delay.
	if apiErr.RetryAfter > 0 {
		appErr := temporal.NewApplicationErrorWithOptions(apiErr.Error(), errType,
			temporal.ApplicationErrorOptions{
				Cause:          err,
				NextRetryDelay: apiErr.RetryAfter,
			})
		return appErr
	}

	return temporal.NewApplicationError(apiErr.Error(), errType, err)
}
