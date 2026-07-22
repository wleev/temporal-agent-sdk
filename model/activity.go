package model

import (
	"context"
	"errors"
	"fmt"
	"sort"

	"go.temporal.io/sdk/activity"
	"go.temporal.io/sdk/temporal"
)

// InvokeModelActivity is the registered name of the model activity. The agent
// loop schedules it by name, so the workflow never links against a provider.
const InvokeModelActivity = "agentsdk_invoke_model"

// ErrorTypeNoProvider is the application error type used when a request names a
// provider that was not registered. It is non-retryable: the worker's provider
// set is fixed at startup, so a later attempt resolves the same way.
const ErrorTypeNoProvider = "AgentSDKNoSuchProvider"

// Activities is the activity-side half of the model seam. It owns the providers
// and the network calls; the workflow side only sends a [Request].
type Activities struct {
	providers   map[string]Provider
	names       []string // sorted; for deterministic error messages
	sinkFactory SinkFactory
}

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
func (a *Activities) invoke(ctx context.Context, p Provider, req Request) (Response, error) {
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
	return sp.InvokeStream(ctx, req, sink)
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
