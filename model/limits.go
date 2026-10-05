package model

import (
	"context"
	"time"
	"unicode/utf8"
)

// ErrorTypeStalled is the application error type of a model call the activity
// cancelled because it exceeded one of its limits: [WithStreamIdle],
// [WithFirstDelta], or [WithUnstreamedLimit]. It is retryable.
const ErrorTypeStalled = "AgentSDKModelStalled"

// DefaultStreamIdle is the stream idle limit of an [Activities] built without
// [WithStreamIdle].
const DefaultStreamIdle = 30 * time.Second

// limits bounds a provider call from inside the model activity. A zero field
// sets no limit.
type limits struct {
	streamIdle time.Duration
	firstDelta time.Duration
	unstreamed time.Duration
}

// WithStreamIdle sets the longest a streamed call may go without a stream event
// once it has received one. Zero or less sets no limit. The default is
// [DefaultStreamIdle].
//
// A call that exceeds a limit is cancelled and fails with [ErrorTypeStalled].
// The activity's StartToCloseTimeout also bounds the call.
func WithStreamIdle(d time.Duration) Option {
	return func(a *Activities) { a.limits.streamIdle = max(d, 0) }
}

// WithFirstDelta sets the longest a streamed call may take to receive its first
// stream event. Zero or less sets no limit, the default.
func WithFirstDelta(d time.Duration) Option {
	return func(a *Activities) { a.limits.firstDelta = max(d, 0) }
}

// WithUnstreamedLimit sets the longest a call that is not streamed may take.
// Zero or less sets no limit, the default.
func WithUnstreamedLimit(d time.Duration) Option {
	return func(a *Activities) { a.limits.unstreamed = max(d, 0) }
}

// StreamEventKind names what a [StreamEvent] carried.
type StreamEventKind string

// The kinds of [StreamEvent].
const (
	// StreamEventText is a piece of the answer's text.
	StreamEventText StreamEventKind = "text"
	// StreamEventToolCall is a piece of a tool call.
	StreamEventToolCall StreamEventKind = "tool_call"
	// StreamEventReasoning is a piece of the model's reasoning.
	StreamEventReasoning StreamEventKind = "reasoning"
	// StreamEventOther is any other event of the stream, such as the start or
	// end of a message or of a content block.
	StreamEventOther StreamEventKind = "other"
)

// StreamEvent is one event a streamed call received from its backend. The model
// activity queues the events of a call and folds the queued ones into the
// [Progress] of its next heartbeat.
type StreamEvent struct {
	// Kind is what the event carried.
	Kind StreamEventKind

	// Chars is the number of characters of text, reasoning, or tool-call
	// arguments the event carried.
	Chars int

	// ToolCallIndex is the [StreamDelta.ToolCallIndex] of the tool call. It is
	// set on a [StreamEventToolCall] event.
	ToolCallIndex int
}

// Event returns the event that delivered d: a [StreamEventText] event when d
// carries text or no tool-call index, and a [StreamEventToolCall] event
// otherwise.
func (d StreamDelta) Event() StreamEvent {
	if d.Text != "" || d.ToolCallIndex < 0 {
		return StreamEvent{Kind: StreamEventText, Chars: utf8.RuneCountInString(d.Text)}
	}
	return StreamEvent{
		Kind:          StreamEventToolCall,
		ToolCallIndex: d.ToolCallIndex,
		Chars:         utf8.RuneCountInString(d.ArgsFragment),
	}
}

type progressFuncKey struct{}

// WithProgressFunc returns a context carrying fn, which [ReportProgress] calls.
// The model activity puts one on the context of every streamed provider call.
func WithProgressFunc(ctx context.Context, fn func(StreamEvent)) context.Context {
	return context.WithValue(ctx, progressFuncKey{}, fn)
}

// ReportProgress reports an event a streamed call received from its backend. It
// does nothing when ctx carries no function from [WithProgressFunc].
//
// A [StreamingProvider] calls it, with the context InvokeStream received, for
// every event of the stream it passes no delta to the sink for. The model
// activity reports the event of each delta the sink receives. A provider that
// withholds a delta from the sink reports the delta's [StreamDelta.Event].
func ReportProgress(ctx context.Context, event StreamEvent) {
	if fn, ok := ctx.Value(progressFuncKey{}).(func(StreamEvent)); ok && fn != nil {
		fn(event)
	}
}
