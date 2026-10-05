package model

import (
	"context"
	"errors"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.temporal.io/sdk/temporal"
	"go.temporal.io/sdk/testsuite"

	"github.com/wleev/temporal-agent-sdk/internal/heartbeat"
)

func TestCallProgress_DetailFoldsTheStreamEvents(t *testing.T) {
	var cp callProgress

	first := cp.Detail(heartbeat.Beat{}, []StreamEvent{
		{Kind: StreamEventReasoning, Chars: 7},
		{Kind: StreamEventText, Chars: 5},
		{Kind: StreamEventText, Chars: 6},
		{Kind: StreamEventToolCall, ToolCallIndex: 0, Chars: 2},
		{Kind: StreamEventToolCall, ToolCallIndex: 0, Chars: 2},
		{Kind: StreamEventToolCall, ToolCallIndex: 1},
		{Kind: StreamEventToolCall, ToolCallIndex: 4},
		{Kind: StreamEventOther},
	})

	assert.Equal(t, Progress{
		Streaming:      true,
		TextChars:      11,
		ToolCalls:      3,
		ReasoningChars: 7,
		Events:         8,
		LastEvent:      StreamEventOther,
	}, first)

	second := cp.Detail(heartbeat.Beat{Keepalive: true, Idle: 3 * time.Second}, nil)

	want := first.(Progress)
	want.Keepalive, want.IdleSeconds = true, 3
	assert.Equal(t, want, second, "a heartbeat with no events keeps the totals")
}

func TestCallProgress_ZeroValueBeforeAnyEvent(t *testing.T) {
	var cp callProgress

	assert.Equal(t, Progress{}, cp.Detail(heartbeat.Beat{}, nil))
}

type recordingSink struct{ deltas []StreamDelta }

func (r *recordingSink) OnDelta(_ context.Context, d StreamDelta) error {
	r.deltas = append(r.deltas, d)
	return nil
}

func TestStreamDelta_Event(t *testing.T) {
	tests := []struct {
		name  string
		delta StreamDelta
		want  StreamEvent
	}{
		{name: "text", delta: StreamDelta{Text: "héllo", ToolCallIndex: -1},
			want: StreamEvent{Kind: StreamEventText, Chars: 5}},
		{name: "text without the tool-call sentinel", delta: StreamDelta{Text: "ab"},
			want: StreamEvent{Kind: StreamEventText, Chars: 2}},
		{name: "tool-call arguments", delta: StreamDelta{ToolCallIndex: 1, ArgsFragment: "{}"},
			want: StreamEvent{Kind: StreamEventToolCall, ToolCallIndex: 1, Chars: 2}},
		{name: "tool-call start", delta: StreamDelta{ToolCallIndex: 0, ToolCallID: "a", ToolName: "read"},
			want: StreamEvent{Kind: StreamEventToolCall}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.want, tt.delta.Event())
		})
	}
}

func TestProgressSink_ForwardsEveryDeltaAndReportsItAsAnEvent(t *testing.T) {
	inner := &recordingSink{}
	var events []StreamEvent
	s := &progressSink{inner: inner, progressed: func(e StreamEvent) { events = append(events, e) }}

	assert.NoError(t, s.OnDelta(context.Background(), StreamDelta{Text: "ab", ToolCallIndex: -1}))
	assert.NoError(t, s.OnDelta(context.Background(), StreamDelta{ToolCallIndex: 1, ToolCallID: "a", ArgsFragment: "{}"}))

	assert.Len(t, inner.deltas, 2, "every delta reached the inner sink")
	assert.Equal(t, []StreamEvent{
		{Kind: StreamEventText, Chars: 2},
		{Kind: StreamEventToolCall, ToolCallIndex: 1, Chars: 2},
	}, events)
}

// beatProvider is a [StreamingProvider] that streams deltas to the sink and waits
// for pause before replying. A stream started on a done context fails. streamed counts its InvokeStream calls when set.
type beatProvider struct {
	deltas   []StreamDelta
	pause    time.Duration
	streamed *atomic.Int32
}

func (beatProvider) Name() string { return "beat" }

func (p beatProvider) Invoke(context.Context, Request) (Response, error) {
	time.Sleep(p.pause)
	return Response{Message: AssistantMessage("done"), FinishReason: FinishStop}, nil
}

func (p beatProvider) InvokeStream(ctx context.Context, _ Request, sink StreamSink) (Response, error) {
	if err := ctx.Err(); err != nil {
		return Response{}, err
	}
	if p.streamed != nil {
		p.streamed.Add(1)
	}
	for _, d := range p.deltas {
		if err := sink.OnDelta(ctx, d); err != nil {
			return Response{}, err
		}
	}
	time.Sleep(p.pause)
	return Response{Message: AssistantMessage("done"), FinishReason: FinishStop}, nil
}

// invokeRecordingBeats runs the model activity for req with a keepalive of 20ms
// and returns the Progress detail of every heartbeat it recorded, in order.
func invokeRecordingBeats(t *testing.T, acts *Activities, req Request) []Progress {
	t.Helper()

	var mu sync.Mutex
	var beats []Progress
	acts.pulse = heartbeat.Config{
		Keepalive:   20 * time.Millisecond,
		MinInterval: time.Nanosecond,
		Record: func(detail any) {
			mu.Lock()
			defer mu.Unlock()
			beats = append(beats, detail.(Progress))
		},
	}

	var s testsuite.WorkflowTestSuite
	env := s.NewTestActivityEnvironment()
	env.RegisterActivity(acts.InvokeModel)
	_, err := env.ExecuteActivity(acts.InvokeModel, req)
	require.NoError(t, err)

	mu.Lock()
	defer mu.Unlock()
	return beats
}

func TestInvokeModel_SilentCallHeartbeatsAtTheStartThenOnATimer(t *testing.T) {
	acts, err := NewActivities([]Provider{beatProvider{pause: 150 * time.Millisecond}})
	require.NoError(t, err)

	beats := invokeRecordingBeats(t, acts, Request{Provider: "beat", Model: "m"})

	if assert.GreaterOrEqual(t, len(beats), 3) {
		assert.Equal(t, Progress{}, beats[0], "the first heartbeat marks the start of the call")
		for _, b := range beats[1:] {
			assert.True(t, b.Keepalive)
			assert.Positive(t, b.IdleSeconds, "a keepalive heartbeat carries the time without progress")
		}
		assert.Greater(t, beats[len(beats)-1].IdleSeconds, beats[1].IdleSeconds)
	}
}

func TestInvokeModel_StreamedDeltasHeartbeatWithTheProgressSoFar(t *testing.T) {
	acts, err := NewActivities([]Provider{beatProvider{
		deltas: []StreamDelta{
			{Text: "ab", ToolCallIndex: -1},
			{Text: "cd", ToolCallIndex: -1},
			{ToolCallIndex: 0, ToolCallID: "call-1"},
		},
	}})
	require.NoError(t, err)

	beats := invokeRecordingBeats(t, acts, Request{Provider: "beat", Model: "m", Stream: true})

	if assert.NotEmpty(t, beats) {
		var progress []Progress
		for _, b := range beats {
			if !b.Keepalive {
				progress = append(progress, b)
			}
		}
		if assert.NotEmpty(t, progress, "heartbeats produced by deltas") {
			last := progress[len(progress)-1]
			assert.True(t, last.Streaming)
			assert.Equal(t, 4, last.TextChars)
			assert.Equal(t, 1, last.ToolCalls)
			assert.Equal(t, 3, last.Events)
			assert.Equal(t, StreamEventToolCall, last.LastEvent)
		}
	}
}

func TestInvokeModel_StalledStreamFailsAsStalled(t *testing.T) {
	block := make(chan struct{})
	t.Cleanup(func() { close(block) })
	acts, err := NewActivities([]Provider{stallingProvider{block: block}}, WithStreamIdle(30*time.Millisecond))
	require.NoError(t, err)
	acts.pulse = heartbeat.Config{Keepalive: 10 * time.Millisecond, Record: func(any) {}}

	var s testsuite.WorkflowTestSuite
	env := s.NewTestActivityEnvironment()
	env.RegisterActivity(acts.InvokeModel)
	_, err = env.ExecuteActivity(acts.InvokeModel, Request{Provider: "stall", Model: "m", Stream: true})

	var appErr *temporal.ApplicationError
	require.ErrorAs(t, err, &appErr)
	assert.Equal(t, ErrorTypeStalled, appErr.Type())
	assert.False(t, appErr.NonRetryable(), "a stalled call is retried")
	assert.Error(t, appErr.Unwrap(), "the stall is the cause")
}

// stallingProvider is a [StreamingProvider] that streams one delta and then
// waits for its context or block.
type stallingProvider struct{ block chan struct{} }

func (stallingProvider) Name() string { return "stall" }

func (stallingProvider) Invoke(context.Context, Request) (Response, error) {
	return Response{}, errors.New("not streamed")
}

func (p stallingProvider) InvokeStream(ctx context.Context, _ Request, sink StreamSink) (Response, error) {
	if err := sink.OnDelta(ctx, StreamDelta{Text: "partial", ToolCallIndex: -1}); err != nil {
		return Response{}, err
	}
	select {
	case <-ctx.Done():
		return Response{}, ctx.Err()
	case <-p.block:
		return Response{}, errors.New("unblocked")
	}
}

func TestInvokeModel_StreamsWithoutAUsableSink(t *testing.T) {
	tests := []struct {
		name    string
		factory SinkFactory
	}{
		{name: "no sink factory"},
		{name: "factory fails", factory: func(context.Context) (StreamSink, error) {
			return nil, errors.New("sink unavailable")
		}},
		{name: "factory returns no sink", factory: func(context.Context) (StreamSink, error) { return nil, nil }},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var streamed atomic.Int32
			acts, err := NewActivities([]Provider{beatProvider{
				deltas:   []StreamDelta{{Text: "ab", ToolCallIndex: -1}},
				streamed: &streamed,
			}}, WithStreamSink(tt.factory))
			require.NoError(t, err)

			invokeRecordingBeats(t, acts, Request{Provider: "beat", Model: "m", Stream: true})

			assert.Equal(t, int32(1), streamed.Load(), "the provider is called through InvokeStream")
		})
	}
}

// failingSink is a [StreamSink] that fails every delta and counts its calls.
type failingSink struct{ calls int }

func (f *failingSink) OnDelta(context.Context, StreamDelta) error {
	f.calls++
	return errors.New("sink unavailable")
}

func TestProgressSink_KeepsReportingAfterTheSinkFails(t *testing.T) {
	inner := &failingSink{}
	reports := 0
	s := &progressSink{inner: inner, progressed: func(StreamEvent) { reports++ }}

	for range 3 {
		assert.NoError(t, s.OnDelta(context.Background(), StreamDelta{Text: "ab", ToolCallIndex: -1}),
			"a sink failure is not passed to the provider")
	}

	assert.Equal(t, 1, inner.calls, "a failed sink gets no further deltas")
	assert.Equal(t, 3, reports)
}

func TestReportProgress_PassesTheEventToTheContextsFunction(t *testing.T) {
	var events []StreamEvent
	ctx := WithProgressFunc(context.Background(), func(e StreamEvent) { events = append(events, e) })

	ReportProgress(ctx, StreamEvent{Kind: StreamEventReasoning, Chars: 4})
	ReportProgress(context.Background(), StreamEvent{Kind: StreamEventOther})

	assert.Equal(t, []StreamEvent{{Kind: StreamEventReasoning, Chars: 4}}, events)
}

func TestNewActivities_LimitOptions(t *testing.T) {
	tests := []struct {
		name string
		opts []Option
		want limits
	}{
		{name: "no limit options", want: limits{streamIdle: DefaultStreamIdle}},
		{
			name: "each limit set",
			opts: []Option{
				WithStreamIdle(5 * time.Second),
				WithFirstDelta(time.Minute),
				WithUnstreamedLimit(time.Hour),
			},
			want: limits{streamIdle: 5 * time.Second, firstDelta: time.Minute, unstreamed: time.Hour},
		},
		{
			name: "zero turns a limit off",
			opts: []Option{WithStreamIdle(0), WithFirstDelta(0), WithUnstreamedLimit(0)},
			want: limits{},
		},
		{
			name: "negative turns a limit off",
			opts: []Option{WithStreamIdle(-1), WithFirstDelta(-1), WithUnstreamedLimit(-1)},
			want: limits{},
		},
		{
			name: "one limit leaves the others at their defaults",
			opts: []Option{WithFirstDelta(time.Minute)},
			want: limits{streamIdle: DefaultStreamIdle, firstDelta: time.Minute},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			acts, err := NewActivities([]Provider{beatProvider{}}, tt.opts...)
			require.NoError(t, err)

			assert.Equal(t, tt.want, acts.limits)
		})
	}
}

func TestInvokeModel_BuildingTheSinkDoesNotCountTowardsTheFirstDelta(t *testing.T) {
	acts, err := NewActivities([]Provider{beatProvider{deltas: []StreamDelta{{Text: "ab", ToolCallIndex: -1}}}},
		WithFirstDelta(50*time.Millisecond),
		WithStreamSink(func(context.Context) (StreamSink, error) {
			time.Sleep(200 * time.Millisecond)
			return discardSink{}, nil
		}),
	)
	require.NoError(t, err)

	invokeRecordingBeats(t, acts, Request{Provider: "beat", Model: "m", Stream: true})
}
