package model

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestProgressTracker_AccumulatesStreamedDeltas(t *testing.T) {
	var pt progressTracker

	pt.observe(StreamDelta{Text: "hello", ToolCallIndex: -1})
	pt.observe(StreamDelta{Text: " world", ToolCallIndex: -1})
	pt.observe(StreamDelta{ToolCallIndex: 0, ToolCallID: "a"})
	pt.observe(StreamDelta{ToolCallIndex: 0, ArgsFragment: "{}"}) // same call, no new count
	pt.observe(StreamDelta{ToolCallIndex: 1, ToolCallID: "b"})

	p := pt.snapshot().(Progress)
	assert.True(t, p.Streaming)
	assert.Equal(t, len("hello world"), p.TextChars)
	assert.Equal(t, 2, p.ToolCalls, "two distinct tool-call indices seen")
}

func TestProgressTracker_ZeroValueBeforeAnyDelta(t *testing.T) {
	var pt progressTracker
	p := pt.snapshot().(Progress)
	assert.False(t, p.Streaming)
	assert.Zero(t, p.TextChars)
	assert.Zero(t, p.ToolCalls)
}

type recordingSink struct{ deltas []StreamDelta }

func (r *recordingSink) OnDelta(_ context.Context, d StreamDelta) error {
	r.deltas = append(r.deltas, d)
	return nil
}

// progressSink forwards every delta to the wrapped sink and tracks progress.
func TestProgressSink_ForwardsAndTracks(t *testing.T) {
	inner := &recordingSink{}
	var pt progressTracker
	s := &progressSink{inner: inner, prog: &pt}

	assert.NoError(t, s.OnDelta(context.Background(), StreamDelta{Text: "hi", ToolCallIndex: -1}))
	assert.NoError(t, s.OnDelta(context.Background(), StreamDelta{ToolCallIndex: 0, ToolCallID: "a"}))

	assert.Len(t, inner.deltas, 2, "both deltas reached the inner sink")
	p := pt.snapshot().(Progress)
	assert.Equal(t, 2, p.TextChars)
	assert.Equal(t, 1, p.ToolCalls)
}
