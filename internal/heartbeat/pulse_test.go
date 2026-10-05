package heartbeat

import (
	"context"
	"strconv"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// recorded is one heartbeat: its beat and the events it drained from the queue.
type recorded struct {
	Beat
	Events []string
}

// recorder collects the heartbeats a pulse records.
type recorder struct {
	mu    sync.Mutex
	beats []recorded
}

// Detail returns the beat and its events as a recorded value.
func (r *recorder) Detail(b Beat, events []string) any {
	return recorded{Beat: b, Events: events}
}

// config returns c with r as its recorder of heartbeats.
func (r *recorder) config(c Config) Config {
	c.Record = func(detail any) {
		r.mu.Lock()
		defer r.mu.Unlock()
		r.beats = append(r.beats, detail.(recorded))
	}
	return c
}

func (r *recorder) all() []recorded {
	r.mu.Lock()
	defer r.mu.Unlock()
	return append([]recorded(nil), r.beats...)
}

func (r *recorder) eventually(t *testing.T, n int) []recorded {
	t.Helper()
	require.Eventually(t, func() bool { return len(r.all()) >= n }, 5*time.Second, time.Millisecond,
		"beats; got %v", r.all())
	return r.all()
}

// events returns every event the recorded heartbeats drained, in order.
func (r *recorder) events() []string {
	var out []string
	for _, b := range r.all() {
		out = append(out, b.Events...)
	}
	return out
}

func TestStart_RecordsABeatWithoutWaitingForTheTimer(t *testing.T) {
	var r recorder
	_, p := Start(t.Context(), &r, r.config(Config{Keepalive: time.Hour}))
	defer p.Stop()

	beats := r.eventually(t, 1)
	assert.False(t, beats[0].Keepalive)
}

func TestProgress_RecordsABeat(t *testing.T) {
	var r recorder
	_, p := Start(t.Context(), &r, r.config(Config{Keepalive: time.Hour, MinInterval: time.Nanosecond}))
	defer p.Stop()

	r.eventually(t, 1)
	p.Progress("event")

	beats := r.eventually(t, 2)
	assert.False(t, beats[1].Keepalive)
	assert.Less(t, beats[1].Idle, time.Second)
}

func TestProgress_CoalescesABurstIntoOneBeatThatHoldsEveryEvent(t *testing.T) {
	var r recorder
	_, p := Start(t.Context(), &r, r.config(Config{Keepalive: time.Hour, MinInterval: 300 * time.Millisecond}))
	defer p.Stop()

	r.eventually(t, 1)
	want := make([]string, 1000)
	for i := range want {
		want[i] = strconv.Itoa(i)
		p.Progress(want[i])
	}
	r.eventually(t, 2)
	time.Sleep(400 * time.Millisecond)

	beats := r.all()
	if assert.Len(t, beats, 2, "the start beat and one beat for the burst") {
		assert.Equal(t, want, beats[1].Events, "the beat holds every queued event, in order")
	}
}

func TestProgress_DeliversEachEventToExactlyOneBeat(t *testing.T) {
	var r recorder
	_, p := Start(t.Context(), &r, r.config(Config{Keepalive: 5 * time.Millisecond, MinInterval: time.Millisecond}))
	defer p.Stop()

	want := make([]string, 200)
	for i := range want {
		want[i] = strconv.Itoa(i)
		p.Progress(want[i])
		if i%20 == 0 {
			time.Sleep(2 * time.Millisecond)
		}
	}

	require.Eventually(t, func() bool { return len(r.events()) >= len(want) }, 5*time.Second, time.Millisecond)
	assert.Equal(t, want, r.events())
}

func TestPulse_KeepaliveBeatsDrainNoEvents(t *testing.T) {
	var r recorder
	_, p := Start(t.Context(), &r, r.config(Config{Keepalive: 20 * time.Millisecond}))
	defer p.Stop()

	for _, b := range r.eventually(t, 3)[1:] {
		assert.True(t, b.Keepalive)
		assert.Empty(t, b.Events)
	}
}

func TestStart_ClampsTheMinimumIntervalToTheKeepalive(t *testing.T) {
	var r recorder
	_, p := Start(t.Context(), &r, r.config(Config{Keepalive: 20 * time.Millisecond, MinInterval: time.Hour}))
	defer p.Stop()

	r.eventually(t, 1)
	p.Progress("event")

	// The first beat a progress report produces.
	require.Eventually(t, func() bool {
		for _, b := range r.all()[1:] {
			if !b.Keepalive {
				return true
			}
		}
		return false
	}, 5*time.Second, time.Millisecond, "a progress beat is recorded within the keepalive interval")
}

func TestPulse_RecordsKeepaliveBeatsWhileSilent(t *testing.T) {
	var r recorder
	_, p := Start(t.Context(), &r, r.config(Config{Keepalive: 20 * time.Millisecond}))
	defer p.Stop()

	beats := r.eventually(t, 3)

	for _, b := range beats[1:] {
		assert.True(t, b.Keepalive)
		assert.GreaterOrEqual(t, b.Idle, 20*time.Millisecond, "a keepalive beat carries the time since the last progress")
	}
	assert.Greater(t, beats[2].Idle, beats[1].Idle)
}

func TestPulse_ProgressResetsTheIdleTime(t *testing.T) {
	var r recorder
	_, p := Start(t.Context(), &r, r.config(Config{Keepalive: 100 * time.Millisecond, MinInterval: time.Nanosecond}))
	defer p.Stop()

	// The start beat and two keepalive beats.
	before := r.eventually(t, 3)[2]
	p.Progress("event")

	// The first keepalive beat after the beat the progress report produced.
	var after *recorded
	require.Eventually(t, func() bool {
		beats := r.all()
		reported := -1
		for i, b := range beats[1:] {
			if !b.Keepalive {
				reported = i + 1
				break
			}
		}
		if reported < 0 || reported+1 >= len(beats) {
			return false
		}
		after = &beats[reported+1]
		return true
	}, 5*time.Second, time.Millisecond)

	assert.True(t, after.Keepalive)
	assert.Less(t, after.Idle, before.Idle, "idle time counts from the progress report")
}

func TestPulse_CancelsACallThatExceedsALimit(t *testing.T) {
	tests := []struct {
		name     string
		config   Config
		progress bool
		want     string
	}{
		{name: "no first progress", config: Config{FirstProgress: 30 * time.Millisecond}, want: LimitFirstProgress},
		{name: "idle after progress", config: Config{Idle: 30 * time.Millisecond}, progress: true, want: LimitIdle},
		{name: "total", config: Config{Total: 30 * time.Millisecond}, want: LimitTotal},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var r recorder
			tt.config.Keepalive = 10 * time.Millisecond
			ctx, p := Start(t.Context(), &r, r.config(tt.config))
			defer p.Stop()
			if tt.progress {
				p.Progress("event")
			}

			select {
			case <-ctx.Done():
			case <-time.After(5 * time.Second):
				t.Fatal("the call context was not cancelled")
			}

			stall := Stalled(ctx)
			require.NotNil(t, stall)
			assert.Equal(t, tt.want, stall.Limit)

			n := len(r.all())
			time.Sleep(50 * time.Millisecond)
			assert.Len(t, r.all(), n, "a pulse that cancelled its call records nothing more")
		})
	}
}

func TestPulse_IdleLimitDoesNotApplyBeforeTheFirstProgress(t *testing.T) {
	var r recorder
	ctx, p := Start(t.Context(), &r, r.config(Config{Keepalive: 10 * time.Millisecond, Idle: 20 * time.Millisecond}))
	defer p.Stop()

	r.eventually(t, 6)

	assert.NoError(t, ctx.Err())
	assert.Nil(t, Stalled(ctx))
}

func TestStop_EndsTheBeatsAndCancelsTheCallContext(t *testing.T) {
	var r recorder
	ctx, p := Start(t.Context(), &r, r.config(Config{Keepalive: 10 * time.Millisecond}))
	r.eventually(t, 2)

	p.Stop()
	p.Stop()

	n := len(r.all())
	time.Sleep(40 * time.Millisecond)
	assert.Len(t, r.all(), n)
	assert.ErrorIs(t, ctx.Err(), context.Canceled)
	assert.Nil(t, Stalled(ctx), "a stopped pulse did not stall")
}

func TestPulse_TotalLimitHoldsUnderContinuousProgress(t *testing.T) {
	var r recorder
	ctx, p := Start(t.Context(), &r, r.config(Config{
		Keepalive: time.Hour, MinInterval: time.Nanosecond, Total: 50 * time.Millisecond,
	}))
	defer p.Stop()

	stop := time.After(5 * time.Second)
	for ctx.Err() == nil {
		p.Progress("event")
		select {
		case <-stop:
			t.Fatal("the call context was not cancelled")
		default:
		}
	}

	stall := Stalled(ctx)
	require.NotNil(t, stall)
	assert.Equal(t, LimitTotal, stall.Limit)
}

func TestKeepaliveFor(t *testing.T) {
	tests := []struct {
		name             string
		heartbeatTimeout time.Duration
		want             time.Duration
	}{
		{name: "no heartbeat timeout", want: DefaultKeepalive},
		{name: "a third of the heartbeat timeout", heartbeatTimeout: 30 * time.Second, want: 10 * time.Second},
		{name: "capped at the default", heartbeatTimeout: time.Hour, want: DefaultKeepalive},
		{name: "never below the minimum", heartbeatTimeout: time.Millisecond, want: MinKeepalive},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.want, KeepaliveFor(tt.heartbeatTimeout))
		})
	}
}

func TestProgress_AnEventQueuedDuringASlowHeartbeatCountsFromWhenItWasQueued(t *testing.T) {
	var r recorder
	config := r.config(Config{Keepalive: time.Hour, MinInterval: time.Nanosecond, Idle: 200 * time.Millisecond})
	record := config.Record
	config.Record = func(detail any) {
		record(detail)
		if events := detail.(recorded).Events; len(events) > 0 && events[0] == "a" {
			time.Sleep(250 * time.Millisecond)
		}
	}
	_, p := Start(t.Context(), &r, config)
	defer p.Stop()

	r.eventually(t, 1)
	p.Progress("a")
	r.eventually(t, 2)
	time.Sleep(150 * time.Millisecond)
	p.Progress("b")

	require.Eventually(t, func() bool { return len(r.events()) == 2 }, 2*time.Second, time.Millisecond,
		"the event queued within the idle limit is recorded; got %v", r.events())
}

func TestPulse_EnforcesALimitShorterThanTheKeepalive(t *testing.T) {
	tests := []struct {
		name   string
		config Config
		want   string
	}{
		{name: "total", config: Config{Total: 20 * time.Millisecond}, want: LimitTotal},
		{name: "first progress", config: Config{FirstProgress: 20 * time.Millisecond}, want: LimitFirstProgress},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var r recorder
			tt.config.Keepalive = time.Hour
			ctx, p := Start(t.Context(), &r, r.config(tt.config))
			defer p.Stop()

			select {
			case <-ctx.Done():
			case <-time.After(2 * time.Second):
				t.Fatal("the call context was not cancelled")
			}

			if stall := Stalled(ctx); assert.NotNil(t, stall) {
				assert.Equal(t, tt.want, stall.Limit)
			}
		})
	}
}

func TestStop_RecordsTheEventsStillQueued(t *testing.T) {
	var r recorder
	_, p := Start(t.Context(), &r, r.config(Config{Keepalive: time.Hour}))

	r.eventually(t, 1)
	p.Progress("a")
	p.Progress("b")
	p.Stop()

	assert.Equal(t, []string{"a", "b"}, r.events())
}

func TestStop_AfterAStallRecordsNothing(t *testing.T) {
	var r recorder
	ctx, p := Start(t.Context(), &r, r.config(Config{Keepalive: time.Hour, Total: 20 * time.Millisecond}))

	select {
	case <-ctx.Done():
	case <-time.After(2 * time.Second):
		t.Fatal("the call context was not cancelled")
	}
	p.Progress("late")
	n := len(r.all())
	p.Stop()

	assert.Len(t, r.all(), n)
}

func TestProgress_AfterStopQueuesNothing(t *testing.T) {
	var r recorder
	_, p := Start(t.Context(), &r, r.config(Config{Keepalive: time.Hour}))
	p.Stop()

	p.Progress("late")

	p.mu.Lock()
	defer p.mu.Unlock()
	assert.Empty(t, p.queue)
}

func TestProgress_FromSeveralGoroutinesDeliversEveryEvent(t *testing.T) {
	var r recorder
	_, p := Start(t.Context(), &r, r.config(Config{Keepalive: time.Hour, MinInterval: time.Millisecond}))

	var wg sync.WaitGroup
	for range 8 {
		wg.Go(func() {
			for range 100 {
				p.Progress("event")
			}
		})
	}
	wg.Wait()
	p.Stop()

	assert.Len(t, r.events(), 800)
}
