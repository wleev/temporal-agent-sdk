// Package heartbeat records Temporal activity heartbeats on a timer for the
// duration of a long call.
//
// Heartbeating lets Temporal detect a dead worker within the activity's
// HeartbeatTimeout instead of waiting for StartToCloseTimeout, and it delivers
// cancellation to the running call. A heartbeat may carry a detail — a progress
// snapshot visible on the activity in the Web UI and readable by the next
// attempt via activity.GetHeartbeatDetails.
package heartbeat

import (
	"context"
	"time"

	"go.temporal.io/sdk/activity"
)

// Beater records heartbeats until Stop is called.
type Beater struct {
	cancel context.CancelFunc
	done   chan struct{}
}

// Start begins heartbeating on ctx and returns a Beater to stop it.
//
// The interval is derived from the activity's HeartbeatTimeout; when none is set
// or ctx is not an activity context, Start does nothing and Stop returns
// immediately. When details is non-nil it is called on each beat to produce the
// heartbeat detail; a nil details beats for liveness only.
func Start(ctx context.Context, details func() any) *Beater {
	if !activity.IsActivity(ctx) {
		return startEvery(ctx, 0, details)
	}
	return startEvery(ctx, beatInterval(activity.GetInfo(ctx).HeartbeatTimeout), details)
}

func startEvery(ctx context.Context, interval time.Duration, details func() any) *Beater {
	b := &Beater{done: make(chan struct{})}
	if interval <= 0 {
		close(b.done)
		return b
	}
	beatCtx, cancel := context.WithCancel(ctx)
	b.cancel = cancel
	go run(beatCtx, interval, details, b.done)
	return b
}

func run(ctx context.Context, interval time.Duration, details func() any, done chan struct{}) {
	defer close(done)
	t := time.NewTicker(interval)
	defer t.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-t.C:
			if details != nil {
				activity.RecordHeartbeat(ctx, details())
			} else {
				activity.RecordHeartbeat(ctx)
			}
		}
	}
}

// Stop ends heartbeating and waits for the beat goroutine to exit.
func (b *Beater) Stop() {
	if b.cancel != nil {
		b.cancel()
	}
	<-b.done
}

// beatInterval beats at a third of the HeartbeatTimeout, so a delayed or dropped
// beat still lands within the window. A zero timeout returns zero, disabling the
// beater.
func beatInterval(timeout time.Duration) time.Duration {
	if timeout <= 0 {
		return 0
	}
	return timeout / 3
}
