// Package heartbeat records the Temporal activity heartbeats of one long call
// and cancels a call that exceeds its limits.
//
// A [Pulse] queues the progress events a call reports. Each heartbeat drains the
// queue and passes the drained events to the call's [Detailer], which folds
// them into the heartbeat's detail. Events that arrive close together are drained by one
// heartbeat. While the call reports nothing, a heartbeat is recorded on a timer
// with an empty queue. A timer heartbeat's [Beat] gives the time since the call
// last reported an event.
//
//	callCtx, pulse := heartbeat.Start(ctx, state, heartbeat.Config{Idle: 30 * time.Second})
//	defer pulse.Stop()
//	result, err := call(callCtx, pulse.Progress)
//	if stall := heartbeat.Stalled(callCtx); stall != nil {
//		return nil, stall
//	}
package heartbeat

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"time"

	"go.temporal.io/sdk/activity"
)

const (
	// DefaultKeepalive is the longest time between heartbeats of an activity
	// with no HeartbeatTimeout, and the cap for one that has it.
	DefaultKeepalive = 10 * time.Second

	// MinKeepalive is the shortest keepalive interval [KeepaliveFor] returns.
	MinKeepalive = 100 * time.Millisecond

	// DefaultMinInterval is the shortest time between heartbeats when
	// [Config.MinInterval] is zero.
	DefaultMinInterval = time.Second
)

// The values of [StallError.Limit].
const (
	LimitFirstProgress = "first progress"
	LimitIdle          = "idle"
	LimitTotal         = "total"
)

// Beat describes one heartbeat.
type Beat struct {
	// Keepalive reports whether the timer produced the heartbeat. It is false
	// for the heartbeat at the start and for one that drains reported events.
	Keepalive bool
	// Idle is the time since the call last reported an event, or since it
	// started when it has reported none. It is zero unless Keepalive is set.
	Idle time.Duration
}

// Detailer builds the details of the heartbeats of a call that reports events
// of type E.
type Detailer[E any] interface {
	// Detail returns the detail of a heartbeat from its beat and the events
	// reported since the previous heartbeat, in the order they were reported.
	// events is empty for a keepalive heartbeat, and for the first heartbeat when
	// the call has reported nothing before it. Detail is called from one
	// goroutine at a time.
	Detail(b Beat, events []E) any
}

// Config configures a [Pulse]. The zero value heartbeats and sets no limit.
type Config struct {
	// Keepalive is the longest time between heartbeats. Zero selects
	// [KeepaliveFor] the activity's HeartbeatTimeout.
	Keepalive time.Duration

	// MinInterval is the shortest time between heartbeats. Events reported
	// sooner after a heartbeat stay queued until the interval has passed. Zero
	// selects [DefaultMinInterval]. A value above Keepalive is lowered to it.
	MinInterval time.Duration

	// FirstProgress is the longest the call may take to report its first
	// progress. Zero sets no limit.
	FirstProgress time.Duration

	// Idle is the longest the call may go without progress once it has reported
	// some. Zero sets no limit.
	Idle time.Duration

	// Total is the longest the call may take. Zero sets no limit.
	Total time.Duration

	// Record records one heartbeat. Nil records an activity heartbeat on the
	// context passed to [Start], or nothing when that is not an activity
	// context.
	Record func(detail any)
}

// KeepaliveFor returns the keepalive interval for an activity with the given
// HeartbeatTimeout: a third of it, at least [MinKeepalive] and at most
// [DefaultKeepalive], and DefaultKeepalive when it is zero.
func KeepaliveFor(heartbeatTimeout time.Duration) time.Duration {
	if heartbeatTimeout <= 0 {
		return DefaultKeepalive
	}
	return max(MinKeepalive, min(heartbeatTimeout/3, DefaultKeepalive))
}

// StallError is the cancellation cause of a call context whose call exceeded a
// limit.
type StallError struct {
	// Limit names the limit: [LimitFirstProgress], [LimitIdle], or [LimitTotal].
	Limit string
	// After is the limit's duration.
	After time.Duration
}

// Error describes the limit the call exceeded.
func (e *StallError) Error() string {
	switch e.Limit {
	case LimitFirstProgress:
		return fmt.Sprintf("no progress within %s of the call starting", e.After)
	case LimitIdle:
		return fmt.Sprintf("no progress for %s", e.After)
	default:
		return fmt.Sprintf("call ran longer than %s", e.After)
	}
}

// Stalled returns the limit the call of ctx exceeded, or nil when ctx is not
// done or ended for another reason. ctx is a context [Start] returned.
func Stalled(ctx context.Context) *StallError {
	var stall *StallError
	if errors.As(context.Cause(ctx), &stall) {
		return stall
	}
	return nil
}

// Pulse queues the progress events of one call and records its heartbeats.
type Pulse[E any] struct {
	config   Config
	detailer Detailer[E]
	callDone <-chan struct{}
	cancel   context.CancelCauseFunc
	progress chan struct{}
	exited   chan struct{}
	stopOnce sync.Once

	// stalled reports whether the call exceeded a limit. It is written before
	// exited is closed and read after.
	stalled bool

	mu sync.Mutex
	// queue holds the events reported since the last heartbeat.
	queue []E
	// lastQueued is the time the most recent event was queued.
	lastQueued time.Time
	// closed reports whether the pulse has ended. A closed pulse queues nothing.
	closed bool
}

// Start begins a pulse for one call and returns the context to make the call
// with. detailer builds the detail of each heartbeat; nil records heartbeats
// without a detail and drops the events. The pulse records a heartbeat as soon
// as it starts. It cancels the
// returned context when the call exceeds a limit of config, with a [StallError]
// as its cause, and records no heartbeat after that. [Pulse.Stop] ends the
// pulse.
func Start[E any](ctx context.Context, detailer Detailer[E], config Config) (context.Context, *Pulse[E]) {
	inActivity := activity.IsActivity(ctx)
	if config.Keepalive <= 0 {
		var heartbeatTimeout time.Duration
		if inActivity {
			heartbeatTimeout = activity.GetInfo(ctx).HeartbeatTimeout
		}
		config.Keepalive = KeepaliveFor(heartbeatTimeout)
	}
	if config.MinInterval <= 0 {
		config.MinInterval = DefaultMinInterval
	}
	config.MinInterval = min(config.MinInterval, config.Keepalive)
	switch {
	case config.Record != nil:
	case inActivity:
		config.Record = func(detail any) {
			if detail == nil {
				activity.RecordHeartbeat(ctx)
				return
			}
			activity.RecordHeartbeat(ctx, detail)
		}
	default:
		config.Record = func(any) {}
	}

	callCtx, cancel := context.WithCancelCause(ctx)
	p := &Pulse[E]{
		config:   config,
		detailer: detailer,
		callDone: callCtx.Done(),
		cancel:   cancel,
		progress: make(chan struct{}, 1),
		exited:   make(chan struct{}),
	}
	go p.run(time.Now())
	return callCtx, p
}

// Progress queues event for the next heartbeat. It never blocks. It does
// nothing once the pulse has ended.
func (p *Pulse[E]) Progress(event E) {
	p.mu.Lock()
	if p.closed {
		p.mu.Unlock()
		return
	}
	p.queue = append(p.queue, event)
	p.lastQueued = time.Now()
	p.mu.Unlock()
	select {
	case p.progress <- struct{}{}:
	default:
	}
}

// Stop ends the pulse, cancels the call context, and waits for the pulse's
// goroutine. It records the events still queued in one last heartbeat, unless
// the call exceeded a limit. Calling it again returns at once.
func (p *Pulse[E]) Stop() {
	p.stopOnce.Do(func() {
		p.cancel(context.Canceled)
		<-p.exited
		if !p.stalled && p.pending() {
			p.record(Beat{})
		}
	})
}

// pending reports whether events are queued.
func (p *Pulse[E]) pending() bool {
	p.mu.Lock()
	defer p.mu.Unlock()
	return len(p.queue) > 0
}

// drain empties the queue and returns its events and the time the last of them
// was queued.
func (p *Pulse[E]) drain() ([]E, time.Time) {
	p.mu.Lock()
	defer p.mu.Unlock()
	events := p.queue
	p.queue = nil
	return events, p.lastQueued
}

// close ends the queuing of events. It drops the queued events when drop is
// set.
func (p *Pulse[E]) close(drop bool) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.closed = true
	if drop {
		p.queue = nil
	}
}

// record drains the queue and records one heartbeat. It returns whether it
// drained events and the time the last of them was queued. A heartbeat that
// drains events is not a keepalive one and has no idle time.
func (p *Pulse[E]) record(b Beat) (drained bool, lastQueued time.Time) {
	events, lastQueued := p.drain()
	if len(events) > 0 {
		b = Beat{}
	}
	var detail any
	if p.detailer != nil {
		detail = p.detailer.Detail(b, events)
	}
	p.config.Record(detail)
	return len(events) > 0, lastQueued
}

// timeline is the state of a pulse's goroutine.
type timeline struct {
	// start is the time the call started.
	start time.Time
	// lastProgress is the time the call last queued an event, or start when it
	// has queued none.
	lastProgress time.Time
	// lastBeat is the time of the last heartbeat.
	lastBeat time.Time
	// progressed reports whether the call has queued an event.
	progressed bool
	// pending reports whether events are queued.
	pending bool
}

// run records heartbeats until the call context is done or the call exceeds a
// limit.
func (p *Pulse[E]) run(start time.Time) {
	defer close(p.exited)

	tl := timeline{start: start, lastProgress: start, lastBeat: start}
	// beat records a heartbeat at now and credits the events it drained.
	beat := func(now time.Time, b Beat) {
		if drained, lastQueued := p.record(b); drained {
			tl.lastProgress, tl.progressed = lastQueued, true
		}
		tl.lastBeat, tl.pending = now, false
	}
	beat(start, Beat{})
	timer := time.NewTimer(time.Until(p.nextWake(tl)))
	defer timer.Stop()

	for {
		select {
		case <-p.callDone:
			p.close(false)
			return
		case <-p.progress:
		case <-timer.C:
		}

		now := time.Now()
		p.credit(&tl)
		if stall := p.exceeded(now, tl); stall != nil {
			p.stalled = true
			p.close(true)
			p.cancel(stall)
			return
		}
		switch {
		case tl.pending && now.Sub(tl.lastBeat) >= p.config.MinInterval:
			beat(now, Beat{})
		case now.Sub(tl.lastBeat) >= p.config.Keepalive:
			beat(now, Beat{Keepalive: true, Idle: now.Sub(tl.lastProgress)})
		}
		timer.Reset(time.Until(p.nextWake(tl)))
	}
}

// credit counts the queued events as progress made when the last of them was
// queued.
func (p *Pulse[E]) credit(tl *timeline) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if len(p.queue) == 0 {
		return
	}
	tl.lastProgress, tl.progressed, tl.pending = p.lastQueued, true, true
}

// exceeded returns the limit the call has exceeded at now, or nil.
func (p *Pulse[E]) exceeded(now time.Time, tl timeline) *StallError {
	c := p.config
	switch {
	case c.Total > 0 && now.Sub(tl.start) >= c.Total:
		return &StallError{Limit: LimitTotal, After: c.Total}
	case !tl.progressed && c.FirstProgress > 0 && now.Sub(tl.start) >= c.FirstProgress:
		return &StallError{Limit: LimitFirstProgress, After: c.FirstProgress}
	case tl.progressed && c.Idle > 0 && now.Sub(tl.lastProgress) >= c.Idle:
		return &StallError{Limit: LimitIdle, After: c.Idle}
	}
	return nil
}

// nextWake returns the earliest time at which a heartbeat is due or a limit is
// reached.
func (p *Pulse[E]) nextWake(tl timeline) time.Time {
	c := p.config
	next := tl.lastBeat.Add(c.Keepalive)
	earlier := func(t time.Time) {
		if t.Before(next) {
			next = t
		}
	}
	if tl.pending {
		earlier(tl.lastBeat.Add(c.MinInterval))
	}
	if c.Total > 0 {
		earlier(tl.start.Add(c.Total))
	}
	if !tl.progressed && c.FirstProgress > 0 {
		earlier(tl.start.Add(c.FirstProgress))
	}
	if tl.progressed && c.Idle > 0 {
		earlier(tl.lastProgress.Add(c.Idle))
	}
	return next
}
