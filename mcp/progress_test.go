package mcp

import (
	"context"
	"encoding/json"
	"errors"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	enumspb "go.temporal.io/api/enums/v1"
	"go.temporal.io/sdk/activity"
	"go.temporal.io/sdk/temporal"
	"go.temporal.io/sdk/testsuite"
	"go.temporal.io/sdk/workflow"

	"github.com/wleev/temporal-agent-sdk/internal/heartbeat"
	"github.com/wleev/temporal-agent-sdk/model"
)

// progressClient is a [Client] whose CallTool reports updates through the
// context before it returns, and whose Ping returns pingErr.
type progressClient struct {
	updates []ProgressUpdate
	// pause is how long CallTool waits after reporting updates.
	pause   time.Duration
	callErr error
	pingErr error
	pings   atomic.Int32
}

func (*progressClient) ListTools(context.Context) ([]*model.Tool, error) { return nil, nil }

func (c *progressClient) CallTool(ctx context.Context, _ string, _ json.RawMessage) (*model.CallToolResult, error) {
	for _, u := range c.updates {
		ReportProgress(ctx, u)
	}
	time.Sleep(c.pause)
	if c.callErr != nil {
		return nil, c.callErr
	}
	return model.TextResult("ok"), nil
}

func (*progressClient) Close() error { return nil }

func (c *progressClient) Ping(context.Context) error {
	c.pings.Add(1)
	return c.pingErr
}

// plainClient is a [Client] that does not implement [Pinger].
type plainClient struct{}

func (plainClient) ListTools(context.Context) ([]*model.Tool, error) { return nil, nil }

func (plainClient) CallTool(context.Context, string, json.RawMessage) (*model.CallToolResult, error) {
	return model.TextResult("ok"), nil
}

func (plainClient) Close() error { return nil }

func activitiesFor(t *testing.T, c Client) *Activities {
	t.Helper()
	acts, err := NewActivities(WithServer("fs", func(context.Context) (Client, error) { return c, nil }))
	require.NoError(t, err)
	return acts
}

// callRecordingBeats runs CallTool on acts with a keepalive of 20ms and returns
// the CallProgress detail of every heartbeat it recorded, in order, and the
// call's error.
func callRecordingBeats(t *testing.T, acts *Activities, in CallToolInput) ([]CallProgress, error) {
	t.Helper()
	var mu sync.Mutex
	var beats []CallProgress
	acts.pulse = heartbeat.Config{
		Keepalive:   20 * time.Millisecond,
		MinInterval: time.Nanosecond,
		Record: func(detail any) {
			mu.Lock()
			defer mu.Unlock()
			beats = append(beats, detail.(CallProgress))
		},
	}
	_, err := acts.CallTool(t.Context(), in)
	mu.Lock()
	defer mu.Unlock()
	return beats, err
}

func TestCallTool_HeartbeatsWithThePhaseAndTheServersProgress(t *testing.T) {
	acts := activitiesFor(t, &progressClient{
		updates: []ProgressUpdate{
			{Progress: 1, Total: 3, Message: "page 1"},
			{Progress: 2, Total: 3, Message: "page 2"},
		},
	})

	beats, err := callRecordingBeats(t, acts, CallToolInput{Server: "fs", Tool: "read"})

	require.NoError(t, err)
	if assert.NotEmpty(t, beats) {
		var progress []CallProgress
		for _, b := range beats {
			if !b.Keepalive {
				progress = append(progress, b)
			}
		}
		if assert.NotEmpty(t, progress, "heartbeats produced by the call") {
			last := progress[len(progress)-1]
			last.IdleSeconds = 0
			assert.Equal(t, CallProgress{
				Server: "fs", Tool: "read", Phase: CallPhaseProgress,
				Notifications: 2, Progress: 2, Total: 3,
			}, last)
		}
	}
}

func TestCallState_DetailFoldsTheCallEvents(t *testing.T) {
	state := &callState{progress: CallProgress{Server: "fs", Tool: "read", Phase: CallPhaseConnecting}}

	assert.Equal(t, CallProgress{Server: "fs", Tool: "read", Phase: CallPhaseConnecting},
		state.Detail(heartbeat.Beat{}, nil))

	got := state.Detail(heartbeat.Beat{}, []callEvent{
		{phase: CallPhaseCalling},
		{phase: CallPhaseProgress, update: ProgressUpdate{Progress: 1, Total: 3, Message: "page 1"}},
		{phase: CallPhaseProgress, update: ProgressUpdate{Progress: 2, Total: 3, Message: "page 2"}},
	})

	want := CallProgress{
		Server: "fs", Tool: "read", Phase: CallPhaseProgress,
		Notifications: 2, Progress: 2, Total: 3,
	}
	assert.Equal(t, want, got, "one heartbeat holds every event queued since the last")

	want.Keepalive, want.IdleSeconds = true, 4
	assert.Equal(t, want, state.Detail(heartbeat.Beat{Keepalive: true, Idle: 4 * time.Second}, nil))
}

func TestCallTool_SilentCallHeartbeatsOnATimer(t *testing.T) {
	acts := activitiesFor(t, &progressClient{pause: 150 * time.Millisecond})

	beats, err := callRecordingBeats(t, acts, CallToolInput{Server: "fs", Tool: "read"})

	require.NoError(t, err)
	var keepalives []CallProgress
	for _, b := range beats {
		if b.Keepalive {
			keepalives = append(keepalives, b)
		}
	}
	if assert.GreaterOrEqual(t, len(keepalives), 2) {
		for _, b := range keepalives {
			assert.Equal(t, CallPhaseCalling, b.Phase)
			assert.Positive(t, b.IdleSeconds)
		}
	}
}

func TestCallTool_FailedCallReturnsTheCallsError(t *testing.T) {
	boom := errors.New("transport closed")
	acts := activitiesFor(t, &progressClient{callErr: boom})

	_, err := callRecordingBeats(t, acts, CallToolInput{Server: "fs", Tool: "read"})

	assert.ErrorIs(t, err, boom)
}

func TestReportProgress_WithoutAFunctionDoesNothing(t *testing.T) {
	ReportProgress(t.Context(), ProgressUpdate{Progress: 1})
}

func TestDefaultCallActivityOptions_SetAHeartbeatTimeout(t *testing.T) {
	opts := defaultCallActivityOptions()

	assert.Equal(t, DefaultCallTimeout, opts.StartToCloseTimeout)
	assert.Equal(t, DefaultCallHeartbeatTimeout, opts.HeartbeatTimeout)
}

// collector gathers the SessionProgress values a session records.
type collector struct {
	mu  sync.Mutex
	got []SessionProgress
}

func (c *collector) record(p SessionProgress) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.got = append(c.got, p)
}

func (c *collector) all() []SessionProgress {
	c.mu.Lock()
	defer c.mu.Unlock()
	return append([]SessionProgress(nil), c.got...)
}

// hold runs holdSession over ops in the background. tick delivers one tick to
// it; stop cancels it and returns its error.
type hold struct {
	ticks  chan time.Time
	cancel context.CancelFunc
	done   chan error
}

// testTiming pings on every tick, tolerates two failed pings in a row, and
// allows a call 5s past its deadline.
var testTiming = holdTiming{pingTimeout: time.Second, maxFailedPings: 3, callGrace: 5 * time.Second}

func startHold(t *testing.T, ops *sessionOps) *hold {
	t.Helper()
	return startHoldWith(t, ops, testTiming)
}

func startHoldWith(t *testing.T, ops *sessionOps, timing holdTiming) *hold {
	t.Helper()
	ctx, cancel := context.WithCancel(t.Context())
	h := &hold{ticks: make(chan time.Time), cancel: cancel, done: make(chan error, 1)}
	go func() { h.done <- holdSession(ctx, ops, h.ticks, timing) }()
	return h
}

func (h *hold) tick(t *testing.T, now time.Time) {
	t.Helper()
	select {
	case h.ticks <- now:
	case err := <-h.done:
		h.done <- err
	case <-time.After(5 * time.Second):
		t.Fatal("holdSession did not take the tick within 5s")
	}
}

func (h *hold) wait(t *testing.T) error {
	t.Helper()
	select {
	case err := <-h.done:
		return err
	case <-time.After(5 * time.Second):
		t.Fatal("holdSession did not return within 5s")
		return nil
	}
}

func (h *hold) stop(t *testing.T) error {
	t.Helper()
	h.cancel()
	return h.wait(t)
}

// eventually waits for c to hold n records and returns them.
func (c *collector) eventually(t *testing.T, n int) []SessionProgress {
	t.Helper()
	require.Eventually(t, func() bool { return len(c.all()) >= n }, 5*time.Second, time.Millisecond,
		"records; got %v", c.all())
	return c.all()
}

func TestHoldSession_RecordsOnceForEachAnsweredPing(t *testing.T) {
	session := &progressClient{}
	var c collector
	ops := &sessionOps{server: "browser", session: session, record: c.record}
	h := startHold(t, ops)

	h.tick(t, time.Now())
	h.tick(t, time.Now())
	got := c.eventually(t, 2)

	require.ErrorIs(t, h.stop(t), context.Canceled)
	assert.Equal(t, []SessionProgress{
		{Server: "browser", Verified: true},
		{Server: "browser", Verified: true},
	}, got)
	assert.Equal(t, int32(2), session.pings.Load())
}

func TestHoldSession_FailsAsSessionLostAfterConsecutiveFailedPings(t *testing.T) {
	session := &progressClient{pingErr: errors.New("connection reset")}
	var c collector
	ops := &sessionOps{server: "browser", session: session, record: c.record}
	h := startHold(t, ops)

	h.tick(t, time.Now())
	h.tick(t, time.Now())
	h.tick(t, time.Now())
	err := h.wait(t)

	var appErr *temporal.ApplicationError
	require.ErrorAs(t, err, &appErr)
	assert.Equal(t, ErrorTypeSessionLost, appErr.Type())
	assert.True(t, appErr.NonRetryable())
	assert.Equal(t, []SessionProgress{
		{Server: "browser", FailedPings: 1},
		{Server: "browser", FailedPings: 2},
	}, c.all(), "the failed pings before the last are recorded")
}

// flakyClient is a [progressClient] whose pings return errs in order, then nil.
type flakyClient struct {
	*progressClient
	errs []error
}

func (f *flakyClient) Ping(ctx context.Context) error {
	_ = f.progressClient.Ping(ctx)
	if len(f.errs) == 0 {
		return nil
	}
	err := f.errs[0]
	f.errs = f.errs[1:]
	return err
}

func TestHoldSession_AnsweredPingResetsTheFailedPings(t *testing.T) {
	reset := errors.New("connection reset")
	session := &flakyClient{progressClient: &progressClient{}, errs: []error{reset, reset, nil, reset}}
	var c collector
	ops := &sessionOps{server: "browser", session: session, record: c.record}
	h := startHold(t, ops)

	for range 4 {
		h.tick(t, time.Now())
	}
	got := c.eventually(t, 4)

	require.ErrorIs(t, h.stop(t), context.Canceled)
	assert.Equal(t, []SessionProgress{
		{Server: "browser", FailedPings: 1},
		{Server: "browser", FailedPings: 2},
		{Server: "browser", Verified: true},
		{Server: "browser", FailedPings: 1},
	}, got)
}

func TestHoldSession_PingsOncePerPingInterval(t *testing.T) {
	session := &progressClient{}
	var c collector
	ops := &sessionOps{server: "browser", session: session, record: c.record}
	timing := testTiming
	timing.pingInterval = 10 * time.Second
	h := startHoldWith(t, ops, timing)

	start := time.Now()
	for _, at := range []time.Duration{0, 3 * time.Second, 6 * time.Second, 10 * time.Second} {
		h.tick(t, start.Add(at))
	}
	got := c.eventually(t, 4)

	require.ErrorIs(t, h.stop(t), context.Canceled)
	assert.Equal(t, int32(2), session.pings.Load())
	assert.Equal(t, []SessionProgress{
		{Server: "browser", Verified: true},
		{Server: "browser"},
		{Server: "browser"},
		{Server: "browser", Verified: true},
	}, got, "a heartbeat is verified only when it follows a ping")
}

func TestHoldSession_DoesNotPingWhileACallIsWithinItsDeadline(t *testing.T) {
	session := &progressClient{}
	var c collector
	ops := &sessionOps{server: "browser", session: session, record: c.record}
	now := time.Now()
	callCtx, cancel := context.WithDeadline(t.Context(), now.Add(time.Minute))
	defer cancel()
	ops.begin(callCtx)
	h := startHold(t, ops)

	h.tick(t, now.Add(30*time.Second))
	got := c.eventually(t, 2)

	require.ErrorIs(t, h.stop(t), context.Canceled)
	assert.Zero(t, session.pings.Load())
	assert.Equal(t, []SessionProgress{
		{Server: "browser", InFlight: 1},
		{Server: "browser", InFlight: 1},
	}, got, "one record when the call began and one for the tick")
}

func TestHoldSession_FailsAsSessionLostWhenACallOutlivesItsDeadline(t *testing.T) {
	var c collector
	ops := &sessionOps{server: "browser", session: &progressClient{}, record: c.record}
	now := time.Now()
	callCtx, cancel := context.WithDeadline(t.Context(), now.Add(time.Minute))
	defer cancel()
	ops.begin(callCtx)
	h := startHold(t, ops)

	h.tick(t, now.Add(2*time.Minute))
	err := h.wait(t)

	var appErr *temporal.ApplicationError
	require.ErrorAs(t, err, &appErr)
	assert.Equal(t, ErrorTypeSessionLost, appErr.Type())
	assert.Len(t, c.all(), 1, "only the call's start is recorded")
}

func TestHoldSession_AllowsACallAGracePeriodPastItsDeadline(t *testing.T) {
	var c collector
	ops := &sessionOps{server: "browser", session: &progressClient{}, record: c.record}
	now := time.Now()
	callCtx, cancel := context.WithDeadline(t.Context(), now.Add(time.Minute))
	defer cancel()
	ops.begin(callCtx)
	h := startHold(t, ops)

	h.tick(t, now.Add(time.Minute+testTiming.callGrace/2))
	got := c.eventually(t, 2)

	require.ErrorIs(t, h.stop(t), context.Canceled)
	assert.Len(t, got, 2, "the call's start and the tick within the grace period")
}

func TestHoldSession_WithoutAPingerRecordsUnverified(t *testing.T) {
	var c collector
	ops := &sessionOps{server: "browser", session: plainClient{}, record: c.record}
	h := startHold(t, ops)

	h.tick(t, time.Now())
	got := c.eventually(t, 1)

	require.ErrorIs(t, h.stop(t), context.Canceled)
	assert.Equal(t, []SessionProgress{{Server: "browser"}}, got)
}

func TestSessionOps_RecordsWhenACallStartsReportsProgressAndFinishes(t *testing.T) {
	var c collector
	session := &progressClient{updates: []ProgressUpdate{{Progress: 1, Total: 2}}}
	ops := &sessionOps{server: "browser", session: session, record: c.record}

	_, err := ops.callTool(t.Context(), CallToolInput{Tool: "click"})
	require.NoError(t, err)
	_, err = ops.listTools(t.Context())
	require.NoError(t, err)

	assert.Equal(t, []SessionProgress{
		{Server: "browser", InFlight: 1},
		{Server: "browser", InFlight: 1},
		{Server: "browser", Calls: 1},
		{Server: "browser", Calls: 1, InFlight: 1},
		{Server: "browser", Calls: 2},
	}, c.all(), "start, one progress notification, finish; then start and finish")
}

func TestSessionTiming_FollowsTheHeartbeatTimeout(t *testing.T) {
	tests := []struct {
		name             string
		heartbeatTimeout time.Duration
		wantInterval     time.Duration
		wantPingTimeout  time.Duration
	}{
		{name: "none set", wantInterval: sessionCheckInterval, wantPingTimeout: sessionPingTimeout},
		{name: "default", heartbeatTimeout: DefaultSessionHeartbeatTimeout,
			wantInterval: sessionCheckInterval, wantPingTimeout: 7500 * time.Millisecond},
		{name: "short", heartbeatTimeout: 8 * time.Second,
			wantInterval: 2 * time.Second, wantPingTimeout: 2 * time.Second},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			interval, timing := sessionTiming(tt.heartbeatTimeout)

			assert.Equal(t, tt.wantInterval, interval)
			assert.Equal(t, tt.wantPingTimeout, timing.pingTimeout)
			assert.Equal(t, max(sessionPingInterval, interval), timing.pingInterval)
			assert.Equal(t, sessionMaxFailedPings, timing.maxFailedPings)
			assert.Equal(t, interval, timing.callGrace)
			if tt.heartbeatTimeout > 0 {
				assert.Less(t, interval+timing.pingTimeout, tt.heartbeatTimeout,
					"a tick and a ping that takes its whole timeout fit inside the heartbeat timeout")
			}
		})
	}
}

func TestStatefulSession_CloseAcceptsAHolderThatEnded(t *testing.T) {
	tests := []struct {
		name    string
		err     error
		wantErr bool
	}{
		{name: "lost", err: temporal.NewNonRetryableApplicationError("session lost", ErrorTypeSessionLost, nil)},
		{name: "heartbeat timeout", err: temporal.NewTimeoutError(enumspb.TIMEOUT_TYPE_HEARTBEAT, nil)},
		{name: "lifetime reached", err: temporal.NewTimeoutError(enumspb.TIMEOUT_TYPE_START_TO_CLOSE, nil)},
		{name: "other failure", err: temporal.NewNonRetryableApplicationError("boom", "Other", nil), wantErr: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var s testsuite.WorkflowTestSuite
			env := s.NewTestWorkflowEnvironment()
			env.RegisterActivityWithOptions(func(context.Context, SessionInput) error {
				return tt.err
			}, activity.RegisterOptions{Name: RunSessionActivity})

			env.ExecuteWorkflow(func(ctx workflow.Context) error {
				session, err := OpenStatefulSession(ctx, "browser")
				if err != nil {
					return err
				}
				if err := workflow.Await(ctx, session.future.IsReady); err != nil {
					return err
				}
				return session.Close(ctx)
			})

			require.True(t, env.IsWorkflowCompleted())
			if tt.wantErr {
				assert.Error(t, env.GetWorkflowError())
			} else {
				assert.NoError(t, env.GetWorkflowError())
			}
		})
	}
}

func TestHolderOptions_DoNotRetryAHolderThatTimedOut(t *testing.T) {
	opts := holderOptions(StatefulOptions{})

	assert.Equal(t, DefaultSessionLifetime, opts.StartToCloseTimeout)
	assert.Equal(t, DefaultSessionHeartbeatTimeout, opts.HeartbeatTimeout)
	assert.True(t, opts.WaitForCancellation)
	if assert.NotNil(t, opts.RetryPolicy) {
		assert.ElementsMatch(t,
			[]string{"TemporalTimeout:StartToClose", "TemporalTimeout:Heartbeat"},
			opts.RetryPolicy.NonRetryableErrorTypes)
		assert.Zero(t, opts.RetryPolicy.MaximumAttempts, "a failed connection is still retried")
	}
}

func TestHolderOptions_KeepACallersRetryPolicy(t *testing.T) {
	policy := &temporal.RetryPolicy{MaximumAttempts: 7}
	opts := holderOptions(StatefulOptions{
		SessionActivityOptions: &workflow.ActivityOptions{RetryPolicy: policy},
	})

	assert.Same(t, policy, opts.RetryPolicy)
	assert.Equal(t, DefaultSessionLifetime, opts.StartToCloseTimeout)
	assert.Equal(t, DefaultSessionHeartbeatTimeout, opts.HeartbeatTimeout)
}

func TestCallOptions_BoundRetries(t *testing.T) {
	s := &StatefulSession{queue: "session-queue"}

	opts := s.callOptions()

	assert.Equal(t, "session-queue", opts.TaskQueue)
	if assert.NotNil(t, opts.RetryPolicy) {
		assert.Equal(t, int32(DefaultMaxAttempts), opts.RetryPolicy.MaximumAttempts)
	}
}

func TestCallOptions_KeepACallersRetryPolicy(t *testing.T) {
	policy := &temporal.RetryPolicy{MaximumAttempts: 9}
	s := &StatefulSession{
		queue: "session-queue",
		opts: StatefulOptions{CallActivityOptions: &workflow.ActivityOptions{
			StartToCloseTimeout: time.Minute,
			RetryPolicy:         policy,
		}},
	}

	assert.Same(t, policy, s.callOptions().RetryPolicy)
}
