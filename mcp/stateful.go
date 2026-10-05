package mcp

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"sync"
	"sync/atomic"
	"time"

	enumspb "go.temporal.io/api/enums/v1"
	"go.temporal.io/sdk/activity"
	"go.temporal.io/sdk/temporal"
	"go.temporal.io/sdk/worker"
	"go.temporal.io/sdk/workflow"

	"github.com/wleev/temporal-agent-sdk/model"
	"github.com/wleev/temporal-agent-sdk/tool"
)

// Stateful MCP keeps one live server connection for the duration of a workflow
// run, so state built up across tool calls — a browser page, an interpreter's
// variables, an open transaction — survives from one call to the next. It is the
// counterpart to the stateless [Activities], which reconnects per call.
//
// A session-holder activity connects the server once and runs a nested Temporal
// worker on a task queue scoped to the run (server@runID). The workflow
// schedules tool calls onto that queue, routing them to the process holding the
// live session. If that process dies, the queue has no poller and a tool call
// times out waiting to start, surfaced as [SessionLostError].
//
// The session lives in one worker's memory and does not survive that worker
// dying: a stateful session trades durability for continuity. Handle
// [SessionLostError] by rebuilding from a known point, and prefer stateless MCP
// wherever the state lives elsewhere.

// Stateful activity names.
const (
	// RunSessionActivity is the session-holder registered on the main worker.
	RunSessionActivity = "agentsdk_mcp_run_session"

	// Op activities are registered only on the per-session nested worker.
	sessionListToolsActivity = "agentsdk_mcp_session_list_tools"
	sessionCallToolActivity  = "agentsdk_mcp_session_call_tool"
)

// ErrorTypeSessionLost is the application-error type of a [SessionLostError]. It
// is non-retryable: the session's in-memory state is gone, so a retry hits a
// fresh session, not the one that held the state.
const ErrorTypeSessionLost = "AgentSDKMCPSessionLost"

// Default timeouts for the stateful path.
const (
	// DefaultSessionLifetime caps how long a session may live; it is the
	// holder activity's StartToClose. Raise it for long sessions.
	DefaultSessionLifetime = time.Hour

	// DefaultSessionHeartbeatTimeout is how long the holder may go without a
	// heartbeat.
	DefaultSessionHeartbeatTimeout = 30 * time.Second

	// DefaultSessionScheduleToStart bounds how long a tool call waits for the
	// session's worker to pick it up. Exceeding it means the worker is gone and
	// signals [SessionLostError].
	DefaultSessionScheduleToStart = 30 * time.Second

	// sessionCheckInterval is the longest interval at which the holder checks
	// the session and records a heartbeat.
	sessionCheckInterval = 3 * time.Second

	// sessionPingTimeout is the longest one ping of an idle session may take.
	sessionPingTimeout = 10 * time.Second

	// sessionPingInterval is the shortest interval between two pings of an idle
	// session.
	sessionPingInterval = 15 * time.Second

	// sessionMaxFailedPings is the number of failed pings in a row that ends a
	// session as lost.
	sessionMaxFailedPings = 3
)

// holdTiming configures how a holder checks its session.
type holdTiming struct {
	// pingInterval is the shortest interval between two pings of an idle
	// session.
	pingInterval time.Duration
	// pingTimeout is the longest one ping may take.
	pingTimeout time.Duration
	// maxFailedPings is the number of failed pings in a row that ends the
	// session as lost.
	maxFailedPings int
	// callGrace is how long a call may run past its deadline before the session
	// is lost.
	callGrace time.Duration
}

// sessionTiming returns the check interval and the check timing for a holder
// with the given HeartbeatTimeout. The interval is [sessionCheckInterval] and
// the ping timeout [sessionPingTimeout], each capped at a quarter of a non-zero
// heartbeatTimeout. Pings are at least [sessionPingInterval] apart, and a call
// gets one interval of grace past its deadline.
func sessionTiming(heartbeatTimeout time.Duration) (interval time.Duration, timing holdTiming) {
	interval, pingTimeout := sessionCheckInterval, sessionPingTimeout
	if heartbeatTimeout > 0 {
		interval = min(interval, heartbeatTimeout/4)
		pingTimeout = min(pingTimeout, heartbeatTimeout/4)
	}
	return interval, holdTiming{
		pingInterval:   max(sessionPingInterval, interval),
		pingTimeout:    pingTimeout,
		maxFailedPings: sessionMaxFailedPings,
		callGrace:      interval,
	}
}

// sessionQueue is the run-scoped task queue that routes tool calls to the worker
// holding a server's live session. It is derived from the run ID, so the
// workflow and the holder activity compute the same name.
func sessionQueue(server, runID string) string {
	return "agentsdk-mcp-session:" + server + "@" + runID
}

// StatefulActivities is the worker-side half of stateful MCP. It owns the server
// factories and registers the session-holder activity.
type StatefulActivities struct {
	servers
}

// NewStatefulActivities creates the stateful activity set for the servers of
// opts. A stateful server's factory is called once per session.
func NewStatefulActivities(opts ...Option) (*StatefulActivities, error) {
	s, err := newServers(opts)
	if err != nil {
		return nil, err
	}
	return &StatefulActivities{servers: s}, nil
}

// RegisterWith wires the session-holder activity into a worker. Only the holder
// is registered here; the per-call tool activities live on each session's nested
// worker.
func (a *StatefulActivities) RegisterWith(r ActivityRegistry) {
	r.RegisterActivityWithOptions(a.RunSession, activity.RegisterOptions{Name: RunSessionActivity})
}

// SessionInput starts a session holder.
type SessionInput struct {
	Server string `json:"server"`
}

// RunSession is the session-holder activity. It connects the server once and
// runs a nested worker that serves this session's tool calls, blocking for the
// session's lifetime until the activity is canceled (on session close) or times
// out.
func (a *StatefulActivities) RunSession(ctx context.Context, in SessionInput) error {
	f, ok := a.factories[in.Server]
	if !ok {
		return temporal.NewNonRetryableApplicationError(
			fmt.Sprintf("no stateful MCP server named %q is registered (have %v)", in.Server, a.names),
			ErrorTypeUnknownServer, nil)
	}

	session, err := f(ctx)
	if err != nil {
		return fmt.Errorf("mcp: connecting stateful server %q: %w", in.Server, err)
	}
	// Closed once when the activity ends, not per call.
	defer closeQuietly(ctx, session, in.Server)

	queue := sessionQueue(in.Server, activity.GetInfo(ctx).WorkflowExecution.RunID)

	// A dedicated worker for this one session: a single poller and a single slot,
	// so the serial in-memory session is never hit concurrently.
	nested := worker.New(activity.GetClient(ctx), queue, worker.Options{
		MaxConcurrentActivityExecutionSize: 1,
		MaxConcurrentActivityTaskPollers:   1,
	})
	ops := &sessionOps{
		server:  in.Server,
		session: session,
		record:  func(p SessionProgress) { activity.RecordHeartbeat(ctx, p) },
	}
	nested.RegisterActivityWithOptions(ops.listTools, activity.RegisterOptions{Name: sessionListToolsActivity})
	nested.RegisterActivityWithOptions(ops.callTool, activity.RegisterOptions{Name: sessionCallToolActivity})

	if err := nested.Start(); err != nil {
		return fmt.Errorf("mcp: starting session worker for %q: %w", in.Server, err)
	}
	defer nested.Stop()

	// The session is connected and its worker is polling.
	ops.record(ops.progress())
	activity.GetLogger(ctx).Info("mcp: stateful session open", "server", in.Server, "queue", queue)

	interval, timing := sessionTiming(activity.GetInfo(ctx).HeartbeatTimeout)
	ticker := time.NewTicker(interval)
	defer ticker.Stop()
	return holdSession(ctx, ops, ticker.C, timing)
}

// holdSession records a heartbeat for the session of ops on every tick until
// ctx is done.
//
// While a call is in flight it records without touching the session, and
// returns a non-retryable [ErrorTypeSessionLost] error once the tick is more
// than timing.callGrace past the call's deadline. With no call in flight it
// pings a session that implements [Pinger] when timing.pingInterval has passed
// since the last ping, holding the session for the ping. A heartbeat that
// follows an answered ping records with Verified set. A failed ping counts
// towards FailedPings, which an answered ping resets; timing.maxFailedPings
// failed pings in a row return a non-retryable [ErrorTypeSessionLost] error.
func holdSession(ctx context.Context, ops *sessionOps, ticks <-chan time.Time, timing holdTiming) error {
	var (
		lastPing    time.Time
		failedPings int
	)
	for {
		var now time.Time
		select {
		case <-ctx.Done():
			return ctx.Err()
		case now = <-ticks:
		}

		if !ops.mu.TryLock() {
			deadline := ops.deadline.Load()
			if deadline != 0 && now.After(time.Unix(0, deadline).Add(timing.callGrace)) {
				return sessionLost(ops.server, errors.New("a call ran past its deadline"))
			}
			p := ops.progress()
			p.FailedPings = failedPings
			ops.record(p)
			continue
		}
		p := ops.progress()
		pinger, canPing := ops.session.(Pinger)
		due := canPing && now.Sub(lastPing) >= timing.pingInterval
		var pingErr error
		if due {
			pingCtx, cancel := context.WithTimeout(ctx, timing.pingTimeout)
			pingErr = pinger.Ping(pingCtx)
			cancel()
		}
		ops.mu.Unlock()

		if ctx.Err() != nil {
			return ctx.Err()
		}
		if due {
			lastPing = now
			if pingErr != nil {
				failedPings++
				if failedPings >= timing.maxFailedPings {
					return sessionLost(ops.server, fmt.Errorf("%d pings in a row failed, the last: %w", failedPings, pingErr))
				}
			} else {
				failedPings = 0
			}
			p.Verified = pingErr == nil
		}
		p.FailedPings = failedPings
		ops.record(p)
	}
}

// sessionLost returns a non-retryable [ErrorTypeSessionLost] error for server.
func sessionLost(server string, cause error) error {
	lost := &SessionLostError{Server: server, cause: cause}
	return temporal.NewNonRetryableApplicationError(lost.Error(), ErrorTypeSessionLost, lost)
}

// sessionOps holds the live session; its methods are the per-call activities the
// nested worker serves. They never close the session. Each call records a
// [SessionProgress] when it starts, on every progress notification from the
// server, and when it finishes.
type sessionOps struct {
	server  string
	session Client
	record  func(SessionProgress)

	// mu is held while a call or a ping runs on the session.
	mu sync.Mutex

	calls    atomic.Int64
	inFlight atomic.Int32
	// deadline is the deadline of the call in flight, in Unix nanoseconds, or
	// zero when there is none.
	deadline atomic.Int64
}

// progress returns the session's current counts.
func (o *sessionOps) progress() SessionProgress {
	return SessionProgress{Server: o.server, Calls: o.calls.Load(), InFlight: o.inFlight.Load()}
}

// begin takes the session for a call bounded by ctx's deadline and records the
// call's start.
func (o *sessionOps) begin(ctx context.Context) {
	o.mu.Lock()
	if deadline, ok := ctx.Deadline(); ok {
		o.deadline.Store(deadline.UnixNano())
	}
	o.inFlight.Add(1)
	o.record(o.progress())
}

// end records the call's finish and releases the session.
func (o *sessionOps) end() {
	o.deadline.Store(0)
	o.inFlight.Add(-1)
	o.calls.Add(1)
	o.record(o.progress())
	o.mu.Unlock()
}

func (o *sessionOps) listTools(ctx context.Context) ([]*model.Tool, error) {
	o.begin(ctx)
	defer o.end()
	return o.session.ListTools(ctx)
}

func (o *sessionOps) callTool(ctx context.Context, in CallToolInput) (*model.CallToolResult, error) {
	o.begin(ctx)
	defer o.end()
	ctx = WithProgressFunc(ctx, func(ProgressUpdate) { o.record(o.progress()) })
	res, err := o.session.CallTool(ctx, in.Tool, in.Arguments)
	if err != nil {
		return nil, fmt.Errorf("mcp: calling tool %q: %w", in.Tool, err)
	}
	if res == nil {
		return nil, fmt.Errorf("mcp: tool %q returned no result", in.Tool)
	}
	return res, nil
}

// SessionLostError reports that a stateful session is gone: its worker died,
// it failed a ping, or a call on it ran past its deadline. The session's
// in-memory state (browser page, interpreter heap, transaction) no longer
// exists. It is a non-retryable application error of type [ErrorTypeSessionLost];
// callers detect it to rebuild the session or fail deliberately.
type SessionLostError struct {
	Server string
	cause  error
}

func (e *SessionLostError) Error() string {
	return fmt.Sprintf("mcp: stateful session for %q was lost: %v", e.Server, e.cause)
}
func (e *SessionLostError) Unwrap() error { return e.cause }

// asSessionLost converts a lost-worker timeout into a [SessionLostError] wrapped
// as a non-retryable application error; other errors pass through unchanged.
//
// A schedule-to-start timeout means the run-scoped queue had no poller (the
// holder died); a heartbeat timeout means the holder stopped heartbeating. Both
// mean the session is gone.
func asSessionLost(server string, err error) error {
	if err == nil {
		return nil
	}
	var to *temporal.TimeoutError
	if errors.As(err, &to) {
		switch to.TimeoutType() {
		case enumspb.TIMEOUT_TYPE_SCHEDULE_TO_START, enumspb.TIMEOUT_TYPE_HEARTBEAT:
			lost := &SessionLostError{Server: server, cause: err}
			return temporal.NewNonRetryableApplicationError(lost.Error(), ErrorTypeSessionLost, lost)
		}
	}
	return err
}

// StatefulOptions configures an opened session.
type StatefulOptions struct {
	// SessionActivityOptions configures the holder activity. Zero values become
	// the defaults: StartToClose [DefaultSessionLifetime], HeartbeatTimeout
	// [DefaultSessionHeartbeatTimeout], and a RetryPolicy that retries a failed
	// connection and does not retry a holder that timed out.
	SessionActivityOptions *workflow.ActivityOptions

	// CallActivityOptions configures the tool-call/list activities. The TaskQueue
	// is always overridden to the run-scoped queue; a zero ScheduleToStartTimeout
	// becomes [DefaultSessionScheduleToStart] and a nil RetryPolicy becomes one
	// bounded at [DefaultMaxAttempts].
	CallActivityOptions *workflow.ActivityOptions

	// NamePrefix is prepended to every tool name from this server.
	NamePrefix string

	// ToolOptions apply to every tool from this server (e.g. approval gating).
	ToolOptions []tool.Option
}

// StatefulSession is a live MCP session opened from workflow code. Close it when
// done; a running session holds a worker slot and an open connection.
type StatefulSession struct {
	server string
	queue  string
	cancel workflow.CancelFunc
	future workflow.Future
	opts   StatefulOptions
}

// OpenStatefulSession starts a session holder for server and returns a handle to
// its tools. The holder runs in the background for the session's lifetime; the
// caller owns teardown via [StatefulSession.Close].
//
// The session is scoped to this workflow run. Do not continue-as-new with a
// session open: the run ID changes, orphaning the session's queue. Close first.
func OpenStatefulSession(ctx workflow.Context, server string) (*StatefulSession, error) {
	return OpenStatefulSessionWith(ctx, server, StatefulOptions{})
}

// OpenStatefulSessionWith is [OpenStatefulSession] with [StatefulOptions].
func OpenStatefulSessionWith(ctx workflow.Context, server string, o StatefulOptions) (*StatefulSession, error) {
	queue := sessionQueue(server, workflow.GetInfo(ctx).WorkflowExecution.RunID)
	holderOpts := holderOptions(o)

	// Run the holder in the background under a cancel scope; hold the future
	// rather than waiting on it, since the activity lives for the session's
	// duration.
	holderCtx, cancel := workflow.WithCancel(ctx)
	holderCtx = workflow.WithActivityOptions(holderCtx, holderOpts)
	future := workflow.ExecuteActivity(holderCtx, RunSessionActivity, SessionInput{Server: server})

	return &StatefulSession{server: server, queue: queue, cancel: cancel, future: future, opts: o}, nil
}

// The error types Temporal gives a timed-out activity, as matched by a retry
// policy's NonRetryableErrorTypes.
const (
	timeoutTypeStartToClose = "TemporalTimeout:StartToClose"
	timeoutTypeHeartbeat    = "TemporalTimeout:Heartbeat"
)

// holderOptions returns the activity options of the session holder. Without a
// RetryPolicy from the caller, a holder that fails to connect is retried and one
// that times out is not.
func holderOptions(o StatefulOptions) workflow.ActivityOptions {
	opts := workflow.ActivityOptions{
		StartToCloseTimeout: DefaultSessionLifetime,
		HeartbeatTimeout:    DefaultSessionHeartbeatTimeout,
	}
	if o.SessionActivityOptions != nil {
		opts = *o.SessionActivityOptions
		if opts.StartToCloseTimeout == 0 {
			opts.StartToCloseTimeout = DefaultSessionLifetime
		}
		if opts.HeartbeatTimeout == 0 {
			opts.HeartbeatTimeout = DefaultSessionHeartbeatTimeout
		}
	}
	if opts.RetryPolicy == nil {
		// A holder that timed out held a session; a new attempt would connect a
		// new one.
		opts.RetryPolicy = &temporal.RetryPolicy{
			NonRetryableErrorTypes: []string{timeoutTypeStartToClose, timeoutTypeHeartbeat},
		}
	}
	// Close waits for the holder to stop its worker and close the connection.
	opts.WaitForCancellation = true
	return opts
}

// callOptions builds the activity options for a tool call: the run-scoped queue,
// a schedule-to-start bound so a dead session surfaces, and, without a
// RetryPolicy from the caller, at most [DefaultMaxAttempts] attempts.
func (s *StatefulSession) callOptions() workflow.ActivityOptions {
	opts := workflow.ActivityOptions{
		StartToCloseTimeout:    DefaultCallTimeout,
		ScheduleToStartTimeout: DefaultSessionScheduleToStart,
	}
	if s.opts.CallActivityOptions != nil {
		opts = *s.opts.CallActivityOptions
	}
	if opts.ScheduleToStartTimeout == 0 {
		opts.ScheduleToStartTimeout = DefaultSessionScheduleToStart
	}
	if opts.RetryPolicy == nil {
		opts.RetryPolicy = &temporal.RetryPolicy{MaximumAttempts: DefaultMaxAttempts}
	}
	opts.TaskQueue = s.queue // always routed to this session's worker
	return opts
}

// Tools lists the session's tools and adapts them into [tool.Tool] values whose
// calls route to the live session.
func (s *StatefulSession) Tools(ctx workflow.Context) ([]tool.Tool, error) {
	ctx = workflow.WithActivityOptions(ctx, s.callOptions())

	var defs []*model.Tool
	err := workflow.ExecuteActivity(ctx, sessionListToolsActivity).Get(ctx, &defs)
	if err != nil {
		return nil, asSessionLost(s.server, err)
	}

	policy := tool.Resolve(s.opts.ToolOptions...)
	out := make([]tool.Tool, 0, len(defs))
	for _, def := range defs {
		if def == nil {
			continue
		}
		out = append(out, newStatefulTool(s, def, policy))
	}
	return out, nil
}

// Close tears the session down: it cancels the holder, which stops the nested
// worker and closes the connection. It returns nil for a holder that was
// canceled and for one that already ended: lost, timed out on its heartbeat (its
// worker is gone), or past its lifetime.
func (s *StatefulSession) Close(ctx workflow.Context) error {
	s.cancel()
	err := s.future.Get(ctx, nil)
	if err == nil || temporal.IsCanceledError(err) {
		return nil
	}
	var appErr *temporal.ApplicationError
	if errors.As(err, &appErr) && appErr.Type() == ErrorTypeSessionLost {
		return nil
	}
	var timeoutErr *temporal.TimeoutError
	if errors.As(err, &timeoutErr) {
		switch timeoutErr.TimeoutType() {
		case enumspb.TIMEOUT_TYPE_HEARTBEAT, enumspb.TIMEOUT_TYPE_START_TO_CLOSE:
			return nil
		}
	}
	return err
}

// statefulTool invokes one tool on a live session, routing to the run-scoped
// queue and translating a lost worker into [SessionLostError].
type statefulTool struct {
	session    *StatefulSession
	def        *model.Tool
	remoteName string
	policy     tool.Policy
}

func newStatefulTool(s *StatefulSession, def *model.Tool, policy tool.Policy) *statefulTool {
	local := *def
	local.Name = s.opts.NamePrefix + def.Name
	if isAbsentSchema(local.InputSchema) {
		local.InputSchema = json.RawMessage(`{"type":"object","properties":{}}`)
	}
	return &statefulTool{session: s, def: &local, remoteName: def.Name, policy: tool.ApplyAnnotations(policy, def.Annotations)}
}

func (t *statefulTool) Def() *model.Tool    { return t.def }
func (t *statefulTool) Policy() tool.Policy { return t.policy }

func (t *statefulTool) Invoke(ctx workflow.Context, args json.RawMessage) (*model.CallToolResult, error) {
	ctx = workflow.WithActivityOptions(ctx, t.session.callOptions())

	var res model.CallToolResult
	err := workflow.ExecuteActivity(ctx, sessionCallToolActivity, CallToolInput{
		Tool:      t.remoteName,
		Arguments: args,
	}).Get(ctx, &res)
	if err != nil {
		return nil, asSessionLost(t.session.server, err)
	}
	return &res, nil
}
