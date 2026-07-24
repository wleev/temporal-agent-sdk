package mcp

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"sort"
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

	// DefaultSessionHeartbeatTimeout detects a dead holder: it stops heartbeating.
	DefaultSessionHeartbeatTimeout = 30 * time.Second

	// DefaultSessionScheduleToStart bounds how long a tool call waits for the
	// session's worker to pick it up. Exceeding it means the worker is gone and
	// signals [SessionLostError].
	DefaultSessionScheduleToStart = 30 * time.Second

	// sessionHeartbeatInterval also bounds how quickly the holder observes
	// cancellation, i.e. how fast Close tears the session down. Kept well under
	// DefaultSessionHeartbeatTimeout.
	sessionHeartbeatInterval = 3 * time.Second
)

// sessionQueue is the run-scoped task queue that routes tool calls to the worker
// holding a server's live session. It is derived from the run ID, so the
// workflow and the holder activity compute the same name.
func sessionQueue(server, runID string) string {
	return "agentsdk-mcp-session:" + server + "@" + runID
}

// StatefulActivities is the worker-side half of stateful MCP. It owns the server
// factories and registers the session-holder activity.
type StatefulActivities struct {
	factories map[string]Factory
	names     []string
}

// NewStatefulActivities creates an empty stateful activity set.
func NewStatefulActivities() *StatefulActivities {
	return &StatefulActivities{factories: make(map[string]Factory)}
}

// Register adds a session factory under a name. The factory connects a server
// whose session persists for the run; unlike the stateless factory it is called
// once per session, not once per call.
func (a *StatefulActivities) Register(name string, f Factory) error {
	if name == "" {
		return fmt.Errorf("mcp: server name must not be empty")
	}
	if f == nil {
		return fmt.Errorf("mcp: factory for %q must not be nil", name)
	}
	if _, dup := a.factories[name]; dup {
		return fmt.Errorf("mcp: stateful server %q is already registered", name)
	}
	a.factories[name] = f
	a.names = append(a.names, name)
	sort.Strings(a.names)
	return nil
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
	ops := &sessionOps{session: session}
	nested.RegisterActivityWithOptions(ops.listTools, activity.RegisterOptions{Name: sessionListToolsActivity})
	nested.RegisterActivityWithOptions(ops.callTool, activity.RegisterOptions{Name: sessionCallToolActivity})

	if err := nested.Start(); err != nil {
		return fmt.Errorf("mcp: starting session worker for %q: %w", in.Server, err)
	}
	defer nested.Stop()

	// Heartbeat so a dead holder is detected as a heartbeat timeout, and so the
	// activity observes server-side cancellation on older servers.
	go beatSession(ctx)

	activity.GetLogger(ctx).Info("mcp: stateful session open", "server", in.Server, "queue", queue)

	// Block until the session is closed (context canceled) or the lifetime cap is
	// reached; the deferred Stop/Close then tear the session down.
	<-ctx.Done()
	return ctx.Err()
}

// beatSession heartbeats the session holder on a fixed, tight cadence so
// cancellation is observed quickly; it does not use internal/heartbeat, whose
// interval derives from the timeout.
func beatSession(ctx context.Context) {
	t := time.NewTicker(sessionHeartbeatInterval)
	defer t.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-t.C:
			activity.RecordHeartbeat(ctx)
		}
	}
}

// sessionOps holds the live session; its methods are the per-call activities the
// nested worker serves. They never close the session.
type sessionOps struct {
	session Client
}

func (o *sessionOps) listTools(ctx context.Context) ([]*model.Tool, error) {
	return o.session.ListTools(ctx)
}

func (o *sessionOps) callTool(ctx context.Context, in CallToolInput) (*model.CallToolResult, error) {
	res, err := o.session.CallTool(ctx, in.Tool, in.Arguments)
	if err != nil {
		return nil, fmt.Errorf("mcp: calling tool %q: %w", in.Tool, err)
	}
	if res == nil {
		return nil, fmt.Errorf("mcp: tool %q returned no result", in.Tool)
	}
	return res, nil
}

// SessionLostError reports that a stateful session's worker is gone, so its
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
	// [DefaultSessionHeartbeatTimeout].
	SessionActivityOptions *workflow.ActivityOptions

	// CallActivityOptions configures the tool-call/list activities. The TaskQueue
	// is always overridden to the run-scoped queue; a zero ScheduleToStartTimeout
	// becomes [DefaultSessionScheduleToStart].
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

	holderOpts := workflow.ActivityOptions{
		StartToCloseTimeout: DefaultSessionLifetime,
		HeartbeatTimeout:    DefaultSessionHeartbeatTimeout,
	}
	if o.SessionActivityOptions != nil {
		holderOpts = *o.SessionActivityOptions
		if holderOpts.StartToCloseTimeout == 0 {
			holderOpts.StartToCloseTimeout = DefaultSessionLifetime
		}
		if holderOpts.HeartbeatTimeout == 0 {
			holderOpts.HeartbeatTimeout = DefaultSessionHeartbeatTimeout
		}
	}
	// Make Close block until the session is actually torn down (connection
	// closed, nested worker stopped) rather than returning when cancellation is
	// merely requested.
	holderOpts.WaitForCancellation = true

	// Run the holder in the background under a cancel scope; hold the future
	// rather than waiting on it, since the activity lives for the session's
	// duration.
	holderCtx, cancel := workflow.WithCancel(ctx)
	holderCtx = workflow.WithActivityOptions(holderCtx, holderOpts)
	future := workflow.ExecuteActivity(holderCtx, RunSessionActivity, SessionInput{Server: server})

	return &StatefulSession{server: server, queue: queue, cancel: cancel, future: future, opts: o}, nil
}

// callOptions builds the activity options for a tool call: the run-scoped queue
// plus a schedule-to-start bound so a dead session surfaces.
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
// worker and closes the connection. A canceled holder is expected, not an error.
func (s *StatefulSession) Close(ctx workflow.Context) error {
	s.cancel()
	err := s.future.Get(ctx, nil)
	if err == nil || temporal.IsCanceledError(err) {
		return nil
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
