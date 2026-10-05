package mcp

import "context"

// CallPhase names a point in a tool call.
type CallPhase string

// The phases of a tool call, in order. [CallPhaseProgress] repeats once per
// progress notification the server sends.
const (
	CallPhaseConnecting CallPhase = "connecting"
	CallPhaseCalling    CallPhase = "calling"
	CallPhaseProgress   CallPhase = "progress"
)

// CallProgress is the detail of a call-tool activity heartbeat. The activity
// records one when it starts connecting, when it sends the call, for progress
// notifications from the server (at most one per second), and on a timer while
// the call reports nothing. It is visible on the activity in the Web UI and
// readable by the next attempt via activity.GetHeartbeatDetails.
type CallProgress struct {
	// Server is the name of the MCP server.
	Server string `json:"server"`
	// Tool is the name of the tool.
	Tool string `json:"tool"`
	// Phase is the step the call has reached.
	Phase CallPhase `json:"phase"`

	// Keepalive reports whether the timer produced the heartbeat, as opposed to
	// a change of phase or a progress notification.
	Keepalive bool `json:"keepalive"`

	// IdleSeconds is the time in seconds since the call last changed phase or
	// reported progress, or since the activity started when it has done neither.
	// It is zero unless Keepalive is set.
	IdleSeconds float64 `json:"idle_seconds"`

	// Notifications is the number of progress notifications the server has sent
	// so far.
	Notifications int `json:"notifications,omitempty"`

	// Progress and Total are the amounts of the server's last progress
	// notification. The notification's message is not recorded.
	// They are set when Phase is [CallPhaseProgress].
	Progress float64 `json:"progress,omitempty"`
	Total    float64 `json:"total,omitempty"`
}

// ProgressUpdate is one progress notification a server sent during a tool call.
type ProgressUpdate struct {
	// Progress is the amount done so far.
	Progress float64
	// Total is the amount to do, or zero when unknown.
	Total float64
	// Message describes the current step.
	Message string
}

type progressFuncKey struct{}

// WithProgressFunc returns a context carrying fn, which [ReportProgress] calls
// with each update.
func WithProgressFunc(ctx context.Context, fn func(ProgressUpdate)) context.Context {
	return context.WithValue(ctx, progressFuncKey{}, fn)
}

// HasProgressFunc reports whether ctx carries a function from
// [WithProgressFunc].
func HasProgressFunc(ctx context.Context) bool {
	fn, ok := ctx.Value(progressFuncKey{}).(func(ProgressUpdate))
	return ok && fn != nil
}

// ReportProgress passes u to the function [WithProgressFunc] put on ctx. It does
// nothing when ctx carries none. A [Client] calls it from CallTool, with the
// context CallTool received, for each progress notification of that call.
func ReportProgress(ctx context.Context, u ProgressUpdate) {
	if fn, ok := ctx.Value(progressFuncKey{}).(func(ProgressUpdate)); ok && fn != nil {
		fn(u)
	}
}

// Pinger is implemented by a [Client] that can check its connection. Ping
// returns an error when the server does not answer.
type Pinger interface {
	Ping(ctx context.Context) error
}

// SessionProgress is the detail of a session-holder activity heartbeat. The
// holder records one when the session opens; when a call on it starts, reports
// progress, or finishes; and every check interval until the session is lost.
type SessionProgress struct {
	Server string `json:"server"`
	// Calls is the number of calls the session has finished.
	Calls int64 `json:"calls"`
	// InFlight is the number of calls running on the session.
	InFlight int32 `json:"in_flight"`
	// Verified reports whether the session answered a ping for this heartbeat.
	Verified bool `json:"verified"`
	// FailedPings is the number of pings in a row the session has failed.
	FailedPings int `json:"failed_pings,omitempty"`
}
