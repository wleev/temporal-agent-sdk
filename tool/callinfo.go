package tool

import "go.temporal.io/sdk/workflow"

// CallInfo identifies the tool call a tool is invoked for.
type CallInfo struct {
	// ID is the model's tool-call ID.
	ID string

	// Turn is the 1-based turn of the run that made the call. It equals the
	// agent loop's TurnEvent.Turn for that model call.
	Turn int

	// Index is the 0-based position of the call among the tool calls of its
	// turn's reply.
	Index int
}

// callInfoKey is the context key WithCallInfo stores a CallInfo under.
type callInfoKey struct{}

// WithCallInfo returns a copy of ctx that carries info.
func WithCallInfo(ctx workflow.Context, info CallInfo) workflow.Context {
	return workflow.WithValue(ctx, callInfoKey{}, info)
}

// CallInfoFrom returns the call information ctx carries, and false when it
// carries none.
func CallInfoFrom(ctx workflow.Context) (CallInfo, bool) {
	info, ok := ctx.Value(callInfoKey{}).(CallInfo)
	return info, ok
}
