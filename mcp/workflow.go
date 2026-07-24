package mcp

import (
	"encoding/json"
	"time"

	"go.temporal.io/sdk/temporal"
	"go.temporal.io/sdk/workflow"

	"github.com/wleev/temporal-agent-sdk/model"
	"github.com/wleev/temporal-agent-sdk/tool"
)

// Default timeouts for MCP activities. Listing is a handshake; a call may do
// real work, so it gets more room.
const (
	DefaultListTimeout = 30 * time.Second
	DefaultCallTimeout = 5 * time.Minute
)

// DefaultMaxAttempts bounds retries of an MCP operation, replacing Temporal's
// unlimited default. Tool errors are not retried at all; see [Activities.CallTool].
const DefaultMaxAttempts = 3

// DefaultCallHeartbeatTimeout detects a dead worker mid tool call, well before
// the longer [DefaultCallTimeout]. The call activity heartbeats while it runs.
const DefaultCallHeartbeatTimeout = 30 * time.Second

// Options configures how MCP tools are invoked from a workflow.
type Options struct {
	// ListActivityOptions configures the list-tools activity.
	ListActivityOptions *workflow.ActivityOptions

	// CallActivityOptions configures the call-tool activity.
	CallActivityOptions *workflow.ActivityOptions

	// NamePrefix is prepended to every tool name from this server. Use it when
	// two servers export the same tool name, which would otherwise collide.
	NamePrefix string

	// ToolOptions apply to every tool from this server, e.g.
	// tool.RequiresApproval to gate a whole server behind human review.
	//
	// A server's own annotations may add approval on top of these, but never
	// remove it. See [tool.ApplyAnnotations].
	ToolOptions []tool.Option
}

func defaultActivityOptions(timeout time.Duration) workflow.ActivityOptions {
	return workflow.ActivityOptions{
		StartToCloseTimeout: timeout,
		RetryPolicy:         &temporal.RetryPolicy{MaximumAttempts: DefaultMaxAttempts},
	}
}

// defaultCallActivityOptions are the call-tool defaults: the call activity
// heartbeats, so it also carries a HeartbeatTimeout.
func defaultCallActivityOptions() workflow.ActivityOptions {
	opts := defaultActivityOptions(DefaultCallTimeout)
	opts.HeartbeatTimeout = DefaultCallHeartbeatTimeout
	return opts
}

// Tools lists a server's tools and adapts them into [tool.Tool] values.
//
// It runs in workflow code and schedules the list-tools activity, so the result
// is recorded in history and stable across replay. Tool descriptions pass
// through untouched: the model sees what the server advertised. Use [ToolsWith]
// to pass [Options].
func Tools(ctx workflow.Context, server string) ([]tool.Tool, error) {
	return ToolsWith(ctx, server, Options{})
}

// ToolsWith is [Tools] with [Options].
func ToolsWith(ctx workflow.Context, server string, o Options) ([]tool.Tool, error) {
	listOpts := defaultActivityOptions(DefaultListTimeout)
	if o.ListActivityOptions != nil {
		listOpts = *o.ListActivityOptions
	}
	listCtx := workflow.WithActivityOptions(ctx, listOpts)

	var defs []*model.Tool
	err := workflow.ExecuteActivity(listCtx, ListToolsActivity, ListToolsInput{Server: server}).
		Get(listCtx, &defs)
	if err != nil {
		return nil, err
	}

	out := make([]tool.Tool, 0, len(defs))
	for _, def := range defs {
		if def == nil {
			continue
		}
		t, err := newMCPTool(server, def, o)
		if err != nil {
			return nil, err
		}
		out = append(out, t)
	}
	return out, nil
}

// newMCPTool adapts one advertised tool into a dynamic tool that dispatches to
// the call-tool activity.
func newMCPTool(server string, def *model.Tool, o Options) (tool.Tool, error) {
	// Copy before mutating so the caller's def is not renamed.
	local := *def
	local.Name = o.NamePrefix + def.Name

	if isAbsentSchema(local.InputSchema) {
		// The model always needs a schema; a tool with no arguments gets an
		// empty object.
		local.InputSchema = json.RawMessage(`{"type":"object","properties":{}}`)
	}

	// The activity addresses the tool by its remote name, before NamePrefix.
	remoteName := def.Name
	callOptions := o.CallActivityOptions
	dispatch := func(ctx workflow.Context, _ string, args json.RawMessage) (*model.CallToolResult, error) {
		if callOptions != nil {
			ctx = workflow.WithActivityOptions(ctx, *callOptions)
		} else {
			ctx = workflow.WithActivityOptions(ctx, defaultCallActivityOptions())
		}

		var res model.CallToolResult
		err := workflow.ExecuteActivity(ctx, CallToolActivity, CallToolInput{
			Server:    server,
			Tool:      remoteName,
			Arguments: args,
		}).Get(ctx, &res)
		if err != nil {
			return nil, err
		}
		// Returned as-is, IsError included; the loop decides what a tool error means.
		return &res, nil
	}

	return tool.Dynamic(&local, dispatch, o.ToolOptions...)
}

// isAbsentSchema reports whether a server advertised no input schema.
//
// InputSchema is typed any, so absence has several shapes: nil from a Go value,
// or the JSON literal null once it has crossed an activity boundary and come
// back as an untyped value.
func isAbsentSchema(s any) bool {
	switch v := s.(type) {
	case nil:
		return true
	case json.RawMessage:
		t := trimSpace(string(v))
		return t == "" || t == "null"
	case string:
		return trimSpace(v) == ""
	case map[string]any:
		return len(v) == 0
	default:
		return false
	}
}

func trimSpace(s string) string {
	start, end := 0, len(s)
	for start < end && isSpace(s[start]) {
		start++
	}
	for end > start && isSpace(s[end-1]) {
		end--
	}
	return s[start:end]
}

func isSpace(b byte) bool {
	return b == ' ' || b == '\t' || b == '\n' || b == '\r'
}
