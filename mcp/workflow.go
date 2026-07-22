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
	listCtx := ctx
	if o.ListActivityOptions != nil {
		listCtx = workflow.WithActivityOptions(ctx, *o.ListActivityOptions)
	} else {
		listCtx = workflow.WithActivityOptions(ctx, defaultActivityOptions(DefaultListTimeout))
	}

	var defs []*model.Tool
	err := workflow.ExecuteActivity(listCtx, ListToolsActivity, ListToolsInput{Server: server}).
		Get(listCtx, &defs)
	if err != nil {
		return nil, err
	}

	basePolicy := tool.Resolve(o.ToolOptions...)
	out := make([]tool.Tool, 0, len(defs))
	for _, def := range defs {
		if def == nil {
			continue
		}
		out = append(out, newMCPTool(server, def, basePolicy, o))
	}
	return out, nil
}

// newMCPTool adapts one advertised tool.
func newMCPTool(server string, def *model.Tool, basePolicy tool.Policy, o Options) *mcpTool {
	// Copy before mutating so the caller's def is not renamed.
	local := *def
	local.Name = o.NamePrefix + def.Name

	if isAbsentSchema(local.InputSchema) {
		// The model always needs a schema; a tool with no arguments gets an
		// empty object.
		local.InputSchema = json.RawMessage(`{"type":"object","properties":{}}`)
	}

	// A server's annotations may escalate approval but never relax it.
	policy := tool.ApplyAnnotations(basePolicy, def.Annotations)

	return &mcpTool{
		def:         &local,
		server:      server,
		remoteName:  def.Name,
		policy:      policy,
		callOptions: o.CallActivityOptions,
	}
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

// mcpTool invokes one MCP tool through the call-tool activity.
type mcpTool struct {
	def         *model.Tool
	server      string
	remoteName  string // the server's name, before NamePrefix
	policy      tool.Policy
	callOptions *workflow.ActivityOptions
}

func (t *mcpTool) Def() *model.Tool    { return t.def }
func (t *mcpTool) Policy() tool.Policy { return t.policy }

func (t *mcpTool) Invoke(ctx workflow.Context, args json.RawMessage) (*model.CallToolResult, error) {
	if t.callOptions != nil {
		ctx = workflow.WithActivityOptions(ctx, *t.callOptions)
	} else {
		ctx = workflow.WithActivityOptions(ctx, defaultActivityOptions(DefaultCallTimeout))
	}

	var res model.CallToolResult
	err := workflow.ExecuteActivity(ctx, CallToolActivity, CallToolInput{
		Server:    t.server,
		Tool:      t.remoteName,
		Arguments: args,
	}).Get(ctx, &res)
	if err != nil {
		return nil, err
	}
	// Returned as-is, IsError included; the loop decides what a tool error means.
	return &res, nil
}
