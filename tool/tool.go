// Package tool defines the tools an agent can call.
//
// A tool's argument type is the single source of truth: the JSON Schema sent to
// the model is reflected from it, and the handler receives a decoded value of
// that same type.
//
// A [Tool] has two halves:
//
//   - [Tool.Def] is the tool's MCP description — name, description, schema,
//     annotations. It is MCP's Tool type, so a tool from an MCP server passes
//     through unchanged.
//   - Its execution policy is everything MCP does not describe: where the body
//     runs, whether a human must approve it, its retry policy.
//
// The body runs in one of two places:
//
//   - [New] runs the handler inside the workflow. It adds nothing to history but
//     must obey workflow determinism rules: no clock, no network, no randomness.
//   - [Activity] runs the handler in an activity. It costs history events but
//     gets retries, timeouts, and freedom from determinism rules.
//
// Use [Activity] for handlers that do I/O; a workflow tool must be deterministic.
package tool

import (
	"encoding/json"
	"fmt"
	"time"

	"go.temporal.io/sdk/workflow"

	"github.com/wleev/temporal-agent-sdk/internal/schema"
	"github.com/wleev/temporal-agent-sdk/model"
)

// Tool is something an agent can invoke.
//
// Implementations are constructed once, outside workflow code, and are safe for
// concurrent use across workflow instances: all state lives in the workflow, not
// the tool.
type Tool interface {
	// Def is the tool's MCP description, sent to the model.
	Def() *model.Tool

	// Invoke runs the tool. A returned error means the call failed; a result
	// with IsError set means the tool ran and reported a failure. The agent loop
	// treats the two differently.
	Invoke(ctx workflow.Context, args json.RawMessage) (*model.CallToolResult, error)

	// Policy is the tool's execution policy: approval and anything else MCP does
	// not describe.
	Policy() Policy
}

// Policy is a tool's execution policy — the half of a tool MCP does not
// describe.
type Policy struct {
	// NeedsApproval gates the tool behind human approval.
	NeedsApproval bool

	// ApprovalPrompt is the text shown to an approver.
	ApprovalPrompt string

	// ActivityOptions overrides activity options for activity-backed tools. Nil
	// means inherit from the workflow context.
	ActivityOptions *workflow.ActivityOptions
}

// Option configures a tool.
type Option func(*Policy)

// Resolve applies options and returns the resulting policy.
func Resolve(opts ...Option) Policy {
	var p Policy
	for _, fn := range opts {
		fn(&p)
	}
	return p
}

// RequiresApproval gates the tool behind human approval. The agent loop parks
// before each invocation until an approver decides.
func RequiresApproval() Option {
	return func(p *Policy) { p.NeedsApproval = true }
}

// WithApprovalPrompt sets the text shown to a human approver, gating the tool as
// [RequiresApproval] does.
func WithApprovalPrompt(prompt string) Option {
	return func(p *Policy) {
		p.NeedsApproval = true
		p.ApprovalPrompt = prompt
	}
}

// WithActivityOptions overrides the activity options for a tool built by
// [Activity]. It has no effect on tools built by [New], which schedule no
// activity.
func WithActivityOptions(opts workflow.ActivityOptions) Option {
	return func(p *Policy) { p.ActivityOptions = &opts }
}

// ApplyAnnotations escalates a policy according to a tool's MCP annotations.
//
// Annotations may only add friction, never remove it: an explicit
// DestructiveHint true on a non-read-only tool turns approval on, and nothing
// here ever turns approval off. MCP annotations are untrusted hints from a
// server the operator does not control, so they cannot disable an operator's
// approval gate.
func ApplyAnnotations(p Policy, a *model.ToolAnnotations) Policy {
	if a == nil || p.NeedsApproval {
		return p
	}
	// DestructiveHint is meaningful only when ReadOnlyHint is false and defaults
	// to true when absent, but an absent hint is not evidence of danger, so only
	// an explicit true escalates.
	if !a.ReadOnlyHint && a.DestructiveHint != nil && *a.DestructiveHint {
		p.NeedsApproval = true
		if p.ApprovalPrompt == "" {
			p.ApprovalPrompt = "This tool reports itself as destructive."
		}
	}
	return p
}

// base holds the fields every tool shares.
type base struct {
	def    *model.Tool
	policy Policy
}

func (b base) Def() *model.Tool { return b.def }
func (b base) Policy() Policy   { return b.policy }

// newBase reflects In's schema and assembles the shared fields. It runs at tool
// construction time, outside workflow code, so schema errors surface at startup.
func newBase[In any](name, desc string, opts []Option) (base, error) {
	if name == "" {
		return base{}, fmt.Errorf("tool: name must not be empty")
	}
	params, err := schema.For[In]()
	if err != nil {
		return base{}, fmt.Errorf("tool %q: %w", name, err)
	}
	return base{
		def:    model.NewTool(name, desc, params),
		policy: Resolve(opts...),
	}, nil
}

// funcTool runs its handler in the workflow.
type funcTool[In, Out any] struct {
	base
	fn func(workflow.Context, In) (Out, error)
}

// New builds a tool whose handler runs inside the workflow.
//
// The handler must obey workflow determinism rules — it replays on every
// workflow task, so it must not read the clock, do I/O, or use randomness. For
// anything that does, use [Activity].
//
// In is reflected into the tool's JSON Schema. Out is encoded to JSON for the
// model, except that a string Out is passed through unquoted.
func New[In, Out any](name, desc string, fn func(workflow.Context, In) (Out, error), opts ...Option) (Tool, error) {
	if fn == nil {
		return nil, fmt.Errorf("tool %q: handler must not be nil", name)
	}
	b, err := newBase[In](name, desc, opts)
	if err != nil {
		return nil, err
	}
	return &funcTool[In, Out]{base: b, fn: fn}, nil
}

func (t *funcTool[In, Out]) Invoke(ctx workflow.Context, args json.RawMessage) (*model.CallToolResult, error) {
	in, err := decodeArgs[In](t.def.Name, args)
	if err != nil {
		// Bad arguments are the model's mistake to fix, so report a tool error
		// rather than a failed call.
		return model.ErrorResult(err.Error()), nil
	}
	out, err := t.fn(ctx, in)
	if err != nil {
		return nil, err
	}
	text, err := encodeResult(t.def.Name, out)
	if err != nil {
		return nil, err
	}
	return model.TextResult(text), nil
}

// activityTool runs its handler in an activity.
type activityTool[In, Out any] struct {
	base
	activity any
}

// Activity builds a tool whose body runs as a Temporal activity: retried under
// the activity's retry policy, bounded by its timeouts, recorded in history, and
// free of determinism rules.
//
// The activity must accept In and return (Out, error). It is referenced the same
// way as any Temporal activity — a function value or a registered name.
//
// Activity options resolve in order: [WithActivityOptions], then any options
// already on the workflow context, then a default 30s StartToCloseTimeout.
// Temporal requires StartToClose or ScheduleToClose to be set.
func Activity[In, Out any](name, desc string, activity any, opts ...Option) (Tool, error) {
	if activity == nil {
		return nil, fmt.Errorf("tool %q: activity must not be nil", name)
	}
	b, err := newBase[In](name, desc, opts)
	if err != nil {
		return nil, err
	}
	return &activityTool[In, Out]{base: b, activity: activity}, nil
}

func (t *activityTool[In, Out]) Invoke(ctx workflow.Context, args json.RawMessage) (*model.CallToolResult, error) {
	in, err := decodeArgs[In](t.def.Name, args)
	if err != nil {
		return model.ErrorResult(err.Error()), nil
	}

	// Temporal rejects an activity with no timeout, so one must always be set,
	// including when WithActivityOptions supplies options that leave the timeouts
	// zero.
	if t.policy.ActivityOptions != nil {
		opts := *t.policy.ActivityOptions
		if opts.StartToCloseTimeout == 0 && opts.ScheduleToCloseTimeout == 0 {
			opts.StartToCloseTimeout = defaultToolActivityTimeout
		}
		ctx = workflow.WithActivityOptions(ctx, opts)
	} else if !hasActivityTimeout(ctx) {
		ctx = workflow.WithActivityOptions(ctx, workflow.ActivityOptions{
			StartToCloseTimeout: defaultToolActivityTimeout,
		})
	}

	var out Out
	if err := workflow.ExecuteActivity(ctx, t.activity, in).Get(ctx, &out); err != nil {
		return nil, err
	}
	text, err := encodeResult(t.def.Name, out)
	if err != nil {
		return nil, err
	}
	return model.TextResult(text), nil
}

const defaultToolActivityTimeout = 30 * time.Second

// hasActivityTimeout reports whether the context already carries a timeout that
// satisfies Temporal's scheduling requirement.
func hasActivityTimeout(ctx workflow.Context) bool {
	opts := workflow.GetActivityOptions(ctx)
	return opts.StartToCloseTimeout > 0 || opts.ScheduleToCloseTimeout > 0
}

// decodeArgs turns the model's raw JSON arguments into In. Decoding fixed bytes
// into a fixed type is deterministic and safe to run in workflow code.
//
//workflowcheck:ignore json.Unmarshal is deterministic for fixed input and type
func decodeArgs[In any](name string, args json.RawMessage) (In, error) {
	var in In
	if len(args) == 0 {
		return in, nil
	}
	if err := json.Unmarshal(args, &in); err != nil {
		return in, fmt.Errorf("tool %q: invalid arguments: %w", name, err)
	}
	return in, nil
}

// encodeResult renders a tool result for the model. A string result is passed
// through unquoted; any other value is JSON-encoded. Marshaling a fixed value is
// deterministic and safe to run in workflow code.
//
//workflowcheck:ignore json.Marshal is deterministic for a fixed value
func encodeResult[Out any](name string, out Out) (string, error) {
	if s, ok := any(out).(string); ok {
		return s, nil
	}
	b, err := json.Marshal(out)
	if err != nil {
		return "", fmt.Errorf("tool %q: cannot encode result: %w", name, err)
	}
	return string(b), nil
}

// Must returns t, panicking if err is non-nil. Use it for package-level tool
// declarations, where a schema error should stop startup.
func Must(t Tool, err error) Tool {
	if err != nil {
		panic(err)
	}
	return t
}
