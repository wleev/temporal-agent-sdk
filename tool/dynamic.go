package tool

import (
	"encoding/json"
	"fmt"

	"go.temporal.io/sdk/workflow"

	"github.com/wleev/temporal-agent-sdk/model"
)

// DispatchFunc invokes a dynamically-resolved tool. It receives the tool's own
// name (matching Def().Name) and the model's raw JSON arguments, so a single
// func can back many dynamic tools, keyed on the name.
//
// It runs in workflow code, so it must schedule any I/O as an activity and obey
// determinism rules. A returned error fails the call; a result with IsError set
// means the tool ran and reported a failure. The agent loop treats the two
// differently.
type DispatchFunc func(ctx workflow.Context, name string, args json.RawMessage) (*model.CallToolResult, error)

// Dynamic builds a tool from a pre-built [model.Tool] description and a dispatch
// function, rather than from a Go argument type and a typed handler as [New] and
// [Activity] do.
//
// Use it when the toolset is resolved at run time and has no static Go type to
// reflect: an MCP-like registry, a plugin system, a per-tenant tool set. The def
// carries the name, description, and JSON Schema the model sees; dispatch does
// the work. One dispatch func can serve every tool in a set, routing on the name
// it receives.
//
// A def's MCP annotations escalate the policy the same way they do for a server
// tool: an explicit destructive hint turns approval on, but never off. See
// [ApplyAnnotations].
func Dynamic(def *model.Tool, dispatch DispatchFunc, opts ...Option) (Tool, error) {
	if def == nil {
		return nil, fmt.Errorf("tool: def must not be nil")
	}
	if def.Name == "" {
		return nil, fmt.Errorf("tool: def name must not be empty")
	}
	if dispatch == nil {
		return nil, fmt.Errorf("tool %q: dispatch must not be nil", def.Name)
	}
	return &dynamicTool{
		base: base{
			def:    def,
			policy: ApplyAnnotations(Resolve(opts...), def.Annotations),
		},
		dispatch: dispatch,
	}, nil
}

// dynamicTool invokes a tool through a caller-supplied dispatch func.
type dynamicTool struct {
	base
	dispatch DispatchFunc
}

func (t *dynamicTool) Invoke(ctx workflow.Context, args json.RawMessage) (*model.CallToolResult, error) {
	return t.dispatch(ctx, t.def.Name, args)
}
