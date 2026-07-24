package agent

import (
	"encoding/json"
	"fmt"
	"sort"
	"time"

	enumspb "go.temporal.io/api/enums/v1"
	"go.temporal.io/sdk/temporal"
	"go.temporal.io/sdk/workflow"

	"github.com/wleev/temporal-agent-sdk/internal/schema"
	"github.com/wleev/temporal-agent-sdk/model"
	"github.com/wleev/temporal-agent-sdk/tool"
)

// WorkflowName is the registered name of the workflow that runs one agent. A
// sub-agent tool starts it as a child workflow.
const WorkflowName = "agentsdk_agent"

// ErrorTypeUnknownAgent marks a reference to an unregistered agent.
const ErrorTypeUnknownAgent = "AgentSDKUnknownAgent"

// WorkflowInput is the argument to [WorkflowName].
//
// It names the agent instead of carrying it, because an [Agent] holds tool
// handlers — Go funcs, which cannot be serialized. The child resolves the name
// against its worker's [Registry], so both workers must register the same
// agents.
type WorkflowInput struct {
	Agent   string          `json:"agent"`
	Input   string          `json:"input"`
	History []model.Message `json:"history,omitempty"`

	// OutputOnly drops the transcript from the returned [Result].
	//
	// A child workflow's return value is serialized into the parent's history as
	// the completion event, so returning the full transcript would copy every
	// child turn into the parent, spending its history budget and risking the 2 MB
	// payload cap. The sub-agent tool needs only the final output, so it sets this.
	OutputOnly bool `json:"output_only,omitempty"`
}

// Registry maps agent names to definitions so that child workflows can resolve
// an agent by name.
//
// Populate it at startup, before starting the worker; it is not safe for
// concurrent modification.
type Registry struct {
	agents map[string]*Agent
}

// NewRegistry creates an empty registry.
func NewRegistry() *Registry {
	return &Registry{agents: make(map[string]*Agent)}
}

// Add registers agents by name. Names must be unique and non-empty.
func (r *Registry) Add(agents ...*Agent) error {
	for _, a := range agents {
		if a == nil {
			return fmt.Errorf("agent: nil agent")
		}
		if _, dup := r.agents[a.name]; dup {
			return fmt.Errorf("agent: %q is already registered", a.name)
		}
		r.agents[a.name] = a
	}
	return nil
}

// Lookup resolves an agent by name.
func (r *Registry) Lookup(name string) (*Agent, bool) {
	a, ok := r.agents[name]
	return a, ok
}

// names lists registered agents in sorted order, for stable error messages. The
// sort makes the result deterministic despite Go's randomized map iteration.
//
//workflowcheck:ignore map iteration order is discarded by the sort
func (r *Registry) names() []string {
	out := make([]string, 0, len(r.agents))
	for name := range r.agents {
		out = append(out, name)
	}
	sort.Strings(out)
	return out
}

// WorkflowRegistry is the part of a worker needed to register workflows.
//
// It is an interface rather than worker.Registry so that test environments,
// which implement only this much, can be passed directly.
type WorkflowRegistry interface {
	RegisterWorkflowWithOptions(w any, options workflow.RegisterOptions)
}

// RegisterWorkflows wires [WorkflowName] into a worker, so that sub-agent child
// workflows can run on it.
func RegisterWorkflows(w WorkflowRegistry, r *Registry) {
	wf := &agentWorkflow{registry: r}
	w.RegisterWorkflowWithOptions(wf.Run, workflow.RegisterOptions{Name: WorkflowName})
}

type agentWorkflow struct {
	registry *Registry
}

// Run executes one registered agent as its own workflow.
func (w *agentWorkflow) Run(ctx workflow.Context, in WorkflowInput) (*Result, error) {
	a, ok := w.registry.Lookup(in.Agent)
	if !ok {
		// The worker's registry is fixed at startup, so this resolves the same
		// way on every attempt.
		return nil, temporal.NewNonRetryableApplicationError(
			fmt.Sprintf("no agent named %q is registered (have %v)", in.Agent, w.registry.names()),
			ErrorTypeUnknownAgent, nil)
	}

	s, err := NewSession(ctx)
	if err != nil {
		return nil, err
	}
	res, err := s.RunWith(ctx, a, in.Input, RunOptions{History: in.History})
	if err != nil {
		return nil, err
	}
	if in.OutputOnly && res != nil {
		// Keep the transcript out of the parent's history; the child retains its
		// own full history. Usage and Turns are still surfaced to the caller.
		res.Messages = nil
	}
	return res, nil
}

// SubAgentOptions configures a sub-agent tool.
type SubAgentOptions struct {
	// ChildOptions configures the child workflow. Its zero value is a reasonable
	// default; see [AsSubAgent] for the fields that are filled in.
	ChildOptions workflow.ChildWorkflowOptions

	// Tool options, e.g. tool.RequiresApproval to gate delegation itself.
	ToolOptions []tool.Option
}

// subAgentInput is the sub-agent tool's argument schema: a single free-text task
// description the parent model writes.
type subAgentInput struct {
	Input string `json:"input" jsonschema_description:"The task or question to delegate to this agent."`
}

// AsSubAgent exposes an agent to a parent agent as a callable tool, backed by a
// child workflow.
//
// The child gets its own workflow execution, and therefore its own 51,200-event
// history budget, its own retry policy, and its own entry in the Web UI, so a
// parent delegating heavy work does not spend its own history on the sub-agent's
// turns.
//
// Sub-agent calls run concurrently when the model requests several at once, since
// the loop dispatches all tool calls in parallel.
//
// A name or description left empty defaults to the sub-agent's name. The
// sub-agent must be registered in the [Registry] used by the worker that runs it,
// or the child fails to resolve it by name. Use [AsSubAgentWith] to override the
// child-workflow or tool options.
func AsSubAgent(sub *Agent, name, description string) (tool.Tool, error) {
	return AsSubAgentWith(sub, name, description, SubAgentOptions{})
}

// AsSubAgentWith is [AsSubAgent] with [SubAgentOptions]. Defaults filled in when
// the options leave them zero:
//
//   - WorkflowID is left empty so Temporal assigns one. Setting it makes
//     concurrent calls to the same sub-agent collide on the ID.
//   - ParentClosePolicy is ABANDON. Temporal's default terminates children when
//     the parent closes, which would kill a sub-agent whose result the parent
//     already returned.
//   - WorkflowExecutionTimeout is 1 hour, bounding a runaway sub-agent.
func AsSubAgentWith(sub *Agent, name, description string, opts SubAgentOptions) (tool.Tool, error) {
	if sub == nil {
		return nil, fmt.Errorf("agent: sub-agent is nil")
	}
	if name == "" {
		name = sub.name
	}
	if description == "" {
		description = fmt.Sprintf("Delegate a task to the %q agent.", sub.name)
	}

	child := opts.ChildOptions
	if child.ParentClosePolicy == enumspb.PARENT_CLOSE_POLICY_UNSPECIFIED {
		child.ParentClosePolicy = enumspb.PARENT_CLOSE_POLICY_ABANDON
	}
	if child.WorkflowExecutionTimeout == 0 {
		child.WorkflowExecutionTimeout = time.Hour
	}

	params, err := schema.For[subAgentInput]()
	if err != nil {
		return nil, err
	}

	t := &subAgentTool{
		def:       model.NewTool(name, description, params),
		agentName: sub.name,
		child:     child,
		policy:    tool.Resolve(opts.ToolOptions...),
	}
	return t, nil
}

// subAgentTool runs a sub-agent as a child workflow.
type subAgentTool struct {
	def       *model.Tool
	agentName string
	child     workflow.ChildWorkflowOptions
	policy    tool.Policy
}

func (t *subAgentTool) Def() *model.Tool    { return t.def }
func (t *subAgentTool) Policy() tool.Policy { return t.policy }

// Invoke starts the sub-agent's child workflow and waits for its result.
//
//workflowcheck:ignore json.Unmarshal is deterministic for fixed input and type
func (t *subAgentTool) Invoke(ctx workflow.Context, args json.RawMessage) (*model.CallToolResult, error) {
	var in subAgentInput
	if len(args) > 0 {
		if err := json.Unmarshal(args, &in); err != nil {
			return model.ErrorResult(
				fmt.Sprintf("sub-agent %q: invalid arguments: %v", t.def.Name, err)), nil
		}
	}

	ctx = workflow.WithChildOptions(ctx, t.child)

	var res Result
	err := workflow.ExecuteChildWorkflow(ctx, WorkflowName, WorkflowInput{
		Agent: t.agentName,
		Input: in.Input,
		// The parent needs only the conclusion; keeping the child's transcript out
		// of the completion payload preserves the parent's history budget.
		OutputOnly: true,
	}).Get(ctx, &res)
	if err != nil {
		return nil, err
	}

	// Only the sub-agent's conclusion goes back to the parent; its full transcript
	// stays in the child's history.
	return model.TextResult(res.Output), nil
}
