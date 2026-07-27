// Package agent runs durable LLM agent loops on Temporal.
//
// The loop runs in workflow code and is replayable: every model call, tool
// result, and approval decision is recorded in workflow history, so a worker can
// crash mid-conversation and another resumes with the full context. Only the LLM
// call and any activity-backed tools leave the workflow.
//
// A minimal agent:
//
//	a, err := agent.NewAgent("assistant", "gpt-5.2",
//		agent.WithInstructions("You are concise and helpful."),
//		agent.WithTools(weatherTool))
//	if err != nil {
//		return err
//	}
//
//	func MyWorkflow(ctx workflow.Context, question string) (string, error) {
//		res, err := agent.Run(ctx, a, question)
//		if err != nil {
//			return "", err
//		}
//		return res.Output, nil
//	}
//
// # History limits
//
// Every turn writes the full prompt and completion into workflow history. A
// Temporal workflow caps at 51,200 events and 50 MB, with 2 MB per individual
// payload, so a long conversation eventually hits that limit. Compact
// [Result.Messages] and continue-as-new to stay under it. Sub-agents run as
// child workflows, each with its own budget.
package agent

import (
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"time"

	"go.temporal.io/sdk/temporal"
	"go.temporal.io/sdk/workflow"

	"github.com/wleev/temporal-agent-sdk/guardrail"
	"github.com/wleev/temporal-agent-sdk/mcp"
	"github.com/wleev/temporal-agent-sdk/model"
	"github.com/wleev/temporal-agent-sdk/tool"
)

// Default loop bounds.
const (
	// DefaultMaxTurns bounds a single Run. Each turn is one model call plus any
	// tools it asked for. The bound stops a model that keeps calling tools without
	// concluding.
	DefaultMaxTurns = 10

	// DefaultModelTimeout bounds one model call attempt.
	DefaultModelTimeout = 2 * time.Minute

	// DefaultModelHeartbeatTimeout is how long Temporal waits for a heartbeat
	// before treating a model call as lost. The activity heartbeats while a call
	// is in flight, so a dead worker is detected within this window rather than at
	// DefaultModelTimeout.
	DefaultModelHeartbeatTimeout = 30 * time.Second

	// DefaultModelMaxAttempts bounds how many times a model call is retried.
	//
	// Temporal otherwise retries an activity indefinitely, so a persistently
	// failing model call would never let the workflow return. Transient errors
	// (429, 5xx, a network blip) still get several retries; raise this or set
	// ScheduleToCloseTimeout on ModelActivityOptions for more resilience.
	DefaultModelMaxAttempts = 5
)

// Error types surfaced as non-retryable application errors.
const (
	ErrorTypeMaxTurns = "AgentSDKMaxTurnsExceeded"
	ErrorTypeConfig   = "AgentSDKInvalidConfig"
	ErrorTypeTripwire = "AgentSDKGuardrailTripwire"
)

// Agent is a declarative agent definition.
//
// It holds no per-run state and is safe to share across workflows: the
// conversation lives in the [Session] and the workflow's history.
type Agent struct {
	name                 string
	instructions         string
	provider             string
	model                string
	tools                []tool.Tool
	mcpServers           []string
	settings             model.Settings
	maxTurns             int
	maxContinuations     int
	modelActivityOptions workflow.ActivityOptions
	approvalTimeout      time.Duration
	stream               bool
	inputGuardrails      []guardrail.Guardrail
	outputGuardrails     []guardrail.Guardrail
}

// Option configures an [Agent] built by [NewAgent].
type Option func(*Agent)

// NewAgent builds an agent. name and model are required; every other field is an
// [Option] with a sensible default. It returns an error only for an empty
// name or model or a nil tool/guardrail, so construction fails at build time
// rather than on the first run.
func NewAgent(name, modelID string, opts ...Option) (*Agent, error) {
	if name == "" {
		return nil, fmt.Errorf("agent: name must not be empty")
	}
	if modelID == "" {
		return nil, fmt.Errorf("agent %q: model must not be empty", name)
	}

	a := &Agent{name: name, model: modelID}
	for _, o := range opts {
		o(a)
	}

	if a.maxTurns <= 0 {
		a.maxTurns = DefaultMaxTurns
	}
	if a.approvalTimeout <= 0 {
		a.approvalTimeout = DefaultApprovalTimeout
	}
	if a.modelActivityOptions.StartToCloseTimeout == 0 && a.modelActivityOptions.ScheduleToCloseTimeout == 0 {
		a.modelActivityOptions.StartToCloseTimeout = DefaultModelTimeout
	}
	if a.modelActivityOptions.RetryPolicy == nil {
		// Cap Temporal's otherwise-unlimited retries at DefaultModelMaxAttempts.
		a.modelActivityOptions.RetryPolicy = &temporal.RetryPolicy{MaximumAttempts: DefaultModelMaxAttempts}
	}
	if a.modelActivityOptions.HeartbeatTimeout == 0 {
		hb := DefaultModelHeartbeatTimeout
		if st := a.modelActivityOptions.StartToCloseTimeout; st > 0 && st < hb {
			hb = st
		}
		a.modelActivityOptions.HeartbeatTimeout = hb
	}

	for i, t := range a.tools {
		if t == nil {
			return nil, fmt.Errorf("agent %q: tool %d is nil", name, i)
		}
	}
	for i, g := range a.inputGuardrails {
		if g == nil {
			return nil, fmt.Errorf("agent %q: input guardrail %d is nil", name, i)
		}
	}
	for i, g := range a.outputGuardrails {
		if g == nil {
			return nil, fmt.Errorf("agent %q: output guardrail %d is nil", name, i)
		}
	}
	return a, nil
}

// WithInstructions sets the system message.
func WithInstructions(instructions string) Option {
	return func(a *Agent) { a.instructions = instructions }
}

// WithProvider selects a registered model provider by name. Empty selects the
// only registered provider.
func WithProvider(name string) Option {
	return func(a *Agent) { a.provider = name }
}

// WithTools adds tools the model may call. It may be called more than once.
func WithTools(tools ...tool.Tool) Option {
	return func(a *Agent) { a.tools = append(a.tools, tools...) }
}

// WithMCPServers names MCP servers whose tools are listed once per run and merged
// with the agent's tools.
func WithMCPServers(servers ...string) Option {
	return func(a *Agent) { a.mcpServers = append(a.mcpServers, servers...) }
}

// WithSettings sets generation parameters. The zero value defers to the provider.
func WithSettings(settings model.Settings) Option {
	return func(a *Agent) { a.settings = settings }
}

// WithMaxTurns bounds the loop. A non-positive value keeps [DefaultMaxTurns].
func WithMaxTurns(n int) Option {
	return func(a *Agent) { a.maxTurns = n }
}

// WithContinueOnLength turns on length-continuation: when a model response ends
// at the output token limit (finish reason [model.FinishLength]) with no tool
// calls, the loop appends the partial answer and re-invokes to continue it, up to
// maxContinuations extra calls. [Result.Output] is the concatenation of the
// fragments in order.
//
// Each continuation is a model call, so it counts as a turn ([Result.Turns]) and
// against [WithMaxTurns]. Exhausting the budget is not an error: the accumulated
// output is returned with [Result.FinishReason] still [model.FinishLength]. A
// non-positive value disables continuation, which is the default.
func WithContinueOnLength(maxContinuations int) Option {
	return func(a *Agent) { a.maxContinuations = maxContinuations }
}

// WithModelActivityOptions configures the model activity. A zero
// StartToCloseTimeout becomes [DefaultModelTimeout], a nil RetryPolicy becomes
// one bounded at [DefaultModelMaxAttempts], and a zero HeartbeatTimeout becomes
// [DefaultModelHeartbeatTimeout] (capped to StartToCloseTimeout). Temporal owns
// model-call retry.
func WithModelActivityOptions(opts workflow.ActivityOptions) Option {
	return func(a *Agent) { a.modelActivityOptions = opts }
}

// WithApprovalTimeout bounds how long an approval-gated tool waits for a human. A
// non-positive value keeps [DefaultApprovalTimeout]. On expiry the tool reports a
// denial to the model rather than failing the workflow.
func WithApprovalTimeout(d time.Duration) Option {
	return func(a *Agent) { a.approvalTimeout = d }
}

// WithStreaming forwards model-call deltas to the sink configured on the worker's
// model activities (see model.Activities.SetStreamSink). The workflow result is
// identical either way; only whether external consumers receive live tokens
// changes.
func WithStreaming() Option {
	return func(a *Agent) { a.stream = true }
}

// WithInputGuardrails validates the user input before the first model call. A
// tripwire blocks the run before any model call and surfaces as a [TripwireError].
// Guardrails run concurrently but are evaluated in declared order.
func WithInputGuardrails(g ...guardrail.Guardrail) Option {
	return func(a *Agent) { a.inputGuardrails = append(a.inputGuardrails, g...) }
}

// WithOutputGuardrails validates the final answer before it is returned. A
// tripwire blocks the answer and surfaces as a [TripwireError], with the
// transcript preserved on the returned Result.
func WithOutputGuardrails(g ...guardrail.Guardrail) Option {
	return func(a *Agent) { a.outputGuardrails = append(a.outputGuardrails, g...) }
}

// Result is the outcome of a [Session.Run].
type Result struct {
	// Output is the model's final message content.
	Output string `json:"output"`

	// Messages is the full conversation, including the system message and every
	// tool exchange. Pass it as [RunOptions.History] to continue, or compact it
	// before a continue-as-new.
	Messages []model.Message `json:"messages"`

	// Usage is the summed token usage across every model call in the run.
	Usage model.Usage `json:"usage"`

	// Turns is the number of model calls made, including any length-continuation
	// calls (see [WithContinueOnLength]).
	Turns int `json:"turns"`

	// FinishReason is the normalized finish reason of the last model call (see
	// [model.FinishReason]). It is [model.FinishLength] when a run stopped at the
	// token limit, including when a continuation budget was exhausted.
	FinishReason model.FinishReason `json:"finish_reason,omitempty"`

	// StructuredOutput is the terminal message as raw JSON, set only when the run
	// carried an OutputSchema. [RunTyped] decodes it for you.
	StructuredOutput json.RawMessage `json:"structured_output,omitempty"`
}

// RunOptions modifies a single run.
type RunOptions struct {
	// History seeds the conversation with prior messages. If it already contains
	// a system message, the agent's Instructions are not prepended again, which
	// makes a Result.Messages round-trip safe.
	History []model.Message

	// OutputSchema, when set, constrains the model's final answer to JSON
	// matching the schema. Usually set for you by [RunTyped]; set it directly
	// only if you are decoding the raw [Result.StructuredOutput] yourself.
	OutputSchema *model.OutputSchema

	// ExtraTools are added to the agent's own tools for this run only, without
	// mutating the [Agent]. It is how tools resolved at runtime — most notably a
	// stateful MCP session's tools (mcp.OpenStatefulSession), which must outlive
	// a single run — are injected: open the session in the workflow, pass its
	// tools here on each run, and close it when done.
	ExtraTools []tool.Tool

	// OnTurn, when set, is invoked once per turn, right after that turn's
	// assistant message (and its tool results, if any) are appended to the
	// conversation, in order, with Turn matching [Result.Turns]. It is how a host
	// observes the loop as it runs — for example to persist each turn to an
	// external store.
	//
	// It runs in workflow code and must be deterministic; schedule an activity for
	// any side effect (persisting a turn is durable and replay-safe that way). A
	// non-nil error aborts the run and is returned with the partial [Result].
	//
	// A nil hook schedules no commands and adds no history events, so a run
	// without it replays byte-identically. Adding a hook to an already-deployed
	// workflow changes the command stream, so gate that with workflow.GetVersion.
	OnTurn func(ctx workflow.Context, e TurnEvent) error

	// ProviderMetadata is opaque key-values set on every model [model.Request] in
	// the run (see [model.Request.Metadata]). Use it to pass a routing provider
	// the context it needs — a tenant, a feature — without encoding it into the
	// model name.
	ProviderMetadata map[string]string
}

// TurnEvent describes one completed model call within a run, passed to
// [RunOptions.OnTurn].
type TurnEvent struct {
	// Turn is 1-based and matches [Result.Turns] at this point in the run.
	Turn int

	// Assistant is the model's message for this turn. It may carry tool calls.
	Assistant model.Message

	// ToolResults are the tool-result messages for this turn's calls, in call
	// order. It is empty on the final turn, where the assistant message concluded
	// the run without calling tools.
	ToolResults []model.Message

	// Usage is this turn's token usage (not the run's accumulated total).
	Usage model.Usage
}

// emitTurn invokes the OnTurn hook if one is set. A nil hook is a no-op, so a run
// without it schedules no extra commands.
func (ro RunOptions) emitTurn(ctx workflow.Context, turn int, assistant model.Message, toolResults []model.Message, usage model.Usage) error {
	if ro.OnTurn == nil {
		return nil
	}
	return ro.OnTurn(ctx, TurnEvent{Turn: turn, Assistant: assistant, ToolResults: toolResults, Usage: usage})
}

// Run executes the agent loop with default options, creating a [Session]. It is
// the common case of one run per workflow; use [RunWith] to pass [RunOptions], or
// [NewSession] directly when several runs share one approval surface.
func Run(ctx workflow.Context, a *Agent, input string) (*Result, error) {
	return RunWith(ctx, a, input, RunOptions{})
}

// RunWith executes the agent loop with the given options, creating a [Session].
//
// On error the returned [Result] is non-nil (except when session creation
// fails): it carries the transcript, usage, and turn count accumulated so far,
// with an empty Output. See [Session.RunWith].
func RunWith(ctx workflow.Context, a *Agent, input string, ro RunOptions) (*Result, error) {
	s, err := NewSession(ctx)
	if err != nil {
		return nil, err
	}
	return s.RunWith(ctx, a, input, ro)
}

// Run executes the agent loop with default options on this session.
func (s *Session) Run(ctx workflow.Context, a *Agent, input string) (*Result, error) {
	return s.RunWith(ctx, a, input, RunOptions{})
}

// RunWith executes the agent loop on this session.
//
// The returned [Result] is always non-nil. On error it carries the transcript
// (Result.Messages), usage, and turn count accumulated up to the failure, with
// an empty Output — so a host with an external conversation store can persist a
// failed run's turns. Use [RunOptions.OnTurn] to observe the loop as it runs.
func (s *Session) RunWith(ctx workflow.Context, a *Agent, input string, ro RunOptions) (*Result, error) {
	res := &Result{}
	if a == nil {
		return res, temporal.NewNonRetryableApplicationError("agent is nil", ErrorTypeConfig, nil)
	}

	tools, err := s.resolveTools(ctx, a, ro)
	if err != nil {
		return res, err
	}

	msgs := initialMessages(a, ro.History, input)
	res.Messages = msgs

	// Screen the input before the first model call.
	if err := s.runGuardrails(ctx, StageInput, a.inputGuardrails, input); err != nil {
		return res, err
	}

	ctx = workflow.WithActivityOptions(ctx, a.modelActivityOptions)
	maxTurns := a.maxTurns
	logger := workflow.GetLogger(ctx)

	// answer accumulates the assistant text across length-continuation fragments;
	// with no continuation it holds the single final answer.
	var answer strings.Builder
	continuations := 0

	for turn := 1; turn <= maxTurns; turn++ {
		resp, err := s.callModel(ctx, a, msgs, tools, ro)
		if err != nil {
			res.Messages = msgs
			return res, err
		}

		res.Turns = turn
		res.FinishReason = resp.FinishReason
		res.Usage.PromptTokens += resp.Usage.PromptTokens
		res.Usage.CompletionTokens += resp.Usage.CompletionTokens
		res.Usage.TotalTokens += resp.Usage.TotalTokens
		msgs = append(msgs, resp.Message)

		calls := resp.Message.ToolCalls()
		// No tool calls means the model is answering rather than calling tools.
		if len(calls) == 0 {
			answer.WriteString(resp.Message.Text())
			res.Messages = msgs

			// Continue a truncated answer while the budget allows. A contentless
			// message (no blocks) is not continued: there is nothing to extend, and
			// re-sending an empty assistant turn is rejected by some providers. That
			// can happen when the whole output budget went to hidden thinking.
			// Structured output is excluded: it is returned whole, not continued.
			canContinue := resp.FinishReason == model.FinishLength &&
				len(resp.Message.Blocks) > 0 &&
				ro.OutputSchema == nil
			if canContinue && continuations < a.maxContinuations {
				continuations++
				logger.Debug("agent continuing truncated answer",
					"agent", a.name, "turn", turn, "continuation", continuations)
				if err := ro.emitTurn(ctx, turn, resp.Message, nil, resp.Usage); err != nil {
					return res, err
				}
				continue
			}

			res.Output = answer.String()
			res.StructuredOutput = resp.StructuredOutput
			if err := ro.emitTurn(ctx, turn, resp.Message, nil, resp.Usage); err != nil {
				return res, err
			}
			// Screen the final answer. The transcript is already on res, so a
			// tripwire returns it alongside the error.
			if err := s.runGuardrails(ctx, StageOutput, a.outputGuardrails, res.Output); err != nil {
				return res, err
			}
			return res, nil
		}

		logger.Debug("agent dispatching tool calls",
			"agent", a.name, "turn", turn, "calls", len(calls))

		results, err := s.dispatch(ctx, a, tools, calls)
		if err != nil {
			res.Messages = msgs
			return res, err
		}
		msgs = append(msgs, results...)
		res.Messages = msgs
		// Only consecutive length-continuations concatenate; a tool round ends the
		// current answer, so drop any accumulated fragments.
		answer.Reset()

		if err := ro.emitTurn(ctx, turn, resp.Message, results, resp.Usage); err != nil {
			return res, err
		}
	}

	// The turn budget ran out. Preserve the transcript and any answer accumulated
	// so far so callers can inspect the run or continue it with a higher bound.
	res.Messages = msgs
	res.Output = answer.String()
	return res, temporal.NewNonRetryableApplicationError(
		fmt.Sprintf("agent %q exceeded %d turns without producing a final answer", a.name, maxTurns),
		ErrorTypeMaxTurns, nil)
}

// initialMessages assembles the starting conversation.
func initialMessages(a *Agent, history []model.Message, input string) []model.Message {
	msgs := make([]model.Message, 0, len(history)+2)

	// Prepend instructions only when the history does not already carry a system
	// message, so feeding Result.Messages back in does not stack them.
	if a.instructions != "" && !hasSystemMessage(history) {
		msgs = append(msgs, model.SystemMessage(a.instructions))
	}
	msgs = append(msgs, history...)
	if input != "" {
		msgs = append(msgs, model.UserMessage(input))
	}
	return msgs
}

func hasSystemMessage(msgs []model.Message) bool {
	for _, m := range msgs {
		if m.Role == model.RoleSystem {
			return true
		}
	}
	return false
}

// callModel schedules the model activity.
func (s *Session) callModel(ctx workflow.Context, a *Agent, msgs []model.Message, tools *toolSet, ro RunOptions) (*model.Response, error) {
	req := model.Request{
		Provider:     a.provider,
		Model:        a.model,
		Messages:     msgs,
		Tools:        tools.defs,
		OutputSchema: ro.OutputSchema,
		Stream:       a.stream,
		Settings:     a.settings,
		Metadata:     ro.ProviderMetadata,
	}
	var resp model.Response
	err := workflow.ExecuteActivity(ctx, model.InvokeModelActivity, req).Get(ctx, &resp)
	if err != nil {
		return nil, err
	}
	return &resp, nil
}

// dispatch invokes every tool the model asked for, concurrently.
//
// workflow.Go schedules coroutines deterministically, so concurrent dispatch is
// replay-safe. Results are written by index and assembled in call order, so
// history does not depend on which tool finished first.
func (s *Session) dispatch(ctx workflow.Context, a *Agent, tools *toolSet, calls []model.ToolCall) ([]model.Message, error) {
	results := make([]toolOutcome, len(calls))
	wg := workflow.NewWaitGroup(ctx)

	for i, call := range calls {
		wg.Add(1)
		workflow.Go(ctx, func(gctx workflow.Context) {
			defer wg.Done()
			results[i] = s.invoke(gctx, a, tools, call)
		})
	}
	wg.Wait(ctx)

	msgs := make([]model.Message, 0, len(results))
	for i, r := range results {
		if r.fatal != nil {
			return nil, r.fatal
		}
		msgs = append(msgs, model.ToolResultMessage(calls[i].ID, r.result))
	}
	return msgs, nil
}

// toolOutcome is one tool's result.
//
// result is the rich result the model sees, carried unflattened so the provider
// renders it. fatal is set only for failures the model cannot act on, which
// abort the run.
type toolOutcome struct {
	result *model.CallToolResult
	fatal  error
}

// invoke runs a single tool call, gating it on approval when required.
//
// Most failures are reported back to the model rather than raised, since a bad
// argument or a failing tool is information the model can act on. Only failures
// classified as fatal abort the run.
func (s *Session) invoke(ctx workflow.Context, a *Agent, tools *toolSet, call model.ToolCall) toolOutcome {
	t, ok := tools.byName[call.Name]
	if !ok {
		// Report an unknown tool back to the model, which usually corrects itself
		// on the next turn.
		return toolOutcome{result: model.ErrorResult(fmt.Sprintf(
			"Error: no tool named %q is available. Available tools: %v.", call.Name, tools.names))}
	}

	if t.Policy().NeedsApproval {
		decision, err := s.awaitApproval(ctx, a, t, call)
		if err != nil {
			return toolOutcome{fatal: err}
		}
		if !decision.Approved {
			return toolOutcome{result: model.TextResult(decision.modelMessage(call.Name))}
		}
	}

	logger := workflow.GetLogger(ctx)
	res, err := t.Invoke(ctx, call.Arguments)
	if err != nil {
		if fatal := classifyToolError(ctx, err); fatal != nil {
			return toolOutcome{fatal: fatal}
		}
		logger.Warn("tool call failed", "agent", a.name, "tool", call.Name, "error", err)
		return toolOutcome{result: model.ErrorResult(fmt.Sprintf("Error: %v", err))}
	}

	// A tool that ran and reported a business error is handed to the model, which
	// is the party that can respond to it, rather than failing the call.
	if res.IsError {
		logger.Debug("tool reported an error", "agent", a.name, "tool", call.Name)
	}

	// The rich result is carried unflattened; the provider flattens it at its edge.
	return toolOutcome{result: res}
}

// classifyToolError reports whether a failed tool call should abort the run. It
// returns non-nil to abort and nil to report the error back to the model.
//
// Business failures (no such user, payment declined) are reported to the model.
// Two classes abort instead: a canceled workflow, which must unwind; and the
// SDK's own configuration errors (an unregistered sub-agent, a malformed agent),
// which are permanent and cannot be fixed by the model rephrasing.
func classifyToolError(ctx workflow.Context, err error) error {
	if errors.Is(ctx.Err(), workflow.ErrCanceled) || temporal.IsCanceledError(err) {
		return err
	}

	var appErr *temporal.ApplicationError
	if errors.As(err, &appErr) && isConfigErrorType(appErr.Type()) {
		return err
	}
	return nil
}

// isConfigErrorType reports whether an application error type is one the model
// cannot recover from — SDK misconfiguration or an unrecoverable session loss —
// and so should abort the run rather than be narrated back to the model.
func isConfigErrorType(t string) bool {
	switch t {
	case ErrorTypeConfig, ErrorTypeUnknownAgent, ErrorTypeApproval,
		model.ErrorTypeNoProvider, mcp.ErrorTypeUnknownServer, mcp.ErrorTypeSessionLost:
		return true
	default:
		return false
	}
}

// toolSet is the resolved, indexed tools for one run.
type toolSet struct {
	defs   []*model.Tool
	byName map[string]tool.Tool
	names  []string // in declaration order; map iteration is nondeterministic
}

// resolveTools merges the agent's static tools, any stateless MCP server tools,
// and the run's ExtraTools (e.g. a stateful MCP session's tools).
func (s *Session) resolveTools(ctx workflow.Context, a *Agent, ro RunOptions) (*toolSet, error) {
	for i, t := range ro.ExtraTools {
		if t == nil {
			return nil, temporal.NewNonRetryableApplicationError(
				fmt.Sprintf("agent %q: extra tool %d is nil", a.name, i), ErrorTypeConfig, nil)
		}
	}
	tools := make([]tool.Tool, 0, len(a.tools)+len(ro.ExtraTools))
	tools = append(tools, a.tools...)
	tools = append(tools, ro.ExtraTools...)

	for _, server := range a.mcpServers {
		mcpTools, err := s.mcpTools(ctx, server)
		if err != nil {
			return nil, err
		}
		tools = append(tools, mcpTools...)
	}

	ts := &toolSet{
		defs:   make([]*model.Tool, 0, len(tools)),
		byName: make(map[string]tool.Tool, len(tools)),
		names:  make([]string, 0, len(tools)),
	}
	for _, t := range tools {
		def := t.Def()
		if def == nil || def.Name == "" {
			return nil, temporal.NewNonRetryableApplicationError(
				fmt.Sprintf("agent %q: a tool has no name", a.name), ErrorTypeConfig, nil)
		}
		if _, dup := ts.byName[def.Name]; dup {
			return nil, temporal.NewNonRetryableApplicationError(
				fmt.Sprintf("agent %q: duplicate tool name %q", a.name, def.Name),
				ErrorTypeConfig, nil)
		}
		ts.byName[def.Name] = t
		ts.defs = append(ts.defs, def)
		ts.names = append(ts.names, def.Name)
	}
	return ts, nil
}
