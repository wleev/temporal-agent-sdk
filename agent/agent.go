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
	"maps"
	"slices"
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
	// before treating the worker running a model call as gone. The model
	// activity heartbeats for as long as the call is in flight.
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
	guardrails           []guardrail.Guardrail
}

// Option configures an [Agent] built by [NewAgent].
type Option func(*Agent)

// NewAgent builds an agent. name and model are required; every other field is an
// [Option] with a sensible default. It returns a non-retryable [ErrorTypeConfig]
// error for an empty name or model, a nil tool, a guardrail [WithGuardrails]
// refuses, or a tool choice in the settings with an invalid mode or with mode
// [model.ToolChoiceRequired] or [model.ToolChoiceTool].
func NewAgent(name, modelID string, opts ...Option) (*Agent, error) {
	if name == "" {
		return nil, configError("agent: name must not be empty")
	}
	if modelID == "" {
		return nil, configError("agent %q: model must not be empty", name)
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
			return nil, configError("agent %q: tool %d is nil", name, i)
		}
	}
	if err := checkGuardrails(a, "guardrail", a.guardrails); err != nil {
		return nil, err
	}
	if err := checkAgentSettings(name, a.settings); err != nil {
		return nil, err
	}
	return a, nil
}

// checkAgentSettings returns a non-retryable [ErrorTypeConfig] error when s
// holds a tool choice with an invalid mode or with mode
// [model.ToolChoiceRequired] or [model.ToolChoiceTool].
func checkAgentSettings(name string, s model.Settings) error {
	c := s.ToolChoice
	switch {
	case c == nil:
		return nil
	case !c.Mode.Valid():
		return configError("agent %q: unknown tool choice mode %q", name, c.Mode)
	case c.Mode == model.ToolChoiceRequired || c.Mode == model.ToolChoiceTool:
		return configError("agent %q: tool choice mode %q applies per call only; use ModelCallPlan.Settings", name, c.Mode)
	default:
		return nil
	}
}

// configError returns a non-retryable [ErrorTypeConfig] application error
// whose message is formatted from format and args.
func configError(format string, args ...any) error {
	return temporal.NewNonRetryableApplicationError(fmt.Sprintf(format, args...), ErrorTypeConfig, nil)
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

// WithSettings sets a copy of the generation parameters. The zero value defers
// to the provider. [NewAgent] rejects a tool choice with an invalid mode or
// with mode [model.ToolChoiceRequired] or [model.ToolChoiceTool];
// [ModelCallPlan.Settings] carries those per call.
func WithSettings(settings model.Settings) Option {
	return func(a *Agent) { a.settings = settings.Clone() }
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
// non-positive value disables continuation, which is the default. [NewAgent]
// refuses continuation together with output-stage guardrails, and a run on such
// an agent refuses a run guardrail that declares the output stage.
func WithContinueOnLength(maxContinuations int) Option {
	return func(a *Agent) { a.maxContinuations = maxContinuations }
}

// WithModelActivityOptions configures the model activity. A zero
// StartToCloseTimeout becomes [DefaultModelTimeout], a nil RetryPolicy becomes
// one bounded at [DefaultModelMaxAttempts], and a zero HeartbeatTimeout becomes
// [DefaultModelHeartbeatTimeout], capped at the StartToCloseTimeout.
//
// The model activity heartbeats for as long as the provider call is in flight
// and cancels a call that exceeds a limit set by [model.WithStreamIdle],
// [model.WithFirstDelta], or [model.WithUnstreamedLimit].
func WithModelActivityOptions(opts workflow.ActivityOptions) Option {
	return func(a *Agent) { a.modelActivityOptions = opts }
}

// WithApprovalTimeout bounds how long an approval-gated tool waits for a human. A
// non-positive value keeps [DefaultApprovalTimeout]. On expiry the tool reports a
// denial to the model rather than failing the workflow.
func WithApprovalTimeout(d time.Duration) Option {
	return func(a *Agent) { a.approvalTimeout = d }
}

// WithStreaming streams model calls and forwards their deltas to the sink
// configured on the worker's model activities (see model.WithStreamSink). A
// streamed call is subject to the limits of model.WithStreamIdle and
// model.WithFirstDelta. The workflow result is identical either way.
func WithStreaming() Option {
	return func(a *Agent) { a.stream = true }
}

// WithGuardrails registers guardrails on the agent, each on every stage its
// Stages declares. It may be called more than once; the guardrails of a stage
// run concurrently and are judged in registration order. [NewAgent] refuses a
// nil guardrail; a guardrail with an empty name, no stages or a stage that is
// not one of the five; a guardrail that declares [guardrail.StageToolResult];
// and a guardrail that declares [guardrail.StageOutput] together with
// [WithContinueOnLength]. A block surfaces as a [TripwireError], except at the
// tool_call stage: a blocked call is not run, its result is an error result
// naming the tool, and the run continues. [RunOptions.OnGuardrail] can end the
// run on such a block by returning an error.
func WithGuardrails(g ...guardrail.Guardrail) Option {
	return func(a *Agent) { a.guardrails = append(a.guardrails, g...) }
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

	// StructuredOutput is the terminal message as raw JSON, set only when the
	// final model call carried an output schema. [RunTyped] decodes it.
	StructuredOutput json.RawMessage `json:"structured_output,omitempty"`
}

// RunOptions modifies a single run.
type RunOptions struct {
	// History seeds the conversation with prior messages. If it already contains
	// a system message, the agent's Instructions are not prepended again, which
	// makes a Result.Messages round-trip safe.
	History []model.Message

	// OutputSchema, when set, constrains the model's final answer to JSON
	// matching the schema. A call's [ModelCallPlan.OutputSchema] replaces it
	// for that call. Usually set for you by [RunTyped]; set it directly only if
	// you are decoding the raw [Result.StructuredOutput] yourself.
	OutputSchema *model.OutputSchema

	// ExtraTools are added to the agent's own tools for this run only, without
	// mutating the [Agent]. It is how tools resolved at runtime — most notably a
	// stateful MCP session's tools (mcp.OpenStatefulSession), which must outlive
	// a single run — are injected: open the session in the workflow, pass its
	// tools here on each run, and close it when done.
	ExtraTools []tool.Tool

	// Guardrails adds guardrails for this run only, each on every stage it
	// declares, after the agent's guardrails for that stage. A guardrail
	// [WithGuardrails] refuses fails the run before the first model call with a
	// non-retryable [ErrorTypeConfig] error.
	Guardrails []guardrail.Guardrail

	// OnReply, when set, is invoked once per model reply, after the reply's
	// guardrail stages and after the reply is appended to the conversation, and
	// before its tool calls are dispatched. It fires for every reply whose
	// output or assistant_text stage did not block, including a reply with a
	// blocked tool call, the final answer and each length-continuation
	// fragment, and before that turn's OnTurn. Its [ReplyEvent] carries a copy
	// of the reply after the guardrail verdicts.
	//
	// It runs in workflow code and must be deterministic. A non-nil error aborts
	// the run and is returned with the partial [Result], whose Messages end with
	// the reply and, for a reply that ends the run with model notes, one user
	// message holding the model notes of its output stage and of the
	// length-continuation fragments before it. A nil hook schedules no
	// commands. Adding a hook to an already-deployed workflow changes the
	// command stream, so gate that with workflow.GetVersion. Its ctx carries the
	// agent's model activity options.
	OnReply func(ctx workflow.Context, e ReplyEvent) error

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

	// PrepareModelCall, when set, is invoked before every model call, including
	// length-continuation calls. It receives a [ModelCallPlan] holding the turn,
	// the run's current tools, a nil Advertised, a copy of the agent's settings
	// and no OutputSchema, Append, CallOnly or Metadata, and returns the plan
	// for the call. Non-empty returned Tools replace the run's dispatchable
	// tools from that call on; nil or empty Tools keep them. Advertised,
	// Settings, OutputSchema, CallOnly and Metadata apply to that call only; nil
	// or empty Advertised sends every tool in Tools. Append adds messages to the
	// conversation, which stay in [Result.Messages] and are not reported
	// through OnReply or OnTurn.
	//
	// It runs in workflow code and must be deterministic. A non-nil error, a
	// returned Turn other than the one passed in, an Advertised name not in
	// Tools, a tool choice with an invalid mode, a required or tool mode on a
	// call that advertises no tools, a tool mode with an empty name or naming a
	// tool the call does not advertise, or an invalid tool (nil, unnamed or a duplicate name) aborts
	// the run and is returned with the partial [Result], whose Messages are
	// those before the hook ran. A nil hook schedules no commands. Adding a hook
	// to an already-deployed workflow changes the command stream, so gate that
	// with workflow.GetVersion. Its ctx carries the agent's model activity
	// options.
	PrepareModelCall func(ctx workflow.Context, plan ModelCallPlan) (ModelCallPlan, error)

	// OnFinal, when set, is invoked after OnTurn for a reply with no tool calls
	// that would end the run, and not for a length-continuation fragment or a
	// tool's final output ([model.FinalResult]). It receives a [FinalEvent] for
	// that reply and returns a [FinalAction]. The zero FinalAction ends the run.
	// With Continue, the loop continues with the next turn, which counts
	// against [WithMaxTurns]; with no turns left the
	// run fails with [ErrorTypeMaxTurns]. Before Append is appended, the reply is
	// removed from the conversation, together with its output-stage model
	// notes, when DropReply is set or the reply has empty or whitespace-only
	// text and no structured output. After a length-continuation chain only the
	// reply, the last fragment, is removed; the earlier fragments stay, and
	// their notes follow the next reply. [Result.Output] and
	// [Result.StructuredOutput] are reset. A FinalAction that sets DropReply or
	// Append without Continue aborts the run with a non-retryable
	// [ErrorTypeConfig] error. The output-stage guardrails screen the reply
	// before OnReply, so OnFinal receives the reply after their verdicts, and
	// each candidate answer after OnFinal continues the run is screened again.
	//
	// It runs in workflow code and must be deterministic. A non-nil error aborts
	// the run and is returned with the partial [Result]. A nil hook schedules no
	// commands. Adding a hook to an already-deployed workflow changes the command
	// stream, so gate that with workflow.GetVersion. Its ctx carries the agent's
	// model activity options.
	OnFinal func(ctx workflow.Context, e FinalEvent) (FinalAction, error)

	// OnRound, when set, is invoked once per round of tool calls, after every
	// call of the round has finished and only when none failed fatally (a
	// tool's tripwire is fatal), before the results are appended to the
	// conversation, before the round's held model notes and before OnTurn. It
	// receives a [RoundEvent] with the reply and its calls after their
	// guardrail verdicts and copies of the round's results in call order: what
	// each tool returned, or the blocked-call error result for a call the
	// tool_call stage blocked. The returned [RoundOutcome.Results] replace those results. A
	// replacement is the whole result the model sees and OnTurn reports; it is
	// not screened again, and a model note a host's [guardrail.Screen] added to
	// the original result is not kept. The kept results are the ones checked
	// for a final output ([model.FinalResult]) of a tool built with
	// [tool.EndsRun]. A Results whose length is not the number of calls aborts
	// the run with a non-retryable [ErrorTypeConfig] error.
	//
	// It runs in workflow code and must be deterministic. A non-nil error aborts
	// the run and is returned with the partial [Result]. A nil hook schedules no
	// commands. Adding a hook to an already-deployed workflow changes the command
	// stream, so gate that with workflow.GetVersion. Its ctx carries the agent's
	// model activity options.
	OnRound func(ctx workflow.Context, e RoundEvent) (RoundOutcome, error)

	// OnGuardrail, when set, is invoked once per acting guardrail verdict — a
	// block, rewrite, annotation or failed check; an allow is not reported — in
	// declared order, before the run acts on the verdicts of that content. A
	// stage that blocks its content reports only that content's block and
	// failed verdicts; the verdicts already reported for earlier content of the
	// same reply stay reported. A block of a reply's text stage leaves the
	// reply's tool calls unscreened; a tool_call block leaves the reply's later
	// calls to be screened. For a reply, the text stage's verdicts come first,
	// then each tool call's in index order. It receives a [GuardrailEvent] with the verdict's
	// [guardrail.Record].
	//
	// It runs in workflow code and must be deterministic. A non-nil error aborts
	// the run and is returned with the partial [Result]. A nil hook schedules no
	// commands. Adding a hook to an already-deployed workflow changes the command
	// stream, so gate that with workflow.GetVersion. Its ctx carries the agent's
	// model activity options.
	OnGuardrail func(ctx workflow.Context, e GuardrailEvent) error

	// OnUnknownTool, when set, is invoked for a tool call whose name is not among
	// the run's dispatchable tools, with the call's context (see
	// [tool.CallInfoFrom]). A non-nil result is the call's result; a nil result
	// keeps the built-in "no tool named" error result. An error aborts the run.
	// The calls of one reply run concurrently in workflow coroutines, so it may
	// be invoked for several calls of a reply before any returns.
	//
	// It runs in workflow code and must be deterministic. A nil hook schedules no
	// commands. Adding a hook to an already-deployed workflow changes the command
	// stream, so gate that with workflow.GetVersion. Its ctx carries the agent's
	// model activity options.
	OnUnknownTool func(ctx workflow.Context, call model.ToolCall) (*model.CallToolResult, error)

	// ProviderMetadata is opaque key-values set on every model [model.Request] in
	// the run (see [model.Request.Metadata]). A call's [ModelCallPlan.Metadata]
	// is overlaid on it for that call only.
	ProviderMetadata map[string]string

	// typed marks a run started by [RunTypedWith]: its final reply must come
	// from a call whose output schema matches OutputSchema.
	typed bool
}

// ReplyEvent describes one model reply, passed to [RunOptions.OnReply].
type ReplyEvent struct {
	// Turn is 1-based and matches [TurnEvent.Turn] for the same model call.
	Turn int

	// Assistant is the model's reply after its guardrail verdicts. It may carry
	// tool calls.
	Assistant model.Message

	// Usage is this reply's token usage.
	Usage model.Usage

	// Attempt is the reply's [model.Response.Attempt]: the model activity
	// attempt, starting at 1, or 0 when unrecorded.
	Attempt int32

	// FinishReason is the reply's [model.Response.FinishReason].
	FinishReason model.FinishReason

	// ResponseID is the reply's [model.Response.ResponseID].
	ResponseID string

	// Model is the reply's [model.Response.Model].
	Model string

	// Metadata is a copy of the reply's [model.Response.Metadata].
	Metadata map[string]string

	// Timing is the reply's [model.Response.Timing].
	Timing *model.Timing
}

// TurnEvent describes one completed model call within a run, passed to
// [RunOptions.OnTurn].
type TurnEvent struct {
	// Turn is 1-based and matches [Result.Turns] at this point in the run.
	Turn int

	// Assistant is the model's message for this turn after its guardrail
	// verdicts. It may carry tool calls.
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

// emitReply invokes the OnReply hook for resp when one is set.
func (ro RunOptions) emitReply(ctx workflow.Context, turn int, resp *model.Response) error {
	if ro.OnReply == nil {
		return nil
	}
	return ro.OnReply(ctx, ReplyEvent{
		Turn:         turn,
		Assistant:    copyMessage(resp.Message),
		Usage:        resp.Usage,
		Attempt:      resp.Attempt,
		FinishReason: resp.FinishReason,
		ResponseID:   resp.ResponseID,
		Model:        resp.Model,
		Metadata:     maps.Clone(resp.Metadata),
		Timing:       resp.Timing,
	})
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
// fails): it carries the transcript, usage, turn count and Output accumulated
// up to the failure. See [Session.RunWith].
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
// (Result.Messages), usage, turn count and Output accumulated up to the
// failure — so a host with an external conversation store can persist a failed
// run's turns. Use [RunOptions.OnTurn] to observe the loop as it runs.
//
// A non-empty input is screened by the input-stage guardrails before the first
// model call; an empty input skips the stage. When the input stage blocks or
// fails, Result.Messages holds the messages before the input.
func (s *Session) RunWith(ctx workflow.Context, a *Agent, input string, ro RunOptions) (*Result, error) {
	res := &Result{}
	if a == nil {
		return res, configError("agent is nil")
	}
	if err := checkGuardrails(a, "run guardrail", ro.Guardrails); err != nil {
		return res, err
	}

	tools, err := s.resolveTools(ctx, a, ro)
	if err != nil {
		return res, err
	}

	ctx = workflow.WithActivityOptions(ctx, a.modelActivityOptions)
	r := &runState{s: s, a: a, ro: ro, tools: tools, res: res, guards: slices.Concat(a.guardrails, ro.Guardrails)}

	msgs := initialMessages(a, ro.History, input)
	if input == "" {
		workflow.GetLogger(ctx).Debug("input stage skipped", "agent", a.name)
	} else {
		screened, err := r.screenInput(ctx, msgs)
		if err != nil {
			res.Messages = msgs[:len(msgs)-1]
			return res, err
		}
		msgs = screened
	}
	res.Messages = msgs

	for turn := 1; turn <= a.maxTurns; turn++ {
		done, err := r.runTurn(ctx, turn)
		if done || err != nil {
			return res, err
		}
	}

	// The turn budget ran out; the result keeps the transcript, the held model
	// notes and the accumulated answer.
	res.Messages = appendNotes(res.Messages, r.heldNotes)
	res.Output = r.answer.String()
	return res, temporal.NewNonRetryableApplicationError(
		fmt.Sprintf("agent %q exceeded %d turns without producing a final answer", a.name, a.maxTurns),
		ErrorTypeMaxTurns, nil)
}

// runState is the state of one run of the agent loop. res.Messages is the
// run's conversation.
type runState struct {
	s     *Session
	a     *Agent
	ro    RunOptions
	tools *toolSet
	res   *Result

	// guards are the agent's guardrails followed by the run's.
	guards []guardrail.Guardrail

	// answer accumulates the assistant text across length-continuation
	// fragments; with no continuation it holds the single final answer.
	answer strings.Builder

	// continuations is the number of length-continuation calls made.
	continuations int

	// heldNotes are the model notes of assistant_text and tool_call verdicts
	// not yet appended to the conversation.
	heldNotes []string
}

// runTurn makes the model call of turn and handles its reply. It reports
// whether the run ended.
func (r *runState) runTurn(ctx workflow.Context, turn int) (bool, error) {
	pc, err := r.ro.prepareCall(ctx, r.a, turn, r.tools, r.res.Messages)
	if err != nil {
		return false, err
	}
	r.tools = pc.tools
	r.res.Messages = pc.msgs

	resp, err := r.s.callModel(ctx, r.a, pc)
	if err != nil {
		return false, err
	}
	r.res.Turns = turn
	r.res.FinishReason = resp.FinishReason
	r.res.Usage = r.res.Usage.Add(resp.Usage)

	// A length-truncated reply with content and no output schema is a
	// fragment, continued while the continuation budget allows.
	canContinue := resp.FinishReason == model.FinishLength &&
		len(resp.Message.Blocks) > 0 &&
		pc.schema == nil
	fragment := !resp.Message.HasToolCalls() && canContinue && r.continuations < r.a.maxContinuations
	final := !resp.Message.HasToolCalls() && !fragment

	sr, err := r.screenReply(ctx, pc, turn, *resp, final)
	if err != nil {
		return false, err
	}
	resp, notes := &sr.resp, sr.notes
	replyAt := len(r.res.Messages)
	r.res.Messages = append(r.res.Messages, resp.Message)
	// heldBefore are the notes held before this reply, held again when the
	// reply is dropped.
	heldBefore := r.heldNotes
	if final {
		r.res.Messages = appendNotes(r.res.Messages, append(slices.Clip(r.heldNotes), notes...))
		r.heldNotes = nil
	} else {
		r.heldNotes = append(r.heldNotes, notes...)
	}
	if err := r.ro.emitReply(ctx, turn, resp); err != nil {
		return false, err
	}

	if len(sr.calls) > 0 {
		return r.finishRound(ctx, turn, &sr)
	}
	r.answer.WriteString(resp.Message.Text())
	if fragment {
		r.continuations++
		workflow.GetLogger(ctx).Debug("agent continuing truncated answer",
			"agent", r.a.name, "turn", turn, "continuation", r.continuations)
		return false, r.ro.emitTurn(ctx, turn, resp.Message, nil, resp.Usage)
	}
	return r.finishReply(ctx, turn, pc.schema, resp, replyAt, heldBefore)
}

// finishReply handles resp, the reply of turn that would end the run, from a
// call with output schema schema. It runs OnTurn and OnFinal and reports
// whether the run ended. When OnFinal continues the run and the reply is
// dropped, the conversation is cut back to replyAt, the reply's index, and
// heldBefore, the notes held before the reply, are held again.
func (r *runState) finishReply(ctx workflow.Context, turn int, schema *model.OutputSchema, resp *model.Response, replyAt int, heldBefore []string) (bool, error) {
	r.res.Output = r.answer.String()
	r.res.StructuredOutput = resp.StructuredOutput
	if err := r.ro.emitTurn(ctx, turn, resp.Message, nil, resp.Usage); err != nil {
		return false, err
	}
	action, err := r.ro.emitFinal(ctx, r.a.name, turn, resp)
	if err != nil {
		return false, err
	}
	if action.Continue {
		// A dropped reply takes its model notes with it.
		if action.DropReply || (!resp.Message.HasText() && len(resp.StructuredOutput) == 0) {
			r.res.Messages = r.res.Messages[:replyAt]
			r.heldNotes = heldBefore
		}
		r.res.Messages = append(r.res.Messages, action.Append...)
		r.answer.Reset()
		r.res.Output = ""
		r.res.StructuredOutput = nil
		return false, nil
	}
	if r.ro.typed && !sameOutputSchema(schema, r.ro.OutputSchema) {
		return false, configError("agent %q: the final reply used output schema %q, not the typed run's schema %q",
			r.a.name, schema.Name, r.ro.OutputSchema.Name)
	}
	return true, nil
}

// finishRound dispatches the tool calls of sr, the reply of turn; runs
// OnRound; appends the kept results and the held model notes; and runs OnTurn.
// It reports whether the run ended with a tool's final output.
func (r *runState) finishRound(ctx workflow.Context, turn int, sr *screenedReply) (bool, error) {
	resp, calls := &sr.resp, sr.calls
	workflow.GetLogger(ctx).Debug("agent dispatching tool calls",
		"agent", r.a.name, "turn", turn, "calls", len(calls))

	returned, err := r.dispatch(ctx, turn, calls, sr.blocked)
	if err != nil {
		return false, err
	}
	kept, err := r.ro.emitRound(ctx, r.a.name, turn, resp.Message, calls, returned)
	if err != nil {
		return false, err
	}
	finalAt, output, ends := r.finalOutput(ctx, calls, kept)
	results := toolResultMessages(calls, withoutFinalOutputs(kept))
	r.res.Messages = append(r.res.Messages, results...)
	r.res.Messages = appendNotes(r.res.Messages, r.heldNotes)
	r.heldNotes = nil
	// A tool round ends the current answer; accumulated fragments are
	// discarded.
	r.answer.Reset()

	if err := r.ro.emitTurn(ctx, turn, resp.Message, results, resp.Usage); err != nil {
		return false, err
	}
	if !ends {
		return false, nil
	}
	if r.ro.typed {
		return false, configError("agent %q: tool %q ended the typed run with host text; a typed run needs structured output",
			r.a.name, calls[finalAt].Name)
	}
	return true, r.finishWithOutput(ctx, turn, output)
}

// finishWithOutput ends the run with output, the final output of a tool result
// of turn. It screens output with the output stage, with the conversation and
// attempt 0; a block returns the [NewTripwireError] of its record with nothing
// appended.
// Otherwise it appends the output after its verdicts as an assistant message,
// unless it is empty, followed by one user message holding the output-stage
// model notes, if any, and sets the run's output to it, its finish reason to
// [model.FinishStop] and its structured output to nil.
func (r *runState) finishWithOutput(ctx workflow.Context, turn int, output string) error {
	out, err := r.screen(ctx, guardrail.CheckInput{
		Stage:        guardrail.StageOutput,
		Text:         output,
		Conversation: slices.Clip(r.res.Messages),
		Turn:         turn,
	}, 0)
	if err != nil {
		return err
	}
	if out.Rewritten {
		output = out.Text
	}
	if output != "" {
		r.res.Messages = append(r.res.Messages, model.AssistantMessage(output))
	}
	r.res.Messages = appendNotes(r.res.Messages, out.ModelNotes)
	r.res.Output = output
	r.res.FinishReason = model.FinishStop
	r.res.StructuredOutput = nil
	return nil
}

// screenInput runs the input stage on the last message of msgs, the user
// input, with the messages before it as the conversation. It returns msgs with
// the input replaced in place by its rewrite, if any, followed by one user
// message holding the model notes, if any.
func (r *runState) screenInput(ctx workflow.Context, msgs []model.Message) ([]model.Message, error) {
	last := len(msgs) - 1
	out, err := r.screen(ctx, guardrail.CheckInput{
		Stage:        guardrail.StageInput,
		Text:         msgs[last].Text(),
		Conversation: slices.Clip(msgs[:last]),
	}, 0)
	if err != nil {
		return nil, err
	}
	if out.Rewritten {
		msgs[last] = model.UserMessage(out.Text)
	}
	return appendNotes(msgs, out.ModelNotes), nil
}

// screenedReply is a reply after its guardrail stages.
type screenedReply struct {
	// resp is the reply with its message and structured output after the
	// verdicts.
	resp model.Response

	// notes are the model notes of the verdicts.
	notes []string

	// calls are the tool calls of resp's message after their verdicts, in
	// index order. They are nil for a final reply.
	calls []model.ToolCall

	// blocked reports, for each entry of calls, whether the tool_call stage
	// blocked it.
	blocked []bool
}

// screenReply runs the reply stages on resp, the reply of turn to the call p,
// with p's messages as the conversation. A block of the output or
// assistant_text stage is returned as the [NewTripwireError] of its record. A
// block of the tool_call stage marks that call blocked, keeps its arguments
// and continues with the next call.
//
// A final reply, one that would end the run, takes the output stage: on its
// structured output, with Structured set, when p has an output schema, and on
// its text otherwise. A call with an output schema whose reply has no
// structured output is screened on the reply's text with Structured set, and
// a rewrite sets the structured output. Any other reply takes the
// assistant_text stage on its text, when it has text, then the tool_call stage
// on each call in index order.
func (r *runState) screenReply(ctx workflow.Context, p preparedCall, turn int, resp model.Response, final bool) (screenedReply, error) {
	conv := slices.Clip(p.msgs)
	structured := final && p.schema != nil
	post := resp.Message
	post.Blocks = slices.Clone(resp.Message.Blocks)
	var notes []string

	if final || post.HasText() {
		in := guardrail.CheckInput{Stage: guardrail.StageAssistantText, Text: post.Text(), Conversation: conv, Turn: turn}
		if final {
			in.Stage = guardrail.StageOutput
			in.Structured = structured
			if structured && len(resp.StructuredOutput) > 0 {
				in.Text = string(resp.StructuredOutput)
			}
		}
		out, err := r.screen(ctx, in, resp.Attempt)
		if err != nil {
			return screenedReply{}, err
		}
		if out.Rewritten {
			post.Blocks = replaceText(post.Blocks, out.Text)
			if structured {
				resp.StructuredOutput = json.RawMessage(out.Text)
			}
		}
		notes = append(notes, out.ModelNotes...)
	}

	var calls []model.ToolCall
	var blocked []bool
	if !final {
		index := 0
		for i := range post.Blocks {
			b := &post.Blocks[i]
			if b.Kind != model.BlockToolCall {
				continue
			}
			call := model.ToolCall{ID: b.ToolCallID, Name: b.ToolName, Arguments: b.Arguments}
			out, err := r.screenOutcome(ctx, guardrail.CheckInput{
				Stage: guardrail.StageToolCall, Call: &call, Conversation: conv, Turn: turn, Index: index,
			}, resp.Attempt)
			if err != nil {
				return screenedReply{}, err
			}
			if out.Rewritten {
				b.Arguments = out.Arguments
			}
			calls = append(calls, model.ToolCall{ID: b.ToolCallID, Name: b.ToolName, Arguments: b.Arguments})
			blocked = append(blocked, out.Blocked != nil)
			notes = append(notes, out.ModelNotes...)
			index++
		}
	}

	resp.Message = post
	return screenedReply{resp: resp, notes: notes, calls: calls, blocked: blocked}, nil
}

// replaceText returns blocks with its text blocks replaced by one text block
// holding text, at the position of the first text block, or appended when
// blocks has no text block.
func replaceText(blocks []model.Block, text string) []model.Block {
	out := make([]model.Block, 0, len(blocks)+1)
	placed := false
	for _, b := range blocks {
		if b.Kind != model.BlockText {
			out = append(out, b)
			continue
		}
		if !placed {
			out = append(out, model.TextBlock(text))
			placed = true
		}
	}
	if !placed {
		out = append(out, model.TextBlock(text))
	}
	return out
}

// noteMessage returns a user message holding notes separated by blank lines.
func noteMessage(notes []string) model.Message {
	return model.UserMessage(strings.Join(notes, "\n\n"))
}

// appendNotes returns msgs followed by the [noteMessage] of notes, or msgs
// unchanged when notes is empty.
func appendNotes(msgs []model.Message, notes []string) []model.Message {
	if len(notes) == 0 {
		return msgs
	}
	return append(msgs, noteMessage(notes))
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

// callModel executes the model activity with p's tool definitions, settings,
// output schema and metadata. The request's messages are p.msgs followed by
// p.callOnly.
func (s *Session) callModel(ctx workflow.Context, a *Agent, p preparedCall) (*model.Response, error) {
	msgs := p.msgs
	if len(p.callOnly) > 0 {
		msgs = slices.Concat(p.msgs, p.callOnly)
	}
	req := model.Request{
		Provider:     a.provider,
		Model:        a.model,
		Messages:     msgs,
		Tools:        p.defs,
		OutputSchema: p.schema,
		Stream:       a.stream,
		Settings:     p.settings,
		Metadata:     p.metadata,
	}
	var resp model.Response
	err := workflow.ExecuteActivity(ctx, model.InvokeModelActivity, req).Get(ctx, &resp)
	if err != nil {
		return nil, err
	}
	return &resp, nil
}

// callMetadata returns run when call is empty, otherwise a new map holding run
// overlaid by call.
//
//workflowcheck:ignore copying map entries into a map does not depend on iteration order
func callMetadata(run, call map[string]string) map[string]string {
	if len(call) == 0 {
		return run
	}
	out := make(map[string]string, len(run)+len(call))
	maps.Copy(out, run)
	maps.Copy(out, call)
	return out
}

// dispatch invokes every tool the model asked for, concurrently, and returns
// their results in call order. blocked holds one entry per call; a call whose
// entry is true is not invoked; its result is the blockedCallResult of its tool. Each call's
// context carries its [tool.CallInfo], with Turn set to turn. It returns the
// first fatal error in call order once every call has finished. The results
// are in call order regardless of which call finishes first.
func (r *runState) dispatch(ctx workflow.Context, turn int, calls []model.ToolCall, blocked []bool) ([]*model.CallToolResult, error) {
	results := make([]toolOutcome, len(calls))
	wg := workflow.NewWaitGroup(ctx)

	for i, call := range calls {
		if blocked[i] {
			results[i] = toolOutcome{result: blockedCallResult(call.Name)}
			continue
		}
		wg.Add(1)
		workflow.Go(ctx, func(gctx workflow.Context) {
			defer wg.Done()
			cctx := tool.WithCallInfo(gctx, tool.CallInfo{ID: call.ID, Turn: turn, Index: i})
			results[i] = r.invoke(cctx, call)
		})
	}
	wg.Wait(ctx)

	out := make([]*model.CallToolResult, 0, len(results))
	for _, res := range results {
		if res.fatal != nil {
			return nil, res.fatal
		}
		out = append(out, res.result)
	}
	return out, nil
}

// finalOutput returns the index and final output of the first result in call
// order that carries one ([model.FinalOutput]) for a call to a tool whose
// policy sets [tool.Policy.CanEndRun], and true, or 0, "" and false when none
// does. results holds one result per call of calls.
func (r *runState) finalOutput(ctx workflow.Context, calls []model.ToolCall, results []*model.CallToolResult) (idx int, output string, ok bool) {
	for i, res := range results {
		final, found := model.FinalOutput(res)
		if !found {
			continue
		}
		if t, known := r.tools.byName[calls[i].Name]; known && t.Policy().CanEndRun {
			return i, final, true
		}
		workflow.GetLogger(ctx).Warn("tool result carries a final output from a tool without tool.EndsRun; the output is ignored",
			"agent", r.a.name, "tool", calls[i].Name)
	}
	return 0, "", false
}

// withoutFinalOutputs returns results with each result passed through
// [model.WithoutFinalOutput].
func withoutFinalOutputs(results []*model.CallToolResult) []*model.CallToolResult {
	out := make([]*model.CallToolResult, len(results))
	for i, r := range results {
		out[i] = model.WithoutFinalOutput(r)
	}
	return out
}

// toolResultMessages returns one tool-result message per call, pairing each
// call's ID with the result at its index.
func toolResultMessages(calls []model.ToolCall, results []*model.CallToolResult) []model.Message {
	msgs := make([]model.Message, 0, len(calls))
	for i, call := range calls {
		msgs = append(msgs, model.ToolResultMessage(call.ID, results[i]))
	}
	return msgs
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
func (r *runState) invoke(ctx workflow.Context, call model.ToolCall) toolOutcome {
	t, ok := r.tools.byName[call.Name]
	if !ok {
		res, err := r.ro.handleUnknownTool(ctx, call)
		if err != nil {
			return toolOutcome{fatal: err}
		}
		if res != nil {
			return toolOutcome{result: res}
		}
		// Report an unknown tool back to the model, which usually corrects itself
		// on the next turn.
		return toolOutcome{result: model.ErrorResult(fmt.Sprintf(
			"Error: no tool named %q is available. Available tools: %v.", call.Name, r.tools.names))}
	}

	if t.Policy().NeedsApproval {
		decision, err := r.s.awaitApproval(ctx, r.a, t, call)
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
		logger.Warn("tool call failed", "agent", r.a.name, "tool", call.Name, "error", err)
		return toolOutcome{result: model.ErrorResult(fmt.Sprintf("Error: %v", err))}
	}

	// A tool that ran and reported a business error is handed to the model, which
	// is the party that can respond to it, rather than failing the call.
	if res.IsError {
		logger.Debug("tool reported an error", "agent", r.a.name, "tool", call.Name)
	}

	// The rich result is carried unflattened; the provider flattens it at its edge.
	return toolOutcome{result: res}
}

// classifyToolError reports whether a failed tool call should abort the run. It
// returns non-nil to abort and nil to report the error back to the model.
//
// Business failures (no such user, payment declined) are reported to the model.
// Four classes abort instead: a canceled workflow; the SDK's own configuration
// errors (an unregistered sub-agent, a malformed agent, an unknown provider or
// MCP server, an unsupported setting); an approval failure or lost MCP
// session; and a guardrail tripwire ([NewTripwireError]).
func classifyToolError(ctx workflow.Context, err error) error {
	if errors.Is(ctx.Err(), workflow.ErrCanceled) || temporal.IsCanceledError(err) {
		return err
	}

	var appErr *temporal.ApplicationError
	if errors.As(err, &appErr) && isFatalErrorType(appErr.Type()) {
		return err
	}
	return nil
}

// isFatalErrorType reports whether an application error type aborts the run: an
// SDK configuration error, an approval failure, an unrecoverable session loss
// or a guardrail tripwire.
func isFatalErrorType(t string) bool {
	switch t {
	case ErrorTypeConfig, ErrorTypeUnknownAgent, ErrorTypeApproval, ErrorTypeTripwire,
		model.ErrorTypeNoProvider, model.ErrorTypeUnsupportedSetting, mcp.ErrorTypeUnknownServer, mcp.ErrorTypeSessionLost:
		return true
	default:
		return false
	}
}

// toolSet is the resolved, indexed tools for one run.
type toolSet struct {
	tools  []tool.Tool // in declaration order
	defs   []*model.Tool
	byName map[string]tool.Tool
	names  []string // in declaration order; map iteration is nondeterministic
}

// newToolSet indexes tools in declaration order. A nil tool, a tool without a
// name, or a duplicate name is a non-retryable [ErrorTypeConfig] error.
func newToolSet(agentName string, tools []tool.Tool) (*toolSet, error) {
	ts := &toolSet{
		tools:  make([]tool.Tool, 0, len(tools)),
		defs:   make([]*model.Tool, 0, len(tools)),
		byName: make(map[string]tool.Tool, len(tools)),
		names:  make([]string, 0, len(tools)),
	}
	for i, t := range tools {
		if t == nil {
			return nil, configError("agent %q: tool %d is nil", agentName, i)
		}
		def := t.Def()
		if def == nil || def.Name == "" {
			return nil, configError("agent %q: a tool has no name", agentName)
		}
		if _, dup := ts.byName[def.Name]; dup {
			return nil, configError("agent %q: duplicate tool name %q", agentName, def.Name)
		}
		ts.tools = append(ts.tools, t)
		ts.byName[def.Name] = t
		ts.defs = append(ts.defs, def)
		ts.names = append(ts.names, def.Name)
	}
	return ts, nil
}

// advertisedDefs returns the definitions of the named tools in declaration
// order. Nil or empty names returns every definition. A name not in the set is
// a non-retryable [ErrorTypeConfig] error.
func (ts *toolSet) advertisedDefs(agentName string, names []string) ([]*model.Tool, error) {
	if len(names) == 0 {
		return ts.defs, nil
	}
	want := make(map[string]bool, len(names))
	for _, name := range names {
		if _, ok := ts.byName[name]; !ok {
			return nil, configError("agent %q: advertised tool %q is not in the plan's tools", agentName, name)
		}
		want[name] = true
	}
	defs := make([]*model.Tool, 0, len(want))
	for _, def := range ts.defs {
		if want[def.Name] {
			defs = append(defs, def)
		}
	}
	return defs, nil
}

// resolveTools merges the agent's static tools, the run's ExtraTools (e.g. a
// stateful MCP session's tools) and the tools of every stateless MCP server, in
// that order.
func (s *Session) resolveTools(ctx workflow.Context, a *Agent, ro RunOptions) (*toolSet, error) {
	for i, t := range ro.ExtraTools {
		if t == nil {
			return nil, configError("agent %q: extra tool %d is nil", a.name, i)
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
	return newToolSet(a.name, tools)
}
