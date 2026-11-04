// Package guardrail screens the content flowing through an agent run.
//
// A guardrail checks content at one or more of five stages:
//
//   - [StageInput]: the user's message, before the first model call.
//   - [StageToolCall]: one tool call's name and arguments, before it runs.
//   - [StageToolResult]: the result of a tool that ran. Hosts screen it inside
//     their tools through [Screen]; the agent loop never runs it.
//   - [StageAssistantText]: the text of a reply that calls tools, or of a
//     length-continuation fragment.
//   - [StageOutput]: the reply that would end the run, or a tool's final
//     output; for a call with an output schema, its JSON structured output.
//
// A check returns a [Verdict]: allow, block, rewrite or annotate; every stage
// accepts all four. A block stops the content, with an optional notice for the
// user; a rewrite replaces it; an annotation attaches a [Notice] for the model,
// the user or both. [Screen] runs the guardrails of one stage on one piece of
// content and merges their verdicts into an [Outcome], with a [Record] per
// acting verdict.
//
// A guardrail is either a deterministic Go function ([Func]) or a model call
// ([LLM]). A Func runs in the workflow and must be deterministic: no clock, no
// network, no randomness. An LLM runs its check as [model.InvokeModelActivity],
// the agent loop's model activity, and needs no extra worker registration.
package guardrail

import (
	"errors"
	"fmt"
	"slices"

	"go.temporal.io/sdk/workflow"

	"github.com/wleev/temporal-agent-sdk/model"
)

// Stage is a point in an agent run where guardrails screen content.
type Stage string

// The stages a guardrail can screen.
const (
	// StageInput screens the user's message.
	StageInput Stage = "input"
	// StageToolCall screens one tool call's name and arguments.
	StageToolCall Stage = "tool_call"
	// StageToolResult screens the result of a tool that ran. Hosts screen it
	// inside their tools through [Screen]; the agent loop never runs it.
	StageToolResult Stage = "tool_result"
	// StageAssistantText screens the text of a reply that calls tools, or of a
	// length-continuation fragment.
	StageAssistantText Stage = "assistant_text"
	// StageOutput screens the reply that would end the run, or a tool's final
	// output.
	StageOutput Stage = "output"
)

// allStages lists every [Stage] in declaration order.
var allStages = []Stage{StageInput, StageToolCall, StageToolResult, StageAssistantText, StageOutput}

// Stages returns the five stages in declaration order: input, tool_call,
// tool_result, assistant_text and output. The returned slice is a copy.
func Stages() []Stage {
	return slices.Clone(allStages)
}

// ValidStage reports whether s is one of the five stages.
func ValidStage(s Stage) bool {
	return slices.Contains(allStages, s)
}

// CheckInput is the content a guardrail screens at one stage.
type CheckInput struct {
	// Stage is the stage being screened.
	Stage Stage

	// Text is the screened text for the input, assistant_text and output
	// stages. For the output stage of a call with an output schema it is the
	// JSON structured output.
	Text string

	// Structured reports that Text is the JSON structured output of a call
	// with an output schema. It is set only for the output stage.
	Structured bool

	// Call is the tool call for the tool_call and tool_result stages. [Screen]
	// gives each check its own copy.
	Call *model.ToolCall

	// Result is the tool's result for the tool_result stage. [Screen] gives
	// each check its own copy, with its own Content and Meta.
	Result *model.CallToolResult

	// Conversation is the messages before the screened content. A check must
	// not modify it.
	Conversation []model.Message

	// Turn is the model turn the content belongs to; 0 for the input stage.
	Turn int

	// Index is the call's index within its reply for the tool stages; 0
	// otherwise.
	Index int
}

// Guardrail screens content at the stages it declares. Check runs in workflow
// code and must be deterministic; I/O runs as an activity, as in [LLM].
type Guardrail interface {
	// Name identifies the guardrail in verdict records, tripwire errors and
	// logs.
	Name() string

	// Stages returns the stages the guardrail screens.
	Stages() []Stage

	// Check returns a verdict on in. A non-nil error is a failure of the check
	// itself (for example, an LLM guardrail's model call failed), distinct
	// from a [Block] verdict.
	Check(ctx workflow.Context, in CheckInput) (Verdict, error)
}

// Func builds a deterministic guardrail from a Go function that screens the
// given stages. It returns an error for an empty name, no stages, a stage that
// is not one of the five, or a nil fn.
//
// The function runs inside the workflow, so it must be deterministic: no clock,
// no network, no randomness.
func Func(name string, stages []Stage, fn func(ctx workflow.Context, in CheckInput) (Verdict, error)) (Guardrail, error) {
	if err := checkDeclaration(name, stages); err != nil {
		return nil, err
	}
	if fn == nil {
		return nil, fmt.Errorf("guardrail %q: nil check function", name)
	}
	return funcGuardrail{name: name, stages: slices.Clone(stages), fn: fn}, nil
}

// checkDeclaration returns an error for an empty name, no stages, or a stage
// that is not one of the five.
func checkDeclaration(name string, stages []Stage) error {
	if name == "" {
		return errors.New("guardrail: empty name")
	}
	if len(stages) == 0 {
		return fmt.Errorf("guardrail %q: no stages", name)
	}
	for _, s := range stages {
		if !ValidStage(s) {
			return fmt.Errorf("guardrail %q: unknown stage %q", name, s)
		}
	}
	return nil
}

type funcGuardrail struct {
	name   string
	stages []Stage
	fn     func(workflow.Context, CheckInput) (Verdict, error)
}

func (g funcGuardrail) Name() string { return g.name }

func (g funcGuardrail) Stages() []Stage { return slices.Clone(g.stages) }

func (g funcGuardrail) Check(ctx workflow.Context, in CheckInput) (Verdict, error) {
	return g.fn(ctx, in)
}
