package agent

import (
	"encoding/json"
	"errors"
	"slices"

	"go.temporal.io/sdk/workflow"

	"github.com/wleev/temporal-agent-sdk/guardrail"
	"github.com/wleev/temporal-agent-sdk/model"
	"github.com/wleev/temporal-agent-sdk/tool"
)

// ModelCallPlan describes one model call, passed to and returned by
// [RunOptions.PrepareModelCall].
type ModelCallPlan struct {
	// Turn is the 1-based loop turn of this call. A returned plan with a
	// different Turn aborts the run.
	Turn int

	// Tools are the tools the run dispatches from this call on, in order. Nil or
	// empty keeps the run's current tools.
	Tools []tool.Tool

	// Advertised names the tools sent to the model on this call. Nil or empty
	// sends all of Tools; a subset is sent in Tools order; a name not in Tools
	// aborts the run.
	Advertised []string

	// Settings are the model settings sent on this call only.
	Settings model.Settings

	// OutputSchema, when non-nil, is the output schema sent on this call only,
	// in place of [RunOptions.OutputSchema]. Nil keeps the run's schema.
	OutputSchema *model.OutputSchema

	// Append are messages added to the conversation before this call. They stay
	// in the conversation for the rest of the run.
	Append []model.Message

	// CallOnly are messages sent after the conversation on this call only. They
	// are not added to the conversation or [Result.Messages]. Nil or empty sends
	// nothing extra.
	CallOnly []model.Message

	// Metadata is overlaid on [RunOptions.ProviderMetadata] for this call only:
	// each key here is added or replaces the run's value. No key is removed, and
	// an empty-string value is sent as is. Nil or empty adds nothing.
	Metadata map[string]string
}

// preparedCall is the tool set, advertised definitions, settings, conversation,
// call-only messages, output schema and request metadata for one model call.
type preparedCall struct {
	tools    *toolSet
	defs     []*model.Tool
	settings model.Settings
	msgs     []model.Message
	callOnly []model.Message
	schema   *model.OutputSchema
	metadata map[string]string
}

// prepareCall applies the PrepareModelCall hook for one model call and
// validates the returned plan as [RunOptions.PrepareModelCall] describes. A nil
// hook returns tools, all of their definitions, the agent's settings, msgs, the
// run's output schema and the run's metadata, with no call-only messages.
func (ro RunOptions) prepareCall(ctx workflow.Context, a *Agent, turn int, tools *toolSet, msgs []model.Message) (preparedCall, error) {
	if ro.PrepareModelCall == nil {
		return preparedCall{
			tools: tools, defs: tools.defs, settings: a.settings, msgs: msgs,
			schema: ro.OutputSchema, metadata: ro.ProviderMetadata,
		}, nil
	}

	plan, err := ro.PrepareModelCall(ctx, ModelCallPlan{
		Turn:     turn,
		Tools:    slices.Clone(tools.tools),
		Settings: a.settings.Clone(),
	})
	if err != nil {
		return preparedCall{}, err
	}
	if plan.Turn != turn {
		return preparedCall{}, configError("agent %q: PrepareModelCall returned a plan for turn %d on turn %d", a.name, plan.Turn, turn)
	}

	ts := tools
	if len(plan.Tools) > 0 {
		ts, err = newToolSet(a.name, plan.Tools)
		if err != nil {
			return preparedCall{}, err
		}
	}
	defs, err := ts.advertisedDefs(a.name, plan.Advertised)
	if err != nil {
		return preparedCall{}, err
	}
	if err := checkToolChoice(a.name, plan.Settings.ToolChoice, defs); err != nil {
		return preparedCall{}, err
	}
	if len(plan.Append) > 0 {
		msgs = append(msgs, plan.Append...)
	}
	schema := ro.OutputSchema
	if plan.OutputSchema != nil {
		schema = plan.OutputSchema
	}
	return preparedCall{
		tools: ts, defs: defs, settings: plan.Settings, msgs: msgs, callOnly: plan.CallOnly,
		schema: schema, metadata: callMetadata(ro.ProviderMetadata, plan.Metadata),
	}, nil
}

// checkToolChoice returns a non-retryable [ErrorTypeConfig] error when
// [model.ResolveToolChoice] rejects choice for defs: an invalid mode, a
// required or tool mode with no definitions, or a tool mode whose name is
// empty or not in defs.
func checkToolChoice(agentName string, choice *model.ToolChoice, defs []*model.Tool) error {
	_, err := model.ResolveToolChoice("", choice, defs)
	var unsupported *model.UnsupportedSettingError
	if errors.As(err, &unsupported) {
		return configError("agent %q: tool choice: %s", agentName, unsupported.Reason)
	}
	return err
}

// FinalEvent describes a reply with no tool calls that would end the run,
// passed to [RunOptions.OnFinal].
type FinalEvent struct {
	// Turn is the 1-based loop turn of the reply.
	Turn int

	// Assistant is the model's reply after its guardrail verdicts. After a
	// length-continuation chain it is the last fragment.
	Assistant model.Message

	// StructuredOutput is the reply's structured output after its guardrail
	// verdicts, set only when the reply's call carried an output schema.
	StructuredOutput json.RawMessage

	// FinishReason is the reply's normalized finish reason.
	FinishReason model.FinishReason

	// Usage is the reply's token usage.
	Usage model.Usage

	// Attempt is the reply's [model.Response.Attempt]: the model activity
	// attempt, starting at 1, or 0 when unrecorded.
	Attempt int32
}

// FinalAction is what [RunOptions.OnFinal] returns for a reply that would end
// the run. The zero value ends the run.
type FinalAction struct {
	// Continue continues the loop with the next turn instead of ending the run.
	// [Result.Output] and [Result.StructuredOutput] are reset.
	Continue bool

	// DropReply removes the reply, and the output-stage model notes that follow
	// it, from the conversation before Append. After a length-continuation
	// chain only the reply, the last fragment, is removed; the earlier
	// fragments stay. It requires Continue.
	DropReply bool

	// Append are messages appended to the conversation before the next turn.
	// They require Continue.
	Append []model.Message
}

// emitFinal invokes the OnFinal hook for resp and returns the hook's action. A
// nil hook returns the zero FinalAction. An action that sets DropReply or
// Append without Continue is a non-retryable [ErrorTypeConfig] error.
func (ro RunOptions) emitFinal(ctx workflow.Context, agentName string, turn int, resp *model.Response) (FinalAction, error) {
	if ro.OnFinal == nil {
		return FinalAction{}, nil
	}
	action, err := ro.OnFinal(ctx, FinalEvent{
		Turn:             turn,
		Assistant:        resp.Message,
		StructuredOutput: resp.StructuredOutput,
		FinishReason:     resp.FinishReason,
		Usage:            resp.Usage,
		Attempt:          resp.Attempt,
	})
	if err != nil {
		return FinalAction{}, err
	}
	if !action.Continue && (action.DropReply || len(action.Append) > 0) {
		return FinalAction{}, configError("agent %q: OnFinal returned DropReply or Append without Continue", agentName)
	}
	return action, nil
}

// RoundEvent describes one finished round of tool calls, passed to
// [RunOptions.OnRound].
type RoundEvent struct {
	// Turn is the 1-based loop turn of the reply that made the calls.
	Turn int

	// Assistant is a copy of the reply after its guardrail verdicts, with its
	// own Blocks and each block's own Arguments, Data and Result.
	Assistant model.Message

	// Calls are copies of the reply's tool calls after their guardrail
	// verdicts, in call order, each with its own Arguments.
	Calls []model.ToolCall

	// Results are copies of the round's results in call order, each with its
	// own Content and Meta: the result each tool returned or, for a call the
	// tool_call stage blocked, the loop's error result naming the tool.
	Results []*model.CallToolResult
}

// RoundOutcome is what [RunOptions.OnRound] returns for a round.
type RoundOutcome struct {
	// Results replace the round's results. Empty keeps every result;
	// otherwise it holds one entry per call, in call order, and a nil entry keeps that
	// call's result. A replacement is the whole result the model sees.
	Results []*model.CallToolResult
}

// emitRound invokes the OnRound hook for the round's results when one is set
// and returns the results the round keeps. A returned Results whose length is
// not the number of calls is a non-retryable [ErrorTypeConfig] error.
func (ro RunOptions) emitRound(ctx workflow.Context, agentName string, turn int, assistant model.Message, calls []model.ToolCall, results []*model.CallToolResult) ([]*model.CallToolResult, error) {
	if ro.OnRound == nil {
		return results, nil
	}
	out, err := ro.OnRound(ctx, RoundEvent{
		Turn:      turn,
		Assistant: copyMessage(assistant),
		Calls:     copyCalls(calls),
		Results:   copyResults(results),
	})
	if err != nil {
		return nil, err
	}
	if len(out.Results) == 0 {
		return results, nil
	}
	if len(out.Results) != len(results) {
		return nil, configError("agent %q: OnRound returned %d results; the round has %d tool calls", agentName, len(out.Results), len(results))
	}
	kept := slices.Clone(results)
	for i, r := range out.Results {
		if r != nil {
			kept[i] = r
		}
	}
	return kept, nil
}

// copyMessage returns m with its own Blocks slice, each block holding its own
// Arguments, Data and the [model.CloneResult] of its Result.
func copyMessage(m model.Message) model.Message {
	m.Blocks = slices.Clone(m.Blocks)
	for i := range m.Blocks {
		b := &m.Blocks[i]
		b.Arguments = slices.Clone(b.Arguments)
		b.Data = slices.Clone(b.Data)
		b.Result = model.CloneResult(b.Result)
	}
	return m
}

// copyCalls returns a copy of calls, each call holding its own Arguments.
func copyCalls(calls []model.ToolCall) []model.ToolCall {
	out := slices.Clone(calls)
	for i := range out {
		out[i].Arguments = slices.Clone(out[i].Arguments)
	}
	return out
}

// copyResults returns the [model.CloneResult] of each result in results.
func copyResults(results []*model.CallToolResult) []*model.CallToolResult {
	out := make([]*model.CallToolResult, len(results))
	for i, r := range results {
		out[i] = model.CloneResult(r)
	}
	return out
}

// GuardrailEvent describes one acting guardrail verdict, passed to
// [RunOptions.OnGuardrail].
type GuardrailEvent struct {
	// Record is the verdict's record.
	Record guardrail.Record

	// Attempt is the model activity attempt of the screened reply, or 0 for
	// content that no model reply produced, such as the input.
	Attempt int32
}

// emitGuardrail invokes the OnGuardrail hook for rec when one is set.
func (ro RunOptions) emitGuardrail(ctx workflow.Context, rec guardrail.Record, attempt int32) error {
	if ro.OnGuardrail == nil {
		return nil
	}
	return ro.OnGuardrail(ctx, GuardrailEvent{Record: rec, Attempt: attempt})
}

// handleUnknownTool invokes the OnUnknownTool hook for a call to a tool outside
// the run's dispatchable tools and returns its result. A nil hook returns
// nil, nil.
func (ro RunOptions) handleUnknownTool(ctx workflow.Context, call model.ToolCall) (*model.CallToolResult, error) {
	if ro.OnUnknownTool == nil {
		return nil, nil
	}
	return ro.OnUnknownTool(ctx, call)
}
