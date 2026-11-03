package agent

import (
	"errors"
	"slices"

	"go.temporal.io/sdk/workflow"

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

	// Assistant is the model's reply. After a length-continuation chain it is
	// the last fragment.
	Assistant model.Message

	// FinishReason is the reply's normalized finish reason.
	FinishReason model.FinishReason

	// Usage is the reply's token usage.
	Usage model.Usage

	// Attempt is the reply's [model.Response.Attempt]: the model activity
	// attempt, starting at 1, or 0 when unrecorded.
	Attempt int32
}

// emitFinal invokes the OnFinal hook for resp and returns the hook's messages.
// A nil hook returns nil, nil.
func (ro RunOptions) emitFinal(ctx workflow.Context, turn int, resp *model.Response) ([]model.Message, error) {
	if ro.OnFinal == nil {
		return nil, nil
	}
	return ro.OnFinal(ctx, FinalEvent{
		Turn:         turn,
		Assistant:    resp.Message,
		FinishReason: resp.FinishReason,
		Usage:        resp.Usage,
		Attempt:      resp.Attempt,
	})
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
