package guardrail

import (
	"encoding/json"

	"github.com/wleev/temporal-agent-sdk/model"
)

// Action is what a [Verdict] does with the screened content.
type Action string

// The actions a verdict can take.
const (
	// ActionAllow lets the content through unchanged.
	ActionAllow Action = "allow"
	// ActionBlock stops the content.
	ActionBlock Action = "block"
	// ActionRewrite replaces the content with the verdict's replacement.
	ActionRewrite Action = "rewrite"
	// ActionAnnotate lets the content through and attaches the verdict's
	// notice.
	ActionAnnotate Action = "annotate"
)

// allActions lists every [Action] in declaration order. Every stage accepts
// each of them.
var allActions = []Action{ActionAllow, ActionBlock, ActionRewrite, ActionAnnotate}

// Audience is who a [Notice] is for.
type Audience string

// The audiences of a notice.
const (
	// AudienceModel addresses the notice to the model.
	AudienceModel Audience = "model"
	// AudienceUser addresses the notice to the user.
	AudienceUser Audience = "user"
	// AudienceBoth addresses the notice to the model and the user.
	AudienceBoth Audience = "both"
)

// Notice is a message a verdict attaches for its audience.
type Notice struct {
	// Text is the notice text.
	Text string `json:"text"`

	// Audience is who the notice is for.
	Audience Audience `json:"audience"`
}

// Verdict is a guardrail's judgement on one piece of content. The zero
// Verdict has no action, so [Screen] records it as a failed check, which
// blocks.
type Verdict struct {
	// Action is what the verdict does with the content.
	Action Action

	// Reason explains the verdict for records and logs.
	Reason string

	// Notice is an optional message for the model, the user or both, carried
	// in the verdict's [Record]. An annotation's model- and both-audience
	// notices also reach the model: in [Outcome.ModelNotes], or as text
	// content of [Outcome.Result] at the tool_result stage. The notice of a
	// block is for the user.
	Notice *Notice

	// Text is the replacement of a rewrite at a text stage.
	Text string

	// Arguments is the replacement of a rewrite at the tool_call stage; a JSON
	// object.
	Arguments json.RawMessage

	// Result is the replacement of a rewrite at the tool_result stage.
	Result *model.CallToolResult

	// Info is optional structured detail for observability. It is carried in
	// [Record.Info] and must survive the Temporal data converter (JSON by
	// default) when a record is passed to an activity.
	Info any
}

// Allow returns a verdict that lets the content through unchanged.
func Allow() Verdict {
	return Verdict{Action: ActionAllow}
}

// Block returns a verdict that stops the content. A non-empty userNotice is
// the verdict's notice, with audience [AudienceUser].
func Block(reason, userNotice string) Verdict {
	v := Verdict{Action: ActionBlock, Reason: reason}
	if userNotice != "" {
		v.Notice = &Notice{Text: userNotice, Audience: AudienceUser}
	}
	return v
}

// RewriteText returns a verdict that replaces the text of a text stage. Its
// notice is carried only in the verdict's [Record].
func RewriteText(text, reason string, notice *Notice) Verdict {
	return Verdict{Action: ActionRewrite, Reason: reason, Notice: notice, Text: text}
}

// RewriteArguments returns a verdict that replaces a tool call's arguments with
// args, a JSON object. Its notice is carried only in the verdict's [Record].
func RewriteArguments(args json.RawMessage, reason string, notice *Notice) Verdict {
	return Verdict{Action: ActionRewrite, Reason: reason, Notice: notice, Arguments: args}
}

// RewriteResult returns a verdict that replaces a tool's result. Its notice is
// carried only in the verdict's [Record].
func RewriteResult(result *model.CallToolResult, reason string, notice *Notice) Verdict {
	return Verdict{Action: ActionRewrite, Reason: reason, Notice: notice, Result: result}
}

// Annotate returns a verdict that lets the content through and attaches notice.
func Annotate(reason string, notice Notice) Verdict {
	return Verdict{Action: ActionAnnotate, Reason: reason, Notice: &notice}
}

// WithInfo returns a copy of v with Info set to info.
func (v Verdict) WithInfo(info any) Verdict {
	v.Info = info
	return v
}
