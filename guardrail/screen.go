package guardrail

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"maps"
	"reflect"
	"slices"
	"strings"

	mcpsdk "github.com/modelcontextprotocol/go-sdk/mcp"
	"go.temporal.io/sdk/temporal"
	"go.temporal.io/sdk/workflow"

	"github.com/wleev/temporal-agent-sdk/model"
)

// Record is one acting verdict of one guardrail on one piece of content.
type Record struct {
	// Stage is the stage of the screened content.
	Stage Stage `json:"stage"`

	// Guardrail is the name of the guardrail.
	Guardrail string `json:"guardrail"`

	// Action is block, rewrite or annotate. A failed check is [ActionBlock]
	// with Failed set.
	Action Action `json:"action"`

	// Reason is the verdict's reason, or the failure for a failed check.
	Reason string `json:"reason"`

	// Notice is the verdict's notice, if any. A failed check has none.
	Notice *Notice `json:"notice,omitempty"`

	// Failed reports that the check returned an error, panicked, or returned
	// a malformed verdict.
	Failed bool `json:"failed,omitempty"`

	// Turn is the [CheckInput] Turn of the content.
	Turn int `json:"turn"`

	// Index is the [CheckInput] Index of the content.
	Index int `json:"index"`

	// Info is the verdict's Info.
	Info any `json:"info,omitempty"`
}

// Outcome is the merged result of screening one piece of content.
type Outcome struct {
	// Blocked is the first block or failed record in declared order; nil when
	// nothing blocks. On a block, only Blocked and Records are set.
	Blocked *Record

	// Text is the text after the verdicts, for the text stages.
	Text string

	// Arguments is the tool call's arguments after the verdicts, for the
	// tool_call stage.
	Arguments json.RawMessage

	// Result is a copy of the tool result after the verdicts, for the
	// tool_result stage, with model- and both-audience notes appended as text
	// content and the Meta of [CheckInput.Result].
	Result *model.CallToolResult

	// Rewritten reports that at least one rewrite was applied.
	Rewritten bool

	// ModelNotes holds the texts of the model- and both-audience annotations
	// in declared order. It is empty for the tool_result stage.
	ModelNotes []string

	// Records holds the acting verdicts in declared order, at most one per
	// guardrail.
	// On a block it holds only the block and failed records.
	Records []Record
}

// judgement is one guardrail's verdict, or its failure, during a [Screen].
type judgement struct {
	verdict Verdict
	failure string
}

func (j judgement) blocks() bool {
	return j.failure != "" || j.verdict.Action == ActionBlock
}

// Screen runs the guardrails among guards whose Stages include in.Stage on in
// and merges their verdicts.
//
// The checks run concurrently, each with its own copy of in.Call and
// in.Result, and are judged in declared order:
//   - A check error, a check panic, or a malformed verdict is a failed check,
//     recorded as a block. Malformed means an action other than allow, block,
//     rewrite and annotate (the zero Verdict included); a block whose notice
//     is not for the user; a rewrite at a text stage with an empty Text or,
//     when in.Structured is set, Text that is not JSON; a rewrite at the
//     tool_call stage with Arguments that are not a JSON object; a rewrite at
//     the tool_result stage with a nil Result; or an annotation without
//     notice text or with an audience other than model, user or both. A check
//     error with an empty message is recorded with the reason "check failed".
//   - A block wins: the outcome carries the first block or failed check in
//     declared order and the records of every block and failed check.
//   - Otherwise rewrites apply in declared order. The first applies as
//     returned; each later rewriting guardrail is checked again, in turn, on
//     the content so far, and its new verdict replaces its first one. When
//     that check blocks or fails, the outcome is a block whose Records hold
//     only that check's record.
//   - Annotations accumulate. Model- and both-audience notices go to
//     [Outcome.ModelNotes], or, for the tool_result stage, are appended to the
//     result as text content.
//
// Screen recovers every panic of a check, including the Temporal SDK's own
// string panics for misuse of workflow APIs, except a panic whose value has an
// SDK-internal type that does not implement error, such as the SDK's
// illegal-state panic on a nondeterministic replay.
//
// Screen returns an error for a nil entry in guards, before any check runs,
// and when a check fails because the workflow was canceled; that error is the
// check's error.
func Screen(ctx workflow.Context, guards []Guardrail, in CheckInput) (Outcome, error) {
	applicable := make([]Guardrail, 0, len(guards))
	for i, g := range guards {
		if g == nil {
			return Outcome{}, fmt.Errorf("guardrail: nil guardrail at index %d", i)
		}
		if slices.Contains(g.Stages(), in.Stage) {
			applicable = append(applicable, g)
		}
	}

	judgements := make([]judgement, len(applicable))
	errs := make([]error, len(applicable))
	wg := workflow.NewWaitGroup(ctx)
	for i, g := range applicable {
		wg.Add(1)
		workflow.Go(ctx, func(gctx workflow.Context) {
			defer wg.Done()
			judgements[i], errs[i] = check(gctx, g, ownCopy(in))
		})
	}
	wg.Wait(ctx)

	for _, err := range errs {
		if err != nil {
			return Outcome{}, err
		}
	}
	if slices.ContainsFunc(judgements, judgement.blocks) {
		return blockedOutcome(ctx, in, applicable, judgements), nil
	}

	var meta mcpsdk.Meta
	if in.Result != nil {
		meta = in.Result.Meta
	}
	text, args, result := in.Text, json.RawMessage(nil), model.CloneResult(in.Result)
	if in.Call != nil {
		args = in.Call.Arguments
	}
	rewritten := false
	for i, g := range applicable {
		if judgements[i].verdict.Action != ActionRewrite {
			continue
		}
		if rewritten {
			again := in
			again.Text = text
			again.Result = result
			if in.Call != nil {
				call := *in.Call
				call.Arguments = args
				again.Call = &call
			}
			j, err := check(ctx, g, ownCopy(again))
			if err != nil {
				return Outcome{}, err
			}
			judgements[i] = j
			if judgements[i].blocks() {
				return blockedOutcome(ctx, in, applicable[i:i+1], judgements[i:i+1]), nil
			}
			if judgements[i].verdict.Action != ActionRewrite {
				continue
			}
		}
		v := judgements[i].verdict
		switch in.Stage {
		case StageToolCall:
			args = v.Arguments
		case StageToolResult:
			result = model.CloneResult(v.Result)
			result.Meta = maps.Clone(meta)
		default:
			text = v.Text
		}
		rewritten = true
	}

	out := Outcome{Text: text, Arguments: args, Result: result, Rewritten: rewritten}
	for i, g := range applicable {
		v := judgements[i].verdict
		if v.Action != ActionRewrite && v.Action != ActionAnnotate {
			continue
		}
		out.Records = append(out.Records, record(in, g, judgements[i]))
		if v.Action != ActionAnnotate || (v.Notice.Audience != AudienceModel && v.Notice.Audience != AudienceBoth) {
			continue
		}
		if in.Stage == StageToolResult {
			if out.Result == nil {
				out.Result = &model.CallToolResult{}
			}
			out.Result.Content = append(out.Result.Content, &mcpsdk.TextContent{Text: v.Notice.Text})
		} else {
			out.ModelNotes = append(out.ModelNotes, v.Notice.Text)
		}
	}
	logRecords(ctx, out.Records)
	return out, nil
}

// sdkInternalPkg is the import path of the Temporal SDK's internal package.
const sdkInternalPkg = "go.temporal.io/sdk/internal"

// check runs g.Check on in and judges its verdict for in. When the check fails
// because the workflow was canceled, check returns the check's error as
// canceled. A panic of the check is a failed judgement holding the panic value;
// a panic whose value does not implement error and has a type from the
// Temporal SDK's internal packages is raised again.
func check(ctx workflow.Context, g Guardrail, in CheckInput) (j judgement, canceled error) {
	name := g.Name()
	defer func() {
		r := recover()
		if r == nil {
			return
		}
		if isSDKPanic(r) {
			panic(r)
		}
		j, canceled = judgement{failure: fmt.Sprintf("guardrail %q panicked: %v", name, r)}, nil
	}()
	v, err := g.Check(ctx, in)
	if err == nil {
		return judgement{verdict: v, failure: malformedReason(in, v)}, nil
	}
	j = judgement{verdict: v, failure: err.Error()}
	if j.failure == "" {
		j.failure = "check failed"
	}
	if isCancellation(ctx, err) {
		return j, err
	}
	return j, nil
}

// isSDKPanic reports whether the panic value r does not implement error and
// its type, or the type it points to, is declared in the Temporal SDK's
// internal packages.
func isSDKPanic(r any) bool {
	if _, ok := r.(error); ok {
		return false
	}
	t := reflect.TypeOf(r)
	if t.Kind() == reflect.Pointer {
		t = t.Elem()
	}
	p := t.PkgPath()
	return p == sdkInternalPkg || strings.HasPrefix(p, sdkInternalPkg+"/")
}

// malformedReason describes why v is not an acceptable verdict on in, or
// returns "" when it is.
func malformedReason(in CheckInput, v Verdict) string {
	if !slices.Contains(allActions, v.Action) {
		return fmt.Sprintf("action %q is not allow, block, rewrite or annotate", v.Action)
	}
	switch v.Action {
	case ActionBlock:
		if v.Notice != nil && v.Notice.Audience != AudienceUser {
			return fmt.Sprintf("block notice audience %q is not user", v.Notice.Audience)
		}
	case ActionRewrite:
		switch in.Stage {
		case StageToolCall:
			if !isJSONObject(v.Arguments) {
				return "rewrite arguments are not a JSON object"
			}
		case StageToolResult:
			if v.Result == nil {
				return "rewrite has no result"
			}
		default:
			if v.Text == "" {
				return "rewrite has no text"
			}
			if in.Structured && !json.Valid([]byte(v.Text)) {
				return "rewrite of structured output is not JSON"
			}
		}
	case ActionAnnotate:
		if v.Notice == nil || v.Notice.Text == "" {
			return "annotation has no notice text"
		}
		if a := v.Notice.Audience; a != AudienceModel && a != AudienceUser && a != AudienceBoth {
			return fmt.Sprintf("annotation audience %q is not model, user or both", a)
		}
	}
	return ""
}

// isJSONObject reports whether raw is a valid JSON object.
func isJSONObject(raw json.RawMessage) bool {
	trimmed := bytes.TrimSpace(raw)
	return len(trimmed) > 0 && trimmed[0] == '{' && json.Valid(trimmed)
}

// isCancellation reports whether err, a non-nil check error, is caused by the
// workflow's cancellation.
func isCancellation(ctx workflow.Context, err error) bool {
	return temporal.IsCanceledError(err) || errors.Is(ctx.Err(), workflow.ErrCanceled)
}

// blockedOutcome builds the outcome of a block: the block and failed records
// of judgements in declared order, the first of them as Blocked.
func blockedOutcome(ctx workflow.Context, in CheckInput, guards []Guardrail, judgements []judgement) Outcome {
	var out Outcome
	for i, g := range guards {
		if judgements[i].blocks() {
			out.Records = append(out.Records, record(in, g, judgements[i]))
		}
	}
	first := out.Records[0]
	out.Blocked = &first
	logRecords(ctx, out.Records)
	return out
}

// record builds the record of g's judgement j on in.
func record(in CheckInput, g Guardrail, j judgement) Record {
	r := Record{
		Stage:     in.Stage,
		Guardrail: g.Name(),
		Action:    j.verdict.Action,
		Reason:    j.verdict.Reason,
		Notice:    j.verdict.Notice,
		Turn:      in.Turn,
		Index:     in.Index,
		Info:      j.verdict.Info,
	}
	if j.failure != "" {
		r.Action = ActionBlock
		r.Reason = j.failure
		r.Notice = nil
		r.Failed = true
	}
	return r
}

// ownCopy returns in with a copy of its Call, holding a copy of the call's
// Arguments, and the [model.CloneResult] of its Result.
func ownCopy(in CheckInput) CheckInput {
	if in.Call != nil {
		c := *in.Call
		c.Arguments = slices.Clone(c.Arguments)
		in.Call = &c
	}
	in.Result = model.CloneResult(in.Result)
	return in
}

// logRecords logs each record at Info level.
func logRecords(ctx workflow.Context, records []Record) {
	logger := workflow.GetLogger(ctx)
	for _, r := range records {
		logger.Info("guardrail acted",
			"stage", r.Stage, "guardrail", r.Guardrail, "action", r.Action, "failed", r.Failed)
	}
}
