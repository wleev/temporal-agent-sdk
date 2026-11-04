package agent_test

import (
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.temporal.io/sdk/temporal"
	"go.temporal.io/sdk/workflow"

	"github.com/wleev/temporal-agent-sdk/agent"
	"github.com/wleev/temporal-agent-sdk/agenttest"
	"github.com/wleev/temporal-agent-sdk/guardrail"
	"github.com/wleev/temporal-agent-sdk/model"
	"github.com/wleev/temporal-agent-sdk/tool"
)

// guardMarkerOptions returns markerOptions(seq) extended with an OnGuardrail
// that appends "guard:<stage>:<guardrail>" and an OnFinal that appends
// "final:<turn>" and returns the zero FinalAction.
func guardMarkerOptions(seq *[]string) agent.RunOptions {
	ro := markerOptions(seq)
	ro.OnGuardrail = func(_ workflow.Context, e agent.GuardrailEvent) error {
		*seq = append(*seq, fmt.Sprintf("guard:%s:%s", e.Record.Stage, e.Record.Guardrail))
		return nil
	}
	ro.OnFinal = func(_ workflow.Context, e agent.FinalEvent) (agent.FinalAction, error) {
		*seq = append(*seq, fmt.Sprintf("final:%d", e.Turn))
		return agent.FinalAction{}, nil
	}
	return ro
}

// hasAssistantMessage reports whether msgs holds an assistant message.
func hasAssistantMessage(msgs []model.Message) bool {
	for _, m := range msgs {
		if m.IsAssistant() {
			return true
		}
	}
	return false
}

// textAndCalls scripts an assistant reply with text followed by tool calls to
// get_weather for the given cities, with call IDs x, y, ….
func textAndCalls(text string, cities ...string) agenttest.Response {
	blocks := []model.Block{model.TextBlock(text)}
	for i, city := range cities {
		blocks = append(blocks, model.ToolCallBlock(string(rune('x'+i)), "get_weather",
			json.RawMessage(fmt.Sprintf(`{"city":%q}`, city))))
	}
	return agenttest.Response{
		Message: model.Message{Role: model.RoleAssistant, Blocks: blocks},
		Usage:   model.Usage{PromptTokens: 10, CompletionTokens: 5, TotalTokens: 15},
	}
}

// cityTool returns a get_weather tool that appends "tool" to seq and the city
// to cities each time it runs.
func cityTool(t *testing.T, seq, cities *[]string) tool.Tool {
	t.Helper()
	tl, err := tool.New("get_weather", "Look up the weather",
		func(_ workflow.Context, in weatherIn) (string, error) {
			*seq = append(*seq, "tool")
			*cities = append(*cities, in.City)
			return "18°C in " + in.City, nil
		})
	require.NoError(t, err)
	return tl
}

func TestRunWith_OutputBlockSkipsEveryReportingHook(t *testing.T) {
	fake := agenttest.NewFakeProvider(agenttest.Says("here is a leaked secret"))
	a, err := agent.NewAgent("assistant", "test-model",
		agent.WithGuardrails(tripOn(t, guardrail.StageOutput, "no-leaks", "here is a leaked secret", "contains a secret")))
	require.NoError(t, err)

	var seq []string
	res, err := runPrepared(t, fake, a, "tell me something", guardMarkerOptions(&seq))

	te, ok := agent.AsTripwire(err)
	if assert.True(t, ok, "got %v", err) {
		assert.Equal(t, guardrail.StageOutput, te.Stage)
		assert.Equal(t, 1, te.Turn)
	}
	assert.Equal(t, []string{"guard:output:no-leaks"}, seq)
	if assert.NotNil(t, res) {
		assert.False(t, hasAssistantMessage(res.Messages), "the blocked reply is not kept")
		assert.Empty(t, res.Output)
	}
}

func TestRunWith_OutputRewriteReachesEveryHookAndTheResult(t *testing.T) {
	fake := agenttest.NewFakeProvider(agenttest.Says("Mild in both."))
	mask := mustFunc(t, "mask", []guardrail.Stage{guardrail.StageOutput}, func(workflow.Context, guardrail.CheckInput) (guardrail.Verdict, error) {
		return guardrail.RewriteText("Mild in [x].", "mask", nil), nil
	})
	a, err := agent.NewAgent("assistant", "test-model", agent.WithGuardrails(mask))
	require.NoError(t, err)

	var seq []string
	var reply agent.ReplyEvent
	var turnEv agent.TurnEvent
	var final agent.FinalEvent
	ro := guardMarkerOptions(&seq)
	onReply, onTurn, onFinal := ro.OnReply, ro.OnTurn, ro.OnFinal
	ro.OnReply = func(ctx workflow.Context, e agent.ReplyEvent) error { reply = e; return onReply(ctx, e) }
	ro.OnTurn = func(ctx workflow.Context, e agent.TurnEvent) error { turnEv = e; return onTurn(ctx, e) }
	ro.OnFinal = func(ctx workflow.Context, e agent.FinalEvent) (agent.FinalAction, error) {
		final = e
		return onFinal(ctx, e)
	}

	res, err := runPrepared(t, fake, a, "Weather?", ro)
	require.NoError(t, err)

	assert.Equal(t, []string{"guard:output:mask", "reply:1", "turn:1", "final:1"}, seq)
	assert.Equal(t, "Mild in [x].", reply.Assistant.Text())
	assert.Equal(t, "Mild in [x].", turnEv.Assistant.Text())
	assert.Equal(t, "Mild in [x].", final.Assistant.Text())
	assert.Equal(t, "Mild in [x].", res.Output)
	if assert.NotEmpty(t, res.Messages) {
		assert.Equal(t, "Mild in [x].", res.Messages[len(res.Messages)-1].Text())
	}
}

func TestRunWith_ToolCallingReplyRunsAssistantTextAndToolCallStages(t *testing.T) {
	fake := agenttest.NewFakeProvider(
		textAndCalls("Checking both.", "Ghent", "Bruges"),
		agenttest.Says("Mild in both."),
	)
	note := mustFunc(t, "a", []guardrail.Stage{guardrail.StageAssistantText}, func(workflow.Context, guardrail.CheckInput) (guardrail.Verdict, error) {
		return guardrail.Annotate("r", guardrail.Notice{Text: "Note A", Audience: guardrail.AudienceModel}), nil
	})
	rw := mustFunc(t, "rw", []guardrail.Stage{guardrail.StageToolCall}, func(_ workflow.Context, in guardrail.CheckInput) (guardrail.Verdict, error) {
		if strings.Contains(string(in.Call.Arguments), "Bruges") {
			return guardrail.RewriteArguments(json.RawMessage(`{"city":"Brugge"}`), "spelling", nil), nil
		}
		return guardrail.Allow(), nil
	})
	var seq, cities []string
	a, err := agent.NewAgent("assistant", "test-model",
		agent.WithTools(cityTool(t, &seq, &cities)),
		agent.WithGuardrails(note),
		agent.WithGuardrails(rw))
	require.NoError(t, err)

	var replies []agent.ReplyEvent
	ro := guardMarkerOptions(&seq)
	onReply := ro.OnReply
	ro.OnReply = func(ctx workflow.Context, e agent.ReplyEvent) error {
		replies = append(replies, e)
		return onReply(ctx, e)
	}
	_, err = runPrepared(t, fake, a, "Weather in Ghent and Bruges?", ro)
	require.NoError(t, err)

	assert.Equal(t, []string{
		"guard:assistant_text:a", "guard:tool_call:rw", "reply:1", "tool", "tool", "turn:1",
		"reply:2", "turn:2", "final:2",
	}, seq)
	if assert.NotEmpty(t, replies) {
		if calls := replies[0].Assistant.ToolCalls(); assert.Len(t, calls, 2) {
			assert.JSONEq(t, `{"city":"Ghent"}`, string(calls[0].Arguments))
			assert.JSONEq(t, `{"city":"Brugge"}`, string(calls[1].Arguments))
		}
	}
	assert.ElementsMatch(t, []string{"Ghent", "Brugge"}, cities)

	if assert.Equal(t, 2, fake.CallCount()) {
		msgs := fake.Calls()[1].Messages
		if assert.GreaterOrEqual(t, len(msgs), 3) {
			assert.NotEmpty(t, msgs[len(msgs)-3].ToolResultID())
			assert.NotEmpty(t, msgs[len(msgs)-2].ToolResultID())
			assert.Equal(t, model.RoleUser, msgs[len(msgs)-1].Role)
			assert.Equal(t, "Note A", msgs[len(msgs)-1].Text())
		}
	}
}

// denyCity returns a tool_call guardrail that blocks calls whose arguments
// mention city.
func denyCity(t *testing.T, city string) guardrail.Guardrail {
	t.Helper()
	return mustFunc(t, "deny", []guardrail.Stage{guardrail.StageToolCall}, func(_ workflow.Context, in guardrail.CheckInput) (guardrail.Verdict, error) {
		if strings.Contains(string(in.Call.Arguments), city) {
			return guardrail.Block("disallowed city", "I can't look that city up."), nil
		}
		return guardrail.Allow(), nil
	})
}

func TestRunWith_ToolCallBlockRefusesOnlyThatCall(t *testing.T) {
	fake := agenttest.NewFakeProvider(
		agenttest.CallsTools(
			agenttest.ToolCall{ID: "x", Name: "get_weather", Args: `{"city":"Ghent"}`},
			agenttest.ToolCall{ID: "y", Name: "get_weather", Args: `{"city":"Bruges"}`},
		),
		agenttest.Says("Bruges is mild."),
	)
	var seq, cities []string
	a, err := agent.NewAgent("assistant", "test-model",
		agent.WithTools(cityTool(t, &seq, &cities)), agent.WithGuardrails(denyCity(t, "Ghent")))
	require.NoError(t, err)

	var events []agent.GuardrailEvent
	var round []string
	ro := agent.RunOptions{
		OnGuardrail: guardrailEvents(&events),
		OnRound: func(_ workflow.Context, e agent.RoundEvent) (agent.RoundOutcome, error) {
			round = resultTexts(e.Results)
			return agent.RoundOutcome{}, nil
		},
	}
	res, err := runPrepared(t, fake, a, "Weather in Ghent and Bruges?", ro)
	require.NoError(t, err)

	blocked := `Error: a guardrail blocked this call to "get_weather".`
	assert.Equal(t, []string{"Bruges"}, cities, "the later call is screened and runs")
	assert.True(t, blockedResult(t, res.Messages, "x").IsError)
	assert.Equal(t, []string{blocked, "18°C in Bruges"}, round)
	assert.Equal(t, []string{blocked, "18°C in Bruges"}, toolResultTexts(res.Messages))
	assert.Equal(t, "Bruges is mild.", res.Output)
	assert.Equal(t, 2, fake.CallCount())
	if assert.Len(t, events, 1) {
		rec := events[0].Record
		assert.Equal(t, guardrail.StageToolCall, rec.Stage)
		assert.Equal(t, guardrail.ActionBlock, rec.Action)
		assert.Equal(t, 0, rec.Index)
		assert.Equal(t, "disallowed city", rec.Reason)
	}
}

// blockedResult returns the result of the tool-result message for call id in
// msgs.
func blockedResult(t *testing.T, msgs []model.Message, id string) *model.CallToolResult {
	t.Helper()
	for _, m := range msgs {
		if m.ToolResultID() != id {
			continue
		}
		for _, b := range m.Blocks {
			if b.Result != nil {
				return b.Result
			}
		}
	}
	require.FailNow(t, "no tool result", "call %q", id)
	return nil
}

// blockAll returns a tool_call guardrail named name that blocks every call.
func blockAll(t *testing.T, name string) guardrail.Guardrail {
	t.Helper()
	return mustFunc(t, name, []guardrail.Stage{guardrail.StageToolCall}, func(workflow.Context, guardrail.CheckInput) (guardrail.Verdict, error) {
		return guardrail.Block("blocked", ""), nil
	})
}

func TestRunWith_ToolCallBlockAfterTextAndARewrittenCall(t *testing.T) {
	fake := agenttest.NewFakeProvider(
		textAndCalls("Checking both.", "Bruges", "Ghent"),
		agenttest.Says("Brugge is mild."),
	)
	rw := mustFunc(t, "rw", []guardrail.Stage{guardrail.StageToolCall}, func(_ workflow.Context, in guardrail.CheckInput) (guardrail.Verdict, error) {
		if strings.Contains(string(in.Call.Arguments), "Ghent") {
			return guardrail.RewriteArguments(json.RawMessage(`{"city":"Gent"}`), "spelling", nil), nil
		}
		return guardrail.RewriteArguments(json.RawMessage(`{"city":"Brugge"}`), "spelling", nil), nil
	})
	note := mustFunc(t, "note", []guardrail.Stage{guardrail.StageToolCall}, func(workflow.Context, guardrail.CheckInput) (guardrail.Verdict, error) {
		return guardrail.Annotate("r", guardrail.Notice{Text: "Checked.", Audience: guardrail.AudienceModel}), nil
	})
	var seq, cities []string
	a, err := agent.NewAgent("assistant", "test-model",
		agent.WithTools(cityTool(t, &seq, &cities)), agent.WithGuardrails(rw, note, denyCity(t, "Ghent")))
	require.NoError(t, err)

	var calls []model.ToolCall
	res, err := runPrepared(t, fake, a, "Weather in Bruges and Ghent?", agent.RunOptions{
		OnRound: func(_ workflow.Context, e agent.RoundEvent) (agent.RoundOutcome, error) {
			calls = e.Calls
			return agent.RoundOutcome{}, nil
		},
	})
	require.NoError(t, err)

	assert.Equal(t, []string{"Brugge"}, cities)
	assert.Equal(t, []string{"18°C in Brugge", `Error: a guardrail blocked this call to "get_weather".`},
		toolResultTexts(res.Messages))
	if assert.Len(t, calls, 2) {
		assert.JSONEq(t, `{"city":"Brugge"}`, string(calls[0].Arguments))
		assert.JSONEq(t, `{"city":"Ghent"}`, string(calls[1].Arguments), "the blocked call keeps its original arguments")
	}
	if assert.Equal(t, 2, fake.CallCount()) {
		msgs := fake.Calls()[1].Messages
		assert.Equal(t, "Checked.", msgs[len(msgs)-1].Text(), "only the allowed call's note is held")
	}
}

func TestRunWith_EveryToolCallBlockedContinuesTheRun(t *testing.T) {
	fake := agenttest.NewFakeProvider(textAndCalls("", "Ghent", "Bruges"), agenttest.Says("I can't look those up."))
	var seq, cities []string
	a, err := agent.NewAgent("assistant", "test-model",
		agent.WithTools(cityTool(t, &seq, &cities)), agent.WithGuardrails(blockAll(t, "all")))
	require.NoError(t, err)

	res, err := runPrepared(t, fake, a, "Weather in Ghent and Bruges?", agent.RunOptions{})
	require.NoError(t, err)

	blocked := `Error: a guardrail blocked this call to "get_weather".`
	assert.Empty(t, cities)
	assert.Equal(t, []string{blocked, blocked}, toolResultTexts(res.Messages))
	assert.Equal(t, "I can't look those up.", res.Output)
}

func TestRunWith_BlockedEndsRunToolDoesNotEndTheRun(t *testing.T) {
	fake := agenttest.NewFakeProvider(agenttest.CallsTool("submit", `{}`), agenttest.Says("Not submitted."))
	a, err := agent.NewAgent("assistant", "test-model",
		agent.WithTools(finalTool(t, "submit", "Submitted.", tool.EndsRun())), agent.WithGuardrails(blockAll(t, "all")))
	require.NoError(t, err)

	res, err := runPrepared(t, fake, a, "Submit it.", agent.RunOptions{})
	require.NoError(t, err)

	assert.Equal(t, "Not submitted.", res.Output)
	assert.Equal(t, 2, fake.CallCount())
}

func TestRunWith_BlockedUnknownToolSkipsOnUnknownTool(t *testing.T) {
	fake := agenttest.NewFakeProvider(agenttest.CallsTool("delete_account", `{}`), agenttest.Says("I can't do that."))
	a, err := agent.NewAgent("assistant", "test-model", agent.WithGuardrails(blockAll(t, "all")))
	require.NoError(t, err)

	var unknown int
	res, err := runPrepared(t, fake, a, "Delete my account.", agent.RunOptions{
		OnUnknownTool: func(workflow.Context, model.ToolCall) (*model.CallToolResult, error) {
			unknown++
			return model.TextResult("deleted"), nil
		},
	})
	require.NoError(t, err)

	assert.Zero(t, unknown)
	assert.Equal(t, []string{`Error: a guardrail blocked this call to "delete_account".`}, toolResultTexts(res.Messages))
}

func TestRunWith_OnGuardrailErrorEndsTheRunOnAToolCallBlock(t *testing.T) {
	fake := agenttest.NewFakeProvider(agenttest.CallsTool("get_weather", `{"city":"Ghent"}`), agenttest.Says("unused"))
	var seq, cities []string
	a, err := agent.NewAgent("assistant", "test-model",
		agent.WithTools(cityTool(t, &seq, &cities)), agent.WithGuardrails(denyCity(t, "Ghent")))
	require.NoError(t, err)

	stop := errors.New("stop on block")
	_, err = runPrepared(t, fake, a, "Weather in Ghent?", agent.RunOptions{
		OnGuardrail: func(_ workflow.Context, e agent.GuardrailEvent) error {
			if e.Record.Action == guardrail.ActionBlock {
				return stop
			}
			return nil
		},
	})

	require.ErrorIs(t, err, stop)
	_, tripped := agent.AsTripwire(err)
	assert.False(t, tripped)
	assert.Empty(t, cities)
	assert.Equal(t, 1, fake.CallCount())
}

func TestRunWith_FailedOutputCheckBlocks(t *testing.T) {
	fake := agenttest.NewFakeProvider(agenttest.Says("Mild."))
	judge := mustFunc(t, "judge", []guardrail.Stage{guardrail.StageOutput}, func(workflow.Context, guardrail.CheckInput) (guardrail.Verdict, error) {
		return guardrail.Verdict{}, errors.New("model unavailable")
	})
	a, err := agent.NewAgent("assistant", "test-model", agent.WithGuardrails(judge))
	require.NoError(t, err)

	_, err = runPrepared(t, fake, a, "Weather?", agent.RunOptions{})

	te, ok := agent.AsTripwire(err)
	if assert.True(t, ok, "got %v", err) {
		assert.True(t, te.Failed)
		assert.Equal(t, guardrail.StageOutput, te.Stage)
		assert.Contains(t, te.Reason, "model unavailable")
	}
}

func TestRunWith_OnFinalContinuationIsScreenedAgain(t *testing.T) {
	fake := agenttest.NewFakeProvider(agenttest.Says("first"), agenttest.Says("second"))
	var screened []string
	record := mustFunc(t, "record", []guardrail.Stage{guardrail.StageOutput}, func(_ workflow.Context, in guardrail.CheckInput) (guardrail.Verdict, error) {
		screened = append(screened, in.Text)
		return guardrail.Allow(), nil
	})
	a, err := agent.NewAgent("assistant", "test-model", agent.WithGuardrails(record))
	require.NoError(t, err)

	res, err := runPrepared(t, fake, a, "Hi", agent.RunOptions{OnFinal: nudgeOnce("nudge")})
	require.NoError(t, err)

	assert.Equal(t, []string{"first", "second"}, screened)
	assert.Equal(t, "second", res.Output)
}

func TestRunWith_GuardrailEventCarriesTheReplyAttempt(t *testing.T) {
	fake := agenttest.NewFakeProvider(
		agenttest.Fails(&model.APIError{StatusCode: 503}),
		agenttest.Says("Mild."),
	)
	note := mustFunc(t, "note", []guardrail.Stage{guardrail.StageOutput}, func(workflow.Context, guardrail.CheckInput) (guardrail.Verdict, error) {
		return guardrail.Annotate("r", guardrail.Notice{Text: "Checked.", Audience: guardrail.AudienceUser}), nil
	})
	a, err := agent.NewAgent("assistant", "test-model", agent.WithGuardrails(note))
	require.NoError(t, err)

	var events []agent.GuardrailEvent
	_, err = runPrepared(t, fake, a, "Weather?", agent.RunOptions{OnGuardrail: guardrailEvents(&events)})
	require.NoError(t, err)

	if assert.Len(t, events, 1) {
		assert.Equal(t, int32(2), events[0].Attempt)
		assert.Equal(t, 1, events[0].Record.Turn)
	}
}

func TestRunWith_NoGuardrailsSchedulesNoExtraCommands(t *testing.T) {
	fake := agenttest.NewFakeProvider(
		agenttest.CallsTools(
			agenttest.ToolCall{ID: "x", Name: "get_weather", Args: `{"city":"Ghent"}`},
			agenttest.ToolCall{ID: "y", Name: "get_weather", Args: `{"city":"Bruges"}`},
		),
		agenttest.Says("Mild in both."),
	)
	var seq []string
	a, err := agent.NewAgent("assistant", "test-model", agent.WithTools(markerWeatherTool(t, &seq)))
	require.NoError(t, err)

	res, err := runPrepared(t, fake, a, "Weather?", markerOptions(&seq))
	require.NoError(t, err)

	assert.Equal(t, []string{"reply:1", "tool", "tool", "turn:1", "reply:2", "turn:2"}, seq)
	assert.Equal(t, "Mild in both.", res.Output)
	assert.Len(t, res.Messages, 5, "user, assistant, two results, answer")
}

func TestRunWith_PerCallSchemaReplyIsScreenedStructured(t *testing.T) {
	fake := agenttest.NewFakeProvider(agenttest.Says(`{"temp":"18°C"}`))
	var structured []bool
	mask := mustFunc(t, "mask", []guardrail.Stage{guardrail.StageOutput}, func(_ workflow.Context, in guardrail.CheckInput) (guardrail.Verdict, error) {
		structured = append(structured, in.Structured)
		return guardrail.RewriteText("not json", "mask", nil), nil
	})
	a, err := agent.NewAgent("assistant", "test-model", agent.WithGuardrails(mask))
	require.NoError(t, err)

	_, err = runPrepared(t, fake, a, "Weather?", agent.RunOptions{
		PrepareModelCall: func(_ workflow.Context, p agent.ModelCallPlan) (agent.ModelCallPlan, error) {
			if p.Turn == 1 {
				p.OutputSchema = &model.OutputSchema{Name: "weather", Schema: json.RawMessage(`{"type":"object"}`)}
			}
			return p, nil
		},
	})

	assert.Equal(t, []bool{true}, structured)
	te, ok := agent.AsTripwire(err)
	if assert.True(t, ok, "got %v", err) {
		assert.True(t, te.Failed)
	}
}

func TestRunWith_StructuredOutputRewriteReplacesTheStructuredOutput(t *testing.T) {
	fake := agenttest.NewFakeProvider(agenttest.Says(`{"temp":"18°C"}`))
	mask := mustFunc(t, "mask", []guardrail.Stage{guardrail.StageOutput}, func(_ workflow.Context, in guardrail.CheckInput) (guardrail.Verdict, error) {
		return guardrail.RewriteText(strings.ReplaceAll(in.Text, "18°C", "[x]"), "mask", nil), nil
	})
	a, err := agent.NewAgent("assistant", "test-model", agent.WithGuardrails(mask))
	require.NoError(t, err)

	var final agent.FinalEvent
	res, err := runPrepared(t, fake, a, "Weather?", agent.RunOptions{
		OutputSchema: &model.OutputSchema{Name: "weather", Schema: json.RawMessage(`{"type":"object"}`)},
		OnFinal: func(_ workflow.Context, e agent.FinalEvent) (agent.FinalAction, error) {
			final = e
			return agent.FinalAction{}, nil
		},
	})
	require.NoError(t, err)

	assert.JSONEq(t, `{"temp":"[x]"}`, string(res.StructuredOutput))
	assert.JSONEq(t, `{"temp":"[x]"}`, res.Output)
	assert.JSONEq(t, `{"temp":"[x]"}`, final.Assistant.Text())
}

func TestRunWith_OutputModelNoteFollowsTheAnswer(t *testing.T) {
	fake := agenttest.NewFakeProvider(agenttest.Says("Mild."))
	note := mustFunc(t, "data", []guardrail.Stage{guardrail.StageOutput}, func(workflow.Context, guardrail.CheckInput) (guardrail.Verdict, error) {
		return guardrail.Annotate("r", guardrail.Notice{Text: "Checked.", Audience: guardrail.AudienceModel}), nil
	})
	a, err := agent.NewAgent("assistant", "test-model", agent.WithGuardrails(note))
	require.NoError(t, err)

	res, err := runPrepared(t, fake, a, "Weather?", agent.RunOptions{})
	require.NoError(t, err)

	assert.Equal(t, []string{"Weather?", "Mild.", "Checked."}, messageTexts(res.Messages))
	assert.Equal(t, "Mild.", res.Output)
}

func TestRunWith_ContinuationFragmentNoteFollowsTheClosingReply(t *testing.T) {
	fake := agenttest.NewFakeProvider(agenttest.Truncated("Part one, "), agenttest.Says("part two."))
	note := mustFunc(t, "a", []guardrail.Stage{guardrail.StageAssistantText}, func(workflow.Context, guardrail.CheckInput) (guardrail.Verdict, error) {
		return guardrail.Annotate("r", guardrail.Notice{Text: "Note A", Audience: guardrail.AudienceModel}), nil
	})
	a, err := agent.NewAgent("assistant", "test-model",
		agent.WithContinueOnLength(1), agent.WithGuardrails(note))
	require.NoError(t, err)

	res, err := runPrepared(t, fake, a, "Weather?", agent.RunOptions{})
	require.NoError(t, err)

	if assert.Equal(t, 2, fake.CallCount()) {
		assert.Equal(t, []string{"Weather?", "Part one, "}, messageTexts(fake.Calls()[1].Messages),
			"the continuation call continues the fragment")
	}
	assert.Equal(t, []string{"Weather?", "Part one, ", "part two.", "Note A"}, messageTexts(res.Messages))
	assert.Equal(t, "Part one, part two.", res.Output)
}

func TestRunWith_DroppedEmptyReplyTakesItsOutputNotes(t *testing.T) {
	fake := agenttest.NewFakeProvider(agenttest.Says(""), agenttest.Says("done"))
	note := mustFunc(t, "data", []guardrail.Stage{guardrail.StageOutput}, func(workflow.Context, guardrail.CheckInput) (guardrail.Verdict, error) {
		return guardrail.Annotate("r", guardrail.Notice{Text: "Checked.", Audience: guardrail.AudienceModel}), nil
	})
	a, err := agent.NewAgent("assistant", "test-model", agent.WithGuardrails(note))
	require.NoError(t, err)

	res, err := runPrepared(t, fake, a, "Hi", agent.RunOptions{OnFinal: nudgeOnce("nudge")})
	require.NoError(t, err)

	if assert.Equal(t, 2, fake.CallCount()) {
		assert.Equal(t, []string{"Hi", "nudge"}, messageTexts(fake.Calls()[1].Messages))
	}
	assert.Equal(t, []string{"Hi", "nudge", "done", "Checked."}, messageTexts(res.Messages))
}

func TestRunWith_AssistantTextRewriteKeepsTheToolCalls(t *testing.T) {
	fake := agenttest.NewFakeProvider(textAndCalls("Checking Ghent.", "Ghent"), agenttest.Says("Mild."))
	mask := mustFunc(t, "mask", []guardrail.Stage{guardrail.StageAssistantText}, func(_ workflow.Context, in guardrail.CheckInput) (guardrail.Verdict, error) {
		return guardrail.RewriteText(strings.ReplaceAll(in.Text, "Ghent", "[city]"), "city", nil), nil
	})
	var seq, cities []string
	a, err := agent.NewAgent("assistant", "test-model",
		agent.WithTools(cityTool(t, &seq, &cities)), agent.WithGuardrails(mask))
	require.NoError(t, err)

	_, err = runPrepared(t, fake, a, "Weather in Ghent?", agent.RunOptions{})
	require.NoError(t, err)

	assert.Equal(t, []string{"Ghent"}, cities, "the call is unchanged")
	if assert.Equal(t, 2, fake.CallCount()) {
		msgs := fake.Calls()[1].Messages
		if assert.Len(t, msgs, 3) {
			reply := msgs[1]
			assert.Equal(t, "Checking [city].", reply.Text())
			if calls := reply.ToolCalls(); assert.Len(t, calls, 1) {
				assert.JSONEq(t, `{"city":"Ghent"}`, string(calls[0].Arguments))
			}
			assert.Equal(t, model.BlockText, reply.Blocks[0].Kind, "the text keeps its position")
		}
	}
}

func TestRunWith_ToolCallModelNoteFollowsTheResults(t *testing.T) {
	fake := agenttest.NewFakeProvider(agenttest.CallsTool("get_weather", `{"city":"Ghent"}`), agenttest.Says("Mild."))
	note := mustFunc(t, "b", []guardrail.Stage{guardrail.StageToolCall}, func(workflow.Context, guardrail.CheckInput) (guardrail.Verdict, error) {
		return guardrail.Annotate("r", guardrail.Notice{Text: "Note B", Audience: guardrail.AudienceModel}), nil
	})
	a, err := agent.NewAgent("assistant", "test-model",
		agent.WithTools(weatherTool(t)), agent.WithGuardrails(note))
	require.NoError(t, err)

	res, err := runPrepared(t, fake, a, "Weather in Ghent?", agent.RunOptions{})
	require.NoError(t, err)

	if assert.Len(t, res.Messages, 5) {
		assert.Equal(t, "18°C in Ghent", res.Messages[2].ToolResultText())
		assert.Equal(t, model.RoleUser, res.Messages[3].Role)
		assert.Equal(t, "Note B", res.Messages[3].Text())
		assert.Equal(t, "Mild.", res.Messages[4].Text())
	}
}

func TestRunWith_HeldNotesAreKeptWhenTheTurnBudgetRunsOut(t *testing.T) {
	fake := agenttest.NewFakeProvider(agenttest.Truncated("Part one, "))
	note := mustFunc(t, "a", []guardrail.Stage{guardrail.StageAssistantText}, func(workflow.Context, guardrail.CheckInput) (guardrail.Verdict, error) {
		return guardrail.Annotate("r", guardrail.Notice{Text: "Note A", Audience: guardrail.AudienceModel}), nil
	})
	a, err := agent.NewAgent("assistant", "test-model",
		agent.WithMaxTurns(1), agent.WithContinueOnLength(1), agent.WithGuardrails(note))
	require.NoError(t, err)

	res, err := runPrepared(t, fake, a, "Weather?", agent.RunOptions{})

	var appErr *temporal.ApplicationError
	if assert.ErrorAs(t, err, &appErr) {
		assert.Equal(t, agent.ErrorTypeMaxTurns, appErr.Type())
	}
	if assert.NotNil(t, res) {
		assert.Equal(t, []string{"Weather?", "Part one, ", "Note A"}, messageTexts(res.Messages))
	}
}

func TestRunWith_DroppedReplyKeepsTheFragmentNotesHeld(t *testing.T) {
	fake := agenttest.NewFakeProvider(agenttest.Truncated("Part one, "), agenttest.Says(""), agenttest.Says("done"))
	note := mustFunc(t, "a", []guardrail.Stage{guardrail.StageAssistantText}, func(workflow.Context, guardrail.CheckInput) (guardrail.Verdict, error) {
		return guardrail.Annotate("r", guardrail.Notice{Text: "Note A", Audience: guardrail.AudienceModel}), nil
	})
	a, err := agent.NewAgent("assistant", "test-model",
		agent.WithContinueOnLength(1), agent.WithGuardrails(note))
	require.NoError(t, err)

	res, err := runPrepared(t, fake, a, "Weather?", agent.RunOptions{OnFinal: nudgeOnce("nudge")})
	require.NoError(t, err)

	assert.Equal(t, []string{"Weather?", "Part one, ", "nudge", "done", "Note A"}, messageTexts(res.Messages))
}
