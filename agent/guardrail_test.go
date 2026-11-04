package agent_test

import (
	"errors"
	"fmt"
	"strings"
	"testing"
	"time"

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

// allow is a check that lets every piece of content through.
func allow(workflow.Context, guardrail.CheckInput) (guardrail.Verdict, error) {
	return guardrail.Allow(), nil
}

// mustFunc builds a Func guardrail and fails the test when it cannot.
func mustFunc(t *testing.T, name string, stages []guardrail.Stage, fn func(workflow.Context, guardrail.CheckInput) (guardrail.Verdict, error)) guardrail.Guardrail {
	t.Helper()
	g, err := guardrail.Func(name, stages, fn)
	require.NoError(t, err)
	return g
}

// tripOn returns a guardrail on stage that blocks the text want with reason.
func tripOn(t *testing.T, stage guardrail.Stage, name, want, reason string) guardrail.Guardrail {
	t.Helper()
	return mustFunc(t, name, []guardrail.Stage{stage}, func(_ workflow.Context, in guardrail.CheckInput) (guardrail.Verdict, error) {
		if in.Text == want {
			return guardrail.Block(reason, ""), nil
		}
		return guardrail.Allow(), nil
	})
}

func TestTripwireError_RoundTripsThroughAnApplicationError(t *testing.T) {
	te := agent.TripwireError{Stage: guardrail.StageToolCall, Guardrail: "blocklist", Reason: "disallowed request",
		Notice: "Blocked.", Turn: 2, Index: 1, Failed: true}
	got, ok := agent.AsTripwire(agent.NewTripwireError(te))
	if assert.True(t, ok) {
		assert.Equal(t, te, *got)
	}
	assert.True(t, errors.Is(&te, &agent.TripwireError{Stage: guardrail.StageToolCall}))
	assert.False(t, errors.Is(&te, &agent.TripwireError{Stage: guardrail.StageOutput}))

	var appErr *temporal.ApplicationError
	if assert.ErrorAs(t, agent.NewTripwireError(te), &appErr) {
		assert.Equal(t, agent.ErrorTypeTripwire, appErr.Type())
		assert.True(t, appErr.NonRetryable())
		assert.Equal(t, `tool_call guardrail "blocklist" tripped: disallowed request`, appErr.Message())
	}
}

func TestNewAgent_RefusesAToolResultGuardrail(t *testing.T) {
	g := mustFunc(t, "mask", []guardrail.Stage{guardrail.StageToolResult}, allow)
	_, err := agent.NewAgent("assistant", "test-model", agent.WithGuardrails(g))
	requireConfigError(t, err, "guardrail.Screen")

	fake := agenttest.NewFakeProvider(agenttest.Says("unused"))
	a, err := agent.NewAgent("assistant", "test-model")
	require.NoError(t, err)
	_, err = runPrepared(t, fake, a, "Weather in Ghent?", agent.RunOptions{Guardrails: []guardrail.Guardrail{g}})
	assertConfigError(t, err, fake, "guardrail.Screen")
}

func TestNewAgent_RefusesOutputChecksWithContinuation(t *testing.T) {
	out := mustFunc(t, "no-leaks", []guardrail.Stage{guardrail.StageOutput}, allow)
	_, err := agent.NewAgent("assistant", "test-model",
		agent.WithContinueOnLength(1), agent.WithGuardrails(out))
	requireConfigError(t, err, "output guardrails cannot be combined with WithContinueOnLength")

	fake := agenttest.NewFakeProvider(agenttest.Says("unused"))
	a, err := agent.NewAgent("assistant", "test-model", agent.WithContinueOnLength(1))
	require.NoError(t, err)
	_, err = runPrepared(t, fake, a, "Weather in Ghent?", agent.RunOptions{Guardrails: []guardrail.Guardrail{out}})
	assertConfigError(t, err, fake, "WithContinueOnLength")
	if assert.Error(t, err) {
		assert.Contains(t, err.Error(), "output")
	}

	text := mustFunc(t, "a", []guardrail.Stage{guardrail.StageAssistantText}, allow)
	_, err = agent.NewAgent("assistant", "test-model",
		agent.WithContinueOnLength(1), agent.WithGuardrails(text))
	assert.NoError(t, err)
}

func TestRunWith_ToolTripwireEndsRunAfterSiblings(t *testing.T) {
	fake := agenttest.NewFakeProvider(
		agenttest.CallsTools(
			agenttest.ToolCall{ID: "x", Name: "get_weather", Args: `{"city":"Ghent"}`},
			agenttest.ToolCall{ID: "y", Name: "get_weather", Args: `{"city":"Bruges"}`},
		),
		agenttest.Says("unused"),
	)
	var ran []string
	weather, err := tool.New("get_weather", "Look up the weather",
		func(_ workflow.Context, in weatherIn) (string, error) {
			if in.City == "Bruges" {
				return "", agent.NewTripwireError(agent.TripwireError{
					Stage: guardrail.StageToolResult, Guardrail: "mask", Index: 1})
			}
			ran = append(ran, in.City)
			return fmt.Sprintf("18°C in %s", in.City), nil
		})
	require.NoError(t, err)
	a, err := agent.NewAgent("assistant", "test-model", agent.WithTools(weather))
	require.NoError(t, err)

	turns := 0
	_, err = runPrepared(t, fake, a, "Weather in Ghent and Bruges?", agent.RunOptions{
		OnTurn: func(workflow.Context, agent.TurnEvent) error {
			turns++
			return nil
		},
	})

	te, ok := agent.AsTripwire(err)
	if assert.True(t, ok, "got %v", err) {
		assert.Equal(t, "mask", te.Guardrail)
		assert.Equal(t, 1, te.Index)
		assert.Equal(t, guardrail.StageToolResult, te.Stage)
	}
	assert.Equal(t, []string{"Ghent"}, ran, "the sibling call ran")
	assert.Equal(t, 0, turns, "OnTurn does not fire")
	assert.Equal(t, 1, fake.CallCount())
}

func TestRunWith_ToolTripwireWaitsForASlowerSibling(t *testing.T) {
	fake := agenttest.NewFakeProvider(
		agenttest.CallsTools(
			agenttest.ToolCall{ID: "x", Name: "get_weather", Args: `{"city":"Ghent"}`},
			agenttest.ToolCall{ID: "y", Name: "get_weather", Args: `{"city":"Bruges"}`},
		),
		agenttest.Says("unused"),
	)
	var finished []string
	weather, err := tool.New("get_weather", "Look up the weather",
		func(ctx workflow.Context, in weatherIn) (string, error) {
			if in.City == "Ghent" {
				return "", agent.NewTripwireError(agent.TripwireError{
					Stage: guardrail.StageToolResult, Guardrail: "mask", Index: 0})
			}
			if err := workflow.Sleep(ctx, time.Minute); err != nil {
				return "", err
			}
			finished = append(finished, in.City)
			return fmt.Sprintf("18°C in %s", in.City), nil
		})
	require.NoError(t, err)
	a, err := agent.NewAgent("assistant", "test-model", agent.WithTools(weather))
	require.NoError(t, err)

	_, err = runPrepared(t, fake, a, "Weather in Ghent and Bruges?", agent.RunOptions{})

	te, ok := agent.AsTripwire(err)
	if assert.True(t, ok, "got %v", err) {
		assert.Equal(t, "mask", te.Guardrail)
		assert.Equal(t, 0, te.Index)
	}
	assert.Equal(t, []string{"Bruges"}, finished, "the run waited for the slower sibling")
	assert.Equal(t, 1, fake.CallCount())
}

func TestRunWith_SubAgentTripwireAbortsTheParentRun(t *testing.T) {
	fake := agenttest.NewFakeProvider(
		agenttest.CallsTool("research", `{"input":"Weather in Ghent?"}`),
		agenttest.CallsTool("get_weather", `{"city":"Ghent"}`),
		agenttest.Says("unused"),
	)
	want := agent.TripwireError{Stage: guardrail.StageToolResult, Guardrail: "strict-gate", Reason: "disallowed request",
		Notice: "Blocked.", Turn: 1, Index: 0}
	gated, err := tool.New("get_weather", "Look up the weather",
		func(workflow.Context, weatherIn) (string, error) {
			return "", agent.NewTripwireError(want)
		})
	require.NoError(t, err)
	researcher, err := agent.NewAgent("researcher", "child-model", agent.WithTools(gated))
	require.NoError(t, err)
	reg := agent.NewRegistry()
	require.NoError(t, reg.Add(researcher))
	sub, err := agent.AsSubAgent(researcher, "research", "Research a topic")
	require.NoError(t, err)
	parent, err := agent.NewAgent("assistant", "test-model", agent.WithTools(sub))
	require.NoError(t, err)

	env := newSubAgentEnv(t, fake, reg)
	var runErr error
	env.ExecuteWorkflow(func(ctx workflow.Context) error {
		_, runErr = agent.Run(ctx, parent, "Weather in Ghent?")
		return runErr
	})
	require.True(t, env.IsWorkflowCompleted())

	te, ok := agent.AsTripwire(runErr)
	if assert.True(t, ok, "got %v", runErr) {
		assert.Equal(t, want, *te)
	}
	parentCalls := 0
	for _, c := range fake.Calls() {
		if c.Model == "test-model" {
			parentCalls++
		}
	}
	assert.Equal(t, 1, parentCalls, "the parent model is not called after the tripwire")
}

// guardrailEvents returns an OnGuardrail hook that appends each event to events.
func guardrailEvents(events *[]agent.GuardrailEvent) func(workflow.Context, agent.GuardrailEvent) error {
	return func(_ workflow.Context, e agent.GuardrailEvent) error {
		*events = append(*events, e)
		return nil
	}
}

func TestRunWith_InputStageScreensTheInputString(t *testing.T) {
	fake := agenttest.NewFakeProvider(agenttest.Says("should never be reached"))
	a, err := agent.NewAgent("assistant", "test-model",
		agent.WithGuardrails(tripOn(t, guardrail.StageInput, "blocklist", "do something bad", "disallowed request")))
	require.NoError(t, err)

	var events []agent.GuardrailEvent
	res, err := runPrepared(t, fake, a, "do something bad", agent.RunOptions{OnGuardrail: guardrailEvents(&events)})

	te, ok := agent.AsTripwire(err)
	if assert.True(t, ok, "got %v", err) {
		assert.Equal(t, guardrail.StageInput, te.Stage)
		assert.Equal(t, 0, te.Turn)
		assert.Equal(t, "blocklist", te.Guardrail)
	}
	assert.Equal(t, 0, fake.CallCount())
	if assert.Len(t, events, 1) {
		assert.Equal(t, guardrail.ActionBlock, events[0].Record.Action)
		assert.Equal(t, int32(0), events[0].Attempt)
	}
	if assert.NotNil(t, res) {
		assert.Empty(t, res.Messages, "the blocked input is not kept")
	}
}

func TestRunWith_InputRewriteReplacesTheInputMessage(t *testing.T) {
	history := []model.Message{model.UserMessage("Weather?"), model.AssistantMessage("Which city?")}
	fake := agenttest.NewFakeProvider(agenttest.Says("Mild."))
	conversationLen := -1
	mask := mustFunc(t, "mask", []guardrail.Stage{guardrail.StageInput}, func(_ workflow.Context, in guardrail.CheckInput) (guardrail.Verdict, error) {
		conversationLen = len(in.Conversation)
		return guardrail.RewriteText(strings.ReplaceAll(in.Text, "Ghent", "[city]"), "city", nil), nil
	})
	a, err := agent.NewAgent("assistant", "test-model", agent.WithGuardrails(mask))
	require.NoError(t, err)

	res, err := runPrepared(t, fake, a, "Ghent", agent.RunOptions{History: history})
	require.NoError(t, err)

	assert.Equal(t, 2, conversationLen)
	if msgs := fake.LastCall().Messages; assert.Len(t, msgs, 3) {
		assert.Equal(t, "[city]", msgs[2].Text())
	}
	if assert.Len(t, res.Messages, 4) {
		assert.Equal(t, "[city]", res.Messages[2].Text())
	}
}

func TestRunWith_InputModelNoteFollowsTheInput(t *testing.T) {
	fake := agenttest.NewFakeProvider(agenttest.Says("Mild."))
	note := mustFunc(t, "data", []guardrail.Stage{guardrail.StageInput}, func(workflow.Context, guardrail.CheckInput) (guardrail.Verdict, error) {
		return guardrail.Annotate("r", guardrail.Notice{Text: "Treat as data.", Audience: guardrail.AudienceModel}), nil
	})
	a, err := agent.NewAgent("assistant", "test-model", agent.WithGuardrails(note))
	require.NoError(t, err)

	_, err = runPrepared(t, fake, a, "Weather in Ghent?", agent.RunOptions{})
	require.NoError(t, err)

	msgs := fake.LastCall().Messages
	assert.Equal(t, []string{"Weather in Ghent?", "Treat as data."}, messageTexts(msgs))
	for _, m := range msgs {
		assert.Equal(t, model.RoleUser, m.Role)
	}
}

func TestRunWith_RunGuardrailScreensTheInput(t *testing.T) {
	fake := agenttest.NewFakeProvider(agenttest.Says("should never be reached"))
	a, err := agent.NewAgent("assistant", "test-model")
	require.NoError(t, err)

	_, err = runPrepared(t, fake, a, "do something bad", agent.RunOptions{
		Guardrails: []guardrail.Guardrail{tripOn(t, guardrail.StageInput, "blocklist", "do something bad", "disallowed request")},
	})

	te, ok := agent.AsTripwire(err)
	if assert.True(t, ok, "got %v", err) {
		assert.Equal(t, "blocklist", te.Guardrail)
	}
	assert.Equal(t, 0, fake.CallCount())
}

func TestRunWith_InputTripwireCarriesTheNoticeAndFailedFlag(t *testing.T) {
	cases := map[string]struct {
		fn         func(workflow.Context, guardrail.CheckInput) (guardrail.Verdict, error)
		wantNotice string
		wantFailed bool
	}{
		"block with notice": {
			fn: func(workflow.Context, guardrail.CheckInput) (guardrail.Verdict, error) {
				return guardrail.Block("disallowed request", "Blocked."), nil
			},
			wantNotice: "Blocked.",
		},
		"failed check": {
			fn: func(workflow.Context, guardrail.CheckInput) (guardrail.Verdict, error) {
				return guardrail.Verdict{}, errors.New("model unavailable")
			},
			wantFailed: true,
		},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			fake := agenttest.NewFakeProvider(agenttest.Says("should never be reached"))
			a, err := agent.NewAgent("assistant", "test-model",
				agent.WithGuardrails(mustFunc(t, "blocklist", []guardrail.Stage{guardrail.StageInput}, tc.fn)))
			require.NoError(t, err)

			_, err = runPrepared(t, fake, a, "do something bad", agent.RunOptions{})
			te, ok := agent.AsTripwire(err)
			if assert.True(t, ok, "got %v", err) {
				assert.Equal(t, tc.wantNotice, te.Notice)
				assert.Equal(t, tc.wantFailed, te.Failed)
			}
			assert.Equal(t, 0, fake.CallCount())
		})
	}
}

func TestRunWith_NoInputSkipsTheInputStage(t *testing.T) {
	fake := agenttest.NewFakeProvider(agenttest.Says("Hello there."))
	a, err := agent.NewAgent("assistant", "test-model",
		agent.WithGuardrails(tripOn(t, guardrail.StageInput, "blocklist", "do something bad", "disallowed request")))
	require.NoError(t, err)

	var events []agent.GuardrailEvent
	res, err := runPrepared(t, fake, a, "", agent.RunOptions{
		History:     []model.Message{model.UserMessage("do something bad")},
		OnGuardrail: guardrailEvents(&events),
	})
	require.NoError(t, err)
	assert.Equal(t, "Hello there.", res.Output)
	assert.Empty(t, events)
}

func TestRunWith_OnGuardrailErrorAbortsTheRun(t *testing.T) {
	fake := agenttest.NewFakeProvider(agenttest.Says("should never be reached"))
	note := mustFunc(t, "note", []guardrail.Stage{guardrail.StageInput}, func(workflow.Context, guardrail.CheckInput) (guardrail.Verdict, error) {
		return guardrail.Annotate("r", guardrail.Notice{Text: "Checked.", Audience: guardrail.AudienceUser}), nil
	})
	a, err := agent.NewAgent("assistant", "test-model", agent.WithGuardrails(note))
	require.NoError(t, err)

	res, err := runPrepared(t, fake, a, "Weather in Ghent?", agent.RunOptions{
		OnGuardrail: func(workflow.Context, agent.GuardrailEvent) error { return errors.New("save failed") },
	})
	if assert.Error(t, err) {
		assert.Contains(t, err.Error(), "save failed")
	}
	assert.Equal(t, 0, fake.CallCount())
	if assert.NotNil(t, res) {
		assert.Empty(t, res.Messages, "the input is not kept when OnGuardrail fails")
	}
}

func TestRunWith_OnGuardrailReportsEachRecordInDeclaredOrder(t *testing.T) {
	fake := agenttest.NewFakeProvider(agenttest.Says("Mild."))
	annotate := func(name string) guardrail.Guardrail {
		return mustFunc(t, name, []guardrail.Stage{guardrail.StageInput}, func(workflow.Context, guardrail.CheckInput) (guardrail.Verdict, error) {
			return guardrail.Annotate("r", guardrail.Notice{Text: "Checked by " + name + ".", Audience: guardrail.AudienceUser}), nil
		})
	}
	a, err := agent.NewAgent("assistant", "test-model",
		agent.WithGuardrails(annotate("first"), annotate("second")))
	require.NoError(t, err)

	var events []agent.GuardrailEvent
	_, err = runPrepared(t, fake, a, "Weather in Ghent?", agent.RunOptions{OnGuardrail: guardrailEvents(&events)})
	require.NoError(t, err)

	if assert.Len(t, events, 2) {
		for i, name := range []string{"first", "second"} {
			assert.Equal(t, guardrail.StageInput, events[i].Record.Stage)
			assert.Equal(t, name, events[i].Record.Guardrail)
			assert.Equal(t, guardrail.ActionAnnotate, events[i].Record.Action)
		}
	}
}

func TestRunWith_ScreensAgentGuardrailsThenRunGuardrailsDeclaringTheStage(t *testing.T) {
	fake := agenttest.NewFakeProvider(agenttest.Says("Mild."))
	annotate := func(name string, stages ...guardrail.Stage) guardrail.Guardrail {
		return mustFunc(t, name, stages, func(workflow.Context, guardrail.CheckInput) (guardrail.Verdict, error) {
			return guardrail.Annotate("r", guardrail.Notice{Text: name, Audience: guardrail.AudienceUser}), nil
		})
	}
	a, err := agent.NewAgent("assistant", "test-model", agent.WithGuardrails(
		annotate("agent-output", guardrail.StageOutput),
		annotate("agent-both", guardrail.StageInput, guardrail.StageOutput)))
	require.NoError(t, err)

	var events []agent.GuardrailEvent
	_, err = runPrepared(t, fake, a, "Weather in Ghent?", agent.RunOptions{
		Guardrails: []guardrail.Guardrail{
			annotate("run-input", guardrail.StageInput),
			annotate("run-both", guardrail.StageInput, guardrail.StageOutput),
		},
		OnGuardrail: guardrailEvents(&events),
	})
	require.NoError(t, err)

	got := make([]string, 0, len(events))
	for _, e := range events {
		got = append(got, string(e.Record.Stage)+":"+e.Record.Guardrail)
	}
	assert.Equal(t, []string{
		"input:agent-both", "input:run-input", "input:run-both",
		"output:agent-output", "output:agent-both", "output:run-both",
	}, got)
}

func TestRunWith_NilRunGuardrailIsAConfigError(t *testing.T) {
	fake := agenttest.NewFakeProvider(agenttest.Says("unused"))
	a, err := agent.NewAgent("assistant", "test-model")
	require.NoError(t, err)

	_, err = runPrepared(t, fake, a, "Weather in Ghent?", agent.RunOptions{Guardrails: []guardrail.Guardrail{nil}})
	assertConfigError(t, err, fake, "run guardrail 0 is nil")
}

// An input tripwire blocks the run before the first model call.
func TestGuardrail_InputTripwireBlocksBeforeModelSpend(t *testing.T) {
	fake := agenttest.NewFakeProvider(agenttest.Says("should never be reached"))
	env := newEnv(t, fake)

	a, err := agent.NewAgent("assistant", "test-model",
		agent.WithGuardrails(tripOn(t, guardrail.StageInput, "blocklist", "do something bad", "disallowed request")))
	require.NoError(t, err)

	var res *agent.Result
	var runErr error
	env.ExecuteWorkflow(func(ctx workflow.Context) error {
		res, runErr = agent.Run(ctx, a, "do something bad")
		return runErr
	})
	require.True(t, env.IsWorkflowCompleted())

	te, ok := agent.AsTripwire(runErr)
	if assert.True(t, ok, "a blocked input must surface as a tripwire error") {
		assert.Equal(t, guardrail.StageInput, te.Stage)
		assert.Equal(t, "blocklist", te.Guardrail)
		assert.Equal(t, "disallowed request", te.Reason)
	}
	assert.True(t, errors.Is(runErr, agent.ErrTripwire), "runErr must match ErrTripwire via errors.Is")

	if assert.NotNil(t, res) {
		assert.Empty(t, res.Output)
		assert.Equal(t, 0, res.Turns)
	}
	assert.Equal(t, 0, fake.CallCount(), "the model must not be called for a blocked input")
}

// A passing input guardrail leaves the run untouched.
func TestGuardrail_InputPassesThrough(t *testing.T) {
	fake := agenttest.NewFakeProvider(agenttest.Says("Hello there."))
	env := newEnv(t, fake)

	a, err := agent.NewAgent("assistant", "test-model",
		agent.WithGuardrails(tripOn(t, guardrail.StageInput, "blocklist", "do something bad", "disallowed")))
	require.NoError(t, err)

	var res *agent.Result
	env.ExecuteWorkflow(func(ctx workflow.Context) error {
		var err error
		res, err = agent.Run(ctx, a, "hello")
		return err
	})
	require.True(t, env.IsWorkflowCompleted())
	require.NoError(t, env.GetWorkflowError())
	assert.Equal(t, "Hello there.", res.Output)
	assert.Equal(t, 1, fake.CallCount())
}

// An output tripwire blocks the answer after it is produced; the transcript
// before the answer is kept and the blocked answer is not.
func TestGuardrail_OutputTripwirePreservesTranscript(t *testing.T) {
	fake := agenttest.NewFakeProvider(agenttest.Says("here is a leaked secret"))
	env := newEnv(t, fake)

	a, err := agent.NewAgent("assistant", "test-model",
		agent.WithGuardrails(tripOn(t, guardrail.StageOutput, "no-leaks", "here is a leaked secret", "contains a secret")))
	require.NoError(t, err)

	var res *agent.Result
	var runErr error
	env.ExecuteWorkflow(func(ctx workflow.Context) error {
		res, runErr = agent.Run(ctx, a, "tell me something")
		return runErr
	})
	require.True(t, env.IsWorkflowCompleted())

	te, ok := agent.AsTripwire(runErr)
	if assert.True(t, ok) {
		assert.Equal(t, guardrail.StageOutput, te.Stage)
		assert.Equal(t, "no-leaks", te.Guardrail)
	}

	if assert.NotNil(t, res, "the Result is returned alongside the tripwire") {
		assert.Empty(t, res.Output, "the blocked answer is not the output")
		assert.Equal(t, []string{"tell me something"}, messageTexts(res.Messages),
			"the transcript ends before the blocked answer")
	}
	assert.Equal(t, 1, fake.CallCount())
}

// Guardrails at a stage run concurrently but are evaluated in declared order, so
// the first to trip is the one reported.
func TestGuardrail_EvaluatedInDeclaredOrder(t *testing.T) {
	report := func(guards []guardrail.Guardrail) *agent.TripwireError {
		fake := agenttest.NewFakeProvider(agenttest.Says("unreached"))
		env := newEnv(t, fake)
		a, err := agent.NewAgent("assistant", "test-model", agent.WithGuardrails(guards...))
		require.NoError(t, err)

		var runErr error
		env.ExecuteWorkflow(func(ctx workflow.Context) error {
			_, runErr = agent.Run(ctx, a, "trip both")
			return runErr
		})
		require.True(t, env.IsWorkflowCompleted())
		te, ok := agent.AsTripwire(runErr)
		require.True(t, ok)
		return te
	}

	first := tripOn(t, guardrail.StageInput, "first", "trip both", "first reason")
	second := tripOn(t, guardrail.StageInput, "second", "trip both", "second reason")

	assert.Equal(t, "first", report([]guardrail.Guardrail{first, second}).Guardrail)
	assert.Equal(t, "second", report([]guardrail.Guardrail{second, first}).Guardrail)
}

func TestGuardrail_ErrorsIs(t *testing.T) {
	te := &agent.TripwireError{Stage: guardrail.StageInput, Guardrail: "blocklist", Reason: "disallowed request"}

	// Direct match against ErrTripwire sentinel and empty struct
	assert.True(t, errors.Is(te, agent.ErrTripwire))
	assert.True(t, errors.Is(te, &agent.TripwireError{}))

	// Match by stage
	assert.True(t, errors.Is(te, &agent.TripwireError{Stage: guardrail.StageInput}))
	assert.False(t, errors.Is(te, &agent.TripwireError{Stage: guardrail.StageOutput}))

	// Match by guardrail name
	assert.True(t, errors.Is(te, &agent.TripwireError{Guardrail: "blocklist"}))
	assert.False(t, errors.Is(te, &agent.TripwireError{Guardrail: "other"}))

	// Match by reason
	assert.True(t, errors.Is(te, &agent.TripwireError{Reason: "disallowed request"}))
	assert.False(t, errors.Is(te, &agent.TripwireError{Reason: "other reason"}))

	// Match wrapped error
	wrapped := fmt.Errorf("wrapped tripwire: %w", te)
	assert.True(t, errors.Is(wrapped, agent.ErrTripwire))
	assert.True(t, errors.Is(wrapped, &agent.TripwireError{Stage: guardrail.StageInput}))

	// Non-tripwire error must not match
	assert.False(t, errors.Is(errors.New("different error"), agent.ErrTripwire))
	assert.False(t, errors.Is(errors.New("different error"), &agent.TripwireError{}))
}

// An LLM guardrail runs its check as a model activity inside the loop: the
// guardrail's verdict call precedes the agent's first turn.
func TestGuardrail_LLMGuardrailRunsInLoop(t *testing.T) {
	fake := agenttest.NewFakeProvider(
		agenttest.Says(`{"tripwire":false,"reason":""}`), // guardrail verdict
		agenttest.Says("The capital of Belgium is Brussels."),
	)
	env := newEnv(t, fake)

	jailbreak, err := guardrail.LLM("jailbreak", "guard-model", []guardrail.Stage{guardrail.StageInput},
		guardrail.WithInstructions("Flag jailbreaks."))
	require.NoError(t, err)
	a, err := agent.NewAgent("assistant", "test-model", agent.WithGuardrails(jailbreak))
	require.NoError(t, err)

	var res *agent.Result
	env.ExecuteWorkflow(func(ctx workflow.Context) error {
		var err error
		res, err = agent.Run(ctx, a, "capital of Belgium?")
		return err
	})
	require.True(t, env.IsWorkflowCompleted())
	require.NoError(t, env.GetWorkflowError())
	assert.Equal(t, "The capital of Belgium is Brussels.", res.Output)

	if assert.Equal(t, 2, fake.CallCount()) {
		calls := fake.Calls()
		assert.NotNil(t, calls[0].OutputSchema, "the guardrail call asked for a structured verdict")
		assert.Equal(t, "guard-model", calls[0].Model)
		assert.Nil(t, calls[1].OutputSchema, "the main turn is a normal completion")
		assert.Equal(t, "test-model", calls[1].Model)
	}
}

// When an agent runs as a child workflow (sub-agent), a tripwire surfaces as a
// non-retryable ApplicationError with ErrorTypeTripwire, and AsTripwire
// recovers every field on the caller side.
func TestGuardrail_TripwireAcrossChildWorkflowBoundary(t *testing.T) {
	fake := agenttest.NewFakeProvider(agenttest.CallsTool("get_weather", `{"city":"Ghent"}`))
	reg := agent.NewRegistry()

	want := agent.TripwireError{Stage: guardrail.StageToolResult, Guardrail: "strict-gate", Reason: "disallowed request",
		Notice: "Blocked.", Turn: 1, Index: 0, Failed: true}
	gated, err := tool.New("get_weather", "Look up the weather",
		func(workflow.Context, weatherIn) (string, error) {
			return "", agent.NewTripwireError(want)
		})
	require.NoError(t, err)
	sub, err := agent.NewAgent("checker", "test-model", agent.WithTools(gated))
	require.NoError(t, err)
	require.NoError(t, reg.Add(sub))

	env := newSubAgentEnv(t, fake, reg)

	env.ExecuteWorkflow(func(ctx workflow.Context) error {
		var res agent.Result
		cctx := workflow.WithChildOptions(ctx, workflow.ChildWorkflowOptions{
			RetryPolicy: &temporal.RetryPolicy{MaximumAttempts: 3},
		})
		return workflow.ExecuteChildWorkflow(cctx, agent.WorkflowName, agent.WorkflowInput{
			Agent: "checker",
			Input: "Weather in Ghent?",
		}).Get(ctx, &res)
	})

	require.True(t, env.IsWorkflowCompleted())
	err = env.GetWorkflowError()
	require.Error(t, err)

	te, ok := agent.AsTripwire(err)
	if assert.True(t, ok, "tripwire in child workflow must be recoverable with AsTripwire across workflow boundary") {
		assert.Equal(t, want, *te)
	}

	var appErr *temporal.ApplicationError
	if assert.True(t, errors.As(err, &appErr), "error across child workflow boundary must be an ApplicationError") {
		assert.Equal(t, agent.ErrorTypeTripwire, appErr.Type(), "tripwire error must have type ErrorTypeTripwire")
		assert.True(t, appErr.NonRetryable(), "tripwire error must be non-retryable so RetryPolicy does not retry it")
	}
	assert.Equal(t, 1, fake.CallCount(), "the child is not retried")
}
