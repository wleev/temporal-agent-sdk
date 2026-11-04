package agent_test

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.temporal.io/sdk/workflow"

	"github.com/wleev/temporal-agent-sdk/agent"
	"github.com/wleev/temporal-agent-sdk/agenttest"
	"github.com/wleev/temporal-agent-sdk/guardrail"
	"github.com/wleev/temporal-agent-sdk/model"
	"github.com/wleev/temporal-agent-sdk/tool"
)

// submitTool returns a dynamic tool named name, built with tool.EndsRun, whose
// result is model.FinalResult(model.TextResult("ok"), output).
func submitTool(t *testing.T, name, output string) tool.Tool {
	t.Helper()
	def := model.NewTool(name, "Submit the answer", []byte(`{"type":"object"}`))
	tl, err := tool.Dynamic(def, func(workflow.Context, string, json.RawMessage) (*model.CallToolResult, error) {
		return model.FinalResult(model.TextResult("ok"), output), nil
	}, tool.EndsRun())
	require.NoError(t, err)
	return tl
}

// submitFake scripts one reply that calls submit_a and submit_b.
func submitFake() *agenttest.FakeProvider {
	return agenttest.NewFakeProvider(agenttest.CallsTools(
		agenttest.ToolCall{Name: "submit_a", Args: `{}`},
		agenttest.ToolCall{Name: "submit_b", Args: `{}`},
	))
}

// hasFinalOutput reports whether a tool result in msgs carries a final output.
func hasFinalOutput(msgs []model.Message) bool {
	for _, m := range msgs {
		for _, b := range m.Blocks {
			if _, ok := model.FinalOutput(b.Result); ok {
				return true
			}
		}
	}
	return false
}

func TestFinalResult_FirstInCallOrderEndsTheRun(t *testing.T) {
	fake := submitFake()
	a, err := agent.NewAgent("assistant", "test-model",
		agent.WithTools(submitTool(t, "submit_a", "submit_a answer"), submitTool(t, "submit_b", "submit_b answer")))
	require.NoError(t, err)

	turns, finals := 0, 0
	res, err := runPrepared(t, fake, a, "Answer.", agent.RunOptions{
		OnTurn: func(workflow.Context, agent.TurnEvent) error {
			turns++
			return nil
		},
		OnFinal: func(workflow.Context, agent.FinalEvent) (agent.FinalAction, error) {
			finals++
			return agent.FinalAction{}, nil
		},
	})
	assert.NoError(t, err)

	assert.Equal(t, "submit_a answer", res.Output)
	assert.Equal(t, model.FinishStop, res.FinishReason)
	assert.Nil(t, res.StructuredOutput)
	assert.Equal(t, 1, fake.CallCount())
	if assert.NotEmpty(t, res.Messages) {
		last := res.Messages[len(res.Messages)-1]
		assert.True(t, last.IsAssistant())
		assert.Equal(t, "submit_a answer", last.Text())
	}
	assert.Equal(t, []string{"ok", "ok"}, toolResultTexts(res.Messages))
	assert.False(t, hasFinalOutput(res.Messages), "the final-output key is stripped")
	assert.Equal(t, 1, turns)
	assert.Equal(t, 0, finals)
}

func TestFinalResult_OutputStageScreensTheHostText(t *testing.T) {
	mask := mustFunc(t, "mask", []guardrail.Stage{guardrail.StageOutput}, func(_ workflow.Context, in guardrail.CheckInput) (guardrail.Verdict, error) {
		return guardrail.RewriteText(strings.ReplaceAll(in.Text, "Ghent", "[x]"), "mask", nil), nil
	})
	a, err := agent.NewAgent("assistant", "test-model",
		agent.WithTools(submitTool(t, "submit_a", "18°C in Ghent"), submitTool(t, "submit_b", "unused")),
		agent.WithGuardrails(mask))
	require.NoError(t, err)

	var events []agent.GuardrailEvent
	res, err := runPrepared(t, submitFake(), a, "Weather?", agent.RunOptions{OnGuardrail: guardrailEvents(&events)})
	require.NoError(t, err)

	assert.Equal(t, "18°C in [x]", res.Output)
	if assert.NotEmpty(t, res.Messages) {
		assert.Equal(t, "18°C in [x]", res.Messages[len(res.Messages)-1].Text())
	}
	if assert.Len(t, events, 1) {
		assert.Equal(t, int32(0), events[0].Attempt)
		assert.Equal(t, guardrail.StageOutput, events[0].Record.Stage)
		assert.Equal(t, 1, events[0].Record.Turn)
	}

	block := tripOn(t, guardrail.StageOutput, "no-city", "18°C in Ghent", "names a city")
	a, err = agent.NewAgent("assistant", "test-model",
		agent.WithTools(submitTool(t, "submit_a", "18°C in Ghent"), submitTool(t, "submit_b", "unused")),
		agent.WithGuardrails(block))
	require.NoError(t, err)

	fake := submitFake()
	events = nil
	res, err = runPrepared(t, fake, a, "Weather?", agent.RunOptions{OnGuardrail: guardrailEvents(&events)})
	te, ok := agent.AsTripwire(err)
	if assert.True(t, ok, "got %v", err) {
		assert.Equal(t, guardrail.StageOutput, te.Stage)
	}
	if assert.NotNil(t, res) {
		assert.Equal(t, "", res.Output)
		if assert.NotEmpty(t, res.Messages) {
			assert.False(t, res.Messages[len(res.Messages)-1].IsAssistant(), "a blocked output is not appended")
		}
	}
	assert.Equal(t, 1, fake.CallCount())
	if assert.Len(t, events, 1) {
		assert.Equal(t, guardrail.ActionBlock, events[0].Record.Action)
	}
}

func TestFinalResult_OutputModelNoteFollowsTheAnswer(t *testing.T) {
	note := mustFunc(t, "data", []guardrail.Stage{guardrail.StageOutput}, func(_ workflow.Context, in guardrail.CheckInput) (guardrail.Verdict, error) {
		return guardrail.Annotate("r", guardrail.Notice{Text: "Checked.", Audience: guardrail.AudienceModel}), nil
	})
	var conversation []string
	look := mustFunc(t, "look", []guardrail.Stage{guardrail.StageOutput}, func(_ workflow.Context, in guardrail.CheckInput) (guardrail.Verdict, error) {
		conversation = toolResultTexts(in.Conversation)
		return guardrail.Allow(), nil
	})
	a, err := agent.NewAgent("assistant", "test-model",
		agent.WithTools(submitTool(t, "submit_a", "The answer."), submitTool(t, "submit_b", "unused")),
		agent.WithGuardrails(note, look))
	require.NoError(t, err)

	res, err := runPrepared(t, submitFake(), a, "Answer.", agent.RunOptions{})
	require.NoError(t, err)

	assert.Equal(t, []string{"ok", "ok"}, conversation, "the output stage sees the round's results")
	if n := len(res.Messages); assert.GreaterOrEqual(t, n, 2) {
		assert.Equal(t, "The answer.", res.Messages[n-2].Text())
		assert.True(t, res.Messages[n-2].IsAssistant())
		assert.Equal(t, "Checked.", res.Messages[n-1].Text())
		assert.Equal(t, model.RoleUser, res.Messages[n-1].Role)
	}
	assert.Equal(t, "The answer.", res.Output)
}

func TestFinalResult_OnRoundReplacementCanEndTheRun(t *testing.T) {
	fake := agenttest.NewFakeProvider(agenttest.CallsTool("get_weather", `{"city":"Ghent"}`))
	weather, err := tool.New("get_weather", "Look up the weather",
		func(_ workflow.Context, in weatherIn) (string, error) { return "18°C in " + in.City, nil }, tool.EndsRun())
	require.NoError(t, err)
	a, err := agent.NewAgent("assistant", "test-model", agent.WithTools(weather))
	require.NoError(t, err)

	res, err := runPrepared(t, fake, a, "Weather in Ghent?", agent.RunOptions{
		OnRound: func(_ workflow.Context, e agent.RoundEvent) (agent.RoundOutcome, error) {
			return agent.RoundOutcome{Results: []*model.CallToolResult{model.FinalResult(e.Results[0], "It is 18°C in Ghent.")}}, nil
		},
	})
	require.NoError(t, err)

	assert.Equal(t, "It is 18°C in Ghent.", res.Output)
	assert.Equal(t, 1, fake.CallCount())
	assert.Equal(t, []string{"18°C in Ghent"}, toolResultTexts(res.Messages))
	assert.False(t, hasFinalOutput(res.Messages))
}

func TestFinalResult_OnRoundReplacingTheFinalContinuesTheRun(t *testing.T) {
	fake := agenttest.NewFakeProvider(agenttest.CallsTool("submit_a", `{}`), agenttest.Says("done"))
	a, err := agent.NewAgent("assistant", "test-model", agent.WithTools(submitTool(t, "submit_a", "submit_a answer")))
	require.NoError(t, err)

	res, err := runPrepared(t, fake, a, "Answer.", agent.RunOptions{
		OnRound: func(workflow.Context, agent.RoundEvent) (agent.RoundOutcome, error) {
			return agent.RoundOutcome{Results: []*model.CallToolResult{model.TextResult("plain")}}, nil
		},
	})
	require.NoError(t, err)

	assert.Equal(t, 2, fake.CallCount())
	assert.Equal(t, "done", res.Output)
	assert.Equal(t, []string{"plain"}, toolResultTexts(res.Messages))
}

func TestFinalResult_FinalCallWaitsForASlowerSibling(t *testing.T) {
	fake := agenttest.NewFakeProvider(agenttest.CallsTools(
		agenttest.ToolCall{Name: "submit_a", Args: `{}`},
		agenttest.ToolCall{Name: "slow", Args: `{"text":"b"}`},
	))
	slow, _ := goldenTools(t)
	a, err := agent.NewAgent("assistant", "test-model", agent.WithTools(submitTool(t, "submit_a", "submit_a answer"), slow))
	require.NoError(t, err)

	res, err := runPrepared(t, fake, a, "Answer.", agent.RunOptions{})
	require.NoError(t, err)

	assert.Equal(t, []string{"ok", "slow:b"}, toolResultTexts(res.Messages), "the slower sibling finished")
	assert.Equal(t, "submit_a answer", res.Output)
	assert.Equal(t, 1, fake.CallCount())
}

func TestFinalResult_TypedRunIsAConfigError(t *testing.T) {
	fake := submitFake()
	a, err := agent.NewAgent("assistant", "test-model",
		agent.WithTools(submitTool(t, "submit_a", "submit_a answer"), submitTool(t, "submit_b", "unused")))
	require.NoError(t, err)

	env := newEnv(t, fake)
	var runErr error
	env.ExecuteWorkflow(func(ctx workflow.Context) error {
		_, runErr = agent.RunTyped[struct {
			Answer string `json:"answer"`
		}](ctx, a, "Answer.")
		return runErr
	})
	require.True(t, env.IsWorkflowCompleted())
	requireConfigError(t, runErr, `tool "submit_a" ended the typed run with host text`)
}

func TestFinalResult_EmptyOutputAppendsNoAssistantMessage(t *testing.T) {
	a, err := agent.NewAgent("assistant", "test-model",
		agent.WithTools(submitTool(t, "submit_a", ""), submitTool(t, "submit_b", "unused")))
	require.NoError(t, err)

	res, err := runPrepared(t, submitFake(), a, "Answer.", agent.RunOptions{})
	require.NoError(t, err)

	assert.Equal(t, "", res.Output)
	if assert.Len(t, res.Messages, 4, "user, the calling reply, two results") {
		assistants := 0
		for _, m := range res.Messages {
			if m.IsAssistant() {
				assistants++
			}
		}
		assert.Equal(t, 1, assistants, "only the calling reply is an assistant message")
		assert.NotEmpty(t, res.Messages[3].ToolResultID(), "the run ends with the tool results")
	}
}
