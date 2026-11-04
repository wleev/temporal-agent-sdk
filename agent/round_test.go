package agent_test

import (
	"encoding/json"
	"fmt"
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

// toolResultTexts returns the text of each tool-result message in msgs, in order.
func toolResultTexts(msgs []model.Message) []string {
	var out []string
	for _, m := range msgs {
		if m.ToolResultID() != "" {
			out = append(out, m.ToolResultText())
		}
	}
	return out
}

// resultTexts returns the flattened text of each result.
func resultTexts(results []*model.CallToolResult) []string {
	out := make([]string, 0, len(results))
	for _, r := range results {
		out = append(out, model.ResultText(r))
	}
	return out
}

func TestOnRound_ReplacesResultsInCallOrder(t *testing.T) {
	fake := agenttest.NewFakeProvider(
		agenttest.CallsTools(
			agenttest.ToolCall{Name: "fast", Args: `{"text":"a"}`},
			agenttest.ToolCall{Name: "slow", Args: `{"text":"b"}`},
		),
		agenttest.Says("done"),
	)
	slow, fast := goldenTools(t)
	a, err := agent.NewAgent("assistant", "test-model", agent.WithTools(fast, slow))
	require.NoError(t, err)

	var turns []agent.TurnEvent
	var received [][]string
	_, err = runPrepared(t, fake, a, "go", agent.RunOptions{
		OnRound: func(_ workflow.Context, e agent.RoundEvent) (agent.RoundOutcome, error) {
			received = append(received, resultTexts(e.Results))
			return agent.RoundOutcome{Results: []*model.CallToolResult{model.TextResult("A"), nil}}, nil
		},
		OnTurn: func(_ workflow.Context, e agent.TurnEvent) error {
			turns = append(turns, e)
			return nil
		},
	})
	require.NoError(t, err)

	if assert.Len(t, received, 1) {
		assert.Equal(t, []string{"fast:a", "slow:b"}, received[0], "results arrive in call order")
	}
	calls := fake.Calls()
	if assert.Len(t, calls, 2) {
		assert.Equal(t, []string{"A", "slow:b"}, toolResultTexts(calls[1].Messages))
	}
	if assert.NotEmpty(t, turns) {
		assert.Equal(t, 1, turns[0].Turn)
		assert.Equal(t, []string{"A", "slow:b"}, toolResultTexts(turns[0].ToolResults))
	}
}

func TestOnRound_ReceivesTheResultsToolsReturned(t *testing.T) {
	fake := agenttest.NewFakeProvider(
		agenttest.CallsTool("get_weather", `{"city":"Ghent"}`),
		agenttest.Says("done"),
	)
	guardCalls := 0
	mask, err := guardrail.Func("mask", []guardrail.Stage{guardrail.StageToolResult},
		func(_ workflow.Context, in guardrail.CheckInput) (guardrail.Verdict, error) {
			guardCalls++
			text := strings.ReplaceAll(model.ResultText(in.Result), "18°C in Ghent", "[masked] in Ghent")
			return guardrail.RewriteResult(model.TextResult(text), "mask", nil), nil
		})
	require.NoError(t, err)

	def := model.NewTool("get_weather", "Look up the weather",
		[]byte(`{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}`))
	weather, err := tool.Dynamic(def, func(ctx workflow.Context, _ string, args json.RawMessage) (*model.CallToolResult, error) {
		var in weatherIn
		if err := json.Unmarshal(args, &in); err != nil {
			return nil, err
		}
		call := model.ToolCall{Name: "get_weather", Arguments: args}
		out, err := guardrail.Screen(ctx, []guardrail.Guardrail{mask}, guardrail.CheckInput{
			Stage: guardrail.StageToolResult, Call: &call, Result: model.TextResult(fmt.Sprintf("18°C in %s", in.City)),
		})
		if err != nil {
			return nil, err
		}
		return out.Result, nil
	})
	require.NoError(t, err)
	a, err := agent.NewAgent("assistant", "test-model", agent.WithTools(weather))
	require.NoError(t, err)

	var seq []string
	var recorded [][]string
	ro := markerOptions(&seq)
	ro.OnRound = func(_ workflow.Context, e agent.RoundEvent) (agent.RoundOutcome, error) {
		seq = append(seq, fmt.Sprintf("round:%d", e.Turn))
		recorded = append(recorded, resultTexts(e.Results))
		return agent.RoundOutcome{Results: []*model.CallToolResult{model.TextResult("observed")}}, nil
	}
	_, err = runPrepared(t, fake, a, "Weather in Ghent?", ro)
	require.NoError(t, err)

	if assert.Len(t, recorded, 1) {
		assert.Equal(t, []string{"[masked] in Ghent"}, recorded[0])
	}
	if calls := fake.Calls(); assert.Len(t, calls, 2) {
		assert.Equal(t, []string{"observed"}, toolResultTexts(calls[1].Messages))
	}
	assert.Equal(t, 1, guardCalls, "the replacement is not screened again")
	assert.Equal(t, []string{"reply:1", "round:1", "turn:1", "reply:2", "turn:2"}, seq)
}

func TestOnRound_NotCalledWhenAToolReturnsATripwire(t *testing.T) {
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
			return "18°C in " + in.City, nil
		})
	require.NoError(t, err)
	a, err := agent.NewAgent("assistant", "test-model", agent.WithTools(weather))
	require.NoError(t, err)

	rounds := 0
	_, err = runPrepared(t, fake, a, "Weather in Ghent and Bruges?", agent.RunOptions{
		OnRound: func(workflow.Context, agent.RoundEvent) (agent.RoundOutcome, error) {
			rounds++
			return agent.RoundOutcome{}, nil
		},
	})

	te, ok := agent.AsTripwire(err)
	if assert.True(t, ok, "got %v", err) {
		assert.Equal(t, 1, te.Index)
	}
	assert.Equal(t, []string{"Ghent"}, ran, "the sibling call finished")
	assert.Equal(t, 0, rounds)
}

func TestOnRound_WrongLengthIsConfigError(t *testing.T) {
	fake := agenttest.NewFakeProvider(
		agenttest.CallsTool("get_weather", `{"city":"Ghent"}`),
		agenttest.Says("unused"),
	)
	a, err := agent.NewAgent("assistant", "test-model", agent.WithTools(weatherTool(t)))
	require.NoError(t, err)

	_, err = runPrepared(t, fake, a, "Weather in Ghent?", agent.RunOptions{
		OnRound: func(workflow.Context, agent.RoundEvent) (agent.RoundOutcome, error) {
			return agent.RoundOutcome{Results: []*model.CallToolResult{model.TextResult("a"), model.TextResult("b")}}, nil
		},
	})
	requireConfigError(t, err, "OnRound returned 2 results; the round has 1 tool calls")
	assert.Equal(t, 1, fake.CallCount())
}

func TestOnRound_ErrorAbortsTheRun(t *testing.T) {
	fake := agenttest.NewFakeProvider(
		agenttest.CallsTool("get_weather", `{"city":"Ghent"}`),
		agenttest.Says("unused"),
	)
	a, err := agent.NewAgent("assistant", "test-model", agent.WithTools(weatherTool(t)))
	require.NoError(t, err)

	res, err := runPrepared(t, fake, a, "Weather in Ghent?", agent.RunOptions{
		OnRound: func(workflow.Context, agent.RoundEvent) (agent.RoundOutcome, error) {
			return agent.RoundOutcome{}, fmt.Errorf("observe failed")
		},
	})
	if assert.Error(t, err) {
		assert.Contains(t, err.Error(), "observe failed")
	}
	assert.Equal(t, 1, fake.CallCount())
	if assert.NotNil(t, res) {
		assert.Empty(t, toolResultTexts(res.Messages), "no result is appended")
	}
}

func TestOnRound_SeesThePostVerdictCallsBeforeTheHeldNotes(t *testing.T) {
	fake := agenttest.NewFakeProvider(
		agenttest.CallsTool("get_weather", `{"city":"Bruges"}`),
		agenttest.Says("done"),
	)
	note := mustFunc(t, "note", []guardrail.Stage{guardrail.StageToolCall}, func(workflow.Context, guardrail.CheckInput) (guardrail.Verdict, error) {
		return guardrail.Annotate("r", guardrail.Notice{Text: "Note B", Audience: guardrail.AudienceModel}), nil
	})
	spell := mustFunc(t, "spell", []guardrail.Stage{guardrail.StageToolCall}, func(workflow.Context, guardrail.CheckInput) (guardrail.Verdict, error) {
		return guardrail.RewriteArguments(json.RawMessage(`{"city":"Brugge"}`), "spelling", nil), nil
	})
	a, err := agent.NewAgent("assistant", "test-model",
		agent.WithTools(weatherTool(t)), agent.WithGuardrails(note, spell))
	require.NoError(t, err)

	var events []agent.RoundEvent
	_, err = runPrepared(t, fake, a, "Weather in Bruges?", agent.RunOptions{
		OnRound: func(_ workflow.Context, e agent.RoundEvent) (agent.RoundOutcome, error) {
			events = append(events, e)
			return agent.RoundOutcome{Results: []*model.CallToolResult{model.TextResult("observed")}}, nil
		},
	})
	require.NoError(t, err)

	if assert.Len(t, events, 1) {
		if assert.Len(t, events[0].Calls, 1) {
			assert.JSONEq(t, `{"city":"Brugge"}`, string(events[0].Calls[0].Arguments))
		}
		if calls := events[0].Assistant.ToolCalls(); assert.Len(t, calls, 1) {
			assert.JSONEq(t, `{"city":"Brugge"}`, string(calls[0].Arguments))
		}
		assert.Equal(t, []string{"18°C in Brugge"}, resultTexts(events[0].Results))
	}
	if calls := fake.Calls(); assert.Len(t, calls, 2) {
		msgs := calls[1].Messages
		if assert.GreaterOrEqual(t, len(msgs), 2) {
			assert.Equal(t, "observed", msgs[len(msgs)-2].ToolResultText())
			assert.Equal(t, "Note B", msgs[len(msgs)-1].Text())
		}
	}
}

func TestOnRound_EmptyResultsKeepEveryResult(t *testing.T) {
	fake := agenttest.NewFakeProvider(agenttest.CallsTool("get_weather", `{"city":"Ghent"}`), agenttest.Says("done"))
	a, err := agent.NewAgent("assistant", "test-model", agent.WithTools(weatherTool(t)))
	require.NoError(t, err)

	res, err := runPrepared(t, fake, a, "Weather in Ghent?", agent.RunOptions{
		OnRound: func(workflow.Context, agent.RoundEvent) (agent.RoundOutcome, error) {
			return agent.RoundOutcome{Results: []*model.CallToolResult{}}, nil
		},
	})
	require.NoError(t, err)
	assert.Equal(t, []string{"18°C in Ghent"}, toolResultTexts(res.Messages))
}
