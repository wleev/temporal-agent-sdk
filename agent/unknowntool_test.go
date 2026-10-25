package agent_test

import (
	"errors"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.temporal.io/sdk/workflow"

	"github.com/wleev/temporal-agent-sdk/agent"
	"github.com/wleev/temporal-agent-sdk/agenttest"
	"github.com/wleev/temporal-agent-sdk/model"
	"github.com/wleev/temporal-agent-sdk/tool"
)

func TestOnUnknownTool_ResultReplacesBuiltIn(t *testing.T) {
	fake := agenttest.NewFakeProvider(
		agenttest.CallsTools(agenttest.ToolCall{ID: "u1", Name: "mcp__app_cal__list", Args: `{"day":"mon"}`}),
		agenttest.Says("done"),
	)
	a, err := agent.NewAgent("assistant", "test-model")
	require.NoError(t, err)

	var calls []model.ToolCall
	var turns []agent.TurnEvent
	_, err = runPrepared(t, fake, a, "go", agent.RunOptions{
		OnUnknownTool: func(_ workflow.Context, call model.ToolCall) (*model.CallToolResult, error) {
			calls = append(calls, call)
			return model.TextResult("Calendar tools are now loaded; call the tool again."), nil
		},
		OnTurn: func(_ workflow.Context, e agent.TurnEvent) error {
			turns = append(turns, e)
			return nil
		},
	})
	require.NoError(t, err)

	if assert.Len(t, calls, 1) {
		assert.Equal(t, "u1", calls[0].ID)
		assert.Equal(t, "mcp__app_cal__list", calls[0].Name)
		assert.JSONEq(t, `{"day":"mon"}`, string(calls[0].Arguments))
	}
	reqs := fake.Calls()
	if assert.Len(t, reqs, 2) {
		last := reqs[1].Messages[len(reqs[1].Messages)-1]
		assert.Equal(t, "Calendar tools are now loaded; call the tool again.", last.ToolResultText())
	}
	if assert.Len(t, turns, 2) && assert.Len(t, turns[0].ToolResults, 1) {
		assert.Equal(t, "u1", turns[0].ToolResults[0].ToolResultID())
		assert.Equal(t, "Calendar tools are now loaded; call the tool again.", turns[0].ToolResults[0].ToolResultText())
	}
}

func TestOnUnknownTool_NilKeepsBuiltIn(t *testing.T) {
	fake := agenttest.NewFakeProvider(
		agenttest.CallsTool("missing", `{}`),
		agenttest.Says("done"),
	)
	a, err := agent.NewAgent("assistant", "test-model")
	require.NoError(t, err)

	hooked := 0
	_, err = runPrepared(t, fake, a, "go", agent.RunOptions{
		OnUnknownTool: func(workflow.Context, model.ToolCall) (*model.CallToolResult, error) {
			hooked++
			return nil, nil
		},
	})
	require.NoError(t, err)

	assert.Equal(t, 1, hooked)
	reqs := fake.Calls()
	if assert.Len(t, reqs, 2) {
		last := reqs[1].Messages[len(reqs[1].Messages)-1]
		assert.Contains(t, last.ToolResultText(), `no tool named "missing"`)
	}
}

func TestOnUnknownTool_NotCalledForKnownTools(t *testing.T) {
	fake := agenttest.NewFakeProvider(
		agenttest.CallsTool("get_weather", `{"city":"Ghent"}`),
		agenttest.Says("done"),
	)
	a, err := agent.NewAgent("assistant", "test-model", agent.WithTools(weatherTool(t)))
	require.NoError(t, err)

	hooked := 0
	_, err = runPrepared(t, fake, a, "go", agent.RunOptions{
		OnUnknownTool: func(workflow.Context, model.ToolCall) (*model.CallToolResult, error) {
			hooked++
			return nil, nil
		},
	})
	require.NoError(t, err)
	assert.Equal(t, 0, hooked)
}

func TestOnUnknownTool_PerCallInfoAndOrder(t *testing.T) {
	fake := agenttest.NewFakeProvider(
		agenttest.CallsTools(
			agenttest.ToolCall{ID: "x", Name: "ghost_a"},
			agenttest.ToolCall{ID: "k", Name: "get_weather", Args: `{"city":"Ghent"}`},
			agenttest.ToolCall{ID: "y", Name: "ghost_b"},
		),
		agenttest.Says("done"),
	)
	a, err := agent.NewAgent("assistant", "test-model", agent.WithTools(weatherTool(t)))
	require.NoError(t, err)

	infos := map[string]tool.CallInfo{}
	_, err = runPrepared(t, fake, a, "go", agent.RunOptions{
		OnUnknownTool: func(ctx workflow.Context, call model.ToolCall) (*model.CallToolResult, error) {
			info, ok := tool.CallInfoFrom(ctx)
			assert.True(t, ok)
			infos[call.ID] = info
			return model.TextResult("handled:" + call.Name), nil
		},
	})
	require.NoError(t, err)

	assert.Equal(t, map[string]tool.CallInfo{
		"x": {ID: "x", Turn: 1, Index: 0},
		"y": {ID: "y", Turn: 1, Index: 2},
	}, infos)
	reqs := fake.Calls()
	if assert.Len(t, reqs, 2) {
		msgs := reqs[1].Messages
		results := msgs[len(msgs)-3:]
		var ids, texts []string
		for _, m := range results {
			ids = append(ids, m.ToolResultID())
			texts = append(texts, m.ToolResultText())
		}
		assert.Equal(t, []string{"x", "k", "y"}, ids)
		assert.Equal(t, []string{"handled:ghost_a", "18°C in Ghent", "handled:ghost_b"}, texts)
	}
}

func TestOnUnknownTool_ErrorAbortsRun(t *testing.T) {
	fake := agenttest.NewFakeProvider(
		agenttest.CallsTool("missing", `{}`),
		agenttest.Says("done"),
	)
	a, err := agent.NewAgent("assistant", "test-model")
	require.NoError(t, err)

	_, err = runPrepared(t, fake, a, "go", agent.RunOptions{
		OnUnknownTool: func(workflow.Context, model.ToolCall) (*model.CallToolResult, error) {
			return nil, errors.New("load failed")
		},
	})
	if assert.Error(t, err) {
		assert.Contains(t, err.Error(), "load failed")
	}
	assert.Equal(t, 1, fake.CallCount())
}

func TestOnUnknownTool_ToolAddedByPlanIsKnown(t *testing.T) {
	fake := agenttest.NewFakeProvider(
		agenttest.CallsTool("echo", `{"text":"hi"}`),
		agenttest.Says("done"),
	)
	a, err := agent.NewAgent("assistant", "test-model")
	require.NoError(t, err)
	echo := echoTool(t)

	hooked := 0
	_, err = runPrepared(t, fake, a, "go", agent.RunOptions{
		PrepareModelCall: func(_ workflow.Context, p agent.ModelCallPlan) (agent.ModelCallPlan, error) {
			if p.Turn == 1 {
				p.Tools = append(p.Tools, echo)
			}
			return p, nil
		},
		OnUnknownTool: func(workflow.Context, model.ToolCall) (*model.CallToolResult, error) {
			hooked++
			return nil, nil
		},
	})
	require.NoError(t, err)

	assert.Equal(t, 0, hooked)
	reqs := fake.Calls()
	if assert.Len(t, reqs, 2) {
		last := reqs[1].Messages[len(reqs[1].Messages)-1]
		assert.Equal(t, "echo:hi", last.ToolResultText())
	}
}
