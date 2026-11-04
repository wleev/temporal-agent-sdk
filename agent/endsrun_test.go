package agent_test

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.temporal.io/sdk/workflow"

	"github.com/wleev/temporal-agent-sdk/agent"
	"github.com/wleev/temporal-agent-sdk/agenttest"
	"github.com/wleev/temporal-agent-sdk/model"
	"github.com/wleev/temporal-agent-sdk/tool"
)

// finalTool returns a dynamic tool named name whose result is
// model.FinalResult(model.TextResult("ok"), output), built with opts.
func finalTool(t *testing.T, name, output string, opts ...tool.Option) tool.Tool {
	t.Helper()
	def := model.NewTool(name, "Look something up", []byte(`{"type":"object"}`))
	tl, err := tool.Dynamic(def, func(workflow.Context, string, json.RawMessage) (*model.CallToolResult, error) {
		return model.FinalResult(model.TextResult("ok"), output), nil
	}, opts...)
	require.NoError(t, err)
	return tl
}

func TestEndsRun_AFinalResultOfAToolWithoutEndsRunDoesNotEndTheRun(t *testing.T) {
	fake := agenttest.NewFakeProvider(agenttest.CallsTool("lookup", `{}`), agenttest.Says("done"))
	a, err := agent.NewAgent("assistant", "test-model", agent.WithTools(finalTool(t, "lookup", "lookup answer")))
	require.NoError(t, err)

	res, err := runPrepared(t, fake, a, "Answer.", agent.RunOptions{})
	require.NoError(t, err)

	assert.Equal(t, "done", res.Output)
	assert.Equal(t, 2, fake.CallCount())
	assert.Equal(t, []string{"ok"}, toolResultTexts(res.Messages))
	assert.False(t, hasFinalOutput(res.Messages), "the final-output key is stripped")
	assert.False(t, hasFinalOutput(fake.LastCall().Messages), "the model is not sent the final-output key")
}

func TestEndsRun_TheFirstToolWithEndsRunEndsTheRun(t *testing.T) {
	fake := agenttest.NewFakeProvider(agenttest.CallsTools(
		agenttest.ToolCall{Name: "lookup", Args: `{}`},
		agenttest.ToolCall{Name: "submit", Args: `{}`},
	))
	a, err := agent.NewAgent("assistant", "test-model", agent.WithTools(
		finalTool(t, "lookup", "lookup answer"),
		finalTool(t, "submit", "submit answer", tool.EndsRun())))
	require.NoError(t, err)

	res, err := runPrepared(t, fake, a, "Answer.", agent.RunOptions{})
	require.NoError(t, err)

	assert.Equal(t, "submit answer", res.Output)
	assert.Equal(t, 1, fake.CallCount())
	assert.Equal(t, []string{"ok", "ok"}, toolResultTexts(res.Messages))
	assert.False(t, hasFinalOutput(res.Messages), "the final-output key is stripped from every result")
}

func TestEndsRun_AnOnRoundFinalResultForAToolWithoutEndsRunDoesNotEndTheRun(t *testing.T) {
	fake := agenttest.NewFakeProvider(agenttest.CallsTool("get_weather", `{"city":"Ghent"}`), agenttest.Says("done"))
	a, err := agent.NewAgent("assistant", "test-model", agent.WithTools(weatherTool(t)))
	require.NoError(t, err)

	res, err := runPrepared(t, fake, a, "Weather in Ghent?", agent.RunOptions{
		OnRound: func(_ workflow.Context, e agent.RoundEvent) (agent.RoundOutcome, error) {
			return agent.RoundOutcome{Results: []*model.CallToolResult{model.FinalResult(e.Results[0], "It is 18°C in Ghent.")}}, nil
		},
	})
	require.NoError(t, err)

	assert.Equal(t, "done", res.Output)
	assert.Equal(t, 2, fake.CallCount())
	assert.False(t, hasFinalOutput(res.Messages))
}

func TestEndsRun_AnUnknownToolFinalResultDoesNotEndTheRun(t *testing.T) {
	fake := agenttest.NewFakeProvider(agenttest.CallsTool("missing", `{}`), agenttest.Says("done"))
	a, err := agent.NewAgent("assistant", "test-model")
	require.NoError(t, err)

	res, err := runPrepared(t, fake, a, "Answer.", agent.RunOptions{
		OnUnknownTool: func(workflow.Context, model.ToolCall) (*model.CallToolResult, error) {
			return model.FinalResult(model.TextResult("ok"), "unknown answer"), nil
		},
	})
	require.NoError(t, err)

	assert.Equal(t, "done", res.Output)
	assert.Equal(t, 2, fake.CallCount())
	assert.False(t, hasFinalOutput(res.Messages))
}

func TestEndsRun_ASubAgentResultDoesNotEndTheParentRun(t *testing.T) {
	fake := agenttest.NewFakeProvider(
		agenttest.CallsTool("research", `{"input":"the capital of Belgium"}`),
		agenttest.CallsTool("submit", `{}`),
		agenttest.Says("The capital of Belgium is Brussels."),
	)
	researcher, err := agent.NewAgent("researcher", "test-model",
		agent.WithTools(finalTool(t, "submit", "Brussels.", tool.EndsRun())))
	require.NoError(t, err)
	reg := agent.NewRegistry()
	require.NoError(t, reg.Add(researcher))
	sub, err := agent.AsSubAgentWith(researcher, "research", "Research a topic",
		agent.SubAgentOptions{ToolOptions: []tool.Option{tool.EndsRun()}})
	require.NoError(t, err)
	parent, err := agent.NewAgent("assistant", "test-model", agent.WithTools(sub))
	require.NoError(t, err)

	env := newSubAgentEnv(t, fake, reg)
	var res *agent.Result
	env.ExecuteWorkflow(func(ctx workflow.Context) (err error) {
		res, err = agent.Run(ctx, parent, "What is the capital of Belgium?")
		return err
	})
	require.True(t, env.IsWorkflowCompleted())
	require.NoError(t, env.GetWorkflowError())

	assert.Equal(t, "The capital of Belgium is Brussels.", res.Output)
	assert.Equal(t, 3, fake.CallCount())
	assert.Equal(t, []string{"Brussels."}, toolResultTexts(res.Messages))
	assert.False(t, hasFinalOutput(res.Messages))
}
