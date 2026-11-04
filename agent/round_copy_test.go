package agent_test

import (
	"testing"

	mcpsdk "github.com/modelcontextprotocol/go-sdk/mcp"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.temporal.io/sdk/workflow"

	"github.com/wleev/temporal-agent-sdk/agent"
	"github.com/wleev/temporal-agent-sdk/agenttest"
)

func TestOnRound_ChangingAnEventResultInPlaceKeepsTheResult(t *testing.T) {
	fake := agenttest.NewFakeProvider(agenttest.CallsTool("get_weather", `{"city":"Ghent"}`), agenttest.Says("done"))
	a, err := agent.NewAgent("assistant", "test-model", agent.WithTools(weatherTool(t)))
	require.NoError(t, err)

	res, err := runPrepared(t, fake, a, "Weather in Ghent?", agent.RunOptions{
		OnRound: func(_ workflow.Context, e agent.RoundEvent) (agent.RoundOutcome, error) {
			e.Results[0].Content[0] = &mcpsdk.TextContent{Text: "changed"}
			e.Results[0].Content = append(e.Results[0].Content, &mcpsdk.TextContent{Text: "extra"})
			e.Results[0].IsError = true
			return agent.RoundOutcome{}, nil
		},
	})
	require.NoError(t, err)

	assert.Equal(t, []string{"18°C in Ghent"}, toolResultTexts(res.Messages))
	for _, m := range res.Messages {
		for _, b := range m.Blocks {
			if b.Result != nil {
				assert.False(t, b.Result.IsError)
			}
		}
	}
}

func TestOnRound_ChangingTheEventReplyInPlaceKeepsTheConversation(t *testing.T) {
	fake := agenttest.NewFakeProvider(agenttest.CallsTool("get_weather", `{"city":"Ghent"}`), agenttest.Says("done"))
	a, err := agent.NewAgent("assistant", "test-model", agent.WithTools(weatherTool(t)))
	require.NoError(t, err)

	res, err := runPrepared(t, fake, a, "Weather in Ghent?", agent.RunOptions{
		OnRound: func(_ workflow.Context, e agent.RoundEvent) (agent.RoundOutcome, error) {
			e.Assistant.Blocks[0].Arguments[2] = 'X'
			e.Assistant.Blocks[0].ToolName = "changed"
			e.Calls[0].Arguments[3] = 'Y'
			return agent.RoundOutcome{}, nil
		},
	})
	require.NoError(t, err)

	var calls int
	for _, m := range res.Messages {
		for _, call := range m.ToolCalls() {
			calls++
			assert.Equal(t, "get_weather", call.Name)
			assert.JSONEq(t, `{"city":"Ghent"}`, string(call.Arguments))
		}
	}
	assert.Equal(t, 1, calls)
}
