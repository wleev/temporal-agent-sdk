package agent_test

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.temporal.io/sdk/workflow"

	"github.com/wleev/temporal-agent-sdk/agent"
	"github.com/wleev/temporal-agent-sdk/agenttest"
	"github.com/wleev/temporal-agent-sdk/model"
)

func TestRunWith_OnReplyCarriesResponseMetadata(t *testing.T) {
	reply := agenttest.Says("Hello.")
	reply.ResponseID = "resp-1"
	reply.Model = "served-model"
	reply.Metadata = map[string]string{"pool": "p1"}
	reasoning := int64(3)
	reply.Usage = model.Usage{PromptTokens: 4, CompletionTokens: 6, TotalTokens: 10, ReasoningTokens: &reasoning}
	env := newEnv(t, agenttest.NewFakeProvider(reply))

	a, err := agent.NewAgent("assistant", "test-model")
	require.NoError(t, err)

	var events []agent.ReplyEvent
	var finals []agent.FinalEvent
	var result *agent.Result
	env.ExecuteWorkflow(func(ctx workflow.Context) error {
		res, runErr := agent.RunWith(ctx, a, "Hi", agent.RunOptions{
			OnReply: func(_ workflow.Context, e agent.ReplyEvent) error {
				events = append(events, e)
				return nil
			},
			OnFinal: func(_ workflow.Context, e agent.FinalEvent) (agent.FinalAction, error) {
				finals = append(finals, e)
				return agent.FinalAction{}, nil
			},
		})
		result = res
		return runErr
	})
	require.True(t, env.IsWorkflowCompleted())
	require.NoError(t, env.GetWorkflowError())

	require.Len(t, events, 1)
	e := events[0]
	assert.Equal(t, model.FinishStop, e.FinishReason)
	assert.Equal(t, "resp-1", e.ResponseID)
	assert.Equal(t, "served-model", e.Model)
	assert.Equal(t, map[string]string{"pool": "p1"}, e.Metadata)
	if assert.NotNil(t, e.Timing) {
		assert.GreaterOrEqual(t, int64(e.Timing.Duration), int64(0))
	}
	assert.Equal(t, &reasoning, e.Usage.ReasoningTokens)
	assert.Equal(t, &reasoning, result.Usage.ReasoningTokens)

	require.Len(t, finals, 1)
	assert.Equal(t, "resp-1", finals[0].ResponseID)
	assert.Equal(t, "served-model", finals[0].Model)
	assert.Equal(t, map[string]string{"pool": "p1"}, finals[0].Metadata)
	assert.NotNil(t, finals[0].Timing)
}
