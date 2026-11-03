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

// runRecordingAttempts runs a one-turn agent over fake and returns the events
// OnReply and OnFinal received.
func runRecordingAttempts(t *testing.T, fake *agenttest.FakeProvider) ([]agent.ReplyEvent, []agent.FinalEvent) {
	t.Helper()
	env := newEnv(t, fake)
	a, err := agent.NewAgent("assistant", "test-model")
	require.NoError(t, err)

	var replies []agent.ReplyEvent
	var finals []agent.FinalEvent
	env.ExecuteWorkflow(func(ctx workflow.Context) error {
		_, runErr := agent.RunWith(ctx, a, "hi", agent.RunOptions{
			OnReply: func(_ workflow.Context, e agent.ReplyEvent) error {
				replies = append(replies, e)
				return nil
			},
			OnFinal: func(_ workflow.Context, e agent.FinalEvent) ([]model.Message, error) {
				finals = append(finals, e)
				return nil, nil
			},
		})
		return runErr
	})
	require.True(t, env.IsWorkflowCompleted())
	require.NoError(t, env.GetWorkflowError())
	return replies, finals
}

func TestOnReply_ReceivesAttempt(t *testing.T) {
	fake := agenttest.NewFakeProvider(
		agenttest.Fails(&model.APIError{StatusCode: 503}),
		agenttest.Says("done"),
	)
	replies, finals := runRecordingAttempts(t, fake)

	if assert.Len(t, replies, 1) {
		assert.Equal(t, int32(2), replies[0].Attempt)
	}
	if assert.Len(t, finals, 1) {
		assert.Equal(t, int32(2), finals[0].Attempt)
	}
}

func TestOnReply_FirstAttemptIsOne(t *testing.T) {
	replies, finals := runRecordingAttempts(t, agenttest.NewFakeProvider(agenttest.Says("done")))

	if assert.Len(t, replies, 1) {
		assert.Equal(t, int32(1), replies[0].Attempt)
	}
	if assert.Len(t, finals, 1) {
		assert.Equal(t, int32(1), finals[0].Attempt)
	}
}
