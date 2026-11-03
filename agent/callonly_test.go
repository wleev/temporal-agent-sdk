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

func messageTexts(msgs []model.Message) []string {
	out := make([]string, 0, len(msgs))
	for _, m := range msgs {
		out = append(out, m.Text())
	}
	return out
}

func callOnlyFake() *agenttest.FakeProvider {
	return agenttest.NewFakeProvider(
		agenttest.CallsTool("get_weather", `{"city":"Ghent"}`),
		agenttest.Says("done"),
	)
}

func TestCallOnly_SentOnceAndNotKept(t *testing.T) {
	fake := callOnlyFake()
	a, err := agent.NewAgent("assistant", "test-model", agent.WithTools(weatherTool(t)))
	require.NoError(t, err)

	res, err := runPrepared(t, fake, a, "Weather in Ghent?", agent.RunOptions{
		PrepareModelCall: func(_ workflow.Context, p agent.ModelCallPlan) (agent.ModelCallPlan, error) {
			if p.Turn == 2 {
				p.CallOnly = []model.Message{model.UserMessage("answer now")}
			}
			return p, nil
		},
	})
	require.NoError(t, err)

	if assert.Equal(t, 2, fake.CallCount()) {
		calls := fake.Calls()
		last := calls[1].Messages[len(calls[1].Messages)-1]
		assert.Equal(t, "answer now", last.Text())
		assert.NotContains(t, messageTexts(calls[0].Messages), "answer now")
		assert.Len(t, calls[1].Messages, len(calls[0].Messages)+3)
	}
	assert.NotContains(t, messageTexts(res.Messages), "answer now")
}

func TestCallOnly_FollowsAppend(t *testing.T) {
	fake := callOnlyFake()
	a, err := agent.NewAgent("assistant", "test-model", agent.WithTools(weatherTool(t)))
	require.NoError(t, err)

	res, err := runPrepared(t, fake, a, "Weather in Ghent?", agent.RunOptions{
		PrepareModelCall: func(_ workflow.Context, p agent.ModelCallPlan) (agent.ModelCallPlan, error) {
			if p.Turn == 2 {
				p.Append = []model.Message{model.UserMessage("kept")}
				p.CallOnly = []model.Message{model.UserMessage("transient")}
			}
			return p, nil
		},
	})
	require.NoError(t, err)

	if assert.Equal(t, 2, fake.CallCount()) {
		texts := messageTexts(fake.Calls()[1].Messages)
		if assert.GreaterOrEqual(t, len(texts), 2) {
			assert.Equal(t, []string{"kept", "transient"}, texts[len(texts)-2:])
		}
	}
	texts := messageTexts(res.Messages)
	assert.Contains(t, texts, "kept")
	assert.NotContains(t, texts, "transient")
}

func TestCallOnly_EmptySendsNothingExtra(t *testing.T) {
	plain := callOnlyFake()
	a, err := agent.NewAgent("assistant", "test-model", agent.WithTools(weatherTool(t)))
	require.NoError(t, err)
	_, err = runPrepared(t, plain, a, "Weather in Ghent?", agent.RunOptions{})
	require.NoError(t, err)

	for name, callOnly := range map[string][]model.Message{"nil": nil, "empty": {}} {
		fake := callOnlyFake()
		_, err = runPrepared(t, fake, a, "Weather in Ghent?", agent.RunOptions{
			PrepareModelCall: func(_ workflow.Context, p agent.ModelCallPlan) (agent.ModelCallPlan, error) {
				p.CallOnly = callOnly
				return p, nil
			},
		})
		require.NoError(t, err, name)
		if assert.Equal(t, plain.CallCount(), fake.CallCount(), name) {
			for i, c := range fake.Calls() {
				assert.Equal(t, plain.Calls()[i].Messages, c.Messages, name)
			}
		}
	}
}

func TestCallOnly_NotCarriedIntoContinuation(t *testing.T) {
	fake := agenttest.NewFakeProvider(
		agenttest.Truncated("The answer "),
		agenttest.Says("is done."),
	)
	a, err := agent.NewAgent("assistant", "test-model", agent.WithContinueOnLength(2))
	require.NoError(t, err)

	var replies []string
	res, err := runPrepared(t, fake, a, "Tell me.", agent.RunOptions{
		PrepareModelCall: func(_ workflow.Context, p agent.ModelCallPlan) (agent.ModelCallPlan, error) {
			if p.Turn == 1 {
				p.CallOnly = []model.Message{model.UserMessage("transient")}
			}
			return p, nil
		},
		OnReply: func(_ workflow.Context, e agent.ReplyEvent) error {
			replies = append(replies, e.Assistant.Text())
			return nil
		},
	})
	require.NoError(t, err)

	if assert.Equal(t, 2, fake.CallCount()) {
		calls := fake.Calls()
		assert.Contains(t, messageTexts(calls[0].Messages), "transient")
		assert.NotContains(t, messageTexts(calls[1].Messages), "transient")
	}
	assert.Equal(t, []string{"The answer ", "is done."}, replies)
	assert.NotContains(t, messageTexts(res.Messages), "transient")
}
