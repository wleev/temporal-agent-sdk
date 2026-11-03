package agent_test

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.temporal.io/sdk/temporal"
	"go.temporal.io/sdk/workflow"

	"github.com/wleev/temporal-agent-sdk/agent"
	"github.com/wleev/temporal-agent-sdk/agenttest"
	"github.com/wleev/temporal-agent-sdk/model"
	"github.com/wleev/temporal-agent-sdk/tool"
)

func newChoiceTool(t *testing.T, name string) tool.Tool {
	t.Helper()
	tl, err := tool.New(name, "Tool "+name,
		func(_ workflow.Context, in echoIn) (string, error) { return in.Text, nil })
	require.NoError(t, err)
	return tl
}

// requireConfigError asserts that err is a non-retryable ErrorTypeConfig
// application error whose message contains want.
func requireConfigError(t *testing.T, err error, want string) {
	t.Helper()
	var appErr *temporal.ApplicationError
	require.ErrorAs(t, err, &appErr)
	assert.Equal(t, agent.ErrorTypeConfig, appErr.Type())
	assert.True(t, appErr.NonRetryable())
	assert.Contains(t, appErr.Message(), want)
}

// assertConfigError asserts that err is a non-retryable ErrorTypeConfig error
// whose message contains want and that fake received no call.
func assertConfigError(t *testing.T, err error, fake *agenttest.FakeProvider, want string) {
	t.Helper()
	requireConfigError(t, err, want)
	assert.Equal(t, 0, fake.CallCount())
}

func TestToolChoice_NameMustBeAdvertised(t *testing.T) {
	fake := agenttest.NewFakeProvider(agenttest.Says("unreached"))
	a, err := agent.NewAgent("assistant", "test-model",
		agent.WithTools(newChoiceTool(t, "fast"), newChoiceTool(t, "slow")))
	require.NoError(t, err)

	_, err = runPrepared(t, fake, a, "hi", agent.RunOptions{
		PrepareModelCall: func(_ workflow.Context, p agent.ModelCallPlan) (agent.ModelCallPlan, error) {
			p.Advertised = []string{"fast"}
			p.Settings.ToolChoice = &model.ToolChoice{Mode: model.ToolChoiceTool, Name: "slow"}
			return p, nil
		},
	})
	assertConfigError(t, err, fake, `tool "slow" is not advertised`)
}

func TestNewAgent_ToolChoiceModes(t *testing.T) {
	cases := []struct {
		name    string
		choice  model.ToolChoice
		wantErr string
	}{
		{name: "auto", choice: model.ToolChoice{Mode: model.ToolChoiceAuto}},
		{name: "none", choice: model.ToolChoice{Mode: model.ToolChoiceNone}},
		{name: "required", choice: model.ToolChoice{Mode: model.ToolChoiceRequired}, wantErr: `tool choice mode "required" applies per call only`},
		{name: "tool", choice: model.ToolChoice{Mode: model.ToolChoiceTool, Name: "fast"}, wantErr: `tool choice mode "tool" applies per call only`},
		{name: "empty", choice: model.ToolChoice{}, wantErr: `unknown tool choice mode ""`},
		{name: "unknown", choice: model.ToolChoice{Mode: "sometimes"}, wantErr: `unknown tool choice mode "sometimes"`},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := agent.NewAgent("assistant", "test-model",
				agent.WithTools(newChoiceTool(t, "fast")),
				agent.WithSettings(model.Settings{ToolChoice: &tc.choice}))
			if tc.wantErr == "" {
				assert.NoError(t, err)
				return
			}
			requireConfigError(t, err, tc.wantErr)
		})
	}
}

func TestToolChoice_InvalidPlanModeFails(t *testing.T) {
	cases := []struct {
		name string
		mode model.ToolChoiceMode
		want string
	}{
		{name: "empty", mode: "", want: `tool choice: unknown mode ""`},
		{name: "unknown", mode: "sometimes", want: `tool choice: unknown mode "sometimes"`},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			fake := agenttest.NewFakeProvider(agenttest.Says("unreached"))
			a, err := agent.NewAgent("assistant", "test-model",
				agent.WithTools(newChoiceTool(t, "fast")))
			require.NoError(t, err)

			_, err = runPrepared(t, fake, a, "hi", agent.RunOptions{
				PrepareModelCall: func(_ workflow.Context, p agent.ModelCallPlan) (agent.ModelCallPlan, error) {
					p.Settings.ToolChoice = &model.ToolChoice{Mode: tc.mode}
					return p, nil
				},
			})
			assertConfigError(t, err, fake, tc.want)
		})
	}
}

func TestToolChoice_RequiredNeedsAdvertisedTools(t *testing.T) {
	fake := agenttest.NewFakeProvider(agenttest.Says("unreached"))
	a, err := agent.NewAgent("assistant", "test-model")
	require.NoError(t, err)

	_, err = runPrepared(t, fake, a, "hi", agent.RunOptions{
		PrepareModelCall: func(_ workflow.Context, p agent.ModelCallPlan) (agent.ModelCallPlan, error) {
			p.Settings.ToolChoice = &model.ToolChoice{Mode: model.ToolChoiceRequired}
			return p, nil
		},
	})
	assertConfigError(t, err, fake, `mode "required" needs tools`)
}

func TestToolChoice_AdvertisedNameReachesTheProvider(t *testing.T) {
	fake := agenttest.NewFakeProvider(agenttest.Says("ok"))
	a, err := agent.NewAgent("assistant", "test-model",
		agent.WithTools(newChoiceTool(t, "fast")))
	require.NoError(t, err)

	res, err := runPrepared(t, fake, a, "hi", agent.RunOptions{
		PrepareModelCall: func(_ workflow.Context, p agent.ModelCallPlan) (agent.ModelCallPlan, error) {
			p.Settings.ToolChoice = &model.ToolChoice{Mode: model.ToolChoiceTool, Name: "fast"}
			return p, nil
		},
	})
	require.NoError(t, err)
	assert.Equal(t, "ok", res.Output)
	if assert.Equal(t, 1, fake.CallCount()) {
		if assert.NotNil(t, fake.Calls()[0].Settings.ToolChoice) {
			assert.Equal(t, "fast", fake.Calls()[0].Settings.ToolChoice.Name)
		}
	}
}
