package agent_test

import (
	"context"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.temporal.io/sdk/converter"
	"go.temporal.io/sdk/workflow"

	"github.com/wleev/temporal-agent-sdk/agent"
	"github.com/wleev/temporal-agent-sdk/agenttest"
	"github.com/wleev/temporal-agent-sdk/model"
	"github.com/wleev/temporal-agent-sdk/tool"
)

// TestReplay_ModelCallHooks records a run that sets PrepareModelCall, OnFinal
// and OnUnknownTool and replays its history.
func TestReplay_ModelCallHooks(t *testing.T) {
	env := startReplayEnv(t)

	const name = "replay_model_call_hooks"
	const loadApp = "agentsdk_test_load_app"

	fast, err := tool.New("fast", "A fast tool",
		func(_ workflow.Context, in echoIn) (string, error) { return "fast:" + in.Text, nil })
	require.NoError(t, err)
	echo := echoTool(t)
	a, err := agent.NewAgent("assistant", "test-model",
		agent.WithTools(fast), agent.WithMaxTurns(6))
	require.NoError(t, err)

	wf := func(ctx workflow.Context) (*agent.Result, error) {
		ctx = workflow.WithActivityOptions(ctx, workflow.ActivityOptions{
			StartToCloseTimeout: 10 * time.Second,
		})
		nudged := false
		return agent.RunWith(ctx, a, "go", agent.RunOptions{
			PrepareModelCall: func(_ workflow.Context, p agent.ModelCallPlan) (agent.ModelCallPlan, error) {
				switch p.Turn {
				case 1:
					p.Advertised = []string{"fast"}
					p.Settings.Temperature = model.Ptr(0.0)
				case 2:
					p.Tools = append(p.Tools, echo)
				}
				return p, nil
			},
			OnUnknownTool: func(ctx workflow.Context, call model.ToolCall) (*model.CallToolResult, error) {
				var out string
				if err := workflow.ExecuteActivity(ctx, loadApp, call.Name).Get(ctx, &out); err != nil {
					return nil, err
				}
				return model.TextResult(out), nil
			},
			OnFinal: func(_ workflow.Context, _ agent.FinalEvent) ([]model.Message, error) {
				if nudged {
					return nil, nil
				}
				nudged = true
				return []model.Message{model.UserMessage("nudge")}, nil
			},
		})
	}

	fake := agenttest.NewFakeProvider(
		agenttest.CallsTools(
			agenttest.ToolCall{ID: "a", Name: "fast", Args: `{"text":"1"}`},
			agenttest.ToolCall{ID: "b", Name: "missing", Args: `{}`},
		),
		agenttest.CallsTool("echo", `{"text":"2"}`),
		agenttest.Says(""),
		agenttest.Says("done"),
	)

	hist := env.runWith(t, fake, nil, wf, name, map[string]any{
		loadApp: func(_ context.Context, n string) (string, error) { return "loaded " + n, nil },
	})

	counts := map[string]int{}
	for _, ev := range hist.Events {
		if attrs := ev.GetActivityTaskScheduledEventAttributes(); attrs != nil {
			counts[attrs.GetActivityType().GetName()]++
		}
	}
	assert.Equal(t, 4, counts[model.InvokeModelActivity])
	assert.Equal(t, 1, counts[loadApp])

	var res agent.Result
	for _, ev := range hist.Events {
		if attrs := ev.GetWorkflowExecutionCompletedEventAttributes(); attrs != nil {
			require.NoError(t, converter.GetDefaultDataConverter().FromPayloads(attrs.GetResult(), &res))
		}
	}
	assert.Equal(t, "done", res.Output)
	assert.Equal(t, 4, res.Turns)
	var texts []string
	for _, m := range res.Messages {
		if m.Role == model.RoleTool {
			texts = append(texts, m.ToolResultText())
		} else {
			texts = append(texts, m.Text())
		}
	}
	assert.Equal(t, []string{"go", "", "fast:1", "loaded missing", "", "echo:2", "nudge", "done"}, texts)

	assertReplays(t, hist, nil, wf, name)
}
