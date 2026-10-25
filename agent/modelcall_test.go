package agent_test

import (
	"encoding/json"
	"errors"
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

// echoTool returns a tool named echo that answers "echo:" followed by its input
// text.
func echoTool(t *testing.T) tool.Tool {
	t.Helper()
	tl, err := tool.New("echo", "Echo the text back",
		func(_ workflow.Context, in echoIn) (string, error) { return "echo:" + in.Text, nil })
	require.NoError(t, err)
	return tl
}

// defNames returns the names of tool definitions, in order.
func defNames(defs []*model.Tool) []string {
	names := make([]string, 0, len(defs))
	for _, d := range defs {
		names = append(names, d.Name)
	}
	return names
}

// toolNames returns the names of tools, in order.
func toolNames(tools []tool.Tool) []string {
	names := make([]string, 0, len(tools))
	for _, t := range tools {
		names = append(names, t.Def().Name)
	}
	return names
}

// unnamedTool is a tool whose definition has no name.
type unnamedTool struct{}

// Def returns a definition with an empty name.
func (unnamedTool) Def() *model.Tool { return &model.Tool{} }

// Invoke returns an empty text result.
func (unnamedTool) Invoke(workflow.Context, json.RawMessage) (*model.CallToolResult, error) {
	return model.TextResult(""), nil
}

// Policy returns the zero policy.
func (unnamedTool) Policy() tool.Policy { return tool.Policy{} }

// runPrepared runs a on input with ro and returns the run's Result and error.
func runPrepared(t *testing.T, fake *agenttest.FakeProvider, a *agent.Agent, input string, ro agent.RunOptions) (*agent.Result, error) {
	t.Helper()
	env := newEnv(t, fake)
	var res *agent.Result
	var runErr error
	env.ExecuteWorkflow(func(ctx workflow.Context) error {
		res, runErr = agent.RunWith(ctx, a, input, ro)
		return runErr
	})
	require.True(t, env.IsWorkflowCompleted())
	return res, runErr
}

func TestPrepareModelCall_ReceivesTurnToolsAndSettings(t *testing.T) {
	fake := agenttest.NewFakeProvider(
		agenttest.CallsTool("get_weather", `{"city":"Ghent"}`),
		agenttest.Says("done"),
	)
	a, err := agent.NewAgent("assistant", "test-model",
		agent.WithTools(weatherTool(t)),
		agent.WithSettings(model.Settings{Temperature: model.Ptr(0.7)}))
	require.NoError(t, err)

	var plans []agent.ModelCallPlan
	res, err := runPrepared(t, fake, a, "Weather in Ghent?", agent.RunOptions{
		ExtraTools: []tool.Tool{echoTool(t)},
		PrepareModelCall: func(_ workflow.Context, p agent.ModelCallPlan) (agent.ModelCallPlan, error) {
			plans = append(plans, p)
			return p, nil
		},
	})
	require.NoError(t, err)
	assert.Equal(t, "done", res.Output)

	if assert.Len(t, plans, 2) {
		for i, p := range plans {
			assert.Equal(t, i+1, p.Turn)
			assert.Equal(t, []string{"get_weather", "echo"}, toolNames(p.Tools))
			assert.Nil(t, p.Advertised)
			if assert.NotNil(t, p.Settings.Temperature) {
				assert.Equal(t, 0.7, *p.Settings.Temperature)
			}
			assert.Empty(t, p.Append)
		}
	}
	if assert.Equal(t, 2, fake.CallCount()) {
		assert.Equal(t, []string{"get_weather", "echo"}, defNames(fake.Calls()[0].Tools))
	}
}

func TestPrepareModelCall_AdvertisedNarrowsOneCall(t *testing.T) {
	fake := agenttest.NewFakeProvider(
		agenttest.CallsTool("get_weather", `{"city":"Ghent"}`),
		agenttest.Says("done"),
	)
	a, err := agent.NewAgent("assistant", "test-model",
		agent.WithTools(weatherTool(t)),
		agent.WithSettings(model.Settings{Temperature: model.Ptr(0.7)}))
	require.NoError(t, err)

	_, err = runPrepared(t, fake, a, "Weather in Ghent?", agent.RunOptions{
		ExtraTools: []tool.Tool{echoTool(t)},
		PrepareModelCall: func(_ workflow.Context, p agent.ModelCallPlan) (agent.ModelCallPlan, error) {
			if p.Turn == 1 {
				p.Advertised = []string{"echo"}
				p.Settings.Temperature = model.Ptr(0.0)
			}
			return p, nil
		},
	})
	require.NoError(t, err)

	calls := fake.Calls()
	if assert.Len(t, calls, 2) {
		assert.Equal(t, []string{"echo"}, defNames(calls[0].Tools))
		assert.Equal(t, model.Ptr(0.0), calls[0].Settings.Temperature)
		assert.Equal(t, []string{"get_weather", "echo"}, defNames(calls[1].Tools))
		assert.Equal(t, model.Ptr(0.7), calls[1].Settings.Temperature)
		last := calls[1].Messages[len(calls[1].Messages)-1]
		assert.Equal(t, "18°C in Ghent", last.ToolResultText())
	}
}

func TestPrepareModelCall_EmptyAdvertisedSendsAllTools(t *testing.T) {
	fake := agenttest.NewFakeProvider(agenttest.Says("done"))
	a, err := agent.NewAgent("assistant", "test-model", agent.WithTools(weatherTool(t)))
	require.NoError(t, err)

	res, err := runPrepared(t, fake, a, "hi", agent.RunOptions{
		ExtraTools: []tool.Tool{echoTool(t)},
		PrepareModelCall: func(_ workflow.Context, p agent.ModelCallPlan) (agent.ModelCallPlan, error) {
			p.Advertised = []string{}
			return p, nil
		},
	})
	require.NoError(t, err)
	assert.Equal(t, "done", res.Output)
	if assert.Equal(t, 1, fake.CallCount()) {
		assert.Equal(t, []string{"get_weather", "echo"}, defNames(fake.Calls()[0].Tools))
	}
}

func TestPrepareModelCall_EmptyToolsKeepRunTools(t *testing.T) {
	a, err := agent.NewAgent("assistant", "test-model", agent.WithTools(weatherTool(t)))
	require.NoError(t, err)

	tests := []struct {
		name  string
		tools []tool.Tool
	}{
		{name: "nil"},
		{name: "empty", tools: []tool.Tool{}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			fake := agenttest.NewFakeProvider(
				agenttest.CallsTool("get_weather", `{"city":"Ghent"}`),
				agenttest.Says("done"),
			)
			res, err := runPrepared(t, fake, a, "Weather in Ghent?", agent.RunOptions{
				ExtraTools: []tool.Tool{echoTool(t)},
				PrepareModelCall: func(_ workflow.Context, p agent.ModelCallPlan) (agent.ModelCallPlan, error) {
					return agent.ModelCallPlan{Turn: p.Turn, Tools: tt.tools}, nil
				},
			})
			require.NoError(t, err)
			assert.Equal(t, "done", res.Output)

			calls := fake.Calls()
			if assert.Len(t, calls, 2) {
				assert.Equal(t, []string{"get_weather", "echo"}, defNames(calls[0].Tools))
				assert.Equal(t, []string{"get_weather", "echo"}, defNames(calls[1].Tools))
				last := calls[1].Messages[len(calls[1].Messages)-1]
				assert.Equal(t, "18°C in Ghent", last.ToolResultText())
			}
		})
	}
}

func TestPrepareModelCall_OtherTurnFails(t *testing.T) {
	fake := agenttest.NewFakeProvider(agenttest.Says("unreached"))
	a, err := agent.NewAgent("assistant", "test-model", agent.WithTools(weatherTool(t)))
	require.NoError(t, err)

	res, err := runPrepared(t, fake, a, "hi", agent.RunOptions{
		PrepareModelCall: func(workflow.Context, agent.ModelCallPlan) (agent.ModelCallPlan, error) {
			return agent.ModelCallPlan{}, nil
		},
	})
	var appErr *temporal.ApplicationError
	if assert.ErrorAs(t, err, &appErr) {
		assert.Equal(t, agent.ErrorTypeConfig, appErr.Type())
		assert.True(t, appErr.NonRetryable())
		assert.Contains(t, appErr.Error(), "turn 0 on turn 1")
	}
	assert.Equal(t, 0, fake.CallCount())
	require.NotNil(t, res)
	if assert.NotEmpty(t, res.Messages) {
		assert.Equal(t, "hi", res.Messages[len(res.Messages)-1].Text())
	}
}

func TestPrepareModelCall_InvalidToolFails(t *testing.T) {
	tests := []struct {
		name string
		tool tool.Tool
		want string
	}{
		{name: "nil", tool: nil, want: "tool 1 is nil"},
		{name: "unnamed", tool: unnamedTool{}, want: "a tool has no name"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			fake := agenttest.NewFakeProvider(agenttest.Says("unreached"))
			a, err := agent.NewAgent("assistant", "test-model", agent.WithTools(weatherTool(t)))
			require.NoError(t, err)

			_, err = runPrepared(t, fake, a, "hi", agent.RunOptions{
				PrepareModelCall: func(_ workflow.Context, p agent.ModelCallPlan) (agent.ModelCallPlan, error) {
					p.Tools = append(p.Tools, tt.tool)
					return p, nil
				},
			})
			var appErr *temporal.ApplicationError
			if assert.ErrorAs(t, err, &appErr) {
				assert.Equal(t, agent.ErrorTypeConfig, appErr.Type())
				assert.True(t, appErr.NonRetryable())
				assert.Contains(t, appErr.Error(), tt.want)
			}
			assert.Equal(t, 0, fake.CallCount())
		})
	}
}

func TestPrepareModelCall_ToolsPersistForTheRun(t *testing.T) {
	fake := agenttest.NewFakeProvider(
		agenttest.CallsTool("echo", `{"text":"hi"}`),
		agenttest.Says("done"),
	)
	a, err := agent.NewAgent("assistant", "test-model", agent.WithTools(weatherTool(t)))
	require.NoError(t, err)
	echo := echoTool(t)

	var plans []agent.ModelCallPlan
	_, err = runPrepared(t, fake, a, "echo hi", agent.RunOptions{
		PrepareModelCall: func(_ workflow.Context, p agent.ModelCallPlan) (agent.ModelCallPlan, error) {
			plans = append(plans, p)
			if p.Turn == 1 {
				p.Tools = append(p.Tools, echo)
			}
			return p, nil
		},
	})
	require.NoError(t, err)

	calls := fake.Calls()
	if assert.Len(t, calls, 2) {
		assert.Equal(t, []string{"get_weather", "echo"}, defNames(calls[0].Tools))
		assert.Equal(t, []string{"get_weather", "echo"}, defNames(calls[1].Tools))
		last := calls[1].Messages[len(calls[1].Messages)-1]
		assert.Equal(t, "echo:hi", last.ToolResultText())
	}
	if assert.Len(t, plans, 2) {
		assert.Len(t, plans[1].Tools, 2)
	}
}

func TestPrepareModelCall_AppendKeptInResult(t *testing.T) {
	fake := agenttest.NewFakeProvider(
		agenttest.CallsTool("get_weather", `{"city":"Ghent"}`),
		agenttest.Says("done"),
	)
	a, err := agent.NewAgent("assistant", "test-model", agent.WithTools(weatherTool(t)))
	require.NoError(t, err)

	res, err := runPrepared(t, fake, a, "Weather in Ghent?", agent.RunOptions{
		PrepareModelCall: func(_ workflow.Context, p agent.ModelCallPlan) (agent.ModelCallPlan, error) {
			if p.Turn == 2 {
				p.Append = []model.Message{model.UserMessage("nudge")}
			}
			return p, nil
		},
	})
	require.NoError(t, err)

	calls := fake.Calls()
	if assert.Len(t, calls, 2) {
		last := calls[1].Messages[len(calls[1].Messages)-1]
		assert.Equal(t, model.RoleUser, last.Role)
		assert.Equal(t, "nudge", last.Text())
	}
	n := len(res.Messages)
	if assert.GreaterOrEqual(t, n, 2) {
		assert.Equal(t, model.RoleUser, res.Messages[n-2].Role)
		assert.Equal(t, "nudge", res.Messages[n-2].Text())
		assert.Equal(t, model.RoleAssistant, res.Messages[n-1].Role)
		assert.Equal(t, "done", res.Messages[n-1].Text())
	}
}

func TestPrepareModelCall_RunsBeforeContinuation(t *testing.T) {
	fake := agenttest.NewFakeProvider(
		agenttest.Truncated("par"),
		agenttest.Says("tial"),
	)
	a, err := agent.NewAgent("assistant", "test-model", agent.WithContinueOnLength(1))
	require.NoError(t, err)

	var turns []int
	res, err := runPrepared(t, fake, a, "go", agent.RunOptions{
		PrepareModelCall: func(_ workflow.Context, p agent.ModelCallPlan) (agent.ModelCallPlan, error) {
			turns = append(turns, p.Turn)
			return p, nil
		},
	})
	require.NoError(t, err)
	assert.Equal(t, []int{1, 2}, turns)
	assert.Equal(t, "partial", res.Output)
}

func TestPrepareModelCall_UnknownAdvertisedNameFails(t *testing.T) {
	fake := agenttest.NewFakeProvider(agenttest.Says("unreached"))
	a, err := agent.NewAgent("assistant", "test-model", agent.WithTools(weatherTool(t)))
	require.NoError(t, err)

	_, err = runPrepared(t, fake, a, "hi", agent.RunOptions{
		PrepareModelCall: func(_ workflow.Context, p agent.ModelCallPlan) (agent.ModelCallPlan, error) {
			p.Advertised = []string{"nope"}
			return p, nil
		},
	})
	var appErr *temporal.ApplicationError
	if assert.ErrorAs(t, err, &appErr) {
		assert.Equal(t, agent.ErrorTypeConfig, appErr.Type())
		assert.True(t, appErr.NonRetryable())
		assert.Contains(t, appErr.Error(), "nope")
	}
	assert.Equal(t, 0, fake.CallCount())
}

func TestPrepareModelCall_DuplicateToolFails(t *testing.T) {
	fake := agenttest.NewFakeProvider(agenttest.Says("unreached"))
	a, err := agent.NewAgent("assistant", "test-model", agent.WithTools(weatherTool(t)))
	require.NoError(t, err)
	dup := weatherTool(t)

	_, err = runPrepared(t, fake, a, "hi", agent.RunOptions{
		PrepareModelCall: func(_ workflow.Context, p agent.ModelCallPlan) (agent.ModelCallPlan, error) {
			p.Tools = append(p.Tools, dup)
			return p, nil
		},
	})
	var appErr *temporal.ApplicationError
	if assert.ErrorAs(t, err, &appErr) {
		assert.Equal(t, agent.ErrorTypeConfig, appErr.Type())
		assert.True(t, appErr.NonRetryable())
		assert.Contains(t, appErr.Error(), `duplicate tool name "get_weather"`)
	}
	assert.Equal(t, 0, fake.CallCount())
}

func TestPrepareModelCall_ErrorAbortsRun(t *testing.T) {
	fake := agenttest.NewFakeProvider(
		agenttest.CallsTool("get_weather", `{"city":"Ghent"}`),
		agenttest.Says("done"),
	)
	a, err := agent.NewAgent("assistant", "test-model", agent.WithTools(weatherTool(t)))
	require.NoError(t, err)

	res, err := runPrepared(t, fake, a, "Weather in Ghent?", agent.RunOptions{
		PrepareModelCall: func(_ workflow.Context, p agent.ModelCallPlan) (agent.ModelCallPlan, error) {
			if p.Turn == 2 {
				return p, errors.New("stop here")
			}
			return p, nil
		},
	})
	if assert.Error(t, err) {
		assert.Contains(t, err.Error(), "stop here")
	}
	assert.Equal(t, 1, fake.CallCount())
	require.NotNil(t, res)
	if assert.NotEmpty(t, res.Messages) {
		last := res.Messages[len(res.Messages)-1]
		assert.Equal(t, model.RoleTool, last.Role)
		assert.Equal(t, "18°C in Ghent", last.ToolResultText())
	}
}

func TestPrepareModelCall_SettingsMutationDoesNotLeak(t *testing.T) {
	fake := agenttest.NewFakeProvider(
		agenttest.CallsTool("get_weather", `{"city":"Ghent"}`),
		agenttest.Says("done"),
	)
	a, err := agent.NewAgent("assistant", "test-model",
		agent.WithTools(weatherTool(t)),
		agent.WithSettings(model.Settings{Temperature: model.Ptr(0.7)}))
	require.NoError(t, err)

	var temps []float64
	_, err = runPrepared(t, fake, a, "Weather in Ghent?", agent.RunOptions{
		PrepareModelCall: func(_ workflow.Context, p agent.ModelCallPlan) (agent.ModelCallPlan, error) {
			temps = append(temps, *p.Settings.Temperature)
			if p.Turn == 1 {
				*p.Settings.Temperature = 0
			}
			return p, nil
		},
	})
	require.NoError(t, err)

	if assert.Len(t, temps, 2) {
		assert.Equal(t, 0.7, temps[1])
	}
	calls := fake.Calls()
	if assert.Len(t, calls, 2) {
		assert.Equal(t, model.Ptr(0.0), calls[0].Settings.Temperature)
		assert.Equal(t, model.Ptr(0.7), calls[1].Settings.Temperature)
	}
}
