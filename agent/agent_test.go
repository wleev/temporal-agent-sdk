package agent_test

import (
	"encoding/json"
	"errors"
	"fmt"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	"go.temporal.io/sdk/activity"
	"go.temporal.io/sdk/testsuite"
	"go.temporal.io/sdk/workflow"

	"github.com/wleev/temporal-agent-sdk/agent"
	"github.com/wleev/temporal-agent-sdk/agenttest"
	"github.com/wleev/temporal-agent-sdk/model"
	"github.com/wleev/temporal-agent-sdk/tool"
)

type weatherIn struct {
	City string `json:"city" jsonschema_description:"City to look up"`
}

type echoIn struct {
	Text string `json:"text" jsonschema_description:"Text to echo"`
}

// newEnv builds a test environment with the fake model activity registered.
func newEnv(t *testing.T, fake *agenttest.FakeProvider) *testsuite.TestWorkflowEnvironment {
	t.Helper()
	var s testsuite.WorkflowTestSuite
	env := s.NewTestWorkflowEnvironment()
	env.RegisterActivityWithOptions(
		agenttest.MustActivities(fake).InvokeModel,
		activity.RegisterOptions{Name: model.InvokeModelActivity},
	)
	return env
}

func weatherTool(t *testing.T) tool.Tool {
	t.Helper()
	tl, err := tool.New("get_weather", "Look up the weather",
		func(_ workflow.Context, in weatherIn) (string, error) {
			return fmt.Sprintf("18°C in %s", in.City), nil
		})
	require.NoError(t, err)
	return tl
}

// mustAgent panics if NewAgent errored — the tests build valid agents, so an
// error is a bug in the test, not a case to handle.
func mustAgent(a *agent.Agent, err error) *agent.Agent {
	if err != nil {
		panic(err)
	}
	return a
}

func TestRun_PlainAnswer(t *testing.T) {
	fake := agenttest.NewFakeProvider(agenttest.Says("Hello there."))
	env := newEnv(t, fake)

	env.ExecuteWorkflow(func(ctx workflow.Context) (*agent.Result, error) {
		return agent.Run(ctx, mustAgent(agent.NewAgent("assistant", "test-model",
			agent.WithInstructions("Be brief."))), "Hi")
	})

	require.True(t, env.IsWorkflowCompleted())
	require.NoError(t, env.GetWorkflowError())

	var res agent.Result
	require.NoError(t, env.GetWorkflowResult(&res))
	require.Equal(t, "Hello there.", res.Output)
	require.Equal(t, 1, res.Turns)
	require.Equal(t, int64(15), res.Usage.TotalTokens)

	// Instructions become the system message, and the user input follows.
	sent := fake.LastCall()
	require.Equal(t, model.RoleSystem, sent.Messages[0].Role)
	require.Equal(t, "Be brief.", sent.Messages[0].Text())
	require.Equal(t, model.RoleUser, sent.Messages[1].Role)
	require.Equal(t, "Hi", sent.Messages[1].Text())
}

func TestRun_ToolCall(t *testing.T) {
	fake := agenttest.NewFakeProvider(
		agenttest.CallsTool("get_weather", `{"city":"Ghent"}`),
		agenttest.Says("It is 18°C in Ghent."),
	)
	env := newEnv(t, fake)

	env.ExecuteWorkflow(func(ctx workflow.Context) (*agent.Result, error) {
		return agent.Run(ctx, mustAgent(agent.NewAgent("assistant", "test-model",
			agent.WithTools(weatherTool(t)))), "Weather in Ghent?")
	})

	require.True(t, env.IsWorkflowCompleted())
	require.NoError(t, env.GetWorkflowError())

	var res agent.Result
	require.NoError(t, env.GetWorkflowResult(&res))
	require.Equal(t, "It is 18°C in Ghent.", res.Output)
	require.Equal(t, 2, res.Turns)
	require.Equal(t, int64(30), res.Usage.TotalTokens, "usage should accumulate across turns")

	// The tool's result must reach the model on the second call, tied to its
	// call ID.
	second := fake.Calls()[1]
	last := second.Messages[len(second.Messages)-1]
	require.Equal(t, model.RoleTool, last.Role)
	require.Equal(t, "18°C in Ghent", last.ToolResultText())
	require.Equal(t, "call-1", last.ToolResultID())
}

func TestRun_ToolSchemaSentToModel(t *testing.T) {
	fake := agenttest.NewFakeProvider(agenttest.Says("done"))
	env := newEnv(t, fake)

	env.ExecuteWorkflow(func(ctx workflow.Context) (*agent.Result, error) {
		return agent.Run(ctx, mustAgent(agent.NewAgent("assistant", "test-model",
			agent.WithTools(weatherTool(t)))), "hi")
	})
	require.NoError(t, env.GetWorkflowError())

	tools := fake.LastCall().Tools
	require.Len(t, tools, 1)
	require.Equal(t, "get_weather", tools[0].Name)
	require.Equal(t, "Look up the weather", tools[0].Description)

	// InputSchema is typed any; it holds the reflector's raw JSON here, and
	// map[string]any once it has crossed an activity boundary.
	raw, err := json.Marshal(tools[0].InputSchema)
	require.NoError(t, err)
	require.JSONEq(t,
		`{"type":"object","properties":{"city":{"type":"string","description":"City to look up"}},`+
			`"required":["city"],"additionalProperties":false}`,
		string(raw))
}

// Tool results must be ordered by the model's call order, not by whichever tool
// finished first. Ordering by completion would replay differently than it ran,
// which is the classic Temporal nondeterminism bug.
func TestRun_ParallelToolCallsPreserveOrder(t *testing.T) {
	fake := agenttest.NewFakeProvider(
		agenttest.CallsTools(
			agenttest.ToolCall{ID: "a", Name: "slow", Args: `{"text":"first"}`},
			agenttest.ToolCall{ID: "b", Name: "fast", Args: `{"text":"second"}`},
			agenttest.ToolCall{ID: "c", Name: "slow", Args: `{"text":"third"}`},
		),
		agenttest.Says("all done"),
	)
	env := newEnv(t, fake)

	// "slow" sleeps in workflow time so it finishes after "fast", making
	// completion order differ from call order.
	slow, err := tool.New("slow", "A slow tool",
		func(ctx workflow.Context, in echoIn) (string, error) {
			if err := workflow.Sleep(ctx, time.Minute); err != nil {
				return "", err
			}
			return "slow:" + in.Text, nil
		})
	require.NoError(t, err)

	fast, err := tool.New("fast", "A fast tool",
		func(_ workflow.Context, in echoIn) (string, error) {
			return "fast:" + in.Text, nil
		})
	require.NoError(t, err)

	env.ExecuteWorkflow(func(ctx workflow.Context) (*agent.Result, error) {
		return agent.Run(ctx, mustAgent(agent.NewAgent("assistant", "test-model",
			agent.WithTools(slow, fast))), "go")
	})

	require.True(t, env.IsWorkflowCompleted())
	require.NoError(t, env.GetWorkflowError())

	second := fake.Calls()[1]
	toolMsgs := filterRole(second.Messages, model.RoleTool)
	require.Len(t, toolMsgs, 3)

	// Call order: a(slow), b(fast), c(slow) — even though b completed first.
	require.Equal(t, []string{"a", "b", "c"},
		[]string{toolMsgs[0].ToolResultID(), toolMsgs[1].ToolResultID(), toolMsgs[2].ToolResultID()},
		"tool results must follow call order, not completion order")
	require.Equal(t, "slow:first", toolMsgs[0].ToolResultText())
	require.Equal(t, "fast:second", toolMsgs[1].ToolResultText())
	require.Equal(t, "slow:third", toolMsgs[2].ToolResultText())
}

// Independent tools should overlap rather than serialize. Three one-minute
// sleeps dispatched together should advance the clock by about a minute, not
// three.
func TestRun_ParallelToolCallsAreConcurrent(t *testing.T) {
	fake := agenttest.NewFakeProvider(
		agenttest.CallsTools(
			agenttest.ToolCall{ID: "a", Name: "slow", Args: `{"text":"1"}`},
			agenttest.ToolCall{ID: "b", Name: "slow", Args: `{"text":"2"}`},
			agenttest.ToolCall{ID: "c", Name: "slow", Args: `{"text":"3"}`},
		),
		agenttest.Says("done"),
	)
	env := newEnv(t, fake)

	slow, err := tool.New("slow", "A slow tool",
		func(ctx workflow.Context, in echoIn) (string, error) {
			if err := workflow.Sleep(ctx, time.Minute); err != nil {
				return "", err
			}
			return in.Text, nil
		})
	require.NoError(t, err)

	var elapsed time.Duration
	env.ExecuteWorkflow(func(ctx workflow.Context) (*agent.Result, error) {
		start := workflow.Now(ctx)
		res, err := agent.Run(ctx, mustAgent(agent.NewAgent("assistant", "test-model",
			agent.WithTools(slow))), "go")
		elapsed = workflow.Now(ctx).Sub(start)
		return res, err
	})

	require.True(t, env.IsWorkflowCompleted())
	require.NoError(t, env.GetWorkflowError())

	// The lower bound proves the sleeps actually ran, so the upper bound is
	// evidence of overlap rather than of nothing having happened.
	require.GreaterOrEqual(t, elapsed, time.Minute,
		"the tools should have slept at all (got %s)", elapsed)
	require.Less(t, elapsed, 2*time.Minute,
		"three 1-minute tools dispatched together should overlap, not serialize (got %s)", elapsed)
}

// A tool error is information the model can act on, so it goes back as a tool
// result rather than failing the workflow and discarding the conversation.
func TestRun_ToolErrorGoesBackToModel(t *testing.T) {
	fake := agenttest.NewFakeProvider(
		agenttest.CallsTool("boom", `{"text":"x"}`),
		agenttest.Says("I could not do that."),
	)
	env := newEnv(t, fake)

	boom, err := tool.New("boom", "Always fails",
		func(_ workflow.Context, _ echoIn) (string, error) {
			return "", errors.New("kaboom")
		})
	require.NoError(t, err)

	env.ExecuteWorkflow(func(ctx workflow.Context) (*agent.Result, error) {
		return agent.Run(ctx, mustAgent(agent.NewAgent("assistant", "test-model",
			agent.WithTools(boom))), "go")
	})

	require.True(t, env.IsWorkflowCompleted())
	require.NoError(t, env.GetWorkflowError(), "a tool error must not fail the workflow")

	second := fake.Calls()[1]
	last := second.Messages[len(second.Messages)-1]
	require.Equal(t, model.RoleTool, last.Role)
	require.Contains(t, last.ToolResultText(), "kaboom")
}

// A hallucinated tool name is likewise recoverable: the model is told what does
// exist and usually corrects itself.
func TestRun_UnknownToolGoesBackToModel(t *testing.T) {
	fake := agenttest.NewFakeProvider(
		agenttest.CallsTool("nonexistent", `{}`),
		agenttest.Says("Sorry, let me try another way."),
	)
	env := newEnv(t, fake)

	env.ExecuteWorkflow(func(ctx workflow.Context) (*agent.Result, error) {
		return agent.Run(ctx, mustAgent(agent.NewAgent("assistant", "test-model",
			agent.WithTools(weatherTool(t)))), "go")
	})

	require.True(t, env.IsWorkflowCompleted())
	require.NoError(t, env.GetWorkflowError())

	second := fake.Calls()[1]
	last := second.Messages[len(second.Messages)-1]
	require.Contains(t, last.ToolResultText(), "no tool named")
	require.Contains(t, last.ToolResultText(), "get_weather", "the model should be told what is available")
}

func TestRun_MaxTurnsExceeded(t *testing.T) {
	fake := agenttest.NewFakeProvider(
		agenttest.CallsTool("get_weather", `{"city":"A"}`),
		agenttest.CallsTool("get_weather", `{"city":"B"}`),
		agenttest.CallsTool("get_weather", `{"city":"C"}`),
	)
	env := newEnv(t, fake)

	env.ExecuteWorkflow(func(ctx workflow.Context) (*agent.Result, error) {
		return agent.Run(ctx, mustAgent(agent.NewAgent("assistant", "test-model",
			agent.WithTools(weatherTool(t)), agent.WithMaxTurns(2))), "go")
	})

	require.True(t, env.IsWorkflowCompleted())
	err := env.GetWorkflowError()
	require.Error(t, err)
	require.Contains(t, err.Error(), "exceeded 2 turns")
	require.Equal(t, 2, fake.CallCount(), "the loop must stop at MaxTurns")
}

func TestRun_HistoryRoundTripDoesNotDuplicateSystemMessage(t *testing.T) {
	fake := agenttest.NewFakeProvider(
		agenttest.Says("first"),
		agenttest.Says("second"),
	)
	env := newEnv(t, fake)

	env.ExecuteWorkflow(func(ctx workflow.Context) (*agent.Result, error) {
		a := mustAgent(agent.NewAgent("assistant", "test-model", agent.WithInstructions("Be brief.")))
		s, err := agent.NewSession(ctx)
		if err != nil {
			return nil, err
		}
		first, err := s.Run(ctx, a, "one")
		if err != nil {
			return nil, err
		}
		// Feeding a prior transcript back must not stack system messages.
		return s.RunWith(ctx, a, "two", agent.RunOptions{History: first.Messages})
	})

	require.True(t, env.IsWorkflowCompleted())
	require.NoError(t, env.GetWorkflowError())

	systems := filterRole(fake.Calls()[1].Messages, model.RoleSystem)
	require.Len(t, systems, 1, "Instructions must not be prepended to a history that already has them")
}

// NewAgent rejects an empty name or model at construction, so an invalid agent
// can never reach a Run.
func TestNewAgent_RejectsInvalidConfig(t *testing.T) {
	_, err := agent.NewAgent("", "m")
	require.ErrorContains(t, err, "name must not be empty")

	_, err = agent.NewAgent("a", "")
	require.ErrorContains(t, err, "model must not be empty")

	_, err = agent.NewAgent("a", "m", agent.WithTools(nil))
	require.ErrorContains(t, err, "tool 0 is nil")
}

func TestRun_RejectsDuplicateToolNames(t *testing.T) {
	fake := agenttest.NewFakeProvider()
	env := newEnv(t, fake)

	env.ExecuteWorkflow(func(ctx workflow.Context) (*agent.Result, error) {
		return agent.Run(ctx, mustAgent(agent.NewAgent("assistant", "test-model",
			agent.WithTools(weatherTool(t), weatherTool(t)))), "go")
	})

	require.True(t, env.IsWorkflowCompleted())
	require.ErrorContains(t, env.GetWorkflowError(), `duplicate tool name "get_weather"`)
}

func filterRole(msgs []model.Message, role model.Role) []model.Message {
	var out []model.Message
	for _, m := range msgs {
		if m.Role == role {
			out = append(out, m)
		}
	}
	return out
}
