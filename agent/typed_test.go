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

type forecast struct {
	City string `json:"city" jsonschema_description:"City name"`
	Temp int    `json:"temp" jsonschema_description:"Temperature in Celsius"`
}

func TestRunTyped_DecodesStructuredAnswer(t *testing.T) {
	fake := agenttest.NewFakeProvider(agenttest.Says(`{"city":"Ghent","temp":18}`))
	env := newEnv(t, fake)

	a, err := agent.NewAgent("weather", "test-model")
	require.NoError(t, err)
	env.ExecuteWorkflow(func(ctx workflow.Context) (*agent.TypedResult[forecast], error) {
		return agent.RunTyped[forecast](ctx, a, "weather in Ghent?")
	})
	require.True(t, env.IsWorkflowCompleted())
	require.NoError(t, env.GetWorkflowError())

	var res agent.TypedResult[forecast]
	require.NoError(t, env.GetWorkflowResult(&res))
	assert.Equal(t, "Ghent", res.Value.City)
	assert.Equal(t, 18, res.Value.Temp)
	assert.Equal(t, 1, res.Turns)

	// The request carried the reflected output schema.
	sent := fake.LastCall()
	if assert.NotNil(t, sent.OutputSchema) {
		assert.Contains(t, string(sent.OutputSchema.Schema), "city")
		assert.Contains(t, string(sent.OutputSchema.Schema), "temp")
	}
}

// Tools run in intermediate turns; only the terminal answer is structured.
func TestRunTyped_ToolsThenStructured(t *testing.T) {
	fake := agenttest.NewFakeProvider(
		agenttest.CallsTool("get_weather", `{"city":"Ghent"}`),
		agenttest.Says(`{"city":"Ghent","temp":18}`),
	)
	env := newEnv(t, fake)

	a, err := agent.NewAgent("weather", "test-model", agent.WithTools(weatherTool(t)))
	require.NoError(t, err)
	env.ExecuteWorkflow(func(ctx workflow.Context) (*agent.TypedResult[forecast], error) {
		return agent.RunTyped[forecast](ctx, a, "weather in Ghent?")
	})
	require.NoError(t, env.GetWorkflowError())

	var res agent.TypedResult[forecast]
	require.NoError(t, env.GetWorkflowResult(&res))
	assert.Equal(t, 18, res.Value.Temp)
	assert.Equal(t, 2, res.Turns)

	// The output schema was sent on every turn, including the tool-call turn.
	assert.NotNil(t, fake.Calls()[0].OutputSchema)
}

// A terminal plain-text answer when a schema was requested is a permanent error.
func TestRunTyped_MissingStructuredOutputErrors(t *testing.T) {
	// The fake surfaces content as structured output when a schema is set, so an
	// empty-content terminal message simulates the model ending on non-structured
	// output.
	fake := agenttest.NewFakeProvider(agenttest.Response{
		Message: model.AssistantMessage(""),
	})
	env := newEnv(t, fake)

	a, err := agent.NewAgent("x", "m")
	require.NoError(t, err)
	env.ExecuteWorkflow(func(ctx workflow.Context) (*agent.TypedResult[forecast], error) {
		return agent.RunTyped[forecast](ctx, a, "hi")
	})
	require.True(t, env.IsWorkflowCompleted())
	assert.ErrorContains(t, env.GetWorkflowError(), "no structured output")
}

// A recursive output type is rejected before any model call.
func TestRunTyped_RejectsBadType(t *testing.T) {
	fake := agenttest.NewFakeProvider()
	env := newEnv(t, fake)

	a, err := agent.NewAgent("x", "m")
	require.NoError(t, err)
	env.ExecuteWorkflow(func(ctx workflow.Context) (*agent.TypedResult[recursiveOut], error) {
		return agent.RunTyped[recursiveOut](ctx, a, "hi")
	})
	require.True(t, env.IsWorkflowCompleted())
	assert.ErrorContains(t, env.GetWorkflowError(), "not a valid schema")
	assert.Equal(t, 0, fake.CallCount(), "a bad output type must fail before calling the model")
}

type recursiveOut struct {
	Next *recursiveOut `json:"next"`
}
