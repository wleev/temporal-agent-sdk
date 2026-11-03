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
)

func answerSchema() *model.OutputSchema {
	return &model.OutputSchema{Name: "answer", Schema: []byte(`{"type":"object"}`), Strict: true}
}

func runSchema() *model.OutputSchema {
	return &model.OutputSchema{Name: "run", Schema: []byte(`{"type":"object"}`), Strict: true}
}

func runTypedForecast(t *testing.T, fake *agenttest.FakeProvider, a *agent.Agent, ro agent.RunOptions) (*agent.TypedResult[forecast], error) {
	t.Helper()
	env := newEnv(t, fake)
	env.ExecuteWorkflow(func(ctx workflow.Context) (*agent.TypedResult[forecast], error) {
		return agent.RunTypedWith[forecast](ctx, a, "weather in Ghent?", ro)
	})
	require.True(t, env.IsWorkflowCompleted())
	if err := env.GetWorkflowError(); err != nil {
		return nil, err
	}
	var res agent.TypedResult[forecast]
	require.NoError(t, env.GetWorkflowResult(&res))
	return &res, nil
}

func TestOutputSchemaPerCall_AppliesToOneCall(t *testing.T) {
	fake := agenttest.NewFakeProvider(
		agenttest.CallsTool("fast", `{"text":"x"}`),
		agenttest.Says(`{"answer":"y"}`),
	)
	a, err := agent.NewAgent("assistant", "test-model",
		agent.WithTools(newChoiceTool(t, "fast")))
	require.NoError(t, err)

	res, err := runPrepared(t, fake, a, "hi", agent.RunOptions{
		PrepareModelCall: func(_ workflow.Context, p agent.ModelCallPlan) (agent.ModelCallPlan, error) {
			if p.Turn == 2 {
				p.OutputSchema = answerSchema()
			}
			return p, nil
		},
	})
	require.NoError(t, err)

	calls := fake.Calls()
	if assert.Len(t, calls, 2) {
		assert.Nil(t, calls[0].OutputSchema)
		if assert.NotNil(t, calls[1].OutputSchema) {
			assert.Equal(t, "answer", calls[1].OutputSchema.Name)
		}
	}
	assert.JSONEq(t, `{"answer":"y"}`, string(res.StructuredOutput))
}

func TestOutputSchemaPerCall_OverridesTheRunSchema(t *testing.T) {
	fake := agenttest.NewFakeProvider(
		agenttest.CallsTool("fast", `{"text":"x"}`),
		agenttest.Says(`{"answer":"y"}`),
	)
	a, err := agent.NewAgent("assistant", "test-model",
		agent.WithTools(newChoiceTool(t, "fast")))
	require.NoError(t, err)

	res, err := runPrepared(t, fake, a, "hi", agent.RunOptions{
		OutputSchema: runSchema(),
		PrepareModelCall: func(_ workflow.Context, p agent.ModelCallPlan) (agent.ModelCallPlan, error) {
			if p.Turn == 2 {
				p.OutputSchema = answerSchema()
			}
			return p, nil
		},
	})
	require.NoError(t, err)

	calls := fake.Calls()
	if assert.Len(t, calls, 2) {
		if assert.NotNil(t, calls[0].OutputSchema) {
			assert.Equal(t, "run", calls[0].OutputSchema.Name)
		}
		if assert.NotNil(t, calls[1].OutputSchema) {
			assert.Equal(t, "answer", calls[1].OutputSchema.Name)
		}
	}
	assert.JSONEq(t, `{"answer":"y"}`, string(res.StructuredOutput))
}

func TestRunTyped_FailsWhenTheFinalCallUsedAnotherSchema(t *testing.T) {
	fake := agenttest.NewFakeProvider(agenttest.Says(`{"city":"Ghent","temp":18}`))
	a, err := agent.NewAgent("weather", "test-model")
	require.NoError(t, err)

	_, err = runTypedForecast(t, fake, a, agent.RunOptions{
		PrepareModelCall: func(_ workflow.Context, p agent.ModelCallPlan) (agent.ModelCallPlan, error) {
			p.OutputSchema = answerSchema()
			return p, nil
		},
	})
	var appErr *temporal.ApplicationError
	if assert.ErrorAs(t, err, &appErr) {
		assert.Equal(t, agent.ErrorTypeConfig, appErr.Type())
		assert.True(t, appErr.NonRetryable())
		assert.Contains(t, appErr.Error(), "answer")
	}
	assert.Equal(t, 1, fake.CallCount())
}

func TestRunTyped_PerCallSchemaOnAToolTurn(t *testing.T) {
	fake := agenttest.NewFakeProvider(
		agenttest.CallsTool("get_weather", `{"city":"Ghent"}`),
		agenttest.Says(`{"city":"Ghent","temp":18}`),
	)
	a, err := agent.NewAgent("weather", "test-model", agent.WithTools(weatherTool(t)))
	require.NoError(t, err)

	res, err := runTypedForecast(t, fake, a, agent.RunOptions{
		PrepareModelCall: func(_ workflow.Context, p agent.ModelCallPlan) (agent.ModelCallPlan, error) {
			if p.Turn == 1 {
				p.OutputSchema = answerSchema()
			}
			return p, nil
		},
	})
	if assert.NoError(t, err) {
		assert.Equal(t, 18, res.Value.Temp)
	}
	calls := fake.Calls()
	if assert.Len(t, calls, 2) {
		if assert.NotNil(t, calls[0].OutputSchema) {
			assert.Equal(t, "answer", calls[0].OutputSchema.Name)
		}
		if assert.NotNil(t, calls[1].OutputSchema) {
			assert.Equal(t, "output", calls[1].OutputSchema.Name)
		}
	}
}

func TestOutputSchemaPerCall_BlocksContinuation(t *testing.T) {
	fake := agenttest.NewFakeProvider(agenttest.Truncated(`{"answer":"y"}`))
	a, err := agent.NewAgent("assistant", "test-model", agent.WithContinueOnLength(2))
	require.NoError(t, err)

	res, err := runPrepared(t, fake, a, "hi", agent.RunOptions{
		PrepareModelCall: func(_ workflow.Context, p agent.ModelCallPlan) (agent.ModelCallPlan, error) {
			p.OutputSchema = answerSchema()
			return p, nil
		},
	})
	require.NoError(t, err)
	assert.Equal(t, 1, fake.CallCount())
	assert.Equal(t, model.FinishLength, res.FinishReason)
}

func TestOutputSchemaPerCall_RunSchemaApplies(t *testing.T) {
	fake := agenttest.NewFakeProvider(agenttest.Says(`{"answer":"y"}`))
	a, err := agent.NewAgent("assistant", "test-model")
	require.NoError(t, err)

	res, err := runPrepared(t, fake, a, "hi", agent.RunOptions{OutputSchema: answerSchema()})
	require.NoError(t, err)
	if assert.NotNil(t, fake.LastCall().OutputSchema) {
		assert.Equal(t, "answer", fake.LastCall().OutputSchema.Name)
	}
	assert.JSONEq(t, `{"answer":"y"}`, string(res.StructuredOutput))
}
