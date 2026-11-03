package agent_test

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.temporal.io/sdk/workflow"

	"github.com/wleev/temporal-agent-sdk/agent"
	"github.com/wleev/temporal-agent-sdk/agenttest"
)

func metadataFake() *agenttest.FakeProvider {
	return agenttest.NewFakeProvider(
		agenttest.CallsTool("get_weather", `{"city":"Ghent"}`),
		agenttest.Says("done"),
	)
}

func TestPrepareModelCall_MetadataIsPerCall(t *testing.T) {
	fake := metadataFake()
	a, err := agent.NewAgent("assistant", "test-model", agent.WithTools(weatherTool(t)))
	require.NoError(t, err)

	ro := agent.RunOptions{
		ProviderMetadata: map[string]string{"region": "eu", "feature": "support"},
		PrepareModelCall: func(_ workflow.Context, p agent.ModelCallPlan) (agent.ModelCallPlan, error) {
			if p.Turn == 1 {
				p.Metadata = map[string]string{"trace": "on", "region": "us"}
			}
			return p, nil
		},
	}
	_, err = runPrepared(t, fake, a, "Weather in Ghent?", ro)
	require.NoError(t, err)

	if assert.Equal(t, 2, fake.CallCount()) {
		calls := fake.Calls()
		assert.Equal(t, map[string]string{"region": "us", "feature": "support", "trace": "on"}, calls[0].Metadata)
		assert.Equal(t, map[string]string{"region": "eu", "feature": "support"}, calls[1].Metadata)
	}
	assert.Equal(t, map[string]string{"region": "eu", "feature": "support"}, ro.ProviderMetadata)
}

func TestPrepareModelCall_NilMetadataKeepsRunMetadata(t *testing.T) {
	meta := map[string]string{"region": "eu", "feature": "support"}
	a, err := agent.NewAgent("assistant", "test-model", agent.WithTools(weatherTool(t)))
	require.NoError(t, err)

	hooked := metadataFake()
	_, err = runPrepared(t, hooked, a, "Weather in Ghent?", agent.RunOptions{
		ProviderMetadata: meta,
		PrepareModelCall: func(_ workflow.Context, p agent.ModelCallPlan) (agent.ModelCallPlan, error) {
			return p, nil
		},
	})
	require.NoError(t, err)

	plain := metadataFake()
	_, err = runPrepared(t, plain, a, "Weather in Ghent?", agent.RunOptions{ProviderMetadata: meta})
	require.NoError(t, err)

	if assert.Equal(t, 2, hooked.CallCount()) {
		for _, c := range hooked.Calls() {
			assert.Equal(t, meta, c.Metadata)
		}
		got, err := json.Marshal(hooked.Calls()[0])
		require.NoError(t, err)
		want, err := json.Marshal(plain.Calls()[0])
		require.NoError(t, err)
		assert.JSONEq(t, string(want), string(got))
	}
}

func TestPrepareModelCall_MetadataWithoutRunMetadata(t *testing.T) {
	fake := metadataFake()
	a, err := agent.NewAgent("assistant", "test-model", agent.WithTools(weatherTool(t)))
	require.NoError(t, err)

	_, err = runPrepared(t, fake, a, "Weather in Ghent?", agent.RunOptions{
		PrepareModelCall: func(_ workflow.Context, p agent.ModelCallPlan) (agent.ModelCallPlan, error) {
			if p.Turn == 1 {
				p.Metadata = map[string]string{"live": "off"}
			}
			return p, nil
		},
	})
	require.NoError(t, err)

	if assert.Equal(t, 2, fake.CallCount()) {
		assert.Equal(t, map[string]string{"live": "off"}, fake.Calls()[0].Metadata)
		assert.Nil(t, fake.Calls()[1].Metadata)
	}
}
