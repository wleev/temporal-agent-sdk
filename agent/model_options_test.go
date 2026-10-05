package agent

import (
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.temporal.io/sdk/workflow"
)

func TestNewAgent_DefaultsTheModelHeartbeatTimeout(t *testing.T) {
	a, err := NewAgent("assistant", "test-model")
	require.NoError(t, err)

	assert.Equal(t, DefaultModelTimeout, a.modelActivityOptions.StartToCloseTimeout)
	assert.Equal(t, DefaultModelHeartbeatTimeout, a.modelActivityOptions.HeartbeatTimeout)
}

func TestNewAgent_CapsTheDefaultHeartbeatTimeoutAtTheCallTimeout(t *testing.T) {
	a, err := NewAgent("assistant", "test-model", WithModelActivityOptions(workflow.ActivityOptions{
		StartToCloseTimeout: 10 * time.Second,
	}))
	require.NoError(t, err)

	assert.Equal(t, 10*time.Second, a.modelActivityOptions.HeartbeatTimeout)
}

func TestNewAgent_KeepsAConfiguredModelHeartbeatTimeout(t *testing.T) {
	a, err := NewAgent("assistant", "test-model", WithModelActivityOptions(workflow.ActivityOptions{
		StartToCloseTimeout: 5 * time.Minute,
		HeartbeatTimeout:    45 * time.Second,
	}))
	require.NoError(t, err)

	assert.Equal(t, 45*time.Second, a.modelActivityOptions.HeartbeatTimeout)
}
