package agent_test

import (
	"context"
	"sync"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.temporal.io/sdk/activity"
	"go.temporal.io/sdk/testsuite"
	"go.temporal.io/sdk/workflow"

	"github.com/wleev/temporal-agent-sdk/agent"
	"github.com/wleev/temporal-agent-sdk/agenttest"
	"github.com/wleev/temporal-agent-sdk/model"
)

// A streaming sink lives in the activity worker process, so a package-level
// collector stands in for an external transport; the test env runs the activity
// in-process.
var streamCollector = struct {
	sync.Mutex
	deltas []model.StreamDelta
}{}

func TestRun_StreamsThroughLoop(t *testing.T) {
	streamCollector.Lock()
	streamCollector.deltas = nil
	streamCollector.Unlock()

	fake := agenttest.NewFakeProvider(agenttest.Says("It is 18°C in Ghent."))
	acts, err := model.NewActivities(fake)
	require.NoError(t, err)
	acts.SetStreamSink(func(context.Context) (model.StreamSink, error) {
		return sink{}, nil
	})

	var s testsuite.WorkflowTestSuite
	env := s.NewTestWorkflowEnvironment()
	env.RegisterActivityWithOptions(acts.InvokeModel, activity.RegisterOptions{Name: model.InvokeModelActivity})

	a, err := agent.NewAgent("assistant", "test-model", agent.WithStreaming())
	require.NoError(t, err)
	env.ExecuteWorkflow(func(ctx workflow.Context) (*agent.Result, error) {
		return agent.Run(ctx, a, "weather?")
	})

	require.True(t, env.IsWorkflowCompleted())
	require.NoError(t, env.GetWorkflowError())

	streamCollector.Lock()
	defer streamCollector.Unlock()
	var text string
	for _, d := range streamCollector.deltas {
		text += d.Text
	}
	assert.Contains(t, text, "It is 18°C in Ghent.", "the final answer streamed through the sink")
}

type sink struct{}

func (sink) OnDelta(_ context.Context, d model.StreamDelta) error {
	streamCollector.Lock()
	streamCollector.deltas = append(streamCollector.deltas, d)
	streamCollector.Unlock()
	return nil
}
