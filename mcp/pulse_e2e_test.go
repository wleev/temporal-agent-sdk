package mcp_test

import (
	"context"
	"encoding/json"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.temporal.io/sdk/client"
	"go.temporal.io/sdk/temporal"
	"go.temporal.io/sdk/worker"
	"go.temporal.io/sdk/workflow"

	"github.com/wleev/temporal-agent-sdk/mcp"
	"github.com/wleev/temporal-agent-sdk/model"
)

const slowCallHeartbeatTimeout = 3 * time.Second

// slowClient is an [mcp.Client] whose CallTool takes twice
// slowCallHeartbeatTimeout.
type slowClient struct{}

func (slowClient) ListTools(context.Context) ([]*model.Tool, error) { return nil, nil }

func (slowClient) CallTool(ctx context.Context, _ string, _ json.RawMessage) (*model.CallToolResult, error) {
	select {
	case <-time.After(2 * slowCallHeartbeatTimeout):
		return model.TextResult("done"), nil
	case <-ctx.Done():
		return nil, ctx.Err()
	}
}

func (slowClient) Close() error { return nil }

func slowCallWorkflow(ctx workflow.Context) (string, error) {
	ctx = workflow.WithActivityOptions(ctx, workflow.ActivityOptions{
		StartToCloseTimeout: time.Minute,
		HeartbeatTimeout:    slowCallHeartbeatTimeout,
		RetryPolicy:         &temporal.RetryPolicy{MaximumAttempts: 1},
	})
	var res model.CallToolResult
	err := workflow.ExecuteActivity(ctx, mcp.CallToolActivity,
		mcp.CallToolInput{Server: "slow", Tool: "wait"}).Get(ctx, &res)
	if err != nil {
		return "", err
	}
	return model.ResultText(&res), nil
}

// TestCallTool_OutlastsTheHeartbeatTimeout runs a tool call that reports no
// progress and takes twice the activity's HeartbeatTimeout on a Temporal dev
// server, and checks that it succeeds.
func TestCallTool_OutlastsTheHeartbeatTimeout(t *testing.T) {
	c := devServer(t)

	acts, err := mcp.NewActivities(mcp.WithServer("slow", func(context.Context) (mcp.Client, error) { return slowClient{}, nil }))
	require.NoError(t, err)

	const taskQueue = "agentsdk-mcp-pulse"
	w := worker.New(c, taskQueue, worker.Options{})
	w.RegisterWorkflow(slowCallWorkflow)
	acts.RegisterWith(w)
	require.NoError(t, w.Start())
	t.Cleanup(w.Stop)

	ctx, cancel := context.WithTimeout(t.Context(), 30*time.Second)
	defer cancel()
	run, err := c.ExecuteWorkflow(ctx, client.StartWorkflowOptions{TaskQueue: taskQueue}, slowCallWorkflow)
	require.NoError(t, err)

	var out string
	require.NoError(t, run.Get(ctx, &out))
	assert.Equal(t, "done", out)
}
