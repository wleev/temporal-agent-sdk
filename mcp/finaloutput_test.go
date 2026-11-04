package mcp_test

import (
	"context"
	"encoding/json"
	"testing"
	"time"

	mcpsdk "github.com/modelcontextprotocol/go-sdk/mcp"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.temporal.io/sdk/client"
	"go.temporal.io/sdk/testsuite"
	"go.temporal.io/sdk/worker"
	"go.temporal.io/sdk/workflow"

	"github.com/wleev/temporal-agent-sdk/mcp"
	"github.com/wleev/temporal-agent-sdk/model"
)

// finalOutputResult returns a result whose Meta carries a final output and a
// "trace" entry.
func finalOutputResult() *model.CallToolResult {
	res := model.TextResult("file contents")
	res.Meta = mcpsdk.Meta{"trace": "t-1"}
	return model.FinalResult(res, "server answer")
}

func TestCallTool_StripsTheFinalOutput(t *testing.T) {
	var s testsuite.WorkflowTestSuite
	env := s.NewTestActivityEnvironment()

	c := &fakeClient{result: finalOutputResult()}
	acts, err := mcp.NewActivities(mcp.WithServer("fs", factory(c)))
	require.NoError(t, err)
	acts.RegisterWith(env)

	val, err := env.ExecuteActivity(mcp.CallToolActivity, mcp.CallToolInput{Server: "fs", Tool: "read_file"})
	require.NoError(t, err)

	var res model.CallToolResult
	require.NoError(t, val.Get(&res))
	_, ok := model.FinalOutput(&res)
	assert.False(t, ok)
	assert.Equal(t, "t-1", res.Meta["trace"])
	assert.Equal(t, "file contents", model.ResultText(&res))
}

// finalOutputFake is a stateful session whose one tool returns
// finalOutputResult.
type finalOutputFake struct{}

func (finalOutputFake) ListTools(context.Context) ([]*model.Tool, error) {
	return []*model.Tool{model.NewTool("read_file", "Read a file", []byte(`{"type":"object","properties":{}}`))}, nil
}

func (finalOutputFake) CallTool(context.Context, string, json.RawMessage) (*model.CallToolResult, error) {
	return finalOutputResult(), nil
}

func (finalOutputFake) Close() error { return nil }

// finalOutputWorkflow invokes the one tool of a stateful "fs" session and
// returns its result.
func finalOutputWorkflow(ctx workflow.Context) (*model.CallToolResult, error) {
	sess, err := mcp.OpenStatefulSession(ctx, "fs")
	if err != nil {
		return nil, err
	}
	defer func() { _ = sess.Close(ctx) }()

	tools, err := sess.Tools(ctx)
	if err != nil {
		return nil, err
	}
	return tools[0].Invoke(ctx, json.RawMessage(`{}`))
}

func TestStateful_ToolStripsTheFinalOutput(t *testing.T) {
	c := devServer(t)

	acts, err := mcp.NewStatefulActivities(mcp.WithServer("fs", func(context.Context) (mcp.Client, error) {
		return finalOutputFake{}, nil
	}))
	require.NoError(t, err)

	const queue = statefulTQ + "-final-output"
	w := worker.New(c, queue, worker.Options{})
	acts.RegisterWith(w)
	w.RegisterWorkflowWithOptions(finalOutputWorkflow, workflow.RegisterOptions{Name: "final_output_wf"})
	require.NoError(t, w.Start())
	t.Cleanup(w.Stop)

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	run, err := c.ExecuteWorkflow(ctx, client.StartWorkflowOptions{TaskQueue: queue}, "final_output_wf")
	require.NoError(t, err)

	var res model.CallToolResult
	require.NoError(t, run.Get(ctx, &res))
	_, ok := model.FinalOutput(&res)
	assert.False(t, ok)
	assert.Equal(t, "t-1", res.Meta["trace"])
	assert.Equal(t, "file contents", model.ResultText(&res))
}
