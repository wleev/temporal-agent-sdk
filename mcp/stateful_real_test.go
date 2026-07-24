package mcp_test

import (
	"context"
	"encoding/json"
	"os"
	"os/exec"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.temporal.io/sdk/client"
	"go.temporal.io/sdk/worker"
	"go.temporal.io/sdk/workflow"

	"github.com/wleev/temporal-agent-sdk/mcp"
	"github.com/wleev/temporal-agent-sdk/mcp/mcpsdk"
	"github.com/wleev/temporal-agent-sdk/model"
)

// This drives a REAL, pre-packaged MCP server over stdio through the full
// stateful path — proving the npx/stdio/protocol plumbing works, not just the
// in-process fake.
//
// The server is @modelcontextprotocol/server-sequentialthinking, chosen because
// it holds its thought history IN MEMORY and returns thoughtHistoryLength on
// every call: across one session that field climbs 1, 2, 3…, but a stateless
// reconnect-per-call would reset it to 1 every time. So a climbing value is
// direct evidence the live session persisted.
//
// It needs `npx` on PATH and network access to fetch the package, so it is
// opt-in via MCP_NPX_E2E=1 to keep CI hermetic.
func TestStateful_RealSequentialThinkingServer(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping: needs a dev server and npx (-short)")
	}
	if os.Getenv("MCP_NPX_E2E") == "" {
		t.Skip("skipping: set MCP_NPX_E2E=1 to run against the real npx MCP server (needs network)")
	}
	if _, err := exec.LookPath("npx"); err != nil {
		t.Skip("skipping: npx not on PATH")
	}

	c := devServer(t)

	acts := mcp.NewStatefulActivities()
	require.NoError(t, acts.Register("thinking", mcpsdk.CommandFactory(
		"npx", "-y", "@modelcontextprotocol/server-sequential-thinking")))

	w := worker.New(c, statefulTQ+"-real", worker.Options{})
	acts.RegisterWith(w)
	w.RegisterWorkflowWithOptions(thinkingWorkflow, workflow.RegisterOptions{Name: "thinking_wf"})
	require.NoError(t, w.Start())
	t.Cleanup(w.Stop)

	// Generous: the first npx run downloads the package.
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Minute)
	defer cancel()
	run, err := c.ExecuteWorkflow(ctx, client.StartWorkflowOptions{TaskQueue: statefulTQ + "-real"}, "thinking_wf")
	require.NoError(t, err)

	var lengths []int
	require.NoError(t, run.Get(ctx, &lengths))

	// The in-memory thought history grew across calls on one live session.
	assert.Equal(t, []int{1, 2}, lengths,
		"thoughtHistoryLength must climb across calls; a reset to 1 would mean the session did not persist")
}

// thinkingWorkflow opens a stateful session to the real server and makes two
// sequentialthinking calls, returning each result's thoughtHistoryLength.
//
//workflowcheck:ignore json marshal/unmarshal are deterministic for fixed values
func thinkingWorkflow(ctx workflow.Context) ([]int, error) {
	// The real server can be slow to boot on first fetch; give the session and
	// its calls room.
	sess, err := mcp.OpenStatefulSessionWith(ctx, "thinking", mcp.StatefulOptions{
		CallActivityOptions: &workflow.ActivityOptions{
			StartToCloseTimeout:    2 * time.Minute,
			ScheduleToStartTimeout: 2 * time.Minute,
		},
	})
	if err != nil {
		return nil, err
	}
	defer func() { _ = sess.Close(ctx) }()

	tools, err := sess.Tools(ctx)
	if err != nil {
		return nil, err
	}
	var think interface {
		Invoke(workflow.Context, json.RawMessage) (*model.CallToolResult, error)
	}
	for _, tl := range tools {
		if tl.Def().Name == "sequentialthinking" {
			think = tl
		}
	}
	if think == nil {
		return nil, workflowError("sequentialthinking tool not advertised")
	}

	var lengths []int
	for i := 1; i <= 2; i++ {
		args, _ := json.Marshal(map[string]any{
			"thought":           "step",
			"nextThoughtNeeded": i < 2,
			"thoughtNumber":     i,
			"totalThoughts":     2,
		})
		res, err := think.Invoke(ctx, args)
		if err != nil {
			return nil, err
		}
		var parsed struct {
			ThoughtHistoryLength int `json:"thoughtHistoryLength"`
		}
		if err := json.Unmarshal([]byte(model.ResultText(res)), &parsed); err != nil {
			return nil, err
		}
		lengths = append(lengths, parsed.ThoughtHistoryLength)
	}
	return lengths, nil
}

type workflowError string

func (e workflowError) Error() string { return string(e) }
