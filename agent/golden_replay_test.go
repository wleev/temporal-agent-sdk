package agent_test

import (
	"os"
	"path/filepath"
	"regexp"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.temporal.io/api/temporalproto"
	"go.temporal.io/sdk/worker"
	"go.temporal.io/sdk/workflow"

	"github.com/wleev/temporal-agent-sdk/agent"
	"github.com/wleev/temporal-agent-sdk/agenttest"
	"github.com/wleev/temporal-agent-sdk/tool"
)

// goldenHistoryGlob matches the recorded goldenWorkflow histories.
const goldenHistoryGlob = "testdata/golden_*.json"

// goldenIdentity and goldenStickyQueue match the worker identity and sticky
// task queue name fields, which embed the recording machine's hostname.
var (
	goldenIdentity    = regexp.MustCompile(`"identity": "[^"]*"`)
	goldenStickyQueue = regexp.MustCompile(`"name": "[^"]*:[0-9a-f-]{36}"`)
)

// goldenWorkflowName is the workflow type goldenWorkflow is registered under.
const goldenWorkflowName = "replay_golden"

// goldenWorkflow returns a workflow that runs an agent with the tools slow and
// fast on the input "go".
func goldenWorkflow(t *testing.T) func(workflow.Context) (*agent.Result, error) {
	t.Helper()
	slow, err := tool.New("slow", "A slow tool",
		func(ctx workflow.Context, in echoIn) (string, error) {
			if err := workflow.Sleep(ctx, 2*time.Second); err != nil {
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
	a, err := agent.NewAgent("assistant", "test-model",
		agent.WithInstructions("Be brief."), agent.WithTools(slow, fast))
	require.NoError(t, err)
	return func(ctx workflow.Context) (*agent.Result, error) {
		return agent.Run(ctx, a, "go")
	}
}

// TestReplay_GoldenHistory replays every testdata/golden_*.json history of
// goldenWorkflow against the current code.
func TestReplay_GoldenHistory(t *testing.T) {
	paths, err := filepath.Glob(goldenHistoryGlob)
	require.NoError(t, err)
	require.NotEmpty(t, paths, "no golden histories match %s", goldenHistoryGlob)
	for _, path := range paths {
		t.Run(filepath.Base(path), func(t *testing.T) {
			r := worker.NewWorkflowReplayer()
			r.RegisterWorkflowWithOptions(goldenWorkflow(t), workflow.RegisterOptions{Name: goldenWorkflowName})
			assert.NoError(t, r.ReplayWorkflowHistoryFromJSONFile(nil, path))
		})
	}
}

// TestReplay_RecordGoldenHistory runs goldenWorkflow on a dev server and writes
// its history to testdata/golden_<version>.json, where <version> is the value
// of AGENTSDK_RECORD_GOLDEN. It runs only when that variable is set and fails
// if the file exists.
func TestReplay_RecordGoldenHistory(t *testing.T) {
	version := os.Getenv("AGENTSDK_RECORD_GOLDEN")
	if version == "" {
		t.Skip("set AGENTSDK_RECORD_GOLDEN=<version> to record a golden history")
	}
	path := filepath.Join("testdata", "golden_"+version+".json")
	if _, err := os.Stat(path); err == nil {
		t.Fatalf("%s already exists", path)
	}
	env := startReplayEnv(t)
	fake := agenttest.NewFakeProvider(
		agenttest.CallsTools(
			agenttest.ToolCall{ID: "a", Name: "slow", Args: `{"text":"1"}`},
			agenttest.ToolCall{ID: "b", Name: "fast", Args: `{"text":"2"}`},
		),
		agenttest.CallsTool("fast", `{"text":"3"}`),
		agenttest.Says("done"),
	)

	hist := env.run(t, fake, nil, goldenWorkflow(t), goldenWorkflowName)

	raw, err := temporalproto.CustomJSONMarshalOptions{Indent: "  "}.Marshal(hist)
	require.NoError(t, err)
	raw = goldenIdentity.ReplaceAll(raw, []byte(`"identity": "agentsdk-golden-recorder"`))
	raw = goldenStickyQueue.ReplaceAll(raw, []byte(`"name": "agentsdk-golden-recorder:sticky"`))
	require.NoError(t, os.MkdirAll(filepath.Dir(path), 0o755))
	require.NoError(t, os.WriteFile(path, append(raw, '\n'), 0o600))
}
