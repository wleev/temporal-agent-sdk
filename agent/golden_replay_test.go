package agent_test

import (
	"context"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.temporal.io/api/temporalproto"
	"go.temporal.io/sdk/worker"
	"go.temporal.io/sdk/workflow"

	"github.com/wleev/temporal-agent-sdk/agent"
	"github.com/wleev/temporal-agent-sdk/agenttest"
	"github.com/wleev/temporal-agent-sdk/model"
	"github.com/wleev/temporal-agent-sdk/tool"
)

// goldenHistoryGlob matches the recorded golden workflow histories.
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

// goldenHooksWorkflowName is the workflow type goldenHooksWorkflow is registered
// under.
const goldenHooksWorkflowName = "replay_golden_hooks"

// goldenPersistActivity is the activity the goldenHooksWorkflow hooks execute.
const goldenPersistActivity = "agentsdk_golden_persist"

// goldenCase is a golden workflow with the fake provider script and activities its recording runs.
type goldenCase struct {
	// name is the workflow type.
	name string
	// workflow builds the workflow function.
	workflow func(t *testing.T) func(workflow.Context) (*agent.Result, error)
	// script builds the fake provider script the recording runs.
	script func() *agenttest.FakeProvider
	// activities are registered by name on the recording worker.
	activities map[string]any
}

// goldenRecordName is the workflow type of the golden case
// TestReplay_RecordGoldenHistory records.
const goldenRecordName = goldenHooksWorkflowName

// goldenCases returns the golden workflows. TestReplay_RecordGoldenHistory
// records the one named goldenRecordName.
func goldenCases() []goldenCase {
	return []goldenCase{
		{
			name:     goldenWorkflowName,
			workflow: goldenWorkflow,
			script: func() *agenttest.FakeProvider {
				return agenttest.NewFakeProvider(
					agenttest.CallsTools(
						agenttest.ToolCall{ID: "a", Name: "slow", Args: `{"text":"1"}`},
						agenttest.ToolCall{ID: "b", Name: "fast", Args: `{"text":"2"}`},
					),
					agenttest.CallsTool("fast", `{"text":"3"}`),
					agenttest.Says("done"),
				)
			},
		},
		{
			name:     goldenHooksWorkflowName,
			workflow: goldenHooksWorkflow,
			script: func() *agenttest.FakeProvider {
				return agenttest.NewFakeProvider(
					agenttest.CallsTools(
						agenttest.ToolCall{ID: "a", Name: "slow", Args: `{"text":"1"}`},
						agenttest.ToolCall{ID: "b", Name: "fast", Args: `{"text":"2"}`},
						agenttest.ToolCall{ID: "c", Name: "missing", Args: `{"text":"3"}`},
					),
					agenttest.Truncated("par"),
					agenttest.Says("tial"),
				)
			},
			activities: map[string]any{
				goldenPersistActivity: func(context.Context, int) error { return nil },
			},
		},
	}
}

// goldenHooksWorkflow returns a workflow that runs a session-based agent with
// the tools slow and fast, seeded history, OnReply and OnTurn hooks that each
// execute goldenPersistActivity, and one length continuation.
func goldenHooksWorkflow(t *testing.T) func(workflow.Context) (*agent.Result, error) {
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
		agent.WithInstructions("Be brief."), agent.WithTools(slow),
		agent.WithContinueOnLength(1))
	require.NoError(t, err)
	return func(ctx workflow.Context) (*agent.Result, error) {
		s, err := agent.NewSession(ctx, agent.WithoutApprovalHandlers())
		if err != nil {
			return nil, err
		}
		return s.RunWith(ctx, a, "go", agent.RunOptions{
			History:    []model.Message{model.UserMessage("earlier"), model.AssistantMessage("noted")},
			ExtraTools: []tool.Tool{fast},
			OnReply: func(ctx workflow.Context, e agent.ReplyEvent) error {
				return workflow.ExecuteActivity(ctx, goldenPersistActivity, e.Turn).Get(ctx, nil)
			},
			OnTurn: func(ctx workflow.Context, e agent.TurnEvent) error {
				return workflow.ExecuteActivity(ctx, goldenPersistActivity, e.Turn).Get(ctx, nil)
			},
		})
	}
}

// TestReplay_GoldenHistory replays every testdata/golden_*.json history against
// the current code. The history's workflow type selects the golden workflow.
func TestReplay_GoldenHistory(t *testing.T) {
	paths, err := filepath.Glob(goldenHistoryGlob)
	require.NoError(t, err)
	require.NotEmpty(t, paths, "no golden histories match %s", goldenHistoryGlob)
	for _, path := range paths {
		t.Run(filepath.Base(path), func(t *testing.T) {
			r := worker.NewWorkflowReplayer()
			for _, c := range goldenCases() {
				r.RegisterWorkflowWithOptions(c.workflow(t), workflow.RegisterOptions{Name: c.name})
			}
			assert.NoError(t, r.ReplayWorkflowHistoryFromJSONFile(nil, path))
		})
	}
}

// TestReplay_RecordGoldenHistory runs the golden case named goldenRecordName on
// a dev server and writes its history to testdata/golden_<version>.json, where
// <version> is the value of AGENTSDK_RECORD_GOLDEN. It runs only when that
// variable is set and fails if the file exists.
func TestReplay_RecordGoldenHistory(t *testing.T) {
	version := os.Getenv("AGENTSDK_RECORD_GOLDEN")
	if version == "" {
		t.Skip("set AGENTSDK_RECORD_GOLDEN=<version> to record a golden history")
	}
	path := filepath.Join("testdata", "golden_"+version+".json")
	if _, err := os.Stat(path); err == nil {
		t.Fatalf("%s already exists", path)
	}
	i := slices.IndexFunc(goldenCases(), func(c goldenCase) bool { return c.name == goldenRecordName })
	require.GreaterOrEqual(t, i, 0, "no golden case named %s", goldenRecordName)
	c := goldenCases()[i]
	env := startReplayEnv(t)

	hist := env.runWith(t, c.script(), nil, c.workflow(t), c.name, c.activities)

	raw, err := temporalproto.CustomJSONMarshalOptions{Indent: "  "}.Marshal(hist)
	require.NoError(t, err)
	raw = goldenIdentity.ReplaceAll(raw, []byte(`"identity": "agentsdk-golden-recorder"`))
	raw = goldenStickyQueue.ReplaceAll(raw, []byte(`"name": "agentsdk-golden-recorder:sticky"`))
	require.NoError(t, os.MkdirAll(filepath.Dir(path), 0o755))
	require.NoError(t, os.WriteFile(path, append(raw, '\n'), 0o600))
}
