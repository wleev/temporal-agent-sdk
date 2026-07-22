package agent_test

import (
	"context"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	"go.temporal.io/api/enums/v1"
	historypb "go.temporal.io/api/history/v1"
	"go.temporal.io/sdk/activity"
	"go.temporal.io/sdk/client"
	"go.temporal.io/sdk/testsuite"
	"go.temporal.io/sdk/worker"
	"go.temporal.io/sdk/workflow"

	"github.com/wleev/temporal-agent-sdk/agent"
	"github.com/wleev/temporal-agent-sdk/agenttest"
	"github.com/wleev/temporal-agent-sdk/model"
	"github.com/wleev/temporal-agent-sdk/tool"
)

// Replay is the real determinism test.
//
// The unit tests prove the loop produces the right answer once. These prove it
// produces the *same* answer when Temporal re-executes it against a history it
// recorded earlier — which is what happens on every worker restart, cache
// eviction, and failover.
//
// They run against a real dev server rather than the in-memory test environment,
// because only a real server produces a real history to replay. A loop that
// ordered tool results by completion, ranged a map, or read the wall clock would
// pass every unit test and fail here.
//
// The dev server binary is downloaded on first run, so these are skipped under
// -short.

const replayTaskQueue = "agentsdk-replay-test"

// replayEnv is a running dev server plus a connected client.
type replayEnv struct {
	client client.Client
	server *testsuite.DevServer
}

func startReplayEnv(t *testing.T) *replayEnv {
	t.Helper()
	if testing.Short() {
		t.Skip("skipping replay test: needs a dev server binary (-short)")
	}

	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Minute)
	defer cancel()

	srv, err := testsuite.StartDevServer(ctx, testsuite.DevServerOptions{
		ClientOptions: &client.Options{},
		LogLevel:      "error",
	})
	if err != nil {
		t.Skipf("skipping replay test: could not start dev server: %v", err)
	}
	t.Cleanup(func() { _ = srv.Stop() })

	return &replayEnv{client: srv.Client(), server: srv}
}

// run executes a workflow on a real worker and returns its recorded history.
func (e *replayEnv) run(t *testing.T, fake *agenttest.FakeProvider, reg *agent.Registry, wf any, name string) *historypb.History {
	t.Helper()

	w := worker.New(e.client, replayTaskQueue, worker.Options{})
	w.RegisterWorkflowWithOptions(wf, workflow.RegisterOptions{Name: name})
	w.RegisterActivityWithOptions(
		agenttest.MustActivities(fake).InvokeModel,
		activity.RegisterOptions{Name: model.InvokeModelActivity},
	)
	if reg != nil {
		agent.RegisterWorkflows(w, reg)
	}

	require.NoError(t, w.Start())
	t.Cleanup(w.Stop)

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	run, err := e.client.ExecuteWorkflow(ctx, client.StartWorkflowOptions{
		TaskQueue: replayTaskQueue,
	}, name)
	require.NoError(t, err)

	var res agent.Result
	require.NoError(t, run.Get(ctx, &res), "workflow should complete successfully")

	return e.history(t, run.GetID(), run.GetRunID())
}

func (e *replayEnv) history(t *testing.T, wfID, runID string) *historypb.History {
	t.Helper()

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	iter := e.client.GetWorkflowHistory(ctx, wfID, runID, false,
		enums.HISTORY_EVENT_FILTER_TYPE_ALL_EVENT)

	var hist historypb.History
	for iter.HasNext() {
		ev, err := iter.Next()
		require.NoError(t, err)
		hist.Events = append(hist.Events, ev)
	}
	require.NotEmpty(t, hist.Events)
	return &hist
}

// assertReplays re-executes a recorded history and fails on nondeterminism.
func assertReplays(t *testing.T, hist *historypb.History, reg *agent.Registry, wf any, name string) {
	t.Helper()

	r := worker.NewWorkflowReplayer()
	r.RegisterWorkflowWithOptions(wf, workflow.RegisterOptions{Name: name})
	if reg != nil {
		agent.RegisterWorkflows(r, reg)
	}
	require.NoError(t, r.ReplayWorkflowHistory(nil, hist),
		"recorded history must replay deterministically against the same code")
}

func TestReplay_ToolCall(t *testing.T) {
	env := startReplayEnv(t)

	const name = "replay_tool_call"
	wf := func(ctx workflow.Context) (*agent.Result, error) {
		tl, err := tool.New("get_weather", "Look up the weather",
			func(_ workflow.Context, in weatherIn) (string, error) {
				return "18°C in " + in.City, nil
			})
		if err != nil {
			return nil, err
		}
		return agent.Run(ctx, mustAgent(agent.NewAgent("assistant", "test-model",
			agent.WithInstructions("Be brief."), agent.WithTools(tl))), "Weather in Ghent?")
	}

	fake := agenttest.NewFakeProvider(
		agenttest.CallsTool("get_weather", `{"city":"Ghent"}`),
		agenttest.Says("It is 18°C in Ghent."),
	)

	hist := env.run(t, fake, nil, wf, name)
	assertReplays(t, hist, nil, wf, name)
}

// A typed run reflects the output schema inside workflow code. Reflection is
// deterministic (proven byte-stable), but this proves it replays.
func TestReplay_RunTyped(t *testing.T) {
	env := startReplayEnv(t)

	const name = "replay_typed"
	wf := func(ctx workflow.Context) (*agent.Result, error) {
		res, err := agent.RunTyped[forecast](ctx, mustAgent(agent.NewAgent("weather", "test-model")), "weather?")
		if err != nil {
			return nil, err
		}
		return &res.Result, nil
	}

	fake := agenttest.NewFakeProvider(agenttest.Says(`{"city":"Ghent","temp":18}`))

	hist := env.run(t, fake, nil, wf, name)
	assertReplays(t, hist, nil, wf, name)
}

// Concurrent dispatch is the likeliest source of nondeterminism, so it gets a
// replay test with tools that deliberately finish out of call order.
func TestReplay_ParallelToolCalls(t *testing.T) {
	env := startReplayEnv(t)

	const name = "replay_parallel"
	wf := func(ctx workflow.Context) (*agent.Result, error) {
		slow, err := tool.New("slow", "A slow tool",
			func(ctx workflow.Context, in echoIn) (string, error) {
				if err := workflow.Sleep(ctx, 2*time.Second); err != nil {
					return "", err
				}
				return "slow:" + in.Text, nil
			})
		if err != nil {
			return nil, err
		}
		fast, err := tool.New("fast", "A fast tool",
			func(_ workflow.Context, in echoIn) (string, error) {
				return "fast:" + in.Text, nil
			})
		if err != nil {
			return nil, err
		}
		return agent.Run(ctx, mustAgent(agent.NewAgent("assistant", "test-model",
			agent.WithTools(slow, fast))), "go")
	}

	fake := agenttest.NewFakeProvider(
		agenttest.CallsTools(
			agenttest.ToolCall{ID: "a", Name: "slow", Args: `{"text":"1"}`},
			agenttest.ToolCall{ID: "b", Name: "fast", Args: `{"text":"2"}`},
			agenttest.ToolCall{ID: "c", Name: "slow", Args: `{"text":"3"}`},
		),
		agenttest.Says("done"),
	)

	hist := env.run(t, fake, nil, wf, name)
	assertReplays(t, hist, nil, wf, name)
}

// A sub-agent adds a child workflow to the parent's history.
func TestReplay_SubAgent(t *testing.T) {
	env := startReplayEnv(t)

	researcher := mustAgent(agent.NewAgent("researcher", "test-model", agent.WithInstructions("Research.")))
	reg := agent.NewRegistry().MustAdd(researcher)

	parent := mustAgent(agent.NewAgent("assistant", "test-model",
		agent.WithTools(agent.MustAsSubAgent(researcher, "research", "Research a topic"))))
	require.NoError(t, reg.Add(parent))

	const name = "replay_subagent"
	wf := func(ctx workflow.Context) (*agent.Result, error) {
		return agent.Run(ctx, parent, "go")
	}

	fake := agenttest.NewFakeProvider(
		agenttest.CallsTool("research", `{"input":"topic"}`),
		agenttest.Says("child answer"),
		agenttest.Says("parent answer"),
	)

	hist := env.run(t, fake, reg, wf, name)
	assertReplays(t, hist, reg, wf, name)
}
