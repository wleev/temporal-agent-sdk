package mcp_test

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	"go.temporal.io/sdk/client"
	"go.temporal.io/sdk/temporal"
	"go.temporal.io/sdk/testsuite"
	"go.temporal.io/sdk/worker"
	"go.temporal.io/sdk/workflow"

	"github.com/wleev/temporal-agent-sdk/mcp"
	"github.com/wleev/temporal-agent-sdk/model"
)

// statefulFake is a session whose value is its in-memory state: a counter that
// increments per call. If the session were reconnected per call (stateless), the
// counter would reset — so a growing counter proves the session persisted.
type statefulFake struct {
	mu      sync.Mutex
	counter int
	closed  bool
}

func (s *statefulFake) ListTools(context.Context) ([]*model.Tool, error) {
	return []*model.Tool{model.NewTool("increment", "Increment the session counter",
		[]byte(`{"type":"object","properties":{}}`))}, nil
}

func (s *statefulFake) CallTool(_ context.Context, name string, _ json.RawMessage) (*model.CallToolResult, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if name != "increment" {
		return model.ErrorResult("no such tool: " + name), nil
	}
	s.counter++
	return model.TextResult(fmt.Sprintf("%d", s.counter)), nil
}

func (s *statefulFake) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.closed = true
	return nil
}

// devServer starts a real Temporal dev server; stateful MCP cannot be tested in
// the in-memory env because nested workers need real task-queue routing.
func devServer(t *testing.T) client.Client {
	t.Helper()
	if testing.Short() {
		t.Skip("skipping stateful MCP test: needs a dev server binary (-short)")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Minute)
	defer cancel()
	srv, err := testsuite.StartDevServer(ctx, testsuite.DevServerOptions{
		ClientOptions: &client.Options{}, LogLevel: "error",
	})
	if err != nil {
		t.Skipf("skipping: could not start dev server: %v", err)
	}
	t.Cleanup(func() { _ = srv.Stop() })
	return srv.Client()
}

const statefulTQ = "agentsdk-stateful-test"

// The core guarantee: across several tool calls in one session, the factory
// connects once and the counter accumulates — the session persisted — and Close
// tears it down.
func TestStateful_SessionPersistsAcrossCalls(t *testing.T) {
	c := devServer(t)

	var connects int32
	sessions := make(chan *statefulFake, 1)
	acts := mcp.NewStatefulActivities()
	acts.MustRegister("counter", func(context.Context) (mcp.Client, error) {
		atomic.AddInt32(&connects, 1)
		s := &statefulFake{}
		sessions <- s
		return s, nil
	})

	w := worker.New(c, statefulTQ, worker.Options{})
	acts.RegisterWith(w)
	w.RegisterWorkflowWithOptions(counterWorkflow, workflow.RegisterOptions{Name: "counter_wf"})
	require.NoError(t, w.Start())
	t.Cleanup(w.Stop)

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	run, err := c.ExecuteWorkflow(ctx, client.StartWorkflowOptions{TaskQueue: statefulTQ}, "counter_wf", 3)
	require.NoError(t, err)

	var got []string
	require.NoError(t, run.Get(ctx, &got))

	// The counter accumulated across calls — this only happens if the session
	// (and its state) survived between calls.
	require.Equal(t, []string{"1", "2", "3"}, got)

	// The factory connected exactly once for the whole session.
	require.EqualValues(t, 1, atomic.LoadInt32(&connects), "the session must connect once, not per call")

	// The session was closed on teardown.
	select {
	case s := <-sessions:
		require.Eventually(t, func() bool { s.mu.Lock(); defer s.mu.Unlock(); return s.closed }, 5*time.Second, 50*time.Millisecond,
			"the session must be closed when the workflow closes it")
	default:
		t.Fatal("no session was created")
	}
}

// counterWorkflow opens a stateful session, calls its increment tool n times, and
// returns the results, then closes the session.
func counterWorkflow(ctx workflow.Context, n int) ([]string, error) {
	sess, err := mcp.OpenStatefulSession(ctx, "counter")
	if err != nil {
		return nil, err
	}
	defer func() { _ = sess.Close(ctx) }()

	tools, err := sess.Tools(ctx)
	if err != nil {
		return nil, err
	}
	if len(tools) != 1 {
		return nil, fmt.Errorf("expected 1 tool, got %d", len(tools))
	}
	inc := tools[0]

	var out []string
	for range n {
		res, err := inc.Invoke(ctx, json.RawMessage(`{}`))
		if err != nil {
			return nil, err
		}
		out = append(out, model.ResultText(res))
	}
	return out, nil
}

// If the session's worker is gone when a tool is called, the call must surface a
// SessionLostError (via a schedule-to-start timeout), not hang.
func TestStateful_LostSessionSurfacesError(t *testing.T) {
	c := devServer(t)

	// No stateful worker is ever started for this server, so the run-scoped queue
	// has no poller — exactly the "holder died" condition.
	w := worker.New(c, statefulTQ+"-lost", worker.Options{})
	w.RegisterWorkflowWithOptions(lostSessionWorkflow, workflow.RegisterOptions{Name: "lost_wf"})
	require.NoError(t, w.Start())
	t.Cleanup(w.Stop)

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	run, err := c.ExecuteWorkflow(ctx, client.StartWorkflowOptions{TaskQueue: statefulTQ + "-lost"}, "lost_wf")
	require.NoError(t, err)

	err = run.Get(ctx, nil)
	require.Error(t, err)
	var appErr *temporal.ApplicationError
	require.ErrorAs(t, err, &appErr)
	require.Equal(t, mcp.ErrorTypeSessionLost, appErr.Type(),
		"a tool call with no session worker must surface a SessionLostError")
}

// lostSessionWorkflow calls a tool on a session whose holder never ran; note it
// does NOT register the stateful activities, so nothing polls the session queue.
func lostSessionWorkflow(ctx workflow.Context) error {
	sess, err := mcp.OpenStatefulSessionWith(ctx, "counter", mcp.StatefulOptions{
		// Short schedule-to-start so the test doesn't wait the 30s default.
		CallActivityOptions: &workflow.ActivityOptions{
			StartToCloseTimeout:    10 * time.Second,
			ScheduleToStartTimeout: 2 * time.Second,
		},
	})
	if err != nil {
		return err
	}
	// The holder activity also won't run (no stateful activities registered), but
	// Tools() schedules on the session queue, which has no poller → timeout.
	_, err = sess.Tools(ctx)
	return err
}
