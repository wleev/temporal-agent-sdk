package main

import (
	"context"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.temporal.io/sdk/client"
	"go.temporal.io/sdk/testsuite"
	"go.temporal.io/sdk/worker"

	"github.com/wleev/temporal-agent-sdk/agent"
	"github.com/wleev/temporal-agent-sdk/agenttest"
	"github.com/wleev/temporal-agent-sdk/model"
)

// These drive the example's real workflow, agents, and activities against a real
// Temporal dev server, with only the LLM replaced by a scripted provider.
//
// That combination is the point: everything the example wires up is exercised
// for real — the worker registration, the approval Update, the child workflows —
// while the one nondeterministic component is pinned so the test can assert
// exact outcomes. A live model would answer differently on every run.

func startExampleWorker(t *testing.T, fake *agenttest.FakeProvider) client.Client {
	t.Helper()
	if testing.Short() {
		t.Skip("skipping end-to-end test: needs a dev server binary (-short)")
	}

	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Minute)
	defer cancel()

	srv, err := testsuite.StartDevServer(ctx, testsuite.DevServerOptions{
		ClientOptions: &client.Options{},
		LogLevel:      "error",
	})
	if err != nil {
		t.Skipf("skipping end-to-end test: could not start dev server: %v", err)
	}
	t.Cleanup(func() { _ = srv.Stop() })

	c := srv.Client()
	w := worker.New(c, taskQueue, worker.Options{})

	// Everything below mirrors runWorker(), except the provider.
	acts, err := model.NewActivities(fake)
	require.NoError(t, err)
	acts.Register(w)

	agent.RegisterWorkflows(w, buildAgents("test-model"))
	w.RegisterWorkflow(SupportWorkflow)
	w.RegisterActivity(LookupOrder)
	w.RegisterActivity(IssueRefund)

	require.NoError(t, w.Start())
	t.Cleanup(w.Stop)
	return c
}

func TestE2E_ActivityToolLookup(t *testing.T) {
	fake := agenttest.NewFakeProvider(
		agenttest.CallsTool("lookup_order", `{"order_id":"ORD-1234"}`),
		agenttest.Says("Your mechanical keyboard (ORD-1234) was delivered."),
	)
	c := startExampleWorker(t, fake)

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	run, err := c.ExecuteWorkflow(ctx, client.StartWorkflowOptions{TaskQueue: taskQueue},
		SupportWorkflow, "where is order ORD-1234?")
	require.NoError(t, err)

	var answer string
	require.NoError(t, run.Get(ctx, &answer))
	assert.Equal(t, "Your mechanical keyboard (ORD-1234) was delivered.", answer)

	// The activity's real output must have reached the model.
	second := fake.Calls()[1]
	last := second.Messages[len(second.Messages)-1]
	assert.Contains(t, last.ToolResultText(), "Mechanical keyboard")
	assert.Contains(t, last.ToolResultText(), "delivered")
}

// A business error from an activity tool goes back to the model, which can then
// ask the customer for a valid ID instead of the workflow failing.
func TestE2E_ActivityToolBusinessErrorReachesModel(t *testing.T) {
	fake := agenttest.NewFakeProvider(
		agenttest.CallsTool("lookup_order", `{"order_id":"nonsense"}`),
		agenttest.Says("I could not find that order. Could you check the ID?"),
	)
	c := startExampleWorker(t, fake)

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	run, err := c.ExecuteWorkflow(ctx, client.StartWorkflowOptions{TaskQueue: taskQueue},
		SupportWorkflow, "where is my order?")
	require.NoError(t, err)

	var answer string
	require.NoError(t, run.Get(ctx, &answer), "a business error must not fail the workflow")
	assert.Contains(t, answer, "could not find")

	second := fake.Calls()[1]
	last := second.Messages[len(second.Messages)-1]
	assert.Contains(t, last.ToolResultText(), "no order found")
}

// The full human-in-the-loop path: the agent parks, an operator queries what is
// pending, approves it via Update, and the refund actually runs.
func TestE2E_ApprovalGateApprove(t *testing.T) {
	fake := agenttest.NewFakeProvider(
		agenttest.CallsTool("issue_refund", `{"order_id":"ORD-1234","amount":89.99,"reason":"faulty"}`),
		agenttest.Says("Your refund has been issued."),
	)
	c := startExampleWorker(t, fake)

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	run, err := c.ExecuteWorkflow(ctx, client.StartWorkflowOptions{TaskQueue: taskQueue},
		SupportWorkflow, "refund order ORD-1234, it is faulty")
	require.NoError(t, err)

	// Poll until the loop has parked and registered its approval.
	var list []agent.PendingApproval
	require.Eventually(t, func() bool {
		val, err := c.QueryWorkflow(ctx, run.GetID(), "", agent.PendingApprovalsQuery)
		if err != nil {
			return false
		}
		if err := val.Get(&list); err != nil {
			return false
		}
		return len(list) == 1
	}, 30*time.Second, 100*time.Millisecond, "the agent should park awaiting approval")

	assert.Equal(t, "issue_refund", list[0].Tool)
	assert.Equal(t, "support", list[0].Agent)
	assert.JSONEq(t, `{"order_id":"ORD-1234","amount":89.99,"reason":"faulty"}`, list[0].Arguments)
	assert.Equal(t, "A refund needs review before it is issued.", list[0].Prompt)

	handle, err := c.UpdateWorkflow(ctx, client.UpdateWorkflowOptions{
		WorkflowID:   run.GetID(),
		UpdateName:   agent.ApproveUpdate,
		WaitForStage: client.WorkflowUpdateStageCompleted,
		Args:         []any{agent.ApprovalRequest{CallID: list[0].CallID, Approved: true}},
	})
	require.NoError(t, err)

	var d agent.Decision
	require.NoError(t, handle.Get(ctx, &d))
	assert.True(t, d.Approved)

	var answer string
	require.NoError(t, run.Get(ctx, &answer))
	assert.Equal(t, "Your refund has been issued.", answer)

	// The refund activity really ran and its output reached the model.
	second := fake.Calls()[1]
	last := second.Messages[len(second.Messages)-1]
	assert.Contains(t, last.ToolResultText(), "Refunded €89.99")
}

func TestE2E_ApprovalGateDeny(t *testing.T) {
	fake := agenttest.NewFakeProvider(
		agenttest.CallsTool("issue_refund", `{"order_id":"ORD-9","amount":500,"reason":"changed mind"}`),
		agenttest.Says("I'm sorry, that refund was not approved."),
	)
	c := startExampleWorker(t, fake)

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	run, err := c.ExecuteWorkflow(ctx, client.StartWorkflowOptions{TaskQueue: taskQueue},
		SupportWorkflow, "refund order ORD-9")
	require.NoError(t, err)

	var list []agent.PendingApproval
	require.Eventually(t, func() bool {
		val, err := c.QueryWorkflow(ctx, run.GetID(), "", agent.PendingApprovalsQuery)
		if err != nil {
			return false
		}
		return val.Get(&list) == nil && len(list) == 1
	}, 30*time.Second, 100*time.Millisecond)

	handle, err := c.UpdateWorkflow(ctx, client.UpdateWorkflowOptions{
		WorkflowID:   run.GetID(),
		UpdateName:   agent.ApproveUpdate,
		WaitForStage: client.WorkflowUpdateStageCompleted,
		Args: []any{agent.ApprovalRequest{
			CallID: list[0].CallID, Approved: false, Reason: "outside the refund window",
		}},
	})
	require.NoError(t, err)
	require.NoError(t, handle.Get(ctx, &agent.Decision{}))

	var answer string
	require.NoError(t, run.Get(ctx, &answer), "a denial must not fail the workflow")

	// The model is told why, so it can explain rather than retry.
	second := fake.Calls()[1]
	last := second.Messages[len(second.Messages)-1]
	assert.Contains(t, last.ToolResultText(), "denied")
	assert.Contains(t, last.ToolResultText(), "outside the refund window")
}

// An unknown call ID is rejected synchronously by the validator and never
// reaches history — the reason approvals are an Update and not a Signal.
func TestE2E_ApprovalValidatorRejectsBadCallID(t *testing.T) {
	fake := agenttest.NewFakeProvider(
		agenttest.CallsTool("issue_refund", `{"order_id":"ORD-1","amount":10,"reason":"x"}`),
		agenttest.Says("done"),
	)
	c := startExampleWorker(t, fake)

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	run, err := c.ExecuteWorkflow(ctx, client.StartWorkflowOptions{TaskQueue: taskQueue},
		SupportWorkflow, "refund ORD-1")
	require.NoError(t, err)

	var list []agent.PendingApproval
	require.Eventually(t, func() bool {
		val, err := c.QueryWorkflow(ctx, run.GetID(), "", agent.PendingApprovalsQuery)
		if err != nil {
			return false
		}
		return val.Get(&list) == nil && len(list) == 1
	}, 30*time.Second, 100*time.Millisecond)

	// A real server may surface a validation rejection either from the call or
	// from the handle, depending on when the rejection is observed; either is a
	// rejection, so accept both.
	bad, err := c.UpdateWorkflow(ctx, client.UpdateWorkflowOptions{
		WorkflowID:   run.GetID(),
		UpdateName:   agent.ApproveUpdate,
		WaitForStage: client.WorkflowUpdateStageCompleted,
		Args:         []any{agent.ApprovalRequest{CallID: "bogus", Approved: true}},
	})
	if err == nil {
		err = bad.Get(ctx, nil)
	}
	if assert.Error(t, err, "an unknown call ID must be rejected") {
		assert.Contains(t, err.Error(), "no pending approval")
	}

	// The real approval still works afterwards.
	handle, err := c.UpdateWorkflow(ctx, client.UpdateWorkflowOptions{
		WorkflowID:   run.GetID(),
		UpdateName:   agent.ApproveUpdate,
		WaitForStage: client.WorkflowUpdateStageCompleted,
		Args:         []any{agent.ApprovalRequest{CallID: list[0].CallID, Approved: true}},
	})
	require.NoError(t, err)
	require.NoError(t, handle.Get(ctx, &agent.Decision{}))

	var answer string
	require.NoError(t, run.Get(ctx, &answer))
}

// Two sub-agents requested in one turn run as concurrent child workflows.
func TestE2E_ConcurrentSubAgents(t *testing.T) {
	fake := agenttest.NewFakeProvider(
		agenttest.CallsTools(
			agenttest.ToolCall{ID: "r1", Name: "research", Args: `{"input":"is this keyboard hot-swappable?"}`},
			agenttest.ToolCall{ID: "p1", Name: "policy", Args: `{"input":"what is the refund window?"}`},
		),
		agenttest.Says("Yes, it is hot-swappable."),     // researcher child
		agenttest.Says("The refund window is 30 days."), // policy child
		agenttest.Says("It is hot-swappable, and you have 30 days to return it."),
	)
	c := startExampleWorker(t, fake)

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	run, err := c.ExecuteWorkflow(ctx, client.StartWorkflowOptions{TaskQueue: taskQueue},
		SupportWorkflow, "is the keyboard hot-swappable, and what is the refund window?")
	require.NoError(t, err)

	var answer string
	require.NoError(t, run.Get(ctx, &answer))
	assert.Equal(t, "It is hot-swappable, and you have 30 days to return it.", answer)

	// Four calls: the parent's opening turn, one per child, and the parent's
	// closing turn.
	require.Equal(t, 4, fake.CallCount())

	// Each child ran with its own agent's instructions, proving they were real
	// separate agents rather than inlined calls.
	var sawResearcher, sawPolicy bool
	for _, call := range fake.Calls() {
		if len(call.Messages) == 0 {
			continue
		}
		switch {
		case call.Messages[0].Text() == researchInstructions:
			sawResearcher = true
		case call.Messages[0].Text() == policyInstructions:
			sawPolicy = true
		}
	}
	assert.True(t, sawResearcher, "the researcher sub-agent should have run")
	assert.True(t, sawPolicy, "the policy sub-agent should have run")

	// Both results reached the parent, in call order.
	final := fake.Calls()[3]
	toolMsgs := 0
	for _, m := range final.Messages {
		if m.Role == model.RoleTool {
			toolMsgs++
		}
	}
	assert.Equal(t, 2, toolMsgs)
}

// The workflow tool runs in-workflow and reads the replay-safe clock.
func TestE2E_WorkflowTool(t *testing.T) {
	fake := agenttest.NewFakeProvider(
		agenttest.CallsTool("get_time", `{"timezone":"Europe/Brussels"}`),
		agenttest.Says("Told you the time."),
	)
	c := startExampleWorker(t, fake)

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	run, err := c.ExecuteWorkflow(ctx, client.StartWorkflowOptions{TaskQueue: taskQueue},
		SupportWorkflow, "what time is it in Brussels?")
	require.NoError(t, err)

	var answer string
	require.NoError(t, run.Get(ctx, &answer))

	second := fake.Calls()[1]
	last := second.Messages[len(second.Messages)-1]
	assert.Contains(t, last.ToolResultText(), time.Now().UTC().Format("2006"),
		"the workflow clock should report the current year")

	// A bad timezone is the model's error to fix, not a workflow failure.
	assert.NotEmpty(t, answer)
}
