package conversation_test

import (
	"fmt"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	"go.temporal.io/sdk/activity"
	"go.temporal.io/sdk/testsuite"
	"go.temporal.io/sdk/workflow"

	"github.com/wleev/temporal-agent-sdk/agent"
	"github.com/wleev/temporal-agent-sdk/agenttest"
	"github.com/wleev/temporal-agent-sdk/conversation"
	"github.com/wleev/temporal-agent-sdk/model"
)

// mustAgent panics if NewAgent errored — the tests build valid agents.
func mustAgent(a *agent.Agent, err error) *agent.Agent {
	if err != nil {
		panic(err)
	}
	return a
}

// updateCallbacks adapts the test env's update callbacks.
type updateCallbacks struct {
	t          *testing.T
	reply      conversation.Reply
	expectFail bool
	failErr    error
}

func (c *updateCallbacks) Accept() {}
func (c *updateCallbacks) Reject(err error) {
	c.failErr = err
	if !c.expectFail {
		c.t.Errorf("update rejected: %v", err)
	}
}
func (c *updateCallbacks) Complete(success any, err error) {
	if err != nil {
		c.failErr = err
		if !c.expectFail {
			c.t.Errorf("update failed: %v", err)
		}
		return
	}
	if r, ok := success.(conversation.Reply); ok {
		c.reply = r
	}
}

func newConvEnv(t *testing.T, fake *agenttest.FakeProvider, cv *conversation.Conversation, reg *agent.Registry) *testsuite.TestWorkflowEnvironment {
	t.Helper()
	var s testsuite.WorkflowTestSuite
	env := s.NewTestWorkflowEnvironment()
	env.RegisterActivityWithOptions(agenttest.MustActivities(fake).InvokeModel,
		activity.RegisterOptions{Name: model.InvokeModelActivity})
	cv.Register(env)
	agent.RegisterWorkflows(env, reg)
	return env
}

func TestConversation_MultiTurn(t *testing.T) {
	fake := agenttest.NewFakeProvider(
		agenttest.Says("Hello there."),
		agenttest.Says("The capital of Belgium is Brussels."),
	)
	a := mustAgent(agent.NewAgent("assistant", "test-model", agent.WithInstructions("Be brief.")))
	reg := agent.NewRegistry().MustAdd(a)
	cv := conversation.New(reg)

	env := newConvEnv(t, fake, cv, reg)

	// Two turns via Update, at different times.
	env.RegisterDelayedCallback(func() {
		env.UpdateWorkflow(conversation.SendMessageUpdate, "u1", &updateCallbacks{t: t},
			conversation.Message{Text: "hi"})
	}, time.Second)
	env.RegisterDelayedCallback(func() {
		env.UpdateWorkflow(conversation.SendMessageUpdate, "u2", &updateCallbacks{t: t},
			conversation.Message{Text: "capital of Belgium?"})
	}, 2*time.Second)

	// Read history near the end.
	var history []model.Message
	env.RegisterDelayedCallback(func() {
		val, err := env.QueryWorkflow(conversation.HistoryQuery)
		require.NoError(t, err)
		require.NoError(t, val.Get(&history))
	}, 3*time.Second)

	// The workflow parks indefinitely; cancel to end the test.
	env.RegisterDelayedCallback(func() { env.CancelWorkflow() }, 4*time.Second)

	env.ExecuteWorkflow(conversation.WorkflowName, conversation.Input{Agent: "assistant"})
	require.True(t, env.IsWorkflowCompleted())

	// The second turn saw the first: history accumulates across turns, and the
	// instructions are not duplicated.
	require.Equal(t, 2, fake.CallCount())
	second := fake.Calls()[1]
	systems := 0
	for _, m := range second.Messages {
		if m.Role == model.RoleSystem {
			systems++
		}
	}
	require.Equal(t, 1, systems, "instructions must not stack across turns")

	// History carries both exchanges.
	require.NotEmpty(t, history)
	require.Equal(t, "The capital of Belgium is Brussels.", history[len(history)-1].Text())
}

// An empty message is rejected by the validator, synchronously.
func TestConversation_RejectsEmptyMessage(t *testing.T) {
	fake := agenttest.NewFakeProvider(agenttest.Says("ok"))
	a := mustAgent(agent.NewAgent("assistant", "test-model"))
	reg := agent.NewRegistry().MustAdd(a)
	cv := conversation.New(reg)
	env := newConvEnv(t, fake, cv, reg)

	cb := &updateCallbacks{t: t, expectFail: true}
	env.RegisterDelayedCallback(func() {
		env.UpdateWorkflow(conversation.SendMessageUpdate, "bad", cb, conversation.Message{Text: "  "})
	}, time.Second)
	env.RegisterDelayedCallback(func() { env.CancelWorkflow() }, 2*time.Second)

	env.ExecuteWorkflow(conversation.WorkflowName, conversation.Input{Agent: "assistant"})
	require.ErrorContains(t, cb.failErr, "must not be empty")
	require.Equal(t, 0, fake.CallCount(), "a rejected turn must not call the model")
}

// When the server suggests continue-as-new, the workflow compacts its history
// and carries the compacted form into the next run — the bound on unbounded
// history.
func TestConversation_ContinuesAsNewWithCompactedHistory(t *testing.T) {
	fake := agenttest.NewFakeProvider(
		agenttest.Says("reply one"), agenttest.Says("reply two"), agenttest.Says("reply three"))
	a := mustAgent(agent.NewAgent("assistant", "test-model", agent.WithInstructions("Be brief.")))
	reg := agent.NewRegistry().MustAdd(a)

	sum := &fixedSummarizer{}
	cv := conversation.New(reg, conversation.WithCompactor(
		conversation.NewSummarizingCompactor(sum, conversation.WithKeepLast(1))))

	var s testsuite.WorkflowTestSuite
	env := s.NewTestWorkflowEnvironment()
	env.RegisterActivityWithOptions(agenttest.MustActivities(fake).InvokeModel,
		activity.RegisterOptions{Name: model.InvokeModelActivity})
	cv.Register(env) // registers the workflow and the compaction activity
	agent.RegisterWorkflows(env, reg)

	// Several turns build up history, then mark CAN suggested so the workflow
	// compacts and continues. Update IDs must be unique or Temporal deduplicates
	// them into one.
	for i, text := range []string{"first message", "second message", "third message"} {
		id := fmt.Sprintf("u%d", i)
		env.RegisterDelayedCallback(func() {
			env.UpdateWorkflow(conversation.SendMessageUpdate, id, &updateCallbacks{t: t},
				conversation.Message{Text: text})
		}, time.Duration(i+1)*time.Second)
	}
	env.RegisterDelayedCallback(func() { env.SetContinueAsNewSuggested(true) }, 4*time.Second)

	env.ExecuteWorkflow(conversation.WorkflowName, conversation.Input{Agent: "assistant"})
	require.True(t, env.IsWorkflowCompleted())

	// The workflow ends by continuing-as-new.
	err := env.GetWorkflowError()
	var canErr *workflow.ContinueAsNewError
	require.ErrorAs(t, err, &canErr, "the conversation must continue-as-new when suggested")

	// The compactor was invoked on the accumulated history.
	require.NotNil(t, sum.got, "history should have been summarized before continue-as-new")
}

func TestConversation_UnknownAgent(t *testing.T) {
	fake := agenttest.NewFakeProvider()
	reg := agent.NewRegistry() // empty
	cv := conversation.New(reg)
	env := newConvEnv(t, fake, cv, reg)

	env.RegisterDelayedCallback(func() { env.CancelWorkflow() }, time.Second)
	env.ExecuteWorkflow(conversation.WorkflowName, conversation.Input{Agent: "ghost"})
	require.True(t, env.IsWorkflowCompleted())
	require.ErrorContains(t, env.GetWorkflowError(), `no agent named "ghost"`)
}
