package agent_test

import (
	"encoding/json"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	enumspb "go.temporal.io/api/enums/v1"
	"go.temporal.io/sdk/testsuite"
	"go.temporal.io/sdk/workflow"

	"github.com/wleev/temporal-agent-sdk/agent"
	"github.com/wleev/temporal-agent-sdk/agenttest"
	"github.com/wleev/temporal-agent-sdk/tool"
)

// newSubAgentEnv registers the model activity and the agent workflow, which is
// what a sub-agent child workflow needs to resolve and run.
func newSubAgentEnv(t *testing.T, fake *agenttest.FakeProvider, reg *agent.Registry) *testsuite.TestWorkflowEnvironment {
	t.Helper()
	env := newEnv(t, fake)
	agent.RegisterWorkflows(env, reg)
	return env
}

func TestSubAgent_DelegatesToChildWorkflow(t *testing.T) {
	// Turn 1: parent delegates. Turn 2: the child answers. Turn 3: the parent
	// wraps up with the child's result.
	fake := agenttest.NewFakeProvider(
		agenttest.CallsTool("research", `{"input":"the capital of Belgium"}`),
		agenttest.Says("Brussels."),
		agenttest.Says("The capital of Belgium is Brussels."),
	)

	researcher, err := agent.NewAgent("researcher", "test-model",
		agent.WithInstructions("You research things."))
	require.NoError(t, err)
	reg := agent.NewRegistry()
	require.NoError(t, reg.Add(researcher))

	sub, err := agent.AsSubAgent(researcher, "research", "Research a topic")
	require.NoError(t, err)
	parent, err := agent.NewAgent("assistant", "test-model", agent.WithTools(sub))
	require.NoError(t, err)
	require.NoError(t, reg.Add(parent))

	env := newSubAgentEnv(t, fake, reg)
	env.ExecuteWorkflow(func(ctx workflow.Context) (*agent.Result, error) {
		return agent.Run(ctx, parent, "What is the capital of Belgium?")
	})

	require.True(t, env.IsWorkflowCompleted())
	require.NoError(t, env.GetWorkflowError())

	var res agent.Result
	require.NoError(t, env.GetWorkflowResult(&res))
	assert.Equal(t, "The capital of Belgium is Brussels.", res.Output)

	// The child's own instructions drove its turn, proving it ran as a real
	// agent rather than as an inlined call.
	childCall := fake.Calls()[1]
	assert.Equal(t, "You research things.", childCall.Messages[0].Text())
	assert.Equal(t, "the capital of Belgium", childCall.Messages[1].Text())

	// The parent sees only the child's conclusion; the child's transcript stays
	// in the child's history. That is the point of the split.
	parentFinal := fake.Calls()[2]
	last := parentFinal.Messages[len(parentFinal.Messages)-1]
	assert.Equal(t, "Brussels.", last.ToolResultText())
}

// The child-completion payload — the value a sub-agent returns — must not carry
// the transcript, because that value is serialized into the *parent's* history.
// Returning Messages would copy every child turn into the parent, spending the
// parent's budget on the child and risking the 2 MB payload cap.
//
// This exercises AgentWorkflow directly, which is the layer the parent actually
// records; asserting on the parent's own Result.Messages would test the wrong
// side of the boundary.
func TestAgentWorkflow_OutputOnlyStripsTranscript(t *testing.T) {
	fake := agenttest.NewFakeProvider(
		agenttest.CallsTool("noop", `{}`),
		agenttest.Says("the conclusion"),
	)
	reg := agent.NewRegistry()

	noop, err := tool.New("noop", "does nothing",
		func(workflow.Context, struct{}) (string, error) { return "ok", nil })
	require.NoError(t, err)

	worker, err := agent.NewAgent("worker", "test-model",
		agent.WithInstructions("Work."), agent.WithTools(noop))
	require.NoError(t, err)
	require.NoError(t, reg.Add(worker))

	env := newEnv(t, fake)
	agent.RegisterWorkflows(env, reg)

	env.ExecuteWorkflow(agent.WorkflowName, agent.WorkflowInput{
		Agent:      "worker",
		Input:      "do the thing",
		OutputOnly: true,
	})
	require.True(t, env.IsWorkflowCompleted())
	require.NoError(t, env.GetWorkflowError())

	var res agent.Result
	require.NoError(t, env.GetWorkflowResult(&res))
	assert.Equal(t, "the conclusion", res.Output, "the output must still be returned")
	assert.Empty(t, res.Messages,
		"OutputOnly must drop the transcript so it is not copied into the parent's history")
	assert.Greater(t, res.Turns, 0, "cheap fields like Turns are still worth returning")
}

// Without OutputOnly, a direct caller still gets the full transcript for
// continue-as-new.
func TestAgentWorkflow_KeepsTranscriptByDefault(t *testing.T) {
	fake := agenttest.NewFakeProvider(agenttest.Says("done"))
	reg := agent.NewRegistry()
	worker, err := agent.NewAgent("worker", "test-model", agent.WithInstructions("Work."))
	require.NoError(t, err)
	require.NoError(t, reg.Add(worker))

	env := newEnv(t, fake)
	agent.RegisterWorkflows(env, reg)

	env.ExecuteWorkflow(agent.WorkflowName, agent.WorkflowInput{Agent: "worker", Input: "hi"})
	require.NoError(t, env.GetWorkflowError())

	var res agent.Result
	require.NoError(t, env.GetWorkflowResult(&res))
	assert.NotEmpty(t, res.Messages, "a direct caller still gets the transcript")
}

// The parent's own history must not carry the sub-agent's turns — that is what
// buys each sub-agent its own event budget.
func TestSubAgent_ParentHistoryExcludesChildTranscript(t *testing.T) {
	fake := agenttest.NewFakeProvider(
		agenttest.CallsTool("research", `{"input":"topic"}`),
		agenttest.Says("child answer"),
		agenttest.Says("parent answer"),
	)

	researcher, err := agent.NewAgent("researcher", "test-model", agent.WithInstructions("Research."))
	require.NoError(t, err)
	reg := agent.NewRegistry()
	require.NoError(t, reg.Add(researcher))
	sub, err := agent.AsSubAgent(researcher, "research", "Research a topic")
	require.NoError(t, err)
	parent, err := agent.NewAgent("assistant", "test-model", agent.WithTools(sub))
	require.NoError(t, err)
	require.NoError(t, reg.Add(parent))

	env := newSubAgentEnv(t, fake, reg)
	env.ExecuteWorkflow(func(ctx workflow.Context) (*agent.Result, error) {
		return agent.Run(ctx, parent, "go")
	})
	require.NoError(t, env.GetWorkflowError())

	var res agent.Result
	require.NoError(t, env.GetWorkflowResult(&res))
	for _, m := range res.Messages {
		assert.NotEqual(t, "Research.", m.Text(),
			"the child's system message must not leak into the parent's transcript")
	}
}

// Independent sub-agents should overlap, which is the main reason to delegate to
// several at once.
func TestSubAgent_ParallelDelegationIsConcurrent(t *testing.T) {
	fake := agenttest.NewFakeProvider(
		agenttest.CallsTools(
			agenttest.ToolCall{ID: "a", Name: "research", Args: `{"input":"one"}`},
			agenttest.ToolCall{ID: "b", Name: "research", Args: `{"input":"two"}`},
		),
		agenttest.Says("answer one"),
		agenttest.Says("answer two"),
		agenttest.Says("combined"),
	)

	researcher, err := agent.NewAgent("researcher", "test-model")
	require.NoError(t, err)
	reg := agent.NewRegistry()
	require.NoError(t, reg.Add(researcher))
	sub, err := agent.AsSubAgent(researcher, "research", "Research a topic")
	require.NoError(t, err)
	parent, err := agent.NewAgent("assistant", "test-model", agent.WithTools(sub))
	require.NoError(t, err)
	require.NoError(t, reg.Add(parent))

	env := newSubAgentEnv(t, fake, reg)
	env.ExecuteWorkflow(func(ctx workflow.Context) (*agent.Result, error) {
		return agent.Run(ctx, parent, "go")
	})

	require.True(t, env.IsWorkflowCompleted())
	require.NoError(t, env.GetWorkflowError())

	var res agent.Result
	require.NoError(t, env.GetWorkflowResult(&res))
	assert.Equal(t, "combined", res.Output)
	assert.Equal(t, 4, fake.CallCount(), "two parent turns plus one turn per child")
}

// An unregistered sub-agent fails permanently: the registry is fixed at startup,
// so retrying cannot help.
func TestSubAgent_UnregisteredAgentFailsNonRetryably(t *testing.T) {
	fake := agenttest.NewFakeProvider(agenttest.CallsTool("research", `{"input":"x"}`))

	researcher, err := agent.NewAgent("researcher", "test-model")
	require.NoError(t, err)
	// Deliberately not registered.
	reg := agent.NewRegistry()

	sub, err := agent.AsSubAgent(researcher, "research", "Research a topic")
	require.NoError(t, err)
	parent, err := agent.NewAgent("assistant", "test-model", agent.WithTools(sub))
	require.NoError(t, err)
	require.NoError(t, reg.Add(parent))

	env := newSubAgentEnv(t, fake, reg)
	env.ExecuteWorkflow(func(ctx workflow.Context) (*agent.Result, error) {
		return agent.Run(ctx, parent, "go")
	})

	require.True(t, env.IsWorkflowCompleted())
	assert.ErrorContains(t, env.GetWorkflowError(), `no agent named "researcher" is registered`)
}

func TestSubAgent_DefaultsToAbandonParentClosePolicy(t *testing.T) {
	researcher, err := agent.NewAgent("researcher", "test-model")
	require.NoError(t, err)

	// Temporal's default is TERMINATE, which would kill a sub-agent when the
	// parent closes. ABANDON is the safer default for delegated work.
	tl, err := agent.AsSubAgent(researcher, "research", "Research a topic")
	require.NoError(t, err)
	assert.NotNil(t, tl)

	// An explicit policy must survive.
	custom, err := agent.AsSubAgentWith(researcher, "research2", "Research", agent.SubAgentOptions{
		ChildOptions: workflow.ChildWorkflowOptions{
			ParentClosePolicy:        enumspb.PARENT_CLOSE_POLICY_TERMINATE,
			WorkflowExecutionTimeout: 5 * time.Minute,
		},
	})
	require.NoError(t, err)
	assert.NotNil(t, custom)
}

func TestSubAgent_SchemaAndNaming(t *testing.T) {
	researcher, err := agent.NewAgent("researcher", "test-model")
	require.NoError(t, err)

	tl, err := agent.AsSubAgent(researcher, "", "")
	require.NoError(t, err)

	def := tl.Def()
	assert.Equal(t, "researcher", def.Name, "an empty name should fall back to the agent's name")
	assert.Contains(t, def.Description, "researcher")

	raw, err := json.Marshal(def.InputSchema)
	require.NoError(t, err)
	assert.JSONEq(t,
		`{"type":"object","additionalProperties":false,"required":["input"],`+
			`"properties":{"input":{"type":"string","description":"The task or question to delegate to this agent."}}}`,
		string(raw))
}

func TestSubAgent_ApprovalGating(t *testing.T) {
	researcher, err := agent.NewAgent("researcher", "test-model")
	require.NoError(t, err)

	tl, err := agent.AsSubAgentWith(researcher, "research", "Research a topic", agent.SubAgentOptions{
		ToolOptions: []tool.Option{tool.WithApprovalPrompt("Delegation needs review.")},
	})
	require.NoError(t, err)
	assert.True(t, tl.Policy().NeedsApproval, "sub-agent delegation should be gateable")
	assert.Equal(t, "Delegation needs review.", tl.Policy().ApprovalPrompt)
}

func TestRegistry_RejectsDuplicatesAndResolves(t *testing.T) {
	reg := agent.NewRegistry()
	a, err := agent.NewAgent("a", "m")
	require.NoError(t, err)
	require.NoError(t, reg.Add(a))

	dup, err := agent.NewAgent("a", "m")
	require.NoError(t, err)
	assert.ErrorContains(t, reg.Add(dup), "already registered")

	got, ok := reg.Lookup("a")
	if assert.True(t, ok) {
		assert.Same(t, a, got, "Get returns the registered agent")
	}

	_, ok = reg.Lookup("missing")
	assert.False(t, ok)
}
