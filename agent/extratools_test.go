package agent_test

import (
	"testing"

	"github.com/stretchr/testify/require"
	"go.temporal.io/sdk/workflow"

	"github.com/wleev/temporal-agent-sdk/agent"
	"github.com/wleev/temporal-agent-sdk/agenttest"
	"github.com/wleev/temporal-agent-sdk/model"
	"github.com/wleev/temporal-agent-sdk/tool"
)

// RunOptions.ExtraTools — how a stateful MCP session's tools are injected into a
// run — must be usable by the model and must not require mutating the Agent.
func TestRun_ExtraToolsAreCallable(t *testing.T) {
	fake := agenttest.NewFakeProvider(
		agenttest.CallsTool("session_tool", `{"text":"hi"}`),
		agenttest.Says("used the session tool"),
	)
	env := newEnv(t, fake)

	extra, err := tool.New("session_tool", "A tool injected for this run",
		func(_ workflow.Context, in echoIn) (string, error) { return "session:" + in.Text, nil })
	require.NoError(t, err)

	env.ExecuteWorkflow(func(ctx workflow.Context) (*agent.Result, error) {
		a := mustAgent(agent.NewAgent("assistant", "test-model")) // no static tools
		return agent.RunWith(ctx, a, "go", agent.RunOptions{ExtraTools: []tool.Tool{extra}})
	})

	require.True(t, env.IsWorkflowCompleted())
	require.NoError(t, env.GetWorkflowError())

	// The extra tool's schema reached the model, and its result came back.
	require.Equal(t, "session_tool", fake.Calls()[0].Tools[0].Name)
	second := fake.Calls()[1]
	last := second.Messages[len(second.Messages)-1]
	require.Equal(t, model.RoleTool, last.Role)
	require.Equal(t, "session:hi", last.ToolResultText())
}
