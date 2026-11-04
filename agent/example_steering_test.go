package agent_test

import (
	"encoding/json"
	"fmt"
	"log/slog"
	"strings"

	"go.temporal.io/sdk/activity"
	"go.temporal.io/sdk/log"
	"go.temporal.io/sdk/testsuite"
	"go.temporal.io/sdk/workflow"

	"github.com/wleev/temporal-agent-sdk/agent"
	"github.com/wleev/temporal-agent-sdk/agenttest"
	"github.com/wleev/temporal-agent-sdk/model"
	"github.com/wleev/temporal-agent-sdk/tool"
)

// runExample runs run in a workflow test environment that discards its logs
// and serves model calls from fake, and returns the run's Result and the
// workflow's error.
func runExample(fake *agenttest.FakeProvider, run func(ctx workflow.Context) (*agent.Result, error)) (*agent.Result, error) {
	var s testsuite.WorkflowTestSuite
	s.SetLogger(log.NewStructuredLogger(slog.New(slog.DiscardHandler)))
	env := s.NewTestWorkflowEnvironment()
	acts, err := model.NewActivities([]model.Provider{fake})
	if err != nil {
		return nil, err
	}
	env.RegisterActivityWithOptions(acts.InvokeModel, activity.RegisterOptions{Name: model.InvokeModelActivity})

	var res *agent.Result
	env.ExecuteWorkflow(func(ctx workflow.Context) (err error) {
		res, err = run(ctx)
		return err
	})
	return res, env.GetWorkflowError()
}

// ExampleRoundOutcome shortens long tool results before the model sees them.
func ExampleRoundOutcome() {
	lookup, err := tool.New("lookup", "Look up a document",
		func(workflow.Context, struct{}) (string, error) { return strings.Repeat("a", 2500), nil })
	if err != nil {
		panic(err)
	}
	a, err := agent.NewAgent("assistant", "test-model", agent.WithTools(lookup))
	if err != nil {
		panic(err)
	}
	fake := agenttest.NewFakeProvider(agenttest.CallsTool("lookup", `{}`), agenttest.Says("done"))

	res, err := runExample(fake, func(ctx workflow.Context) (*agent.Result, error) {
		return agent.RunWith(ctx, a, "Summarize the document.", agent.RunOptions{
			OnRound: func(ctx workflow.Context, e agent.RoundEvent) (agent.RoundOutcome, error) {
				out := agent.RoundOutcome{Results: make([]*model.CallToolResult, len(e.Results))}
				for i, r := range e.Results {
					if text := []rune(model.ResultText(r)); len(text) > 2000 {
						out.Results[i] = model.TextResult(string(text[:2000]) + " [truncated]")
					}
				}
				return out, nil
			},
		})
	})
	result := res.Messages[len(res.Messages)-2].ToolResultText()
	fmt.Println(len(result), strings.HasSuffix(result, " [truncated]"), res.Output, err)
	// Output: 2012 true done <nil>
}

// newSubmitTool returns a submit tool whose result ends the run with the
// submitted answer.
func newSubmitTool() (tool.Tool, error) {
	submitSchema := []byte(`{"type":"object","properties":{"answer":{"type":"string"}},"required":["answer"]}`)
	submit, err := tool.Dynamic(model.NewTool("submit", "Submit the final answer", submitSchema),
		func(ctx workflow.Context, _ string, args json.RawMessage) (*model.CallToolResult, error) {
			var in struct {
				Answer string `json:"answer"`
			}
			if err := json.Unmarshal(args, &in); err != nil {
				return nil, err
			}
			return model.FinalResult(model.TextResult("Submitted."), in.Answer), nil
		}, tool.EndsRun())
	return submit, err
}

// Example_endingARunFromATool runs an agent whose submit tool ends the run.
func Example_endingARunFromATool() {
	submit, err := newSubmitTool()
	if err != nil {
		panic(err)
	}
	a, err := agent.NewAgent("assistant", "test-model", agent.WithTools(submit))
	if err != nil {
		panic(err)
	}
	fake := agenttest.NewFakeProvider(agenttest.CallsTool("submit", `{"answer":"It is 18°C in Ghent."}`))

	res, err := runExample(fake, func(ctx workflow.Context) (*agent.Result, error) {
		return agent.Run(ctx, a, "Weather in Ghent?")
	})
	fmt.Println(res.Output, res.FinishReason, fake.CallCount(), err)
	// Output: It is 18°C in Ghent. stop 1 <nil>
}
