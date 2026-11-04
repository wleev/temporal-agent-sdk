package agent_test

import (
	"fmt"

	"go.temporal.io/sdk/workflow"

	"github.com/wleev/temporal-agent-sdk/agent"
	"github.com/wleev/temporal-agent-sdk/agenttest"
	"github.com/wleev/temporal-agent-sdk/model"
)

// ExampleFinalAction continues a run once when the final reply has no text.
func ExampleFinalAction() {
	a, err := agent.NewAgent("assistant", "test-model")
	if err != nil {
		panic(err)
	}
	fake := agenttest.NewFakeProvider(agenttest.Says(""), agenttest.Says("It is 18°C in Ghent."))

	res, err := runExample(fake, func(ctx workflow.Context) (*agent.Result, error) {
		asked := false
		return agent.RunWith(ctx, a, "Weather in Ghent?", agent.RunOptions{
			OnFinal: func(ctx workflow.Context, e agent.FinalEvent) (agent.FinalAction, error) {
				if e.Assistant.Text() == "" && !asked {
					asked = true
					return agent.FinalAction{
						Continue: true,
						Append:   []model.Message{model.UserMessage("Answer in text.")},
					}, nil
				}
				return agent.FinalAction{}, nil
			},
		})
	})
	fmt.Println(res.Output, fake.CallCount(), err)
	// Output: It is 18°C in Ghent. 2 <nil>
}
