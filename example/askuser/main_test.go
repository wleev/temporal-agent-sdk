package main

import (
	"context"
	"encoding/json"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.temporal.io/sdk/activity"
	"go.temporal.io/sdk/testsuite"
	"go.temporal.io/sdk/workflow"

	"github.com/wleev/temporal-agent-sdk/model"
)

// echoProvider calls ask_user once, then echoes the human's answer back in its
// final message.
type echoProvider struct{}

func (echoProvider) Name() string { return "echo" }

func (echoProvider) Invoke(_ context.Context, req model.Request) (model.Response, error) {
	last := req.Messages[len(req.Messages)-1]
	if last.Role == model.RoleTool {
		return model.Response{
			Message:      model.AssistantMessage("Booked with " + last.ToolResultText()),
			FinishReason: model.FinishStop,
		}, nil
	}
	return model.Response{
		Message: model.Message{Role: model.RoleAssistant, Blocks: []model.Block{
			model.ToolCallBlock("call-1", "ask_user", json.RawMessage(`{"question":"which city?"}`)),
		}},
		FinishReason: model.FinishToolCalls,
	}, nil
}

type updateCb struct{ t *testing.T }

func (updateCb) Accept()            {}
func (c updateCb) Reject(err error) { c.t.Errorf("answer update rejected: %v", err) }
func (c updateCb) Complete(_ any, err error) {
	if err != nil {
		c.t.Errorf("answer update failed: %v", err)
	}
}

// The full round trip: the agent calls ask_user and parks, the question surfaces
// on the Query, an Update delivers the answer, and it reaches the model.
func TestAskUser_RoundTrip(t *testing.T) {
	var s testsuite.WorkflowTestSuite
	env := s.NewTestWorkflowEnvironment()

	acts, err := model.NewActivities(echoProvider{})
	require.NoError(t, err)
	env.RegisterActivityWithOptions(acts.InvokeModel, activity.RegisterOptions{Name: model.InvokeModelActivity})

	a, err := buildAgent("test-model")
	require.NoError(t, err)

	// The run is parked on the open question when this fires.
	env.RegisterDelayedCallback(func() {
		val, err := env.QueryWorkflow(PendingQuestionsQuery)
		require.NoError(t, err)
		var list []PendingQuestion
		require.NoError(t, val.Get(&list))
		require.Len(t, list, 1, "one question is open")
		assert.Equal(t, "which city?", list[0].Question)

		env.UpdateWorkflow(AnswerQuestionUpdate, "u-1", updateCb{t: t},
			AnswerRequest{ID: list[0].ID, Answer: json.RawMessage(`{"city":"Ghent"}`)})
	}, time.Second)

	env.ExecuteWorkflow(func(ctx workflow.Context) (string, error) {
		return runAgent(ctx, a, "book a flight")
	})

	require.True(t, env.IsWorkflowCompleted())
	require.NoError(t, env.GetWorkflowError())

	var out string
	require.NoError(t, env.GetWorkflowResult(&out))
	assert.Contains(t, out, "Ghent", "the human's answer reached the model")
}

// An answer for an unknown question ID is rejected by the validator.
func TestAskUser_ValidatorRejectsUnknownID(t *testing.T) {
	desk := newQuestionDesk()
	desk.pending["q-1"] = &PendingQuestion{ID: "q-1", Question: "which city?"}

	assert.Error(t, desk.validateAnswer(AnswerRequest{ID: "q-2", Answer: json.RawMessage(`{}`)}),
		"an unknown question ID is rejected")
	assert.Error(t, desk.validateAnswer(AnswerRequest{ID: "q-1", Answer: json.RawMessage(`not json`)}),
		"a non-JSON answer is rejected")
	assert.NoError(t, desk.validateAnswer(AnswerRequest{ID: "q-1", Answer: json.RawMessage(`{"city":"Ghent"}`)}))
}
