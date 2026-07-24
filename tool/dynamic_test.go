package tool_test

import (
	"context"
	"encoding/json"
	"errors"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.temporal.io/sdk/activity"
	"go.temporal.io/sdk/testsuite"
	"go.temporal.io/sdk/workflow"

	"github.com/wleev/temporal-agent-sdk/model"
	"github.com/wleev/temporal-agent-sdk/tool"
)

func ptr[T any](v T) *T { return &v }

// okDispatch is a dispatch func that returns a fixed result.
func okDispatch(workflow.Context, string, json.RawMessage) (*model.CallToolResult, error) {
	return model.TextResult("ok"), nil
}

func TestDynamic_Validation(t *testing.T) {
	_, err := tool.Dynamic(nil, okDispatch)
	assert.ErrorContains(t, err, "def must not be nil")

	_, err = tool.Dynamic(model.NewTool("", "desc", nil), okDispatch)
	assert.ErrorContains(t, err, "name must not be empty")

	_, err = tool.Dynamic(model.NewTool("t", "desc", nil), nil)
	assert.ErrorContains(t, err, "dispatch must not be nil")
}

func TestDynamic_DefAndSchemaPassThrough(t *testing.T) {
	schema := []byte(`{"type":"object","properties":{"q":{"type":"string"}}}`)
	tl, err := tool.Dynamic(model.NewTool("search", "Search the web", schema), okDispatch)
	require.NoError(t, err)

	def := tl.Def()
	assert.Equal(t, "search", def.Name)
	assert.Equal(t, "Search the web", def.Description)
	assert.JSONEq(t, string(schema), schemaJSON(t, def))
	assert.False(t, tl.Policy().NeedsApproval)
}

func TestDynamic_OptionsSetPolicy(t *testing.T) {
	tl, err := tool.Dynamic(model.NewTool("refund", "Issue a refund", nil), okDispatch,
		tool.WithApprovalPrompt("Refunds need review."))
	require.NoError(t, err)

	assert.True(t, tl.Policy().NeedsApproval)
	assert.Equal(t, "Refunds need review.", tl.Policy().ApprovalPrompt)
}

// A destructive annotation on the def escalates the policy, just as it does for
// a server tool: it can add approval, never remove it.
func TestDynamic_AnnotationsEscalatePolicy(t *testing.T) {
	def := model.NewTool("delete", "Delete a record", nil)
	def.Annotations = &model.ToolAnnotations{DestructiveHint: ptr(true)}

	tl, err := tool.Dynamic(def, okDispatch)
	require.NoError(t, err)
	assert.True(t, tl.Policy().NeedsApproval, "a destructive hint gates the tool")
}

// The dispatch func receives the tool's own name and the raw args, so one func
// can back a whole toolset, routing on the name.
func TestDynamic_DispatchReceivesNameAndArgs(t *testing.T) {
	var gotName, gotArgs string
	tl, err := tool.Dynamic(model.NewTool("lookup", "Look up", nil),
		func(_ workflow.Context, name string, args json.RawMessage) (*model.CallToolResult, error) {
			gotName, gotArgs = name, string(args)
			return model.TextResult("dispatched " + name), nil
		})
	require.NoError(t, err)

	out, err := runTool(t, tl, `{"id":42}`, nil)
	require.NoError(t, err)
	assert.Equal(t, "lookup", gotName)
	assert.JSONEq(t, `{"id":42}`, gotArgs)
	assert.Equal(t, "dispatched lookup", out)
}

// An IsError result is handed back to the model, so runTool surfaces it the same
// way a failed lookup would read to the caller.
func TestDynamic_ToolErrorResult(t *testing.T) {
	tl, err := tool.Dynamic(model.NewTool("lookup", "Look up", nil),
		func(workflow.Context, string, json.RawMessage) (*model.CallToolResult, error) {
			return model.ErrorResult("not found"), nil
		})
	require.NoError(t, err)

	_, err = runTool(t, tl, `{}`, nil)
	assert.ErrorContains(t, err, "not found")
}

// An error returned by dispatch fails the call, distinct from an IsError result.
func TestDynamic_DispatchErrorFailsCall(t *testing.T) {
	tl, err := tool.Dynamic(model.NewTool("boom", "Fails", nil),
		func(workflow.Context, string, json.RawMessage) (*model.CallToolResult, error) {
			return nil, errors.New("dispatch exploded")
		})
	require.NoError(t, err)

	_, err = runTool(t, tl, `{}`, nil)
	assert.ErrorContains(t, err, "dispatch exploded")
}

func echoActivity(_ context.Context, args json.RawMessage) (string, error) {
	return "activity saw " + string(args), nil
}

// The intended use: dispatch schedules an activity, so the tool's work runs off
// the workflow thread with the activity's retries and timeouts.
func TestDynamic_DispatchSchedulesActivity(t *testing.T) {
	tl, err := tool.Dynamic(model.NewTool("run", "Run", nil),
		func(ctx workflow.Context, _ string, args json.RawMessage) (*model.CallToolResult, error) {
			ctx = workflow.WithActivityOptions(ctx, workflow.ActivityOptions{
				StartToCloseTimeout: time.Minute,
			})
			var out string
			if err := workflow.ExecuteActivity(ctx, "echoActivity", args).Get(ctx, &out); err != nil {
				return nil, err
			}
			return model.TextResult(out), nil
		})
	require.NoError(t, err)

	out, err := runTool(t, tl, `{"x":1}`, func(env *testsuite.TestWorkflowEnvironment) {
		env.RegisterActivityWithOptions(echoActivity, activity.RegisterOptions{Name: "echoActivity"})
	})
	require.NoError(t, err)
	assert.Contains(t, out, "activity saw")
}
