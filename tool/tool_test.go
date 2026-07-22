package tool_test

import (
	"context"
	"encoding/json"
	"errors"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	"go.temporal.io/sdk/activity"
	"go.temporal.io/sdk/temporal"
	"go.temporal.io/sdk/testsuite"
	"go.temporal.io/sdk/workflow"

	"github.com/wleev/temporal-agent-sdk/model"
	"github.com/wleev/temporal-agent-sdk/tool"
)

type addIn struct {
	A int `json:"a" jsonschema_description:"First addend"`
	B int `json:"b" jsonschema_description:"Second addend"`
}

type addOut struct {
	Sum int `json:"sum"`
}

type recursiveIn struct {
	Next *recursiveIn `json:"next"`
}

// schemaJSON renders a tool's input schema for comparison.
//
// MCP types InputSchema as any, so it holds whatever the producer put there —
// raw JSON here, since that is what the reflector emits.
func schemaJSON(t *testing.T, def *model.Tool) string {
	t.Helper()
	raw, err := json.Marshal(def.InputSchema)
	require.NoError(t, err)
	return string(raw)
}

func TestNew_SchemaFromArgumentType(t *testing.T) {
	tl, err := tool.New("add", "Add two numbers",
		func(_ workflow.Context, in addIn) (int, error) { return in.A + in.B, nil })
	require.NoError(t, err)

	def := tl.Def()
	require.Equal(t, "add", def.Name)
	require.Equal(t, "Add two numbers", def.Description)
	require.JSONEq(t,
		`{"type":"object","additionalProperties":false,"required":["a","b"],
		  "properties":{"a":{"type":"integer","description":"First addend"},
		                "b":{"type":"integer","description":"Second addend"}}}`,
		schemaJSON(t, def))
	require.False(t, tl.Policy().NeedsApproval)
}

func TestNew_Validation(t *testing.T) {
	_, err := tool.New[addIn, int]("", "desc", func(workflow.Context, addIn) (int, error) { return 0, nil })
	require.ErrorContains(t, err, "name must not be empty")

	_, err = tool.New[addIn, int]("add", "desc", nil)
	require.ErrorContains(t, err, "handler must not be nil")

	// A recursive argument type would overflow the reflector's stack; the guard
	// turns that into an error at construction.
	_, err = tool.New("bad", "desc",
		func(workflow.Context, recursiveIn) (int, error) { return 0, nil })
	require.ErrorContains(t, err, "recursive type")
}

func TestOptions(t *testing.T) {
	plain := tool.Resolve()
	require.False(t, plain.NeedsApproval)

	gated := tool.Resolve(tool.RequiresApproval())
	require.True(t, gated.NeedsApproval)
	require.Empty(t, gated.ApprovalPrompt)

	// A prompt implies gating; setting one without gating would be a footgun.
	prompted := tool.Resolve(tool.WithApprovalPrompt("Careful."))
	require.True(t, prompted.NeedsApproval)
	require.Equal(t, "Careful.", prompted.ApprovalPrompt)

	withAct := tool.Resolve(tool.WithActivityOptions(workflow.ActivityOptions{
		StartToCloseTimeout: time.Minute,
	}))
	require.NotNil(t, withAct.ActivityOptions)
	require.Equal(t, time.Minute, withAct.ActivityOptions.StartToCloseTimeout)
}

func TestMust_PanicsOnError(t *testing.T) {
	require.Panics(t, func() {
		tool.Must(tool.New("bad", "d", func(workflow.Context, recursiveIn) (int, error) { return 0, nil }))
	})
	require.NotPanics(t, func() {
		tool.Must(tool.New("ok", "d", func(workflow.Context, addIn) (int, error) { return 0, nil }))
	})
}

// runTool invokes a tool inside a throwaway workflow, which is the only place a
// tool can legally run.
func runTool(t *testing.T, tl tool.Tool, args string, register func(*testsuite.TestWorkflowEnvironment)) (string, error) {
	t.Helper()
	var s testsuite.WorkflowTestSuite
	env := s.NewTestWorkflowEnvironment()
	if register != nil {
		register(env)
	}
	env.ExecuteWorkflow(func(ctx workflow.Context) (*model.CallToolResult, error) {
		return tl.Invoke(ctx, []byte(args))
	})
	require.True(t, env.IsWorkflowCompleted())
	if err := env.GetWorkflowError(); err != nil {
		return "", err
	}
	var res model.CallToolResult
	require.NoError(t, env.GetWorkflowResult(&res))
	if res.IsError {
		// The tool ran and reported failure; surface it as an error so the
		// tests can assert on it the same way as a failed call.
		return "", errors.New(model.ResultText(&res))
	}
	return model.ResultText(&res), nil
}

// A string result is what the model reads, so it must not arrive JSON-quoted.
func TestInvoke_StringResultPassesThroughUnquoted(t *testing.T) {
	tl, err := tool.New("echo", "Echo",
		func(_ workflow.Context, in addIn) (string, error) { return "plain text", nil })
	require.NoError(t, err)

	out, err := runTool(t, tl, `{"a":1,"b":2}`, nil)
	require.NoError(t, err)
	require.Equal(t, "plain text", out, `a string result must not be quoted`)
}

func TestInvoke_StructResultIsJSONEncoded(t *testing.T) {
	tl, err := tool.New("add", "Add",
		func(_ workflow.Context, in addIn) (addOut, error) { return addOut{Sum: in.A + in.B}, nil })
	require.NoError(t, err)

	out, err := runTool(t, tl, `{"a":2,"b":3}`, nil)
	require.NoError(t, err)
	require.JSONEq(t, `{"sum":5}`, out)
}

func TestInvoke_DecodesArguments(t *testing.T) {
	var got addIn
	tl, err := tool.New("add", "Add",
		func(_ workflow.Context, in addIn) (int, error) {
			got = in
			return in.A + in.B, nil
		})
	require.NoError(t, err)

	out, err := runTool(t, tl, `{"a":7,"b":5}`, nil)
	require.NoError(t, err)
	require.Equal(t, addIn{A: 7, B: 5}, got)
	require.Equal(t, "12", out)
}

func TestInvoke_EmptyArgumentsUseZeroValue(t *testing.T) {
	tl, err := tool.New("add", "Add",
		func(_ workflow.Context, in addIn) (int, error) { return in.A + in.B, nil })
	require.NoError(t, err)

	out, err := runTool(t, tl, ``, nil)
	require.NoError(t, err)
	require.Equal(t, "0", out)
}

func TestInvoke_MalformedArgumentsError(t *testing.T) {
	tl, err := tool.New("add", "Add",
		func(_ workflow.Context, in addIn) (int, error) { return 0, nil })
	require.NoError(t, err)

	_, err = runTool(t, tl, `{not json`, nil)
	require.ErrorContains(t, err, "invalid arguments")
}

func TestInvoke_HandlerErrorPropagates(t *testing.T) {
	tl, err := tool.New("boom", "Fails",
		func(_ workflow.Context, in addIn) (int, error) { return 0, errors.New("kaboom") })
	require.NoError(t, err)

	_, err = runTool(t, tl, `{"a":1,"b":1}`, nil)
	require.ErrorContains(t, err, "kaboom")
}

// addActivity is the activity behind the activity-tool tests.
func addActivity(_ context.Context, in addIn) (addOut, error) {
	return addOut{Sum: in.A + in.B}, nil
}

func TestActivity_RunsAsActivity(t *testing.T) {
	tl, err := tool.Activity[addIn, addOut]("add", "Add two numbers", addActivity)
	require.NoError(t, err)

	out, err := runTool(t, tl, `{"a":4,"b":6}`, func(env *testsuite.TestWorkflowEnvironment) {
		env.RegisterActivityWithOptions(addActivity, activity.RegisterOptions{Name: "addActivity"})
	})
	require.NoError(t, err)
	require.JSONEq(t, `{"sum":10}`, out)
}

// Temporal rejects an activity with no timeout, so a tool invoked on a bare
// context needs a default rather than a scheduling failure.
func TestActivity_DefaultTimeoutOnBareContext(t *testing.T) {
	tl, err := tool.Activity[addIn, addOut]("add", "Add", addActivity)
	require.NoError(t, err)

	_, err = runTool(t, tl, `{"a":1,"b":1}`, func(env *testsuite.TestWorkflowEnvironment) {
		env.RegisterActivityWithOptions(addActivity, activity.RegisterOptions{Name: "addActivity"})
	})
	require.NoError(t, err, "an activity tool must supply its own timeout when the context has none")
}

// A caller may pass WithActivityOptions to set a retry policy while leaving the
// timeouts zero. Temporal rejects an activity with no timeout, so the default
// must still be injected rather than the schedule failing.
func TestActivity_DefaultTimeoutWhenOptionsOmitIt(t *testing.T) {
	tl, err := tool.Activity[addIn, addOut]("add", "Add", addActivity,
		tool.WithActivityOptions(workflow.ActivityOptions{
			RetryPolicy: &temporal.RetryPolicy{MaximumAttempts: 2},
			// No StartToClose or ScheduleToClose set.
		}))
	require.NoError(t, err)

	out, err := runTool(t, tl, `{"a":1,"b":1}`, func(env *testsuite.TestWorkflowEnvironment) {
		env.RegisterActivityWithOptions(addActivity, activity.RegisterOptions{Name: "addActivity"})
	})
	require.NoError(t, err,
		"an activity tool must supply a default timeout even when explicit options omit it")
	require.JSONEq(t, `{"sum":2}`, out)
}

func TestActivity_Validation(t *testing.T) {
	_, err := tool.Activity[addIn, addOut]("add", "Add", nil)
	require.ErrorContains(t, err, "activity must not be nil")
}
