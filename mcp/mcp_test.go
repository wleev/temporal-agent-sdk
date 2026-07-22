package mcp_test

import (
	"context"
	"encoding/json"
	"errors"
	"sync"
	"testing"

	mcpsdk "github.com/modelcontextprotocol/go-sdk/mcp"
	"github.com/stretchr/testify/require"
	"go.temporal.io/sdk/temporal"
	"go.temporal.io/sdk/testsuite"
	"go.temporal.io/sdk/workflow"

	"github.com/wleev/temporal-agent-sdk/mcp"
	"github.com/wleev/temporal-agent-sdk/model"
	"github.com/wleev/temporal-agent-sdk/tool"
)

// fakeClient is a scripted MCP server connection that records its lifecycle, so
// the tests can assert the connect/op/close discipline.
type fakeClient struct {
	mu     sync.Mutex
	tools  []*model.Tool
	result *model.CallToolResult
	err    error
	calls  int

	closed    bool
	calledFor string
	args      json.RawMessage
}

func (c *fakeClient) ListTools(context.Context) ([]*model.Tool, error) {
	if c.err != nil {
		return nil, c.err
	}
	return c.tools, nil
}

func (c *fakeClient) CallTool(_ context.Context, name string, args json.RawMessage) (*model.CallToolResult, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.calls++
	c.calledFor = name
	c.args = args
	if c.err != nil {
		return nil, c.err
	}
	return c.result, nil
}

func (c *fakeClient) Close() error {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.closed = true
	return nil
}

func factory(c *fakeClient) mcp.Factory {
	return func(context.Context) (mcp.Client, error) { return c, nil }
}

func TestActivities_Registration(t *testing.T) {
	acts := mcp.NewActivities()
	f := factory(&fakeClient{})

	require.NoError(t, acts.Register("fs", f))
	require.ErrorContains(t, acts.Register("fs", f), "already registered")
	require.ErrorContains(t, acts.Register("", f), "name must not be empty")
	require.ErrorContains(t, acts.Register("x", nil), "must not be nil")
}

func TestListTools_ConnectsAndCloses(t *testing.T) {
	var s testsuite.WorkflowTestSuite
	env := s.NewTestActivityEnvironment()

	c := &fakeClient{tools: []*model.Tool{{
		Name:        "read_file",
		Description: "Read a file",
		InputSchema: json.RawMessage(`{"type":"object","properties":{"path":{"type":"string"}}}`),
	}}}
	mcp.NewActivities().MustRegister("fs", factory(c)).RegisterWith(env)

	val, err := env.ExecuteActivity(mcp.ListToolsActivity, mcp.ListToolsInput{Server: "fs"})
	require.NoError(t, err)

	var got []*model.Tool
	require.NoError(t, val.Get(&got))
	require.Len(t, got, 1)
	require.Equal(t, "read_file", got[0].Name)
	require.True(t, c.closed, "the connection must be closed after the operation")
}

// The whole reason for using the SDK's types: a modern server's fields must
// survive, including ones this SDK has no opinion about.
func TestListTools_ServerFieldsSurviveIntact(t *testing.T) {
	var s testsuite.WorkflowTestSuite
	env := s.NewTestActivityEnvironment()

	destructive := true
	c := &fakeClient{tools: []*model.Tool{{
		Name:        "delete_file",
		Title:       "Delete a file",
		Description: "Permanently delete a file",
		InputSchema: json.RawMessage(`{"type":"object","properties":{"path":{"type":"string"}}}`),
		OutputSchema: json.RawMessage(
			`{"type":"object","properties":{"deleted":{"type":"boolean"}}}`),
		Annotations: &model.ToolAnnotations{
			DestructiveHint: &destructive,
			IdempotentHint:  true,
			Title:           "Delete",
		},
	}}}
	mcp.NewActivities().MustRegister("fs", factory(c)).RegisterWith(env)

	val, err := env.ExecuteActivity(mcp.ListToolsActivity, mcp.ListToolsInput{Server: "fs"})
	require.NoError(t, err)

	var got []*model.Tool
	require.NoError(t, val.Get(&got))
	require.Len(t, got, 1)

	// Every field round-trips through the activity boundary. A hand-maintained
	// struct would have silently dropped title, outputSchema, and annotations.
	require.Equal(t, "Delete a file", got[0].Title)
	require.NotNil(t, got[0].OutputSchema, "outputSchema must survive")
	require.NotNil(t, got[0].Annotations, "annotations must survive")
	require.NotNil(t, got[0].Annotations.DestructiveHint)
	require.True(t, *got[0].Annotations.DestructiveHint)
	require.True(t, got[0].Annotations.IdempotentHint)
	require.Equal(t, "Delete", got[0].Annotations.Title)
}

func TestCallTool_PassesArgumentsAndCloses(t *testing.T) {
	var s testsuite.WorkflowTestSuite
	env := s.NewTestActivityEnvironment()

	c := &fakeClient{result: model.TextResult("file contents")}
	mcp.NewActivities().MustRegister("fs", factory(c)).RegisterWith(env)

	val, err := env.ExecuteActivity(mcp.CallToolActivity, mcp.CallToolInput{
		Server:    "fs",
		Tool:      "read_file",
		Arguments: json.RawMessage(`{"path":"/tmp/x"}`),
	})
	require.NoError(t, err)

	var res model.CallToolResult
	require.NoError(t, val.Get(&res))
	require.Equal(t, "file contents", model.ResultText(&res))
	require.Equal(t, "read_file", c.calledFor)
	require.JSONEq(t, `{"path":"/tmp/x"}`, string(c.args))
	require.True(t, c.closed)
}

// A tool that ran and reported failure is not an activity failure. Retrying it
// would re-run the tool, burn the retry budget, and end the same way.
func TestCallTool_IsErrorIsNotAnActivityFailure(t *testing.T) {
	var s testsuite.WorkflowTestSuite
	env := s.NewTestActivityEnvironment()

	c := &fakeClient{result: model.ErrorResult("no such file: /tmp/nope")}
	mcp.NewActivities().MustRegister("fs", factory(c)).RegisterWith(env)

	val, err := env.ExecuteActivity(mcp.CallToolActivity, mcp.CallToolInput{Server: "fs", Tool: "read_file"})
	require.NoError(t, err, "a tool error must not fail the activity")

	var res model.CallToolResult
	require.NoError(t, val.Get(&res))
	require.True(t, res.IsError, "the IsError flag must survive to the loop")
	require.Contains(t, model.ResultText(&res), "no such file")
	require.Equal(t, 1, c.calls, "a tool error must be called exactly once, never retried")
}

// The connection must close even when the call fails, or a failing server would
// leak a connection per retry.
func TestCallTool_ClosesOnError(t *testing.T) {
	var s testsuite.WorkflowTestSuite
	env := s.NewTestActivityEnvironment()

	c := &fakeClient{err: errors.New("transport exploded")}
	mcp.NewActivities().MustRegister("fs", factory(c)).RegisterWith(env)

	_, err := env.ExecuteActivity(mcp.CallToolActivity, mcp.CallToolInput{Server: "fs", Tool: "boom"})
	require.ErrorContains(t, err, "transport exploded")
	require.True(t, c.closed, "the connection must close even when the call fails")
}

func TestActivities_UnknownServerIsNonRetryable(t *testing.T) {
	var s testsuite.WorkflowTestSuite
	env := s.NewTestActivityEnvironment()

	mcp.NewActivities().MustRegister("fs", factory(&fakeClient{})).RegisterWith(env)

	_, err := env.ExecuteActivity(mcp.ListToolsActivity, mcp.ListToolsInput{Server: "ghost"})
	require.Error(t, err)

	var appErr *temporal.ApplicationError
	require.ErrorAs(t, err, &appErr)
	require.True(t, appErr.NonRetryable(), "the factory set is fixed at startup")
	require.Equal(t, mcp.ErrorTypeUnknownServer, appErr.Type())
}

func TestActivities_ConnectFailureIsRetryable(t *testing.T) {
	var s testsuite.WorkflowTestSuite
	env := s.NewTestActivityEnvironment()

	acts := mcp.NewActivities().MustRegister("fs", func(context.Context) (mcp.Client, error) {
		return nil, errors.New("connection refused")
	})
	acts.RegisterWith(env)

	_, err := env.ExecuteActivity(mcp.ListToolsActivity, mcp.ListToolsInput{Server: "fs"})
	require.ErrorContains(t, err, "connection refused")

	var appErr *temporal.ApplicationError
	if errors.As(err, &appErr) {
		require.False(t, appErr.NonRetryable(), "a connect failure should be retried")
	}
}

// toolView is a serializable projection for asserting tool defs out of a
// workflow result.
type toolView struct {
	Name          string `json:"name"`
	Title         string `json:"title"`
	Params        string `json:"params"`
	NeedsApproval bool   `json:"needs_approval"`
	Prompt        string `json:"prompt"`
}

//workflowcheck:ignore json.Marshal is deterministic for a fixed value
func viewTools(ctx workflow.Context, server string, opts ...mcp.Options) ([]toolView, error) {
	tools, err := mcp.Tools(ctx, server)
	if len(opts) > 0 {
		tools, err = mcp.ToolsWith(ctx, server, opts[0])
	}
	if err != nil {
		return nil, err
	}
	out := make([]toolView, 0, len(tools))
	for _, tl := range tools {
		d := tl.Def()
		raw, _ := json.Marshal(d.InputSchema)
		out = append(out, toolView{
			Name:          d.Name,
			Title:         d.Title,
			Params:        string(raw),
			NeedsApproval: tl.Policy().NeedsApproval,
			Prompt:        tl.Policy().ApprovalPrompt,
		})
	}
	return out, nil
}

func TestTools_SchemaPassesThroughUntouched(t *testing.T) {
	var s testsuite.WorkflowTestSuite
	env := s.NewTestWorkflowEnvironment()

	const serverSchema = `{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}`
	c := &fakeClient{tools: []*model.Tool{{
		Name:        "read_file",
		Title:       "Read a file",
		Description: "Read a file",
		InputSchema: json.RawMessage(serverSchema),
	}}}
	mcp.NewActivities().MustRegister("fs", factory(c)).RegisterWith(env)

	env.ExecuteWorkflow(func(ctx workflow.Context) ([]toolView, error) {
		return viewTools(ctx, "fs")
	})
	require.NoError(t, env.GetWorkflowError())

	var got []toolView
	require.NoError(t, env.GetWorkflowResult(&got))
	require.Len(t, got, 1)
	require.Equal(t, "read_file", got[0].Name)
	require.Equal(t, "Read a file", got[0].Title)

	// The server owns the schema; we must not reflect or rewrite it.
	require.JSONEq(t, serverSchema, got[0].Params)
}

func TestTools_NamePrefixAvoidsCollisions(t *testing.T) {
	var s testsuite.WorkflowTestSuite
	env := s.NewTestWorkflowEnvironment()

	c := &fakeClient{tools: []*model.Tool{{Name: "read", InputSchema: json.RawMessage(`{"type":"object"}`)}}}
	mcp.NewActivities().MustRegister("fs", factory(c)).RegisterWith(env)

	env.ExecuteWorkflow(func(ctx workflow.Context) ([]toolView, error) {
		return viewTools(ctx, "fs", mcp.Options{NamePrefix: "fs_"})
	})
	require.NoError(t, env.GetWorkflowError())

	var got []toolView
	require.NoError(t, env.GetWorkflowResult(&got))
	require.Equal(t, "fs_read", got[0].Name)
}

func TestTools_EmptySchemaGetsEmptyObject(t *testing.T) {
	var s testsuite.WorkflowTestSuite
	env := s.NewTestWorkflowEnvironment()

	c := &fakeClient{tools: []*model.Tool{{Name: "ping"}}} // no InputSchema
	mcp.NewActivities().MustRegister("fs", factory(c)).RegisterWith(env)

	env.ExecuteWorkflow(func(ctx workflow.Context) ([]toolView, error) {
		return viewTools(ctx, "fs")
	})
	require.NoError(t, env.GetWorkflowError())

	var got []toolView
	require.NoError(t, env.GetWorkflowResult(&got))
	require.JSONEq(t, `{"type":"object","properties":{}}`, got[0].Params)
}

// A server's destructive hint should add an approval gate the operator did not
// configure.
func TestTools_DestructiveHintEscalatesToApproval(t *testing.T) {
	var s testsuite.WorkflowTestSuite
	env := s.NewTestWorkflowEnvironment()

	destructive := true
	c := &fakeClient{tools: []*model.Tool{{
		Name:        "delete_everything",
		InputSchema: json.RawMessage(`{"type":"object"}`),
		Annotations: &model.ToolAnnotations{DestructiveHint: &destructive},
	}}}
	mcp.NewActivities().MustRegister("fs", factory(c)).RegisterWith(env)

	env.ExecuteWorkflow(func(ctx workflow.Context) ([]toolView, error) {
		return viewTools(ctx, "fs") // no ToolOptions: not gated by the operator
	})
	require.NoError(t, env.GetWorkflowError())

	var got []toolView
	require.NoError(t, env.GetWorkflowResult(&got))
	require.True(t, got[0].NeedsApproval, "a destructive hint should escalate to approval")
	require.Contains(t, got[0].Prompt, "destructive")
}

// The security-critical direction. A server saying "not destructive" must never
// switch off a gate the operator configured — annotations are unverified hints
// from a party we do not control, so they may only add friction.
func TestTools_HintsCannotBypassOperatorApproval(t *testing.T) {
	var s testsuite.WorkflowTestSuite
	env := s.NewTestWorkflowEnvironment()

	notDestructive := false
	c := &fakeClient{tools: []*model.Tool{{
		Name:        "totally_safe_honest",
		InputSchema: json.RawMessage(`{"type":"object"}`),
		Annotations: &model.ToolAnnotations{
			DestructiveHint: &notDestructive,
			ReadOnlyHint:    true,
		},
	}}}
	mcp.NewActivities().MustRegister("fs", factory(c)).RegisterWith(env)

	env.ExecuteWorkflow(func(ctx workflow.Context) ([]toolView, error) {
		return viewTools(ctx, "fs", mcp.Options{
			ToolOptions: []tool.Option{tool.WithApprovalPrompt("operator says review this")},
		})
	})
	require.NoError(t, env.GetWorkflowError())

	var got []toolView
	require.NoError(t, env.GetWorkflowResult(&got))
	require.True(t, got[0].NeedsApproval,
		"a server's hints must not be able to bypass an operator-configured approval gate")
	require.Equal(t, "operator says review this", got[0].Prompt,
		"the operator's prompt must win over the server's")
}

// An absent hint is not evidence of safety, but neither is it evidence of
// danger: only an explicit destructive hint escalates.
func TestTools_NoAnnotationsLeavesPolicyAlone(t *testing.T) {
	var s testsuite.WorkflowTestSuite
	env := s.NewTestWorkflowEnvironment()

	c := &fakeClient{tools: []*model.Tool{{Name: "read", InputSchema: json.RawMessage(`{"type":"object"}`)}}}
	mcp.NewActivities().MustRegister("fs", factory(c)).RegisterWith(env)

	env.ExecuteWorkflow(func(ctx workflow.Context) ([]toolView, error) {
		return viewTools(ctx, "fs")
	})
	require.NoError(t, env.GetWorkflowError())

	var got []toolView
	require.NoError(t, env.GetWorkflowResult(&got))
	require.False(t, got[0].NeedsApproval)
}

func TestTools_ApprovalGatingAppliesToWholeServer(t *testing.T) {
	var s testsuite.WorkflowTestSuite
	env := s.NewTestWorkflowEnvironment()

	c := &fakeClient{tools: []*model.Tool{{Name: "write", InputSchema: json.RawMessage(`{"type":"object"}`)}}}
	mcp.NewActivities().MustRegister("fs", factory(c)).RegisterWith(env)

	env.ExecuteWorkflow(func(ctx workflow.Context) ([]toolView, error) {
		return viewTools(ctx, "fs", mcp.Options{ToolOptions: []tool.Option{tool.RequiresApproval()}})
	})
	require.NoError(t, env.GetWorkflowError())

	var got []toolView
	require.NoError(t, env.GetWorkflowResult(&got))
	require.True(t, got[0].NeedsApproval, "an MCP server should be gateable as a whole")
}

// Non-text content has nowhere to go in a text-only model, but silently dropping
// it would let the model claim it saw something it did not.
func TestResultText_NonTextContentIsNamedNotDropped(t *testing.T) {
	res := &model.CallToolResult{Content: []mcpsdk.Content{
		&mcpsdk.TextContent{Text: "here is the chart:"},
		&mcpsdk.ImageContent{MIMEType: "image/png", Data: []byte{1, 2, 3}},
	}}

	got := model.ResultText(res)
	require.Contains(t, got, "here is the chart:")
	require.Contains(t, got, "image omitted")
	require.Contains(t, got, "image/png")
}

func TestResultText_StructuredContentFallback(t *testing.T) {
	res := &model.CallToolResult{StructuredContent: map[string]any{"deleted": true}}
	require.JSONEq(t, `{"deleted":true}`, model.ResultText(res))
}
