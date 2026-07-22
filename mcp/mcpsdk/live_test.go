package mcpsdk_test

import (
	"context"
	"encoding/json"
	"testing"
	"time"

	mcpgo "github.com/modelcontextprotocol/go-sdk/mcp"
	"github.com/stretchr/testify/require"

	"github.com/wleev/temporal-agent-sdk/mcp/mcpsdk"
	"github.com/wleev/temporal-agent-sdk/model"
)

type greetIn struct {
	Name string `json:"name" jsonschema:"the name to greet"`
}

// Drives the adapter against a real in-process MCP server over the SDK's own
// transport, so the protocol round trip is genuinely exercised.
func TestAdapter_AgainstRealServer(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	server := mcpgo.NewServer(&mcpgo.Implementation{Name: "test-server", Version: "1"}, nil)
	destructive := true
	mcpgo.AddTool(server, &mcpgo.Tool{
		Name:        "greet",
		Title:       "Greet someone",
		Description: "Say hello",
		Annotations: &mcpgo.ToolAnnotations{DestructiveHint: &destructive},
	}, func(ctx context.Context, req *mcpgo.CallToolRequest, in greetIn) (*mcpgo.CallToolResult, any, error) {
		return &mcpgo.CallToolResult{
			Content: []mcpgo.Content{&mcpgo.TextContent{Text: "hello " + in.Name}},
		}, nil, nil
	})

	ct, st := mcpgo.NewInMemoryTransports()
	go func() { _ = server.Run(ctx, st) }()

	client := mcpgo.NewClient(mcpsdk.DefaultImplementation, nil)
	cs, err := client.Connect(ctx, ct, nil)
	require.NoError(t, err)

	session := mcpsdk.Adapt(cs)
	defer session.Close()

	tools, err := session.ListTools(ctx)
	require.NoError(t, err)
	require.Len(t, tools, 1)
	require.Equal(t, "greet", tools[0].Name)
	require.Equal(t, "Greet someone", tools[0].Title, "title must survive the real protocol")
	require.NotNil(t, tools[0].Annotations, "annotations must survive the real protocol")
	require.True(t, *tools[0].Annotations.DestructiveHint)
	require.NotNil(t, tools[0].InputSchema, "the server's inferred schema must survive")

	res, err := session.CallTool(ctx, "greet", json.RawMessage(`{"name":"Wesley"}`))
	require.NoError(t, err)
	require.False(t, res.IsError)
	require.Equal(t, "hello Wesley", model.ResultText(res))
}

// A real server reporting a tool error must arrive as IsError, not as a Go error.
func TestAdapter_ToolErrorArrivesAsIsError(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	server := mcpgo.NewServer(&mcpgo.Implementation{Name: "test-server", Version: "1"}, nil)
	mcpgo.AddTool(server, &mcpgo.Tool{Name: "fail", Description: "Always fails"},
		func(ctx context.Context, req *mcpgo.CallToolRequest, in greetIn) (*mcpgo.CallToolResult, any, error) {
			return &mcpgo.CallToolResult{
				IsError: true,
				Content: []mcpgo.Content{&mcpgo.TextContent{Text: "no such user"}},
			}, nil, nil
		})

	ct, st := mcpgo.NewInMemoryTransports()
	go func() { _ = server.Run(ctx, st) }()

	client := mcpgo.NewClient(mcpsdk.DefaultImplementation, nil)
	cs, err := client.Connect(ctx, ct, nil)
	require.NoError(t, err)
	session := mcpsdk.Adapt(cs)
	defer session.Close()

	res, err := session.CallTool(ctx, "fail", json.RawMessage(`{"name":"x"}`))
	require.NoError(t, err, "a tool error must not surface as a call failure")
	require.True(t, res.IsError, "IsError must survive the real protocol")
	require.Contains(t, model.ResultText(res), "no such user")
}
