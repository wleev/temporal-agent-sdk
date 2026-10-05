package mcpsdk_test

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	gosdk "github.com/modelcontextprotocol/go-sdk/mcp"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/wleev/temporal-agent-sdk/mcp"
	"github.com/wleev/temporal-agent-sdk/mcp/mcpsdk"
	"github.com/wleev/temporal-agent-sdk/model"
)

// connectProgressServer returns a client connected to an in-memory MCP server
// with one tool, "slow". A call carrying a progress token gets two progress
// notifications and its result 100ms later. tokens counts those calls.
func connectProgressServer(t *testing.T, tokens *atomic.Int32, opts ...mcpsdk.Option) mcp.Client {
	t.Helper()

	server := gosdk.NewServer(&gosdk.Implementation{Name: "progress-test", Version: "1"}, nil)
	gosdk.AddTool(server, &gosdk.Tool{Name: "slow"},
		func(ctx context.Context, req *gosdk.CallToolRequest, _ struct{}) (*gosdk.CallToolResult, any, error) {
			if token := req.Params.GetProgressToken(); token != nil {
				tokens.Add(1)
				for i := 1; i <= 2; i++ {
					err := req.Session.NotifyProgress(ctx, &gosdk.ProgressNotificationParams{
						ProgressToken: token,
						Progress:      float64(i),
						Total:         2,
						Message:       fmt.Sprintf("step %d", i),
					})
					if err != nil {
						return nil, nil, err
					}
				}
				// The tool keeps running after its last notification.
				time.Sleep(100 * time.Millisecond)
			}
			return &gosdk.CallToolResult{Content: []gosdk.Content{&gosdk.TextContent{Text: "done"}}}, nil, nil
		})

	clientTransport, serverTransport := gosdk.NewInMemoryTransports()
	_, err := server.Connect(t.Context(), serverTransport, nil)
	require.NoError(t, err)

	factory := mcpsdk.Factory(
		func(context.Context) (gosdk.Transport, error) { return clientTransport, nil }, opts...)
	c, err := factory(t.Context())
	require.NoError(t, err)
	t.Cleanup(func() { _ = c.Close() })
	return c
}

func TestSession_ReportsServerProgressToTheCallsContext(t *testing.T) {
	var tokens atomic.Int32
	c := connectProgressServer(t, &tokens)

	var mu sync.Mutex
	var got []mcp.ProgressUpdate
	ctx := mcp.WithProgressFunc(t.Context(), func(u mcp.ProgressUpdate) {
		mu.Lock()
		defer mu.Unlock()
		got = append(got, u)
	})

	res, err := c.CallTool(ctx, "slow", json.RawMessage(`{}`))
	require.NoError(t, err)
	assert.Equal(t, "done", model.ResultText(res))

	require.Eventually(t, func() bool {
		mu.Lock()
		defer mu.Unlock()
		return len(got) == 2
	}, 5*time.Second, 5*time.Millisecond, "both progress notifications reach the function")
	mu.Lock()
	defer mu.Unlock()
	assert.ElementsMatch(t, []mcp.ProgressUpdate{
		{Progress: 1, Total: 2, Message: "step 1"},
		{Progress: 2, Total: 2, Message: "step 2"},
	}, got)
}

func TestSession_SendsNoProgressTokenWithoutAProgressFunction(t *testing.T) {
	var tokens atomic.Int32
	c := connectProgressServer(t, &tokens)

	_, err := c.CallTool(t.Context(), "slow", json.RawMessage(`{}`))

	require.NoError(t, err)
	assert.Zero(t, tokens.Load())
}

func TestSession_PingReachesTheServer(t *testing.T) {
	var tokens atomic.Int32
	c := connectProgressServer(t, &tokens)

	pinger, ok := c.(mcp.Pinger)
	require.True(t, ok, "the adapter's session implements mcp.Pinger")
	assert.NoError(t, pinger.Ping(t.Context()))
}

func TestSession_KeepsTheCallersProgressNotificationHandler(t *testing.T) {
	var tokens, handled atomic.Int32
	c := connectProgressServer(t, &tokens, mcpsdk.WithClientOptions(&gosdk.ClientOptions{
		ProgressNotificationHandler: func(context.Context, *gosdk.ProgressNotificationClientRequest) {
			handled.Add(1)
		},
	}))

	var reported atomic.Int32
	ctx := mcp.WithProgressFunc(t.Context(), func(mcp.ProgressUpdate) { reported.Add(1) })
	_, err := c.CallTool(ctx, "slow", json.RawMessage(`{}`))
	require.NoError(t, err)

	require.Eventually(t, func() bool { return handled.Load() == 2 && reported.Load() == 2 },
		5*time.Second, 5*time.Millisecond, "both the caller's handler and the progress function get each notification")
}
