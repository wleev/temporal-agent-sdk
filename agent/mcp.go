package agent

import (
	"go.temporal.io/sdk/workflow"

	"github.com/wleev/temporal-agent-sdk/mcp"
	"github.com/wleev/temporal-agent-sdk/tool"
)

// MCPOptions configures how an agent reaches MCP servers, keyed by server name.
// The zero value uses defaults for every server.
type MCPOptions map[string]mcp.Options

// mcpTools returns a server's tools, listing them at most once per session.
//
// A server's tool list is stable for the workflow's lifetime, so it is cached and
// reused across runs rather than issuing a list_tools activity per run.
func (s *Session) mcpTools(ctx workflow.Context, server string) ([]tool.Tool, error) {
	if cached, ok := s.mcpToolCache[server]; ok {
		return cached, nil
	}

	tools, err := mcp.ToolsWith(ctx, server, s.mcpOptions[server])
	if err != nil {
		return nil, err
	}
	s.mcpToolCache[server] = tools
	return tools, nil
}

// SetMCPOptions configures per-server MCP options for the session. Call it
// before the first [Session.Run] that uses those servers; a server already
// listed keeps the options in force at that time.
func (s *Session) SetMCPOptions(opts MCPOptions) {
	s.mcpOptions = opts
}
