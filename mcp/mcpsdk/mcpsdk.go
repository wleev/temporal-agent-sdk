// Package mcpsdk adapts the official MCP Go SDK to [mcp.Client].
//
// It is a separate package so the SDK's client — its transports, OAuth, and
// JSON-RPC machinery — is pulled in only by programs that connect to a server.
//
// A local server over stdio:
//
//	acts := mcp.NewActivities()
//	err := acts.Register("filesystem", mcpsdk.CommandFactory(
//		"npx", "-y", "@modelcontextprotocol/server-filesystem", "/data"))
//
// A remote one:
//
//	err = acts.Register("docs", mcpsdk.StreamableFactory("https://example.com/mcp"))
//
// For anything else — custom transports, auth, client options — use [Factory]
// and build the transport yourself.
package mcpsdk

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"os/exec"

	mcpsdk "github.com/modelcontextprotocol/go-sdk/mcp"

	"github.com/wleev/temporal-agent-sdk/mcp"
	"github.com/wleev/temporal-agent-sdk/model"
)

// DefaultImplementation identifies this client to MCP servers.
var DefaultImplementation = &mcpsdk.Implementation{
	Name:    "temporal-agent-sdk",
	Version: "0.1.0",
}

// Session is a connected MCP client session, satisfying [mcp.Client].
type Session struct {
	cs *mcpsdk.ClientSession
}

// Adapt wraps an already-connected session.
//
// Ownership transfers: [Session.Close] closes the underlying session, which the
// MCP activities always do.
func Adapt(cs *mcpsdk.ClientSession) *Session { return &Session{cs: cs} }

// ListTools implements [mcp.Client]. Tools are returned exactly as the server
// described them.
func (s *Session) ListTools(ctx context.Context) ([]*model.Tool, error) {
	res, err := s.cs.ListTools(ctx, &mcpsdk.ListToolsParams{})
	if err != nil {
		return nil, err
	}
	return res.Tools, nil
}

// CallTool implements [mcp.Client].
//
// A returned error means the call failed. A result with IsError set means the
// tool ran and reported a failure; the distinction is preserved because the
// agent loop retries one and not the other.
func (s *Session) CallTool(ctx context.Context, name string, args json.RawMessage) (*model.CallToolResult, error) {
	params := &mcpsdk.CallToolParams{Name: name}
	if len(args) > 0 {
		// Arguments is typed any and marshaled by the SDK; raw JSON is passed
		// through without decoding and re-encoding.
		params.Arguments = args
	}
	return s.cs.CallTool(ctx, params)
}

// Close implements [mcp.Client].
func (s *Session) Close() error { return s.cs.Close() }

// Factory builds an [mcp.Factory] that connects over the transport returned by
// newTransport.
//
// newTransport is called per activity attempt, so each attempt gets its own
// connection.
func Factory(newTransport func(ctx context.Context) (mcpsdk.Transport, error), opts ...Option) mcp.Factory {
	cfg := resolve(opts)
	return func(ctx context.Context) (mcp.Client, error) {
		t, err := newTransport(ctx)
		if err != nil {
			return nil, err
		}
		client := mcpsdk.NewClient(cfg.impl, cfg.clientOptions)
		cs, err := client.Connect(ctx, t, cfg.sessionOptions)
		if err != nil {
			return nil, fmt.Errorf("mcpsdk: connecting: %w", err)
		}
		return Adapt(cs), nil
	}
}

// CommandFactory connects to a local MCP server run as a subprocess over stdio.
func CommandFactory(name string, args ...string) mcp.Factory {
	return CommandFactoryWith(nil, name, args...)
}

// CommandFactoryWith is [CommandFactory] with client options, and a hook to
// adjust the command — environment, working directory — before it is started.
func CommandFactoryWith(configure func(*exec.Cmd), name string, args ...string) mcp.Factory {
	return Factory(func(ctx context.Context) (mcpsdk.Transport, error) {
		cmd := exec.CommandContext(ctx, name, args...)
		if configure != nil {
			configure(cmd)
		}
		return &mcpsdk.CommandTransport{Command: cmd}, nil
	})
}

// StreamableFactory connects to a remote MCP server over streamable HTTP.
func StreamableFactory(url string, opts ...Option) mcp.Factory {
	return Factory(func(ctx context.Context) (mcpsdk.Transport, error) {
		cfg := resolve(opts)
		t := &mcpsdk.StreamableClientTransport{Endpoint: url}
		if cfg.httpClient != nil {
			t.HTTPClient = cfg.httpClient
		}
		return t, nil
	}, opts...)
}

// Option configures a factory.
type Option func(*config)

type config struct {
	impl           *mcpsdk.Implementation
	clientOptions  *mcpsdk.ClientOptions
	sessionOptions *mcpsdk.ClientSessionOptions
	httpClient     *http.Client
}

func resolve(opts []Option) config {
	c := config{impl: DefaultImplementation}
	for _, o := range opts {
		o(&c)
	}
	if c.impl == nil {
		c.impl = DefaultImplementation
	}
	return c
}

// WithImplementation overrides how this client identifies itself to servers.
func WithImplementation(impl *mcpsdk.Implementation) Option {
	return func(c *config) { c.impl = impl }
}

// WithClientOptions passes options to the underlying MCP client.
func WithClientOptions(opts *mcpsdk.ClientOptions) Option {
	return func(c *config) { c.clientOptions = opts }
}

// WithSessionOptions passes options to the underlying client session.
func WithSessionOptions(opts *mcpsdk.ClientSessionOptions) Option {
	return func(c *config) { c.sessionOptions = opts }
}

// WithHTTPClient sets the HTTP client used by HTTP-based transports, e.g. to
// add authentication.
func WithHTTPClient(h *http.Client) Option {
	return func(c *config) { c.httpClient = h }
}

// compile-time check that a Session is a usable Client.
var _ mcp.Client = (*Session)(nil)
