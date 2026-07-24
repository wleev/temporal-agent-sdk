// Package model defines the provider-neutral seam between the agent loop and
// the LLM.
//
// The agent loop runs inside a Temporal workflow and must be deterministic; the
// LLM call is network I/O and must not. Everything in this package is plain data
// that survives serialization into workflow history.
//
// [Request] carries tool descriptions rather than tools: a tool holds a Go func,
// which cannot be serialized and has no meaning in the activity process. Only
// the name, description, and argument schema cross the wire; the func stays in
// the workflow, where the loop invokes it after the model asks for it.
//
// Tool descriptions use MCP's Tool type directly, so a tool from any MCP server
// passes through without loss and local Go tools produce the same type.
// Provider-specific loss happens at the provider edge (for example, Chat
// Completions tool results are text-only and must be flattened), not in this
// seam.
//
// MCP does not describe execution policy — whether a tool runs in the workflow
// or an activity, whether it needs approval, its retry policy. Those live on the
// runtime object in the tool package.
package model

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"

	mcpsdk "github.com/modelcontextprotocol/go-sdk/mcp"
)

// Tool is the wire description of a callable tool, MCP's Tool type. Construct one
// with [NewTool] for the common case.
type Tool = mcpsdk.Tool

// ToolAnnotations are MCP's behavioral hints about a tool.
//
// Per the MCP spec these are hints and "are not guaranteed to provide a
// faithful description of tool behavior"; they come from the server. Never let
// them weaken a safety decision; see tool.ApprovalPolicy.
type ToolAnnotations = mcpsdk.ToolAnnotations

// CallToolResult is the result of invoking a tool, MCP's result type. A server's
// response — text, images, embedded resources, structured content, and the
// IsError flag — survives intact up to the provider edge.
type CallToolResult = mcpsdk.CallToolResult

// NewTool builds a [Tool] from a name, description, and JSON Schema. The schema
// is raw JSON; MCP's InputSchema is typed any and accepts it directly.
func NewTool(name, description string, inputSchema []byte) *Tool {
	return &Tool{
		Name:        name,
		Description: description,
		InputSchema: json.RawMessage(inputSchema),
	}
}

// TextResult builds a plain text [CallToolResult].
func TextResult(text string) *CallToolResult {
	return &CallToolResult{
		Content: []mcpsdk.Content{&mcpsdk.TextContent{Text: text}},
	}
}

// ErrorResult builds a [CallToolResult] marked as a tool error.
//
// This reports a failure of the tool, not of the call: the agent loop hands the
// text to the model rather than failing the workflow, so the model can act on it.
func ErrorResult(text string) *CallToolResult {
	r := TextResult(text)
	r.IsError = true
	return r
}

// ResultText flattens a tool result to plain text for a model that cannot accept
// anything richer (for example, text-only Chat Completions tool results).
//
// Non-text content is rendered as a short placeholder naming what was omitted, so
// the model can relay it rather than treating it as seen. Marshaling a fixed
// value is deterministic, so this is replay-safe.
//
//workflowcheck:ignore json.Marshal is deterministic for a fixed value
func ResultText(r *CallToolResult) string {
	if r == nil {
		return ""
	}

	var b strings.Builder
	for _, c := range r.Content {
		if b.Len() > 0 {
			b.WriteString("\n")
		}
		switch v := c.(type) {
		case *mcpsdk.TextContent:
			b.WriteString(v.Text)
		case *mcpsdk.ImageContent:
			b.WriteString(fmt.Sprintf("[image omitted: %s content cannot be shown to this model]", v.MIMEType))
		case *mcpsdk.AudioContent:
			b.WriteString(fmt.Sprintf("[audio omitted: %s content cannot be shown to this model]", v.MIMEType))
		case *mcpsdk.ResourceLink:
			b.WriteString(fmt.Sprintf("[resource: %s]", v.URI))
		case *mcpsdk.EmbeddedResource:
			b.WriteString(embeddedResourceText(v))
		default:
			b.WriteString(fmt.Sprintf("[unsupported content type %T omitted]", c))
		}
	}

	// A result carrying only structured content falls back to its JSON.
	if b.Len() == 0 && r.StructuredContent != nil {
		if raw, err := json.Marshal(r.StructuredContent); err == nil {
			return string(raw)
		}
	}
	return b.String()
}

// embeddedResourceText renders an embedded resource, inlining it when it is
// text and naming it otherwise.
func embeddedResourceText(v *mcpsdk.EmbeddedResource) string {
	if v.Resource == nil {
		return "[embedded resource omitted]"
	}
	if v.Resource.Text != "" {
		return v.Resource.Text
	}
	return fmt.Sprintf("[embedded resource omitted: %s (%s)]", v.Resource.URI, v.Resource.MIMEType)
}

// Role identifies the author of a [Message].
type Role string

// The roles a [Message] author can have.
const (
	RoleSystem    Role = "system"
	RoleUser      Role = "user"
	RoleAssistant Role = "assistant"
	RoleTool      Role = "tool"
)

// ToolCall is a model's request to invoke a tool. It is the convenience view of
// a [BlockToolCall] block, returned by [Message.ToolCalls].
type ToolCall struct {
	ID        string          `json:"id"`
	Name      string          `json:"name"`
	Arguments json.RawMessage `json:"arguments"`
}

// BlockKind is the type of a content [Block].
type BlockKind string

// The kinds of content a [Block] can hold.
const (
	BlockText       BlockKind = "text"
	BlockToolCall   BlockKind = "tool_call"
	BlockToolResult BlockKind = "tool_result"
	BlockThinking   BlockKind = "thinking"
	BlockMedia      BlockKind = "media"
)

// Block is one ordered piece of a message's content. Ordering is preserved, so
// an assistant turn can be thinking, then text, then a tool call, then more text.
// It is a struct union — a Kind discriminator plus per-kind fields — so it
// serializes as plain JSON with no custom marshaling, keeping workflow history
// deterministic.
type Block struct {
	Kind BlockKind `json:"kind"`

	// Text is the content of a text or thinking block.
	Text string `json:"text,omitempty"`

	// ToolCallID identifies a tool call (BlockToolCall) or the call a result
	// answers (BlockToolResult).
	ToolCallID string          `json:"tool_call_id,omitempty"`
	ToolName   string          `json:"tool_name,omitempty"` // BlockToolCall
	Arguments  json.RawMessage `json:"arguments,omitempty"` // BlockToolCall

	// Result is a tool's rich output (BlockToolResult), carried unflattened so a
	// provider can render it however it can; text-only providers flatten it with
	// [ResultText] at their edge.
	Result *CallToolResult `json:"result,omitempty"`

	// Signature is the opaque signature of a thinking block (BlockThinking). It
	// must be echoed back verbatim on the next turn.
	Signature string `json:"signature,omitempty"`

	// MIMEType and Data carry binary content (BlockMedia) — an image or audio clip
	// in a user message. Data marshals as base64, so a media block survives
	// workflow history deterministically. Providers that cannot accept the MIME
	// type flatten it to a named placeholder at their edge.
	MIMEType string `json:"mime_type,omitempty"` // BlockMedia
	Data     []byte `json:"data,omitempty"`      // BlockMedia
}

// Message is one entry in a conversation: an author and an ordered list of
// content blocks.
type Message struct {
	Role   Role    `json:"role"`
	Blocks []Block `json:"blocks,omitempty"`
}

// TextBlock builds a text block.
func TextBlock(text string) Block { return Block{Kind: BlockText, Text: text} }

// ToolCallBlock builds a tool-call block.
func ToolCallBlock(id, name string, args json.RawMessage) Block {
	return Block{Kind: BlockToolCall, ToolCallID: id, ToolName: name, Arguments: args}
}

// ThinkingBlock builds a thinking block; signature is echoed back verbatim.
func ThinkingBlock(text, signature string) Block {
	return Block{Kind: BlockThinking, Text: text, Signature: signature}
}

// SystemMessage builds a system message from plain text.
func SystemMessage(text string) Message {
	return Message{Role: RoleSystem, Blocks: []Block{TextBlock(text)}}
}

// UserMessage builds a user message from plain text.
func UserMessage(text string) Message {
	return Message{Role: RoleUser, Blocks: []Block{TextBlock(text)}}
}

// AssistantMessage builds a plain-text assistant message.
func AssistantMessage(text string) Message {
	return Message{Role: RoleAssistant, Blocks: []Block{TextBlock(text)}}
}

// ToolResultMessage builds a tool-result message carrying a rich result, which
// stays unflattened until a provider renders it.
func ToolResultMessage(callID string, result *CallToolResult) Message {
	return Message{Role: RoleTool, Blocks: []Block{{
		Kind: BlockToolResult, ToolCallID: callID, Result: result,
	}}}
}

// ImageBlock builds a media block holding an image, e.g.
// ImageBlock("image/png", pngBytes). Put it in a user message with [UserContent].
func ImageBlock(mimeType string, data []byte) Block {
	return Block{Kind: BlockMedia, MIMEType: mimeType, Data: data}
}

// AudioBlock builds a media block holding an audio clip, e.g.
// AudioBlock("audio/wav", wavBytes). Not every provider can accept audio; those
// that cannot render it as a named placeholder.
func AudioBlock(mimeType string, data []byte) Block {
	return Block{Kind: BlockMedia, MIMEType: mimeType, Data: data}
}

// UserContent builds a user message from an ordered list of blocks — the way to
// mix text and media in one turn, e.g.
// UserContent(TextBlock("what is this?"), ImageBlock("image/png", pngBytes)).
func UserContent(blocks ...Block) Message {
	return Message{Role: RoleUser, Blocks: blocks}
}

// MediaPlaceholder names a media block a provider cannot render, so the model is
// told what was omitted rather than the content being dropped silently. Providers
// call it for any MIME type they do not map.
func MediaPlaceholder(b Block) string {
	kind := "media"
	switch {
	case strings.HasPrefix(b.MIMEType, "image/"):
		kind = "image"
	case strings.HasPrefix(b.MIMEType, "audio/"):
		kind = "audio"
	case strings.HasPrefix(b.MIMEType, "video/"):
		kind = "video"
	}
	return fmt.Sprintf("[%s omitted: %s content cannot be shown to this model]", kind, b.MIMEType)
}

// Text returns the message's text, concatenating its text blocks; thinking and
// tool blocks are excluded. A final answer is read from this.
func (m Message) Text() string {
	var b strings.Builder
	for _, blk := range m.Blocks {
		if blk.Kind == BlockText {
			b.WriteString(blk.Text)
		}
	}
	return b.String()
}

// ToolCalls returns the message's tool-call blocks as [ToolCall] values, in
// order. An empty result means the model produced no tool calls this turn.
func (m Message) ToolCalls() []ToolCall {
	var out []ToolCall
	for _, blk := range m.Blocks {
		if blk.Kind == BlockToolCall {
			out = append(out, ToolCall{ID: blk.ToolCallID, Name: blk.ToolName, Arguments: blk.Arguments})
		}
	}
	return out
}

// ToolResultText returns the flattened text of the message's tool-result blocks,
// the same rendering a text-only provider sends to the model.
func (m Message) ToolResultText() string {
	var b strings.Builder
	for _, blk := range m.Blocks {
		if blk.Kind == BlockToolResult {
			b.WriteString(ResultText(blk.Result))
		}
	}
	return b.String()
}

// ToolResultID returns the call ID of the message's first tool-result block, or
// "" if it has none, tying a result back to the call that produced it.
func (m Message) ToolResultID() string {
	for _, blk := range m.Blocks {
		if blk.Kind == BlockToolResult {
			return blk.ToolCallID
		}
	}
	return ""
}

// Settings are provider-neutral generation parameters: sampling temperature,
// nucleus sampling, and output length. A nil field means "provider default",
// distinct from an explicit zero (as Temperature 0 relies on).
//
// Provider-specific parameters are not included here. Configure them where the
// provider is built: the standard providers expose common knobs as construction
// options (e.g. openai.WithParams) and accept an injected, pre-wrapped client
// (NewWithClient); anything they do not cover is handled by implementing
// [Provider].
type Settings struct {
	Temperature *float64 `json:"temperature,omitempty"`
	TopP        *float64 `json:"top_p,omitempty"`
	MaxTokens   *int64   `json:"max_tokens,omitempty"`
}

// Ptr returns a pointer to v, for setting [Settings]' optional fields inline:
// model.Ptr(0.7), model.Ptr[int64](256).
func Ptr[T any](v T) *T { return &v }

// Request is the input to the model activity.
type Request struct {
	// Provider selects a registered [Provider] by name. Empty uses the sole
	// registered provider, which is the common single-provider case.
	Provider string `json:"provider,omitempty"`

	Model    string    `json:"model"`
	Messages []Message `json:"messages"`

	// Tools the model may call. Providers map these to their own format and are
	// responsible for any loss that mapping entails.
	Tools []*Tool `json:"tools,omitempty"`

	// OutputSchema, when set, constrains the model's final answer to JSON
	// matching the schema. Providers wire it to their native structured-output
	// field. It binds only the terminal (non-tool) message; the model still
	// calls tools normally in intermediate turns.
	OutputSchema *OutputSchema `json:"output_schema,omitempty"`

	// Stream requests that the activity forward live deltas to a configured
	// [StreamSink] while producing this response. It affects only whether external
	// consumers get live tokens; the aggregated [Response] is identical either way,
	// so the workflow and replay are unaffected.
	Stream bool `json:"stream,omitempty"`

	Settings Settings `json:"settings,omitzero"`

	// Metadata is opaque per-request key-values the SDK passes to the [Provider]
	// unchanged and never interprets. A provider that routes — say a multi-tenant
	// gateway that resolves the concrete model, endpoint, and credentials from a
	// tenant and feature — reads its routing keys here instead of overloading
	// Model. Set it per run with agent.RunOptions.ProviderMetadata.
	Metadata map[string]string `json:"metadata,omitempty"`
}

// OutputSchema describes a required structured output shape.
//
// Schema is a JSON Schema, typically reflected from a Go type (see the tool
// package's use of internal/schema). Strict requests the provider's strict mode
// where available.
type OutputSchema struct {
	Name   string          `json:"name"`
	Schema json.RawMessage `json:"schema"`
	Strict bool            `json:"strict,omitempty"`
}

// Usage reports token consumption for a single call.
type Usage struct {
	PromptTokens     int64 `json:"prompt_tokens"`
	CompletionTokens int64 `json:"completion_tokens"`
	TotalTokens      int64 `json:"total_tokens"`
}

// Response is the output of the model activity.
type Response struct {
	Message      Message `json:"message"`
	Usage        Usage   `json:"usage"`
	FinishReason string  `json:"finish_reason,omitempty"`

	// StructuredOutput is the model's final answer as raw JSON, set only when the
	// request carried an [OutputSchema] and the model produced a terminal
	// (non-tool) message. It is the JSON the caller decodes into their type.
	StructuredOutput json.RawMessage `json:"structured_output,omitempty"`
}

// SetStructuredOutput fills resp.StructuredOutput from the terminal message.
//
// When the request asked for structured output and the model produced a terminal
// (non-tool) message, that message's content is the JSON answer. Providers call
// this after mapping the response.
func SetStructuredOutput(resp *Response, req Request) {
	if req.OutputSchema == nil || len(resp.Message.ToolCalls()) > 0 {
		return
	}
	text := resp.Message.Text()
	if text == "" {
		return
	}
	resp.StructuredOutput = json.RawMessage(text)
}

// StreamDelta is one incremental piece of a streamed response.
//
// Deltas are best-effort live output for external consumers; the durable answer
// is always the aggregated [Response] the activity returns. A delta carries
// either a text fragment or a tool-call argument fragment (keyed by
// ToolCallIndex, since tool arguments arrive split across deltas).
type StreamDelta struct {
	// Text is a content fragment, appended to the assistant message.
	Text string

	// ToolCallIndex identifies which tool call an argument fragment belongs to;
	// -1 when the delta is not a tool-call fragment.
	ToolCallIndex int
	ToolCallID    string // set when first known for this index
	ToolName      string // set when first known for this index
	ArgsFragment  string // a piece of the tool call's JSON arguments
}

// StreamSink receives live [StreamDelta]s during a streamed model call.
//
// Implementations run inside the model activity and typically forward each delta
// to an external transport (SSE, websocket, pub-sub) keyed by workflow ID, so a
// UI can show tokens as they arrive. A sink error is logged and does not fail the
// call — streaming is best-effort; the durable result is unaffected.
type StreamSink interface {
	OnDelta(ctx context.Context, d StreamDelta) error
}

// StreamingProvider is an optional capability: a [Provider] that can stream live
// deltas while producing its response.
//
// The activity uses it only when the request asks to stream and a sink is
// configured; otherwise it uses [Provider.Invoke]. InvokeStream must return the
// same fully-aggregated [Response] Invoke would, so the workflow and replay see
// identical results whether or not streaming happened.
type StreamingProvider interface {
	Provider
	InvokeStream(ctx context.Context, req Request, sink StreamSink) (Response, error)
}

// Provider talks to one LLM backend.
//
// Implementations run inside an activity and receive a standard
// [context.Context], so they may do network I/O freely. They must not import
// Temporal; retry posture is communicated with [APIError], which the activity
// layer translates. This keeps providers usable and testable outside a workflow.
type Provider interface {
	Name() string
	Invoke(ctx context.Context, req Request) (Response, error)
}

// APIError reports a failed provider call with enough information for the
// activity layer to decide whether Temporal should retry it.
//
// Providers return this instead of Temporal's error types, so the retry decision
// lives in one place and providers stay dependency-free.
type APIError struct {
	// StatusCode is the HTTP status, or 0 if the request never got a response
	// (a dial or timeout failure, which is retryable).
	StatusCode int

	// RetryAfter, when non-zero, is the backend's requested delay before the next
	// attempt. It is passed through to Temporal as an explicit next-retry delay.
	RetryAfter time.Duration

	// ShouldRetry, when non-nil, is the backend's explicit instruction and
	// overrides any status-code inference.
	ShouldRetry *bool

	Err error
}

func (e *APIError) Error() string {
	if e.StatusCode == 0 {
		return fmt.Sprintf("model provider: %v", e.Err)
	}
	return fmt.Sprintf("model provider: http %d: %v", e.StatusCode, e.Err)
}

func (e *APIError) Unwrap() error { return e.Err }

// Retryable reports whether another attempt could plausibly succeed.
//
// An explicit instruction from the backend (ShouldRetry) wins. Otherwise a
// missing response (StatusCode 0) means a transport failure and is retryable;
// among real responses only timeout, conflict, rate limiting, and server errors
// are transient — except 501 Not Implemented, which is permanent. Every other
// 4xx is treated as a request bug and not retried.
func (e *APIError) Retryable() bool {
	if e.ShouldRetry != nil {
		return *e.ShouldRetry
	}
	switch {
	case e.StatusCode == 0:
		return true
	case e.StatusCode == http.StatusRequestTimeout,
		e.StatusCode == http.StatusConflict,
		e.StatusCode == http.StatusTooManyRequests:
		return true
	case e.StatusCode >= http.StatusInternalServerError && e.StatusCode != http.StatusNotImplemented:
		return true
	default:
		return false
	}
}
