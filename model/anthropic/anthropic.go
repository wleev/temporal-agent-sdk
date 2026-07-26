// Package anthropic implements a [model.Provider] against the Anthropic Messages
// API.
//
// The Messages API differs from Chat Completions: the system prompt is a
// top-level parameter, tool calls and tool results are content blocks rather
// than separate fields, and max_tokens is required.
package anthropic

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strconv"
	"strings"
	"time"

	anth "github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"

	"github.com/wleev/temporal-agent-sdk/model"
)

// DefaultMaxTokens is used when a request does not set [model.Settings.MaxTokens].
// Anthropic requires max_tokens, which [model.Settings] leaves optional, so the
// provider supplies this value when the request omits one.
const DefaultMaxTokens = 4096

// Provider calls the Anthropic Messages API.
type Provider struct {
	client           anth.Client // NewClient returns a value, not a pointer
	name             string
	defaultMaxTokens int64
	customize        func(*anth.MessageNewParams)
}

// Option configures a [Provider].
type Option func(*config)

type config struct {
	name      string
	apiKey    string
	baseURL   string
	maxTokens int64
	http      *http.Client
	extra     []option.RequestOption
	customize func(*anth.MessageNewParams)
}

// WithName sets the provider's registered name. Defaults to "anthropic".
func WithName(name string) Option { return func(c *config) { c.name = name } }

// WithAPIKey sets the API key. Defaults to $ANTHROPIC_API_KEY.
func WithAPIKey(key string) Option { return func(c *config) { c.apiKey = key } }

// WithBaseURL points the provider at an Anthropic-compatible endpoint, e.g. a
// gateway.
func WithBaseURL(url string) Option { return func(c *config) { c.baseURL = url } }

// WithDefaultMaxTokens overrides [DefaultMaxTokens] for requests that do not set
// their own limit.
func WithDefaultMaxTokens(n int64) Option { return func(c *config) { c.maxTokens = n } }

// WithHTTPClient supplies a custom HTTP client.
func WithHTTPClient(h *http.Client) Option { return func(c *config) { c.http = h } }

// WithRequestOptions passes options straight through to the Anthropic client.
//
// The provider sets WithMaxRetries(0) first, so passing a retry count here
// re-enables client-side retries beneath Temporal's — see [New].
func WithRequestOptions(opts ...option.RequestOption) Option {
	return func(c *config) { c.extra = append(c.extra, opts...) }
}

// WithParams sets provider-specific request parameters the neutral
// [model.Settings] does not model — top_k, stop sequences, a thinking config,
// service tier, and the like.
//
// The hook runs on every request after the neutral settings are applied. It is
// the typed, construction-time home for Anthropic-specific config; for anything
// beyond request shaping, implement [model.Provider].
//
//	anthropic.New(anthropic.WithParams(func(p *anthropic.MessageNewParams) {
//	    p.TopK = anthropic.Int(40)
//	    p.StopSequences = []string{"STOP"}
//	}))
func WithParams(fn func(*anth.MessageNewParams)) Option {
	return func(c *config) { c.customize = fn }
}

// New builds a provider.
//
// Client-side retries are disabled so Temporal owns retry: the Anthropic client
// would otherwise default to two internal retries beneath the activity retry
// policy.
func New(opts ...Option) (*Provider, error) {
	cfg := config{name: "anthropic", maxTokens: DefaultMaxTokens}
	for _, o := range opts {
		o(&cfg)
	}
	if cfg.name == "" {
		return nil, errors.New("anthropic: provider name must not be empty")
	}
	if cfg.maxTokens <= 0 {
		cfg.maxTokens = DefaultMaxTokens
	}

	reqOpts := []option.RequestOption{option.WithMaxRetries(0)}
	if cfg.apiKey != "" {
		reqOpts = append(reqOpts, option.WithAPIKey(cfg.apiKey))
	}
	if cfg.baseURL != "" {
		reqOpts = append(reqOpts, option.WithBaseURL(cfg.baseURL))
	}
	if cfg.http != nil {
		reqOpts = append(reqOpts, option.WithHTTPClient(cfg.http))
	}
	reqOpts = append(reqOpts, cfg.extra...)

	return &Provider{
		client:           anth.NewClient(reqOpts...),
		name:             cfg.name,
		defaultMaxTokens: cfg.maxTokens,
		customize:        cfg.customize,
	}, nil
}

// NewWithClient builds a provider around a client you construct yourself.
//
// This is the full-control injection point: build the [anth.Client] with whatever
// middleware, authentication, or transport you need — or wrap an existing one —
// and hand it over. Name, default max tokens, and WithParams still apply;
// client-construction options like WithAPIKey and WithBaseURL do not, since the
// client is already built.
//
// You own the client's retry setting: pass option.WithMaxRetries(0) when
// building it unless you want retries beneath Temporal's.
func NewWithClient(client anth.Client, opts ...Option) (*Provider, error) {
	cfg := config{name: "anthropic", maxTokens: DefaultMaxTokens}
	for _, o := range opts {
		o(&cfg)
	}
	if cfg.name == "" {
		return nil, errors.New("anthropic: provider name must not be empty")
	}
	if cfg.maxTokens <= 0 {
		cfg.maxTokens = DefaultMaxTokens
	}
	return &Provider{
		client:           client,
		name:             cfg.name,
		defaultMaxTokens: cfg.maxTokens,
		customize:        cfg.customize,
	}, nil
}

// Name implements [model.Provider].
func (p *Provider) Name() string { return p.name }

// Invoke implements [model.Provider].
func (p *Provider) Invoke(ctx context.Context, req model.Request) (model.Response, error) {
	params, err := p.buildParams(req)
	if err != nil {
		return model.Response{}, err
	}

	resp, err := p.client.Messages.New(ctx, params)
	if err != nil {
		return model.Response{}, toAPIError(err)
	}
	out := fromMessage(resp)
	model.SetStructuredOutput(&out, req)
	return out, nil
}

// InvokeStream implements [model.StreamingProvider].
//
// It forwards text and tool-input deltas to the sink as they arrive and returns
// the fully aggregated response — identical to what Invoke would return — so the
// workflow sees the same result. A sink error stops forwarding but never fails
// the call.
func (p *Provider) InvokeStream(ctx context.Context, req model.Request, sink model.StreamSink) (model.Response, error) {
	params, err := p.buildParams(req)
	if err != nil {
		return model.Response{}, err
	}

	stream := p.client.Messages.NewStreaming(ctx, params)
	defer func() { _ = stream.Close() }()

	var msg anth.Message
	sinkFailed := false
	for stream.Next() {
		event := stream.Current()
		if err := msg.Accumulate(event); err != nil {
			return model.Response{}, &model.APIError{StatusCode: 0, Err: err}
		}
		if sinkFailed {
			continue
		}
		if cbd, ok := event.AsAny().(anth.ContentBlockDeltaEvent); ok {
			var d model.StreamDelta
			switch delta := cbd.Delta.AsAny().(type) {
			case anth.TextDelta:
				d = model.StreamDelta{Text: delta.Text, ToolCallIndex: -1}
			case anth.InputJSONDelta:
				// Tool-use argument fragment; the block index identifies the call.
				d = model.StreamDelta{ToolCallIndex: int(cbd.Index), ArgsFragment: delta.PartialJSON}
			default:
				continue
			}
			if err := sink.OnDelta(ctx, d); err != nil {
				sinkFailed = true
			}
		}
	}
	if err := stream.Err(); err != nil {
		return model.Response{}, toAPIError(err)
	}

	out := fromMessage(&msg)
	model.SetStructuredOutput(&out, req)
	return out, nil
}

func (p *Provider) buildParams(req model.Request) (anth.MessageNewParams, error) {
	params, err := p.toParams(req)
	if err != nil {
		return anth.MessageNewParams{}, err
	}
	if p.customize != nil {
		// Runs after the neutral mapping, so it may add or override fields.
		p.customize(&params)
	}
	return params, nil
}

func (p *Provider) toParams(req model.Request) (anth.MessageNewParams, error) {
	system, msgs, err := toMessages(req.Messages)
	if err != nil {
		return anth.MessageNewParams{}, err
	}

	// Model is a defined string type, so gateway aliases work alongside the
	// library's constants.
	params := anth.MessageNewParams{
		Model:    req.Model,
		Messages: msgs,
	}
	if len(system) > 0 {
		params.System = system
	}

	tools, err := toTools(req.Tools)
	if err != nil {
		return anth.MessageNewParams{}, err
	}
	if len(tools) > 0 {
		params.Tools = tools
	}

	s := req.Settings

	// max_tokens is required by Anthropic but optional in [model.Settings].
	if s.MaxTokens != nil {
		params.MaxTokens = *s.MaxTokens
	} else {
		params.MaxTokens = p.defaultMaxTokens
	}

	if s.Temperature != nil {
		params.Temperature = anth.Float(*s.Temperature)
	}
	if s.TopP != nil {
		params.TopP = anth.Float(*s.TopP)
	}

	if sch := req.OutputSchema; sch != nil {
		// Anthropic's native structured output: OutputConfig.Format.Schema. The
		// field is map[string]any, so the raw JSON schema is decoded into a map.
		// Like OpenAI's response_format, it binds only the terminal message.
		var schema map[string]any
		if len(sch.Schema) > 0 {
			if err := json.Unmarshal(sch.Schema, &schema); err != nil {
				return anth.MessageNewParams{}, fmt.Errorf("anthropic: invalid output schema: %w", err)
			}
		}
		params.OutputConfig = anth.OutputConfigParam{
			Format: anth.JSONOutputFormatParam{Schema: schema},
		}
	}

	// Anthropic-specific knobs (top_k, stop sequences, thinking) are set through
	// WithParams at construction, not the neutral Settings.

	return params, nil
}

// toMessages converts the neutral message list into Anthropic's shape.
//
// Two structural differences are handled here:
//
//   - System messages are not a role in Anthropic; they are hoisted into the
//     top-level system parameter.
//   - A tool result is not its own message; it is a tool_result block inside a
//     user message. The neutral loop emits one message per tool result (OpenAI's
//     shape), so consecutive tool results are coalesced into a single user turn,
//     which is what Anthropic expects after a parallel tool-use turn.
func toMessages(msgs []model.Message) (system []anth.TextBlockParam, out []anth.MessageParam, err error) {
	var pendingResults []anth.ContentBlockParamUnion

	flush := func() {
		if len(pendingResults) > 0 {
			out = append(out, anth.NewUserMessage(pendingResults...))
			pendingResults = nil
		}
	}

	for i, m := range msgs {
		switch m.Role {
		case model.RoleSystem:
			flush()
			system = append(system, anth.TextBlockParam{Text: m.Text()})
		case model.RoleUser:
			flush()
			out = append(out, userMessage(m))
		case model.RoleAssistant:
			flush()
			out = append(out, toAssistantMessage(m))
		case model.RoleTool:
			// Batch, don't flush: multiple tool results for one assistant turn
			// belong in the same user message. Chat-style tool results are
			// text, so the rich result is flattened here at the edge.
			for _, blk := range m.Blocks {
				if blk.Kind == model.BlockToolResult {
					pendingResults = append(pendingResults,
						anth.NewToolResultBlock(blk.ToolCallID, model.ResultText(blk.Result), false))
				}
			}
		default:
			return nil, nil, fmt.Errorf("anthropic: message %d has unknown role %q", i, m.Role)
		}
	}
	flush()
	return system, out, nil
}

// userMessage builds a user message, keeping the plain-text form unless the
// message carries media. Anthropic takes image input as a base64 image block;
// audio and other MIME types it cannot show become a named placeholder, the same
// degradation [model.ResultText] applies to non-text tool output.
func userMessage(m model.Message) anth.MessageParam {
	hasMedia := false
	for _, blk := range m.Blocks {
		if blk.Kind == model.BlockMedia {
			hasMedia = true
			break
		}
	}
	if !hasMedia {
		return anth.NewUserMessage(anth.NewTextBlock(m.Text()))
	}

	var blocks []anth.ContentBlockParamUnion
	for _, blk := range m.Blocks {
		switch blk.Kind {
		case model.BlockText:
			if blk.Text != "" {
				blocks = append(blocks, anth.NewTextBlock(blk.Text))
			}
		case model.BlockMedia:
			if len(blk.Data) > 0 && strings.HasPrefix(blk.MIMEType, "image/") {
				blocks = append(blocks, anth.NewImageBlockBase64(blk.MIMEType, base64.StdEncoding.EncodeToString(blk.Data)))
			} else {
				// No inline bytes (a URI block for a scheme this provider cannot
				// fetch) or an unsupported type flattens to a named placeholder.
				blocks = append(blocks, anth.NewTextBlock(model.MediaPlaceholder(blk)))
			}
		}
	}
	if len(blocks) == 0 {
		blocks = append(blocks, anth.NewTextBlock(""))
	}
	return anth.NewUserMessage(blocks...)
}

// toAssistantMessage rebuilds a prior assistant turn, preserving block order. A
// thinking block must be echoed back with its signature before the tool_use it
// preceded, or the API rejects the turn.
func toAssistantMessage(m model.Message) anth.MessageParam {
	var blocks []anth.ContentBlockParamUnion
	for _, blk := range m.Blocks {
		switch blk.Kind {
		case model.BlockThinking:
			blocks = append(blocks, anth.NewThinkingBlock(blk.Signature, blk.Text))
		case model.BlockText:
			if blk.Text != "" {
				blocks = append(blocks, anth.NewTextBlock(blk.Text))
			}
		case model.BlockToolCall:
			// Input is any; json.RawMessage marshals as its raw bytes, so the tool
			// arguments pass through unchanged.
			var input any = json.RawMessage(`{}`)
			if len(blk.Arguments) > 0 {
				input = blk.Arguments
			}
			blocks = append(blocks, anth.NewToolUseBlock(blk.ToolCallID, input, blk.ToolName))
		}
	}
	if len(blocks) == 0 {
		// Anthropic rejects an empty content array; a contentless assistant turn
		// only arises from malformed history, but guard it anyway.
		blocks = append(blocks, anth.NewTextBlock(""))
	}
	return anth.NewAssistantMessage(blocks...)
}

func toTools(tools []*model.Tool) ([]anth.ToolUnionParam, error) {
	if len(tools) == 0 {
		return nil, nil
	}
	out := make([]anth.ToolUnionParam, 0, len(tools))
	for _, t := range tools {
		if t == nil {
			continue
		}
		schema, err := toInputSchema(t)
		if err != nil {
			return nil, err
		}
		tp := anth.ToolParam{Name: t.Name, InputSchema: schema}
		if t.Description != "" {
			tp.Description = anth.String(t.Description)
		}
		out = append(out, anth.ToolUnionParam{OfTool: &tp})
	}
	return out, nil
}

// toInputSchema decomposes a JSON Schema into Anthropic's ToolInputSchemaParam,
// which splits it into properties, required, and a defaulted type. Remaining keys
// (additionalProperties and the like) are carried through as extra fields.
func toInputSchema(t *model.Tool) (anth.ToolInputSchemaParam, error) {
	raw, err := model.ToolInputSchemaJSON(t)
	if err != nil {
		return anth.ToolInputSchemaParam{}, fmt.Errorf("anthropic: tool %q: %w", t.Name, err)
	}

	schema := anth.ToolInputSchemaParam{}
	if len(raw) == 0 {
		return schema, nil
	}

	var m map[string]any
	if err := json.Unmarshal(raw, &m); err != nil {
		return anth.ToolInputSchemaParam{}, fmt.Errorf("anthropic: tool %q has an invalid input schema: %w", t.Name, err)
	}

	if props, ok := m["properties"]; ok {
		schema.Properties = props
	}
	if req, ok := m["required"].([]any); ok {
		schema.Required = toStringSlice(req)
	}

	// Keys Anthropic does not model explicitly are preserved so
	// additionalProperties:false and any future keyword still reach the model.
	// Map iteration order does not matter: this runs in the activity and only
	// affects the request, not replay.
	var extra map[string]any
	for k, v := range m {
		switch k {
		case "type", "properties", "required":
		default:
			if extra == nil {
				extra = map[string]any{}
			}
			extra[k] = v
		}
	}
	schema.ExtraFields = extra

	return schema, nil
}

func toStringSlice(vs []any) []string {
	out := make([]string, 0, len(vs))
	for _, v := range vs {
		if s, ok := v.(string); ok {
			out = append(out, s)
		}
	}
	return out
}

// fromMessage converts an Anthropic response into the neutral shape, preserving
// content-block order: thinking, text, and tool_use become neutral blocks in the
// same sequence. Thinking blocks are captured with their signature so they can be
// echoed back on the next turn.
func fromMessage(m *anth.Message) model.Response {
	out := model.Message{Role: model.RoleAssistant}

	for _, block := range m.Content {
		switch v := block.AsAny().(type) {
		case anth.ThinkingBlock:
			out.Blocks = append(out.Blocks, model.ThinkingBlock(v.Thinking, v.Signature))
		case anth.TextBlock:
			out.Blocks = append(out.Blocks, model.TextBlock(v.Text))
		case anth.ToolUseBlock:
			args := v.Input
			if len(args) == 0 {
				args = json.RawMessage(`{}`)
			}
			out.Blocks = append(out.Blocks, model.ToolCallBlock(v.ID, v.Name, args))
		}
	}

	return model.Response{
		Message:      out,
		FinishReason: finishReason(m.StopReason),
		Usage: model.Usage{
			PromptTokens:     m.Usage.InputTokens,
			CompletionTokens: m.Usage.OutputTokens,
			TotalTokens:      m.Usage.InputTokens + m.Usage.OutputTokens,
		},
	}
}

// finishReason maps an Anthropic stop reason to a [model.FinishReason].
func finishReason(sr anth.StopReason) model.FinishReason {
	switch sr {
	case anth.StopReasonEndTurn, anth.StopReasonStopSequence, "":
		return model.FinishStop
	case anth.StopReasonMaxTokens:
		return model.FinishLength
	case anth.StopReasonToolUse:
		return model.FinishToolCalls
	case anth.StopReasonRefusal:
		return model.FinishContentFilter
	default:
		// pause_turn and any reason a newer API adds.
		return model.FinishOther
	}
}

// toAPIError converts an Anthropic client error into the neutral retry-posture
// form. 429 (rate limit) and 529 (overloaded) both fall under the existing
// retryable rules (429, or status >= 500). Anthropic sends no x-should-retry
// header, so ShouldRetry stays nil and the status-based decision applies.
func toAPIError(err error) error {
	var e *anth.Error
	if !errors.As(err, &e) {
		// No HTTP response: dial failure, timeout, or context error. Status 0
		// marks it transport-level, which Retryable treats as retryable.
		return &model.APIError{StatusCode: 0, Err: err}
	}

	apiErr := &model.APIError{StatusCode: e.StatusCode, Err: err}
	if e.Response != nil {
		apiErr.RetryAfter = parseRetryAfter(e.Response.Header)
	}
	return apiErr
}

// parseRetryAfter reads Anthropic's retry-after header (seconds, or an HTTP
// date).
func parseRetryAfter(h http.Header) time.Duration {
	v := h.Get("retry-after")
	if v == "" {
		return 0
	}
	if secs, err := strconv.ParseFloat(v, 64); err == nil && secs > 0 {
		return time.Duration(secs * float64(time.Second))
	}
	if t, err := http.ParseTime(v); err == nil {
		if d := time.Until(t); d > 0 {
			return d
		}
	}
	return 0
}
