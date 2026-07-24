// Package openai implements a [model.Provider] against the OpenAI Chat
// Completions API.
//
// It also serves any OpenAI-compatible backend, such as vLLM, via [WithBaseURL].
package openai

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

	oai "github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
	"github.com/openai/openai-go/v3/shared"

	"github.com/wleev/temporal-agent-sdk/model"
)

// Provider calls an OpenAI-compatible Chat Completions endpoint.
type Provider struct {
	client    oai.Client // NewClient returns a value, not a pointer
	name      string
	customize func(*oai.ChatCompletionNewParams)
}

// Option configures a [Provider].
type Option func(*config)

type config struct {
	name      string
	apiKey    string
	baseURL   string
	http      *http.Client
	extra     []option.RequestOption
	customize func(*oai.ChatCompletionNewParams)
}

// WithName sets the provider's registered name. Defaults to "openai". Use this
// to register several backends — say an OpenAI provider and a vLLM one — in the
// same worker.
func WithName(name string) Option { return func(c *config) { c.name = name } }

// WithAPIKey sets the API key. Defaults to $OPENAI_API_KEY. Backends that do not
// authenticate, such as a local vLLM, still expect a non-empty value; any
// placeholder works.
func WithAPIKey(key string) Option { return func(c *config) { c.apiKey = key } }

// WithBaseURL points the provider at an OpenAI-compatible endpoint, e.g.
// "http://localhost:8000/v1" for vLLM.
func WithBaseURL(url string) Option { return func(c *config) { c.baseURL = url } }

// WithHTTPClient supplies a custom HTTP client.
func WithHTTPClient(h *http.Client) Option { return func(c *config) { c.http = h } }

// WithRequestOptions passes options straight through to the OpenAI client.
//
// Note that the provider sets WithMaxRetries(0) first, so passing your own
// retry count here re-enables client-side retries and stacks them under
// Temporal's — see [New].
func WithRequestOptions(opts ...option.RequestOption) Option {
	return func(c *config) { c.extra = append(c.extra, opts...) }
}

// WithParams sets provider-specific request parameters that the neutral
// [model.Settings] does not model — seed, penalties, tool_choice, response
// format, and the like.
//
// The hook runs on every request after the neutral settings are applied, so it
// can add fields or override them. It is the typed, construction-time home for
// OpenAI-specific config; for a parameter this SDK has no field for at all, use
// [WithRequestOptions] with option.WithJSONSet. For anything beyond request
// shaping, implement [model.Provider].
//
//	openai.New(openai.WithParams(func(p *openai.ChatCompletionNewParams) {
//	    p.Seed = openai.Int(42)
//	    p.FrequencyPenalty = openai.Float(0.5)
//	}))
func WithParams(fn func(*oai.ChatCompletionNewParams)) Option {
	return func(c *config) { c.customize = fn }
}

// New builds a provider.
//
// Client-side retries are disabled so Temporal owns retry: the provider makes
// exactly one HTTP call per attempt and reports retry posture through
// [model.APIError]. The OpenAI client would otherwise default to two internal
// retries beneath Temporal's activity retry policy.
func New(opts ...Option) (*Provider, error) {
	cfg := config{name: "openai"}
	for _, o := range opts {
		o(&cfg)
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

	if cfg.name == "" {
		return nil, errors.New("openai: provider name must not be empty")
	}

	return &Provider{client: oai.NewClient(reqOpts...), name: cfg.name, customize: cfg.customize}, nil
}

// NewWithClient builds a provider around a client you construct yourself.
//
// Build the [oai.Client] with whatever middleware, authentication, transport, or
// retry behavior you need, then hand it over. Name and WithParams still apply;
// client-construction options like WithAPIKey and WithBaseURL do not, since the
// client is already built.
//
// You own the client's retry setting: pass option.WithMaxRetries(0) when
// building it unless you want retries beneath Temporal's.
func NewWithClient(client oai.Client, opts ...Option) (*Provider, error) {
	cfg := config{name: "openai"}
	for _, o := range opts {
		o(&cfg)
	}
	if cfg.name == "" {
		return nil, errors.New("openai: provider name must not be empty")
	}
	return &Provider{client: client, name: cfg.name, customize: cfg.customize}, nil
}

// Name implements [model.Provider].
func (p *Provider) Name() string { return p.name }

// params builds the request params, including the construction-time customizer.
func (p *Provider) params(req model.Request) (oai.ChatCompletionNewParams, error) {
	params, err := toParams(req)
	if err != nil {
		return oai.ChatCompletionNewParams{}, err
	}
	if p.customize != nil {
		// Runs after the neutral mapping, so it may add or override fields.
		p.customize(&params)
	}
	return params, nil
}

// Invoke implements [model.Provider].
func (p *Provider) Invoke(ctx context.Context, req model.Request) (model.Response, error) {
	params, err := p.params(req)
	if err != nil {
		return model.Response{}, err
	}

	resp, err := p.client.Chat.Completions.New(ctx, params)
	if err != nil {
		return model.Response{}, toAPIError(err)
	}
	if len(resp.Choices) == 0 {
		// No choices is not an HTTP failure but leaves nothing to act on;
		// status 0 marks it retryable.
		return model.Response{}, &model.APIError{
			StatusCode: 0,
			Err:        errors.New("model returned no choices"),
		}
	}

	out := fromCompletion(resp)
	model.SetStructuredOutput(&out, req)
	return out, nil
}

// InvokeStream implements [model.StreamingProvider].
//
// It forwards each content and tool-argument delta to the sink as it arrives,
// and returns the fully aggregated response — byte-for-byte what Invoke would
// return — so the workflow sees an identical result. A sink error stops
// forwarding but never fails the call: the durable result is unaffected.
func (p *Provider) InvokeStream(ctx context.Context, req model.Request, sink model.StreamSink) (model.Response, error) {
	params, err := p.params(req)
	if err != nil {
		return model.Response{}, err
	}
	// Opt in to usage on the terminal chunk; without this the stream omits it.
	params.StreamOptions = oai.ChatCompletionStreamOptionsParam{IncludeUsage: oai.Bool(true)}

	stream := p.client.Chat.Completions.NewStreaming(ctx, params)
	defer func() { _ = stream.Close() }()

	acc := oai.ChatCompletionAccumulator{}
	sinkFailed := false
	for stream.Next() {
		chunk := stream.Current()
		acc.AddChunk(chunk)
		if sinkFailed || len(chunk.Choices) == 0 {
			continue
		}
		delta := chunk.Choices[0].Delta
		if delta.Content != "" {
			if err := sink.OnDelta(ctx, model.StreamDelta{Text: delta.Content, ToolCallIndex: -1}); err != nil {
				sinkFailed = true
				continue
			}
		}
		for _, tc := range delta.ToolCalls {
			d := model.StreamDelta{
				ToolCallIndex: int(tc.Index),
				ToolCallID:    tc.ID,
				ToolName:      tc.Function.Name,
				ArgsFragment:  tc.Function.Arguments,
			}
			if err := sink.OnDelta(ctx, d); err != nil {
				sinkFailed = true
				break
			}
		}
	}
	if err := stream.Err(); err != nil {
		return model.Response{}, toAPIError(err)
	}
	if len(acc.Choices) == 0 {
		return model.Response{}, &model.APIError{StatusCode: 0, Err: errors.New("model returned no choices")}
	}

	out := fromCompletion(&acc.ChatCompletion)
	model.SetStructuredOutput(&out, req)
	return out, nil
}

func toParams(req model.Request) (oai.ChatCompletionNewParams, error) {
	msgs, err := toMessages(req.Messages)
	if err != nil {
		return oai.ChatCompletionNewParams{}, err
	}

	// ChatModel is a defined string type, so arbitrary names (vLLM model paths,
	// gateway aliases) work alongside the library's constants.
	params := oai.ChatCompletionNewParams{
		Model:    req.Model,
		Messages: msgs,
	}

	tools, err := toTools(req.Tools)
	if err != nil {
		return oai.ChatCompletionNewParams{}, err
	}
	if len(tools) > 0 {
		params.Tools = tools
	}

	s := req.Settings
	if s.Temperature != nil {
		params.Temperature = oai.Float(*s.Temperature)
	}
	if s.TopP != nil {
		params.TopP = oai.Float(*s.TopP)
	}
	if s.MaxTokens != nil {
		// max_tokens is deprecated and rejected by newer models; the successor
		// field is the portable choice.
		params.MaxCompletionTokens = oai.Int(*s.MaxTokens)
	}

	if sch := req.OutputSchema; sch != nil {
		// Schema is typed any; the raw JSON implements json.Marshaler so it is
		// emitted verbatim. This constrains only the terminal message — tools
		// still work in intermediate turns.
		params.ResponseFormat = oai.ChatCompletionNewParamsResponseFormatUnion{
			OfJSONSchema: &shared.ResponseFormatJSONSchemaParam{
				JSONSchema: shared.ResponseFormatJSONSchemaJSONSchemaParam{
					Name:   schemaName(sch.Name),
					Schema: sch.Schema,
					Strict: oai.Bool(sch.Strict),
				},
			},
		}
	}

	return params, nil
}

// schemaName supplies a non-empty name; OpenAI requires one on a json_schema
// response format.
func schemaName(name string) string {
	if name == "" {
		return "output"
	}
	return name
}

func toMessages(msgs []model.Message) ([]oai.ChatCompletionMessageParamUnion, error) {
	out := make([]oai.ChatCompletionMessageParamUnion, 0, len(msgs))
	for i, m := range msgs {
		switch m.Role {
		case model.RoleSystem:
			out = append(out, oai.SystemMessage(m.Text()))
		case model.RoleUser:
			out = append(out, userMessage(m))
		case model.RoleTool:
			// Chat Completions tool results are text, so the rich result is
			// flattened here at the provider edge.
			for _, blk := range m.Blocks {
				if blk.Kind == model.BlockToolResult {
					out = append(out, oai.ToolMessage(model.ResultText(blk.Result), blk.ToolCallID))
				}
			}
		case model.RoleAssistant:
			out = append(out, toAssistantMessage(m))
		default:
			return nil, fmt.Errorf("openai: message %d has unknown role %q", i, m.Role)
		}
	}
	return out, nil
}

// userMessage builds a user message, keeping the plain-string form unless the
// message carries media. Chat Completions takes image input as an image_url part
// with a data URL; audio and other MIME types it cannot show become a named
// placeholder, the same degradation [model.ResultText] applies to tool output.
func userMessage(m model.Message) oai.ChatCompletionMessageParamUnion {
	hasMedia := false
	for _, blk := range m.Blocks {
		if blk.Kind == model.BlockMedia {
			hasMedia = true
			break
		}
	}
	if !hasMedia {
		return oai.UserMessage(m.Text())
	}

	var parts []oai.ChatCompletionContentPartUnionParam
	for _, blk := range m.Blocks {
		switch blk.Kind {
		case model.BlockText:
			parts = append(parts, oai.TextContentPart(blk.Text))
		case model.BlockMedia:
			if strings.HasPrefix(blk.MIMEType, "image/") {
				url := fmt.Sprintf("data:%s;base64,%s", blk.MIMEType, base64.StdEncoding.EncodeToString(blk.Data))
				parts = append(parts, oai.ImageContentPart(oai.ChatCompletionContentPartImageImageURLParam{URL: url}))
			} else {
				parts = append(parts, oai.TextContentPart(model.MediaPlaceholder(blk)))
			}
		}
	}
	return oai.UserMessage(parts)
}

// toAssistantMessage rebuilds a prior assistant turn, including any tool calls.
//
// Replaying tool calls back to the model is mandatory: the API rejects a tool
// result whose originating call is absent from the history.
func toAssistantMessage(m model.Message) oai.ChatCompletionMessageParamUnion {
	// OpenAI's assistant message is content plus tool_calls; block order does not
	// matter to it, and it has no thinking-block echo, so thinking blocks are
	// dropped (OpenAI never produced them).
	text := m.Text()
	toolCalls := m.ToolCalls()

	msg := oai.AssistantMessage(text)
	if len(toolCalls) == 0 {
		return msg
	}

	calls := make([]oai.ChatCompletionMessageToolCallUnionParam, 0, len(toolCalls))
	for _, tc := range toolCalls {
		args := string(tc.Arguments)
		if args == "" {
			args = "{}"
		}
		calls = append(calls, oai.ChatCompletionMessageToolCallUnionParam{
			OfFunction: &oai.ChatCompletionMessageFunctionToolCallParam{
				ID: tc.ID,
				Function: oai.ChatCompletionMessageFunctionToolCallFunctionParam{
					Name:      tc.Name,
					Arguments: args,
				},
			},
		})
	}
	msg.OfAssistant.ToolCalls = calls

	// An assistant turn that only calls tools has no prose. Leaving an empty
	// string in place makes some backends reject the message.
	if text == "" {
		msg.OfAssistant.Content = oai.ChatCompletionAssistantMessageParamContentUnion{}
	}
	return msg
}

// toTools maps MCP tool descriptions onto OpenAI function tools.
//
// Chat Completions has no place for MCP annotations, output schemas, or titles,
// so only name, description, and input schema survive.
func toTools(tools []*model.Tool) ([]oai.ChatCompletionToolUnionParam, error) {
	if len(tools) == 0 {
		return nil, nil
	}
	out := make([]oai.ChatCompletionToolUnionParam, 0, len(tools))
	for _, t := range tools {
		if t == nil {
			continue
		}
		params, err := toFunctionParameters(t)
		if err != nil {
			return nil, err
		}
		def := oai.FunctionDefinitionParam{Name: t.Name, Parameters: params}
		if t.Description != "" {
			def.Description = oai.String(t.Description)
		}
		out = append(out, oai.ChatCompletionFunctionTool(def))
	}
	return out, nil
}

// toFunctionParameters converts an MCP input schema into OpenAI's parameter map.
//
// InputSchema is typed any: it arrives as raw JSON from a local tool's reflector
// or as map[string]any after crossing an activity boundary. Both are handled.
func toFunctionParameters(t *model.Tool) (oai.FunctionParameters, error) {
	switch v := t.InputSchema.(type) {
	case nil:
		return nil, nil
	case oai.FunctionParameters:
		return v, nil
	case map[string]any:
		// The common case after a round trip: already exactly what OpenAI wants.
		return oai.FunctionParameters(v), nil
	case json.RawMessage:
		return unmarshalParameters(t.Name, v)
	case []byte:
		return unmarshalParameters(t.Name, v)
	case string:
		return unmarshalParameters(t.Name, []byte(v))
	default:
		// Some other Go value that marshals to a schema; round-trip it.
		raw, err := json.Marshal(v)
		if err != nil {
			return nil, fmt.Errorf("openai: tool %q has an unmarshalable input schema (%T): %w", t.Name, v, err)
		}
		return unmarshalParameters(t.Name, raw)
	}
}

func unmarshalParameters(name string, raw []byte) (oai.FunctionParameters, error) {
	if len(raw) == 0 {
		return nil, nil
	}
	var params oai.FunctionParameters
	if err := json.Unmarshal(raw, &params); err != nil {
		return nil, fmt.Errorf("openai: tool %q has an invalid input schema: %w", name, err)
	}
	return params, nil
}

func fromCompletion(resp *oai.ChatCompletion) model.Response {
	choice := resp.Choices[0]

	out := model.Message{Role: model.RoleAssistant}
	if choice.Message.Content != "" {
		out.Blocks = append(out.Blocks, model.TextBlock(choice.Message.Content))
	}
	for _, tc := range choice.Message.ToolCalls {
		args := tc.Function.Arguments // raw JSON, carried as a string
		if args == "" {
			args = "{}"
		}
		out.Blocks = append(out.Blocks, model.ToolCallBlock(tc.ID, tc.Function.Name, json.RawMessage(args)))
	}

	return model.Response{
		Message:      out,
		FinishReason: choice.FinishReason,
		Usage: model.Usage{
			PromptTokens:     resp.Usage.PromptTokens,
			CompletionTokens: resp.Usage.CompletionTokens,
			TotalTokens:      resp.Usage.TotalTokens,
		},
	}
}

// toAPIError converts an OpenAI client error into the neutral form the activity
// layer uses to decide retry posture.
func toAPIError(err error) error {
	var oaiErr *oai.Error
	if !errors.As(err, &oaiErr) {
		// No HTTP response: a dial failure, timeout, or context error. Status 0
		// marks it transport-level, which Retryable treats as retryable.
		return &model.APIError{StatusCode: 0, Err: err}
	}

	apiErr := &model.APIError{StatusCode: oaiErr.StatusCode, Err: err}
	if oaiErr.Response != nil {
		apiErr.ShouldRetry = parseShouldRetry(oaiErr.Response.Header)
		apiErr.RetryAfter = parseRetryAfter(oaiErr.Response.Header)
	}
	return apiErr
}

// parseShouldRetry reads the backend's explicit retry instruction, which
// outranks status-code inference.
func parseShouldRetry(h http.Header) *bool {
	switch h.Get("x-should-retry") {
	case "true":
		return ptr(true)
	case "false":
		return ptr(false)
	default:
		return nil
	}
}

// parseRetryAfter reads a requested backoff. The millisecond header is checked
// first because it is more precise; retry-after may also be an HTTP date.
func parseRetryAfter(h http.Header) time.Duration {
	if ms := h.Get("retry-after-ms"); ms != "" {
		if f, err := strconv.ParseFloat(ms, 64); err == nil && f > 0 {
			return time.Duration(f * float64(time.Millisecond))
		}
	}
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

func ptr[T any](v T) *T { return &v }
