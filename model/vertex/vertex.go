// Package vertex implements a [model.Provider] against Google's Gemini models
// through the google.golang.org/genai SDK.
//
// One provider covers both backends the SDK unifies: Vertex AI (a GCP project and
// location, authenticated with Application Default Credentials) and the Gemini
// Developer API (an API key). Vertex is the documented primary; select the Gemini
// API backend with [WithAPIKey].
//
// Gemini's wire shape differs from both Chat Completions and the Messages API:
// the system prompt is a top-level SystemInstruction, tool calls and results are
// Parts inside Contents, function responses live in a user turn, and content is
// natively multimodal.
package vertex

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strings"

	"google.golang.org/genai"

	"github.com/wleev/temporal-agent-sdk/model"
)

// Provider calls Gemini through the genai SDK.
type Provider struct {
	client    *genai.Client
	name      string
	customize func(*genai.GenerateContentConfig)
}

// Option configures a [Provider].
type Option func(*config)

type config struct {
	name      string
	apiKey    string
	project   string
	location  string
	baseURL   string
	http      *http.Client
	customize func(*genai.GenerateContentConfig)
}

// WithName sets the provider's registered name. Defaults to "vertex".
func WithName(name string) Option { return func(c *config) { c.name = name } }

// WithAPIKey selects the Gemini Developer API backend with an API key. Defaults
// to $GEMINI_API_KEY / $GOOGLE_API_KEY as the SDK resolves them.
func WithAPIKey(key string) Option { return func(c *config) { c.apiKey = key } }

// WithProject selects the Vertex AI backend with a GCP project ID. Credentials
// come from Application Default Credentials.
func WithProject(project string) Option { return func(c *config) { c.project = project } }

// WithLocation sets the Vertex AI region, e.g. "us-central1". Selects the Vertex
// AI backend.
func WithLocation(location string) Option { return func(c *config) { c.location = location } }

// WithBaseURL points the client at a custom endpoint. Mainly for tests and
// gateways.
func WithBaseURL(url string) Option { return func(c *config) { c.baseURL = url } }

// WithHTTPClient supplies a custom HTTP client.
func WithHTTPClient(h *http.Client) Option { return func(c *config) { c.http = h } }

// WithParams sets Gemini-specific request config the neutral [model.Settings]
// does not model — a thinking config, safety settings, top_k, and the like. The
// hook runs on every request after the neutral mapping, so it may add or override
// fields.
//
//	vertex.New(vertex.WithParams(func(c *genai.GenerateContentConfig) {
//	    c.ThinkingConfig = &genai.ThinkingConfig{IncludeThoughts: true}
//	}))
func WithParams(fn func(*genai.GenerateContentConfig)) Option {
	return func(c *config) { c.customize = fn }
}

// New builds a provider, constructing the genai client from the options.
//
// The backend is chosen from the options: a project or location selects Vertex
// AI; an API key selects the Gemini Developer API; with neither, the SDK
// auto-detects from the environment (ADC, GOOGLE_GENAI_USE_VERTEXAI, and friends).
// The genai client does not retry GenerateContent internally, so Temporal owns
// retry with nothing to disable.
func New(opts ...Option) (*Provider, error) {
	cfg := config{name: "vertex"}
	for _, o := range opts {
		o(&cfg)
	}
	if cfg.name == "" {
		return nil, errors.New("vertex: provider name must not be empty")
	}

	cc := &genai.ClientConfig{
		APIKey:     cfg.apiKey,
		Project:    cfg.project,
		Location:   cfg.location,
		HTTPClient: cfg.http,
	}
	switch {
	case cfg.project != "" || cfg.location != "":
		cc.Backend = genai.BackendVertexAI
	case cfg.apiKey != "":
		cc.Backend = genai.BackendGeminiAPI
	}
	if cfg.baseURL != "" {
		cc.HTTPOptions = genai.HTTPOptions{BaseURL: cfg.baseURL}
	}

	client, err := genai.NewClient(context.Background(), cc)
	if err != nil {
		return nil, fmt.Errorf("vertex: %w", err)
	}
	return &Provider{client: client, name: cfg.name, customize: cfg.customize}, nil
}

// NewWithClient builds a provider around a genai client you construct yourself.
//
// This is the full-control injection point: build the [genai.Client] with
// whatever credentials, transport, or middleware you need — or hand over one you
// already have. Name and WithParams still apply; the client-construction options
// (WithAPIKey, WithProject, WithLocation, WithBaseURL, WithHTTPClient) do not,
// since the client is already built.
func NewWithClient(client *genai.Client, opts ...Option) (*Provider, error) {
	cfg := config{name: "vertex"}
	for _, o := range opts {
		o(&cfg)
	}
	if cfg.name == "" {
		return nil, errors.New("vertex: provider name must not be empty")
	}
	if client == nil {
		return nil, errors.New("vertex: client must not be nil")
	}
	return &Provider{client: client, name: cfg.name, customize: cfg.customize}, nil
}

// Name implements [model.Provider].
func (p *Provider) Name() string { return p.name }

// Invoke implements [model.Provider].
func (p *Provider) Invoke(ctx context.Context, req model.Request) (model.Response, error) {
	contents, cfg, err := p.build(req)
	if err != nil {
		return model.Response{}, err
	}
	resp, err := p.client.Models.GenerateContent(ctx, req.Model, contents, cfg)
	if err != nil {
		return model.Response{}, toAPIError(err)
	}
	out := fromResponse(resp)
	model.SetStructuredOutput(&out, req)
	return out, nil
}

// InvokeStream implements [model.StreamingProvider].
//
// It forwards text deltas as they arrive and returns the fully aggregated
// response — identical to what Invoke would return — so the workflow sees the
// same result. Gemini streams text in fragments and tool calls as whole parts, so
// the aggregate coalesces text into one block; thinking, text, then tool calls is
// the canonical order. A sink error stops forwarding but never fails the call.
func (p *Provider) InvokeStream(ctx context.Context, req model.Request, sink model.StreamSink) (model.Response, error) {
	contents, cfg, err := p.build(req)
	if err != nil {
		return model.Response{}, err
	}

	var text, thinking strings.Builder
	var thinkingSig []byte
	var funcCalls []*genai.FunctionCall
	var usage *genai.GenerateContentResponseUsageMetadata
	var finish genai.FinishReason
	sinkFailed := false

	for resp, err := range p.client.Models.GenerateContentStream(ctx, req.Model, contents, cfg) {
		if err != nil {
			return model.Response{}, toAPIError(err)
		}
		if resp.UsageMetadata != nil {
			usage = resp.UsageMetadata
		}
		if len(resp.Candidates) == 0 {
			continue
		}
		cand := resp.Candidates[0]
		if cand.FinishReason != "" {
			finish = cand.FinishReason
		}
		if cand.Content == nil {
			continue
		}
		for _, part := range cand.Content.Parts {
			switch {
			case part.FunctionCall != nil:
				funcCalls = append(funcCalls, part.FunctionCall)
			case part.Thought && part.Text != "":
				thinking.WriteString(part.Text)
				if len(part.ThoughtSignature) > 0 {
					thinkingSig = part.ThoughtSignature
				}
			case part.Text != "":
				text.WriteString(part.Text)
				if !sinkFailed {
					if err := sink.OnDelta(ctx, model.StreamDelta{Text: part.Text, ToolCallIndex: -1}); err != nil {
						sinkFailed = true
					}
				}
			}
		}
	}

	out := model.Message{Role: model.RoleAssistant}
	if thinking.Len() > 0 {
		out.Blocks = append(out.Blocks, model.ThinkingBlock(thinking.String(), encodeSignature(thinkingSig)))
	}
	if text.Len() > 0 {
		out.Blocks = append(out.Blocks, model.TextBlock(text.String()))
	}
	for i, fc := range funcCalls {
		id := fc.ID
		if id == "" {
			id = fmt.Sprintf("call_%d", i)
		}
		args := encodeArgs(fc.Args)
		out.Blocks = append(out.Blocks, model.ToolCallBlock(id, fc.Name, args))
		if !sinkFailed {
			_ = sink.OnDelta(ctx, model.StreamDelta{
				ToolCallIndex: i, ToolCallID: id, ToolName: fc.Name, ArgsFragment: string(args),
			})
		}
	}

	r := model.Response{Message: out, FinishReason: finishReason(finish)}
	if usage != nil {
		r.Usage = usageFrom(usage)
	}
	model.SetStructuredOutput(&r, req)
	return r, nil
}

func (p *Provider) build(req model.Request) ([]*genai.Content, *genai.GenerateContentConfig, error) {
	system, contents, err := toContents(req.Messages)
	if err != nil {
		return nil, nil, err
	}

	cfg := &genai.GenerateContentConfig{}
	if system != nil {
		cfg.SystemInstruction = system
	}

	tools, err := toTools(req.Tools)
	if err != nil {
		return nil, nil, err
	}
	if len(tools) > 0 {
		cfg.Tools = tools
	}

	s := req.Settings
	if s.Temperature != nil {
		t := float32(*s.Temperature)
		cfg.Temperature = &t
	}
	if s.TopP != nil {
		t := float32(*s.TopP)
		cfg.TopP = &t
	}
	if s.MaxTokens != nil {
		cfg.MaxOutputTokens = int32(*s.MaxTokens) //nolint:gosec // a token limit fits in int32
	}

	if sch := req.OutputSchema; sch != nil {
		// Gemini's controlled generation: a JSON MIME type plus a JSON Schema. The
		// SDK accepts raw JSON Schema via ResponseJsonSchema, so our schema passes
		// through unchanged, like OpenAI's response_format.
		cfg.ResponseMIMEType = "application/json"
		if len(sch.Schema) > 0 {
			var schema any
			if err := json.Unmarshal(sch.Schema, &schema); err != nil {
				return nil, nil, fmt.Errorf("vertex: invalid output schema: %w", err)
			}
			cfg.ResponseJsonSchema = schema
		}
	}

	if p.customize != nil {
		p.customize(cfg)
	}
	return contents, cfg, nil
}

// toContents converts the neutral messages into Gemini's shape.
//
// Like Anthropic, Gemini has no system role (the system prompt is hoisted into
// SystemInstruction) and expects tool results in a user turn — so consecutive
// tool-result messages are coalesced into one user Content of FunctionResponse
// parts. Gemini's FunctionResponse needs the function name, which a tool-result
// block does not carry, so the name is resolved from the matching tool call.
func toContents(msgs []model.Message) (*genai.Content, []*genai.Content, error) {
	callName := map[string]string{}
	for _, m := range msgs {
		for _, blk := range m.Blocks {
			if blk.Kind == model.BlockToolCall {
				callName[blk.ToolCallID] = blk.ToolName
			}
		}
	}

	var system *genai.Content
	var out []*genai.Content
	var pending []*genai.Part

	flush := func() {
		if len(pending) > 0 {
			out = append(out, &genai.Content{Role: genai.RoleUser, Parts: pending})
			pending = nil
		}
	}

	for i, m := range msgs {
		switch m.Role {
		case model.RoleSystem:
			flush()
			if system == nil {
				system = &genai.Content{}
			}
			system.Parts = append(system.Parts, &genai.Part{Text: m.Text()})
		case model.RoleUser:
			flush()
			out = append(out, &genai.Content{Role: genai.RoleUser, Parts: userParts(m)})
		case model.RoleAssistant:
			flush()
			parts := assistantParts(m)
			// Coalesce consecutive model turns into one Content; Gemini requires
			// user and model turns to alternate.
			if n := len(out); n > 0 && out[n-1].Role == genai.RoleModel {
				out[n-1].Parts = append(out[n-1].Parts, parts...)
			} else {
				out = append(out, &genai.Content{Role: genai.RoleModel, Parts: parts})
			}
		case model.RoleTool:
			for _, blk := range m.Blocks {
				if blk.Kind == model.BlockToolResult {
					pending = append(pending, &genai.Part{FunctionResponse: &genai.FunctionResponse{
						ID:       blk.ToolCallID,
						Name:     callName[blk.ToolCallID],
						Response: map[string]any{"result": model.ResultText(blk.Result)},
					}})
				}
			}
		default:
			return nil, nil, fmt.Errorf("vertex: message %d has unknown role %q", i, m.Role)
		}
	}
	flush()
	return system, out, nil
}

// userParts maps a user message's blocks to Gemini parts. Gemini is natively
// multimodal, so media rides through as inline_data with no flattening.
func userParts(m model.Message) []*genai.Part {
	var parts []*genai.Part
	for _, blk := range m.Blocks {
		switch blk.Kind {
		case model.BlockText:
			parts = append(parts, &genai.Part{Text: blk.Text})
		case model.BlockMedia:
			if blk.URI != "" && len(blk.Data) == 0 {
				// A gs:// object survives resolution and rides through natively.
				parts = append(parts, &genai.Part{FileData: &genai.FileData{FileURI: blk.URI, MIMEType: blk.MIMEType}})
			} else {
				parts = append(parts, &genai.Part{InlineData: &genai.Blob{Data: blk.Data, MIMEType: blk.MIMEType}})
			}
		}
	}
	if len(parts) == 0 {
		parts = append(parts, &genai.Part{Text: ""})
	}
	return parts
}

// assistantParts rebuilds a prior assistant turn, preserving block order so a
// thinking part and its signature are echoed back ahead of the tool call they
// preceded.
func assistantParts(m model.Message) []*genai.Part {
	var parts []*genai.Part
	for _, blk := range m.Blocks {
		switch blk.Kind {
		case model.BlockThinking:
			parts = append(parts, &genai.Part{
				Text: blk.Text, Thought: true, ThoughtSignature: decodeSignature(blk.Signature),
			})
		case model.BlockText:
			if blk.Text != "" {
				parts = append(parts, &genai.Part{Text: blk.Text})
			}
		case model.BlockToolCall:
			parts = append(parts, &genai.Part{FunctionCall: &genai.FunctionCall{
				ID: blk.ToolCallID, Name: blk.ToolName, Args: decodeArgs(blk.Arguments),
			}})
		}
	}
	return parts
}

func toTools(tools []*model.Tool) ([]*genai.Tool, error) {
	if len(tools) == 0 {
		return nil, nil
	}
	var decls []*genai.FunctionDeclaration
	for _, t := range tools {
		if t == nil {
			continue
		}
		raw, err := model.ToolInputSchemaJSON(t)
		if err != nil {
			return nil, fmt.Errorf("vertex: tool %q: %w", t.Name, err)
		}
		decl := &genai.FunctionDeclaration{Name: t.Name, Description: t.Description}
		if len(raw) > 0 {
			var schema any
			if err := json.Unmarshal(raw, &schema); err != nil {
				return nil, fmt.Errorf("vertex: tool %q has an invalid input schema: %w", t.Name, err)
			}
			decl.ParametersJsonSchema = schema
		}
		decls = append(decls, decl)
	}
	if len(decls) == 0 {
		return nil, nil
	}
	return []*genai.Tool{{FunctionDeclarations: decls}}, nil
}

// fromResponse converts a Gemini response into the neutral shape, walking the
// candidate's parts in order. A thought part becomes a thinking block carrying
// its signature (base64-encoded, since the neutral signature is a string); a
// function-call part becomes a tool-call block, synthesizing an ID when Gemini
// omits one, since tool results are keyed by ID.
func fromResponse(resp *genai.GenerateContentResponse) model.Response {
	out := model.Message{Role: model.RoleAssistant}
	var finish genai.FinishReason

	if len(resp.Candidates) > 0 {
		cand := resp.Candidates[0]
		finish = cand.FinishReason
		if cand.Content != nil {
			fnIdx := 0
			for _, part := range cand.Content.Parts {
				switch {
				case part.FunctionCall != nil:
					fc := part.FunctionCall
					id := fc.ID
					if id == "" {
						id = fmt.Sprintf("call_%d", fnIdx)
					}
					fnIdx++
					out.Blocks = append(out.Blocks, model.ToolCallBlock(id, fc.Name, encodeArgs(fc.Args)))
				case part.Thought && part.Text != "":
					out.Blocks = append(out.Blocks, model.ThinkingBlock(part.Text, encodeSignature(part.ThoughtSignature)))
				case part.Text != "":
					out.Blocks = append(out.Blocks, model.TextBlock(part.Text))
				}
			}
		}
	}

	r := model.Response{Message: out, FinishReason: finishReason(finish)}
	if resp.UsageMetadata != nil {
		r.Usage = usageFrom(resp.UsageMetadata)
	}
	return r
}

// finishReason maps a Gemini finish reason to a [model.FinishReason]. Gemini
// reports STOP even when the turn is a function call, so tool calls are detected
// from the response parts, not here.
func finishReason(fr genai.FinishReason) model.FinishReason {
	switch fr {
	case genai.FinishReasonStop, genai.FinishReasonUnspecified, "":
		return model.FinishStop
	case genai.FinishReasonMaxTokens:
		return model.FinishLength
	case genai.FinishReasonSafety, genai.FinishReasonRecitation, genai.FinishReasonBlocklist,
		genai.FinishReasonProhibitedContent, genai.FinishReasonSPII, genai.FinishReasonImageSafety,
		genai.FinishReasonImageProhibitedContent, genai.FinishReasonImageRecitation:
		return model.FinishContentFilter
	default:
		return model.FinishOther
	}
}

func usageFrom(u *genai.GenerateContentResponseUsageMetadata) model.Usage {
	return model.Usage{
		PromptTokens:     int64(u.PromptTokenCount),
		CompletionTokens: int64(u.CandidatesTokenCount),
		TotalTokens:      int64(u.TotalTokenCount),
	}
}

// toAPIError maps a genai error into the neutral retry-posture form. The SDK's
// APIError carries the HTTP status in Code, which is all model.APIError needs to
// classify retryability; a non-API error (dial failure, context) is status 0,
// which Retryable treats as retryable.
func toAPIError(err error) error {
	var e genai.APIError
	if errors.As(err, &e) {
		return &model.APIError{StatusCode: e.Code, Err: err}
	}
	return &model.APIError{StatusCode: 0, Err: err}
}

// decodeArgs turns a tool call's raw JSON arguments into the map Gemini expects.
func decodeArgs(raw json.RawMessage) map[string]any {
	if len(raw) == 0 {
		return nil
	}
	var m map[string]any
	if err := json.Unmarshal(raw, &m); err != nil {
		return nil
	}
	return m
}

// encodeArgs turns Gemini's argument map back into raw JSON, defaulting to an
// empty object so a downstream provider never sees nil arguments.
func encodeArgs(args map[string]any) json.RawMessage {
	if len(args) == 0 {
		return json.RawMessage(`{}`)
	}
	raw, err := json.Marshal(args)
	if err != nil {
		return json.RawMessage(`{}`)
	}
	return raw
}

// encodeSignature renders Gemini's opaque []byte thought signature as the string
// a neutral thinking block carries; decodeSignature reverses it.
func encodeSignature(sig []byte) string {
	if len(sig) == 0 {
		return ""
	}
	return base64.StdEncoding.EncodeToString(sig)
}

func decodeSignature(sig string) []byte {
	if sig == "" {
		return nil
	}
	if b, err := base64.StdEncoding.DecodeString(sig); err == nil {
		return b
	}
	return nil
}
