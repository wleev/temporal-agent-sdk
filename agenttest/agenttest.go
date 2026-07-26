// Package agenttest provides test doubles for agent workflows.
//
// The centerpiece is [FakeProvider], a scripted [model.Provider] that returns
// canned responses instead of calling an LLM, making agent tests fast, offline,
// and deterministic.
//
//	fake := agenttest.NewFakeProvider(
//		agenttest.CallsTool("get_weather", `{"city":"Ghent"}`),
//		agenttest.Says("It is 18°C in Ghent."),
//	)
//	acts, err := model.NewActivities(fake)
//	// ... handle err ...
//	env.RegisterActivityWithOptions(
//		acts.InvokeModel,
//		activity.RegisterOptions{Name: model.InvokeModelActivity},
//	)
package agenttest

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"

	"github.com/wleev/temporal-agent-sdk/model"
)

// FakeProvider is a scripted [model.Provider].
//
// It hands out responses in order, one per call. Calling more times than there
// are scripted responses returns an error rather than a default, so a test
// catches an agent that loops more than expected.
type FakeProvider struct {
	mu        sync.Mutex
	responses []Response
	calls     []model.Request
	name      string
}

// Response is one scripted turn.
type Response struct {
	// Message is what the model returns.
	Message model.Message

	// Err, if set, is returned instead of Message. Use it to exercise retry and
	// failure paths.
	Err error

	// Usage is the reported token usage.
	Usage model.Usage

	// FinishReason overrides the default finish reason (stop, or tool_calls when
	// the message calls tools). Set it to [model.FinishLength] to exercise
	// length-continuation.
	FinishReason model.FinishReason
}

// Option configures a [FakeProvider].
type Option func(*FakeProvider)

// WithName sets the provider's name. Defaults to "fake".
func WithName(name string) Option {
	return func(p *FakeProvider) { p.name = name }
}

// NewFakeProvider builds a provider that replays responses in order.
func NewFakeProvider(responses ...Response) *FakeProvider {
	return &FakeProvider{responses: responses, name: "fake"}
}

// NewFakeProviderWith builds a provider with options.
func NewFakeProviderWith(opts []Option, responses ...Response) *FakeProvider {
	p := NewFakeProvider(responses...)
	for _, o := range opts {
		o(p)
	}
	return p
}

// Says scripts a plain text answer, which ends the loop.
func Says(content string) Response {
	return Response{
		Message: model.AssistantMessage(content),
		Usage:   model.Usage{PromptTokens: 10, CompletionTokens: 5, TotalTokens: 15},
	}
}

// CallsTool scripts a single tool call. args must be valid JSON.
func CallsTool(name, args string) Response {
	return CallsTools(ToolCall{Name: name, Args: args})
}

// ToolCall describes one scripted tool call.
type ToolCall struct {
	// ID is the tool call ID. Empty auto-generates "call-<n>" per call.
	ID   string
	Name string
	Args string
}

// CallsTools scripts several tool calls in one assistant turn, exercising
// parallel dispatch.
func CallsTools(calls ...ToolCall) Response {
	blocks := make([]model.Block, 0, len(calls))
	for i, c := range calls {
		id := c.ID
		if id == "" {
			id = fmt.Sprintf("call-%d", i+1)
		}
		args := c.Args
		if args == "" {
			args = "{}"
		}
		blocks = append(blocks, model.ToolCallBlock(id, c.Name, json.RawMessage(args)))
	}
	return Response{
		Message: model.Message{Role: model.RoleAssistant, Blocks: blocks},
		Usage:   model.Usage{PromptTokens: 10, CompletionTokens: 5, TotalTokens: 15},
	}
}

// Truncated scripts a partial text answer that stopped at the token limit
// (finish reason [model.FinishLength]), to exercise length-continuation.
func Truncated(content string) Response {
	r := Says(content)
	r.FinishReason = model.FinishLength
	return r
}

// Fails scripts an error.
func Fails(err error) Response { return Response{Err: err} }

// Name implements [model.Provider].
func (p *FakeProvider) Name() string { return p.name }

// Invoke implements [model.Provider], returning the next scripted response.
func (p *FakeProvider) Invoke(_ context.Context, req model.Request) (model.Response, error) {
	p.mu.Lock()
	defer p.mu.Unlock()

	p.calls = append(p.calls, req)

	if len(p.calls) > len(p.responses) {
		return model.Response{}, fmt.Errorf(
			"agenttest: model called %d time(s) but only %d response(s) were scripted; "+
				"the agent looped more than the test expected",
			len(p.calls), len(p.responses))
	}

	r := p.responses[len(p.calls)-1]
	if r.Err != nil {
		return model.Response{}, r.Err
	}

	finish := r.FinishReason
	if finish == "" {
		finish = model.FinishStop
		if len(r.Message.ToolCalls()) > 0 {
			finish = model.FinishToolCalls
		}
	}
	resp := model.Response{Message: r.Message, Usage: r.Usage, FinishReason: finish}

	// Mirror a real provider: when the request asked for structured output and
	// this is a terminal answer, surface the content as structured output.
	model.SetStructuredOutput(&resp, req)
	return resp, nil
}

// InvokeStream implements [model.StreamingProvider], forwarding the scripted
// message content as a single delta and returning the same aggregated response
// Invoke would — so streaming paths can be exercised end to end.
func (p *FakeProvider) InvokeStream(ctx context.Context, req model.Request, sink model.StreamSink) (model.Response, error) {
	resp, err := p.Invoke(ctx, req)
	if err != nil {
		return resp, err
	}
	if text := resp.Message.Text(); text != "" {
		_ = sink.OnDelta(ctx, model.StreamDelta{Text: text, ToolCallIndex: -1})
	}
	for i, tc := range resp.Message.ToolCalls() {
		_ = sink.OnDelta(ctx, model.StreamDelta{
			ToolCallIndex: i, ToolCallID: tc.ID, ToolName: tc.Name, ArgsFragment: string(tc.Arguments),
		})
	}
	return resp, nil
}

// Calls returns the requests the provider received, in order. Use it to assert
// what the agent actually sent — tool schemas, accumulated history, settings.
func (p *FakeProvider) Calls() []model.Request {
	p.mu.Lock()
	defer p.mu.Unlock()
	return append([]model.Request(nil), p.calls...)
}

// CallCount returns how many times the model was invoked.
func (p *FakeProvider) CallCount() int {
	p.mu.Lock()
	defer p.mu.Unlock()
	return len(p.calls)
}

// LastCall returns the most recent request. It panics if there was none.
func (p *FakeProvider) LastCall() model.Request {
	p.mu.Lock()
	defer p.mu.Unlock()
	if len(p.calls) == 0 {
		panic("agenttest: no model calls were made")
	}
	return p.calls[len(p.calls)-1]
}
