package conversation

import (
	"context"
	"fmt"
	"strings"

	"github.com/wleev/temporal-agent-sdk/model"
)

// Compactor shrinks a conversation's history before it is carried into the next
// workflow run.
//
// Its Compact method runs as an activity (it typically calls the model), so it
// takes a plain [context.Context] and may do I/O. It must return a valid message
// list — in particular it must not orphan a tool result by cutting between an
// assistant tool call and its result.
type Compactor interface {
	Compact(ctx context.Context, msgs []model.Message) ([]model.Message, error)
}

// Summarizer produces a short natural-language summary of a run of messages. It
// is the model-dependent piece of [SummarizingCompactor].
type Summarizer interface {
	Summarize(ctx context.Context, msgs []model.Message) (string, error)
}

// SummarizingCompactor keeps the leading system messages and the most recent
// turns verbatim, and replaces everything in between with a single summary
// system message.
//
// KeepLast is a lower bound on the trailing messages retained; the actual cut is
// moved earlier to the nearest turn boundary (a user message) so a tool call and
// its result are never split.
type SummarizingCompactor struct {
	summarizer Summarizer
	keepLast   int
}

// DefaultKeepLast is the default trailing-message floor for
// [SummarizingCompactor].
const DefaultKeepLast = 8

// SummarizingCompactorOption configures a [SummarizingCompactor].
type SummarizingCompactorOption func(*SummarizingCompactor)

// WithKeepLast sets the minimum number of trailing messages kept verbatim. A
// non-positive value keeps [DefaultKeepLast].
func WithKeepLast(n int) SummarizingCompactorOption {
	return func(c *SummarizingCompactor) { c.keepLast = n }
}

// NewSummarizingCompactor builds a compactor that replaces the middle of a
// conversation with a summary produced by summarizer, keeping the leading system
// messages and recent turns verbatim.
func NewSummarizingCompactor(summarizer Summarizer, opts ...SummarizingCompactorOption) *SummarizingCompactor {
	c := &SummarizingCompactor{summarizer: summarizer, keepLast: DefaultKeepLast}
	for _, o := range opts {
		o(c)
	}
	if c.keepLast <= 0 {
		c.keepLast = DefaultKeepLast
	}
	return c
}

// Compact implements [Compactor].
func (c *SummarizingCompactor) Compact(ctx context.Context, msgs []model.Message) ([]model.Message, error) {
	keep := c.keepLast
	if keep <= 0 {
		keep = DefaultKeepLast
	}

	// Leading system messages (the agent instructions, prior summaries) are
	// always preserved.
	head := 0
	for head < len(msgs) && msgs[head].Role == model.RoleSystem {
		head++
	}

	cut := turnBoundary(msgs, len(msgs)-keep, head)
	middle := msgs[head:cut]
	if len(middle) == 0 {
		// Nothing old enough to summarize.
		return msgs, nil
	}

	summary, err := c.summarizer.Summarize(ctx, middle)
	if err != nil {
		return nil, fmt.Errorf("conversation: summarizing history: %w", err)
	}

	out := make([]model.Message, 0, head+1+(len(msgs)-cut))
	out = append(out, msgs[:head]...)
	out = append(out, model.SystemMessage("Summary of earlier conversation:\n"+summary))
	out = append(out, msgs[cut:]...)
	return out, nil
}

// turnBoundary returns a cut index at or before target (but not before min) that
// falls on the start of a turn — a user message — so the retained tail never
// begins with an orphaned tool result. If no such boundary exists it returns
// min, summarizing everything after the head.
func turnBoundary(msgs []model.Message, target, min int) int {
	if target < min {
		target = min
	}
	for i := target; i > min; i-- {
		if msgs[i].Role == model.RoleUser {
			return i
		}
	}
	return min
}

// LLMSummarizer summarizes with a model provider. It is provider-neutral: hand it
// any [model.Provider] (the same one the agent uses, or a cheaper model).
type LLMSummarizer struct {
	provider     model.Provider
	model        string
	instructions string
}

const defaultSummaryInstructions = "Summarize the following conversation concisely, " +
	"preserving facts, decisions, names, and any state the assistant must remember to " +
	"continue helpfully. Write the summary as notes, not dialogue."

// LLMSummarizerOption configures an [LLMSummarizer].
type LLMSummarizerOption func(*LLMSummarizer)

// WithInstructions overrides the default summarization prompt.
func WithInstructions(instructions string) LLMSummarizerOption {
	return func(s *LLMSummarizer) { s.instructions = instructions }
}

// NewLLMSummarizer builds a summarizer backed by a model provider. Hand it any
// [model.Provider] — the agent's own or a cheaper model — and a model identifier.
func NewLLMSummarizer(provider model.Provider, modelID string, opts ...LLMSummarizerOption) *LLMSummarizer {
	s := &LLMSummarizer{provider: provider, model: modelID}
	for _, o := range opts {
		o(s)
	}
	return s
}

// Summarize implements [Summarizer].
func (s *LLMSummarizer) Summarize(ctx context.Context, msgs []model.Message) (string, error) {
	instructions := s.instructions
	if instructions == "" {
		instructions = defaultSummaryInstructions
	}

	resp, err := s.provider.Invoke(ctx, model.Request{
		Model: s.model,
		Messages: []model.Message{
			model.SystemMessage(instructions),
			model.UserMessage(renderTranscript(msgs)),
		},
	})
	if err != nil {
		return "", err
	}
	return strings.TrimSpace(resp.Message.Text()), nil
}

// renderTranscript flattens messages into plain text for the summarizer prompt.
func renderTranscript(msgs []model.Message) string {
	var b strings.Builder
	for _, m := range msgs {
		switch m.Role {
		case model.RoleUser:
			fmt.Fprintf(&b, "User: %s\n", m.Text())
		case model.RoleAssistant:
			if text := m.Text(); text != "" {
				fmt.Fprintf(&b, "Assistant: %s\n", text)
			}
			for _, tc := range m.ToolCalls() {
				fmt.Fprintf(&b, "Assistant called tool %s(%s)\n", tc.Name, string(tc.Arguments))
			}
		case model.RoleTool:
			for _, blk := range m.Blocks {
				if blk.Kind == model.BlockToolResult {
					fmt.Fprintf(&b, "Tool result: %s\n", model.ResultText(blk.Result))
				}
			}
		case model.RoleSystem:
			fmt.Fprintf(&b, "System: %s\n", m.Text())
		}
	}
	return b.String()
}
