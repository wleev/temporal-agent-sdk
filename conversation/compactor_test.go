package conversation_test

import (
	"context"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/wleev/temporal-agent-sdk/conversation"
	"github.com/wleev/temporal-agent-sdk/model"
)

// fixedSummarizer returns a canned summary and records what it was asked to
// summarize.
type fixedSummarizer struct {
	got []model.Message
}

func (s *fixedSummarizer) Summarize(_ context.Context, msgs []model.Message) (string, error) {
	s.got = msgs
	return "earlier: user asked about weather", nil
}

func msg(role model.Role, content string) model.Message {
	return model.Message{Role: role, Blocks: []model.Block{model.TextBlock(content)}}
}

func TestSummarizingCompactor_KeepsHeadAndTailSummarizesMiddle(t *testing.T) {
	sum := &fixedSummarizer{}
	c := conversation.NewSummarizingCompactor(sum, conversation.WithKeepLast(2))

	history := []model.Message{
		msg(model.RoleSystem, "You are helpful."),
		msg(model.RoleUser, "turn 1"),
		msg(model.RoleAssistant, "a1"),
		msg(model.RoleUser, "turn 2"),
		msg(model.RoleAssistant, "a2"),
		msg(model.RoleUser, "turn 3"),
		msg(model.RoleAssistant, "a3"),
	}

	out, err := c.Compact(context.Background(), history)
	require.NoError(t, err)

	// System instructions preserved, then a summary, then the recent tail.
	require.Equal(t, model.RoleSystem, out[0].Role)
	require.Equal(t, "You are helpful.", out[0].Text())
	require.Equal(t, model.RoleSystem, out[1].Role)
	require.Contains(t, out[1].Text(), "earlier: user asked about weather")

	// Fewer messages than before.
	require.Less(t, len(out), len(history))

	// The tail begins on a turn boundary (a user message), never an orphaned
	// tool result.
	require.Equal(t, model.RoleUser, out[2].Role)
}

// The cut must never split an assistant tool call from its tool result.
func TestSummarizingCompactor_DoesNotOrphanToolResults(t *testing.T) {
	sum := &fixedSummarizer{}
	c := conversation.NewSummarizingCompactor(sum, conversation.WithKeepLast(2))

	history := []model.Message{
		msg(model.RoleSystem, "sys"),
		msg(model.RoleUser, "u1"),
		msg(model.RoleAssistant, "a1"),
		msg(model.RoleUser, "u2"),
		{Role: model.RoleAssistant, Blocks: []model.Block{model.ToolCallBlock("x", "t", nil)}},
		model.ToolResultMessage("x", model.TextResult("result")),
		msg(model.RoleAssistant, "final"),
	}

	out, err := c.Compact(context.Background(), history)
	require.NoError(t, err)

	// No tool message may appear without its preceding assistant tool call.
	for i, m := range out {
		if m.Role == model.RoleTool {
			require.Greater(t, i, 0)
			require.NotEqual(t, model.RoleSystem, out[i-1].Role,
				"a tool result must not directly follow a summary (orphaned)")
		}
	}
}

// Short histories are returned unchanged.
func TestSummarizingCompactor_NoopWhenShort(t *testing.T) {
	sum := &fixedSummarizer{}
	c := conversation.NewSummarizingCompactor(sum, conversation.WithKeepLast(8))

	history := []model.Message{msg(model.RoleSystem, "sys"), msg(model.RoleUser, "hi"), msg(model.RoleAssistant, "hello")}
	out, err := c.Compact(context.Background(), history)
	require.NoError(t, err)
	require.Equal(t, history, out)
	require.Nil(t, sum.got, "summarizer should not be called for a short history")
}

// The LLM summarizer flattens the transcript and returns the model's summary.
func TestLLMSummarizer(t *testing.T) {
	prov := &recordingProvider{reply: "the summary"}
	s := conversation.NewLLMSummarizer(prov, "m")

	out, err := s.Summarize(context.Background(), []model.Message{
		msg(model.RoleUser, "what's the weather?"),
		msg(model.RoleAssistant, "18C"),
	})
	require.NoError(t, err)
	require.Equal(t, "the summary", out)

	// The transcript was rendered into the user message.
	require.Contains(t, prov.lastReq.Messages[1].Text(), "what's the weather?")
	require.Contains(t, prov.lastReq.Messages[1].Text(), "18C")
}

type recordingProvider struct {
	reply   string
	lastReq model.Request
}

func (p *recordingProvider) Name() string { return "rec" }
func (p *recordingProvider) Invoke(_ context.Context, req model.Request) (model.Response, error) {
	p.lastReq = req
	return model.Response{Message: model.AssistantMessage(p.reply)}, nil
}
