package agent_test

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.temporal.io/sdk/workflow"

	"github.com/wleev/temporal-agent-sdk/agent"
	"github.com/wleev/temporal-agent-sdk/agenttest"
	"github.com/wleev/temporal-agent-sdk/guardrail"
	"github.com/wleev/temporal-agent-sdk/model"
	"github.com/wleev/temporal-agent-sdk/tool"
)

// TestReplay_SteeringControls records a dev-server run that uses tool choice,
// parallel-call, metadata, per-call schema, call-only, strict-tool, guardrail,
// OnRound, FinalAction and final-result features, and replays its history.
func TestReplay_SteeringControls(t *testing.T) {
	env := startReplayEnv(t)

	const name = "replay_steering_controls"
	const saveRecord = "agentsdk_test_save_guardrail_record"

	maskResult := mustFunc(t, "mask-result", []guardrail.Stage{guardrail.StageToolResult},
		func(_ workflow.Context, in guardrail.CheckInput) (guardrail.Verdict, error) {
			text := strings.ReplaceAll(model.ResultText(in.Result), "secret", "[masked]")
			return guardrail.RewriteResult(model.TextResult(text), "mask", nil), nil
		})
	fastDef := model.NewTool("fast", "A fast tool",
		[]byte(`{"type":"object","properties":{"text":{"type":"string"}},"required":["text"]}`))
	fast, err := tool.Dynamic(fastDef, func(ctx workflow.Context, _ string, args json.RawMessage) (*model.CallToolResult, error) {
		var in echoIn
		if err := json.Unmarshal(args, &in); err != nil {
			return nil, err
		}
		info, _ := tool.CallInfoFrom(ctx)
		call := model.ToolCall{ID: info.ID, Name: "fast", Arguments: args}
		out, err := guardrail.Screen(ctx, []guardrail.Guardrail{maskResult}, guardrail.CheckInput{
			Stage: guardrail.StageToolResult, Call: &call, Result: model.TextResult("fast:" + in.Text + " secret"),
			Turn: info.Turn, Index: info.Index,
		})
		if err != nil {
			return nil, err
		}
		return out.Result, nil
	})
	require.NoError(t, err)

	submitDef := model.StrictTool(model.NewTool("submit", "Submit the answer",
		[]byte(`{"type":"object","properties":{"answer":{"type":"string"}},"required":["answer"],"additionalProperties":false}`)))
	submit, err := tool.Dynamic(submitDef, func(_ workflow.Context, _ string, args json.RawMessage) (*model.CallToolResult, error) {
		var in struct {
			Answer string `json:"answer"`
		}
		if err := json.Unmarshal(args, &in); err != nil {
			return nil, err
		}
		return model.FinalResult(model.TextResult("accepted"), in.Answer), nil
	}, tool.EndsRun())
	require.NoError(t, err)

	inputNote := mustFunc(t, "input-note", []guardrail.Stage{guardrail.StageInput},
		func(workflow.Context, guardrail.CheckInput) (guardrail.Verdict, error) {
			return guardrail.Annotate("r", guardrail.Notice{Text: "Input checked.", Audience: guardrail.AudienceModel}), nil
		})
	textNote := mustFunc(t, "text-note", []guardrail.Stage{guardrail.StageAssistantText},
		func(workflow.Context, guardrail.CheckInput) (guardrail.Verdict, error) {
			return guardrail.Annotate("r", guardrail.Notice{Text: "Note A", Audience: guardrail.AudienceModel}), nil
		})
	outputNote := mustFunc(t, "output-note", []guardrail.Stage{guardrail.StageOutput},
		func(workflow.Context, guardrail.CheckInput) (guardrail.Verdict, error) {
			return guardrail.Annotate("r", guardrail.Notice{Text: "Checked.", Audience: guardrail.AudienceUser}), nil
		})
	rewriteFast := mustFunc(t, "rewrite-fast", []guardrail.Stage{guardrail.StageToolCall},
		func(_ workflow.Context, in guardrail.CheckInput) (guardrail.Verdict, error) {
			if in.Turn == 1 && in.Call.Name == "fast" {
				return guardrail.RewriteArguments(json.RawMessage(`{"text":"1-masked"}`), "mask", nil), nil
			}
			return guardrail.Allow(), nil
		})

	a, err := agent.NewAgent("assistant", "test-model",
		agent.WithTools(fast, submit), agent.WithMaxTurns(4),
		agent.WithGuardrails(inputNote, textNote, outputNote))
	require.NoError(t, err)

	schema := &model.OutputSchema{Name: "draft", Schema: json.RawMessage(`{"type":"object"}`), Strict: true}
	var output string
	var records []string
	var rounds [][]string
	wf := func(ctx workflow.Context) (*agent.Result, error) {
		ctx = workflow.WithActivityOptions(ctx, workflow.ActivityOptions{StartToCloseTimeout: 10 * time.Second})
		res, err := agent.RunWith(ctx, a, "go", agent.RunOptions{
			Guardrails:       []guardrail.Guardrail{rewriteFast},
			ProviderMetadata: map[string]string{"tenant": "replay"},
			PrepareModelCall: func(_ workflow.Context, p agent.ModelCallPlan) (agent.ModelCallPlan, error) {
				switch p.Turn {
				case 1:
					p.Advertised = []string{"fast"}
					p.Settings.ToolChoice = &model.ToolChoice{Mode: model.ToolChoiceRequired}
					p.Settings.ParallelToolCalls = model.Ptr(false)
					p.Metadata = map[string]string{"route": "turn1"}
				case 2:
					p.OutputSchema = schema
					p.CallOnly = []model.Message{model.UserMessage("answer as JSON")}
				case 3:
					p.Settings.ToolChoice = &model.ToolChoice{Mode: model.ToolChoiceTool, Name: "submit"}
				}
				return p, nil
			},
			OnGuardrail: func(ctx workflow.Context, e agent.GuardrailEvent) error {
				return workflow.ExecuteActivity(ctx, saveRecord, e).Get(ctx, nil)
			},
			OnRound: func(ctx workflow.Context, e agent.RoundEvent) (agent.RoundOutcome, error) {
				if !workflow.IsReplaying(ctx) {
					rounds = append(rounds, resultTexts(e.Results))
				}
				if e.Turn != 1 {
					return agent.RoundOutcome{}, nil
				}
				return agent.RoundOutcome{Results: []*model.CallToolResult{model.TextResult("observed")}}, nil
			},
			OnFinal: func(_ workflow.Context, e agent.FinalEvent) (agent.FinalAction, error) {
				if e.Turn == 2 {
					return agent.FinalAction{Continue: true, DropReply: true}, nil
				}
				return agent.FinalAction{}, nil
			},
		})
		if res != nil {
			output = res.Output
		}
		return res, err
	}

	fake := agenttest.NewFakeProvider(
		agenttest.Response{
			Message: model.Message{Role: model.RoleAssistant, Blocks: []model.Block{
				model.TextBlock("Checking."),
				model.ToolCallBlock("a", "fast", json.RawMessage(`{"text":"1"}`)),
			}},
			Usage: model.Usage{PromptTokens: 10, CompletionTokens: 5, TotalTokens: 15},
		},
		agenttest.Fails(&model.APIError{StatusCode: 503}),
		agenttest.Says(`{"draft":true}`),
		agenttest.CallsTool("submit", `{"answer":"Ghent is mild."}`),
	)

	hist := env.runWith(t, fake, nil, wf, name, map[string]any{
		saveRecord: func(_ context.Context, e agent.GuardrailEvent) error {
			records = append(records, fmt.Sprintf("%s:%s:%s:%d", e.Record.Stage, e.Record.Guardrail, e.Record.Action, e.Attempt))
			return nil
		},
	})

	assert.Equal(t, "Ghent is mild.", output)
	assert.Equal(t, []string{
		"input:input-note:annotate:0",
		"assistant_text:text-note:annotate:1",
		"tool_call:rewrite-fast:rewrite:1",
		"output:output-note:annotate:2",
		"output:output-note:annotate:0",
	}, records)
	assert.Equal(t, [][]string{{"fast:1-masked [masked]"}, {"accepted"}}, rounds)

	calls := fake.Calls()
	if assert.Len(t, calls, 4, "turn 1, the failed and the retried turn 2, turn 3") {
		turn1 := calls[0]
		if assert.Len(t, turn1.Tools, 1) {
			assert.Equal(t, "fast", turn1.Tools[0].Name)
		}
		if assert.NotNil(t, turn1.Settings.ToolChoice) {
			assert.Equal(t, model.ToolChoiceRequired, turn1.Settings.ToolChoice.Mode)
		}
		if assert.NotNil(t, turn1.Settings.ParallelToolCalls) {
			assert.False(t, *turn1.Settings.ParallelToolCalls)
		}
		assert.Equal(t, map[string]string{"tenant": "replay", "route": "turn1"}, turn1.Metadata)
		assert.Equal(t, []string{"go", "Input checked."}, messageTexts(turn1.Messages))

		failed, turn2 := calls[1], calls[2]
		assert.Equal(t, schema, failed.OutputSchema)
		assert.Equal(t, messageTexts(turn2.Messages), messageTexts(failed.Messages), "the retry sends the same request")
		assert.Equal(t, schema, turn2.OutputSchema)
		if assert.NotEmpty(t, turn2.Messages) {
			assert.Equal(t, "answer as JSON", turn2.Messages[len(turn2.Messages)-1].Text())
		}
		assert.Equal(t, []string{"observed"}, toolResultTexts(turn2.Messages))

		turn3 := calls[3]
		if assert.NotNil(t, turn3.Settings.ToolChoice) {
			assert.Equal(t, model.ToolChoiceTool, turn3.Settings.ToolChoice.Mode)
			assert.Equal(t, "submit", turn3.Settings.ToolChoice.Name)
		}
		assert.NotContains(t, messageTexts(turn3.Messages), `{"draft":true}`, "the dropped reply is not sent")
		var advertised []string
		var sentSubmit *model.Tool
		for _, tl := range turn3.Tools {
			advertised = append(advertised, tl.Name)
			if tl.Name == "submit" {
				sentSubmit = tl
			}
		}
		assert.Equal(t, []string{"fast", "submit"}, advertised)
		if assert.NotNil(t, sentSubmit) {
			assert.True(t, model.IsStrictTool(sentSubmit))
		}
	}

	assertReplays(t, hist, nil, wf, name)
}
