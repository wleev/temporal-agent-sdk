package guardrail_test

import (
	"testing"

	mcpsdk "github.com/modelcontextprotocol/go-sdk/mcp"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.temporal.io/sdk/workflow"

	"github.com/wleev/temporal-agent-sdk/guardrail"
	"github.com/wleev/temporal-agent-sdk/model"
)

func TestScreen_EachCheckGetsItsOwnCallAndResult(t *testing.T) {
	results := stagesOf(guardrail.StageToolResult)
	mutate, err := guardrail.Func("mutate", results, func(_ workflow.Context, in guardrail.CheckInput) (guardrail.Verdict, error) {
		in.Call.Name = "delete_account"
		in.Call.Arguments[9] = 'X'
		in.Result.Content[0] = &mcpsdk.TextContent{Text: "changed"}
		in.Result.Content = append(in.Result.Content, &mcpsdk.TextContent{Text: "extra"})
		in.Result.Meta["trace"] = "changed"
		in.Result.IsError = true
		return guardrail.Allow(), nil
	})
	require.NoError(t, err)
	var seenCall model.ToolCall
	var seenResult model.CallToolResult
	look, err := guardrail.Func("look", results, func(_ workflow.Context, in guardrail.CheckInput) (guardrail.Verdict, error) {
		seenCall, seenResult = *in.Call, *in.Result
		return guardrail.Allow(), nil
	})
	require.NoError(t, err)

	call := weatherCall()
	result := model.TextResult("18°C in Ghent")
	result.Meta = mcpsdk.Meta{"trace": "t-1"}
	out, err := screen(t, []guardrail.Guardrail{mutate, look},
		guardrail.CheckInput{Stage: guardrail.StageToolResult, Call: call, Result: result})
	require.NoError(t, err)

	want := weatherCall()
	assert.Equal(t, *want, seenCall, "a sibling check sees the call unchanged")
	assert.Equal(t, *want, *call, "the host's call is unchanged")
	for name, r := range map[string]*model.CallToolResult{"sibling": &seenResult, "host": result, "outcome": out.Result} {
		if assert.Len(t, r.Content, 1, name) {
			assert.Equal(t, "18°C in Ghent", model.ResultText(r), name)
		}
		assert.False(t, r.IsError, name)
		assert.Equal(t, "t-1", r.Meta["trace"], name)
	}
}
