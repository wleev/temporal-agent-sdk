package agent_test

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.temporal.io/sdk/activity"
	"go.temporal.io/sdk/testsuite"
	"go.temporal.io/sdk/workflow"

	"github.com/wleev/temporal-agent-sdk/agent"
	"github.com/wleev/temporal-agent-sdk/model"
	anthropicprovider "github.com/wleev/temporal-agent-sdk/model/anthropic"
)

// This is the seam's real exam: the whole agent loop — turn sequencing, tool
// dispatch, history accumulation — driven through the REAL Anthropic provider
// (not the fake), over a stub Anthropic endpoint. If the neutral seam had
// baked in OpenAI's message shape, the second turn's request would be malformed
// Anthropic and the round trip would fail here.

// anthropicStub serves /v1/messages and returns a scripted response per call.
type anthropicStub struct {
	*httptest.Server
	mu        sync.Mutex
	responses []string
	bodies    []map[string]any
}

func newAnthropicStub(t *testing.T, responses ...string) *anthropicStub {
	t.Helper()
	s := &anthropicStub{responses: responses}
	s.Server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		var body map[string]any
		_ = json.Unmarshal(raw, &body)

		s.mu.Lock()
		i := len(s.bodies)
		s.bodies = append(s.bodies, body)
		var resp string
		if i < len(s.responses) {
			resp = s.responses[i]
		}
		s.mu.Unlock()

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(200)
		_, _ = io.WriteString(w, resp)
	}))
	t.Cleanup(s.Close)
	return s
}

func (s *anthropicStub) request(i int) map[string]any {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.bodies[i]
}

func anthropicToolUse(id, name, inputJSON string) string {
	return `{"id":"msg","type":"message","role":"assistant","model":"claude-test",
	  "content":[{"type":"tool_use","id":"` + id + `","name":"` + name + `","input":` + inputJSON + `}],
	  "stop_reason":"tool_use","usage":{"input_tokens":10,"output_tokens":5}}`
}

func anthropicText(text string) string {
	return `{"id":"msg","type":"message","role":"assistant","model":"claude-test",
	  "content":[{"type":"text","text":"` + text + `"}],
	  "stop_reason":"end_turn","usage":{"input_tokens":12,"output_tokens":6}}`
}

// newAnthropicEnv registers the model activity backed by the real Anthropic
// provider pointed at the stub.
func newAnthropicEnv(t *testing.T, stub *anthropicStub) *testsuite.TestWorkflowEnvironment {
	t.Helper()
	provider, err := anthropicprovider.New(
		anthropicprovider.WithBaseURL(stub.URL),
		anthropicprovider.WithAPIKey("test-key"),
	)
	require.NoError(t, err)

	acts, err := model.NewActivities(provider)
	require.NoError(t, err)

	var s testsuite.WorkflowTestSuite
	env := s.NewTestWorkflowEnvironment()
	env.RegisterActivityWithOptions(acts.InvokeModel, activity.RegisterOptions{Name: model.InvokeModelActivity})
	return env
}

// A full tool round trip through the Anthropic provider. The second model call
// must carry the tool result back in Anthropic's format (a user message with a
// tool_result block), which only works if the neutral history survives the
// round trip through the seam.
func TestAnthropicProvider_ToolRoundTripThroughLoop(t *testing.T) {
	stub := newAnthropicStub(t,
		anthropicToolUse("toolu_1", "get_weather", `{"city":"Ghent"}`),
		anthropicText("It is 18C in Ghent."),
	)
	env := newAnthropicEnv(t, stub)

	a, err := agent.NewAgent("assistant", "claude-test",
		agent.WithInstructions("Be brief."), agent.WithTools(weatherTool(t)))
	require.NoError(t, err)
	env.ExecuteWorkflow(func(ctx workflow.Context) (*agent.Result, error) {
		return agent.Run(ctx, a, "Weather in Ghent?")
	})

	require.True(t, env.IsWorkflowCompleted())
	require.NoError(t, env.GetWorkflowError())

	var res agent.Result
	require.NoError(t, env.GetWorkflowResult(&res))
	assert.Equal(t, "It is 18C in Ghent.", res.Output)
	assert.Equal(t, 2, res.Turns)

	// Turn 1: system hoisted, one user message.
	first := stub.request(0)
	sys := first["system"].([]any)
	assert.Equal(t, "Be brief.", sys[0].(map[string]any)["text"])

	// Turn 2: the request must contain the assistant tool_use turn AND a user
	// message carrying the tool_result — the neutral tool result, rendered as
	// Anthropic expects.
	second := stub.request(1)
	msgs := second["messages"].([]any)
	if assert.Len(t, msgs, 3, "user, assistant(tool_use), user(tool_result)") {
		toolResultMsg := msgs[2].(map[string]any)
		assert.Equal(t, "user", toolResultMsg["role"])
		block := toolResultMsg["content"].([]any)[0].(map[string]any)
		assert.Equal(t, "tool_result", block["type"])
		assert.Equal(t, "toolu_1", block["tool_use_id"])
		// The tool's actual output made it back into the Anthropic tool_result.
		assert.Contains(t, mustJSON(t, block), "18°C in Ghent")
	}
}

// Parallel tool calls exercise the coalescing path through the live loop: two
// tool_use blocks in one assistant turn must come back as two tool_result blocks
// in one user message.
func TestAnthropicProvider_ParallelToolResultsThroughLoop(t *testing.T) {
	stub := newAnthropicStub(t,
		`{"id":"msg","type":"message","role":"assistant","model":"claude-test",
		  "content":[
		    {"type":"tool_use","id":"a","name":"get_weather","input":{"city":"Ghent"}},
		    {"type":"tool_use","id":"b","name":"get_weather","input":{"city":"Bruges"}}
		  ],
		  "stop_reason":"tool_use","usage":{"input_tokens":10,"output_tokens":5}}`,
		anthropicText("Both are mild."),
	)
	env := newAnthropicEnv(t, stub)

	a, err := agent.NewAgent("assistant", "claude-test", agent.WithTools(weatherTool(t)))
	require.NoError(t, err)
	env.ExecuteWorkflow(func(ctx workflow.Context) (*agent.Result, error) {
		return agent.Run(ctx, a, "weather?")
	})

	require.True(t, env.IsWorkflowCompleted())
	require.NoError(t, env.GetWorkflowError())

	var res agent.Result
	require.NoError(t, env.GetWorkflowResult(&res))
	assert.Equal(t, "Both are mild.", res.Output)

	second := stub.request(1)
	msgs := second["messages"].([]any)
	// The two tool results must be ONE user message with two tool_result blocks.
	last := msgs[len(msgs)-1].(map[string]any)
	assert.Equal(t, "user", last["role"])
	blocks := last["content"].([]any)
	if assert.Len(t, blocks, 2, "two parallel tool results coalesce into one user message") {
		assert.Equal(t, "a", blocks[0].(map[string]any)["tool_use_id"])
		assert.Equal(t, "b", blocks[1].(map[string]any)["tool_use_id"])
	}
}

func mustJSON(t *testing.T, v any) string {
	t.Helper()
	b, err := json.Marshal(v)
	require.NoError(t, err)
	return string(b)
}
