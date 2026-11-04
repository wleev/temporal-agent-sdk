package agent_test

import (
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.temporal.io/sdk/workflow"

	"github.com/wleev/temporal-agent-sdk/agent"
	"github.com/wleev/temporal-agent-sdk/agenttest"
	"github.com/wleev/temporal-agent-sdk/guardrail"
	"github.com/wleev/temporal-agent-sdk/model"
)

// customGuardrail is a Guardrail implementation that allows everything, with a
// name and stages of its own.
type customGuardrail struct {
	name   string
	stages []guardrail.Stage
}

func (g customGuardrail) Name() string { return g.name }

func (g customGuardrail) Stages() []guardrail.Stage { return g.stages }

func (customGuardrail) Check(workflow.Context, guardrail.CheckInput) (guardrail.Verdict, error) {
	return guardrail.Allow(), nil
}

func TestRunWith_RefusesAnUnnamedOrUndeclaredRunGuardrail(t *testing.T) {
	for name, tc := range map[string]struct {
		g    guardrail.Guardrail
		want string
	}{
		"empty name":    {customGuardrail{stages: []guardrail.Stage{guardrail.StageInput}}, "run guardrail 0 has no name"},
		"no stages":     {customGuardrail{name: "custom"}, `run guardrail "custom" declares no stages`},
		"unknown stage": {customGuardrail{name: "custom", stages: []guardrail.Stage{"reasoning"}}, `run guardrail "custom" declares unknown stage "reasoning"`},
	} {
		t.Run(name, func(t *testing.T) {
			fake := agenttest.NewFakeProvider(agenttest.Says("unused"))
			a, err := agent.NewAgent("assistant", "test-model")
			require.NoError(t, err)
			_, err = runPrepared(t, fake, a, "Weather in Ghent?", agent.RunOptions{Guardrails: []guardrail.Guardrail{tc.g}})
			assertConfigError(t, err, fake, tc.want)
		})
	}
}

func TestWithGuardrails_RegistersOnEveryDeclaredStage(t *testing.T) {
	mask := mustFunc(t, "mask", []guardrail.Stage{guardrail.StageAssistantText, guardrail.StageOutput},
		func(_ workflow.Context, in guardrail.CheckInput) (guardrail.Verdict, error) {
			if !strings.Contains(in.Text, "secret") {
				return guardrail.Allow(), nil
			}
			return guardrail.RewriteText(strings.ReplaceAll(in.Text, "secret", "[x]"), "mask", nil), nil
		})
	fake := agenttest.NewFakeProvider(textAndCalls("a secret plan", "Ghent"), agenttest.Says("the secret answer"))
	a, err := agent.NewAgent("assistant", "test-model", agent.WithTools(weatherTool(t)), agent.WithGuardrails(mask))
	require.NoError(t, err)

	res, err := runPrepared(t, fake, a, "Weather in Ghent?", agent.RunOptions{})
	require.NoError(t, err)

	assert.Equal(t, "the [x] answer", res.Output)
	var texts []string
	for _, m := range filterRole(res.Messages, model.RoleAssistant) {
		texts = append(texts, m.Text())
	}
	assert.Equal(t, []string{"a [x] plan", "the [x] answer"}, texts)
}

func TestNewAgent_RefusesGuardrails(t *testing.T) {
	output := mustFunc(t, "no-leaks", []guardrail.Stage{guardrail.StageOutput}, allow)
	for name, tc := range map[string]struct {
		opts []agent.Option
		want string
	}{
		"nil": {[]agent.Option{agent.WithGuardrails(nil)}, "guardrail 0 is nil"},
		"tool_result": {[]agent.Option{agent.WithGuardrails(customGuardrail{name: "mask",
			stages: []guardrail.Stage{guardrail.StageOutput, guardrail.StageToolResult}})}, `guardrail "mask": the agent loop does not run the tool_result stage`},
		"output with continuation": {[]agent.Option{agent.WithContinueOnLength(1), agent.WithGuardrails(output)},
			`guardrail "no-leaks": output guardrails cannot be combined with WithContinueOnLength`},
		"empty name": {[]agent.Option{agent.WithGuardrails(customGuardrail{stages: []guardrail.Stage{guardrail.StageInput}})},
			"guardrail 0 has no name"},
		"no stages": {[]agent.Option{agent.WithGuardrails(customGuardrail{name: "custom"})}, `guardrail "custom" declares no stages`},
		"unknown stage": {[]agent.Option{agent.WithGuardrails(customGuardrail{name: "custom", stages: []guardrail.Stage{"reasoning"}})},
			`guardrail "custom" declares unknown stage "reasoning"`},
	} {
		t.Run(name, func(t *testing.T) {
			_, err := agent.NewAgent("assistant", "test-model", tc.opts...)
			requireConfigError(t, err, tc.want)
		})
	}

	text := mustFunc(t, "a", []guardrail.Stage{guardrail.StageAssistantText}, allow)
	_, err := agent.NewAgent("assistant", "test-model", agent.WithContinueOnLength(1), agent.WithGuardrails(text))
	assert.NoError(t, err)
}
