package model

import "maps"

// StrictToolMetaKey is the [Tool.Meta] key whose true value asks the provider
// to make the tool's arguments adhere strictly to its input schema. OpenAI
// sends it as strict: true on the function; Anthropic and Vertex ignore it.
const StrictToolMetaKey = "temporal-agent-sdk/strict"

// StrictTool returns a shallow copy of t whose Meta carries [StrictToolMetaKey]
// set to true, in addition to the entries of t.Meta. It leaves t unchanged and
// returns nil for a nil t.
func StrictTool(t *Tool) *Tool {
	if t == nil {
		return nil
	}
	c := *t
	c.Meta = maps.Clone(t.Meta)
	if c.Meta == nil {
		c.Meta = make(map[string]any, 1)
	}
	c.Meta[StrictToolMetaKey] = true
	return &c
}

// IsStrictTool reports whether t is marked strict by [StrictTool].
func IsStrictTool(t *Tool) bool {
	if t == nil {
		return false
	}
	v, _ := t.Meta[StrictToolMetaKey].(bool)
	return v
}

// FinalOutputMetaKey is the [CallToolResult.Meta] key whose string value is a
// final output. The agent loop ends a run with the first final output in call
// order among the round's results whose tool is built with tool.EndsRun, after
// the output-stage guardrails, which may rewrite or block it, once every call
// of the round has finished; RunOptions.OnReply and RunOptions.OnFinal do not
// run for it. A run started by agent.RunTyped fails with a non-retryable
// configuration error instead. The loop removes the key from every result, and
// the MCP tool adapters remove it from the results of MCP servers.
const FinalOutputMetaKey = "temporal-agent-sdk/final_output"

// FinalResult returns a copy of result whose Meta holds output under
// [FinalOutputMetaKey], in addition to the entries of result.Meta. It leaves
// result unchanged. A nil result gives a copy of TextResult(""). Returned by a
// tool built with tool.EndsRun, it ends a run as [FinalOutputMetaKey]
// describes.
func FinalResult(result *CallToolResult, output string) *CallToolResult {
	if result == nil {
		result = TextResult("")
	}
	c := *result
	c.Meta = maps.Clone(result.Meta)
	if c.Meta == nil {
		c.Meta = make(map[string]any, 1)
	}
	c.Meta[FinalOutputMetaKey] = output
	return &c
}

// FinalOutput returns the final output result carries under
// [FinalOutputMetaKey] and true, or "" and false when it carries none.
func FinalOutput(result *CallToolResult) (string, bool) {
	if result == nil {
		return "", false
	}
	out, ok := result.Meta[FinalOutputMetaKey].(string)
	return out, ok
}

// WithoutFinalOutput returns r when its Meta holds no [FinalOutputMetaKey], and
// otherwise a copy of r whose Meta lacks the key, or is nil when the key was
// its only entry. A nil r gives nil.
func WithoutFinalOutput(r *CallToolResult) *CallToolResult {
	if r == nil {
		return nil
	}
	if _, ok := r.Meta[FinalOutputMetaKey]; !ok {
		return r
	}
	c := *r
	c.Meta = maps.Clone(r.Meta)
	delete(c.Meta, FinalOutputMetaKey)
	if len(c.Meta) == 0 {
		c.Meta = nil
	}
	return &c
}
