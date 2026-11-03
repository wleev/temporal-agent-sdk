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
