package model

import "strconv"

// Valid reports whether m is [ToolChoiceAuto], [ToolChoiceNone],
// [ToolChoiceRequired] or [ToolChoiceTool].
func (m ToolChoiceMode) Valid() bool {
	switch m {
	case ToolChoiceAuto, ToolChoiceNone, ToolChoiceRequired, ToolChoiceTool:
		return true
	default:
		return false
	}
}

// ResolveToolChoice validates c against tools and returns the tool choice to
// send; provider names the provider in a returned [UnsupportedSettingError].
// It returns nil for a nil c, and for an auto or none c
// when tools holds no non-nil tool. It returns an [UnsupportedSettingError] for
// an invalid mode, for a required or tool mode when tools holds no non-nil
// tool, and for a tool mode whose Name is empty or names no tool in tools.
// Otherwise it returns c.
func ResolveToolChoice(provider string, c *ToolChoice, tools []*Tool) (*ToolChoice, error) {
	if c == nil {
		return nil, nil
	}
	if !c.Mode.Valid() {
		return nil, unsupportedToolChoice(provider, "unknown mode "+strconv.Quote(string(c.Mode)))
	}
	if !hasAnyTool(tools) {
		if c.Mode == ToolChoiceAuto || c.Mode == ToolChoiceNone {
			return nil, nil
		}
		return nil, unsupportedToolChoice(provider, "mode "+strconv.Quote(string(c.Mode))+" needs tools and the request has none")
	}
	if c.Mode == ToolChoiceTool {
		if c.Name == "" {
			return nil, unsupportedToolChoice(provider, "mode tool needs a tool name")
		}
		if !hasToolNamed(tools, c.Name) {
			return nil, unsupportedToolChoice(provider, "tool "+strconv.Quote(c.Name)+" is not advertised")
		}
	}
	return c, nil
}

// hasAnyTool reports whether tools holds a non-nil tool.
func hasAnyTool(tools []*Tool) bool {
	for _, t := range tools {
		if t != nil {
			return true
		}
	}
	return false
}

// hasToolNamed reports whether tools holds a tool named name.
func hasToolNamed(tools []*Tool, name string) bool {
	for _, t := range tools {
		if t != nil && t.Name == name {
			return true
		}
	}
	return false
}

// unsupportedToolChoice returns an [UnsupportedSettingError] for provider's
// tool_choice setting, with Reason set to reason.
func unsupportedToolChoice(provider, reason string) error {
	return &UnsupportedSettingError{Provider: provider, Setting: "tool_choice", Reason: reason}
}
