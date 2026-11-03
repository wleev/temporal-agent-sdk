package model

import "fmt"

// ErrorTypeUnsupportedSetting is the Temporal application error type of a
// non-retryable failure caused by a [Settings] value a provider cannot honor.
const ErrorTypeUnsupportedSetting = "AgentSDKUnsupportedSetting"

// UnsupportedSettingError reports a [Settings] value that a provider cannot
// honor. [ResolveToolChoice] returns it for a [Settings.ToolChoice] that
// cannot be sent.
type UnsupportedSettingError struct {
	// Provider is the provider backend that rejected the setting, such as
	// "openai".
	Provider string
	// Setting is the name of the rejected setting, such as "tool_choice".
	Setting string
	// Reason describes why the provider cannot honor the setting.
	Reason string
}

// Error implements error.
func (e *UnsupportedSettingError) Error() string {
	return fmt.Sprintf("model: provider %q cannot honor %s: %s", e.Provider, e.Setting, e.Reason)
}
