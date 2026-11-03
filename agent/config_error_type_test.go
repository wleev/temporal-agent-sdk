package agent

import (
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/wleev/temporal-agent-sdk/model"
)

func TestIsConfigErrorType_UnsupportedSetting(t *testing.T) {
	assert.True(t, isConfigErrorType(model.ErrorTypeUnsupportedSetting))
	assert.True(t, isConfigErrorType(model.ErrorTypeNoProvider))
	assert.False(t, isConfigErrorType("SomethingElse"))
}
