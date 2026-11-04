package agent

import (
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/wleev/temporal-agent-sdk/model"
)

func TestIsFatalErrorType_UnsupportedSetting(t *testing.T) {
	assert.True(t, isFatalErrorType(model.ErrorTypeUnsupportedSetting))
	assert.True(t, isFatalErrorType(model.ErrorTypeNoProvider))
	assert.True(t, isFatalErrorType(ErrorTypeTripwire))
	assert.False(t, isFatalErrorType("SomethingElse"))
}
