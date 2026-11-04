package tool_test

import (
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/wleev/temporal-agent-sdk/model"
	"github.com/wleev/temporal-agent-sdk/tool"
)

func TestEndsRun_SetsCanEndRun(t *testing.T) {
	assert.True(t, tool.Resolve(tool.EndsRun()).CanEndRun)
	assert.False(t, tool.Resolve().CanEndRun)

	destructive := true
	p := tool.ApplyAnnotations(tool.Resolve(tool.EndsRun()), &model.ToolAnnotations{DestructiveHint: &destructive})
	assert.True(t, p.CanEndRun, "annotations keep CanEndRun")
	assert.True(t, p.NeedsApproval)
}
