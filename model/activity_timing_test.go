package model_test

import (
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.temporal.io/sdk/testsuite"

	"github.com/wleev/temporal-agent-sdk/model"
)

func TestInvokeModel_Timing(t *testing.T) {
	for _, tc := range []struct {
		name             string
		stream           bool
		wantFirstContent bool
	}{
		{name: "unstreamed", stream: false, wantFirstContent: false},
		{name: "streamed", stream: true, wantFirstContent: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var s testsuite.WorkflowTestSuite
			env := s.NewTestActivityEnvironment()
			prov := &streamingStub{stubProvider: stubProvider{
				name: "s",
				resp: model.Response{Message: model.AssistantMessage("hi")},
			}}
			acts, err := model.NewActivities([]model.Provider{prov})
			require.NoError(t, err)
			acts.Register(env)

			val, err := env.ExecuteActivity(model.InvokeModelActivity, model.Request{Model: "m", Stream: tc.stream})
			require.NoError(t, err)

			var resp model.Response
			require.NoError(t, val.Get(&resp))
			require.NotNil(t, resp.Timing)
			assert.GreaterOrEqual(t, resp.Timing.Duration, time.Duration(0))
			if tc.wantFirstContent {
				require.NotNil(t, resp.Timing.FirstContent)
				assert.Positive(t, *resp.Timing.FirstContent)
				assert.LessOrEqual(t, *resp.Timing.FirstContent, resp.Timing.Duration)
			} else {
				assert.Nil(t, resp.Timing.FirstContent)
			}
		})
	}
}
