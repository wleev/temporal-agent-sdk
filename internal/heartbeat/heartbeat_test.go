package heartbeat

import (
	"context"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.temporal.io/sdk/activity"
	"go.temporal.io/sdk/converter"
	"go.temporal.io/sdk/testsuite"
)

func TestBeatInterval(t *testing.T) {
	assert.Equal(t, time.Duration(0), beatInterval(0), "no timeout disables the beater")
	assert.Equal(t, 10*time.Second, beatInterval(30*time.Second), "beats at a third of the timeout")
}

// startEvery records heartbeats on its interval until Stop, and each carries the
// detail from the details func.
func TestStartEvery_RecordsHeartbeatDetails(t *testing.T) {
	var s testsuite.WorkflowTestSuite
	env := s.NewTestActivityEnvironment()

	var beats int32
	var last atomic.Value
	env.SetOnActivityHeartbeatListener(func(_ *activity.Info, details converter.EncodedValues) {
		atomic.AddInt32(&beats, 1)
		var d string
		if details.HasValues() {
			require.NoError(t, details.Get(&d))
		}
		last.Store(d)
	})

	act := func(ctx context.Context) error {
		b := startEvery(ctx, 20*time.Millisecond, func() any { return "tick" })
		defer b.Stop()
		time.Sleep(300 * time.Millisecond)
		return nil
	}
	env.RegisterActivity(act)

	_, err := env.ExecuteActivity(act)
	require.NoError(t, err)

	assert.GreaterOrEqual(t, atomic.LoadInt32(&beats), int32(1), "at least one heartbeat was recorded")
	assert.Equal(t, "tick", last.Load(), "the heartbeat carried the detail")
}

// With no HeartbeatTimeout on the activity, Start is inert and Stop is safe.
func TestStart_NoTimeoutIsInert(t *testing.T) {
	var s testsuite.WorkflowTestSuite
	env := s.NewTestActivityEnvironment()

	var beats int32
	env.SetOnActivityHeartbeatListener(func(*activity.Info, converter.EncodedValues) {
		atomic.AddInt32(&beats, 1)
	})

	act := func(ctx context.Context) error {
		b := Start(ctx, func() any { return "tick" })
		defer b.Stop()
		time.Sleep(50 * time.Millisecond)
		return nil
	}
	env.RegisterActivity(act)

	_, err := env.ExecuteActivity(act)
	require.NoError(t, err)
	assert.Equal(t, int32(0), atomic.LoadInt32(&beats), "no timeout means no heartbeats")
}
