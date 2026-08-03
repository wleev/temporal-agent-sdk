package workflowstreamsink_test

import (
	"context"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.temporal.io/sdk/client"
	"go.temporal.io/sdk/contrib/workflowstreams"
	"go.temporal.io/sdk/converter"
	"go.temporal.io/sdk/testsuite"
	"go.temporal.io/sdk/worker"
	"go.temporal.io/sdk/workflow"

	"github.com/wleev/temporal-agent-sdk/agent"
	"github.com/wleev/temporal-agent-sdk/model"
	"github.com/wleev/temporal-agent-sdk/model/streamsink/workflowstreamsink"
)

var streamWords = []string{"Hello ", "there ", "world"}

// streamingProvider streams a fixed sequence of text deltas, so a subscriber can
// assert both content and order.
type streamingProvider struct{}

func (streamingProvider) Name() string { return "fake" }

func (streamingProvider) Invoke(context.Context, model.Request) (model.Response, error) {
	return model.Response{Message: model.AssistantMessage("Hello there world"), FinishReason: model.FinishStop}, nil
}

func (streamingProvider) InvokeStream(ctx context.Context, _ model.Request, sink model.StreamSink) (model.Response, error) {
	for _, w := range streamWords {
		if err := sink.OnDelta(ctx, model.StreamDelta{Text: w, ToolCallIndex: -1}); err != nil {
			return model.Response{}, err
		}
	}
	return model.Response{Message: model.AssistantMessage("Hello there world"), FinishReason: model.FinishStop}, nil
}

const doneSignal = "done"

// streamHostWorkflow hosts the stream so the model activity's published deltas
// have a handler, runs one agent turn, then waits for the test to release it —
// keeping the stream (and its workflow) alive while the subscriber reads.
//
//workflowcheck:ignore NewWorkflowStream's internal maps.Copy is an order-independent copy; the contrib is built for workflow use.
func streamHostWorkflow(ctx workflow.Context) (*agent.Result, error) {
	if _, err := workflowstreams.NewWorkflowStream(ctx, nil); err != nil {
		return nil, err
	}
	a, err := agent.NewAgent("assistant", "test-model", agent.WithStreaming())
	if err != nil {
		return nil, err
	}
	res, err := agent.Run(ctx, a, "hi")
	if err != nil {
		return nil, err
	}
	var done bool
	workflow.GetSignalChannel(ctx, doneSignal).Receive(ctx, &done)
	return res, nil
}

// New always yields a usable factory, including when the topic name defaults.
func TestNew_ReturnsFactory(t *testing.T) {
	for _, topic := range []string{"", "custom"} {
		require.NotNil(t, workflowstreamsink.New(topic, workflowstreams.Options{}),
			"New(%q) must return a non-nil SinkFactory", topic)
	}
}

// The sink publishes each streamed delta to the workflow's stream, in order, and
// a subscriber decodes them back to model.StreamDelta.
func TestSink_PublishesOrderedDeltas(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping workflowstreams sink e2e: needs a dev server binary (-short)")
	}

	acts, err := model.NewActivities(streamingProvider{})
	require.NoError(t, err)
	// A short flush keeps the e2e quick; Close drains the rest at call end.
	acts.SetStreamSink(workflowstreamsink.New("model",
		workflowstreams.Options{BatchInterval: 200 * time.Millisecond}))

	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Minute)
	defer cancel()
	srv, err := testsuite.StartDevServer(ctx, testsuite.DevServerOptions{LogLevel: "error"})
	if err != nil {
		t.Skipf("skipping: could not start dev server: %v", err)
	}
	t.Cleanup(func() { _ = srv.Stop() })
	c := srv.Client()

	const tq = "agentsdk-wsink-test"
	w := worker.New(c, tq, worker.Options{})
	acts.Register(w)
	w.RegisterWorkflowWithOptions(streamHostWorkflow, workflow.RegisterOptions{Name: "stream_host"})
	require.NoError(t, w.Start())
	t.Cleanup(w.Stop)

	run, err := c.ExecuteWorkflow(ctx, client.StartWorkflowOptions{TaskQueue: tq}, "stream_host")
	require.NoError(t, err)

	sub := workflowstreams.NewClient(c, run.GetID(), workflowstreams.Options{})
	defer func() { _ = sub.Close(ctx) }()

	var got []string
	dc := converter.GetDefaultDataConverter()
	for item, err := range sub.Topic("model").Subscribe(ctx, 0) {
		require.NoError(t, err)
		var d model.StreamDelta
		require.NoError(t, dc.FromPayload(item.Data, &d))
		got = append(got, d.Text)
		if len(got) == len(streamWords) {
			break
		}
	}
	assert.Equal(t, streamWords, got, "deltas arrive in publish order")

	// Release the host workflow so it can finish, and confirm the durable result.
	require.NoError(t, c.SignalWorkflow(ctx, run.GetID(), "", doneSignal, true))
	var res agent.Result
	require.NoError(t, run.Get(ctx, &res))
	assert.Equal(t, "Hello there world", res.Output)
}
