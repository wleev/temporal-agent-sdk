// Package workflowstreamsink is an opt-in [model.SinkFactory] that streams model
// deltas through the workflowstreams contrib, giving durable, ordered,
// exactly-once, cross-language delivery to subscribers.
//
// The default activity-side sink forwards deltas straight to a transport of your
// choice and never touches workflow history; this one publishes each delta to a
// durable log hosted inside the agent's workflow, so deltas become workflow
// events. That buys durability and cross-language subscribers at the cost of
// history growth — batch (the default is a 2s flush) and truncate for a chatty
// stream. Reach for it when a subscriber must not miss or reorder deltas, or
// when consumers are not in Go; keep the default sink for plain UI streaming.
//
// # Wiring
//
// Activity side — build the factory and set it on the model activity (or pass it
// as plugin.Config.StreamSink):
//
//	acts, _ := model.NewActivities(provider)
//	acts.SetStreamSink(workflowstreamsink.New("model", workflowstreams.Options{}))
//
// Workflow side — the agent's workflow must host the stream, so the deltas its
// model activity publishes have somewhere to land. Construct it once at the top
// and carry its state across continue-as-new:
//
//	func AgentWorkflow(ctx workflow.Context, in Input) (*agent.Result, error) {
//		if _, err := workflowstreams.NewWorkflowStream(ctx, in.StreamState); err != nil {
//			return nil, err
//		}
//		return agent.Run(ctx, myAgent, in.Question)
//	}
//
// Subscriber side — read the topic from any client:
//
//	c := workflowstreams.NewClient(temporalClient, workflowID, workflowstreams.Options{})
//	defer c.Close(ctx)
//	for item, err := range c.Topic("model").Subscribe(ctx, 0) {
//		// item.Data decodes to a model.StreamDelta
//	}
//
// workflowcheck reports NewWorkflowStream as non-deterministic because it copies
// a map internally; that copy is order-independent and the call is safe in
// workflow code. Suppress the report with a //workflowcheck:ignore directive on
// the hosting workflow.
package workflowstreamsink

import (
	"context"

	"go.temporal.io/sdk/contrib/workflowstreams"

	"github.com/wleev/temporal-agent-sdk/model"
)

// defaultTopic is the topic used when New is given an empty name.
const defaultTopic = "model"

// New returns a [model.SinkFactory] that publishes each streamed delta to the
// named workflowstreams topic on the model activity's own workflow. An empty
// topic defaults to "model". opts tunes batching and serialization; the zero
// value uses the contrib defaults (a 2s flush interval).
//
// The workflow that runs the model activity must host the stream with
// [workflowstreams.NewWorkflowStream]; without it, published deltas have no
// handler and are dropped.
func New(topic string, opts workflowstreams.Options) model.SinkFactory {
	if topic == "" {
		topic = defaultTopic
	}
	return func(ctx context.Context) (model.StreamSink, error) {
		c, err := workflowstreams.NewClientFromActivity(ctx, opts)
		if err != nil {
			return nil, err
		}
		return &sink{client: c, topic: c.Topic(topic)}, nil
	}
}

// sink publishes deltas to one topic for the duration of a single streamed call.
type sink struct {
	client *workflowstreams.Client
	topic  *workflowstreams.TopicHandle
}

// OnDelta buffers the delta for publishing. It goes out on the next flush, so the
// call itself never blocks on the network; delivery errors surface at Close.
func (s *sink) OnDelta(_ context.Context, d model.StreamDelta) error {
	s.topic.Publish(d, false)
	return nil
}

// Close flushes any buffered deltas and stops the publisher. The activity calls
// it once the streamed call ends, via [model.StreamSinkCloser].
func (s *sink) Close(ctx context.Context) error {
	return s.client.Close(ctx)
}
