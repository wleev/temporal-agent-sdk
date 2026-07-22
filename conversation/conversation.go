// Package conversation runs a durable multi-turn agent conversation as a single
// long-lived workflow — one workflow per session, its workflow ID the session
// ID.
//
// The workflow's accumulated messages are the conversation memory; Temporal
// persists them across worker restarts and between turns. A user turn arrives as
// an Update ([SendMessageUpdate]) and returns the reply synchronously; the
// transcript is readable via a Query ([HistoryQuery]).
//
// To stay within Temporal's 51,200-event / 50 MB history limit, the workflow
// continues-as-new when the server suggests it, carrying its history forward
// through an optional [Compactor] that summarizes older turns to keep the
// carried state small.
//
// Wiring:
//
//	reg := agent.NewRegistry().MustAdd(myAgent)
//	conv := conversation.New(reg)                       // optionally conversation.WithCompactor(...)
//	conv.Register(w)                                    // registers the workflow (and compaction activity)
//	agent.RegisterWorkflows(w, reg)                     // for sub-agents, if any
//
// Start a session and take a turn:
//
//	c.ExecuteWorkflow(ctx, client.StartWorkflowOptions{ID: sessionID, TaskQueue: tq},
//	    conversation.WorkflowName, conversation.Input{Agent: "my-agent"})
//	h, _ := c.UpdateWorkflow(ctx, client.UpdateWorkflowOptions{
//	    WorkflowID: sessionID, UpdateName: conversation.SendMessageUpdate,
//	    WaitForStage: client.WorkflowUpdateStageCompleted,
//	    Args: []any{conversation.Message{Text: "hello"}}})
package conversation

import (
	"fmt"
	"strings"
	"time"

	"go.temporal.io/sdk/activity"
	"go.temporal.io/sdk/temporal"
	"go.temporal.io/sdk/workflow"

	"github.com/wleev/temporal-agent-sdk/agent"
	"github.com/wleev/temporal-agent-sdk/model"
)

// DefaultCompactionTimeout bounds a compaction activity (it calls the model).
const DefaultCompactionTimeout = 2 * time.Minute

// Registered names.
const (
	WorkflowName       = "agentsdk_conversation"
	SendMessageUpdate  = "send_message"
	HistoryQuery       = "history"
	CompactionActivity = "agentsdk_compact"
)

// ErrorTypeConfig marks permanent configuration errors.
const ErrorTypeConfig = "AgentSDKConversationConfig"

// Input starts a conversation. It names the agent (resolved by name, since an
// [agent.Agent] holds Go funcs that cannot be serialized) and carries any prior
// history, which is how continue-as-new preserves memory.
type Input struct {
	Agent   string          `json:"agent"`
	History []model.Message `json:"history,omitempty"`
}

// Message is one user turn — the [SendMessageUpdate] argument.
type Message struct {
	Text string `json:"text"`
}

// Reply is the result of a turn.
type Reply struct {
	Output string `json:"output"`
	Turns  int    `json:"turns"`
}

// Conversation registers the conversation workflow and its compaction activity.
type Conversation struct {
	registry  *agent.Registry
	compactor Compactor
}

// Option configures a [Conversation].
type Option func(*Conversation)

// WithCompactor sets the compaction strategy used before continue-as-new. With
// none, history is carried forward verbatim (fine for short conversations).
func WithCompactor(c Compactor) Option {
	return func(cv *Conversation) { cv.compactor = c }
}

// New builds a conversation runner. Agents referenced by [Input.Agent] must be
// in the registry.
func New(registry *agent.Registry, opts ...Option) *Conversation {
	cv := &Conversation{registry: registry}
	for _, o := range opts {
		o(cv)
	}
	return cv
}

// WorkflowRegistry is the part of a worker needed to register the workflow.
type WorkflowRegistry interface {
	RegisterWorkflowWithOptions(w any, options workflow.RegisterOptions)
	RegisterActivityWithOptions(a any, options activity.RegisterOptions)
}

// Register wires the conversation workflow and, if a compactor was set, its
// compaction activity into a worker.
func (cv *Conversation) Register(r WorkflowRegistry) {
	r.RegisterWorkflowWithOptions(cv.run, workflow.RegisterOptions{Name: WorkflowName})
	if cv.compactor != nil {
		r.RegisterActivityWithOptions(cv.compactor.Compact, activity.RegisterOptions{Name: CompactionActivity})
	}
}

// state is the per-run conversation memory.
type state struct {
	history []model.Message
	turns   int
}

// run is the conversation workflow.
func (cv *Conversation) run(ctx workflow.Context, in Input) error {
	a, ok := cv.registry.Get(in.Agent)
	if !ok {
		return temporal.NewNonRetryableApplicationError(
			fmt.Sprintf("conversation: no agent named %q is registered", in.Agent),
			ErrorTypeConfig, nil)
	}

	// One Session for the workflow's lifetime, so the approval surface is shared
	// across every turn.
	s, err := agent.NewSession(ctx)
	if err != nil {
		return err
	}

	st := &state{history: in.History}

	// Each user turn runs one agent turn synchronously and returns the reply.
	// Update (not Signal) so the caller gets the answer and validation.
	err = workflow.SetUpdateHandlerWithOptions(ctx, SendMessageUpdate,
		func(ctx workflow.Context, msg Message) (Reply, error) {
			res, err := s.RunWith(ctx, a, msg.Text, agent.RunOptions{History: st.history})
			if err != nil {
				return Reply{}, err
			}
			// res.Messages is the full transcript for this turn, already excluding
			// any sub-agent child transcripts. It becomes the running history.
			st.history = res.Messages
			st.turns++
			return Reply{Output: res.Output, Turns: st.turns}, nil
		},
		workflow.UpdateHandlerOptions{
			Validator: func(_ workflow.Context, msg Message) error {
				if strings.TrimSpace(msg.Text) == "" {
					return fmt.Errorf("message text must not be empty")
				}
				return nil
			},
			Description: "Send a user message and receive the agent's reply.",
		})
	if err != nil {
		return err
	}

	if err := workflow.SetQueryHandler(ctx, HistoryQuery,
		func() ([]model.Message, error) { return st.history, nil }); err != nil {
		return err
	}

	// Park until the server suggests continue-as-new AND no turn is mid-flight,
	// so a turn is never cut off. Between turns this simply waits; Update
	// handlers fire independently of this Await.
	err = workflow.Await(ctx, func() bool {
		return workflow.GetInfo(ctx).GetContinueAsNewSuggested() && workflow.AllHandlersFinished(ctx)
	})
	if err != nil {
		// Canceled or terminated while parked.
		return err
	}

	// Compact before carrying history into the next run, keeping the workflow
	// within Temporal's history limits.
	carried := st.history
	if cv.compactor != nil {
		ctx = workflow.WithActivityOptions(ctx, workflow.ActivityOptions{
			StartToCloseTimeout: DefaultCompactionTimeout,
		})
		if err := workflow.ExecuteActivity(ctx, CompactionActivity, st.history).Get(ctx, &carried); err != nil {
			return err
		}
	}

	return workflow.NewContinueAsNewError(ctx, WorkflowName, Input{Agent: in.Agent, History: carried})
}
