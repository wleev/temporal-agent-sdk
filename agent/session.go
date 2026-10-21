package agent

import (
	"fmt"
	"time"

	"go.temporal.io/sdk/temporal"
	"go.temporal.io/sdk/workflow"

	"github.com/wleev/temporal-agent-sdk/model"
	"github.com/wleev/temporal-agent-sdk/tool"
)

// Handler names a [Session] registers unless it is created with
// [WithoutApprovalHandlers].
const (
	// ApproveUpdate decides one pending approval. It is an Update, so the approver
	// learns synchronously whether the decision was accepted or rejected.
	ApproveUpdate = "agent_approve"

	// PendingApprovalsQuery lists approvals awaiting a decision.
	PendingApprovalsQuery = "agent_pending_approvals"
)

// DefaultApprovalTimeout bounds how long a gated tool waits for a human.
const DefaultApprovalTimeout = 24 * time.Hour

// ErrorTypeApproval marks approval-machinery failures.
const ErrorTypeApproval = "AgentSDKApprovalError"

// Session holds the per-workflow state shared across agent runs: the approval
// surface and the MCP tool cache.
//
// Create one Session per workflow and reuse it across runs.
type Session struct {
	// approvalHandlers reports whether NewSession registered ApproveUpdate and
	// PendingApprovalsQuery.
	approvalHandlers bool

	// pending is keyed by tool call ID for lookup; order comes from pendingOrder,
	// since ranging a map is nondeterministic.
	pending      map[string]*PendingApproval
	pendingOrder []string

	decisions map[string]*Decision

	// mcpToolCache caches each MCP server's tool list for the workflow's
	// lifetime, so N agent runs cost one list_tools activity per server rather
	// than N.
	mcpToolCache map[string][]tool.Tool

	// mcpOptions holds per-server MCP options, keyed by server name.
	mcpOptions MCPOptions
}

// PendingApproval describes a tool call awaiting a human decision. It is
// returned by the [PendingApprovalsQuery] query.
type PendingApproval struct {
	// CallID is the model's tool call ID, and the key an approver echoes back in
	// an [ApprovalRequest].
	CallID string `json:"call_id"`

	Agent string `json:"agent"`
	Tool  string `json:"tool"`

	// Arguments is the JSON the model wants to call the tool with.
	Arguments string `json:"arguments"`

	// Prompt is the tool's approval prompt, if it set one.
	Prompt string `json:"prompt,omitempty"`

	// RequestedAt is workflow time, not wall-clock time.
	RequestedAt time.Time `json:"requested_at"`
}

// ApprovalRequest is the argument to the [ApproveUpdate] update.
type ApprovalRequest struct {
	// CallID identifies the pending approval. Required.
	CallID string `json:"call_id"`

	// Approved is the decision.
	Approved bool `json:"approved"`

	// Reason is shown to the model. On a denial it lets the model understand the
	// refusal and choose another route.
	Reason string `json:"reason,omitempty"`
}

// Decision is the recorded outcome of an approval.
type Decision struct {
	Approved bool   `json:"approved"`
	Reason   string `json:"reason,omitempty"`

	// TimedOut reports that no human answered in time.
	TimedOut bool `json:"timed_out,omitempty"`
}

// modelMessage renders a refusal for the model.
func (d *Decision) modelMessage(toolName string) string {
	switch {
	case d.TimedOut:
		return fmt.Sprintf(
			"Tool call %q was not approved: the approval request timed out. "+
				"Do not retry this tool; continue without it or ask the user how to proceed.", toolName)
	case d.Reason != "":
		return fmt.Sprintf("Tool call %q was denied by a human reviewer. Reason: %s", toolName, d.Reason)
	default:
		return fmt.Sprintf("Tool call %q was denied by a human reviewer.", toolName)
	}
}

// SessionOption configures NewSession.
type SessionOption func(*sessionConfig)

// sessionConfig holds the settings SessionOptions apply.
type sessionConfig struct {
	withoutApprovalHandlers bool
}

// WithoutApprovalHandlers makes NewSession register neither the ApproveUpdate
// update nor the PendingApprovalsQuery query. A call to a tool whose Policy
// needs approval then fails the run with a non-retryable ErrorTypeApproval
// error. Other calls in the same reply still run.
func WithoutApprovalHandlers() SessionOption {
	return func(c *sessionConfig) { c.withoutApprovalHandlers = true }
}

// NewSession creates a session. Without options it registers the ApproveUpdate
// update and the PendingApprovalsQuery query on the workflow; with
// WithoutApprovalHandlers it registers neither. Registering a session's handlers
// replaces handlers registered earlier in the workflow under the same names: a
// later default NewSession, or the package-level Run, RunWith, and RunTyped,
// which create a default session, takes over ApproveUpdate and
// PendingApprovalsQuery.
func NewSession(ctx workflow.Context, opts ...SessionOption) (*Session, error) {
	var cfg sessionConfig
	for _, o := range opts {
		o(&cfg)
	}

	s := &Session{
		approvalHandlers: !cfg.withoutApprovalHandlers,
		pending:          make(map[string]*PendingApproval),
		decisions:        make(map[string]*Decision),
		mcpToolCache:     make(map[string][]tool.Tool),
	}
	if !s.approvalHandlers {
		return s, nil
	}

	err := workflow.SetUpdateHandlerWithOptions(ctx, ApproveUpdate, s.handleApprove,
		workflow.UpdateHandlerOptions{
			Validator:   s.validateApprove,
			Description: "Approve or deny a pending agent tool call.",
		})
	if err != nil {
		return nil, temporal.NewNonRetryableApplicationError(
			fmt.Sprintf("registering %q update handler: %v", ApproveUpdate, err),
			ErrorTypeApproval, err)
	}

	if err := workflow.SetQueryHandler(ctx, PendingApprovalsQuery, s.handlePendingQuery); err != nil {
		return nil, temporal.NewNonRetryableApplicationError(
			fmt.Sprintf("registering %q query handler: %v", PendingApprovalsQuery, err),
			ErrorTypeApproval, err)
	}

	return s, nil
}

// validateApprove rejects a decision before it reaches history.
//
// Per the Temporal contract, a validator must not mutate workflow state; it only
// reads here. Rejected updates are never written to history, and the approver
// gets a synchronous rejection.
func (s *Session) validateApprove(req ApprovalRequest) error {
	if req.CallID == "" {
		return fmt.Errorf("call_id must not be empty")
	}
	if _, ok := s.pending[req.CallID]; !ok {
		if _, decided := s.decisions[req.CallID]; decided {
			return fmt.Errorf("tool call %q has already been decided", req.CallID)
		}
		return fmt.Errorf("no pending approval for tool call %q", req.CallID)
	}
	return nil
}

// handleApprove records a decision. The waiting tool observes it via Await.
func (s *Session) handleApprove(ctx workflow.Context, req ApprovalRequest) (*Decision, error) {
	d := &Decision{Approved: req.Approved, Reason: req.Reason}
	s.decisions[req.CallID] = d
	s.removePending(req.CallID)

	workflow.GetLogger(ctx).Info("approval decided",
		"call_id", req.CallID, "approved", req.Approved)
	return d, nil
}

// handlePendingQuery lists pending approvals in request order.
func (s *Session) handlePendingQuery() ([]PendingApproval, error) {
	out := make([]PendingApproval, 0, len(s.pendingOrder))
	for _, id := range s.pendingOrder {
		if p, ok := s.pending[id]; ok {
			out = append(out, *p)
		}
	}
	return out, nil
}

// PendingApprovals returns the approvals currently awaiting a decision, for
// workflow code that wants to surface them itself.
func (s *Session) PendingApprovals() []PendingApproval {
	out, _ := s.handlePendingQuery()
	return out
}

func (s *Session) addPending(p *PendingApproval) {
	s.pending[p.CallID] = p
	s.pendingOrder = append(s.pendingOrder, p.CallID)
}

func (s *Session) removePending(callID string) {
	delete(s.pending, callID)
	for i, id := range s.pendingOrder {
		if id == callID {
			s.pendingOrder = append(s.pendingOrder[:i], s.pendingOrder[i+1:]...)
			return
		}
	}
}

// awaitApproval parks until a human decides, or until the timeout expires. It
// fails with a non-retryable ErrorTypeApproval error when the session has no
// approval handlers.
//
// The wait is durable: the workflow holds no worker resources while parked and
// survives worker restarts.
func (s *Session) awaitApproval(ctx workflow.Context, a *Agent, t tool.Tool, call model.ToolCall) (*Decision, error) {
	if !s.approvalHandlers {
		return nil, temporal.NewNonRetryableApplicationError(
			fmt.Sprintf("tool %q needs approval, but the session has no approval handlers", call.Name),
			ErrorTypeApproval, nil)
	}

	// Reject a duplicate call ID, which would let one approval satisfy two calls.
	if _, exists := s.decisions[call.ID]; exists {
		return nil, temporal.NewNonRetryableApplicationError(
			fmt.Sprintf("tool call ID %q was reused within one session", call.ID),
			ErrorTypeApproval, nil)
	}

	p := &PendingApproval{
		CallID:      call.ID,
		Agent:       a.name,
		Tool:        call.Name,
		Arguments:   string(call.Arguments),
		Prompt:      t.Policy().ApprovalPrompt,
		RequestedAt: workflow.Now(ctx),
	}
	s.addPending(p)

	timeout := a.approvalTimeout
	if timeout <= 0 {
		timeout = DefaultApprovalTimeout
	}

	logger := workflow.GetLogger(ctx)
	logger.Info("awaiting approval",
		"agent", a.name, "tool", call.Name, "call_id", call.ID, "timeout", timeout)

	ok, err := workflow.AwaitWithTimeout(ctx, timeout, func() bool {
		_, decided := s.decisions[call.ID]
		return decided
	})
	if err != nil {
		// Workflow canceled or otherwise torn down while parked.
		s.removePending(call.ID)
		return nil, err
	}

	if !ok {
		// A timeout is a denial: the model is told the call was not approved and
		// continues without it rather than failing the workflow.
		d := &Decision{Approved: false, TimedOut: true}
		s.decisions[call.ID] = d
		s.removePending(call.ID)
		logger.Warn("approval timed out",
			"agent", a.name, "tool", call.Name, "call_id", call.ID)
		return d, nil
	}

	return s.decisions[call.ID], nil
}
