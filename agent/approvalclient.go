package agent

import (
	"context"
	"fmt"

	"go.temporal.io/sdk/client"
)

// ApprovalClient is the approver-side surface for a running agent workflow. It
// wraps a Temporal client and lists, approves, or denies pending tool calls by ID
// through [ApproveUpdate] and [PendingApprovalsQuery].
type ApprovalClient struct {
	client client.Client
}

// NewApprovalClient returns an [ApprovalClient] over c.
func NewApprovalClient(c client.Client) ApprovalClient {
	return ApprovalClient{client: c}
}

// Pending lists the tool calls awaiting a decision on the workflow's latest run.
func (ac ApprovalClient) Pending(ctx context.Context, workflowID string) ([]PendingApproval, error) {
	val, err := ac.client.QueryWorkflow(ctx, workflowID, "", PendingApprovalsQuery)
	if err != nil {
		return nil, fmt.Errorf("querying pending approvals: %w", err)
	}
	var pending []PendingApproval
	if err := val.Get(&pending); err != nil {
		return nil, fmt.Errorf("decoding pending approvals: %w", err)
	}
	return pending, nil
}

// Approve approves the pending tool call with the given ID. It blocks until the
// decision is recorded and returns an error if the call ID is stale, unknown, or
// already decided.
func (ac ApprovalClient) Approve(ctx context.Context, workflowID, callID string) error {
	return ac.decide(ctx, workflowID, ApprovalRequest{CallID: callID, Approved: true})
}

// Deny denies the pending tool call with the given ID. A non-empty reason is
// passed to the model as the tool result.
func (ac ApprovalClient) Deny(ctx context.Context, workflowID, callID, reason string) error {
	return ac.decide(ctx, workflowID, ApprovalRequest{CallID: callID, Approved: false, Reason: reason})
}

func (ac ApprovalClient) decide(ctx context.Context, workflowID string, req ApprovalRequest) error {
	handle, err := ac.client.UpdateWorkflow(ctx, client.UpdateWorkflowOptions{
		WorkflowID:   workflowID,
		UpdateName:   ApproveUpdate,
		WaitForStage: client.WorkflowUpdateStageCompleted,
		Args:         []any{req},
	})
	if err != nil {
		return fmt.Errorf("sending approval decision: %w", err)
	}
	// Wait for the handler to record the decision; a rejected update (unknown or
	// already-decided call ID) surfaces here.
	if err := handle.Get(ctx, nil); err != nil {
		return fmt.Errorf("approval decision rejected: %w", err)
	}
	return nil
}
