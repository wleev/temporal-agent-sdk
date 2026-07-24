// Package main is a runnable end-to-end demo of the agent SDK.
//
// It defines a support agent with four kinds of tool:
//
//   - a workflow tool (get_time) that runs in the workflow
//   - an activity tool (lookup_order) that runs in an activity
//   - an approval-gated tool (issue_refund) that parks for a human
//   - two sub-agents (research, policy) that run as child workflows
//
// See README.md for how to run it.
package main

import (
	"context"
	"fmt"
	"strings"
	"time"

	"go.temporal.io/sdk/temporal"
	"go.temporal.io/sdk/workflow"

	"github.com/wleev/temporal-agent-sdk/agent"
	"github.com/wleev/temporal-agent-sdk/tool"
)

// --- Tool argument types -----------------------------------------------------
//
// Each struct is the single source of truth: its shape becomes the JSON Schema
// the model sees, and the handler receives a decoded value of the same type.
// Field descriptions matter — they are how the model learns what to pass.

type timeIn struct {
	Timezone string `json:"timezone" jsonschema_description:"IANA timezone name, e.g. Europe/Brussels"`
}

type orderIn struct {
	OrderID string `json:"order_id" jsonschema_description:"The customer's order ID, e.g. ORD-1234"`
}

type orderOut struct {
	OrderID string  `json:"order_id"`
	Item    string  `json:"item"`
	Total   float64 `json:"total"`
	Status  string  `json:"status"`
}

type refundIn struct {
	OrderID string  `json:"order_id" jsonschema_description:"The order to refund"`
	Amount  float64 `json:"amount" jsonschema_description:"Amount to refund in EUR"`
	Reason  string  `json:"reason" jsonschema_description:"Why the refund is being issued"`
}

// --- Activities --------------------------------------------------------------

// LookupOrder is a Temporal activity, so it may do real I/O and is retried by
// Temporal if it fails. A workflow tool could not do this.
func LookupOrder(ctx context.Context, in orderIn) (orderOut, error) {
	// A real implementation would query a database here.
	if !strings.HasPrefix(in.OrderID, "ORD-") {
		// A business error the model can act on: it will usually ask the
		// customer for a valid ID rather than give up.
		return orderOut{}, fmt.Errorf("no order found with ID %q", in.OrderID)
	}
	return orderOut{
		OrderID: in.OrderID,
		Item:    "Mechanical keyboard",
		Total:   89.99,
		Status:  "delivered",
	}, nil
}

// IssueRefund is the side effect the approval gate protects.
func IssueRefund(ctx context.Context, in refundIn) (string, error) {
	// A real implementation would call a payment provider here.
	return fmt.Sprintf("Refunded €%.2f for order %s (%s).", in.Amount, in.OrderID, in.Reason), nil
}

// --- Tools -------------------------------------------------------------------

// getTimeTool runs inside the workflow.
//
// It reads the clock through workflow.Now rather than time.Now: the workflow
// clock replays to the same instant, while the wall clock would return a
// different value on every replay and break determinism.
var getTimeTool = must(tool.New("get_time",
	"Get the current date and time in a given timezone.",
	func(ctx workflow.Context, in timeIn) (string, error) {
		loc, err := time.LoadLocation(in.Timezone)
		if err != nil {
			return "", fmt.Errorf("unknown timezone %q", in.Timezone)
		}
		return workflow.Now(ctx).In(loc).Format(time.RFC1123), nil
	}))

// lookupOrderTool runs in an activity: it is I/O, so it gets retries and
// timeouts and is free of determinism rules.
var lookupOrderTool = must(tool.Activity[orderIn, orderOut]("lookup_order",
	"Look up an order by its ID.",
	LookupOrder,
	tool.WithActivityOptions(workflow.ActivityOptions{
		StartToCloseTimeout: 10 * time.Second,
		RetryPolicy: &temporal.RetryPolicy{
			MaximumAttempts: 3,
			// A missing order is permanent; retrying it wastes the budget.
			NonRetryableErrorTypes: []string{"NotFound"},
		},
	})))

// refundTool is gated: the loop parks before every call until a human decides.
// The wait is durable — the workflow holds no worker while parked and survives
// restarts, so a 24-hour window costs nothing.
var refundTool = must(tool.Activity[refundIn, string]("issue_refund",
	"Issue a refund for an order. Requires human approval.",
	IssueRefund,
	tool.WithApprovalPrompt("A refund needs review before it is issued."),
	tool.WithActivityOptions(workflow.ActivityOptions{
		StartToCloseTimeout: 30 * time.Second,
	})))

// --- Agents ------------------------------------------------------------------
//
// Agents are built by buildAgents rather than declared as package-level vars,
// because the model name comes from a flag and package vars initialize before
// flag.Parse runs — a flag read at init time is always the default.

var (
	supportAgent  *agent.Agent
	researchAgent *agent.Agent
	policyAgent   *agent.Agent
)

// Sub-agent instructions, shared with the test that identifies which sub-agent
// ran by its system message.
const (
	researchInstructions = "You research product questions. Answer concisely and factually."
	policyInstructions   = "You answer questions about company policy. " +
		"The refund window is 30 days from delivery. Shipping is free above €50."
)

// must panics if construction failed. Startup wiring builds fixed tools and
// agents, so an error is a programming bug, not a runtime condition. The SDK
// ships no such helper; a consumer declares its own and chooses whether to panic.
func must[T any](v T, err error) T {
	if err != nil {
		panic(err)
	}
	return v
}

// buildAgents constructs the agent graph for a given model. Call it after flags
// are parsed and before registering workflows.
func buildAgents(modelName string) *agent.Registry {
	// researchAgent runs as a child workflow when the support agent delegates,
	// so it gets its own history budget and its turns never touch the parent's.
	researchAgent = must(agent.NewAgent(
		"researcher",
		modelName,
		agent.WithInstructions(researchInstructions),
		agent.WithTools(getTimeTool),
		agent.WithMaxTurns(5),
	))

	policyAgent = must(agent.NewAgent(
		"policy",
		modelName,
		agent.WithInstructions(policyInstructions),
		agent.WithMaxTurns(5),
	))

	supportAgent = must(agent.NewAgent(
		"support",
		modelName,
		agent.WithInstructions("You are a customer support agent for an online electronics store. "+
			"Use lookup_order to find orders. Delegate product questions to the research agent "+
			"and policy questions to the policy agent — you may consult both at once. "+
			"Refunds require human approval; if one is denied, explain why to the customer."),
		agent.WithTools(
			getTimeTool,
			lookupOrderTool,
			refundTool,
			// Both sub-agents present as ordinary tools; the model cannot tell
			// they are child workflows. If it asks for both in one turn, they
			// run concurrently.
			must(agent.AsSubAgent(researchAgent, "research",
				"Delegate a product research question to a specialist agent.")),
			must(agent.AsSubAgent(policyAgent, "policy",
				"Delegate a company policy question to a specialist agent.")),
		),
		agent.WithMaxTurns(10),
		agent.WithApprovalTimeout(24*time.Hour),
	))

	// Child workflows resolve agents by name, since an Agent holds Go funcs and
	// cannot be serialized.
	reg := agent.NewRegistry()
	if err := reg.Add(supportAgent, researchAgent, policyAgent); err != nil {
		panic(err)
	}
	return reg
}
