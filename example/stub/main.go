package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"
	"strings"
	"time"

	"go.temporal.io/sdk/client"
	"go.temporal.io/sdk/interceptor"
	"go.temporal.io/sdk/worker"
	"go.temporal.io/sdk/workflow"

	"github.com/wleev/temporal-agent-sdk/agent"
	"github.com/wleev/temporal-agent-sdk/model"
	oaiprovider "github.com/wleev/temporal-agent-sdk/model/openai"
	"github.com/wleev/temporal-agent-sdk/observability"
)

const taskQueue = "agent-example"

var (
	flagBaseURL = flag.String("base-url", envOr("OPENAI_BASE_URL", ""),
		"OpenAI-compatible base URL (e.g. http://localhost:8000/v1 for vLLM)")
	flagModel = flag.String("model", envOr("AGENT_MODEL", "gpt-5.2"),
		"model name")
	// 127.0.0.1, not localhost: `temporal server start-dev` binds IPv4-only, and
	// on macOS localhost resolves to IPv6 (::1) first, which hangs the dial.
	flagAddress = flag.String("address", envOr("TEMPORAL_ADDRESS", "127.0.0.1:7233"),
		"Temporal frontend address")
)

func envOr(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

// SupportWorkflow is the entry-point workflow. The durable agent loop lives in
// agent.Run; this only returns its result. Real applications typically park here
// and take user turns via Update instead of returning after one.
func SupportWorkflow(ctx workflow.Context, question string) (string, error) {
	res, err := agent.Run(ctx, supportAgent, question)
	if err != nil {
		return "", err
	}
	workflow.GetLogger(ctx).Info("agent finished",
		"turns", res.Turns, "tokens", res.Usage.TotalTokens)
	return res.Output, nil
}

func main() {
	flag.Parse()

	if len(flag.Args()) == 0 {
		usage()
		os.Exit(2)
	}

	switch flag.Arg(0) {
	case "worker":
		runWorker()
	case "ask":
		if len(flag.Args()) < 2 {
			log.Fatal("ask needs a question, e.g. ask \"where is order ORD-1234?\"")
		}
		ask(strings.Join(flag.Args()[1:], " "))
	case "pending":
		pending()
	case "approve", "deny":
		if len(flag.Args()) < 3 {
			log.Fatalf("%s needs a workflow ID and a call ID", flag.Arg(0))
		}
		decide(flag.Arg(1), flag.Arg(2), flag.Arg(0) == "approve")
	default:
		usage()
		os.Exit(2)
	}
}

func usage() {
	fmt.Fprintf(os.Stderr, `Usage:
  example worker                              start the worker
  example ask "<question>"                    run the agent
  example pending <workflow-id>               list approvals awaiting a decision
  example approve <workflow-id> <call-id>     approve a pending tool call
  example deny <workflow-id> <call-id>        deny a pending tool call

Flags:
`)
	flag.PrintDefaults()
}

// tracingInterceptor builds the OpenTelemetry interceptor, shared by the client
// and the worker. It uses the global TracerProvider — register an exporter
// (OTLP, stdout) in your app to see spans; with none it is a no-op. Spans for
// the whole run, each model call (with GenAI attributes), and each tool then
// appear automatically.
func tracingInterceptor() interceptor.Interceptor {
	ti, err := observability.TracingInterceptor()
	if err != nil {
		log.Fatalf("building tracing interceptor: %v", err)
	}
	return ti
}

func newClient() client.Client {
	c, err := client.Dial(client.Options{
		HostPort:     *flagAddress,
		Interceptors: []interceptor.ClientInterceptor{tracingInterceptor()},
	})
	if err != nil {
		log.Fatalf("connecting to Temporal at %s: %v\n\nIs it running? Try: temporal server start-dev",
			*flagAddress, err)
	}
	return c
}

func runWorker() {
	c := newClient()
	defer c.Close()

	opts := []oaiprovider.Option{}
	if *flagBaseURL != "" {
		// The vLLM path: an OpenAI-compatible server at a custom address.
		opts = append(opts, oaiprovider.WithBaseURL(*flagBaseURL))
		log.Printf("using OpenAI-compatible endpoint at %s", *flagBaseURL)
	} else if os.Getenv("OPENAI_API_KEY") == "" {
		log.Fatal("set OPENAI_API_KEY, or pass -base-url for an OpenAI-compatible endpoint")
	}

	provider, err := oaiprovider.New(opts...)
	if err != nil {
		log.Fatalf("building provider: %v", err)
	}

	acts, err := model.NewActivities(provider)
	if err != nil {
		log.Fatalf("building model activities: %v", err)
	}

	w := worker.New(c, taskQueue, worker.Options{
		Interceptors: []interceptor.WorkerInterceptor{tracingInterceptor()},
	})

	// The model seam.
	acts.Register(w)

	// The agent workflow, so sub-agent child workflows can run.
	agent.RegisterWorkflows(w, buildAgents(*flagModel))

	// The entry-point workflow and the activities behind activity-backed tools.
	w.RegisterWorkflow(SupportWorkflow)
	w.RegisterActivity(LookupOrder)
	w.RegisterActivity(IssueRefund)

	log.Printf("worker listening on task queue %q (model %q)", taskQueue, *flagModel)
	if err := w.Run(worker.InterruptCh()); err != nil {
		log.Fatalf("worker stopped: %v", err)
	}
}

func ask(question string) {
	c := newClient()
	defer c.Close()

	ctx := context.Background()
	wfID := fmt.Sprintf("support-%d", time.Now().UnixNano())

	run, err := c.ExecuteWorkflow(ctx, client.StartWorkflowOptions{
		ID:        wfID,
		TaskQueue: taskQueue,
	}, SupportWorkflow, question)
	if err != nil {
		log.Fatalf("starting workflow: %v", err)
	}

	fmt.Printf("workflow: %s\n", run.GetID())
	fmt.Printf("waiting for the agent...\n")
	fmt.Printf("(if it needs approval: example pending %s)\n\n", run.GetID())

	var answer string
	if err := run.Get(ctx, &answer); err != nil {
		log.Fatalf("agent failed: %v", err)
	}
	fmt.Printf("%s\n", answer)
}

func pending() {
	c := newClient()
	defer c.Close()

	wfID := flag.Arg(1)
	if wfID == "" {
		log.Fatal("pending needs a workflow ID")
	}

	val, err := c.QueryWorkflow(context.Background(), wfID, "", agent.PendingApprovalsQuery)
	if err != nil {
		log.Fatalf("querying: %v", err)
	}

	var list []agent.PendingApproval
	if err := val.Get(&list); err != nil {
		log.Fatalf("decoding: %v", err)
	}
	if len(list) == 0 {
		fmt.Println("nothing awaiting approval")
		return
	}

	for _, p := range list {
		var pretty any
		if err := json.Unmarshal([]byte(p.Arguments), &pretty); err != nil {
			pretty = p.Arguments // fall back to the raw arguments
		}
		args, _ := json.MarshalIndent(pretty, "  ", "  ")

		fmt.Printf("call-id: %s\n", p.CallID)
		fmt.Printf("  agent: %s\n", p.Agent)
		fmt.Printf("  tool:  %s\n", p.Tool)
		if p.Prompt != "" {
			fmt.Printf("  note:  %s\n", p.Prompt)
		}
		fmt.Printf("  args:  %s\n\n", args)
	}
	fmt.Printf("approve with: example approve %s <call-id>\n", wfID)
}

func decide(wfID, callID string, approved bool) {
	c := newClient()
	defer c.Close()

	reason := ""
	if !approved {
		reason = "denied by the operator"
	}

	// Update rather than Signal: the decision is validated and returns a result,
	// so a stale or unknown call ID is rejected instead of vanishing.
	handle, err := c.UpdateWorkflow(context.Background(), client.UpdateWorkflowOptions{
		WorkflowID:   wfID,
		UpdateName:   agent.ApproveUpdate,
		WaitForStage: client.WorkflowUpdateStageCompleted,
		Args: []any{agent.ApprovalRequest{
			CallID:   callID,
			Approved: approved,
			Reason:   reason,
		}},
	})
	if err != nil {
		log.Fatalf("sending decision: %v", err)
	}

	var d agent.Decision
	if err := handle.Get(context.Background(), &d); err != nil {
		log.Fatalf("decision rejected: %v", err)
	}

	if d.Approved {
		fmt.Printf("approved %s\n", callID)
	} else {
		fmt.Printf("denied %s\n", callID)
	}
}
