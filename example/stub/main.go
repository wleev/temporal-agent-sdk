package main

import (
	"context"
	"encoding/json"
	"errors"
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
	if len(os.Args) < 2 {
		usage()
		os.Exit(2)
	}
	cmd := os.Args[1]
	// Flags follow the command: `stub worker -base-url ... -model ...`.
	_ = flag.CommandLine.Parse(os.Args[2:])
	args := flag.Args()

	switch cmd {
	case "worker":
		runWorker()
	case "ask":
		if len(args) == 0 {
			log.Fatal("ask needs a question, e.g. ask \"where is order ORD-1234?\"")
		}
		ask(strings.Join(args, " "))
	case "pending":
		if len(args) == 0 {
			log.Fatal("pending needs a workflow ID")
		}
		pending(args[0])
	case "approve", "deny":
		if len(args) < 2 {
			log.Fatalf("%s needs a workflow ID and a call ID", cmd)
		}
		decide(args[0], args[1], cmd == "approve")
	default:
		usage()
		os.Exit(2)
	}
}

func usage() {
	fmt.Fprintf(os.Stderr, `Usage: stub <command> [flags]

  worker                              start the worker
  ask "<question>"                    run the agent
  pending <workflow-id>               list approvals awaiting a decision
  approve <workflow-id> <call-id>     approve a pending tool call
  deny <workflow-id> <call-id>        deny a pending tool call

Flags (after the command):
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

// newProvider builds the OpenAI(-compatible) provider from -base-url, falling
// back to OPENAI_API_KEY.
func newProvider() (*oaiprovider.Provider, error) {
	var opts []oaiprovider.Option
	switch {
	case *flagBaseURL != "":
		opts = append(opts, oaiprovider.WithBaseURL(*flagBaseURL))
		log.Printf("using OpenAI-compatible endpoint at %s", *flagBaseURL)
	case os.Getenv("OPENAI_API_KEY") == "":
		return nil, errors.New("set OPENAI_API_KEY, or -base-url for an OpenAI-compatible endpoint")
	}
	return oaiprovider.New(opts...)
}

func runWorker() {
	c := newClient()
	defer c.Close()

	provider, err := newProvider()
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
	fmt.Printf("(if it needs approval: stub pending %s)\n\n", run.GetID())

	var answer string
	if err := run.Get(ctx, &answer); err != nil {
		log.Fatalf("agent failed: %v", err)
	}
	fmt.Printf("%s\n", answer)
}

func pending(wfID string) {
	c := newClient()
	defer c.Close()

	list, err := agent.NewApprovalClient(c).Pending(context.Background(), wfID)
	if err != nil {
		log.Fatalf("listing pending approvals: %v", err)
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
	fmt.Printf("approve with: stub approve %s <call-id>\n", wfID)
}

func decide(wfID, callID string, approved bool) {
	c := newClient()
	defer c.Close()

	// ApprovalClient wraps the Update: the decision is validated and awaited, and a
	// stale or unknown call ID returns an error.
	ac := agent.NewApprovalClient(c)
	var err error
	if approved {
		err = ac.Approve(context.Background(), wfID, callID)
	} else {
		err = ac.Deny(context.Background(), wfID, callID, "denied by the operator")
	}
	if err != nil {
		log.Fatalf("sending decision: %v", err)
	}

	if approved {
		fmt.Printf("approved %s\n", callID)
	} else {
		fmt.Printf("denied %s\n", callID)
	}
}
