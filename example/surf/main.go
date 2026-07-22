package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"os/exec"
	"strings"
	"time"

	"go.temporal.io/sdk/activity"
	"go.temporal.io/sdk/client"
	"go.temporal.io/sdk/interceptor"
	"go.temporal.io/sdk/worker"
	"go.temporal.io/sdk/workflow"

	"github.com/wleev/temporal-agent-sdk/agent"
	"github.com/wleev/temporal-agent-sdk/mcp"
	"github.com/wleev/temporal-agent-sdk/mcp/mcpsdk"
	"github.com/wleev/temporal-agent-sdk/model"
	oaiprovider "github.com/wleev/temporal-agent-sdk/model/openai"
	"github.com/wleev/temporal-agent-sdk/observability"
)

const taskQueue = "surf-example"

// SurfWorkflow is the entry point: it runs one agent turn and returns the reply.
// A real chat app would use the conversation package for a durable multi-turn
// session; this keeps the demo a single request/response.
func SurfWorkflow(ctx workflow.Context, question string) (string, error) {
	res, err := agent.Run(ctx, surfAgent, question)
	if err != nil {
		return "", err
	}
	return res.Output, nil
}

func main() {
	// 127.0.0.1, not localhost: `temporal server start-dev` binds IPv4-only, and
	// on macOS localhost resolves to IPv6 (::1) first, which hangs the dial.
	address := envOr("TEMPORAL_ADDRESS", "127.0.0.1:7233")
	modelName := envOr("AGENT_MODEL", "gpt-5.2")
	baseURL := os.Getenv("OPENAI_BASE_URL")
	weatherKey := os.Getenv("WEATHERAPI_KEY")
	dataPath := envOr("SURF_DATA", "surf-sessions.json")
	spotsPath := envOr("SURF_SPOTS", "surf-spots.json")

	args := os.Args[1:]
	if len(args) == 0 {
		usage()
		os.Exit(2)
	}

	switch args[0] {
	case "worker":
		runWorker(address, modelName, baseURL, weatherKey, dataPath, spotsPath)
	case "ask":
		if len(args) < 2 {
			log.Fatal(`ask needs a question, e.g. ask "what's the surf at Ericeira this week?"`)
		}
		ask(address, strings.Join(args[1:], " "))
	default:
		usage()
		os.Exit(2)
	}
}

func usage() {
	fmt.Fprint(os.Stderr, `surf — an agent combining a real MCP marine forecast with a local surf log

Usage:
  surf worker            start the worker
  surf ask "<question>"  ask the surf coach

Environment:
  WEATHERAPI_KEY   WeatherAPI.com key (free tier works) — https://www.weatherapi.com/signup.aspx
  OPENAI_API_KEY   OpenAI key, or set OPENAI_BASE_URL for a compatible endpoint
  AGENT_MODEL      model name (default gpt-5.2)
  SURF_DATA        session log file (default surf-sessions.json)
  SURF_SPOTS       saved spots file (default surf-spots.json)
  TEMPORAL_ADDRESS Temporal frontend (default localhost:7233)

The worker spawns the WeatherAPI MCP server via: npx -y weatherapi-mcp
(needs Node/npx on PATH).
`)
}

func envOr(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

func tracing() interceptor.Interceptor {
	ti, err := observability.TracingInterceptor()
	if err != nil {
		log.Fatalf("tracing: %v", err)
	}
	return ti
}

func dial(address string) client.Client {
	c, err := client.Dial(client.Options{
		HostPort:     address,
		Interceptors: []interceptor.ClientInterceptor{tracing()},
	})
	if err != nil {
		log.Fatalf("connecting to Temporal at %s: %v\n\nIs it running? Try: temporal server start-dev", address, err)
	}
	return c
}

func runWorker(address, modelName, baseURL, weatherKey, dataPath, spotsPath string) {
	if weatherKey == "" {
		log.Fatal("set WEATHERAPI_KEY (free at https://www.weatherapi.com/signup.aspx)")
	}

	var provOpts []oaiprovider.Option
	if baseURL != "" {
		provOpts = append(provOpts, oaiprovider.WithBaseURL(baseURL))
	} else if os.Getenv("OPENAI_API_KEY") == "" {
		log.Fatal("set OPENAI_API_KEY, or OPENAI_BASE_URL for a compatible endpoint")
	}
	provider, err := oaiprovider.New(provOpts...)
	if err != nil {
		log.Fatalf("provider: %v", err)
	}

	c := dial(address)
	defer c.Close()

	w := worker.New(c, taskQueue, worker.Options{
		Interceptors: []interceptor.WorkerInterceptor{tracing()},
	})

	// Model seam.
	acts, err := model.NewActivities(provider)
	if err != nil {
		log.Fatalf("model activities: %v", err)
	}
	acts.Register(w)

	// The real MCP server: spawn `npx -y weatherapi-mcp` over stdio, with the API
	// key in the child environment. The library's stateless MCP integration
	// connects, calls a tool, and disconnects per invocation.
	mcpActs := mcp.NewActivities()
	mcpActs.MustRegister(mcpServer, mcpsdk.CommandFactoryWith(
		func(cmd *exec.Cmd) { cmd.Env = append(os.Environ(), "WEATHERAPI_KEY="+weatherKey) },
		"npx", "-y", "weatherapi-mcp"))
	mcpActs.RegisterWith(w)

	// Local, file-backed tools, registered under the names the agent references.
	registerStore(w, &Store{Path: dataPath, SpotsPath: spotsPath})

	// Agent + entry workflow.
	agent.RegisterWorkflows(w, buildAgent(modelName))
	w.RegisterWorkflow(SurfWorkflow)

	log.Printf("surf worker on %q (model %q, log %q)", taskQueue, modelName, dataPath)
	if err := w.Run(worker.InterruptCh()); err != nil {
		log.Fatalf("worker stopped: %v", err)
	}
}

// registerStore wires the store's methods as the local tool activities.
func registerStore(w worker.Worker, store *Store) {
	w.RegisterActivityWithOptions(store.Log, activity.RegisterOptions{Name: logActivityName})
	w.RegisterActivityWithOptions(store.Trends, activity.RegisterOptions{Name: trendsActivityName})
	w.RegisterActivityWithOptions(store.AddSpot, activity.RegisterOptions{Name: addSpotActivityName})
	w.RegisterActivityWithOptions(store.ListSpots, activity.RegisterOptions{Name: listSpotsActivityName})
}

func ask(address, question string) {
	c := dial(address)
	defer c.Close()

	ctx := context.Background()
	run, err := c.ExecuteWorkflow(ctx, client.StartWorkflowOptions{
		ID:        fmt.Sprintf("surf-%d", time.Now().UnixNano()),
		TaskQueue: taskQueue,
	}, SurfWorkflow, question)
	if err != nil {
		log.Fatalf("starting workflow: %v", err)
	}
	fmt.Printf("workflow: %s\n\n", run.GetID())

	var answer string
	if err := run.Get(ctx, &answer); err != nil {
		log.Fatalf("agent failed: %v", err)
	}
	fmt.Println(answer)
}
