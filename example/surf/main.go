package main

import (
	"context"
	"errors"
	"flag"
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

var (
	flagBaseURL = flag.String("base-url", envOr("OPENAI_BASE_URL", ""),
		"OpenAI-compatible base URL (e.g. http://localhost:8000/v1 for vLLM)")
	flagModel = flag.String("model", envOr("AGENT_MODEL", "gpt-5.2"), "model name")
	// 127.0.0.1, not localhost: `temporal server start-dev` binds IPv4-only, and
	// on macOS localhost resolves to IPv6 (::1) first, which hangs the dial.
	flagAddress = flag.String("address", envOr("TEMPORAL_ADDRESS", "127.0.0.1:7233"),
		"Temporal frontend address")
	flagWeatherKey = flag.String("weather-key", os.Getenv("WEATHERAPI_KEY"),
		"WeatherAPI.com key (free at https://www.weatherapi.com/signup.aspx)")
	flagData  = flag.String("data", envOr("SURF_DATA", "surf-sessions.json"), "session log file")
	flagSpots = flag.String("spots", envOr("SURF_SPOTS", "surf-spots.json"), "saved spots file")
)

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
	if len(os.Args) < 2 {
		usage()
		os.Exit(2)
	}
	cmd := os.Args[1]
	// Flags follow the command: `surf worker -base-url ... -model ...`.
	_ = flag.CommandLine.Parse(os.Args[2:])
	args := flag.Args()

	switch cmd {
	case "worker":
		runWorker()
	case "ask":
		if len(args) == 0 {
			log.Fatal(`ask needs a question, e.g. ask "what's the surf at Ericeira this week?"`)
		}
		ask(strings.Join(args, " "))
	default:
		usage()
		os.Exit(2)
	}
}

func usage() {
	fmt.Fprint(os.Stderr, `surf — an agent combining a real MCP marine forecast with a local surf log

Usage: surf <command> [flags]

  worker            start the worker
  ask "<question>"  ask the surf coach

Set -weather-key (or WEATHERAPI_KEY) — free at https://www.weatherapi.com/signup.aspx.
Set OPENAI_API_KEY, or -base-url for an OpenAI-compatible endpoint. The worker
spawns the WeatherAPI MCP server via: npx -y weatherapi-mcp (needs Node/npx on PATH).

Flags (after the command):
`)
	flag.PrintDefaults()
}

func envOr(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

func tracingInterceptor() interceptor.Interceptor {
	ti, err := observability.TracingInterceptor()
	if err != nil {
		log.Fatalf("tracing: %v", err)
	}
	return ti
}

func newClient() client.Client {
	c, err := client.Dial(client.Options{
		HostPort:     *flagAddress,
		Interceptors: []interceptor.ClientInterceptor{tracingInterceptor()},
	})
	if err != nil {
		log.Fatalf("connecting to Temporal at %s: %v\n\nIs it running? Try: temporal server start-dev", *flagAddress, err)
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
	if *flagWeatherKey == "" {
		log.Fatal("set -weather-key or WEATHERAPI_KEY (free at https://www.weatherapi.com/signup.aspx)")
	}

	provider, err := newProvider()
	if err != nil {
		log.Fatalf("provider: %v", err)
	}

	c := newClient()
	defer c.Close()

	w := worker.New(c, taskQueue, worker.Options{
		Interceptors: []interceptor.WorkerInterceptor{tracingInterceptor()},
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
	if err := mcpActs.Register(mcpServer, mcpsdk.CommandFactoryWith(
		func(cmd *exec.Cmd) { cmd.Env = append(os.Environ(), "WEATHERAPI_KEY="+*flagWeatherKey) },
		"npx", "-y", "weatherapi-mcp")); err != nil {
		log.Fatal(err)
	}
	mcpActs.RegisterWith(w)

	// Local, file-backed tools, registered under the names the agent references.
	registerStore(w, &Store{Path: *flagData, SpotsPath: *flagSpots})

	// Agent + entry workflow.
	agent.RegisterWorkflows(w, buildAgent(*flagModel))
	w.RegisterWorkflow(SurfWorkflow)

	log.Printf("surf worker on %q (model %q, log %q)", taskQueue, *flagModel, *flagData)
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

func ask(question string) {
	c := newClient()
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
