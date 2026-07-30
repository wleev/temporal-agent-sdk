// Command askuser is a sample showing how a consumer builds a free-form
// human-input tool — the model asks the user a question mid-run and consumes
// their typed (JSON) answer — on the SDK's primitives. The SDK ships the agent
// loop and the workflow tool surface, not this tool.
//
// The ask_user tool parks the run on a durable workflow Await, exposes the open
// question through a Query, and takes the human's typed answer through an Update —
// the same shape agent approval uses, retyped for a JSON payload instead of a
// bool.
//
//	go run ./example/askuser worker                                  # terminal 1
//	go run ./example/askuser ask "book me a flight"                  # terminal 2 -> prints a workflow ID
//	go run ./example/askuser questions <workflow-id>                 # what is the agent asking?
//	go run ./example/askuser answer <workflow-id> q-1 '{"city":"Ghent"}'
//	go run ./example/askuser result <workflow-id>                    # the final answer
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
	"go.temporal.io/sdk/worker"
	"go.temporal.io/sdk/workflow"

	"github.com/wleev/temporal-agent-sdk/agent"
	"github.com/wleev/temporal-agent-sdk/model"
	oaiprovider "github.com/wleev/temporal-agent-sdk/model/openai"
	"github.com/wleev/temporal-agent-sdk/tool"
)

const (
	taskQueue = "agent-askuser"

	// AnswerQuestionUpdate delivers a human's typed answer to an open question.
	AnswerQuestionUpdate = "answer_question"

	// PendingQuestionsQuery lists the questions awaiting an answer.
	PendingQuestionsQuery = "pending_questions"

	// answerTimeout bounds how long the tool parks waiting for a human.
	answerTimeout = time.Hour
)

var (
	flagBaseURL = flag.String("base-url", envOr("OPENAI_BASE_URL", ""),
		"OpenAI-compatible base URL (e.g. http://localhost:8000/v1 for vLLM)")
	flagModel = flag.String("model", envOr("AGENT_MODEL", "gpt-5.2"), "model name")
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

// AskUserIn is the ask_user tool's argument. Only the question is the model's to
// set; every tool field is sent to the model as required, so an arbitrary-JSON
// "answer schema" field would reflect to a typeless required property that
// OpenAI rejects. A real desk attaches the expected answer schema itself, per
// question type.
type AskUserIn struct {
	Question string `json:"question" jsonschema_description:"The question to put to the human."`
}

// PendingQuestion is one open question, returned by [PendingQuestionsQuery].
type PendingQuestion struct {
	ID       string `json:"id"`
	Question string `json:"question"`
}

// AnswerRequest is the argument to [AnswerQuestionUpdate]: the human's answer to
// one open question.
type AnswerRequest struct {
	ID     string          `json:"id"`
	Answer json.RawMessage `json:"answer"`
}

// deskKey carries the per-run [questionDesk] on the workflow context. The
// package-level ask_user tool reads it to find its run's desk.
type deskKey struct{}

// questionDesk holds one run's open questions and their answers, and serves the
// Update and Query that connect a human to them. Its methods run in workflow code
// and are not synchronized.
type questionDesk struct {
	pending map[string]*PendingQuestion
	order   []string
	answers map[string]json.RawMessage
	counter int
}

func newQuestionDesk() *questionDesk {
	return &questionDesk{
		pending: make(map[string]*PendingQuestion),
		answers: make(map[string]json.RawMessage),
	}
}

// register wires the desk's Update and Query onto the workflow.
func (d *questionDesk) register(ctx workflow.Context) error {
	err := workflow.SetUpdateHandlerWithOptions(ctx, AnswerQuestionUpdate, d.handleAnswer,
		workflow.UpdateHandlerOptions{Validator: d.validateAnswer})
	if err != nil {
		return err
	}
	return workflow.SetQueryHandler(ctx, PendingQuestionsQuery, d.handlePending)
}

// ask records a question, parks until it is answered or the timeout expires, and
// returns the human's answer as the tool result. On timeout it returns an
// explanatory note and no error.
func (d *questionDesk) ask(ctx workflow.Context, in AskUserIn) (string, error) {
	d.counter++
	id := fmt.Sprintf("q-%d", d.counter)
	d.pending[id] = &PendingQuestion{ID: id, Question: in.Question}
	d.order = append(d.order, id)

	answered, err := workflow.AwaitWithTimeout(ctx, answerTimeout, func() bool {
		_, ok := d.answers[id]
		return ok
	})
	d.removePending(id)
	if err != nil {
		return "", err
	}
	if !answered {
		return "No answer was provided: the request timed out.", nil
	}
	ans := d.answers[id]
	delete(d.answers, id)
	return string(ans), nil
}

func (d *questionDesk) validateAnswer(req AnswerRequest) error {
	if _, ok := d.pending[req.ID]; !ok {
		return fmt.Errorf("no pending question %q", req.ID)
	}
	if !json.Valid(req.Answer) {
		return errors.New("answer must be valid JSON")
	}
	// A production desk would also validate the answer against the schema it
	// expects for this question, before it is recorded.
	return nil
}

func (d *questionDesk) handleAnswer(_ workflow.Context, req AnswerRequest) error {
	d.answers[req.ID] = req.Answer
	return nil
}

func (d *questionDesk) handlePending() ([]PendingQuestion, error) {
	out := make([]PendingQuestion, 0, len(d.order))
	for _, id := range d.order {
		if p, ok := d.pending[id]; ok {
			out = append(out, *p)
		}
	}
	return out, nil
}

func (d *questionDesk) removePending(id string) {
	delete(d.pending, id)
	for i, x := range d.order {
		if x == id {
			d.order = append(d.order[:i], d.order[i+1:]...)
			break
		}
	}
}

// askUser is the ask_user tool body. It resolves the run's desk from the context
// and delegates to it.
func askUser(ctx workflow.Context, in AskUserIn) (string, error) {
	desk, ok := ctx.Value(deskKey{}).(*questionDesk)
	if !ok {
		return "", errors.New("ask_user: no question desk on the workflow context")
	}
	return desk.ask(ctx, in)
}

// runAgent is the workflow: it opens a per-run question desk, puts it on the
// context for the ask_user tool, and runs the agent.
func runAgent(ctx workflow.Context, a *agent.Agent, question string) (string, error) {
	desk := newQuestionDesk()
	if err := desk.register(ctx); err != nil {
		return "", err
	}
	ctx = workflow.WithValue(ctx, deskKey{}, desk)

	res, err := agent.Run(ctx, a, question)
	if err != nil {
		return "", err
	}
	return res.Output, nil
}

func buildAgent(modelID string) (*agent.Agent, error) {
	// A strong tool description and an imperative system prompt matter more than
	// the schema for getting the model to actually call the tool: a model tends to
	// ask its question in plain prose unless told that prose is a dead end and the
	// tool is the only channel.
	askTool, err := tool.New("ask_user",
		"Ask the human operator a question and wait for their typed answer. This is the "+
			"ONLY way to reach the user; use it for any information, decision, or "+
			"clarification you need from them.", askUser)
	if err != nil {
		return nil, err
	}
	return agent.NewAgent("assistant", modelID,
		agent.WithInstructions("You run inside a Temporal workflow and cannot talk to the user "+
			"directly — anything you write as a reply is discarded and never reaches them. When a "+
			"request needs information, a decision, or a preference that only the user can provide, "+
			"you MUST call ask_user and wait for the answer before continuing. Never phrase a "+
			"question as your reply; always route it through ask_user."),
		agent.WithTools(askTool))
}

func main() {
	if len(os.Args) < 2 {
		usage()
		os.Exit(2)
	}
	cmd := os.Args[1]
	// Flags follow the command: `askuser worker -base-url ... -model ...`.
	_ = flag.CommandLine.Parse(os.Args[2:])
	args := flag.Args()

	switch cmd {
	case "worker":
		runWorker()
	case "ask":
		if len(args) == 0 {
			log.Fatal(`ask needs a question, e.g. ask "book me a flight"`)
		}
		ask(strings.Join(args, " "))
	case "questions":
		if len(args) == 0 {
			log.Fatal("questions needs a workflow ID")
		}
		questions(args[0])
	case "answer":
		if len(args) < 3 {
			log.Fatal(`answer needs: answer <workflow-id> <question-id> '<json>'`)
		}
		answer(args[0], args[1], args[2])
	case "result":
		if len(args) == 0 {
			log.Fatal("result needs a workflow ID")
		}
		result(args[0])
	default:
		usage()
		os.Exit(2)
	}
}

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
	a, err := buildAgent(*flagModel)
	if err != nil {
		log.Fatalf("building agent: %v", err)
	}

	w := worker.New(c, taskQueue, worker.Options{})
	acts.Register(w)
	w.RegisterWorkflowWithOptions(func(ctx workflow.Context, question string) (string, error) {
		return runAgent(ctx, a, question)
	}, workflow.RegisterOptions{Name: "AskUserWorkflow"})

	log.Printf("worker listening on task queue %q (model %q)", taskQueue, *flagModel)
	if err := w.Run(worker.InterruptCh()); err != nil {
		log.Fatalf("worker stopped: %v", err)
	}
}

func ask(question string) {
	c := newClient()
	defer c.Close()

	run, err := c.ExecuteWorkflow(context.Background(),
		client.StartWorkflowOptions{TaskQueue: taskQueue}, "AskUserWorkflow", question)
	if err != nil {
		log.Fatalf("starting run: %v", err)
	}
	fmt.Printf("started: %s\n", run.GetID())
	fmt.Printf("check questions with: askuser questions %s\n", run.GetID())
}

func questions(wfID string) {
	c := newClient()
	defer c.Close()

	val, err := c.QueryWorkflow(context.Background(), wfID, "", PendingQuestionsQuery)
	if err != nil {
		log.Fatalf("querying: %v", err)
	}
	var list []PendingQuestion
	if err := val.Get(&list); err != nil {
		log.Fatalf("decoding: %v", err)
	}
	if len(list) == 0 {
		fmt.Println("no open questions")
		return
	}
	for _, q := range list {
		fmt.Printf("id:     %s\n", q.ID)
		fmt.Printf("  asks: %s\n", q.Question)
	}
	fmt.Printf("answer with: askuser answer %s <id> '<json>'\n", wfID)
}

func answer(wfID, questionID, answerJSON string) {
	c := newClient()
	defer c.Close()

	handle, err := c.UpdateWorkflow(context.Background(), client.UpdateWorkflowOptions{
		WorkflowID:   wfID,
		UpdateName:   AnswerQuestionUpdate,
		WaitForStage: client.WorkflowUpdateStageCompleted,
		Args:         []any{AnswerRequest{ID: questionID, Answer: json.RawMessage(answerJSON)}},
	})
	if err != nil {
		log.Fatalf("sending answer: %v", err)
	}
	if err := handle.Get(context.Background(), nil); err != nil {
		log.Fatalf("answer rejected: %v", err)
	}
	fmt.Printf("answered %s; get the result with: askuser result %s\n", questionID, wfID)
}

func result(wfID string) {
	c := newClient()
	defer c.Close()

	var out string
	if err := c.GetWorkflow(context.Background(), wfID, "").Get(context.Background(), &out); err != nil {
		log.Fatalf("getting result: %v", err)
	}
	fmt.Println(out)
}

func newClient() client.Client {
	c, err := client.Dial(client.Options{HostPort: *flagAddress})
	if err != nil {
		log.Fatalf("connecting to Temporal at %s: %v\n\nIs it running? Try: temporal server start-dev",
			*flagAddress, err)
	}
	return c
}

func usage() {
	fmt.Fprint(os.Stderr, `askuser — free-form human input as a workflow tool

Usage: askuser <command> [flags]

  worker                                       start the worker
  ask "<question>"                             start a run
  questions <workflow-id>                      list open questions
  answer <workflow-id> <question-id> '<json>'  answer a question
  result <workflow-id>                         print the final answer

Flags (after the command):
`)
	flag.PrintDefaults()
}
