# askuser — free-form human input as a workflow tool

The SDK ships a batteries-included **approval** surface (`agent.ApprovalClient`,
`tool.RequiresApproval`) — a human decides yes/no on a tool call. This sample
shows the other half: **free-form human input**, where the model asks the user a
question mid-run and consumes their typed (JSON) answer. It is **not** an SDK
feature — it is built here from the primitives the SDK already provides, so you
can copy and adapt it. The answer is validated as JSON; a real desk would also
enforce a per-question schema.

## The pattern

An `ask_user` workflow tool (`tool.New`) that, when the model calls it:

1. records the question under a run-scoped id,
2. parks the run on a durable `workflow.AwaitWithTimeout`,
3. returns the human's answer to the model as the tool result.

A human reaches the parked run through two handlers the workflow registers — the
same shape agent approval uses, retyped for a JSON payload instead of a bool:

- a **Query** (`pending_questions`) lists what the agent is asking, and
- an **Update** (`answer_question`) delivers a typed, validated answer.

The per-run question state lives on a `questionDesk` placed on the workflow
context, so the package-level tool resolves the desk for the run it runs in.
Because it all runs in workflow code, the wait is durable: the run survives worker
restarts and can park for as long as the timeout allows, costing nothing while
idle.

## Getting the model to actually call it

The wiring is the easy part; the reliability lever is the **prompt and tool
description**, not the schema. Left to a soft prompt, a model tends to ask its
question in plain prose — which, in a headless workflow, just becomes the run's
output and reaches no one. Two things fix that, and this example does both:

- the **tool description** says the tool is the *only* way to reach the user, and
- the **system prompt** is imperative: prose replies are discarded, so the model
  *must* call `ask_user` for anything only the user can provide, and never phrase
  a question as its answer.

With those, a mid-size local model calls the tool on genuine need-input requests
essentially every time; with a vague prompt it rarely does. (It still, correctly,
answers on its own when the request needs no user input.)

## Run it

```sh
temporal server start-dev                              # terminal 1

export OPENAI_API_KEY=sk-...                           # terminal 2
go run ./example/askuser worker

go run ./example/askuser ask "book me a flight"        # terminal 3 -> prints a workflow ID
go run ./example/askuser questions <workflow-id>       # what is the agent asking?
go run ./example/askuser answer <workflow-id> q-1 '{"city":"Ghent"}'
go run ./example/askuser result <workflow-id>          # the final answer
```

Against an OpenAI-compatible endpoint instead:
`go run ./example/askuser worker -base-url http://localhost:8000/v1 -model my-model`.
