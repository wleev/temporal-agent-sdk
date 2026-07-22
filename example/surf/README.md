# surf — real MCP + local tools

A surf-coach agent that combines a **real MCP server** (the official
[WeatherAPI.com marine forecast](https://github.com/weatherapicom/weatherapi-mcp))
with **local, file-backed tools** that log surf sessions and compute trends.

The domain is fictional; the wiring is real:

- **Marine forecast** — `get_marine_weather` comes from an external MCP server
  the worker spawns over stdio (`npx -y weatherapi-mcp`). The library's stateless
  MCP integration connects, calls the tool, and disconnects per invocation.
- **Surf log** — `log_surf_session` and `surf_trends` are local
  [`tool.Activity`](../../tool) tools. They persist to a JSON file, so they run in
  an activity (file I/O is not allowed in workflow code) and get retries and
  timeouts for free.
- **Spots** — `add_surf_spot` and `list_surf_spots` let the user teach the agent
  new locations. A spot is a name plus `lat,lon`; the agent looks one up to
  resolve a name to coordinates before calling the marine forecast, and saves new
  ones the user mentions. Bad coordinates come back as a non-retryable error the
  model relays to the user (rather than Temporal retrying deterministic-invalid
  input forever).

The agent ties them together: check the surf, record how a session went, and
learn which wave heights and swell periods you surf best in.

## Run it

```sh
temporal server start-dev                                   # terminal 1

export OPENAI_API_KEY=sk-...                                # terminal 2
export WEATHERAPI_KEY=...        # free at https://www.weatherapi.com/signup.aspx
go run ./example/surf worker                                # needs npx on PATH

# terminal 3
go run ./example/surf ask "add Supertubos at 39.36,-9.37"
go run ./example/surf ask "what's the surf at Supertubos this week?"
go run ./example/surf ask "surfed Supertubos today, ~1.3m and clean, 8 out of 10"
go run ./example/surf ask "what conditions do I surf best in?"
```

The marine tool needs coordinates (`lat,lon`); the agent knows a few spots
(Ericeira, Nazaré, Peniche, Trestles) and passes their coordinates. Wave and
swell data work on the free WeatherAPI tier; **tides and water temperature need
a Pro+ plan**.

Against an OpenAI-compatible endpoint (vLLM, gateway) instead of OpenAI:

```sh
go run ./example/surf worker -o OPENAI_BASE_URL=http://localhost:8000/v1  # or export it
```

Sessions are stored in `surf-sessions.json` (override with `SURF_DATA`). Swap the
`Store` for SQLite or a database and nothing else changes — the tools stay the
same.

## Tests

`surf_test.go` runs the whole agent offline: a fake model provider and a fake MCP
client stand in for OpenAI and the weather server, so `go test ./example/surf`
exercises the MCP tool and both local tools end to end against a real dev server
without a network or API keys. `go test -short` skips the dev-server tests.

## The simpler example

[`../stub`](../stub) is the minimal version — a support agent showing tools,
approval, and sub-agents without an external MCP server.
