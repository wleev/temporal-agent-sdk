package main

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.temporal.io/sdk/client"
	"go.temporal.io/sdk/testsuite"
	"go.temporal.io/sdk/worker"

	"github.com/wleev/temporal-agent-sdk/agent"
	"github.com/wleev/temporal-agent-sdk/agenttest"
	"github.com/wleev/temporal-agent-sdk/mcp"
	"github.com/wleev/temporal-agent-sdk/model"
)

// fakeMarineClient is a stand-in MCP server: it advertises get_marine_weather and
// returns canned marine JSON, so the whole agent — MCP tool + local tools — runs
// offline. The real worker spawns `npx weatherapi-mcp` instead.
type fakeMarineClient struct{ called string }

func (c *fakeMarineClient) ListTools(context.Context) ([]*model.Tool, error) {
	return []*model.Tool{model.NewTool(
		"get_marine_weather",
		"Marine forecast: wave height, swell, wind.",
		[]byte(`{"type":"object","properties":{"q":{"type":"string"},"days":{"type":"integer"}},"required":["q"]}`),
	)}, nil
}

func (c *fakeMarineClient) CallTool(_ context.Context, name string, args json.RawMessage) (*model.CallToolResult, error) {
	c.called = name
	return model.TextResult(`{"location":{"name":"Ericeira"},"forecast":{"forecastday":[{"date":"2026-07-21",` +
		`"hour":[{"time":"2026-07-21 09:00","sig_ht_mt":1.4,"swell_ht_mt":1.1,"swell_dir_16_point":"WSW",` +
		`"swell_period_secs":8.2,"wind_kph":18.0,"condition":{"text":"Sunny"}}]}]}}`), nil
}

func (c *fakeMarineClient) Close() error { return nil }

// startWorker spins up a real dev server and a worker wired like the production
// worker, except the model and the MCP server are faked. The store keeps its
// sessions and spots files alongside dataPath.
func startWorker(t *testing.T, fake *agenttest.FakeProvider, marine *fakeMarineClient, dataPath string) client.Client {
	t.Helper()
	if testing.Short() {
		t.Skip("skipping surf e2e: needs a dev server binary (-short)")
	}

	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Minute)
	defer cancel()
	srv, err := testsuite.StartDevServer(ctx, testsuite.DevServerOptions{
		ClientOptions: &client.Options{}, LogLevel: "error",
	})
	if err != nil {
		t.Skipf("skipping: could not start dev server: %v", err)
	}
	t.Cleanup(func() { _ = srv.Stop() })
	c := srv.Client()

	w := worker.New(c, taskQueue, worker.Options{})

	acts, err := model.NewActivities(fake)
	require.NoError(t, err)
	acts.Register(w)

	// Fake MCP server under the same name the agent lists.
	mcpActs := mcp.NewActivities()
	require.NoError(t, mcpActs.Register(mcpServer, func(context.Context) (mcp.Client, error) { return marine, nil }))
	mcpActs.RegisterWith(w)

	registerStore(w, &Store{Path: dataPath, SpotsPath: dataPath + ".spots.json"})
	agent.RegisterWorkflows(w, buildAgent("test-model"))
	w.RegisterWorkflow(SurfWorkflow)

	require.NoError(t, w.Start())
	t.Cleanup(w.Stop)
	return c
}

// The full flow: the agent checks the marine forecast (MCP), then logs a session
// (local activity), which must land in the JSON file.
func TestSurf_ForecastThenLog(t *testing.T) {
	dataPath := filepath.Join(t.TempDir(), "sessions.json")

	fake := agenttest.NewFakeProvider(
		agenttest.CallsTool("get_marine_weather", `{"q":"38.99,-9.42","days":3}`),
		agenttest.CallsTool("log_surf_session",
			`{"spot":"Ericeira","date":"2026-07-21","wave_height_m":1.4,"swell_period_s":8.2,"wind_kph":18,"rating":8,"notes":"fun"}`),
		agenttest.Says("Logged your Ericeira session — solid 8/10."),
	)
	marine := &fakeMarineClient{}
	c := startWorker(t, fake, marine, dataPath)

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	run, err := c.ExecuteWorkflow(ctx, client.StartWorkflowOptions{TaskQueue: taskQueue},
		SurfWorkflow, "surfed Ericeira today, it was fun, check what it was doing and log it 8/10")
	require.NoError(t, err)

	var answer string
	require.NoError(t, run.Get(ctx, &answer))
	assert.Contains(t, answer, "Ericeira")

	// The marine MCP tool was actually called.
	assert.Equal(t, "get_marine_weather", marine.called)

	// The session was persisted by the local activity tool.
	sessions := readSessions(t, dataPath)
	if assert.Len(t, sessions, 1) {
		assert.Equal(t, "Ericeira", sessions[0].Spot)
		assert.Equal(t, 8, sessions[0].Rating)
	}

	// The marine forecast reached the model on the second turn (tool result).
	second := fake.Calls()[1]
	last := second.Messages[len(second.Messages)-1]
	assert.Contains(t, last.ToolResultText(), "sig_ht_mt")
}

// Trends: pre-seed the log, the agent calls surf_trends, and the computed summary
// reaches the model.
func TestSurf_Trends(t *testing.T) {
	dataPath := filepath.Join(t.TempDir(), "sessions.json")
	seed := []SurfSession{
		{Spot: "Ericeira", Date: "2026-07-01", WaveHeightM: 1.2, SwellPeriodS: 8, WindKph: 15, Rating: 9, Notes: "epic"},
		{Spot: "Ericeira", Date: "2026-07-02", WaveHeightM: 0.6, SwellPeriodS: 6, WindKph: 25, Rating: 4, Notes: "small"},
		{Spot: "Ericeira", Date: "2026-07-03", WaveHeightM: 1.3, SwellPeriodS: 9, WindKph: 12, Rating: 8, Notes: "clean"},
	}
	writeSessions(t, dataPath, seed)

	fake := agenttest.NewFakeProvider(
		agenttest.CallsTool("surf_trends", `{"spot":"Ericeira"}`),
		agenttest.Says("You surf best in chest-high waves with a longer swell period."),
	)
	c := startWorker(t, fake, &fakeMarineClient{}, dataPath)

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	run, err := c.ExecuteWorkflow(ctx, client.StartWorkflowOptions{TaskQueue: taskQueue},
		SurfWorkflow, "what conditions do I surf best at Ericeira?")
	require.NoError(t, err)

	var answer string
	require.NoError(t, run.Get(ctx, &answer))
	assert.NotEmpty(t, answer)

	// The computed trends reached the model: it saw the average rating and the
	// best-conditions summary.
	second := fake.Calls()[1]
	last := second.Messages[len(second.Messages)-1]
	assert.Contains(t, last.ToolResultText(), "best_conditions")
	assert.Contains(t, last.ToolResultText(), "avg_rating")
}

// The user adds a new spot; it must persist and then be resolvable so a forecast
// can use its coordinates.
func TestSurf_AddThenUseSpot(t *testing.T) {
	dataPath := filepath.Join(t.TempDir(), "sessions.json")
	spotsPath := dataPath + ".spots.json"

	fake := agenttest.NewFakeProvider(
		agenttest.CallsTool("add_surf_spot",
			`{"name":"Supertubos","coordinates":"39.36,-9.37","notes":"heavy beach break"}`),
		agenttest.CallsTool("list_surf_spots", `{}`),
		agenttest.CallsTool("get_marine_weather", `{"q":"39.36,-9.37","days":3}`),
		agenttest.Says("Saved Supertubos and pulled its forecast."),
	)
	marine := &fakeMarineClient{}
	c := startWorker(t, fake, marine, dataPath)

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	run, err := c.ExecuteWorkflow(ctx, client.StartWorkflowOptions{TaskQueue: taskQueue},
		SurfWorkflow, "add Supertubos at 39.36,-9.37 and check the surf there")
	require.NoError(t, err)
	require.NoError(t, run.Get(ctx, new(string)))

	// The spot was persisted to the spots file.
	data, err := os.ReadFile(spotsPath)
	require.NoError(t, err)
	var spots []SurfSpot
	require.NoError(t, json.Unmarshal(data, &spots))
	if assert.Len(t, spots, 1) {
		assert.Equal(t, "Supertubos", spots[0].Name)
	}

	// list_surf_spots returned the new spot merged with the built-in defaults.
	listTurn := fake.Calls()[2] // the turn after list_surf_spots ran
	last := listTurn.Messages[len(listTurn.Messages)-1]
	assert.Contains(t, last.ToolResultText(), "Supertubos")
	assert.Contains(t, last.ToolResultText(), "Ericeira") // a default is still present

	// The forecast used the coordinates.
	assert.Equal(t, "get_marine_weather", marine.called)
}

// A bad coordinate string is a tool error the model can recover from, not a
// workflow failure.
func TestSurf_AddSpotRejectsBadCoordinates(t *testing.T) {
	dataPath := filepath.Join(t.TempDir(), "sessions.json")

	fake := agenttest.NewFakeProvider(
		agenttest.CallsTool("add_surf_spot", `{"name":"Nowhere","coordinates":"not-coords","notes":""}`),
		agenttest.Says("That doesn't look like coordinates — what's the lat,lon?"),
	)
	c := startWorker(t, fake, &fakeMarineClient{}, dataPath)

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	run, err := c.ExecuteWorkflow(ctx, client.StartWorkflowOptions{TaskQueue: taskQueue},
		SurfWorkflow, "add a spot called Nowhere")
	require.NoError(t, err)
	require.NoError(t, run.Get(ctx, new(string)), "a validation error must not fail the workflow")

	// The validation error reached the model as a tool result.
	second := fake.Calls()[1]
	last := second.Messages[len(second.Messages)-1]
	assert.Contains(t, last.ToolResultText(), "lat,lon")
}

func TestValidateCoordinates(t *testing.T) {
	assert.NoError(t, validateCoordinates("39.36,-9.37"))
	assert.NoError(t, validateCoordinates(" 39.36 , -9.37 "))
	assert.Error(t, validateCoordinates("39.36"))
	assert.Error(t, validateCoordinates("abc,def"))
	assert.Error(t, validateCoordinates("200,0"))  // lat out of range
	assert.Error(t, validateCoordinates("0,-200")) // lon out of range
}

// computeTrends is pure; unit-test its bucketing directly.
func TestComputeTrends(t *testing.T) {
	tr := computeTrends([]SurfSession{
		{WaveHeightM: 1.2, SwellPeriodS: 9, Rating: 9},
		{WaveHeightM: 1.3, SwellPeriodS: 9, Rating: 7},
		{WaveHeightM: 0.5, SwellPeriodS: 6, Rating: 3},
	})
	assert.Equal(t, 3, tr.Sessions)
	assert.InDelta(t, 6.3, tr.AvgRating, 0.1)
	assert.Contains(t, tr.Best, "1-1.5m") // the higher-rated bucket
}

func readSessions(t *testing.T, path string) []SurfSession {
	t.Helper()
	data, err := os.ReadFile(path)
	require.NoError(t, err)
	var s []SurfSession
	require.NoError(t, json.Unmarshal(data, &s))
	return s
}

func writeSessions(t *testing.T, path string, s []SurfSession) {
	t.Helper()
	data, err := json.MarshalIndent(s, "", "  ")
	require.NoError(t, err)
	require.NoError(t, os.WriteFile(path, data, 0o644))
}
