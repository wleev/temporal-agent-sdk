// Command surf is a fuller example: an agent that combines a *real* MCP server
// (the official WeatherAPI.com marine forecast) with local, file-backed tools
// that log surf sessions and compute trends.
//
// The domain is fictional but the wiring is real. The marine forecast comes from
// an external MCP server run over stdio; logging and trend analysis are local
// activity tools that persist to a JSON file. The agent ties them together: check
// the surf, record how a session went, and learn which conditions you surf best
// in.
//
// See the sibling example/stub for the minimal version.
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"sort"
	"strconv"
	"strings"
	"sync"

	"go.temporal.io/sdk/temporal"

	"github.com/wleev/temporal-agent-sdk/agent"
	"github.com/wleev/temporal-agent-sdk/tool"
)

// Registered activity names for the local, file-backed tools. The agent's tools
// reference these names, so the agent definition does not depend on the concrete
// store (which is created on the worker with a data-file path).
const (
	logActivityName       = "surf_log_session"
	trendsActivityName    = "surf_trends"
	addSpotActivityName   = "surf_add_spot"
	listSpotsActivityName = "surf_list_spots"

	// mcpServer is the name the agent lists in MCPServers; the worker registers a
	// factory under it that spawns the WeatherAPI MCP server.
	mcpServer = "weather"
)

// SurfSession is one logged surf. Its fields are exactly what the agent fills in
// when it records a session — marine conditions plus the surfer's own rating —
// and the JSON Schema the model sees is reflected from this struct.
type SurfSession struct {
	Spot         string  `json:"spot" jsonschema_description:"Surf spot name, e.g. Ericeira"`
	Date         string  `json:"date" jsonschema_description:"Date surfed, YYYY-MM-DD"`
	WaveHeightM  float64 `json:"wave_height_m" jsonschema_description:"Significant wave height in metres"`
	SwellPeriodS float64 `json:"swell_period_s" jsonschema_description:"Swell period in seconds"`
	WindKph      float64 `json:"wind_kph" jsonschema_description:"Wind speed in kph"`
	Rating       int     `json:"rating" jsonschema_description:"How the session went, 1 (poor) to 10 (epic)"`
	Notes        string  `json:"notes" jsonschema_description:"Freeform notes about the session"`
}

// LogResult confirms a stored session.
type LogResult struct {
	Stored bool `json:"stored"`
	Total  int  `json:"total_sessions"`
}

// TrendsInput filters the trend analysis. An empty Spot covers every spot.
type TrendsInput struct {
	Spot string `json:"spot" jsonschema_description:"Limit to one spot, or empty for all spots"`
}

// Trends summarizes logged sessions: overall rating and how the rating varies
// with conditions, so the agent can say which conditions surf best.
type Trends struct {
	Sessions     int      `json:"sessions"`
	AvgRating    float64  `json:"avg_rating"`
	Best         string   `json:"best_conditions"`
	ByWaveHeight []Bucket `json:"by_wave_height"`
	BySwell      []Bucket `json:"by_swell_period"`
}

// Bucket is the average rating for one range of a condition.
type Bucket struct {
	Range     string  `json:"range"`
	Sessions  int     `json:"sessions"`
	AvgRating float64 `json:"avg_rating"`
}

// SurfSpot is a named location with the coordinates the marine forecast needs.
// The agent learns spots from the user and reads them back to resolve a name to
// coordinates before calling get_marine_weather.
type SurfSpot struct {
	Name        string `json:"name" jsonschema_description:"Spot name, e.g. Supertubos"`
	Coordinates string `json:"coordinates" jsonschema_description:"Latitude,longitude, e.g. 39.36,-9.37"`
	Notes       string `json:"notes" jsonschema_description:"Optional notes about the spot, e.g. best swell direction"`
}

// AddSpotResult confirms a saved spot.
type AddSpotResult struct {
	Added bool `json:"added"`
	Total int  `json:"total_spots"`
}

// SpotList is the known spots — the built-in defaults plus any the user added.
type SpotList struct {
	Spots []SurfSpot `json:"spots"`
}

// defaultSpots are well-known spots the agent knows out of the box, so the
// example works before the user adds any.
func defaultSpots() []SurfSpot {
	return []SurfSpot{
		{Name: "Ericeira", Coordinates: "38.99,-9.42", Notes: "Portugal"},
		{Name: "Nazaré", Coordinates: "39.60,-9.07", Notes: "Portugal — big-wave"},
		{Name: "Trestles", Coordinates: "33.38,-117.59", Notes: "California"},
	}
}

// Store persists sessions and spots to JSON files.
//
// The mutex guards concurrent activity executions within one worker; the JSON
// files are not safe across multiple workers — fine for an example, and the point
// where a real app would swap in SQLite or a database. Because it runs as a
// Temporal activity it may do this file I/O freely; a workflow tool could not.
type Store struct {
	mu        sync.Mutex
	Path      string // sessions file
	SpotsPath string // spots file
}

// Log appends a session. It is registered as an activity.
func (s *Store) Log(_ context.Context, in SurfSession) (LogResult, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	sessions, err := s.read()
	if err != nil {
		return LogResult{}, err
	}
	sessions = append(sessions, in)
	if err := s.write(sessions); err != nil {
		return LogResult{}, err
	}
	return LogResult{Stored: true, Total: len(sessions)}, nil
}

// Trends computes the summary. It is registered as an activity.
func (s *Store) Trends(_ context.Context, in TrendsInput) (Trends, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	all, err := s.read()
	if err != nil {
		return Trends{}, err
	}
	sessions := all[:0:0]
	for _, x := range all {
		if in.Spot == "" || x.Spot == in.Spot {
			sessions = append(sessions, x)
		}
	}
	return computeTrends(sessions), nil
}

// AddSpot validates the coordinates and saves a new spot. It is registered as an
// activity.
func (s *Store) AddSpot(_ context.Context, in SurfSpot) (AddSpotResult, error) {
	// Validation failures are deterministic, so mark them non-retryable — a plain
	// error would make Temporal retry the same bad input forever. Non-retryable
	// means the agent loop hands the message straight to the model, which can ask
	// the user to correct it. Genuine file I/O errors below stay retryable.
	if in.Name == "" {
		return AddSpotResult{}, temporal.NewNonRetryableApplicationError(
			"spot name must not be empty", "SurfValidation", nil)
	}
	if err := validateCoordinates(in.Coordinates); err != nil {
		return AddSpotResult{}, temporal.NewNonRetryableApplicationError(err.Error(), "SurfValidation", err)
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	spots, err := s.readSpots()
	if err != nil {
		return AddSpotResult{}, err
	}
	// Replace an existing spot of the same name rather than duplicate it.
	replaced := false
	for i := range spots {
		if strings.EqualFold(spots[i].Name, in.Name) {
			spots[i] = in
			replaced = true
			break
		}
	}
	if !replaced {
		spots = append(spots, in)
	}
	if err := s.writeSpots(spots); err != nil {
		return AddSpotResult{}, err
	}
	return AddSpotResult{Added: true, Total: len(spots)}, nil
}

// ListSpots returns the known spots — built-in defaults overlaid with any the
// user saved (a saved spot of the same name wins). It is registered as an
// activity and takes no arguments.
func (s *Store) ListSpots(_ context.Context, _ struct{}) (SpotList, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	saved, err := s.readSpots()
	if err != nil {
		return SpotList{}, err
	}

	// Defaults first, then overlay saved by name (case-insensitive).
	byName := map[string]SurfSpot{}
	order := []string{}
	add := func(sp SurfSpot) {
		k := strings.ToLower(sp.Name)
		if _, seen := byName[k]; !seen {
			order = append(order, k)
		}
		byName[k] = sp
	}
	for _, sp := range defaultSpots() {
		add(sp)
	}
	for _, sp := range saved {
		add(sp)
	}

	out := make([]SurfSpot, 0, len(order))
	for _, k := range order {
		out = append(out, byName[k])
	}
	return SpotList{Spots: out}, nil
}

// validateCoordinates checks a "lat,lon" string is well-formed and in range.
func validateCoordinates(s string) error {
	parts := strings.Split(s, ",")
	if len(parts) != 2 {
		return fmt.Errorf("coordinates %q must be \"lat,lon\"", s)
	}
	lat, err := strconv.ParseFloat(strings.TrimSpace(parts[0]), 64)
	if err != nil || lat < -90 || lat > 90 {
		return fmt.Errorf("latitude in %q must be a number between -90 and 90", s)
	}
	lon, err := strconv.ParseFloat(strings.TrimSpace(parts[1]), 64)
	if err != nil || lon < -180 || lon > 180 {
		return fmt.Errorf("longitude in %q must be a number between -180 and 180", s)
	}
	return nil
}

func (s *Store) readSpots() ([]SurfSpot, error) {
	if s.SpotsPath == "" {
		return nil, nil
	}
	data, err := os.ReadFile(s.SpotsPath)
	if os.IsNotExist(err) || len(data) == 0 {
		return nil, nil
	}
	if err != nil {
		return nil, fmt.Errorf("reading %s: %w", s.SpotsPath, err)
	}
	var spots []SurfSpot
	if err := json.Unmarshal(data, &spots); err != nil {
		return nil, fmt.Errorf("parsing %s: %w", s.SpotsPath, err)
	}
	return spots, nil
}

func (s *Store) writeSpots(spots []SurfSpot) error {
	if s.SpotsPath == "" {
		return fmt.Errorf("no spots file configured")
	}
	data, err := json.MarshalIndent(spots, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(s.SpotsPath, data, 0o644)
}

func (s *Store) read() ([]SurfSession, error) {
	data, err := os.ReadFile(s.Path)
	if os.IsNotExist(err) {
		return nil, nil
	}
	if err != nil {
		return nil, fmt.Errorf("reading %s: %w", s.Path, err)
	}
	if len(data) == 0 {
		return nil, nil
	}
	var sessions []SurfSession
	if err := json.Unmarshal(data, &sessions); err != nil {
		return nil, fmt.Errorf("parsing %s: %w", s.Path, err)
	}
	return sessions, nil
}

func (s *Store) write(sessions []SurfSession) error {
	data, err := json.MarshalIndent(sessions, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(s.Path, data, 0o644)
}

// computeTrends buckets sessions by wave height and swell period and averages the
// rating in each, then names the best-rated conditions.
func computeTrends(sessions []SurfSession) Trends {
	t := Trends{Sessions: len(sessions)}
	if len(sessions) == 0 {
		t.Best = "no sessions logged yet"
		return t
	}

	var total int
	for _, s := range sessions {
		total += s.Rating
	}
	t.AvgRating = round1(float64(total) / float64(len(sessions)))

	t.ByWaveHeight = bucketize(sessions, waveBucket, []string{"<1m", "1-1.5m", "1.5-2m", "2m+"})
	t.BySwell = bucketize(sessions, swellBucket, []string{"<8s", "8-11s", "11s+"})

	// Best conditions = the wave-height bucket with the highest average rating.
	best := ""
	bestRating := -1.0
	for _, b := range t.ByWaveHeight {
		if b.Sessions > 0 && b.AvgRating > bestRating {
			bestRating, best = b.AvgRating, b.Range
		}
	}
	t.Best = fmt.Sprintf("waves around %s (avg rating %.1f)", best, bestRating)
	return t
}

func waveBucket(s SurfSession) int {
	switch {
	case s.WaveHeightM < 1:
		return 0
	case s.WaveHeightM < 1.5:
		return 1
	case s.WaveHeightM < 2:
		return 2
	default:
		return 3
	}
}

func swellBucket(s SurfSession) int {
	switch {
	case s.SwellPeriodS < 8:
		return 0
	case s.SwellPeriodS < 11:
		return 1
	default:
		return 2
	}
}

func bucketize(sessions []SurfSession, bucket func(SurfSession) int, labels []string) []Bucket {
	sum := make([]int, len(labels))
	count := make([]int, len(labels))
	for _, s := range sessions {
		i := bucket(s)
		sum[i] += s.Rating
		count[i]++
	}
	var out []Bucket
	for i, label := range labels {
		if count[i] == 0 {
			continue
		}
		out = append(out, Bucket{
			Range:     label,
			Sessions:  count[i],
			AvgRating: round1(float64(sum[i]) / float64(count[i])),
		})
	}
	sort.Slice(out, func(a, b int) bool { return out[a].AvgRating > out[b].AvgRating })
	return out
}

func round1(f float64) float64 { return math.Round(f*10) / 10 }

// localTools are the file-backed tools, referencing the activities by name so the
// agent definition stays independent of the concrete [Store].
func localTools() []tool.Tool {
	return []tool.Tool{
		tool.Must(tool.Activity[SurfSession, LogResult](
			"log_surf_session",
			"Record a completed surf session: the spot, date, the marine conditions "+
				"(wave height, swell period, wind), a rating from 1 to 10, and notes.",
			logActivityName)),
		tool.Must(tool.Activity[TrendsInput, Trends](
			"surf_trends",
			"Summarize logged sessions into trends — which wave heights and swell "+
				"periods produce the best-rated sessions. Empty spot means all spots.",
			trendsActivityName)),
		tool.Must(tool.Activity[SurfSpot, AddSpotResult](
			"add_surf_spot",
			"Save a new surf spot with its coordinates (\"lat,lon\") so future "+
				"forecasts can use it. Call this when the user wants to add a location.",
			addSpotActivityName)),
		tool.Must(tool.Activity[struct{}, SpotList](
			"list_surf_spots",
			"List known surf spots and their coordinates. Use it to resolve a spot "+
				"name to coordinates before calling get_marine_weather.",
			listSpotsActivityName)),
	}
}

// surfAgent is the entry-point agent, set by [buildAgent].
var surfAgent *agent.Agent

// must panics if agent construction failed — startup wiring builds a fixed
// agent, so an error is a programming bug.
func must(a *agent.Agent, err error) *agent.Agent {
	if err != nil {
		panic(err)
	}
	return a
}

// buildAgent constructs the surf-coach agent and a registry holding it. The
// agent uses the WeatherAPI MCP server for marine forecasts and the local tools
// for logging and trends.
func buildAgent(modelName string) *agent.Registry {
	surfAgent = must(agent.NewAgent(
		"surf-coach",
		modelName,
		agent.WithInstructions("You are a surf coach and logbook assistant.\n"+
			"- get_marine_weather needs coordinates as \"lat,lon\". To find a spot's "+
			"coordinates, call list_surf_spots. If the spot isn't listed, ask the user "+
			"for its coordinates and call add_surf_spot to save it, then use it.\n"+
			"- When the user wants to add or remember a new surf location, call "+
			"add_surf_spot with its name and coordinates.\n"+
			"- When the surfer reports how a session went, call log_surf_session with the "+
			"conditions (use the forecast where the surfer is vague) and their 1-10 rating.\n"+
			"- When asked what conditions they surf best in, call surf_trends and explain the pattern.\n"+
			"Be concise and encouraging."),
		agent.WithMCPServers(mcpServer),
		agent.WithTools(localTools()...),
		agent.WithMaxTurns(8),
	))
	return agent.NewRegistry().MustAdd(surfAgent)
}
