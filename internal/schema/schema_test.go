package schema

import (
	"encoding/json"
	"strings"
	"testing"
)

type weatherIn struct {
	City string `json:"city" jsonschema_description:"City name, e.g. Ghent"`
	Unit string `json:"unit" jsonschema:"enum=celsius,enum=fahrenheit"`
	Days int    `json:"days" jsonschema_description:"Forecast horizon"`
}

type nestedIn struct {
	Inner weatherIn `json:"inner"`
	Tags  []string  `json:"tags"`
}

type cyclicIn struct {
	Name  string    `json:"name"`
	Child *cyclicIn `json:"child"`
}

type cyclicViaSlice struct {
	Children []cyclicViaSlice `json:"children"`
}

type omitemptyIn struct {
	A string  `json:"a"`
	B string  `json:"b,omitempty"`
	C *int    `json:"c,omitempty"`
	D nestedO `json:"d,omitempty"`
}

type nestedO struct {
	E string `json:"e,omitempty"`
}

// OpenAI strict mode requires every property in required, including fields
// tagged omitempty and fields of nested objects. invopop omits omitempty fields
// from required, so the schema layer must force them back — recursively.
func TestFor_AllFieldsRequiredDespiteOmitempty(t *testing.T) {
	got, err := For[omitemptyIn]()
	if err != nil {
		t.Fatalf("For: %v", err)
	}

	var s struct {
		Required   []string `json:"required"`
		Properties map[string]struct {
			Required []string `json:"required"`
		} `json:"properties"`
	}
	if err := json.Unmarshal(got, &s); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	require := func(got []string, want ...string) {
		t.Helper()
		set := map[string]bool{}
		for _, k := range got {
			set[k] = true
		}
		for _, w := range want {
			if !set[w] {
				t.Errorf("required %v is missing %q", got, w)
			}
		}
		if len(got) != len(want) {
			t.Errorf("required = %v, want exactly %v", got, want)
		}
	}

	// Field order must be preserved for determinism.
	require(s.Required, "a", "b", "c", "d")
	require(s.Properties["d"].Required, "e")
}

// The forced-required pass must not disturb byte stability.
func TestFor_ForcedRequiredIsDeterministic(t *testing.T) {
	first, err := For[omitemptyIn]()
	if err != nil {
		t.Fatalf("For: %v", err)
	}
	for range 50 {
		next, err := For[omitemptyIn]()
		if err != nil {
			t.Fatal(err)
		}
		if string(next) != string(first) {
			t.Fatalf("not byte-stable:\n%s\n%s", first, next)
		}
	}
}

// The exact bytes matter: this is what the model sees, and OpenAI strict mode
// rejects $schema/$ref and requires additionalProperties:false with every
// property in required.
func TestFor_OpenAIStrictShape(t *testing.T) {
	got, err := For[weatherIn]()
	if err != nil {
		t.Fatalf("For: %v", err)
	}

	const want = `{"properties":{"city":{"type":"string","description":"City name, e.g. Ghent"},` +
		`"unit":{"type":"string","enum":["celsius","fahrenheit"]},` +
		`"days":{"type":"integer","description":"Forecast horizon"}},` +
		`"additionalProperties":false,"type":"object","required":["city","unit","days"]}`

	if string(got) != want {
		t.Errorf("schema mismatch\n got: %s\nwant: %s", got, want)
	}
}

// Guards the four Reflector flags and the manual s.Version reset. Each of these
// has bitten us or would silently break strict mode if a flag regressed.
func TestFor_ForbiddenKeywords(t *testing.T) {
	got, err := For[nestedIn]()
	if err != nil {
		t.Fatalf("For: %v", err)
	}
	for _, kw := range []string{"$schema", "$ref", "$defs", "$id", "definitions"} {
		if strings.Contains(string(got), kw) {
			t.Errorf("schema contains %q, which OpenAI strict mode rejects: %s", kw, got)
		}
	}
}

func TestFor_NestedStructInlined(t *testing.T) {
	got, err := For[nestedIn]()
	if err != nil {
		t.Fatalf("For: %v", err)
	}
	var s struct {
		Properties map[string]struct {
			Type       string          `json:"type"`
			Properties json.RawMessage `json:"properties"`
		} `json:"properties"`
	}
	if err := json.Unmarshal(got, &s); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if s.Properties["inner"].Properties == nil {
		t.Errorf("nested struct was not inlined: %s", got)
	}
	if s.Properties["tags"].Type != "array" {
		t.Errorf("slice field: got type %q, want array", s.Properties["tags"].Type)
	}
}

// Reflection output feeds workflow history via the tool schema, so unstable
// serialization would surface as nondeterministic replay.
func TestFor_Deterministic(t *testing.T) {
	first, err := For[nestedIn]()
	if err != nil {
		t.Fatalf("For: %v", err)
	}
	for i := range 50 {
		next, err := For[nestedIn]()
		if err != nil {
			t.Fatalf("For (iter %d): %v", i, err)
		}
		if string(next) != string(first) {
			t.Fatalf("schema not byte-stable across calls\nfirst: %s\n iter %d: %s", first, i, next)
		}
	}
}

// Without $defs the reflector has no cycle base case, so these would overflow
// the stack rather than return.
func TestFor_RejectsRecursiveTypes(t *testing.T) {
	for _, tc := range []struct {
		name string
		fn   func() (json.RawMessage, error)
	}{
		{"via pointer", For[cyclicIn]},
		{"via slice", For[cyclicViaSlice]},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := tc.fn(); err == nil {
				t.Fatal("expected error for recursive type, got nil")
			} else if !strings.Contains(err.Error(), "recursive type") {
				t.Errorf("unexpected error: %v", err)
			}
		})
	}
}

type anyFieldIn struct {
	Question string          `json:"question"`
	Answer   json.RawMessage `json:"answer"`
}

type nestedAnyIn struct {
	Inner struct {
		Blob any `json:"blob"`
	} `json:"inner"`
}

type sliceAnyIn struct {
	Items []any `json:"items"`
}

// For returns an error naming the offending field when a struct field or slice
// element has type any, interface{}, or json.RawMessage.
func TestFor_RejectsUntypedProperties(t *testing.T) {
	for _, tc := range []struct {
		name string
		fn   func() (json.RawMessage, error)
		want string
	}{
		{"json.RawMessage", For[anyFieldIn], "Answer"},
		{"nested any", For[nestedAnyIn], "Inner.Blob"},
		{"slice of any", For[sliceAnyIn], "Items.[]"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, err := tc.fn()
			if err == nil {
				t.Fatal("expected an error for an untyped field, got nil")
			}
			if !strings.Contains(err.Error(), "no type") ||
				!strings.Contains(err.Error(), tc.want) {
				t.Errorf("error should name the untyped field %q: %v", tc.want, err)
			}
		})
	}
}

// A fully typed struct passes the guard.
func TestFor_AcceptsTypedProperties(t *testing.T) {
	if _, err := For[nestedIn](); err != nil {
		t.Fatalf("a fully typed struct must pass: %v", err)
	}
}

type mapFieldsIn struct {
	Free  map[string]any    `json:"free"`
	Typed map[string]string `json:"typed"`
}

// A map value is a free object, not an untyped leaf: map[string]any is allowed
// and reflects to an open object.
func TestFor_AllowsMapFreeObject(t *testing.T) {
	got, err := For[mapFieldsIn]()
	if err != nil {
		t.Fatalf("a map field must be allowed as a free object: %v", err)
	}
	if !strings.Contains(string(got), `"free":{"type":"object"}`) {
		t.Errorf("map[string]any should reflect to an open object: %s", got)
	}
}

// A top-level untyped type (a tool whose whole argument is any or json.RawMessage)
// is rejected, not just untyped fields.
func TestFor_RejectsUntypedRoot(t *testing.T) {
	if _, err := For[json.RawMessage](); err == nil {
		t.Fatal("expected an error for a json.RawMessage argument type, got nil")
	}
}
