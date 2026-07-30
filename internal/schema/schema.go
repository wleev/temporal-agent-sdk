// Package schema derives JSON Schema from Go types for LLM tool definitions.
//
// The Go type is the single source of truth: a tool's argument struct defines
// both the schema sent to the model and the value the handler receives, so the
// two cannot drift.
//
// Schemas are emitted in OpenAI strict function-calling shape: no $schema, no
// $ref/$defs, additionalProperties:false, and every property listed in
// required.
package schema

import (
	"encoding/json"
	"fmt"
	"reflect"
	"strings"

	"github.com/invopop/jsonschema"
)

// reflector produces flat, self-contained schemas. DoNotReference inlines every
// nested type and suppresses $defs/$ref, which OpenAI strict mode rejects.
// AllowAdditionalProperties false emits additionalProperties:false.
func reflector() *jsonschema.Reflector {
	return &jsonschema.Reflector{
		ExpandedStruct:            true,
		DoNotReference:            true,
		Anonymous:                 true,
		AllowAdditionalProperties: false,
	}
}

// For derives the JSON Schema for T.
//
// Call this at tool-construction time, never inside workflow code. The output is
// stable for a given binary but depends on T: reordering or renaming a field
// changes the schema and breaks replay of in-flight workflows, so version tool
// argument types rather than editing them.
//
// It returns an error for a field that reflects to an untyped schema — one of
// type any, interface{}, or json.RawMessage — which providers reject. Map value
// schemas (additionalProperties) are not inspected.
func For[T any]() (json.RawMessage, error) {
	var zero T
	t := reflect.TypeOf(&zero).Elem()

	if err := validateType(t); err != nil {
		return nil, err
	}

	s := reflector().ReflectFromType(t)

	// invopop emits $schema unconditionally and offers no option to disable it.
	s.Version = ""

	// OpenAI strict function calling requires every property to appear in
	// required; optionality is expressed with a nullable type, not by omission.
	// invopop omits fields tagged omitempty, so force the invariant.
	forceAllRequired(s)

	b, err := json.Marshal(s)
	if err != nil {
		return nil, fmt.Errorf("marshal schema for %s: %w", t, err)
	}
	return b, nil
}

// rawMessageType is json.RawMessage: it implements json.Marshaler with no
// JSONSchema method, so invopop reflects it to a schema with no type.
var rawMessageType = reflect.TypeOf(json.RawMessage{})

// untypedLeaf reports whether a type used as a struct field or slice element
// reflects to a schema with no type — any, interface{}, or json.RawMessage. A
// map value is not a leaf: it becomes a free object (additionalProperties).
func untypedLeaf(t reflect.Type) bool {
	for t.Kind() == reflect.Pointer {
		t = t.Elem()
	}
	return t.Kind() == reflect.Interface || t == rawMessageType
}

func untypedError(path []string, t reflect.Type) error {
	return fmt.Errorf("field %s has type %s, which reflects to a schema with no type; any, "+
		"interface{}, and json.RawMessage are not valid tool schema fields — use a concrete type",
		strings.Join(path, "."), t)
}

// forceAllRequired makes every property of every object schema required,
// recursively. Required is rebuilt from the ordered property map in declaration
// order, so the output stays byte-stable across runs.
func forceAllRequired(s *jsonschema.Schema) {
	if s == nil {
		return
	}
	if s.Properties != nil {
		required := make([]string, 0, s.Properties.Len())
		for pair := s.Properties.Oldest(); pair != nil; pair = pair.Next() {
			required = append(required, pair.Key)
			forceAllRequired(pair.Value)
		}
		s.Required = required
	}
	// Arrays reintroduce object schemas through their element type.
	forceAllRequired(s.Items)
}

// validateType rejects a type strict JSON Schema cannot represent from Go: a
// self-referential type (which needs $ref) or a field or slice element that
// reflects to a schema with no type. It runs at tool-construction time.
func validateType(t reflect.Type) error {
	if untypedLeaf(t) {
		return fmt.Errorf("type %s reflects to a schema with no type; a tool argument type "+
			"must be a struct with concrete fields", t)
	}
	return walk(t, map[reflect.Type]bool{}, nil)
}

func walk(t reflect.Type, onPath map[reflect.Type]bool, path []string) error {
	for t.Kind() == reflect.Pointer {
		t = t.Elem()
	}

	switch t.Kind() {
	case reflect.Slice, reflect.Array:
		// The element becomes the array's item schema; an untyped element (as []any
		// reflects) reflects to an untyped item schema.
		if untypedLeaf(t.Elem()) {
			return untypedError(append(path, "[]"), t.Elem())
		}
		return walk(t.Elem(), onPath, path)
	case reflect.Map:
		// A map value becomes additionalProperties — a free object — so an interface
		// value is valid. Recurse only to catch a cycle through the value type.
		return walk(t.Elem(), onPath, path)
	case reflect.Struct:
	default:
		return nil
	}

	// time.Time is special-cased by the reflector into a date-time string, so its
	// unexported internals are never walked.
	if t == reflect.TypeOf(jsonschema.Schema{}) || t.String() == "time.Time" {
		return nil
	}

	if onPath[t] {
		return fmt.Errorf(
			"recursive type %s (via %v): tool argument types must not be self-referential, "+
				"because strict JSON Schema cannot express a cycle without $ref",
			t, path)
	}
	onPath[t] = true
	defer delete(onPath, t)

	for i := range t.NumField() {
		f := t.Field(i)
		if !f.IsExported() || f.Tag.Get("json") == "-" {
			continue
		}
		if untypedLeaf(f.Type) {
			return untypedError(append(path, f.Name), f.Type)
		}
		if err := walk(f.Type, onPath, append(path, f.Name)); err != nil {
			return err
		}
	}
	return nil
}
