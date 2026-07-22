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
func For[T any]() (json.RawMessage, error) {
	var zero T
	t := reflect.TypeOf(&zero).Elem()

	if err := checkAcyclic(t); err != nil {
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

// checkAcyclic rejects self-referential types, which strict JSON Schema cannot
// express without $ref and which would otherwise recurse until the stack
// overflows. It returns an error the caller can act on at registration time.
func checkAcyclic(t reflect.Type) error {
	return walk(t, map[reflect.Type]bool{}, nil)
}

func walk(t reflect.Type, onPath map[reflect.Type]bool, path []string) error {
	for t.Kind() == reflect.Ptr {
		t = t.Elem()
	}

	switch t.Kind() {
	case reflect.Slice, reflect.Array, reflect.Map:
		// Only the element type can reintroduce a cycle; map keys are scalars.
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
		if err := walk(f.Type, onPath, append(path, f.Name)); err != nil {
			return err
		}
	}
	return nil
}
