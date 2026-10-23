package model_test

import (
	"encoding/json"
	"reflect"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/wleev/temporal-agent-sdk/model"
)

func TestSettings_ZeroMarshalsEmpty(t *testing.T) {
	raw, err := json.Marshal(model.Settings{})
	require.NoError(t, err)
	assert.JSONEq(t, `{}`, string(raw))
}

func TestSettings_ReasoningEffortJSON(t *testing.T) {
	raw, err := json.Marshal(model.Settings{ReasoningEffort: model.Ptr("high")})
	require.NoError(t, err)
	assert.JSONEq(t, `{"reasoning_effort":"high"}`, string(raw))

	var got model.Settings
	require.NoError(t, json.Unmarshal(raw, &got))
	if assert.NotNil(t, got.ReasoningEffort) {
		assert.Equal(t, "high", *got.ReasoningEffort)
	}
}

// nonZero returns a non-zero value of type typ, or false for an unsupported
// kind.
func nonZero(typ reflect.Type) (reflect.Value, bool) {
	v := reflect.New(typ).Elem()
	switch typ.Kind() {
	case reflect.Pointer:
		elem, ok := nonZero(typ.Elem())
		if !ok {
			return reflect.Value{}, false
		}
		p := reflect.New(typ.Elem())
		p.Elem().Set(elem)
		return p, true
	case reflect.Float32, reflect.Float64:
		v.SetFloat(0.5)
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		v.SetInt(7)
	case reflect.String:
		v.SetString("set")
	case reflect.Bool:
		v.SetBool(true)
	default:
		return reflect.Value{}, false
	}
	return v, true
}

func TestSettings_CloneCopiesEveryField(t *testing.T) {
	var s model.Settings
	sv := reflect.ValueOf(&s).Elem()
	for i := range sv.NumField() {
		f := sv.Type().Field(i)
		v, ok := nonZero(f.Type)
		if !assert.True(t, ok, "field %s has unsupported type %s", f.Name, f.Type) {
			continue
		}
		sv.Field(i).Set(v)
	}

	c := s.Clone()
	assert.Equal(t, s, c)

	cv := reflect.ValueOf(c)
	for i := range sv.NumField() {
		if sv.Field(i).Kind() == reflect.Pointer {
			assert.NotEqual(t, sv.Field(i).Pointer(), cv.Field(i).Pointer(),
				"field %s shares its pointer", sv.Type().Field(i).Name)
		}
	}
}
