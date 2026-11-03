// Package jsontest reads values out of JSON documents in tests.
package jsontest

import (
	"encoding/json"
	"strconv"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"
)

// Path returns the raw JSON at the dotted path in the JSON document raw, or ""
// when the path is absent. Numeric segments index arrays.
func Path(t testing.TB, raw []byte, path string) string {
	t.Helper()
	var cur any
	require.NoError(t, json.Unmarshal(raw, &cur))
	for _, seg := range strings.Split(path, ".") {
		switch v := cur.(type) {
		case map[string]any:
			next, ok := v[seg]
			if !ok {
				return ""
			}
			cur = next
		case []any:
			i, err := strconv.Atoi(seg)
			if err != nil || i < 0 || i >= len(v) {
				return ""
			}
			cur = v[i]
		default:
			return ""
		}
	}
	out, err := json.Marshal(cur)
	require.NoError(t, err)
	return string(out)
}
