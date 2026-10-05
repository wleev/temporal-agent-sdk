package storage_test

import (
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/wleev/temporal-agent-sdk/storage"
)

func TestNewConfig_StartsFromTheDefaults(t *testing.T) {
	c, err := storage.NewConfig()

	require.NoError(t, err)
	assert.Equal(t, storage.Config{
		DriverName:  storage.DefaultDriverName,
		KeyPrefix:   storage.DefaultKeyPrefix,
		Concurrency: storage.DefaultConcurrency,
		Timeout:     storage.DefaultTimeout,
	}, c)
}

func TestNewConfig_AppliesTheOptions(t *testing.T) {
	c, err := storage.NewConfig(
		storage.WithDriverName("s3-primary"),
		storage.WithKeyPrefix("tenant-a/"),
		storage.WithThreshold(64<<10),
		storage.WithConcurrency(2),
		storage.WithTimeout(time.Second),
	)

	require.NoError(t, err)
	assert.Equal(t, storage.Config{
		DriverName:  "s3-primary",
		KeyPrefix:   "tenant-a/",
		Threshold:   64 << 10,
		Concurrency: 2,
		Timeout:     time.Second,
	}, c)
}

func TestNewConfig_RejectsInvalidOptions(t *testing.T) {
	tests := []struct {
		name string
		opt  storage.Option
	}{
		{name: "empty driver name", opt: storage.WithDriverName("")},
		{name: "negative threshold", opt: storage.WithThreshold(-1)},
		{name: "zero concurrency", opt: storage.WithConcurrency(0)},
		{name: "zero timeout", opt: storage.WithTimeout(0)},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := storage.NewConfig(tt.opt)
			assert.Error(t, err)
		})
	}
}

// TestPackage_ImportsOnlyTheStandardLibrary checks that the non-test files of
// this package import no module outside the standard library.
func TestPackage_ImportsOnlyTheStandardLibrary(t *testing.T) {
	files, err := filepath.Glob("*.go")
	require.NoError(t, err)

	for _, name := range files {
		if strings.HasSuffix(name, "_test.go") {
			continue
		}
		src, err := os.ReadFile(name)
		require.NoError(t, err)
		f, err := parser.ParseFile(token.NewFileSet(), name, src, parser.ImportsOnly)
		require.NoError(t, err)
		for _, imp := range f.Imports {
			path, err := strconv.Unquote(imp.Path.Value)
			require.NoError(t, err)
			first, _, _ := strings.Cut(path, "/")
			assert.NotContains(t, first, ".", "%s imports %s", name, path)
		}
	}
}
