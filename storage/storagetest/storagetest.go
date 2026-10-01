// Package storagetest provides an in-memory [storage.Store] for tests.
package storagetest

import (
	"context"
	"fmt"
	"slices"
	"sync"

	"github.com/wleev/temporal-agent-sdk/storage"
)

// MemoryStore is a [storage.Store] that keeps objects in a map. It is safe for
// concurrent use, and its zero value is an empty store.
type MemoryStore struct {
	mu      sync.Mutex
	objects map[string][]byte
}

var _ storage.Store = (*MemoryStore)(nil)

// NewMemoryStore returns an empty MemoryStore.
func NewMemoryStore() *MemoryStore {
	return &MemoryStore{}
}

// Put stores a copy of data under key. It returns ctx.Err() if ctx is done.
func (m *MemoryStore) Put(ctx context.Context, key string, data []byte) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.objects == nil {
		m.objects = map[string][]byte{}
	}
	m.objects[key] = slices.Clone(data)
	return nil
}

// Get returns a copy of the bytes under key, or an error wrapping
// [storage.ErrNotFound] if key holds no object. It returns ctx.Err() if ctx is
// done.
func (m *MemoryStore) Get(ctx context.Context, key string) ([]byte, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	data, ok := m.objects[key]
	if !ok {
		return nil, fmt.Errorf("%s: %w", key, storage.ErrNotFound)
	}
	return slices.Clone(data), nil
}

// Delete removes the object under key, if any.
func (m *MemoryStore) Delete(key string) {
	m.mu.Lock()
	defer m.mu.Unlock()
	delete(m.objects, key)
}

// Keys returns the stored keys in sorted order.
func (m *MemoryStore) Keys() []string {
	m.mu.Lock()
	defer m.mu.Unlock()
	keys := make([]string, 0, len(m.objects))
	for k := range m.objects {
		keys = append(keys, k)
	}
	slices.Sort(keys)
	return keys
}
