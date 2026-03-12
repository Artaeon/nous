package memory

import (
	"encoding/json"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/artaeon/nous/internal/safefile"
)

// Entry represents a single piece of long-term knowledge.
type Entry struct {
	Key       string    `json:"key"`
	Value     string    `json:"value"`
	Category  string    `json:"category"`
	CreatedAt time.Time `json:"created_at"`
	AccessCount int     `json:"access_count"`
}

// LongTermMemory provides persistent key-value storage backed by a JSON file.
// In future iterations this will use mmap for zero-copy access, but the
// JSON approach keeps the initial implementation simple and portable.
type LongTermMemory struct {
	mu      sync.RWMutex
	entries map[string]Entry
	path    string
	dirty   bool
}

func NewLongTermMemory(storePath string) *LongTermMemory {
	ltm := &LongTermMemory{
		entries: make(map[string]Entry),
		path:    filepath.Join(storePath, "longterm.json"),
	}
	ltm.load()
	return ltm
}

// Store persists a key-value pair.
func (ltm *LongTermMemory) Store(key, value, category string) {
	ltm.mu.Lock()
	ltm.entries[key] = Entry{
		Key:       key,
		Value:     value,
		Category:  category,
		CreatedAt: time.Now(),
	}
	ltm.dirty = true
	ltm.mu.Unlock()

	_ = ltm.Flush()
}

// Retrieve looks up a key and increments its access counter.
func (ltm *LongTermMemory) Retrieve(key string) (string, bool) {
	ltm.mu.Lock()
	defer ltm.mu.Unlock()

	entry, ok := ltm.entries[key]
	if !ok {
		return "", false
	}

	entry.AccessCount++
	ltm.entries[key] = entry
	ltm.dirty = true

	return entry.Value, true
}

// Search returns all entries matching a category.
func (ltm *LongTermMemory) Search(category string) []Entry {
	ltm.mu.RLock()
	defer ltm.mu.RUnlock()

	var results []Entry
	for _, e := range ltm.entries {
		if e.Category == category {
			results = append(results, e)
		}
	}
	return results
}

// All returns every entry in long-term memory.
func (ltm *LongTermMemory) All() []Entry {
	ltm.mu.RLock()
	defer ltm.mu.RUnlock()

	results := make([]Entry, 0, len(ltm.entries))
	for _, e := range ltm.entries {
		results = append(results, e)
	}
	return results
}

// Size returns the number of stored entries.
func (ltm *LongTermMemory) Size() int {
	ltm.mu.RLock()
	defer ltm.mu.RUnlock()
	return len(ltm.entries)
}

// Flush writes dirty state to disk.
func (ltm *LongTermMemory) Flush() error {
	ltm.mu.Lock()
	defer ltm.mu.Unlock()

	if !ltm.dirty {
		return nil
	}

	if err := os.MkdirAll(filepath.Dir(ltm.path), 0755); err != nil {
		return err
	}

	data, err := json.MarshalIndent(ltm.entries, "", "  ")
	if err != nil {
		return err
	}

	ltm.dirty = false
	return safefile.WriteAtomic(ltm.path, data, 0644)
}

func (ltm *LongTermMemory) load() {
	data, err := os.ReadFile(ltm.path)
	if err != nil {
		return
	}
	_ = json.Unmarshal(data, &ltm.entries)
}
