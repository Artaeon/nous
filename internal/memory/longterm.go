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
	mu            sync.RWMutex
	entries       map[string]Entry
	categoryIndex map[string][]string // category → list of entry keys for O(1) lookups
	path          string
	dirty         bool
}

func NewLongTermMemory(storePath string) *LongTermMemory {
	ltm := &LongTermMemory{
		entries:       make(map[string]Entry),
		categoryIndex: make(map[string][]string),
		path:          filepath.Join(storePath, "longterm.json"),
	}
	ltm.load()
	return ltm
}

// Store persists a key-value pair.
func (ltm *LongTermMemory) Store(key, value, category string) {
	ltm.mu.Lock()
	// Remove old entry from category index if it existed with a different category
	if old, ok := ltm.entries[key]; ok && old.Category != category {
		ltm.removeCategoryKeyLocked(key, old.Category)
	}
	ltm.entries[key] = Entry{
		Key:       key,
		Value:     value,
		Category:  category,
		CreatedAt: time.Now(),
	}
	ltm.addCategoryKeyLocked(key, category)
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

// Search returns all entries matching a category. Uses O(1) category index.
func (ltm *LongTermMemory) Search(category string) []Entry {
	ltm.mu.RLock()
	defer ltm.mu.RUnlock()

	keys := ltm.categoryIndex[category]
	results := make([]Entry, 0, len(keys))
	for _, k := range keys {
		if e, ok := ltm.entries[k]; ok {
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
	return safefile.WriteAtomicWithBackup(ltm.path, data, 0644, safefile.MaxBackups)
}

func (ltm *LongTermMemory) load() {
	data, err := os.ReadFile(ltm.path)
	if err != nil {
		return
	}
	_ = json.Unmarshal(data, &ltm.entries)
	ltm.rebuildCategoryIndex()
}

// rebuildCategoryIndex rebuilds the category index from all entries.
func (ltm *LongTermMemory) rebuildCategoryIndex() {
	ltm.categoryIndex = make(map[string][]string)
	for k, e := range ltm.entries {
		ltm.categoryIndex[e.Category] = append(ltm.categoryIndex[e.Category], k)
	}
}

// addCategoryKeyLocked adds a key to the category index. Must hold write lock.
func (ltm *LongTermMemory) addCategoryKeyLocked(key, category string) {
	for _, k := range ltm.categoryIndex[category] {
		if k == key {
			return // already indexed
		}
	}
	ltm.categoryIndex[category] = append(ltm.categoryIndex[category], key)
}

// removeCategoryKeyLocked removes a key from the category index. Must hold write lock.
func (ltm *LongTermMemory) removeCategoryKeyLocked(key, category string) {
	keys := ltm.categoryIndex[category]
	for i, k := range keys {
		if k == key {
			ltm.categoryIndex[category] = append(keys[:i], keys[i+1:]...)
			return
		}
	}
}
