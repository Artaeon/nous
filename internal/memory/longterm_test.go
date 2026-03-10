package memory

import (
	"os"
	"path/filepath"
	"testing"
)

func tempDir(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	return dir
}

func TestNewLongTermMemory(t *testing.T) {
	dir := tempDir(t)
	ltm := NewLongTermMemory(dir)
	if ltm == nil {
		t.Fatal("NewLongTermMemory returned nil")
	}
	if ltm.Size() != 0 {
		t.Errorf("expected size 0, got %d", ltm.Size())
	}
}

func TestLTMStoreAndRetrieve(t *testing.T) {
	dir := tempDir(t)
	ltm := NewLongTermMemory(dir)

	ltm.Store("project.language", "Go", "project")

	val, ok := ltm.Retrieve("project.language")
	if !ok {
		t.Fatal("expected to retrieve stored entry")
	}
	if val != "Go" {
		t.Errorf("expected 'Go', got %q", val)
	}
}

func TestLTMRetrieveNonExistent(t *testing.T) {
	dir := tempDir(t)
	ltm := NewLongTermMemory(dir)

	_, ok := ltm.Retrieve("nonexistent")
	if ok {
		t.Error("expected ok=false for nonexistent key")
	}
}

func TestLTMRetrieveIncrementsAccessCount(t *testing.T) {
	dir := tempDir(t)
	ltm := NewLongTermMemory(dir)

	ltm.Store("key", "value", "test")

	// Access 3 times
	ltm.Retrieve("key")
	ltm.Retrieve("key")
	ltm.Retrieve("key")

	ltm.mu.RLock()
	entry := ltm.entries["key"]
	ltm.mu.RUnlock()

	if entry.AccessCount != 3 {
		t.Errorf("expected access count 3, got %d", entry.AccessCount)
	}
}

func TestLTMSearchByCategory(t *testing.T) {
	dir := tempDir(t)
	ltm := NewLongTermMemory(dir)

	ltm.Store("k1", "v1", "project")
	ltm.Store("k2", "v2", "project")
	ltm.Store("k3", "v3", "user")

	results := ltm.Search("project")
	if len(results) != 2 {
		t.Fatalf("expected 2 results for category 'project', got %d", len(results))
	}

	// Verify all results have the correct category
	for _, r := range results {
		if r.Category != "project" {
			t.Errorf("expected category 'project', got %q", r.Category)
		}
	}
}

func TestLTMSearchNoResults(t *testing.T) {
	dir := tempDir(t)
	ltm := NewLongTermMemory(dir)

	ltm.Store("k1", "v1", "project")

	results := ltm.Search("nonexistent")
	if len(results) != 0 {
		t.Errorf("expected 0 results, got %d", len(results))
	}
}

func TestLTMAll(t *testing.T) {
	dir := tempDir(t)
	ltm := NewLongTermMemory(dir)

	ltm.Store("a", "1", "cat1")
	ltm.Store("b", "2", "cat2")
	ltm.Store("c", "3", "cat1")

	all := ltm.All()
	if len(all) != 3 {
		t.Errorf("expected 3 entries, got %d", len(all))
	}
}

func TestLTMSize(t *testing.T) {
	dir := tempDir(t)
	ltm := NewLongTermMemory(dir)

	ltm.Store("a", "1", "test")
	ltm.Store("b", "2", "test")

	if ltm.Size() != 2 {
		t.Errorf("expected size 2, got %d", ltm.Size())
	}
}

func TestLTMStoreOverwrite(t *testing.T) {
	dir := tempDir(t)
	ltm := NewLongTermMemory(dir)

	ltm.Store("key", "v1", "cat")
	ltm.Store("key", "v2", "cat")

	val, _ := ltm.Retrieve("key")
	if val != "v2" {
		t.Errorf("expected 'v2', got %q", val)
	}

	if ltm.Size() != 1 {
		t.Errorf("expected size 1 after overwrite, got %d", ltm.Size())
	}
}

func TestLTMPersistence(t *testing.T) {
	dir := tempDir(t)

	// Store data
	ltm1 := NewLongTermMemory(dir)
	ltm1.Store("project", "nous", "meta")
	ltm1.Store("language", "Go", "meta")

	// Verify the file was written
	jsonPath := filepath.Join(dir, "longterm.json")
	if _, err := os.Stat(jsonPath); os.IsNotExist(err) {
		t.Fatal("expected longterm.json to be written to disk")
	}

	// Load from disk in a new instance
	ltm2 := NewLongTermMemory(dir)

	if ltm2.Size() != 2 {
		t.Fatalf("expected 2 entries after reload, got %d", ltm2.Size())
	}

	val, ok := ltm2.Retrieve("project")
	if !ok {
		t.Fatal("expected 'project' key to persist")
	}
	if val != "nous" {
		t.Errorf("expected 'nous', got %q", val)
	}

	val, ok = ltm2.Retrieve("language")
	if !ok {
		t.Fatal("expected 'language' key to persist")
	}
	if val != "Go" {
		t.Errorf("expected 'Go', got %q", val)
	}
}

func TestLTMFlushIdempotent(t *testing.T) {
	dir := tempDir(t)
	ltm := NewLongTermMemory(dir)

	// Flush with no changes should be a no-op
	if err := ltm.Flush(); err != nil {
		t.Errorf("expected no error on clean flush, got %v", err)
	}
}

func TestLTMLoadFromEmptyDir(t *testing.T) {
	dir := tempDir(t)
	// No longterm.json file exists — should just start empty
	ltm := NewLongTermMemory(dir)
	if ltm.Size() != 0 {
		t.Errorf("expected size 0 from empty dir, got %d", ltm.Size())
	}
}
