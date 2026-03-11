package memory

import (
	"os"
	"path/filepath"
	"testing"
)

func TestProjectMemoryRememberAndRecall(t *testing.T) {
	dir := t.TempDir()
	pm := NewProjectMemory(dir)

	pm.Remember("language", "Go", "user", 1.0)
	pm.Remember("framework", "none", "inferred", 0.8)

	fact, ok := pm.Recall("language")
	if !ok {
		t.Fatal("expected to recall 'language'")
	}
	if fact.Value != "Go" {
		t.Errorf("expected 'Go', got %q", fact.Value)
	}
	if fact.Source != "user" {
		t.Errorf("expected source 'user', got %q", fact.Source)
	}
	if fact.Confidence != 1.0 {
		t.Errorf("expected confidence 1.0, got %f", fact.Confidence)
	}
}

func TestProjectMemoryUpdate(t *testing.T) {
	dir := t.TempDir()
	pm := NewProjectMemory(dir)

	pm.Remember("language", "Go", "user", 0.8)
	original, _ := pm.Recall("language")
	created := original.CreatedAt

	pm.Remember("language", "Go 1.22+", "user", 1.0)
	updated, _ := pm.Recall("language")

	if updated.Value != "Go 1.22+" {
		t.Errorf("expected updated value, got %q", updated.Value)
	}
	if !updated.CreatedAt.Equal(created) {
		t.Error("expected CreatedAt to be preserved on update")
	}
	if updated.Confidence != 1.0 {
		t.Errorf("expected confidence 1.0 after update, got %f", updated.Confidence)
	}
}

func TestProjectMemorySearch(t *testing.T) {
	dir := t.TempDir()
	pm := NewProjectMemory(dir)

	pm.Remember("language", "Go", "user", 1.0)
	pm.Remember("build_tool", "make", "inferred", 0.9)
	pm.Remember("test_cmd", "go test", "inferred", 0.9)

	results := pm.Search("go")
	if len(results) < 2 {
		t.Errorf("expected at least 2 results for 'go', got %d", len(results))
	}

	results = pm.Search("make")
	if len(results) != 1 {
		t.Errorf("expected 1 result for 'make', got %d", len(results))
	}

	results = pm.Search("nonexistent")
	if len(results) != 0 {
		t.Errorf("expected 0 results for 'nonexistent', got %d", len(results))
	}
}

func TestProjectMemoryForget(t *testing.T) {
	dir := t.TempDir()
	pm := NewProjectMemory(dir)

	pm.Remember("language", "Go", "user", 1.0)
	if !pm.Forget("language") {
		t.Error("expected Forget to return true for existing key")
	}
	if pm.Forget("language") {
		t.Error("expected Forget to return false for already-deleted key")
	}

	_, ok := pm.Recall("language")
	if ok {
		t.Error("expected key to be forgotten")
	}
}

func TestProjectMemorySize(t *testing.T) {
	dir := t.TempDir()
	pm := NewProjectMemory(dir)

	if pm.Size() != 0 {
		t.Error("expected empty memory")
	}

	pm.Remember("a", "1", "user", 1.0)
	pm.Remember("b", "2", "user", 1.0)

	if pm.Size() != 2 {
		t.Errorf("expected size 2, got %d", pm.Size())
	}
}

func TestProjectMemoryAll(t *testing.T) {
	dir := t.TempDir()
	pm := NewProjectMemory(dir)

	pm.Remember("a", "1", "user", 1.0)
	pm.Remember("b", "2", "user", 1.0)

	all := pm.All()
	if len(all) != 2 {
		t.Errorf("expected 2 facts, got %d", len(all))
	}
}

func TestProjectMemoryFlushAndReload(t *testing.T) {
	dir := t.TempDir()
	pm := NewProjectMemory(dir)

	pm.Remember("language", "Go", "user", 1.0)
	pm.Remember("build", "make", "inferred", 0.9)

	if err := pm.Flush(); err != nil {
		t.Fatalf("flush error: %v", err)
	}

	// Verify file exists
	jsonPath := filepath.Join(dir, ".nous", "project_memory.json")
	if _, err := os.Stat(jsonPath); err != nil {
		t.Fatalf("expected JSON file to exist: %v", err)
	}

	// Reload
	pm2 := NewProjectMemory(dir)
	if pm2.Size() != 2 {
		t.Errorf("expected 2 facts after reload, got %d", pm2.Size())
	}

	fact, ok := pm2.Recall("language")
	if !ok || fact.Value != "Go" {
		t.Error("expected to recall 'language' = 'Go' after reload")
	}
}

func TestProjectMemoryEmptyReload(t *testing.T) {
	dir := t.TempDir()
	pm := NewProjectMemory(dir)
	if pm.Size() != 0 {
		t.Error("expected empty memory when no file exists")
	}
}
