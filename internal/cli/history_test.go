package cli

import (
	"os"
	"path/filepath"
	"testing"
)

func tempHistoryPath(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	return filepath.Join(dir, "history")
}

func TestNewHistory(t *testing.T) {
	h := NewHistory("/tmp/test-hist", 100)
	if h.maxSize != 100 {
		t.Errorf("maxSize = %d, want 100", h.maxSize)
	}
	if h.Size() != 0 {
		t.Errorf("Size = %d, want 0", h.Size())
	}
}

func TestNewHistoryDefaultMax(t *testing.T) {
	h := NewHistory("/tmp/test-hist", 0)
	if h.maxSize != 1000 {
		t.Errorf("maxSize = %d, want 1000 (default)", h.maxSize)
	}
}

func TestAddAndEntries(t *testing.T) {
	path := tempHistoryPath(t)
	h := NewHistory(path, 100)

	h.Add("hello")
	h.Add("world")

	entries := h.Entries()
	if len(entries) != 2 {
		t.Fatalf("len(entries) = %d, want 2", len(entries))
	}
	if entries[0] != "hello" || entries[1] != "world" {
		t.Errorf("entries = %v, want [hello world]", entries)
	}
}

func TestAddSkipsEmptyAndWhitespace(t *testing.T) {
	path := tempHistoryPath(t)
	h := NewHistory(path, 100)

	h.Add("")
	h.Add("   ")
	h.Add("\t")

	if h.Size() != 0 {
		t.Errorf("Size = %d, want 0 (empty entries should be skipped)", h.Size())
	}
}

func TestAddSkipsConsecutiveDuplicates(t *testing.T) {
	path := tempHistoryPath(t)
	h := NewHistory(path, 100)

	h.Add("hello")
	h.Add("hello")
	h.Add("world")
	h.Add("world")
	h.Add("hello")

	entries := h.Entries()
	if len(entries) != 3 {
		t.Fatalf("len(entries) = %d, want 3", len(entries))
	}
	expected := []string{"hello", "world", "hello"}
	for i, e := range expected {
		if entries[i] != e {
			t.Errorf("entries[%d] = %q, want %q", i, entries[i], e)
		}
	}
}

func TestAddPrunesOldest(t *testing.T) {
	path := tempHistoryPath(t)
	h := NewHistory(path, 3)

	h.Add("a")
	h.Add("b")
	h.Add("c")
	h.Add("d")

	entries := h.Entries()
	if len(entries) != 3 {
		t.Fatalf("len(entries) = %d, want 3", len(entries))
	}
	if entries[0] != "b" {
		t.Errorf("entries[0] = %q, want %q (oldest should be pruned)", entries[0], "b")
	}
}

func TestSaveAndLoad(t *testing.T) {
	path := tempHistoryPath(t)
	h := NewHistory(path, 100)

	h.Add("first")
	h.Add("second")
	h.Add("third")

	if err := h.Save(); err != nil {
		t.Fatalf("Save: %v", err)
	}

	// Load into a fresh history
	h2 := NewHistory(path, 100)
	if err := h2.Load(); err != nil {
		t.Fatalf("Load: %v", err)
	}

	entries := h2.Entries()
	if len(entries) != 3 {
		t.Fatalf("loaded len(entries) = %d, want 3", len(entries))
	}
	if entries[2] != "third" {
		t.Errorf("entries[2] = %q, want %q", entries[2], "third")
	}
}

func TestLoadNonexistentFile(t *testing.T) {
	h := NewHistory("/tmp/does-not-exist-"+t.Name(), 100)
	if err := h.Load(); err != nil {
		t.Errorf("Load on missing file should return nil, got: %v", err)
	}
	if h.Size() != 0 {
		t.Errorf("Size = %d, want 0", h.Size())
	}
}

func TestLoadPrunesExcess(t *testing.T) {
	path := tempHistoryPath(t)

	// Write a big history file manually
	h := NewHistory(path, 1000)
	for i := 0; i < 50; i++ {
		h.Add(filepath.Join("entry", string(rune('a'+i%26))))
	}
	_ = h.Save()

	// Load with a smaller max
	h2 := NewHistory(path, 5)
	if err := h2.Load(); err != nil {
		t.Fatalf("Load: %v", err)
	}
	if h2.Size() > 5 {
		t.Errorf("Size = %d, want <= 5", h2.Size())
	}
}

func TestPreviousAndNext(t *testing.T) {
	path := tempHistoryPath(t)
	h := NewHistory(path, 100)

	h.Add("a")
	h.Add("b")
	h.Add("c")

	// Navigate backwards
	if got := h.Previous(); got != "c" {
		t.Errorf("Previous() = %q, want %q", got, "c")
	}
	if got := h.Previous(); got != "b" {
		t.Errorf("Previous() = %q, want %q", got, "b")
	}
	if got := h.Previous(); got != "a" {
		t.Errorf("Previous() = %q, want %q", got, "a")
	}
	// At the beginning, stays at first entry
	if got := h.Previous(); got != "a" {
		t.Errorf("Previous() at start = %q, want %q", got, "a")
	}

	// Navigate forward
	if got := h.Next(); got != "b" {
		t.Errorf("Next() = %q, want %q", got, "b")
	}
	if got := h.Next(); got != "c" {
		t.Errorf("Next() = %q, want %q", got, "c")
	}
	// Past the end, empty
	if got := h.Next(); got != "" {
		t.Errorf("Next() past end = %q, want empty", got)
	}
}

func TestPreviousEmptyHistory(t *testing.T) {
	h := NewHistory(tempHistoryPath(t), 100)
	if got := h.Previous(); got != "" {
		t.Errorf("Previous() on empty = %q, want empty", got)
	}
	if got := h.Next(); got != "" {
		t.Errorf("Next() on empty = %q, want empty", got)
	}
}

func TestSearch(t *testing.T) {
	path := tempHistoryPath(t)
	h := NewHistory(path, 100)

	h.Add("/help")
	h.Add("hello world")
	h.Add("/history")
	h.Add("something else")
	h.Add("/help again")

	// Should find most recent matching entry
	if got := h.Search("/h"); got != "/help again" {
		t.Errorf("Search('/h') = %q, want %q", got, "/help again")
	}
	if got := h.Search("/hist"); got != "/history" {
		t.Errorf("Search('/hist') = %q, want %q", got, "/history")
	}
	if got := h.Search("hello"); got != "hello world" {
		t.Errorf("Search('hello') = %q, want %q", got, "hello world")
	}
	if got := h.Search("nonexistent"); got != "" {
		t.Errorf("Search('nonexistent') = %q, want empty", got)
	}
	if got := h.Search(""); got != "" {
		t.Errorf("Search('') = %q, want empty", got)
	}
}

func TestClear(t *testing.T) {
	path := tempHistoryPath(t)
	h := NewHistory(path, 100)

	h.Add("a")
	h.Add("b")
	_ = h.Save()

	if err := h.Clear(); err != nil {
		t.Fatalf("Clear: %v", err)
	}

	if h.Size() != 0 {
		t.Errorf("Size after Clear = %d, want 0", h.Size())
	}

	// File should be gone
	if _, err := os.Stat(path); !os.IsNotExist(err) {
		t.Errorf("history file should be deleted after Clear")
	}
}

func TestEntry(t *testing.T) {
	path := tempHistoryPath(t)
	h := NewHistory(path, 100)

	h.Add("first")
	h.Add("second")
	h.Add("third")

	if got := h.Entry(1); got != "first" {
		t.Errorf("Entry(1) = %q, want %q", got, "first")
	}
	if got := h.Entry(3); got != "third" {
		t.Errorf("Entry(3) = %q, want %q", got, "third")
	}
	if got := h.Entry(0); got != "" {
		t.Errorf("Entry(0) = %q, want empty", got)
	}
	if got := h.Entry(99); got != "" {
		t.Errorf("Entry(99) = %q, want empty", got)
	}
}

func TestResetPosition(t *testing.T) {
	path := tempHistoryPath(t)
	h := NewHistory(path, 100)

	h.Add("a")
	h.Add("b")

	h.Previous() // position -> "b"
	h.Previous() // position -> "a"

	h.ResetPosition()

	// After reset, Previous should return last entry again
	if got := h.Previous(); got != "b" {
		t.Errorf("Previous() after reset = %q, want %q", got, "b")
	}
}

func TestAddAutoSaves(t *testing.T) {
	path := tempHistoryPath(t)
	h := NewHistory(path, 100)

	h.Add("auto-saved")

	// Verify file exists and has content
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("ReadFile: %v", err)
	}
	if !contains(string(data), "auto-saved") {
		t.Errorf("history file should contain 'auto-saved', got: %q", string(data))
	}
}

func TestConcurrentAccess(t *testing.T) {
	path := tempHistoryPath(t)
	h := NewHistory(path, 1000)

	done := make(chan struct{})
	for i := 0; i < 10; i++ {
		go func(n int) {
			defer func() { done <- struct{}{} }()
			for j := 0; j < 50; j++ {
				h.Add("entry")
				h.Previous()
				h.Next()
				h.Entries()
				h.Size()
			}
		}(i)
	}
	for i := 0; i < 10; i++ {
		<-done
	}
	// Just verify no panics/races
}

func contains(s, substr string) bool {
	return len(s) >= len(substr) && searchString(s, substr)
}

func searchString(s, substr string) bool {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}
