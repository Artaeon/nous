package hands

import (
	"path/filepath"
	"sync"
	"testing"
)

func TestHandStateGetSet(t *testing.T) {
	dir := t.TempDir()
	store, err := NewHandStateStore(dir)
	if err != nil {
		t.Fatalf("failed to create store: %v", err)
	}

	// Get from empty store returns empty string
	if v := store.Get("myhand", "key1"); v != "" {
		t.Errorf("expected empty string for non-existent key, got %q", v)
	}

	// Set and get
	if err := store.Set("myhand", "key1", "value1"); err != nil {
		t.Fatalf("set failed: %v", err)
	}
	if v := store.Get("myhand", "key1"); v != "value1" {
		t.Errorf("expected 'value1', got %q", v)
	}

	// Overwrite
	if err := store.Set("myhand", "key1", "value2"); err != nil {
		t.Fatalf("set failed: %v", err)
	}
	if v := store.Get("myhand", "key1"); v != "value2" {
		t.Errorf("expected 'value2', got %q", v)
	}
}

func TestHandStateGetAll(t *testing.T) {
	dir := t.TempDir()
	store, err := NewHandStateStore(dir)
	if err != nil {
		t.Fatalf("failed to create store: %v", err)
	}

	// GetAll for nonexistent hand returns nil
	if m := store.GetAll("nohand"); m != nil {
		t.Errorf("expected nil for nonexistent hand, got %v", m)
	}

	store.Set("h1", "a", "1")
	store.Set("h1", "b", "2")

	all := store.GetAll("h1")
	if len(all) != 2 {
		t.Fatalf("expected 2 entries, got %d", len(all))
	}
	if all["a"] != "1" || all["b"] != "2" {
		t.Errorf("unexpected values: %v", all)
	}

	// Verify it's a copy (modifying it doesn't affect store)
	all["a"] = "modified"
	if v := store.Get("h1", "a"); v != "1" {
		t.Error("expected GetAll to return a copy")
	}
}

func TestHandStatePersistence(t *testing.T) {
	dir := t.TempDir()

	// Create store and set values
	store1, err := NewHandStateStore(dir)
	if err != nil {
		t.Fatalf("failed to create store: %v", err)
	}
	store1.Set("agent1", "last_file", "/tmp/out.txt")
	store1.Set("agent1", "run_count", "5")
	store1.Set("agent2", "status", "ok")

	// Create a new store from the same directory — should load persisted data
	store2, err := NewHandStateStore(dir)
	if err != nil {
		t.Fatalf("failed to create second store: %v", err)
	}

	if v := store2.Get("agent1", "last_file"); v != "/tmp/out.txt" {
		t.Errorf("expected '/tmp/out.txt' after reload, got %q", v)
	}
	if v := store2.Get("agent1", "run_count"); v != "5" {
		t.Errorf("expected '5' after reload, got %q", v)
	}
	if v := store2.Get("agent2", "status"); v != "ok" {
		t.Errorf("expected 'ok' after reload, got %q", v)
	}
}

func TestHandStatePersistenceFileExists(t *testing.T) {
	dir := t.TempDir()

	store, _ := NewHandStateStore(dir)
	store.Set("testhand", "key", "val")

	// Verify the JSON file was created
	path := filepath.Join(dir, "testhand.json")
	if _, err := filepath.Glob(path); err != nil {
		t.Errorf("expected state file at %s", path)
	}
}

func TestHandStateConcurrentAccess(t *testing.T) {
	dir := t.TempDir()
	store, err := NewHandStateStore(dir)
	if err != nil {
		t.Fatalf("failed to create store: %v", err)
	}

	var wg sync.WaitGroup
	const goroutines = 20
	const iterations = 50

	// Concurrent writes
	for i := 0; i < goroutines; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			for j := 0; j < iterations; j++ {
				handName := "hand"
				key := "key"
				value := "value"
				store.Set(handName, key, value)
				store.Get(handName, key)
				store.GetAll(handName)
			}
		}(i)
	}

	wg.Wait()

	// Verify data is consistent
	v := store.Get("hand", "key")
	if v != "value" {
		t.Errorf("expected 'value' after concurrent access, got %q", v)
	}
}

func TestExtractState(t *testing.T) {
	tests := []struct {
		name   string
		output string
		want   map[string]string
	}{
		{
			name:   "single state",
			output: "Some output\n[STATE last_file=/tmp/result.txt]\nMore output",
			want:   map[string]string{"last_file": "/tmp/result.txt"},
		},
		{
			name:   "multiple states",
			output: "[STATE count=3]\n[STATE status=done]",
			want:   map[string]string{"count": "3", "status": "done"},
		},
		{
			name:   "no state",
			output: "Just regular output with no state markers",
			want:   nil,
		},
		{
			name:   "state with spaces in value",
			output: "[STATE message=hello world]",
			want:   map[string]string{"message": "hello world"},
		},
		{
			name:   "empty output",
			output: "",
			want:   nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := ExtractState(tt.output)
			if tt.want == nil {
				if got != nil {
					t.Errorf("expected nil, got %v", got)
				}
				return
			}
			if len(got) != len(tt.want) {
				t.Fatalf("expected %d entries, got %d: %v", len(tt.want), len(got), got)
			}
			for k, v := range tt.want {
				if got[k] != v {
					t.Errorf("key %q: expected %q, got %q", k, v, got[k])
				}
			}
		})
	}
}

func TestFormatStatePrompt(t *testing.T) {
	// Empty state
	if s := FormatStatePrompt(nil); s != "" {
		t.Errorf("expected empty string for nil state, got %q", s)
	}
	if s := FormatStatePrompt(map[string]string{}); s != "" {
		t.Errorf("expected empty string for empty state, got %q", s)
	}

	// Non-empty state
	state := map[string]string{"key1": "val1"}
	s := FormatStatePrompt(state)
	if s == "" {
		t.Error("expected non-empty prompt")
	}
	if !containsSubstring(s, "Previous run context") {
		t.Errorf("expected 'Previous run context' in prompt, got %q", s)
	}
	if !containsSubstring(s, "key1 = val1") {
		t.Errorf("expected 'key1 = val1' in prompt, got %q", s)
	}
}

func containsSubstring(s, sub string) bool {
	return len(s) >= len(sub) && (s == sub || len(s) > 0 && findSubstring(s, sub))
}

func findSubstring(s, sub string) bool {
	for i := 0; i <= len(s)-len(sub); i++ {
		if s[i:i+len(sub)] == sub {
			return true
		}
	}
	return false
}
