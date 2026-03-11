package sentinel

import (
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"testing"
	"time"
)

func TestNewWatcher(t *testing.T) {
	dir := t.TempDir()
	w, err := NewWatcher(dir, 50*time.Millisecond, nil)
	if err != nil {
		t.Fatalf("NewWatcher: %v", err)
	}
	defer w.Stop()

	if w.WatchCount() < 1 {
		t.Error("should watch at least the root directory")
	}
}

func TestWatcherIgnoresHiddenDirs(t *testing.T) {
	dir := t.TempDir()
	os.MkdirAll(filepath.Join(dir, ".git", "objects"), 0755)
	os.MkdirAll(filepath.Join(dir, "src"), 0755)

	w, err := NewWatcher(dir, 50*time.Millisecond, nil)
	if err != nil {
		t.Fatalf("NewWatcher: %v", err)
	}
	defer w.Stop()

	// Should watch root + src, but not .git
	count := w.WatchCount()
	if count != 2 {
		t.Errorf("expected 2 watches (root + src), got %d", count)
	}
}

func TestWatcherDetectsFileCreate(t *testing.T) {
	dir := t.TempDir()

	var mu sync.Mutex
	var received []FileEvent

	w, err := NewWatcher(dir, 50*time.Millisecond, func(events []FileEvent) {
		mu.Lock()
		received = append(received, events...)
		mu.Unlock()
	})
	if err != nil {
		t.Fatalf("NewWatcher: %v", err)
	}

	// Start watcher in background
	go w.Run()
	defer w.Stop()

	// Give inotify a moment to register
	time.Sleep(50 * time.Millisecond)

	// Create a file
	os.WriteFile(filepath.Join(dir, "test.go"), []byte("package main"), 0644)

	// Wait for debounce
	time.Sleep(200 * time.Millisecond)

	mu.Lock()
	defer mu.Unlock()

	if len(received) == 0 {
		t.Fatal("expected at least one event")
	}

	found := false
	for _, ev := range received {
		if ev.Path == "test.go" {
			found = true
			break
		}
	}
	if !found {
		t.Errorf("expected event for test.go, got: %v", received)
	}
}

func TestWatcherDetectsFileModify(t *testing.T) {
	dir := t.TempDir()

	// Pre-create file
	testFile := filepath.Join(dir, "main.go")
	os.WriteFile(testFile, []byte("package main\n"), 0644)

	var mu sync.Mutex
	var received []FileEvent

	w, err := NewWatcher(dir, 50*time.Millisecond, func(events []FileEvent) {
		mu.Lock()
		received = append(received, events...)
		mu.Unlock()
	})
	if err != nil {
		t.Fatalf("NewWatcher: %v", err)
	}

	go w.Run()
	defer w.Stop()

	time.Sleep(50 * time.Millisecond)

	// Modify file
	os.WriteFile(testFile, []byte("package main\nfunc main() {}\n"), 0644)

	time.Sleep(200 * time.Millisecond)

	mu.Lock()
	defer mu.Unlock()

	if len(received) == 0 {
		t.Fatal("expected modify event")
	}

	foundModified := false
	for _, ev := range received {
		if ev.Path == "main.go" && (ev.Type == EventModified || ev.Type == EventCreated) {
			foundModified = true
		}
	}
	if !foundModified {
		t.Errorf("expected modified event for main.go, got: %v", received)
	}
}

func TestWatcherDetectsFileDelete(t *testing.T) {
	dir := t.TempDir()

	testFile := filepath.Join(dir, "doomed.go")
	os.WriteFile(testFile, []byte("package main\n"), 0644)

	var mu sync.Mutex
	var received []FileEvent

	w, err := NewWatcher(dir, 50*time.Millisecond, func(events []FileEvent) {
		mu.Lock()
		received = append(received, events...)
		mu.Unlock()
	})
	if err != nil {
		t.Fatalf("NewWatcher: %v", err)
	}

	go w.Run()
	defer w.Stop()

	time.Sleep(50 * time.Millisecond)

	os.Remove(testFile)

	time.Sleep(200 * time.Millisecond)

	mu.Lock()
	defer mu.Unlock()

	if len(received) == 0 {
		t.Fatal("expected delete event")
	}

	found := false
	for _, ev := range received {
		if ev.Path == "doomed.go" && ev.Type == EventDeleted {
			found = true
		}
	}
	if !found {
		t.Errorf("expected deleted event for doomed.go, got: %v", received)
	}
}

func TestWatcherIgnoresSwapFiles(t *testing.T) {
	dir := t.TempDir()

	var mu sync.Mutex
	var received []FileEvent

	w, err := NewWatcher(dir, 50*time.Millisecond, func(events []FileEvent) {
		mu.Lock()
		received = append(received, events...)
		mu.Unlock()
	})
	if err != nil {
		t.Fatalf("NewWatcher: %v", err)
	}

	go w.Run()
	defer w.Stop()

	time.Sleep(50 * time.Millisecond)

	// Create a .swp file (vim swap) — should be ignored
	os.WriteFile(filepath.Join(dir, ".main.go.swp"), []byte("swap data"), 0644)

	time.Sleep(200 * time.Millisecond)

	mu.Lock()
	defer mu.Unlock()

	for _, ev := range received {
		if ev.Path == ".main.go.swp" {
			t.Error("should not receive events for .swp files")
		}
	}
}

func TestWatcherDebounce(t *testing.T) {
	dir := t.TempDir()

	var mu sync.Mutex
	callCount := 0

	w, err := NewWatcher(dir, 100*time.Millisecond, func(events []FileEvent) {
		mu.Lock()
		callCount++
		mu.Unlock()
	})
	if err != nil {
		t.Fatalf("NewWatcher: %v", err)
	}

	go w.Run()
	defer w.Stop()

	time.Sleep(50 * time.Millisecond)

	// Rapid-fire 5 file creates within debounce window
	for i := 0; i < 5; i++ {
		os.WriteFile(filepath.Join(dir, fmt.Sprintf("file%d.go", i)), []byte("package main"), 0644)
		time.Sleep(10 * time.Millisecond)
	}

	// Wait for debounce to fire
	time.Sleep(300 * time.Millisecond)

	mu.Lock()
	defer mu.Unlock()

	// Should be called once or twice, not 5 times
	if callCount > 2 {
		t.Errorf("debounce failed: callback called %d times (expected 1-2)", callCount)
	}
}

func TestWatcherSubdirectoryWatch(t *testing.T) {
	dir := t.TempDir()
	subDir := filepath.Join(dir, "pkg", "core")
	os.MkdirAll(subDir, 0755)

	var mu sync.Mutex
	var received []FileEvent

	w, err := NewWatcher(dir, 50*time.Millisecond, func(events []FileEvent) {
		mu.Lock()
		received = append(received, events...)
		mu.Unlock()
	})
	if err != nil {
		t.Fatalf("NewWatcher: %v", err)
	}

	go w.Run()
	defer w.Stop()

	time.Sleep(50 * time.Millisecond)

	// Create file in subdirectory
	os.WriteFile(filepath.Join(subDir, "core.go"), []byte("package core"), 0644)

	time.Sleep(200 * time.Millisecond)

	mu.Lock()
	defer mu.Unlock()

	if len(received) == 0 {
		t.Fatal("expected event for subdirectory file")
	}

	found := false
	for _, ev := range received {
		if ev.Path == filepath.Join("pkg", "core", "core.go") {
			found = true
		}
	}
	if !found {
		t.Errorf("expected event with relative path pkg/core/core.go, got: %v", received)
	}
}

func TestChangedGoFiles(t *testing.T) {
	events := []FileEvent{
		{Path: "main.go", Type: EventModified},
		{Path: "README.md", Type: EventModified},
		{Path: "internal/foo.go", Type: EventCreated},
		{Path: "go.sum", Type: EventModified},
		{Path: "main.go", Type: EventModified}, // duplicate
	}

	got := ChangedGoFiles(events)
	if len(got) != 2 {
		t.Errorf("expected 2 unique .go files, got %d: %v", len(got), got)
	}
}

func TestEventTypeString(t *testing.T) {
	tests := []struct {
		t    EventType
		want string
	}{
		{EventCreated, "created"},
		{EventModified, "modified"},
		{EventDeleted, "deleted"},
		{EventRenamed, "renamed"},
	}

	for _, tt := range tests {
		if got := tt.t.String(); got != tt.want {
			t.Errorf("EventType(%d).String() = %q, want %q", tt.t, got, tt.want)
		}
	}
}

func TestWatcherStop(t *testing.T) {
	dir := t.TempDir()

	w, err := NewWatcher(dir, 50*time.Millisecond, nil)
	if err != nil {
		t.Fatalf("NewWatcher: %v", err)
	}

	done := make(chan struct{})
	go func() {
		w.Run()
		close(done)
	}()

	time.Sleep(50 * time.Millisecond)
	w.Stop()

	select {
	case <-done:
		// OK — Run returned
	case <-time.After(2 * time.Second):
		t.Error("Run did not return after Stop")
	}
}
