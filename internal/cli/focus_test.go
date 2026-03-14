package cli

import (
	"path/filepath"
	"testing"
	"time"
)

func TestFocusManager_StartAndStop(t *testing.T) {
	dir := t.TempDir()
	fm := NewFocusManager(filepath.Join(dir, "focus.json"))

	err := fm.Start("write tests", 25*time.Minute)
	if err != nil {
		t.Fatalf("Start failed: %v", err)
	}

	s := fm.Status()
	if s == nil {
		t.Fatal("Status returned nil during active session")
	}
	if s.Task != "write tests" {
		t.Errorf("expected task 'write tests', got %q", s.Task)
	}
	if s.Remaining() <= 0 {
		t.Error("expected positive remaining time")
	}

	// Should not be able to start another
	err = fm.Start("another task", 10*time.Minute)
	if err == nil {
		t.Error("expected error starting second session")
	}

	// Stop
	result := fm.Stop()
	if result == nil {
		t.Fatal("Stop returned nil")
	}
	if result.Task != "write tests" {
		t.Errorf("expected task 'write tests', got %q", result.Task)
	}
	if !result.Completed {
		t.Error("expected session to be marked as completed")
	}

	// Status should be nil after stop
	if fm.Status() != nil {
		t.Error("expected nil status after stop")
	}
}

func TestFocusManager_AddNote(t *testing.T) {
	dir := t.TempDir()
	fm := NewFocusManager(filepath.Join(dir, "focus.json"))

	// Note without session should fail
	if err := fm.AddNote("test"); err == nil {
		t.Error("expected error adding note without session")
	}

	_ = fm.Start("coding", 25*time.Minute)
	if err := fm.AddNote("found a bug"); err != nil {
		t.Fatalf("AddNote failed: %v", err)
	}
	if err := fm.AddNote("fixed it"); err != nil {
		t.Fatalf("AddNote failed: %v", err)
	}

	s := fm.Status()
	if len(s.Notes) != 2 {
		t.Errorf("expected 2 notes, got %d", len(s.Notes))
	}
}

func TestFocusManager_History(t *testing.T) {
	dir := t.TempDir()
	fm := NewFocusManager(filepath.Join(dir, "focus.json"))

	// Complete two sessions
	_ = fm.Start("task one", 25*time.Minute)
	fm.Stop()

	_ = fm.Start("task two", 15*time.Minute)
	fm.Stop()

	hist := fm.History()
	if len(hist) != 2 {
		t.Errorf("expected 2 history entries, got %d", len(hist))
	}
	// Most recent first
	if hist[0].Task != "task two" {
		t.Errorf("expected most recent first, got %q", hist[0].Task)
	}
}

func TestFocusManager_Active(t *testing.T) {
	dir := t.TempDir()
	fm := NewFocusManager(filepath.Join(dir, "focus.json"))

	if fm.Active() {
		t.Error("expected not active initially")
	}

	_ = fm.Start("test", 25*time.Minute)
	if !fm.Active() {
		t.Error("expected active after start")
	}

	fm.Stop()
	if fm.Active() {
		t.Error("expected not active after stop")
	}
}

func TestFocusManager_PromptTag(t *testing.T) {
	dir := t.TempDir()
	fm := NewFocusManager(filepath.Join(dir, "focus.json"))

	if tag := fm.PromptTag(); tag != "" {
		t.Errorf("expected empty prompt tag, got %q", tag)
	}

	_ = fm.Start("test", 25*time.Minute)
	tag := fm.PromptTag()
	if tag == "" {
		t.Error("expected non-empty prompt tag during session")
	}
}

func TestFocusManager_Persistence(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "focus.json")

	fm := NewFocusManager(path)
	_ = fm.Start("persistent task", 25*time.Minute)

	// Load into new manager
	fm2 := NewFocusManager(path)
	s := fm2.Status()
	if s == nil {
		t.Fatal("expected active session after reload")
	}
	if s.Task != "persistent task" {
		t.Errorf("expected task 'persistent task', got %q", s.Task)
	}
}

func TestFocusSession_Expired(t *testing.T) {
	s := &FocusSession{
		StartTime: time.Now().Add(-30 * time.Minute),
		Duration:  25 * time.Minute,
	}
	if !s.IsExpired() {
		t.Error("expected session to be expired")
	}
	if s.Remaining() != 0 {
		t.Error("expected zero remaining")
	}
}

func TestFormatSummary(t *testing.T) {
	s := &FocusSession{
		Task:      "testing",
		StartTime: time.Now().Add(-20 * time.Minute),
		EndTime:   time.Now(),
		Duration:  25 * time.Minute,
		Notes:     []string{"note one", "note two"},
		Completed: true,
	}
	out := FormatSummary(s)
	if out == "" {
		t.Error("expected non-empty summary")
	}

	// Nil should be empty
	if FormatSummary(nil) != "" {
		t.Error("expected empty for nil")
	}
}
