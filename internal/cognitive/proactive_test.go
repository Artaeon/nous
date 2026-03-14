package cognitive

import (
	"testing"
	"time"

	"github.com/artaeon/nous/internal/blackboard"
)

func TestProactiveEngine_CooldownRespected(t *testing.T) {
	board := blackboard.New()
	pe := NewProactiveEngine(board)

	// First check should work (if conditions met)
	_ = pe.Check()

	// Immediate second check should return nil (cooldown)
	suggestions := pe.Check()
	if suggestions != nil {
		t.Error("expected nil during cooldown")
	}
}

func TestProactiveEngine_ErrorPatternDetection(t *testing.T) {
	board := blackboard.New()
	pe := NewProactiveEngine(board)
	pe.SetCooldown(0) // disable cooldown for testing

	// Record 3 failed actions
	board.RecordAction(blackboard.ActionRecord{Tool: "shell", Success: false, Timestamp: time.Now()})
	board.RecordAction(blackboard.ActionRecord{Tool: "read_file", Success: false, Timestamp: time.Now()})
	board.RecordAction(blackboard.ActionRecord{Tool: "write_file", Success: false, Timestamp: time.Now()})

	suggestions := pe.Check()
	found := false
	for _, s := range suggestions {
		if s.Type == SuggestError {
			found = true
			break
		}
	}
	if !found {
		t.Error("expected error pattern suggestion after 3 failures")
	}
}

func TestProactiveEngine_IdleDetection(t *testing.T) {
	board := blackboard.New()
	pe := NewProactiveEngine(board)
	pe.SetCooldown(0)

	// Simulate 6 minutes of inactivity
	pe.mu.Lock()
	pe.lastInput = time.Now().Add(-6 * time.Minute)
	pe.mu.Unlock()

	suggestions := pe.Check()
	found := false
	for _, s := range suggestions {
		if s.Type == SuggestIdle {
			found = true
			break
		}
	}
	if !found {
		t.Error("expected idle suggestion after 6 minutes of inactivity")
	}
}

func TestProactiveEngine_Suppressed(t *testing.T) {
	board := blackboard.New()
	pe := NewProactiveEngine(board)
	pe.SetCooldown(0)
	pe.SetSuppressed(true)

	// Even with all conditions met, should return nil
	pe.mu.Lock()
	pe.lastInput = time.Now().Add(-10 * time.Minute)
	pe.mu.Unlock()

	suggestions := pe.Check()
	if suggestions != nil {
		t.Error("expected nil when suppressed")
	}
}

func TestProactiveEngine_RecordInput(t *testing.T) {
	board := blackboard.New()
	pe := NewProactiveEngine(board)

	// Set old last input
	pe.mu.Lock()
	pe.lastInput = time.Now().Add(-10 * time.Minute)
	pe.mu.Unlock()

	// Record fresh input
	pe.RecordInput()

	pe.SetCooldown(0)
	suggestions := pe.Check()
	// Should NOT get idle suggestion since we just recorded input
	for _, s := range suggestions {
		if s.Type == SuggestIdle {
			t.Error("should not get idle suggestion right after RecordInput")
		}
	}
}

func TestFormatSuggestions(t *testing.T) {
	out := FormatSuggestions(nil)
	if out != "" {
		t.Error("expected empty string for nil suggestions")
	}

	suggestions := []Suggestion{
		{Type: SuggestIdle, Message: "test message", Action: "/test"},
	}
	out = FormatSuggestions(suggestions)
	if out == "" {
		t.Error("expected non-empty output for suggestions")
	}
}
