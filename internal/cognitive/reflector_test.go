package cognitive

import (
	"context"
	"testing"
	"time"

	"github.com/artaeon/nous/internal/blackboard"
)

func TestReflectorDetectsFailure(t *testing.T) {
	board := blackboard.New()
	r := NewReflector(board, nil)

	action := blackboard.ActionRecord{
		StepID:  "step-1",
		Tool:    "read",
		Input:   "nonexistent.go",
		Output:  "Error: no such file or directory",
		Success: false,
	}

	r.reflect(action)

	reflection, ok := board.Get("reflection")
	if !ok {
		t.Fatal("Reflector should post reflection on failure")
	}
	msg := reflection.(string)
	if msg == "" {
		t.Error("reflection message should not be empty")
	}
}

func TestReflectorConsecutiveFailuresTriggerReplan(t *testing.T) {
	board := blackboard.New()
	r := NewReflector(board, nil)

	for i := 0; i < 3; i++ {
		r.reflect(blackboard.ActionRecord{
			StepID:  "step",
			Tool:    "read",
			Output:  "Error: permission denied",
			Success: false,
		})
	}

	replan, ok := board.Get("needs_replan")
	if !ok {
		t.Fatal("3 consecutive failures should trigger needs_replan")
	}
	if replan.(string) != "step" {
		t.Errorf("expected step ID 'step', got %v", replan)
	}
}

func TestReflectorDetectsRepeatedTool(t *testing.T) {
	board := blackboard.New()
	r := NewReflector(board, nil)

	// Two consecutive grep calls should trigger a warning
	r.reflect(blackboard.ActionRecord{
		Tool: "grep", Output: "found something", Success: true,
	})
	r.reflect(blackboard.ActionRecord{
		Tool: "grep", Output: "found more", Success: true,
	})

	reflection, ok := board.Get("reflection")
	if !ok {
		t.Fatal("repeated tool should trigger reflection")
	}
	msg := reflection.(string)
	if msg == "" {
		t.Error("reflection should warn about repeated tool")
	}
}

func TestReflectorAllowsRepeatedRead(t *testing.T) {
	board := blackboard.New()
	r := NewReflector(board, nil)

	// Reading multiple files is normal
	r.reflect(blackboard.ActionRecord{
		Tool: "read", Output: "file content A", Success: true,
	})
	r.reflect(blackboard.ActionRecord{
		Tool: "read", Output: "file content B", Success: true,
	})

	_, ok := board.Get("reflection")
	if ok {
		t.Error("repeated read calls should be allowed (reading multiple files is normal)")
	}
}

func TestReflectorResetsFailCountOnSuccess(t *testing.T) {
	board := blackboard.New()
	r := NewReflector(board, nil)

	r.reflect(blackboard.ActionRecord{
		Tool: "read", Output: "Error: not found", Success: false,
	})
	r.reflect(blackboard.ActionRecord{
		Tool: "read", Output: "Error: not found", Success: false,
	})

	// Success resets the counter
	r.reflect(blackboard.ActionRecord{
		Tool: "read", Output: "file content", Success: true,
	})
	board.Delete("reflection")

	// One more failure should NOT trigger 3-consecutive (was reset)
	r.reflect(blackboard.ActionRecord{
		StepID: "new", Tool: "read", Output: "Error: not found", Success: false,
	})

	_, ok := board.Get("needs_replan")
	if ok {
		t.Error("success should reset consecutive failure counter")
	}
}

func TestReflectorDiagnosesSpecificErrors(t *testing.T) {
	board := blackboard.New()
	r := NewReflector(board, nil)

	tests := []struct {
		output   string
		contains string
	}{
		{"Error: no such file or directory", "ls or glob"},
		{"Error: permission denied", "permission"},
		{"Error: is a directory", "ls"},
		{"Error: match is not unique, found 3 times in file.go", "ambiguous"},
	}

	for _, tt := range tests {
		board.Delete("reflection")
		r.consecutiveFails = 0

		r.reflect(blackboard.ActionRecord{
			Tool: "read", Output: tt.output, Success: false,
		})

		reflection, ok := board.Get("reflection")
		if !ok {
			t.Errorf("expected reflection for output %q", tt.output)
			continue
		}
		msg := reflection.(string)
		if !contains(msg, tt.contains) {
			t.Errorf("reflection %q should contain %q", msg, tt.contains)
		}
	}
}

func TestReflectorStats(t *testing.T) {
	board := blackboard.New()
	r := NewReflector(board, nil)

	r.reflect(blackboard.ActionRecord{Tool: "read", Output: "ok", Success: true})
	r.reflect(blackboard.ActionRecord{Tool: "read", Output: "err", Success: false})

	checks, issues := r.Stats()
	if checks != 2 {
		t.Errorf("expected 2 checks, got %d", checks)
	}
	if issues != 1 {
		t.Errorf("expected 1 issue, got %d", issues)
	}
}

func TestReflectorRunsAsStream(t *testing.T) {
	board := blackboard.New()
	r := NewReflector(board, nil)

	ctx, cancel := context.WithCancel(context.Background())
	go r.Run(ctx)

	// Give it time to subscribe
	time.Sleep(50 * time.Millisecond)

	// Post an action — the Reflector should pick it up
	board.RecordAction(blackboard.ActionRecord{
		StepID:  "test-1",
		Tool:    "grep",
		Output:  "Error: permission denied",
		Success: false,
	})

	time.Sleep(100 * time.Millisecond)
	cancel()

	// Check that the Reflector posted a reflection
	reflection, ok := board.Get("reflection")
	if !ok {
		t.Fatal("Reflector should react to action_recorded events")
	}
	if reflection.(string) == "" {
		t.Error("reflection should not be empty")
	}
}

func contains(s, substr string) bool {
	return len(s) >= len(substr) && (s == substr || len(s) > 0 && containsLower(s, substr))
}

func containsLower(s, substr string) bool {
	for i := 0; i+len(substr) <= len(s); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}
