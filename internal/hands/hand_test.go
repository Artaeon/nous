package hands

import (
	"testing"
	"time"
)

func TestDefaultConfig(t *testing.T) {
	cfg := DefaultConfig()
	if cfg.MaxSteps != 8 {
		t.Errorf("DefaultConfig.MaxSteps = %d, want 8", cfg.MaxSteps)
	}
	if cfg.Timeout != 120 {
		t.Errorf("DefaultConfig.Timeout = %d, want 120", cfg.Timeout)
	}
	if cfg.RequiresApproval {
		t.Error("DefaultConfig.RequiresApproval should be false")
	}
	if len(cfg.Tools) != 0 {
		t.Errorf("DefaultConfig.Tools should be empty, got %v", cfg.Tools)
	}
}

func TestHandState(t *testing.T) {
	states := []HandState{HandIdle, HandRunning, HandPaused, HandFailed, HandCompleted}
	expected := []string{"idle", "running", "paused", "failed", "completed"}
	for i, s := range states {
		if string(s) != expected[i] {
			t.Errorf("HandState %d = %q, want %q", i, s, expected[i])
		}
	}
}

func TestHandResultDuration(t *testing.T) {
	r := HandResult{
		Output:    "done",
		Duration:  2 * time.Second,
		ToolCalls: 3,
	}
	if r.Duration != 2*time.Second {
		t.Errorf("Duration = %v, want 2s", r.Duration)
	}
	if r.ToolCalls != 3 {
		t.Errorf("ToolCalls = %d, want 3", r.ToolCalls)
	}
	if r.Error != "" {
		t.Errorf("Error should be empty, got %q", r.Error)
	}
}

func TestRunRecordFields(t *testing.T) {
	now := time.Now()
	rec := RunRecord{
		HandName:  "researcher",
		StartedAt: now,
		Duration:  5000,
		Success:   true,
		Output:    "report generated",
		ToolCalls: 4,
	}
	if rec.HandName != "researcher" {
		t.Errorf("HandName = %q, want researcher", rec.HandName)
	}
	if rec.Duration != 5000 {
		t.Errorf("Duration = %d, want 5000", rec.Duration)
	}
	if !rec.Success {
		t.Error("Success should be true")
	}
}
