package tools

import (
	"strings"
	"testing"
	"time"
)

func TestTimerDurationParsing(t *testing.T) {
	tm := newTimerManager()

	tests := []struct {
		duration string
		wantErr  bool
	}{
		{"5m", false},
		{"1h30m", false},
		{"25m", false},
		{"500ms", false},
		{"invalid", true},
		{"-5m", true},
		{"0s", true},
	}

	for _, tt := range tests {
		t.Run(tt.duration, func(t *testing.T) {
			result, err := tm.StartTimer(tt.duration, "test")
			if tt.wantErr {
				if err == nil {
					t.Errorf("expected error for duration %q, got: %s", tt.duration, result)
				}
			} else {
				if err != nil {
					t.Errorf("unexpected error for duration %q: %v", tt.duration, err)
				}
			}
		})
	}
}

func TestTimerStartListStop(t *testing.T) {
	tm := newTimerManager()

	// Start a timer.
	result, err := tm.StartTimer("10m", "Test Timer")
	if err != nil {
		t.Fatalf("StartTimer: %v", err)
	}
	if !strings.Contains(result, "Test Timer") {
		t.Errorf("start result should contain timer name, got: %s", result)
	}
	if !strings.Contains(result, "10m0s") {
		t.Errorf("start result should contain duration, got: %s", result)
	}
	if !strings.Contains(result, "fires at") {
		t.Errorf("start result should contain fire time, got: %s", result)
	}

	// List timers.
	list, err := tm.ListTimers()
	if err != nil {
		t.Fatalf("ListTimers: %v", err)
	}
	if !strings.Contains(list, "Test Timer") {
		t.Errorf("list should contain timer, got: %s", list)
	}
	if !strings.Contains(list, "1 active timer") {
		t.Errorf("list should show count, got: %s", list)
	}

	// Status.
	status, err := tm.StatusTimer("Test Timer")
	if err != nil {
		t.Fatalf("StatusTimer: %v", err)
	}
	if !strings.Contains(status, "Test Timer") {
		t.Errorf("status should contain timer name, got: %s", status)
	}
	if !strings.Contains(status, "remaining") {
		t.Errorf("status should contain 'remaining', got: %s", status)
	}

	// Stop the timer.
	stopResult, err := tm.StopTimer("Test Timer")
	if err != nil {
		t.Fatalf("StopTimer: %v", err)
	}
	if !strings.Contains(stopResult, "stopped") {
		t.Errorf("stop result should contain 'stopped', got: %s", stopResult)
	}

	// List should be empty now.
	// Give the goroutine a moment to clean up.
	time.Sleep(10 * time.Millisecond)
	list, err = tm.ListTimers()
	if err != nil {
		t.Fatalf("ListTimers after stop: %v", err)
	}
	if !strings.Contains(list, "No active timers") {
		t.Errorf("list should be empty after stop, got: %s", list)
	}
}

func TestTimerStopByID(t *testing.T) {
	tm := newTimerManager()

	result, err := tm.StartTimer("10m", "ID Test")
	if err != nil {
		t.Fatalf("StartTimer: %v", err)
	}

	// Extract ID from result.
	// Format: "Timer 'ID Test' started: 10m0s (fires at HH:MM) [id=t1]"
	idStart := strings.Index(result, "[id=")
	idEnd := strings.Index(result, "]")
	if idStart < 0 || idEnd < 0 {
		t.Fatalf("could not extract ID from: %s", result)
	}
	id := result[idStart+4 : idEnd]

	stopResult, err := tm.StopTimer(id)
	if err != nil {
		t.Fatalf("StopTimer by ID: %v", err)
	}
	if !strings.Contains(stopResult, "stopped") {
		t.Errorf("stop result should contain 'stopped', got: %s", stopResult)
	}
}

func TestTimerStopNotFound(t *testing.T) {
	tm := newTimerManager()

	_, err := tm.StopTimer("nonexistent")
	if err == nil {
		t.Error("expected error when stopping nonexistent timer")
	}
}

func TestTimerStatusNotFound(t *testing.T) {
	tm := newTimerManager()

	_, err := tm.StatusTimer("nonexistent")
	if err == nil {
		t.Error("expected error for status of nonexistent timer")
	}
}

func TestTimerListEmpty(t *testing.T) {
	tm := newTimerManager()

	list, err := tm.ListTimers()
	if err != nil {
		t.Fatalf("ListTimers: %v", err)
	}
	if list != "No active timers." {
		t.Errorf("expected empty message, got: %s", list)
	}
}

func TestTimerPomodoroShortcut(t *testing.T) {
	tm := newTimerManager()

	result, err := toolTimer(tm, map[string]string{"action": "pomodoro"})
	if err != nil {
		t.Fatalf("pomodoro: %v", err)
	}
	if !strings.Contains(result, "Pomodoro") {
		t.Errorf("pomodoro should use 'Pomodoro' name, got: %s", result)
	}
	if !strings.Contains(result, "25m0s") {
		t.Errorf("pomodoro should be 25m, got: %s", result)
	}

	// Clean up.
	tm.StopTimer("Pomodoro")
}

func TestTimerPomodoroCustomName(t *testing.T) {
	tm := newTimerManager()

	result, err := toolTimer(tm, map[string]string{"action": "pomodoro", "name": "Work Sprint"})
	if err != nil {
		t.Fatalf("pomodoro custom: %v", err)
	}
	if !strings.Contains(result, "Work Sprint") {
		t.Errorf("pomodoro should use custom name, got: %s", result)
	}

	// Clean up.
	tm.StopTimer("Work Sprint")
}

func TestTimerToolUnknownAction(t *testing.T) {
	tm := newTimerManager()

	_, err := toolTimer(tm, map[string]string{"action": "invalid"})
	if err == nil {
		t.Error("expected error for unknown action")
	}
}

func TestTimerToolMissingArgs(t *testing.T) {
	tm := newTimerManager()

	// Start without duration.
	_, err := toolTimer(tm, map[string]string{"action": "start"})
	if err == nil {
		t.Error("expected error for start without duration")
	}

	// Stop without name.
	_, err = toolTimer(tm, map[string]string{"action": "stop"})
	if err == nil {
		t.Error("expected error for stop without name")
	}

	// Status without name.
	_, err = toolTimer(tm, map[string]string{"action": "status"})
	if err == nil {
		t.Error("expected error for status without name")
	}
}

func TestTimerToolRegistration(t *testing.T) {
	r := NewRegistry()
	RegisterTimerTools(r)

	tool, err := r.Get("timer")
	if err != nil {
		t.Fatal("timer tool not registered")
	}
	if tool.Name != "timer" {
		t.Errorf("tool name = %q, want %q", tool.Name, "timer")
	}
}
