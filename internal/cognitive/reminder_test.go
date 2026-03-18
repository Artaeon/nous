package cognitive

import (
	"sync"
	"testing"
	"time"
)

func TestParseDuration(t *testing.T) {
	tests := []struct {
		input    string
		expected time.Duration
	}{
		{"30 minutes", 30 * time.Minute},
		{"2 hours", 2 * time.Hour},
		{"1 hour 30 minutes", 90 * time.Minute},
		{"45 seconds", 45 * time.Second},
		{"in 5 min", 5 * time.Minute},
		{"10 secs", 10 * time.Second},
		{"1 day", 24 * time.Hour},
		{"2 weeks", 14 * 24 * time.Hour},
		{"3 hrs", 3 * time.Hour},
		{"1 hour 15 min 30 seconds", time.Hour + 15*time.Minute + 30*time.Second},
	}

	for _, tt := range tests {
		result, err := ParseDuration(tt.input)
		if err != nil {
			t.Errorf("ParseDuration(%q) error: %v", tt.input, err)
			continue
		}
		if result != tt.expected {
			t.Errorf("ParseDuration(%q) = %v, want %v", tt.input, result, tt.expected)
		}
	}
}

func TestParseDurationInvalid(t *testing.T) {
	_, err := ParseDuration("hello world")
	if err == nil {
		t.Error("expected error for unparseable duration")
	}

	_, err = ParseDuration("")
	if err == nil {
		t.Error("expected error for empty string")
	}
}

func TestAddReminderFires(t *testing.T) {
	var mu sync.Mutex
	var received string

	rm := NewReminderManagerWithNotify(func(msg string) {
		mu.Lock()
		received = msg
		mu.Unlock()
	})

	rm.AddReminder("test reminder", 50*time.Millisecond)

	// Wait for it to fire
	time.Sleep(150 * time.Millisecond)

	mu.Lock()
	got := received
	mu.Unlock()

	if got != "test reminder" {
		t.Errorf("expected 'test reminder', got %q", got)
	}
}

func TestAddReminderAt(t *testing.T) {
	var mu sync.Mutex
	var received string

	rm := NewReminderManagerWithNotify(func(msg string) {
		mu.Lock()
		received = msg
		mu.Unlock()
	})

	rm.AddReminderAt("at reminder", time.Now().Add(50*time.Millisecond))

	time.Sleep(150 * time.Millisecond)

	mu.Lock()
	got := received
	mu.Unlock()

	if got != "at reminder" {
		t.Errorf("expected 'at reminder', got %q", got)
	}
}

func TestListReminders(t *testing.T) {
	rm := NewReminderManagerWithNotify(func(msg string) {})

	rm.AddReminder("first", 1*time.Hour)
	rm.AddReminder("second", 2*time.Hour)

	active := rm.ListReminders()
	if len(active) != 2 {
		t.Errorf("expected 2 active reminders, got %d", len(active))
	}
}

func TestCancelReminder(t *testing.T) {
	var mu sync.Mutex
	var received string

	rm := NewReminderManagerWithNotify(func(msg string) {
		mu.Lock()
		received = msg
		mu.Unlock()
	})

	r := rm.AddReminder("cancelled", 100*time.Millisecond)

	// Cancel before it fires
	ok := rm.CancelReminder(r.ID)
	if !ok {
		t.Error("CancelReminder returned false")
	}

	// Verify it's gone from active list
	active := rm.ListReminders()
	if len(active) != 0 {
		t.Errorf("expected 0 active reminders after cancel, got %d", len(active))
	}

	// Wait past the fire time
	time.Sleep(200 * time.Millisecond)

	mu.Lock()
	got := received
	mu.Unlock()

	if got != "" {
		t.Errorf("cancelled reminder should not fire, but got %q", got)
	}
}

func TestCancelNonExistentReminder(t *testing.T) {
	rm := NewReminderManagerWithNotify(func(msg string) {})

	ok := rm.CancelReminder(9999)
	if ok {
		t.Error("CancelReminder should return false for non-existent ID")
	}
}

func TestReminderIDsIncrement(t *testing.T) {
	rm := NewReminderManagerWithNotify(func(msg string) {})

	r1 := rm.AddReminder("a", 1*time.Hour)
	r2 := rm.AddReminder("b", 1*time.Hour)

	if r2.ID <= r1.ID {
		t.Errorf("IDs should increment: r1=%d, r2=%d", r1.ID, r2.ID)
	}
}

func TestListRemindersExcludesFired(t *testing.T) {
	rm := NewReminderManagerWithNotify(func(msg string) {})

	rm.AddReminder("fires soon", 10*time.Millisecond)
	rm.AddReminder("fires later", 1*time.Hour)

	// Wait for first to fire
	time.Sleep(50 * time.Millisecond)

	active := rm.ListReminders()
	if len(active) != 1 {
		t.Errorf("expected 1 active reminder, got %d", len(active))
	}
	if len(active) > 0 && active[0].Message != "fires later" {
		t.Errorf("expected 'fires later', got %q", active[0].Message)
	}
}
