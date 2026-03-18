package tools

import (
	"fmt"
	"os/exec"
	"strings"
	"sync"
	"time"
)

// timerEntry represents a single active timer.
type timerEntry struct {
	ID        string
	Name      string
	Duration  time.Duration
	StartTime time.Time
	EndTime   time.Time
	Done      chan struct{}
}

// timerManager manages active timers with thread-safe access.
type timerManager struct {
	mu     sync.Mutex
	timers map[string]*timerEntry
	nextID int
}

var globalTimerManager = &timerManager{
	timers: make(map[string]*timerEntry),
}

// newTimerManager creates a new timer manager (for testing).
func newTimerManager() *timerManager {
	return &timerManager{
		timers: make(map[string]*timerEntry),
	}
}

// StartTimer parses the duration string, creates a timer, and launches a goroutine
// that sends a desktop notification when it fires.
func (tm *timerManager) StartTimer(durationStr, name string) (string, error) {
	d, err := time.ParseDuration(durationStr)
	if err != nil {
		return "", fmt.Errorf("timer: invalid duration %q: %w", durationStr, err)
	}
	if d <= 0 {
		return "", fmt.Errorf("timer: duration must be positive")
	}

	tm.mu.Lock()
	tm.nextID++
	id := fmt.Sprintf("t%d", tm.nextID)
	if name == "" {
		name = fmt.Sprintf("Timer %s", id)
	}

	now := time.Now()
	entry := &timerEntry{
		ID:        id,
		Name:      name,
		Duration:  d,
		StartTime: now,
		EndTime:   now.Add(d),
		Done:      make(chan struct{}),
	}
	tm.timers[id] = entry
	tm.mu.Unlock()

	go func() {
		select {
		case <-time.After(d):
			// Timer fired — send desktop notification.
			_ = exec.Command("notify-send", "Nous Timer", fmt.Sprintf("Timer '%s' finished (%s)", entry.Name, entry.Duration)).Run()
		case <-entry.Done:
			// Timer was cancelled.
		}
		tm.mu.Lock()
		delete(tm.timers, id)
		tm.mu.Unlock()
	}()

	return fmt.Sprintf("Timer '%s' started: %s (fires at %s) [id=%s]",
		name, d, entry.EndTime.Format("15:04"), id), nil
}

// StopTimer cancels an active timer by ID or name.
func (tm *timerManager) StopTimer(identifier string) (string, error) {
	tm.mu.Lock()
	defer tm.mu.Unlock()

	// Try by ID first.
	if entry, ok := tm.timers[identifier]; ok {
		close(entry.Done)
		delete(tm.timers, identifier)
		return fmt.Sprintf("Timer '%s' stopped.", entry.Name), nil
	}

	// Try by name (case-insensitive partial match).
	lowerID := strings.ToLower(identifier)
	for id, entry := range tm.timers {
		if strings.Contains(strings.ToLower(entry.Name), lowerID) {
			close(entry.Done)
			delete(tm.timers, id)
			return fmt.Sprintf("Timer '%s' stopped.", entry.Name), nil
		}
	}

	return "", fmt.Errorf("timer: no active timer matching %q", identifier)
}

// StatusTimer shows time remaining for a specific timer.
func (tm *timerManager) StatusTimer(identifier string) (string, error) {
	tm.mu.Lock()
	defer tm.mu.Unlock()

	entry := tm.findTimer(identifier)
	if entry == nil {
		return "", fmt.Errorf("timer: no active timer matching %q", identifier)
	}

	remaining := time.Until(entry.EndTime)
	if remaining < 0 {
		remaining = 0
	}
	return fmt.Sprintf("Timer '%s': %s remaining (fires at %s)",
		entry.Name, remaining.Round(time.Second), entry.EndTime.Format("15:04")), nil
}

// ListTimers returns all active timers with remaining time.
func (tm *timerManager) ListTimers() (string, error) {
	tm.mu.Lock()
	defer tm.mu.Unlock()

	if len(tm.timers) == 0 {
		return "No active timers.", nil
	}

	var sb strings.Builder
	fmt.Fprintf(&sb, "%d active timer(s):\n", len(tm.timers))
	for _, entry := range tm.timers {
		remaining := time.Until(entry.EndTime)
		if remaining < 0 {
			remaining = 0
		}
		fmt.Fprintf(&sb, "- [%s] '%s': %s remaining (fires at %s)\n",
			entry.ID, entry.Name, remaining.Round(time.Second), entry.EndTime.Format("15:04"))
	}
	return sb.String(), nil
}

// findTimer locates a timer by ID or name (case-insensitive partial match).
// Must be called with tm.mu held.
func (tm *timerManager) findTimer(identifier string) *timerEntry {
	if entry, ok := tm.timers[identifier]; ok {
		return entry
	}
	lowerID := strings.ToLower(identifier)
	for _, entry := range tm.timers {
		if strings.Contains(strings.ToLower(entry.Name), lowerID) {
			return entry
		}
	}
	return nil
}

// RegisterTimerTools adds the timer tool to the registry.
func RegisterTimerTools(r *Registry) {
	r.Register(Tool{
		Name:        "timer",
		Description: "Countdown timer and stopwatch. Args: action (start/stop/status/list/pomodoro), duration (e.g. '5m', '1h30m'), name (optional label).",
		Execute: func(args map[string]string) (string, error) {
			return toolTimer(globalTimerManager, args)
		},
	})
}

func toolTimer(tm *timerManager, args map[string]string) (string, error) {
	action := strings.ToLower(strings.TrimSpace(args["action"]))

	switch action {
	case "start":
		duration := args["duration"]
		if duration == "" {
			return "", fmt.Errorf("timer start requires 'duration' (e.g. '5m', '1h30m')")
		}
		return tm.StartTimer(duration, args["name"])

	case "stop":
		name := args["name"]
		if name == "" {
			return "", fmt.Errorf("timer stop requires 'name' or timer ID")
		}
		return tm.StopTimer(name)

	case "status":
		name := args["name"]
		if name == "" {
			return "", fmt.Errorf("timer status requires 'name' or timer ID")
		}
		return tm.StatusTimer(name)

	case "list":
		return tm.ListTimers()

	case "pomodoro":
		name := args["name"]
		if name == "" {
			name = "Pomodoro"
		}
		return tm.StartTimer("25m", name)

	default:
		return "", fmt.Errorf("timer: unknown action %q (use start/stop/status/list/pomodoro)", action)
	}
}
