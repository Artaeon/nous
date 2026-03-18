package cognitive

import (
	"fmt"
	"os/exec"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"time"
)

// Reminder represents a scheduled reminder.
type Reminder struct {
	ID      int
	Message string
	FireAt  time.Time
	Fired   bool
}

// ReminderManager manages in-process reminders.
type ReminderManager struct {
	reminders  []Reminder
	mu         sync.Mutex
	notifyFunc func(msg string)
	nextID     int
	cancelMap  map[int]chan struct{} // channels to cancel pending reminders
}

// NewReminderManager creates a new ReminderManager with a default notification function.
func NewReminderManager() *ReminderManager {
	return &ReminderManager{
		notifyFunc: func(msg string) {
			fmt.Printf("[REMINDER] %s\n", msg)
		},
		cancelMap: make(map[int]chan struct{}),
	}
}

// NewReminderManagerWithNotify creates a new ReminderManager with a custom notification function.
func NewReminderManagerWithNotify(notify func(msg string)) *ReminderManager {
	return &ReminderManager{
		notifyFunc: notify,
		cancelMap:  make(map[int]chan struct{}),
	}
}

// AddReminder adds a reminder that fires after the given duration.
func (rm *ReminderManager) AddReminder(message string, duration time.Duration) *Reminder {
	return rm.AddReminderAt(message, time.Now().Add(duration))
}

// AddReminderAt adds a reminder that fires at a specific time.
func (rm *ReminderManager) AddReminderAt(message string, at time.Time) *Reminder {
	rm.mu.Lock()
	rm.nextID++
	r := Reminder{
		ID:      rm.nextID,
		Message: message,
		FireAt:  at,
		Fired:   false,
	}
	rm.reminders = append(rm.reminders, r)
	cancelCh := make(chan struct{})
	rm.cancelMap[r.ID] = cancelCh
	rm.mu.Unlock()

	go rm.scheduleReminder(r.ID, at, cancelCh)

	return &r
}

func (rm *ReminderManager) scheduleReminder(id int, at time.Time, cancelCh chan struct{}) {
	delay := time.Until(at)
	if delay < 0 {
		delay = 0
	}

	timer := time.NewTimer(delay)
	defer timer.Stop()

	select {
	case <-timer.C:
		rm.fireReminder(id)
	case <-cancelCh:
		// Cancelled
	}
}

func (rm *ReminderManager) fireReminder(id int) {
	rm.mu.Lock()
	var message string
	for i := range rm.reminders {
		if rm.reminders[i].ID == id && !rm.reminders[i].Fired {
			rm.reminders[i].Fired = true
			message = rm.reminders[i].Message
			break
		}
	}
	delete(rm.cancelMap, id)
	notify := rm.notifyFunc
	rm.mu.Unlock()

	if message != "" && notify != nil {
		notify(message)
		// Try desktop notification (best-effort)
		sendDesktopNotification("Nous Reminder", message)
	}
}

// sendDesktopNotification attempts to send a desktop notification via notify-send.
func sendDesktopNotification(title, body string) {
	cmd := exec.Command("notify-send", title, body)
	_ = cmd.Run() // Best effort, ignore errors
}

// ListReminders returns all active (unfired) reminders.
func (rm *ReminderManager) ListReminders() []Reminder {
	rm.mu.Lock()
	defer rm.mu.Unlock()

	var active []Reminder
	for _, r := range rm.reminders {
		if !r.Fired {
			active = append(active, r)
		}
	}
	return active
}

// CancelReminder cancels a reminder by ID. Returns true if found and cancelled.
func (rm *ReminderManager) CancelReminder(id int) bool {
	rm.mu.Lock()
	defer rm.mu.Unlock()

	for i := range rm.reminders {
		if rm.reminders[i].ID == id && !rm.reminders[i].Fired {
			rm.reminders[i].Fired = true // Mark as fired so it won't appear in ListReminders
			if ch, ok := rm.cancelMap[id]; ok {
				close(ch)
				delete(rm.cancelMap, id)
			}
			return true
		}
	}
	return false
}

// ParseDuration on ReminderManager delegates to the package-level ParseDuration.
func (rm *ReminderManager) ParseDuration(input string) (time.Duration, error) {
	return ParseDuration(input)
}

// reDurationPart matches individual duration components like "2 hours", "30 min".
var reDurationPart = regexp.MustCompile(`(\d+)\s*(seconds?|secs?|s|minutes?|mins?|min|hours?|hrs?|h|days?|d|weeks?|w)`)

// ParseDuration parses a human-readable duration string.
// Supports: "30 minutes", "2 hours", "1 hour 30 minutes", "45 seconds", "in 5 min"
func ParseDuration(input string) (time.Duration, error) {
	input = strings.ToLower(strings.TrimSpace(input))
	// Strip leading "in " if present
	input = strings.TrimPrefix(input, "in ")
	input = strings.TrimSpace(input)

	matches := reDurationPart.FindAllStringSubmatch(input, -1)
	if len(matches) == 0 {
		return 0, fmt.Errorf("could not parse duration from %q — expected format like '30 minutes', '2 hours 15 min'", input)
	}

	var total time.Duration
	for _, m := range matches {
		n, err := strconv.Atoi(m[1])
		if err != nil {
			continue
		}
		unit := m[2]
		switch {
		case strings.HasPrefix(unit, "s"):
			total += time.Duration(n) * time.Second
		case strings.HasPrefix(unit, "min") || unit == "m":
			// "m" alone is ambiguous but we skip it since the regex requires at least "min"
			total += time.Duration(n) * time.Minute
		case strings.HasPrefix(unit, "h"):
			total += time.Duration(n) * time.Hour
		case strings.HasPrefix(unit, "d"):
			total += time.Duration(n) * 24 * time.Hour
		case strings.HasPrefix(unit, "w"):
			total += time.Duration(n) * 7 * 24 * time.Hour
		}
	}

	if total == 0 {
		return 0, fmt.Errorf("parsed zero duration from %q", input)
	}

	return total, nil
}
