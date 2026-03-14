package cli

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/artaeon/nous/internal/safefile"
)

// FocusSession tracks a single Pomodoro-style focus block.
type FocusSession struct {
	Task      string    `json:"task"`
	StartTime time.Time `json:"start_time"`
	Duration  time.Duration `json:"duration"`
	EndTime   time.Time `json:"end_time,omitempty"`
	Breaks    int       `json:"breaks"`
	Notes     []string  `json:"notes,omitempty"`
	Completed bool      `json:"completed"`
}

// Remaining returns how much time is left in the session.
func (fs *FocusSession) Remaining() time.Duration {
	if fs == nil {
		return 0
	}
	end := fs.StartTime.Add(fs.Duration)
	rem := time.Until(end)
	if rem < 0 {
		return 0
	}
	return rem
}

// IsExpired returns true if the focus timer has run out.
func (fs *FocusSession) IsExpired() bool {
	if fs == nil {
		return true
	}
	return time.Now().After(fs.StartTime.Add(fs.Duration))
}

// FocusManager handles focus sessions with persistence.
type FocusManager struct {
	mu      sync.Mutex
	current *FocusSession
	history []FocusSession
	file    string
}

// NewFocusManager creates a focus manager that persists to the given path.
func NewFocusManager(path string) *FocusManager {
	fm := &FocusManager{
		file: path,
	}
	_ = fm.load()
	return fm
}

// Start begins a new focus session. Returns an error if one is already active.
func (fm *FocusManager) Start(task string, duration time.Duration) error {
	fm.mu.Lock()
	defer fm.mu.Unlock()

	if fm.current != nil && !fm.current.IsExpired() {
		return fmt.Errorf("focus session already active: %q (%s remaining)",
			fm.current.Task, fm.current.Remaining().Truncate(time.Second))
	}

	// If there's an expired session, archive it
	if fm.current != nil {
		fm.archiveCurrent()
	}

	fm.current = &FocusSession{
		Task:      task,
		StartTime: time.Now(),
		Duration:  duration,
	}
	return fm.save()
}

// Stop ends the current focus session and returns it for summary display.
func (fm *FocusManager) Stop() *FocusSession {
	fm.mu.Lock()
	defer fm.mu.Unlock()

	if fm.current == nil {
		return nil
	}

	session := *fm.current
	session.EndTime = time.Now()
	session.Completed = true
	fm.history = append(fm.history, session)
	fm.current = nil
	_ = fm.save()
	return &session
}

// Status returns the current focus session, or nil if none is active.
func (fm *FocusManager) Status() *FocusSession {
	fm.mu.Lock()
	defer fm.mu.Unlock()

	if fm.current == nil {
		return nil
	}

	// Auto-expire
	if fm.current.IsExpired() {
		fm.archiveCurrent()
		_ = fm.save()
		return nil
	}

	cp := *fm.current
	return &cp
}

// Active returns true if a focus session is currently running.
func (fm *FocusManager) Active() bool {
	return fm.Status() != nil
}

// AddNote appends a note to the current focus session.
func (fm *FocusManager) AddNote(note string) error {
	fm.mu.Lock()
	defer fm.mu.Unlock()

	if fm.current == nil {
		return fmt.Errorf("no active focus session")
	}
	fm.current.Notes = append(fm.current.Notes, note)
	return fm.save()
}

// History returns all completed focus sessions, most recent first.
func (fm *FocusManager) History() []FocusSession {
	fm.mu.Lock()
	defer fm.mu.Unlock()

	out := make([]FocusSession, len(fm.history))
	copy(out, fm.history)

	// Reverse for most recent first
	for i, j := 0, len(out)-1; i < j; i, j = i+1, j-1 {
		out[i], out[j] = out[j], out[i]
	}
	return out
}

// PromptTag returns a short string for the REPL prompt when focus is active,
// e.g. "[15:23 remaining]". Returns empty string if no session.
func (fm *FocusManager) PromptTag() string {
	s := fm.Status()
	if s == nil {
		return ""
	}
	rem := s.Remaining().Truncate(time.Second)
	mins := int(rem.Minutes())
	secs := int(rem.Seconds()) % 60
	return fmt.Sprintf("[%02d:%02d remaining]", mins, secs)
}

// FormatSummary produces a readable summary of a completed focus session.
func FormatSummary(s *FocusSession) string {
	if s == nil {
		return ""
	}
	var b strings.Builder
	elapsed := s.EndTime.Sub(s.StartTime).Truncate(time.Second)
	b.WriteString(fmt.Sprintf("  Task:     %s\n", s.Task))
	b.WriteString(fmt.Sprintf("  Duration: %s (of %s planned)\n", elapsed, s.Duration.Truncate(time.Second)))
	if len(s.Notes) > 0 {
		b.WriteString("  Notes:\n")
		for _, n := range s.Notes {
			b.WriteString(fmt.Sprintf("    - %s\n", n))
		}
	}
	return b.String()
}

func (fm *FocusManager) archiveCurrent() {
	if fm.current == nil {
		return
	}
	archived := *fm.current
	archived.EndTime = fm.current.StartTime.Add(fm.current.Duration)
	archived.Completed = fm.current.IsExpired()
	fm.history = append(fm.history, archived)
	fm.current = nil
}

func (fm *FocusManager) save() error {
	if fm.file == "" {
		return nil
	}
	state := struct {
		Current *FocusSession  `json:"current,omitempty"`
		History []FocusSession `json:"history"`
	}{
		Current: fm.current,
		History: fm.history,
	}
	data, err := json.MarshalIndent(state, "", "  ")
	if err != nil {
		return err
	}
	return safefile.WriteAtomic(fm.file, data, 0644)
}

func (fm *FocusManager) load() error {
	if fm.file == "" {
		return nil
	}
	data, err := os.ReadFile(fm.file)
	if err != nil {
		return err
	}
	var state struct {
		Current *FocusSession  `json:"current,omitempty"`
		History []FocusSession `json:"history"`
	}
	if err := json.Unmarshal(data, &state); err != nil {
		return err
	}
	fm.current = state.Current
	fm.history = state.History
	return nil
}
