package memory

import (
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"time"
)

// UndoEntry records a single file modification that can be rolled back.
type UndoEntry struct {
	Path      string    `json:"path"`
	Action    string    `json:"action"` // "write", "edit", "mkdir"
	Before    string    `json:"before"` // original content (empty for new files)
	After     string    `json:"after"`  // new content
	Timestamp time.Time `json:"timestamp"`
	WasNew    bool      `json:"was_new"` // true if file didn't exist before
}

// UndoStack tracks file modifications so they can be rolled back.
type UndoStack struct {
	mu      sync.Mutex
	entries []UndoEntry
	maxSize int
}

// NewUndoStack creates an undo stack with the given maximum capacity.
func NewUndoStack(maxSize int) *UndoStack {
	return &UndoStack{
		entries: make([]UndoEntry, 0),
		maxSize: maxSize,
	}
}

// Push records a change. If the stack is at capacity, the oldest entry is dropped.
func (u *UndoStack) Push(entry UndoEntry) {
	u.mu.Lock()
	defer u.mu.Unlock()

	if entry.Timestamp.IsZero() {
		entry.Timestamp = time.Now()
	}

	u.entries = append(u.entries, entry)

	if len(u.entries) > u.maxSize {
		u.entries = u.entries[len(u.entries)-u.maxSize:]
	}
}

// Pop removes and returns the most recent entry.
func (u *UndoStack) Pop() (UndoEntry, bool) {
	u.mu.Lock()
	defer u.mu.Unlock()

	if len(u.entries) == 0 {
		return UndoEntry{}, false
	}

	last := u.entries[len(u.entries)-1]
	u.entries = u.entries[:len(u.entries)-1]
	return last, true
}

// Undo reverts the most recent file change.
func (u *UndoStack) Undo() (string, error) {
	entry, ok := u.Pop()
	if !ok {
		return "", fmt.Errorf("nothing to undo")
	}

	if entry.WasNew {
		if err := os.Remove(entry.Path); err != nil && !os.IsNotExist(err) {
			return "", fmt.Errorf("undo: remove %s: %w", entry.Path, err)
		}
		return fmt.Sprintf("undone: removed %s (was newly created)", entry.Path), nil
	}

	if err := os.MkdirAll(filepath.Dir(entry.Path), 0755); err != nil {
		return "", fmt.Errorf("undo: mkdir for %s: %w", entry.Path, err)
	}

	if err := os.WriteFile(entry.Path, []byte(entry.Before), 0644); err != nil {
		return "", fmt.Errorf("undo: restore %s: %w", entry.Path, err)
	}

	return fmt.Sprintf("undone: restored %s (%s)", entry.Path, entry.Action), nil
}

// Peek returns the most recent entry without removing it.
func (u *UndoStack) Peek() (UndoEntry, bool) {
	u.mu.Lock()
	defer u.mu.Unlock()

	if len(u.entries) == 0 {
		return UndoEntry{}, false
	}

	return u.entries[len(u.entries)-1], true
}

// List returns all entries, newest first.
func (u *UndoStack) List() []UndoEntry {
	u.mu.Lock()
	defer u.mu.Unlock()

	result := make([]UndoEntry, len(u.entries))
	for i, e := range u.entries {
		result[len(u.entries)-1-i] = e
	}
	return result
}

// Size returns the number of entries in the stack.
func (u *UndoStack) Size() int {
	u.mu.Lock()
	defer u.mu.Unlock()
	return len(u.entries)
}

// Clear removes all entries from the stack.
func (u *UndoStack) Clear() {
	u.mu.Lock()
	defer u.mu.Unlock()
	u.entries = u.entries[:0]
}
