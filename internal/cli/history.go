package cli

import (
	"bufio"
	"os"
	"path/filepath"
	"strings"
	"sync"
)

// History provides persistent readline-like command history.
// Entries are stored one per line in a plain text file (default ~/.nous/history).
type History struct {
	entries  []string
	file     string
	maxSize  int
	position int
	mu       sync.Mutex
}

// NewHistory creates a History backed by the given file path.
// maxSize limits how many entries are kept (oldest are pruned on save).
func NewHistory(path string, maxSize int) *History {
	if maxSize <= 0 {
		maxSize = 1000
	}
	return &History{
		file:    path,
		maxSize: maxSize,
	}
}

// DefaultHistoryPath returns ~/.nous/history.
func DefaultHistoryPath() string {
	home, err := os.UserHomeDir()
	if err != nil {
		return filepath.Join(".nous", "history")
	}
	return filepath.Join(home, ".nous", "history")
}

// Add appends an entry to history, skipping consecutive duplicates.
// It auto-saves to the backing file after each addition.
func (h *History) Add(entry string) {
	entry = strings.TrimSpace(entry)
	if entry == "" {
		return
	}

	h.mu.Lock()
	defer h.mu.Unlock()

	// Skip consecutive duplicates
	if len(h.entries) > 0 && h.entries[len(h.entries)-1] == entry {
		h.position = len(h.entries)
		return
	}

	h.entries = append(h.entries, entry)

	// Prune oldest if over max
	if len(h.entries) > h.maxSize {
		h.entries = h.entries[len(h.entries)-h.maxSize:]
	}

	h.position = len(h.entries)

	// Best-effort save
	_ = h.saveLocked()
}

// Load reads history entries from the backing file.
func (h *History) Load() error {
	h.mu.Lock()
	defer h.mu.Unlock()

	f, err := os.Open(h.file)
	if err != nil {
		if os.IsNotExist(err) {
			return nil // no history yet
		}
		return err
	}
	defer f.Close()

	var entries []string
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := scanner.Text()
		if line != "" {
			entries = append(entries, line)
		}
	}
	if err := scanner.Err(); err != nil {
		return err
	}

	// Keep only most recent maxSize entries
	if len(entries) > h.maxSize {
		entries = entries[len(entries)-h.maxSize:]
	}

	h.entries = entries
	h.position = len(h.entries)
	return nil
}

// Save writes all entries to the backing file.
func (h *History) Save() error {
	h.mu.Lock()
	defer h.mu.Unlock()
	return h.saveLocked()
}

func (h *History) saveLocked() error {
	// Ensure parent directory exists
	dir := filepath.Dir(h.file)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}

	f, err := os.Create(h.file)
	if err != nil {
		return err
	}
	defer f.Close()

	w := bufio.NewWriter(f)
	for _, entry := range h.entries {
		_, _ = w.WriteString(entry)
		_ = w.WriteByte('\n')
	}
	return w.Flush()
}

// Previous returns the previous entry (navigating up through history).
// Returns "" if at the beginning.
func (h *History) Previous() string {
	h.mu.Lock()
	defer h.mu.Unlock()

	if len(h.entries) == 0 {
		return ""
	}
	if h.position > 0 {
		h.position--
	}
	return h.entries[h.position]
}

// Next returns the next entry (navigating down through history).
// Returns "" when past the most recent entry.
func (h *History) Next() string {
	h.mu.Lock()
	defer h.mu.Unlock()

	if len(h.entries) == 0 {
		return ""
	}
	if h.position < len(h.entries)-1 {
		h.position++
		return h.entries[h.position]
	}
	// Past the end — return empty (current input)
	h.position = len(h.entries)
	return ""
}

// Search performs a reverse search for the most recent entry matching the prefix.
// Returns "" if no match is found.
func (h *History) Search(prefix string) string {
	h.mu.Lock()
	defer h.mu.Unlock()

	prefix = strings.TrimSpace(prefix)
	if prefix == "" {
		return ""
	}

	for i := len(h.entries) - 1; i >= 0; i-- {
		if strings.HasPrefix(h.entries[i], prefix) {
			h.position = i
			return h.entries[i]
		}
	}
	return ""
}

// Entries returns a copy of all history entries.
func (h *History) Entries() []string {
	h.mu.Lock()
	defer h.mu.Unlock()

	out := make([]string, len(h.entries))
	copy(out, h.entries)
	return out
}

// Size returns the number of entries.
func (h *History) Size() int {
	h.mu.Lock()
	defer h.mu.Unlock()
	return len(h.entries)
}

// Clear removes all entries and deletes the history file.
func (h *History) Clear() error {
	h.mu.Lock()
	defer h.mu.Unlock()

	h.entries = nil
	h.position = 0
	return os.Remove(h.file)
}

// Entry returns the entry at position n (1-indexed) or "" if out of range.
func (h *History) Entry(n int) string {
	h.mu.Lock()
	defer h.mu.Unlock()

	idx := n - 1 // convert to 0-indexed
	if idx < 0 || idx >= len(h.entries) {
		return ""
	}
	return h.entries[idx]
}

// ResetPosition resets the navigation cursor to the end of history.
func (h *History) ResetPosition() {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.position = len(h.entries)
}
