package sandbox

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"time"
)

const (
	auditFileName   = "audit.json"
	maxAuditFileSize = 10 * 1024 * 1024 // 10MB
)

// AuditEntry records a single sandboxed execution.
type AuditEntry struct {
	Timestamp time.Time     `json:"timestamp"`
	Command   string        `json:"command"`
	Args      []string      `json:"args,omitempty"`
	ExitCode  int           `json:"exit_code"`
	Duration  time.Duration `json:"duration_ns"`
	Status    string        `json:"status"` // "ok", "denied: ...", "timeout", "error: ..."
}

// Auditor writes execution records to ~/.nous/audit.json.
type Auditor struct {
	path string
	mu   sync.Mutex
}

// NewAuditor creates an auditor that writes to the given base directory.
// The directory is created if it does not exist.
func NewAuditor(baseDir string) *Auditor {
	return &Auditor{
		path: filepath.Join(baseDir, auditFileName),
	}
}

// Log appends an entry to the audit log. It rotates the file if it
// exceeds maxAuditFileSize.
func (a *Auditor) Log(entry AuditEntry) {
	a.mu.Lock()
	defer a.mu.Unlock()

	// Ensure parent directory exists
	if err := os.MkdirAll(filepath.Dir(a.path), 0755); err != nil {
		return // best-effort
	}

	// Check file size and rotate if needed
	a.rotateIfNeeded()

	// Append entry as a single JSON line
	f, err := os.OpenFile(a.path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0600)
	if err != nil {
		return
	}
	defer f.Close()

	data, err := json.Marshal(entry)
	if err != nil {
		return
	}

	f.Write(data)
	f.Write([]byte("\n"))
}

// rotateIfNeeded renames the current audit log to audit.json.old if
// it exceeds maxAuditFileSize. Must be called with mu held.
func (a *Auditor) rotateIfNeeded() {
	info, err := os.Stat(a.path)
	if err != nil {
		return // file doesn't exist yet, nothing to rotate
	}

	if info.Size() < maxAuditFileSize {
		return
	}

	rotatedPath := a.path + ".old"
	_ = os.Remove(rotatedPath)
	_ = os.Rename(a.path, rotatedPath)
}

// Recent returns the last n audit entries. Returns what it can on error.
func (a *Auditor) Recent(n int) []AuditEntry {
	a.mu.Lock()
	defer a.mu.Unlock()

	data, err := os.ReadFile(a.path)
	if err != nil {
		return nil
	}

	// Parse JSONL
	var entries []AuditEntry
	for _, line := range splitLines(data) {
		if len(line) == 0 {
			continue
		}
		var entry AuditEntry
		if err := json.Unmarshal(line, &entry); err != nil {
			continue
		}
		entries = append(entries, entry)
	}

	// Return last n
	if len(entries) <= n {
		return entries
	}
	return entries[len(entries)-n:]
}

// splitLines splits data by newline bytes without importing strings.
func splitLines(data []byte) [][]byte {
	var lines [][]byte
	start := 0
	for i, b := range data {
		if b == '\n' {
			lines = append(lines, data[start:i])
			start = i + 1
		}
	}
	if start < len(data) {
		lines = append(lines, data[start:])
	}
	return lines
}

// FormatEntry returns a human-readable one-line summary of an audit entry.
func FormatEntry(e AuditEntry) string {
	return fmt.Sprintf("[%s] %s (exit=%d, %v) %s",
		e.Timestamp.Format("15:04:05"),
		e.Command,
		e.ExitCode,
		e.Duration.Round(time.Millisecond),
		e.Status,
	)
}
