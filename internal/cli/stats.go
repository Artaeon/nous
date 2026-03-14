package cli

import (
	"fmt"
	"runtime"
	"sync"
	"time"
)

// SessionStatsSnapshot is a point-in-time copy of session metrics,
// safe to pass by value (no mutex).
type SessionStatsSnapshot struct {
	StartTime  time.Time
	Messages   int
	ToolCalls  int
	HandsRun   int
	TokensUsed int
}

// SessionStats tracks per-session metrics for the REPL.
type SessionStats struct {
	StartTime  time.Time
	Messages   int
	ToolCalls  int
	HandsRun   int
	TokensUsed int
	mu         sync.Mutex
}

// NewSessionStats creates a new stats tracker starting now.
func NewSessionStats() *SessionStats {
	return &SessionStats{
		StartTime: time.Now(),
	}
}

// RecordMessage increments the message counter.
func (s *SessionStats) RecordMessage() {
	s.mu.Lock()
	s.Messages++
	s.mu.Unlock()
}

// RecordToolCall increments the tool call counter.
func (s *SessionStats) RecordToolCall() {
	s.mu.Lock()
	s.ToolCalls++
	s.mu.Unlock()
}

// RecordHandRun increments the hands-run counter.
func (s *SessionStats) RecordHandRun() {
	s.mu.Lock()
	s.HandsRun++
	s.mu.Unlock()
}

// RecordTokens adds to the token counter.
func (s *SessionStats) RecordTokens(n int) {
	s.mu.Lock()
	s.TokensUsed += n
	s.mu.Unlock()
}

// Uptime returns the duration since the session started.
func (s *SessionStats) Uptime() time.Duration {
	return time.Since(s.StartTime)
}

// Snapshot returns a point-in-time copy of the current stats (safe for display).
func (s *SessionStats) Snapshot() SessionStatsSnapshot {
	s.mu.Lock()
	defer s.mu.Unlock()
	return SessionStatsSnapshot{
		StartTime:  s.StartTime,
		Messages:   s.Messages,
		ToolCalls:  s.ToolCalls,
		HandsRun:   s.HandsRun,
		TokensUsed: s.TokensUsed,
	}
}

// FormatUptime returns a human-friendly uptime string like "2h 14m".
func (s *SessionStats) FormatUptime() string {
	return formatDuration(s.Uptime())
}

// MemoryUsageMB returns the current process heap allocation in megabytes.
func MemoryUsageMB() float64 {
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	return float64(m.Alloc) / (1024 * 1024)
}

// FormatSummary returns a one-line summary of session stats.
func (s *SessionStats) FormatSummary() string {
	snap := s.Snapshot()
	return fmt.Sprintf("%d messages  %d tool calls  %d hands  %d tokens",
		snap.Messages, snap.ToolCalls, snap.HandsRun, snap.TokensUsed)
}

func formatDuration(d time.Duration) string {
	d = d.Truncate(time.Second)
	h := int(d.Hours())
	m := int(d.Minutes()) % 60
	s := int(d.Seconds()) % 60

	switch {
	case h > 0:
		return fmt.Sprintf("%dh %dm", h, m)
	case m > 0:
		return fmt.Sprintf("%dm %ds", m, s)
	default:
		return fmt.Sprintf("%ds", s)
	}
}
