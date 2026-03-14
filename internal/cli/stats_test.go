package cli

import (
	"strings"
	"testing"
	"time"
)

func TestNewSessionStats(t *testing.T) {
	s := NewSessionStats()
	if s.Messages != 0 || s.ToolCalls != 0 || s.HandsRun != 0 || s.TokensUsed != 0 {
		t.Errorf("new stats should be zero: %+v", s)
	}
	if s.StartTime.IsZero() {
		t.Error("StartTime should be set")
	}
}

func TestRecordMessage(t *testing.T) {
	s := NewSessionStats()
	s.RecordMessage()
	s.RecordMessage()
	s.RecordMessage()
	if s.Messages != 3 {
		t.Errorf("Messages = %d, want 3", s.Messages)
	}
}

func TestRecordToolCall(t *testing.T) {
	s := NewSessionStats()
	s.RecordToolCall()
	s.RecordToolCall()
	if s.ToolCalls != 2 {
		t.Errorf("ToolCalls = %d, want 2", s.ToolCalls)
	}
}

func TestRecordHandRun(t *testing.T) {
	s := NewSessionStats()
	s.RecordHandRun()
	if s.HandsRun != 1 {
		t.Errorf("HandsRun = %d, want 1", s.HandsRun)
	}
}

func TestRecordTokens(t *testing.T) {
	s := NewSessionStats()
	s.RecordTokens(100)
	s.RecordTokens(250)
	if s.TokensUsed != 350 {
		t.Errorf("TokensUsed = %d, want 350", s.TokensUsed)
	}
}

func TestUptime(t *testing.T) {
	s := NewSessionStats()
	// Uptime should be positive (we just created it)
	time.Sleep(10 * time.Millisecond)
	if s.Uptime() < 10*time.Millisecond {
		t.Errorf("Uptime = %v, expected >= 10ms", s.Uptime())
	}
}

func TestSnapshot(t *testing.T) {
	s := NewSessionStats()
	s.RecordMessage()
	s.RecordToolCall()
	s.RecordTokens(42)

	snap := s.Snapshot()
	if snap.Messages != 1 || snap.ToolCalls != 1 || snap.TokensUsed != 42 {
		t.Errorf("Snapshot = %+v, want Messages=1 ToolCalls=1 Tokens=42", snap)
	}

	// Modifying original shouldn't affect snapshot
	s.RecordMessage()
	if snap.Messages != 1 {
		t.Error("snapshot should be independent of original")
	}
}

func TestFormatUptime(t *testing.T) {
	s := &SessionStats{StartTime: time.Now().Add(-2*time.Hour - 14*time.Minute)}
	got := s.FormatUptime()
	if got != "2h 14m" {
		t.Errorf("FormatUptime = %q, want %q", got, "2h 14m")
	}
}

func TestFormatUptimeMinutes(t *testing.T) {
	s := &SessionStats{StartTime: time.Now().Add(-5*time.Minute - 30*time.Second)}
	got := s.FormatUptime()
	if got != "5m 30s" {
		t.Errorf("FormatUptime = %q, want %q", got, "5m 30s")
	}
}

func TestFormatUptimeSeconds(t *testing.T) {
	s := &SessionStats{StartTime: time.Now().Add(-15 * time.Second)}
	got := s.FormatUptime()
	if got != "15s" {
		t.Errorf("FormatUptime = %q, want %q", got, "15s")
	}
}

func TestStatsFormatSummary(t *testing.T) {
	s := NewSessionStats()
	s.RecordMessage()
	s.RecordMessage()
	s.RecordToolCall()
	s.RecordTokens(1234)

	summary := s.FormatSummary()
	if !strings.Contains(summary, "2 messages") {
		t.Errorf("summary missing messages: %q", summary)
	}
	if !strings.Contains(summary, "1 tool calls") {
		t.Errorf("summary missing tool calls: %q", summary)
	}
	if !strings.Contains(summary, "1234 tokens") {
		t.Errorf("summary missing tokens: %q", summary)
	}
}

func TestMemoryUsageMB(t *testing.T) {
	mb := MemoryUsageMB()
	if mb <= 0 {
		t.Errorf("MemoryUsageMB = %f, want > 0", mb)
	}
}

func TestConcurrentStats(t *testing.T) {
	s := NewSessionStats()
	done := make(chan struct{})
	for i := 0; i < 10; i++ {
		go func() {
			defer func() { done <- struct{}{} }()
			for j := 0; j < 100; j++ {
				s.RecordMessage()
				s.RecordToolCall()
				s.RecordHandRun()
				s.RecordTokens(1)
				s.Snapshot()
				s.FormatSummary()
			}
		}()
	}
	for i := 0; i < 10; i++ {
		<-done
	}
	// Just verify no panics/races
	if s.Messages != 1000 {
		t.Errorf("Messages = %d, want 1000", s.Messages)
	}
}

func TestFormatDuration(t *testing.T) {
	tests := []struct {
		d    time.Duration
		want string
	}{
		{0, "0s"},
		{30 * time.Second, "30s"},
		{90 * time.Second, "1m 30s"},
		{3600 * time.Second, "1h 0m"},
		{2*time.Hour + 30*time.Minute, "2h 30m"},
	}
	for _, tt := range tests {
		got := formatDuration(tt.d)
		if got != tt.want {
			t.Errorf("formatDuration(%v) = %q, want %q", tt.d, got, tt.want)
		}
	}
}
