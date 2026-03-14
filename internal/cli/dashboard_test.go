package cli

import (
	"strings"
	"testing"
	"time"
)

func TestRenderDashboardEmpty(t *testing.T) {
	stats := NewSessionStats()
	d := DashboardData{
		Model: "test-model",
		Stats: stats,
		Hands: nil,
		Memory: MemoryInfo{
			WorkingUsed: 0, WorkingMax: 64,
		},
	}

	out := RenderDashboard(d)
	if !strings.Contains(out, "Nous Dashboard") {
		t.Error("output should contain 'Nous Dashboard'")
	}
	if !strings.Contains(out, "test-model") {
		t.Error("output should contain model name")
	}
	if !strings.Contains(out, "(none registered)") {
		t.Error("output should show no hands registered")
	}
	if !strings.Contains(out, "(none active)") {
		t.Error("output should show no channels active")
	}
}

func TestRenderDashboardWithHands(t *testing.T) {
	stats := NewSessionStats()
	stats.RecordMessage()
	stats.RecordToolCall()
	stats.RecordTokens(500)

	d := DashboardData{
		Model: "qwen2.5:1.5b",
		Stats: stats,
		Hands: []HandInfo{
			{Name: "researcher", Enabled: true, State: "idle", LastRun: time.Now().Add(-18 * time.Second), TotalRuns: 3},
			{Name: "monitor", Enabled: true, State: "running", LastRun: time.Now().Add(-5 * time.Minute), TotalRuns: 12},
			{Name: "guardian", Enabled: false, State: "", RequiresApproval: true},
		},
		Memory: MemoryInfo{
			WorkingUsed:  12,
			WorkingMax:   64,
			LongTerm:     47,
			Episodic:     234,
			ProjectFacts: 8,
		},
		ChannelCnt: 2,
	}

	out := RenderDashboard(d)

	if !strings.Contains(out, "qwen2.5:1.5b") {
		t.Error("output should contain model name")
	}
	if !strings.Contains(out, "researcher") {
		t.Error("output should contain researcher hand")
	}
	if !strings.Contains(out, "monitor") {
		t.Error("output should contain monitor hand")
	}
	if !strings.Contains(out, "guardian") {
		t.Error("output should contain guardian hand")
	}
	if !strings.Contains(out, "12/64 slots") {
		t.Error("output should contain working memory stats")
	}
	if !strings.Contains(out, "234 memories") {
		t.Error("output should contain episodic count")
	}
	if !strings.Contains(out, "2 active") {
		t.Error("output should show 2 active channels")
	}
}

func TestVisibleLen(t *testing.T) {
	tests := []struct {
		input string
		want  int
	}{
		{"hello", 5},
		{"\033[31mred\033[0m", 3},
		{"", 0},
		{"\033[1m\033[36mtest\033[0m", 4},
		{"no escapes here", 15},
	}
	for _, tt := range tests {
		got := visibleLen(tt.input)
		if got != tt.want {
			t.Errorf("visibleLen(%q) = %d, want %d", tt.input, got, tt.want)
		}
	}
}

func TestFormatDurationShort(t *testing.T) {
	tests := []struct {
		d    time.Duration
		want string
	}{
		{5 * time.Second, "5s"},
		{90 * time.Second, "1m"},
		{3600 * time.Second, "1h"},
		{2*time.Hour + 30*time.Minute, "2h"},
	}
	for _, tt := range tests {
		got := formatDurationShort(tt.d)
		if got != tt.want {
			t.Errorf("formatDurationShort(%v) = %q, want %q", tt.d, got, tt.want)
		}
	}
}
