package main

import (
	"strings"
	"testing"
	"time"

	"github.com/artaeon/nous/internal/assistant"
	"github.com/artaeon/nous/internal/blackboard"
	"github.com/artaeon/nous/internal/cognitive"
	"github.com/artaeon/nous/internal/memory"
	"github.com/artaeon/nous/internal/tools"
	"github.com/artaeon/nous/internal/training"
)

func TestRenderHelpIncludesKeyCommands(t *testing.T) {
	help := renderHelp()
	checks := []string{"/dashboard", "/today", "/remind", "/status", "/plan <goal>", "/tools", "/quit"}
	for _, check := range checks {
		if !strings.Contains(help, check) {
			t.Fatalf("renderHelp() should contain %q", check)
		}
	}
}

func TestRenderDashboardIncludesRuntimeMemoryAndTraining(t *testing.T) {
	board := blackboard.New()
	board.PostPercept(blackboard.Percept{Raw: "hello", Timestamp: time.Now()})
	board.PushGoal(blackboard.Goal{ID: "g1", Description: "ship feature", Status: "active", CreatedAt: time.Now()})
	board.RecordAction(blackboard.ActionRecord{StepID: "s1", Tool: "read", Success: true, Timestamp: time.Now()})

	wm := memory.NewWorkingMemory(8)
	wm.Store("recent", "value", 0.9)

	baseDir := t.TempDir()
	ltm := memory.NewLongTermMemory(baseDir)
	ltm.Store("arch", "hexagonal", "project")

	projMem := memory.NewProjectMemory(baseDir)
	projMem.Remember("framework", "go", "test", 1.0)

	undo := memory.NewUndoStack(5)
	undo.Push(memory.UndoEntry{Path: "README.md", Action: "write", Timestamp: time.Now()})

	episodic := memory.NewEpisodicMemory(baseDir, nil)
	episodic.Record(memory.Episode{Timestamp: time.Now(), Input: "test", Output: "ok", Success: true})

	collector := training.NewCollector(baseDir)
	collector.Collect("sys", "input", "output", []string{"read"}, 0.9)
	autoTuner := training.NewAutoTuner(collector, "qwen2.5:1.5b")
	assistantStore := assistant.NewStore(baseDir)
	_, _ = assistantStore.AddTask("Call mom", time.Now().Add(time.Hour), "")
	_ = assistantStore.SetPreference("language", "de")

	session := &cognitive.Session{ID: "sess-1", Name: "Session One"}
	dashboard := renderDashboard(board, wm, ltm, projMem, undo, session, episodic, collector, autoTuner, assistantStore)

	checks := []string{"Runtime", "Memory", "Learning loop", "Assistant", "Session", "Training pairs", "Episodes", "sess-1"}
	for _, check := range checks {
		if !strings.Contains(dashboard, check) {
			t.Fatalf("renderDashboard() should contain %q", check)
		}
	}
}

func TestRenderToolCatalogGroupsToolsByCategory(t *testing.T) {
	reg := tools.NewRegistry()
	tools.RegisterBuiltins(reg, t.TempDir(), false)

	catalog := renderToolCatalog(reg)
	checks := []string{"Explore", "Modify", "System", "Git", "Web", "read", "write", "git", "fetch"}
	for _, check := range checks {
		if !strings.Contains(catalog, check) {
			t.Fatalf("renderToolCatalog() should contain %q", check)
		}
	}
}

func TestRenderProjectViewIncludesStructureAndKeyFiles(t *testing.T) {
	project := &cognitive.ProjectInfo{
		Name:      "nous",
		Language:  "Go",
		FileCount: 42,
		KeyFiles:  []string{"README.md", "go.mod", "cmd/nous/main.go"},
		Tree:      "├── cmd\n└── internal\n",
	}

	view := renderProjectView(project)
	checks := []string{"Project", "Language  Go", "Files     42", "README.md", "Structure", "cmd", "internal"}
	for _, check := range checks {
		if !strings.Contains(view, check) {
			t.Fatalf("renderProjectView() should contain %q", check)
		}
	}
}

func TestRenderTodayIncludesUnreadNotificationsAndUpcomingTasks(t *testing.T) {
	store := assistant.NewStore(t.TempDir())
	now := time.Date(2026, 3, 11, 8, 30, 0, 0, time.UTC)
	_, _ = store.AddTask("Dentist", now.Add(-time.Minute), "")
	_, _ = store.AddTask("Pay rent", now.Add(3*time.Hour), "")
	_, _ = store.TriggerDue(now)

	out := renderToday(store, now)
	checks := []string{"Inbox", "Today", "Upcoming", "Reminder: Dentist", "Pay rent"}
	for _, check := range checks {
		if !strings.Contains(out, check) {
			t.Fatalf("renderToday() should contain %q", check)
		}
	}
}

func TestParseReminderInputSupportsRelativeDailyAndCalendarFormats(t *testing.T) {
	now := time.Date(2026, 3, 11, 8, 0, 0, 0, time.UTC)

	tests := []struct {
		input      string
		wantTitle  string
		wantRecurs string
		wantDue    time.Time
	}{
		{"in 2h call mom", "call mom", "", now.Add(2 * time.Hour)},
		{"daily 09:30 standup", "standup", "daily", time.Date(2026, 3, 11, 9, 30, 0, 0, time.UTC)},
		{"2026-03-12 18:00 dentist", "dentist", "", time.Date(2026, 3, 12, 18, 0, 0, 0, time.UTC)},
	}

	for _, tt := range tests {
		due, recurrence, title, err := parseReminderInput(tt.input, now)
		if err != nil {
			t.Fatalf("parseReminderInput(%q) error = %v", tt.input, err)
		}
		if title != tt.wantTitle || recurrence != tt.wantRecurs || !due.Equal(tt.wantDue) {
			t.Fatalf("parseReminderInput(%q) = (%v, %q, %q)", tt.input, due, recurrence, title)
		}
	}
}

func TestScoreInteractionQualityRewardsFastSuccessfulAnswers(t *testing.T) {
	board := blackboard.New()
	board.RecordAction(blackboard.ActionRecord{StepID: "1", Tool: "read", Success: true, Timestamp: time.Now()})

	quality := scoreInteractionQuality("This is a successful answer with enough detail to count as substantive.", 2*time.Second, board)
	if quality <= 0.7 {
		t.Fatalf("expected high quality score, got %.2f", quality)
	}
}

func TestScoreInteractionQualityPenalizesFailures(t *testing.T) {
	board := blackboard.New()
	board.Set("reflection", "tool loop warning")

	quality := scoreInteractionQuality("Error: reached maximum tool iterations and failed", 15*time.Second, board)
	if quality >= 0.5 {
		t.Fatalf("expected penalized quality score, got %.2f", quality)
	}
}