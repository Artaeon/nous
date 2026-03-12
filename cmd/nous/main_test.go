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
	checks := []string{"/dashboard", "/today", "/remind", "/routine", "/status", "/plan <goal>", "/tools", "/quit"}
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

func TestRenderRoutinesIncludesConfiguredRoutines(t *testing.T) {
	store := assistant.NewStore(t.TempDir())
	_, _ = store.AddRoutine("Morning review", "daily", "08:30")

	out := renderRoutines(store)
	checks := []string{"Routines", "Morning review", "daily", "08:30"}
	for _, check := range checks {
		if !strings.Contains(out, check) {
			t.Fatalf("renderRoutines() should contain %q", check)
		}
	}
}

func TestRenderBriefingRespectsGermanPreference(t *testing.T) {
	store := assistant.NewStore(t.TempDir())
	now := time.Date(2026, 3, 12, 10, 0, 0, 0, time.UTC)
	_ = store.SetPreference("language", "de")
	_, _ = store.AddTask("Bericht fertigstellen", now.Add(30*time.Minute), "")

	out := renderBriefing(store, now)
	checks := []string{"Guten Morgen", "Heute (1)", "Bericht fertigstellen"}
	for _, check := range checks {
		if !strings.Contains(out, check) {
			t.Fatalf("renderBriefing() should contain %q, got %q", check, out)
		}
	}
}

func TestWhatShouldIDoNowRespectsGermanPreference(t *testing.T) {
	store := assistant.NewStore(t.TempDir())
	now := time.Date(2026, 3, 12, 10, 0, 0, 0, time.UTC)
	_ = store.SetPreference("language", "de")
	_, _ = store.AddTask("Design Review", now.Add(45*time.Minute), "")

	out := whatShouldIDoNow(store, now)
	checks := []string{"In 45 Min. steht", "Design Review"}
	for _, check := range checks {
		if !strings.Contains(out, check) {
			t.Fatalf("whatShouldIDoNow() should contain %q, got %q", check, out)
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

func TestParseRoutineInputSupportsDailyAndWeekdays(t *testing.T) {
	tests := []struct {
		input        string
		wantSchedule string
		wantTime     string
		wantTitle    string
	}{
		{"daily 08:30 Morning review", "daily", "08:30", "Morning review"},
		{"weekdays 09:15 Inbox zero", "weekdays", "09:15", "Inbox zero"},
	}

	for _, tt := range tests {
		schedule, clock, title, err := parseRoutineInput(tt.input)
		if err != nil {
			t.Fatalf("parseRoutineInput(%q) error = %v", tt.input, err)
		}
		if schedule != tt.wantSchedule || clock != tt.wantTime || title != tt.wantTitle {
			t.Fatalf("parseRoutineInput(%q) = (%q, %q, %q)", tt.input, schedule, clock, title)
		}
	}
}

func TestBuildAssistantContextIncludesPreferencesTasksAndRoutines(t *testing.T) {
	store := assistant.NewStore(t.TempDir())
	now := time.Date(2026, 3, 11, 10, 0, 0, 0, time.UTC)
	_ = store.SetPreference("language", "de")
	_, _ = store.AddTask("Call dentist", now.Add(2*time.Hour), "")
	_, _ = store.AddRoutine("Morning review", "daily", "08:30")

	out := buildAssistantContext(store, "What matters today?", "User: I want a calm day\nAssistant: We'll keep it simple.", now)
	checks := []string{"[Assistant Memory]", "Current time:", "Recent conversation:", "language=de", "Active reminders/tasks:", "Call dentist", "Active routines today:", "Morning review"}
	for _, check := range checks {
		if !strings.Contains(out, check) {
			t.Fatalf("buildAssistantContext() should contain %q", check)
		}
	}
}

func TestBuildAssistantContextSkipsCodeQueries(t *testing.T) {
	store := assistant.NewStore(t.TempDir())
	_ = store.SetPreference("language", "de")
	out := buildAssistantContext(store, "show me the main.go file", "", time.Now())
	if out != "" {
		t.Fatalf("expected no assistant context for code query, got %q", out)
	}
}

func TestAnswerAssistantQueryUsesPreferencesAndReminders(t *testing.T) {
	store := assistant.NewStore(t.TempDir())
	now := time.Date(2026, 3, 11, 10, 0, 0, 0, time.UTC)
	_ = store.SetPreference("language", "de")
	_, _ = store.AddTask("Call dentist", now.Add(2*time.Hour), "")

	answer, ok := answerAssistantQuery(store, "What reminder do I currently have?", "", now)
	if !ok {
		t.Fatal("expected deterministic assistant answer")
	}
	checks := []string{"Call dentist", "12:00"}
	for _, check := range checks {
		if !strings.Contains(answer, check) {
			t.Fatalf("answerAssistantQuery() should contain %q, got %q", check, answer)
		}
	}

	langAnswer, ok := answerAssistantQuery(store, "What language do I prefer?", "", now)
	if !ok || (!strings.Contains(langAnswer, "de") && !strings.Contains(strings.ToLower(langAnswer), "deutsch")) {
		t.Fatalf("expected language answer to mention de/Deutsch, got %q", langAnswer)
	}
}

func TestAnswerAssistantQuerySupportsCompanionPlanningAndFocus(t *testing.T) {
	store := assistant.NewStore(t.TempDir())
	now := time.Date(2026, 3, 11, 10, 0, 0, 0, time.UTC)
	_ = store.SetPreference("focus", "Deep work before meetings")
	_, _ = store.AddTask("Finish report", now.Add(-30*time.Minute), "")
	_, _ = store.AddRoutine("Review priorities", "daily", "08:00")

	planAnswer, ok := answerAssistantQuery(store, "Help me plan my day", "", now)
	if !ok {
		t.Fatal("expected planning answer")
	}
	for _, check := range []string{"Finish report", "Deep work before meetings", "Review priorities"} {
		if !strings.Contains(planAnswer, check) {
			t.Fatalf("planning answer should contain %q, got %q", check, planAnswer)
		}
	}

	focusAnswer, ok := answerAssistantQuery(store, "I feel overwhelmed", "", now)
	if !ok {
		t.Fatal("expected focus answer")
	}
	for _, check := range []string{"Finish report", "Deep work before meetings"} {
		if !strings.Contains(focusAnswer, check) {
			t.Fatalf("focus answer should contain %q, got %q", check, focusAnswer)
		}
	}
}

func TestAnswerAssistantQuerySupportsGreetingAndPreferenceSummary(t *testing.T) {
	store := assistant.NewStore(t.TempDir())
	now := time.Date(2026, 3, 11, 10, 0, 0, 0, time.UTC)
	_ = store.SetPreference("language", "de")
	_ = store.SetPreference("focus", "Deep work before meetings")
	_ = store.SetPreference("profile.meetings", "you get anxious before meetings")
	_, _ = store.AddTask("Send update", now.Add(90*time.Minute), "")

	greeting, ok := answerAssistantQuery(store, "hello", "", now)
	if !ok {
		t.Fatal("expected greeting answer")
	}
	for _, check := range []string{"Send update", "11:30"} {
		if !strings.Contains(greeting, check) {
			t.Fatalf("greeting answer should contain %q, got %q", check, greeting)
		}
	}

	prefAnswer, ok := answerAssistantQuery(store, "What do you know about my preferences?", "", now)
	if !ok {
		t.Fatal("expected preference summary answer")
	}
	for _, check := range []string{"language=de", "focus=Deep work before meetings", "you get anxious before meetings", "Send update"} {
		if !strings.Contains(prefAnswer, check) {
			t.Fatalf("preference summary should contain %q, got %q", check, prefAnswer)
		}
	}
}

func TestAnswerAssistantQueryCanRememberPersonalNotes(t *testing.T) {
	store := assistant.NewStore(t.TempDir())
	now := time.Date(2026, 3, 12, 10, 0, 0, 0, time.UTC)

	answer, ok := answerAssistantQuery(store, "remember that I get anxious before meetings", "", now)
	if !ok {
		t.Fatal("expected remember-note answer")
	}
	if !strings.Contains(strings.ToLower(answer), "anxious before meetings") {
		t.Fatalf("remember answer should echo stored note, got %q", answer)
	}

	summary, ok := answerAssistantQuery(store, "what do you know about me now", "", now)
	if !ok {
		t.Fatal("expected self-summary answer")
	}
	if !strings.Contains(strings.ToLower(summary), "anxious before meetings") {
		t.Fatalf("self-summary should include remembered note, got %q", summary)
	}

	ctx := buildAssistantContext(store, "help me focus", "", now)
	if !strings.Contains(strings.ToLower(ctx), "personal notes") || !strings.Contains(strings.ToLower(ctx), "anxious before meetings") {
		t.Fatalf("assistant context should include remembered personal notes, got %q", ctx)
	}
}

func TestAnswerAssistantQueryUsesRecentConversationForFollowUps(t *testing.T) {
	store := assistant.NewStore(t.TempDir())
	now := time.Date(2026, 3, 12, 10, 0, 0, 0, time.UTC)
	recent := "User: I feel overwhelmed\nAssistant: Let's reduce today to one next step: finish the report before lunch."

	answer, ok := answerAssistantQuery(store, "tell me more", recent, now)
	if !ok {
		t.Fatal("expected follow-up answer")
	}
	for _, check := range []string{"Picking up from our last point", "finish the report"} {
		if !strings.Contains(answer, check) {
			t.Fatalf("follow-up answer should contain %q, got %q", check, answer)
		}
	}

	_, _ = store.AddTask("Finish report", now.Add(2*time.Hour), "")
	answer, ok = answerAssistantQuery(store, "okay", recent, now)
	if !ok {
		t.Fatal("expected short confirmation follow-up answer")
	}
	for _, check := range []string{"report", "next small step"} {
		if !strings.Contains(strings.ToLower(answer), strings.ToLower(check)) {
			t.Fatalf("confirmation follow-up should contain %q, got %q", check, answer)
		}
	}

	answer, ok = answerAssistantQuery(store, "that sounds right", recent, now)
	if !ok {
		t.Fatal("expected affirmation follow-up answer")
	}
	if !strings.Contains(strings.ToLower(answer), "report") {
		t.Fatalf("affirmation follow-up should point to next step, got %q", answer)
	}
}

func TestAnswerAssistantQueryHandlesReflectiveProcrastination(t *testing.T) {
	store := assistant.NewStore(t.TempDir())
	now := time.Date(2026, 3, 12, 10, 0, 0, 0, time.UTC)

	first, ok := answerAssistantQuery(store, "I have been procrastinating on my report for days", "", now)
	if !ok {
		t.Fatal("expected procrastination reply")
	}
	for _, check := range []string{"report", "three rough bullet points"} {
		if !strings.Contains(strings.ToLower(first), strings.ToLower(check)) {
			t.Fatalf("procrastination reply should contain %q, got %q", check, first)
		}
	}

	recent := "User: I have been procrastinating on my report for days\nAssistant: " + first
	why, ok := answerAssistantQuery(store, "why do you think that keeps happening", recent, now)
	if !ok {
		t.Fatal("expected reflective follow-up reply")
	}
	for _, check := range []string{"report", "pressure task", "not mean you're lazy"} {
		if !strings.Contains(strings.ToLower(why), strings.ToLower(check)) {
			t.Fatalf("reflective reply should contain %q, got %q", check, why)
		}
	}

	affirmRecent := recent + "\nUser: why do you think that keeps happening\nAssistant: " + why
	affirm, ok := answerAssistantQuery(store, "that sounds right", affirmRecent, now)
	if !ok {
		t.Fatal("expected affirmation next-step reply")
	}
	if !strings.Contains(strings.ToLower(affirm), "report") {
		t.Fatalf("affirmation reply should stay on the report topic, got %q", affirm)
	}
}

func TestAnswerAssistantQueryHandlesGermanReflectivePrompts(t *testing.T) {
	store := assistant.NewStore(t.TempDir())
	now := time.Date(2026, 3, 12, 10, 0, 0, 0, time.UTC)
	_ = store.SetPreference("language", "de")
	_ = store.SetPreference("focus", "Tiefe Arbeit vor Meetings")
	_ = store.SetPreference("profile.meetings", "ich werde vor meetings nervös")

	summary, ok := answerAssistantQuery(store, "was weißt du über mich", "", now)
	if !ok {
		t.Fatal("expected German self-summary")
	}
	for _, check := range []string{"Tiefe Arbeit vor Meetings", "ich werde vor meetings nervös"} {
		if !strings.Contains(summary, check) {
			t.Fatalf("German summary should contain %q, got %q", check, summary)
		}
	}

	first, ok := answerAssistantQuery(store, "ich prokrastiniere seit tagen bei meinem bericht", "", now)
	if !ok {
		t.Fatal("expected German procrastination reply")
	}
	for _, check := range []string{"bericht", "drei grobe stichpunkte"} {
		if !strings.Contains(strings.ToLower(first), check) {
			t.Fatalf("German procrastination reply should contain %q, got %q", check, first)
		}
	}

	recent := "User: ich prokrastiniere seit tagen bei meinem bericht\nAssistant: " + first
	why, ok := answerAssistantQuery(store, "warum passiert das immer wieder", recent, now)
	if !ok {
		t.Fatal("expected German reflection reply")
	}
	for _, check := range []string{"bericht", "druck-aufgabe", "nicht, dass du faul bist"} {
		if !strings.Contains(strings.ToLower(why), check) {
			t.Fatalf("German reflection reply should contain %q, got %q", check, why)
		}
	}

	affirmRecent := recent + "\nUser: warum passiert das immer wieder\nAssistant: " + why
	affirm, ok := answerAssistantQuery(store, "das klingt richtig", affirmRecent, now)
	if !ok {
		t.Fatal("expected German affirmation reply")
	}
	if !strings.Contains(strings.ToLower(affirm), "bericht") {
		t.Fatalf("German affirmation reply should stay on the report topic, got %q", affirm)
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

func TestAssistantQueryDoesNotHijackCodeQuestions(t *testing.T) {
	store := assistant.NewStore(t.TempDir())
	now := time.Now()

	codeQueries := []string{
		"what does renderBriefing include",
		"explain the correctArgNames function",
		"how does semantic ranking work in working.go",
		"read file internal/cognitive/reasoner.go",
		"what is the function signature of SemanticSearch",
		"show me the struct definition for Pipeline",
		"where is the type WorkingMemory defined",
		"how does the code in main.go work",
	}

	for _, query := range codeQueries {
		_, ok := answerAssistantQuery(store, query, "", now)
		if ok {
			t.Errorf("answerAssistantQuery should NOT handle code query %q", query)
		}
	}
}

func TestAssistantContextNotInjectedForCodeQuestions(t *testing.T) {
	codeQueries := []string{
		"what does renderBriefing include",
		"explain the function correctArgNames",
		"how does semantic ranking work in working.go",
		"read the file main.go",
		"search for SemanticSearch in the codebase",
	}

	for _, query := range codeQueries {
		if shouldInjectAssistantContext(query) {
			t.Errorf("shouldInjectAssistantContext should return false for %q", query)
		}
	}
}

func TestAssistantQueryStillWorksForRealAssistantQuestions(t *testing.T) {
	store := assistant.NewStore(t.TempDir())
	now := time.Date(2026, 3, 11, 10, 0, 0, 0, time.UTC)
	_, _ = store.AddTask("Buy groceries", now.Add(2*time.Hour), "")

	assistantQueries := []struct {
		input    string
		wantOK   bool
	}{
		{"what reminder do I currently have?", true},
		{"what's on my plate", true},
		{"what should I do now", true},
		{"good morning", true},
		{"evening review", true},
		{"overdue tasks", true},
	}

	for _, tt := range assistantQueries {
		_, ok := answerAssistantQuery(store, tt.input, "", now)
		if ok != tt.wantOK {
			t.Errorf("answerAssistantQuery(%q) ok=%v, want %v", tt.input, ok, tt.wantOK)
		}
	}
}
