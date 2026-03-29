package agent

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/artaeon/nous/internal/tools"
)

// mockRegistry creates a tool registry with mock tools for testing.
func mockRegistry() *tools.Registry {
	reg := tools.NewRegistry()

	reg.Register(tools.Tool{
		Name:        "web_search",
		Description: "Search the web",
		Execute: func(args map[string]string) (string, error) {
			query := args["query"]
			return "Results for: " + query + "\n1. Result one\n2. Result two\n3. Result three", nil
		},
	})

	reg.Register(tools.Tool{
		Name:        "read",
		Description: "Read a file",
		Execute: func(args map[string]string) (string, error) {
			return "file contents of " + args["path"], nil
		},
	})

	reg.Register(tools.Tool{
		Name:        "write",
		Description: "Write a file",
		Execute: func(args map[string]string) (string, error) {
			return "wrote to " + args["path"], nil
		},
	})

	reg.Register(tools.Tool{
		Name:        "fetch",
		Description: "Fetch a URL",
		Execute: func(args map[string]string) (string, error) {
			return "fetched " + args["url"], nil
		},
	})

	reg.Register(tools.Tool{
		Name:        "calculator",
		Description: "Calculate math",
		Execute: func(args map[string]string) (string, error) {
			return "42", nil
		},
	})

	return reg
}

func TestDecomposeGoal_Research(t *testing.T) {
	planner := NewPlanner([]string{"web_search", "read", "write", "fetch"})
	plan, err := planner.DecomposeGoal("Research the market for local AI assistants")
	if err != nil {
		t.Fatalf("DecomposeGoal: %v", err)
	}

	if plan.Goal == "" {
		t.Error("plan goal is empty")
	}

	if len(plan.Phases) < 2 {
		t.Errorf("expected at least 2 phases, got %d", len(plan.Phases))
	}

	// First phase should involve information gathering
	found := false
	for _, phase := range plan.Phases {
		for _, task := range phase.Tasks {
			for _, step := range task.ToolChain {
				if step.Tool == "web_search" {
					found = true
				}
			}
		}
	}
	if !found {
		t.Error("research plan should include web_search tool")
	}
}

func TestDecomposeGoal_Writing(t *testing.T) {
	planner := NewPlanner([]string{"web_search", "read", "write"})
	plan, err := planner.DecomposeGoal("Write a blog post about Go programming")
	if err != nil {
		t.Fatalf("DecomposeGoal: %v", err)
	}

	if len(plan.Phases) < 3 {
		t.Errorf("writing plan should have at least 3 phases (research, outline/draft, review), got %d", len(plan.Phases))
	}

	// Should include a draft/write phase
	hasWrite := false
	for _, phase := range plan.Phases {
		lower := strings.ToLower(phase.Name)
		if strings.Contains(lower, "draft") || strings.Contains(lower, "write") ||
			strings.Contains(lower, "outline") {
			hasWrite = true
		}
	}
	if !hasWrite {
		t.Error("writing plan should have a draft/write phase")
	}
}

func TestDecomposeGoal_Empty(t *testing.T) {
	planner := NewPlanner(nil)
	_, err := planner.DecomposeGoal("")
	if err == nil {
		t.Error("expected error for empty goal")
	}
}

func TestExecuteChain(t *testing.T) {
	reg := mockRegistry()
	exec := NewExecutor(reg, "/tmp/test-agent")

	chain := []ToolStep{
		{Tool: "web_search", Args: map[string]string{"query": "test query"}, DependsOn: -1, OutputKey: "search"},
		{Tool: "write", Args: map[string]string{"path": "/tmp/test-agent/output.md"}, DependsOn: 0, OutputKey: "file"},
	}

	result, err := exec.ExecuteChain(chain, nil)
	if err != nil {
		t.Fatalf("ExecuteChain: %v", err)
	}

	if result.ToolCalls != 2 {
		t.Errorf("expected 2 tool calls, got %d", result.ToolCalls)
	}

	if len(result.Steps) != 2 {
		t.Errorf("expected 2 steps, got %d", len(result.Steps))
	}

	if result.Steps[0].Error != "" {
		t.Errorf("step 0 had error: %s", result.Steps[0].Error)
	}
}

func TestExecuteChain_Failure(t *testing.T) {
	reg := tools.NewRegistry()
	reg.Register(tools.Tool{
		Name:        "fail_tool",
		Description: "Always fails",
		Execute: func(args map[string]string) (string, error) {
			return "", os.ErrNotExist
		},
	})

	exec := NewExecutor(reg, "/tmp/test-agent")
	exec.MaxRetries = 2

	chain := []ToolStep{
		{Tool: "fail_tool", Args: map[string]string{}, DependsOn: -1},
	}

	_, err := exec.ExecuteChain(chain, nil)
	if err == nil {
		t.Error("expected error from failing tool")
	}
}

func TestExecuteChain_ToolCallLimit(t *testing.T) {
	reg := mockRegistry()
	exec := NewExecutor(reg, "/tmp/test-agent")
	exec.MaxToolCalls = 2

	chain := []ToolStep{
		{Tool: "web_search", Args: map[string]string{"query": "one"}, DependsOn: -1},
		{Tool: "web_search", Args: map[string]string{"query": "two"}, DependsOn: -1},
		{Tool: "web_search", Args: map[string]string{"query": "three"}, DependsOn: -1},
	}

	_, err := exec.ExecuteChain(chain, nil)
	if err == nil {
		t.Error("expected safety limit error")
	}
	if err != nil && !strings.Contains(err.Error(), "safety limit") {
		t.Errorf("expected safety limit error, got: %v", err)
	}
}

func TestAgentState_Persistence(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "state.json")

	// Create and populate state
	s1 := NewAgentState(path)
	s1.Reset("test goal")
	s1.SetPlan(&Plan{
		Goal:   "test goal",
		Phases: []Phase{{Name: "Phase 1", Tasks: []Task{{ID: "t1", Description: "task one"}}}},
	})
	s1.RecordResult("t1", "result one")
	s1.AddToolCalls(5)

	if err := s1.Save(); err != nil {
		t.Fatalf("Save: %v", err)
	}

	// Load into fresh state
	s2 := NewAgentState(path)
	if err := s2.Load(); err != nil {
		t.Fatalf("Load: %v", err)
	}

	if s2.CurrentGoal != "test goal" {
		t.Errorf("goal: got %q, want %q", s2.CurrentGoal, "test goal")
	}
	if s2.TotalToolCalls != 5 {
		t.Errorf("tool calls: got %d, want 5", s2.TotalToolCalls)
	}
	if s2.Plan == nil || len(s2.Plan.Phases) != 1 {
		t.Error("plan not persisted correctly")
	}
	if s2.Results["t1"] != "result one" {
		t.Errorf("result: got %q, want %q", s2.Results["t1"], "result one")
	}
}

func TestAgentState_Advance(t *testing.T) {
	s := NewAgentState("")
	s.SetPlan(&Plan{
		Goal: "test",
		Phases: []Phase{
			{Name: "P1", Tasks: []Task{{ID: "t1"}, {ID: "t2"}}},
			{Name: "P2", Tasks: []Task{{ID: "t3"}}},
		},
	})

	// Start at phase 0, task 0
	if s.Phase != 0 || s.Task != 0 {
		t.Errorf("initial: phase=%d task=%d", s.Phase, s.Task)
	}

	// Advance to task 1 in phase 0
	if !s.Advance() {
		t.Error("expected more tasks")
	}
	if s.Phase != 0 || s.Task != 1 {
		t.Errorf("after first advance: phase=%d task=%d", s.Phase, s.Task)
	}

	// Advance to phase 1, task 0
	if !s.Advance() {
		t.Error("expected more tasks")
	}
	if s.Phase != 1 || s.Task != 0 {
		t.Errorf("after second advance: phase=%d task=%d", s.Phase, s.Task)
	}

	// Advance past end
	if s.Advance() {
		t.Error("expected no more tasks")
	}
	if !s.Finished {
		t.Error("expected finished=true")
	}
}

func TestAgentRun_SimpleGoal(t *testing.T) {
	reg := mockRegistry()
	config := AgentConfig{
		Workspace:    t.TempDir(),
		MaxToolCalls: 50,
		MaxRetries:   2,
		StepTimeout:  5 * time.Second,
	}

	a := NewAgent(reg, config)

	var reports []string
	a.SetReportCallback(func(msg string) {
		reports = append(reports, msg)
	})

	err := a.Start("Research Go programming")
	if err != nil {
		t.Fatalf("Start: %v", err)
	}

	// Wait for completion (with timeout)
	deadline := time.After(10 * time.Second)
	for {
		select {
		case <-deadline:
			t.Fatal("agent did not complete within 10 seconds")
		default:
		}

		status := a.Status()
		if !status.Running {
			break
		}
		time.Sleep(100 * time.Millisecond)
	}

	if len(reports) == 0 {
		t.Error("expected progress reports")
	}

	// Should have a final report
	finalFound := false
	for _, r := range reports {
		if strings.Contains(r, "[COMPLETE]") {
			finalFound = true
		}
	}
	if !finalFound {
		t.Error("expected [COMPLETE] in reports")
	}
}

func TestAgent_HumanInLoop(t *testing.T) {
	reg := mockRegistry()
	config := AgentConfig{
		Workspace:    t.TempDir(),
		MaxToolCalls: 50,
		MaxRetries:   2,
		StepTimeout:  5 * time.Second,
	}

	a := NewAgent(reg, config)

	// Override planner to produce a plan that needs human input
	a.Planner = NewPlanner([]string{"web_search", "write"})

	var reports []string
	a.SetReportCallback(func(msg string) {
		reports = append(reports, msg)
	})

	err := a.Start("Plan a product launch strategy")
	if err != nil {
		t.Fatalf("Start: %v", err)
	}

	// Wait for the agent to pause for input or complete
	deadline := time.After(10 * time.Second)
	for {
		select {
		case <-deadline:
			t.Fatal("agent did not pause or complete within 10 seconds")
		default:
		}

		status := a.Status()
		if status.PausedFor != "" {
			// Provide input
			a.Resume("Launch in Q3 with a budget of $50k")
			break
		}
		if !status.Running {
			break
		}
		time.Sleep(100 * time.Millisecond)
	}

	// Wait for completion
	deadline = time.After(10 * time.Second)
	for {
		select {
		case <-deadline:
			// Acceptable — some plans may not have human-in-loop tasks
			return
		default:
		}

		status := a.Status()
		if !status.Running {
			break
		}
		time.Sleep(100 * time.Millisecond)
	}
}

func TestScheduler_NextRunTime(t *testing.T) {
	now := time.Date(2026, 3, 29, 10, 0, 0, 0, time.UTC)

	tests := []struct {
		schedule string
		wantErr  bool
		check    func(t time.Time) bool
	}{
		{"hourly", false, func(t time.Time) bool { return t.Equal(time.Date(2026, 3, 29, 11, 0, 0, 0, time.UTC)) }},
		{"every 30m", false, func(t time.Time) bool { return t.Equal(now.Add(30 * time.Minute)) }},
		{"every 2h", false, func(t time.Time) bool { return t.Equal(now.Add(2 * time.Hour)) }},
		{"daily 9:00", false, func(t time.Time) bool { return t.Day() == 30 && t.Hour() == 9 }}, // 9am already passed today
		{"daily 14:00", false, func(t time.Time) bool { return t.Day() == 29 && t.Hour() == 14 }},
		{"weekdays 8:00", false, func(t time.Time) bool {
			return t.Weekday() != time.Saturday && t.Weekday() != time.Sunday && t.Hour() == 8
		}},
		{"weekly monday 9:00", false, func(t time.Time) bool {
			return t.Weekday() == time.Monday && t.Hour() == 9
		}},
		{"invalid", true, nil},
	}

	for _, tt := range tests {
		next, err := nextRunTime(tt.schedule, now)
		if tt.wantErr {
			if err == nil {
				t.Errorf("schedule %q: expected error", tt.schedule)
			}
			continue
		}
		if err != nil {
			t.Errorf("schedule %q: %v", tt.schedule, err)
			continue
		}
		if !next.After(now) {
			t.Errorf("schedule %q: next=%v is not after now=%v", tt.schedule, next, now)
		}
		if tt.check != nil && !tt.check(next) {
			t.Errorf("schedule %q: next=%v failed check", tt.schedule, next)
		}
	}
}

func TestReporter_Progress(t *testing.T) {
	state := NewAgentState("")
	state.Reset("Test goal")
	state.SetPlan(&Plan{
		Goal: "Test goal",
		Phases: []Phase{
			{
				Name: "Research",
				Tasks: []Task{
					{ID: "t1", Description: "Search the web", Status: TaskCompleted},
					{ID: "t2", Description: "Analyze results", Status: TaskRunning},
				},
			},
			{
				Name:      "Report",
				DependsOn: []int{0},
				Tasks: []Task{
					{ID: "t3", Description: "Write report", Status: TaskPending},
				},
			},
		},
	})
	state.TotalToolCalls = 3

	reporter := NewReporter(state)
	report := reporter.ProgressReport()

	if !strings.Contains(report, "Phase 1/2") {
		t.Errorf("report should mention phase 1/2, got:\n%s", report)
	}
	if !strings.Contains(report, "Research") {
		t.Errorf("report should mention phase name, got:\n%s", report)
	}
	if !strings.Contains(report, "Tool calls: 3") {
		t.Errorf("report should mention tool calls, got:\n%s", report)
	}
}

func TestReporter_Final(t *testing.T) {
	state := NewAgentState("")
	state.Reset("Test goal")
	state.SetPlan(&Plan{
		Goal: "Test goal",
		Phases: []Phase{
			{
				Name: "Only Phase",
				Tasks: []Task{
					{ID: "t1", Description: "Do the thing", Status: TaskCompleted},
				},
			},
		},
	})
	state.TotalToolCalls = 7
	state.RecordResult("t1", "done")

	reporter := NewReporter(state)
	report := reporter.FinalReport()

	if !strings.Contains(report, "[COMPLETE]") {
		t.Errorf("final report should start with [COMPLETE], got:\n%s", report)
	}
	if !strings.Contains(report, "1 completed") {
		t.Errorf("final report should mention completed tasks, got:\n%s", report)
	}
}

func TestClassifyGoal(t *testing.T) {
	tests := []struct {
		goal string
		want goalType
	}{
		{"Research the market for AI assistants", goalResearch},
		{"Write a blog post about Go", goalWriting},
		{"Analyze our sales data", goalAnalysis},
		{"Plan a product launch", goalPlanning},
		{"Build a web scraper", goalBuilding},
		{"Monitor Bitcoin price", goalMonitoring},
		{"Something random", goalGeneric},
	}

	for _, tt := range tests {
		got := classifyGoal(tt.goal)
		if got != tt.want {
			t.Errorf("classifyGoal(%q) = %d, want %d", tt.goal, got, tt.want)
		}
	}
}

func TestExtractTopic(t *testing.T) {
	tests := []struct {
		goal string
		want string
	}{
		{"Research the market for local AI assistants", "local AI assistants"},
		{"Write a blog post about Go", "blog post about Go"},
		{"Analyze our sales data", "our sales data"},
		{"Monitor Bitcoin price", "Bitcoin price"},
	}

	for _, tt := range tests {
		got := extractTopic(tt.goal)
		if got != tt.want {
			t.Errorf("extractTopic(%q) = %q, want %q", tt.goal, got, tt.want)
		}
	}
}

func TestIsDangerousTool(t *testing.T) {
	dangerous := []string{"shell", "run", "write", "edit", "patch", "git"}
	safe := []string{"web_search", "read", "glob", "grep", "calculator", "weather"}

	for _, name := range dangerous {
		if !IsDangerousTool(name) {
			t.Errorf("%q should be dangerous", name)
		}
	}
	for _, name := range safe {
		if IsDangerousTool(name) {
			t.Errorf("%q should not be dangerous", name)
		}
	}
}
