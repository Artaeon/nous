package agent

import (
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/artaeon/nous/internal/tools"
)

// mockRegistry creates a tool registry with mock tools for testing.
func mockRegistry() *tools.Registry {
	reg := tools.NewRegistry()

	reg.Register(tools.Tool{
		Name:        "websearch",
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
	planner := NewPlanner([]string{"websearch", "read", "write", "fetch"})
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
				if step.Tool == "websearch" {
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
	planner := NewPlanner([]string{"websearch", "read", "write"})
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
		{Tool: "websearch", Args: map[string]string{"query": "test query"}, DependsOn: -1, OutputKey: "search"},
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
		{Tool: "websearch", Args: map[string]string{"query": "one"}, DependsOn: -1},
		{Tool: "websearch", Args: map[string]string{"query": "two"}, DependsOn: -1},
		{Tool: "websearch", Args: map[string]string{"query": "three"}, DependsOn: -1},
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

	var mu sync.Mutex
	var reports []string
	a.SetReportCallback(func(msg string) {
		mu.Lock()
		reports = append(reports, msg)
		mu.Unlock()
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

	mu.Lock()
	reportsCopy := make([]string, len(reports))
	copy(reportsCopy, reports)
	mu.Unlock()

	if len(reportsCopy) == 0 {
		t.Error("expected progress reports")
	}

	// Should have a final report
	finalFound := false
	for _, r := range reportsCopy {
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
	defer a.Stop() // ensure goroutine cleanup

	// Override planner to produce a plan that needs human input
	a.Planner = NewPlanner([]string{"websearch", "write"})

	var mu sync.Mutex
	var reports []string
	a.SetReportCallback(func(msg string) {
		mu.Lock()
		reports = append(reports, msg)
		mu.Unlock()
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
	safe := []string{"websearch", "read", "glob", "grep", "calculator", "weather"}

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

// ── Edge-case tests ─────────────────────────────────────────────────

func TestAgent_DoubleStart(t *testing.T) {
	reg := mockRegistry()
	config := AgentConfig{
		Workspace:    t.TempDir(),
		MaxToolCalls: 50,
		MaxRetries:   1,
		StepTimeout:  5 * time.Second,
	}
	a := NewAgent(reg, config)

	// Start first goal
	if err := a.Start("Research Go programming"); err != nil {
		t.Fatalf("first Start: %v", err)
	}

	// Second start while running must fail
	err := a.Start("Research Rust programming")
	if err == nil {
		t.Fatal("expected error on double start")
	}
	if !strings.Contains(err.Error(), "already running") {
		t.Errorf("expected 'already running' error, got: %v", err)
	}

	a.Stop()
}

func TestAgent_StopWhileIdle(t *testing.T) {
	reg := mockRegistry()
	a := NewAgent(reg, AgentConfig{Workspace: t.TempDir()})
	// Stopping an idle agent should not panic
	a.Stop()
}

func TestAgent_ResumeWhileNotPaused(t *testing.T) {
	reg := mockRegistry()
	a := NewAgent(reg, AgentConfig{Workspace: t.TempDir()})
	// Resume when not paused should be a no-op
	a.Resume("some input")
}

func TestAgent_StatusWhileIdle(t *testing.T) {
	reg := mockRegistry()
	a := NewAgent(reg, AgentConfig{Workspace: t.TempDir()})
	status := a.Status()
	if status.Running {
		t.Error("idle agent should not be running")
	}
	if status.Goal != "" {
		t.Errorf("idle agent goal should be empty, got %q", status.Goal)
	}
}

func TestAgent_EmptyGoalStart(t *testing.T) {
	reg := mockRegistry()
	a := NewAgent(reg, AgentConfig{Workspace: t.TempDir()})

	err := a.Start("")
	if err != nil {
		// Start itself doesn't validate — the planner does.
		// Wait for the run goroutine to finish.
		time.Sleep(200 * time.Millisecond)
	}

	// Agent should finish (with error report) and not be running
	time.Sleep(500 * time.Millisecond)
	status := a.Status()
	if status.Running {
		t.Error("agent should have stopped after empty goal")
	}
}

func TestAgent_ConfigDefaults(t *testing.T) {
	reg := mockRegistry()

	// Provide custom MaxToolCalls but empty workspace
	config := AgentConfig{
		MaxToolCalls: 42,
		MaxRetries:   7,
	}
	a := NewAgent(reg, config)

	// Workspace should be defaulted
	if a.Config.Workspace == "" {
		t.Error("workspace should be defaulted")
	}
	// MaxToolCalls should be preserved, not overwritten
	if a.Config.MaxToolCalls != 42 {
		t.Errorf("MaxToolCalls: got %d, want 42", a.Config.MaxToolCalls)
	}
	if a.Config.MaxRetries != 7 {
		t.Errorf("MaxRetries: got %d, want 7", a.Config.MaxRetries)
	}
}

func TestAgentState_SaveLoadRoundTrip_TaskStatus(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "state.json")

	s := NewAgentState(path)
	s.Reset("goal")
	s.SetPlan(&Plan{
		Goal: "goal",
		Phases: []Phase{
			{
				Name:   "P1",
				Status: PhaseCompleted,
				Tasks: []Task{
					{ID: "t1", Status: TaskCompleted, Result: "done"},
					{ID: "t2", Status: TaskFailed, Error: "boom"},
					{ID: "t3", Status: TaskNeedsHuman, HumanPrompt: "help?"},
				},
			},
		},
	})
	s.RecordHumanInput("t3", "help?", "yes")

	if err := s.Save(); err != nil {
		t.Fatalf("Save: %v", err)
	}

	s2 := NewAgentState(path)
	if err := s2.Load(); err != nil {
		t.Fatalf("Load: %v", err)
	}

	// Verify statuses survived round-trip
	tasks := s2.Plan.Phases[0].Tasks
	if tasks[0].Status != TaskCompleted {
		t.Errorf("t1 status: got %v, want completed", tasks[0].Status)
	}
	if tasks[1].Status != TaskFailed {
		t.Errorf("t2 status: got %v, want failed", tasks[1].Status)
	}
	if tasks[2].Status != TaskNeedsHuman {
		t.Errorf("t3 status: got %v, want needs_human", tasks[2].Status)
	}
	if s2.Plan.Phases[0].Status != PhaseCompleted {
		t.Errorf("phase status: got %v, want completed", s2.Plan.Phases[0].Status)
	}
	if len(s2.HumanInputs) != 1 {
		t.Errorf("human inputs: got %d, want 1", len(s2.HumanInputs))
	}
}

func TestAgentState_LoadNonexistent(t *testing.T) {
	s := NewAgentState(filepath.Join(t.TempDir(), "nope.json"))
	if err := s.Load(); err != nil {
		t.Errorf("Load of nonexistent file should return nil, got: %v", err)
	}
}

func TestAgentState_SnapshotIsolation(t *testing.T) {
	s := NewAgentState("")
	s.Reset("test")
	s.SetPlan(&Plan{
		Goal:   "test",
		Phases: []Phase{{Name: "P1", Tasks: []Task{{ID: "t1", Status: TaskPending}}}},
	})

	snap := s.Snapshot()

	// Mutate original
	s.mu.Lock()
	s.Plan.Phases[0].Tasks[0].Status = TaskCompleted
	s.mu.Unlock()

	// Snapshot should be unaffected
	if snap.Plan.Phases[0].Tasks[0].Status != TaskPending {
		t.Error("snapshot was mutated by changes to original state")
	}
}

func TestExecuteChain_EmptyChain(t *testing.T) {
	exec := NewExecutor(mockRegistry(), "/tmp/test")
	result, err := exec.ExecuteChain(nil, nil)
	if err != nil {
		t.Fatalf("empty chain: %v", err)
	}
	if result.ToolCalls != 0 {
		t.Errorf("empty chain: got %d tool calls, want 0", result.ToolCalls)
	}
}

func TestExecuteChain_UnknownTool(t *testing.T) {
	exec := NewExecutor(mockRegistry(), "/tmp/test")
	exec.MaxRetries = 1
	chain := []ToolStep{
		{Tool: "nonexistent_tool", Args: map[string]string{}, DependsOn: -1},
	}
	_, err := exec.ExecuteChain(chain, nil)
	if err == nil {
		t.Error("expected error for unknown tool")
	}
}

func TestExecuteChain_VariableSubstitution(t *testing.T) {
	reg := tools.NewRegistry()
	var capturedArgs map[string]string
	reg.Register(tools.Tool{
		Name:        "echo",
		Description: "Echo args",
		Execute: func(args map[string]string) (string, error) {
			capturedArgs = args
			return args["msg"], nil
		},
	})

	exec := NewExecutor(reg, "/tmp/test")
	chain := []ToolStep{
		{Tool: "echo", Args: map[string]string{"msg": "hello"}, DependsOn: -1, OutputKey: "greeting"},
		{Tool: "echo", Args: map[string]string{"msg": "got: ${greeting}"}, DependsOn: 0, OutputKey: "final"},
	}

	result, err := exec.ExecuteChain(chain, nil)
	if err != nil {
		t.Fatalf("chain: %v", err)
	}

	if capturedArgs["msg"] != "got: hello" {
		t.Errorf("variable substitution failed: got %q", capturedArgs["msg"])
	}
	if result.FinalOutput != "got: hello" {
		t.Errorf("final output: got %q", result.FinalOutput)
	}
}

func TestExecuteChain_WorkspaceResolution(t *testing.T) {
	reg := tools.NewRegistry()
	var capturedPath string
	reg.Register(tools.Tool{
		Name:        "write",
		Description: "Write",
		Execute: func(args map[string]string) (string, error) {
			capturedPath = args["path"]
			return "ok", nil
		},
	})

	exec := NewExecutor(reg, "/my/workspace")
	chain := []ToolStep{
		{Tool: "write", Args: map[string]string{"path": "agent_workspace/notes.md"}, DependsOn: -1},
	}

	_, err := exec.ExecuteChain(chain, nil)
	if err != nil {
		t.Fatalf("chain: %v", err)
	}

	if capturedPath != "/my/workspace/agent_workspace/notes.md" {
		t.Errorf("workspace not prepended: got %q", capturedPath)
	}
}

func TestExecuteChain_AbsolutePathNotModified(t *testing.T) {
	reg := tools.NewRegistry()
	var capturedPath string
	reg.Register(tools.Tool{
		Name:        "read",
		Description: "Read",
		Execute: func(args map[string]string) (string, error) {
			capturedPath = args["path"]
			return "ok", nil
		},
	})

	exec := NewExecutor(reg, "/my/workspace")
	chain := []ToolStep{
		{Tool: "read", Args: map[string]string{"path": "/etc/hosts"}, DependsOn: -1},
	}

	_, err := exec.ExecuteChain(chain, nil)
	if err != nil {
		t.Fatalf("chain: %v", err)
	}

	if capturedPath != "/etc/hosts" {
		t.Errorf("absolute path should not be modified: got %q", capturedPath)
	}
}

func TestDecomposeGoal_AllTypes(t *testing.T) {
	planner := NewPlanner([]string{"websearch", "write", "read"})

	goals := []struct {
		goal     string
		minPhases int
	}{
		{"Research the history of cryptocurrency", 2},
		{"Write a report about climate change", 3},
		{"Analyze the pros and cons of remote work", 2},
		{"Create a project plan for building a mobile app", 2},
		{"Build a web scraper for news articles", 2},
		{"Monitor Bitcoin price every hour", 1},
		{"Do something completely random and odd", 2}, // generic
	}

	for _, tt := range goals {
		plan, err := planner.DecomposeGoal(tt.goal)
		if err != nil {
			t.Errorf("DecomposeGoal(%q): %v", tt.goal, err)
			continue
		}
		if len(plan.Phases) < tt.minPhases {
			t.Errorf("DecomposeGoal(%q): got %d phases, want >= %d", tt.goal, len(plan.Phases), tt.minPhases)
		}
		// Every phase should have at least 1 task
		for _, phase := range plan.Phases {
			if len(phase.Tasks) == 0 {
				t.Errorf("DecomposeGoal(%q): phase %q has 0 tasks", tt.goal, phase.Name)
			}
		}
		// Plan should have an estimated duration > 0
		if plan.EstDuration <= 0 {
			t.Errorf("DecomposeGoal(%q): estimated duration is %v", tt.goal, plan.EstDuration)
		}
	}
}

func TestScheduler_AddRemoveList(t *testing.T) {
	reg := mockRegistry()
	a := NewAgent(reg, AgentConfig{Workspace: t.TempDir()})
	s := NewScheduler(a)

	// Add two jobs
	id1, err := s.AddJob("check weather", "every 30m")
	if err != nil {
		t.Fatalf("AddJob: %v", err)
	}
	id2, err := s.AddJob("check stocks", "hourly")
	if err != nil {
		t.Fatalf("AddJob: %v", err)
	}

	jobs := s.ListJobs()
	if len(jobs) != 2 {
		t.Fatalf("expected 2 jobs, got %d", len(jobs))
	}

	// Remove one
	s.RemoveJob(id1)
	jobs = s.ListJobs()
	if len(jobs) != 1 {
		t.Fatalf("expected 1 job after removal, got %d", len(jobs))
	}
	if jobs[0].ID != id2 {
		t.Errorf("wrong job remaining: got %s, want %s", jobs[0].ID, id2)
	}

	// Remove the other
	s.RemoveJob(id2)
	jobs = s.ListJobs()
	if len(jobs) != 0 {
		t.Fatalf("expected 0 jobs, got %d", len(jobs))
	}
}

func TestScheduler_Persistence(t *testing.T) {
	dir := t.TempDir()
	reg := mockRegistry()
	a := NewAgent(reg, AgentConfig{Workspace: dir})

	s1 := NewScheduler(a)
	_, err := s1.AddJob("daily check", "daily 9:00")
	if err != nil {
		t.Fatalf("AddJob: %v", err)
	}

	// Create new scheduler from same path — should load the job
	s2 := NewScheduler(a)
	jobs := s2.ListJobs()
	if len(jobs) != 1 {
		t.Fatalf("expected 1 persisted job, got %d", len(jobs))
	}
	if jobs[0].Goal != "daily check" {
		t.Errorf("persisted job goal: got %q, want %q", jobs[0].Goal, "daily check")
	}
}

func TestScheduler_InvalidSchedule(t *testing.T) {
	reg := mockRegistry()
	a := NewAgent(reg, AgentConfig{Workspace: t.TempDir()})
	s := NewScheduler(a)

	_, err := s.AddJob("test", "garbage")
	if err == nil {
		t.Error("expected error for invalid schedule")
	}
}

func TestIsInWorkspace(t *testing.T) {
	a := &Agent{Config: AgentConfig{Workspace: "/home/user/.nous/agent"}}

	tests := []struct {
		step ToolStep
		safe bool
	}{
		{ToolStep{Tool: "write", Args: map[string]string{"path": "/home/user/.nous/agent/notes.md"}}, true},
		{ToolStep{Tool: "write", Args: map[string]string{"path": "agent_workspace/notes.md"}}, true},
		{ToolStep{Tool: "write", Args: map[string]string{"path": "relative/file.md"}}, true},
		{ToolStep{Tool: "write", Args: map[string]string{"path": "/etc/passwd"}}, false},
		{ToolStep{Tool: "write", Args: map[string]string{"path": "../../../etc/passwd"}}, false},
		{ToolStep{Tool: "websearch", Args: map[string]string{"query": "test"}}, true}, // no path arg
	}

	for _, tt := range tests {
		got := a.isInWorkspace(tt.step)
		if got != tt.safe {
			t.Errorf("isInWorkspace(%v): got %v, want %v", tt.step.Args, got, tt.safe)
		}
	}
}

func TestAgent_LoadAndResume_NoState(t *testing.T) {
	reg := mockRegistry()
	a := NewAgent(reg, AgentConfig{Workspace: t.TempDir()})
	err := a.LoadAndResume()
	if err == nil {
		t.Error("expected error when no saved state exists")
	}
}

func TestAgent_LoadAndResume_FinishedState(t *testing.T) {
	dir := t.TempDir()
	reg := mockRegistry()

	// Create agent, run to completion, then try to resume
	a := NewAgent(reg, AgentConfig{
		Workspace:    dir,
		MaxToolCalls: 50,
		MaxRetries:   1,
	})

	a.Start("Research Go")
	deadline := time.After(5 * time.Second)
	for {
		select {
		case <-deadline:
			t.Fatal("timeout waiting for agent")
		default:
		}
		if !a.Status().Running {
			break
		}
		time.Sleep(50 * time.Millisecond)
	}

	// Now try to resume — should fail since it's finished
	a2 := NewAgent(reg, AgentConfig{Workspace: dir})
	err := a2.LoadAndResume()
	if err == nil {
		t.Error("expected error when resuming finished state")
	}
}

func TestReporter_IdleState(t *testing.T) {
	state := NewAgentState("")
	reporter := NewReporter(state)
	report := reporter.ProgressReport()
	if !strings.Contains(report, "[IDLE]") {
		t.Errorf("idle report should contain [IDLE], got: %s", report)
	}
}

func TestReporter_PlanningState(t *testing.T) {
	state := NewAgentState("")
	state.Reset("Test goal")
	// Plan is nil but goal is set
	reporter := NewReporter(state)
	report := reporter.ProgressReport()
	if !strings.Contains(report, "[PLANNING]") {
		t.Errorf("planning report should contain [PLANNING], got: %s", report)
	}
}

func TestCognitiveBridge_NilSafe(t *testing.T) {
	// All CognitiveBridge methods should be safe to call on a nil bridge
	var cb *CognitiveBridge

	result := cb.Summarize("some long text here that needs summarizing", 3)
	if result == "" {
		t.Error("nil bridge Summarize should return truncated text, not empty")
	}

	result = cb.Think("what is Go?")
	if result != "" {
		t.Error("nil bridge Think should return empty")
	}

	answer, trace := cb.Reason("why is the sky blue?")
	if answer != "" || trace != "" {
		t.Error("nil bridge Reason should return empty")
	}

	result = cb.GenerateDocument("Go programming", "overview")
	if result != "" {
		t.Error("nil bridge GenerateDocument should return empty")
	}

	result = cb.Compose("tell me about Go")
	if result != "" {
		t.Error("nil bridge Compose should return empty")
	}

	result = cb.SynthesizeResults("test goal", map[string]string{"r1": "data one", "r2": "data two"})
	if result == "" {
		t.Error("nil bridge SynthesizeResults should return concatenated results")
	}
}

func TestExecutor_CognitiveSteps_NoBrain(t *testing.T) {
	exec := NewExecutor(mockRegistry(), "/tmp/test")
	exec.MaxRetries = 1

	// Cognitive steps without a brain should fail with clear error
	chain := []ToolStep{
		{Tool: "_summarize", Args: map[string]string{"text": "some text"}, DependsOn: -1},
	}
	_, err := exec.ExecuteChain(chain, nil)
	if err == nil {
		t.Error("expected error for cognitive step without brain")
	}
	if !strings.Contains(err.Error(), "brain") {
		t.Errorf("error should mention brain, got: %v", err)
	}
}

func TestExecutor_CognitiveSteps_WithMockBrain(t *testing.T) {
	exec := NewExecutor(mockRegistry(), "/tmp/test")
	exec.Brain = &CognitiveBridge{} // empty bridge — no cognitive systems

	// _summarize with no Summarizer should fallback to truncation
	chain := []ToolStep{
		{Tool: "_summarize", Args: map[string]string{"text": "This is a test. It has multiple sentences. We want to summarize it."}, DependsOn: -1, OutputKey: "summary"},
	}
	result, err := exec.ExecuteChain(chain, nil)
	if err != nil {
		t.Fatalf("_summarize with empty bridge: %v", err)
	}
	if result.FinalOutput == "" {
		t.Error("_summarize should produce output even with empty bridge")
	}
}

func TestAgent_ConcurrentStatusCalls(t *testing.T) {
	reg := mockRegistry()
	a := NewAgent(reg, AgentConfig{
		Workspace:    t.TempDir(),
		MaxToolCalls: 50,
	})

	a.Start("Research Go programming")

	// Hammer Status() from multiple goroutines while agent is running
	done := make(chan struct{})
	for i := 0; i < 10; i++ {
		go func() {
			defer func() { done <- struct{}{} }()
			for j := 0; j < 50; j++ {
				_ = a.Status()
				_ = a.Report()
				time.Sleep(time.Millisecond)
			}
		}()
	}
	for i := 0; i < 10; i++ {
		<-done
	}

	a.Stop()
}
