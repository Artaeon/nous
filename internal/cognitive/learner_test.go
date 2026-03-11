package cognitive

import (
	"context"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/artaeon/nous/internal/blackboard"
)

func TestLearnerLearnFromCompletedGoal(t *testing.T) {
	board := blackboard.New()
	dir := t.TempDir()
	learner := NewLearner(board, nil, dir)

	// Post a percept for context
	board.PostPercept(blackboard.Percept{
		Raw:    "read the file and fix the bug",
		Intent: "fix",
	})

	// Create and complete a goal+plan
	goalID := "test-goal-1"
	board.PushGoal(blackboard.Goal{
		ID:          goalID,
		Description: "fix the bug",
		Status:      "pending",
		CreatedAt:   time.Now(),
	})
	board.SetPlan(blackboard.Plan{
		GoalID: goalID,
		Steps: []blackboard.Step{
			{ID: "s1", Tool: "read", Description: "read main.go", Status: "done"},
			{ID: "s2", Tool: "edit", Description: "fix the bug", Status: "done"},
		},
		Status: "completed",
	})

	learner.learnFromGoal(goalID)

	if learner.PatternCount() != 1 {
		t.Fatalf("expected 1 pattern, got %d", learner.PatternCount())
	}

	patterns := learner.Patterns()
	if patterns[0].Trigger != "fix" {
		t.Errorf("expected trigger 'fix', got %q", patterns[0].Trigger)
	}
	if len(patterns[0].ToolChain) != 2 {
		t.Errorf("expected 2-step tool chain, got %d", len(patterns[0].ToolChain))
	}
	if patterns[0].ToolChain[0] != "read" || patterns[0].ToolChain[1] != "edit" {
		t.Errorf("expected read→edit chain, got %v", patterns[0].ToolChain)
	}
}

func TestLearnerReinforcesExistingPattern(t *testing.T) {
	board := blackboard.New()
	learner := NewLearner(board, nil, "")

	board.PostPercept(blackboard.Percept{Raw: "test", Intent: "explore"})

	// Same tool chain twice
	for i := 0; i < 2; i++ {
		goalID := "goal-" + string(rune('a'+i))
		board.PushGoal(blackboard.Goal{ID: goalID, Status: "pending"})
		board.SetPlan(blackboard.Plan{
			GoalID: goalID,
			Steps: []blackboard.Step{
				{Tool: "grep", Status: "done"},
				{Tool: "read", Status: "done"},
			},
			Status: "completed",
		})
		learner.learnFromGoal(goalID)
	}

	if learner.PatternCount() != 1 {
		t.Fatalf("same tool chain should be reinforced, not duplicated; got %d patterns", learner.PatternCount())
	}

	p := learner.Patterns()[0]
	if p.Uses != 2 {
		t.Errorf("expected 2 uses, got %d", p.Uses)
	}
	if p.Confidence != 1.0 {
		t.Errorf("expected confidence 1.0, got %.2f", p.Confidence)
	}
}

func TestLearnerFindRelevantPatterns(t *testing.T) {
	board := blackboard.New()
	learner := NewLearner(board, nil, "")

	board.PostPercept(blackboard.Percept{Raw: "fix", Intent: "fix"})
	board.PushGoal(blackboard.Goal{ID: "g1", Status: "pending"})
	board.SetPlan(blackboard.Plan{
		GoalID: "g1",
		Steps:  []blackboard.Step{{Tool: "read", Status: "done"}},
		Status: "completed",
	})
	learner.learnFromGoal("g1")

	relevant := learner.FindRelevantPatterns("fix")
	if len(relevant) != 1 {
		t.Errorf("expected 1 relevant pattern for 'fix', got %d", len(relevant))
	}

	irrelevant := learner.FindRelevantPatterns("create")
	if len(irrelevant) != 0 {
		t.Errorf("expected 0 patterns for 'create', got %d", len(irrelevant))
	}
}

func TestLearnerSkipsIncompleteGoals(t *testing.T) {
	board := blackboard.New()
	learner := NewLearner(board, nil, "")

	board.PushGoal(blackboard.Goal{ID: "g1", Status: "pending"})
	board.SetPlan(blackboard.Plan{
		GoalID: "g1",
		Steps:  []blackboard.Step{{Tool: "read", Status: "running"}},
		Status: "executing",
	})
	learner.learnFromGoal("g1")

	if learner.PatternCount() != 0 {
		t.Error("should not learn from incomplete goals")
	}
}

func TestLearnerPersistence(t *testing.T) {
	dir := t.TempDir()
	board := blackboard.New()

	// Create learner, learn something, save
	l1 := NewLearner(board, nil, dir)
	board.PostPercept(blackboard.Percept{Raw: "test", Intent: "verify"})
	board.PushGoal(blackboard.Goal{ID: "g1", Status: "pending"})
	board.SetPlan(blackboard.Plan{
		GoalID: "g1",
		Steps:  []blackboard.Step{{Tool: "shell", Status: "done", Description: "run tests"}},
		Status: "completed",
	})
	l1.learnFromGoal("g1")

	// Check file was saved
	_, err := os.Stat(filepath.Join(dir, "patterns.json"))
	if err != nil {
		t.Fatal("patterns.json should be saved")
	}

	// Create new learner from same path — should load patterns
	l2 := NewLearner(board, nil, dir)
	if l2.PatternCount() != 1 {
		t.Errorf("loaded learner should have 1 pattern, got %d", l2.PatternCount())
	}
}

func TestLearnerRunsAsStream(t *testing.T) {
	board := blackboard.New()
	learner := NewLearner(board, nil, "")

	ctx, cancel := context.WithCancel(context.Background())
	go learner.Run(ctx)
	time.Sleep(50 * time.Millisecond)

	board.PostPercept(blackboard.Percept{Raw: "test", Intent: "verify"})

	goalID := "stream-goal"
	board.PushGoal(blackboard.Goal{ID: goalID, Status: "pending"})
	board.SetPlan(blackboard.Plan{
		GoalID: goalID,
		Steps:  []blackboard.Step{{Tool: "read", Status: "done"}},
		Status: "completed",
	})
	board.UpdateGoalStatus(goalID, "completed")

	time.Sleep(100 * time.Millisecond)
	cancel()

	if learner.PatternCount() != 1 {
		t.Errorf("Learner stream should learn from goal_updated events, got %d patterns", learner.PatternCount())
	}
}

func TestClassifyTrigger(t *testing.T) {
	tests := []struct {
		input string
		want  string
	}{
		{"read the file", "explore"},
		{"show me the code", "explore"},
		{"fix the bug in main.go", "fix"},
		{"create a new test", "create"},
		{"write a function", "create"},
		{"refactor the handler", "modify"},
		{"run the tests", "verify"},
		{"hello there", "general"},
	}

	for _, tt := range tests {
		got := classifyTrigger(tt.input)
		if got != tt.want {
			t.Errorf("classifyTrigger(%q) = %q, want %q", tt.input, got, tt.want)
		}
	}
}
