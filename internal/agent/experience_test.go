package agent

import (
	"os"
	"path/filepath"
	"testing"
)

func TestExperienceMemory(t *testing.T) {
	dir := t.TempDir()
	em := NewExperienceMemory(dir)

	// Record some experiences.
	em.Record(ExperienceEntry{
		GoalType:    "research",
		ToolChain:   []string{"websearch", "summarize"},
		Succeeded:   true,
		OutputWords: 150,
		Goal:        "research machine learning",
	})
	em.Record(ExperienceEntry{
		GoalType:    "research",
		ToolChain:   []string{"websearch", "fetch"},
		Succeeded:   false,
		OutputWords: 0,
		Goal:        "research quantum computing",
	})
	em.Record(ExperienceEntry{
		GoalType:    "research",
		ToolChain:   []string{"websearch", "summarize"},
		Succeeded:   true,
		OutputWords: 200,
		Goal:        "research artificial intelligence",
	})

	if len(em.Entries) != 3 {
		t.Fatalf("Expected 3 entries, got %d", len(em.Entries))
	}

	// Test tool scores.
	scores := em.ToolScoresForGoal("research")
	if len(scores) == 0 {
		t.Fatal("Expected tool scores for research goals")
	}

	// websearch: 3 uses, 2 successes = 66%
	for _, s := range scores {
		if s.Tool == "websearch" {
			if s.Uses != 3 {
				t.Errorf("websearch uses = %d, want 3", s.Uses)
			}
			if s.Successes != 2 {
				t.Errorf("websearch successes = %d, want 2", s.Successes)
			}
		}
	}

	// Test best tools.
	best := em.BestToolsForGoal("research", 5)
	if len(best) == 0 {
		t.Error("Expected best tools for research")
	}

	// Test persistence — reload from file.
	em2 := NewExperienceMemory(dir)
	if len(em2.Entries) != 3 {
		t.Errorf("Loaded entries = %d, want 3", len(em2.Entries))
	}

	// Verify file exists.
	if _, err := os.Stat(filepath.Join(dir, "experience.json")); os.IsNotExist(err) {
		t.Error("experience.json not created")
	}
}

func TestExperienceSimilarGoals(t *testing.T) {
	dir := t.TempDir()
	em := NewExperienceMemory(dir)

	em.Record(ExperienceEntry{
		GoalType:  "research",
		ToolChain: []string{"websearch"},
		Succeeded: true,
		Goal:      "research machine learning trends",
	})
	em.Record(ExperienceEntry{
		GoalType:  "writing",
		ToolChain: []string{"write"},
		Succeeded: true,
		Goal:      "write a blog post about cooking",
	})

	similar := em.SimilarGoalOutcomes("research machine learning")
	if len(similar) != 1 {
		t.Errorf("Expected 1 similar goal, got %d", len(similar))
	}
}

func TestExperienceFailedTools(t *testing.T) {
	dir := t.TempDir()
	em := NewExperienceMemory(dir)

	// Record 3+ failures for a tool.
	for i := 0; i < 4; i++ {
		em.Record(ExperienceEntry{
			GoalType:  "analysis",
			ToolChain: []string{"badtool"},
			Succeeded: false,
			Goal:      "analyze something",
		})
	}

	failed := em.FailedToolsForGoal("analysis")
	found := false
	for _, f := range failed {
		if f == "badtool" {
			found = true
		}
	}
	if !found {
		t.Error("Expected badtool to be in failed tools list")
	}
}

func TestGoalTypeString(t *testing.T) {
	tests := []struct {
		gt   goalType
		want string
	}{
		{goalResearch, "research"},
		{goalWriting, "writing"},
		{goalAnalysis, "analysis"},
		{goalGeneric, "generic"},
	}
	for _, tt := range tests {
		got := goalTypeString(tt.gt)
		if got != tt.want {
			t.Errorf("goalTypeString(%d) = %q, want %q", tt.gt, got, tt.want)
		}
	}
}
