package cognitive

import (
	"os"
	"path/filepath"
	"testing"
)

func TestRecipeBookRecord(t *testing.T) {
	dir := t.TempDir()
	rb := NewRecipeBook(dir)

	pipe := NewPipeline("show me the reasoner code")
	pipe.AddStep("grep", "internal/cognitive/reasoner.go:1:package cognitive")
	pipe.AddStep("read", "package cognitive\n\nimport (...)\n\ntype Reasoner struct {\n...")

	rb.Record(pipe, "question", "show me the reasoner code")

	if rb.Size() != 1 {
		t.Errorf("expected 1 recipe, got %d", rb.Size())
	}

	recipes := rb.List()
	if recipes[0].Trigger != "question" {
		t.Errorf("trigger = %q, want %q", recipes[0].Trigger, "question")
	}
	if recipes[0].Uses != 1 {
		t.Errorf("uses = %d, want 1", recipes[0].Uses)
	}
}

func TestRecipeBookSkipSingleStep(t *testing.T) {
	rb := NewRecipeBook("")

	pipe := NewPipeline("list files")
	pipe.AddStep("ls", "main.go\nREADME.md")

	rb.Record(pipe, "command", "list files")

	if rb.Size() != 0 {
		t.Error("single-step sequences should not be recorded")
	}
}

func TestRecipeBookSkipErrors(t *testing.T) {
	rb := NewRecipeBook("")

	pipe := NewPipeline("read something")
	pipe.AddStep("read", "Error: file not found")
	pipe.AddStep("ls", "main.go")

	rb.Record(pipe, "command", "read something")

	if rb.Size() != 0 {
		t.Error("sequences with errors should not be recorded")
	}
}

func TestRecipeBookDeduplicate(t *testing.T) {
	rb := NewRecipeBook("")

	// Record same sequence twice
	for i := 0; i < 2; i++ {
		pipe := NewPipeline("show code")
		pipe.AddStep("grep", "found matches")
		pipe.AddStep("read", "file content")
		rb.Record(pipe, "question", "show code")
	}

	if rb.Size() != 1 {
		t.Errorf("expected 1 recipe (deduplicated), got %d", rb.Size())
	}

	recipes := rb.List()
	if recipes[0].Uses != 2 {
		t.Errorf("uses = %d, want 2", recipes[0].Uses)
	}
	if recipes[0].Successes != 2 {
		t.Errorf("successes = %d, want 2", recipes[0].Successes)
	}
}

func TestRecipeBookMatch(t *testing.T) {
	rb := NewRecipeBook("")

	pipe := NewPipeline("find the function definition for NewReasoner")
	pipe.AddStep("grep", "internal/cognitive/reasoner.go:52:func NewReasoner")
	pipe.AddStep("read", "func NewReasoner(board *blackboard.Blackboard...")

	rb.Record(pipe, "question", "find the function definition for NewReasoner")

	// Should match a similar query
	matches := rb.Match("question", "find the function definition for NewPerceiver")
	if len(matches) == 0 {
		t.Fatal("expected at least one match for similar query")
	}
}

func TestRecipeBookMatchNoResults(t *testing.T) {
	rb := NewRecipeBook("")

	pipe := NewPipeline("find function")
	pipe.AddStep("grep", "matches")
	pipe.AddStep("read", "content")
	rb.Record(pipe, "question", "find function definition")

	// Completely different query
	matches := rb.Match("command", "delete all temp files")
	if len(matches) > 0 {
		t.Errorf("expected no matches for unrelated query, got %d", len(matches))
	}
}

func TestRecipeBookReplay(t *testing.T) {
	rb := NewRecipeBook("")

	pipe := NewPipeline("read main.go")
	pipe.AddStep("glob", "Found 1 files matching pattern")
	pipe.AddStep("read", "Read main.go: package main... (50 lines)")

	rb.Record(pipe, "command", "read main.go")

	recipes := rb.List()
	if len(recipes) == 0 {
		t.Fatal("no recipes")
	}

	steps, err := rb.Replay(recipes[0].ID, map[string]string{"$FILE": "server.go"})
	if err != nil {
		t.Fatalf("Replay: %v", err)
	}

	if len(steps) != 2 {
		t.Errorf("expected 2 steps, got %d", len(steps))
	}
}

func TestRecipeBookReplayNotFound(t *testing.T) {
	rb := NewRecipeBook("")

	_, err := rb.Replay("nonexistent", nil)
	if err == nil {
		t.Error("expected error for nonexistent recipe")
	}
}

func TestRecipeBookPersistence(t *testing.T) {
	dir := t.TempDir()

	// Create and record
	rb1 := NewRecipeBook(dir)
	pipe := NewPipeline("test")
	pipe.AddStep("ls", "files")
	pipe.AddStep("read", "content")
	rb1.Record(pipe, "command", "test query")

	// Load from disk
	rb2 := NewRecipeBook(dir)
	if rb2.Size() != 1 {
		t.Errorf("expected 1 recipe after reload, got %d", rb2.Size())
	}

	// Verify file exists
	path := filepath.Join(dir, "recipes.json")
	if _, err := os.Stat(path); err != nil {
		t.Errorf("recipes.json not found: %v", err)
	}
}

func TestRecipeConfidence(t *testing.T) {
	r := Recipe{Uses: 10, Successes: 8}
	if c := r.Confidence(); c != 0.8 {
		t.Errorf("Confidence() = %f, want 0.8", c)
	}

	r2 := Recipe{Uses: 0}
	if c := r2.Confidence(); c != 0.5 {
		t.Errorf("Confidence() = %f, want 0.5 for new recipe", c)
	}
}

func TestExtractKeywords(t *testing.T) {
	keywords := extractKeywords("How can I read the main.go file please?")
	// Should include "read", "main.go", "file" but not "how", "can", "the", "please"
	found := make(map[string]bool)
	for _, k := range keywords {
		found[k] = true
	}
	if found["how"] || found["can"] || found["the"] || found["please"] {
		t.Errorf("stop words not filtered: %v", keywords)
	}
	if !found["read"] || !found["file"] {
		t.Errorf("expected 'read' and 'file' in keywords: %v", keywords)
	}
}

func TestKeywordOverlap(t *testing.T) {
	a := []string{"read", "file", "main", "code"}
	b := []string{"read", "file", "server"}

	overlap := keywordOverlap(a, b)
	// 2 out of 4 match
	if overlap != 0.5 {
		t.Errorf("overlap = %f, want 0.5", overlap)
	}

	// Empty cases
	if keywordOverlap(nil, b) != 0 {
		t.Error("nil a should return 0")
	}
	if keywordOverlap(a, nil) != 0 {
		t.Error("nil b should return 0")
	}
}

func TestGenerateRecipeName(t *testing.T) {
	steps := []StepResult{
		{ToolName: "grep"},
		{ToolName: "read"},
	}
	name := generateRecipeName(steps)
	if name != "grep→read" {
		t.Errorf("name = %q, want %q", name, "grep→read")
	}

	// Empty
	if generateRecipeName(nil) != "empty" {
		t.Error("empty steps should produce 'empty'")
	}
}

func TestRecipeBookPrune(t *testing.T) {
	rb := NewRecipeBook("")

	// Add 55 recipes (exceeds cap of 50)
	for i := 0; i < 55; i++ {
		pipe := NewPipeline("query")
		pipe.AddStep("ls", "files")
		pipe.AddStep("read", "content")
		// Make each unique by using different tool names
		toolName := "read"
		if i%2 == 0 {
			toolName = "grep"
		}
		if i%3 == 0 {
			toolName = "glob"
		}
		pipe.steps[1].ToolName = toolName
		rb.mu.Lock()
		rb.recipes = append(rb.recipes, Recipe{
			ID:        "r" + string(rune('A'+i%26)) + string(rune('0'+i/26)),
			Steps:     []RecipeStep{{Tool: "ls"}, {Tool: toolName}},
			Uses:      i + 1,
			Successes: i,
		})
		rb.mu.Unlock()
	}

	rb.mu.Lock()
	rb.prune()
	rb.mu.Unlock()

	if rb.Size() > 40 {
		t.Errorf("prune should reduce to 40, got %d", rb.Size())
	}
}

func TestReportSuccess(t *testing.T) {
	rb := NewRecipeBook("")

	pipe := NewPipeline("test")
	pipe.AddStep("ls", "files")
	pipe.AddStep("read", "content")
	rb.Record(pipe, "command", "test")

	recipes := rb.List()
	id := recipes[0].ID

	rb.ReportSuccess(id)

	rb.mu.RLock()
	for _, r := range rb.recipes {
		if r.ID == id {
			if r.Successes != 2 { // 1 from record + 1 from report
				t.Errorf("successes = %d, want 2", r.Successes)
			}
		}
	}
	rb.mu.RUnlock()
}
