package cognitive

import (
	"os"
	"path/filepath"
	"strings"
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

func TestRecipeBookReplayParameterSubstitution(t *testing.T) {
	rb := NewRecipeBook("")

	// Manually add a recipe with parameters
	rb.mu.Lock()
	rb.recipes = append(rb.recipes, Recipe{
		ID:      "test_param",
		Name:    "grep→read",
		Trigger: "question",
		Steps: []RecipeStep{
			{Tool: "grep", Args: map[string]string{"pattern": "$PATTERN"}},
			{Tool: "read", Args: map[string]string{"path": "$FILE"}},
		},
		Params:    []string{"$PATTERN", "$FILE"},
		Uses:      1,
		Successes: 1,
	})
	rb.mu.Unlock()

	steps, err := rb.Replay("test_param", map[string]string{
		"$PATTERN": "NewReasoner",
		"$FILE":    "internal/cognitive/reasoner.go",
	})
	if err != nil {
		t.Fatalf("Replay error: %v", err)
	}
	if steps[0].Args["pattern"] != "NewReasoner" {
		t.Errorf("pattern not substituted: %s", steps[0].Args["pattern"])
	}
	if steps[1].Args["path"] != "internal/cognitive/reasoner.go" {
		t.Errorf("path not substituted: %s", steps[1].Args["path"])
	}
}

func TestRecipeBookMatchSortsByScore(t *testing.T) {
	rb := NewRecipeBook("")

	// Add recipes with different relevance
	rb.mu.Lock()
	rb.recipes = append(rb.recipes,
		Recipe{
			ID: "low", Trigger: "question",
			Keywords: []string{"unrelated", "stuff"},
			Steps:    []RecipeStep{{Tool: "ls"}, {Tool: "read"}},
			Uses: 10, Successes: 9,
		},
		Recipe{
			ID: "high", Trigger: "question",
			Keywords: []string{"find", "function", "definition"},
			Steps:    []RecipeStep{{Tool: "grep"}, {Tool: "read"}},
			Uses: 5, Successes: 5,
		},
	)
	rb.mu.Unlock()

	matches := rb.Match("question", "find the function definition")
	if len(matches) == 0 {
		t.Fatal("expected matches")
	}
	if matches[0].ID != "high" {
		t.Errorf("expected 'high' recipe first, got %q", matches[0].ID)
	}
}

func TestRecipeBookMatchMaxResults(t *testing.T) {
	rb := NewRecipeBook("")

	// Add 10 matching recipes
	rb.mu.Lock()
	for i := 0; i < 10; i++ {
		rb.recipes = append(rb.recipes, Recipe{
			ID:       string(rune('A' + i)),
			Trigger:  "question",
			Keywords: []string{"find", "function"},
			Steps:    []RecipeStep{{Tool: "grep"}, {Tool: "read"}},
			Uses:     1,
			Successes: 1,
		})
	}
	rb.mu.Unlock()

	matches := rb.Match("question", "find the function")
	if len(matches) > 3 {
		t.Errorf("expected max 3 matches, got %d", len(matches))
	}
}

func TestRecipeBookEmptyInputMatch(t *testing.T) {
	rb := NewRecipeBook("")
	rb.mu.Lock()
	rb.recipes = append(rb.recipes, Recipe{
		ID: "x", Trigger: "question", Keywords: []string{"test"},
		Steps: []RecipeStep{{Tool: "ls"}, {Tool: "read"}},
	})
	rb.mu.Unlock()

	// Empty input should return nil
	matches := rb.Match("question", "")
	if len(matches) != 0 {
		t.Errorf("empty input should return no matches, got %d", len(matches))
	}
}

func TestLooksLikePathVariants(t *testing.T) {
	tests := []struct {
		input string
		want  bool
	}{
		{"internal/cognitive/reasoner.go", true},
		{"main.go", true},
		{"README.md", true},
		{"script.py", true},
		{"index.js", true},
		{"style.ts", true},
		{"some-dir/", true},
		{"plain-text", false},
		{"1234", false},
		{"", false},
	}
	for _, tt := range tests {
		if got := looksLikePath(tt.input); got != tt.want {
			t.Errorf("looksLikePath(%q) = %v, want %v", tt.input, got, tt.want)
		}
	}
}

func TestGenerateRecipeNameLong(t *testing.T) {
	var steps []StepResult
	for i := 0; i < 5; i++ {
		steps = append(steps, StepResult{ToolName: "tool" + string(rune('A'+i))})
	}
	name := generateRecipeName(steps)
	if name == "" {
		t.Error("expected non-empty name for long recipe")
	}
	// Should contain step count for >3 unique tools
	if len(steps) > 3 {
		if !strings.Contains(name, "steps") {
			t.Errorf("long recipe name should mention step count, got: %s", name)
		}
	}
}

func TestRecipeConfidenceEdgeCases(t *testing.T) {
	tests := []struct {
		name       string
		uses       int
		successes  int
		wantConf   float64
	}{
		{"new recipe", 0, 0, 0.5},
		{"perfect", 10, 10, 1.0},
		{"half", 10, 5, 0.5},
		{"poor", 10, 1, 0.1},
		{"single use success", 1, 1, 1.0},
		{"single use failure", 1, 0, 0.0},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			r := Recipe{Uses: tt.uses, Successes: tt.successes}
			if got := r.Confidence(); got != tt.wantConf {
				t.Errorf("Confidence() = %f, want %f", got, tt.wantConf)
			}
		})
	}
}

func TestExtractKeywordsStopWords(t *testing.T) {
	// All stop words should be filtered
	keywords := extractKeywords("the and for that this with from are was have has can will how what where when who which does please could would should")
	if len(keywords) != 0 {
		t.Errorf("all stop words should be filtered, got: %v", keywords)
	}
}

func TestExtractKeywordsShortWords(t *testing.T) {
	// Words < 3 chars should be filtered
	keywords := extractKeywords("a i go do to be it of on at")
	if len(keywords) != 0 {
		t.Errorf("short words should be filtered, got: %v", keywords)
	}
}

func TestExtractKeywordsPunctuation(t *testing.T) {
	keywords := extractKeywords("find, the 'function' definition! in (main.go)")
	found := make(map[string]bool)
	for _, k := range keywords {
		found[k] = true
	}
	if !found["find"] || !found["function"] || !found["definition"] {
		t.Errorf("punctuation should be stripped, got: %v", keywords)
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
