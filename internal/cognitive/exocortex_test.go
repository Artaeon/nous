package cognitive

import (
	"strings"
	"testing"
)

// --- Exocortex Tests ---

func TestExocortexCreation(t *testing.T) {
	ex := NewExocortex(nil, nil, nil, nil, nil, nil)
	if ex == nil {
		t.Fatal("NewExocortex should not return nil")
	}
}

func TestExocortexTier3Default(t *testing.T) {
	ex := NewExocortex(nil, nil, nil, nil, nil, nil)

	result := ex.Process("hello, how are you?")
	if result.Tier != 3 {
		t.Errorf("conversational query should be tier 3, got %d", result.Tier)
	}
	if result.LLMBypassed {
		t.Error("tier 3 should not bypass LLM")
	}
}

func TestExocortexTier2Classification(t *testing.T) {
	ex := NewExocortex(nil, nil, nil, nil, nil, nil)

	tier2Queries := []string{
		"how does the Pipeline work",
		"explain how the reasoner processes queries",
		"what does NewReasoner do",
		"describe the architecture",
		"summarize the codebase",
	}

	for _, q := range tier2Queries {
		result := ex.Process(q)
		if result.Tier != 2 {
			t.Errorf("%q should be tier 2, got tier %d", q, result.Tier)
		}
	}
}

func TestExocortexTier3Conversational(t *testing.T) {
	ex := NewExocortex(nil, nil, nil, nil, nil, nil)

	tier3Queries := []string{
		"what do you think about Go",
		"can you explain polymorphism",
		"how would you solve this",
		"should I use channels here",
		"tell me about concurrency patterns",
	}

	for _, q := range tier3Queries {
		result := ex.Process(q)
		if result.Tier != 3 {
			t.Errorf("%q should be tier 3, got tier %d", q, result.Tier)
		}
	}
}

func TestExocortexFormatStatusLine(t *testing.T) {
	tests := []struct {
		tier     int
		contains string
	}{
		{1, "bypass"},
		{2, "scaffold"},
		{3, "full"},
	}

	for _, tt := range tests {
		result := &ExoResult{Tier: tt.tier}
		status := result.FormatStatusLine()
		if !strings.Contains(status, tt.contains) {
			t.Errorf("tier %d status should contain %q, got %q", tt.tier, tt.contains, status)
		}
	}
}

// --- Neural Scaffold Tests ---

func TestNeuralScaffoldCreation(t *testing.T) {
	ns := NewNeuralScaffold()
	if ns == nil {
		t.Fatal("NewNeuralScaffold should not return nil")
	}
}

func TestScaffoldGrepWithResults(t *testing.T) {
	ns := NewNeuralScaffold()
	result := "file.go:10:func main()\nfile.go:20:func helper()"
	prompt := ns.BuildFromToolResult("find main", "grep", map[string]string{"pattern": "main"}, result)

	if prompt.ResponseSeed == "" {
		t.Error("grep with results should have a response seed")
	}
	if !strings.Contains(prompt.ResponseSeed, "2 matches") {
		t.Errorf("seed should mention match count, got: %s", prompt.ResponseSeed)
	}
	if !strings.Contains(prompt.ResponseSeed, "main") {
		t.Error("seed should mention pattern")
	}
}

func TestScaffoldGrepNoResults(t *testing.T) {
	ns := NewNeuralScaffold()
	prompt := ns.BuildFromToolResult("find xyz", "grep", map[string]string{"pattern": "xyz"}, "")

	if !strings.Contains(prompt.ResponseSeed, "no matches") {
		t.Errorf("empty grep should say no matches, got: %s", prompt.ResponseSeed)
	}
}

func TestScaffoldRead(t *testing.T) {
	ns := NewNeuralScaffold()
	content := "package main\n\nfunc main() {}\n"
	prompt := ns.BuildFromToolResult("read main.go", "read", map[string]string{"path": "main.go"}, content)

	if !strings.Contains(prompt.ResponseSeed, "main.go") {
		t.Errorf("read seed should mention file, got: %s", prompt.ResponseSeed)
	}
	if !strings.Contains(prompt.ResponseSeed, "3 lines") {
		t.Errorf("read seed should mention line count, got: %s", prompt.ResponseSeed)
	}
}

func TestScaffoldLs(t *testing.T) {
	ns := NewNeuralScaffold()
	result := "cmd/\ninternal/\ngo.mod"
	prompt := ns.BuildFromToolResult("list files", "ls", map[string]string{"path": "."}, result)

	if !strings.Contains(prompt.ResponseSeed, "3 entries") {
		t.Errorf("ls seed should count entries, got: %s", prompt.ResponseSeed)
	}
}

func TestScaffoldTree(t *testing.T) {
	ns := NewNeuralScaffold()
	result := ".\n├── cmd\n└── internal"
	prompt := ns.BuildFromToolResult("show structure", "tree", map[string]string{}, result)

	if !strings.Contains(prompt.ResponseSeed, "structure") {
		t.Errorf("tree seed should mention structure, got: %s", prompt.ResponseSeed)
	}
}

func TestScaffoldGlob(t *testing.T) {
	ns := NewNeuralScaffold()
	result := "a.go\nb.go\nc.go"
	prompt := ns.BuildFromToolResult("find go files", "glob", map[string]string{"pattern": "*.go"}, result)

	if !strings.Contains(prompt.ResponseSeed, "3 files") {
		t.Errorf("glob seed should count files, got: %s", prompt.ResponseSeed)
	}
}

func TestScaffoldGitStatus(t *testing.T) {
	ns := NewNeuralScaffold()
	result := "nothing to commit, working tree clean"
	prompt := ns.BuildFromToolResult("git status", "git", map[string]string{"command": "status"}, result)

	if !strings.Contains(prompt.ResponseSeed, "clean") {
		t.Errorf("clean git should say clean, got: %s", prompt.ResponseSeed)
	}
}

func TestScaffoldGitLog(t *testing.T) {
	ns := NewNeuralScaffold()
	result := "abc123 Initial commit"
	prompt := ns.BuildFromToolResult("show log", "git", map[string]string{"command": "log"}, result)

	if !strings.Contains(prompt.ResponseSeed, "commits") {
		t.Errorf("git log seed should mention commits, got: %s", prompt.ResponseSeed)
	}
}

func TestScaffoldMultipleResults(t *testing.T) {
	ns := NewNeuralScaffold()
	steps := []synthStep{
		{Tool: "grep", Args: map[string]string{"pattern": "Pipeline"}, Result: "pipe.go:5:type Pipeline"},
		{Tool: "read", Args: map[string]string{"path": "pipe.go"}, Result: "package cog\ntype Pipeline struct{}"},
	}

	prompt := ns.BuildFromMultipleResults("find Pipeline", steps)
	if prompt.ResponseSeed == "" {
		t.Error("multi-result scaffold should have a seed")
	}
	if !strings.Contains(prompt.UserMessage, "Pipeline") {
		t.Error("user message should contain the query")
	}
}

// --- Validation Tests ---

func TestValidateResponseHallucination(t *testing.T) {
	ns := NewNeuralScaffold()

	// Model says "no results" but results exist
	response := "I searched but no results found for your query."
	seed := "I found 5 matches for `Pipeline`:"
	result := "pipe.go:5:type Pipeline\npipe.go:10:func NewPipeline"

	validated := ns.ValidateResponse(response, seed, "grep", result)
	if validated == response {
		t.Error("should reject hallucinated 'no results' response")
	}
	if validated != seed {
		t.Errorf("should return seed, got: %s", validated)
	}
}

func TestValidateResponseCorrect(t *testing.T) {
	ns := NewNeuralScaffold()

	response := "Found 3 matches in reasoner.go for Pipeline."
	seed := "I found 3 matches for `Pipeline`:"
	result := "reasoner.go:5:Pipeline\nreasoner.go:10:Pipeline\nreasoner.go:15:Pipeline"

	validated := ns.ValidateResponse(response, seed, "grep", result)
	if validated != response {
		t.Error("correct response should not be modified")
	}
}

func TestValidateResponseEmptyResult(t *testing.T) {
	ns := NewNeuralScaffold()

	// When results are genuinely empty, "no results" is correct
	response := "No matches found for xyz."
	seed := "I searched for `xyz` but found no matches."

	validated := ns.ValidateResponse(response, seed, "grep", "")
	if validated != response {
		t.Error("should accept 'no results' when results are genuinely empty")
	}
}

func TestValidateResponseVariousHallucinations(t *testing.T) {
	ns := NewNeuralScaffold()
	seed := "I found 3 matches:"
	result := "a.go:1:match\nb.go:2:match\nc.go:3:match"

	hallucinations := []string{
		"I couldn't find any matches",
		"The search returned no results",
		"No occurrences were found",
		"The file does not contain the pattern",
		"Nothing was found matching your query",
	}

	for _, h := range hallucinations {
		validated := ns.ValidateResponse(h, seed, "grep", result)
		if validated == h {
			t.Errorf("should reject hallucination: %q", h)
		}
	}
}

// --- Helper Tests ---

func TestCountNonEmpty(t *testing.T) {
	lines := []string{"a", "", "b", "", "", "c"}
	if countNonEmpty(lines) != 3 {
		t.Errorf("should count 3 non-empty, got %d", countNonEmpty(lines))
	}
}

func TestTruncateEvidence(t *testing.T) {
	short := "short text"
	if truncateEvidence(short, 100) != short {
		t.Error("short text should not be truncated")
	}

	long := strings.Repeat("x", 200)
	result := truncateEvidence(long, 50)
	if len(result) > 70 { // 50 + "... (truncated)"
		t.Errorf("should be truncated, got length %d", len(result))
	}
}

// --- Benchmark ---

func BenchmarkExocortexProcess(b *testing.B) {
	ex := NewExocortex(nil, nil, nil, nil, nil, nil)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ex.Process("what do you think about Go?")
	}
}

func BenchmarkNeuralScaffold(b *testing.B) {
	ns := NewNeuralScaffold()
	result := "file.go:10:func main()\nfile.go:20:func helper()\nfile.go:30:func init()"
	args := map[string]string{"pattern": "func"}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ns.BuildFromToolResult("find functions", "grep", args, result)
	}
}
