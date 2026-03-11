package cognitive

import (
	"errors"
	"strings"
	"testing"

	"github.com/artaeon/nous/internal/ollama"
)

// --- ContextBudget tests ---

func TestContextBudgetEstimateTokens(t *testing.T) {
	b := DefaultBudget()
	// 400 chars / 4.0 chars per token = 100 tokens
	got := b.EstimateTokens(strings.Repeat("a", 400))
	if got != 100 {
		t.Errorf("EstimateTokens(400 chars) = %d, want 100", got)
	}
}

func TestContextBudgetEstimateMessages(t *testing.T) {
	b := DefaultBudget()
	msgs := []ollama.Message{
		{Role: "system", Content: strings.Repeat("x", 800)},  // 200 + 4
		{Role: "user", Content: strings.Repeat("x", 400)},    // 100 + 4
	}
	got := b.EstimateMessages(msgs)
	want := 308 // (200+4) + (100+4)
	if got != want {
		t.Errorf("EstimateMessages = %d, want %d", got, want)
	}
}

func TestContextBudgetRemaining(t *testing.T) {
	b := &ContextBudget{MaxTokens: 1000, CharsPerToken: 4.0}
	msgs := []ollama.Message{
		{Role: "user", Content: strings.Repeat("x", 2000)}, // 500 + 4 = 504
	}
	got := b.Remaining(msgs)
	want := 496
	if got != want {
		t.Errorf("Remaining = %d, want %d", got, want)
	}
}

func TestContextBudgetShouldCompress(t *testing.T) {
	b := &ContextBudget{MaxTokens: 100, CharsPerToken: 1.0}
	// 80 chars + 4 overhead = 84 tokens = 84% usage > 75%
	msgs := []ollama.Message{
		{Role: "user", Content: strings.Repeat("x", 80)},
	}
	if !b.ShouldCompress(msgs) {
		t.Error("ShouldCompress should be true at 84% usage")
	}
}

func TestContextBudgetShouldForceAnswer(t *testing.T) {
	b := &ContextBudget{MaxTokens: 100, CharsPerToken: 1.0}
	// 90 chars + 4 overhead = 94 tokens = 94% usage > 85%
	msgs := []ollama.Message{
		{Role: "user", Content: strings.Repeat("x", 90)},
	}
	if !b.ShouldForceAnswer(msgs) {
		t.Error("ShouldForceAnswer should be true at 94% usage")
	}
}

func TestContextBudgetNotYetFull(t *testing.T) {
	b := &ContextBudget{MaxTokens: 1000, CharsPerToken: 4.0}
	msgs := []ollama.Message{
		{Role: "user", Content: "hello"},
	}
	if b.ShouldCompress(msgs) {
		t.Error("ShouldCompress should be false for small messages")
	}
	if b.ShouldForceAnswer(msgs) {
		t.Error("ShouldForceAnswer should be false for small messages")
	}
}

// --- SmartTruncate tests ---

func TestSmartTruncateRead(t *testing.T) {
	// Create a file with 60 lines
	var lines []string
	for i := 0; i < 60; i++ {
		lines = append(lines, "line content here")
	}
	result := SmartTruncate("read", strings.Join(lines, "\n"))
	if !strings.Contains(result, "[...20 lines omitted...]") {
		t.Error("read truncation should show omitted line count")
	}
	// Should have first 20 + marker + last 20
	parts := strings.Split(result, "[...20 lines omitted...]")
	if len(parts) != 2 {
		t.Error("should split into head and tail around the marker")
	}
}

func TestSmartTruncateGrep(t *testing.T) {
	var lines []string
	for i := 0; i < 30; i++ {
		lines = append(lines, "match line")
	}
	result := SmartTruncate("grep", strings.Join(lines, "\n"))
	if !strings.Contains(result, "...and 15 more") {
		t.Error("grep truncation should cap at 15 results")
	}
}

func TestSmartTruncateNoOp(t *testing.T) {
	short := "just a few lines\nof output"
	if SmartTruncate("read", short) != short {
		t.Error("short content should not be truncated")
	}
}

func TestSmartTruncateHardLimit(t *testing.T) {
	huge := strings.Repeat("x", 3000)
	result := SmartTruncate("sysinfo", huge)
	if len(result) > 2100 {
		t.Errorf("hard limit should cap at ~2048, got %d", len(result))
	}
	if !strings.HasSuffix(result, "... (truncated)") {
		t.Error("truncated result should end with truncation marker")
	}
}

// --- ValidateToolResult tests ---

func TestValidateToolResultError(t *testing.T) {
	result, hint := ValidateToolResult("read", "", errors.New("no such file"))
	if !strings.Contains(result, "Error:") {
		t.Error("error result should contain Error prefix")
	}
	if !strings.Contains(hint, "ls or glob") {
		t.Errorf("file-not-found hint should suggest ls/glob, got: %s", hint)
	}
}

func TestValidateToolResultEmptyRead(t *testing.T) {
	_, hint := ValidateToolResult("read", "  \n  ", nil)
	if !strings.Contains(hint, "empty") {
		t.Error("empty read result should warn about empty file")
	}
}

func TestValidateToolResultEmptyGrep(t *testing.T) {
	result, _ := ValidateToolResult("grep", "", nil)
	if result != "No matches found." {
		t.Errorf("empty grep should return 'No matches found.', got: %s", result)
	}
}

func TestValidateToolResultNormal(t *testing.T) {
	result, hint := ValidateToolResult("read", "file content here", nil)
	if result != "file content here" {
		t.Error("normal result should pass through unchanged")
	}
	if hint != "" {
		t.Error("normal result should have no hint")
	}
}

// --- ReflectionGate tests ---

func TestReflectionGateReset(t *testing.T) {
	g := &ReflectionGate{}
	g.toolCallCount = 10
	g.consecutiveEmpty = 5
	g.Reset()
	if g.toolCallCount != 0 || g.consecutiveEmpty != 0 {
		t.Error("Reset should zero all counters")
	}
}

func TestReflectionGateNormalUsage(t *testing.T) {
	g := &ReflectionGate{}
	hint := g.Check("read", "some file content", nil)
	if hint != "" {
		t.Errorf("normal tool result should produce no hint, got: %s", hint)
	}
}

func TestReflectionGateConsecutiveEmpty(t *testing.T) {
	g := &ReflectionGate{}
	g.Check("grep", "", nil)
	hint := g.Check("grep", "", nil)
	if !strings.Contains(hint, "empty results") {
		t.Errorf("two empty results should trigger warning, got: %s", hint)
	}
}

func TestReflectionGateRepetition(t *testing.T) {
	g := &ReflectionGate{}
	g.Check("read", "content A", nil)
	g.Check("ls", "dir listing", nil)
	hint := g.Check("read", "content A", nil) // same result as first
	if !strings.Contains(hint, "repeating") {
		t.Errorf("repeated tool call should be detected, got: %s", hint)
	}
}

func TestReflectionGateTooManyIterations(t *testing.T) {
	g := &ReflectionGate{}
	// Make 5 unique calls
	for i := 0; i < 4; i++ {
		g.Check("read", strings.Repeat("x", i+1), nil)
	}
	hint := g.Check("grep", "unique result 5", nil)
	if !strings.Contains(hint, "many tool calls") {
		t.Errorf("5th tool call should trigger convergence hint, got: %s", hint)
	}
}

func TestReflectionGateEmptyResetOnContent(t *testing.T) {
	g := &ReflectionGate{}
	g.Check("grep", "", nil) // empty
	g.Check("read", "found it!", nil) // non-empty resets counter
	hint := g.Check("grep", "", nil) // empty again but counter was reset
	if strings.Contains(hint, "empty results") {
		t.Error("non-empty result should reset consecutive empty counter")
	}
}
