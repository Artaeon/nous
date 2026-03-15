package cognitive

import (
	"errors"
	"fmt"
	"strings"
	"testing"

	"github.com/artaeon/nous/internal/ollama"
)

// --- ContextBudget tests ---

func TestContextBudgetEstimateTokens(t *testing.T) {
	b := DefaultBudget()
	if b.MaxTokens != 8192 {
		t.Errorf("DefaultBudget().MaxTokens = %d, want 8192", b.MaxTokens)
	}
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
		lines = append(lines, fmt.Sprintf("line %d content here", i+1))
	}
	result := SmartTruncate("read", strings.Join(lines, "\n"))
	if !strings.Contains(result, "lines omitted") {
		t.Error("read truncation should show omitted line count")
	}
	if !strings.Contains(result, "Landmarks") {
		t.Error("read truncation should include landmark lines from middle")
	}
	if !strings.Contains(result, "[line") {
		t.Error("read truncation should include line number markers")
	}
	if !strings.Contains(result, "read with offset/limit") {
		t.Error("read truncation should hint about offset/limit")
	}
}

func TestSmartTruncateGrep(t *testing.T) {
	var lines []string
	for i := 0; i < 30; i++ {
		lines = append(lines, "match line")
	}
	result := SmartTruncate("grep", strings.Join(lines, "\n"))
	if !strings.Contains(result, "...and 10 more") {
		t.Error("grep truncation should cap at 20 results")
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
	cr := g.Check("read", "some file content", nil)
	if cr.Hint != "" {
		t.Errorf("normal tool result should produce no hint, got: %s", cr.Hint)
	}
	if cr.ForceStop {
		t.Error("normal result should not force stop")
	}
}

func TestReflectionGateConsecutiveEmpty(t *testing.T) {
	g := &ReflectionGate{}
	g.Check("grep", "", nil)
	cr := g.Check("grep", "", nil)
	if !strings.Contains(cr.Hint, "empty") {
		t.Errorf("two empty results should trigger warning, got: %s", cr.Hint)
	}
}

func TestReflectionGateRepetition(t *testing.T) {
	g := &ReflectionGate{}
	g.Check("read", "content A", nil)
	cr := g.Check("read", "content A", nil) // same result = repetition
	if !strings.Contains(cr.Hint, "already have") {
		t.Errorf("repeated tool call should be detected, got: %s", cr.Hint)
	}
}

func TestReflectionGateForceStopOnTripleRepeat(t *testing.T) {
	g := &ReflectionGate{}
	g.Check("ls", "dir listing", nil)
	g.Check("ls", "dir listing", nil)
	cr := g.Check("ls", "dir listing", nil) // 3rd repeat
	if !cr.ForceStop {
		t.Error("3 repeated calls should force stop")
	}
}

func TestReflectionGateForceStopOnManyIterations(t *testing.T) {
	g := &ReflectionGate{}
	g.Check("read", "a", nil)
	g.Check("ls", "b", nil)
	g.Check("grep", "c", nil)
	g.Check("tree", "d", nil)
	g.Check("glob", "e", nil)
	cr := g.Check("diff", "f", nil) // 6th unique call
	if !cr.ForceStop {
		t.Error("6th tool call should force stop")
	}
}

func TestReflectionGateEmptyResetOnContent(t *testing.T) {
	g := &ReflectionGate{}
	g.Check("grep", "", nil) // empty
	g.Check("read", "found it!", nil) // non-empty resets counter
	cr := g.Check("grep", "", nil) // empty again but counter was reset
	if strings.Contains(cr.Hint, "empty") {
		t.Error("non-empty result should reset consecutive empty counter")
	}
}

// --- Extended edge case tests ---

func TestContextBudgetZeroCharsPerToken(t *testing.T) {
	b := &ContextBudget{MaxTokens: 1000, CharsPerToken: 0}
	// Should fall back to default (len/4)
	got := b.EstimateTokens("1234567890") // 10 chars / 4 = 2
	if got != 2 {
		t.Errorf("zero CharsPerToken should fallback to 4, got %d", got)
	}
}

func TestContextBudgetZeroMaxTokens(t *testing.T) {
	b := &ContextBudget{MaxTokens: 0, CharsPerToken: 4.0}
	// UsagePercent with 0 MaxTokens should return 1.0 (full)
	msgs := []ollama.Message{{Role: "user", Content: "test"}}
	if pct := b.UsagePercent(msgs); pct != 1.0 {
		t.Errorf("UsagePercent with 0 MaxTokens = %f, want 1.0", pct)
	}
}

func TestContextBudgetRemainingOverflow(t *testing.T) {
	b := &ContextBudget{MaxTokens: 10, CharsPerToken: 1.0}
	msgs := []ollama.Message{{Role: "user", Content: strings.Repeat("x", 100)}}
	// Usage exceeds max → remaining should be 0
	remaining := b.Remaining(msgs)
	if remaining != 0 {
		t.Errorf("Remaining with overflow = %d, want 0", remaining)
	}
}

func TestSmartTruncateTree(t *testing.T) {
	var lines []string
	for i := 0; i < 40; i++ {
		lines = append(lines, fmt.Sprintf("entry_%d", i))
	}
	result := SmartTruncate("tree", strings.Join(lines, "\n"))
	if !strings.Contains(result, "10 more entries") {
		t.Errorf("tree with 40 lines should be capped at 30, got: %s", result[:100])
	}
}

func TestSmartTruncateGlob(t *testing.T) {
	var lines []string
	for i := 0; i < 25; i++ {
		lines = append(lines, fmt.Sprintf("file_%d.go", i))
	}
	result := SmartTruncate("glob", strings.Join(lines, "\n"))
	if !strings.Contains(result, "5 more") {
		t.Errorf("glob with 25 matches should cap at 20, got: %s", result[:100])
	}
}

func TestSmartTruncateReadWithLandmarks(t *testing.T) {
	var lines []string
	for i := 0; i < 100; i++ {
		if i == 50 {
			lines = append(lines, "func ImportantFunction() error {")
		} else {
			lines = append(lines, fmt.Sprintf("// line %d", i))
		}
	}
	result := SmartTruncate("read", strings.Join(lines, "\n"))

	// Should contain landmark from middle
	if !strings.Contains(result, "[line") {
		t.Error("read truncation should include landmark line numbers")
	}
	// Should preserve head
	if !strings.Contains(result, "// line 0") {
		t.Error("read truncation should preserve head")
	}
}

func TestValidateToolResultPermissionDenied(t *testing.T) {
	_, hint := ValidateToolResult("write", "", errors.New("permission denied"))
	if !strings.Contains(hint, "Permission") {
		t.Errorf("permission error should hint about permissions, got: %s", hint)
	}
}

func TestValidateToolResultIsDirectory(t *testing.T) {
	_, hint := ValidateToolResult("read", "", errors.New("is a directory"))
	if !strings.Contains(hint, "directory") {
		t.Errorf("directory error should hint about ls, got: %s", hint)
	}
}

func TestValidateToolResultGenericError(t *testing.T) {
	_, hint := ValidateToolResult("shell", "", errors.New("something went wrong"))
	if !strings.Contains(hint, "different approach") {
		t.Errorf("generic error should suggest different approach, got: %s", hint)
	}
}

func TestValidateToolResultEmptyGlob(t *testing.T) {
	result, _ := ValidateToolResult("glob", "", nil)
	if result != "No files matched the pattern." {
		t.Errorf("empty glob = %q, want 'No files matched the pattern.'", result)
	}
}

func TestReflectionGateErrorSkipped(t *testing.T) {
	g := &ReflectionGate{}
	cr := g.Check("read", "", errors.New("some error"))
	// Errors are handled by ValidateToolResult, gate returns empty
	if cr.Hint != "" || cr.ForceStop {
		t.Error("gate should not add hints for errors (handled elsewhere)")
	}
}

func TestReflectionGateCircularBuffer(t *testing.T) {
	g := &ReflectionGate{}
	// Fill the circular buffer with 4 unique calls
	g.Check("read", "a", nil)
	g.Check("ls", "b", nil)
	g.Check("grep", "c", nil)
	g.Check("glob", "d", nil)
	// 5th call should overwrite the oldest in the buffer
	cr := g.Check("tree", "e", nil)
	if cr.ForceStop {
		t.Error("5 unique calls should not force stop (buffer wraps)")
	}
}

func TestShortHash(t *testing.T) {
	h1 := shortHash("hello")
	h2 := shortHash("hello")
	h3 := shortHash("world")

	if h1 != h2 {
		t.Error("same input should produce same hash")
	}
	if h1 == h3 {
		t.Error("different inputs should produce different hashes")
	}
	if len(h1) != 8 { // 4 bytes = 8 hex chars
		t.Errorf("hash length = %d, want 8", len(h1))
	}
}
