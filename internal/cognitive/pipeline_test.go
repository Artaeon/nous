package cognitive

import (
	"fmt"
	"strings"
	"testing"
)

func TestNewPipeline(t *testing.T) {
	p := NewPipeline("what is in go.mod?")
	if p == nil {
		t.Fatal("NewPipeline returned nil")
	}
	if p.userQuery != "what is in go.mod?" {
		t.Errorf("expected userQuery 'what is in go.mod?', got %q", p.userQuery)
	}
	if p.StepCount() != 0 {
		t.Errorf("expected 0 steps, got %d", p.StepCount())
	}
	if p.LastResult() != "" {
		t.Errorf("expected empty LastResult, got %q", p.LastResult())
	}
	if p.BuildContext() != "" {
		t.Errorf("expected empty BuildContext, got %q", p.BuildContext())
	}
}

func TestCompressStepRead(t *testing.T) {
	// Simulate a multiline file read
	var lines []string
	lines = append(lines, "module github.com/artaeon/nous")
	lines = append(lines, "")
	lines = append(lines, "go 1.22")
	for i := 0; i < 50; i++ {
		lines = append(lines, fmt.Sprintf("require something/v%d", i))
	}
	result := strings.Join(lines, "\n")

	summary := CompressStep("read", result)

	if !strings.Contains(summary, "Read") {
		t.Errorf("read summary should start with 'Read', got: %s", summary)
	}
	if !strings.Contains(summary, "module github.com/artaeon/nous") {
		t.Errorf("read summary should contain first meaningful line, got: %s", summary)
	}
	if !strings.Contains(summary, "lines") {
		t.Errorf("read summary should contain line count, got: %s", summary)
	}
}

func TestCompressStepGrep(t *testing.T) {
	result := `internal/cognitive/reasoner.go:82:func (r *Reasoner) reason(ctx context.Context, percept blackboard.Percept) error {
internal/cognitive/reasoner.go:230:func (r *Reasoner) callLLM() (string, error) {
internal/tools/builtin.go:15:func init() {
cmd/nous/main.go:20:func main() {`

	summary := CompressStep("grep", result)

	if !strings.Contains(summary, "4 matches") {
		t.Errorf("grep summary should contain match count, got: %s", summary)
	}
	if !strings.Contains(summary, "internal/cognitive/reasoner.go") {
		t.Errorf("grep summary should contain file names, got: %s", summary)
	}
}

func TestCompressStepGlob(t *testing.T) {
	result := `internal/cognitive/reasoner.go
internal/cognitive/conversation.go
internal/cognitive/grounding.go
internal/cognitive/pipeline.go
internal/cognitive/tool_selector.go`

	summary := CompressStep("glob", result)

	if !strings.Contains(summary, "5 files") {
		t.Errorf("glob summary should contain file count, got: %s", summary)
	}
	if !strings.Contains(summary, "matching pattern") {
		t.Errorf("glob summary should say 'matching pattern', got: %s", summary)
	}
}

func TestCompressStepLs(t *testing.T) {
	result := `blackboard/
cognitive/
compress/
memory/
ollama/
tools/`

	summary := CompressStep("ls", result)

	if !strings.Contains(summary, "6 entries") {
		t.Errorf("ls summary should contain entry count, got: %s", summary)
	}
	if !strings.Contains(summary, "blackboard/") {
		t.Errorf("ls summary should include first entries, got: %s", summary)
	}
}

func TestCompressStepGit(t *testing.T) {
	result := `b2cd043 Fix streaming token filter to suppress tool call JSON from output
8f08a72 Update README for v0.4.0: 18 tools, project memory, undo stack`

	summary := CompressStep("git", result)

	if !strings.HasPrefix(summary, "Git: ") {
		t.Errorf("git summary should start with 'Git: ', got: %s", summary)
	}
	if !strings.Contains(summary, "b2cd043") {
		t.Errorf("git summary should contain first line of output, got: %s", summary)
	}
}

func TestCompressStepError(t *testing.T) {
	result := "Error: no such file or directory: /nonexistent/path.go"

	summary := CompressStep("read", result)

	if !strings.HasPrefix(summary, "Error: ") {
		t.Errorf("error summary should start with 'Error: ', got: %s", summary)
	}
	if !strings.Contains(summary, "no such file") {
		t.Errorf("error summary should contain the error message, got: %s", summary)
	}
}

func TestCompressStepDefault(t *testing.T) {
	// Short result should pass through as-is
	result := "Operation completed successfully"
	summary := CompressStep("unknown_tool", result)
	if summary != result {
		t.Errorf("short default result should pass through, got: %s", summary)
	}

	// Long result should be truncated at 80 chars
	longResult := strings.Repeat("abcdefghij", 20) // 200 chars
	summary = CompressStep("unknown_tool", longResult)
	if len(summary) > 84 { // 80 + "..."
		t.Errorf("long default result should be truncated, got len %d: %s", len(summary), summary)
	}
	if !strings.HasSuffix(summary, "...") {
		t.Errorf("truncated result should end with '...', got: %s", summary)
	}
}

func TestCompressStepWrite(t *testing.T) {
	result := "Wrote 42 bytes to internal/cognitive/pipeline.go"
	summary := CompressStep("write", result)
	if !strings.HasPrefix(summary, "Modified") {
		t.Errorf("write summary should start with 'Modified', got: %s", summary)
	}
}

func TestCompressStepEdit(t *testing.T) {
	result := "Replaced content in internal/cognitive/reasoner.go"
	summary := CompressStep("edit", result)
	if !strings.HasPrefix(summary, "Modified") {
		t.Errorf("edit summary should start with 'Modified', got: %s", summary)
	}
}

func TestCompressStepSysinfo(t *testing.T) {
	result := "Linux x86_64 8 CPUs\n16GB RAM\nGo 1.22"
	summary := CompressStep("sysinfo", result)
	if !strings.HasPrefix(summary, "System: ") {
		t.Errorf("sysinfo summary should start with 'System: ', got: %s", summary)
	}
	if !strings.Contains(summary, "Linux") {
		t.Errorf("sysinfo summary should contain OS info, got: %s", summary)
	}
}

func TestBuildContext(t *testing.T) {
	p := NewPipeline("what does this project do?")

	p.AddStep("read", "module github.com/artaeon/nous\n\ngo 1.22\n")
	p.AddStep("ls", "blackboard/\ncognitive/\ncompress/\nmemory/\nollama/\ntools/\n")
	p.AddStep("grep", "cmd/nous/main.go:20:func main() {\n")

	ctx := p.BuildContext()

	if !strings.HasPrefix(ctx, "[Previous steps]") {
		t.Errorf("context should start with '[Previous steps]', got: %s", ctx)
	}
	if !strings.Contains(ctx, "1. ") {
		t.Errorf("context should contain numbered steps, got: %s", ctx)
	}
	if !strings.Contains(ctx, "2. ") {
		t.Errorf("context should contain step 2, got: %s", ctx)
	}
	if !strings.Contains(ctx, "3. ") {
		t.Errorf("context should contain step 3, got: %s", ctx)
	}
}

func TestBuildContextEmpty(t *testing.T) {
	p := NewPipeline("hello")
	ctx := p.BuildContext()
	if ctx != "" {
		t.Errorf("empty pipeline should return empty context, got: %q", ctx)
	}
}

func TestPipelineMultiStep(t *testing.T) {
	p := NewPipeline("explain the architecture of this project")

	// Step 1: Read go.mod
	p.AddStep("read", "module github.com/artaeon/nous\n\ngo 1.22\n\nrequire (\n\tnothing\n)")

	// Step 2: List directories
	p.AddStep("ls", "blackboard/\ncognitive/\ncompress/\nmemory/\nollama/\ntools/")

	// Step 3: Grep for main
	p.AddStep("grep", "cmd/nous/main.go:20:func main() {")

	// Step 4: Read a specific file
	var longFile []string
	for i := 0; i < 100; i++ {
		longFile = append(longFile, fmt.Sprintf("line %d: some code here", i))
	}
	p.AddStep("read", strings.Join(longFile, "\n"))

	// Step 5: Glob for test files
	p.AddStep("glob", "internal/cognitive/reasoner_test.go\ninternal/cognitive/conversation_test.go\ninternal/cognitive/grounding_test.go\ninternal/cognitive/pipeline_test.go\ninternal/tools/builtin_test.go")

	// Verify step count
	if p.StepCount() != 5 {
		t.Fatalf("expected 5 steps, got %d", p.StepCount())
	}

	// Verify context stays compact
	ctx := p.BuildContext()
	lines := strings.Split(ctx, "\n")

	// Should be header + 5 step lines = 6 lines total
	if len(lines) != 6 {
		t.Errorf("expected 6 lines in context (header + 5 steps), got %d:\n%s", len(lines), ctx)
	}

	// Total context should be small (under 500 chars for 5 steps)
	if len(ctx) > 500 {
		t.Errorf("context should be compact (<500 chars), got %d chars:\n%s", len(ctx), ctx)
	}

	// Only the last step should have a RawResult
	for i, step := range p.steps {
		if i < len(p.steps)-1 {
			if step.RawResult != "" {
				t.Errorf("step %d should have empty RawResult (cleared), got %d chars", i+1, len(step.RawResult))
			}
		} else {
			if step.RawResult == "" {
				t.Error("last step should have non-empty RawResult")
			}
		}
	}

	// LastResult should return the last step's raw result
	lastResult := p.LastResult()
	if !strings.HasPrefix(lastResult, "internal/cognitive/reasoner_test.go") {
		t.Errorf("LastResult should be the glob output, got: %s", lastResult[:50])
	}

	// Verify each step has a numbered summary
	for i := 1; i <= 5; i++ {
		prefix := fmt.Sprintf("%d. ", i)
		if !strings.Contains(ctx, prefix) {
			t.Errorf("context missing step %d prefix", i)
		}
	}
}

func TestPipelineLastResultClearedOnNewStep(t *testing.T) {
	p := NewPipeline("test")
	p.AddStep("read", "first result content")

	if p.LastResult() != "first result content" {
		t.Errorf("expected first result, got: %s", p.LastResult())
	}

	p.AddStep("ls", "second result content")

	if p.LastResult() != "second result content" {
		t.Errorf("expected second result, got: %s", p.LastResult())
	}

	// First step's raw result should be cleared
	if p.steps[0].RawResult != "" {
		t.Error("first step RawResult should be cleared after adding second step")
	}
}
