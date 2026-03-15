package cognitive

import (
	"errors"
	"strings"
	"testing"
)

// --- Response Synthesizer Tests ---

func TestSynthesizerCreation(t *testing.T) {
	rs := NewResponseSynthesizer()
	if rs == nil {
		t.Fatal("NewResponseSynthesizer should not return nil")
	}
}

func TestCanSynthesize(t *testing.T) {
	rs := NewResponseSynthesizer()

	supported := []string{"grep", "read", "ls", "tree", "glob", "git", "write", "edit"}
	for _, tool := range supported {
		if !rs.CanSynthesize(tool) {
			t.Errorf("should synthesize %s", tool)
		}
	}

	unsupported := []string{"unknown", "shell", "custom"}
	for _, tool := range unsupported {
		if rs.CanSynthesize(tool) {
			t.Errorf("should not synthesize %s", tool)
		}
	}
}

// --- Grep Synthesis ---

func TestSynthesizeGrepResults(t *testing.T) {
	rs := NewResponseSynthesizer()
	result := "internal/cognitive/reasoner.go:66:func NewReasoner(...)\ninternal/cognitive/reasoner.go:78:func (r *Reasoner) Run..."
	args := map[string]string{"pattern": "NewReasoner"}

	response := rs.Synthesize("grep", args, result, nil)
	if !strings.Contains(response, "2 matches") {
		t.Errorf("should mention match count, got: %s", response)
	}
	if !strings.Contains(response, "NewReasoner") {
		t.Error("should mention the search pattern")
	}
	if !strings.Contains(response, "reasoner.go") {
		t.Error("should mention the file")
	}
}

func TestSynthesizeGrepNoResults(t *testing.T) {
	rs := NewResponseSynthesizer()
	args := map[string]string{"pattern": "NonexistentThing"}

	response := rs.Synthesize("grep", args, "", nil)
	if !strings.Contains(response, "No matches") {
		t.Errorf("should say no matches, got: %s", response)
	}
	if !strings.Contains(response, "NonexistentThing") {
		t.Error("should mention the pattern")
	}
}

func TestSynthesizeGrepWithGlob(t *testing.T) {
	rs := NewResponseSynthesizer()
	result := "main.go:10:func main()"
	args := map[string]string{"pattern": "main", "glob": "*.go"}

	response := rs.Synthesize("grep", args, result, nil)
	if !strings.Contains(response, "*.go") {
		t.Errorf("should mention file filter, got: %s", response)
	}
}

func TestSynthesizeGrepSingleMatch(t *testing.T) {
	rs := NewResponseSynthesizer()
	result := "internal/cognitive/crystal.go:15:type CrystalBook struct {"
	args := map[string]string{"pattern": "CrystalBook"}

	response := rs.Synthesize("grep", args, result, nil)
	if !strings.Contains(response, "1 match") {
		t.Errorf("should say 1 match, got: %s", response)
	}
}

func TestSynthesizeGrepGroupsByFile(t *testing.T) {
	rs := NewResponseSynthesizer()
	result := "a.go:1:line1\na.go:5:line5\nb.go:3:line3"
	args := map[string]string{"pattern": "test"}

	response := rs.Synthesize("grep", args, result, nil)
	if !strings.Contains(response, "a.go") {
		t.Error("should group by file a.go")
	}
	if !strings.Contains(response, "b.go") {
		t.Error("should group by file b.go")
	}
}

// --- Read Synthesis ---

func TestSynthesizeReadShortFile(t *testing.T) {
	rs := NewResponseSynthesizer()
	content := "package main\n\nfunc main() {\n\tfmt.Println(\"hello\")\n}"
	args := map[string]string{"path": "main.go"}

	response := rs.Synthesize("read", args, content, nil)
	if !strings.Contains(response, "main.go") {
		t.Error("should mention file name")
	}
	if !strings.Contains(response, "```") {
		t.Error("should use code block for short files")
	}
}

func TestSynthesizeReadLongFile(t *testing.T) {
	rs := NewResponseSynthesizer()
	var lines []string
	for i := 0; i < 100; i++ {
		lines = append(lines, "line content")
	}
	content := strings.Join(lines, "\n")
	args := map[string]string{"path": "long_file.go"}

	response := rs.Synthesize("read", args, content, nil)
	if !strings.Contains(response, "100 lines") {
		t.Error("should mention line count")
	}
	if !strings.Contains(response, "more lines") {
		t.Error("should indicate truncation for long files")
	}
}

// --- Ls Synthesis ---

func TestSynthesizeLs(t *testing.T) {
	rs := NewResponseSynthesizer()
	result := "cmd/\ninternal/\ngo.mod\ngo.sum\nREADME.md"
	args := map[string]string{"path": "."}

	response := rs.Synthesize("ls", args, result, nil)
	if !strings.Contains(response, "5 entries") {
		t.Errorf("should count entries, got: %s", response)
	}
}

func TestSynthesizeLsEmpty(t *testing.T) {
	rs := NewResponseSynthesizer()
	args := map[string]string{"path": "empty_dir"}

	response := rs.Synthesize("ls", args, "", nil)
	if !strings.Contains(response, "empty") {
		t.Errorf("should say empty, got: %s", response)
	}
}

// --- Tree Synthesis ---

func TestSynthesizeTree(t *testing.T) {
	rs := NewResponseSynthesizer()
	result := ".\n├── cmd\n│   └── nous\n├── internal\n│   ├── cognitive\n│   └── tools"
	args := map[string]string{"path": "."}

	response := rs.Synthesize("tree", args, result, nil)
	if !strings.Contains(response, "```") {
		t.Error("should use code block for tree output")
	}
	if !strings.Contains(response, "entries") {
		t.Error("should mention entry count")
	}
}

// --- Glob Synthesis ---

func TestSynthesizeGlob(t *testing.T) {
	rs := NewResponseSynthesizer()
	result := "internal/cognitive/reasoner.go\ninternal/cognitive/grammar.go\ninternal/cognitive/intent.go"
	args := map[string]string{"pattern": "**/*.go"}

	response := rs.Synthesize("glob", args, result, nil)
	if !strings.Contains(response, "3 files") {
		t.Errorf("should count files, got: %s", response)
	}
}

func TestSynthesizeGlobEmpty(t *testing.T) {
	rs := NewResponseSynthesizer()
	args := map[string]string{"pattern": "**/*.xyz"}

	response := rs.Synthesize("glob", args, "", nil)
	if !strings.Contains(response, "No files") {
		t.Errorf("should say no files, got: %s", response)
	}
}

// --- Git Synthesis ---

func TestSynthesizeGitStatus(t *testing.T) {
	rs := NewResponseSynthesizer()
	result := "On branch main\nnothing to commit, working tree clean"
	args := map[string]string{"command": "status"}

	response := rs.Synthesize("git", args, result, nil)
	if !strings.Contains(response, "clean") {
		t.Errorf("should indicate clean tree, got: %s", response)
	}
}

func TestSynthesizeGitLog(t *testing.T) {
	rs := NewResponseSynthesizer()
	result := "abc1234 First commit\ndef5678 Second commit\nghi9012 Third commit"
	args := map[string]string{"command": "log --oneline -10"}

	response := rs.Synthesize("git", args, result, nil)
	if !strings.Contains(response, "3") {
		t.Errorf("should count commits, got: %s", response)
	}
	if !strings.Contains(response, "First commit") {
		t.Error("should include commit messages")
	}
}

func TestSynthesizeGitDiff(t *testing.T) {
	rs := NewResponseSynthesizer()
	result := "diff --git a/file.go b/file.go\n+new line"
	args := map[string]string{"command": "diff"}

	response := rs.Synthesize("git", args, result, nil)
	if !strings.Contains(response, "```diff") {
		t.Error("should use diff code block")
	}
}

// --- Write/Edit Synthesis ---

func TestSynthesizeWrite(t *testing.T) {
	rs := NewResponseSynthesizer()
	args := map[string]string{"path": "output.txt"}

	response := rs.Synthesize("write", args, "ok", nil)
	if !strings.Contains(response, "output.txt") {
		t.Error("should mention file path")
	}
	if !strings.Contains(response, "written") {
		t.Error("should confirm write")
	}
}

func TestSynthesizeEdit(t *testing.T) {
	rs := NewResponseSynthesizer()
	args := map[string]string{"path": "main.go"}

	response := rs.Synthesize("edit", args, "ok", nil)
	if !strings.Contains(response, "main.go") {
		t.Error("should mention file path")
	}
	if !strings.Contains(response, "edited") {
		t.Error("should confirm edit")
	}
}

// --- Error Handling ---

func TestSynthesizeError(t *testing.T) {
	rs := NewResponseSynthesizer()

	response := rs.Synthesize("read", map[string]string{"path": "missing.go"}, "", errors.New("file not found"))
	if !strings.Contains(response, "missing.go") {
		t.Error("should mention file")
	}
	if !strings.Contains(response, "file not found") {
		t.Error("should include error message")
	}
}

// --- Multi-Step Synthesis ---

func TestSynthesizeMultiSingleStep(t *testing.T) {
	rs := NewResponseSynthesizer()
	steps := []synthStep{
		{Tool: "grep", Args: map[string]string{"pattern": "main"}, Result: "main.go:1:package main"},
	}

	response := rs.SynthesizeMulti("find main", steps)
	if !strings.Contains(response, "1 match") {
		t.Errorf("single step should use direct synthesis, got: %s", response)
	}
}

func TestSynthesizeMultiMultipleSteps(t *testing.T) {
	rs := NewResponseSynthesizer()
	steps := []synthStep{
		{Tool: "grep", Args: map[string]string{"pattern": "Pipeline"}, Result: "pipeline.go:5:type Pipeline struct {"},
		{Tool: "read", Args: map[string]string{"path": "pipeline.go"}, Result: "package cognitive\n\ntype Pipeline struct{}"},
	}

	response := rs.SynthesizeMulti("find and read Pipeline", steps)
	if !strings.Contains(response, "Pipeline") {
		t.Error("should mention Pipeline")
	}
	if !strings.Contains(response, "pipeline.go") {
		t.Error("should mention the file")
	}
}

func TestSynthesizeMultiSkipsErrors(t *testing.T) {
	rs := NewResponseSynthesizer()
	steps := []synthStep{
		{Tool: "grep", Args: map[string]string{"pattern": "x"}, Result: "a.go:1:x", Err: nil},
		{Tool: "read", Args: map[string]string{"path": "b.go"}, Err: errors.New("not found")},
	}

	response := rs.SynthesizeMulti("test", steps)
	if strings.Contains(response, "not found") {
		t.Error("should skip error steps in multi-step")
	}
}

// --- Group By File ---

func TestGroupByFile(t *testing.T) {
	lines := []string{
		"a.go:10:func main()",
		"a.go:20:func helper()",
		"b.go:5:type Config struct",
	}

	groups := groupByFile(lines)
	if len(groups) != 2 {
		t.Errorf("should have 2 file groups, got %d", len(groups))
	}
	if groups[0].file != "a.go" {
		t.Errorf("first group file = %q, want a.go", groups[0].file)
	}
	if len(groups[0].matches) != 2 {
		t.Errorf("a.go should have 2 matches, got %d", len(groups[0].matches))
	}
}

func TestGroupByFileSingleFile(t *testing.T) {
	lines := []string{"file.go:1:line1", "file.go:2:line2"}
	groups := groupByFile(lines)
	if len(groups) != 1 {
		t.Errorf("should have 1 group, got %d", len(groups))
	}
}

// --- Benchmark ---

func BenchmarkSynthesizeGrep(b *testing.B) {
	rs := NewResponseSynthesizer()
	result := "a.go:1:line1\na.go:5:line5\nb.go:3:line3\nc.go:10:line10"
	args := map[string]string{"pattern": "test", "glob": "*.go"}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		rs.Synthesize("grep", args, result, nil)
	}
}
