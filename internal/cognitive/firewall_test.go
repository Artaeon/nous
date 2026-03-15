package cognitive

import (
	"regexp"
	"strings"
	"testing"
)

// --- Cognitive Firewall Tests ---

func TestFirewallCreation(t *testing.T) {
	fw := NewCognitiveFirewall(nil)
	if fw == nil {
		t.Fatal("NewCognitiveFirewall should not return nil")
	}
}

// --- Language Impossibility Tests ---

func TestFirewallGoTryCatch(t *testing.T) {
	fw := NewCognitiveFirewall(nil)
	ctx := &FirewallContext{
		Query:    "how to handle errors in Go",
		Response: "Use try-catch blocks to handle errors in Go.",
		Language: "Go",
	}

	corrected, violations := fw.Validate(ctx)
	if len(violations) == 0 {
		t.Error("should detect try-catch as impossible in Go")
	}
	if strings.Contains(corrected, "try-catch") {
		t.Error("should replace try-catch")
	}
}

func TestFirewallGoTryCatchCode(t *testing.T) {
	fw := NewCognitiveFirewall(nil)
	ctx := &FirewallContext{
		Query:    "handle errors",
		Response: "```go\ntry {\n  result := doSomething()\n} catch (err) {\n  log.Fatal(err)\n}\n```",
		Language: "Go",
	}

	_, violations := fw.Validate(ctx)
	found := false
	for _, v := range violations {
		if v.Type == "language_impossible" {
			found = true
		}
	}
	if !found {
		t.Error("should detect try-catch in code block")
	}
}

func TestFirewallGoWhileLoop(t *testing.T) {
	fw := NewCognitiveFirewall(nil)
	ctx := &FirewallContext{
		Query:    "loop in Go",
		Response: "Use while (condition) { ... } for looping.",
		Language: "Go",
	}

	corrected, violations := fw.Validate(ctx)
	if len(violations) == 0 {
		t.Error("should detect while loop as impossible in Go")
	}
	if strings.Contains(corrected, "while (") {
		t.Error("should replace while loop")
	}
}

func TestFirewallGoNull(t *testing.T) {
	fw := NewCognitiveFirewall(nil)
	ctx := &FirewallContext{
		Query:    "check for null",
		Response: "Check if the value is null before using it.",
		Language: "Go",
	}

	corrected, _ := fw.Validate(ctx)
	if strings.Contains(corrected, "null") {
		t.Error("should replace null with nil in Go context")
	}
}

func TestFirewallGoThis(t *testing.T) {
	fw := NewCognitiveFirewall(nil)
	ctx := &FirewallContext{
		Query:    "access fields",
		Response: "Access the field using this.name in the method.",
		Language: "Go",
	}

	_, violations := fw.Validate(ctx)
	found := false
	for _, v := range violations {
		if strings.Contains(v.Description, "this") {
			found = true
		}
	}
	if !found {
		t.Error("should detect this keyword as impossible in Go")
	}
}

func TestFirewallGoAsyncAwait(t *testing.T) {
	fw := NewCognitiveFirewall(nil)
	ctx := &FirewallContext{
		Query:    "concurrent code",
		Response: "Use async func doWork() and await result to handle concurrency.",
		Language: "Go",
	}

	_, violations := fw.Validate(ctx)
	if len(violations) == 0 {
		t.Error("should detect async/await as impossible in Go")
	}
}

func TestFirewallNoViolationsForCorrectGo(t *testing.T) {
	fw := NewCognitiveFirewall(nil)
	ctx := &FirewallContext{
		Query:    "handle errors in Go",
		Response: "Use if err != nil { return err } for error handling in Go. Go uses goroutines and channels for concurrency.",
		Language: "Go",
	}

	_, violations := fw.Validate(ctx)
	if len(violations) != 0 {
		t.Errorf("correct Go code should have no violations, got %d: %v", len(violations), violations)
	}
}

func TestFirewallNoLanguageNoViolation(t *testing.T) {
	fw := NewCognitiveFirewall(nil)
	ctx := &FirewallContext{
		Query:    "how to handle errors",
		Response: "Use try-catch blocks for error handling.",
		Language: "", // no language detected
	}

	_, violations := fw.Validate(ctx)
	if len(violations) != 0 {
		t.Error("without language context, should not flag language-specific violations")
	}
}

// --- Numerical Consistency Tests ---

func TestFirewallNumericalMismatchGlob(t *testing.T) {
	fw := NewCognitiveFirewall(nil)
	ctx := &FirewallContext{
		Query:    "how many Go files",
		Response: "There are 7287 Go files in the directory.",
		ToolResults: []FirewallToolResult{
			{
				Tool: "glob",
				Args: map[string]string{"pattern": "*.go"},
				Result: "a.go\nb.go\nc.go\nd.go\ne.go\nf.go\ng.go\nh.go\ni.go\nj.go\n" +
					"k.go\nl.go\nm.go\nn.go\no.go\np.go\nq.go\nr.go\ns.go\nt.go\n" +
					"u.go\nv.go\nw.go\nx.go\ny.go\nz.go\naa.go\nbb.go\ncc.go\ndd.go\nee.go",
			},
		},
	}

	corrected, violations := fw.Validate(ctx)
	if len(violations) == 0 {
		t.Error("should detect 7287 vs 31 mismatch")
	}
	if strings.Contains(corrected, "7287") {
		t.Error("should replace wrong count")
	}
	if !strings.Contains(corrected, "31") {
		t.Errorf("should insert correct count 31, got: %s", corrected)
	}
}

func TestFirewallNumericalCorrectCount(t *testing.T) {
	fw := NewCognitiveFirewall(nil)
	ctx := &FirewallContext{
		Query:    "how many matches",
		Response: "Found 3 matches for the pattern.",
		ToolResults: []FirewallToolResult{
			{
				Tool:   "grep",
				Args:   map[string]string{"pattern": "test"},
				Result: "a.go:1:test\nb.go:2:test\nc.go:3:test",
			},
		},
	}

	_, violations := fw.Validate(ctx)
	for _, v := range violations {
		if v.Type == "numerical_mismatch" {
			t.Error("correct count should not trigger numerical mismatch")
		}
	}
}

// --- Hallucination Marker Tests ---

func TestFirewallHallucinationNoResults(t *testing.T) {
	fw := NewCognitiveFirewall(nil)
	ctx := &FirewallContext{
		Query:    "find Pipeline",
		Response: "I couldn't find any matches for Pipeline in the codebase.",
		ToolResults: []FirewallToolResult{
			{
				Tool:   "grep",
				Args:   map[string]string{"pattern": "Pipeline"},
				Result: "pipeline.go:5:type Pipeline struct {\npipeline.go:10:func NewPipeline(",
			},
		},
	}

	corrected, violations := fw.Validate(ctx)
	found := false
	for _, v := range violations {
		if v.Type == "hallucination" {
			found = true
		}
	}
	if !found {
		t.Error("should detect 'couldn't find' as hallucination when results exist")
	}
	if strings.Contains(corrected, "couldn't find") {
		t.Error("should replace hallucinated response")
	}
}

func TestFirewallHallucinationCorrectlyEmpty(t *testing.T) {
	fw := NewCognitiveFirewall(nil)
	ctx := &FirewallContext{
		Query:    "find XyzNonexistent",
		Response: "No results found for XyzNonexistent.",
		ToolResults: []FirewallToolResult{
			{
				Tool:   "grep",
				Args:   map[string]string{"pattern": "XyzNonexistent"},
				Result: "",
			},
		},
	}

	_, violations := fw.Validate(ctx)
	for _, v := range violations {
		if v.Type == "hallucination" {
			t.Error("should not flag 'no results' when results are genuinely empty")
		}
	}
}

// --- Tool Contradiction Tests ---

func TestFirewallWriteContradiction(t *testing.T) {
	fw := NewCognitiveFirewall(nil)
	ctx := &FirewallContext{
		Query:    "create test.txt",
		Response: "I'm sorry, but I can't create the file because it doesn't exist.",
		ToolResults: []FirewallToolResult{
			{
				Tool:   "write",
				Args:   map[string]string{"path": "/tmp/test.txt"},
				Result: "ok",
			},
		},
	}

	corrected, violations := fw.Validate(ctx)
	found := false
	for _, v := range violations {
		if v.Type == "tool_contradiction" {
			found = true
		}
	}
	if !found {
		t.Error("should detect 'can't create' as contradiction for write tool")
	}
	if strings.Contains(corrected, "sorry") {
		t.Error("should replace apology with confirmation")
	}
	if !strings.Contains(corrected, "written successfully") {
		t.Errorf("should confirm write, got: %s", corrected)
	}
}

func TestFirewallEditContradiction(t *testing.T) {
	fw := NewCognitiveFirewall(nil)
	ctx := &FirewallContext{
		Query:    "edit main.go",
		Response: "No changes need to be made, the comment is already present.",
		ToolResults: []FirewallToolResult{
			{
				Tool:   "edit",
				Args:   map[string]string{"path": "main.go"},
				Result: "ok",
			},
		},
	}

	corrected, violations := fw.Validate(ctx)
	found := false
	for _, v := range violations {
		if v.Type == "tool_contradiction" {
			found = true
		}
	}
	if !found {
		t.Error("should detect 'no changes needed' as contradiction when edit was executed")
	}
	if !strings.Contains(corrected, "edited successfully") {
		t.Errorf("should confirm edit, got: %s", corrected)
	}
}

// --- Multiple Violations ---

func TestFirewallMultipleViolations(t *testing.T) {
	fw := NewCognitiveFirewall(nil)
	ctx := &FirewallContext{
		Query:    "explain error handling",
		Response: "Use try-catch blocks. There are 9999 files. The value is null.",
		Language: "Go",
		ToolResults: []FirewallToolResult{
			{
				Tool:   "glob",
				Args:   map[string]string{"pattern": "*.go"},
				Result: "a.go\nb.go\nc.go",
			},
		},
	}

	_, violations := fw.Validate(ctx)
	if len(violations) < 2 {
		t.Errorf("should find multiple violations, got %d", len(violations))
	}
}

// --- Edge Cases ---

func TestFirewallEmptyResponse(t *testing.T) {
	fw := NewCognitiveFirewall(nil)
	ctx := &FirewallContext{
		Query:    "test",
		Response: "",
	}

	corrected, violations := fw.Validate(ctx)
	if corrected != "" {
		t.Error("empty response should stay empty")
	}
	if len(violations) != 0 {
		t.Error("empty response should have no violations")
	}
}

func TestFirewallNoToolResults(t *testing.T) {
	fw := NewCognitiveFirewall(nil)
	ctx := &FirewallContext{
		Query:    "hello",
		Response: "Hello! How can I help you?",
	}

	corrected, violations := fw.Validate(ctx)
	if corrected != "Hello! How can I help you?" {
		t.Error("response without tool results should be unchanged")
	}
	if len(violations) != 0 {
		t.Error("friendly greeting should have no violations")
	}
}

func TestFirewallCustomLanguageRules(t *testing.T) {
	fw := NewCognitiveFirewall(nil)
	fw.RegisterLanguageRules("rust", []LanguageRule{
		{
			Pattern:    compileRe(`\bgc\b`),
			Impossible: "garbage collection",
			Correction: "ownership system",
		},
	})

	ctx := &FirewallContext{
		Query:    "memory management in Rust",
		Response: "Rust uses gc for memory management.",
		Language: "Rust",
	}

	_, violations := fw.Validate(ctx)
	if len(violations) == 0 {
		t.Error("should detect gc as impossible in Rust")
	}
}

func compileRe(pattern string) *regexp.Regexp {
	re := regexp.MustCompile(pattern)
	return re
}

// --- Benchmark ---

func BenchmarkFirewallValidate(b *testing.B) {
	fw := NewCognitiveFirewall(nil)
	ctx := &FirewallContext{
		Query:    "explain error handling in Go",
		Response: "Use if err != nil for error handling. Found 5 matches across 3 files.",
		Language: "Go",
		ToolResults: []FirewallToolResult{
			{Tool: "grep", Args: map[string]string{"pattern": "err"}, Result: "a.go:1:err\nb.go:2:err\nc.go:3:err\nd.go:4:err\ne.go:5:err"},
		},
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		fw.Validate(ctx)
	}
}
