package cognitive

import (
	"strings"
	"testing"

	"github.com/artaeon/nous/internal/tools"
)

// --- Speculative Executor Tests ---

func mockToolRegistry() *tools.Registry {
	reg := tools.NewRegistry()
	reg.Register(tools.Tool{
		Name:        "read",
		Description: "Read file",
		Execute: func(args map[string]string) (string, error) {
			return "file content: " + args["path"], nil
		},
	})
	reg.Register(tools.Tool{
		Name:        "grep",
		Description: "Search files",
		Execute: func(args map[string]string) (string, error) {
			return "match: " + args["pattern"] + " found in test.go", nil
		},
	})
	reg.Register(tools.Tool{
		Name:        "ls",
		Description: "List directory",
		Execute: func(args map[string]string) (string, error) {
			return "main.go\ngo.mod\nREADME.md", nil
		},
	})
	reg.Register(tools.Tool{
		Name:        "glob",
		Description: "Glob files",
		Execute: func(args map[string]string) (string, error) {
			return "cmd/main.go\ninternal/core.go", nil
		},
	})
	reg.Register(tools.Tool{
		Name:        "git",
		Description: "Git commands",
		Execute: func(args map[string]string) (string, error) {
			return "On branch main\nnothing to commit", nil
		},
	})
	reg.Register(tools.Tool{
		Name:        "tree",
		Description: "Directory tree",
		Execute: func(args map[string]string) (string, error) {
			return ".\n├── cmd\n└── internal", nil
		},
	})
	// Dangerous tools (should never be speculatively executed)
	reg.Register(tools.Tool{
		Name:        "write",
		Description: "Write file",
		Execute: func(args map[string]string) (string, error) {
			return "written", nil
		},
	})
	reg.Register(tools.Tool{
		Name:        "shell",
		Description: "Shell command",
		Execute: func(args map[string]string) (string, error) {
			return "executed", nil
		},
	})
	return reg
}

func TestSpeculativeExecuteSearch(t *testing.T) {
	reg := mockToolRegistry()
	se := NewSpeculativeExecutor(reg, nil)

	bundle := se.Execute(`search for "ReflectionGate" in go files`)
	if bundle == nil {
		t.Fatal("expected results for search query")
	}

	found := false
	for _, r := range bundle.Results {
		if r.Tool == "grep" {
			found = true
			if !strings.Contains(r.Result, "ReflectionGate") {
				t.Errorf("grep result should contain search term, got %q", r.Result)
			}
		}
	}
	if !found {
		t.Error("expected grep in results")
	}
}

func TestSpeculativeExecuteFileRef(t *testing.T) {
	reg := mockToolRegistry()
	se := NewSpeculativeExecutor(reg, nil)

	bundle := se.Execute("read the file main.go")
	if bundle == nil {
		t.Fatal("expected results for file query")
	}

	found := false
	for _, r := range bundle.Results {
		if r.Tool == "read" {
			found = true
		}
	}
	if !found {
		t.Error("expected read in results")
	}
}

func TestSpeculativeExecuteGit(t *testing.T) {
	reg := mockToolRegistry()
	se := NewSpeculativeExecutor(reg, nil)

	bundle := se.Execute("show git status")
	if bundle == nil {
		t.Fatal("expected results for git query")
	}

	found := false
	for _, r := range bundle.Results {
		if r.Tool == "git" {
			found = true
		}
	}
	if !found {
		t.Error("expected git in results")
	}
}

func TestSpeculativeNeverWriteTools(t *testing.T) {
	reg := mockToolRegistry()
	se := NewSpeculativeExecutor(reg, nil)

	// Even if the query mentions "write", speculative should NOT execute write
	bundle := se.Execute("write hello to test.txt")
	if bundle != nil {
		for _, r := range bundle.Results {
			if r.Tool == "write" || r.Tool == "shell" || r.Tool == "edit" {
				t.Errorf("dangerous tool %q should never be speculatively executed", r.Tool)
			}
		}
	}
}

func TestSpeculativeMaxParallel(t *testing.T) {
	reg := mockToolRegistry()
	se := NewSpeculativeExecutor(reg, nil)
	se.maxParallel = 2

	candidates := se.analyzeCandidates("search for Pipeline and show git status and list files")
	// analyzeCandidates doesn't enforce max, Execute does
	if len(candidates) == 0 {
		t.Error("should find candidates")
	}
}

func TestSpeculativeEmptyQuery(t *testing.T) {
	reg := mockToolRegistry()
	se := NewSpeculativeExecutor(reg, nil)

	if bundle := se.Execute(""); bundle != nil {
		t.Error("empty query should return nil")
	}
}

func TestSpeculativeUnrelatedQuery(t *testing.T) {
	reg := mockToolRegistry()
	se := NewSpeculativeExecutor(reg, nil)

	bundle := se.Execute("what is the meaning of life?")
	if bundle != nil {
		t.Errorf("unrelated query should return nil or no results, got %d results", len(bundle.Results))
	}
}

func TestSpeculativeFormatEvidence(t *testing.T) {
	bundle := &SpecBundle{
		Query: "test",
		Results: []SpecResult{
			{Tool: "grep", Args: map[string]string{"pattern": "TODO"}, Result: "match1\nmatch2"},
			{Tool: "ls", Args: map[string]string{}, Result: "file1.go\nfile2.go"},
		},
	}

	evidence := bundle.FormatEvidence()
	if !strings.Contains(evidence, "Speculative results") {
		t.Error("evidence should have header")
	}
	if !strings.Contains(evidence, "grep") {
		t.Error("evidence should contain grep result")
	}
	if !strings.Contains(evidence, "match1") {
		t.Error("evidence should contain actual results")
	}
}

func TestSpeculativeFormatEvidenceNil(t *testing.T) {
	var sb *SpecBundle
	if sb.FormatEvidence() != "" {
		t.Error("nil bundle should return empty evidence")
	}
}

func TestSpeculativeFormatEvidenceTruncation(t *testing.T) {
	bundle := &SpecBundle{
		Query: "test",
		Results: []SpecResult{
			{Tool: "read", Args: map[string]string{"path": "big.go"}, Result: strings.Repeat("x", 3000)},
		},
	}

	evidence := bundle.FormatEvidence()
	if !strings.Contains(evidence, "truncated") {
		t.Error("long results should be truncated")
	}
}

// --- Helper Function Tests ---

func TestIsReadOnlyTool(t *testing.T) {
	readOnly := []string{"read", "ls", "tree", "glob", "grep", "sysinfo", "diff", "git"}
	writeTools := []string{"write", "edit", "shell", "mkdir", "patch", "find_replace"}

	for _, tool := range readOnly {
		if !isReadOnlyTool(tool) {
			t.Errorf("%q should be read-only", tool)
		}
	}
	for _, tool := range writeTools {
		if isReadOnlyTool(tool) {
			t.Errorf("%q should NOT be read-only", tool)
		}
	}
}

func TestExtractSearchTerm(t *testing.T) {
	tests := []struct {
		name  string
		query string
		want  string
	}{
		{"double quoted", `find "ReflectionGate"`, "ReflectionGate"},
		{"single quoted", `find 'Pipeline'`, "Pipeline"},
		{"backtick", "find `SmartTruncate`", "SmartTruncate"},
		{"camelCase unquoted", "find where ReflectionGate is used", "ReflectionGate"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := extractSearchTerm(tt.query)
			if got != tt.want {
				t.Errorf("extractSearchTerm(%q) = %q, want %q", tt.query, got, tt.want)
			}
		})
	}
}

func TestIsCamelCase(t *testing.T) {
	tests := []struct {
		s    string
		want bool
	}{
		{"ReflectionGate", true},
		{"camelCase", true},
		{"lowercase", false},
		{"UPPERCASE", false},
		{"Ab", false}, // too short
		{"SmartTruncate", true},
	}

	for _, tt := range tests {
		t.Run(tt.s, func(t *testing.T) {
			if got := isCamelCase(tt.s); got != tt.want {
				t.Errorf("isCamelCase(%q) = %v, want %v", tt.s, got, tt.want)
			}
		})
	}
}

func TestExtractFilePath(t *testing.T) {
	tests := []struct {
		name  string
		query string
		want  string
	}{
		{"go file", "read main.go", "main.go"},
		{"path with dir", "read internal/cognitive/reasoner.go", "internal/cognitive/reasoner.go"},
		{"no file", "hello world", ""},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := extractFilePath(tt.query)
			if got != tt.want {
				t.Errorf("extractFilePath(%q) = %q, want %q", tt.query, got, tt.want)
			}
		})
	}
}

func TestContainsAny(t *testing.T) {
	if !containsAny("hello world", "hello", "foo") {
		t.Error("should match")
	}
	if containsAny("hello world", "foo", "bar") {
		t.Error("should not match")
	}
}

// --- Benchmark ---

func BenchmarkSpeculativeExecute(b *testing.B) {
	reg := mockToolRegistry()
	se := NewSpeculativeExecutor(reg, nil)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		se.Execute(`search for "Pipeline" in all go files`)
	}
}
