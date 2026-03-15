package cognitive

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// --- Intent Compiler Tests ---

func setupTestCompiler(t *testing.T) (*IntentCompiler, string) {
	t.Helper()
	dir := t.TempDir()
	// Create a realistic project structure
	dirs := []string{
		"cmd/nous",
		"internal/cognitive",
		"internal/tools",
		"internal/ollama",
		"internal/memory",
	}
	for _, d := range dirs {
		os.MkdirAll(filepath.Join(dir, d), 0o755)
	}
	files := []string{
		"go.mod",
		"go.sum",
		"README.md",
		"Makefile",
		"cmd/nous/main.go",
		"internal/cognitive/reasoner.go",
		"internal/cognitive/pipeline.go",
		"internal/cognitive/grounding.go",
		"internal/cognitive/intent.go",
		"internal/tools/builtin.go",
		"internal/tools/registry.go",
		"internal/ollama/client.go",
		"internal/memory/episodic.go",
	}
	for _, f := range files {
		os.WriteFile(filepath.Join(dir, f), []byte("// "+f), 0o644)
	}
	ic := NewIntentCompiler(dir)
	return ic, dir
}

// --- Read Intent Tests ---

func TestIntentCompileReadFile(t *testing.T) {
	ic, _ := setupTestCompiler(t)

	tests := []struct {
		name  string
		input string
		want  string // expected resolved path
	}{
		{"read go.mod", "read the file go.mod", "go.mod"},
		{"read with quotes", `read "go.mod"`, "go.mod"},
		{"show me file", "show me README.md", "README.md"},
		{"look at file", "look at go.mod", "go.mod"},
		{"cat file", "cat go.mod", "go.mod"},
		{"view file", "view go.mod", "go.mod"},
		{"read nested", "read internal/cognitive/reasoner.go", "internal/cognitive/reasoner.go"},
		{"what is in", "what's in go.mod", "go.mod"},
		{"show file", "show go.mod", "go.mod"},
		{"open file", "open Makefile", "Makefile"},
		{"read partial path", "read reasoner.go", "internal/cognitive/reasoner.go"},
		{"read partial with dir", "read cognitive/reasoner.go", "internal/cognitive/reasoner.go"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			actions := ic.Compile(tt.input)
			if len(actions) == 0 {
				t.Fatalf("expected action for %q, got none", tt.input)
			}
			if actions[0].Tool != "read" {
				t.Errorf("tool = %q, want read", actions[0].Tool)
			}
			if actions[0].Args["path"] != tt.want {
				t.Errorf("path = %q, want %q", actions[0].Args["path"], tt.want)
			}
			if actions[0].Confidence < 0.8 {
				t.Errorf("confidence = %f, want >= 0.8", actions[0].Confidence)
			}
		})
	}
}

// --- Search Intent Tests ---

func TestIntentCompileSearch(t *testing.T) {
	ic, _ := setupTestCompiler(t)

	tests := []struct {
		name        string
		input       string
		wantPattern string
		wantGlob    string
	}{
		{"search quoted", `search for "ReflectionGate"`, "ReflectionGate", ""},
		{"find quoted", `find "Pipeline"`, "Pipeline", ""},
		{"grep for", "grep for TODO", "TODO", ""},
		{"search in go files", "search for ReflectionGate in all go files", "ReflectionGate", "*.go"},
		{"where is defined", "where is Pipeline defined", "Pipeline", ""},
		{"search with backticks", "search for `SmartTruncate`", "SmartTruncate", ""},
		{"find pattern", "find handleRequest in all python files", "handleRequest", "*.py"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			actions := ic.Compile(tt.input)
			if len(actions) == 0 {
				t.Fatalf("expected action for %q, got none", tt.input)
			}
			if actions[0].Tool != "grep" {
				t.Errorf("tool = %q, want grep", actions[0].Tool)
			}
			if actions[0].Args["pattern"] != tt.wantPattern {
				t.Errorf("pattern = %q, want %q", actions[0].Args["pattern"], tt.wantPattern)
			}
			if tt.wantGlob != "" && actions[0].Args["glob"] != tt.wantGlob {
				t.Errorf("glob = %q, want %q", actions[0].Args["glob"], tt.wantGlob)
			}
		})
	}
}

// --- Semantic Grep Intent Tests ---

func TestIntentCompileSemanticGrep(t *testing.T) {
	ic, _ := setupTestCompiler(t)

	tests := []struct {
		name        string
		input       string
		wantPattern string
		wantPath    string
	}{
		{"show structs", "show all structs in reasoner.go", `type \w+ struct`, "internal/cognitive/reasoner.go"},
		{"find functions", "find functions in pipeline.go", `^func `, "internal/cognitive/pipeline.go"},
		{"list methods", "list all methods in grounding.go", `func \(`, "internal/cognitive/grounding.go"},
		{"get interfaces", "get interfaces in intent.go", `type \w+ interface`, "internal/cognitive/intent.go"},
		{"show types", "show types in client.go", `^type `, "internal/ollama/client.go"},
		{"extract imports", "extract imports in go.mod", `^import`, "go.mod"},
		{"find constants", "find all constants in builtin.go", `^const `, "internal/tools/builtin.go"},
		{"show variables", "show variables in registry.go", `^var `, "internal/tools/registry.go"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			actions := ic.Compile(tt.input)
			if len(actions) == 0 {
				t.Fatalf("expected action for %q, got none", tt.input)
			}
			if actions[0].Tool != "grep" {
				t.Errorf("tool = %q, want grep", actions[0].Tool)
			}
			if actions[0].Args["pattern"] != tt.wantPattern {
				t.Errorf("pattern = %q, want %q", actions[0].Args["pattern"], tt.wantPattern)
			}
			if actions[0].Args["path"] != tt.wantPath {
				t.Errorf("path = %q, want %q", actions[0].Args["path"], tt.wantPath)
			}
			if actions[0].Source != "semantic-grep-pattern" {
				t.Errorf("source = %q, want semantic-grep-pattern", actions[0].Source)
			}
			if actions[0].Confidence < 0.85 {
				t.Errorf("confidence = %f, want >= 0.85", actions[0].Confidence)
			}
		})
	}
}

// --- List Intent Tests ---

func TestIntentCompileList(t *testing.T) {
	ic, _ := setupTestCompiler(t)

	tests := []struct {
		name     string
		input    string
		wantPath string // empty = current dir
	}{
		{"list files", "list files", ""},
		{"what files current", "what files are in the current directory", ""},
		{"ls dir", "list files in internal/cognitive", "internal/cognitive"},
		{"show directory", "show the directory", ""},
		{"whats in current dir", "what's in the current folder", ""},
		{"what files in dir", "what files are in internal/cognitive", "internal/cognitive"},
		{"what files inside dir", "what files are inside internal/tools", "internal/tools"},
		{"what files under dir", "what files under internal/ollama", "internal/ollama"},
		{"whats in dir slash", "what's in internal/cognitive/", "internal/cognitive"},
		{"what is inside dir slash", "what is inside internal/memory/", "internal/memory"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			actions := ic.Compile(tt.input)
			if len(actions) == 0 {
				t.Fatalf("expected action for %q, got none", tt.input)
			}
			if actions[0].Tool != "ls" {
				t.Errorf("tool = %q, want ls", actions[0].Tool)
			}
			gotPath := actions[0].Args["path"]
			if tt.wantPath != "" && gotPath != tt.wantPath {
				t.Errorf("path = %q, want %q", gotPath, tt.wantPath)
			}
		})
	}
}

// --- Write Intent Tests ---

func TestIntentCompileWrite(t *testing.T) {
	ic, _ := setupTestCompiler(t)

	tests := []struct {
		name        string
		input       string
		wantPath    string
		wantContent string
	}{
		{
			"create with text",
			`create a file called /tmp/test.txt with the text "hello world"`,
			"/tmp/test.txt",
			"hello world",
		},
		{
			"write to file",
			`write "test content" to /tmp/output.txt`,
			"/tmp/output.txt",
			"test content",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			actions := ic.Compile(tt.input)
			if len(actions) == 0 {
				t.Fatalf("expected action for %q, got none", tt.input)
			}
			if actions[0].Tool != "write" {
				t.Errorf("tool = %q, want write", actions[0].Tool)
			}
			if actions[0].Args["path"] != tt.wantPath {
				t.Errorf("path = %q, want %q", actions[0].Args["path"], tt.wantPath)
			}
			if actions[0].Args["content"] != tt.wantContent {
				t.Errorf("content = %q, want %q", actions[0].Args["content"], tt.wantContent)
			}
		})
	}
}

// --- Git Intent Tests ---

func TestIntentCompileGit(t *testing.T) {
	ic, _ := setupTestCompiler(t)

	tests := []struct {
		name    string
		input   string
		wantCmd string
	}{
		{"git status", "git status", "status"},
		{"show status", "show the git status", "status"},
		{"what changed", "what are the changes", "status"},
		{"git log", "show the commit history", "log --oneline -15"},
		{"git diff", "git diff", "diff"},
		{"show recent commits", "show recent commits", "log --oneline -15"},
		{"last commits", "show last commits", "log --oneline -15"},
		{"latest commits", "show latest commits", "log --oneline -15"},
		{"recent changes", "show recent changes", "log --oneline -15"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			actions := ic.Compile(tt.input)
			if len(actions) == 0 {
				t.Fatalf("expected action for %q, got none", tt.input)
			}
			if actions[0].Tool != "git" {
				t.Errorf("tool = %q, want git", actions[0].Tool)
			}
			if actions[0].Args["command"] != tt.wantCmd {
				t.Errorf("command = %q, want %q", actions[0].Args["command"], tt.wantCmd)
			}
		})
	}
}

// --- Tree Intent Tests ---

func TestIntentCompileTree(t *testing.T) {
	ic, _ := setupTestCompiler(t)

	actions := ic.Compile("show the project structure")
	if len(actions) == 0 {
		t.Fatal("expected action for tree query")
	}
	if actions[0].Tool != "tree" {
		t.Errorf("tool = %q, want tree", actions[0].Tool)
	}
}

// --- Glob Intent Tests ---

func TestIntentCompileGlobCount(t *testing.T) {
	ic, _ := setupTestCompiler(t)

	tests := []struct {
		name        string
		input       string
		wantPattern string
		wantSource  string
	}{
		{"how many test files", "how many test files are there?", "**/*_test.go", "glob-count-pattern"},
		{"count test files", "count test files", "**/*_test.go", "glob-count-pattern"},
		{"how many files", "how many files are there", "**/*", "glob-count-pattern"},
		{"how many files exist", "how many files exist", "**/*", "glob-count-pattern"},
		{"count files", "count files", "**/*", "glob-count-pattern"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			actions := ic.Compile(tt.input)
			if len(actions) == 0 {
				t.Fatalf("expected action for %q, got none", tt.input)
			}
			if actions[0].Tool != "glob" {
				t.Errorf("tool = %q, want glob", actions[0].Tool)
			}
			if actions[0].Args["pattern"] != tt.wantPattern {
				t.Errorf("pattern = %q, want %q", actions[0].Args["pattern"], tt.wantPattern)
			}
			if actions[0].Source != tt.wantSource {
				t.Errorf("source = %q, want %q", actions[0].Source, tt.wantSource)
			}
		})
	}
}

func TestIntentCompileGlobSuperlative(t *testing.T) {
	ic, _ := setupTestCompiler(t)

	tests := []struct {
		name        string
		input       string
		wantPattern string
		wantSource  string
	}{
		{"largest go files", "find the largest go files", "**/*.go", "glob-superlative-pattern"},
		{"biggest files", "find the biggest files", "**/*", "glob-superlative-pattern"},
		{"smallest python files", "show the smallest python files", "**/*.py", "glob-superlative-pattern"},
		{"newest files", "find the newest files", "**/*", "glob-superlative-pattern"},
		{"oldest rust files", "list oldest rust files", "**/*.rs", "glob-superlative-pattern"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			actions := ic.Compile(tt.input)
			if len(actions) == 0 {
				t.Fatalf("expected action for %q, got none", tt.input)
			}
			if actions[0].Tool != "glob" {
				t.Errorf("tool = %q, want glob", actions[0].Tool)
			}
			if actions[0].Args["pattern"] != tt.wantPattern {
				t.Errorf("pattern = %q, want %q", actions[0].Args["pattern"], tt.wantPattern)
			}
			if actions[0].Source != tt.wantSource {
				t.Errorf("source = %q, want %q", actions[0].Source, tt.wantSource)
			}
		})
	}
}

func TestIntentCompileGlob(t *testing.T) {
	ic, _ := setupTestCompiler(t)

	tests := []struct {
		name        string
		input       string
		wantPattern string
	}{
		{"find go files", "find all go files", "**/*.go"},
		{"find matching", `find files matching "*.test.js"`, "*.test.js"},
		{"list py files", "list all python files", "**/*.py"},
		{"find test files", "find all test files", "**/*_test*"},
		{"find spec files", "find all spec files", "**/*_spec*"},
		{"find mock files", "find all mock files", "**/*mock*"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			actions := ic.Compile(tt.input)
			if len(actions) == 0 {
				t.Fatalf("expected action for %q, got none", tt.input)
			}
			if actions[0].Tool != "glob" {
				t.Errorf("tool = %q, want glob", actions[0].Tool)
			}
			if actions[0].Args["pattern"] != tt.wantPattern {
				t.Errorf("pattern = %q, want %q", actions[0].Args["pattern"], tt.wantPattern)
			}
		})
	}
}

// --- Path Resolution Tests ---

func TestResolvePathExact(t *testing.T) {
	ic, _ := setupTestCompiler(t)
	got := ic.ResolvePath("go.mod")
	if got != "go.mod" {
		t.Errorf("ResolvePath(go.mod) = %q, want go.mod", got)
	}
}

func TestResolvePathSuffix(t *testing.T) {
	ic, _ := setupTestCompiler(t)
	got := ic.ResolvePath("main.go")
	if got != "cmd/nous/main.go" {
		t.Errorf("ResolvePath(main.go) = %q, want cmd/nous/main.go", got)
	}
}

func TestResolvePathPartial(t *testing.T) {
	ic, _ := setupTestCompiler(t)
	got := ic.ResolvePath("cognitive/reasoner.go")
	if got != "internal/cognitive/reasoner.go" {
		t.Errorf("ResolvePath(cognitive/reasoner.go) = %q, want internal/cognitive/reasoner.go", got)
	}
}

func TestResolvePathNonExistent(t *testing.T) {
	ic, _ := setupTestCompiler(t)
	got := ic.ResolvePath("nonexistent.xyz")
	if got != "" {
		t.Errorf("ResolvePath(nonexistent) = %q, want empty", got)
	}
}

func TestResolvePathEmpty(t *testing.T) {
	ic, _ := setupTestCompiler(t)
	got := ic.ResolvePath("")
	if got != "" {
		t.Errorf("ResolvePath('') = %q, want empty", got)
	}
}

func TestResolveDir(t *testing.T) {
	ic, _ := setupTestCompiler(t)

	tests := []struct {
		name string
		frag string
		want string
	}{
		{"exact dir", "cmd/nous", "cmd/nous"},
		{"partial dir", "cognitive", "internal/cognitive"},
		{"nested dir", "internal/tools", "internal/tools"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := ic.ResolveDir(tt.frag)
			if got != tt.want {
				t.Errorf("ResolveDir(%q) = %q, want %q", tt.frag, got, tt.want)
			}
		})
	}
}

// --- Response Recovery Tests ---

func TestCompileResponseRecovery(t *testing.T) {
	ic, _ := setupTestCompiler(t)

	tests := []struct {
		name     string
		response string
		input    string
		wantTool string
	}{
		{
			"model mentions reading",
			"Let me read the go.mod file to check the version",
			"what Go version does this use?",
			"read",
		},
		{
			"model mentions searching",
			"I need to search for ReflectionGate in the codebase",
			"find where ReflectionGate is used",
			"grep",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			actions := ic.CompileResponse(tt.response, tt.input)
			if len(actions) == 0 {
				t.Fatalf("expected action from response recovery")
			}
			if actions[0].Tool != tt.wantTool {
				t.Errorf("tool = %q, want %q", actions[0].Tool, tt.wantTool)
			}
			if !strings.HasPrefix(actions[0].Source, "response-recovery:") {
				t.Errorf("source should be prefixed with response-recovery:, got %q", actions[0].Source)
			}
			if actions[0].Confidence >= 0.9 {
				t.Errorf("response recovery confidence should be reduced, got %f", actions[0].Confidence)
			}
		})
	}
}

// --- Edge Cases ---

func TestCompileEmptyInput(t *testing.T) {
	ic, _ := setupTestCompiler(t)
	if actions := ic.Compile(""); actions != nil {
		t.Errorf("empty input should return nil, got %v", actions)
	}
}

func TestCompileNoMatch(t *testing.T) {
	ic, _ := setupTestCompiler(t)
	actions := ic.Compile("hello, how are you?")
	if len(actions) != 0 {
		t.Errorf("greeting should not match any intent, got %v", actions)
	}
}

func TestCompileConversationalNoMatch(t *testing.T) {
	ic, _ := setupTestCompiler(t)
	noMatch := []string{
		"what is the meaning of life?",
		"explain quantum computing",
		"thank you very much",
		"yes",
		"no",
		"2+2",
	}
	for _, input := range noMatch {
		if actions := ic.Compile(input); len(actions) != 0 {
			t.Errorf("Compile(%q) should not match, got tool=%s", input, actions[0].Tool)
		}
	}
}

// --- FileTree and DirTree Tests ---

func TestFileTreePopulated(t *testing.T) {
	ic, _ := setupTestCompiler(t)
	tree := ic.FileTree()
	if len(tree) < 5 {
		t.Errorf("FileTree should have files, got %d", len(tree))
	}
	// Should contain go.mod
	found := false
	for _, f := range tree {
		if f == "go.mod" {
			found = true
		}
	}
	if !found {
		t.Error("FileTree should contain go.mod")
	}
}

func TestDirTreePopulated(t *testing.T) {
	ic, _ := setupTestCompiler(t)
	tree := ic.DirTree()
	if len(tree) < 3 {
		t.Errorf("DirTree should have directories, got %d", len(tree))
	}
}

// --- Noise Directory Filtering ---

func TestNoiseDirSkipped(t *testing.T) {
	dir := t.TempDir()
	os.MkdirAll(filepath.Join(dir, "node_modules/foo"), 0o755)
	os.MkdirAll(filepath.Join(dir, ".git/objects"), 0o755)
	os.MkdirAll(filepath.Join(dir, "src"), 0o755)
	os.WriteFile(filepath.Join(dir, "node_modules/foo/bar.js"), []byte("x"), 0o644)
	os.WriteFile(filepath.Join(dir, ".git/objects/abc"), []byte("x"), 0o644)
	os.WriteFile(filepath.Join(dir, "src/main.go"), []byte("x"), 0o644)

	ic := NewIntentCompiler(dir)
	tree := ic.FileTree()

	for _, f := range tree {
		if strings.Contains(f, "node_modules") || strings.Contains(f, ".git") {
			t.Errorf("noise file %q should be excluded", f)
		}
	}
	found := false
	for _, f := range tree {
		if f == "src/main.go" {
			found = true
		}
	}
	if !found {
		t.Error("src/main.go should be in FileTree")
	}
}

// --- File Extension Mapping ---

func TestFileExtToGlob(t *testing.T) {
	tests := []struct {
		ext  string
		want string
	}{
		{"go", "*.go"},
		{"python", "*.py"},
		{"py", "*.py"},
		{"javascript", "*.js"},
		{"typescript", "*.ts"},
		{"rust", "*.rs"},
	}

	for _, tt := range tests {
		t.Run(tt.ext, func(t *testing.T) {
			got, ok := fileExtToGlob[tt.ext]
			if !ok {
				t.Fatalf("fileExtToGlob missing entry for %q", tt.ext)
			}
			if got != tt.want {
				t.Errorf("fileExtToGlob[%q] = %q, want %q", tt.ext, got, tt.want)
			}
		})
	}
}

// --- Confidence Scoring ---

func TestConfidenceLevels(t *testing.T) {
	ic, _ := setupTestCompiler(t)

	// Read should have high confidence
	read := ic.Compile("read go.mod")
	if len(read) > 0 && read[0].Confidence < 0.85 {
		t.Errorf("read confidence = %f, want >= 0.85", read[0].Confidence)
	}

	// Write should have slightly lower confidence
	write := ic.Compile(`create a file called /tmp/test.txt with the text "hello"`)
	if len(write) > 0 && write[0].Confidence > 0.85 {
		t.Errorf("write confidence = %f, want <= 0.85", write[0].Confidence)
	}
}

// --- Benchmark ---

func BenchmarkIntentCompile(b *testing.B) {
	dir := b.TempDir()
	for _, d := range []string{"cmd/nous", "internal/cognitive", "internal/tools"} {
		os.MkdirAll(filepath.Join(dir, d), 0o755)
	}
	for _, f := range []string{"go.mod", "cmd/nous/main.go", "internal/cognitive/reasoner.go"} {
		os.WriteFile(filepath.Join(dir, f), []byte("x"), 0o644)
	}
	ic := NewIntentCompiler(dir)

	queries := []string{
		"read go.mod",
		`search for "Pipeline" in all go files`,
		"list files in internal/cognitive",
		"hello how are you",
		"show the project structure",
		"git status",
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ic.Compile(queries[i%len(queries)])
	}
}

func BenchmarkResolvePath(b *testing.B) {
	dir := b.TempDir()
	for _, d := range []string{"cmd/nous", "internal/cognitive", "internal/tools"} {
		os.MkdirAll(filepath.Join(dir, d), 0o755)
	}
	for _, f := range []string{"go.mod", "cmd/nous/main.go", "internal/cognitive/reasoner.go"} {
		os.WriteFile(filepath.Join(dir, f), []byte("x"), 0o644)
	}
	ic := NewIntentCompiler(dir)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ic.ResolvePath("reasoner.go")
	}
}
