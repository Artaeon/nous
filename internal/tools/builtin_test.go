package tools

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

func setupTempDir(t *testing.T) string {
	t.Helper()
	return t.TempDir()
}

func writeFile(t *testing.T, path, content string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
		t.Fatalf("failed to create parent dir: %v", err)
	}
	if err := os.WriteFile(path, []byte(content), 0644); err != nil {
		t.Fatalf("failed to write file: %v", err)
	}
}

// --- read tool ---

func TestToolRead(t *testing.T) {
	dir := setupTempDir(t)
	writeFile(t, filepath.Join(dir, "hello.txt"), "line one\nline two\nline three\n")

	r := NewRegistry()
	RegisterBuiltins(r, dir, false)

	tool, _ := r.Get("read")
	result, err := tool.Execute(map[string]string{"path": "hello.txt"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if !strings.Contains(result, "line one") {
		t.Error("expected result to contain 'line one'")
	}
	if !strings.Contains(result, "line two") {
		t.Error("expected result to contain 'line two'")
	}
	// Should have line numbers
	if !strings.Contains(result, "1 |") {
		t.Error("expected line numbers in output")
	}
}

func TestToolReadWithOffsetAndLimit(t *testing.T) {
	dir := setupTempDir(t)
	content := "line1\nline2\nline3\nline4\nline5\n"
	writeFile(t, filepath.Join(dir, "file.txt"), content)

	r := NewRegistry()
	RegisterBuiltins(r, dir, false)

	tool, _ := r.Get("read")
	result, err := tool.Execute(map[string]string{
		"path":   "file.txt",
		"offset": "2",
		"limit":  "2",
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if !strings.Contains(result, "line3") {
		t.Error("expected result to contain 'line3' (offset=2)")
	}
	if !strings.Contains(result, "line4") {
		t.Error("expected result to contain 'line4'")
	}
	if strings.Contains(result, "line1") {
		t.Error("expected result to NOT contain 'line1' (before offset)")
	}
	if strings.Contains(result, "line5") {
		t.Error("expected result to NOT contain 'line5' (after limit)")
	}
}

func TestToolReadNonExistent(t *testing.T) {
	dir := setupTempDir(t)
	r := NewRegistry()
	RegisterBuiltins(r, dir, false)

	tool, _ := r.Get("read")
	_, err := tool.Execute(map[string]string{"path": "nope.txt"})
	if err == nil {
		t.Fatal("expected error reading nonexistent file")
	}
}

func TestToolReadAbsolutePath(t *testing.T) {
	dir := setupTempDir(t)
	absPath := filepath.Join(dir, "abs.txt")
	writeFile(t, absPath, "absolute content")

	r := NewRegistry()
	RegisterBuiltins(r, dir, false)

	tool, _ := r.Get("read")
	result, err := tool.Execute(map[string]string{"path": absPath})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.Contains(result, "absolute content") {
		t.Error("expected to read file via absolute path")
	}
}

// --- write tool ---

func TestToolWrite(t *testing.T) {
	dir := setupTempDir(t)
	r := NewRegistry()
	RegisterBuiltins(r, dir, false)

	tool, _ := r.Get("write")
	result, err := tool.Execute(map[string]string{
		"path":    "output.txt",
		"content": "hello world",
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.Contains(result, "wrote") {
		t.Error("expected confirmation message")
	}

	// Verify the file was created
	data, err := os.ReadFile(filepath.Join(dir, "output.txt"))
	if err != nil {
		t.Fatalf("failed to read written file: %v", err)
	}
	if string(data) != "hello world" {
		t.Errorf("expected 'hello world', got %q", string(data))
	}
}

func TestToolWriteCreatesParentDirs(t *testing.T) {
	dir := setupTempDir(t)
	r := NewRegistry()
	RegisterBuiltins(r, dir, false)

	tool, _ := r.Get("write")
	_, err := tool.Execute(map[string]string{
		"path":    "sub/deep/file.txt",
		"content": "nested",
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	data, _ := os.ReadFile(filepath.Join(dir, "sub", "deep", "file.txt"))
	if string(data) != "nested" {
		t.Errorf("expected 'nested', got %q", string(data))
	}
}

func TestToolWriteOverwrite(t *testing.T) {
	dir := setupTempDir(t)
	writeFile(t, filepath.Join(dir, "file.txt"), "original")

	r := NewRegistry()
	RegisterBuiltins(r, dir, false)

	tool, _ := r.Get("write")
	_, err := tool.Execute(map[string]string{
		"path":    "file.txt",
		"content": "replaced",
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	data, _ := os.ReadFile(filepath.Join(dir, "file.txt"))
	if string(data) != "replaced" {
		t.Errorf("expected 'replaced', got %q", string(data))
	}
}

// --- edit tool ---

func TestToolEdit(t *testing.T) {
	dir := setupTempDir(t)
	writeFile(t, filepath.Join(dir, "code.go"), "func main() {\n\tfmt.Println(\"hello\")\n}\n")

	r := NewRegistry()
	RegisterBuiltins(r, dir, false)

	tool, _ := r.Get("edit")
	result, err := tool.Execute(map[string]string{
		"path": "code.go",
		"old":  `fmt.Println("hello")`,
		"new":  `fmt.Println("world")`,
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.Contains(result, "edited") {
		t.Error("expected confirmation message")
	}

	data, _ := os.ReadFile(filepath.Join(dir, "code.go"))
	if !strings.Contains(string(data), `fmt.Println("world")`) {
		t.Error("expected file to contain replaced string")
	}
	if strings.Contains(string(data), `fmt.Println("hello")`) {
		t.Error("expected original string to be replaced")
	}
}

func TestToolEditNotFound(t *testing.T) {
	dir := setupTempDir(t)
	writeFile(t, filepath.Join(dir, "file.txt"), "some content")

	r := NewRegistry()
	RegisterBuiltins(r, dir, false)

	tool, _ := r.Get("edit")
	_, err := tool.Execute(map[string]string{
		"path": "file.txt",
		"old":  "nonexistent string",
		"new":  "replacement",
	})
	if err == nil {
		t.Fatal("expected error when old string not found")
	}
	if !strings.Contains(err.Error(), "not found") {
		t.Errorf("expected 'not found' in error, got %q", err.Error())
	}
}

func TestToolEditAmbiguous(t *testing.T) {
	dir := setupTempDir(t)
	writeFile(t, filepath.Join(dir, "file.txt"), "hello\nhello\n")

	r := NewRegistry()
	RegisterBuiltins(r, dir, false)

	tool, _ := r.Get("edit")
	_, err := tool.Execute(map[string]string{
		"path": "file.txt",
		"old":  "hello",
		"new":  "world",
	})
	if err == nil {
		t.Fatal("expected error when old string matches multiple times")
	}
	if !strings.Contains(err.Error(), "2 times") {
		t.Errorf("expected error about multiple matches, got %q", err.Error())
	}
}

// --- glob tool ---

func TestToolGlob(t *testing.T) {
	dir := setupTempDir(t)
	writeFile(t, filepath.Join(dir, "main.go"), "package main")
	writeFile(t, filepath.Join(dir, "util.go"), "package main")
	writeFile(t, filepath.Join(dir, "readme.md"), "# Readme")

	r := NewRegistry()
	RegisterBuiltins(r, dir, false)

	tool, _ := r.Get("glob")
	result, err := tool.Execute(map[string]string{"pattern": "*.go"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if !strings.Contains(result, "main.go") {
		t.Error("expected 'main.go' in glob results")
	}
	if !strings.Contains(result, "util.go") {
		t.Error("expected 'util.go' in glob results")
	}
	if strings.Contains(result, "readme.md") {
		t.Error("expected 'readme.md' to NOT match *.go pattern")
	}
}

func TestToolGlobNoMatches(t *testing.T) {
	dir := setupTempDir(t)
	writeFile(t, filepath.Join(dir, "file.txt"), "content")

	r := NewRegistry()
	RegisterBuiltins(r, dir, false)

	tool, _ := r.Get("glob")
	result, err := tool.Execute(map[string]string{"pattern": "*.rs"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.Contains(result, "no files matched") {
		t.Errorf("expected 'no files matched', got %q", result)
	}
}

func TestToolGlobMissingPattern(t *testing.T) {
	dir := setupTempDir(t)
	r := NewRegistry()
	RegisterBuiltins(r, dir, false)

	tool, _ := r.Get("glob")
	_, err := tool.Execute(map[string]string{})
	if err == nil {
		t.Fatal("expected error for missing pattern")
	}
}

// --- grep tool ---

func TestToolGrep(t *testing.T) {
	dir := setupTempDir(t)
	writeFile(t, filepath.Join(dir, "main.go"), "package main\n\nfunc main() {\n}\n")
	writeFile(t, filepath.Join(dir, "util.go"), "package main\n\nfunc helper() {\n}\n")

	r := NewRegistry()
	RegisterBuiltins(r, dir, false)

	tool, _ := r.Get("grep")
	result, err := tool.Execute(map[string]string{"pattern": "func"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if !strings.Contains(result, "func main") {
		t.Error("expected grep to find 'func main'")
	}
	if !strings.Contains(result, "func helper") {
		t.Error("expected grep to find 'func helper'")
	}
}

func TestToolGrepNoMatches(t *testing.T) {
	dir := setupTempDir(t)
	writeFile(t, filepath.Join(dir, "file.txt"), "hello world")

	r := NewRegistry()
	RegisterBuiltins(r, dir, false)

	tool, _ := r.Get("grep")
	result, err := tool.Execute(map[string]string{"pattern": "nonexistent_string_xyz"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.Contains(result, "no matches") {
		t.Errorf("expected 'no matches', got %q", result)
	}
}

func TestToolGrepWithGlobFilter(t *testing.T) {
	dir := setupTempDir(t)
	writeFile(t, filepath.Join(dir, "main.go"), "func main() {}")
	writeFile(t, filepath.Join(dir, "notes.txt"), "func is a keyword")

	r := NewRegistry()
	RegisterBuiltins(r, dir, false)

	tool, _ := r.Get("grep")
	result, err := tool.Execute(map[string]string{
		"pattern": "func",
		"glob":    "*.go",
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if !strings.Contains(result, "main.go") {
		t.Error("expected grep to find match in main.go")
	}
	if strings.Contains(result, "notes.txt") {
		t.Error("expected grep to NOT match notes.txt with *.go filter")
	}
}

// --- ls tool ---

func TestToolLs(t *testing.T) {
	dir := setupTempDir(t)
	writeFile(t, filepath.Join(dir, "file1.txt"), "content1")
	writeFile(t, filepath.Join(dir, "file2.go"), "content2")
	os.MkdirAll(filepath.Join(dir, "subdir"), 0755)

	r := NewRegistry()
	RegisterBuiltins(r, dir, false)

	tool, _ := r.Get("ls")
	result, err := tool.Execute(map[string]string{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if !strings.Contains(result, "file1.txt") {
		t.Error("expected 'file1.txt' in ls output")
	}
	if !strings.Contains(result, "file2.go") {
		t.Error("expected 'file2.go' in ls output")
	}
	if !strings.Contains(result, "subdir") {
		t.Error("expected 'subdir' in ls output")
	}
	// Directory entries should have "d" prefix
	if !strings.Contains(result, "d ") {
		t.Error("expected directory prefix 'd' in output")
	}
}

func TestToolLsSubdir(t *testing.T) {
	dir := setupTempDir(t)
	os.MkdirAll(filepath.Join(dir, "sub"), 0755)
	writeFile(t, filepath.Join(dir, "sub", "nested.txt"), "nested")

	r := NewRegistry()
	RegisterBuiltins(r, dir, false)

	tool, _ := r.Get("ls")
	result, err := tool.Execute(map[string]string{"path": "sub"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if !strings.Contains(result, "nested.txt") {
		t.Error("expected 'nested.txt' in ls output for subdirectory")
	}
}

func TestToolLsNonExistent(t *testing.T) {
	dir := setupTempDir(t)
	r := NewRegistry()
	RegisterBuiltins(r, dir, false)

	tool, _ := r.Get("ls")
	_, err := tool.Execute(map[string]string{"path": "nonexistent"})
	if err == nil {
		t.Fatal("expected error listing nonexistent directory")
	}
}

// --- mkdir tool ---

func TestToolMkdir(t *testing.T) {
	dir := setupTempDir(t)
	r := NewRegistry()
	RegisterBuiltins(r, dir, false)

	tool, _ := r.Get("mkdir")
	result, err := tool.Execute(map[string]string{"path": "newdir"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.Contains(result, "created") {
		t.Error("expected confirmation message")
	}

	info, err := os.Stat(filepath.Join(dir, "newdir"))
	if err != nil {
		t.Fatal("expected directory to exist")
	}
	if !info.IsDir() {
		t.Error("expected path to be a directory")
	}
}

func TestToolMkdirNested(t *testing.T) {
	dir := setupTempDir(t)
	r := NewRegistry()
	RegisterBuiltins(r, dir, false)

	tool, _ := r.Get("mkdir")
	_, err := tool.Execute(map[string]string{"path": "a/b/c"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	info, err := os.Stat(filepath.Join(dir, "a", "b", "c"))
	if err != nil {
		t.Fatal("expected nested directory to exist")
	}
	if !info.IsDir() {
		t.Error("expected path to be a directory")
	}
}

// --- tree tool ---

func TestToolTree(t *testing.T) {
	dir := setupTempDir(t)
	writeFile(t, filepath.Join(dir, "main.go"), "package main")
	os.MkdirAll(filepath.Join(dir, "internal", "pkg"), 0755)
	writeFile(t, filepath.Join(dir, "internal", "pkg", "util.go"), "package pkg")

	r := NewRegistry()
	RegisterBuiltins(r, dir, false)

	tool, _ := r.Get("tree")
	result, err := tool.Execute(map[string]string{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if !strings.Contains(result, "main.go") {
		t.Error("expected 'main.go' in tree output")
	}
	if !strings.Contains(result, "internal") {
		t.Error("expected 'internal' in tree output")
	}
	if !strings.Contains(result, "pkg") {
		t.Error("expected 'pkg' in tree output")
	}
}

func TestToolTreeDepthLimit(t *testing.T) {
	dir := setupTempDir(t)
	// Create a deeply nested structure
	os.MkdirAll(filepath.Join(dir, "a", "b", "c", "d", "e"), 0755)
	writeFile(t, filepath.Join(dir, "a", "b", "c", "d", "e", "deep.txt"), "deep")

	r := NewRegistry()
	RegisterBuiltins(r, dir, false)

	tool, _ := r.Get("tree")
	result, err := tool.Execute(map[string]string{"depth": "2"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// At depth 2, we should see "a" and "b" but not "c" or "d"
	if !strings.Contains(result, "a") {
		t.Error("expected 'a' at depth 0")
	}
	if !strings.Contains(result, "b") {
		t.Error("expected 'b' at depth 1")
	}
	// "c" is at depth 2 which means it should NOT appear since
	// buildTree returns when depth >= maxDepth
	if strings.Contains(result, "deep.txt") {
		t.Error("expected 'deep.txt' to be beyond depth limit")
	}
}

func TestToolTreeHiddenFilesExcluded(t *testing.T) {
	dir := setupTempDir(t)
	os.MkdirAll(filepath.Join(dir, ".git"), 0755)
	writeFile(t, filepath.Join(dir, ".git", "config"), "git config")
	writeFile(t, filepath.Join(dir, "visible.txt"), "visible")

	r := NewRegistry()
	RegisterBuiltins(r, dir, false)

	tool, _ := r.Get("tree")
	result, err := tool.Execute(map[string]string{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if strings.Contains(result, ".git") {
		t.Error("expected .git to be excluded from tree")
	}
	if !strings.Contains(result, "visible.txt") {
		t.Error("expected 'visible.txt' in tree output")
	}
}

// --- shell tool ---

func TestToolShellDisabled(t *testing.T) {
	dir := setupTempDir(t)
	r := NewRegistry()
	RegisterBuiltins(r, dir, false) // allowShell=false

	tool, _ := r.Get("shell")
	_, err := tool.Execute(map[string]string{"command": "echo hello"})
	if err == nil {
		t.Fatal("expected error when shell is disabled")
	}
	if !strings.Contains(err.Error(), "disabled") {
		t.Errorf("expected 'disabled' in error, got %q", err.Error())
	}
}

func TestToolShellEnabled(t *testing.T) {
	dir := setupTempDir(t)
	r := NewRegistry()
	RegisterBuiltins(r, dir, true) // allowShell=true

	tool, _ := r.Get("shell")
	result, err := tool.Execute(map[string]string{"command": "echo hello"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.Contains(result, "hello") {
		t.Errorf("expected 'hello' in output, got %q", result)
	}
}

// --- RegisterBuiltins completeness ---

func TestRegisterBuiltinsRegistersAllTools(t *testing.T) {
	dir := setupTempDir(t)
	r := NewRegistry()
	RegisterBuiltins(r, dir, false)

	expectedTools := []string{"read", "write", "edit", "glob", "grep", "ls", "shell", "mkdir", "tree", "fetch", "git", "patch", "replace_all", "diff"}
	tools := r.List()

	registered := map[string]bool{}
	for _, tool := range tools {
		registered[tool.Name] = true
	}

	for _, name := range expectedTools {
		if !registered[name] {
			t.Errorf("expected tool %q to be registered", name)
		}
	}

	if len(tools) != len(expectedTools) {
		t.Errorf("expected %d tools, got %d", len(expectedTools), len(tools))
	}
}

// --- resolvePath ---

func TestResolvePath(t *testing.T) {
	tests := []struct {
		workDir  string
		path     string
		expected string
	}{
		{"/home/user/project", "file.txt", "/home/user/project/file.txt"},
		{"/home/user/project", "sub/file.txt", "/home/user/project/sub/file.txt"},
		{"/home/user/project", "/absolute/path.txt", "/absolute/path.txt"},
	}

	for _, tt := range tests {
		got := resolvePath(tt.workDir, tt.path)
		if got != tt.expected {
			t.Errorf("resolvePath(%q, %q) = %q, want %q", tt.workDir, tt.path, got, tt.expected)
		}
	}
}

// --- fetch tool ---

func TestToolFetch(t *testing.T) {
	dir := setupTempDir(t)
	r := NewRegistry()
	RegisterBuiltins(r, dir, false)

	tool, _ := r.Get("fetch")

	// Test missing URL argument
	_, err := tool.Execute(map[string]string{})
	if err == nil {
		t.Fatal("expected error for missing url argument")
	}
	if !strings.Contains(err.Error(), "url") {
		t.Errorf("expected error about 'url', got %q", err.Error())
	}

	// Test with an invalid URL (connection refused on localhost)
	_, err = tool.Execute(map[string]string{"url": "http://127.0.0.1:1"})
	if err == nil {
		t.Fatal("expected error fetching from invalid URL")
	}

	// Test with a completely malformed URL
	_, err = tool.Execute(map[string]string{"url": "not-a-url"})
	if err != nil {
		// Good -- invalid URL should fail at request creation or connection
		if !strings.Contains(err.Error(), "fetch") {
			t.Errorf("expected 'fetch' in error, got %q", err.Error())
		}
	}
}
