package tools

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/artaeon/nous/internal/memory"
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
		"old":  "fmt.Println(\"hello\")",
		"new":  "fmt.Println(\"world\")",
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.Contains(result, "edited") {
		t.Error("expected confirmation message")
	}

	data, _ := os.ReadFile(filepath.Join(dir, "code.go"))
	if !strings.Contains(string(data), "fmt.Println(\"world\")") {
		t.Error("expected file to contain replaced string")
	}
	if strings.Contains(string(data), "fmt.Println(\"hello\")") {
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
	os.MkdirAll(filepath.Join(dir, "a", "b", "c", "d", "e"), 0755)
	writeFile(t, filepath.Join(dir, "a", "b", "c", "d", "e", "deep.txt"), "deep")

	r := NewRegistry()
	RegisterBuiltins(r, dir, false)

	tool, _ := r.Get("tree")
	result, err := tool.Execute(map[string]string{"depth": "2"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if !strings.Contains(result, "a") {
		t.Error("expected 'a' at depth 0")
	}
	if !strings.Contains(result, "b") {
		t.Error("expected 'b' at depth 1")
	}
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
	RegisterBuiltins(r, dir, false)

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
	RegisterBuiltins(r, dir, true)

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

	expectedTools := []string{"read", "write", "edit", "glob", "grep", "ls", "shell", "mkdir", "tree", "fetch", "run", "sysinfo", "find_replace", "git", "patch", "replace_all", "diff", "clipboard"}
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

	_, err := tool.Execute(map[string]string{})
	if err == nil {
		t.Fatal("expected error for missing url argument")
	}
	if !strings.Contains(err.Error(), "url") {
		t.Errorf("expected error about 'url', got %q", err.Error())
	}

	_, err = tool.Execute(map[string]string{"url": "http://127.0.0.1:1"})
	if err == nil {
		t.Fatal("expected error fetching from invalid URL")
	}
}

// --- run tool ---

func TestToolRun(t *testing.T) {
	dir := setupTempDir(t)
	r := NewRegistry()
	RegisterBuiltins(r, dir, true)

	tool, _ := r.Get("run")

	result, err := tool.Execute(map[string]string{"command": "echo hello"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.Contains(result, "hello") {
		t.Errorf("expected 'hello' in output, got %q", result)
	}

	result, err = tool.Execute(map[string]string{
		"command": "cat",
		"stdin":   "from stdin",
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.Contains(result, "from stdin") {
		t.Errorf("expected 'from stdin' in output, got %q", result)
	}

	_, err = tool.Execute(map[string]string{})
	if err == nil {
		t.Fatal("expected error for missing command argument")
	}
}

func TestToolRunDisabled(t *testing.T) {
	dir := setupTempDir(t)
	r := NewRegistry()
	RegisterBuiltins(r, dir, false)

	tool, _ := r.Get("run")
	_, err := tool.Execute(map[string]string{"command": "echo hello"})
	if err == nil {
		t.Fatal("expected error when shell is disabled")
	}
	if !strings.Contains(err.Error(), "disabled") {
		t.Errorf("expected 'disabled' in error, got %q", err.Error())
	}
}

// --- sysinfo tool ---

func TestToolSysinfo(t *testing.T) {
	dir := setupTempDir(t)
	r := NewRegistry()
	RegisterBuiltins(r, dir, false)

	tool, _ := r.Get("sysinfo")
	result, err := tool.Execute(map[string]string{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if !strings.Contains(result, "OS:") {
		t.Error("expected 'OS:' in sysinfo output")
	}
	if !strings.Contains(result, "Architecture:") {
		t.Error("expected 'Architecture:' in sysinfo output")
	}
	if !strings.Contains(result, "CPU cores:") {
		t.Error("expected 'CPU cores:' in sysinfo output")
	}
	if !strings.Contains(result, "Go version:") {
		t.Error("expected 'Go version:' in sysinfo output")
	}
	if !strings.Contains(result, "Disk:") {
		t.Error("expected 'Disk:' in sysinfo output")
	}
	if !strings.Contains(result, "linux") {
		t.Error("expected 'linux' in sysinfo output on a Linux system")
	}
}

// --- find_replace tool ---

func TestToolFindReplace(t *testing.T) {
	dir := setupTempDir(t)
	r := NewRegistry()
	RegisterBuiltins(r, dir, false)

	writeFile(t, filepath.Join(dir, "code.go"), "func hello() {\n\tfmt.Println(\"hello\")\n\tfmt.Println(\"hello world\")\n}\n")

	tool, _ := r.Get("find_replace")

	result, err := tool.Execute(map[string]string{
		"path":        "code.go",
		"pattern":     "hello",
		"replacement": "goodbye",
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.Contains(result, "1 occurrence") {
		t.Errorf("expected '1 occurrence' in result, got %q", result)
	}

	data, _ := os.ReadFile(filepath.Join(dir, "code.go"))
	content := string(data)
	if !strings.Contains(content, "goodbye") {
		t.Error("expected 'goodbye' in file after replacement")
	}
	if !strings.Contains(content, "hello") {
		t.Error("expected remaining 'hello' in file (only first should be replaced)")
	}
}

func TestToolFindReplaceAll(t *testing.T) {
	dir := setupTempDir(t)
	r := NewRegistry()
	RegisterBuiltins(r, dir, false)

	writeFile(t, filepath.Join(dir, "text.txt"), "foo bar foo baz foo")

	tool, _ := r.Get("find_replace")
	result, err := tool.Execute(map[string]string{
		"path":        "text.txt",
		"pattern":     "foo",
		"replacement": "qux",
		"all":         "true",
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.Contains(result, "3 occurrence") {
		t.Errorf("expected '3 occurrence' in result, got %q", result)
	}

	data, _ := os.ReadFile(filepath.Join(dir, "text.txt"))
	if strings.Contains(string(data), "foo") {
		t.Error("expected all 'foo' to be replaced")
	}
	if string(data) != "qux bar qux baz qux" {
		t.Errorf("expected 'qux bar qux baz qux', got %q", string(data))
	}
}

func TestToolFindReplaceRegex(t *testing.T) {
	dir := setupTempDir(t)
	r := NewRegistry()
	RegisterBuiltins(r, dir, false)

	writeFile(t, filepath.Join(dir, "numbers.txt"), "item1 item2 item3")

	tool, _ := r.Get("find_replace")
	result, err := tool.Execute(map[string]string{
		"path":        "numbers.txt",
		"pattern":     "item(\\d+)",
		"replacement": "thing${1}",
		"all":         "true",
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.Contains(result, "3 occurrence") {
		t.Errorf("expected '3 occurrence' in result, got %q", result)
	}

	data, _ := os.ReadFile(filepath.Join(dir, "numbers.txt"))
	if string(data) != "thing1 thing2 thing3" {
		t.Errorf("expected 'thing1 thing2 thing3', got %q", string(data))
	}
}

func TestToolFindReplaceNoMatch(t *testing.T) {
	dir := setupTempDir(t)
	r := NewRegistry()
	RegisterBuiltins(r, dir, false)

	writeFile(t, filepath.Join(dir, "file.txt"), "hello world")

	tool, _ := r.Get("find_replace")
	_, err := tool.Execute(map[string]string{
		"path":        "file.txt",
		"pattern":     "nonexistent",
		"replacement": "replaced",
	})
	if err == nil {
		t.Fatal("expected error when pattern not found")
	}
	if !strings.Contains(err.Error(), "not found") {
		t.Errorf("expected 'not found' in error, got %q", err.Error())
	}
}

// --- undo integration with tools ---

func TestToolWriteWithUndo(t *testing.T) {
	dir := setupTempDir(t)
	undo := memory.NewUndoStack(10)
	r := NewRegistry()
	RegisterBuiltins(r, dir, false, undo)

	// Write a new file
	tool, _ := r.Get("write")
	_, err := tool.Execute(map[string]string{
		"path":    "new.txt",
		"content": "hello",
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if undo.Size() != 1 {
		t.Fatalf("expected 1 undo entry, got %d", undo.Size())
	}

	entry, _ := undo.Peek()
	if entry.Action != "write" {
		t.Errorf("expected 'write' action, got %q", entry.Action)
	}
	if !entry.WasNew {
		t.Error("expected WasNew=true for new file")
	}
}

func TestToolEditWithUndo(t *testing.T) {
	dir := setupTempDir(t)
	writeFile(t, filepath.Join(dir, "file.txt"), "hello world")

	undo := memory.NewUndoStack(10)
	r := NewRegistry()
	RegisterBuiltins(r, dir, false, undo)

	tool, _ := r.Get("edit")
	_, err := tool.Execute(map[string]string{
		"path": "file.txt",
		"old":  "hello",
		"new":  "goodbye",
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if undo.Size() != 1 {
		t.Fatalf("expected 1 undo entry, got %d", undo.Size())
	}

	entry, _ := undo.Peek()
	if entry.Before != "hello world" {
		t.Errorf("expected before='hello world', got %q", entry.Before)
	}
}

func TestToolUndoReverts(t *testing.T) {
	dir := setupTempDir(t)
	writeFile(t, filepath.Join(dir, "file.txt"), "original")

	undo := memory.NewUndoStack(10)
	r := NewRegistry()
	RegisterBuiltins(r, dir, false, undo)

	// Edit the file
	tool, _ := r.Get("edit")
	_, _ = tool.Execute(map[string]string{
		"path": "file.txt",
		"old":  "original",
		"new":  "modified",
	})

	// Undo
	msg, err := undo.Undo()
	if err != nil {
		t.Fatalf("undo error: %v", err)
	}
	if msg == "" {
		t.Error("expected undo message")
	}

	// Verify file is restored
	data, _ := os.ReadFile(filepath.Join(dir, "file.txt"))
	if string(data) != "original" {
		t.Errorf("expected 'original' after undo, got %q", string(data))
	}
}

// --- clipboard tool ---

func TestToolClipboard(t *testing.T) {
	dir := setupTempDir(t)
	r := NewRegistry()
	RegisterBuiltins(r, dir, false)

	tool, _ := r.Get("clipboard")

	// Test missing action
	_, err := tool.Execute(map[string]string{})
	if err == nil {
		t.Fatal("expected error for missing action argument")
	}
	if !strings.Contains(err.Error(), "action") {
		t.Errorf("expected 'action' in error, got %q", err.Error())
	}

	// Test invalid action
	_, err = tool.Execute(map[string]string{"action": "invalid"})
	if err != nil {
		errStr := err.Error()
		// Either "neither xclip nor xsel" (no clipboard tool) or "unknown action"
		if !strings.Contains(errStr, "neither") && !strings.Contains(errStr, "unknown action") {
			t.Errorf("expected clipboard-related error, got %q", errStr)
		}
	}

	// Test write without content -- may fail due to no xclip/xsel, that's OK
	_, err = tool.Execute(map[string]string{"action": "write"})
	if err != nil {
		errStr := err.Error()
		if strings.Contains(errStr, "neither") {
			t.Logf("clipboard not available (expected in CI): %v", err)
		} else if strings.Contains(errStr, "content") {
			t.Logf("correctly requires content for write: %v", err)
		}
	}

	// Test read -- will likely fail due to no clipboard tool in test env
	_, err = tool.Execute(map[string]string{"action": "read"})
	if err != nil {
		errStr := err.Error()
		if strings.Contains(errStr, "neither") {
			t.Logf("clipboard not available (expected in CI): %v", err)
		}
	}
}
