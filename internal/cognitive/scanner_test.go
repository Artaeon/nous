package cognitive

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func setupScanDir(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()

	// Create a realistic Go project structure
	writeTestFile(t, filepath.Join(dir, "go.mod"), "module example.com/test\n\ngo 1.22\n")
	writeTestFile(t, filepath.Join(dir, "main.go"), "package main\n\nfunc main() {}\n")
	writeTestFile(t, filepath.Join(dir, "README.md"), "# Test Project\n")

	os.MkdirAll(filepath.Join(dir, "internal", "pkg"), 0755)
	writeTestFile(t, filepath.Join(dir, "internal", "pkg", "util.go"), "package pkg\n")

	os.MkdirAll(filepath.Join(dir, "cmd"), 0755)
	writeTestFile(t, filepath.Join(dir, "cmd", "app.go"), "package main\n")

	return dir
}

func writeTestFile(t *testing.T, path, content string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
		t.Fatalf("failed to create dir: %v", err)
	}
	if err := os.WriteFile(path, []byte(content), 0644); err != nil {
		t.Fatalf("failed to write file: %v", err)
	}
}

func TestScanProjectBasic(t *testing.T) {
	dir := setupScanDir(t)

	info := ScanProject(dir)
	if info == nil {
		t.Fatal("ScanProject returned nil")
	}

	if info.RootDir != dir {
		t.Errorf("expected RootDir %q, got %q", dir, info.RootDir)
	}

	if info.Name != filepath.Base(dir) {
		t.Errorf("expected Name=%q, got %q", filepath.Base(dir), info.Name)
	}
}

func TestScanProjectDetectsGoLanguage(t *testing.T) {
	dir := setupScanDir(t)
	info := ScanProject(dir)

	if info.Language != "Go" {
		t.Errorf("expected language 'Go', got %q", info.Language)
	}
}

func TestScanProjectDetectsRust(t *testing.T) {
	dir := t.TempDir()
	writeTestFile(t, filepath.Join(dir, "Cargo.toml"), "[package]\nname = \"test\"\n")
	writeTestFile(t, filepath.Join(dir, "src", "main.rs"), "fn main() {}\n")
	writeTestFile(t, filepath.Join(dir, "src", "lib.rs"), "pub fn hello() {}\n")

	info := ScanProject(dir)
	if info.Language != "Rust" {
		t.Errorf("expected language 'Rust', got %q", info.Language)
	}
}

func TestScanProjectDetectsPython(t *testing.T) {
	dir := t.TempDir()
	writeTestFile(t, filepath.Join(dir, "setup.py"), "from setuptools import setup\n")
	writeTestFile(t, filepath.Join(dir, "app.py"), "print('hello')\n")
	writeTestFile(t, filepath.Join(dir, "utils.py"), "def helper(): pass\n")

	info := ScanProject(dir)
	if info.Language != "Python" {
		t.Errorf("expected language 'Python', got %q", info.Language)
	}
}

func TestScanProjectDetectsTypeScript(t *testing.T) {
	dir := t.TempDir()
	writeTestFile(t, filepath.Join(dir, "package.json"), "{}\n")
	writeTestFile(t, filepath.Join(dir, "tsconfig.json"), "{}\n")
	writeTestFile(t, filepath.Join(dir, "src", "index.ts"), "console.log('hi')\n")
	writeTestFile(t, filepath.Join(dir, "src", "utils.ts"), "export const x = 1\n")

	info := ScanProject(dir)
	if info.Language != "TypeScript" {
		t.Errorf("expected language 'TypeScript', got %q", info.Language)
	}
}

func TestScanProjectDetectsJavaScript(t *testing.T) {
	dir := t.TempDir()
	writeTestFile(t, filepath.Join(dir, "package.json"), "{}\n")
	writeTestFile(t, filepath.Join(dir, "index.js"), "console.log('hi')\n")
	writeTestFile(t, filepath.Join(dir, "utils.js"), "module.exports = {}\n")

	info := ScanProject(dir)
	if info.Language != "JavaScript" {
		t.Errorf("expected language 'JavaScript', got %q", info.Language)
	}
}

func TestScanProjectCountsFiles(t *testing.T) {
	dir := setupScanDir(t)
	info := ScanProject(dir)

	// We created: go.mod, main.go, README.md, internal/pkg/util.go, cmd/app.go = 5 files
	if info.FileCount != 5 {
		t.Errorf("expected 5 files, got %d", info.FileCount)
	}
}

func TestScanProjectFindsKeyFiles(t *testing.T) {
	dir := setupScanDir(t)
	info := ScanProject(dir)

	keyFileSet := map[string]bool{}
	for _, kf := range info.KeyFiles {
		keyFileSet[kf] = true
	}

	if !keyFileSet["go.mod"] {
		t.Error("expected 'go.mod' in key files")
	}
	if !keyFileSet["README.md"] {
		t.Error("expected 'README.md' in key files")
	}
}

func TestScanProjectKeyFilesVariants(t *testing.T) {
	dir := t.TempDir()
	writeTestFile(t, filepath.Join(dir, "Dockerfile"), "FROM golang:1.22\n")
	writeTestFile(t, filepath.Join(dir, "Makefile"), "build:\n\tgo build\n")
	writeTestFile(t, filepath.Join(dir, "main.go"), "package main\n")
	writeTestFile(t, filepath.Join(dir, ".env.example"), "KEY=value\n")
	writeTestFile(t, filepath.Join(dir, "claude.md"), "# Claude\n")

	info := ScanProject(dir)

	keyFileSet := map[string]bool{}
	for _, kf := range info.KeyFiles {
		keyFileSet[kf] = true
	}

	// Note: Dockerfile and Makefile detection uses ToLower
	if !keyFileSet["Dockerfile"] {
		// Check lowercase variant
		if !keyFileSet["dockerfile"] {
			t.Error("expected 'Dockerfile' or 'dockerfile' in key files")
		}
	}
}

func TestScanProjectSkipsHiddenDirs(t *testing.T) {
	dir := t.TempDir()
	os.MkdirAll(filepath.Join(dir, ".git", "objects"), 0755)
	writeTestFile(t, filepath.Join(dir, ".git", "config"), "git config")
	writeTestFile(t, filepath.Join(dir, "visible.go"), "package main\n")

	info := ScanProject(dir)

	// .git contents should not be counted
	if info.FileCount != 1 {
		t.Errorf("expected 1 file (excluding .git), got %d", info.FileCount)
	}
}

func TestScanProjectSkipsNodeModules(t *testing.T) {
	dir := t.TempDir()
	os.MkdirAll(filepath.Join(dir, "node_modules", "pkg"), 0755)
	writeTestFile(t, filepath.Join(dir, "node_modules", "pkg", "index.js"), "module.exports = {}")
	writeTestFile(t, filepath.Join(dir, "index.js"), "console.log('app')\n")

	info := ScanProject(dir)
	if info.FileCount != 1 {
		t.Errorf("expected 1 file (excluding node_modules), got %d", info.FileCount)
	}
}

func TestScanProjectTree(t *testing.T) {
	dir := setupScanDir(t)
	info := ScanProject(dir)

	if info.Tree == "" {
		t.Fatal("expected non-empty tree")
	}
	if !strings.Contains(info.Tree, "internal") {
		t.Error("expected 'internal' in tree")
	}
	if !strings.Contains(info.Tree, "main.go") {
		t.Error("expected 'main.go' in tree")
	}
}

func TestContextString(t *testing.T) {
	dir := setupScanDir(t)
	info := ScanProject(dir)
	ctx := info.ContextString()

	if !strings.Contains(ctx, "Project:") {
		t.Error("expected 'Project:' in context string")
	}
	if !strings.Contains(ctx, "Language: Go") {
		t.Error("expected 'Language: Go' in context string")
	}
	if !strings.Contains(ctx, "Files:") {
		t.Error("expected 'Files:' in context string")
	}
	if !strings.Contains(ctx, "Key files:") {
		t.Error("expected 'Key files:' in context string")
	}
	if !strings.Contains(ctx, "Structure:") {
		t.Error("expected 'Structure:' in context string")
	}
}

func TestContextStringEmptyProject(t *testing.T) {
	dir := t.TempDir()
	info := ScanProject(dir)
	ctx := info.ContextString()

	if !strings.Contains(ctx, "Project:") {
		t.Error("expected 'Project:' even for empty project")
	}
	// Should not contain Key files if none found
	if strings.Contains(ctx, "Key files:") {
		t.Error("expected no 'Key files:' for empty project")
	}
}

func TestDetectLanguageUnknown(t *testing.T) {
	dir := t.TempDir()
	// No files at all
	info := ScanProject(dir)
	if info.Language != "unknown" {
		t.Errorf("expected language 'unknown' for empty dir, got %q", info.Language)
	}
}

func TestDetectLanguageFallback(t *testing.T) {
	dir := t.TempDir()
	// Create files with an uncommon extension
	writeTestFile(t, filepath.Join(dir, "a.xyz"), "data")
	writeTestFile(t, filepath.Join(dir, "b.xyz"), "data")

	info := ScanProject(dir)
	if info.Language != "xyz" {
		t.Errorf("expected language 'xyz' as fallback, got %q", info.Language)
	}
}
