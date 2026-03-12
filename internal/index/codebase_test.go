package index

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

const testGoFile = `package example

import (
	"fmt"
	"strings"
)

// Greeter says hello.
type Greeter struct {
	Name string
}

// Greet returns a greeting.
func (g *Greeter) Greet() string {
	return "Hello, " + g.Name
}

// Helper is a standalone function.
func Helper() {
	fmt.Println(strings.TrimSpace("hi"))
}

// Speaker can speak.
type Speaker interface {
	Speak() string
}

// MaxRetries is the default retry count.
const MaxRetries = 3

// DefaultName is the fallback name.
var DefaultName = "World"
`

func setupTestDir(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()

	// Write a Go file
	err := os.WriteFile(filepath.Join(dir, "example.go"), []byte(testGoFile), 0644)
	if err != nil {
		t.Fatal(err)
	}

	return dir
}

func buildTestIndex(t *testing.T, dir string) *CodebaseIndex {
	t.Helper()
	store := t.TempDir()
	idx := NewCodebaseIndex(store)
	if err := idx.Build(dir); err != nil {
		t.Fatal(err)
	}
	return idx
}

func TestBuildIndex(t *testing.T) {
	dir := setupTestDir(t)
	idx := buildTestIndex(t, dir)

	if idx.Size() == 0 {
		t.Fatal("expected symbols in index, got 0")
	}

	// Should have: Greeter (struct), Greet (method), Helper (func), Speaker (interface), MaxRetries (const), DefaultName (var)
	if idx.Size() != 6 {
		t.Errorf("expected 6 symbols, got %d", idx.Size())
		for _, s := range idx.Symbols {
			t.Logf("  %s (%s)", s.Name, s.Kind)
		}
	}
}

func TestBuildIndexFunction(t *testing.T) {
	dir := setupTestDir(t)
	idx := buildTestIndex(t, dir)

	var found *Symbol
	for i := range idx.Symbols {
		if idx.Symbols[i].Name == "Helper" {
			found = &idx.Symbols[i]
			break
		}
	}

	if found == nil {
		t.Fatal("expected to find symbol 'Helper'")
	}
	if found.Kind != "func" {
		t.Errorf("expected kind 'func', got %q", found.Kind)
	}
	if found.Package != "example" {
		t.Errorf("expected package 'example', got %q", found.Package)
	}
	if found.File != "example.go" {
		t.Errorf("expected file 'example.go', got %q", found.File)
	}
	if found.Line == 0 {
		t.Error("expected non-zero line number")
	}
}

func TestBuildIndexStruct(t *testing.T) {
	dir := setupTestDir(t)
	idx := buildTestIndex(t, dir)

	var found *Symbol
	for i := range idx.Symbols {
		if idx.Symbols[i].Name == "Greeter" {
			found = &idx.Symbols[i]
			break
		}
	}

	if found == nil {
		t.Fatal("expected to find symbol 'Greeter'")
	}
	if found.Kind != "struct" {
		t.Errorf("expected kind 'struct', got %q", found.Kind)
	}
	if found.Signature != "type Greeter struct" {
		t.Errorf("expected signature 'type Greeter struct', got %q", found.Signature)
	}
}

func TestBuildIndexMethod(t *testing.T) {
	dir := setupTestDir(t)
	idx := buildTestIndex(t, dir)

	var found *Symbol
	for i := range idx.Symbols {
		if idx.Symbols[i].Name == "Greet" {
			found = &idx.Symbols[i]
			break
		}
	}

	if found == nil {
		t.Fatal("expected to find symbol 'Greet'")
	}
	if found.Kind != "method" {
		t.Errorf("expected kind 'method', got %q", found.Kind)
	}
	if !strings.Contains(found.Signature, "Greet") {
		t.Errorf("expected signature to contain 'Greet', got %q", found.Signature)
	}
}

func TestBuildIndexInterface(t *testing.T) {
	dir := setupTestDir(t)
	idx := buildTestIndex(t, dir)

	var found *Symbol
	for i := range idx.Symbols {
		if idx.Symbols[i].Name == "Speaker" {
			found = &idx.Symbols[i]
			break
		}
	}

	if found == nil {
		t.Fatal("expected to find symbol 'Speaker'")
	}
	if found.Kind != "interface" {
		t.Errorf("expected kind 'interface', got %q", found.Kind)
	}
	if found.Signature != "type Speaker interface" {
		t.Errorf("expected signature 'type Speaker interface', got %q", found.Signature)
	}
}

func TestBuildIndexImports(t *testing.T) {
	dir := setupTestDir(t)
	idx := buildTestIndex(t, dir)

	if len(idx.Files) == 0 {
		t.Fatal("expected files in index")
	}

	fi := idx.Files[0]
	if fi.Package != "example" {
		t.Errorf("expected package 'example', got %q", fi.Package)
	}

	hasImport := func(name string) bool {
		for _, imp := range fi.Imports {
			if imp == name {
				return true
			}
		}
		return false
	}

	if !hasImport("fmt") {
		t.Error("expected import 'fmt'")
	}
	if !hasImport("strings") {
		t.Error("expected import 'strings'")
	}
}

func TestBuildIndexDocComment(t *testing.T) {
	dir := setupTestDir(t)
	idx := buildTestIndex(t, dir)

	var found *Symbol
	for i := range idx.Symbols {
		if idx.Symbols[i].Name == "Greeter" {
			found = &idx.Symbols[i]
			break
		}
	}

	if found == nil {
		t.Fatal("expected to find symbol 'Greeter'")
	}
	if !strings.Contains(found.Doc, "Greeter says hello") {
		t.Errorf("expected doc to contain 'Greeter says hello', got %q", found.Doc)
	}

	// Also check function doc
	for i := range idx.Symbols {
		if idx.Symbols[i].Name == "Greet" {
			if !strings.Contains(idx.Symbols[i].Doc, "Greet returns a greeting") {
				t.Errorf("expected doc to contain 'Greet returns a greeting', got %q", idx.Symbols[i].Doc)
			}
			break
		}
	}
}

func TestLookup(t *testing.T) {
	dir := setupTestDir(t)
	idx := buildTestIndex(t, dir)

	results := idx.Lookup("greet")
	if len(results) == 0 {
		t.Fatal("expected lookup results for 'greet'")
	}

	// Should find both Greeter and Greet
	names := make(map[string]bool)
	for _, r := range results {
		names[r.Name] = true
	}
	if !names["Greeter"] {
		t.Error("expected Greeter in results")
	}
	if !names["Greet"] {
		t.Error("expected Greet in results")
	}
}

func TestLookupCaseInsensitive(t *testing.T) {
	dir := setupTestDir(t)
	idx := buildTestIndex(t, dir)

	upper := idx.Lookup("HELPER")
	lower := idx.Lookup("helper")
	mixed := idx.Lookup("Helper")

	if len(upper) == 0 || len(lower) == 0 || len(mixed) == 0 {
		t.Fatal("expected case-insensitive lookup to find results")
	}

	if upper[0].Name != lower[0].Name || lower[0].Name != mixed[0].Name {
		t.Error("expected case-insensitive results to match")
	}
}

func TestFileContext(t *testing.T) {
	dir := setupTestDir(t)
	idx := buildTestIndex(t, dir)

	ctx := idx.FileContext("example.go")
	if ctx == "" {
		t.Fatal("expected non-empty file context")
	}

	if !strings.HasPrefix(ctx, "package example:") {
		t.Errorf("expected context to start with 'package example:', got %q", ctx)
	}

	// Should contain symbol names
	for _, name := range []string{"Greeter", "Greet", "Helper", "Speaker"} {
		if !strings.Contains(ctx, name) {
			t.Errorf("expected context to contain %q", name)
		}
	}
}

func TestRelevantContext(t *testing.T) {
	dir := setupTestDir(t)
	idx := buildTestIndex(t, dir)

	ctx := idx.RelevantContext("how does Greeter work", 5)
	if ctx == "" {
		t.Fatal("expected non-empty relevant context")
	}

	if !strings.HasPrefix(ctx, "[Codebase]") {
		t.Errorf("expected context to start with '[Codebase]', got %q", ctx)
	}

	if !strings.Contains(ctx, "Greeter") {
		t.Error("expected context to contain 'Greeter'")
	}

	// Should contain file:line references
	if !strings.Contains(ctx, "example.go:") {
		t.Error("expected context to contain file:line reference")
	}
}

func TestIncrementalUpdate(t *testing.T) {
	dir := setupTestDir(t)
	store := t.TempDir()
	idx := NewCodebaseIndex(store)
	if err := idx.Build(dir); err != nil {
		t.Fatal(err)
	}

	origSize := idx.Size()

	// Modify the file — add a new function
	modified := testGoFile + `
// Extra is a new function.
func Extra() int {
	return 42
}
`
	if err := os.WriteFile(filepath.Join(dir, "example.go"), []byte(modified), 0644); err != nil {
		t.Fatal(err)
	}

	// Incremental update
	if err := idx.IncrementalUpdate(dir, []string{"example.go"}); err != nil {
		t.Fatal(err)
	}

	if idx.Size() != origSize+1 {
		t.Errorf("expected %d symbols after incremental update, got %d", origSize+1, idx.Size())
	}

	// Verify the new function is found
	results := idx.Lookup("Extra")
	if len(results) == 0 {
		t.Error("expected to find 'Extra' after incremental update")
	}
}

func TestSaveAndLoad(t *testing.T) {
	dir := setupTestDir(t)
	store := t.TempDir()
	idx := NewCodebaseIndex(store)
	if err := idx.Build(dir); err != nil {
		t.Fatal(err)
	}

	originalSize := idx.Size()
	if originalSize == 0 {
		t.Fatal("expected non-empty index before save")
	}

	// Save
	if err := idx.Save(); err != nil {
		t.Fatal("save failed:", err)
	}

	// Load into a fresh index
	idx2 := NewCodebaseIndex(store)
	if idx2.Size() != originalSize {
		t.Errorf("expected %d symbols after load, got %d", originalSize, idx2.Size())
	}

	// Verify symbols match
	if len(idx2.Files) != len(idx.Files) {
		t.Errorf("expected %d files after load, got %d", len(idx.Files), len(idx2.Files))
	}
}

func TestBuildIndexSkipsVendor(t *testing.T) {
	dir := t.TempDir()

	// Create a Go file in the root
	if err := os.WriteFile(filepath.Join(dir, "main.go"), []byte(`package main

func Main() {}
`), 0644); err != nil {
		t.Fatal(err)
	}

	// Create vendor directory with a Go file
	vendorDir := filepath.Join(dir, "vendor", "lib")
	if err := os.MkdirAll(vendorDir, 0755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(vendorDir, "lib.go"), []byte(`package lib

func VendorFunc() {}
`), 0644); err != nil {
		t.Fatal(err)
	}

	// Create .git directory with a Go file
	gitDir := filepath.Join(dir, ".git", "hooks")
	if err := os.MkdirAll(gitDir, 0755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(gitDir, "hook.go"), []byte(`package hooks

func GitHook() {}
`), 0644); err != nil {
		t.Fatal(err)
	}

	store := t.TempDir()
	idx := NewCodebaseIndex(store)
	if err := idx.Build(dir); err != nil {
		t.Fatal(err)
	}

	// Should only have Main from root
	if idx.Size() != 1 {
		t.Errorf("expected 1 symbol (skipping vendor/.git), got %d", idx.Size())
		for _, s := range idx.Symbols {
			t.Logf("  %s (%s) in %s", s.Name, s.Kind, s.File)
		}
	}

	results := idx.Lookup("VendorFunc")
	if len(results) != 0 {
		t.Error("expected vendor functions to be skipped")
	}

	results = idx.Lookup("GitHook")
	if len(results) != 0 {
		t.Error("expected .git functions to be skipped")
	}
}

func TestBestFileForQuery(t *testing.T) {
	dir := setupTestDir(t)
	idx := buildTestIndex(t, dir)

	// Query about "Greeter" should return example.go
	best := idx.BestFileForQuery("what does Greeter do")
	if best != "example.go" {
		t.Errorf("expected 'example.go', got %q", best)
	}

	// Query about "Helper" should also return example.go
	best = idx.BestFileForQuery("explain Helper function")
	if best != "example.go" {
		t.Errorf("expected 'example.go', got %q", best)
	}

	// No match
	best = idx.BestFileForQuery("something completely unrelated xyz")
	if best != "" {
		t.Errorf("expected empty for unrelated query, got %q", best)
	}
}

func TestBestSymbolForQuery(t *testing.T) {
	dir := setupTestDir(t)
	idx := buildTestIndex(t, dir)

	// Query about "Greeter" should return the Greeter symbol with line number
	sym := idx.BestSymbolForQuery("what does Greeter do")
	if sym == nil {
		t.Fatal("expected non-nil symbol for 'Greeter'")
	}
	if sym.Name != "Greeter" {
		t.Errorf("expected symbol name 'Greeter', got %q", sym.Name)
	}
	if sym.File != "example.go" {
		t.Errorf("expected file 'example.go', got %q", sym.File)
	}
	if sym.Line == 0 {
		t.Error("expected non-zero line number")
	}

	// Query about "Helper" should return the Helper symbol
	sym = idx.BestSymbolForQuery("explain Helper function")
	if sym == nil {
		t.Fatal("expected non-nil symbol for 'Helper'")
	}
	if sym.Name != "Helper" {
		t.Errorf("expected symbol name 'Helper', got %q", sym.Name)
	}

	// No match
	sym = idx.BestSymbolForQuery("something completely unrelated xyz")
	if sym != nil {
		t.Errorf("expected nil for unrelated query, got %v", sym)
	}
}

func TestBestFileForQueryExactNameBoost(t *testing.T) {
	dir := t.TempDir()

	// Create two files with overlapping but different symbols
	if err := os.WriteFile(filepath.Join(dir, "alpha.go"), []byte(`package test
func Alpha() {}
func AlphaHelper() {}
`), 0644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "beta.go"), []byte(`package test
func Beta() {}
func BetaRunner() {}
`), 0644); err != nil {
		t.Fatal(err)
	}

	idx := buildTestIndex(t, dir)

	// Exact name "Alpha" should prefer alpha.go
	best := idx.BestFileForQuery("Alpha")
	if best != "alpha.go" {
		t.Errorf("expected 'alpha.go' for exact name match, got %q", best)
	}

	// Exact name "Beta" should prefer beta.go
	best = idx.BestFileForQuery("Beta")
	if best != "beta.go" {
		t.Errorf("expected 'beta.go' for exact name match, got %q", best)
	}
}
