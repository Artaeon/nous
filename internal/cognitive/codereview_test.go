package cognitive

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func writeReviewFile(t *testing.T, name, content string) string {
	t.Helper()
	dir := t.TempDir()
	path := filepath.Join(dir, name)
	if err := os.WriteFile(path, []byte(content), 0644); err != nil {
		t.Fatal(err)
	}
	return path
}

func hasRule(results []ReviewResult, rule string) bool {
	for _, r := range results {
		if r.Rule == rule {
			return true
		}
	}
	return false
}

func countRule(results []ReviewResult, rule string) int {
	n := 0
	for _, r := range results {
		if r.Rule == rule {
			n++
		}
	}
	return n
}

func TestReview_UncheckedError(t *testing.T) {
	src := `package main

import "os"

func bad() {
	_, _ = os.Open("file.txt")
}
`
	path := writeReviewFile(t, "unchecked.go", src)
	cr := NewCodeReviewer()

	results, err := cr.ReviewFile(path)
	if err != nil {
		t.Fatalf("ReviewFile: %v", err)
	}

	if !hasRule(results, "unchecked-error") {
		t.Error("expected unchecked-error finding for os.Open with _ error")
	}
}

func TestReview_UnusedParameter(t *testing.T) {
	src := `package main

func process(data string, count int) string {
	return "fixed"
}
`
	path := writeReviewFile(t, "unused.go", src)
	cr := NewCodeReviewer()

	results, err := cr.ReviewFile(path)
	if err != nil {
		t.Fatalf("ReviewFile: %v", err)
	}

	if !hasRule(results, "unused-parameter") {
		t.Error("expected unused-parameter finding")
	}

	// Both data and count are unused
	c := countRule(results, "unused-parameter")
	if c < 2 {
		t.Errorf("expected at least 2 unused-parameter findings, got %d", c)
	}
}

func TestReview_UnusedParameter_Used(t *testing.T) {
	src := `package main

import "fmt"

func greet(name string) string {
	return fmt.Sprintf("hello %s", name)
}
`
	path := writeReviewFile(t, "used.go", src)
	cr := NewCodeReviewer()

	results, err := cr.ReviewFile(path)
	if err != nil {
		t.Fatalf("ReviewFile: %v", err)
	}

	if hasRule(results, "unused-parameter") {
		t.Error("did not expect unused-parameter finding when parameter is used")
	}
}

func TestReview_EmptyErrorHandler(t *testing.T) {
	src := `package main

import "os"

func doSomething() {
	_, err := os.Open("file.txt")
	if err != nil {
	}
}
`
	path := writeReviewFile(t, "emptyerr.go", src)
	cr := NewCodeReviewer()

	results, err := cr.ReviewFile(path)
	if err != nil {
		t.Fatalf("ReviewFile: %v", err)
	}

	if !hasRule(results, "empty-error-handler") {
		t.Error("expected empty-error-handler finding")
	}
}

func TestReview_EmptyErrorHandler_BareReturn(t *testing.T) {
	src := `package main

import "os"

func doSomething() {
	_, err := os.Open("file.txt")
	if err != nil {
		return
	}
}
`
	path := writeReviewFile(t, "barereturn.go", src)
	cr := NewCodeReviewer()

	results, err := cr.ReviewFile(path)
	if err != nil {
		t.Fatalf("ReviewFile: %v", err)
	}

	if !hasRule(results, "empty-error-handler") {
		t.Error("expected empty-error-handler finding for bare return")
	}
}

func TestReview_EmptyErrorHandler_ProperHandling(t *testing.T) {
	src := `package main

import (
	"fmt"
	"os"
)

func doSomething() error {
	_, err := os.Open("file.txt")
	if err != nil {
		return fmt.Errorf("open failed: %w", err)
	}
	return nil
}
`
	path := writeReviewFile(t, "propererr.go", src)
	cr := NewCodeReviewer()

	results, err := cr.ReviewFile(path)
	if err != nil {
		t.Fatalf("ReviewFile: %v", err)
	}

	if hasRule(results, "empty-error-handler") {
		t.Error("did not expect empty-error-handler for proper error handling")
	}
}

func TestReview_HardcodedSecret(t *testing.T) {
	src := `package main

func config() {
	password := "super_secret_123"
	apiKey := "sk-1234567890abcdef"
	token := "eyJhbGciOiJIUzI1NiJ9"
	_ = password
	_ = apiKey
	_ = token
}
`
	path := writeReviewFile(t, "secrets.go", src)
	cr := NewCodeReviewer()

	results, err := cr.ReviewFile(path)
	if err != nil {
		t.Fatalf("ReviewFile: %v", err)
	}

	if !hasRule(results, "hardcoded-secret") {
		t.Error("expected hardcoded-secret finding")
	}

	c := countRule(results, "hardcoded-secret")
	if c < 3 {
		t.Errorf("expected at least 3 hardcoded-secret findings (password, apiKey, token), got %d", c)
	}
}

func TestReview_HardcodedSecret_VarDecl(t *testing.T) {
	src := `package main

var dbPassword = "letmein"
`
	path := writeReviewFile(t, "varsecret.go", src)
	cr := NewCodeReviewer()

	results, err := cr.ReviewFile(path)
	if err != nil {
		t.Fatalf("ReviewFile: %v", err)
	}

	if !hasRule(results, "hardcoded-secret") {
		t.Error("expected hardcoded-secret finding for var declaration")
	}
}

func TestReview_HardcodedSecret_NoFalsePositive(t *testing.T) {
	src := `package main

import "os"

func config() string {
	password := os.Getenv("DB_PASSWORD")
	return password
}
`
	path := writeReviewFile(t, "envsecret.go", src)
	cr := NewCodeReviewer()

	results, err := cr.ReviewFile(path)
	if err != nil {
		t.Fatalf("ReviewFile: %v", err)
	}

	if hasRule(results, "hardcoded-secret") {
		t.Error("did not expect hardcoded-secret for os.Getenv call")
	}
}

func TestReview_TodoComment(t *testing.T) {
	src := `package main

// TODO: implement this properly
func stub() {
	// FIXME: broken logic
	// HACK: temporary workaround
	// XXX: needs review
}
`
	path := writeReviewFile(t, "todos.go", src)
	cr := NewCodeReviewer()

	results, err := cr.ReviewFile(path)
	if err != nil {
		t.Fatalf("ReviewFile: %v", err)
	}

	if !hasRule(results, "todo-comment") {
		t.Error("expected todo-comment finding")
	}

	c := countRule(results, "todo-comment")
	if c < 4 {
		t.Errorf("expected at least 4 todo-comment findings (TODO, FIXME, HACK, XXX), got %d", c)
	}
}

func TestReview_LongFunction(t *testing.T) {
	// Generate a function with 60 lines
	var lines []string
	lines = append(lines, `package main`)
	lines = append(lines, ``)
	lines = append(lines, `import "fmt"`)
	lines = append(lines, ``)
	lines = append(lines, `func longFunc() {`)
	for i := 0; i < 55; i++ {
		lines = append(lines, `	fmt.Println("line")`)
	}
	lines = append(lines, `}`)

	src := strings.Join(lines, "\n")
	path := writeReviewFile(t, "longfunc.go", src)
	cr := NewCodeReviewer()

	results, err := cr.ReviewFile(path)
	if err != nil {
		t.Fatalf("ReviewFile: %v", err)
	}

	if !hasRule(results, "long-function") {
		t.Error("expected long-function finding for 55+ line function")
	}
}

func TestReview_LongFunction_Short(t *testing.T) {
	src := `package main

import "fmt"

func shortFunc() {
	fmt.Println("hello")
	fmt.Println("world")
}
`
	path := writeReviewFile(t, "shortfunc.go", src)
	cr := NewCodeReviewer()

	results, err := cr.ReviewFile(path)
	if err != nil {
		t.Fatalf("ReviewFile: %v", err)
	}

	if hasRule(results, "long-function") {
		t.Error("did not expect long-function finding for short function")
	}
}

func TestReview_DeepNesting(t *testing.T) {
	src := `package main

func deep() {
	if true {
		if true {
			if true {
				if true {
					if true {
						_ = 1
					}
				}
			}
		}
	}
}
`
	path := writeReviewFile(t, "deep.go", src)
	cr := NewCodeReviewer()

	results, err := cr.ReviewFile(path)
	if err != nil {
		t.Fatalf("ReviewFile: %v", err)
	}

	if !hasRule(results, "deep-nesting") {
		t.Error("expected deep-nesting finding for 5-level nesting")
	}
}

func TestReview_DeepNesting_Shallow(t *testing.T) {
	src := `package main

func shallow() {
	if true {
		if true {
			_ = 1
		}
	}
}
`
	path := writeReviewFile(t, "shallow.go", src)
	cr := NewCodeReviewer()

	results, err := cr.ReviewFile(path)
	if err != nil {
		t.Fatalf("ReviewFile: %v", err)
	}

	if hasRule(results, "deep-nesting") {
		t.Error("did not expect deep-nesting for 2-level nesting")
	}
}

func TestReview_MissingDoc(t *testing.T) {
	src := `package main

func ExportedNoDoc() {}

// Documented is properly documented.
func Documented() {}

type UndocumentedType struct{}

// DocumentedType is properly documented.
type DocumentedType struct{}
`
	path := writeReviewFile(t, "missingdoc.go", src)
	cr := NewCodeReviewer()

	results, err := cr.ReviewFile(path)
	if err != nil {
		t.Fatalf("ReviewFile: %v", err)
	}

	if !hasRule(results, "missing-doc") {
		t.Error("expected missing-doc finding for ExportedNoDoc and UndocumentedType")
	}

	// Check specific messages
	foundFunc := false
	foundType := false
	for _, r := range results {
		if r.Rule == "missing-doc" {
			if strings.Contains(r.Message, "ExportedNoDoc") {
				foundFunc = true
			}
			if strings.Contains(r.Message, "UndocumentedType") {
				foundType = true
			}
		}
	}
	if !foundFunc {
		t.Error("expected missing-doc for ExportedNoDoc")
	}
	if !foundType {
		t.Error("expected missing-doc for UndocumentedType")
	}

	// Should NOT flag Documented or DocumentedType
	for _, r := range results {
		if r.Rule == "missing-doc" && strings.Contains(r.Message, "Documented()") {
			t.Error("should not flag Documented (has doc comment)")
		}
		if r.Rule == "missing-doc" && strings.Contains(r.Message, "DocumentedType") {
			t.Error("should not flag DocumentedType (has doc comment)")
		}
	}
}

func TestReview_MissingDoc_Unexported(t *testing.T) {
	src := `package main

func unexported() {}

type unexportedType struct{}
`
	path := writeReviewFile(t, "unexported.go", src)
	cr := NewCodeReviewer()

	results, err := cr.ReviewFile(path)
	if err != nil {
		t.Fatalf("ReviewFile: %v", err)
	}

	if hasRule(results, "missing-doc") {
		t.Error("did not expect missing-doc for unexported symbols")
	}
}

func TestReviewDir(t *testing.T) {
	dir := t.TempDir()

	// File with issues
	src1 := `package main

// TODO: fix this

func ExportedNoDoc() {}
`
	if err := os.WriteFile(filepath.Join(dir, "a.go"), []byte(src1), 0644); err != nil {
		t.Fatal(err)
	}

	// Clean file
	src2 := `package main

// Clean does nothing wrong.
func Clean() {}
`
	if err := os.WriteFile(filepath.Join(dir, "b.go"), []byte(src2), 0644); err != nil {
		t.Fatal(err)
	}

	// Test file should be skipped
	srcTest := `package main

// TODO: should not appear
func TestSomething() {}
`
	if err := os.WriteFile(filepath.Join(dir, "c_test.go"), []byte(srcTest), 0644); err != nil {
		t.Fatal(err)
	}

	cr := NewCodeReviewer()
	results, err := cr.ReviewDir(dir)
	if err != nil {
		t.Fatalf("ReviewDir: %v", err)
	}

	if !hasRule(results, "todo-comment") {
		t.Error("expected todo-comment from a.go")
	}
	if !hasRule(results, "missing-doc") {
		t.Error("expected missing-doc from a.go")
	}

	// Verify test file was skipped
	for _, r := range results {
		if strings.Contains(r.File, "c_test.go") {
			t.Error("test file c_test.go should have been skipped")
		}
	}
}

func TestReview_ParseError(t *testing.T) {
	path := writeReviewFile(t, "bad.go", "not valid go {{{")
	cr := NewCodeReviewer()

	_, err := cr.ReviewFile(path)
	if err == nil {
		t.Error("expected parse error, got nil")
	}
}

func TestReview_CleanFile(t *testing.T) {
	src := `package main

import "fmt"

// Hello prints a greeting.
func Hello(name string) {
	fmt.Printf("Hello, %s!\n", name)
}
`
	path := writeReviewFile(t, "clean.go", src)
	cr := NewCodeReviewer()

	results, err := cr.ReviewFile(path)
	if err != nil {
		t.Fatalf("ReviewFile: %v", err)
	}

	// Should have no warnings or errors (only possible info-level from missing-doc)
	for _, r := range results {
		if r.Severity == "error" || r.Severity == "warning" {
			t.Errorf("unexpected %s finding in clean file: %s (%s)", r.Severity, r.Message, r.Rule)
		}
	}
}

func TestReview_Severity(t *testing.T) {
	src := `package main

func config() {
	password := "hunter2"
	_ = password
}
`
	path := writeReviewFile(t, "severity.go", src)
	cr := NewCodeReviewer()

	results, err := cr.ReviewFile(path)
	if err != nil {
		t.Fatalf("ReviewFile: %v", err)
	}

	for _, r := range results {
		if r.Rule == "hardcoded-secret" && r.Severity != "error" {
			t.Errorf("hardcoded-secret should have severity 'error', got %q", r.Severity)
		}
	}
}
