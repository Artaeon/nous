package cognitive

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// sampleGoSource is a Go file with various function signatures for testing.
const sampleGoSource = `package demo

import "errors"

// Add adds two integers.
func Add(a, b int) int {
	return a + b
}

// IsValid checks whether an email address is valid.
func IsValid(email string) bool {
	return email != ""
}

// ParseURL parses a raw URL string.
func ParseURL(rawURL string) (*URL, error) {
	if rawURL == "" {
		return nil, errors.New("empty")
	}
	return &URL{}, nil
}

// NewServer constructs a server.
func NewServer(addr string, port int) *Server {
	return &Server{}
}

// ListItems returns a slice of items.
func ListItems(count int) []string {
	return nil
}

// Greet returns a greeting.
func Greet(name string) string {
	return "hello " + name
}

// internal is unexported and should be skipped.
func internal() {}

type URL struct{}
type Server struct{}
`

func TestSmartTestGeneratorBasic(t *testing.T) {
	dir := t.TempDir()
	srcPath := filepath.Join(dir, "demo.go")
	if err := os.WriteFile(srcPath, []byte(sampleGoSource), 0o644); err != nil {
		t.Fatal(err)
	}

	stg := NewSmartTestGenerator()
	out, err := stg.GenerateTests(srcPath)
	if err != nil {
		t.Fatalf("GenerateTests: %v", err)
	}

	// Package name must match source.
	if !strings.HasPrefix(out, "package demo\n") {
		t.Errorf("expected package demo, got first line: %s", strings.SplitN(out, "\n", 2)[0])
	}

	// Must import testing.
	if !strings.Contains(out, `"testing"`) {
		t.Error("missing testing import")
	}

	// Must have test functions for each exported func.
	for _, name := range []string{"TestAdd", "TestIsValid", "TestParseURL", "TestNewServer", "TestListItems", "TestGreet"} {
		if !strings.Contains(out, "func "+name+"(t *testing.T)") {
			t.Errorf("missing test function: %s", name)
		}
	}

	// Must NOT have a test for unexported func.
	if strings.Contains(out, "Testinternal") {
		t.Error("should not generate test for unexported function")
	}
}

func TestSmartTestGeneratorTableDriven(t *testing.T) {
	dir := t.TempDir()
	srcPath := filepath.Join(dir, "demo.go")
	if err := os.WriteFile(srcPath, []byte(sampleGoSource), 0o644); err != nil {
		t.Fatal(err)
	}

	stg := NewSmartTestGenerator()
	out, err := stg.GenerateTests(srcPath)
	if err != nil {
		t.Fatalf("GenerateTests: %v", err)
	}

	// Table-driven structure markers.
	if !strings.Contains(out, "tests := []struct {") {
		t.Error("missing table-driven struct definition")
	}
	if !strings.Contains(out, "t.Run(tt.name") {
		t.Error("missing t.Run subtests")
	}
	if !strings.Contains(out, "for _, tt := range tests") {
		t.Error("missing range loop over test cases")
	}
}

func TestSmartTestGeneratorEdgeCases(t *testing.T) {
	dir := t.TempDir()
	srcPath := filepath.Join(dir, "demo.go")
	if err := os.WriteFile(srcPath, []byte(sampleGoSource), 0o644); err != nil {
		t.Fatal(err)
	}

	stg := NewSmartTestGenerator()
	out, err := stg.GenerateTests(srcPath)
	if err != nil {
		t.Fatalf("GenerateTests: %v", err)
	}

	// Empty string edge case (from string params).
	if !strings.Contains(out, `""`) {
		t.Error("missing empty string edge case")
	}

	// Zero edge case (from int params like port).
	if !strings.Contains(out, "0") {
		t.Error("missing zero edge case")
	}

	// Nil check for pointer returns.
	if !strings.Contains(out, "nil") {
		t.Error("missing nil check for pointer return")
	}

	// wantErr for error returns.
	if !strings.Contains(out, "wantErr") {
		t.Error("missing wantErr field for error return types")
	}
}

func TestSmartTestGeneratorErrorReturn(t *testing.T) {
	src := `package errpkg

import "errors"

func Validate(token string) (bool, error) {
	if token == "" {
		return false, errors.New("empty")
	}
	return true, nil
}
`
	dir := t.TempDir()
	srcPath := filepath.Join(dir, "err.go")
	if err := os.WriteFile(srcPath, []byte(src), 0o644); err != nil {
		t.Fatal(err)
	}

	stg := NewSmartTestGenerator()
	out, err := stg.GenerateTests(srcPath)
	if err != nil {
		t.Fatalf("GenerateTests: %v", err)
	}

	if !strings.HasPrefix(out, "package errpkg\n") {
		t.Error("package name mismatch")
	}
	if !strings.Contains(out, "wantErr") {
		t.Error("missing wantErr for error-returning function")
	}
	if !strings.Contains(out, "TestValidate") {
		t.Error("missing TestValidate function")
	}
}

func TestSmartTestGeneratorPortParam(t *testing.T) {
	src := `package netpkg

func Listen(addr string, port int) error {
	return nil
}
`
	dir := t.TempDir()
	srcPath := filepath.Join(dir, "net.go")
	if err := os.WriteFile(srcPath, []byte(src), 0o644); err != nil {
		t.Fatal(err)
	}

	stg := NewSmartTestGenerator()
	out, err := stg.GenerateTests(srcPath)
	if err != nil {
		t.Fatalf("GenerateTests: %v", err)
	}

	// Port-specific values.
	if !strings.Contains(out, "8080") {
		t.Error("missing port-specific value 8080")
	}
}

func TestSmartTestGeneratorNoExported(t *testing.T) {
	src := `package hidden

func private() {}
func alsoPrivate(x int) int { return x }
`
	dir := t.TempDir()
	srcPath := filepath.Join(dir, "hidden.go")
	if err := os.WriteFile(srcPath, []byte(src), 0o644); err != nil {
		t.Fatal(err)
	}

	stg := NewSmartTestGenerator()
	_, err := stg.GenerateTests(srcPath)
	if err == nil {
		t.Error("expected error for file with no exported functions")
	}
}

func TestSmartTestGeneratorBoolReturn(t *testing.T) {
	src := `package boolpkg

func IsEmpty(s string) bool {
	return s == ""
}
`
	dir := t.TempDir()
	srcPath := filepath.Join(dir, "bool.go")
	if err := os.WriteFile(srcPath, []byte(src), 0o644); err != nil {
		t.Fatal(err)
	}

	stg := NewSmartTestGenerator()
	out, err := stg.GenerateTests(srcPath)
	if err != nil {
		t.Fatalf("GenerateTests: %v", err)
	}

	if !strings.Contains(out, "want bool") || !strings.Contains(out, "want:") {
		t.Error("missing bool expectation in test struct")
	}
}
