package cognitive

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestGenerateDoc_Constructor(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "example.go")
	os.WriteFile(path, []byte(`package example

func NewRouter(addr string) *Router {
	return &Router{addr: addr}
}
`), 0644)

	g := NewAutoDocGenerator()
	doc := g.GenerateDoc(path, "NewRouter")
	if doc == "" {
		t.Fatal("expected doc for NewRouter")
	}
	if !strings.HasPrefix(doc, "// NewRouter") {
		t.Errorf("doc should start with function name, got: %s", doc)
	}
	if !strings.Contains(doc, "creates a new") {
		t.Errorf("constructor doc should contain 'creates a new', got: %s", doc)
	}
	if !strings.Contains(doc, "Router") {
		t.Errorf("constructor doc should mention the type name, got: %s", doc)
	}
}

func TestGenerateDoc_Getter(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "example.go")
	os.WriteFile(path, []byte(`package example

func GetName(id int) string {
	return ""
}
`), 0644)

	g := NewAutoDocGenerator()
	doc := g.GenerateDoc(path, "GetName")
	if doc == "" {
		t.Fatal("expected doc for GetName")
	}
	if !strings.HasPrefix(doc, "// GetName") {
		t.Errorf("doc should start with function name, got: %s", doc)
	}
	if !strings.Contains(doc, "returns the") {
		t.Errorf("getter doc should contain 'returns the', got: %s", doc)
	}
}

func TestGenerateDoc_Handler(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "example.go")
	os.WriteFile(path, []byte(`package example

import "net/http"

func HandleAuth(w http.ResponseWriter, r *http.Request) {
}
`), 0644)

	g := NewAutoDocGenerator()
	doc := g.GenerateDoc(path, "HandleAuth")
	if doc == "" {
		t.Fatal("expected doc for HandleAuth")
	}
	if !strings.HasPrefix(doc, "// HandleAuth") {
		t.Errorf("doc should start with function name, got: %s", doc)
	}
	if !strings.Contains(doc, "handles") {
		t.Errorf("handler doc should contain 'handles', got: %s", doc)
	}
}

func TestGenerateDoc_BoolReturn(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "example.go")
	os.WriteFile(path, []byte(`package example

func IsValid(token string) bool {
	return token != ""
}
`), 0644)

	g := NewAutoDocGenerator()
	doc := g.GenerateDoc(path, "IsValid")
	if doc == "" {
		t.Fatal("expected doc for IsValid")
	}
	if !strings.Contains(doc, "reports whether") {
		t.Errorf("bool func doc should mention 'reports whether', got: %s", doc)
	}
}

func TestGenerateDoc_ErrorReturn(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "example.go")
	os.WriteFile(path, []byte(`package example

func LoadConfig(path string) (*Config, error) {
	return nil, nil
}

type Config struct{}
`), 0644)

	g := NewAutoDocGenerator()
	doc := g.GenerateDoc(path, "LoadConfig")
	if doc == "" {
		t.Fatal("expected doc for LoadConfig")
	}
	if !strings.Contains(doc, "error") {
		t.Errorf("error-returning func doc should mention error, got: %s", doc)
	}
	if !strings.Contains(doc, "loads") {
		t.Errorf("Load* func doc should contain 'loads', got: %s", doc)
	}
}

func TestGenerateDoc_SkipsDocumented(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "example.go")
	os.WriteFile(path, []byte(`package example

// LoadConfig loads the config. Already documented.
func LoadConfig(path string) error {
	return nil
}
`), 0644)

	g := NewAutoDocGenerator()
	doc := g.GenerateDoc(path, "LoadConfig")
	if doc != "" {
		t.Errorf("should skip already-documented function, got: %s", doc)
	}
}

func TestGenerateDoc_Validate(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "example.go")
	os.WriteFile(path, []byte(`package example

func ValidateInput(data string) error {
	return nil
}
`), 0644)

	g := NewAutoDocGenerator()
	doc := g.GenerateDoc(path, "ValidateInput")
	if doc == "" {
		t.Fatal("expected doc for ValidateInput")
	}
	if !strings.Contains(doc, "validates") {
		t.Errorf("Validate* doc should contain 'validates', got: %s", doc)
	}
}

func TestGenerateDocForFile(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "example.go")
	os.WriteFile(path, []byte(`package example

func NewFoo() *Foo {
	return &Foo{}
}

// Bar is already documented.
func Bar() {}

func DeleteItem(id int) error {
	return nil
}

type Foo struct{}
`), 0644)

	g := NewAutoDocGenerator()
	diff := g.GenerateDocForFile(path)
	if diff == "" {
		t.Fatal("expected diff output")
	}
	if !strings.Contains(diff, "NewFoo") {
		t.Error("diff should include NewFoo")
	}
	if !strings.Contains(diff, "DeleteItem") {
		t.Error("diff should include DeleteItem")
	}
	// Bar is documented, should not appear.
	if strings.Contains(diff, "// Bar") {
		t.Error("diff should not include already-documented Bar")
	}
}

func TestDocumentPackage(t *testing.T) {
	dir := t.TempDir()

	// File 1: two undocumented exported functions.
	os.WriteFile(filepath.Join(dir, "a.go"), []byte(`package mypkg

func NewService(addr string) *Service {
	return nil
}

func ParseRequest(data []byte) (*Request, error) {
	return nil, nil
}

type Service struct{}
type Request struct{}
`), 0644)

	// File 2: one undocumented, one documented.
	os.WriteFile(filepath.Join(dir, "b.go"), []byte(`package mypkg

// ReadFile reads a file.
func ReadFile(path string) ([]byte, error) {
	return nil, nil
}

func WriteOutput(path string, data []byte) error {
	return nil
}
`), 0644)

	g := NewAutoDocGenerator()
	count, err := g.DocumentPackage(dir)
	if err != nil {
		t.Fatalf("DocumentPackage: %v", err)
	}
	if count != 3 {
		t.Errorf("expected 3 functions documented, got %d", count)
	}

	// Verify the files were actually modified.
	contentA, _ := os.ReadFile(filepath.Join(dir, "a.go"))
	if !strings.Contains(string(contentA), "// NewService") {
		t.Error("a.go should now contain doc for NewService")
	}
	if !strings.Contains(string(contentA), "// ParseRequest") {
		t.Error("a.go should now contain doc for ParseRequest")
	}

	contentB, _ := os.ReadFile(filepath.Join(dir, "b.go"))
	if !strings.Contains(string(contentB), "// WriteOutput") {
		t.Error("b.go should now contain doc for WriteOutput")
	}
}

func TestGenerateDoc_SkipsUnexported(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "example.go")
	os.WriteFile(path, []byte(`package example

func helperFunc(x int) int {
	return x
}
`), 0644)

	g := NewAutoDocGenerator()
	// GenerateDoc can find unexported by name, but GenerateDocForFile skips them.
	diff := g.GenerateDocForFile(path)
	if diff != "" {
		t.Errorf("should not generate docs for unexported functions, got: %s", diff)
	}
}

func TestSplitCamelCase(t *testing.T) {
	tests := []struct {
		in, want string
	}{
		{"ValidateToken", "validate token"},
		{"HTTP", "h t t p"},
		{"A", "a"},
		{"", ""},
		{"loadConfig", "load config"},
	}
	for _, tt := range tests {
		got := splitCamelCase(tt.in)
		if got != tt.want {
			t.Errorf("splitCamelCase(%q) = %q, want %q", tt.in, got, tt.want)
		}
	}
}
