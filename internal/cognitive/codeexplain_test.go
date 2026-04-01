package cognitive

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func writeTempGo(t *testing.T, name, content string) string {
	t.Helper()
	dir := t.TempDir()
	path := filepath.Join(dir, name)
	if err := os.WriteFile(path, []byte(content), 0644); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestExplainFile_Basic(t *testing.T) {
	src := `package auth

import (
	"net/http"
	"encoding/json"
)

// Claims holds user identity data.
type Claims struct {
	UserID string
	Email  string
}

// ValidateToken validates a JWT token and returns the claims.
func ValidateToken(token string) (*Claims, error) {
	return nil, nil
}

// AuthMiddleware wraps an HTTP handler with authentication.
func AuthMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w)
		next.ServeHTTP(w, r)
	})
}
`
	path := writeTempGo(t, "middleware.go", src)
	ce := NewCodeExplainer()

	result, err := ce.ExplainFile(path)
	if err != nil {
		t.Fatalf("ExplainFile: %v", err)
	}

	// Check key parts are present
	checks := []string{
		"Package: auth",
		"net/http",
		"encoding/json",
		"Claims",
		"ValidateToken",
		"AuthMiddleware",
	}
	for _, check := range checks {
		if !strings.Contains(result, check) {
			t.Errorf("ExplainFile output missing %q\nGot:\n%s", check, result)
		}
	}
}

func TestExplainFile_EmptyFile(t *testing.T) {
	src := `package empty
`
	path := writeTempGo(t, "empty.go", src)
	ce := NewCodeExplainer()

	result, err := ce.ExplainFile(path)
	if err != nil {
		t.Fatalf("ExplainFile: %v", err)
	}

	if !strings.Contains(result, "Package: empty") {
		t.Errorf("expected package name in output, got:\n%s", result)
	}
}

func TestExplainFunction_Patterns(t *testing.T) {
	src := `package main

import (
	"fmt"
	"net/http"
	"encoding/json"
)

// Serve starts the HTTP server with error handling.
func Serve(addr string) error {
	if err := http.ListenAndServe(addr, nil); err != nil {
		return err
	}
	defer fmt.Println("done")
	for _, v := range []string{"a", "b"} {
		fmt.Println(v)
	}
	data, _ := json.Marshal(nil)
	fmt.Println(string(data))
	return nil
}
`
	path := writeTempGo(t, "server.go", src)
	ce := NewCodeExplainer()

	result, err := ce.ExplainFunction(path, "Serve")
	if err != nil {
		t.Fatalf("ExplainFunction: %v", err)
	}

	checks := []string{
		"Serve",
		"error handling",
		"deferred cleanup",
		"collection",
		"Parameters:",
		"addr",
	}
	for _, check := range checks {
		lower := strings.ToLower(result)
		if !strings.Contains(lower, strings.ToLower(check)) {
			t.Errorf("ExplainFunction output missing %q\nGot:\n%s", check, result)
		}
	}
}

func TestExplainFunction_NotFound(t *testing.T) {
	src := `package main

func Hello() {}
`
	path := writeTempGo(t, "hello.go", src)
	ce := NewCodeExplainer()

	_, err := ce.ExplainFunction(path, "Goodbye")
	if err == nil {
		t.Error("expected error for missing function, got nil")
	}
	if !strings.Contains(err.Error(), "not found") {
		t.Errorf("expected 'not found' in error, got: %v", err)
	}
}

func TestExplainFunction_Goroutine(t *testing.T) {
	src := `package main

func Worker() {
	go func() {
		select {}
	}()
}
`
	path := writeTempGo(t, "worker.go", src)
	ce := NewCodeExplainer()

	result, err := ce.ExplainFunction(path, "Worker")
	if err != nil {
		t.Fatalf("ExplainFunction: %v", err)
	}

	if !strings.Contains(strings.ToLower(result), "goroutine") {
		t.Errorf("expected mention of goroutine, got:\n%s", result)
	}
	if !strings.Contains(strings.ToLower(result), "channel selection") {
		t.Errorf("expected mention of channel selection, got:\n%s", result)
	}
}

func TestExplainFunction_HTTP(t *testing.T) {
	src := `package main

import "net/http"

func Handler(w http.ResponseWriter, r *http.Request) {
	http.Error(w, "not found", 404)
}
`
	path := writeTempGo(t, "handler.go", src)
	ce := NewCodeExplainer()

	result, err := ce.ExplainFunction(path, "Handler")
	if err != nil {
		t.Fatalf("ExplainFunction: %v", err)
	}

	if !strings.Contains(strings.ToLower(result), "http") {
		t.Errorf("expected mention of HTTP, got:\n%s", result)
	}
}

func TestExplainLine_Assignment(t *testing.T) {
	src := `package main

func main() {
	x := 42
	_ = x
}
`
	path := writeTempGo(t, "assign.go", src)
	ce := NewCodeExplainer()

	result, err := ce.ExplainLine(path, 4)
	if err != nil {
		t.Fatalf("ExplainLine: %v", err)
	}

	lower := strings.ToLower(result)
	if !strings.Contains(lower, "x") {
		t.Errorf("expected mention of x, got: %s", result)
	}
	if !strings.Contains(lower, "42") || !strings.Contains(lower, "defin") {
		t.Errorf("expected assignment description, got: %s", result)
	}
}

func TestExplainLine_IfStatement(t *testing.T) {
	src := `package main

import "fmt"

func main() {
	err := fmt.Errorf("bad")
	if err != nil {
		fmt.Println(err)
	}
}
`
	path := writeTempGo(t, "ifstmt.go", src)
	ce := NewCodeExplainer()

	result, err := ce.ExplainLine(path, 7)
	if err != nil {
		t.Fatalf("ExplainLine: %v", err)
	}

	lower := strings.ToLower(result)
	if !strings.Contains(lower, "err") {
		t.Errorf("expected mention of err check, got: %s", result)
	}
}

func TestExplainLine_Return(t *testing.T) {
	src := `package main

func add(a, b int) int {
	return a + b
}
`
	path := writeTempGo(t, "ret.go", src)
	ce := NewCodeExplainer()

	result, err := ce.ExplainLine(path, 4)
	if err != nil {
		t.Fatalf("ExplainLine: %v", err)
	}

	if !strings.Contains(strings.ToLower(result), "return") {
		t.Errorf("expected 'return' in output, got: %s", result)
	}
}

func TestExplainLine_ForRange(t *testing.T) {
	src := `package main

import "fmt"

func main() {
	items := []string{"a", "b"}
	for _, v := range items {
		fmt.Println(v)
	}
}
`
	path := writeTempGo(t, "forrange.go", src)
	ce := NewCodeExplainer()

	result, err := ce.ExplainLine(path, 7)
	if err != nil {
		t.Fatalf("ExplainLine: %v", err)
	}

	if !strings.Contains(strings.ToLower(result), "iterat") {
		t.Errorf("expected iteration description, got: %s", result)
	}
}

func TestExplainLine_Comment(t *testing.T) {
	src := `package main

// This is a comment
func main() {}
`
	path := writeTempGo(t, "comment.go", src)
	ce := NewCodeExplainer()

	result, err := ce.ExplainLine(path, 3)
	if err != nil {
		t.Fatalf("ExplainLine: %v", err)
	}

	// Should at least not error; might describe the func or comment
	if result == "" {
		t.Error("expected non-empty explanation for comment line")
	}
}

func TestExplainLine_EmptyLine(t *testing.T) {
	src := `package main

func main() {

}
`
	path := writeTempGo(t, "emptyline.go", src)
	ce := NewCodeExplainer()

	result, err := ce.ExplainLine(path, 4)
	if err != nil {
		t.Fatalf("ExplainLine: %v", err)
	}

	if result == "" {
		t.Error("expected non-empty result for empty line")
	}
}

func TestExplainLine_InvalidLine(t *testing.T) {
	src := `package main

func main() {}
`
	path := writeTempGo(t, "short.go", src)
	ce := NewCodeExplainer()

	_, err := ce.ExplainLine(path, 9999)
	if err == nil {
		t.Error("expected error for line number beyond file, got nil")
	}
}

func TestExplainFile_Types(t *testing.T) {
	src := `package models

// User represents a registered user.
type User struct {
	Name  string
	Email string
}

// Stringer is something that can produce a string.
type Stringer interface {
	String() string
}
`
	path := writeTempGo(t, "models.go", src)
	ce := NewCodeExplainer()

	result, err := ce.ExplainFile(path)
	if err != nil {
		t.Fatalf("ExplainFile: %v", err)
	}

	if !strings.Contains(result, "User") {
		t.Errorf("missing User in output:\n%s", result)
	}
	if !strings.Contains(result, "struct") {
		t.Errorf("missing struct kind in output:\n%s", result)
	}
	if !strings.Contains(result, "Stringer") {
		t.Errorf("missing Stringer in output:\n%s", result)
	}
	if !strings.Contains(result, "interface") {
		t.Errorf("missing interface kind in output:\n%s", result)
	}
}

func TestExplainFunction_Method(t *testing.T) {
	src := `package main

import "fmt"

type Dog struct{ Name string }

func (d *Dog) Bark() string {
	return fmt.Sprintf("Woof! I am %s", d.Name)
}
`
	path := writeTempGo(t, "dog.go", src)
	ce := NewCodeExplainer()

	result, err := ce.ExplainFunction(path, "Bark")
	if err != nil {
		t.Fatalf("ExplainFunction: %v", err)
	}

	if !strings.Contains(result, "Bark") {
		t.Errorf("expected Bark in output, got:\n%s", result)
	}
	if !strings.Contains(result, "Dog") {
		t.Errorf("expected receiver type Dog in output, got:\n%s", result)
	}
}

func TestExplainFile_ParseError(t *testing.T) {
	path := writeTempGo(t, "bad.go", "not valid go code {{{}}")
	ce := NewCodeExplainer()

	_, err := ce.ExplainFile(path)
	if err == nil {
		t.Error("expected parse error, got nil")
	}
}
