package cognitive

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func writeTempGoFiles(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()

	// main.go: main calls SetupRouter and Run.
	os.WriteFile(filepath.Join(dir, "main.go"), []byte(`package myapp

func main() {
	r := SetupRouter()
	Run(r)
}

func Run(r *Router) {
	r.Start()
}
`), 0644)

	// router.go: SetupRouter calls AuthMiddleware and HandleIndex.
	os.WriteFile(filepath.Join(dir, "router.go"), []byte(`package myapp

type Router struct{}

func SetupRouter() *Router {
	AuthMiddleware()
	HandleIndex()
	return &Router{}
}

func (r *Router) Start() {}
`), 0644)

	// middleware.go: AuthMiddleware calls ValidateToken and ExtractClaims.
	os.WriteFile(filepath.Join(dir, "middleware.go"), []byte(`package myapp

func AuthMiddleware() {
	ValidateToken("abc")
	ExtractClaims("abc")
}

func ValidateToken(token string) bool {
	return token != ""
}

func ExtractClaims(token string) map[string]string {
	return nil
}

func HandleIndex() {}
`), 0644)

	return dir
}

func TestDepGraph_Build(t *testing.T) {
	dir := writeTempGoFiles(t)
	dg := NewDepGraph()
	if err := dg.Build(dir); err != nil {
		t.Fatalf("Build: %v", err)
	}

	// Should have recorded packages.
	if len(dg.Packages) == 0 {
		t.Error("expected at least one package")
	}
	fns, ok := dg.Packages["myapp"]
	if !ok {
		t.Fatal("expected 'myapp' package")
	}
	if len(fns) == 0 {
		t.Error("expected functions in myapp package")
	}
}

func TestDepGraph_WhoCalls(t *testing.T) {
	dir := writeTempGoFiles(t)
	dg := NewDepGraph()
	if err := dg.Build(dir); err != nil {
		t.Fatalf("Build: %v", err)
	}

	// ValidateToken is called by AuthMiddleware.
	callers := dg.WhoCalls("ValidateToken")
	if len(callers) == 0 {
		t.Fatal("expected callers for ValidateToken")
	}
	found := false
	for _, c := range callers {
		if strings.Contains(c, "AuthMiddleware") {
			found = true
		}
	}
	if !found {
		t.Errorf("expected AuthMiddleware in callers of ValidateToken, got %v", callers)
	}
}

func TestDepGraph_WhatCalls(t *testing.T) {
	dir := writeTempGoFiles(t)
	dg := NewDepGraph()
	if err := dg.Build(dir); err != nil {
		t.Fatalf("Build: %v", err)
	}

	// AuthMiddleware calls ValidateToken and ExtractClaims.
	callees := dg.WhatCalls("AuthMiddleware")
	if len(callees) == 0 {
		t.Fatal("expected callees for AuthMiddleware")
	}

	hasValidate := false
	hasClaims := false
	for _, c := range callees {
		if strings.Contains(c, "ValidateToken") {
			hasValidate = true
		}
		if strings.Contains(c, "ExtractClaims") {
			hasClaims = true
		}
	}
	if !hasValidate {
		t.Errorf("expected ValidateToken in callees, got %v", callees)
	}
	if !hasClaims {
		t.Errorf("expected ExtractClaims in callees, got %v", callees)
	}
}

func TestDepGraph_Impact(t *testing.T) {
	dir := writeTempGoFiles(t)
	dg := NewDepGraph()
	if err := dg.Build(dir); err != nil {
		t.Fatalf("Build: %v", err)
	}

	// If ValidateToken changes, AuthMiddleware and SetupRouter and main
	// should all be in the impact set (transitive callers).
	impact := dg.Impact("ValidateToken")
	if len(impact) == 0 {
		t.Fatal("expected impact for ValidateToken")
	}

	hasAuth := false
	hasSetup := false
	hasMain := false
	for _, f := range impact {
		if strings.Contains(f, "AuthMiddleware") {
			hasAuth = true
		}
		if strings.Contains(f, "SetupRouter") {
			hasSetup = true
		}
		if strings.Contains(f, "main") {
			hasMain = true
		}
	}
	if !hasAuth {
		t.Errorf("expected AuthMiddleware in impact, got %v", impact)
	}
	if !hasSetup {
		t.Errorf("expected SetupRouter in impact, got %v", impact)
	}
	if !hasMain {
		t.Errorf("expected main in impact, got %v", impact)
	}
}

func TestDepGraph_Render(t *testing.T) {
	dir := writeTempGoFiles(t)
	dg := NewDepGraph()
	if err := dg.Build(dir); err != nil {
		t.Fatalf("Build: %v", err)
	}

	out := dg.Render("AuthMiddleware", 2)
	if out == "" {
		t.Fatal("expected render output")
	}

	// Should contain the function name.
	if !strings.Contains(out, "AuthMiddleware") {
		t.Error("render should contain AuthMiddleware")
	}
	// Should show calls.
	if !strings.Contains(out, "calls:") {
		t.Error("render should show calls")
	}
	// Should show called by.
	if !strings.Contains(out, "called by:") {
		t.Error("render should show called by")
	}
	// With depth 2, should show impact.
	if !strings.Contains(out, "impact:") {
		t.Error("render with depth 2 should show impact")
	}
}

func TestDepGraph_RenderNotFound(t *testing.T) {
	dg := NewDepGraph()
	out := dg.Render("NonExistent", 1)
	if !strings.Contains(out, "not found") {
		t.Errorf("expected 'not found' for unknown function, got: %s", out)
	}
}

func TestDepGraph_EmptyGraph(t *testing.T) {
	dg := NewDepGraph()
	callers := dg.WhoCalls("Foo")
	if callers != nil {
		t.Errorf("expected nil for empty graph, got %v", callers)
	}
	callees := dg.WhatCalls("Foo")
	if callees != nil {
		t.Errorf("expected nil for empty graph, got %v", callees)
	}
	impact := dg.Impact("Foo")
	if impact != nil {
		t.Errorf("expected nil impact for empty graph, got %v", impact)
	}
}
