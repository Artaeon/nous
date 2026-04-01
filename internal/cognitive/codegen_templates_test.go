package cognitive

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestGenerateHTTPServer_Basic(t *testing.T) {
	cg := NewCodeGenerator()
	routes := []Route{
		{Method: "GET", Path: "/api/users", Handler: "ListUsers"},
		{Method: "POST", Path: "/api/users", Handler: "CreateUser"},
		{Method: "DELETE", Path: "/api/users/{id}", Handler: "DeleteUser"},
	}

	out := cg.GenerateHTTPServer("myapi", 8080, routes)

	for _, want := range []string{
		"package main",
		"http.ListenAndServe",
		":8080",
		"ListUsersHandler",
		"CreateUserHandler",
		"DeleteUserHandler",
		"net/http",
		"encoding/json",
		"mux.HandleFunc",
		"myapi starting on",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("output missing %q", want)
		}
	}

	if out == "" {
		t.Fatal("output is empty")
	}
}

func TestGenerateHTTPServer_StatusCodes(t *testing.T) {
	cg := NewCodeGenerator()
	routes := []Route{
		{Method: "POST", Path: "/items", Handler: "CreateItem"},
	}

	out := cg.GenerateHTTPServer("svc", 9090, routes)

	if !strings.Contains(out, "http.StatusCreated") {
		t.Error("POST handler should use StatusCreated")
	}
}

func TestGenerateCRUD_Basic(t *testing.T) {
	cg := NewCodeGenerator()
	fields := []Field{
		{Name: "name", Type: "string"},
		{Name: "email", Type: "string"},
		{Name: "age", Type: "int"},
	}

	out := cg.GenerateCRUD("user", fields)

	for _, want := range []string{
		"package main",
		"CreateUserHandler",
		"ListUserHandler",
		"GetUserHandler",
		"UpdateUserHandler",
		"DeleteUserHandler",
		`json:"name"`,
		`json:"email"`,
		`json:"age"`,
		"sync.RWMutex",
		"http.StatusCreated",
		"http.StatusNotFound",
		"http.StatusNoContent",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("CRUD output missing %q", want)
		}
	}

	if out == "" {
		t.Fatal("output is empty")
	}
}

func TestGenerateCLI_Basic(t *testing.T) {
	cg := NewCodeGenerator()
	commands := []CLICommand{
		{
			Name:        "serve",
			Description: "Start the server",
			Flags: []CLIFlag{
				{Name: "port", Type: "int", Default: "8080", Usage: "Port to listen on"},
				{Name: "host", Type: "string", Default: "localhost", Usage: "Host to bind to"},
			},
		},
		{
			Name:        "version",
			Description: "Print version",
			Flags:       nil,
		},
	}

	out := cg.GenerateCLI("mytool", commands)

	for _, want := range []string{
		"package main",
		"flag",
		"os.Args",
		"runServe",
		"runVersion",
		"Start the server",
		"Print version",
		`"port"`,
		`"host"`,
		"mytool",
		"printUsage",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("CLI output missing %q", want)
		}
	}

	if out == "" {
		t.Fatal("output is empty")
	}
}

func TestGenerateTestSuite_RealFile(t *testing.T) {
	// Write a small Go file to a temp dir so we have a known target.
	dir := t.TempDir()
	src := `package example

func Add(a, b int) int { return a + b }
func Multiply(x, y int) int { return x * y }
func helper() {} // unexported, should be skipped
`
	path := filepath.Join(dir, "math.go")
	if err := os.WriteFile(path, []byte(src), 0644); err != nil {
		t.Fatal(err)
	}

	cg := NewCodeGenerator()
	out, err := cg.GenerateTestSuite(path)
	if err != nil {
		t.Fatalf("GenerateTestSuite: %v", err)
	}

	for _, want := range []string{
		"package example",
		"TestAdd",
		"TestMultiply",
		"BenchmarkAdd",
		"BenchmarkMultiply",
		"t.Run",
		"b.N",
		`"testing"`,
	} {
		if !strings.Contains(out, want) {
			t.Errorf("test suite output missing %q", want)
		}
	}

	// Must NOT contain the unexported helper
	if strings.Contains(out, "Testhelper") || strings.Contains(out, "TestHelper") {
		t.Error("test suite should skip unexported functions")
	}

	if out == "" {
		t.Fatal("output is empty")
	}
}

func TestGenerateTestSuite_NoExportedFuncs(t *testing.T) {
	dir := t.TempDir()
	src := `package example

func helper() {}
`
	path := filepath.Join(dir, "internal.go")
	if err := os.WriteFile(path, []byte(src), 0644); err != nil {
		t.Fatal(err)
	}

	cg := NewCodeGenerator()
	_, err := cg.GenerateTestSuite(path)
	if err == nil {
		t.Error("expected error for file with no exported functions")
	}
}

func TestGenerateTestSuite_MissingFile(t *testing.T) {
	cg := NewCodeGenerator()
	_, err := cg.GenerateTestSuite("/nonexistent/path.go")
	if err == nil {
		t.Error("expected error for missing file")
	}
}

func TestGenerateWorkerPool_Basic(t *testing.T) {
	cg := NewCodeGenerator()
	out := cg.GenerateWorkerPool("indexer", "string", "int", 4)

	for _, want := range []string{
		"package main",
		"IndexerJob",
		"IndexerResult",
		"IndexerPool",
		"NewIndexerPool",
		"chan IndexerJob",
		"chan IndexerResult",
		"sync.WaitGroup",
		"context.Context",
		"go p.worker",
		"close(p.jobs)",
		"close(p.results)",
		"processIndexer",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("worker pool output missing %q", want)
		}
	}

	if !strings.Contains(out, "4") {
		t.Error("worker count not found in output")
	}

	if out == "" {
		t.Fatal("output is empty")
	}
}

func TestGenerateMiddleware_Basic(t *testing.T) {
	cg := NewCodeGenerator()
	out := cg.GenerateMiddleware("myapp", []string{"cors", "auth", "ratelimit"})

	for _, want := range []string{
		"package main",
		"LoggingMiddleware",
		"RecoveryMiddleware",
		"CORSMiddleware",
		"AuthMiddleware",
		"RateLimitMiddleware",
		"Chain",
		"http.Handler",
		"time.Since",
		"recover()",
		"Access-Control-Allow-Origin",
		"Authorization",
		"StatusTooManyRequests",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("middleware output missing %q", want)
		}
	}

	if out == "" {
		t.Fatal("output is empty")
	}
}

func TestGenerateMiddleware_MinimalChecks(t *testing.T) {
	cg := NewCodeGenerator()
	out := cg.GenerateMiddleware("plain", nil)

	// Should always have logging and recovery even with no checks
	if !strings.Contains(out, "LoggingMiddleware") {
		t.Error("should always include LoggingMiddleware")
	}
	if !strings.Contains(out, "RecoveryMiddleware") {
		t.Error("should always include RecoveryMiddleware")
	}
}

func TestGenerateConfigLoader_Basic(t *testing.T) {
	cg := NewCodeGenerator()
	fields := []Field{
		{Name: "host", Type: "string"},
		{Name: "port", Type: "int"},
		{Name: "debug", Type: "bool"},
	}

	out := cg.GenerateConfigLoader("app", fields)

	for _, want := range []string{
		"package main",
		"type App struct",
		"NewApp",
		"json.Unmarshal",
		"os.Getenv",
		"Validate",
		"APP_HOST",
		"APP_PORT",
		"APP_DEBUG",
		`json:"host"`,
		`json:"port"`,
		`json:"debug"`,
		"strconv",
		"os.ReadFile",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("config loader output missing %q", want)
		}
	}

	if out == "" {
		t.Fatal("output is empty")
	}
}

func TestAllGeneratorsNonEmpty(t *testing.T) {
	cg := NewCodeGenerator()

	tests := []struct {
		name   string
		output string
	}{
		{"HTTPServer", cg.GenerateHTTPServer("s", 80, []Route{{Method: "GET", Path: "/", Handler: "Index"}})},
		{"CRUD", cg.GenerateCRUD("item", []Field{{Name: "name", Type: "string"}})},
		{"CLI", cg.GenerateCLI("cli", []CLICommand{{Name: "run", Description: "run it"}})},
		{"WorkerPool", cg.GenerateWorkerPool("w", "int", "int", 2)},
		{"Middleware", cg.GenerateMiddleware("m", []string{"cors"})},
		{"ConfigLoader", cg.GenerateConfigLoader("c", []Field{{Name: "key", Type: "string"}})},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.output == "" {
				t.Errorf("%s produced empty output", tt.name)
			}
			if !strings.Contains(tt.output, "package main") {
				t.Errorf("%s missing 'package main'", tt.name)
			}
		})
	}
}
