package cognitive

import (
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"strings"
	"unicode"
)

// -----------------------------------------------------------------------
// Go Code Generation Templates
//
// Each method on CodeGenerator produces complete, compilable Go source
// code for a common pattern. No ML — pure template expansion with
// idiomatic Go output.
// -----------------------------------------------------------------------

// Route describes an HTTP route for GenerateHTTPServer.
type Route struct {
	Method  string // GET, POST, PUT, DELETE
	Path    string // /api/users
	Handler string // CreateUser
}

// Field describes a struct field for GenerateCRUD and GenerateConfigLoader.
type Field struct {
	Name string
	Type string
}

// CLICommand describes a subcommand for GenerateCLI.
type CLICommand struct {
	Name        string
	Description string
	Flags       []CLIFlag
}

// CLIFlag describes a flag within a CLI command.
type CLIFlag struct {
	Name    string
	Type    string // string, int, bool
	Default string
	Usage   string
}

// GenerateHTTPServer produces a complete main.go with mux routing,
// handler stubs, and server startup.
func (cg *CodeGenerator) GenerateHTTPServer(name string, port int, routes []Route) string {
	var handlers strings.Builder
	var registrations strings.Builder

	for _, r := range routes {
		handlerName := r.Handler + "Handler"

		// Registration line
		registrations.WriteString(fmt.Sprintf("\tmux.HandleFunc(\"%s %s\", %s)\n", r.Method, r.Path, handlerName))

		// Handler function
		statusCode := "http.StatusOK"
		statusJSON := `"ok"`
		if r.Method == "POST" {
			statusCode = "http.StatusCreated"
			statusJSON = `"created"`
		}
		handlers.WriteString(fmt.Sprintf(`
// %s handles %s %s.
func %s(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(%s)
	json.NewEncoder(w).Encode(map[string]string{"status": %s})
}
`, handlerName, r.Method, r.Path, handlerName, statusCode, statusJSON))
	}

	return fmt.Sprintf(`package main

import (
	"encoding/json"
	"log"
	"net/http"
)

func main() {
	mux := http.NewServeMux()

%s
	addr := ":%d"
	log.Printf("%s starting on %%s", addr)
	if err := http.ListenAndServe(addr, mux); err != nil {
		log.Fatal(err)
	}
}
%s`, registrations.String(), port, name, handlers.String())
}

// GenerateCRUD produces complete CRUD handlers (Create, Read, Update,
// Delete, List) for an entity with JSON encoding and in-memory storage.
func (cg *CodeGenerator) GenerateCRUD(entity string, fields []Field) string {
	upper := exportedName(entity)
	lower := unexportedName(entity)

	// Build struct fields
	var structFields strings.Builder
	for _, f := range fields {
		structFields.WriteString(fmt.Sprintf("\t%s %s `json:\"%s\"`\n", exportedName(f.Name), f.Type, strings.ToLower(f.Name)))
	}

	return fmt.Sprintf(`package main

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"sync"
)

// %s is the data model.
type %s struct {
	ID int `+"`json:\"id\"`"+`
%s}

// %sStore holds %s records in memory.
type %sStore struct {
	mu     sync.RWMutex
	items  map[int]%s
	nextID int
}

var store = &%sStore{items: make(map[int]%s), nextID: 1}

func main() {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /%s", List%sHandler)
	mux.HandleFunc("POST /%s", Create%sHandler)
	mux.HandleFunc("GET /%s/{id}", Get%sHandler)
	mux.HandleFunc("PUT /%s/{id}", Update%sHandler)
	mux.HandleFunc("DELETE /%s/{id}", Delete%sHandler)

	log.Printf("%s CRUD API on :8080")
	log.Fatal(http.ListenAndServe(":8080", mux))
}

// Create%sHandler handles POST /%s.
func Create%sHandler(w http.ResponseWriter, r *http.Request) {
	var item %s
	if err := json.NewDecoder(r.Body).Decode(&item); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	store.mu.Lock()
	item.ID = store.nextID
	store.nextID++
	store.items[item.ID] = item
	store.mu.Unlock()

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusCreated)
	json.NewEncoder(w).Encode(item)
}

// List%sHandler handles GET /%s.
func List%sHandler(w http.ResponseWriter, r *http.Request) {
	store.mu.RLock()
	items := make([]%s, 0, len(store.items))
	for _, item := range store.items {
		items = append(items, item)
	}
	store.mu.RUnlock()

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(items)
}

// Get%sHandler handles GET /%s/{id}.
func Get%sHandler(w http.ResponseWriter, r *http.Request) {
	id := 0
	fmt.Sscanf(r.PathValue("id"), "%%d", &id)

	store.mu.RLock()
	item, ok := store.items[id]
	store.mu.RUnlock()

	if !ok {
		http.Error(w, "not found", http.StatusNotFound)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(item)
}

// Update%sHandler handles PUT /%s/{id}.
func Update%sHandler(w http.ResponseWriter, r *http.Request) {
	id := 0
	fmt.Sscanf(r.PathValue("id"), "%%d", &id)

	store.mu.Lock()
	defer store.mu.Unlock()

	if _, ok := store.items[id]; !ok {
		http.Error(w, "not found", http.StatusNotFound)
		return
	}

	var item %s
	if err := json.NewDecoder(r.Body).Decode(&item); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	item.ID = id
	store.items[id] = item

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(item)
}

// Delete%sHandler handles DELETE /%s/{id}.
func Delete%sHandler(w http.ResponseWriter, r *http.Request) {
	id := 0
	fmt.Sscanf(r.PathValue("id"), "%%d", &id)

	store.mu.Lock()
	defer store.mu.Unlock()

	if _, ok := store.items[id]; !ok {
		http.Error(w, "not found", http.StatusNotFound)
		return
	}
	delete(store.items, id)

	w.WriteHeader(http.StatusNoContent)
}
`,
		// struct
		upper, upper, structFields.String(),
		// store
		upper, lower, upper, upper,
		// store var
		upper, upper,
		// routes
		lower, upper,
		lower, upper,
		lower, upper,
		lower, upper,
		lower, upper,
		// log
		lower,
		// Create
		upper, lower, upper, upper,
		// List
		upper, lower, upper, upper,
		// Get
		upper, lower, upper,
		// Update
		upper, lower, upper,
		upper,
		// Delete
		upper, lower, upper,
	)
}

// GenerateCLI produces a complete CLI tool with flag parsing, help text,
// and command routing using only the standard library.
func (cg *CodeGenerator) GenerateCLI(name string, commands []CLICommand) string {
	var commandCases strings.Builder
	var commandFuncs strings.Builder
	var helpLines strings.Builder

	for _, cmd := range commands {
		helpLines.WriteString(fmt.Sprintf("\t\tfmt.Println(\"  %-14s %s\")\n", cmd.Name, cmd.Description))

		// Build flag definitions and parsing
		var flagDefs strings.Builder
		var flagVars strings.Builder
		var flagParseSetup strings.Builder

		flagSetName := cmd.Name + "Flags"
		flagDefs.WriteString(fmt.Sprintf("\t%s := flag.NewFlagSet(\"%s\", flag.ExitOnError)\n", flagSetName, cmd.Name))

		for _, f := range cmd.Flags {
			varName := cmd.Name + exportedName(f.Name)
			switch f.Type {
			case "int":
				flagVars.WriteString(fmt.Sprintf("\t%s := %s.Int(\"%s\", %s, \"%s\")\n",
					varName, flagSetName, f.Name, f.Default, f.Usage))
			case "bool":
				flagVars.WriteString(fmt.Sprintf("\t%s := %s.Bool(\"%s\", %s, \"%s\")\n",
					varName, flagSetName, f.Name, f.Default, f.Usage))
			default: // string
				flagVars.WriteString(fmt.Sprintf("\t%s := %s.String(\"%s\", \"%s\", \"%s\")\n",
					varName, flagSetName, f.Name, f.Default, f.Usage))
			}
		}

		flagParseSetup.WriteString(fmt.Sprintf("\t%s.Parse(args)\n", flagSetName))

		// Build the run function call with flag variable names
		var flagArgs strings.Builder
		for i, f := range cmd.Flags {
			if i > 0 {
				flagArgs.WriteString(", ")
			}
			flagArgs.WriteString(fmt.Sprintf("*%s%s", cmd.Name, exportedName(f.Name)))
		}

		commandCases.WriteString(fmt.Sprintf("\tcase \"%s\":\n", cmd.Name))
		commandCases.WriteString(fmt.Sprintf("%s%s%s", flagDefs.String(), flagVars.String(), flagParseSetup.String()))
		commandCases.WriteString(fmt.Sprintf("\t\trun%s(%s)\n", exportedName(cmd.Name), flagArgs.String()))

		// Build the command function
		var funcParams strings.Builder
		for i, f := range cmd.Flags {
			if i > 0 {
				funcParams.WriteString(", ")
			}
			funcParams.WriteString(fmt.Sprintf("%s %s", f.Name, f.Type))
		}
		commandFuncs.WriteString(fmt.Sprintf(`
// run%s executes the %s command.
func run%s(%s) {
	fmt.Println("running %s")
	// TODO: implement %s logic
}
`, exportedName(cmd.Name), cmd.Name, exportedName(cmd.Name), funcParams.String(), cmd.Name, cmd.Name))
	}

	return fmt.Sprintf(`package main

import (
	"flag"
	"fmt"
	"os"
)

func main() {
	if len(os.Args) < 2 {
		printUsage()
		os.Exit(1)
	}

	command := os.Args[1]
	args := os.Args[2:]

	switch command {
%s	case "help", "-h", "--help":
		printUsage()
	default:
		fmt.Fprintf(os.Stderr, "%s: unknown command %%q\n", command)
		printUsage()
		os.Exit(1)
	}
}

func printUsage() {
	fmt.Println("Usage: %s <command> [flags]")
	fmt.Println()
	fmt.Println("Commands:")
%s	fmt.Println()
	fmt.Println("Use \"%s <command> -h\" for help on a command.")
}
%s`, commandCases.String(), name, name, helpLines.String(), name, commandFuncs.String())
}

// GenerateTestSuite reads a Go source file, finds all exported functions,
// and generates a complete test file with table-driven tests, subtests,
// and benchmark stubs.
func (cg *CodeGenerator) GenerateTestSuite(path string) (string, error) {
	src, err := os.ReadFile(path)
	if err != nil {
		return "", fmt.Errorf("reading source: %w", err)
	}

	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, path, src, parser.ParseComments)
	if err != nil {
		return "", fmt.Errorf("parsing source: %w", err)
	}

	pkgName := f.Name.Name

	type funcInfo struct {
		Name   string
		Params string // "(a int, b string)"
		Return string // "int" or "(int, error)"
	}

	var funcs []funcInfo

	for _, decl := range f.Decls {
		fn, ok := decl.(*ast.FuncDecl)
		if !ok || fn.Recv != nil {
			continue // skip methods and non-functions
		}
		name := fn.Name.Name
		if !unicode.IsUpper(rune(name[0])) {
			continue // skip unexported
		}

		// Extract parameter signature from source
		params := extractFieldList(src, fset, fn.Type.Params)
		returns := ""
		if fn.Type.Results != nil {
			returns = extractFieldList(src, fset, fn.Type.Results)
		}

		funcs = append(funcs, funcInfo{Name: name, Params: params, Return: returns})
	}

	if len(funcs) == 0 {
		return "", fmt.Errorf("no exported functions found in %s", path)
	}

	var tests strings.Builder
	tests.WriteString(fmt.Sprintf("package %s\n\nimport \"testing\"\n", pkgName))

	for _, fn := range funcs {
		tests.WriteString(fmt.Sprintf(`
func Test%s(t *testing.T) {
	tests := []struct {
		name string
		// TODO: add input fields matching %s
		// TODO: add expected output fields
	}{
		{
			name: "basic",
			// TODO: fill in test case
		},
		{
			name: "edge case",
			// TODO: fill in test case
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// TODO: call %s and check results
		})
	}
}
`, fn.Name, fn.Params, fn.Name))
	}

	// Add benchmark stubs for each function
	for _, fn := range funcs {
		tests.WriteString(fmt.Sprintf(`
func Benchmark%s(b *testing.B) {
	// TODO: set up benchmark inputs
	for i := 0; i < b.N; i++ {
		// TODO: call %s
	}
}
`, fn.Name, fn.Name))
	}

	return tests.String(), nil
}

// GenerateWorkerPool produces a complete concurrent worker pool with
// job channel, result channel, graceful shutdown, and error handling.
func (cg *CodeGenerator) GenerateWorkerPool(name string, jobType, resultType string, workers int) string {
	upper := exportedName(name)

	return fmt.Sprintf(`package main

import (
	"context"
	"fmt"
	"log"
	"sync"
)

// %sJob is a unit of work for the pool.
type %sJob struct {
	ID    int
	Input %s
}

// %sResult is the outcome of processing a job.
type %sResult struct {
	JobID int
	Value %s
	Err   error
}

// %sPool manages a fixed number of workers processing jobs concurrently.
type %sPool struct {
	workers int
	jobs    chan %sJob
	results chan %sResult
	wg      sync.WaitGroup
}

// New%sPool creates a worker pool with the given worker count.
func New%sPool(workers int) *%sPool {
	return &%sPool{
		workers: workers,
		jobs:    make(chan %sJob, workers*2),
		results: make(chan %sResult, workers*2),
	}
}

// Start launches workers and returns when all have exited. The context
// can be used for graceful shutdown.
func (p *%sPool) Start(ctx context.Context) {
	for i := 0; i < p.workers; i++ {
		p.wg.Add(1)
		go p.worker(ctx, i)
	}
}

// Submit sends a job to the pool. It blocks if the job buffer is full.
func (p *%sPool) Submit(job %sJob) {
	p.jobs <- job
}

// Close signals that no more jobs will be submitted and waits for all
// workers to finish.
func (p *%sPool) Close() {
	close(p.jobs)
	p.wg.Wait()
	close(p.results)
}

// Results returns the result channel for consumption.
func (p *%sPool) Results() <-chan %sResult {
	return p.results
}

func (p *%sPool) worker(ctx context.Context, id int) {
	defer p.wg.Done()
	for {
		select {
		case <-ctx.Done():
			log.Printf("worker %%d: shutting down", id)
			return
		case job, ok := <-p.jobs:
			if !ok {
				return // channel closed
			}
			result := process%s(job)
			p.results <- result
		}
	}
}

// process%s handles a single job. Edit this to add your logic.
func process%s(job %sJob) %sResult {
	// TODO: implement actual processing logic
	return %sResult{
		JobID: job.ID,
		Value: *new(%s), // zero value placeholder
		Err:   nil,
	}
}

func main() {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	pool := New%sPool(%d)
	pool.Start(ctx)

	// Submit jobs
	go func() {
		for i := 0; i < 20; i++ {
			pool.Submit(%sJob{
				ID:    i,
				Input: *new(%s), // TODO: real input
			})
		}
		pool.Close()
	}()

	// Collect results
	for result := range pool.Results() {
		if result.Err != nil {
			log.Printf("job %%d failed: %%v", result.JobID, result.Err)
			continue
		}
		fmt.Printf("job %%d: %%v\n", result.JobID, result.Value)
	}
}
`,
		// Job type
		upper, upper, jobType,
		// Result type
		upper, upper, resultType,
		// Pool struct
		upper, upper, upper, upper,
		// Constructor
		upper, upper, upper, upper, upper, upper,
		// Start
		upper,
		// Submit
		upper, upper,
		// Close
		upper,
		// Results
		upper, upper,
		// worker
		upper,
		upper,
		// process func
		upper, upper, upper, upper, upper, resultType,
		// main
		upper, workers,
		upper, jobType,
	)
}

// GenerateMiddleware produces composable HTTP middleware with logging,
// recovery, CORS, auth check, and rate limiting.
func (cg *CodeGenerator) GenerateMiddleware(name string, checks []string) string {
	upper := exportedName(name)

	// Build middleware chain call
	var chain strings.Builder
	chain.WriteString("handler := http.HandlerFunc(appHandler)\n")

	// Always include recovery and logging; add requested checks
	allMiddleware := []string{"LoggingMiddleware", "RecoveryMiddleware"}
	for _, c := range checks {
		switch strings.ToLower(c) {
		case "cors":
			allMiddleware = append(allMiddleware, "CORSMiddleware")
		case "auth":
			allMiddleware = append(allMiddleware, "AuthMiddleware")
		case "ratelimit", "rate_limit", "rate-limit":
			allMiddleware = append(allMiddleware, "RateLimitMiddleware")
		}
	}

	// Apply in reverse so outermost is first in list
	for i := len(allMiddleware) - 1; i >= 0; i-- {
		chain.WriteString(fmt.Sprintf("\thandler = %s(handler).(http.HandlerFunc)\n", allMiddleware[i]))
	}

	return fmt.Sprintf(`package main

import (
	"log"
	"net/http"
	"sync"
	"time"
)

// Middleware is a function that wraps an http.Handler.
type Middleware func(http.Handler) http.Handler

// Chain composes multiple middleware into a single wrapper. The first
// middleware in the list is the outermost (runs first).
func Chain(middlewares ...Middleware) Middleware {
	return func(final http.Handler) http.Handler {
		for i := len(middlewares) - 1; i >= 0; i-- {
			final = middlewares[i](final)
		}
		return final
	}
}

// LoggingMiddleware logs each request's method, path, and duration.
func LoggingMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		next.ServeHTTP(w, r)
		log.Printf("%%s %%s %%s", r.Method, r.URL.Path, time.Since(start))
	})
}

// RecoveryMiddleware recovers from panics and returns 500.
func RecoveryMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		defer func() {
			if err := recover(); err != nil {
				log.Printf("panic recovered: %%v", err)
				http.Error(w, "Internal Server Error", http.StatusInternalServerError)
			}
		}()
		next.ServeHTTP(w, r)
	})
}

// CORSMiddleware adds permissive CORS headers.
func CORSMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization")
		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusNoContent)
			return
		}
		next.ServeHTTP(w, r)
	})
}

// AuthMiddleware rejects requests without a valid Authorization header.
func AuthMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		token := r.Header.Get("Authorization")
		if token == "" {
			http.Error(w, "Unauthorized", http.StatusUnauthorized)
			return
		}
		// TODO: validate token
		next.ServeHTTP(w, r)
	})
}

// RateLimitMiddleware limits requests to one per second per IP.
func RateLimitMiddleware(next http.Handler) http.Handler {
	var mu sync.Mutex
	visitors := make(map[string]time.Time)

	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		ip := r.RemoteAddr

		mu.Lock()
		last, exists := visitors[ip]
		if exists && time.Since(last) < time.Second {
			mu.Unlock()
			http.Error(w, "Too Many Requests", http.StatusTooManyRequests)
			return
		}
		visitors[ip] = time.Now()
		mu.Unlock()

		next.ServeHTTP(w, r)
	})
}

// appHandler is the final application handler.
func appHandler(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	w.Write([]byte(` + "`" + `{"service":"` + upper + `","status":"ok"}` + "`" + `))
}

func main() {
	%s
	log.Printf("%s middleware server on :8080")
	log.Fatal(http.ListenAndServe(":8080", handler))
}
`, chain.String(), name)
}

// GenerateConfigLoader produces a config struct with JSON/env loading,
// validation, defaults, and a NewConfig constructor.
func (cg *CodeGenerator) GenerateConfigLoader(name string, fields []Field) string {
	upper := exportedName(name)

	// Build struct fields
	var structFields strings.Builder
	var defaults strings.Builder
	var envOverrides strings.Builder
	var validation strings.Builder

	for _, f := range fields {
		fname := exportedName(f.Name)
		envKey := strings.ToUpper(name) + "_" + strings.ToUpper(f.Name)
		jsonTag := strings.ToLower(f.Name)

		structFields.WriteString(fmt.Sprintf("\t%s %s `json:\"%s\"`\n", fname, f.Type, jsonTag))

		// Default values
		switch f.Type {
		case "string":
			defaults.WriteString(fmt.Sprintf("\t\t%s: \"\",\n", fname))
			envOverrides.WriteString(fmt.Sprintf("\tif v := os.Getenv(\"%s\"); v != \"\" {\n\t\tcfg.%s = v\n\t}\n", envKey, fname))
		case "int":
			defaults.WriteString(fmt.Sprintf("\t\t%s: 0,\n", fname))
			envOverrides.WriteString(fmt.Sprintf("\tif v := os.Getenv(\"%s\"); v != \"\" {\n\t\tif n, err := strconv.Atoi(v); err == nil {\n\t\t\tcfg.%s = n\n\t\t}\n\t}\n", envKey, fname))
		case "bool":
			defaults.WriteString(fmt.Sprintf("\t\t%s: false,\n", fname))
			envOverrides.WriteString(fmt.Sprintf("\tif v := os.Getenv(\"%s\"); v != \"\" {\n\t\tcfg.%s = v == \"true\" || v == \"1\"\n\t}\n", envKey, fname))
		default:
			defaults.WriteString(fmt.Sprintf("\t\t// %s: set default for %s\n", fname, f.Type))
		}

		// Validation for string fields — require non-empty
		if f.Type == "string" {
			validation.WriteString(fmt.Sprintf("\tif cfg.%s == \"\" {\n\t\terrs = append(errs, \"%s is required\")\n\t}\n", fname, fname))
		}
	}

	return fmt.Sprintf(`package main

import (
	"encoding/json"
	"fmt"
	"os"
	"strconv"
	"strings"
)

// %s holds application configuration.
type %s struct {
%s}

// New%s loads configuration from a JSON file, then applies environment
// variable overrides, and finally validates the result.
func New%s(path string) (*%s, error) {
	cfg := &%s{
%s	}

	// Load from JSON file if it exists
	if path != "" {
		data, err := os.ReadFile(path)
		if err != nil && !os.IsNotExist(err) {
			return nil, fmt.Errorf("reading config: %%w", err)
		}
		if err == nil {
			if err := json.Unmarshal(data, cfg); err != nil {
				return nil, fmt.Errorf("parsing config: %%w", err)
			}
		}
	}

	// Override from environment variables
%s
	// Validate
	if errs := cfg.Validate(); len(errs) > 0 {
		return nil, fmt.Errorf("config validation: %%s", strings.Join(errs, "; "))
	}

	return cfg, nil
}

// Validate checks that required fields are set.
func (cfg *%s) Validate() []string {
	var errs []string
%s	return errs
}

func main() {
	cfg, err := New%s("config.json")
	if err != nil {
		fmt.Fprintf(os.Stderr, "config error: %%v\n", err)
		os.Exit(1)
	}
	data, _ := json.MarshalIndent(cfg, "", "  ")
	fmt.Println(string(data))

	// suppress unused import warnings
	_ = strconv.Atoi
}
`,
		upper, upper, structFields.String(),
		upper, upper, upper, upper, defaults.String(),
		envOverrides.String(),
		upper, validation.String(),
		upper,
	)
}

// -----------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------

// exportedName upper-cases the first rune of s.
func exportedName(s string) string {
	if s == "" {
		return s
	}
	runes := []rune(s)
	runes[0] = unicode.ToUpper(runes[0])
	return string(runes)
}

// unexportedName lower-cases the first rune of s.
func unexportedName(s string) string {
	if s == "" {
		return s
	}
	runes := []rune(s)
	runes[0] = unicode.ToLower(runes[0])
	return string(runes)
}

// extractFieldList returns the source text of a field list (params or results).
func extractFieldList(src []byte, fset *token.FileSet, fl *ast.FieldList) string {
	if fl == nil || len(fl.List) == 0 {
		return ""
	}
	start := fset.Position(fl.Opening)
	end := fset.Position(fl.Closing)
	if start.IsValid() && end.IsValid() {
		// include parentheses
		return string(src[start.Offset : end.Offset+1])
	}
	// fall back to just the type names
	var parts []string
	for _, field := range fl.List {
		typStr := string(src[fset.Position(field.Type.Pos()).Offset:fset.Position(field.Type.End()).Offset])
		for range field.Names {
			parts = append(parts, typStr)
		}
		if len(field.Names) == 0 {
			parts = append(parts, typStr)
		}
	}
	return strings.Join(parts, ", ")
}
