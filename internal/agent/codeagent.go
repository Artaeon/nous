package agent

import (
	"bytes"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"
)

// CodeAgent is an autonomous coding agent that generates complete Go
// projects from natural language descriptions. No LLM — uses template-based
// generation with smart entity extraction and proper Go idioms.
type CodeAgent struct {
	OutputDir string
	Verbose   bool
}

// CodePlan describes what the agent will build.
type CodePlan struct {
	ProjectName string        `json:"project_name"`
	Description string        `json:"description"`
	Type        string        `json:"type"` // "api", "cli", "library", "worker"
	Entities    []Entity      `json:"entities"`
	Storage     string        `json:"storage"` // "memory", "file"
	Features    []string      `json:"features"`
	Files       []PlannedFile `json:"files"`
}

// Entity is a data model extracted from the description.
type Entity struct {
	Name   string  `json:"name"`
	Fields []Field `json:"fields"`
}

// Field is a struct field.
type Field struct {
	Name string `json:"name"`
	Type string `json:"type"`
	JSON string `json:"json"`
}

// PlannedFile describes a file to be generated.
type PlannedFile struct {
	Path        string `json:"path"`
	Description string `json:"description"`
}

// BuildResult is the outcome of the coding agent.
type BuildResult struct {
	Plan        *CodePlan     `json:"plan"`
	Files       []string      `json:"files"`
	Compiled    bool          `json:"compiled"`
	CompileErr  string        `json:"compile_error,omitempty"`
	TestsPassed int           `json:"tests_passed"`
	TestsFailed int           `json:"tests_failed"`
	Duration    time.Duration `json:"duration"`
}

// NewCodeAgent creates a coding agent that writes to the given directory.
func NewCodeAgent(outputDir string) *CodeAgent {
	return &CodeAgent{OutputDir: outputDir, Verbose: true}
}

// Build takes a natural language description and generates a working Go project.
func (ca *CodeAgent) Build(description string) (*BuildResult, error) {
	start := time.Now()
	result := &BuildResult{}

	// Step 1: Parse the request
	ca.log("Planning...")
	plan := ca.ParseRequest(description)
	result.Plan = plan
	ca.log("  Project: %s (%s)", plan.ProjectName, plan.Type)
	for _, e := range plan.Entities {
		ca.log("  Entity: %s (%d fields)", e.Name, len(e.Fields))
	}
	ca.log("  Storage: %s", plan.Storage)
	ca.log("")

	// Step 2: Create output directory
	if err := os.MkdirAll(ca.OutputDir, 0755); err != nil {
		return nil, fmt.Errorf("create dir: %w", err)
	}

	// Step 3: Generate files based on project type
	ca.log("Generating code...")
	var files []string
	var err error

	switch plan.Type {
	case "api":
		files, err = ca.generateAPI(plan)
	case "cli":
		files, err = ca.generateCLI(plan)
	case "library":
		files, err = ca.generateLibrary(plan)
	case "worker":
		files, err = ca.generateWorker(plan)
	default:
		files, err = ca.generateAPI(plan) // default to API
	}
	if err != nil {
		return nil, fmt.Errorf("generate: %w", err)
	}

	result.Files = files
	for _, f := range files {
		ca.log("  Created: %s", f)
	}
	ca.log("")

	// Step 4: Generate go.mod
	modPath := filepath.Join(ca.OutputDir, "go.mod")
	modContent := fmt.Sprintf("module %s\n\ngo 1.22\n", plan.ProjectName)
	os.WriteFile(modPath, []byte(modContent), 0644)

	// Step 5: Compile
	ca.log("Compiling...")
	compiled, compileOut := ca.compile()
	result.Compiled = compiled
	if !compiled {
		result.CompileErr = compileOut
		ca.log("  FAILED: %s", compileOut)
		// Try auto-fix
		ca.log("  Attempting auto-fix...")
		ca.autoFix()
		compiled, compileOut = ca.compile()
		result.Compiled = compiled
		if compiled {
			ca.log("  Fixed and compiled successfully")
		} else {
			result.CompileErr = compileOut
			ca.log("  Still failing: %s", compileOut)
		}
	} else {
		ca.log("  ok")
	}

	// Step 6: Run tests
	if result.Compiled {
		ca.log("Running tests...")
		passed, failed, testOut := ca.runTests()
		result.TestsPassed = passed
		result.TestsFailed = failed
		if failed > 0 {
			ca.log("  %d passed, %d failed", passed, failed)
			ca.log("  %s", testOut)
		} else if passed > 0 {
			ca.log("  %d/%d passed", passed, passed)
		} else {
			ca.log("  no tests found")
		}
	}

	result.Duration = time.Since(start)
	ca.log("")
	ca.log("Project ready in %s/ (%s)", ca.OutputDir, result.Duration.Round(time.Millisecond))

	return result, nil
}

func (ca *CodeAgent) log(format string, args ...interface{}) {
	if ca.Verbose {
		fmt.Printf(format+"\n", args...)
	}
}

// -----------------------------------------------------------------------
// Request Parsing
// -----------------------------------------------------------------------

// ParseRequest parses a natural language description into a code plan.
func (ca *CodeAgent) ParseRequest(desc string) *CodePlan {
	lower := strings.ToLower(desc)
	plan := &CodePlan{Description: desc}

	// Detect project type
	switch {
	case containsAnyWord(lower, "rest api", "http server", "web server", "api server", "web api", "api"):
		plan.Type = "api"
	case containsAnyWord(lower, "cli", "command line", "terminal", "console"):
		plan.Type = "cli"
	case containsAnyWord(lower, "library", "package", "module", "pkg"):
		plan.Type = "library"
	case containsAnyWord(lower, "worker", "background", "queue", "job", "processor"):
		plan.Type = "worker"
	default:
		plan.Type = "api"
	}

	// Detect storage
	switch {
	case containsAnyWord(lower, "sqlite", "database", "postgres", "mysql", "db"):
		plan.Storage = "database"
	case containsAnyWord(lower, "file", "json file", "disk"):
		plan.Storage = "file"
	default:
		plan.Storage = "memory"
	}

	// Extract entities
	plan.Entities = extractEntities(lower)

	// Generate project name
	if len(plan.Entities) > 0 {
		plan.ProjectName = strings.ToLower(plan.Entities[0].Name) + "-" + plan.Type
	} else {
		plan.ProjectName = "myproject"
	}

	// Detect features
	if containsAnyWord(lower, "auth", "authentication", "jwt", "login") {
		plan.Features = append(plan.Features, "auth")
	}
	if containsAnyWord(lower, "cors") {
		plan.Features = append(plan.Features, "cors")
	}
	if containsAnyWord(lower, "logging", "log", "logs") {
		plan.Features = append(plan.Features, "logging")
	}

	return plan
}

// Known entity templates with sensible default fields.
var knownEntities = map[string][]Field{
	"todo":    {{Name: "Title", Type: "string", JSON: "title"}, {Name: "Completed", Type: "bool", JSON: "completed"}},
	"todos":   {{Name: "Title", Type: "string", JSON: "title"}, {Name: "Completed", Type: "bool", JSON: "completed"}},
	"user":    {{Name: "Name", Type: "string", JSON: "name"}, {Name: "Email", Type: "string", JSON: "email"}},
	"users":   {{Name: "Name", Type: "string", JSON: "name"}, {Name: "Email", Type: "string", JSON: "email"}},
	"product": {{Name: "Name", Type: "string", JSON: "name"}, {Name: "Price", Type: "float64", JSON: "price"}, {Name: "Description", Type: "string", JSON: "description"}},
	"products": {{Name: "Name", Type: "string", JSON: "name"}, {Name: "Price", Type: "float64", JSON: "price"}, {Name: "Description", Type: "string", JSON: "description"}},
	"article": {{Name: "Title", Type: "string", JSON: "title"}, {Name: "Body", Type: "string", JSON: "body"}, {Name: "Author", Type: "string", JSON: "author"}},
	"articles": {{Name: "Title", Type: "string", JSON: "title"}, {Name: "Body", Type: "string", JSON: "body"}, {Name: "Author", Type: "string", JSON: "author"}},
	"post":    {{Name: "Title", Type: "string", JSON: "title"}, {Name: "Content", Type: "string", JSON: "content"}, {Name: "Author", Type: "string", JSON: "author"}},
	"posts":   {{Name: "Title", Type: "string", JSON: "title"}, {Name: "Content", Type: "string", JSON: "content"}, {Name: "Author", Type: "string", JSON: "author"}},
	"task":    {{Name: "Title", Type: "string", JSON: "title"}, {Name: "Done", Type: "bool", JSON: "done"}, {Name: "Priority", Type: "int", JSON: "priority"}},
	"tasks":   {{Name: "Title", Type: "string", JSON: "title"}, {Name: "Done", Type: "bool", JSON: "done"}, {Name: "Priority", Type: "int", JSON: "priority"}},
	"book":    {{Name: "Title", Type: "string", JSON: "title"}, {Name: "Author", Type: "string", JSON: "author"}, {Name: "Year", Type: "int", JSON: "year"}},
	"books":   {{Name: "Title", Type: "string", JSON: "title"}, {Name: "Author", Type: "string", JSON: "author"}, {Name: "Year", Type: "int", JSON: "year"}},
	"note":    {{Name: "Title", Type: "string", JSON: "title"}, {Name: "Body", Type: "string", JSON: "body"}},
	"notes":   {{Name: "Title", Type: "string", JSON: "title"}, {Name: "Body", Type: "string", JSON: "body"}},
	"event":   {{Name: "Name", Type: "string", JSON: "name"}, {Name: "Date", Type: "string", JSON: "date"}, {Name: "Location", Type: "string", JSON: "location"}},
	"events":  {{Name: "Name", Type: "string", JSON: "name"}, {Name: "Date", Type: "string", JSON: "date"}, {Name: "Location", Type: "string", JSON: "location"}},
	"image":   {{Name: "URL", Type: "string", JSON: "url"}, {Name: "Width", Type: "int", JSON: "width"}, {Name: "Height", Type: "int", JSON: "height"}},
	"images":  {{Name: "URL", Type: "string", JSON: "url"}, {Name: "Width", Type: "int", JSON: "width"}, {Name: "Height", Type: "int", JSON: "height"}},
}

func extractEntities(desc string) []Entity {
	// Look for "for managing X", "for X", "X manager", "X service"
	patterns := []string{
		"for managing ", "for ", "manage ", " manager",
	}

	for _, p := range patterns {
		idx := strings.Index(desc, p)
		if idx < 0 {
			continue
		}
		after := desc[idx+len(p):]
		// Take first word (the entity name)
		word := strings.Fields(after)[0]
		word = strings.Trim(word, ".,;:!?\"'")

		if fields, ok := knownEntities[word]; ok {
			name := strings.Title(strings.TrimSuffix(word, "s"))
			return []Entity{{Name: name, Fields: fields}}
		}
	}

	// Default entity
	return []Entity{{
		Name: "Item",
		Fields: []Field{
			{Name: "Name", Type: "string", JSON: "name"},
			{Name: "Value", Type: "string", JSON: "value"},
		},
	}}
}

func containsAnyWord(s string, words ...string) bool {
	for _, w := range words {
		if strings.Contains(s, w) {
			return true
		}
	}
	return false
}

// -----------------------------------------------------------------------
// API Generation
// -----------------------------------------------------------------------

func (ca *CodeAgent) generateAPI(plan *CodePlan) ([]string, error) {
	entity := plan.Entities[0]
	lower := strings.ToLower(entity.Name)
	var files []string

	// model.go
	if err := ca.writeFile("model.go", ca.genModel(entity)); err != nil {
		return nil, err
	}
	files = append(files, "model.go")

	// store.go
	if err := ca.writeFile("store.go", ca.genStore(entity)); err != nil {
		return nil, err
	}
	files = append(files, "store.go")

	// handlers.go
	if err := ca.writeFile("handlers.go", ca.genHandlers(entity)); err != nil {
		return nil, err
	}
	files = append(files, "handlers.go")

	// main.go
	if err := ca.writeFile("main.go", ca.genMain(entity, plan)); err != nil {
		return nil, err
	}
	files = append(files, "main.go")

	// handlers_test.go
	if err := ca.writeFile("handlers_test.go", ca.genTests(entity)); err != nil {
		return nil, err
	}
	files = append(files, "handlers_test.go")

	_ = lower
	return files, nil
}

func (ca *CodeAgent) genModel(e Entity) string {
	var sb strings.Builder
	sb.WriteString("package main\n\nimport \"time\"\n\n")
	sb.WriteString(fmt.Sprintf("// %s represents a %s in the system.\n", e.Name, strings.ToLower(e.Name)))
	sb.WriteString(fmt.Sprintf("type %s struct {\n", e.Name))
	sb.WriteString("\tID        int       `json:\"id\"`\n")
	for _, f := range e.Fields {
		sb.WriteString(fmt.Sprintf("\t%s %s `json:\"%s\"`\n", f.Name, f.Type, f.JSON))
	}
	sb.WriteString("\tCreatedAt time.Time `json:\"created_at\"`\n")
	sb.WriteString("\tUpdatedAt time.Time `json:\"updated_at\"`\n")
	sb.WriteString("}\n")
	return sb.String()
}

func (ca *CodeAgent) genStore(e Entity) string {
	name := e.Name
	lower := strings.ToLower(name)
	return fmt.Sprintf(`package main

import (
	"fmt"
	"sync"
	"time"
)

// %sStore provides thread-safe in-memory storage for %ss.
type %sStore struct {
	mu     sync.RWMutex
	items  map[int]*%s
	nextID int
}

// New%sStore creates a new empty store.
func New%sStore() *%sStore {
	return &%sStore{
		items:  make(map[int]*%s),
		nextID: 1,
	}
}

// Create adds a new %s and returns it with an assigned ID.
func (s *%sStore) Create(item *%s) *%s {
	s.mu.Lock()
	defer s.mu.Unlock()
	item.ID = s.nextID
	s.nextID++
	item.CreatedAt = time.Now()
	item.UpdatedAt = item.CreatedAt
	s.items[item.ID] = item
	return item
}

// Get returns a %s by ID, or an error if not found.
func (s *%sStore) Get(id int) (*%s, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	item, ok := s.items[id]
	if !ok {
		return nil, fmt.Errorf("%s %%d not found", id)
	}
	return item, nil
}

// List returns all %ss.
func (s *%sStore) List() []*%s {
	s.mu.RLock()
	defer s.mu.RUnlock()
	result := make([]*%s, 0, len(s.items))
	for _, item := range s.items {
		result = append(result, item)
	}
	return result
}

// Update replaces a %s by ID.
func (s *%sStore) Update(id int, updated *%s) (*%s, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, ok := s.items[id]; !ok {
		return nil, fmt.Errorf("%s %%d not found", id)
	}
	updated.ID = id
	updated.UpdatedAt = time.Now()
	s.items[id] = updated
	return updated, nil
}

// Delete removes a %s by ID.
func (s *%sStore) Delete(id int) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, ok := s.items[id]; !ok {
		return fmt.Errorf("%s %%d not found", id)
	}
	delete(s.items, id)
	return nil
}
`,
		name, lower,
		name, name,
		name, name, name, name, name,
		lower, name, name, name,
		lower, name, name, lower,
		lower, name, name, name,
		lower, name, name, name, lower,
		lower, name, lower,
	)
}

func (ca *CodeAgent) genHandlers(e Entity) string {
	name := e.Name
	lower := strings.ToLower(name)
	firstField := "Name"
	if len(e.Fields) > 0 {
		firstField = e.Fields[0].Name
	}
	firstJSON := strings.ToLower(firstField)

	return fmt.Sprintf(`package main

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strconv"
	"strings"
)

// Server holds the application dependencies.
type Server struct {
	store *%sStore
}

// NewServer creates a new server with an empty store.
func NewServer() *Server {
	return &Server{store: New%sStore()}
}

// List%ss handles GET /api/%ss — returns all %ss.
func (s *Server) List%ss(w http.ResponseWriter, r *http.Request) {
	items := s.store.List()
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(items)
}

// Create%s handles POST /api/%ss — creates a new %s.
func (s *Server) Create%s(w http.ResponseWriter, r *http.Request) {
	var item %s
	if err := json.NewDecoder(r.Body).Decode(&item); err != nil {
		http.Error(w, `+"`"+`{"error":"invalid JSON"}`+"`"+`, http.StatusBadRequest)
		return
	}
	if item.%s == "" {
		http.Error(w, `+"`"+`{"error":"%s is required"}`+"`"+`, http.StatusBadRequest)
		return
	}
	created := s.store.Create(&item)
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusCreated)
	json.NewEncoder(w).Encode(created)
}

// Get%s handles GET /api/%ss/{id} — returns a single %s.
func (s *Server) Get%s(w http.ResponseWriter, r *http.Request) {
	id, err := extractID(r.URL.Path)
	if err != nil {
		http.Error(w, `+"`"+`{"error":"invalid id"}`+"`"+`, http.StatusBadRequest)
		return
	}
	item, err := s.store.Get(id)
	if err != nil {
		http.Error(w, fmt.Sprintf(`+"`"+`{"error":"%%s"}`+"`"+`, err), http.StatusNotFound)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(item)
}

// Update%s handles PUT /api/%ss/{id} — updates a %s.
func (s *Server) Update%s(w http.ResponseWriter, r *http.Request) {
	id, err := extractID(r.URL.Path)
	if err != nil {
		http.Error(w, `+"`"+`{"error":"invalid id"}`+"`"+`, http.StatusBadRequest)
		return
	}
	var item %s
	if err := json.NewDecoder(r.Body).Decode(&item); err != nil {
		http.Error(w, `+"`"+`{"error":"invalid JSON"}`+"`"+`, http.StatusBadRequest)
		return
	}
	updated, err := s.store.Update(id, &item)
	if err != nil {
		http.Error(w, fmt.Sprintf(`+"`"+`{"error":"%%s"}`+"`"+`, err), http.StatusNotFound)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(updated)
}

// Delete%s handles DELETE /api/%ss/{id} — removes a %s.
func (s *Server) Delete%s(w http.ResponseWriter, r *http.Request) {
	id, err := extractID(r.URL.Path)
	if err != nil {
		http.Error(w, `+"`"+`{"error":"invalid id"}`+"`"+`, http.StatusBadRequest)
		return
	}
	if err := s.store.Delete(id); err != nil {
		http.Error(w, fmt.Sprintf(`+"`"+`{"error":"%%s"}`+"`"+`, err), http.StatusNotFound)
		return
	}
	w.WriteHeader(http.StatusNoContent)
}

func extractID(path string) (int, error) {
	parts := strings.Split(strings.TrimRight(path, "/"), "/")
	if len(parts) == 0 {
		return 0, fmt.Errorf("no id")
	}
	return strconv.Atoi(parts[len(parts)-1])
}
`,
		name, name,
		name, lower, lower, name,
		name, lower, lower, name, name,
		firstField, firstJSON,
		name, lower, lower, name,
		name, lower, lower, name, name,
		name, lower, lower, name,
	)
}

func (ca *CodeAgent) genMain(e Entity, plan *CodePlan) string {
	lower := strings.ToLower(e.Name)
	return fmt.Sprintf(`package main

import (
	"fmt"
	"log"
	"net/http"
	"strings"
)

func main() {
	srv := NewServer()
	mux := http.NewServeMux()

	// Route: /api/%ss
	mux.HandleFunc("/api/%ss", func(w http.ResponseWriter, r *http.Request) {
		switch r.Method {
		case http.MethodGet:
			srv.List%ss(w, r)
		case http.MethodPost:
			srv.Create%s(w, r)
		default:
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		}
	})

	// Route: /api/%ss/{id}
	mux.HandleFunc("/api/%ss/", func(w http.ResponseWriter, r *http.Request) {
		// Only handle paths with an ID segment
		parts := strings.Split(strings.TrimRight(r.URL.Path, "/"), "/")
		if len(parts) < 4 {
			http.NotFound(w, r)
			return
		}
		switch r.Method {
		case http.MethodGet:
			srv.Get%s(w, r)
		case http.MethodPut:
			srv.Update%s(w, r)
		case http.MethodDelete:
			srv.Delete%s(w, r)
		default:
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		}
	})

	addr := ":8080"
	fmt.Printf("%s API server starting on %%s\n", addr)
	log.Fatal(http.ListenAndServe(addr, mux))
}
`,
		lower, lower, e.Name, e.Name,
		lower, lower, e.Name, e.Name, e.Name,
		plan.ProjectName,
	)
}

func (ca *CodeAgent) genTests(e Entity) string {
	name := e.Name
	lower := strings.ToLower(name)
	firstField := "Name"
	firstJSON := "name"
	if len(e.Fields) > 0 {
		firstField = e.Fields[0].Name
		firstJSON = e.Fields[0].JSON
	}

	return fmt.Sprintf(`package main

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestCreate%s(t *testing.T) {
	srv := NewServer()
	body := []byte(`+"`"+`{"%s": "Test %s"}`+"`"+`)
	req := httptest.NewRequest(http.MethodPost, "/api/%ss", bytes.NewReader(body))
	w := httptest.NewRecorder()
	srv.Create%s(w, req)

	if w.Code != http.StatusCreated {
		t.Errorf("Create: got status %%d, want %%d", w.Code, http.StatusCreated)
	}

	var created %s
	json.NewDecoder(w.Body).Decode(&created)
	if created.ID != 1 {
		t.Errorf("ID: got %%d, want 1", created.ID)
	}
	if created.%s != "Test %s" {
		t.Errorf("%s: got %%q, want %%q", created.%s, "Test %s")
	}
}

func TestList%ss(t *testing.T) {
	srv := NewServer()
	// Create two items
	srv.store.Create(&%s{%s: "First"})
	srv.store.Create(&%s{%s: "Second"})

	req := httptest.NewRequest(http.MethodGet, "/api/%ss", nil)
	w := httptest.NewRecorder()
	srv.List%ss(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("List: got status %%d, want %%d", w.Code, http.StatusOK)
	}

	var items []%s
	json.NewDecoder(w.Body).Decode(&items)
	if len(items) != 2 {
		t.Errorf("List: got %%d items, want 2", len(items))
	}
}

func TestGet%s(t *testing.T) {
	srv := NewServer()
	srv.store.Create(&%s{%s: "FindMe"})

	req := httptest.NewRequest(http.MethodGet, "/api/%ss/1", nil)
	w := httptest.NewRecorder()
	srv.Get%s(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("Get: got status %%d, want %%d", w.Code, http.StatusOK)
	}

	var item %s
	json.NewDecoder(w.Body).Decode(&item)
	if item.%s != "FindMe" {
		t.Errorf("%s: got %%q, want %%q", item.%s, "FindMe")
	}
}

func TestGet%sNotFound(t *testing.T) {
	srv := NewServer()
	req := httptest.NewRequest(http.MethodGet, "/api/%ss/999", nil)
	w := httptest.NewRecorder()
	srv.Get%s(w, req)

	if w.Code != http.StatusNotFound {
		t.Errorf("GetNotFound: got status %%d, want %%d", w.Code, http.StatusNotFound)
	}
}

func TestDelete%s(t *testing.T) {
	srv := NewServer()
	srv.store.Create(&%s{%s: "DeleteMe"})

	req := httptest.NewRequest(http.MethodDelete, "/api/%ss/1", nil)
	w := httptest.NewRecorder()
	srv.Delete%s(w, req)

	if w.Code != http.StatusNoContent {
		t.Errorf("Delete: got status %%d, want %%d", w.Code, http.StatusNoContent)
	}

	// Verify deleted
	_, err := srv.store.Get(1)
	if err == nil {
		t.Error("expected error after delete, got nil")
	}
}

func TestCreate%sInvalidJSON(t *testing.T) {
	srv := NewServer()
	req := httptest.NewRequest(http.MethodPost, "/api/%ss", bytes.NewReader([]byte("not json")))
	w := httptest.NewRecorder()
	srv.Create%s(w, req)

	if w.Code != http.StatusBadRequest {
		t.Errorf("InvalidJSON: got status %%d, want %%d", w.Code, http.StatusBadRequest)
	}
}

func TestCreate%sMissing%s(t *testing.T) {
	srv := NewServer()
	req := httptest.NewRequest(http.MethodPost, "/api/%ss", bytes.NewReader([]byte("{}")))
	w := httptest.NewRecorder()
	srv.Create%s(w, req)

	if w.Code != http.StatusBadRequest {
		t.Errorf("Missing%s: got status %%d, want %%d", w.Code, http.StatusBadRequest)
	}
}
`,
		name, firstJSON, lower, lower, name, name,
		firstField, lower, firstField, firstField, lower,
		name, name, firstField, name, firstField, lower, name,
		name,
		name, name, firstField, lower, name, name,
		firstField, firstField, firstField,
		name, lower, name,
		name, name, firstField, lower, name,
		name, lower, name,
		name, firstField, lower, name, firstField,
	)
}

// -----------------------------------------------------------------------
// CLI Generation
// -----------------------------------------------------------------------

func (ca *CodeAgent) generateCLI(plan *CodePlan) ([]string, error) {
	name := plan.ProjectName
	content := fmt.Sprintf(`package main

import (
	"flag"
	"fmt"
	"os"
)

func main() {
	if len(os.Args) < 2 {
		fmt.Println("Usage: %s <command> [flags]")
		fmt.Println("Commands: help, version")
		os.Exit(1)
	}

	switch os.Args[1] {
	case "help":
		fmt.Println("%s - a command line tool")
		fmt.Println("Commands:")
		fmt.Println("  help     Show this help")
		fmt.Println("  version  Show version")
	case "version":
		fmt.Println("%s v1.0.0")
	default:
		fmt.Fprintf(os.Stderr, "unknown command: %%s\n", os.Args[1])
		os.Exit(1)
	}
	_ = flag.NewFlagSet("", flag.ExitOnError) // suppress unused import
}
`, name, name, name)

	if err := ca.writeFile("main.go", content); err != nil {
		return nil, err
	}
	return []string{"main.go"}, nil
}

// -----------------------------------------------------------------------
// Library Generation
// -----------------------------------------------------------------------

func (ca *CodeAgent) generateLibrary(plan *CodePlan) ([]string, error) {
	entity := plan.Entities[0]
	lower := strings.ToLower(entity.Name)

	content := fmt.Sprintf(`package %s

// %s represents the core type.
type %s struct {
`, lower, entity.Name, entity.Name)

	for _, f := range entity.Fields {
		content += fmt.Sprintf("\t%s %s\n", f.Name, f.Type)
	}
	content += "}\n"

	if err := ca.writeFile(lower+".go", content); err != nil {
		return nil, err
	}
	return []string{lower + ".go"}, nil
}

// -----------------------------------------------------------------------
// Worker Generation
// -----------------------------------------------------------------------

func (ca *CodeAgent) generateWorker(plan *CodePlan) ([]string, error) {
	content := `package main

import (
	"context"
	"fmt"
	"sync"
	"time"
)

type Job struct {
	ID   int
	Data string
}

type Result struct {
	JobID  int
	Output string
}

func worker(ctx context.Context, id int, jobs <-chan Job, results chan<- Result, wg *sync.WaitGroup) {
	defer wg.Done()
	for {
		select {
		case <-ctx.Done():
			return
		case job, ok := <-jobs:
			if !ok {
				return
			}
			// Process the job
			output := fmt.Sprintf("worker %d processed job %d: %s", id, job.ID, job.Data)
			results <- Result{JobID: job.ID, Output: output}
		}
	}
}

func main() {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	jobs := make(chan Job, 100)
	results := make(chan Result, 100)
	var wg sync.WaitGroup

	// Start workers
	numWorkers := 4
	for i := 0; i < numWorkers; i++ {
		wg.Add(1)
		go worker(ctx, i, jobs, results, &wg)
	}

	// Send jobs
	go func() {
		for i := 0; i < 10; i++ {
			jobs <- Job{ID: i, Data: fmt.Sprintf("task-%d", i)}
		}
		close(jobs)
	}()

	// Collect results
	go func() {
		wg.Wait()
		close(results)
	}()

	for r := range results {
		fmt.Println(r.Output)
	}
}
`
	if err := ca.writeFile("main.go", content); err != nil {
		return nil, err
	}
	return []string{"main.go"}, nil
}

// -----------------------------------------------------------------------
// Build Helpers
// -----------------------------------------------------------------------

func (ca *CodeAgent) writeFile(name, content string) error {
	path := filepath.Join(ca.OutputDir, name)
	return os.WriteFile(path, []byte(content), 0644)
}

func (ca *CodeAgent) compile() (bool, string) {
	cmd := exec.Command("go", "build", "./...")
	cmd.Dir = ca.OutputDir
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	err := cmd.Run()
	if err != nil {
		return false, strings.TrimSpace(stderr.String())
	}
	return true, ""
}

func (ca *CodeAgent) autoFix() {
	// Run goimports-like fix: gofmt
	cmd := exec.Command("gofmt", "-w", ".")
	cmd.Dir = ca.OutputDir
	cmd.Run()
}

func (ca *CodeAgent) runTests() (passed, failed int, output string) {
	cmd := exec.Command("go", "test", "-v", "-count=1", "./...")
	cmd.Dir = ca.OutputDir
	var stdout bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stdout
	cmd.Run()

	out := stdout.String()
	for _, line := range strings.Split(out, "\n") {
		if strings.HasPrefix(line, "--- PASS") {
			passed++
		} else if strings.HasPrefix(line, "--- FAIL") {
			failed++
		}
	}
	return passed, failed, out
}
