package cognitive

import (
	"strings"
	"testing"
)

func TestExplainHTTPHandler(t *testing.T) {
	ce := NewCodeExplainer()
	src := `package main

import (
	"encoding/json"
	"log"
	"net/http"
)

func handleHealth(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
}

func main() {
	http.HandleFunc("/health", handleHealth)
	log.Fatal(http.ListenAndServe(":8080", nil))
}
`
	results, err := ce.ExplainSource(src)
	if err != nil {
		t.Fatalf("ExplainSource: %v", err)
	}
	if len(results) != 2 {
		t.Fatalf("expected 2 functions, got %d", len(results))
	}

	// handleHealth should mention HTTP headers and JSON
	handler := results[0]
	if handler.Name != "handleHealth" {
		t.Errorf("expected handleHealth, got %s", handler.Name)
	}
	found := strings.Join(handler.Patterns, " | ")
	if !strings.Contains(found, "HTTP") && !strings.Contains(found, "header") {
		t.Errorf("handleHealth patterns should mention HTTP headers, got: %s", found)
	}
	if !strings.Contains(found, "JSON") && !strings.Contains(found, "json") {
		t.Errorf("handleHealth patterns should mention JSON, got: %s", found)
	}
	if handler.Summary == "" {
		t.Error("handleHealth summary is empty")
	}

	// main should mention HTTP server
	main := results[1]
	if main.Name != "main" {
		t.Errorf("expected main, got %s", main.Name)
	}
	mainFound := strings.Join(main.Patterns, " | ")
	if !strings.Contains(mainFound, "HTTP") {
		t.Errorf("main patterns should mention HTTP, got: %s", mainFound)
	}
}

func TestExplainRangeLoop(t *testing.T) {
	ce := NewCodeExplainer()
	src := `package main

import "fmt"

func processLayers(layers []Layer) {
	for _, layer := range layers {
		layer.Forward()
		fmt.Println(layer.Name)
	}
}
`
	// Note: This won't fully parse since Layer is undefined, but
	// parser.ParseFile in lenient mode should still work
	results, err := ce.ExplainSource(src)
	if err != nil {
		t.Fatalf("ExplainSource: %v", err)
	}
	if len(results) != 1 {
		t.Fatalf("expected 1 function, got %d", len(results))
	}
	found := strings.Join(results[0].Patterns, " | ")
	if !strings.Contains(found, "layer") && !strings.Contains(found, "Layer") {
		t.Errorf("should mention layers, got: %s", found)
	}
}

func TestExplainErrorHandling(t *testing.T) {
	ce := NewCodeExplainer()
	src := `package main

import "os"

func readConfig(path string) ([]byte, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	return data, nil
}
`
	results, err := ce.ExplainSource(src)
	if err != nil {
		t.Fatalf("ExplainSource: %v", err)
	}
	if len(results) != 1 {
		t.Fatalf("expected 1 function, got %d", len(results))
	}
	found := strings.Join(results[0].Patterns, " | ")
	// Should detect os.ReadFile and error handling
	if !strings.Contains(found, "file") && !strings.Contains(found, "File") {
		t.Errorf("should mention reading file, got: %s", found)
	}
	if !strings.Contains(strings.ToLower(found), "error") && !strings.Contains(found, "err") {
		t.Errorf("should mention error handling, got: %s", found)
	}
}

func TestExplainConcurrency(t *testing.T) {
	ce := NewCodeExplainer()
	src := `package main

import "sync"

func processAll(items []string) {
	var wg sync.WaitGroup
	for _, item := range items {
		wg.Add(1)
		go func(s string) {
			defer wg.Done()
			process(s)
		}(item)
	}
	wg.Wait()
}
`
	results, err := ce.ExplainSource(src)
	if err != nil {
		t.Fatalf("ExplainSource: %v", err)
	}
	if len(results) != 1 {
		t.Fatalf("expected 1 function, got %d", len(results))
	}
	found := strings.Join(results[0].Patterns, " | ")
	if !strings.Contains(found, "goroutine") {
		t.Errorf("should mention goroutine, got: %s", found)
	}
}

func TestExplainKnownFunctionsCount(t *testing.T) {
	if len(knownFunctions) < 150 {
		t.Errorf("knownFunctions should have 150+ entries, got %d", len(knownFunctions))
	}
}

func TestExplainSummaryGeneration(t *testing.T) {
	ce := NewCodeExplainer()
	src := `package main

import (
	"database/sql"
	"log"
)

func fetchUser(db *sql.DB, id int) (string, error) {
	var name string
	err := db.QueryRow("SELECT name FROM users WHERE id = ?", id).Scan(&name)
	if err != nil {
		log.Printf("failed to fetch user %d: %v", id, err)
		return "", err
	}
	return name, nil
}
`
	results, err := ce.ExplainSource(src)
	if err != nil {
		t.Fatalf("ExplainSource: %v", err)
	}
	if len(results) != 1 {
		t.Fatalf("expected 1 function, got %d", len(results))
	}
	r := results[0]
	if r.Summary == "" {
		t.Error("summary is empty")
	}
	// Summary should be a sentence, starting with the function name
	if !strings.HasPrefix(r.Summary, "fetchUser") {
		t.Errorf("summary should start with function name, got: %s", r.Summary)
	}
	// Should contain descriptive text, not just the function name
	if len(r.Summary) < 20 {
		t.Errorf("summary too short: %s", r.Summary)
	}
}

func TestExplainPurposeFromName(t *testing.T) {
	ce := NewCodeExplainer()
	tests := []struct {
		name string
		want string // substring that should appear
	}{
		{"handleRequest", "handles"},
		{"processItems", "processes"},
		{"validateInput", "validates"},
		{"fetchData", "fetches"},
		{"loadConfig", "loads"},
	}
	for _, tt := range tests {
		purpose := ce.purposeFromName(tt.name)
		if !strings.Contains(purpose, tt.want) {
			t.Errorf("purposeFromName(%q) = %q, want to contain %q", tt.name, purpose, tt.want)
		}
	}
}

func TestExplainCamelToWords(t *testing.T) {
	tests := []struct {
		input string
		want  string
	}{
		{"CamelCase", "Camel Case"},
		{"HTTPHandler", "HTTP Handler"},
		{"getID", "get ID"},
		{"", ""},
		{"simple", "simple"},
	}
	for _, tt := range tests {
		got := camelToWords(tt.input)
		if got != tt.want {
			t.Errorf("camelToWords(%q) = %q, want %q", tt.input, got, tt.want)
		}
	}
}

func TestExplainCompositeHTTPServer(t *testing.T) {
	ce := NewCodeExplainer()
	src := `package main

import (
	"log"
	"net/http"
)

func main() {
	http.HandleFunc("/", handleIndex)
	http.HandleFunc("/health", handleHealth)
	log.Fatal(http.ListenAndServe(":8080", nil))
}
`
	results, err := ce.ExplainSource(src)
	if err != nil {
		t.Fatalf("ExplainSource: %v", err)
	}
	if len(results) != 1 {
		t.Fatalf("expected 1 function, got %d", len(results))
	}
	r := results[0]
	compositeStr := strings.Join(r.Composite, " | ")
	if !strings.Contains(compositeStr, "HTTP server") {
		t.Errorf("should detect HTTP server composite pattern, got: %v", r.Composite)
	}
}

func TestExplainCompositeMutex(t *testing.T) {
	ce := NewCodeExplainer()
	src := `package main

import "sync"

var mu sync.Mutex
var count int

func increment() {
	mu.Lock()
	defer mu.Unlock()
	count++
}
`
	results, err := ce.ExplainSource(src)
	if err != nil {
		t.Fatalf("ExplainSource: %v", err)
	}
	if len(results) != 1 {
		t.Fatalf("expected 1 function, got %d", len(results))
	}
	compositeStr := strings.Join(results[0].Composite, " | ")
	if !strings.Contains(compositeStr, "mutex") && !strings.Contains(compositeStr, "Thread-safe") {
		t.Errorf("should detect mutex composite, got: %v", results[0].Composite)
	}
}

func TestExplainCompositeTimeoutContext(t *testing.T) {
	ce := NewCodeExplainer()
	src := `package main

import (
	"context"
	"time"
)

func doWork() {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	_ = ctx
}
`
	results, err := ce.ExplainSource(src)
	if err != nil {
		t.Fatalf("ExplainSource: %v", err)
	}
	if len(results) != 1 {
		t.Fatalf("expected 1 function, got %d", len(results))
	}
	compositeStr := strings.Join(results[0].Composite, " | ")
	if !strings.Contains(compositeStr, "timeout") && !strings.Contains(compositeStr, "context") {
		t.Errorf("should detect timeout context composite, got: %v", results[0].Composite)
	}
}

func TestExplainDataFlowBasic(t *testing.T) {
	ce := NewCodeExplainer()
	src := `package main

import "os"

func readConfig(path string) ([]byte, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	return data, nil
}
`
	results, err := ce.ExplainSource(src)
	if err != nil {
		t.Fatalf("ExplainSource: %v", err)
	}
	if len(results) != 1 {
		t.Fatalf("expected 1 function, got %d", len(results))
	}
	r := results[0]
	if r.DataFlow == "" {
		t.Error("DataFlow should not be empty for readConfig")
	}
	// Should mention "Takes" and "returns"
	if !strings.Contains(r.DataFlow, "Takes") {
		t.Errorf("DataFlow should start with 'Takes', got: %s", r.DataFlow)
	}
	if !strings.Contains(r.DataFlow, "returns") {
		t.Errorf("DataFlow should mention returns, got: %s", r.DataFlow)
	}
}

func TestExplainDataFlowWithTransform(t *testing.T) {
	ce := NewCodeExplainer()
	src := `package main

import (
	"encoding/json"
	"os"
)

func loadJSON(path string) (map[string]interface{}, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var result map[string]interface{}
	err = json.Unmarshal(data, &result)
	if err != nil {
		return nil, err
	}
	return result, nil
}
`
	results, err := ce.ExplainSource(src)
	if err != nil {
		t.Fatalf("ExplainSource: %v", err)
	}
	if len(results) != 1 {
		t.Fatalf("expected 1 function, got %d", len(results))
	}
	r := results[0]
	if r.DataFlow == "" {
		t.Error("DataFlow should describe the transformation pipeline")
	}
	// Should mention the file reading and JSON parsing steps
	lower := strings.ToLower(r.DataFlow)
	if !strings.Contains(lower, "file") && !strings.Contains(lower, "read") {
		t.Errorf("DataFlow should mention file reading, got: %s", r.DataFlow)
	}
}

func TestExplainCompositeWaitGroup(t *testing.T) {
	ce := NewCodeExplainer()
	src := `package main

import "sync"

func processAll(items []string) {
	var wg sync.WaitGroup
	for _, item := range items {
		wg.Add(1)
		go func(s string) {
			defer wg.Done()
			process(s)
		}(item)
	}
	wg.Wait()
}
`
	results, err := ce.ExplainSource(src)
	if err != nil {
		t.Fatalf("ExplainSource: %v", err)
	}
	if len(results) != 1 {
		t.Fatalf("expected 1 function, got %d", len(results))
	}
	compositeStr := strings.Join(results[0].Composite, " | ")
	if !strings.Contains(compositeStr, "WaitGroup") && !strings.Contains(compositeStr, "concurrent") {
		t.Errorf("should detect WaitGroup composite, got: %v", results[0].Composite)
	}
}

func TestExplainFullSummaryWithComposite(t *testing.T) {
	ce := NewCodeExplainer()
	src := `package main

import (
	"encoding/json"
	"log"
	"net/http"
)

func handleHealth(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
}

func main() {
	http.HandleFunc("/health", handleHealth)
	log.Fatal(http.ListenAndServe(":8080", nil))
}
`
	results, err := ce.ExplainSource(src)
	if err != nil {
		t.Fatalf("ExplainSource: %v", err)
	}
	// The main function should have a composite pattern about HTTP server
	for _, r := range results {
		if r.Name == "main" {
			if len(r.Composite) == 0 {
				t.Error("main should have composite patterns")
			}
			// Summary should incorporate composite patterns
			if !strings.Contains(r.Summary, "HTTP") {
				t.Errorf("main summary should mention HTTP, got: %s", r.Summary)
			}
		}
	}
}

func TestExplainCompositeTimeMeasurement(t *testing.T) {
	ce := NewCodeExplainer()
	src := `package main

import (
	"fmt"
	"time"
)

func benchmark() {
	start := time.Now()
	doWork()
	elapsed := time.Since(start)
	fmt.Println(elapsed)
}
`
	results, err := ce.ExplainSource(src)
	if err != nil {
		t.Fatalf("ExplainSource: %v", err)
	}
	if len(results) != 1 {
		t.Fatalf("expected 1 function, got %d", len(results))
	}
	compositeStr := strings.Join(results[0].Composite, " | ")
	if !strings.Contains(compositeStr, "execution time") && !strings.Contains(compositeStr, "Measures") {
		t.Errorf("should detect time measurement composite, got: %v", results[0].Composite)
	}
}
