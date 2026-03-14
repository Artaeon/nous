package compress

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/artaeon/nous/internal/ollama"
)

func TestParseAtomValid(t *testing.T) {
	input := "TRIGGER: code review\nKNOWLEDGE: Always check error returns in Go."
	atom := parseAtom(input)

	if atom.Trigger != "code review" {
		t.Errorf("Trigger = %q, want %q", atom.Trigger, "code review")
	}
	if atom.Content != "Always check error returns in Go." {
		t.Errorf("Content = %q, want %q", atom.Content, "Always check error returns in Go.")
	}
	if atom.Weight != 0.5 {
		t.Errorf("Weight = %f, want 0.5", atom.Weight)
	}
}

func TestParseAtomCaseInsensitive(t *testing.T) {
	input := "trigger: debugging\nknowledge: Use delve for Go debugging."
	atom := parseAtom(input)

	if atom.Trigger != "debugging" {
		t.Errorf("Trigger = %q, want %q", atom.Trigger, "debugging")
	}
	if atom.Content != "Use delve for Go debugging." {
		t.Errorf("Content = %q, want %q", atom.Content, "Use delve for Go debugging.")
	}
}

func TestParseAtomMixedCase(t *testing.T) {
	input := "Trigger: testing\nKnowledge: Table-driven tests are idiomatic."
	atom := parseAtom(input)

	if atom.Trigger != "testing" {
		t.Errorf("Trigger = %q, want %q", atom.Trigger, "testing")
	}
	if atom.Content != "Table-driven tests are idiomatic." {
		t.Errorf("Content = %q, want %q", atom.Content, "Table-driven tests are idiomatic.")
	}
}

func TestParseAtomFallback(t *testing.T) {
	input := "Some random response without the expected format"
	atom := parseAtom(input)

	if atom.Trigger != "general" {
		t.Errorf("Trigger = %q, want %q", atom.Trigger, "general")
	}
	if atom.Content != input {
		t.Errorf("Content = %q, want %q", atom.Content, input)
	}
}

func TestParseAtomEmpty(t *testing.T) {
	atom := parseAtom("")

	if atom.Trigger != "general" {
		t.Errorf("Trigger = %q, want %q", atom.Trigger, "general")
	}
	if atom.Content != "" {
		t.Errorf("Content = %q, want empty", atom.Content)
	}
}

func TestParseAtomExtraWhitespace(t *testing.T) {
	input := "  TRIGGER:   networking  \n  KNOWLEDGE:   Use context for timeouts.  "
	atom := parseAtom(input)

	if atom.Trigger != "networking" {
		t.Errorf("Trigger = %q, want %q", atom.Trigger, "networking")
	}
	if atom.Content != "Use context for timeouts." {
		t.Errorf("Content = %q, want %q", atom.Content, "Use context for timeouts.")
	}
}

func TestParseAtomTriggerOnly(t *testing.T) {
	input := "TRIGGER: something"
	atom := parseAtom(input)

	// No KNOWLEDGE line, so Content falls back to entire response
	if atom.Content != input {
		t.Errorf("Content = %q, want %q", atom.Content, input)
	}
	if atom.Trigger != "general" {
		t.Errorf("Trigger = %q, want %q", atom.Trigger, "general")
	}
}

func TestOverlapScoreIdentical(t *testing.T) {
	a := []string{"hello", "world"}
	b := []string{"hello", "world"}
	score := overlapScore(a, b)
	if score != 1.0 {
		t.Errorf("score = %f, want 1.0", score)
	}
}

func TestOverlapScorePartial(t *testing.T) {
	a := []string{"hello", "world", "foo"}
	b := []string{"hello", "bar"}
	score := overlapScore(a, b)
	// 1 match out of 2 trigger words
	if score != 0.5 {
		t.Errorf("score = %f, want 0.5", score)
	}
}

func TestOverlapScoreNoMatch(t *testing.T) {
	a := []string{"hello"}
	b := []string{"world"}
	score := overlapScore(a, b)
	if score != 0.0 {
		t.Errorf("score = %f, want 0.0", score)
	}
}

func TestOverlapScoreEmptyB(t *testing.T) {
	a := []string{"hello"}
	score := overlapScore(a, nil)
	if score != 0.0 {
		t.Errorf("score = %f, want 0.0", score)
	}
}

func TestOverlapScoreEmptyA(t *testing.T) {
	b := []string{"hello"}
	score := overlapScore(nil, b)
	if score != 0.0 {
		t.Errorf("score = %f, want 0.0", score)
	}
}

func TestOverlapScoreBothEmpty(t *testing.T) {
	score := overlapScore(nil, nil)
	if score != 0.0 {
		t.Errorf("score = %f, want 0.0", score)
	}
}

func TestNewCompressor(t *testing.T) {
	c := NewCompressor(nil)
	if c == nil {
		t.Fatal("NewCompressor returned nil")
	}
	if c.Count() != 0 {
		t.Errorf("Count = %d, want 0", c.Count())
	}
}

func TestRelevantEmpty(t *testing.T) {
	c := NewCompressor(nil)
	results := c.Relevant("anything", 5)
	if results != nil {
		t.Errorf("Relevant on empty compressor = %v, want nil", results)
	}
}

func TestRelevantFindsMatching(t *testing.T) {
	c := NewCompressor(nil)
	c.atoms = []Atom{
		{Trigger: "go testing", Content: "Use table-driven tests", Weight: 0.8},
		{Trigger: "python debugging", Content: "Use pdb", Weight: 0.5},
		{Trigger: "go concurrency", Content: "Use channels over mutexes", Weight: 0.7},
	}

	results := c.Relevant("go testing patterns", 2)
	if len(results) == 0 {
		t.Fatal("expected at least one result")
	}
	// "go testing" has the most overlap with "go testing patterns"
	if results[0].Content != "Use table-driven tests" {
		t.Errorf("top result = %q, want %q", results[0].Content, "Use table-driven tests")
	}
}

func TestRelevantRespectsMaxAtoms(t *testing.T) {
	c := NewCompressor(nil)
	c.atoms = []Atom{
		{Trigger: "go error", Content: "Wrap errors", Weight: 0.5},
		{Trigger: "go testing", Content: "Table tests", Weight: 0.5},
		{Trigger: "go modules", Content: "Use go mod tidy", Weight: 0.5},
	}

	results := c.Relevant("go", 2)
	if len(results) != 2 {
		t.Errorf("got %d results, want 2", len(results))
	}
}

func TestRelevantNoMatch(t *testing.T) {
	c := NewCompressor(nil)
	c.atoms = []Atom{
		{Trigger: "python debugging", Content: "Use pdb", Weight: 0.5},
	}

	results := c.Relevant("go testing", 5)
	if len(results) != 0 {
		t.Errorf("got %d results, want 0", len(results))
	}
}

func TestRelevantWeightAffectsOrder(t *testing.T) {
	c := NewCompressor(nil)
	c.atoms = []Atom{
		{Trigger: "go patterns", Content: "Low weight", Weight: 0.1},
		{Trigger: "go patterns", Content: "High weight", Weight: 1.0},
	}

	results := c.Relevant("go patterns", 2)
	if len(results) != 2 {
		t.Fatalf("got %d results, want 2", len(results))
	}
	if results[0].Content != "High weight" {
		t.Errorf("first result = %q, want %q", results[0].Content, "High weight")
	}
	if results[1].Content != "Low weight" {
		t.Errorf("second result = %q, want %q", results[1].Content, "Low weight")
	}
}

func TestRelevantMaxAtomsExceedsResults(t *testing.T) {
	c := NewCompressor(nil)
	c.atoms = []Atom{
		{Trigger: "go testing", Content: "Use table tests", Weight: 0.5},
	}

	results := c.Relevant("go testing", 10)
	if len(results) != 1 {
		t.Errorf("got %d results, want 1", len(results))
	}
}

func TestCount(t *testing.T) {
	c := NewCompressor(nil)
	if c.Count() != 0 {
		t.Errorf("Count = %d, want 0", c.Count())
	}
	c.atoms = append(c.atoms, Atom{Trigger: "test", Content: "test", Weight: 0.5})
	if c.Count() != 1 {
		t.Errorf("Count = %d, want 1", c.Count())
	}
}

func TestCompressIntegration(t *testing.T) {
	// Mock Ollama server
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		resp := ollama.GenerateResponse{
			Message: ollama.Message{
				Role:    "assistant",
				Content: "TRIGGER: error handling\nKNOWLEDGE: Always wrap errors with context using fmt.Errorf.",
			},
			Done: true,
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
	}))
	defer srv.Close()

	client := ollama.New(ollama.WithHost(srv.URL))
	comp := NewCompressor(client)

	atom, err := comp.Compress("How do I handle errors?", "Use fmt.Errorf to wrap errors.")
	if err != nil {
		t.Fatalf("Compress error: %v", err)
	}
	if atom.Trigger != "error handling" {
		t.Errorf("Trigger = %q, want %q", atom.Trigger, "error handling")
	}
	if atom.Content != "Always wrap errors with context using fmt.Errorf." {
		t.Errorf("Content = %q, want %q", atom.Content, "Always wrap errors with context using fmt.Errorf.")
	}
	if comp.Count() != 1 {
		t.Errorf("Count = %d, want 1", comp.Count())
	}
}

func TestCompressServerError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "internal error", http.StatusInternalServerError)
	}))
	defer srv.Close()

	client := ollama.New(ollama.WithHost(srv.URL))
	comp := NewCompressor(client)

	_, err := comp.Compress("input", "response")
	if err == nil {
		t.Fatal("expected error from server failure")
	}
}
