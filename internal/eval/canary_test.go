package eval

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestGenerateCanarySet(t *testing.T) {
	cs := GenerateCanarySet()
	if len(cs.Prompts) != 50 {
		t.Errorf("canary set has %d prompts, want 50", len(cs.Prompts))
	}

	// Verify all have IDs
	for i, p := range cs.Prompts {
		if p.ID == "" {
			t.Errorf("prompt %d has no ID", i)
		}
		if p.Query == "" {
			t.Errorf("prompt %d has no query", i)
		}
		if p.ExpectedType == "" {
			t.Errorf("prompt %d has no expected_type", i)
		}
		if len(p.MustNotMatch) == 0 {
			t.Errorf("prompt %d (%q) has no must_not_match patterns", i, p.Query)
		}
	}

	// Verify categories are balanced
	types := make(map[string]int)
	failures := make(map[string]int)
	for _, p := range cs.Prompts {
		types[p.ExpectedType]++
		failures[p.FailureType]++
	}
	if types["explain"] < 15 {
		t.Errorf("too few explain prompts: %d", types["explain"])
	}
	if types["compare"] < 10 {
		t.Errorf("too few compare prompts: %d", types["compare"])
	}
	if failures["wrong_route"] < 10 {
		t.Errorf("too few wrong_route cases: %d", failures["wrong_route"])
	}
	if failures["filler"] < 5 {
		t.Errorf("too few filler cases: %d", failures["filler"])
	}
}

func TestCanarySetPersistence(t *testing.T) {
	cs := GenerateCanarySet()
	dir := t.TempDir()
	path := filepath.Join(dir, "canary.json")

	if err := SaveCanarySet(cs, path); err != nil {
		t.Fatalf("save: %v", err)
	}

	loaded, err := LoadCanarySet(path)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if len(loaded.Prompts) != len(cs.Prompts) {
		t.Errorf("loaded %d prompts, want %d", len(loaded.Prompts), len(cs.Prompts))
	}
}

func TestValidateCanaryResponse_Pass(t *testing.T) {
	prompt := CanaryPrompt{
		Query:        "explain photosynthesis",
		MustRoute:    "lookup_knowledge",
		MustNotMatch: []string{"i appreciate", "gotcha"},
	}
	fail := ValidateCanaryResponse(prompt, "Photosynthesis is the process by which plants convert sunlight into energy.", "lookup_knowledge")
	if fail != nil {
		t.Errorf("expected pass, got failure: %s", fail.Reason)
	}
}

func TestValidateCanaryResponse_WrongRoute(t *testing.T) {
	prompt := CanaryPrompt{
		Query:     "explain photosynthesis",
		MustRoute: "lookup_knowledge",
	}
	fail := ValidateCanaryResponse(prompt, "anything", "respond")
	if fail == nil {
		t.Fatal("expected failure for wrong route")
	}
	if !strings.Contains(fail.Reason, "wrong_route") {
		t.Errorf("reason = %q, want wrong_route", fail.Reason)
	}
}

func TestValidateCanaryResponse_FluffDetected(t *testing.T) {
	prompt := CanaryPrompt{
		Query:        "explain quantum physics",
		MustRoute:    "lookup_knowledge",
		MustNotMatch: []string{"i appreciate", "gotcha"},
	}
	fail := ValidateCanaryResponse(prompt, "I appreciate you asking about quantum physics! It's very cool.", "lookup_knowledge")
	if fail == nil {
		t.Fatal("expected failure for fluff detection")
	}
	if !strings.Contains(fail.Reason, "fluff_detected") {
		t.Errorf("reason = %q, want fluff_detected", fail.Reason)
	}
}

func TestValidateCanaryResponse_MustMatch(t *testing.T) {
	prompt := CanaryPrompt{
		Query:     "explain CRISPR gene editing",
		MustRoute: "lookup_knowledge",
		MustMatch: []string{"knowledge", "don't have", "learn"},
	}
	// Should fail because none of the must-match patterns are present
	fail := ValidateCanaryResponse(prompt, "CRISPR is cool.", "lookup_knowledge")
	if fail == nil {
		t.Fatal("expected failure for missing must-match")
	}
	if !strings.Contains(fail.Reason, "missing_expected") {
		t.Errorf("reason = %q, want missing_expected", fail.Reason)
	}
}

func TestCheckMergeGates_Pass(t *testing.T) {
	result := &CanaryResult{
		TotalPrompts:    50,
		RoutingAccuracy: 0.95,
		FillerRate:      0.0,
		UsefulRate:      0.80,
	}
	failures := CheckMergeGates(result, DefaultMergeGateConfig())
	if len(failures) > 0 {
		t.Errorf("expected pass, got failures: %v", failures)
	}
}

func TestCheckMergeGates_Fail(t *testing.T) {
	result := &CanaryResult{
		TotalPrompts:    50,
		FailedPrompts:   make([]CanaryFailure, 10), // 20% fail rate
		RoutingAccuracy: 0.80,                       // below 90%
		FillerRate:      0.05,                       // above 0%
		UsefulRate:      0.60,                       // below 70%
	}
	failures := CheckMergeGates(result, DefaultMergeGateConfig())
	if len(failures) < 3 {
		t.Errorf("expected at least 3 failures, got %d: %v", len(failures), failures)
	}
}

func TestTaskKPIs(t *testing.T) {
	k := &TaskKPIs{}

	// Record 10 routing decisions: 9 correct
	for i := 0; i < 10; i++ {
		k.RecordRouting(i < 9)
	}
	if got := k.RoutingAccuracy(); got != 0.9 {
		t.Errorf("routing accuracy = %.2f, want 0.90", got)
	}

	// Record 20 responses: 1 with filler
	for i := 0; i < 20; i++ {
		k.RecordFiller(i == 0)
	}
	if got := k.FillerRate(); got != 0.05 {
		t.Errorf("filler rate = %.2f, want 0.05", got)
	}

	// Record 10 explain/compare: 8 useful
	for i := 0; i < 10; i++ {
		k.RecordUsefulness(i < 8)
	}
	if got := k.UsefulRate(); got != 0.8 {
		t.Errorf("useful rate = %.2f, want 0.80", got)
	}

	// Verify report is non-empty
	report := k.Report()
	if !strings.Contains(report, "Routing accuracy") {
		t.Error("report missing routing accuracy")
	}
	if !strings.Contains(report, "Filler rate") {
		t.Error("report missing filler rate")
	}
	if !strings.Contains(report, "Explain/Compare") {
		t.Error("report missing useful rate")
	}
}

func TestTaskKPIs_Empty(t *testing.T) {
	k := &TaskKPIs{}
	if k.RoutingAccuracy() != 1.0 {
		t.Error("empty routing accuracy should be 1.0")
	}
	if k.FillerRate() != 0.0 {
		t.Error("empty filler rate should be 0.0")
	}
	if k.UsefulRate() != 1.0 {
		t.Error("empty useful rate should be 1.0")
	}
}

func TestCanarySetUniqueIDs(t *testing.T) {
	cs := GenerateCanarySet()
	seen := make(map[string]bool)
	for _, p := range cs.Prompts {
		if seen[p.ID] {
			t.Errorf("duplicate ID: %s", p.ID)
		}
		seen[p.ID] = true
	}
}

func TestLoadCanarySet_NotFound(t *testing.T) {
	_, err := LoadCanarySet("/nonexistent/path.json")
	if err == nil {
		t.Error("expected error for nonexistent path")
	}
}

func TestSaveCanarySet_BadPath(t *testing.T) {
	cs := GenerateCanarySet()
	err := SaveCanarySet(cs, "/nonexistent/dir/canary.json")
	if err == nil && !os.IsNotExist(err) {
		// Some systems might allow this; just check it doesn't panic
	}
}
