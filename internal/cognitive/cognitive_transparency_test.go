package cognitive

import (
	"strings"
	"testing"
)

func TestBuildTrace(t *testing.T) {
	ct := NewCognitiveTransparency()
	trace := ct.BuildTrace(
		"quantum physics",
		[]string{"knowledge_graph"},
		3,
		[]string{"quantum physics involves wave-particle duality"},
		[]string{"I don't have data on recent experimental results"},
		0.78,
	)

	if trace == nil {
		t.Fatal("expected non-nil trace")
	}
	if trace.Confidence != 0.78 {
		t.Errorf("confidence = %f, want 0.78", trace.Confidence)
	}
	if len(trace.Sources) != 1 {
		t.Fatalf("sources = %d, want 1", len(trace.Sources))
	}
	if trace.Sources[0].Name != "knowledge_graph" {
		t.Errorf("source name = %q, want %q", trace.Sources[0].Name, "knowledge_graph")
	}
	if trace.Sources[0].FactCount != 3 {
		t.Errorf("fact count = %d, want 3", trace.Sources[0].FactCount)
	}
	if len(trace.Inferences) != 1 {
		t.Fatalf("inferences = %d, want 1", len(trace.Inferences))
	}
	if trace.Inferences[0].Conclusion != "quantum physics involves wave-particle duality" {
		t.Errorf("inference conclusion = %q", trace.Inferences[0].Conclusion)
	}
	if len(trace.Uncertainties) != 1 {
		t.Fatalf("uncertainties = %d, want 1", len(trace.Uncertainties))
	}
	if trace.Summary == "" {
		t.Error("expected non-empty summary")
	}
	// Steps should include: 1 retrieval + 1 inference + 1 fallback/uncertainty = 3
	if len(trace.Steps) != 3 {
		t.Errorf("steps = %d, want 3", len(trace.Steps))
	}
}

func TestFormatCompact(t *testing.T) {
	ct := NewCognitiveTransparency()
	trace := ct.BuildTrace(
		"quantum physics",
		[]string{"knowledge_graph"},
		3,
		nil,
		nil,
		0.85,
	)

	compact := trace.Format(false)
	if !strings.HasPrefix(compact, "[") || !strings.HasSuffix(compact, "]") {
		t.Errorf("compact format should be bracketed, got: %s", compact)
	}
	if !strings.Contains(compact, "knowledge_graph") {
		t.Errorf("compact should mention source, got: %s", compact)
	}
	if !strings.Contains(compact, "3 facts") {
		t.Errorf("compact should mention fact count, got: %s", compact)
	}
	if !strings.Contains(compact, "high") {
		t.Errorf("compact should classify confidence as high, got: %s", compact)
	}
}

func TestFormatVerbose(t *testing.T) {
	ct := NewCognitiveTransparency()
	trace := ct.BuildTrace(
		"quantum physics",
		[]string{"knowledge_graph"},
		3,
		[]string{"quantum physics involves wave-particle duality"},
		[]string{"I don't have data on recent experimental results"},
		0.78,
	)

	verbose := trace.Format(true)

	// Must start with the header.
	if !strings.HasPrefix(verbose, "How I arrived at this:") {
		t.Errorf("verbose should start with header, got: %s", verbose[:50])
	}

	// Must contain numbered steps.
	if !strings.Contains(verbose, "1. ") {
		t.Error("verbose should have numbered steps")
	}

	// Must mention retrieved facts.
	if !strings.Contains(verbose, "Retrieved") {
		t.Error("verbose should mention retrieval")
	}

	// Must mention inference.
	if !strings.Contains(verbose, "Inferred") {
		t.Error("verbose should mention inference")
	}

	// Must mention uncertainty.
	if !strings.Contains(verbose, "Uncertainty") {
		t.Error("verbose should mention uncertainty")
	}

	// Must have Sources line.
	if !strings.Contains(verbose, "Sources:") {
		t.Error("verbose should have Sources section")
	}
	if !strings.Contains(verbose, "knowledge_graph (3 facts)") {
		t.Errorf("verbose should list source details, got:\n%s", verbose)
	}
	if !strings.Contains(verbose, "inference (1 conclusion)") {
		t.Errorf("verbose should list inference count, got:\n%s", verbose)
	}

	// Must have Confidence line.
	if !strings.Contains(verbose, "Confidence: 0.78") {
		t.Errorf("verbose should show confidence value, got:\n%s", verbose)
	}
}

func TestClassifyConfidence(t *testing.T) {
	tests := []struct {
		score float64
		want  string
	}{
		{0.95, "very high"},
		{0.91, "very high"},
		{0.90, "high"},    // boundary: >= 0.7
		{0.85, "high"},
		{0.70, "high"},    // boundary
		{0.65, "moderate"},
		{0.50, "moderate"}, // boundary
		{0.45, "low"},
		{0.30, "low"},     // boundary
		{0.25, "very low"},
		{0.10, "very low"},
		{0.0, "very low"},
	}

	for _, tt := range tests {
		got := ClassifyConfidence(tt.score)
		if got != tt.want {
			t.Errorf("ClassifyConfidence(%f) = %q, want %q", tt.score, got, tt.want)
		}
	}
}

func TestAddStepAndInference(t *testing.T) {
	trace := &TransparencyTrace{
		Uncertainties: []string{},
	}

	// Add steps.
	trace.AddStep("retrieved", "Found 5 facts about Go", 0.9)
	trace.AddStep("synthesized", "Combined facts into response", 0.85)

	if len(trace.Steps) != 2 {
		t.Fatalf("steps = %d, want 2", len(trace.Steps))
	}
	if trace.Steps[0].Action != "retrieved" {
		t.Errorf("step[0].Action = %q, want %q", trace.Steps[0].Action, "retrieved")
	}
	if trace.Steps[1].Confidence != 0.85 {
		t.Errorf("step[1].Confidence = %f, want 0.85", trace.Steps[1].Confidence)
	}

	// Add inference.
	trace.AddInference(
		"Go is a compiled language; compiled languages are fast",
		"Go is fast",
		"transitivity",
		0.88,
	)

	if len(trace.Inferences) != 1 {
		t.Fatalf("inferences = %d, want 1", len(trace.Inferences))
	}
	inf := trace.Inferences[0]
	if inf.Rule != "transitivity" {
		t.Errorf("inference rule = %q, want %q", inf.Rule, "transitivity")
	}
	if inf.Conclusion != "Go is fast" {
		t.Errorf("inference conclusion = %q, want %q", inf.Conclusion, "Go is fast")
	}
	if inf.Confidence != 0.88 {
		t.Errorf("inference confidence = %f, want 0.88", inf.Confidence)
	}

	// Add uncertainty.
	trace.AddUncertainty("benchmark results may vary by workload")
	if len(trace.Uncertainties) != 1 {
		t.Fatalf("uncertainties = %d, want 1", len(trace.Uncertainties))
	}
	if trace.Uncertainties[0] != "benchmark results may vary by workload" {
		t.Errorf("uncertainty = %q", trace.Uncertainties[0])
	}
}

func TestMergeTraces(t *testing.T) {
	ct := NewCognitiveTransparency()

	t1 := ct.BuildTrace(
		"Go concurrency",
		[]string{"knowledge_graph"},
		2,
		nil,
		[]string{"limited data on goroutine scheduling internals"},
		0.80,
	)

	t2 := ct.BuildTrace(
		"Go channels",
		[]string{"knowledge_graph", "episodic_memory"},
		4,
		[]string{"channels provide type-safe communication"},
		nil,
		0.90,
	)

	merged := MergeTraces(t1, t2)

	if merged == nil {
		t.Fatal("expected non-nil merged trace")
	}

	// Steps from both traces.
	if len(merged.Steps) < 3 {
		t.Errorf("merged steps = %d, expected at least 3", len(merged.Steps))
	}

	// Sources should be deduplicated: knowledge_graph appears in both,
	// episodic_memory only in t2.
	foundKG := false
	foundEM := false
	for _, s := range merged.Sources {
		switch s.Name {
		case "knowledge_graph":
			foundKG = true
			// knowledge_graph: 2 from t1 + 2 from t2 (4/2=2 per source in t2) = 4
			// Actually: t1 has 2 facts in knowledge_graph; t2 distributes 4 across 2 sources = 2 each
			// So merged knowledge_graph = 2 + 2 = 4
			if s.FactCount < 2 {
				t.Errorf("knowledge_graph fact count = %d, expected >= 2", s.FactCount)
			}
		case "episodic_memory":
			foundEM = true
		}
	}
	if !foundKG {
		t.Error("merged trace missing knowledge_graph source")
	}
	if !foundEM {
		t.Error("merged trace missing episodic_memory source")
	}

	// Confidence should be average of 0.80 and 0.90 = 0.85.
	if merged.Confidence < 0.84 || merged.Confidence > 0.86 {
		t.Errorf("merged confidence = %f, want ~0.85", merged.Confidence)
	}

	// Uncertainties should include the one from t1.
	if len(merged.Uncertainties) != 1 {
		t.Errorf("merged uncertainties = %d, want 1", len(merged.Uncertainties))
	}

	// Inferences from t2.
	if len(merged.Inferences) != 1 {
		t.Errorf("merged inferences = %d, want 1", len(merged.Inferences))
	}

	// Summary should be non-empty.
	if merged.Summary == "" {
		t.Error("merged summary should not be empty")
	}
}

func TestMergeTracesWithNil(t *testing.T) {
	ct := NewCognitiveTransparency()
	t1 := ct.BuildTrace("test", []string{"kg"}, 1, nil, nil, 0.7)

	// Merging with nil traces should not panic.
	merged := MergeTraces(t1, nil, nil)
	if merged.Confidence != 0.7 {
		t.Errorf("confidence = %f, want 0.7", merged.Confidence)
	}
}

func TestEmptyTrace(t *testing.T) {
	ct := NewCognitiveTransparency()

	// No sources, no inferences, no uncertainties.
	trace := ct.BuildTrace("", nil, 0, nil, nil, 0.0)

	if trace == nil {
		t.Fatal("expected non-nil trace even with empty inputs")
	}
	if len(trace.Steps) != 0 {
		t.Errorf("steps = %d, want 0", len(trace.Steps))
	}
	if len(trace.Sources) != 0 {
		t.Errorf("sources = %d, want 0", len(trace.Sources))
	}
	if len(trace.Inferences) != 0 {
		t.Errorf("inferences = %d, want 0", len(trace.Inferences))
	}
	if trace.Uncertainties == nil {
		t.Error("uncertainties should be initialized to empty slice, not nil")
	}
	if trace.Confidence != 0.0 {
		t.Errorf("confidence = %f, want 0.0", trace.Confidence)
	}

	// Compact format should still produce valid output.
	compact := trace.Format(false)
	if compact == "" {
		t.Error("compact format should not be empty even for empty trace")
	}
	if !strings.Contains(compact, "very low") {
		t.Errorf("empty trace should have very low confidence, got: %s", compact)
	}

	// Verbose format should still produce valid output.
	verbose := trace.Format(true)
	if verbose == "" {
		t.Error("verbose format should not be empty even for empty trace")
	}

	// Merge of zero traces.
	merged := MergeTraces()
	if merged == nil {
		t.Fatal("merge of zero traces should return non-nil")
	}
	if len(merged.Steps) != 0 {
		t.Errorf("empty merge steps = %d, want 0", len(merged.Steps))
	}
}
