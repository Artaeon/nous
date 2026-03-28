package training

import (
	"strings"
	"testing"
	"time"
)

// ---------------------------------------------------------------------------
// Distillation tests
// ---------------------------------------------------------------------------

func TestDistillationStore_RecordAndLoad(t *testing.T) {
	dir := t.TempDir()
	store := NewDistillationStore(dir)

	trace := &TeacherTrace{
		Input:    "What is photosynthesis?",
		Intent:   "explain",
		Slots:    map[string]string{"topic": "photosynthesis"},
		Plan:     []string{"define process", "describe inputs", "describe outputs"},
		Response: "Photosynthesis is the process by which plants convert light energy into chemical energy.",
		Quality:  0.9,
	}

	if err := store.RecordTrace(trace); err != nil {
		t.Fatalf("RecordTrace: %v", err)
	}

	// Load into a new store instance
	store2 := NewDistillationStore(dir)
	traces, err := store2.LoadTraces()
	if err != nil {
		t.Fatalf("LoadTraces: %v", err)
	}

	if len(traces) != 1 {
		t.Fatalf("expected 1 trace, got %d", len(traces))
	}
	if traces[0].Intent != "explain" {
		t.Errorf("intent = %q, want %q", traces[0].Intent, "explain")
	}
	if traces[0].Slots["topic"] != "photosynthesis" {
		t.Errorf("slot topic = %q, want %q", traces[0].Slots["topic"], "photosynthesis")
	}
}

func TestDistillationStore_FilterByQuality(t *testing.T) {
	store := NewDistillationStore("")

	traces := []*TeacherTrace{
		{Input: "low", Intent: "a", Quality: 0.3},
		{Input: "med", Intent: "b", Quality: 0.6},
		{Input: "high", Intent: "c", Quality: 0.9},
		{Input: "top", Intent: "d", Quality: 1.0},
	}
	for _, tr := range traces {
		if err := store.RecordTrace(tr); err != nil {
			t.Fatalf("RecordTrace: %v", err)
		}
	}

	filtered := store.FilterByQuality(0.7)
	if len(filtered) != 2 {
		t.Errorf("expected 2 traces >= 0.7, got %d", len(filtered))
	}

	filtered = store.FilterByQuality(0.0)
	if len(filtered) != 4 {
		t.Errorf("expected 4 traces >= 0.0, got %d", len(filtered))
	}
}

func TestIntentDistiller_Distill(t *testing.T) {
	store := NewDistillationStore("")

	// Create enough traces with varied intents and quality
	intents := []string{"explain", "compare", "define", "explain", "compare",
		"explain", "define", "explain", "compare", "explain"}
	for i, intent := range intents {
		quality := 0.75 + float64(i)*0.02
		if quality > 1.0 {
			quality = 1.0
		}
		tr := &TeacherTrace{
			Input:    "Tell me about topic " + intent + " " + strings.Repeat("word ", i+1),
			Intent:   intent,
			Slots:    map[string]string{"topic": "topic_" + intent},
			Plan:     []string{"claim1", "claim2"},
			Response: "Response for " + intent,
			Quality:  quality,
		}
		if err := store.RecordTrace(tr); err != nil {
			t.Fatalf("RecordTrace: %v", err)
		}
	}

	config := DefaultDistillationConfig()
	config.MinQuality = 0.7
	config.ValidationSplit = 0.3

	distiller := NewIntentDistiller(store, config)
	result, examples, err := distiller.Distill()
	if err != nil {
		t.Fatalf("Distill: %v", err)
	}

	if result == nil {
		t.Fatal("result is nil")
	}
	if len(examples) == 0 {
		t.Fatal("no examples produced")
	}
	if result.TracesUsed == 0 {
		t.Error("TracesUsed should be > 0")
	}
	if result.Epochs != config.Epochs {
		t.Errorf("Epochs = %d, want %d", result.Epochs, config.Epochs)
	}

	// Every example should have intent and confidence
	for i, ex := range examples {
		if ex.Intent == "" {
			t.Errorf("example[%d] has empty intent", i)
		}
		if ex.Confidence <= 0 || ex.Confidence > 1.0 {
			t.Errorf("example[%d] confidence = %.2f, want (0,1]", i, ex.Confidence)
		}
	}
}

func TestPlanDistiller_Distill(t *testing.T) {
	store := NewDistillationStore("")

	for i := 0; i < 10; i++ {
		tr := &TeacherTrace{
			Input:    "Explain about plants and photosynthesis " + strings.Repeat("extra ", i),
			Intent:   "explain",
			Plan:     []string{"define photosynthesis", "describe light reaction", "describe carbon fixation"},
			Response: "Photosynthesis is a complex process involving light and dark reactions.",
			Quality:  0.8 + float64(i)*0.01,
		}
		if err := store.RecordTrace(tr); err != nil {
			t.Fatalf("RecordTrace: %v", err)
		}
	}

	config := DefaultDistillationConfig()
	config.MinQuality = 0.7

	distiller := NewPlanDistiller(store, config)
	result, examples, err := distiller.Distill()
	if err != nil {
		t.Fatalf("Distill: %v", err)
	}

	if result == nil {
		t.Fatal("result is nil")
	}
	if len(examples) == 0 {
		t.Fatal("no plan examples produced")
	}
	if result.PlanQuality < 0 || result.PlanQuality > 1 {
		t.Errorf("PlanQuality = %.2f, want [0, 1]", result.PlanQuality)
	}

	for i, ex := range examples {
		if len(ex.Claims) == 0 {
			t.Errorf("example[%d] has no claims", i)
		}
		if ex.Quality <= 0 {
			t.Errorf("example[%d] quality = %.2f, want > 0", i, ex.Quality)
		}
	}
}

// ---------------------------------------------------------------------------
// Preference optimization tests
// ---------------------------------------------------------------------------

func TestPreferenceStore_RecordAndLoad(t *testing.T) {
	dir := t.TempDir()
	store := NewPreferenceStore(dir)

	pair := &PreferencePair{
		Input:    "What is 2+2?",
		Chosen:   "2+2 equals 4.",
		Rejected: "I think it might be 5.",
		Margin:   0.8,
		Source:   "human",
	}

	if err := store.RecordPair(pair); err != nil {
		t.Fatalf("RecordPair: %v", err)
	}

	store2 := NewPreferenceStore(dir)
	pairs, err := store2.LoadPairs()
	if err != nil {
		t.Fatalf("LoadPairs: %v", err)
	}

	if len(pairs) != 1 {
		t.Fatalf("expected 1 pair, got %d", len(pairs))
	}
	if pairs[0].Chosen != "2+2 equals 4." {
		t.Errorf("chosen = %q, want %q", pairs[0].Chosen, "2+2 equals 4.")
	}
	if pairs[0].Source != "human" {
		t.Errorf("source = %q, want %q", pairs[0].Source, "human")
	}
}

func TestPreferenceStore_Stats(t *testing.T) {
	store := NewPreferenceStore("")

	pairs := []*PreferencePair{
		{Input: "a", Chosen: "good a", Rejected: "bad a", Margin: 0.3, Source: "human"},
		{Input: "b", Chosen: "good b", Rejected: "bad b", Margin: 0.5, Source: "human"},
		{Input: "c", Chosen: "good c", Rejected: "bad c", Margin: 0.9, Source: "auto_quality"},
		{Input: "d", Chosen: "good d", Rejected: "bad d", Margin: 0.7, Source: "correction"},
	}
	for _, p := range pairs {
		if err := store.RecordPair(p); err != nil {
			t.Fatalf("RecordPair: %v", err)
		}
	}

	stats := store.Stats()
	if stats.TotalPairs != 4 {
		t.Errorf("TotalPairs = %d, want 4", stats.TotalPairs)
	}
	if stats.BySource["human"] != 2 {
		t.Errorf("BySource[human] = %d, want 2", stats.BySource["human"])
	}
	if stats.BySource["auto_quality"] != 1 {
		t.Errorf("BySource[auto_quality] = %d, want 1", stats.BySource["auto_quality"])
	}
	if stats.MinMargin != 0.3 {
		t.Errorf("MinMargin = %.1f, want 0.3", stats.MinMargin)
	}
	if stats.MaxMargin != 0.9 {
		t.Errorf("MaxMargin = %.1f, want 0.9", stats.MaxMargin)
	}

	// Average margin: (0.3 + 0.5 + 0.9 + 0.7) / 4 = 0.6
	if stats.AverageMargin < 0.59 || stats.AverageMargin > 0.61 {
		t.Errorf("AverageMargin = %.2f, want ~0.6", stats.AverageMargin)
	}
}

func TestPreferenceOptimizer_Optimize(t *testing.T) {
	store := NewPreferenceStore("")

	// Create pairs where chosen is clearly better (longer, more specific, on-topic)
	for i := 0; i < 20; i++ {
		pair := &PreferencePair{
			Input:    "Tell me about quantum physics and wave-particle duality.",
			Chosen:   "Quantum physics describes the behavior of particles at the subatomic level. Wave-particle duality means particles like electrons exhibit both wave and particle properties, for example in the double-slit experiment.",
			Rejected: "I don't know much about that topic.",
			Margin:   0.7,
			Source:   "auto_quality",
		}
		if err := store.RecordPair(pair); err != nil {
			t.Fatalf("RecordPair: %v", err)
		}
	}

	config := DefaultPreferenceConfig()
	config.MinMargin = 0.1

	optimizer := NewPreferenceOptimizer(store, config)
	result, weights, err := optimizer.Optimize()
	if err != nil {
		t.Fatalf("Optimize: %v", err)
	}

	if result == nil {
		t.Fatal("result is nil")
	}
	if weights == nil {
		t.Fatal("weights is nil")
	}

	if result.PairsUsed != 20 {
		t.Errorf("PairsUsed = %d, want 20", result.PairsUsed)
	}

	// The chosen response should win most of the time after optimization
	if result.ChosenWinRate < 0.5 {
		t.Errorf("ChosenWinRate = %.2f, want >= 0.5", result.ChosenWinRate)
	}

	// Weights should all be non-negative
	if weights.Correctness < 0 || weights.Helpfulness < 0 || weights.Conciseness < 0 ||
		weights.Specificity < 0 || weights.Safety < 0 || weights.Coherence < 0 {
		t.Errorf("weights contain negative values: %+v", weights)
	}
}

func TestAutoGeneratePairs(t *testing.T) {
	outputs := []ScoredOutput{
		{Input: "What is Go?", Response: "Go is a programming language created at Google. It features strong typing, garbage collection, and built-in concurrency.", Quality: 0.9},
		{Input: "What is Go?", Response: "Go is a language.", Quality: 0.3},
		{Input: "What is Go?", Response: "Go is a statically typed, compiled programming language designed at Google.", Quality: 0.7},
		{Input: "What is Rust?", Response: "Rust is a systems programming language.", Quality: 0.6},
		// Only one response for Python, so no pair should be generated
		{Input: "What is Python?", Response: "Python is interpreted.", Quality: 0.5},
	}

	pairs := AutoGeneratePairs(outputs)

	if len(pairs) == 0 {
		t.Fatal("expected pairs to be generated")
	}

	// For "What is Go?" we should get pairs: (0.9 vs 0.3), (0.9 vs 0.7), (0.7 vs 0.3)
	goCount := 0
	for _, p := range pairs {
		if p.Input == "What is Go?" {
			goCount++
		}
		if p.Source != "auto_quality" {
			t.Errorf("source = %q, want %q", p.Source, "auto_quality")
		}
		if p.Margin < 0.05 {
			t.Errorf("margin = %.2f, want >= 0.05", p.Margin)
		}
	}
	if goCount < 2 {
		t.Errorf("expected at least 2 pairs for 'What is Go?', got %d", goCount)
	}

	// "What is Python?" has only one response, should not generate pairs
	for _, p := range pairs {
		if p.Input == "What is Python?" {
			t.Error("should not generate pairs for single-response inputs")
		}
	}
}

func TestRewardWeights_ScoreResponse(t *testing.T) {
	weights := &RewardWeights{
		Correctness: 1.0,
		Helpfulness: 1.0,
		Conciseness: 1.0,
		Specificity: 1.0,
		Safety:      1.0,
		Coherence:   1.0,
	}

	query := "What is machine learning?"

	// Good response should score higher than bad response
	good := "Machine learning is a subset of artificial intelligence. It involves algorithms that learn patterns from data, for example neural networks, and can make predictions without explicit programming."
	bad := "I don't know."

	goodScore := weights.ScoreResponse(good, query)
	badScore := weights.ScoreResponse(bad, query)

	if goodScore <= badScore {
		t.Errorf("good score (%.3f) should be > bad score (%.3f)", goodScore, badScore)
	}

	if goodScore < 0 || goodScore > 1 {
		t.Errorf("good score = %.3f, want [0, 1]", goodScore)
	}
	if badScore < 0 || badScore > 1 {
		t.Errorf("bad score = %.3f, want [0, 1]", badScore)
	}
}

// ---------------------------------------------------------------------------
// Failure mining tests
// ---------------------------------------------------------------------------

func TestFailureStore_RecordAndLoad(t *testing.T) {
	dir := t.TempDir()
	store := NewFailureStore(dir)

	record := &FailureRecord{
		Input:       "Tell me about quantum mechanics",
		Response:    "According to a 2019 study at MIT, quantum mechanics is magic.",
		FailureType: "hallucination",
		Severity:    "major",
		DetectedBy:  "quality_gate",
	}

	if err := store.RecordFailure(record); err != nil {
		t.Fatalf("RecordFailure: %v", err)
	}

	store2 := NewFailureStore(dir)
	records, err := store2.LoadRecords()
	if err != nil {
		t.Fatalf("LoadRecords: %v", err)
	}

	if len(records) != 1 {
		t.Fatalf("expected 1 record, got %d", len(records))
	}
	if records[0].FailureType != "hallucination" {
		t.Errorf("failure_type = %q, want %q", records[0].FailureType, "hallucination")
	}
}

func TestFailureStore_Analyze(t *testing.T) {
	store := NewFailureStore("")

	records := []*FailureRecord{
		{Input: "explain quantum mechanics please", Response: "According to a study...", FailureType: "hallucination", Severity: "major"},
		{Input: "explain quantum physics topic", Response: "Dr. Smith found...", FailureType: "hallucination", Severity: "major"},
		{Input: "what is quantum computing", Response: "That's a great question.", FailureType: "filler_response", Severity: "minor"},
		{Input: "how does quantum entanglement work", Response: "I'm not sure.", FailureType: "filler_response", Severity: "minor"},
		{Input: "explain dark matter", Response: "", FailureType: "crash", Severity: "critical"},
	}

	for _, r := range records {
		if err := store.RecordFailure(r); err != nil {
			t.Fatalf("RecordFailure: %v", err)
		}
	}

	analysis := store.Analyze()

	if analysis.TotalFailures != 5 {
		t.Errorf("TotalFailures = %d, want 5", analysis.TotalFailures)
	}
	if analysis.ByType["hallucination"] != 2 {
		t.Errorf("hallucination count = %d, want 2", analysis.ByType["hallucination"])
	}
	if analysis.ByType["filler_response"] != 2 {
		t.Errorf("filler_response count = %d, want 2", analysis.ByType["filler_response"])
	}
	if analysis.BySeverity["critical"] != 1 {
		t.Errorf("critical count = %d, want 1", analysis.BySeverity["critical"])
	}

	// Should have recommended fixes for the detected types
	if len(analysis.RecommendedFixes) == 0 {
		t.Error("expected recommended fixes")
	}
}

func TestMineRegressionSuite(t *testing.T) {
	store := NewFailureStore("")

	records := []*FailureRecord{
		{
			Input:        "What is the capital of France?",
			Response:     "I think it might be London.",
			FailureType:  "hallucination",
			Severity:     "critical",
			UserFeedback: "The capital is Paris, not London!",
			DetectedBy:   "user_correction",
		},
		{
			Input:       "Tell me about dogs",
			Response:    "That's a great question! Let me think about that.",
			FailureType: "filler_response",
			Severity:    "major",
			DetectedBy:  "quality_gate",
		},
	}

	for _, r := range records {
		if err := store.RecordFailure(r); err != nil {
			t.Fatalf("RecordFailure: %v", err)
		}
	}

	cases := store.MineRegressionSuite()
	if len(cases) != 2 {
		t.Fatalf("expected 2 regression cases, got %d", len(cases))
	}

	// Critical failure should come first (higher priority)
	if cases[0].Priority != 3 {
		t.Errorf("first case priority = %d, want 3 (critical)", cases[0].Priority)
	}
	if cases[0].FailureType != "hallucination" {
		t.Errorf("first case type = %q, want %q", cases[0].FailureType, "hallucination")
	}

	// Must contain hints from user feedback and input
	if len(cases[0].MustContain) == 0 {
		t.Error("expected must-contain hints for hallucination case")
	}
}

func TestMineTrainingData(t *testing.T) {
	store := NewFailureStore("")

	records := []*FailureRecord{
		{
			Input:        "Explain photosynthesis",
			Response:     "I don't really know about that topic.",
			FailureType:  "filler_response",
			Severity:     "major",
			UserFeedback: "Please give a real answer about how plants make energy",
		},
		{
			Input:       "What is gravity?",
			Response:    "According to Dr. Fakename at Fake University, gravity is caused by magnets.",
			FailureType: "hallucination",
			Severity:    "critical",
		},
	}

	for _, r := range records {
		if err := store.RecordFailure(r); err != nil {
			t.Fatalf("RecordFailure: %v", err)
		}
	}

	targets := store.MineTrainingData()
	if len(targets) != 2 {
		t.Fatalf("expected 2 training targets, got %d", len(targets))
	}

	for _, target := range targets {
		if target.Input == "" {
			t.Error("target has empty input")
		}
		if target.BadResponse == "" {
			t.Error("target has empty bad response")
		}
		if len(target.GoodHints) == 0 {
			t.Errorf("target for %q has no good hints", target.FailureType)
		}
	}

	// Filler response target should have hints about substantive answers
	for _, target := range targets {
		if target.FailureType == "filler_response" {
			found := false
			for _, h := range target.GoodHints {
				if strings.Contains(h, "substantive") {
					found = true
					break
				}
			}
			if !found {
				t.Error("filler_response target should hint at giving substantive answers")
			}
		}
	}
}

func TestFailureDetector_Hallucination(t *testing.T) {
	fd := NewFailureDetector()

	tests := []struct {
		name     string
		input    string
		response string
		wantType string
	}{
		{
			name:     "fabricated study",
			input:    "What causes cancer?",
			response: "According to a 2023 study at Harvard University, cancer is caused by excess vitamin C. Dr. Johnson found that 95.7% of people who take supplements develop tumors.",
			wantType: "hallucination",
		},
		{
			name:     "fabricated journal",
			input:    "How does gravity work?",
			response: "As published in the Physical Review Letters, gravity is actually caused by tiny invisible strings. Scientists have proven this conclusively.",
			wantType: "hallucination",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			failures := fd.Detect(tt.input, tt.response, 100)
			found := false
			for _, f := range failures {
				if f.Type == tt.wantType {
					found = true
					if f.Evidence == "" {
						t.Error("expected evidence for hallucination detection")
					}
					if f.Confidence <= 0 {
						t.Error("expected positive confidence")
					}
				}
			}
			if !found {
				t.Errorf("expected %q failure to be detected, got: %v", tt.wantType, failures)
			}
		})
	}
}

func TestFailureDetector_FillerResponse(t *testing.T) {
	fd := NewFailureDetector()

	tests := []struct {
		name     string
		input    string
		response string
	}{
		{
			name:     "not sure",
			input:    "What is dark matter?",
			response: "I'm not sure about that. That's a great question though!",
		},
		{
			name:     "don't know",
			input:    "Explain string theory",
			response: "I don't know much about string theory. Let me think about that.",
		},
		{
			name:     "as an ai",
			input:    "Write a poem",
			response: "As an AI, I'll do my best to write something for you.",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			failures := fd.Detect(tt.input, tt.response, 100)
			found := false
			for _, f := range failures {
				if f.Type == "filler_response" {
					found = true
					break
				}
			}
			if !found {
				t.Errorf("expected filler_response detection for %q", tt.name)
			}
		})
	}
}

func TestFailureDetector_LowQuality(t *testing.T) {
	fd := NewFailureDetector()

	// Very short response
	failures := fd.Detect("Explain quantum computing in detail", "Ok.", 100)
	found := false
	for _, f := range failures {
		if f.Type == "low_quality" {
			found = true
			break
		}
	}
	if !found {
		t.Error("expected low_quality detection for very short response")
	}

	// Empty response should detect crash
	failures = fd.Detect("What is AI?", "", 100)
	foundCrash := false
	for _, f := range failures {
		if f.Type == "crash" {
			foundCrash = true
			break
		}
	}
	if !foundCrash {
		t.Error("expected crash detection for empty response")
	}

	// Timeout detection
	failures = fd.Detect("query", "response here", 35000)
	foundTimeout := false
	for _, f := range failures {
		if f.Type == "timeout" {
			foundTimeout = true
			break
		}
	}
	if !foundTimeout {
		t.Error("expected timeout detection for 35s latency")
	}
}

// ---------------------------------------------------------------------------
// Additional edge case tests
// ---------------------------------------------------------------------------

func TestDistillationStore_EmptyFilter(t *testing.T) {
	store := NewDistillationStore("")
	filtered := store.FilterByQuality(0.5)
	if len(filtered) != 0 {
		t.Errorf("expected 0 from empty store, got %d", len(filtered))
	}
}

func TestPreferenceStore_EmptyStats(t *testing.T) {
	store := NewPreferenceStore("")
	stats := store.Stats()
	if stats.TotalPairs != 0 {
		t.Errorf("TotalPairs = %d, want 0", stats.TotalPairs)
	}
	if stats.MinMargin != 0 {
		t.Errorf("MinMargin = %.1f, want 0", stats.MinMargin)
	}
	if stats.MaxMargin != 0 {
		t.Errorf("MaxMargin = %.1f, want 0", stats.MaxMargin)
	}
}

func TestFailureStore_EmptyAnalyze(t *testing.T) {
	store := NewFailureStore("")
	analysis := store.Analyze()
	if analysis.TotalFailures != 0 {
		t.Errorf("TotalFailures = %d, want 0", analysis.TotalFailures)
	}
}

func TestIntentDistiller_InsufficientQuality(t *testing.T) {
	store := NewDistillationStore("")

	// All traces below quality threshold
	for i := 0; i < 5; i++ {
		tr := &TeacherTrace{
			Input:   "test",
			Intent:  "a",
			Quality: 0.3,
		}
		_ = store.RecordTrace(tr)
	}

	config := DefaultDistillationConfig()
	config.MinQuality = 0.7

	distiller := NewIntentDistiller(store, config)
	_, _, err := distiller.Distill()
	if err == nil {
		t.Error("expected error when no traces meet quality threshold")
	}
}

func TestPreferenceStore_FilterByMargin(t *testing.T) {
	store := NewPreferenceStore("")

	pairs := []*PreferencePair{
		{Input: "a", Margin: 0.05, Source: "human"},
		{Input: "b", Margin: 0.15, Source: "human"},
		{Input: "c", Margin: 0.5, Source: "auto_quality"},
		{Input: "d", Margin: 0.9, Source: "correction"},
	}
	for _, p := range pairs {
		_ = store.RecordPair(p)
	}

	filtered := store.FilterByMargin(0.1)
	if len(filtered) != 3 {
		t.Errorf("expected 3 pairs >= 0.1, got %d", len(filtered))
	}

	filtered = store.FilterByMargin(0.5)
	if len(filtered) != 2 {
		t.Errorf("expected 2 pairs >= 0.5, got %d", len(filtered))
	}
}

func TestFailureDetector_CleanResponse(t *testing.T) {
	fd := NewFailureDetector()

	// A good response should not trigger any failures
	failures := fd.Detect(
		"What is photosynthesis?",
		"Photosynthesis is the process by which green plants and certain other organisms transform light energy into chemical energy. During photosynthesis in green plants, light energy is captured and used to convert water, carbon dioxide, and minerals into oxygen and energy-rich organic compounds.",
		200,
	)

	// Should have no critical failures
	for _, f := range failures {
		if f.Severity == "critical" {
			t.Errorf("clean response should not have critical failures, got: %+v", f)
		}
	}
}

func TestTeacherTrace_TimestampSet(t *testing.T) {
	store := NewDistillationStore("")
	trace := &TeacherTrace{
		Input:   "test",
		Intent:  "test",
		Quality: 0.8,
	}
	_ = store.RecordTrace(trace)

	traces := store.FilterByQuality(0)
	if len(traces) != 1 {
		t.Fatalf("expected 1 trace, got %d", len(traces))
	}
	if traces[0].Timestamp.IsZero() {
		t.Error("timestamp should be set automatically")
	}
}

func TestPreferencePair_TimestampSet(t *testing.T) {
	store := NewPreferenceStore("")
	pair := &PreferencePair{
		Input:  "test",
		Margin: 0.5,
		Source: "human",
	}
	_ = store.RecordPair(pair)

	pairs, _ := store.LoadPairs()
	if len(pairs) != 1 {
		t.Fatalf("expected 1 pair, got %d", len(pairs))
	}
	if pairs[0].Timestamp.IsZero() {
		t.Error("timestamp should be set automatically")
	}
}

func TestAutoGeneratePairs_SingleResponse(t *testing.T) {
	outputs := []ScoredOutput{
		{Input: "What is Go?", Response: "Go is a language.", Quality: 0.5},
	}
	pairs := AutoGeneratePairs(outputs)
	if len(pairs) != 0 {
		t.Errorf("single response should produce 0 pairs, got %d", len(pairs))
	}
}

func TestAutoGeneratePairs_NegligibleMargin(t *testing.T) {
	outputs := []ScoredOutput{
		{Input: "What is Go?", Response: "Response A.", Quality: 0.50},
		{Input: "What is Go?", Response: "Response B.", Quality: 0.51},
	}
	pairs := AutoGeneratePairs(outputs)
	if len(pairs) != 0 {
		t.Errorf("negligible margin should produce 0 pairs, got %d", len(pairs))
	}
}

// ---------------------------------------------------------------------------
// Benchmarks
// ---------------------------------------------------------------------------

func BenchmarkDistillation(b *testing.B) {
	store := NewDistillationStore("")

	for i := 0; i < 100; i++ {
		tr := &TeacherTrace{
			Input:     "Explain the concept of neural networks and deep learning " + strings.Repeat("word ", i),
			Intent:    "explain",
			Slots:     map[string]string{"topic": "neural_networks"},
			Plan:      []string{"define neural networks", "explain layers", "describe training"},
			Response:  "Neural networks are computational models inspired by the brain.",
			Quality:   0.8 + float64(i%20)*0.01,
			Timestamp: time.Now(),
		}
		_ = store.RecordTrace(tr)
	}

	config := DefaultDistillationConfig()
	config.MinQuality = 0.7

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		distiller := NewIntentDistiller(store, config)
		_, _, _ = distiller.Distill()
	}
}

func BenchmarkPreferenceOptimize(b *testing.B) {
	store := NewPreferenceStore("")

	for i := 0; i < 50; i++ {
		pair := &PreferencePair{
			Input:    "Tell me about artificial intelligence and machine learning applications.",
			Chosen:   "Artificial intelligence encompasses machine learning, deep learning, and other techniques. For example, neural networks can classify images and natural language processing powers chatbots.",
			Rejected: "AI is a thing. It does stuff.",
			Margin:   0.6,
			Source:   "auto_quality",
		}
		_ = store.RecordPair(pair)
	}

	config := DefaultPreferenceConfig()
	config.MinMargin = 0.1

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		optimizer := NewPreferenceOptimizer(store, config)
		_, _, _ = optimizer.Optimize()
	}
}

func BenchmarkFailureDetect(b *testing.B) {
	fd := NewFailureDetector()

	input := "What is the theory of relativity?"
	response := "According to a 2020 study at MIT, the theory of relativity states that E=mc2. Dr. Einstein found that 99.9% of physicists agree. As published in the Physical Review, this is well established. Scientists have proven that time dilation occurs near massive objects."

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		fd.Detect(input, response, 500)
	}
}
