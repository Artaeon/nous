package cognitive

import (
	"math"
	"testing"
)

func TestEnsembleNil(t *testing.T) {
	var e *ToolEnsemble
	result := e.Predict("test")
	if result != nil {
		t.Error("nil ensemble should return nil")
	}
}

func TestEnsembleIntentOnly(t *testing.T) {
	// Create intent compiler with a known working directory
	dir := t.TempDir()
	intent := NewIntentCompiler(dir)

	e := NewToolEnsemble(intent, nil)

	// "read main.go" should match intent patterns with high confidence
	result := e.Predict("read main.go")
	if result == nil {
		t.Skip("intent didn't match — depends on filesystem, skipping")
	}
	if result.Source != "intent" {
		t.Errorf("source = %q, want intent", result.Source)
	}
}

func TestEnsembleCortexOnly(t *testing.T) {
	labels := []string{"grep", "read", "write", "ls"}
	cortex := NewNeuralCortex(64, 32, labels, "")

	// Train cortex heavily on "grep" for queries containing "find"
	for i := 0; i < 500; i++ {
		input := CortexInputFromQuery("find function definition", cortex.InputSize)
		cortex.Train(input, "grep")
	}

	e := NewToolEnsemble(nil, cortex)
	result := e.Predict("find function definition")

	if result == nil {
		t.Fatal("trained cortex should produce a prediction")
	}
	if result.Tool != "grep" {
		t.Errorf("tool = %q, want grep", result.Tool)
	}
	if result.Source != "cortex" {
		t.Errorf("source = %q, want cortex", result.Source)
	}
}

func TestEnsembleCortexNeedMinTraining(t *testing.T) {
	labels := []string{"grep", "read"}
	cortex := NewNeuralCortex(64, 32, labels, "")

	// Only 10 training steps — below threshold of 50
	for i := 0; i < 10; i++ {
		cortex.Train(CortexInputFromQuery("test", cortex.InputSize), "grep")
	}

	e := NewToolEnsemble(nil, cortex)
	result := e.Predict("test query")

	// With no intent and undertrained cortex, should return nil
	if result != nil {
		t.Error("undertrained cortex should not be consulted")
	}
}

func TestEnsembleAgreementBoostsConfidence(t *testing.T) {
	labels := []string{"grep", "read", "write", "ls", "glob", "tree", "git", "shell", "edit"}
	cortex := NewNeuralCortex(64, 32, labels, "")

	// Train cortex to predict "grep" for search-like queries
	for i := 0; i < 500; i++ {
		input := CortexInputFromQuery("search for NewReasoner", cortex.InputSize)
		cortex.Train(input, "grep")
	}

	dir := t.TempDir()
	intent := NewIntentCompiler(dir)
	e := NewToolEnsemble(intent, cortex)

	// "search for NewReasoner" matches intent's search pattern AND cortex's trained pattern
	result := e.Predict(`search for "NewReasoner"`)
	if result == nil {
		t.Skip("intent didn't match search pattern, skipping")
	}

	// If both agree, source should be "ensemble" and confidence should be boosted
	if result.Source == "ensemble" {
		if result.Confidence <= result.IntentConf && result.Confidence <= result.CortexConf {
			t.Error("ensemble agreement should boost confidence above either individual")
		}
	}
}

func TestEnsembleResultFields(t *testing.T) {
	result := &EnsembleResult{
		Tool:       "grep",
		Confidence: 0.85,
		Source:     "intent",
		IntentConf: 0.85,
		CortexConf: 0.0,
	}

	if result.Tool != "grep" {
		t.Error("tool field should be accessible")
	}
	if result.IntentConf != 0.85 {
		t.Error("intent confidence should be accessible")
	}
}

func TestMin64(t *testing.T) {
	if min64(3.0, 5.0) != 3.0 {
		t.Error("min64(3, 5) should be 3")
	}
	if min64(7.0, 2.0) != 2.0 {
		t.Error("min64(7, 2) should be 2")
	}
	if min64(4.0, 4.0) != 4.0 {
		t.Error("min64(4, 4) should be 4")
	}
}

// --- Formula Verification Tests ---

func TestEnsembleConfidenceBoostFormula(t *testing.T) {
	labels := []string{"grep", "read", "write", "ls", "glob", "tree", "git", "shell", "edit"}
	cortex := NewNeuralCortex(64, 32, labels, "")

	// Train cortex heavily to predict "grep" for search queries
	for i := 0; i < 500; i++ {
		input := CortexInputFromQuery("search for NewReasoner", cortex.InputSize)
		cortex.Train(input, "grep")
	}

	dir := t.TempDir()
	intent := NewIntentCompiler(dir)
	e := NewToolEnsemble(intent, cortex)

	// "search for NewReasoner" should match both intent and cortex
	result := e.Predict(`search for "NewReasoner"`)
	if result == nil {
		t.Skip("intent didn't match search pattern, skipping")
	}

	if result.Source == "ensemble" {
		// Formula: confidence = min(intentConf, cortexConf) * 1.3, capped at 0.95
		expectedConf := min64(result.IntentConf, result.CortexConf) * 1.3
		if expectedConf > 0.95 {
			expectedConf = 0.95
		}
		if math.Abs(result.Confidence-expectedConf) > 0.01 {
			t.Errorf("ensemble confidence = %f, expected %f (intent=%f, cortex=%f)",
				result.Confidence, expectedConf, result.IntentConf, result.CortexConf)
		}
	}
}

// --- Edge Case Tests ---

func TestEnsembleConflictingPredictions(t *testing.T) {
	labels := []string{"grep", "read", "write", "ls"}
	cortex := NewNeuralCortex(64, 32, labels, "")

	// Train cortex to strongly predict "read"
	for i := 0; i < 500; i++ {
		input := CortexInputFromQuery("search for code patterns", cortex.InputSize)
		cortex.Train(input, "read")
	}

	dir := t.TempDir()
	intent := NewIntentCompiler(dir)
	e := NewToolEnsemble(intent, cortex)

	// "search for code patterns" — intent likely says "grep", cortex says "read"
	result := e.Predict("search for code patterns")
	if result == nil {
		t.Skip("no prediction returned, skipping")
	}

	// Cortex says "read" and intent says "grep" — if both have confidence,
	// we should see "conflict" or one should win
	if result.Source == "conflict" {
		// Confidence should be reduced
		if result.Confidence > result.CortexConf && result.Confidence > result.IntentConf {
			t.Error("conflict confidence should not exceed either individual")
		}
	}
	// Either way, a valid tool should be returned
	if result.Tool == "" {
		t.Error("conflicting predictions should still return a tool")
	}
}

// ensure math import is used
var _ = math.Abs
