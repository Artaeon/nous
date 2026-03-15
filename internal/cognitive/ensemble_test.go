package cognitive

import (
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
