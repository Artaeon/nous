package cognitive

import (
	"strings"
	"testing"
)

// -----------------------------------------------------------------------
// Helper: build a small graph for testing synthesis.
// -----------------------------------------------------------------------

func buildSynthesisTestGraph() *CognitiveGraph {
	g := NewCognitiveGraph("")

	// Category hierarchy: Rust is_a programming language
	g.AddEdge("Rust", "programming language", RelIsA, "test")
	// Programming language has properties
	g.AddEdge("programming language", "syntax", RelHas, "test")
	g.AddEdge("programming language", "compiler", RelHas, "test")
	g.AddEdge("programming language", "type system", RelHas, "test")
	g.AddEdge("programming language", "software development", RelUsedFor, "test")

	// Sibling: Go is also a programming language with known properties
	g.AddEdge("Go", "programming language", RelIsA, "test")
	g.AddEdge("Go", "concurrency", RelHas, "test")
	g.AddEdge("Go", "garbage collector", RelHas, "test")
	g.AddEdge("Go", "Google", RelCreatedBy, "test")

	// Similar_to relationship
	g.AddEdge("Rust", "C++", RelSimilarTo, "test")
	g.AddEdge("C++", "systems programming", RelUsedFor, "test")
	g.AddEdge("C++", "manual memory management", RelHas, "test")

	// Causal chain: deforestation causes erosion causes flooding
	g.AddEdge("deforestation", "soil erosion", RelCauses, "test")
	g.AddEdge("soil erosion", "flooding", RelCauses, "test")

	// Opposite: democracy opposite_of authoritarianism
	g.AddEdge("democracy", "authoritarianism", RelOppositeOf, "test")
	g.AddEdge("authoritarianism", "censorship", RelHas, "test")
	g.AddEdge("authoritarianism", "central control", RelHas, "test")

	// For compound topic decomposition: "quantum" and "computing" exist
	g.AddEdge("quantum", "physics", RelPartOf, "test")
	g.AddEdge("quantum", "superposition", RelHas, "test")
	g.AddEdge("computing", "calculation", RelUsedFor, "test")
	g.AddEdge("computing", "algorithms", RelHas, "test")

	// Compositional: a topic with both incoming and outgoing
	g.AddEdge("Vienna", "Austria", RelLocatedIn, "test")
	g.AddEdge("Mozart", "Vienna", RelLocatedIn, "test")

	return g
}

// -----------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------

func TestSynthesize_Generalization(t *testing.T) {
	g := buildSynthesisTestGraph()
	ks := NewKnowledgeSynthesizer(g, nil)

	result := ks.Synthesize("Rust")
	if result == nil {
		t.Fatal("expected non-nil result")
	}

	// Should find generalization via "Rust is_a programming language"
	var found bool
	for _, sk := range result.Synthesized {
		if sk.Strategy == StratGeneralization {
			found = true
			if !strings.Contains(sk.Qualifier, "programming language") {
				t.Errorf("generalization qualifier should mention parent category, got: %s", sk.Qualifier)
			}
			if !strings.Contains(sk.Claim, "programming language") {
				t.Errorf("generalization claim should mention parent, got: %s", sk.Claim)
			}
			if sk.Confidence > maxSynthesisConfidence {
				t.Errorf("confidence %f exceeds cap %f", sk.Confidence, maxSynthesisConfidence)
			}
			if sk.Caveat == "" {
				t.Error("generalization should include a caveat")
			}
			if len(sk.Evidence) == 0 {
				t.Error("generalization should include evidence")
			}
			break
		}
	}
	if !found {
		t.Error("expected at least one generalization synthesis for Rust")
	}
}

func TestSynthesize_Decomposition(t *testing.T) {
	g := buildSynthesisTestGraph()
	ks := NewKnowledgeSynthesizer(g, nil)

	result := ks.Synthesize("quantum computing")
	if result == nil {
		t.Fatal("expected non-nil result")
	}

	var found bool
	for _, sk := range result.Synthesized {
		if sk.Strategy == StratDecomposition {
			found = true
			if !strings.Contains(sk.Qualifier, "quantum") || !strings.Contains(sk.Qualifier, "computing") {
				t.Errorf("decomposition qualifier should mention both components, got: %s", sk.Qualifier)
			}
			if sk.Confidence > maxSynthesisConfidence {
				t.Errorf("confidence %f exceeds cap %f", sk.Confidence, maxSynthesisConfidence)
			}
			if sk.Caveat == "" {
				t.Error("decomposition should include a caveat")
			}
			break
		}
	}
	if !found {
		t.Error("expected decomposition synthesis for 'quantum computing'")
	}
}

func TestSynthesize_WithGraph(t *testing.T) {
	g := buildSynthesisTestGraph()
	ae := NewAnalogyEngine(g, nil)
	ks := NewKnowledgeSynthesizer(g, ae)

	// Test Rust — should get generalization + analogy + compositional
	result := ks.Synthesize("Rust")
	if result == nil {
		t.Fatal("expected non-nil result")
	}
	if len(result.Synthesized) == 0 {
		t.Fatal("expected at least one synthesized claim for Rust")
	}

	strategies := make(map[SynthesisStrategy]bool)
	for _, sk := range result.Synthesized {
		strategies[sk.Strategy] = true
		// Every claim must have qualifier and caveat.
		if sk.Qualifier == "" {
			t.Errorf("claim missing qualifier: %s", sk.Claim)
		}
		if sk.Caveat == "" {
			t.Errorf("claim missing caveat: %s", sk.Claim)
		}
	}

	if !strategies[StratGeneralization] {
		t.Error("expected generalization strategy for Rust")
	}

	// Overall confidence should be set.
	if result.OverallConf <= 0 {
		t.Error("overall confidence should be positive")
	}
	if result.OverallConf > maxSynthesisConfidence {
		t.Errorf("overall confidence %f exceeds cap %f", result.OverallConf, maxSynthesisConfidence)
	}

	// Explanation should be non-empty.
	if result.Explanation == "" {
		t.Error("expected non-empty explanation")
	}

	// Test causal chain with deforestation.
	causalResult := ks.Synthesize("deforestation")
	var causalFound bool
	for _, sk := range causalResult.Synthesized {
		if sk.Strategy == StratCausalChain {
			causalFound = true
			if !strings.Contains(sk.Claim, "flooding") {
				t.Errorf("causal chain should mention flooding, got: %s", sk.Claim)
			}
			break
		}
	}
	if !causalFound {
		t.Error("expected causal chain synthesis for deforestation")
	}

	// Test contrastive with democracy.
	contrastResult := ks.Synthesize("democracy")
	var contrastFound bool
	for _, sk := range contrastResult.Synthesized {
		if sk.Strategy == StratContrastive {
			contrastFound = true
			if !strings.Contains(sk.Claim, "authoritarianism") {
				t.Errorf("contrastive should mention authoritarianism, got: %s", sk.Claim)
			}
			break
		}
	}
	if !contrastFound {
		t.Error("expected contrastive synthesis for democracy")
	}
}

func TestSynthesize_EmptyGraph(t *testing.T) {
	g := NewCognitiveGraph("")
	ks := NewKnowledgeSynthesizer(g, nil)

	result := ks.Synthesize("unknown topic")
	if result == nil {
		t.Fatal("expected non-nil result even for empty graph")
	}
	if len(result.Synthesized) != 0 {
		t.Errorf("expected no synthesized claims from empty graph, got %d", len(result.Synthesized))
	}
	if result.Explanation == "" {
		t.Error("expected non-empty explanation even when nothing is found")
	}

	// Nil graph should not panic.
	ksNil := NewKnowledgeSynthesizer(nil, nil)
	nilResult := ksNil.Synthesize("anything")
	if nilResult == nil {
		t.Fatal("expected non-nil result even with nil graph")
	}
	if len(nilResult.Synthesized) != 0 {
		t.Error("expected no claims with nil graph")
	}
}

func TestFormatSynthesis(t *testing.T) {
	g := buildSynthesisTestGraph()
	ks := NewKnowledgeSynthesizer(g, nil)

	result := ks.Synthesize("Rust")
	if len(result.Synthesized) == 0 {
		t.Fatal("expected synthesized claims to format")
	}

	formatted := ks.FormatSynthesis(result)
	if formatted == "" {
		t.Fatal("expected non-empty formatted output")
	}

	// Should contain the "I can reason about it" preamble.
	if !strings.Contains(formatted, "reason") {
		t.Error("formatted output should indicate reasoning")
	}

	// Should contain confidence levels.
	if !strings.Contains(formatted, "Confidence:") {
		t.Error("formatted output should show confidence")
	}

	// Test formatting with no results.
	emptyResult := &SynthesisResult{
		Topic: "nothing",
	}
	emptyFormatted := ks.FormatSynthesis(emptyResult)
	if !strings.Contains(emptyFormatted, "nothing") {
		t.Error("empty format should still mention the topic")
	}
	if !strings.Contains(emptyFormatted, "couldn't find") {
		t.Errorf("empty format should indicate failure, got: %s", emptyFormatted)
	}
}

func TestShouldSynthesize(t *testing.T) {
	g := buildSynthesisTestGraph()
	ks := NewKnowledgeSynthesizer(g, nil)

	// Should NOT synthesize when we have enough direct facts.
	if ks.ShouldSynthesize("Rust", 5) {
		t.Error("should not synthesize when directFactCount >= 2")
	}
	if ks.ShouldSynthesize("Rust", 2) {
		t.Error("should not synthesize when directFactCount == 2")
	}

	// Should synthesize when direct facts are insufficient but adjacent
	// knowledge exists.
	if !ks.ShouldSynthesize("Rust", 0) {
		t.Error("should synthesize for Rust with 0 direct facts")
	}
	if !ks.ShouldSynthesize("Rust", 1) {
		t.Error("should synthesize for Rust with 1 direct fact")
	}

	// Should NOT synthesize when there's no adjacent knowledge at all.
	if ks.ShouldSynthesize("completely unknown xyz topic", 0) {
		t.Error("should not synthesize when no adjacent knowledge exists")
	}

	// Compound topics should be synthesizable via decomposition.
	if !ks.ShouldSynthesize("quantum computing", 0) {
		t.Error("should synthesize 'quantum computing' via component words")
	}

	// Nil graph should not synthesize.
	ksNil := NewKnowledgeSynthesizer(nil, nil)
	if ksNil.ShouldSynthesize("anything", 0) {
		t.Error("should not synthesize with nil graph")
	}
}

func TestSynthesisConfidenceCapped(t *testing.T) {
	g := buildSynthesisTestGraph()
	ae := NewAnalogyEngine(g, nil)
	ks := NewKnowledgeSynthesizer(g, ae)

	// Test every topic in our graph.
	topics := []string{"Rust", "Go", "deforestation", "democracy", "quantum computing", "Vienna"}

	for _, topic := range topics {
		result := ks.Synthesize(topic)
		if result == nil {
			continue
		}

		for _, sk := range result.Synthesized {
			if sk.Confidence > maxSynthesisConfidence {
				t.Errorf("topic %q: claim confidence %f exceeds cap %f: %s",
					topic, sk.Confidence, maxSynthesisConfidence, sk.Claim)
			}
			if sk.Confidence < 0 {
				t.Errorf("topic %q: claim confidence %f is negative: %s",
					topic, sk.Confidence, sk.Claim)
			}
		}

		if result.OverallConf > maxSynthesisConfidence {
			t.Errorf("topic %q: overall confidence %f exceeds cap %f",
				topic, result.OverallConf, maxSynthesisConfidence)
		}
	}
}

func TestSynthesisStrategy_String(t *testing.T) {
	tests := []struct {
		strat SynthesisStrategy
		want  string
	}{
		{StratAnalogy, "analogy"},
		{StratDecomposition, "decomposition"},
		{StratGeneralization, "generalization"},
		{StratCausalChain, "causal chain"},
		{StratContrastive, "contrastive reasoning"},
		{StratCompositional, "compositional reasoning"},
		{SynthesisStrategy(99), "unknown"},
	}

	for _, tt := range tests {
		got := tt.strat.String()
		if got != tt.want {
			t.Errorf("Strategy(%d).String() = %q, want %q", tt.strat, got, tt.want)
		}
	}
}

func TestSynthesize_CausalChainDirect(t *testing.T) {
	g := NewCognitiveGraph("")
	g.AddEdge("smoking", "lung damage", RelCauses, "test")
	g.AddEdge("lung damage", "breathing difficulty", RelCauses, "test")

	ks := NewKnowledgeSynthesizer(g, nil)
	result := ks.Synthesize("smoking")

	var found bool
	for _, sk := range result.Synthesized {
		if sk.Strategy == StratCausalChain {
			found = true
			if !strings.Contains(sk.Claim, "breathing difficulty") {
				t.Errorf("causal chain should reach breathing difficulty, got: %s", sk.Claim)
			}
			if !strings.Contains(sk.Claim, "lung damage") {
				t.Errorf("causal chain should mention intermediate, got: %s", sk.Claim)
			}
			if len(sk.Evidence) != 2 {
				t.Errorf("causal chain should have 2 evidence items, got %d", len(sk.Evidence))
			}
		}
	}
	if !found {
		t.Error("expected causal chain synthesis for smoking")
	}
}

func TestSynthesize_ContrastiveDirect(t *testing.T) {
	g := NewCognitiveGraph("")
	g.AddEdge("introvert", "extrovert", RelOppositeOf, "test")
	g.AddEdge("extrovert", "social gatherings", RelPrefers, "test")
	g.AddEdge("extrovert", "large groups", RelHas, "test")

	ks := NewKnowledgeSynthesizer(g, nil)
	result := ks.Synthesize("introvert")

	var found bool
	for _, sk := range result.Synthesized {
		if sk.Strategy == StratContrastive {
			found = true
			if !strings.Contains(sk.Claim, "extrovert") {
				t.Errorf("contrastive should mention opposite, got: %s", sk.Claim)
			}
		}
	}
	if !found {
		t.Error("expected contrastive synthesis for introvert")
	}
}

func TestCapConfidence(t *testing.T) {
	if c := capConfidence(0.9); c != maxSynthesisConfidence {
		t.Errorf("capConfidence(0.9) = %f, want %f", c, maxSynthesisConfidence)
	}
	if c := capConfidence(0.5); c != 0.5 {
		t.Errorf("capConfidence(0.5) = %f, want 0.5", c)
	}
	if c := capConfidence(-0.1); c != 0 {
		t.Errorf("capConfidence(-0.1) = %f, want 0", c)
	}
}
