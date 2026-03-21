package cognitive

import (
	"strings"
	"testing"
)

// -----------------------------------------------------------------------
// Enhanced Reasoning Tests — new decomposition patterns and step types
// -----------------------------------------------------------------------

// helper builds a graph with philosophers and physics knowledge.
func buildTestGraph() (*CognitiveGraph, *SemanticEngine, *AnalogyEngine) {
	cg := NewCognitiveGraph("")
	se := NewSemanticEngine()

	// Philosophers
	cg.EnsureNode("Socrates", NodeEntity)
	cg.EnsureNode("philosopher", NodeConcept)
	cg.EnsureNode("questioning", NodeProperty)
	cg.EnsureNode("ethics", NodeConcept)
	cg.EnsureNode("wisdom", NodeProperty)
	cg.AddEdge("socrates", "philosopher", RelIsA, "test")
	cg.AddEdge("socrates", "questioning", RelHas, "test")
	cg.AddEdge("socrates", "ethics", RelRelatedTo, "test")
	cg.AddEdge("socrates", "wisdom", RelDescribedAs, "test")

	// Stoicism
	cg.EnsureNode("Stoicism", NodeConcept)
	cg.EnsureNode("philosophy", NodeConcept)
	cg.EnsureNode("resilience", NodeProperty)
	cg.EnsureNode("virtue", NodeProperty)
	cg.EnsureNode("inner peace", NodeProperty)
	cg.AddEdge("stoicism", "philosophy", RelIsA, "test")
	cg.AddEdge("stoicism", "resilience", RelCauses, "test")
	cg.AddEdge("stoicism", "virtue", RelHas, "test")
	cg.AddEdge("stoicism", "inner peace", RelUsedFor, "test")
	cg.AddEdge("stoicism", "ethics", RelRelatedTo, "test")

	// Physics
	cg.EnsureNode("Einstein", NodeEntity)
	cg.EnsureNode("physicist", NodeConcept)
	cg.EnsureNode("relativity", NodeConcept)
	cg.EnsureNode("physics", NodeConcept)
	cg.EnsureNode("science", NodeConcept)
	cg.AddEdge("einstein", "physicist", RelIsA, "test")
	cg.AddEdge("einstein", "relativity", RelCreatedBy, "test")
	cg.AddEdge("relativity", "physics", RelPartOf, "test")
	cg.AddEdge("physics", "science", RelIsA, "test")
	cg.AddEdge("einstein", "physics", RelRelatedTo, "test")

	// Newton
	cg.EnsureNode("Newton", NodeEntity)
	cg.EnsureNode("gravity", NodeConcept)
	cg.EnsureNode("mathematics", NodeConcept)
	cg.AddEdge("newton", "physicist", RelIsA, "test")
	cg.AddEdge("newton", "gravity", RelCreatedBy, "test")
	cg.AddEdge("gravity", "physics", RelPartOf, "test")
	cg.AddEdge("newton", "physics", RelRelatedTo, "test")
	cg.AddEdge("newton", "mathematics", RelRelatedTo, "test")

	// Quantum mechanics
	cg.EnsureNode("quantum mechanics", NodeConcept)
	cg.EnsureNode("uncertainty", NodeProperty)
	cg.EnsureNode("wave-particle duality", NodeConcept)
	cg.AddEdge("quantum mechanics", "physics", RelPartOf, "test")
	cg.AddEdge("quantum mechanics", "uncertainty", RelCauses, "test")
	cg.AddEdge("quantum mechanics", "wave-particle duality", RelCauses, "test")

	ae := NewAnalogyEngine(cg, se)
	return cg, se, ae
}

func TestReasoningWhatWouldXSay(t *testing.T) {
	cg, se, ae := buildTestGraph()

	re := NewReasoningEngine(cg, se)
	re.Analogy = ae

	chain := re.Reason("What would Socrates say about technology?")
	if chain == nil {
		t.Fatal("should produce a reasoning chain")
	}
	t.Logf("Trace:\n%s", chain.Trace)
	t.Logf("Answer: %s", chain.Answer)

	if chain.Answer == "" {
		t.Error("answer should not be empty")
	}
	// The answer should reference Socrates' attributes in some form
	lower := strings.ToLower(chain.Answer)
	if !strings.Contains(lower, "socrates") && !strings.Contains(lower, "questioning") &&
		!strings.Contains(lower, "wisdom") && !strings.Contains(lower, "ethics") {
		t.Errorf("answer should reference Socrates' principles, got %q", chain.Answer)
	}
}

func TestReasoningWhyImportant(t *testing.T) {
	cg, se, _ := buildTestGraph()

	re := NewReasoningEngine(cg, se)

	chain := re.Reason("Why is Stoicism important?")
	if chain == nil {
		t.Fatal("should produce a reasoning chain")
	}
	t.Logf("Trace:\n%s", chain.Trace)
	t.Logf("Answer: %s", chain.Answer)

	if chain.Answer == "" {
		t.Error("answer should not be empty")
	}
}

func TestReasoningImplications(t *testing.T) {
	cg, se, _ := buildTestGraph()

	re := NewReasoningEngine(cg, se)

	chain := re.Reason("What are the implications of quantum mechanics?")
	if chain == nil {
		t.Fatal("should produce a reasoning chain")
	}
	t.Logf("Trace:\n%s", chain.Trace)
	t.Logf("Answer: %s", chain.Answer)

	if chain.Answer == "" {
		t.Error("answer should not be empty")
	}
	// Should find uncertainty and/or wave-particle duality
	lower := strings.ToLower(chain.Answer)
	if !strings.Contains(lower, "uncertainty") && !strings.Contains(lower, "wave") {
		t.Errorf("answer should mention causal effects of quantum mechanics, got %q", chain.Answer)
	}
}

func TestReasoningPathFind(t *testing.T) {
	cg, se, _ := buildTestGraph()

	re := NewReasoningEngine(cg, se)

	chain := re.Reason("How is Einstein related to physics?")
	if chain == nil {
		t.Fatal("should produce a reasoning chain")
	}
	t.Logf("Trace:\n%s", chain.Trace)
	t.Logf("Answer: %s", chain.Answer)

	if chain.Answer == "" {
		t.Error("answer should not be empty")
	}
	lower := strings.ToLower(chain.Answer)
	if !strings.Contains(lower, "einstein") || !strings.Contains(lower, "physics") {
		t.Errorf("answer should mention both Einstein and physics, got %q", chain.Answer)
	}
}

func TestReasoningWhatIfRemoved(t *testing.T) {
	cg, se, _ := buildTestGraph()

	// Add some dependency edges so the counterfactual has material
	cg.AddEdge("socrates", "philosophy", RelCauses, "test")
	cg.AddEdge("philosophy", "ethics", RelCauses, "test")

	re := NewReasoningEngine(cg, se)

	chain := re.Reason("What if there was no Socrates?")
	if chain == nil {
		t.Fatal("should produce a reasoning chain")
	}
	t.Logf("Trace:\n%s", chain.Trace)
	t.Logf("Answer: %s", chain.Answer)

	if chain.Answer == "" {
		t.Error("answer should not be empty")
	}
}

func TestReasoningWhatInCommon(t *testing.T) {
	cg, se, _ := buildTestGraph()

	re := NewReasoningEngine(cg, se)

	chain := re.Reason("What do Einstein and Newton have in common?")
	if chain == nil {
		t.Fatal("should produce a reasoning chain")
	}
	t.Logf("Trace:\n%s", chain.Trace)
	t.Logf("Answer: %s", chain.Answer)

	if chain.Answer == "" {
		t.Error("answer should not be empty")
	}
	// Both are physicists related to physics
	lower := strings.ToLower(chain.Answer)
	if !strings.Contains(lower, "physicist") && !strings.Contains(lower, "physics") {
		t.Errorf("answer should mention shared physicist/physics trait, got %q", chain.Answer)
	}
}
