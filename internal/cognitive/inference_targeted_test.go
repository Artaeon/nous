package cognitive

import (
	"testing"
)

// TestTransitiveFrom verifies targeted transitive inference from specific nodes.
func TestTransitiveFrom(t *testing.T) {
	g := NewCognitiveGraph("")

	// A is_a B, B is_a C
	aID := g.EnsureNode("Sparrow", NodeEntity)
	bID := g.EnsureNode("Bird", NodeConcept)
	cID := g.EnsureNode("Animal", NodeConcept)

	g.AddEdge("Sparrow", "Bird", RelIsA, "test")
	g.AddEdge("Bird", "Animal", RelIsA, "test")

	ie := NewInferenceEngine(g)
	infs := ie.TransitiveFrom([]string{aID})

	if len(infs) == 0 {
		t.Fatal("expected at least one transitive inference, got none")
	}

	found := false
	for _, inf := range infs {
		if inf.Subject == "Sparrow" && inf.Relation == RelIsA && inf.Object == "Animal" {
			found = true
			if inf.Confidence <= 0 || inf.Confidence > 1.0 {
				t.Errorf("unexpected confidence: %f", inf.Confidence)
			}
			if inf.Reason == "" {
				t.Error("expected non-empty reason")
			}
			t.Logf("inferred: %s %s %s (conf=%.3f, reason=%s)",
				inf.Subject, inf.Relation, inf.Object, inf.Confidence, inf.Reason)
		}
	}
	if !found {
		t.Error("expected inference: Sparrow is_a Animal")
	}

	// Running again should not produce duplicates (edge already exists).
	infs2 := ie.TransitiveFrom([]string{aID})
	if len(infs2) != 0 {
		t.Errorf("expected no new inferences on second run, got %d", len(infs2))
	}

	// Verify the edge was actually added to the graph.
	edges := g.EdgesFrom("Sparrow")
	foundEdge := false
	for _, e := range edges {
		if e.To == cID && e.Relation == RelIsA && e.Inferred {
			foundEdge = true
		}
	}
	if !foundEdge {
		t.Error("expected inferred edge Sparrow->Animal in graph")
	}

	_ = bID // used in setup
}

// TestTransitiveFrom_ThreeHops verifies the 3-hop depth limit.
func TestTransitiveFrom_ThreeHops(t *testing.T) {
	g := NewCognitiveGraph("")

	g.EnsureNode("A", NodeEntity)
	g.EnsureNode("B", NodeConcept)
	g.EnsureNode("C", NodeConcept)
	g.EnsureNode("D", NodeConcept)
	g.EnsureNode("E", NodeConcept)

	g.AddEdge("A", "B", RelIsA, "test")
	g.AddEdge("B", "C", RelIsA, "test")
	g.AddEdge("C", "D", RelIsA, "test")
	g.AddEdge("D", "E", RelIsA, "test") // 4th hop — should NOT be reached

	ie := NewInferenceEngine(g)
	infs := ie.TransitiveFrom([]string{nodeID("A")})

	for _, inf := range infs {
		if inf.Object == "E" {
			t.Error("should NOT infer A is_a E — exceeds 3 hop limit")
		}
	}

	// A->C (2 hops) and A->D (3 hops) should be inferred
	subjects := map[string]bool{}
	for _, inf := range infs {
		if inf.Subject == "A" {
			subjects[inf.Object] = true
		}
	}
	if !subjects["C"] {
		t.Error("expected A is_a C (2 hops)")
	}
	if !subjects["D"] {
		t.Error("expected A is_a D (3 hops)")
	}
}

// TestAnalogicalFrom verifies analogical inference transfers properties between siblings.
func TestAnalogicalFrom(t *testing.T) {
	g := NewCognitiveGraph("")

	g.EnsureNode("Python", NodeEntity)
	g.EnsureNode("Go", NodeEntity)
	g.EnsureNode("programming language", NodeConcept)
	g.EnsureNode("software development", NodeConcept)

	g.AddEdge("Python", "programming language", RelIsA, "test")
	g.AddEdge("Go", "programming language", RelIsA, "test")
	g.AddEdge("Python", "software development", RelDomain, "test")

	ie := NewInferenceEngine(g)
	infs := ie.AnalogicalFrom([]string{nodeID("Python")})

	if len(infs) == 0 {
		t.Fatal("expected at least one analogical inference, got none")
	}

	found := false
	for _, inf := range infs {
		if inf.Subject == "Go" && inf.Relation == RelDomain && inf.Object == "software development" {
			found = true
			if inf.Confidence <= 0 {
				t.Errorf("unexpected confidence: %f", inf.Confidence)
			}
			t.Logf("inferred: %s %s %s (conf=%.3f, reason=%s)",
				inf.Subject, inf.Relation, inf.Object, inf.Confidence, inf.Reason)
		}
	}
	if !found {
		t.Error("expected inference: Go domain software development")
	}

	// Second run should not duplicate
	infs2 := ie.AnalogicalFrom([]string{nodeID("Python")})
	if len(infs2) != 0 {
		t.Errorf("expected no new inferences on second run, got %d", len(infs2))
	}
}

// TestInferAt verifies the combined convenience method deduplicates results.
func TestInferAt(t *testing.T) {
	g := NewCognitiveGraph("")

	g.EnsureNode("Cat", NodeEntity)
	g.EnsureNode("Dog", NodeEntity)
	g.EnsureNode("Pet", NodeConcept)
	g.EnsureNode("Mammal", NodeConcept)
	g.EnsureNode("companionship", NodeConcept)

	g.AddEdge("Cat", "Pet", RelIsA, "test")
	g.AddEdge("Dog", "Pet", RelIsA, "test")
	g.AddEdge("Pet", "Mammal", RelIsA, "test")          // transitive chain
	g.AddEdge("Cat", "companionship", RelRelatedTo, "test") // analogical transfer

	ie := NewInferenceEngine(g)
	infs := ie.InferAt([]string{nodeID("Cat")})

	if len(infs) == 0 {
		t.Fatal("expected inferences from InferAt, got none")
	}

	// Check for transitive: Cat is_a Mammal
	foundTransitive := false
	// Check for analogical: Dog related_to companionship
	foundAnalogical := false

	seen := make(map[string]bool)
	for _, inf := range infs {
		key := inf.Subject + "|" + string(inf.Relation) + "|" + inf.Object
		if seen[key] {
			t.Errorf("duplicate inference: %s", key)
		}
		seen[key] = true

		if inf.Subject == "Cat" && inf.Relation == RelIsA && inf.Object == "Mammal" {
			foundTransitive = true
		}
		if inf.Subject == "Dog" && inf.Relation == RelRelatedTo && inf.Object == "companionship" {
			foundAnalogical = true
		}
	}

	if !foundTransitive {
		t.Error("expected transitive inference: Cat is_a Mammal")
	}
	if !foundAnalogical {
		t.Error("expected analogical inference: Dog related_to companionship")
	}

	t.Logf("InferAt returned %d unique inferences", len(infs))
	for _, inf := range infs {
		t.Logf("  %s %s %s (conf=%.3f)", inf.Subject, inf.Relation, inf.Object, inf.Confidence)
	}
}
