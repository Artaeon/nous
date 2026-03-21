package cognitive

import (
	"strings"
	"testing"
)

// helper to build a test graph with edges and nodes.
func setupAnalogyGraph() (*CognitiveGraph, *SemanticEngine) {
	cg := NewCognitiveGraph("")
	se := NewSemanticEngine()

	// Physics domain
	cg.EnsureNode("Einstein", NodeEntity)
	cg.EnsureNode("Newton", NodeEntity)
	cg.EnsureNode("physicist", NodeConcept)
	cg.EnsureNode("relativity", NodeConcept)
	cg.EnsureNode("classical mechanics", NodeConcept)
	cg.EnsureNode("physics", NodeConcept)

	cg.AddEdge("Einstein", "physicist", RelIsA, "test")
	cg.AddEdge("Newton", "physicist", RelIsA, "test")
	cg.AddEdge("Einstein", "relativity", RelHas, "test")
	cg.AddEdge("Newton", "classical mechanics", RelHas, "test")
	cg.AddEdge("Einstein", "physics", RelDomain, "test")
	cg.AddEdge("Newton", "physics", RelDomain, "test")
	cg.AddEdge("relativity", "physics", RelDomain, "test")
	cg.AddEdge("classical mechanics", "physics", RelDomain, "test")

	// Philosophy domain
	cg.EnsureNode("Socrates", NodeEntity)
	cg.EnsureNode("philosopher", NodeConcept)
	cg.EnsureNode("critical thinking", NodeConcept)
	cg.EnsureNode("ethics", NodeConcept)
	cg.EnsureNode("humility", NodeConcept)
	cg.EnsureNode("philosophy", NodeConcept)

	cg.AddEdge("Socrates", "philosopher", RelIsA, "test")
	cg.AddEdge("Socrates", "critical thinking", RelHas, "test")
	cg.AddEdge("Socrates", "ethics", RelHas, "test")
	cg.AddEdge("Socrates", "humility", RelDescribedAs, "test")
	cg.AddEdge("Socrates", "philosophy", RelDomain, "test")

	return cg, se
}

func TestAnalogyCompleteAnalogy(t *testing.T) {
	cg, se := setupAnalogyGraph()
	ae := NewAnalogyEngine(cg, se)

	// Einstein:relativity :: Newton:?
	// Einstein --has--> relativity, Newton --has--> classical mechanics
	result, conf := ae.CompleteAnalogy("Einstein", "relativity", "Newton")
	if result == "" {
		t.Fatal("CompleteAnalogy returned empty result")
	}
	if conf <= 0 {
		t.Fatal("CompleteAnalogy returned zero confidence")
	}
	if !strings.Contains(strings.ToLower(result), "classical mechanics") {
		t.Errorf("expected 'classical mechanics', got %q", result)
	}
	t.Logf("Einstein:relativity :: Newton:%s (conf=%.2f)", result, conf)
}

func TestAnalogyRelationalSkeleton(t *testing.T) {
	cg, se := setupAnalogyGraph()
	ae := NewAnalogyEngine(cg, se)

	skeleton := ae.relationalSkeleton("einstein")
	if len(skeleton) == 0 {
		t.Fatal("relationalSkeleton returned empty map")
	}

	// Einstein has outgoing: is_a (physicist), has (relativity), domain (physics)
	if skeleton["out:is_a"] != 1 {
		t.Errorf("expected out:is_a=1, got %d", skeleton["out:is_a"])
	}
	if skeleton["out:has"] != 1 {
		t.Errorf("expected out:has=1, got %d", skeleton["out:has"])
	}
	if skeleton["out:domain"] != 1 {
		t.Errorf("expected out:domain=1, got %d", skeleton["out:domain"])
	}
	t.Logf("Einstein skeleton: %v", skeleton)
}

func TestAnalogyStructuralSimilarity(t *testing.T) {
	cg, se := setupAnalogyGraph()
	ae := NewAnalogyEngine(cg, se)

	skelA := ae.relationalSkeleton("einstein")
	skelB := ae.relationalSkeleton("newton")

	sim := ae.structuralSimilarity(skelA, skelB)
	if sim < 0.5 {
		t.Errorf("expected high similarity between Einstein and Newton, got %.2f", sim)
	}
	t.Logf("structural similarity Einstein/Newton: %.4f", sim)

	// Compare with a very different node.
	skelC := ae.relationalSkeleton("relativity")
	simDiff := ae.structuralSimilarity(skelA, skelC)
	if simDiff >= sim {
		t.Errorf("expected lower similarity for Einstein/relativity (%.2f) than Einstein/Newton (%.2f)", simDiff, sim)
	}
	t.Logf("structural similarity Einstein/relativity: %.4f", simDiff)

	// Edge cases.
	empty := make(map[string]int)
	if ae.structuralSimilarity(empty, skelA) != 0 {
		t.Error("expected 0 similarity with empty skeleton")
	}
	if ae.structuralSimilarity(skelA, skelA) < 0.99 {
		t.Error("expected ~1.0 self-similarity")
	}
}

func TestAnalogyApplyPrinciples(t *testing.T) {
	cg, se := setupAnalogyGraph()
	ae := NewAnalogyEngine(cg, se)

	result := ae.ApplyPrinciples("Socrates", "technology")
	if result == "" {
		t.Fatal("ApplyPrinciples returned empty string")
	}
	if !strings.Contains(result, "Socrates") {
		t.Error("expected result to mention Socrates")
	}
	if !strings.Contains(result, "technology") {
		t.Error("expected result to mention technology")
	}
	t.Logf("ApplyPrinciples output: %s", result)

	// Non-existent entity returns empty.
	if ae.ApplyPrinciples("Nobody", "anything") != "" {
		t.Error("expected empty result for unknown entity")
	}
}

func TestAnalogyMapDomains(t *testing.T) {
	cg, se := setupAnalogyGraph()

	// Add a second science domain to map against.
	cg.EnsureNode("biology", NodeConcept)
	cg.EnsureNode("Darwin", NodeEntity)
	cg.EnsureNode("biologist", NodeConcept)
	cg.EnsureNode("evolution", NodeConcept)

	cg.AddEdge("Darwin", "biologist", RelIsA, "test")
	cg.AddEdge("Darwin", "evolution", RelHas, "test")
	cg.AddEdge("Darwin", "biology", RelDomain, "test")
	cg.AddEdge("evolution", "biology", RelDomain, "test")

	ae := NewAnalogyEngine(cg, se)

	mappings := ae.MapDomains("physics", "biology")
	if len(mappings) == 0 {
		t.Fatal("MapDomains returned no mappings")
	}

	t.Log("Domain mappings (physics → biology):")
	for _, m := range mappings {
		t.Logf("  %s → %s (conf=%.2f)", m.SourceNode, m.TargetNode, m.Confidence)
	}

	// Expect at least one mapping with non-trivial confidence.
	found := false
	for _, m := range mappings {
		if m.Confidence > 0.1 {
			found = true
			break
		}
	}
	if !found {
		t.Error("expected at least one mapping with confidence > 0.1")
	}
}

func TestAnalogyFindAnalogy(t *testing.T) {
	cg, se := setupAnalogyGraph()
	ae := NewAnalogyEngine(cg, se)

	// "Einstein is to relativity" — find analogous pair in physics domain.
	result := ae.FindAnalogy("Einstein", "relativity", "physics")
	if result == nil {
		t.Fatal("FindAnalogy returned nil")
	}
	if len(result.Mappings) == 0 {
		t.Fatal("FindAnalogy returned no mappings")
	}
	t.Logf("FindAnalogy: %s", result.Explanation)

	// Non-existent relation returns nil.
	nilResult := ae.FindAnalogy("Einstein", "biology", "physics")
	if nilResult != nil {
		t.Error("expected nil for unconnected pair")
	}
}

func TestAnalogyExtractSubgraph(t *testing.T) {
	cg, se := setupAnalogyGraph()
	ae := NewAnalogyEngine(cg, se)

	nodes, edges := ae.extractSubgraph("einstein", 1)
	if len(nodes) == 0 {
		t.Fatal("extractSubgraph returned no nodes")
	}
	if len(edges) == 0 {
		t.Fatal("extractSubgraph returned no edges")
	}
	// Einstein at depth 1 should reach physicist, relativity, physics.
	if len(nodes) < 4 {
		t.Errorf("expected at least 4 nodes (einstein + 3 neighbors), got %d", len(nodes))
	}
	t.Logf("subgraph from einstein (depth=1): %d nodes, %d edges", len(nodes), len(edges))

	// Non-existent root.
	n2, e2 := ae.extractSubgraph("nonexistent", 2)
	if len(n2) != 0 || len(e2) != 0 {
		t.Error("expected empty result for nonexistent root")
	}
}
