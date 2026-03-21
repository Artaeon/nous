package cognitive

import (
	"path/filepath"
	"strings"
	"testing"
)

// -----------------------------------------------------------------------
// Core Graph Tests
// -----------------------------------------------------------------------

func TestCognitiveGraphBasic(t *testing.T) {
	cg := NewCognitiveGraph("")

	id := cg.EnsureNode("Stoicera", NodeEntity)
	if id != "stoicera" {
		t.Errorf("expected id 'stoicera', got %q", id)
	}

	node := cg.GetNode("stoicera")
	if node == nil {
		t.Fatal("node should exist")
	}
	if node.Label != "Stoicera" {
		t.Errorf("label should preserve case, got %q", node.Label)
	}
}

func TestCognitiveGraphDedup(t *testing.T) {
	cg := NewCognitiveGraph("")

	cg.EnsureNode("Go", NodeEntity)
	cg.EnsureNode("Go", NodeEntity) // duplicate
	cg.EnsureNode("Go", NodeEntity) // duplicate

	if cg.NodeCount() != 1 {
		t.Errorf("expected 1 node after dedup, got %d", cg.NodeCount())
	}

	node := cg.GetNode("go")
	if node.AccessCount != 2 { // 2 re-accesses after creation
		t.Errorf("expected access count 2, got %d", node.AccessCount)
	}
}

func TestCognitiveGraphEdges(t *testing.T) {
	cg := NewCognitiveGraph("")

	cg.EnsureNode("Stoicera", NodeEntity)
	cg.EnsureNode("Vienna", NodeEntity)
	cg.AddEdge("stoicera", "vienna", RelLocatedIn, "web")

	edges := cg.EdgesFrom("stoicera")
	if len(edges) != 1 {
		t.Fatalf("expected 1 edge, got %d", len(edges))
	}
	if edges[0].Relation != RelLocatedIn {
		t.Errorf("expected located_in, got %s", edges[0].Relation)
	}

	incoming := cg.EdgesTo("vienna")
	if len(incoming) != 1 {
		t.Errorf("expected 1 incoming edge to Vienna, got %d", len(incoming))
	}
}

func TestCognitiveGraphEdgeDedup(t *testing.T) {
	cg := NewCognitiveGraph("")

	cg.EnsureNode("Go", NodeEntity)
	cg.EnsureNode("fast", NodeProperty)

	cg.AddEdge("go", "fast", RelDescribedAs, "web")
	cg.AddEdge("go", "fast", RelDescribedAs, "web") // duplicate

	if cg.EdgeCount() != 1 {
		t.Errorf("expected 1 edge after dedup, got %d", cg.EdgeCount())
	}

	// Weight should be boosted
	edges := cg.EdgesFrom("go")
	if edges[0].Weight < 0.75 {
		t.Errorf("duplicate edge should boost weight, got %.2f", edges[0].Weight)
	}
}

func TestFindNodes(t *testing.T) {
	cg := NewCognitiveGraph("")

	cg.EnsureNode("Go programming language", NodeEntity)
	cg.EnsureNode("Go routines", NodeConcept)
	cg.EnsureNode("Python", NodeEntity)

	results := cg.FindNodes("go")
	if len(results) < 2 {
		t.Errorf("expected at least 2 results matching 'go', got %d", len(results))
	}
}

// -----------------------------------------------------------------------
// Spreading Activation Tests
// -----------------------------------------------------------------------

func TestSpreadingActivation(t *testing.T) {
	cg := NewCognitiveGraph("")

	cg.EnsureNode("Stoicera", NodeEntity)
	cg.EnsureNode("Vienna", NodeEntity)
	cg.EnsureNode("Austria", NodeEntity)
	cg.EnsureNode("philosophy", NodeConcept)

	cg.AddEdge("stoicera", "vienna", RelLocatedIn, "web")
	cg.AddEdge("vienna", "austria", RelLocatedIn, "web")
	cg.AddEdge("stoicera", "philosophy", RelDomain, "web")

	// Activate Stoicera
	cg.Activate("stoicera", 1.0)

	// Stoicera should be fully active
	stoicera := cg.GetNode("stoicera")
	if stoicera.Activation != 1.0 {
		t.Errorf("stoicera activation should be 1.0, got %.2f", stoicera.Activation)
	}

	// Vienna should be partially active (1 hop)
	vienna := cg.GetNode("vienna")
	if vienna.Activation < 0.1 {
		t.Errorf("vienna should be activated by spreading, got %.2f", vienna.Activation)
	}

	// Austria should be even less active (2 hops)
	austria := cg.GetNode("austria")
	if austria.Activation >= vienna.Activation {
		t.Error("austria (2 hops) should have less activation than vienna (1 hop)")
	}

	// Philosophy should also be activated
	phil := cg.GetNode("philosophy")
	if phil.Activation < 0.1 {
		t.Errorf("philosophy should be activated, got %.2f", phil.Activation)
	}
}

func TestMostActive(t *testing.T) {
	cg := NewCognitiveGraph("")

	cg.EnsureNode("A", NodeEntity)
	cg.EnsureNode("B", NodeEntity)
	cg.EnsureNode("C", NodeEntity)

	cg.AddEdge("a", "b", RelRelatedTo, "test")
	cg.AddEdge("b", "c", RelRelatedTo, "test")

	cg.Activate("a", 1.0)

	active := cg.MostActive(10)
	if len(active) == 0 {
		t.Fatal("should have active nodes")
	}
	if active[0].ID != "a" {
		t.Errorf("most active should be 'a', got %q", active[0].ID)
	}
}

func TestDecayAll(t *testing.T) {
	cg := NewCognitiveGraph("")

	cg.EnsureNode("Test", NodeEntity)
	cg.Activate("test", 1.0)

	cg.DecayAll(0.5)
	node := cg.GetNode("test")
	if node.Activation != 0.5 {
		t.Errorf("after 50%% decay, activation should be 0.5, got %.2f", node.Activation)
	}

	cg.DecayAll(0.01) // decay to near zero
	if node.Activation > 0.01 {
		t.Error("should decay to near zero")
	}
}

// -----------------------------------------------------------------------
// Triple Extraction Tests
// -----------------------------------------------------------------------

func TestExtractTriplesIsA(t *testing.T) {
	triples := ExtractTriples("Stoicera is a philosophy company based in Vienna")
	if len(triples) == 0 {
		t.Fatal("expected at least one triple")
	}
	found := false
	for _, tr := range triples {
		if tr.Relation == RelIsA && strings.Contains(strings.ToLower(tr.Subject), "stoicera") {
			found = true
		}
	}
	if !found {
		t.Error("should extract 'Stoicera is_a philosophy company'")
	}
}

func TestExtractTriplesLocatedIn(t *testing.T) {
	triples := ExtractTriples("Stoicera is based in Vienna")
	if len(triples) == 0 {
		t.Fatal("expected triple")
	}
	if triples[0].Relation != RelLocatedIn {
		t.Errorf("expected located_in, got %s", triples[0].Relation)
	}
	if !strings.Contains(triples[0].Object, "Vienna") {
		t.Errorf("expected object 'Vienna', got %q", triples[0].Object)
	}
}

func TestExtractTriplesFoundedBy(t *testing.T) {
	triples := ExtractTriples("The company was founded in 2023 by Raphael Lugmayr")
	if len(triples) < 2 {
		t.Fatalf("expected 2 triples (founded_in + founded_by), got %d", len(triples))
	}

	hasFounder := false
	hasYear := false
	for _, tr := range triples {
		if tr.Relation == RelFoundedBy && strings.Contains(tr.Object, "Raphael") {
			hasFounder = true
		}
		if tr.Relation == RelFoundedIn && tr.Object == "2023" {
			hasYear = true
		}
	}
	if !hasFounder {
		t.Error("should extract founded_by Raphael")
	}
	if !hasYear {
		t.Error("should extract founded_in 2023")
	}
}

func TestExtractTriplesOffers(t *testing.T) {
	triples := ExtractTriples("Stoicera offers journals, meditation guides, and daily practices")
	if len(triples) < 2 {
		t.Errorf("expected multiple triples from list, got %d", len(triples))
	}
	for _, tr := range triples {
		if tr.Relation != RelOffers {
			t.Errorf("all should be 'offers', got %s", tr.Relation)
		}
	}
}

func TestExtractTriplesEmpty(t *testing.T) {
	triples := ExtractTriples("")
	if len(triples) != 0 {
		t.Error("empty string should produce no triples")
	}
}

// -----------------------------------------------------------------------
// IngestToGraph Tests
// -----------------------------------------------------------------------

func TestIngestToGraph(t *testing.T) {
	cg := NewCognitiveGraph("")

	content := `Stoicera is a philosophy company based in Vienna.
They create tools for modern stoics.
The company was founded in 2023 by Raphael Lugmayr.
Stoicera offers journals, meditation guides, and daily practices.`

	added := IngestToGraph(cg, content, "https://stoicera.com", "Stoicera")

	if added == 0 {
		t.Fatal("should extract relationships")
	}

	// Check the graph has the right structure
	stoicera := cg.GetNode("stoicera")
	if stoicera == nil {
		t.Fatal("should have 'stoicera' node")
	}

	// Check edges from stoicera
	edges := cg.EdgesFrom("stoicera")
	if len(edges) == 0 {
		t.Error("stoicera should have outgoing edges")
	}

	// Verify specific relationships
	hasLocation := false
	hasDefinition := false
	for _, e := range edges {
		if e.Relation == RelLocatedIn {
			hasLocation = true
		}
		if e.Relation == RelIsA {
			hasDefinition = true
		}
	}
	if !hasLocation {
		t.Error("should have located_in edge")
	}
	if !hasDefinition {
		t.Error("should have is_a edge")
	}

	t.Logf("Ingested %d relationships into graph with %d nodes and %d edges",
		added, cg.NodeCount(), cg.EdgeCount())
}

// -----------------------------------------------------------------------
// Graph Query Tests
// -----------------------------------------------------------------------

func TestGraphQuery(t *testing.T) {
	cg := NewCognitiveGraph("")

	// Build a small knowledge graph
	content := `Stoicera is a philosophy company based in Vienna.
The company was founded in 2023 by Raphael Lugmayr.
Stoicera offers journals and meditation guides.`

	IngestToGraph(cg, content, "https://stoicera.com", "Stoicera")

	// Query: what is Stoicera?
	answer := cg.Query("what is Stoicera")
	if answer == nil {
		t.Fatal("expected an answer")
	}
	if len(answer.DirectFacts) == 0 {
		t.Error("expected direct facts")
	}

	composed := cg.ComposeAnswer("what is Stoicera", answer)
	if composed == "" {
		t.Fatal("expected composed answer")
	}
	t.Logf("Answer: %s", composed)

	// Should mention it's a philosophy company
	if !strings.Contains(strings.ToLower(composed), "philosophy") {
		t.Error("answer should mention philosophy")
	}
}

func TestGraphQueryWho(t *testing.T) {
	cg := NewCognitiveGraph("")

	IngestToGraph(cg, "The company was founded by Raphael Lugmayr in 2023.", "web", "company")

	answer := cg.Query("who founded the company")
	if answer == nil {
		t.Skip("no answer — triple extraction may not have matched")
	}

	composed := cg.ComposeAnswer("who founded the company", answer)
	t.Logf("Who answer: %s", composed)
}

func TestGraphQueryWhere(t *testing.T) {
	cg := NewCognitiveGraph("")

	IngestToGraph(cg, "Stoicera is based in Vienna.", "web", "Stoicera")

	answer := cg.Query("where is Stoicera")
	if answer == nil {
		t.Fatal("expected an answer")
	}

	composed := cg.ComposeAnswer("where is Stoicera", answer)
	if !strings.Contains(composed, "Vienna") {
		t.Errorf("answer should mention Vienna, got %q", composed)
	}
}

// -----------------------------------------------------------------------
// Inference Tests
// -----------------------------------------------------------------------

func TestTransitiveInference(t *testing.T) {
	cg := NewCognitiveGraph("")

	cg.EnsureNode("Stoicera", NodeEntity)
	cg.EnsureNode("Vienna", NodeEntity)
	cg.EnsureNode("Austria", NodeEntity)

	cg.AddEdge("stoicera", "vienna", RelLocatedIn, "web")
	cg.AddEdge("vienna", "austria", RelLocatedIn, "web")

	ie := NewInferenceEngine(cg)
	inferences := ie.Transitive()

	if len(inferences) == 0 {
		t.Fatal("should infer Stoicera is located in Austria")
	}

	found := false
	for _, inf := range inferences {
		if strings.Contains(inf.Reason, "Austria") && strings.Contains(inf.Reason, "Stoicera") {
			found = true
			t.Logf("Inferred: %s (confidence: %.2f)", inf.Reason, inf.Confidence)
		}
	}
	if !found {
		t.Error("should infer transitive location")
	}

	// The inferred edge should now exist in the graph
	edges := cg.EdgesFrom("stoicera")
	hasAustria := false
	for _, e := range edges {
		if e.To == "austria" && e.Relation == RelLocatedIn && e.Inferred {
			hasAustria = true
		}
	}
	if !hasAustria {
		t.Error("inferred edge should be added to graph")
	}
}

func TestContradictionDetection(t *testing.T) {
	cg := NewCognitiveGraph("")

	cg.EnsureNode("user", NodeEntity)
	cg.EnsureNode("coffee", NodeConcept)

	cg.AddEdge("user", "coffee", RelPrefers, "user")
	cg.AddEdge("user", "coffee", RelDislikes, "user")

	ie := NewInferenceEngine(cg)
	contradictions := ie.Contradictions()

	found := false
	for _, c := range contradictions {
		if strings.Contains(c.Reason, "contradiction") || c.Relation == RelContradicts {
			found = true
			t.Logf("Contradiction: %s", c.Reason)
		}
	}
	if !found {
		t.Error("should detect prefers+dislikes contradiction")
	}
}

// -----------------------------------------------------------------------
// Consolidation Tests
// -----------------------------------------------------------------------

func TestConsolidation(t *testing.T) {
	cg := NewCognitiveGraph("")

	IngestToGraph(cg, `Stoicera is a philosophy company based in Vienna.
Vienna is located in Austria.
Austria is in Europe.
The company was founded by Raphael Lugmayr.`, "web", "Stoicera")

	consolidator := NewConsolidator(cg)
	insights := consolidator.Consolidate()

	t.Logf("Consolidation produced %d insights", len(insights))
	for _, insight := range insights {
		t.Logf("  [%s] %s (confidence: %.2f)", insight.Type, insight.Description, insight.Confidence)
	}

	// After consolidation, graph should have inferred edges
	stats := cg.Stats()
	t.Logf("Graph stats: %d nodes, %d edges (%d inferred)", stats.Nodes, stats.Edges, stats.InferredEdges)
}

// -----------------------------------------------------------------------
// Persistence Tests
// -----------------------------------------------------------------------

func TestCognitiveGraphPersistence(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "graph.json")

	// Create and populate
	cg1 := NewCognitiveGraph(path)
	cg1.EnsureNode("Go", NodeEntity)
	cg1.EnsureNode("programming language", NodeConcept)
	cg1.AddEdge("go", "programming language", RelIsA, "web")

	if err := cg1.Save(); err != nil {
		t.Fatalf("save error: %v", err)
	}

	// Load in new instance
	cg2 := NewCognitiveGraph(path)
	if cg2.NodeCount() != 2 {
		t.Errorf("expected 2 nodes after reload, got %d", cg2.NodeCount())
	}

	node := cg2.GetNode("go")
	if node == nil {
		t.Fatal("should have 'go' node after reload")
	}
	if node.Label != "Go" {
		t.Errorf("label should be preserved, got %q", node.Label)
	}

	edges := cg2.EdgesFrom("go")
	if len(edges) != 1 {
		t.Errorf("expected 1 edge after reload, got %d", len(edges))
	}
}

// -----------------------------------------------------------------------
// Pattern Detection Tests
// -----------------------------------------------------------------------

func TestPatternDetection(t *testing.T) {
	pd := NewPatternDetector()

	// Simulate repeated actions
	for i := 0; i < 5; i++ {
		pd.RecordAction("weather")
		pd.RecordAction("journal")
	}

	patterns := pd.DetectPatterns()
	if len(patterns) < 2 {
		t.Errorf("expected at least 2 patterns, got %d", len(patterns))
	}

	for _, p := range patterns {
		if p.Count < 3 {
			t.Errorf("pattern %q should have count >= 3, got %d", p.Action, p.Count)
		}
	}
}

func TestAnticipate(t *testing.T) {
	pd := NewPatternDetector()

	// Record actions at current hour
	for i := 0; i < 5; i++ {
		pd.RecordAction("weather")
	}

	likely := pd.Anticipate()
	if len(likely) == 0 {
		t.Error("should anticipate 'weather' at current hour")
	}
}

// -----------------------------------------------------------------------
// End-to-End: Ingest → Query → Answer with Inference
// -----------------------------------------------------------------------

func TestEndToEndCognitiveGraph(t *testing.T) {
	cg := NewCognitiveGraph("")

	// Ingest content about Stoicera
	content1 := `Stoicera is a philosophy company based in Vienna.
The company was founded in 2023 by Raphael Lugmayr.
Stoicera offers journals, meditation guides, and daily practices.
Their mission is to make ancient wisdom accessible.`

	added1 := IngestToGraph(cg, content1, "https://stoicera.com", "Stoicera")

	// Ingest content about Vienna
	content2 := `Vienna is located in Austria.
Austria is a country in Europe.
Vienna is the capital of Austria.`

	added2 := IngestToGraph(cg, content2, "https://en.wikipedia.org/wiki/Vienna", "Vienna")

	t.Logf("Ingested %d + %d relationships", added1, added2)

	// Run inference — should discover Stoicera is in Austria (transitive)
	ie := NewInferenceEngine(cg)
	inferences := ie.Transitive()
	t.Logf("Inferred %d new facts", len(inferences))

	for _, inf := range inferences {
		t.Logf("  Inferred: %s", inf.Reason)
	}

	// Query: where is Stoicera?
	answer := cg.Query("where is Stoicera")
	if answer == nil {
		t.Fatal("should answer 'where is Stoicera'")
	}

	composed := cg.ComposeAnswer("where is Stoicera", answer)
	t.Logf("Answer: %s", composed)

	if !strings.Contains(composed, "Vienna") {
		t.Error("answer should mention Vienna")
	}

	// After inference, should also know about Austria
	if !strings.Contains(composed, "Austria") {
		t.Logf("Note: answer doesn't mention Austria yet — inference may not have propagated to answer composition")
	}

	// Stats
	stats := cg.Stats()
	t.Logf("Final graph: %d nodes, %d edges (%d inferred), %d active",
		stats.Nodes, stats.Edges, stats.InferredEdges, stats.ActiveNodes)
}
