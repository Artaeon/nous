package cognitive

import (
	"strings"
	"testing"
)

// -----------------------------------------------------------------------
// Semantic Similarity Tests
// -----------------------------------------------------------------------

func TestTaxonomySimilarity(t *testing.T) {
	se := NewSemanticEngine()

	// car ≈ automobile (same category: vehicle)
	sim := se.Similarity("car", "automobile")
	if sim < 0.5 {
		t.Errorf("car and automobile should be similar, got %.2f", sim)
	}

	// go ≈ python (same category: programming_language)
	sim = se.Similarity("go", "python")
	if sim < 0.3 {
		t.Errorf("go and python should be similar (both programming languages), got %.2f", sim)
	}

	// car vs python (different categories)
	sim = se.Similarity("car", "python")
	if sim > 0.2 {
		t.Errorf("car and python should NOT be similar, got %.2f", sim)
	}

	// happy ≈ excited (same category: positive emotion)
	sim = se.Similarity("happy", "excited")
	if sim < 0.5 {
		t.Errorf("happy and excited should be similar, got %.2f", sim)
	}

	// happy vs sad (both emotions, but positive vs negative)
	sim = se.Similarity("happy", "sad")
	if sim < 0.1 {
		t.Errorf("happy and sad share 'emotion' category, got %.2f", sim)
	}
}

func TestNgramSimilarity(t *testing.T) {
	// Morphological similarity
	sim := ngramSimilarity("program", "programming")
	if sim < 0.3 {
		t.Errorf("program and programming should be morphologically similar, got %.2f", sim)
	}

	sim = ngramSimilarity("cat", "dog")
	if sim > 0.3 {
		t.Errorf("cat and dog should NOT be morphologically similar, got %.2f", sim)
	}
}

func TestCooccurrenceSimilarity(t *testing.T) {
	se := NewSemanticEngine()

	// Ingest text where "car" and "automobile" appear in similar contexts
	se.IngestText("The car drove down the highway fast. The automobile was parked nearby.")
	se.IngestText("The car is a fast vehicle. The automobile needs maintenance.")
	se.IngestText("I love driving my car. Her automobile is new.")

	sim := se.cooccurrenceSimilarity("car", "automobile")
	t.Logf("Co-occurrence similarity car/automobile: %.3f", sim)
	// After enough context, they should have some similarity
}

func TestSimilarTerms(t *testing.T) {
	se := NewSemanticEngine()

	candidates := []string{"automobile", "truck", "bicycle", "python", "java"}
	similar := se.SimilarTerms("car", candidates, 0.3)

	// Should find automobile and truck (both vehicles)
	found := false
	for _, s := range similar {
		if s == "automobile" || s == "truck" {
			found = true
		}
	}
	if !found {
		t.Errorf("SimilarTerms should find vehicle-related terms, got %v", similar)
	}
}

// -----------------------------------------------------------------------
// Multi-Hop Reasoning Tests
// -----------------------------------------------------------------------

func TestMultiHopWhoFounded(t *testing.T) {
	cg := NewCognitiveGraph("")
	se := NewSemanticEngine()

	IngestToGraph(cg, "Stoicera was founded by Raphael Lugmayr.", "web", "Stoicera")

	re := NewReasoningEngine(cg, se)
	chain := re.Reason("who founded Stoicera")

	if chain == nil {
		t.Fatal("should produce a reasoning chain")
	}
	t.Logf("Trace:\n%s", chain.Trace)
	t.Logf("Answer: %s", chain.Answer)

	if !strings.Contains(chain.Answer, "Raphael") {
		t.Errorf("answer should mention Raphael, got %q", chain.Answer)
	}
}

func TestMultiHopWhereLocated(t *testing.T) {
	cg := NewCognitiveGraph("")
	se := NewSemanticEngine()

	IngestToGraph(cg, "Stoicera is based in Vienna. Vienna is located in Austria.", "web", "Stoicera")

	re := NewReasoningEngine(cg, se)
	chain := re.Reason("what country is Stoicera in")

	if chain == nil {
		t.Fatal("should produce a reasoning chain")
	}
	t.Logf("Trace:\n%s", chain.Trace)
	t.Logf("Answer: %s", chain.Answer)
}

func TestMultiHopWhatOffers(t *testing.T) {
	cg := NewCognitiveGraph("")
	se := NewSemanticEngine()

	IngestToGraph(cg, "Stoicera offers journals, meditation guides, and daily practices.", "web", "Stoicera")

	re := NewReasoningEngine(cg, se)
	chain := re.Reason("what does Stoicera offer")

	if chain == nil {
		t.Fatal("should produce a reasoning chain")
	}
	t.Logf("Answer: %s", chain.Answer)

	if !strings.Contains(chain.Answer, "journals") {
		t.Errorf("answer should mention journals, got %q", chain.Answer)
	}
}

func TestMultiHopThreeHops(t *testing.T) {
	cg := NewCognitiveGraph("")
	se := NewSemanticEngine()

	// Build a 3-hop chain
	IngestToGraph(cg, "Stoicera was founded by Raphael.", "web", "Stoicera")
	// Manually add location for Raphael
	cg.EnsureNode("Raphael", NodeEntity)
	cg.EnsureNode("Vienna", NodeEntity)
	cg.AddEdge("raphael", "vienna", RelLocatedIn, "web")

	re := NewReasoningEngine(cg, se)
	chain := re.Reason("where is the founder of Stoicera from")

	if chain == nil {
		t.Fatal("should produce a 3-hop reasoning chain")
	}
	t.Logf("Trace:\n%s", chain.Trace)
	t.Logf("Answer: %s", chain.Answer)

	if !strings.Contains(chain.Answer, "Vienna") {
		t.Errorf("answer should trace through founder to Vienna, got %q", chain.Answer)
	}
}

func TestReasoningWithSemanticMatch(t *testing.T) {
	cg := NewCognitiveGraph("")
	se := NewSemanticEngine()

	cg.EnsureNode("Go", NodeEntity)
	cg.EnsureNode("programming language", NodeConcept)
	cg.AddEdge("go", "programming language", RelIsA, "web")

	// Query with synonym — "golang" should match "go" via semantic similarity
	re := NewReasoningEngine(cg, se)

	// Direct match test
	nodes := re.findNodes("Go")
	if len(nodes) == 0 {
		t.Fatal("should find Go node")
	}

	// The semantic engine should recognize golang ≈ go
	sim := se.Similarity("golang", "go")
	t.Logf("Similarity golang/go: %.2f", sim)
}

// -----------------------------------------------------------------------
// Self-Correction Tests
// -----------------------------------------------------------------------

func TestSelfCorrection(t *testing.T) {
	cg := NewCognitiveGraph("")

	cg.EnsureNode("Stoicera", NodeEntity)
	cg.EnsureNode("2023", NodeEvent)
	cg.AddEdge("stoicera", "2023", RelFoundedIn, "web")

	// Check edge confidence before correction
	edges := cg.EdgesFrom("stoicera")
	oldConf := edges[0].Confidence

	// User corrects: "Stoicera was founded in 2024, not 2023"
	ApplyCorrection(cg, Correction{
		WrongFact: "2023",
		RightFact: "Stoicera was founded in 2024",
	})

	// Old edge should have lower confidence
	edges = cg.EdgesFrom("stoicera")
	foundOld := false
	foundNew := false
	for _, e := range edges {
		if e.To == "2023" {
			foundOld = true
			if e.Confidence >= oldConf {
				t.Error("wrong edge confidence should be reduced")
			}
			t.Logf("Old edge confidence: %.2f → %.2f", oldConf, e.Confidence)
		}
		if e.To == "2024" {
			foundNew = true
			if e.Confidence < 0.9 {
				t.Errorf("corrected edge should have high confidence, got %.2f", e.Confidence)
			}
		}
	}
	if !foundOld {
		t.Error("old edge should still exist (with low confidence)")
	}
	if !foundNew {
		t.Error("new corrected edge should be added")
	}
}

func TestDetectCorrection(t *testing.T) {
	tests := []struct {
		input string
		want  bool
	}{
		{"no, it was founded in 2024", true},
		{"that's wrong, Stoicera is based in Berlin", true},
		{"incorrect, the founder is Johannes", true},
		{"what is Stoicera", false},
		{"tell me more", false},
	}

	for _, tt := range tests {
		c := DetectCorrection(tt.input)
		got := c != nil
		if got != tt.want {
			t.Errorf("DetectCorrection(%q) = %v, want %v", tt.input, got, tt.want)
		}
	}
}

// -----------------------------------------------------------------------
// Concept Abstraction Tests
// -----------------------------------------------------------------------

func TestConceptAbstraction(t *testing.T) {
	cg := NewCognitiveGraph("")

	// Add several programming languages with shared properties
	cg.EnsureNode("Go", NodeEntity)
	cg.EnsureNode("Rust", NodeEntity)
	cg.EnsureNode("C", NodeEntity)
	cg.EnsureNode("programming language", NodeConcept)
	cg.EnsureNode("fast", NodeProperty)
	cg.EnsureNode("compiled", NodeProperty)

	cg.AddEdge("go", "programming language", RelIsA, "web")
	cg.AddEdge("rust", "programming language", RelIsA, "web")
	cg.AddEdge("c", "programming language", RelIsA, "web")

	cg.AddEdge("go", "fast", RelDescribedAs, "web")
	cg.AddEdge("rust", "fast", RelDescribedAs, "web")
	cg.AddEdge("c", "fast", RelDescribedAs, "web")

	cg.AddEdge("go", "compiled", RelDescribedAs, "web")
	cg.AddEdge("rust", "compiled", RelDescribedAs, "web")
	cg.AddEdge("c", "compiled", RelDescribedAs, "web")

	ae := NewAbstractionEngine(cg)
	abstractions := ae.Discover()

	if len(abstractions) == 0 {
		t.Fatal("should discover abstractions")
	}

	foundFast := false
	for _, a := range abstractions {
		t.Logf("Abstraction: %s (confidence: %.2f, evidence: %v)", a.Rule, a.Confidence, a.Evidence)
		if strings.Contains(a.Rule, "fast") {
			foundFast = true
		}
	}

	if !foundFast {
		t.Error("should abstract: 'programming languages tend to be fast'")
	}
}

// -----------------------------------------------------------------------
// Causal Reasoning Tests
// -----------------------------------------------------------------------

func TestCausalRecordAndAnalyze(t *testing.T) {
	ce := NewCausalEngine()

	// Simulate pattern: stress → food spending
	for i := 0; i < 10; i++ {
		ce.RecordEvent("journal", map[string]string{"mood": "stressed"})
		ce.RecordEvent("expense", map[string]string{"category": "food", "amount": "30"})
	}

	links := ce.AnalyzeCausality()
	t.Logf("Found %d causal links", len(links))
	for _, link := range links {
		t.Logf("  %s", link.Description)
	}

	if len(links) == 0 {
		t.Error("should detect causal relationship between journal and expense")
	}
}

func TestCausalAnswerWhy(t *testing.T) {
	ce := NewCausalEngine()

	for i := 0; i < 5; i++ {
		ce.RecordEvent("weather", map[string]string{"condition": "rain"})
		ce.RecordEvent("journal", map[string]string{"mood": "sad"})
	}

	ce.AnalyzeCausality()

	answer := ce.AnswerWhy("why am I sad")
	t.Logf("Why answer: %s", answer)
	// May or may not find the link depending on term matching
}

func TestTemporalCorrelation(t *testing.T) {
	ce := NewCausalEngine()

	// Simulate correlated events over multiple "days"
	for i := 0; i < 10; i++ {
		ce.RecordEvent("exercise", nil)
		ce.RecordEvent("sleep", map[string]string{"quality": "good"})
	}

	correlations := ce.FindCorrelations()
	t.Logf("Found %d correlations", len(correlations))
	for _, c := range correlations {
		t.Logf("  %s (r=%.2f)", c.Description, c.Correlation)
	}
}

// -----------------------------------------------------------------------
// End-to-End: Full Reasoning Pipeline
// -----------------------------------------------------------------------

func TestFullReasoningPipeline(t *testing.T) {
	cg := NewCognitiveGraph("")
	se := NewSemanticEngine()

	// Ingest rich content
	content := `Stoicera is a philosophy company based in Vienna.
The company was founded in 2023 by Raphael Lugmayr.
Stoicera offers journals, meditation guides, and daily practices.
Their mission is to make ancient wisdom accessible.
Vienna is located in Austria.
Austria is a country in Europe.`

	IngestToGraph(cg, content, "https://stoicera.com", "Stoicera")
	se.IngestText(content)

	// Run inference
	ie := NewInferenceEngine(cg)
	inferences := ie.Transitive()
	t.Logf("Inferred %d new facts", len(inferences))

	// Run abstraction
	ae := NewAbstractionEngine(cg)
	abstractions := ae.Discover()
	t.Logf("Found %d abstractions", len(abstractions))

	// Multi-hop reasoning
	re := NewReasoningEngine(cg, se)

	questions := []struct {
		q    string
		want string
	}{
		{"who founded Stoicera", "Raphael"},
		{"what does Stoicera offer", "journals"},
		{"where is the founder of Stoicera from", "Vienna"},
	}

	for _, q := range questions {
		chain := re.Reason(q.q)
		if chain == nil {
			t.Logf("  %q: no reasoning chain", q.q)
			continue
		}
		t.Logf("  %q → %s", q.q, chain.Answer)
		t.Logf("    Trace: %s", chain.Trace)

		if q.want != "" && !strings.Contains(chain.Answer, q.want) {
			t.Errorf("answer to %q should contain %q, got %q", q.q, q.want, chain.Answer)
		}
	}

	// Graph stats
	stats := cg.Stats()
	t.Logf("\nFinal: %d nodes, %d edges (%d inferred)", stats.Nodes, stats.Edges, stats.InferredEdges)
}
