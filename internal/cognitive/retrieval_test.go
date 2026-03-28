package cognitive

import (
	"math"
	"strings"
	"testing"
	"time"
)

// ---------- helpers ----------

// testKV creates a KnowledgeVec with mockEmbed and some seeded documents.
func testKV(t *testing.T, docs map[string]string) *KnowledgeVec {
	t.Helper()
	kv := NewKnowledgeVec(mockEmbed, "")
	for text, source := range docs {
		if err := kv.AddChunk(text, source); err != nil {
			t.Fatalf("AddChunk failed: %v", err)
		}
	}
	return kv
}

// testGraph creates a CognitiveGraph with some seeded knowledge.
func testGraph(t *testing.T) *CognitiveGraph {
	t.Helper()
	g := NewCognitiveGraph("")
	g.AddEdge("Go", "programming language", RelIsA, "test")
	g.AddEdge("Go", "Google", RelCreatedBy, "test")
	g.AddEdge("Go", "concurrency", RelKnownFor, "test")
	g.AddEdge("Go", "simplicity", RelKnownFor, "test")
	g.AddEdge("Python", "programming language", RelIsA, "test")
	g.AddEdge("Python", "machine learning", RelUsedFor, "test")
	g.AddEdge("Python", "Guido van Rossum", RelCreatedBy, "test")
	g.AddEdge("Rust", "programming language", RelIsA, "test")
	g.AddEdge("Rust", "memory safety", RelKnownFor, "test")
	g.AddEdge("Vienna", "Austria", RelLocatedIn, "test")
	g.AddEdge("Vienna", "city", RelIsA, "test")
	return g
}

// testRetriever creates a TwoTierRetriever with seeded data.
func testRetriever(t *testing.T) *TwoTierRetriever {
	t.Helper()
	docs := map[string]string{
		"Go is a statically typed compiled programming language designed at Google.":           "wiki-go",
		"Python is a high-level general-purpose programming language.":                         "wiki-python",
		"Rust is a multi-paradigm systems programming language focused on memory safety.":      "wiki-rust",
		"Vienna is the capital and largest city of Austria.":                                    "wiki-vienna",
		"Machine learning is a subset of artificial intelligence using statistical methods.":    "wiki-ml",
		"Quantum computing uses quantum mechanical phenomena to perform computations.":          "wiki-quantum",
		"The population of France is approximately 67 million.":                                 "wiki-france",
		"Go was designed by Robert Griesemer, Rob Pike, and Ken Thompson at Google.":            "wiki-go-2",
		"Go features goroutines for lightweight concurrent programming.":                        "wiki-go-3",
		"Climate change refers to long-term shifts in global temperatures and weather patterns.": "wiki-climate",
	}
	kv := testKV(t, docs)
	g := testGraph(t)
	r := NewTwoTierRetriever(kv, g)

	for text, source := range docs {
		r.IndexDocument(text, source)
	}
	return r
}

// ---------- TwoTierRetriever tests ----------

func TestTwoTierRetriever_Lexical(t *testing.T) {
	r := NewTwoTierRetriever(nil, nil)

	r.IndexDocument("Go is a statically typed compiled programming language.", "wiki-go")
	r.IndexDocument("Python is a high-level general-purpose programming language.", "wiki-python")
	r.IndexDocument("Rust is a systems programming language focused on memory safety.", "wiki-rust")
	r.IndexDocument("Quantum computing uses quantum bits called qubits.", "wiki-quantum")

	// Lexical retrieval should rank "Go" document highest for "Go programming".
	results := r.retrieveLexical("Go programming language", 4)
	if len(results) == 0 {
		t.Fatal("expected lexical results")
	}

	// All programming language documents should appear.
	foundGo := false
	for _, rr := range results {
		if strings.Contains(rr.Text, "Go is") {
			foundGo = true
		}
		if rr.Tier != TierLexical {
			t.Errorf("tier = %d, want TierLexical", rr.Tier)
		}
	}
	if !foundGo {
		t.Error("Go document should appear in lexical results for 'Go programming'")
	}

	// Verify BM25 scores are positive and ordered.
	for i := 1; i < len(results); i++ {
		if results[i].Score > results[i-1].Score {
			t.Errorf("results not sorted: score[%d]=%f > score[%d]=%f",
				i, results[i].Score, i-1, results[i-1].Score)
		}
	}
}

func TestTwoTierRetriever_Semantic(t *testing.T) {
	docs := map[string]string{
		"Go is a compiled programming language designed at Google.":   "wiki-go",
		"Python is a high-level interpreted programming language.":    "wiki-python",
		"The weather today is sunny with clear skies.":                "weather",
	}
	kv := testKV(t, docs)
	r := NewTwoTierRetriever(kv, nil)

	results := r.retrieveSemantic("programming language Go", 3)
	if len(results) == 0 {
		t.Fatal("expected semantic results")
	}
	for _, rr := range results {
		if rr.Tier != TierSemantic {
			t.Errorf("tier = %d, want TierSemantic", rr.Tier)
		}
		if rr.Score <= 0 {
			t.Error("semantic score should be positive")
		}
	}
}

func TestTwoTierRetriever_Combined(t *testing.T) {
	r := testRetriever(t)

	results := r.Retrieve("Go programming language", 5)
	if len(results) == 0 {
		t.Fatal("expected combined results")
	}

	// Verify RRF scoring: all results should have positive scores.
	for _, rr := range results {
		if rr.Score <= 0 {
			t.Errorf("RRF score should be positive, got %f", rr.Score)
		}
	}

	// Results should be sorted by score descending.
	for i := 1; i < len(results); i++ {
		if results[i].Score > results[i-1].Score {
			t.Errorf("results not sorted: score[%d]=%f > score[%d]=%f",
				i, results[i].Score, i-1, results[i-1].Score)
		}
	}
}

func TestIndexDocument(t *testing.T) {
	r := NewTwoTierRetriever(nil, nil)

	r.IndexDocument("Go is a programming language.", "src-1")
	r.IndexDocument("Python is also a programming language.", "src-2")
	r.IndexDocument("", "empty") // should be ignored

	r.mu.RLock()
	defer r.mu.RUnlock()

	if r.docCount != 2 {
		t.Errorf("docCount = %d, want 2", r.docCount)
	}
	if r.avgDocLen <= 0 {
		t.Error("avgDocLen should be positive")
	}

	// Check that document frequency was tracked.
	if r.docFreq["programming"] != 2 {
		t.Errorf("docFreq[programming] = %d, want 2", r.docFreq["programming"])
	}
}

func TestTwoTierRetriever_GraphRetrieval(t *testing.T) {
	g := testGraph(t)
	r := NewTwoTierRetriever(nil, g)

	results := r.retrieveGraph("Go programming", 10)
	if len(results) == 0 {
		t.Fatal("expected graph results for 'Go programming'")
	}

	foundIsA := false
	for _, rr := range results {
		if rr.Tier != TierGraph {
			t.Errorf("tier = %d, want TierGraph", rr.Tier)
		}
		if strings.Contains(strings.ToLower(rr.Text), "programming language") {
			foundIsA = true
		}
	}
	if !foundIsA {
		t.Error("expected to find 'is a programming language' fact in graph results")
	}
}

func TestTwoTierRetriever_EmptyQuery(t *testing.T) {
	r := testRetriever(t)

	results := r.Retrieve("", 5)
	// Should handle gracefully: may return empty or graph-only results.
	_ = results
}

func TestTwoTierRetriever_NoDocuments(t *testing.T) {
	r := NewTwoTierRetriever(nil, nil)
	results := r.Retrieve("test query", 5)
	if len(results) != 0 {
		t.Errorf("expected no results from empty retriever, got %d", len(results))
	}
}

// ---------- QueryRewriter tests ----------

func TestQueryDecompose_Comparison(t *testing.T) {
	qr := NewQueryRewriter()

	tests := []struct {
		query string
	}{
		{"compare Go vs Python"},
		{"Go versus Python"},
		{"difference between Go and Python"},
		{"Go compared to Python"},
	}

	for _, tc := range tests {
		d := qr.Decompose(tc.query)
		if d.QueryType != "comparison" {
			t.Errorf("Decompose(%q): type = %q, want comparison", tc.query, d.QueryType)
			continue
		}
		if !d.IsComplex {
			t.Errorf("Decompose(%q): IsComplex should be true", tc.query)
		}
		if len(d.SubQueries) < 2 {
			t.Errorf("Decompose(%q): expected at least 2 sub-queries, got %d", tc.query, len(d.SubQueries))
			continue
		}

		// Check purposes.
		purposes := make(map[string]bool)
		for _, sq := range d.SubQueries {
			purposes[sq.Purpose] = true
		}
		if !purposes["entity_a"] || !purposes["entity_b"] {
			t.Errorf("Decompose(%q): missing entity_a or entity_b sub-queries; got %v", tc.query, purposes)
		}
	}
}

func TestQueryDecompose_Causal(t *testing.T) {
	qr := NewQueryRewriter()

	tests := []struct {
		query string
	}{
		{"why did the Roman Empire fall"},
		{"what causes inflation"},
		{"how does pollution cause climate change"},
	}

	for _, tc := range tests {
		d := qr.Decompose(tc.query)
		if d.QueryType != "causal" {
			t.Errorf("Decompose(%q): type = %q, want causal", tc.query, d.QueryType)
		}
		if !d.IsComplex {
			t.Errorf("Decompose(%q): IsComplex should be true", tc.query)
		}
		if len(d.SubQueries) < 2 {
			t.Errorf("Decompose(%q): expected at least 2 sub-queries, got %d", tc.query, len(d.SubQueries))
		}
	}
}

func TestQueryDecompose_Simple(t *testing.T) {
	qr := NewQueryRewriter()

	d := qr.Decompose("what is photosynthesis")
	if d.QueryType != "simple" {
		t.Errorf("type = %q, want simple", d.QueryType)
	}
	if d.IsComplex {
		t.Error("simple query should not be complex")
	}
	if len(d.SubQueries) != 1 {
		t.Errorf("expected 1 sub-query, got %d", len(d.SubQueries))
	}
}

func TestQueryDecompose_Temporal(t *testing.T) {
	qr := NewQueryRewriter()

	d := qr.Decompose("how has artificial intelligence changed over time")
	if d.QueryType != "temporal" {
		t.Errorf("type = %q, want temporal", d.QueryType)
	}
	if !d.IsComplex {
		t.Error("temporal query should be complex")
	}
	if len(d.SubQueries) < 3 {
		t.Errorf("expected at least 3 sub-queries for temporal, got %d", len(d.SubQueries))
	}
}

func TestQueryDecompose_MultiEntity(t *testing.T) {
	qr := NewQueryRewriter()

	// "what is X and how does Y relate to Z" triggers multi-entity decomposition.
	d := qr.Decompose("what is Go and how does it relate to systems programming")
	if d.QueryType != "multi_entity" {
		t.Errorf("type = %q, want multi_entity", d.QueryType)
	}
	if !d.IsComplex {
		t.Error("multi-entity query should be complex")
	}
	if len(d.SubQueries) < 2 {
		t.Errorf("expected at least 2 sub-queries, got %d", len(d.SubQueries))
	}
}

func TestRewriteForRetrieval(t *testing.T) {
	qr := NewQueryRewriter()

	tests := []struct {
		input    string
		notWant  []string // words that should be stripped
		mustHave []string // words that must be present
	}{
		{
			input:   "what is machine learning?",
			notWant: []string{"what", "is"},
		},
		{
			input:    "how does AI work?",
			mustHave: []string{"artificial intelligence"},
			notWant:  []string{"how", "does"},
		},
		{
			input:   "tell me about actually basically quantum physics",
			notWant: []string{"actually", "basically", "tell", "me", "about"},
		},
		{
			input:    "explain NLP please",
			mustHave: []string{"natural language processing"},
			notWant:  []string{"explain", "please"},
		},
	}

	for _, tc := range tests {
		result := qr.RewriteForRetrieval(tc.input)
		lower := strings.ToLower(result)

		for _, nw := range tc.notWant {
			if strings.Contains(lower, nw) {
				t.Errorf("RewriteForRetrieval(%q) = %q, should not contain %q", tc.input, result, nw)
			}
		}
		for _, mh := range tc.mustHave {
			if !strings.Contains(lower, mh) {
				t.Errorf("RewriteForRetrieval(%q) = %q, should contain %q", tc.input, result, mh)
			}
		}
	}
}

func TestRewriteForRetrieval_Empty(t *testing.T) {
	qr := NewQueryRewriter()
	result := qr.RewriteForRetrieval("")
	if result != "" {
		t.Errorf("expected empty, got %q", result)
	}
}

// ---------- Evidence attribution tests ----------

func TestEvidenceAttribution(t *testing.T) {
	r := testRetriever(t)
	ea := NewEvidenceAttributor(r)

	plan := &ContentPlan{
		Topic:  "Go",
		Thesis: "Go is a programming language.",
		Claims: []PlanClaim{
			{Text: "Go is a statically typed compiled programming language.", Priority: 1},
			{Text: "Go was designed at Google.", Priority: 2},
		},
	}

	attrs := ea.AttributeClaims(plan)
	if len(attrs) != 2 {
		t.Fatalf("expected 2 attributions, got %d", len(attrs))
	}

	// The first claim should have evidence since we indexed that exact text.
	if !attrs[0].Supported {
		t.Error("claim 'Go is a statically typed...' should be supported")
	}
	if attrs[0].Confidence <= 0 {
		t.Error("confidence should be positive for supported claim")
	}
}

func TestEvidenceAttribution_NilPlan(t *testing.T) {
	r := testRetriever(t)
	ea := NewEvidenceAttributor(r)

	attrs := ea.AttributeClaims(nil)
	if attrs != nil {
		t.Error("expected nil for nil plan")
	}
}

func TestContradictionDetection_Evidence(t *testing.T) {
	r := testRetriever(t)
	ea := NewEvidenceAttributor(r)

	t.Run("direct_negation", func(t *testing.T) {
		claims := []string{
			"Go is a compiled language.",
			"Go is not a compiled language.",
		}
		contradictions := ea.CheckContradictions(claims)
		if len(contradictions) == 0 {
			t.Error("expected a direct negation contradiction")
			return
		}
		if contradictions[0].Type != "direct_negation" {
			t.Errorf("type = %q, want direct_negation", contradictions[0].Type)
		}
	})

	t.Run("incompatible_values", func(t *testing.T) {
		claims := []string{
			"The population is 50 million.",
			"The population is 67 million.",
		}
		contradictions := ea.CheckContradictions(claims)
		if len(contradictions) == 0 {
			t.Error("expected an incompatible values contradiction")
			return
		}
		if contradictions[0].Type != "incompatible_values" {
			t.Errorf("type = %q, want incompatible_values", contradictions[0].Type)
		}
	})

	t.Run("temporal_conflict", func(t *testing.T) {
		claims := []string{
			"The Renaissance happened before the Industrial Revolution.",
			"The Industrial Revolution happened before the Renaissance.",
		}
		contradictions := ea.CheckContradictions(claims)
		if len(contradictions) == 0 {
			t.Error("expected a temporal conflict")
			return
		}
		if contradictions[0].Type != "temporal_conflict" {
			t.Errorf("type = %q, want temporal_conflict", contradictions[0].Type)
		}
	})

	t.Run("no_contradiction", func(t *testing.T) {
		claims := []string{
			"Go is fast.",
			"Python is readable.",
		}
		contradictions := ea.CheckContradictions(claims)
		if len(contradictions) != 0 {
			t.Errorf("expected no contradictions, got %d", len(contradictions))
		}
	})

	t.Run("single_claim", func(t *testing.T) {
		contradictions := ea.CheckContradictions([]string{"just one claim"})
		if contradictions != nil {
			t.Error("single claim cannot contradict itself")
		}
	})
}

func TestValidateResponse(t *testing.T) {
	r := testRetriever(t)
	ea := NewEvidenceAttributor(r)

	response := "Go is a compiled programming language. Go was designed at Google. Go features goroutines for concurrency."
	sources := r.Retrieve("Go programming language", 10)

	vr := ea.ValidateResponse(response, sources)
	if vr == nil {
		t.Fatal("expected non-nil validation result")
	}

	if len(vr.Attributions) == 0 {
		t.Error("expected at least one attribution")
	}

	if vr.SupportRate < 0 || vr.SupportRate > 1 {
		t.Errorf("SupportRate = %f, should be in [0,1]", vr.SupportRate)
	}

	if vr.ContradictionRate < 0 || vr.ContradictionRate > 1 {
		t.Errorf("ContradictionRate = %f, should be in [0,1]", vr.ContradictionRate)
	}
}

func TestValidateResponse_EmptyResponse(t *testing.T) {
	r := testRetriever(t)
	ea := NewEvidenceAttributor(r)

	vr := ea.ValidateResponse("", nil)
	if vr == nil {
		t.Fatal("expected non-nil validation result")
	}
	if vr.SupportRate != 1.0 {
		t.Errorf("empty response should have 100%% support rate, got %f", vr.SupportRate)
	}
}

// ---------- Freshness tests ----------

func TestFreshnessClassification(t *testing.T) {
	fc := NewFreshnessClassifier()

	tests := []struct {
		query    string
		expected TopicFreshness
	}{
		{"what is photosynthesis", FreshnessStatic},
		{"history of ancient Rome", FreshnessStatic},
		{"the law of thermodynamics", FreshnessStatic},
		{"population of France", FreshnessSlow},
		{"how many people live in Japan", FreshnessSlow},
		{"GDP per capita ranking", FreshnessSlow},
		{"weather today in London", FreshnessDynamic},
		{"latest news about technology", FreshnessDynamic},
		{"current state of AI research", FreshnessDynamic},
		{"stock price right now", FreshnessRealtime},
		{"AAPL stock ticker", FreshnessRealtime},
		{"live election results", FreshnessRealtime},
	}

	for _, tc := range tests {
		result := fc.Classify(tc.query)
		if result != tc.expected {
			t.Errorf("Classify(%q) = %s, want %s", tc.query, result, tc.expected)
		}
	}
}

func TestFreshnessClassification_Empty(t *testing.T) {
	fc := NewFreshnessClassifier()
	if fc.Classify("") != FreshnessStatic {
		t.Error("empty query should be classified as static")
	}
}

func TestTrustScoring(t *testing.T) {
	t.Run("fresh_authoritative", func(t *testing.T) {
		ts := ScoreTrust("kb:physics", 1*time.Hour, 3)
		if ts.Score < 0.7 {
			t.Errorf("fresh authoritative source should score high, got %f", ts.Score)
		}
		if ts.Factors["authority"] < 0.9 {
			t.Errorf("knowledge base should have high authority, got %f", ts.Factors["authority"])
		}
	})

	t.Run("old_unknown", func(t *testing.T) {
		ts := ScoreTrust("", 720*time.Hour, 0) // 30 days, unknown source, no cross-refs
		if ts.Score > 0.5 {
			t.Errorf("old unknown source should score low, got %f", ts.Score)
		}
	})

	t.Run("recency_decay", func(t *testing.T) {
		fresh := ScoreTrust("test", 0, 1)
		old := ScoreTrust("test", 48*time.Hour, 1)
		if old.Factors["recency"] >= fresh.Factors["recency"] {
			t.Error("older source should have lower recency factor")
		}
	})

	t.Run("cross_reference_boost", func(t *testing.T) {
		single := ScoreTrust("test", 1*time.Hour, 0)
		multi := ScoreTrust("test", 1*time.Hour, 5)
		if multi.Factors["consistency"] <= single.Factors["consistency"] {
			t.Error("more cross-references should boost consistency")
		}
	})

	t.Run("source_authority_levels", func(t *testing.T) {
		kb := ScoreTrust("kb:physics", 1*time.Hour, 1)
		graph := ScoreTrust("graph:entity", 1*time.Hour, 1)
		inferred := ScoreTrust("inferred:guess", 1*time.Hour, 1)

		if kb.Factors["authority"] <= graph.Factors["authority"] {
			t.Error("kb should have higher authority than graph")
		}
		if graph.Factors["authority"] <= inferred.Factors["authority"] {
			t.Error("graph should have higher authority than inferred")
		}
	})
}

func TestRetrieveWithFreshness(t *testing.T) {
	r := testRetriever(t)
	fr := NewFreshnessRetriever(r)

	t.Run("static_query", func(t *testing.T) {
		results, freshness := fr.RetrieveWithFreshness("what is quantum computing", 5)
		if freshness != FreshnessStatic {
			t.Errorf("freshness = %s, want static", freshness)
		}
		// Should return results (static content is always acceptable).
		_ = results
	})

	t.Run("dynamic_query", func(t *testing.T) {
		results, freshness := fr.RetrieveWithFreshness("latest news about Go programming", 5)
		if freshness != FreshnessDynamic {
			t.Errorf("freshness = %s, want dynamic", freshness)
		}
		// All results should be recent (our test data is freshly indexed).
		for _, rr := range results {
			if time.Since(rr.Freshness) > 2*time.Hour {
				t.Error("dynamic query should only return recent results")
			}
		}
	})

	t.Run("freshness_scoring", func(t *testing.T) {
		results, _ := fr.RetrieveWithFreshness("Go programming", 5)
		// Results should be sorted by score descending.
		for i := 1; i < len(results); i++ {
			if results[i].Score > results[i-1].Score {
				t.Errorf("results not sorted: score[%d]=%f > score[%d]=%f",
					i, results[i].Score, i-1, results[i-1].Score)
			}
		}
	})
}

func TestRetrieveWithFreshness_EmptyRetriever(t *testing.T) {
	base := NewTwoTierRetriever(nil, nil)
	fr := NewFreshnessRetriever(base)

	results, freshness := fr.RetrieveWithFreshness("test", 5)
	if len(results) != 0 {
		t.Errorf("expected no results, got %d", len(results))
	}
	_ = freshness
}

func TestDefaultFreshnessConfig(t *testing.T) {
	cfg := DefaultFreshnessConfig()
	if cfg.MaxAge != 24*time.Hour {
		t.Errorf("MaxAge = %v, want 24h", cfg.MaxAge)
	}
	if cfg.TimeoutMs != 2000 {
		t.Errorf("TimeoutMs = %d, want 2000", cfg.TimeoutMs)
	}
	if cfg.MinTrust != 0.5 {
		t.Errorf("MinTrust = %f, want 0.5", cfg.MinTrust)
	}
}

// ---------- Internal utility tests ----------

func TestTokenizeForRetrieval(t *testing.T) {
	tokens := tokenizeForRetrieval("The quick brown fox jumps over the lazy dog!")
	// "the" should be removed (stop word).
	for _, tok := range tokens {
		if tok == "the" {
			t.Error("stop word 'the' should be removed")
		}
	}
	// Content words should be present.
	expected := map[string]bool{"quick": true, "brown": true, "fox": true, "jumps": true, "lazy": true, "dog": true}
	found := make(map[string]bool)
	for _, tok := range tokens {
		found[tok] = true
	}
	for w := range expected {
		if !found[w] {
			t.Errorf("expected token %q not found in %v", w, tokens)
		}
	}
}

func TestExtractEntities(t *testing.T) {
	entities := extractEntities("compare Go vs Python")
	if len(entities) < 2 {
		t.Fatalf("expected at least 2 entities, got %d: %v", len(entities), entities)
	}

	foundGo, foundPython := false, false
	for _, e := range entities {
		if strings.Contains(e, "go") {
			foundGo = true
		}
		if strings.Contains(e, "python") {
			foundPython = true
		}
	}
	if !foundGo || !foundPython {
		t.Errorf("expected both 'go' and 'python' in entities: %v", entities)
	}
}

func TestSplitSentences_Retrieval(t *testing.T) {
	text := "Go is fast. Python is readable. Rust is safe."
	sentences := splitSentences(text)
	if len(sentences) != 3 {
		t.Errorf("expected 3 sentences, got %d: %v", len(sentences), sentences)
	}
}

func TestSplitSentences_RetrievalEmpty(t *testing.T) {
	sentences := splitSentences("")
	if sentences != nil {
		t.Error("empty text should return nil")
	}
}

func TestTextSimilarity(t *testing.T) {
	a := tokenizeForRetrieval("Go is a programming language")
	b := tokenizeForRetrieval("Go is a compiled programming language")
	c := tokenizeForRetrieval("The weather is sunny today")

	simAB := textSimilarity(a, b)
	simAC := textSimilarity(a, c)

	if simAB <= simAC {
		t.Errorf("similar texts should have higher similarity: AB=%f, AC=%f", simAB, simAC)
	}
	if simAB <= 0 || simAB > 1 {
		t.Errorf("similarity should be in (0,1], got %f", simAB)
	}
}

func TestJaccardSimilarity(t *testing.T) {
	a := []string{"go", "programming", "language"}
	b := []string{"go", "programming", "language"}
	c := []string{"weather", "sunny", "today"}

	if jaccardSimilarity(a, b) != 1.0 {
		t.Error("identical sets should have Jaccard = 1.0")
	}
	if jaccardSimilarity(a, c) != 0.0 {
		t.Error("disjoint sets should have Jaccard = 0.0")
	}
	if jaccardSimilarity(nil, nil) != 0.0 {
		t.Error("empty sets should have Jaccard = 0.0")
	}
}

func TestBM25Scoring(t *testing.T) {
	r := NewTwoTierRetriever(nil, nil)
	r.IndexDocument("Go programming language concurrency", "doc1")
	r.IndexDocument("Python programming language data science", "doc2")
	r.IndexDocument("weather forecast sunny cloudy", "doc3")

	r.mu.RLock()
	defer r.mu.RUnlock()

	queryTerms := tokenizeForRetrieval("Go programming")
	score1 := r.bm25Score(queryTerms, r.Documents[0])
	score3 := r.bm25Score(queryTerms, r.Documents[2])

	if score1 <= 0 {
		t.Error("matching document should have positive BM25 score")
	}
	if score3 != 0 {
		t.Error("non-matching document should have zero BM25 score")
	}
	if score1 <= score3 {
		t.Error("matching document should score higher than non-matching")
	}
}

func TestRRFMerging(t *testing.T) {
	r := NewTwoTierRetriever(nil, nil)

	lexical := []RetrievalResult{
		{Text: "doc A", Source: "src", Score: 1.0, Tier: TierLexical},
		{Text: "doc B", Source: "src", Score: 0.8, Tier: TierLexical},
	}
	semantic := []RetrievalResult{
		{Text: "doc B", Source: "src", Score: 0.9, Tier: TierSemantic},
		{Text: "doc C", Source: "src", Score: 0.7, Tier: TierSemantic},
	}
	graph := []RetrievalResult{
		{Text: "doc A", Source: "src", Score: 0.6, Tier: TierGraph},
	}

	merged := r.reciprocalRankFusion(lexical, semantic, graph, 5)

	// "doc A" appears in lexical and graph, "doc B" in lexical and semantic.
	// Both should appear, with "doc A" or "doc B" at the top.
	if len(merged) != 3 {
		t.Errorf("expected 3 merged results, got %d", len(merged))
	}

	// Verify no duplicates.
	seen := make(map[string]bool)
	for _, rr := range merged {
		if seen[rr.Text] {
			t.Errorf("duplicate in merged results: %q", rr.Text)
		}
		seen[rr.Text] = true
	}
}

// ---------- Benchmarks ----------

func BenchmarkTwoTierRetriever(b *testing.B) {
	kv := NewKnowledgeVec(mockEmbed, "")
	g := NewCognitiveGraph("")

	// Seed some data.
	for i := 0; i < 100; i++ {
		text := "This is document number " + strings.Repeat("x", i%20+5) + " about various topics."
		kv.AddChunk(text, "bench")
	}
	g.AddEdge("Go", "programming language", RelIsA, "bench")
	g.AddEdge("Python", "programming language", RelIsA, "bench")

	r := NewTwoTierRetriever(kv, g)
	for i := 0; i < 100; i++ {
		text := "This is document number " + strings.Repeat("x", i%20+5) + " about various topics."
		r.IndexDocument(text, "bench")
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		r.Retrieve("programming language document", 10)
	}
}

func BenchmarkQueryDecompose(b *testing.B) {
	qr := NewQueryRewriter()
	queries := []string{
		"compare Go vs Python",
		"what is photosynthesis",
		"why did the Roman Empire fall",
		"how has AI changed over time",
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		qr.Decompose(queries[i%len(queries)])
	}
}

func BenchmarkContradictionCheck(b *testing.B) {
	r := NewTwoTierRetriever(nil, nil)
	ea := NewEvidenceAttributor(r)

	claims := []string{
		"Go is a compiled language.",
		"Python is an interpreted language.",
		"Rust focuses on memory safety.",
		"Go was created by Google.",
		"Python was created by Guido van Rossum.",
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ea.CheckContradictions(claims)
	}
}

func BenchmarkRewriteForRetrieval(b *testing.B) {
	qr := NewQueryRewriter()
	queries := []string{
		"what is machine learning?",
		"how does AI work?",
		"explain NLP please",
		"tell me about quantum physics",
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		qr.RewriteForRetrieval(queries[i%len(queries)])
	}
}

// ---------- Edge case and regression tests ----------

func TestGraphRelationRelevance(t *testing.T) {
	if graphRelationRelevance(RelIsA) != 1.0 {
		t.Error("IsA should have relevance 1.0")
	}
	if graphRelationRelevance(RelRelatedTo) >= graphRelationRelevance(RelIsA) {
		t.Error("RelatedTo should be less relevant than IsA")
	}
}

func TestTopicFreshnessString(t *testing.T) {
	tests := []struct {
		f    TopicFreshness
		want string
	}{
		{FreshnessStatic, "static"},
		{FreshnessSlow, "slow"},
		{FreshnessDynamic, "dynamic"},
		{FreshnessRealtime, "realtime"},
		{TopicFreshness(99), "unknown"},
	}
	for _, tc := range tests {
		if got := tc.f.String(); got != tc.want {
			t.Errorf("TopicFreshness(%d).String() = %q, want %q", tc.f, got, tc.want)
		}
	}
}

func TestMaxAgeFor(t *testing.T) {
	if maxAgeFor(FreshnessRealtime) >= maxAgeFor(FreshnessDynamic) {
		t.Error("realtime maxAge should be less than dynamic")
	}
	if maxAgeFor(FreshnessDynamic) >= maxAgeFor(FreshnessSlow) {
		t.Error("dynamic maxAge should be less than slow")
	}
	if maxAgeFor(FreshnessSlow) >= maxAgeFor(FreshnessStatic) {
		t.Error("slow maxAge should be less than static")
	}
}

func TestSourceAuthority(t *testing.T) {
	if sourceAuthority("kb:physics") <= sourceAuthority("graph:entity") {
		t.Error("kb should rank above graph")
	}
	if sourceAuthority("graph:entity") <= sourceAuthority("inferred:guess") {
		t.Error("graph should rank above inferred")
	}
	if sourceAuthority("inferred:guess") <= sourceAuthority("") {
		t.Error("inferred should rank above empty")
	}
}

func TestScoreTrustBounds(t *testing.T) {
	// Test that trust scores are always in [0,1].
	testCases := []struct {
		source string
		age    time.Duration
		refs   int
	}{
		{"kb:test", 0, 100},
		{"", 10000 * time.Hour, 0},
		{"wiki:article", 1 * time.Hour, 5},
	}

	for _, tc := range testCases {
		ts := ScoreTrust(tc.source, tc.age, tc.refs)
		if ts.Score < 0 || ts.Score > 1 {
			t.Errorf("ScoreTrust(%q, %v, %d).Score = %f, out of [0,1]",
				tc.source, tc.age, tc.refs, ts.Score)
		}
		for name, factor := range ts.Factors {
			if factor < 0 || factor > 1 {
				t.Errorf("ScoreTrust factor %q = %f, out of [0,1]", name, factor)
			}
		}
	}
}

func TestAbsFloat(t *testing.T) {
	if absFloat(-5.0) != 5.0 {
		t.Error("absFloat(-5) should be 5")
	}
	if absFloat(3.0) != 3.0 {
		t.Error("absFloat(3) should be 3")
	}
}

// Ensure isStopWord covers basic cases.
func TestIsRetrieverStop(t *testing.T) {
	if !isRetrieverStop("the") {
		t.Error("'the' should be a stop word")
	}
	if !isRetrieverStop("is") {
		t.Error("'is' should be a stop word")
	}
	if isRetrieverStop("quantum") {
		t.Error("'quantum' should not be a stop word")
	}
}

// Sanity check: NewQueryRewriter should not return nil.
func TestNewQueryRewriter(t *testing.T) {
	qr := NewQueryRewriter()
	if qr == nil {
		t.Fatal("NewQueryRewriter returned nil")
	}
	if qr.compareRe == nil || qr.causalRe == nil || qr.temporalRe == nil || qr.multiRe == nil {
		t.Fatal("patterns should be pre-compiled")
	}
}

// Test the comparison entity extraction directly.
func TestExtractComparisonEntities(t *testing.T) {
	tests := []struct {
		input string
		wantA string
		wantB string
	}{
		{"go vs python", "go", "python"},
		{"go versus python", "go", "python"},
		{"difference between cats and dogs", "cats", "dogs"},
	}

	for _, tc := range tests {
		a, b := extractComparisonEntities(tc.input)
		if a != tc.wantA || b != tc.wantB {
			t.Errorf("extractComparisonEntities(%q) = (%q, %q), want (%q, %q)",
				tc.input, a, b, tc.wantA, tc.wantB)
		}
	}
}

// Verify the absFloat helper is consistent with math.Abs.
func TestAbsFloatConsistency(t *testing.T) {
	vals := []float64{-1.5, 0, 1.5, -0.001, 100.0}
	for _, v := range vals {
		if absFloat(v) != math.Abs(v) {
			t.Errorf("absFloat(%f) != math.Abs(%f)", v, v)
		}
	}
}
