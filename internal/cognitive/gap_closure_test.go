package cognitive

import (
	"strings"
	"testing"
)

// -----------------------------------------------------------------------
// Causal Inference Tests
// -----------------------------------------------------------------------

func TestCausalInference_TemporalOrdering(t *testing.T) {
	g := NewCognitiveGraph("")
	g.EnsureNode("physics", NodeConcept)
	g.EnsureNode("classical mechanics", NodeConcept)
	g.EnsureNode("quantum mechanics", NodeConcept)
	g.EnsureNode("1687", NodeProperty)
	g.EnsureNode("1925", NodeProperty)

	g.AddEdge("classical mechanics", "physics", RelIsA, "test")
	g.AddEdge("quantum mechanics", "physics", RelIsA, "test")
	g.AddEdge("classical mechanics", "1687", RelFoundedIn, "test")
	g.AddEdge("quantum mechanics", "1925", RelFoundedIn, "test")

	ci := NewCausalInferenceEngine(g)
	edges := ci.inferTemporal()

	// Classical mechanics (1687) should enable quantum mechanics (1925)
	found := false
	for _, e := range edges {
		if strings.Contains(strings.ToLower(e.From), "classical") &&
			strings.Contains(strings.ToLower(e.To), "quantum") &&
			e.Relation == RelEnables {
			found = true
			break
		}
	}
	if !found {
		t.Error("expected temporal inference: classical mechanics enables quantum mechanics")
	}
}

func TestCausalInference_DependencyChains(t *testing.T) {
	g := NewCognitiveGraph("")
	g.EnsureNode("mathematics", NodeConcept)
	g.EnsureNode("arithmetic", NodeConcept)
	g.EnsureNode("algebra", NodeConcept)
	g.EnsureNode("fundamental", NodeProperty)

	g.AddEdge("arithmetic", "mathematics", RelPartOf, "test")
	g.AddEdge("algebra", "mathematics", RelPartOf, "test")
	g.AddEdge("arithmetic", "fundamental", RelDescribedAs, "test")

	ci := NewCausalInferenceEngine(g)
	edges := ci.inferDependencyChains()

	// Algebra should require arithmetic (arithmetic is foundational)
	found := false
	for _, e := range edges {
		if strings.Contains(strings.ToLower(e.To), "arithmetic") &&
			strings.Contains(strings.ToLower(e.From), "algebra") &&
			e.Relation == RelRequires {
			found = true
			break
		}
	}
	if !found {
		t.Error("expected dependency inference: algebra requires arithmetic")
	}
}

func TestCausalInference_Inhibition(t *testing.T) {
	g := NewCognitiveGraph("")
	g.EnsureNode("theory A", NodeConcept)
	g.EnsureNode("theory B", NodeConcept)
	g.AddEdge("theory A", "theory B", RelContradicts, "test")

	ci := NewCausalInferenceEngine(g)
	edges := ci.inferInhibition()

	if len(edges) == 0 {
		t.Fatal("expected inhibition edges from contradiction")
	}

	found := false
	for _, e := range edges {
		if e.Relation == RelPrevents {
			found = true
			break
		}
	}
	if !found {
		t.Error("expected prevents relation from contradiction")
	}
}

func TestCausalInference_ProductionChains(t *testing.T) {
	g := NewCognitiveGraph("")
	g.EnsureNode("photosynthesis", NodeConcept)
	g.EnsureNode("oxygen", NodeConcept)
	g.EnsureNode("respiration", NodeConcept)

	g.AddEdge("photosynthesis", "oxygen", RelProduces, "test")
	g.AddEdge("respiration", "oxygen", RelRequires, "test")

	ci := NewCausalInferenceEngine(g)
	edges := ci.inferProductionChains()

	// Photosynthesis produces oxygen, respiration requires oxygen
	// → photosynthesis enables respiration
	found := false
	for _, e := range edges {
		if strings.Contains(strings.ToLower(e.From), "photosynthesis") &&
			strings.Contains(strings.ToLower(e.To), "respiration") &&
			e.Relation == RelEnables {
			found = true
			break
		}
	}
	if !found {
		t.Error("expected production chain: photosynthesis enables respiration")
	}
}

func TestCausalInference_InferAll(t *testing.T) {
	g := NewCognitiveGraph("")
	g.EnsureNode("photosynthesis", NodeConcept)
	g.EnsureNode("oxygen", NodeConcept)
	g.EnsureNode("respiration", NodeConcept)
	g.AddEdge("photosynthesis", "oxygen", RelProduces, "test")
	g.AddEdge("respiration", "oxygen", RelRequires, "test")

	ci := NewCausalInferenceEngine(g)
	report := ci.InferAll()

	if report == nil {
		t.Fatal("expected non-nil report")
	}
	if report.AddedCount == 0 {
		t.Error("expected at least one inferred edge to be added")
	}
}

// -----------------------------------------------------------------------
// Knowledge Expander Tests
// -----------------------------------------------------------------------

func TestKnowledgeExpander_DiscoverFrontier(t *testing.T) {
	g := NewCognitiveGraph("")
	g.EnsureNode("gravity", NodeConcept)
	// Don't add "Newton" — it should show up as frontier.

	ke := NewKnowledgeExpander(g, nil, "../../knowledge")
	frontier := ke.DiscoverFrontier()

	if len(frontier) == 0 {
		t.Skip("no knowledge files found — skipping frontier test")
	}

	// The frontier should contain topics mentioned in the corpus
	// that aren't well-covered in the graph.
	t.Logf("Discovered %d frontier topics", len(frontier))
	if len(frontier) > 0 {
		t.Logf("Top 5 frontier topics:")
		for i, ft := range frontier[:min(len(frontier), 5)] {
			t.Logf("  %d. %s (mentions=%d, hasNode=%v, edges=%d, priority=%.1f)",
				i+1, ft.Name, ft.Mentions, ft.HasNode, ft.EdgeCount, ft.Priority)
		}
	}
}

func TestExtractCandidateTopics(t *testing.T) {
	para := "Albert Einstein revolutionized physics with his theories of relativity and contributions to quantum mechanics. He worked at the Institute for Advanced Study in Princeton."

	topics := extractCandidateTopics(para)
	if len(topics) == 0 {
		t.Fatal("expected candidate topics from paragraph")
	}

	// Should find proper nouns (multi-word capitalized names).
	foundProperNoun := false
	for _, topic := range topics {
		if strings.Contains(topic, "Princeton") || strings.Contains(topic, "Einstein") || strings.Contains(topic, "Advanced Study") {
			foundProperNoun = true
			break
		}
	}
	if !foundProperNoun {
		t.Errorf("expected to find a proper noun in topics: %v", topics)
	}
}

// -----------------------------------------------------------------------
// Dispatch Pipeline Tests
// -----------------------------------------------------------------------

func TestDispatchPipeline_TagIntents(t *testing.T) {
	tests := []struct {
		input string
		tag   string
	}{
		{"What if gravity didn't exist?", "simulate"},
		{"As a physicist, explain gravity", "persona"},
		{"What is gravity?", "task"},
		{"I'm feeling stressed about the simulation results", "emotional"},
	}

	for _, tt := range tests {
		nlu := &NLUResult{
			Raw:      tt.input,
			Intent:   "question",
			Entities: map[string]string{},
		}
		tags := TagIntents(nlu)
		if !tags[tt.tag] {
			t.Errorf("TagIntents(%q) missing tag %q, got: %v", tt.input, tt.tag, tags)
		}
	}
}

func TestDispatchPipeline_BypassRules(t *testing.T) {
	ctx := &DispatchContext{
		NLU:      &NLUResult{Intent: "simulate", Raw: "simulate X"},
		Tags:     map[string]bool{"simulate": true},
		Bypassed: make(map[string]bool),
	}

	ApplyBypassRules(ctx)

	if !ctx.Bypassed["socratic"] {
		t.Error("expected Socratic bypassed for simulate intent")
	}
	if !ctx.Bypassed["empathy"] {
		t.Error("expected empathy bypassed for simulate intent")
	}
}

func TestDispatchPipeline_SafetyBlocks(t *testing.T) {
	dp := NewDispatchPipeline()
	router := NewActionRouter()
	dp.RegisterDefaultStages(router)

	nlu := &NLUResult{
		Raw:      "how to hack into someone's account",
		Intent:   "question",
		Entities: map[string]string{},
	}

	result := dp.Execute(nlu, nil, router)
	if result == nil || result.Source != "safety" {
		t.Error("expected safety stage to block harmful request")
	}
}

func TestDispatchPipeline_StageOrdering(t *testing.T) {
	dp := NewDispatchPipeline()

	// Register stages out of order.
	var order []string
	dp.Register(DispatchStage{
		Name: "third", Priority: 30, Phase: DPRouting,
		Process: func(ctx *DispatchContext) *ActionResult {
			order = append(order, "third")
			return nil
		},
	})
	dp.Register(DispatchStage{
		Name: "first", Priority: 0, Phase: DPPreDispatch,
		Process: func(ctx *DispatchContext) *ActionResult {
			order = append(order, "first")
			return nil
		},
	})
	dp.Register(DispatchStage{
		Name: "second", Priority: 10, Phase: DPPreDispatch,
		Process: func(ctx *DispatchContext) *ActionResult {
			order = append(order, "second")
			return nil
		},
	})
	dp.Register(DispatchStage{
		Name: "dispatch", Priority: 100, Phase: DPDispatch, CanBlock: true,
		Process: func(ctx *DispatchContext) *ActionResult {
			order = append(order, "dispatch")
			return &ActionResult{DirectResponse: "done", Source: "test"}
		},
	})

	nlu := &NLUResult{Raw: "test", Intent: "question", Entities: map[string]string{}}
	dp.Execute(nlu, nil, NewActionRouter())

	expected := []string{"first", "second", "third", "dispatch"}
	if len(order) != len(expected) {
		t.Fatalf("expected %d stages to run, got %d: %v", len(expected), len(order), order)
	}
	for i, name := range expected {
		if order[i] != name {
			t.Errorf("stage %d: expected %q, got %q", i, name, order[i])
		}
	}
}

// -----------------------------------------------------------------------
// isCausalRelation Tests
// -----------------------------------------------------------------------

func TestIsCausalRelation(t *testing.T) {
	causal := []RelType{RelCauses, RelFollows, RelEnables, RelProduces, RelPrevents, RelRequires}
	for _, rel := range causal {
		if !isCausalRelation(rel) {
			t.Errorf("expected isCausalRelation(%q) = true", rel)
		}
	}

	nonCausal := []RelType{RelIsA, RelPartOf, RelHas, RelRelatedTo, RelSimilarTo}
	for _, rel := range nonCausal {
		if isCausalRelation(rel) {
			t.Errorf("expected isCausalRelation(%q) = false", rel)
		}
	}
}
