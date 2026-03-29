package cognitive

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// setupTestGraph creates a small graph with known facts for testing.
func setupTestGraph() *CognitiveGraph {
	cg := NewCognitiveGraph("")

	// Sky / light scattering facts.
	cg.EnsureNode("sky", NodeConcept)
	cg.EnsureNode("blue", NodeProperty)
	cg.EnsureNode("light scattering", NodeConcept)
	cg.EnsureNode("Rayleigh scattering", NodeConcept)
	cg.EnsureNode("atmosphere", NodeConcept)
	cg.EnsureNode("sunlight", NodeConcept)

	cg.AddEdge("sky", "blue", RelHas, "test")
	cg.AddEdge("sky", "atmosphere", RelPartOf, "test")
	cg.AddEdge("light scattering", "Rayleigh scattering", RelIsA, "test")
	cg.AddEdge("Rayleigh scattering", "blue", RelCauses, "test")
	cg.AddEdge("sunlight", "light scattering", RelCauses, "test")
	cg.AddEdge("atmosphere", "light scattering", RelHas, "test")

	// Gravity facts.
	cg.EnsureNode("gravity", NodeConcept)
	cg.EnsureNode("mass", NodeProperty)
	cg.EnsureNode("orbits", NodeConcept)
	cg.EnsureNode("tides", NodeConcept)
	cg.EnsureNode("general relativity", NodeConcept)
	cg.EnsureNode("Albert Einstein", NodeEntity)

	cg.AddEdge("gravity", "mass", RelCauses, "test")
	cg.AddEdge("gravity", "orbits", RelCauses, "test")
	cg.AddEdge("gravity", "tides", RelCauses, "test")
	cg.AddEdge("gravity", "general relativity", RelRelatedTo, "test")
	cg.AddEdge("general relativity", "Albert Einstein", RelCreatedBy, "test")

	// Programming language facts.
	cg.EnsureNode("Python", NodeEntity)
	cg.EnsureNode("Rust", NodeEntity)
	cg.EnsureNode("programming language", NodeConcept)
	cg.EnsureNode("memory safety", NodeProperty)
	cg.EnsureNode("dynamic typing", NodeProperty)
	cg.EnsureNode("machine learning", NodeConcept)
	cg.EnsureNode("systems programming", NodeConcept)

	cg.AddEdge("Python", "programming language", RelIsA, "test")
	cg.AddEdge("Rust", "programming language", RelIsA, "test")
	cg.AddEdge("Rust", "memory safety", RelHas, "test")
	cg.AddEdge("Python", "dynamic typing", RelHas, "test")
	cg.AddEdge("Python", "machine learning", RelUsedFor, "test")
	cg.AddEdge("Rust", "systems programming", RelUsedFor, "test")

	return cg
}

func TestDeepReasoner_Decompose(t *testing.T) {
	dr := NewDeepReasoner(setupTestGraph(), "")

	tests := []struct {
		question string
		minSubs  int
		contains []string
	}{
		{
			question: "Why is the sky blue?",
			minSubs:  2,
			contains: []string{"What is", "What causes"},
		},
		{
			question: "How does gravity affect tides?",
			minSubs:  3,
			contains: []string{"What is gravity", "What is tides", "related"},
		},
		{
			question: "What would happen if gravity disappeared?",
			minSubs:  3,
			contains: []string{"What is", "depends on", "effects"},
		},
		{
			question: "Is Rust better than Python?",
			minSubs:  3,
			contains: []string{"properties of Rust", "properties of Python", "compare"},
		},
		{
			question: "What is the relationship between gravity and orbits?",
			minSubs:  3,
			contains: []string{"What is gravity", "What is orbits", "related"},
		},
		{
			question: "How does photosynthesis work?",
			minSubs:  3,
			contains: []string{"What is", "parts of", "function"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.question, func(t *testing.T) {
			subs := dr.Decompose(tt.question)
			if len(subs) < tt.minSubs {
				t.Errorf("Decompose(%q) returned %d sub-questions, want >= %d: %v",
					tt.question, len(subs), tt.minSubs, subs)
				return
			}

			joined := strings.Join(subs, " | ")
			for _, want := range tt.contains {
				if !strings.Contains(strings.ToLower(joined), strings.ToLower(want)) {
					t.Errorf("Decompose(%q) sub-questions %q missing %q",
						tt.question, joined, want)
				}
			}
		})
	}
}

func TestDeepReasoner_Decompose_Empty(t *testing.T) {
	dr := NewDeepReasoner(setupTestGraph(), "")
	subs := dr.Decompose("")
	if len(subs) != 0 {
		t.Errorf("Decompose empty should return nil, got %v", subs)
	}
}

func TestDeepReasoner_Decompose_NonDeep(t *testing.T) {
	dr := NewDeepReasoner(setupTestGraph(), "")
	// A simple factual question should not decompose.
	subs := dr.Decompose("What is the capital of France?")
	if len(subs) != 0 {
		t.Errorf("Decompose(factual) should return nil, got %v", subs)
	}
}

func TestDeepReasoner_Why(t *testing.T) {
	graph := setupTestGraph()
	dr := NewDeepReasoner(graph, "")

	// "Why does gravity cause tides?" decomposes to:
	//   "What is gravity cause tides?" → topic "gravity" (found in graph)
	//   "What causes gravity cause tides?" → topic "gravity" (found)
	// We use a question whose captured topic maps to a graph node.
	result := dr.Reason("Why does gravity exist?")
	if result == nil {
		t.Fatal("Reason('Why does gravity exist?') returned nil")
	}

	if result.Question != "Why does gravity exist?" {
		t.Errorf("Question = %q, want %q", result.Question, "Why does gravity exist?")
	}

	if len(result.Steps) < 2 {
		t.Errorf("expected >= 2 steps, got %d", len(result.Steps))
	}

	if result.FinalAnswer == "" {
		t.Error("FinalAnswer is empty")
	}

	if result.Trace == "" {
		t.Error("Trace is empty")
	}

	if result.Confidence <= 0 {
		t.Errorf("Confidence should be > 0, got %f", result.Confidence)
	}

	// The trace should contain step markers.
	if !strings.Contains(result.Trace, "Step 1") {
		t.Errorf("Trace should contain 'Step 1', got: %s", result.Trace)
	}
}

func TestDeepReasoner_HowAffect(t *testing.T) {
	graph := setupTestGraph()
	dr := NewDeepReasoner(graph, "")

	result := dr.Reason("How does gravity affect orbits?")
	if result == nil {
		t.Fatal("Reason('How does gravity affect orbits?') returned nil")
	}

	if len(result.Steps) < 2 {
		t.Errorf("expected >= 2 steps, got %d", len(result.Steps))
	}

	if result.FinalAnswer == "" {
		t.Error("FinalAnswer is empty")
	}

	// Should mention gravity or orbits in the answer.
	ansLower := strings.ToLower(result.FinalAnswer)
	if !strings.Contains(ansLower, "gravity") && !strings.Contains(ansLower, "orbit") {
		t.Errorf("FinalAnswer should mention gravity or orbits, got: %s", result.FinalAnswer)
	}
}

func TestDeepReasoner_Chain(t *testing.T) {
	graph := setupTestGraph()
	dr := NewDeepReasoner(graph, "")

	// "Why does gravity cause orbits?" → decomposes to sub-questions about "gravity"
	result := dr.Reason("Why does gravity cause orbits?")
	if result == nil {
		t.Fatal("Reason returned nil")
	}

	// Verify chain: each step after the first should reference the previous.
	foundChained := false
	for i := 1; i < len(result.Steps); i++ {
		step := result.Steps[i]
		if step.Conclusion == "" {
			continue // skip steps with no data
		}
		if step.Premise != "" {
			foundChained = true
		}
	}
	if len(result.Steps) > 1 && !foundChained {
		t.Error("multi-step chain should have at least one step referencing a previous conclusion")
	}

	// The trace should have multiple steps.
	stepCount := strings.Count(result.Trace, "Step ")
	if stepCount < 2 {
		t.Errorf("Trace should have >= 2 steps, got %d in: %s", stepCount, result.Trace)
	}
}

func TestDeepReasoner_WithKnowledgeText(t *testing.T) {
	// Create a temporary knowledge directory with a test file.
	dir := t.TempDir()
	content := `Gravity is a fundamental force of nature that attracts objects with mass toward each other. It is responsible for keeping planets in orbit around the sun and for the formation of tides on Earth.

Light scattering occurs when light interacts with particles in the atmosphere. Rayleigh scattering is the primary reason the sky appears blue, as shorter wavelengths of light are scattered more than longer wavelengths.`

	err := os.WriteFile(filepath.Join(dir, "physics.txt"), []byte(content), 0644)
	if err != nil {
		t.Fatal(err)
	}

	// Use an empty graph so we can confirm knowledge_text is used.
	graph := NewCognitiveGraph("")
	dr := NewDeepReasoner(graph, dir)

	// "Why does gravity exist?" decomposes to sub-questions about "gravity"
	// which should be found in the knowledge text file.
	result := dr.Reason("Why does gravity exist?")
	if result == nil {
		t.Fatal("Reason with knowledge text returned nil")
	}

	if result.FinalAnswer == "" {
		t.Error("FinalAnswer is empty even with knowledge text available")
	}

	// Check that knowledge_text was used as a source.
	hasTextSource := false
	for _, s := range result.Steps {
		if s.Source == "knowledge_text" {
			hasTextSource = true
			break
		}
	}
	if !hasTextSource {
		t.Error("expected at least one step sourced from knowledge_text")
	}
}

func TestDeepReasoner_NilGraph(t *testing.T) {
	dr := NewDeepReasoner(nil, "")
	result := dr.Reason("Why is the sky blue?")
	if result != nil {
		t.Error("expected nil result with nil graph")
	}
}

func TestDeepReasoner_EmptyQuestion(t *testing.T) {
	dr := NewDeepReasoner(setupTestGraph(), "")
	result := dr.Reason("")
	if result != nil {
		t.Error("expected nil result for empty question")
	}
}

func TestIsDeepQuestion(t *testing.T) {
	deep := []string{
		"Why is the sky blue?",
		"How does gravity affect tides?",
		"What would happen if the sun disappeared?",
		"What is the relationship between DNA and RNA?",
		"How does photosynthesis work?",
		"Is Python better than Rust?",
	}
	for _, q := range deep {
		if !IsDeepQuestion(q) {
			t.Errorf("IsDeepQuestion(%q) = false, want true", q)
		}
	}

	notDeep := []string{
		"What is the capital of France?",
		"Tell me a joke",
		"Hello there",
		"How are you?",
		"Set a timer for 5 minutes",
	}
	for _, q := range notDeep {
		if IsDeepQuestion(q) {
			t.Errorf("IsDeepQuestion(%q) = true, want false", q)
		}
	}
}

func TestDeepReasoner_LowerFirst(t *testing.T) {
	tests := []struct {
		in   string
		want string
	}{
		{"", ""},
		{"Hello", "hello"},
		{"hello", "hello"},
		{"ABC", "ABC"}, // second char uppercase → keep unchanged
	}
	for _, tt := range tests {
		got := lowerFirst(tt.in)
		if got != tt.want {
			t.Errorf("lowerFirst(%q) = %q, want %q", tt.in, got, tt.want)
		}
	}
}

func TestDeepReasoner_SplitSentences(t *testing.T) {
	text := "First sentence. Second sentence. Third."
	got := splitSentences(text)
	if len(got) < 2 {
		t.Errorf("splitSentences returned %d sentences, want >= 2: %v", len(got), got)
	}
	// The existing splitSentences strips trailing periods during split.
	if len(got) > 0 && !strings.Contains(got[0], "First") {
		t.Errorf("first sentence should contain 'First', got %q", got[0])
	}
}

func TestDeepReasoner_Confidence(t *testing.T) {
	graph := setupTestGraph()
	dr := NewDeepReasoner(graph, "")

	result := dr.Reason("Why does gravity exist?")
	if result == nil {
		t.Fatal("Reason returned nil")
	}

	if result.Confidence <= 0 || result.Confidence > 1.0 {
		t.Errorf("Confidence should be in (0, 1.0], got %f", result.Confidence)
	}
}
