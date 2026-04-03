package cognitive

import (
	"strings"
	"testing"
)

func TestSimulationEngine_Simulate(t *testing.T) {
	graph := newTestGraph()
	causal := NewGraphCausalReasoner(graph)
	sim := NewSimulationEngine(graph, causal, nil, nil)

	result := sim.Simulate("What if quantum mechanics didn't exist?", 3)
	if result == nil {
		t.Fatal("expected non-nil simulation result")
	}

	if result.Scenario != "What if quantum mechanics didn't exist?" {
		t.Fatalf("scenario mismatch: %q", result.Scenario)
	}

	if result.Report == "" {
		t.Fatal("expected non-empty report")
	}

	if !strings.Contains(result.Report, "Simulation Report") {
		t.Fatalf("report missing header: %q", result.Report[:min(len(result.Report), 200)])
	}

	if result.Duration == 0 {
		t.Fatal("expected non-zero duration")
	}
}

func TestSimulationEngine_SimulateRemoval(t *testing.T) {
	graph := newTestGraph()
	causal := NewGraphCausalReasoner(graph)
	sim := NewSimulationEngine(graph, causal, nil, nil)

	result := sim.SimulateRemoval("physics")
	if result == nil {
		t.Fatal("expected non-nil simulation result")
	}

	if !strings.Contains(result.Scenario, "removed") {
		t.Fatalf("expected removal scenario, got: %q", result.Scenario)
	}

	if result.Report == "" {
		t.Fatal("expected non-empty report")
	}
}

func TestSimulationEngine_EmptyScenario(t *testing.T) {
	sim := NewSimulationEngine(nil, nil, nil, nil)
	result := sim.Simulate("unknown topic with no graph", 3)
	if result == nil {
		t.Fatal("expected non-nil result even with nil graph")
	}
}

func TestSimulationEngine_StepLimits(t *testing.T) {
	graph := newTestGraph()
	causal := NewGraphCausalReasoner(graph)
	sim := NewSimulationEngine(graph, causal, nil, nil)

	// Steps clamped to max 10
	result := sim.Simulate("test scenario", 100)
	if result == nil {
		t.Fatal("expected non-nil result")
	}
	if len(result.Steps) > 10 {
		t.Fatalf("expected max 10 steps, got %d", len(result.Steps))
	}

	// Steps min 1
	result = sim.Simulate("test", 0)
	if result == nil {
		t.Fatal("expected non-nil result")
	}
}

func TestExtractHypothesis(t *testing.T) {
	tests := []struct {
		input    string
		expected string
	}{
		{"What if gravity didn't exist?", "gravity didn't exist"},
		{"Simulate the impact of AI", "the impact of AI"},
		{"What would happen if the sun exploded?", "the sun exploded"},
		{"plain statement", "plain statement"},
	}

	for _, tt := range tests {
		got := extractHypothesis(tt.input)
		if got != tt.expected {
			t.Errorf("extractHypothesis(%q) = %q, want %q", tt.input, got, tt.expected)
		}
	}
}

func TestIsSimulationQuery(t *testing.T) {
	positives := []string{
		"What if the internet went down?",
		"Simulate a world without electricity",
		"What would happen if gravity reversed?",
		"Predict what happens when AI surpasses human intelligence",
		"What are the consequences of deforestation?",
	}
	for _, q := range positives {
		if !IsSimulationQuery(q) {
			t.Errorf("expected IsSimulationQuery(%q) = true", q)
		}
	}

	negatives := []string{
		"What is gravity?",
		"Hello there",
		"How do I cook pasta?",
		"Calculate 2+2",
	}
	for _, q := range negatives {
		if IsSimulationQuery(q) {
			t.Errorf("expected IsSimulationQuery(%q) = false", q)
		}
	}
}

func TestIsRemovalQuery(t *testing.T) {
	if !IsRemovalQuery("What if gravity were removed?") {
		t.Error("expected removal query detected")
	}
	if !IsRemovalQuery("What if the sun disappeared?") {
		t.Error("expected removal query detected")
	}
	if IsRemovalQuery("What if gravity were stronger?") {
		t.Error("expected non-removal query")
	}
}

// newTestGraph creates a small test graph with physics/science topics.
func newTestGraph() *CognitiveGraph {
	g := NewCognitiveGraph("")

	// EnsureNode creates nodes; AddEdge auto-creates if needed.
	g.EnsureNode("physics", NodeConcept)
	g.EnsureNode("quantum mechanics", NodeConcept)
	g.EnsureNode("relativity", NodeConcept)
	g.EnsureNode("Albert Einstein", NodeEntity)
	g.EnsureNode("Isaac Newton", NodeEntity)
	g.EnsureNode("gravity", NodeConcept)
	g.EnsureNode("science", NodeConcept)

	g.AddEdge("quantum mechanics", "physics", RelIsA, "test")
	g.AddEdge("relativity", "physics", RelIsA, "test")
	g.AddEdge("Albert Einstein", "relativity", RelCreatedBy, "test")
	g.AddEdge("Albert Einstein", "quantum mechanics", RelInfluencedBy, "test")
	g.AddEdge("Isaac Newton", "gravity", RelCreatedBy, "test")
	g.AddEdge("gravity", "physics", RelIsA, "test")
	g.AddEdge("physics", "science", RelIsA, "test")
	g.AddEdge("relativity", "gravity", RelRelatedTo, "test")
	g.AddEdge("quantum mechanics", "relativity", RelRelatedTo, "test")

	return g
}
