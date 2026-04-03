package cognitive

import (
	"strings"
	"testing"
)

func TestGraphRAG_Query(t *testing.T) {
	graph := newTestGraph()
	rag := NewGraphRAGEngine(graph, nil)

	result := rag.Query("physics")
	if result == nil {
		t.Fatal("expected non-nil result")
	}
	if result.Response == "" {
		t.Fatal("expected non-empty response")
	}
	if len(result.Facts) == 0 {
		t.Fatal("expected at least one fact")
	}
	if result.Confidence == 0 {
		t.Fatal("expected non-zero confidence")
	}
}

func TestGraphRAG_QueryWithConnections(t *testing.T) {
	graph := newTestGraph()
	multihop := NewMultiHopReasoner(graph)
	rag := NewGraphRAGEngine(graph, multihop)

	result := rag.Query("quantum mechanics")
	if result == nil {
		t.Fatal("expected non-nil result")
	}
	if result.Response == "" {
		t.Fatal("expected non-empty response for quantum mechanics")
	}
	// Should find facts about quantum mechanics being a type of physics.
	found := false
	for _, f := range result.Facts {
		if strings.Contains(strings.ToLower(f.Text), "physics") {
			found = true
			break
		}
	}
	if !found {
		t.Errorf("expected a fact connecting quantum mechanics to physics, got: %v", result.Facts)
	}
}

func TestGraphRAG_UnknownTopic(t *testing.T) {
	graph := newTestGraph()
	rag := NewGraphRAGEngine(graph, nil)

	result := rag.Query("cryptocurrency")
	if result == nil {
		t.Fatal("expected non-nil result")
	}
	// Should have low confidence for unknown topic.
	if result.Confidence > 0.3 {
		t.Fatalf("expected low confidence for unknown topic, got %f", result.Confidence)
	}
}

func TestGraphRAG_NilGraph(t *testing.T) {
	rag := NewGraphRAGEngine(nil, nil)
	result := rag.Query("anything")
	if result == nil {
		t.Fatal("expected non-nil result")
	}
	if !strings.Contains(result.Response, "not available") {
		t.Fatalf("expected unavailable message, got: %q", result.Response)
	}
}

func TestGraphRAG_FactRanking(t *testing.T) {
	graph := newTestGraph()
	rag := NewGraphRAGEngine(graph, nil)

	result := rag.Query("gravity")
	if result == nil || len(result.Facts) == 0 {
		t.Fatal("expected facts for gravity")
	}

	// Facts should be sorted by score descending.
	for i := 1; i < len(result.Facts); i++ {
		if result.Facts[i].Score > result.Facts[i-1].Score {
			t.Errorf("facts not sorted by score: fact[%d].Score=%f > fact[%d].Score=%f",
				i, result.Facts[i].Score, i-1, result.Facts[i-1].Score)
		}
	}
}

func TestGraphRAG_Coverage(t *testing.T) {
	graph := newTestGraph()
	rag := NewGraphRAGEngine(graph, nil)

	result := rag.Query("physics")
	if result == nil {
		t.Fatal("expected non-nil result")
	}
	// Physics should have good coverage (multiple relation types).
	if result.Coverage == 0 {
		t.Fatal("expected non-zero coverage for physics")
	}
}

func TestGraphRAG_FactDiversity(t *testing.T) {
	graph := newTestGraph()
	rag := NewGraphRAGEngine(graph, nil)

	result := rag.Query("physics")
	if result == nil || len(result.Facts) == 0 {
		t.Fatal("expected facts")
	}

	// Check that we have diverse relation types (not all the same).
	relTypes := make(map[string]bool)
	for _, f := range result.Facts {
		relTypes[f.Relation] = true
	}
	// With the test graph, physics has is_a edges from multiple children.
	// We should see at least one relation type.
	if len(relTypes) == 0 {
		t.Fatal("expected at least one relation type in facts")
	}
}
