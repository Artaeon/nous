package cognitive

import (
	"strings"
	"testing"
)

// -----------------------------------------------------------------------
// Multi-Hop Reasoning Tests
// -----------------------------------------------------------------------

// buildMultiHopTestGraph creates a small graph for multi-hop tests:
//
//	Einstein --created_by--> photoelectric effect
//	photoelectric effect --part_of--> quantum mechanics
//	Einstein --is_a--> physicist
//	Bohr --is_a--> physicist
//	Bohr --known_for--> atomic model
//	Go --is_a--> programming language
//	Rust --is_a--> programming language
//	Go --created_by--> Google
//	Rust --created_by--> Mozilla
//	Go --has--> concurrency
//	Rust --has--> memory safety
func buildMultiHopTestGraph() *CognitiveGraph {
	cg := NewCognitiveGraph("")

	cg.EnsureNode("Einstein", NodeEntity)
	cg.EnsureNode("photoelectric effect", NodeConcept)
	cg.EnsureNode("quantum mechanics", NodeConcept)
	cg.EnsureNode("Bohr", NodeEntity)
	cg.EnsureNode("physicist", NodeConcept)
	cg.EnsureNode("atomic model", NodeConcept)
	cg.EnsureNode("Go", NodeEntity)
	cg.EnsureNode("Rust", NodeEntity)
	cg.EnsureNode("programming language", NodeConcept)
	cg.EnsureNode("Google", NodeEntity)
	cg.EnsureNode("Mozilla", NodeEntity)
	cg.EnsureNode("concurrency", NodeProperty)
	cg.EnsureNode("memory safety", NodeProperty)

	cg.AddEdge("Einstein", "photoelectric effect", RelKnownFor, "test")
	cg.AddEdge("photoelectric effect", "quantum mechanics", RelPartOf, "test")
	cg.AddEdge("Einstein", "physicist", RelIsA, "test")
	cg.AddEdge("Bohr", "physicist", RelIsA, "test")
	cg.AddEdge("Bohr", "atomic model", RelKnownFor, "test")
	cg.AddEdge("Go", "programming language", RelIsA, "test")
	cg.AddEdge("Rust", "programming language", RelIsA, "test")
	cg.AddEdge("Go", "Google", RelCreatedBy, "test")
	cg.AddEdge("Rust", "Mozilla", RelCreatedBy, "test")
	cg.AddEdge("Go", "concurrency", RelHas, "test")
	cg.AddEdge("Rust", "memory safety", RelHas, "test")

	return cg
}

func TestMultiHopDirectConnection(t *testing.T) {
	cg := buildMultiHopTestGraph()
	mhr := NewMultiHopReasoner(cg)

	conn := mhr.FindConnection("Einstein", "photoelectric effect")
	if conn == nil {
		t.Fatal("expected a connection result")
	}
	if len(conn.Direct) == 0 {
		t.Fatal("expected at least one direct edge")
	}

	found := false
	for _, d := range conn.Direct {
		if d.Relation == RelKnownFor {
			found = true
			break
		}
	}
	if !found {
		t.Errorf("expected direct known_for edge, got: %+v", conn.Direct)
	}

	if !strings.Contains(conn.Summary, "directly connected") {
		t.Errorf("summary should mention direct connection, got: %q", conn.Summary)
	}
}

func TestMultiHopTwoHopConnection(t *testing.T) {
	cg := buildMultiHopTestGraph()
	mhr := NewMultiHopReasoner(cg)

	// Einstein → photoelectric effect → quantum mechanics
	conn := mhr.FindConnection("Einstein", "quantum mechanics")
	if conn == nil {
		t.Fatal("expected a connection result")
	}

	if len(conn.TwoHop) == 0 {
		t.Fatal("expected at least one two-hop path")
	}

	found := false
	for _, th := range conn.TwoHop {
		if strings.EqualFold(th.Via, "photoelectric effect") {
			found = true
			break
		}
	}
	if !found {
		t.Errorf("expected two-hop through 'photoelectric effect', got: %+v", conn.TwoHop)
	}

	if !strings.Contains(conn.Summary, "photoelectric effect") {
		t.Errorf("summary should mention intermediate node, got: %q", conn.Summary)
	}
}

func TestMultiHopSharedProperties(t *testing.T) {
	cg := buildMultiHopTestGraph()
	mhr := NewMultiHopReasoner(cg)

	// Go and Rust both is_a programming language
	conn := mhr.FindConnection("Go", "Rust")
	if conn == nil {
		t.Fatal("expected a connection result")
	}

	if len(conn.SharedProps) == 0 {
		t.Fatal("expected shared properties")
	}

	found := false
	for _, prop := range conn.SharedProps {
		if strings.EqualFold(prop, "programming language") {
			found = true
			break
		}
	}
	if !found {
		t.Errorf("expected 'programming language' as shared property, got: %v", conn.SharedProps)
	}

	if !strings.Contains(conn.Summary, "share") {
		t.Errorf("summary should mention shared properties, got: %q", conn.Summary)
	}
}

func TestMultiHopNoConnection(t *testing.T) {
	cg := buildMultiHopTestGraph()
	mhr := NewMultiHopReasoner(cg)

	// concurrency and Mozilla have no path between them
	conn := mhr.FindConnection("concurrency", "Mozilla")
	if conn == nil {
		t.Fatal("expected a connection result (even if empty)")
	}

	if len(conn.Direct) != 0 {
		t.Errorf("expected no direct edges, got %d", len(conn.Direct))
	}
	if len(conn.TwoHop) != 0 {
		t.Errorf("expected no two-hop paths, got %d", len(conn.TwoHop))
	}
	if len(conn.SharedProps) != 0 {
		t.Errorf("expected no shared props, got %d", len(conn.SharedProps))
	}

	if !strings.Contains(conn.Summary, "don't see a direct connection") {
		t.Errorf("summary should indicate no connection, got: %q", conn.Summary)
	}
}

func TestExplainRelationFormat(t *testing.T) {
	cg := buildMultiHopTestGraph()
	mhr := NewMultiHopReasoner(cg)

	// Direct connection
	explanation := mhr.ExplainRelation("Einstein", "photoelectric effect")
	if !strings.Contains(explanation, "Einstein") || !strings.Contains(explanation, "photoelectric effect") {
		t.Errorf("explanation should mention both entities, got: %q", explanation)
	}
	if !strings.Contains(explanation, "directly connected") {
		t.Errorf("direct link should produce 'directly connected' text, got: %q", explanation)
	}

	// No connection
	explanation = mhr.ExplainRelation("concurrency", "Mozilla")
	if !strings.Contains(explanation, "don't see") {
		t.Errorf("no-connection explanation should say so, got: %q", explanation)
	}
}

func TestMultiHopNilGraph(t *testing.T) {
	mhr := NewMultiHopReasoner(nil)
	if mhr != nil {
		t.Fatal("expected nil reasoner for nil graph")
	}
}

func TestMultiHopEmptyEntities(t *testing.T) {
	cg := buildMultiHopTestGraph()
	mhr := NewMultiHopReasoner(cg)

	conn := mhr.FindConnection("", "Einstein")
	if conn != nil {
		t.Errorf("expected nil for empty entity A, got: %+v", conn)
	}

	conn = mhr.FindConnection("Einstein", "")
	if conn != nil {
		t.Errorf("expected nil for empty entity B, got: %+v", conn)
	}
}

func TestMultiHopSharedPhysicists(t *testing.T) {
	cg := buildMultiHopTestGraph()
	mhr := NewMultiHopReasoner(cg)

	// Einstein and Bohr both is_a physicist
	conn := mhr.FindConnection("Einstein", "Bohr")
	if conn == nil {
		t.Fatal("expected a connection result")
	}

	found := false
	for _, prop := range conn.SharedProps {
		if strings.EqualFold(prop, "physicist") {
			found = true
			break
		}
	}
	if !found {
		t.Errorf("Einstein and Bohr should share 'physicist', got: %v", conn.SharedProps)
	}
}

func TestRelVerb(t *testing.T) {
	tests := []struct {
		rel  RelType
		want string
	}{
		{RelIsA, "is a"},
		{RelHas, "has"},
		{RelLocatedIn, "is located in"},
		{RelCreatedBy, "was created by"},
		{RelKnownFor, "is known for"},
		{RelPartOf, "is part of"},
		{RelRelatedTo, "is related to"},
	}

	for _, tt := range tests {
		got := relVerb(tt.rel)
		if got != tt.want {
			t.Errorf("relVerb(%q) = %q, want %q", tt.rel, got, tt.want)
		}
	}
}
