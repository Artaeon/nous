package cognitive

import (
	"strings"
	"testing"
)

func TestPersonaSpecialization(t *testing.T) {
	g := newTestGraph()
	g.EnsureNode("dark matter", NodeConcept)
	g.EnsureNode("physics", NodeConcept)
	g.AddEdge("dark matter", "physics", RelIsA, "test")
	g.EnsureNode("universe", NodeConcept)
	g.AddEdge("dark matter", "universe", RelPartOf, "test")

	pe := NewPersonaEngine(g)

	// First query.
	result1 := pe.Answer("explain dark matter", "physicist")
	if result1.Confidence == 0 {
		t.Error("Expected non-zero confidence for physicist answering dark matter")
	}

	physicist := pe.GetPersona("physicist")
	if physicist.QueriesHandled != 1 {
		t.Errorf("Expected 1 query handled, got %d", physicist.QueriesHandled)
	}

	// Second query on same topic should have higher confidence.
	result2 := pe.Answer("explain dark matter", "physicist")
	if result2.Confidence < result1.Confidence {
		t.Errorf("Second query should have >= confidence (%.3f < %.3f)", result2.Confidence, result1.Confidence)
	}

	if physicist.QueriesHandled != 2 {
		t.Errorf("Expected 2 queries handled, got %d", physicist.QueriesHandled)
	}

	// Expertise profile should list dark matter.
	profile := pe.ExpertiseProfile("physicist")
	if !strings.Contains(profile, "Physicist") {
		t.Error("Profile should contain persona name")
	}
	if !strings.Contains(profile, "2") {
		t.Error("Profile should mention queries handled")
	}
}

func TestPersonaRecordInteraction(t *testing.T) {
	g := newTestGraph()
	pe := NewPersonaEngine(g)

	pe.RecordInteraction("historian", "roman empire", []string{"founded in 753 BC"})
	pe.RecordInteraction("historian", "roman empire", []string{"founded in 753 BC"})
	pe.RecordInteraction("historian", "greek philosophy", []string{"Socrates"})

	historian := pe.GetPersona("historian")
	if historian == nil {
		t.Fatal("Historian persona not found")
	}
	if historian.QueriesHandled != 3 {
		t.Errorf("Expected 3 queries, got %d", historian.QueriesHandled)
	}
	if historian.TopicsAnswered["roman empire"] != 2 {
		t.Errorf("Expected roman empire answered 2 times, got %d", historian.TopicsAnswered["roman empire"])
	}
}

func TestPersonaFrameVerbUsage(t *testing.T) {
	g := newTestGraph()
	g.EnsureNode("inflation", NodeConcept)
	g.EnsureNode("economics", NodeConcept)
	g.AddEdge("inflation", "economics", RelIsA, "test")

	pe := NewPersonaEngine(g)

	// Answer should use frame verbs for variety.
	result := pe.Answer("explain inflation", "economist")
	if result.Response == "" {
		t.Error("Expected non-empty response from economist")
	}
	t.Logf("Economist response: %s", result.Response)
}
