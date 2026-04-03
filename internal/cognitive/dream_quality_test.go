package cognitive

import (
	"testing"
)

func TestDreamQualityScoring(t *testing.T) {
	g := newTestGraph()
	// Set up two domains.
	g.EnsureNode("physics", NodeConcept)
	g.EnsureNode("quantum mechanics", NodeConcept)
	g.AddEdge("quantum mechanics", "physics", RelIsA, "test")
	g.EnsureNode("history", NodeConcept)
	g.EnsureNode("roman empire", NodeConcept)
	g.AddEdge("roman empire", "history", RelIsA, "test")
	g.EnsureNode("gravity", NodeConcept)
	g.AddEdge("gravity", "physics", RelDomain, "test")

	de := NewDreamEngine(g, nil, nil, nil, nil)

	// Test cross-domain surprise scoring.
	crossDomain := &DreamDiscovery{
		Type:     "connection",
		Summary:  "test",
		Entities: []string{"quantum mechanics", "roman empire"},
		Novel:    true,
	}
	surprise := de.scoreSurprise(crossDomain)
	if surprise < 0.3 {
		t.Errorf("Cross-domain discovery should have high surprise, got %.2f", surprise)
	}

	// Test same-domain surprise scoring.
	sameDomain := &DreamDiscovery{
		Type:     "connection",
		Summary:  "test",
		Entities: []string{"quantum mechanics", "gravity"},
		Novel:    true,
	}
	sameSurprise := de.scoreSurprise(sameDomain)
	// Same-domain may still have surprise from entity rarity, but less than cross-domain.
	t.Logf("Cross-domain surprise: %.2f, Same-domain surprise: %.2f", surprise, sameSurprise)

	// Test single-entity discovery.
	single := &DreamDiscovery{
		Type:     "expansion",
		Summary:  "test",
		Entities: []string{"physics"},
	}
	singleSurprise := de.scoreSurprise(single)
	if singleSurprise > 0.5 {
		t.Errorf("Single-entity discovery should have low surprise, got %.2f", singleSurprise)
	}
}

func TestDreamDeduplication(t *testing.T) {
	g := newTestGraph()
	g.EnsureNode("physics", NodeConcept)
	g.EnsureNode("math", NodeConcept)

	de := NewDreamEngine(g, nil, nil, nil, nil)

	d1 := &DreamDiscovery{
		Type:     "connection",
		Entities: []string{"physics", "math"},
	}
	d2 := &DreamDiscovery{
		Type:     "connection",
		Entities: []string{"math", "physics"}, // same pair, reversed order
	}

	key1 := de.discoveryKey(d1)
	key2 := de.discoveryKey(d2)

	if key1 != key2 {
		t.Errorf("Reversed entity pairs should produce same key: %q != %q", key1, key2)
	}
}

func TestDreamTopDiscoveries(t *testing.T) {
	g := newTestGraph()
	de := NewDreamEngine(g, nil, nil, nil, nil)

	// Manually add discoveries with different quality.
	de.discoveries = []DreamDiscovery{
		{Type: "insight", Summary: "high quality", Confidence: 0.8, Surprise: 0.9, Novel: true},
		{Type: "pattern", Summary: "low quality", Confidence: 0.2, Surprise: 0.1, Novel: false},
		{Type: "connection", Summary: "medium quality", Confidence: 0.5, Surprise: 0.5, Novel: true},
	}

	top := de.TopDiscoveries(2)
	if len(top) != 2 {
		t.Fatalf("Expected 2 top discoveries, got %d", len(top))
	}
	if top[0].Summary != "high quality" {
		t.Errorf("First discovery should be highest quality, got %q", top[0].Summary)
	}
}

func TestDreamIsDifferentDomain(t *testing.T) {
	tests := []struct {
		a, b     string
		expected bool
	}{
		{"quantum mechanics", "classical music", true},
		{"quantum mechanics", "quantum physics", false}, // "quantum" shared
		{"very long sentence that is definitely not a concept at all in any way", "test", false},
	}

	for _, tt := range tests {
		result := isDifferentDomain(tt.a, tt.b)
		if result != tt.expected {
			t.Errorf("isDifferentDomain(%q, %q) = %v, want %v", tt.a, tt.b, result, tt.expected)
		}
	}
}
