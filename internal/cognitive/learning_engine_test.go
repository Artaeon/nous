package cognitive

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestLearningFactExtraction(t *testing.T) {
	graph := NewCognitiveGraph("")
	le := NewLearningEngine(graph, nil, "")

	// User states a fact
	n := le.LearnFromConversation("Go was created by Google.")
	if n == 0 {
		t.Error("should extract fact from 'Go was created by Google'")
	}

	// Check the graph has the knowledge
	nodes := graph.FindNodes("Go")
	if len(nodes) == 0 {
		t.Error("graph should contain a 'Go' node")
	}

	nodes = graph.FindNodes("Google")
	if len(nodes) == 0 {
		t.Error("graph should contain a 'Google' node")
	}

	t.Logf("Nodes after learning: %d, Edges: %d", graph.NodeCount(), graph.EdgeCount())
}

func TestLearningMultipleFacts(t *testing.T) {
	graph := NewCognitiveGraph("")
	le := NewLearningEngine(graph, nil, "")

	// Multiple facts in one message
	n := le.LearnFromConversation("Stoicera was founded by Raphael. It is based in Vienna. Stoicera is a philosophy company.")
	if n < 2 {
		t.Errorf("should extract at least 2 facts, got %d", n)
	}

	t.Logf("Learned %d facts from compound sentence", n)
	t.Logf("Graph now has %d nodes, %d edges", graph.NodeCount(), graph.EdgeCount())
}

func TestLearningPreferences(t *testing.T) {
	graph := NewCognitiveGraph("")
	le := NewLearningEngine(graph, nil, "")

	n := le.LearnFromConversation("I love programming in Go!")
	if n == 0 {
		t.Error("should learn user preference from 'I love programming in Go'")
	}

	// Check for preference edge
	edges := graph.EdgesFrom("user")
	found := false
	for _, e := range edges {
		if e.Relation == RelPrefers {
			found = true
			t.Logf("Preference: user prefers %s", e.To)
		}
	}
	if !found {
		t.Error("should store user preference in graph")
	}
}

func TestLearningPersonalInfo(t *testing.T) {
	graph := NewCognitiveGraph("")
	le := NewLearningEngine(graph, nil, "")

	le.LearnFromConversation("I live in Vienna.")
	le.LearnFromConversation("I work at Stoicera.")

	edges := graph.EdgesFrom("user")
	hasLocation := false
	hasWork := false
	for _, e := range edges {
		if e.Relation == RelLocatedIn {
			hasLocation = true
			t.Logf("Location: %s", e.To)
		}
		if e.Relation == RelPartOf {
			hasWork = true
			t.Logf("Work: %s", e.To)
		}
	}
	if !hasLocation {
		t.Error("should learn user location")
	}
	if !hasWork {
		t.Error("should learn user workplace")
	}
}

func TestLearningTeaching(t *testing.T) {
	graph := NewCognitiveGraph("")
	le := NewLearningEngine(graph, nil, "")

	n := le.LearnFromConversation("Let me teach you: Rust is a systems programming language.")
	if n == 0 {
		t.Error("should learn from teaching mode input")
	}

	nodes := graph.FindNodes("Rust")
	if len(nodes) == 0 {
		t.Error("graph should contain 'Rust' after teaching")
	}
	t.Logf("Learned %d facts from teaching", n)
}

func TestLearningCorrections(t *testing.T) {
	graph := NewCognitiveGraph("")
	le := NewLearningEngine(graph, nil, "")

	// First learn something
	le.LearnFromConversation("Python was created by Guido.")

	// Then correct
	le.LearnFromConversation("Actually Python was created by Guido van Rossum.")

	// The corrected fact should exist
	nodes := graph.FindNodes("Guido van Rossum")
	if len(nodes) == 0 {
		t.Error("should store corrected fact")
	}
}

func TestLearningPatternAbsorption(t *testing.T) {
	graph := NewCognitiveGraph("")
	le := NewLearningEngine(graph, nil, "")

	// Send multiple well-formed sentences with punctuation (required for absorption)
	le.LearnFromConversation("I think programming is a wonderful activity for creative people.")
	le.LearnFromConversation("I think philosophy is a powerful discipline for understanding reality.")
	le.LearnFromConversation("I believe engineering is the future of modern civilization.")

	patterns := le.LearnedPatterns()
	if len(patterns) == 0 {
		t.Error("should absorb sentence patterns from user speech")
	}
	for _, p := range patterns {
		t.Logf("Pattern: %q (category: %s, usage: %d)", p.Template, p.Category, p.UsageCount)
	}
}

func TestLearningTopicInterest(t *testing.T) {
	graph := NewCognitiveGraph("")
	le := NewLearningEngine(graph, nil, "")

	// Talk about programming a lot (extractKeywords filters words < 3 chars)
	le.LearnFromConversation("programming is great for concurrency")
	le.LearnFromConversation("programming has goroutines")
	le.LearnFromConversation("programming is compiled")
	le.LearnFromConversation("I love programming")

	// Talk about philosophy once
	le.LearnFromConversation("philosophy exists")

	progInterest := le.TopicInterest("programming")
	philInterest := le.TopicInterest("philosophy")

	t.Logf("Programming interest: %d, Philosophy interest: %d", progInterest, philInterest)
	if progInterest == 0 {
		t.Error("programming should have non-zero interest after 4 mentions")
	}
	if progInterest <= philInterest {
		t.Error("programming (4 mentions) should rank higher than philosophy (1 mention)")
	}
}

func TestLearningStats(t *testing.T) {
	graph := NewCognitiveGraph("")
	le := NewLearningEngine(graph, nil, "")

	le.LearnFromConversation("Go was created by Google.")
	le.LearnFromConversation("Stoicera is based in Vienna.")
	le.LearnFromConversation("I love programming.")

	stats := le.Stats()
	t.Logf("Stats: %+v", stats)

	if stats.TotalFacts == 0 {
		t.Error("should have learned some facts")
	}
}

func TestLearningPersistence(t *testing.T) {
	dir := t.TempDir()
	graph := NewCognitiveGraph("")

	// Create engine and learn
	le := NewLearningEngine(graph, nil, dir)
	le.LearnFromConversation("Go was created by Google.")
	le.LearnFromConversation("I think Go is a great language for building things.")

	// Check file was created
	path := filepath.Join(dir, "learning_engine.json")
	if _, err := os.Stat(path); os.IsNotExist(err) {
		t.Error("should persist learning data to disk")
	}

	// Create new engine from same dir — should load patterns
	le2 := NewLearningEngine(graph, nil, dir)
	stats := le2.Stats()
	if stats.PatternsLearned == 0 && stats.FactsFromChat == 0 {
		t.Error("should load learning data from disk")
	}
	t.Logf("Loaded from disk: %d patterns, %d facts", stats.PatternsLearned, stats.FactsFromChat)
}

func TestLearningReport(t *testing.T) {
	graph := NewCognitiveGraph("")
	le := NewLearningEngine(graph, nil, "")

	le.LearnFromConversation("Go was created by Google.")
	le.LearnFromConversation("Stoicera was founded by Raphael.")
	le.LearnFromConversation("I love philosophy and programming.")

	report := le.FormatLearningReport()
	t.Log(report)

	if !strings.Contains(report, "Learning Report") {
		t.Error("report should contain header")
	}
	if !strings.Contains(report, "Knowledge nodes") {
		t.Error("report should show knowledge node count")
	}
}

func TestAbstractToTemplate(t *testing.T) {
	tests := []struct {
		input string
		want  bool // should produce a template
	}{
		{"I think Go is a great language", true},
		{"hi", false},                              // too short
		{"the cat sat on the mat today", true},      // has content words
	}

	for _, tt := range tests {
		tmpl := abstractToTemplate(tt.input)
		got := tmpl != ""
		if got != tt.want {
			t.Errorf("abstractToTemplate(%q) = %q, wantNonEmpty=%v", tt.input, tmpl, tt.want)
		}
		if tmpl != "" {
			t.Logf("%q → %q", tt.input, tmpl)
		}
	}
}

func TestLearningDidYouKnow(t *testing.T) {
	graph := NewCognitiveGraph("")
	le := NewLearningEngine(graph, nil, "")

	n := le.LearnFromConversation("Did you know Go was created by Google?")
	if n == 0 {
		t.Error("should extract fact from 'did you know' pattern")
	}
	t.Logf("Learned %d facts from 'did you know'", n)
}

func TestLearningFavorite(t *testing.T) {
	graph := NewCognitiveGraph("")
	le := NewLearningEngine(graph, nil, "")

	n := le.LearnFromConversation("My favorite language is Go.")
	if n == 0 {
		t.Error("should learn from 'my favorite' pattern")
	}

	nodes := graph.FindNodes("Go")
	if len(nodes) == 0 {
		t.Error("should have Go in graph after favorite declaration")
	}
}
