package cognitive

import (
	"fmt"
	"strings"
	"testing"
)

func TestArticleCurrentLength(t *testing.T) {
	graph := NewCognitiveGraph("")
	semantic := NewSemanticEngine()
	causal := NewCausalEngine()
	patterns := NewPatternDetector()
	composer := NewComposer(graph, semantic, causal, patterns)
	learning := NewLearningEngine(graph, composer, "")

	// Teach rich knowledge about Go
	facts := []string{
		"Go is a programming language.",
		"Go was created by Google.",
		"Go was designed by Robert Griesemer, Rob Pike, and Ken Thompson.",
		"Go is used for backend development.",
		"Go is used for cloud computing.",
		"Go features goroutines for concurrency.",
		"Go has a simple syntax.",
		"Go is statically typed.",
		"Go compiles to machine code.",
		"Go was first released in 2009.",
		"Go is open source.",
		"Go is used by Uber, Dropbox, and Docker.",
		"Go has a built-in garbage collector.",
		"Go supports interfaces for polymorphism.",
		"Go is fast and efficient.",
		"Go is described as simple and readable.",
	}
	for _, f := range facts {
		learning.LearnFromConversation(f)
	}

	// Test 1: Current factual response
	t.Log("=== Current Factual Response ===")
	resp := composer.Compose("tell me everything about Go", RespFactual, &ComposeContext{UserName: "Raphael"})
	if resp != nil {
		words := len(strings.Fields(resp.Text))
		t.Logf("Words: %d", words)
		t.Logf("Text:\n%s", resp.Text)
	}

	// Test 2: Current creative text
	t.Log("\n=== Current Creative Text ===")
	graphFacts := gatherAllFacts(graph, "Go")
	if composer.Generative != nil && len(graphFacts) > 0 {
		text := composer.Generative.ComposeCreativeText("Go", graphFacts)
		words := len(strings.Fields(text))
		t.Logf("Words: %d", words)
		t.Logf("Text:\n%s", text)
	}

	// Test 3: Article writer (to be built)
	t.Log("\n=== Article (to be built) ===")
	if composer.Generative != nil && len(graphFacts) > 0 {
		article := composer.Generative.ComposeArticle("Go", graphFacts)
		words := len(strings.Fields(article))
		t.Logf("Words: %d", words)
		t.Logf("Text:\n%s", article)
	}
}

// gatherAllFacts collects all facts about a topic from the graph.
func gatherAllFacts(graph *CognitiveGraph, topic string) []edgeFact {
	var facts []edgeFact
	graph.mu.RLock()
	defer graph.mu.RUnlock()

	// Find all nodes matching the topic
	topicLower := strings.ToLower(topic)
	var nodeIDs []string
	if ids, ok := graph.byLabel[topicLower]; ok {
		nodeIDs = append(nodeIDs, ids...)
	}
	id := nodeID(topicLower)
	if _, exists := graph.nodes[id]; exists {
		found := false
		for _, nid := range nodeIDs {
			if nid == id {
				found = true
				break
			}
		}
		if !found {
			nodeIDs = append(nodeIDs, id)
		}
	}

	for _, nid := range nodeIDs {
		node := graph.nodes[nid]
		if node == nil {
			continue
		}
		for _, edge := range graph.outEdges[nid] {
			to := graph.nodes[edge.To]
			if to == nil {
				continue
			}
			facts = append(facts, edgeFact{
				Subject:  node.Label,
				Relation: edge.Relation,
				Object:   to.Label,
			})
		}
	}
	return facts
}

func TestArticleWriter(t *testing.T) {
	g := NewGenerativeEngine()

	facts := []edgeFact{
		{Subject: "Go", Relation: RelIsA, Object: "programming language"},
		{Subject: "Go", Relation: RelCreatedBy, Object: "Google"},
		{Subject: "Go", Relation: RelUsedFor, Object: "backend development"},
		{Subject: "Go", Relation: RelUsedFor, Object: "cloud computing"},
		{Subject: "Go", Relation: RelHas, Object: "goroutines"},
		{Subject: "Go", Relation: RelDescribedAs, Object: "fast and efficient"},
		{Subject: "Go", Relation: RelDescribedAs, Object: "simple and readable"},
		{Subject: "Go", Relation: RelHas, Object: "garbage collector"},
		{Subject: "Go", Relation: RelFoundedIn, Object: "2009"},
		{Subject: "Go", Relation: RelRelatedTo, Object: "systems programming"},
	}

	t.Log("=== Article: Go ===")
	article := g.ComposeArticle("Go", facts)
	words := len(strings.Fields(article))
	t.Logf("Words: %d\n", words)
	t.Log(article)

	if words < 200 {
		t.Errorf("article too short: %d words, want at least 200", words)
	}

	// Test with different topic
	t.Log("\n=== Article: Stoicera ===")
	stoiceraFacts := []edgeFact{
		{Subject: "Stoicera", Relation: RelIsA, Object: "philosophy company"},
		{Subject: "Stoicera", Relation: RelFoundedBy, Object: "Raphael"},
		{Subject: "Stoicera", Relation: RelLocatedIn, Object: "Vienna"},
		{Subject: "Stoicera", Relation: RelUsedFor, Object: "inner peace"},
		{Subject: "Stoicera", Relation: RelRelatedTo, Object: "Stoicism"},
		{Subject: "Stoicera", Relation: RelDescribedAs, Object: "innovative"},
	}
	article2 := g.ComposeArticle("Stoicera", stoiceraFacts)
	words2 := len(strings.Fields(article2))
	t.Logf("Words: %d\n", words2)
	t.Log(article2)

	if words2 < 150 {
		t.Errorf("article too short: %d words, want at least 150", words2)
	}

	// Uniqueness test — generate same article 3 times
	t.Log("\n=== Uniqueness Check ===")
	articles := make(map[string]bool)
	for i := 0; i < 5; i++ {
		a := g.ComposeArticle("Go", facts)
		articles[a] = true
		w := len(strings.Fields(a))
		t.Logf("Version %d: %d words", i+1, w)
	}
	t.Logf("Unique articles: %d/5", len(articles))
	if len(articles) < 3 {
		t.Errorf("expected at least 3 unique articles, got %d", len(articles))
	}
}

func TestArticleWordCount(t *testing.T) {
	g := NewGenerativeEngine()

	// Rich fact set — 15 facts to work with
	facts := []edgeFact{
		{Subject: "Rust", Relation: RelIsA, Object: "systems programming language"},
		{Subject: "Rust", Relation: RelCreatedBy, Object: "Mozilla"},
		{Subject: "Rust", Relation: RelUsedFor, Object: "memory-safe systems programming"},
		{Subject: "Rust", Relation: RelUsedFor, Object: "WebAssembly"},
		{Subject: "Rust", Relation: RelUsedFor, Object: "embedded systems"},
		{Subject: "Rust", Relation: RelHas, Object: "ownership model"},
		{Subject: "Rust", Relation: RelHas, Object: "pattern matching"},
		{Subject: "Rust", Relation: RelHas, Object: "zero-cost abstractions"},
		{Subject: "Rust", Relation: RelDescribedAs, Object: "safe and concurrent"},
		{Subject: "Rust", Relation: RelDescribedAs, Object: "fast as C"},
		{Subject: "Rust", Relation: RelRelatedTo, Object: "C++"},
		{Subject: "Rust", Relation: RelRelatedTo, Object: "systems programming"},
		{Subject: "Rust", Relation: RelFoundedIn, Object: "2010"},
		{Subject: "Rust", Relation: RelCauses, Object: "fewer memory bugs"},
		{Subject: "Rust", Relation: RelOffers, Object: "fearless concurrency"},
	}

	article := g.ComposeArticle("Rust", facts)
	words := len(strings.Fields(article))
	fmt.Printf("Rust article: %d words\n", words)
	t.Log(article)

	if words < 300 {
		t.Errorf("want at least 300 words for 15 facts, got %d", words)
	}
}
