package cognitive

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// -----------------------------------------------------------------------
// Full Pipeline Test — simulates real conversations with Nous end-to-end.
// Tests: knowledge packages, learning from conversation, memory recall,
// world knowledge queries, multi-turn conversations, response quality.
//
// This is the closest thing to "talking to Nous" in a test.
// -----------------------------------------------------------------------

// setupFullPipeline creates the complete cognitive stack with packages loaded.
func setupFullPipeline(t *testing.T) (*Composer, *LearningEngine, *ActionRouter, *CognitiveGraph, *PackageLoader) {
	t.Helper()
	dir := t.TempDir()

	graph := NewCognitiveGraph(filepath.Join(dir, "graph.json"))
	semantic := NewSemanticEngine()
	causal := NewCausalEngine()
	patterns := NewPatternDetector()
	composer := NewComposer(graph, semantic, causal, patterns)
	learning := NewLearningEngine(graph, composer, dir)

	ar := NewActionRouter()
	ar.CogGraph = graph
	ar.Composer = composer
	ar.Semantic = semantic
	ar.Causal = causal
	ar.Patterns = patterns

	// Load real knowledge packages
	packDir := filepath.Join("..", "..", "packages")
	loader := NewPackageLoader(graph, composer.Generative, composer, packDir)
	if _, err := os.Stat(packDir); err == nil {
		results, err := loader.LoadAll()
		if err != nil {
			t.Fatalf("failed to load packages: %v", err)
		}
		totalFacts := 0
		for _, r := range results {
			totalFacts += r.FactsLoaded
		}
		t.Logf("Loaded %d packages with %d facts", len(results), totalFacts)
	}

	return composer, learning, ar, graph, loader
}

func TestFullPipelineConversation(t *testing.T) {
	composer, learning, ar, graph, _ := setupFullPipeline(t)

	ctx := &ComposeContext{
		UserName: "Raphael",
	}

	t.Log("\n========================================")
	t.Log("  NOUS FULL PIPELINE TEST")
	t.Log("  Testing: learning, memory, knowledge")
	t.Log("========================================\n")

	// ---------------------------------------------------------------
	// Phase 1: Teach Nous about the user
	// ---------------------------------------------------------------
	t.Log("=== PHASE 1: Teaching Nous (User Facts) ===\n")

	teachInputs := []string{
		"My name is Raphael and I'm from Vienna.",
		"I'm building an AI called Nous.",
		"I love philosophy, especially Stoicism.",
		"I work at Stoicera, a philosophy company.",
		"My favorite programming language is Go.",
		"I enjoy hiking in the Austrian Alps.",
	}

	for _, input := range teachInputs {
		learning.LearnFromConversation(input)
		resp := composer.Compose(input, RespConversational, ctx)
		t.Logf("  User: %s", input)
		if resp != nil && resp.Text != "" {
			t.Logf("  Nous: %s", resp.Text)
			composer.RecordTurn(input, resp.Text)
		}
		t.Log()
	}

	// ---------------------------------------------------------------
	// Phase 2: Test Memory — does Nous remember what we taught?
	// ---------------------------------------------------------------
	t.Log("\n=== PHASE 2: Memory Recall ===\n")

	// The learning engine stores personal facts under "user" as the subject
	// (e.g. "My name is Raphael" → user -[described_as]-> Raphael)
	// and extracts triples from structured sentences
	userEdges := graph.EdgesFrom("user")
	t.Logf("  Facts learned about user: %d", len(userEdges))
	for _, e := range userEdges {
		t.Logf("    %s -[%s]-> %s (source: %s)", e.From, e.Relation, e.To, e.Source)
	}

	// Also check for explicit entity facts from triple extraction
	allPersonalSubjects := []string{"Raphael", "raphael", "Nous", "nous", "Stoicera", "stoicera",
		"Go", "philosophy", "Stoicism", "Vienna", "hiking"}
	totalPersonal := 0
	for _, subj := range allPersonalSubjects {
		edges := graph.EdgesFrom(subj)
		if len(edges) > 0 {
			for _, e := range edges {
				if e.Source == "conversation" || e.Source == "teaching" {
					t.Logf("    %s -[%s]-> %s (source: %s)", e.From, e.Relation, e.To, e.Source)
					totalPersonal++
				}
			}
		}
	}
	t.Logf("  Total personal facts learned: %d (user: %d + entities: %d)", len(userEdges)+totalPersonal, len(userEdges), totalPersonal)

	if len(userEdges)+totalPersonal == 0 {
		t.Error("FAIL: Nous learned nothing from conversation!")
	}

	// ---------------------------------------------------------------
	// Phase 3: Test World Knowledge (from packages)
	// ---------------------------------------------------------------
	t.Log("\n=== PHASE 3: World Knowledge Queries ===\n")

	knowledgeQueries := []struct {
		query    string
		topic    string
		respType ResponseType
	}{
		{"what is Stoicism?", "Stoicism", RespFactual},
		{"tell me about Albert Einstein", "Albert Einstein", RespFactual},
		{"what do you know about Vienna?", "Vienna", RespFactual},
		{"what is quantum mechanics?", "quantum mechanics", RespFactual},
		{"tell me about Go", "Go", RespFactual},
		{"what is the Milky Way?", "the Milky Way", RespFactual},
		{"who was Socrates?", "Socrates", RespFactual},
		{"tell me about Linux", "Linux", RespFactual},
		{"what was the Renaissance?", "the Renaissance", RespFactual},
		{"who was Mozart?", "Mozart", RespFactual},
	}

	knowledgeHits := 0
	for _, q := range knowledgeQueries {
		edges := graph.EdgesFrom(q.topic)
		resp := composer.Compose(q.query, q.respType, ctx)

		hasKnowledge := len(edges) > 0
		hasResponse := resp != nil && resp.Text != "" && len(resp.Text) > 20

		status := "PASS"
		if !hasKnowledge {
			status = "NO KNOWLEDGE"
		} else if !hasResponse {
			status = "WEAK RESPONSE"
		} else {
			knowledgeHits++
		}

		t.Logf("  [%s] %s", status, q.query)
		t.Logf("    Graph facts: %d", len(edges))
		if hasResponse {
			// Truncate for readability
			text := resp.Text
			if len(text) > 200 {
				text = text[:200] + "..."
			}
			t.Logf("    Response: %s", text)
			composer.RecordTurn(q.query, resp.Text)
		}
		t.Log()
	}

	t.Logf("  Knowledge coverage: %d/%d queries answered from packages", knowledgeHits, len(knowledgeQueries))

	// ---------------------------------------------------------------
	// Phase 4: Generate Full Articles from Package Knowledge
	// ---------------------------------------------------------------
	t.Log("\n=== PHASE 4: Full Article Generation ===\n")

	articleTopics := []string{"Stoicism", "Albert Einstein", "Vienna", "Python", "the Roman Empire"}

	for _, topic := range articleTopics {
		edges := graph.EdgesFrom(topic)
		if len(edges) == 0 {
			t.Logf("  [SKIP] %s — no edges in graph", topic)
			continue
		}

		var facts []edgeFact
		for _, e := range edges {
			facts = append(facts, edgeFact{Subject: e.From, Relation: e.Relation, Object: e.To})
		}

		article := composer.Generative.ComposeArticle(topic, facts)
		words := len(strings.Fields(article))
		t.Logf("  === %s (%d words, %d facts) ===", topic, words, len(facts))
		// Show first 300 chars
		preview := article
		if len(preview) > 400 {
			preview = preview[:400] + "..."
		}
		t.Logf("  %s", preview)
		t.Log()

		if words < 100 {
			t.Errorf("Article about %s too short: %d words", topic, words)
		}
	}

	// ---------------------------------------------------------------
	// Phase 5: Multi-turn Conversation Flow
	// ---------------------------------------------------------------
	t.Log("\n=== PHASE 5: Natural Conversation ===\n")

	conversation := []struct {
		input    string
		respType ResponseType
	}{
		{"hey Nous!", RespGreeting},
		{"what do you know about philosophy?", RespFactual},
		{"that's interesting! what about Stoicism specifically?", RespFactual},
		{"who were the main Stoic philosophers?", RespFactual},
		{"I've been reading Marcus Aurelius lately", RespConversational},
		{"can you tell me something about Vienna?", RespFactual},
		{"thanks for all that knowledge!", RespThankYou},
		{"see you later!", RespFarewell},
	}

	t.Log("  --- Conversation Start ---")
	responses := make([]string, 0)
	for _, turn := range conversation {
		resp := composer.Compose(turn.input, turn.respType, ctx)
		text := ""
		if resp != nil {
			text = resp.Text
		}

		t.Logf("  User: %s", turn.input)
		t.Logf("  Nous: %s", text)
		t.Log()

		if text != "" {
			composer.RecordTurn(turn.input, text)
			responses = append(responses, text)
		}
	}
	t.Log("  --- Conversation End ---")

	// Check uniqueness
	seen := make(map[string]bool)
	for _, r := range responses {
		seen[r] = true
	}
	t.Logf("\n  Response uniqueness: %d/%d unique", len(seen), len(responses))

	// ---------------------------------------------------------------
	// Phase 6: Response Uniqueness Across Runs
	// ---------------------------------------------------------------
	t.Log("\n=== PHASE 6: Uniqueness Test ===\n")

	// Ask the same question 10 times
	question := "what is Stoicism?"
	stoicismResponses := make(map[string]bool)
	for i := 0; i < 10; i++ {
		resp := composer.Compose(question, RespFactual, ctx)
		if resp != nil && resp.Text != "" {
			stoicismResponses[resp.Text] = true
		}
	}
	t.Logf("  'What is Stoicism?' — %d/10 unique responses", len(stoicismResponses))

	// Ask about Einstein 10 times
	einsteinResponses := make(map[string]bool)
	for i := 0; i < 10; i++ {
		resp := composer.Compose("tell me about Albert Einstein", RespFactual, ctx)
		if resp != nil && resp.Text != "" {
			einsteinResponses[resp.Text] = true
		}
	}
	t.Logf("  'Tell me about Einstein' — %d/10 unique responses", len(einsteinResponses))

	// ---------------------------------------------------------------
	// Phase 7: Total Knowledge Assessment
	// ---------------------------------------------------------------
	t.Log("\n=== PHASE 7: Knowledge Assessment ===\n")

	totalNodes := graph.NodeCount()
	totalEdges := graph.EdgeCount()

	// Count topics we can write articles about
	topicCandidates := []string{
		"Stoicism", "Go", "Rust", "Python", "JavaScript", "Linux",
		"Albert Einstein", "Socrates", "Plato", "Aristotle",
		"Vienna", "Paris", "Japan", "Australia",
		"quantum mechanics", "general relativity", "DNA", "evolution",
		"the Renaissance", "the Industrial Revolution", "World War II",
		"Shakespeare", "Mozart", "Beethoven",
		"philosophy", "physics", "chemistry", "biology",
		"artificial intelligence", "machine learning",
	}

	articulateTopics := 0
	for _, topic := range topicCandidates {
		edges := graph.EdgesFrom(topic)
		if len(edges) >= 3 {
			articulateTopics++
		}
	}

	t.Logf("  Graph nodes: %d", totalNodes)
	t.Logf("  Graph edges: %d", totalEdges)
	t.Logf("  Topics with 3+ facts: %d/%d (%.0f%%)",
		articulateTopics, len(topicCandidates),
		float64(articulateTopics)/float64(len(topicCandidates))*100)
	t.Logf("  Adjective pool: %d words", len(adjSlots))
	t.Logf("  Quality noun pool: %d words", len(qualityNouns))
	t.Logf("  Impact noun pool: %d words", len(impactNouns))

	// ---------------------------------------------------------------
	// Summary
	// ---------------------------------------------------------------
	t.Log("\n========================================")
	t.Log("  SUMMARY")
	t.Log("========================================")
	t.Logf("  Engine: Rule-based generative (zero LLM)")
	t.Logf("  Knowledge: %d nodes, %d edges", totalNodes, totalEdges)
	t.Logf("  Topics articulated: %d/%d", articulateTopics, len(topicCandidates))
	t.Logf("  Memory: learned from %d user messages", len(teachInputs))
	t.Logf("  World knowledge queries: %d/%d answered", knowledgeHits, len(knowledgeQueries))
	t.Logf("  Stoicism uniqueness: %d/10", len(stoicismResponses))
	t.Logf("  Einstein uniqueness: %d/10", len(einsteinResponses))
	t.Logf("  Conversation uniqueness: %d/%d", len(seen), len(responses))
	t.Log("========================================")

	// Hard assertions
	if articulateTopics < 15 {
		t.Errorf("Expected at least 15 articulable topics, got %d", articulateTopics)
	}
	if len(stoicismResponses) < 3 {
		t.Errorf("Expected at least 3 unique Stoicism responses, got %d", len(stoicismResponses))
	}

	_ = ar
	_ = fmt.Sprintf // use fmt
}

func TestNLUToComposerPipeline(t *testing.T) {
	composer, _, ar, _, _ := setupFullPipeline(t)

	ctx := &ComposeContext{
		UserName: "Raphael",
	}

	t.Log("\n=== NLU → Composer Pipeline ===\n")

	// Test that ClassifyForComposer works for various inputs
	inputs := []struct {
		query    string
		expected string // rough category
	}{
		{"hello", "greeting"},
		{"what is Stoicism?", "factual"},
		{"tell me about Go", "factual/explain"},
		{"how does quantum mechanics work?", "explain"},
		{"thanks!", "thank_you"},
		{"goodbye", "farewell"},
		{"I feel stressed today", "empathetic"},
		{"what do you think about AI?", "opinion"},
	}

	for _, tc := range inputs {
		respType := ar.ClassifyForComposer(tc.query)
		resp := composer.Compose(tc.query, respType, ctx)

		text := "(no response)"
		if resp != nil && resp.Text != "" {
			text = resp.Text
			if len(text) > 150 {
				text = text[:150] + "..."
			}
		}

		t.Logf("  Input:    %s", tc.query)
		t.Logf("  Type:     %d (%s)", respType, tc.expected)
		t.Logf("  Response: %s", text)
		t.Log()

		if resp == nil || resp.Text == "" {
			t.Errorf("No response for %q (classified as %d)", tc.query, respType)
		}
	}
}
