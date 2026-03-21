package cognitive

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// -----------------------------------------------------------------------
// Stress Test — hammers Nous hard to find every grammar, fluency, and
// logic issue. Tests articles, conversations, edge cases, and quality.
// -----------------------------------------------------------------------

func setupStress(t *testing.T) (*Composer, *LearningEngine, *CognitiveGraph, *GenerativeEngine) {
	t.Helper()
	dir := t.TempDir()
	graph := NewCognitiveGraph(filepath.Join(dir, "graph.json"))
	semantic := NewSemanticEngine()
	causal := NewCausalEngine()
	patterns := NewPatternDetector()
	composer := NewComposer(graph, semantic, causal, patterns)
	learning := NewLearningEngine(graph, composer, dir)

	packDir := filepath.Join("..", "..", "packages")
	loader := NewPackageLoader(graph, composer.Generative, composer, packDir)
	if _, err := os.Stat(packDir); err == nil {
		loader.LoadAll()
	}

	return composer, learning, graph, composer.Generative
}

// TestStressArticleQuality generates articles for EVERY topic in the packages
// and checks for grammar issues, weird phrases, and quality.
func TestStressArticleQuality(t *testing.T) {
	_, _, graph, gen := setupStress(t)

	// Collect all topics with 3+ facts
	allTopics := []string{
		"Stoicism", "philosophy", "ethics", "epistemology", "logic",
		"existentialism", "Marcus Aurelius", "Seneca", "Socrates", "Plato",
		"Aristotle", "Friedrich Nietzsche", "Immanuel Kant",
		"physics", "chemistry", "biology", "astronomy",
		"quantum mechanics", "general relativity", "Albert Einstein",
		"DNA", "evolution", "thermodynamics", "entropy",
		"the Sun", "the solar system", "the Milky Way", "black hole",
		"gravity", "light", "atom", "speed of light",
		"Vienna", "Austria", "Paris", "France", "Germany",
		"Japan", "China", "India", "Brazil", "Australia", "Antarctica",
		"United States", "Europe", "Africa",
		"the Pacific Ocean", "Mount Everest", "the Nile", "the Mediterranean Sea",
		"Amazon rainforest",
		"the internet", "artificial intelligence", "machine learning",
		"Python", "JavaScript", "C", "Linux", "Docker", "Kubernetes",
		"Git", "blockchain", "cloud computing", "cybersecurity",
		"computer science", "open source",
		"ancient Egypt", "ancient Greece", "the Roman Empire",
		"the Renaissance", "the Industrial Revolution", "World War II",
		"the Cold War", "the space race", "the Moon landing",
		"the French Revolution", "democracy", "the Silk Road",
		"Leonardo da Vinci",
		"music", "classical music", "Mozart", "Beethoven",
		"literature", "Shakespeare", "painting", "the Mona Lisa",
		"architecture", "cinema",
	}

	// Grammar issues to detect
	grammarIssues := []struct {
		pattern string
		desc    string
	}{
		{"you has", "wrong verb agreement with 'you'"},
		{"it are", "wrong verb agreement with 'it'"},
		{"they is", "wrong verb agreement with 'they'"},
		{"a unique", ""}, // this is actually CORRECT, don't flag
		{"an universal", "wrong article (should be 'a universal')"},
		{"an unique", "wrong article (should be 'a unique')"},
		{"an European", "wrong article"},
		{"a hour", "wrong article (should be 'an hour')"},
		{"a honest", "wrong article (should be 'an honest')"},
		{"growed", "wrong past tense of grow"},
		{"builded", "wrong past tense of build"},
		{"runned", "wrong past tense of run"},
		{"  ", "double space"},
		{" .", "space before period"},
		{" ,", "space before comma"},
		{"..", "double period (not ellipsis)"},
		{"the the ", "doubled 'the'"},
		// "a a" removed — "DNA a pivotal" causes false positives
		// "is is" removed — "This is" followed by "is" in different sentences causes false positives
		{"the Roman the Roman", "repeated phrase"},
		{"in in ", "doubled 'in'"},
		{"of of ", "doubled 'of'"},
		{"meriting", "meriting used as predicate adjective"},
	}

	totalArticles := 0
	totalIssues := 0
	totalWords := 0
	shortArticles := 0

	for _, topic := range allTopics {
		edges := graph.EdgesFrom(topic)
		if len(edges) < 2 {
			continue
		}

		var facts []edgeFact
		for _, e := range edges {
			subj := graph.NodeLabel(e.From)
			obj := graph.NodeLabel(e.To)
			if subj == "" {
				subj = e.From
			}
			if obj == "" {
				obj = e.To
			}
			facts = append(facts, edgeFact{Subject: subj, Relation: e.Relation, Object: obj})
		}

		article := gen.ComposeArticle(topic, facts)
		words := len(strings.Fields(article))
		totalArticles++
		totalWords += words

		if words < 100 {
			shortArticles++
			t.Logf("  [SHORT] %s: only %d words", topic, words)
		}

		// Check for grammar issues
		lower := strings.ToLower(article)
		for _, gi := range grammarIssues {
			if gi.pattern == "a unique" {
				continue // correct usage, skip
			}
			if strings.Contains(lower, strings.ToLower(gi.pattern)) {
				totalIssues++
				// Find context
				idx := strings.Index(lower, strings.ToLower(gi.pattern))
				start := idx - 20
				if start < 0 {
					start = 0
				}
				end := idx + len(gi.pattern) + 20
				if end > len(article) {
					end = len(article)
				}
				t.Logf("  [GRAMMAR] %s in '%s': ...%s...", gi.desc, topic, article[start:end])
			}
		}

		// Check for awkward phrases
		awkward := []string{
			"When it comes to in ",
			"Establishing 20",
			"the the ",
			"is known as is",
			"was what",
			"you has to",
			"be used for be",
		}
		for _, awk := range awkward {
			if strings.Contains(article, awk) {
				totalIssues++
				t.Logf("  [AWKWARD] '%s' found in %s article", awk, topic)
			}
		}
	}

	avgWords := 0
	if totalArticles > 0 {
		avgWords = totalWords / totalArticles
	}

	t.Logf("\n=== Article Quality Summary ===")
	t.Logf("  Articles generated: %d", totalArticles)
	t.Logf("  Total words: %d", totalWords)
	t.Logf("  Average words: %d", avgWords)
	t.Logf("  Short articles (<100 words): %d", shortArticles)
	t.Logf("  Grammar issues: %d", totalIssues)

	if totalIssues > 5 {
		t.Errorf("Too many grammar issues: %d", totalIssues)
	}
}

// TestStressConversationalKnowledge tests whether conversational responses
// actually USE the knowledge in the graph (the main weakness we saw).
func TestStressConversationalKnowledge(t *testing.T) {
	composer, _, graph, _ := setupStress(t)
	ctx := &ComposeContext{UserName: "Raphael"}

	// These are factual questions — the Composer should return actual knowledge
	queries := []struct {
		query string
		topic string
	}{
		{"what is Stoicism?", "Stoicism"},
		{"tell me about Albert Einstein", "Albert Einstein"},
		{"what do you know about Vienna?", "Vienna"},
		{"explain quantum mechanics", "quantum mechanics"},
		{"who was Socrates?", "Socrates"},
		{"what is Python?", "Python"},
		{"tell me about the Renaissance", "the Renaissance"},
		{"what is Linux?", "Linux"},
		{"who was Mozart?", "Mozart"},
		{"what is DNA?", "DNA"},
		{"tell me about ancient Greece", "ancient Greece"},
		{"what is gravity?", "gravity"},
		{"tell me about Beethoven", "Beethoven"},
		{"what is the internet?", "the internet"},
		{"what is evolution?", "evolution"},
	}

	knowledgeInResponse := 0
	emptyResponses := 0

	for _, q := range queries {
		// Get facts from graph
		edges := graph.EdgesFrom(q.topic)
		factWords := make(map[string]bool)
		for _, e := range edges {
			for _, w := range strings.Fields(strings.ToLower(e.To)) {
				if len(w) > 3 {
					factWords[w] = true
				}
			}
		}

		resp := composer.Compose(q.query, RespFactual, ctx)
		if resp == nil || resp.Text == "" {
			emptyResponses++
			t.Logf("  [EMPTY] %s", q.query)
			continue
		}

		// Check if the response contains any actual facts
		respLower := strings.ToLower(resp.Text)
		factHits := 0
		for w := range factWords {
			if strings.Contains(respLower, w) {
				factHits++
			}
		}

		hasKnowledge := factHits >= 2
		if hasKnowledge {
			knowledgeInResponse++
		}

		status := "KNOWLEDGE"
		if !hasKnowledge {
			status = "GENERIC"
		}

		text := resp.Text
		if len(text) > 120 {
			text = text[:120] + "..."
		}
		t.Logf("  [%s] %s → %s (fact hits: %d/%d)", status, q.query, text, factHits, len(factWords))
	}

	t.Logf("\n  Knowledge in responses: %d/%d (%.0f%%)",
		knowledgeInResponse, len(queries),
		float64(knowledgeInResponse)/float64(len(queries))*100)
	t.Logf("  Empty responses: %d", emptyResponses)
}

// TestStressLearningAndRecall teaches Nous many facts and tests recall.
func TestStressLearningAndRecall(t *testing.T) {
	_, learning, graph, _ := setupStress(t)

	// Teach Nous a bunch of things
	teachings := []struct {
		input    string
		expectSubj string
		expectObj  string
	}{
		{"My name is Raphael", "user", "raphael"},
		{"I live in Vienna", "user", "vienna"},
		{"I work at Stoicera", "user", "stoicera"},
		{"I love philosophy", "user", "philosophy"},
		{"I enjoy hiking", "user", "hiking"},
		{"I prefer Go over Python", "user", "go"},
		{"My favorite book is Meditations", "user", "meditations"},
		{"Stoicera is a philosophy company", "stoicera", "philosophy"},
		{"Stoicera was founded in 2020", "stoicera", "2020"},
		{"Nous is an AI assistant", "nous", "ai"},
		{"Nous was built by Raphael", "nous", "raphael"},
		{"Vienna is the capital of Austria", "vienna", "austria"},
	}

	for _, teach := range teachings {
		learning.LearnFromConversation(teach.input)
	}

	// Check what was learned
	learned := 0
	for _, teach := range teachings {
		edges := graph.EdgesFrom(teach.expectSubj)
		found := false
		for _, e := range edges {
			if strings.Contains(strings.ToLower(e.To), teach.expectObj) {
				found = true
				break
			}
		}
		status := "LEARNED"
		if !found {
			status = "MISSED"
		} else {
			learned++
		}
		t.Logf("  [%s] %q → %s has %s?", status, teach.input, teach.expectSubj, teach.expectObj)
	}

	t.Logf("\n  Learning accuracy: %d/%d (%.0f%%)",
		learned, len(teachings),
		float64(learned)/float64(len(teachings))*100)
}

// TestStressProseNaturalness generates many responses and checks for
// unnatural patterns that make it obvious it's rule-based.
func TestStressProseNaturalness(t *testing.T) {
	_, _, graph, gen := setupStress(t)

	topics := []string{"Stoicism", "Albert Einstein", "Python", "Vienna", "the Renaissance"}

	for _, topic := range topics {
		edges := graph.EdgesFrom(topic)
		if len(edges) == 0 {
			continue
		}
		var facts []edgeFact
		for _, e := range edges {
			subj := graph.NodeLabel(e.From)
			obj := graph.NodeLabel(e.To)
			if subj == "" {
				subj = e.From
			}
			if obj == "" {
				obj = e.To
			}
			facts = append(facts, edgeFact{Subject: subj, Relation: e.Relation, Object: obj})
		}

		t.Logf("\n=== %s — 3 consecutive articles ===", topic)
		for i := 0; i < 3; i++ {
			article := gen.ComposeArticle(topic, facts)
			words := len(strings.Fields(article))

			// Count unique sentences
			sentences := strings.Split(article, ".")
			uniqueSentences := make(map[string]bool)
			for _, s := range sentences {
				s = strings.TrimSpace(s)
				if len(s) > 10 {
					uniqueSentences[s] = true
				}
			}

			// Check for repetitive patterns
			repeatedPhrases := 0
			chunks := strings.Fields(article)
			for j := 0; j < len(chunks)-5; j++ {
				trigram := strings.Join(chunks[j:j+4], " ")
				count := strings.Count(article, trigram)
				if count > 1 && len(trigram) > 15 {
					repeatedPhrases++
				}
			}

			t.Logf("  [%d] %d words, %d unique sentences, %d repeated 4-grams",
				i+1, words, len(uniqueSentences), repeatedPhrases)

			// Show first 200 chars
			preview := article
			if len(preview) > 250 {
				preview = preview[:250] + "..."
			}
			t.Logf("      %s", preview)
		}
	}
}

// TestStressEdgeCases tests weird inputs and edge cases.
func TestStressEdgeCases(t *testing.T) {
	composer, learning, _, _ := setupStress(t)
	ctx := &ComposeContext{UserName: "Raphael"}

	edgeCases := []struct {
		input    string
		respType ResponseType
		desc     string
	}{
		{"", RespConversational, "empty input"},
		{"a", RespConversational, "single letter"},
		{"???", RespConversational, "just punctuation"},
		{"tell me about something that doesn't exist", RespFactual, "unknown topic"},
		{"asdfghjkl", RespConversational, "gibberish"},
		{"what is the meaning of life?", RespFactual, "philosophical question"},
		{"hello hello hello hello", RespGreeting, "repeated greeting"},
		{"I'm so happy today!!!", RespEmpathetic, "strong positive emotion"},
		{"I'm devastated", RespEmpathetic, "strong negative emotion"},
		{"thank you so much you're amazing", RespThankYou, "effusive thanks"},
		{"goodbye forever", RespFarewell, "dramatic farewell"},
	}

	for _, ec := range edgeCases {
		resp := composer.Compose(ec.input, ec.respType, ctx)
		text := "(nil response)"
		if resp != nil {
			text = resp.Text
			if text == "" {
				text = "(empty string)"
			}
		}
		if len(text) > 100 {
			text = text[:100] + "..."
		}
		t.Logf("  [%s] %q → %s", ec.desc, ec.input, text)

		// Learn from it too
		if ec.input != "" {
			learning.LearnFromConversation(ec.input)
		}
	}
}

// TestStress50TopicArticles generates articles for 50 different topics
// to test vocabulary exhaustion and repetitiveness.
func TestStress50TopicArticles(t *testing.T) {
	_, _, graph, gen := setupStress(t)

	allTopics := []string{
		"Stoicism", "physics", "Vienna", "Python", "Einstein",
		"Mozart", "democracy", "DNA", "Linux", "the Renaissance",
		"gravity", "Shakespeare", "Japan", "evolution", "Beethoven",
		"the internet", "Socrates", "chemistry", "Australia", "cinema",
		"blockchain", "France", "the Nile", "logic", "painting",
	}

	allArticles := make([]string, 0)
	totalWords := 0

	for _, topic := range allTopics {
		// Try both exact and case variations
		edges := graph.EdgesFrom(topic)
		if len(edges) < 2 {
			continue
		}
		var facts []edgeFact
		for _, e := range edges {
			subj := graph.NodeLabel(e.From)
			obj := graph.NodeLabel(e.To)
			if subj == "" {
				subj = e.From
			}
			if obj == "" {
				obj = e.To
			}
			facts = append(facts, edgeFact{Subject: subj, Relation: e.Relation, Object: obj})
		}

		article := gen.ComposeArticle(topic, facts)
		allArticles = append(allArticles, article)
		totalWords += len(strings.Fields(article))
	}

	// Check how many articles share the same opening sentence
	openings := make(map[string]int)
	for _, a := range allArticles {
		dot := strings.Index(a, ".")
		if dot > 0 {
			opening := a[:dot]
			openings[opening]++
		}
	}

	duplicateOpenings := 0
	for opening, count := range openings {
		if count > 1 {
			duplicateOpenings++
			t.Logf("  [REPEATED OPENING x%d] %s", count, opening)
		}
	}

	// Vocabulary diversity across ALL articles
	allText := strings.Join(allArticles, " ")
	allWords := strings.Fields(strings.ToLower(allText))
	uniqueWords := make(map[string]bool)
	for _, w := range allWords {
		uniqueWords[w] = true
	}

	diversity := float64(len(uniqueWords)) / float64(len(allWords)) * 100

	t.Logf("\n=== 50-Topic Stress Test ===")
	t.Logf("  Articles generated: %d", len(allArticles))
	t.Logf("  Total words: %d", totalWords)
	t.Logf("  Average words: %d", totalWords/maxInt(len(allArticles), 1))
	t.Logf("  Unique words: %d", len(uniqueWords))
	t.Logf("  Vocabulary diversity: %.1f%%", diversity)
	t.Logf("  Duplicate openings: %d", duplicateOpenings)

	_ = fmt.Sprint // use fmt
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}
