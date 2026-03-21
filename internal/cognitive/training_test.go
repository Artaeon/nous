package cognitive

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// TestTrainAndCompare trains all NLG subsystems on the knowledge corpus
// and compares article quality before and after training.
func TestTrainAndCompare(t *testing.T) {
	// ── Setup: load knowledge packages into graph ──
	dir := t.TempDir()
	graph := NewCognitiveGraph(filepath.Join(dir, "graph.json"))
	semantic := NewSemanticEngine()
	causal := NewCausalEngine()
	patterns := NewPatternDetector()

	packDir := filepath.Join("..", "..", "packages")
	if _, err := os.Stat(packDir); err != nil {
		t.Skip("packages directory not found")
	}

	// Create UNTRAINED composer for baseline
	composerBaseline := NewComposer(graph, semantic, causal, patterns)
	loaderBaseline := NewPackageLoader(graph, composerBaseline.Generative, composerBaseline, packDir)
	loaderBaseline.LoadAll()

	// ── Baseline: generate articles WITHOUT corpus training ──
	topics := []string{
		"Stoicism", "Albert Einstein", "DNA", "Python",
		"the Renaissance", "Mozart", "quantum mechanics",
		"evolution", "Linux", "Socrates",
	}

	type articleResult struct {
		Topic     string
		Text      string
		WordCount int
	}

	generateArticles := func(gen *GenerativeEngine, label string) []articleResult {
		var results []articleResult
		for _, topic := range topics {
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
			results = append(results, articleResult{Topic: topic, Text: article, WordCount: words})
		}
		return results
	}

	baselineArticles := generateArticles(composerBaseline.Generative, "BASELINE")

	// ── Train: load all corpus files ──
	corpusDir := filepath.Join("..", "..", "knowledge")
	if _, err := os.Stat(corpusDir); err != nil {
		t.Skip("knowledge directory not found")
	}

	// Create a TRAINED composer
	graph2 := NewCognitiveGraph(filepath.Join(dir, "graph2.json"))
	semantic2 := NewSemanticEngine()
	composerTrained := NewComposer(graph2, semantic2, causal, patterns)
	loaderTrained := NewPackageLoader(graph2, composerTrained.Generative, composerTrained, packDir)
	loaderTrained.LoadAll()

	// Load and train on all corpus files
	corpusFiles, _ := filepath.Glob(filepath.Join(corpusDir, "*.txt"))
	totalBytes := 0
	totalSentences := 0
	for _, f := range corpusFiles {
		data, err := os.ReadFile(f)
		if err != nil {
			continue
		}
		text := string(data)
		totalBytes += len(data)

		// Feed to semantic engine (builds co-occurrence)
		semantic2.IngestText(text)

		// Feed to Markov + templates via composer
		composerTrained.IngestContent(text)

		// Count sentences trained
		sentences := splitIntoSentences(text)
		totalSentences += len(sentences)
	}

	// Rebuild embeddings from updated co-occurrence
	if composerTrained.Generative.embeddings != nil {
		semantic2.mu.RLock()
		composerTrained.Generative.embeddings.BuildFromCooccurrence(semantic2.cooccurrence)
		semantic2.mu.RUnlock()
	}

	t.Logf("\n=== TRAINING STATS ===")
	t.Logf("  Corpus files: %d", len(corpusFiles))
	t.Logf("  Corpus size: %.1f KB", float64(totalBytes)/1024)
	t.Logf("  Sentences trained: %d", totalSentences)
	if composerTrained.Generative.markov != nil {
		t.Logf("  Markov trigrams: %d", composerTrained.Generative.markov.Size())
		t.Logf("  Markov tokens: %d", composerTrained.Generative.markov.TotalTokens())
	}
	if composerTrained.Generative.templates != nil {
		t.Logf("  Learned templates: %d", composerTrained.Generative.templates.Size())
	}
	if composerTrained.Generative.embeddings != nil {
		t.Logf("  Embedding vectors: %d", composerTrained.Generative.embeddings.Size())
	}

	// ── Generate trained articles ──
	trainedArticles := generateArticles(composerTrained.Generative, "TRAINED")

	// ── Compare ──
	t.Logf("\n=== ARTICLE COMPARISON ===")

	baselineUniqueWords := make(map[string]bool)
	trainedUniqueWords := make(map[string]bool)
	baselineTotalWords := 0
	trainedTotalWords := 0

	for i := 0; i < len(baselineArticles) && i < len(trainedArticles); i++ {
		ba := baselineArticles[i]
		ta := trainedArticles[i]

		t.Logf("\n--- %s ---", ba.Topic)
		t.Logf("  Baseline: %d words", ba.WordCount)
		t.Logf("  Trained:  %d words", ta.WordCount)

		// Show first 200 chars of each
		bPreview := ba.Text
		if len(bPreview) > 250 {
			bPreview = bPreview[:250] + "..."
		}
		tPreview := ta.Text
		if len(tPreview) > 250 {
			tPreview = tPreview[:250] + "..."
		}
		t.Logf("  [BASELINE] %s", bPreview)
		t.Logf("  [TRAINED]  %s", tPreview)

		for _, w := range strings.Fields(strings.ToLower(ba.Text)) {
			baselineUniqueWords[w] = true
		}
		for _, w := range strings.Fields(strings.ToLower(ta.Text)) {
			trainedUniqueWords[w] = true
		}
		baselineTotalWords += ba.WordCount
		trainedTotalWords += ta.WordCount
	}

	baselineDiversity := float64(len(baselineUniqueWords)) / float64(maxInt(baselineTotalWords, 1)) * 100
	trainedDiversity := float64(len(trainedUniqueWords)) / float64(maxInt(trainedTotalWords, 1)) * 100

	t.Logf("\n=== QUALITY METRICS ===")
	t.Logf("  Baseline: %d articles, %d total words, %d unique words, %.1f%% diversity",
		len(baselineArticles), baselineTotalWords, len(baselineUniqueWords), baselineDiversity)
	t.Logf("  Trained:  %d articles, %d total words, %d unique words, %.1f%% diversity",
		len(trainedArticles), trainedTotalWords, len(trainedUniqueWords), trainedDiversity)

	// Check no grammar issues
	grammarIssues := 0
	for _, ta := range trainedArticles {
		issues := checkGrammarIssues(ta.Text)
		grammarIssues += len(issues)
		for _, issue := range issues {
			t.Logf("  [GRAMMAR] %s: %s", ta.Topic, issue)
		}
	}
	t.Logf("  Grammar issues (trained): %d", grammarIssues)
}

// TestTrainedConversation runs a full conversation through the trained pipeline.
func TestTrainedConversation(t *testing.T) {
	dir := t.TempDir()
	graph := NewCognitiveGraph(filepath.Join(dir, "graph.json"))
	semantic := NewSemanticEngine()
	causal := NewCausalEngine()
	patterns := NewPatternDetector()
	composer := NewComposer(graph, semantic, causal, patterns)
	learning := NewLearningEngine(graph, composer, dir)

	// Load packages
	packDir := filepath.Join("..", "..", "packages")
	if _, err := os.Stat(packDir); err != nil {
		t.Skip("packages directory not found")
	}
	loader := NewPackageLoader(graph, composer.Generative, composer, packDir)
	loader.LoadAll()

	// Train on corpus
	corpusDir := filepath.Join("..", "..", "knowledge")
	if _, err := os.Stat(corpusDir); err != nil {
		t.Skip("knowledge directory not found")
	}
	corpusFiles, _ := filepath.Glob(filepath.Join(corpusDir, "*.txt"))
	for _, f := range corpusFiles {
		data, _ := os.ReadFile(f)
		if len(data) > 0 {
			semantic.IngestText(string(data))
			composer.IngestContent(string(data))
		}
	}
	// Rebuild embeddings
	if composer.Generative.embeddings != nil {
		semantic.mu.RLock()
		composer.Generative.embeddings.BuildFromCooccurrence(semantic.cooccurrence)
		semantic.mu.RUnlock()
	}

	// Setup router
	router := NewActionRouter()
	router.CogGraph = graph
	router.Composer = composer
	router.Semantic = semantic
	router.Causal = causal
	router.Patterns = patterns

	turn := 0
	chat := func(msg string) string {
		turn++
		learning.LearnFromConversation(msg)

		nlu := NewNLU().Understand(msg)
		result := router.Execute(nlu, nil)

		text := ""
		if result != nil {
			text = result.DirectResponse
			if text == "" {
				text = result.Data
			}
		}
		if text == "" {
			text = "(no response)"
		}

		// Record turn for self-improvement
		composer.RecordTurn(msg, text)

		return text
	}

	fmt.Println("\n" + strings.Repeat("=", 70))
	fmt.Println("  TRAINED CONVERSATION TEST")
	fmt.Println(strings.Repeat("=", 70))

	conversations := []struct {
		section  string
		messages []string
	}{
		{
			"Getting to know each other",
			[]string{
				"Hey Nous, good morning!",
				"My name is Raphael and I'm building a local AI",
				"I love philosophy, especially Stoicism",
			},
		},
		{
			"Knowledge queries",
			[]string{
				"Tell me about Stoicism",
				"Who was Socrates?",
				"What is quantum mechanics?",
				"Tell me about Albert Einstein",
				"What is DNA?",
			},
		},
		{
			"Emotional & personal",
			[]string{
				"I had a really tough day today",
				"Thanks, that helps a lot",
			},
		},
		{
			"Deep dives",
			[]string{
				"What is evolution?",
				"Tell me about the Renaissance",
				"What is Python?",
			},
		},
	}

	totalResponses := 0
	knowledgeHits := 0
	emptyResponses := 0

	for _, conv := range conversations {
		fmt.Printf("\n--- %s ---\n", conv.section)
		for _, msg := range conv.messages {
			resp := chat(msg)
			totalResponses++

			if resp == "(no response)" {
				emptyResponses++
			}

			// Check if knowledge queries get substantive responses
			isKnowledge := strings.Contains(strings.ToLower(msg), "tell me") ||
				strings.Contains(strings.ToLower(msg), "what is") ||
				strings.Contains(strings.ToLower(msg), "who was")
			if isKnowledge && len(resp) > 100 {
				knowledgeHits++
			}

			preview := resp
			if len(preview) > 150 {
				preview = preview[:150] + "..."
			}
			fmt.Printf("  [%d] Raphael: %s\n", turn, msg)
			fmt.Printf("      Nous:    %s\n", preview)
		}
	}

	fmt.Printf("\n--- STATS ---\n")
	fmt.Printf("  Total turns: %d\n", turn)
	fmt.Printf("  Responses: %d\n", totalResponses)
	fmt.Printf("  Knowledge hits: %d\n", knowledgeHits)
	fmt.Printf("  Empty responses: %d\n", emptyResponses)
	if composer.Generative.markov != nil {
		fmt.Printf("  Markov trigrams: %d\n", composer.Generative.markov.Size())
	}
	if composer.Generative.templates != nil {
		fmt.Printf("  Learned templates: %d\n", composer.Generative.templates.Size())
	}
	if composer.Generative.embeddings != nil {
		fmt.Printf("  Embedding vectors: %d\n", composer.Generative.embeddings.Size())
	}

	if emptyResponses > 0 {
		t.Errorf("got %d empty responses", emptyResponses)
	}
}

// checkGrammarIssues scans text for known grammar problems.
func checkGrammarIssues(text string) []string {
	var issues []string
	lower := strings.ToLower(text)

	// Check for double articles (word-boundary aware)
	words := strings.Fields(lower)
	for i := 0; i < len(words)-1; i++ {
		w1, w2 := words[i], words[i+1]
		if (w1 == "the" && w2 == "the") ||
			(w1 == "a" && w2 == "a") ||
			(w1 == "a" && w2 == "an") ||
			(w1 == "an" && w2 == "a") ||
			(w1 == "an" && w2 == "an") {
			issues = append(issues, fmt.Sprintf("double article: %s %s (at word %d)", w1, w2, i))
		}
	}

	// Check for missing spaces before punctuation-separated words
	for _, bad := range []string{"beginningof", "startof", "endof"} {
		if strings.Contains(lower, bad) {
			issues = append(issues, "missing space: "+bad)
		}
	}

	// Check for pronoun errors
	for _, bad := range []string{"behind he ", "sets he ", "makes he ", "gives he "} {
		if strings.Contains(lower, bad) {
			issues = append(issues, "pronoun error: "+bad)
		}
	}

	// Check for double "which which" or "who who"
	for _, bad := range []string{"which which", "who who"} {
		if strings.Contains(lower, bad) {
			issues = append(issues, "double relative: "+bad)
		}
	}

	return issues
}
