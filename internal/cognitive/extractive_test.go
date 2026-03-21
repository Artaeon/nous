package cognitive

import (
	"strings"
	"testing"
)

func TestSplitSentences(t *testing.T) {
	text := "Nous is a local AI. It runs on your machine. It has 68 tools."
	sentences := splitSentences(text)
	if len(sentences) != 3 {
		t.Errorf("expected 3 sentences, got %d: %v", len(sentences), sentences)
	}
}

func TestSplitSentencesParagraphs(t *testing.T) {
	text := "First paragraph about AI.\n\nSecond paragraph about tools. They work great."
	sentences := splitSentences(text)
	if len(sentences) < 3 {
		t.Errorf("expected at least 3 sentences, got %d: %v", len(sentences), sentences)
	}
}

func TestExtractFacts(t *testing.T) {
	content := `Stoicera is a philosophy company based in Vienna.
They create tools for modern stoics.
The company was founded in 2023 by Raphael Lugmayr.
Stoicera offers journals, meditation guides, and daily practices.
Their mission is to make ancient wisdom accessible to everyone.`

	facts := ExtractFacts(content, "https://stoicera.com", "stoicera")
	if len(facts) == 0 {
		t.Fatal("expected facts to be extracted")
	}

	// First fact should be the definition
	hasDefinition := false
	for _, f := range facts {
		if f.IsDefinition && strings.Contains(f.Text, "philosophy company") {
			hasDefinition = true
			break
		}
	}
	if !hasDefinition {
		t.Error("expected a definition fact about Stoicera")
	}

	// Should have facts about the topic
	topicMentions := 0
	for _, f := range facts {
		if strings.Contains(strings.ToLower(f.Text), "stoicera") {
			topicMentions++
		}
	}
	if topicMentions < 2 {
		t.Errorf("expected multiple facts mentioning the topic, got %d", topicMentions)
	}
}

func TestExtractFactsFiltersBoilerplate(t *testing.T) {
	content := `Welcome to our site. Accept cookies to continue.
Nous is an AI assistant that runs locally.
Subscribe to our newsletter for updates.
It processes queries in microseconds.`

	facts := ExtractFacts(content, "test", "nous")
	for _, f := range facts {
		if strings.Contains(strings.ToLower(f.Text), "cookie") {
			t.Error("boilerplate should be filtered out")
		}
		if strings.Contains(strings.ToLower(f.Text), "newsletter") {
			t.Error("boilerplate should be filtered out")
		}
	}
}

func TestExtractiveQAAnswer(t *testing.T) {
	eqa := NewExtractiveQA()

	facts := []Fact{
		{Text: "Stoicera is a philosophy company based in Vienna", Score: 0.9, IsDefinition: true, Position: 0},
		{Text: "The company was founded in 2023 by Raphael Lugmayr", Score: 0.7, Position: 2},
		{Text: "Stoicera offers journals and meditation guides", Score: 0.6, Position: 3},
		{Text: "Their website uses a dark theme with green accents", Score: 0.3, Position: 10},
		{Text: "The weather in Vienna is cold in winter", Score: 0.2, Position: 15},
	}

	// Question about what Stoicera is
	results := eqa.Answer("what is stoicera", facts, 3)
	if len(results) == 0 {
		t.Fatal("expected answers")
	}
	if !strings.Contains(results[0].Text, "philosophy company") {
		t.Errorf("top answer should be the definition, got %q", results[0].Text)
	}

	// Question about founding
	results = eqa.Answer("who founded stoicera", facts, 3)
	if len(results) == 0 {
		t.Fatal("expected answers for founding question")
	}
	foundFounding := false
	for _, r := range results {
		if strings.Contains(r.Text, "founded") || strings.Contains(r.Text, "Raphael") {
			foundFounding = true
			break
		}
	}
	if !foundFounding {
		t.Error("should find the founding fact")
	}
}

func TestExtractiveQAIrrelevant(t *testing.T) {
	eqa := NewExtractiveQA()

	facts := []Fact{
		{Text: "The weather in Vienna is cold in winter", Score: 0.5, Position: 0},
		{Text: "Apples are a popular fruit worldwide", Score: 0.3, Position: 1},
	}

	results := eqa.Answer("what is quantum computing", facts, 3)
	// Should return no results or very low relevance
	for _, r := range results {
		if r.Relevance > 0.3 {
			t.Errorf("irrelevant facts should have low relevance, got %.2f for %q", r.Relevance, r.Text)
		}
	}
}

func TestComposeResponse(t *testing.T) {
	facts := []ScoredFact{
		{Fact: Fact{Text: "Stoicera is a philosophy company based in Vienna", IsDefinition: true}, Relevance: 0.9},
		{Fact: Fact{Text: "They create tools for modern stoics"}, Relevance: 0.7},
	}

	response := ComposeResponse("what is stoicera", facts, "https://stoicera.com")
	if response == "" {
		t.Fatal("expected non-empty response")
	}
	if !strings.Contains(response, "philosophy company") {
		t.Error("response should contain the definition")
	}
	if !strings.Contains(response, "tools") {
		t.Error("response should include relevant facts")
	}
}

func TestComposeTopicSummary(t *testing.T) {
	facts := []Fact{
		{Text: "Go is a programming language created by Google", Score: 0.9, IsDefinition: true},
		{Text: "Go compiles to native machine code", Score: 0.7},
		{Text: "Go has built-in concurrency with goroutines", Score: 0.8},
		{Text: "The Go standard library is comprehensive", Score: 0.6},
	}

	summary := ComposeTopicSummary("Go", facts)
	if summary == "" {
		t.Fatal("expected non-empty summary")
	}
	if !strings.Contains(summary, "programming language") {
		t.Error("summary should lead with definition")
	}
	if !strings.Contains(summary, "goroutines") {
		t.Error("summary should include key facts")
	}
}

func TestFactStore(t *testing.T) {
	store := NewFactStore()
	store.Add(Fact{Text: "Go is fast", Topic: "golang", Source: "web"})
	store.Add(Fact{Text: "Rust is safe", Topic: "rust", Source: "web"})
	store.Add(Fact{Text: "Go has goroutines", Topic: "golang", Source: "docs"})

	if store.Size() != 3 {
		t.Errorf("expected 3 facts, got %d", store.Size())
	}

	goFacts := store.FactsAbout("golang")
	if len(goFacts) != 2 {
		t.Errorf("expected 2 Go facts, got %d", len(goFacts))
	}

	webFacts := store.FactsFromSource("web")
	if len(webFacts) != 2 {
		t.Errorf("expected 2 web facts, got %d", len(webFacts))
	}
}

func TestTokenize(t *testing.T) {
	tokens := tokenize("What is the capital of France?")
	// Should remove stop words: what, is, the, of
	for _, tok := range tokens {
		if isStopWord(tok) {
			t.Errorf("stop word %q should be filtered", tok)
		}
	}
	found := false
	for _, tok := range tokens {
		if tok == "france" {
			found = true
		}
	}
	if !found {
		t.Error("should contain 'france'")
	}
}

func TestIsDefinition(t *testing.T) {
	tests := []struct {
		sent  string
		topic string
		want  bool
	}{
		{"Nous is a local AI assistant", "nous", true},
		{"The weather is cold today", "nous", false},
		{"Go is a programming language", "go", true},
		{"I like programming in Go", "go", false},
	}

	for _, tt := range tests {
		got := isDefinition(tt.sent, tt.topic)
		if got != tt.want {
			t.Errorf("isDefinition(%q, %q) = %v, want %v", tt.sent, tt.topic, got, tt.want)
		}
	}
}

func TestIsBoilerplate(t *testing.T) {
	tests := []struct {
		sent string
		want bool
	}{
		{"Accept cookies to continue browsing", true},
		{"Subscribe to our newsletter", true},
		{"Nous is a powerful AI assistant", false},
		{"The company was founded in 2023", false},
	}

	for _, tt := range tests {
		got := isBoilerplate(tt.sent)
		if got != tt.want {
			t.Errorf("isBoilerplate(%q) = %v, want %v", tt.sent, got, tt.want)
		}
	}
}
