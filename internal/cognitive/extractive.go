package cognitive

import (
	"math"
	"regexp"
	"sort"
	"strings"
	"unicode"
)

// ExtractiveQA answers questions by finding relevant sentences in stored content.
// Zero LLM calls. Microsecond latency. Pure code.
type ExtractiveQA struct {
	// IDF cache — built from all stored documents
	docCount int
	docFreq  map[string]int // word → number of documents containing it
}

// NewExtractiveQA creates a new extractive QA engine.
func NewExtractiveQA() *ExtractiveQA {
	return &ExtractiveQA{
		docFreq: make(map[string]int),
	}
}

// Fact represents a single extracted fact from content.
type Fact struct {
	Text       string  // the sentence or fact text
	Source     string  // where it came from (URL, tool, etc.)
	Topic      string  // extracted topic/entity this fact is about
	Score      float64 // importance score (0-1)
	IsDefinition bool  // "X is..." pattern
	IsList       bool  // bullet/numbered list item
	Position     int   // position in original document (0=first)
}

// FactStore holds extracted facts for querying.
type FactStore struct {
	facts    []Fact
	byTopic  map[string][]int // topic → indices into facts
	bySource map[string][]int // source → indices into facts
}

// NewFactStore creates an empty fact store.
func NewFactStore() *FactStore {
	return &FactStore{
		byTopic:  make(map[string][]int),
		bySource: make(map[string][]int),
	}
}

// Add stores a fact and indexes it.
func (fs *FactStore) Add(f Fact) {
	idx := len(fs.facts)
	fs.facts = append(fs.facts, f)

	if f.Topic != "" {
		key := strings.ToLower(f.Topic)
		fs.byTopic[key] = append(fs.byTopic[key], idx)
	}
	if f.Source != "" {
		fs.bySource[f.Source] = append(fs.bySource[f.Source], idx)
	}
}

// Size returns number of stored facts.
func (fs *FactStore) Size() int {
	return len(fs.facts)
}

// FactsAbout returns facts related to a topic.
func (fs *FactStore) FactsAbout(topic string) []Fact {
	key := strings.ToLower(topic)
	var results []Fact

	// Direct topic match
	if indices, ok := fs.byTopic[key]; ok {
		for _, i := range indices {
			results = append(results, fs.facts[i])
		}
	}

	// Partial match — topic contains query or vice versa
	for t, indices := range fs.byTopic {
		if t == key {
			continue
		}
		if strings.Contains(t, key) || strings.Contains(key, t) {
			for _, i := range indices {
				results = append(results, fs.facts[i])
			}
		}
	}

	return results
}

// FactsFromSource returns all facts from a given source.
func (fs *FactStore) FactsFromSource(source string) []Fact {
	var results []Fact
	if indices, ok := fs.bySource[source]; ok {
		for _, i := range indices {
			results = append(results, fs.facts[i])
		}
	}
	return results
}

// AllFacts returns all stored facts.
func (fs *FactStore) AllFacts() []Fact {
	return fs.facts
}

// -----------------------------------------------------------------------
// Fact Extraction — pull structured facts from raw text
// -----------------------------------------------------------------------

// ExtractFacts splits content into scored, structured facts.
func ExtractFacts(content, source, topic string) []Fact {
	sentences := splitSentences(content)
	if len(sentences) == 0 {
		return nil
	}

	var facts []Fact
	topicLower := strings.ToLower(topic)

	for i, sent := range sentences {
		sent = strings.TrimSpace(sent)
		if len(sent) < 10 || len(sent) > 500 {
			continue
		}

		// Skip navigation/boilerplate
		if isBoilerplate(sent) {
			continue
		}

		f := Fact{
			Text:     sent,
			Source:   source,
			Topic:    topic,
			Position: i,
		}

		// Score the sentence
		f.Score = scoreSentence(sent, i, len(sentences), topicLower)
		f.IsDefinition = isDefinition(sent, topicLower)
		f.IsList = isListItem(sent)

		// Boost definitions and early sentences
		if f.IsDefinition {
			f.Score += 0.3
		}
		if f.IsList {
			f.Score += 0.1
		}

		// Cap at 1.0
		if f.Score > 1.0 {
			f.Score = 1.0
		}

		// Only keep sentences with reasonable score
		if f.Score >= 0.15 {
			facts = append(facts, f)
		}
	}

	// Sort by score descending
	sort.Slice(facts, func(i, j int) bool {
		return facts[i].Score > facts[j].Score
	})

	// Keep top facts (avoid storing too much)
	if len(facts) > 50 {
		facts = facts[:50]
	}

	return facts
}

// -----------------------------------------------------------------------
// Extractive Question Answering
// -----------------------------------------------------------------------

// ScoredFact is a fact with a query-relevance score.
type ScoredFact struct {
	Fact
	Relevance float64
}

// Answer finds the most relevant facts for a question.
func (eqa *ExtractiveQA) Answer(question string, facts []Fact, maxResults int) []ScoredFact {
	if len(facts) == 0 {
		return nil
	}

	queryWords := tokenize(question)
	querySet := make(map[string]bool)
	for _, w := range queryWords {
		querySet[w] = true
	}

	// Build IDF from these facts
	docFreq := make(map[string]int)
	for _, f := range facts {
		seen := make(map[string]bool)
		for _, w := range tokenize(f.Text) {
			if !seen[w] {
				docFreq[w]++
				seen[w] = true
			}
		}
	}
	totalDocs := len(facts)

	var scored []ScoredFact
	for _, f := range facts {
		rel := scoreFact(f, queryWords, querySet, docFreq, totalDocs)
		if rel > 0.05 {
			scored = append(scored, ScoredFact{Fact: f, Relevance: rel})
		}
	}

	// Sort by relevance
	sort.Slice(scored, func(i, j int) bool {
		return scored[i].Relevance > scored[j].Relevance
	})

	if len(scored) > maxResults {
		scored = scored[:maxResults]
	}

	return scored
}

// scoreFact computes relevance of a fact to a query.
func scoreFact(f Fact, queryWords []string, querySet map[string]bool, docFreq map[string]int, totalDocs int) float64 {
	factWords := tokenize(f.Text)
	factSet := make(map[string]bool)
	for _, w := range factWords {
		factSet[w] = true
	}

	// 1. Keyword overlap (Jaccard-like)
	overlap := 0
	for _, qw := range queryWords {
		if factSet[qw] {
			overlap++
		}
	}
	if len(queryWords) == 0 {
		return 0
	}
	jaccardScore := float64(overlap) / float64(len(queryWords))

	// 2. TF-IDF weighted overlap
	var tfidfScore float64
	for _, qw := range queryWords {
		if !factSet[qw] {
			continue
		}
		// TF: frequency in this fact
		tf := 0
		for _, fw := range factWords {
			if fw == qw {
				tf++
			}
		}
		tfNorm := float64(tf) / float64(len(factWords))

		// IDF: inverse document frequency
		df := docFreq[qw]
		if df == 0 {
			df = 1
		}
		idf := math.Log(float64(totalDocs+1) / float64(df))
		tfidfScore += tfNorm * idf
	}

	// 3. Position bonus (earlier content is usually more important)
	posBonus := 0.0
	if f.Position < 5 {
		posBonus = 0.15 * (1.0 - float64(f.Position)/5.0)
	}

	// 4. Definition bonus
	defBonus := 0.0
	if f.IsDefinition {
		defBonus = 0.2
	}

	// 5. Base importance
	importanceBonus := f.Score * 0.1

	// Combine
	relevance := jaccardScore*0.4 + tfidfScore*0.3 + posBonus + defBonus + importanceBonus
	return relevance
}

// -----------------------------------------------------------------------
// Response Composition — turn extracted facts into readable responses
// -----------------------------------------------------------------------

// QuestionType classifies what kind of answer is expected.
type QuestionType int

const (
	QGeneral QuestionType = iota
	QWho                  // who founded, who created
	QWhat                 // what is, what does
	QWhen                 // when was, when did
	QWhere                // where is, where was
	QWhy                  // why does, why is
	QHow                  // how does, how to
	QList                 // what products, what features, list
)

// classifyQuestion determines what kind of answer the user expects.
func classifyQuestion(q string) QuestionType {
	lower := strings.ToLower(q)
	words := strings.Fields(lower)
	if len(words) == 0 {
		return QGeneral
	}

	// Check for list-expecting questions
	for _, marker := range []string{"what products", "what features", "what tools",
		"list ", "which ones", "what are the", "what do they offer",
		"what services", "what options"} {
		if strings.Contains(lower, marker) {
			return QList
		}
	}

	switch words[0] {
	case "who":
		return QWho
	case "what", "what's":
		return QWhat
	case "when":
		return QWhen
	case "where":
		return QWhere
	case "why":
		return QWhy
	case "how":
		return QHow
	}

	// Check interior question words
	for _, w := range words {
		switch w {
		case "who":
			return QWho
		case "when":
			return QWhen
		case "where":
			return QWhere
		}
	}

	return QGeneral
}

// responseConnectors provides natural transitions between facts.
var responseConnectors = []string{
	"", // first fact needs no connector
	"Additionally, ",
	"Also, ",
	"Furthermore, ",
	"Moreover, ",
}

// ComposeResponse builds a natural-sounding response from scored facts.
// It is question-type-aware: "who" questions lead with people,
// "what" questions lead with definitions, "list" questions use bullets.
func ComposeResponse(question string, facts []ScoredFact, source string) string {
	if len(facts) == 0 {
		return ""
	}

	qtype := classifyQuestion(question)

	// For list questions, use bullet format
	if qtype == QList {
		return composeListResponse(facts, source)
	}

	// For who/when/where, try to find the most specific answer fact
	if qtype == QWho || qtype == QWhen || qtype == QWhere {
		return composeSpecificResponse(facts, qtype, source)
	}

	// General / what / why / how — lead with most relevant, add context
	return composeNarrativeResponse(facts, source)
}

// composeListResponse formats facts as a bullet list.
func composeListResponse(facts []ScoredFact, source string) string {
	var b strings.Builder
	count := 0
	for _, f := range facts {
		if count >= 6 {
			break
		}
		b.WriteString("- ")
		b.WriteString(ensurePeriod(f.Text))
		b.WriteString("\n")
		count++
	}
	return strings.TrimRight(b.String(), "\n")
}

// composeSpecificResponse answers who/when/where questions directly.
func composeSpecificResponse(facts []ScoredFact, qtype QuestionType, source string) string {
	if len(facts) == 0 {
		return ""
	}

	var b strings.Builder

	// Lead with the top fact (most relevant to the specific question)
	b.WriteString(ensurePeriod(facts[0].Text))

	// Add 1-2 supporting context facts
	for i := 1; i < len(facts) && i <= 2; i++ {
		if facts[i].Relevance < facts[0].Relevance*0.5 {
			break // don't add weakly relevant facts
		}
		b.WriteString(" ")
		b.WriteString(ensurePeriod(facts[i].Text))
	}

	return b.String()
}

// composeNarrativeResponse builds a flowing paragraph answer.
func composeNarrativeResponse(facts []ScoredFact, source string) string {
	if len(facts) == 0 {
		return ""
	}

	var b strings.Builder
	used := 0

	// Lead with definition if the top fact is one
	if facts[0].IsDefinition {
		b.WriteString(ensurePeriod(facts[0].Text))
		facts = facts[1:]
		used++
	}

	// Add supporting facts with natural connectors
	for i, f := range facts {
		if used >= 4 {
			break
		}
		if b.Len() > 0 {
			b.WriteString(" ")
		}

		// Use connectors for facts after the first
		if used > 0 && i < len(responseConnectors) {
			connector := responseConnectors[i]
			if connector != "" {
				// Only use connector if fact doesn't start with the topic name
				firstWord := strings.ToLower(strings.Fields(f.Text)[0])
				if firstWord != "the" && firstWord != "it" && firstWord != "they" {
					b.WriteString(connector)
					// Lowercase the first letter of the fact after a connector
					text := f.Text
					if len(text) > 0 {
						text = strings.ToLower(text[:1]) + text[1:]
					}
					b.WriteString(ensurePeriod(text))
					used++
					continue
				}
			}
		}

		b.WriteString(ensurePeriod(f.Text))
		used++
	}

	return b.String()
}

// ensurePeriod adds a period if the sentence doesn't end with punctuation.
func ensurePeriod(s string) string {
	s = strings.TrimSpace(s)
	if s == "" {
		return s
	}
	last := s[len(s)-1]
	if last != '.' && last != '!' && last != '?' {
		return s + "."
	}
	return s
}

// ComposeTopicSummary creates a summary from all facts about a topic.
func ComposeTopicSummary(topic string, facts []Fact) string {
	if len(facts) == 0 {
		return ""
	}

	// Sort by score
	sorted := make([]Fact, len(facts))
	copy(sorted, facts)
	sort.Slice(sorted, func(i, j int) bool {
		return sorted[i].Score > sorted[j].Score
	})

	var b strings.Builder

	// Lead with definition if available — no "Here's what I know" preamble
	defIdx := -1
	for i, f := range sorted {
		if f.IsDefinition {
			b.WriteString(ensurePeriod(f.Text))
			b.WriteString("\n\n")
			defIdx = i
			break
		}
	}

	// Add top facts as bullets
	count := 0
	for i, f := range sorted {
		if count >= 5 {
			break
		}
		if i == defIdx {
			continue
		}
		b.WriteString("- ")
		b.WriteString(ensurePeriod(f.Text))
		b.WriteString("\n")
		count++
	}

	// If there's a definition, total shown = count + 1
	shown := count
	if defIdx >= 0 {
		shown++
	}
	remaining := len(facts) - shown
	if remaining > 1 {
		b.WriteString("\n")
		b.WriteString(intToWord(remaining))
		b.WriteString(" more facts available — ask me anything about this!")
	}

	return strings.TrimRight(b.String(), "\n")
}

func intToWord(n int) string {
	if n <= 0 {
		return "No"
	}
	words := []string{"Zero", "One", "Two", "Three", "Four", "Five", "Six",
		"Seven", "Eight", "Nine", "Ten", "Eleven", "Twelve"}
	if n < len(words) {
		return words[n]
	}
	return "Many"
}

// -----------------------------------------------------------------------
// Text processing helpers
// -----------------------------------------------------------------------

var sentenceEndRe = regexp.MustCompile(`[.!?]+[\s]+|[.!?]+$`)
var sentenceBreakRe = regexp.MustCompile(`\n\n+|\n[-*•]\s+|\n\d+[.)]\s+`)

// splitSentences breaks text into sentences.
func splitSentences(text string) []string {
	// First split on paragraph boundaries and list items
	blocks := sentenceBreakRe.Split(text, -1)

	var sentences []string
	for _, block := range blocks {
		block = strings.TrimSpace(block)
		if block == "" {
			continue
		}

		// Split on sentence endings
		parts := sentenceEndRe.Split(block, -1)
		for _, p := range parts {
			p = strings.TrimSpace(p)
			if p != "" {
				sentences = append(sentences, p)
			}
		}
	}

	return sentences
}

// tokenize splits text into lowercase words, removing stop words.
func tokenize(text string) []string {
	lower := strings.ToLower(text)
	words := strings.FieldsFunc(lower, func(r rune) bool {
		return !unicode.IsLetter(r) && !unicode.IsDigit(r)
	})

	var filtered []string
	for _, w := range words {
		if len(w) < 2 {
			continue
		}
		if !isExtractiveStop(w) {
			filtered = append(filtered, w)
		}
	}
	return filtered
}

// isExtractiveStopWord is a broader stop word list for extractive QA tokenization.
// The package also has isStopWord in predictive.go — this one is used specifically
// for the extractive QA pipeline and covers more common words.
var extractiveStopWords = map[string]bool{
	"the": true, "a": true, "an": true, "is": true, "are": true,
	"was": true, "were": true, "be": true, "been": true, "being": true,
	"have": true, "has": true, "had": true, "do": true, "does": true,
	"did": true, "will": true, "would": true, "could": true, "should": true,
	"may": true, "might": true, "can": true, "shall": true, "must": true,
	"of": true, "in": true, "to": true, "for": true, "with": true,
	"on": true, "at": true, "by": true, "from": true, "up": true,
	"about": true, "into": true, "through": true, "during": true, "before": true,
	"after": true, "above": true, "below": true, "between": true,
	"and": true, "but": true, "or": true, "not": true, "no": true,
	"so": true, "if": true, "then": true, "than": true, "too": true,
	"very": true, "just": true, "also": true, "that": true, "this": true,
	"these": true, "those": true, "it": true, "its": true, "what": true,
	"which": true, "who": true, "whom": true, "where": true, "when": true,
	"how": true, "why": true, "me": true, "my": true, "your": true,
	"you": true, "he": true, "she": true, "we": true, "they": true,
	"them": true, "their": true, "our": true, "his": true, "her": true,
	"i": true, "am": true,
}

func isExtractiveStop(w string) bool {
	return extractiveStopWords[w]
}

// scoreSentence scores a sentence's importance (0-1).
func scoreSentence(sent string, position, total int, topicLower string) float64 {
	lower := strings.ToLower(sent)
	score := 0.2 // base

	// Position: first sentences are more important
	if position < 3 {
		score += 0.3 * (1.0 - float64(position)/3.0)
	}

	// Contains topic words
	topicWords := strings.Fields(topicLower)
	matchCount := 0
	for _, tw := range topicWords {
		if strings.Contains(lower, tw) {
			matchCount++
		}
	}
	if len(topicWords) > 0 {
		score += 0.3 * float64(matchCount) / float64(len(topicWords))
	}

	// Contains numbers/data (factual content)
	if hasNumbers(sent) {
		score += 0.1
	}

	// Contains proper nouns (capitalized words mid-sentence)
	if hasProperNouns(sent) {
		score += 0.05
	}

	return score
}

// isDefinition detects "X is..." or "X refers to..." patterns.
func isDefinition(sent, topicLower string) bool {
	lower := strings.ToLower(sent)
	for _, pattern := range []string{" is a ", " is an ", " is the ", " are a ", " are the ",
		" refers to ", " means ", " defined as ", " known as "} {
		if strings.Contains(lower, pattern) {
			// Check if it's about our topic
			if topicLower == "" || strings.Contains(lower, topicLower) {
				return true
			}
		}
	}
	return false
}

// isListItem detects bullet/numbered list items.
func isListItem(sent string) bool {
	trimmed := strings.TrimSpace(sent)
	if len(trimmed) < 3 {
		return false
	}
	if trimmed[0] == '-' || trimmed[0] == '*' {
		return true
	}
	if trimmed[0] >= '1' && trimmed[0] <= '9' && (trimmed[1] == '.' || trimmed[1] == ')') {
		return true
	}
	return false
}

// isBoilerplate detects navigation/UI/cookie text.
var boilerplatePatterns = []string{
	"cookie", "privacy policy", "terms of service", "subscribe",
	"sign up", "log in", "sign in", "newsletter", "copyright",
	"all rights reserved", "click here", "read more", "loading",
	"javascript", "skip to content", "menu", "navigation",
	"accept cookies", "we use cookies",
}

func isBoilerplate(sent string) bool {
	lower := strings.ToLower(sent)
	for _, p := range boilerplatePatterns {
		if strings.Contains(lower, p) {
			return true
		}
	}
	// Very short sentences that are likely UI elements
	if len(sent) < 20 && !strings.Contains(sent, " ") {
		return true
	}
	return false
}

func hasNumbers(s string) bool {
	for _, r := range s {
		if unicode.IsDigit(r) {
			return true
		}
	}
	return false
}

func hasProperNouns(s string) bool {
	words := strings.Fields(s)
	for i, w := range words {
		if i == 0 {
			continue // skip first word (always capitalized)
		}
		if len(w) > 1 && unicode.IsUpper(rune(w[0])) {
			return true
		}
	}
	return false
}
