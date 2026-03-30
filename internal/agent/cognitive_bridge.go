package agent

import (
	"fmt"
	"strings"
	"unicode"

	"github.com/artaeon/nous/internal/cognitive"
)

// CognitiveBridge connects the autonomous agent to Nous's cognitive systems.
// It gives the agent the ability to THINK — summarize, reason, generate
// documents, and compose natural language — not just run tools.
type CognitiveBridge struct {
	Summarizer *cognitive.Summarizer
	Thinker    *cognitive.ThinkingEngine
	DeepReason *cognitive.DeepReasoner
	DocGen     *cognitive.DocumentGenerator
	Composer   *cognitive.Composer
	NLG        *cognitive.NLGEngine
	Graph      *cognitive.CognitiveGraph
}

// NewCognitiveBridge creates a bridge from an ActionRouter's cognitive systems.
// Returns nil if the ActionRouter has no cognitive systems wired.
func NewCognitiveBridge(ar *cognitive.ActionRouter) *CognitiveBridge {
	if ar == nil {
		return nil
	}
	return &CognitiveBridge{
		Summarizer: ar.Summarizer,
		Thinker:    ar.Thinker,
		DeepReason: ar.DeepReason,
		DocGen:     ar.DocGen,
		Composer:   ar.Composer,
		NLG:        ar.NLG,
		Graph:      ar.CogGraph,
	}
}

// Summarize condenses text to the given number of sentences.
// Falls back to simple truncation if the summarizer is not available.
func (cb *CognitiveBridge) Summarize(text string, maxSentences int) string {
	if cb == nil || cb.Summarizer == nil {
		return truncateBridge(text, 500)
	}
	return cb.Summarizer.Summarize(text, maxSentences)
}

// SummarizeToLength condenses text to approximately maxWords words.
func (cb *CognitiveBridge) SummarizeToLength(text string, maxWords int) string {
	if cb == nil || cb.Summarizer == nil {
		return truncateBridge(text, maxWords*6)
	}
	return cb.Summarizer.SummarizeToLength(text, maxWords)
}

// Think uses the ThinkingEngine to generate a thoughtful response to a query.
// Returns empty string if the engine is not available or can't handle the query.
func (cb *CognitiveBridge) Think(query string) string {
	if cb == nil || cb.Thinker == nil {
		return ""
	}
	result := cb.Thinker.Think(query, nil)
	if result == nil {
		return ""
	}
	return result.Text
}

// ThinkAbout is like Think but filters the result to only include sentences
// relevant to the given topic. This prevents knowledge graph contamination.
func (cb *CognitiveBridge) ThinkAbout(topic, query string) string {
	raw := cb.Think(query)
	if raw == "" {
		return ""
	}
	return filterByTopic(raw, topic)
}

// Reason performs deep multi-step reasoning on a question.
func (cb *CognitiveBridge) Reason(question string) (answer string, trace string) {
	if cb == nil || cb.DeepReason == nil {
		return "", ""
	}
	result := cb.DeepReason.Reason(question)
	if result == nil {
		return "", ""
	}
	return result.FinalAnswer, result.Trace
}

// GenerateDocument creates a structured multi-section document about a topic.
func (cb *CognitiveBridge) GenerateDocument(topic, style string) string {
	if cb == nil || cb.DocGen == nil {
		return ""
	}
	doc := cb.DocGen.Generate(topic, style)
	if doc == nil || len(doc.Sections) == 0 {
		return ""
	}
	return formatDocument(doc)
}

// Compose generates a natural language response for a query.
func (cb *CognitiveBridge) Compose(query string) string {
	if cb == nil || cb.Composer == nil {
		return ""
	}
	resp := cb.Composer.Compose(query, cognitive.RespFactual, nil)
	if resp == nil || resp.Text == "" {
		return ""
	}
	return resp.Text
}

// -----------------------------------------------------------------------
// SmartSynthesize — the real synthesis pipeline
// -----------------------------------------------------------------------

// SmartSynthesize merges web search results and knowledge graph content
// into a coherent, topic-filtered synthesis. This is the core function
// that transforms the agent from a tool runner into a thinking worker.
//
// Pipeline:
//  1. Extract key sentences from each web result
//  2. Extract key sentences from knowledge graph
//  3. Filter out sentences unrelated to the topic
//  4. Remove near-duplicate sentences (Jaccard > 0.5)
//  5. Order semantically: definition → history → properties → applications → outlook
//  6. Join into coherent paragraphs with connectors
func (cb *CognitiveBridge) SmartSynthesize(topic string, webResults []string, graphFacts []string) string {
	if cb == nil {
		return joinNonEmpty(append(webResults, graphFacts...))
	}

	// 1. Extract key sentences from each source
	var allSentences []string

	for _, result := range webResults {
		if result == "" {
			continue
		}
		// Extract top 3 sentences per web result
		summary := cb.Summarize(result, 3)
		for _, s := range splitSentences(summary) {
			s = strings.TrimSpace(s)
			if len(s) > 20 { // skip tiny fragments
				allSentences = append(allSentences, s)
			}
		}
	}

	for _, fact := range graphFacts {
		if fact == "" {
			continue
		}
		for _, s := range splitSentences(fact) {
			s = strings.TrimSpace(s)
			if len(s) > 20 {
				allSentences = append(allSentences, s)
			}
		}
	}

	if len(allSentences) == 0 {
		return "No relevant information found for: " + topic
	}

	// 2. Topic-relevance filter: drop sentences that share no content words with the topic
	topicWords := contentWordSet(topic)
	var relevant []string
	for _, s := range allSentences {
		if isRelevantToTopic(s, topicWords) {
			relevant = append(relevant, s)
		}
	}

	// If filtering killed everything, be less strict — keep sentences with ANY overlap
	if len(relevant) == 0 {
		relevant = allSentences
	}

	// 3. Dedup: remove near-duplicate sentences (Jaccard > 0.5)
	deduped := deduplicateSentences(relevant, 0.5)

	// 4. Semantic ordering: definition → history → properties → applications → outlook
	ordered := semanticOrder(deduped)

	// 5. Join into paragraphs with connectors
	return joinWithConnectors(topic, ordered)
}

// SynthesizeResults takes raw tool outputs and produces a coherent summary.
// This replaces the old cascade approach with SmartSynthesize.
func (cb *CognitiveBridge) SynthesizeResults(goal string, results map[string]string) string {
	if cb == nil {
		return concatenateResults(results)
	}

	topic := extractSynthesisTopic(goal)

	// Separate web results from other outputs
	var webResults []string
	for _, v := range results {
		if v == "" {
			continue
		}
		webResults = append(webResults, v)
	}

	// Get knowledge graph content if available
	var graphFacts []string
	if cb.Graph != nil {
		graphFacts = cb.queryGraphFacts(topic)
	}

	result := cb.SmartSynthesize(topic, webResults, graphFacts)
	if result == "" {
		return concatenateResults(results)
	}
	return result
}

// WriteReport generates a full document from accumulated results.
func (cb *CognitiveBridge) WriteReport(topic, style string, results map[string]string) string {
	doc := cb.GenerateDocument(topic, style)
	if doc != "" {
		return doc
	}
	return cb.SynthesizeResults("write a "+style+" about "+topic, results)
}

// queryGraphFacts retrieves relevant knowledge paragraphs from the cognitive graph.
func (cb *CognitiveBridge) queryGraphFacts(topic string) []string {
	if cb.Graph == nil {
		return nil
	}

	// Use the graph's FindParagraph method if it has one, or compose from edges
	// The graph stores facts as edges; we want prose paragraphs.
	// Try Compose which queries the graph internally.
	composed := cb.Compose("What is " + topic + "?")
	if composed != "" && len(composed) > 30 {
		return []string{composed}
	}
	return nil
}

// -----------------------------------------------------------------------
// Sentence processing helpers
// -----------------------------------------------------------------------

// splitSentences splits text on sentence boundaries (". ", "! ", "? ").
func splitSentences(text string) []string {
	text = strings.TrimSpace(text)
	if text == "" {
		return nil
	}

	var sentences []string
	remaining := text

	for len(remaining) > 0 {
		bestIdx := -1
		for _, punct := range []string{". ", "! ", "? "} {
			idx := strings.Index(remaining, punct)
			if idx >= 0 && (bestIdx < 0 || idx < bestIdx) {
				bestIdx = idx
			}
		}

		if bestIdx < 0 {
			s := strings.TrimSpace(remaining)
			if s != "" {
				sentences = append(sentences, s)
			}
			break
		}

		s := strings.TrimSpace(remaining[:bestIdx+1])
		if s != "" {
			sentences = append(sentences, s)
		}
		remaining = remaining[bestIdx+2:]
	}

	return sentences
}

// contentWordSet returns a set of lowercased content words (no stopwords).
func contentWordSet(text string) map[string]bool {
	words := strings.Fields(text)
	set := make(map[string]bool)
	for _, w := range words {
		w = strings.Trim(w, ".,;:!?\"'()[]{}")
		lower := strings.ToLower(w)
		if len(lower) <= 2 || synthStopWords[lower] {
			continue
		}
		set[lower] = true
	}
	return set
}

// isRelevantToTopic checks if a sentence shares at least one content word with the topic.
func isRelevantToTopic(sentence string, topicWords map[string]bool) bool {
	if len(topicWords) == 0 {
		return true // no topic filter
	}

	words := strings.Fields(strings.ToLower(sentence))
	for _, w := range words {
		w = strings.Trim(w, ".,;:!?\"'()[]{}")
		if len(w) <= 2 {
			continue
		}
		if topicWords[w] {
			return true
		}
	}
	return false
}

// jaccardSimilarity computes word-level Jaccard similarity between two sentences.
func jaccardSimilarity(a, b string) float64 {
	setA := contentWordSet(a)
	setB := contentWordSet(b)

	if len(setA) == 0 || len(setB) == 0 {
		return 0.0
	}

	intersection := 0
	for w := range setA {
		if setB[w] {
			intersection++
		}
	}

	union := len(setA) + len(setB) - intersection
	if union == 0 {
		return 0.0
	}
	return float64(intersection) / float64(union)
}

// deduplicateSentences removes near-duplicate sentences using Jaccard similarity.
// Keeps the first (longest) version of each cluster.
func deduplicateSentences(sentences []string, threshold float64) []string {
	if len(sentences) <= 1 {
		return sentences
	}

	var result []string
	for _, candidate := range sentences {
		duplicate := false
		for _, kept := range result {
			if jaccardSimilarity(candidate, kept) > threshold {
				duplicate = true
				break
			}
		}
		if !duplicate {
			result = append(result, candidate)
		}
	}
	return result
}

// semanticOrder sorts sentences by semantic role: definition → history → properties → applications → outlook.
func semanticOrder(sentences []string) []string {
	type scored struct {
		text  string
		order int
	}

	var items []scored
	for _, s := range sentences {
		items = append(items, scored{s, classifySentence(s)})
	}

	// Stable sort by category order
	for i := 1; i < len(items); i++ {
		for j := i; j > 0 && items[j].order < items[j-1].order; j-- {
			items[j], items[j-1] = items[j-1], items[j]
		}
	}

	result := make([]string, len(items))
	for i, item := range items {
		result[i] = item.text
	}
	return result
}

// classifySentence returns a category order (0 = definition, 1 = history, etc.)
func classifySentence(s string) int {
	lower := strings.ToLower(s)

	// Definition signals (order 0)
	defSignals := []string{" is a ", " is an ", " are a ", " refers to ", " defined as ", " means "}
	for _, sig := range defSignals {
		if strings.Contains(lower, sig) {
			return 0
		}
	}

	// History signals (order 1)
	histSignals := []string{"founded", "created", "invented", "established", "originated", "history", "began", "started", "first", "1950", "1960", "1970", "1980", "1990", "2000", "2010", "2020"}
	for _, sig := range histSignals {
		if strings.Contains(lower, sig) {
			return 1
		}
	}

	// Property/feature signals (order 2)
	propSignals := []string{" has ", " have ", " includes ", " features ", " consists ", " contains ", " comprises ", "key ", "main ", "three ", "types "}
	for _, sig := range propSignals {
		if strings.Contains(lower, sig) {
			return 2
		}
	}

	// Application signals (order 3)
	appSignals := []string{"used for", "used in", "application", "enables", "allows", "helps", "transforms", "revolutioniz", "impact"}
	for _, sig := range appSignals {
		if strings.Contains(lower, sig) {
			return 3
		}
	}

	// Outlook/trend signals (order 4)
	trendSignals := []string{"future", "trend", "growing", "expected", "projected", "market", "billion", "trillion", "will ", "2025", "2026", "2027", "2028", "2030"}
	for _, sig := range trendSignals {
		if strings.Contains(lower, sig) {
			return 4
		}
	}

	return 3 // default: middle
}

// joinWithConnectors assembles ordered sentences into coherent paragraphs.
func joinWithConnectors(topic string, sentences []string) string {
	if len(sentences) == 0 {
		return ""
	}
	if len(sentences) == 1 {
		return sentences[0]
	}

	var b strings.Builder
	lastCategory := -1

	for i, s := range sentences {
		category := classifySentence(s)

		if i == 0 {
			b.WriteString(s)
			lastCategory = category
			continue
		}

		// New paragraph when category changes
		if category != lastCategory {
			b.WriteString("\n\n")
			lastCategory = category
		} else {
			b.WriteString(" ")
		}

		// Add a connector if we're continuing in the same category
		if category == lastCategory && i > 0 {
			s = addConnector(s, i)
		}

		b.WriteString(s)
	}

	return b.String()
}

// addConnector prepends a transitional word to a sentence when appropriate.
func addConnector(s string, position int) string {
	lower := strings.ToLower(s)

	// Don't add connectors if sentence already starts with one
	startsWithConnector := false
	connectorStarts := []string{"additionally", "furthermore", "moreover", "also", "in addition", "however", "meanwhile", "specifically", "notably"}
	for _, c := range connectorStarts {
		if strings.HasPrefix(lower, c) {
			startsWithConnector = true
			break
		}
	}
	if startsWithConnector {
		return s
	}

	// Pick connector based on position (vary to avoid repetition)
	connectors := []string{"Additionally, ", "Furthermore, ", "Moreover, ", ""}
	connector := connectors[position%len(connectors)]
	if connector == "" {
		return s
	}

	// Lowercase the first letter of the original sentence after connector
	if len(s) > 0 && unicode.IsUpper(rune(s[0])) {
		s = string(unicode.ToLower(rune(s[0]))) + s[1:]
	}
	return connector + s
}

// extractSynthesisTopic extracts the core topic from a synthesis goal string.
func extractSynthesisTopic(goal string) string {
	lower := strings.ToLower(goal)
	prefixes := []string{
		"analyze key findings about ",
		"analyze ",
		"summarize findings about ",
		"summarize ",
		"write a report about ",
		"write a summary of ",
		"verify implementation of ",
	}
	for _, p := range prefixes {
		if strings.HasPrefix(lower, p) {
			return strings.TrimSpace(goal[len(p):])
		}
	}
	return goal
}

// -----------------------------------------------------------------------
// Formatting helpers
// -----------------------------------------------------------------------

// formatDocument converts a GeneratedDocument into readable markdown text.
func formatDocument(doc *cognitive.GeneratedDocument) string {
	var b strings.Builder
	fmt.Fprintf(&b, "# %s\n\n", doc.Title)
	for _, sec := range doc.Sections {
		fmt.Fprintf(&b, "## %s\n\n%s\n\n", sec.Heading, sec.Content)
	}
	return b.String()
}

// concatenateResults joins all results into a simple text block.
func concatenateResults(results map[string]string) string {
	var parts []string
	for _, v := range results {
		if v != "" {
			parts = append(parts, v)
		}
	}
	return strings.Join(parts, "\n\n")
}

// joinNonEmpty joins non-empty strings with newlines.
func joinNonEmpty(parts []string) string {
	var result []string
	for _, p := range parts {
		if p = strings.TrimSpace(p); p != "" {
			result = append(result, p)
		}
	}
	return strings.Join(result, "\n\n")
}

// truncateBridge shortens text to maxLen characters.
func truncateBridge(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
}

// filterByTopic removes sentences from text that don't relate to the topic.
func filterByTopic(text, topic string) string {
	topicWords := contentWordSet(topic)
	if len(topicWords) == 0 {
		return text
	}

	var kept []string
	for _, s := range splitSentences(text) {
		if isRelevantToTopic(s, topicWords) {
			kept = append(kept, s)
		}
	}
	if len(kept) == 0 {
		return text // filtering removed everything — return unfiltered
	}
	return strings.Join(kept, " ")
}

// synthStopWords for synthesis content word extraction.
var synthStopWords = map[string]bool{
	"the": true, "and": true, "for": true, "are": true, "but": true,
	"not": true, "you": true, "all": true, "can": true, "had": true,
	"her": true, "was": true, "one": true, "our": true, "out": true,
	"has": true, "its": true, "his": true, "how": true, "man": true,
	"new": true, "now": true, "old": true, "see": true, "way": true,
	"who": true, "did": true, "get": true, "let": true, "say": true,
	"she": true, "too": true, "use": true, "that": true, "with": true,
	"have": true, "this": true, "will": true, "your": true, "from": true,
	"they": true, "been": true, "said": true, "each": true, "which": true,
	"their": true, "there": true, "about": true, "would": true, "these": true,
	"other": true, "into": true, "more": true, "some": true, "such": true,
	"than": true, "when": true, "what": true, "also": true,
	"were": true, "then": true, "them": true,
	// Agent-specific noise words
	"based": true, "information": true, "results": true, "wrote": true,
	"search": true, "found": true, "bytes": true,
}
