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
// into a structured, readable report. This is the core function that
// transforms the agent from a link dumper into an analyst.
//
// Pipeline:
//  1. CLEAN raw search output (strip URLs, numbering, truncation markers)
//  2. EXTRACT content sentences (claims, facts, data points)
//  3. FILTER for topic relevance
//  4. DEDUPLICATE near-identical sentences
//  5. CLUSTER by subtopic (group related sentences)
//  6. FORMAT as structured markdown with section headers
func (cb *CognitiveBridge) SmartSynthesize(topic string, webResults []string, graphFacts []string) string {
	// 1. Clean and extract content from raw search results
	var extracted []extractedFact
	var sources []string

	for _, raw := range webResults {
		if raw == "" {
			continue
		}
		facts, srcs := extractFromSearchResult(raw)
		extracted = append(extracted, facts...)
		sources = append(sources, srcs...)
	}

	// Add knowledge graph content
	for _, fact := range graphFacts {
		for _, s := range splitSentences(fact) {
			s = strings.TrimSpace(s)
			if len(s) > 25 {
				extracted = append(extracted, extractedFact{text: s, theme: classifyTheme(s)})
			}
		}
	}

	if len(extracted) == 0 {
		return "No relevant information found for: " + topic
	}

	// 2. Filter for topic relevance
	topicWords := contentWordSet(topic)
	var relevant []extractedFact
	for _, f := range extracted {
		if isRelevantToTopic(f.text, topicWords) {
			relevant = append(relevant, f)
		}
	}
	if len(relevant) == 0 {
		relevant = extracted
	}

	// 3. Deduplicate
	relevant = deduplicateFacts(relevant, 0.5)

	// 4. Cluster by theme
	clusters := clusterByTheme(relevant)

	// 5. Format as structured markdown
	return formatAsReport(topic, clusters, sources)
}

// extractedFact is a single content claim extracted from search results.
type extractedFact struct {
	text  string // the actual content sentence
	theme string // "overview", "strategy", "data", "challenge", "trend", "example"
}

// extractFromSearchResult cleans a raw DuckDuckGo search result and
// extracts content sentences. Strips URLs, numbering, title lines.
func extractFromSearchResult(raw string) ([]extractedFact, []string) {
	var facts []extractedFact
	var sources []string

	lines := strings.Split(raw, "\n")
	for _, line := range lines {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}

		// Extract source URLs for citation
		if strings.HasPrefix(line, "http") || strings.HasPrefix(line, "   http") {
			url := strings.TrimSpace(line)
			if url != "" {
				// Extract domain for citation
				domain := extractDomain(url)
				if domain != "" && !containsString(sources, domain) {
					sources = append(sources, domain)
				}
			}
			continue
		}

		// Strip numbered prefix: "1. ", "2. ", etc.
		cleaned := line
		if len(cleaned) > 3 && cleaned[0] >= '0' && cleaned[0] <= '9' {
			dotIdx := strings.Index(cleaned, ". ")
			if dotIdx >= 0 && dotIdx <= 3 {
				cleaned = strings.TrimSpace(cleaned[dotIdx+2:])
			}
		}

		// Strip leading whitespace (indented snippets)
		cleaned = strings.TrimSpace(cleaned)

		// Skip if it's just a title (short, no period, title-case)
		if len(cleaned) < 40 && !strings.Contains(cleaned, ".") {
			continue
		}

		// Skip truncated snippets ending in "..."
		if strings.HasSuffix(cleaned, "...") {
			cleaned = strings.TrimSuffix(cleaned, "...")
			// Only keep if substantial
			if len(cleaned) < 30 {
				continue
			}
		}

		// Skip very short fragments
		if len(cleaned) < 25 {
			continue
		}

		// Skip article titles masquerading as content
		if isArticleTitle(cleaned) {
			continue
		}

		// Skip author bylines: "Author Name N min read · Date"
		if strings.Contains(cleaned, "min read") || strings.Contains(cleaned, "· ") {
			continue
		}

		// Skip sentences that end abruptly (truncated snippets)
		if !strings.HasSuffix(cleaned, ".") && !strings.HasSuffix(cleaned, "!") &&
			!strings.HasSuffix(cleaned, "?") && !strings.HasSuffix(cleaned, ":") {
			// Check if it looks like a truncated sentence
			words := strings.Fields(cleaned)
			lastWord := ""
			if len(words) > 0 {
				lastWord = strings.ToLower(words[len(words)-1])
			}
			// Articles, prepositions, conjunctions at the end = truncated
			truncMarkers := map[string]bool{
				"the": true, "a": true, "an": true, "and": true, "or": true,
				"but": true, "that": true, "which": true, "many": true,
				"with": true, "for": true, "from": true, "into": true,
			}
			if truncMarkers[lastWord] {
				continue
			}
		}

		// Split into sentences and classify each
		for _, sent := range splitSentences(cleaned) {
			sent = strings.TrimSpace(sent)
			if len(sent) < 25 {
				continue
			}
			if isMetaSentence(sent) {
				continue
			}
			if isArticleTitle(sent) {
				continue
			}
			facts = append(facts, extractedFact{
				text:  sent,
				theme: classifyTheme(sent),
			})
		}
	}

	return facts, sources
}

// classifyTheme determines what kind of content a sentence represents.
func classifyTheme(s string) string {
	lower := strings.ToLower(s)

	// Data/statistics
	for _, sig := range []string{"billion", "million", "percent", "%", "revenue", "market", "growth", "projected", "forecast", "cagr"} {
		if strings.Contains(lower, sig) {
			return "data"
		}
	}

	// Strategies/approaches
	for _, sig := range []string{"strategy", "model", "approach", "method", "way to", "option", "path", "pricing", "monetiz", "license", "freemium", "subscription", "open core", "dual licens", "support contract"} {
		if strings.Contains(lower, sig) {
			return "strategy"
		}
	}

	// Challenges/problems
	for _, sig := range []string{"challenge", "problem", "risk", "difficult", "struggle", "threat", "concern", "drawback", "disadvantage", "limitation"} {
		if strings.Contains(lower, sig) {
			return "challenge"
		}
	}

	// Trends/future
	for _, sig := range []string{"trend", "future", "emerging", "growing", "2025", "2026", "2027", "2028", "2030", "increasingly", "shift"} {
		if strings.Contains(lower, sig) {
			return "trend"
		}
	}

	// Examples/case studies
	for _, sig := range []string{"example", "such as", "like ", "including", "case study", "for instance"} {
		if strings.Contains(lower, sig) {
			return "example"
		}
	}

	// Definitions/overview
	for _, sig := range []string{" is a ", " is an ", " are a ", " refers to", " means ", " defined as"} {
		if strings.Contains(lower, sig) {
			return "overview"
		}
	}

	return "overview" // default
}

// isMetaSentence returns true for sentences about the article, not the topic.
func isMetaSentence(s string) bool {
	lower := strings.ToLower(s)
	meta := []string{
		"in this article", "in this post", "in this blog", "in this guide",
		"read more", "click here", "subscribe", "sign up", "learn more",
		"table of contents", "share this", "related articles",
		"we will explore", "we'll cover", "let's look at", "let's dive",
		"this article", "this post", "this guide", "this report",
		"in the following sections", "you can find the", "for more insights",
		"explore the ", "what specifically about",
		"no results found for:", "what angle are you",
		"min read", "would you like to explore",
		"let me share what i know", "context really matters",
		"that's an interesting area",
	}
	for _, m := range meta {
		if strings.Contains(lower, m) {
			return true
		}
	}
	return false
}

// isArticleTitle returns true for lines that are article titles, not content.
func isArticleTitle(s string) bool {
	lower := strings.ToLower(s)

	// Site name patterns anywhere: "- Medium", "| Forbes", "- GitHub Blog"
	sitePatterns := []string{
		" - medium", " | medium", " - github", " | github",
		" - forbes", " | forbes", "- the github blog", " - dev.to",
		" - reviews", " - comparison", "| forecast", "| trends",
		" - sourceforge", " - wikipedia", ": the complete guide",
		": a comprehensive guide", ": practical tools",
	}
	for _, p := range sitePatterns {
		if strings.Contains(lower, p) {
			return true
		}
	}

	// Promotional/CTA starts
	promoStarts := []string{
		"discover the", "explore the", "learn how", "find out how",
		"find the highest", "see the key", "see how",
	}
	for _, p := range promoStarts {
		if strings.HasPrefix(lower, p) {
			return true
		}
	}

	// Title-case heuristic: if most words are capitalized and no period, likely a title
	words := strings.Fields(s)
	if len(words) >= 3 && len(words) <= 15 && !strings.Contains(s, ".") {
		capCount := 0
		for _, w := range words {
			if len(w) > 0 && w[0] >= 'A' && w[0] <= 'Z' {
				capCount++
			}
		}
		// If >60% of words start with uppercase, it's a title
		if float64(capCount)/float64(len(words)) > 0.6 {
			return true
		}
	}

	return false
}

// deduplicateFacts removes near-duplicate extracted facts.
func deduplicateFacts(facts []extractedFact, threshold float64) []extractedFact {
	var result []extractedFact
	for _, f := range facts {
		dup := false
		for _, kept := range result {
			if jaccardSimilarity(f.text, kept.text) > threshold {
				dup = true
				break
			}
		}
		if !dup {
			result = append(result, f)
		}
	}
	return result
}

// clusterByTheme groups facts by their theme.
func clusterByTheme(facts []extractedFact) map[string][]string {
	clusters := make(map[string][]string)
	for _, f := range facts {
		clusters[f.theme] = append(clusters[f.theme], f.text)
	}
	return clusters
}

// formatAsReport builds a structured markdown report from clustered facts.
func formatAsReport(topic string, clusters map[string][]string, sources []string) string {
	var b strings.Builder

	// Title
	fmt.Fprintf(&b, "# %s\n\n", capitalizeTitle(topic))

	// Executive summary — first 2-3 overview sentences
	if overview, ok := clusters["overview"]; ok && len(overview) > 0 {
		fmt.Fprintf(&b, "## Overview\n\n")
		max := 3
		if len(overview) < max {
			max = len(overview)
		}
		for _, s := range overview[:max] {
			fmt.Fprintf(&b, "%s ", s)
		}
		fmt.Fprintf(&b, "\n\n")
	}

	// Strategies/approaches
	if strategies, ok := clusters["strategy"]; ok && len(strategies) > 0 {
		fmt.Fprintf(&b, "## Key Findings\n\n")
		for _, s := range strategies {
			fmt.Fprintf(&b, "- %s\n", s)
		}
		fmt.Fprintf(&b, "\n")
	}

	// Data/statistics
	if data, ok := clusters["data"]; ok && len(data) > 0 {
		fmt.Fprintf(&b, "## Market Data\n\n")
		for _, s := range data {
			fmt.Fprintf(&b, "- %s\n", s)
		}
		fmt.Fprintf(&b, "\n")
	}

	// Trends
	if trends, ok := clusters["trend"]; ok && len(trends) > 0 {
		fmt.Fprintf(&b, "## Trends\n\n")
		for _, s := range trends {
			fmt.Fprintf(&b, "%s ", s)
		}
		fmt.Fprintf(&b, "\n\n")
	}

	// Challenges
	if challenges, ok := clusters["challenge"]; ok && len(challenges) > 0 {
		fmt.Fprintf(&b, "## Challenges\n\n")
		for _, s := range challenges {
			fmt.Fprintf(&b, "- %s\n", s)
		}
		fmt.Fprintf(&b, "\n")
	}

	// Examples
	if examples, ok := clusters["example"]; ok && len(examples) > 0 {
		fmt.Fprintf(&b, "## Examples\n\n")
		for _, s := range examples {
			fmt.Fprintf(&b, "- %s\n", s)
		}
		fmt.Fprintf(&b, "\n")
	}

	// Sources — deduplicated
	if len(sources) > 0 {
		seen := make(map[string]bool)
		var uniqueSources []string
		for _, s := range sources {
			if !seen[s] {
				seen[s] = true
				uniqueSources = append(uniqueSources, s)
			}
		}
		fmt.Fprintf(&b, "## Sources\n\n")
		for _, s := range uniqueSources {
			fmt.Fprintf(&b, "- %s\n", s)
		}
		fmt.Fprintf(&b, "\n")
	}

	result := b.String()
	if strings.Count(result, "\n") <= 3 {
		return "No substantial content found for: " + topic
	}
	return result
}

// extractDomain extracts the domain name from a URL for citation.
func extractDomain(url string) string {
	// Strip protocol
	u := url
	for _, prefix := range []string{"https://", "http://", "www."} {
		u = strings.TrimPrefix(u, prefix)
	}
	// Take just the domain
	if idx := strings.IndexByte(u, '/'); idx > 0 {
		u = u[:idx]
	}
	return strings.TrimSpace(u)
}

// containsString checks if a slice contains a string.
func containsString(slice []string, s string) bool {
	for _, v := range slice {
		if v == s {
			return true
		}
	}
	return false
}

// capitalizeTitle capitalizes each word in a title.
func capitalizeTitle(s string) string {
	words := strings.Fields(s)
	for i, w := range words {
		if len(w) > 0 {
			words[i] = strings.ToUpper(w[:1]) + w[1:]
		}
	}
	return strings.Join(words, " ")
}

// SynthesizeResults takes raw tool outputs and produces a structured report.
func (cb *CognitiveBridge) SynthesizeResults(goal string, results map[string]string) string {
	if cb == nil {
		return concatenateResults(results)
	}

	topic := extractSynthesisTopic(goal)

	// Collect all raw results
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
	// Try DocumentGenerator first (knowledge graph content)
	doc := cb.GenerateDocument(topic, style)

	// Try synthesis from web results
	synth := cb.SynthesizeResults("write a "+style+" about "+topic, results)

	// Merge: if both produced content, combine them
	if doc != "" && synth != "" && len(synth) > 100 {
		return doc + "\n\n---\n\n## Web Research Findings\n\n" + synth
	}
	if doc != "" {
		return doc
	}
	if synth != "" {
		return synth
	}
	return "Report generation pending — insufficient data for: " + topic
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
// Groups sentences by semantic category, adds connectors only between groups.
func joinWithConnectors(topic string, sentences []string) string {
	if len(sentences) == 0 {
		return ""
	}
	if len(sentences) == 1 {
		return sentences[0]
	}

	// Group sentences by category
	type group struct {
		category  int
		sentences []string
	}
	var groups []group
	lastCat := -1
	for _, s := range sentences {
		cat := classifySentence(s)
		if cat != lastCat || len(groups) == 0 {
			groups = append(groups, group{category: cat})
			lastCat = cat
		}
		groups[len(groups)-1].sentences = append(groups[len(groups)-1].sentences, s)
	}

	// Build paragraphs — one per group
	var paragraphs []string
	for gi, g := range groups {
		var para strings.Builder
		for si, s := range g.sentences {
			if si > 0 {
				para.WriteString(" ")
			}
			para.WriteString(s)
		}

		text := para.String()

		// Add a transition only at the start of a NEW group (not every sentence)
		if gi > 0 {
			text = addGroupTransition(g.category, text)
		}

		paragraphs = append(paragraphs, text)
	}

	return strings.Join(paragraphs, "\n\n")
}

// addGroupTransition prepends a category-appropriate transition to a paragraph.
func addGroupTransition(category int, text string) string {
	var prefix string
	switch category {
	case 1: // history
		prefix = "Historically, "
	case 2: // properties
		prefix = "In terms of key characteristics, "
	case 3: // applications
		prefix = "In practice, "
	case 4: // outlook
		prefix = "Looking ahead, "
	default:
		return text
	}

	// Lowercase the first char of the original text
	if len(text) > 0 && unicode.IsUpper(rune(text[0])) {
		text = string(unicode.ToLower(rune(text[0]))) + text[1:]
	}
	return prefix + text
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
