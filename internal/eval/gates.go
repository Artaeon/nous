package eval

import (
	"fmt"
	"math"
	"strings"
	"time"
	"unicode"
)

// GateResult holds the outcome of a single quality gate check.
type GateResult struct {
	Gate    string
	Score   float64
	Pass    bool
	Details string
}

// QualityGates defines the minimum thresholds for each quality dimension.
type QualityGates struct {
	Correctness       float64 // minimum correctness score (default 0.85)
	HallucinationRate float64 // maximum hallucination rate (default 0.05)
	Helpfulness       float64 // minimum helpfulness (default 0.80)
	Coherence         float64 // minimum coherence (default 0.85)
	LatencyMs         int64   // maximum response latency in ms (default 500)
	FailureQuality    float64 // minimum quality when failing gracefully (default 0.70)
}

// DefaultGates returns quality gates with production-ready thresholds.
func DefaultGates() *QualityGates {
	return &QualityGates{
		Correctness:       0.85,
		HallucinationRate: 0.05,
		Helpfulness:       0.80,
		Coherence:         0.85,
		LatencyMs:         500,
		FailureQuality:    0.70,
	}
}

// CheckCorrectness evaluates factual alignment between response and gold answer.
// Uses token overlap, key-phrase matching, and negation detection.
func CheckCorrectness(response, gold string) GateResult {
	if gold == "" {
		return GateResult{Gate: "correctness", Score: 1.0, Pass: true, Details: "no gold answer provided; skipped"}
	}

	respLower := strings.ToLower(response)
	goldLower := strings.ToLower(gold)

	// Extract meaningful tokens (3+ chars, not stopwords).
	goldTokens := extractTokens(goldLower)
	respTokens := extractTokens(respLower)

	if len(goldTokens) == 0 {
		return GateResult{Gate: "correctness", Score: 1.0, Pass: true, Details: "gold answer has no meaningful tokens"}
	}

	// Token overlap score.
	respSet := toSet(respTokens)
	matched := 0
	for _, t := range goldTokens {
		if respSet[t] {
			matched++
		}
	}
	tokenOverlap := float64(matched) / float64(len(goldTokens))

	// Bigram overlap for phrase matching.
	goldBigrams := bigrams(goldTokens)
	respBigrams := bigrams(respTokens)
	bigramScore := 0.0
	if len(goldBigrams) > 0 {
		respBigramSet := toSet(respBigrams)
		bigramMatched := 0
		for _, b := range goldBigrams {
			if respBigramSet[b] {
				bigramMatched++
			}
		}
		bigramScore = float64(bigramMatched) / float64(len(goldBigrams))
	}

	// Negation penalty: if response negates key gold terms.
	negationPenalty := 0.0
	negators := []string{"not", "no", "never", "neither", "nor", "isn't", "aren't", "wasn't", "weren't", "doesn't", "don't", "didn't", "won't", "can't", "couldn't", "shouldn't", "wouldn't"}
	respWords := strings.Fields(respLower)
	for i, w := range respWords {
		for _, neg := range negators {
			if w == neg && i+1 < len(respWords) {
				// Check if the negated word is a key gold token.
				nextWord := cleanToken(respWords[i+1])
				for _, gt := range goldTokens {
					if nextWord == gt {
						negationPenalty += 0.15
					}
				}
			}
		}
	}
	if negationPenalty > 0.5 {
		negationPenalty = 0.5
	}

	// Weighted combination.
	score := 0.6*tokenOverlap + 0.4*bigramScore - negationPenalty
	if score < 0 {
		score = 0
	}
	if score > 1 {
		score = 1
	}

	return GateResult{
		Gate:    "correctness",
		Score:   score,
		Pass:    score >= DefaultGates().Correctness,
		Details: formatDetails("token_overlap=%.2f bigram=%.2f negation_penalty=%.2f", tokenOverlap, bigramScore, negationPenalty),
	}
}

// CheckHallucination detects unsupported claims in the response.
// Compares response claims against provided source material.
func CheckHallucination(response string, sources []string) GateResult {
	if len(sources) == 0 {
		// Without sources, use heuristic: check for hedging and uncertainty markers.
		return checkHallucinationHeuristic(response)
	}

	respLower := strings.ToLower(response)

	// Build source vocabulary.
	sourceVocab := make(map[string]bool)
	for _, src := range sources {
		for _, t := range extractTokens(strings.ToLower(src)) {
			sourceVocab[t] = true
		}
	}

	// Extract response claims (sentences with factual assertions).
	sentences := splitSentences(respLower)
	if len(sentences) == 0 {
		return GateResult{Gate: "hallucination", Score: 1.0, Pass: true, Details: "no sentences to evaluate"}
	}

	unsupported := 0
	total := 0

	for _, sent := range sentences {
		tokens := extractTokens(sent)
		if len(tokens) < 3 {
			continue // Skip very short fragments.
		}
		total++

		// Check what fraction of the sentence's content tokens appear in sources.
		supported := 0
		for _, t := range tokens {
			if sourceVocab[t] {
				supported++
			}
		}
		coverage := float64(supported) / float64(len(tokens))
		if coverage < 0.3 {
			unsupported++
		}
	}

	if total == 0 {
		return GateResult{Gate: "hallucination", Score: 1.0, Pass: true, Details: "no substantial sentences found"}
	}

	hallucinationRate := float64(unsupported) / float64(total)
	// Score is inverted: lower hallucination rate = higher score.
	score := 1.0 - hallucinationRate

	return GateResult{
		Gate:    "hallucination",
		Score:   score,
		Pass:    hallucinationRate <= DefaultGates().HallucinationRate,
		Details: formatDetails("unsupported=%d total=%d rate=%.3f", unsupported, total, hallucinationRate),
	}
}

// checkHallucinationHeuristic evaluates hallucination risk without source material
// by looking for confidence markers and specificity of claims.
func checkHallucinationHeuristic(response string) GateResult {
	respLower := strings.ToLower(response)
	sentences := splitSentences(respLower)
	if len(sentences) == 0 {
		return GateResult{Gate: "hallucination", Score: 1.0, Pass: true, Details: "empty response"}
	}

	// Hedging phrases indicate appropriate uncertainty.
	hedges := []string{
		"approximately", "roughly", "about", "around",
		"generally", "typically", "usually", "often",
		"it is believed", "according to", "research suggests",
		"commonly", "in most cases", "tends to",
	}

	// Overly specific unsupported claims are red flags.
	specificityMarkers := []string{
		"exactly", "precisely", "always", "never",
		"100%", "certainly", "undoubtedly", "proven fact",
		"everyone knows", "it is certain",
	}

	hedgeCount := 0
	specificCount := 0

	for _, h := range hedges {
		if strings.Contains(respLower, h) {
			hedgeCount++
		}
	}
	for _, s := range specificityMarkers {
		if strings.Contains(respLower, s) {
			specificCount++
		}
	}

	// More hedging = less likely to hallucinate.
	// More unsupported specificity = more likely to hallucinate.
	sentenceCount := float64(len(sentences))
	hedgeRatio := float64(hedgeCount) / sentenceCount
	specificRatio := float64(specificCount) / sentenceCount

	score := 0.8 + 0.1*math.Min(hedgeRatio, 1.0) - 0.2*math.Min(specificRatio, 1.0)
	if score < 0 {
		score = 0
	}
	if score > 1 {
		score = 1
	}

	return GateResult{
		Gate:    "hallucination",
		Score:   score,
		Pass:    score >= (1.0 - DefaultGates().HallucinationRate),
		Details: formatDetails("heuristic_mode hedges=%d specificity_flags=%d sentences=%d", hedgeCount, specificCount, len(sentences)),
	}
}

// CheckHelpfulness evaluates whether the response addresses the user's query.
// Uses query term coverage, response length adequacy, and question-answer alignment.
func CheckHelpfulness(response, query string) GateResult {
	if query == "" {
		return GateResult{Gate: "helpfulness", Score: 1.0, Pass: true, Details: "no query provided"}
	}

	queryLower := strings.ToLower(query)
	respLower := strings.ToLower(response)
	queryTokens := extractTokens(queryLower)
	respTokens := extractTokens(respLower)

	if len(respTokens) == 0 {
		return GateResult{Gate: "helpfulness", Score: 0.0, Pass: false, Details: "empty response"}
	}

	// 1. Query term coverage: how many query terms appear in the response.
	respSet := toSet(respTokens)
	queryMatched := 0
	for _, qt := range queryTokens {
		if respSet[qt] {
			queryMatched++
		}
	}
	queryCoverage := 0.0
	if len(queryTokens) > 0 {
		queryCoverage = float64(queryMatched) / float64(len(queryTokens))
	}

	// 2. Length adequacy: very short responses are less helpful.
	wordCount := len(strings.Fields(response))
	lengthScore := 1.0
	if wordCount < 5 {
		lengthScore = 0.2
	} else if wordCount < 10 {
		lengthScore = 0.5
	} else if wordCount < 20 {
		lengthScore = 0.8
	}

	// 3. Question-answer alignment: check if question type is addressed.
	qaScore := checkQAAlignment(queryLower, respLower)

	// 4. Refusal detection: penalize non-answers.
	refusalPenalty := 0.0
	refusalPhrases := []string{
		"i cannot", "i can't", "i'm unable", "i am unable",
		"i don't know", "i do not know", "i'm not sure",
		"as an ai", "as a language model",
	}
	for _, rp := range refusalPhrases {
		if strings.Contains(respLower, rp) {
			refusalPenalty = 0.3
			break
		}
	}

	score := 0.40*queryCoverage + 0.25*lengthScore + 0.35*qaScore - refusalPenalty
	if score < 0 {
		score = 0
	}
	if score > 1 {
		score = 1
	}

	return GateResult{
		Gate:    "helpfulness",
		Score:   score,
		Pass:    score >= DefaultGates().Helpfulness,
		Details: formatDetails("query_coverage=%.2f length=%.2f qa_align=%.2f refusal_penalty=%.2f", queryCoverage, lengthScore, qaScore, refusalPenalty),
	}
}

// checkQAAlignment detects if the response type matches the question type.
func checkQAAlignment(query, response string) float64 {
	score := 0.5 // baseline

	// "What is" questions should have definitional answers.
	if strings.HasPrefix(query, "what is") || strings.HasPrefix(query, "what are") {
		definitional := []string{"is ", "are ", "refers to", "means", "defined as", "a type of", "the process"}
		for _, d := range definitional {
			if strings.Contains(response, d) {
				score = 1.0
				break
			}
		}
	}

	// "How" questions should have procedural or explanatory answers.
	if strings.HasPrefix(query, "how") {
		procedural := []string{"by ", "through ", "using ", "step", "first", "then", "process", "method", "way"}
		for _, p := range procedural {
			if strings.Contains(response, p) {
				score = 1.0
				break
			}
		}
	}

	// "Why" questions should have causal answers.
	if strings.HasPrefix(query, "why") {
		causal := []string{"because", "due to", "since", "as a result", "caused by", "reason", "leads to"}
		for _, c := range causal {
			if strings.Contains(response, c) {
				score = 1.0
				break
			}
		}
	}

	// "When" questions should have temporal answers.
	if strings.HasPrefix(query, "when") {
		temporal := []string{"in ", "on ", "during ", "year", "century", "date", "time", "period", "era", "ago"}
		for _, t := range temporal {
			if strings.Contains(response, t) {
				score = 1.0
				break
			}
		}
	}

	// "Who" questions should mention people or organizations.
	if strings.HasPrefix(query, "who") {
		// Check for capitalized words (proper nouns) in response.
		words := strings.Fields(response)
		for _, w := range words {
			if len(w) > 1 && unicode.IsUpper(rune(w[0])) {
				score = 1.0
				break
			}
		}
	}

	// Compare/vs questions should address both sides.
	if strings.Contains(query, " vs ") || strings.Contains(query, "compare") || strings.Contains(query, "pros and cons") {
		contrastive := []string{"while", "whereas", "however", "on the other hand", "but", "advantage", "disadvantage", "pros", "cons"}
		for _, c := range contrastive {
			if strings.Contains(response, c) {
				score = 1.0
				break
			}
		}
	}

	return score
}

// CheckCoherence evaluates the structural coherence of a response.
// Checks sentence structure, vocabulary diversity, repetition, and logical connectors.
func CheckCoherence(response string) GateResult {
	if len(strings.TrimSpace(response)) == 0 {
		return GateResult{Gate: "coherence", Score: 0.0, Pass: false, Details: "empty response"}
	}

	sentences := splitSentences(response)
	words := strings.Fields(strings.ToLower(response))

	if len(words) == 0 {
		return GateResult{Gate: "coherence", Score: 0.0, Pass: false, Details: "no words found"}
	}

	// 1. Sentence structure: sentences should start with uppercase and end properly.
	structureScore := 0.0
	if len(sentences) > 0 {
		wellFormed := 0
		for _, s := range sentences {
			s = strings.TrimSpace(s)
			if len(s) == 0 {
				continue
			}
			hasUpperStart := unicode.IsUpper(rune(s[0]))
			wordCount := len(strings.Fields(s))
			reasonable := wordCount >= 2 && wordCount <= 80
			if hasUpperStart && reasonable {
				wellFormed++
			}
		}
		if len(sentences) > 0 {
			structureScore = float64(wellFormed) / float64(len(sentences))
		}
	}

	// 2. Vocabulary diversity (type-token ratio, capped for long texts).
	uniqueWords := toSet(words)
	sampleSize := len(words)
	if sampleSize > 100 {
		sampleSize = 100
	}
	sample := words[:sampleSize]
	sampleUnique := toSet(sample)
	ttr := float64(len(sampleUnique)) / float64(len(sample))
	diversityScore := math.Min(ttr/0.6, 1.0) // Normalized so 0.6 TTR = 1.0.

	// 3. Repetition detection: penalize repeated consecutive sentences or phrases.
	repetitionPenalty := 0.0
	for i := 1; i < len(sentences); i++ {
		if strings.TrimSpace(sentences[i]) == strings.TrimSpace(sentences[i-1]) && len(strings.TrimSpace(sentences[i])) > 0 {
			repetitionPenalty += 0.2
		}
	}
	// Also check for word-level repetition (same word 3+ times consecutively).
	for i := 2; i < len(words); i++ {
		if words[i] == words[i-1] && words[i] == words[i-2] && len(words[i]) > 2 {
			repetitionPenalty += 0.1
		}
	}
	if repetitionPenalty > 0.6 {
		repetitionPenalty = 0.6
	}

	// 4. Logical connectors: presence indicates structured thinking.
	connectors := []string{
		"therefore", "however", "furthermore", "additionally", "moreover",
		"because", "since", "thus", "consequently", "although",
		"first", "second", "finally", "in addition", "for example",
		"specifically", "in particular", "as a result", "on the other hand",
	}
	connectorCount := 0
	lowerResp := strings.ToLower(response)
	for _, c := range connectors {
		if strings.Contains(lowerResp, c) {
			connectorCount++
		}
	}
	connectorScore := math.Min(float64(connectorCount)/3.0, 1.0)

	// 5. Length sanity: extremely short or absurdly long responses.
	lengthFactor := 1.0
	if len(words) < 3 {
		lengthFactor = 0.3
	}

	_ = uniqueWords // used for diversity in a different context

	score := (0.30*structureScore + 0.25*diversityScore + 0.20*connectorScore + 0.25*lengthFactor) - repetitionPenalty
	if score < 0 {
		score = 0
	}
	if score > 1 {
		score = 1
	}

	return GateResult{
		Gate:    "coherence",
		Score:   score,
		Pass:    score >= DefaultGates().Coherence,
		Details: formatDetails("structure=%.2f diversity=%.2f connectors=%d repetition_penalty=%.2f length_factor=%.2f", structureScore, diversityScore, connectorCount, repetitionPenalty, lengthFactor),
	}
}

// CheckLatency verifies the response was generated within the latency budget.
func CheckLatency(elapsed time.Duration) GateResult {
	gates := DefaultGates()
	elapsedMs := elapsed.Milliseconds()
	budgetMs := gates.LatencyMs

	// Score degrades linearly from 1.0 at 0ms to 0.0 at 2x budget.
	score := 1.0 - float64(elapsedMs)/float64(2*budgetMs)
	if score < 0 {
		score = 0
	}
	if score > 1 {
		score = 1
	}

	pass := elapsedMs <= budgetMs

	return GateResult{
		Gate:    "latency",
		Score:   score,
		Pass:    pass,
		Details: formatDetails("elapsed=%dms budget=%dms", elapsedMs, budgetMs),
	}
}

// CheckFailureQuality evaluates the quality of a failure/fallback response.
// A good failure should still be helpful, honest about limitations, and suggest alternatives.
func CheckFailureQuality(response string, isFailure bool) GateResult {
	if !isFailure {
		return GateResult{Gate: "failure_quality", Score: 1.0, Pass: true, Details: "not a failure response"}
	}

	respLower := strings.ToLower(response)
	words := strings.Fields(response)

	// 1. Honesty: acknowledges the limitation.
	honestyScore := 0.0
	honestyMarkers := []string{
		"i'm not sure", "i don't know", "i cannot", "i'm unable",
		"i don't have", "i lack", "beyond my", "outside my",
		"not certain", "uncertain", "may not be accurate",
		"i apologize", "sorry", "unfortunately",
	}
	for _, m := range honestyMarkers {
		if strings.Contains(respLower, m) {
			honestyScore = 1.0
			break
		}
	}

	// 2. Helpfulness despite failure: offers alternatives or partial info.
	alternativeScore := 0.0
	alternativeMarkers := []string{
		"however", "but", "instead", "you could", "you might",
		"try", "consider", "suggest", "recommend", "alternatively",
		"what i can tell you", "here's what i know",
	}
	for _, m := range alternativeMarkers {
		if strings.Contains(respLower, m) {
			alternativeScore = 1.0
			break
		}
	}

	// 3. Non-empty and reasonable length.
	lengthScore := 0.0
	if len(words) >= 10 {
		lengthScore = 1.0
	} else if len(words) >= 5 {
		lengthScore = 0.6
	} else if len(words) > 0 {
		lengthScore = 0.3
	}

	// 4. Politeness and professionalism.
	politenessScore := 0.5 // baseline
	politeMarkers := []string{"please", "thank", "happy to", "glad to", "hope", "help"}
	for _, m := range politeMarkers {
		if strings.Contains(respLower, m) {
			politenessScore = 1.0
			break
		}
	}

	score := 0.30*honestyScore + 0.30*alternativeScore + 0.20*lengthScore + 0.20*politenessScore
	if score > 1 {
		score = 1
	}

	return GateResult{
		Gate:    "failure_quality",
		Score:   score,
		Pass:    score >= DefaultGates().FailureQuality,
		Details: formatDetails("honesty=%.2f alternatives=%.2f length=%.2f politeness=%.2f", honestyScore, alternativeScore, lengthScore, politenessScore),
	}
}

// RunAllGates executes all quality gates and returns the results.
func RunAllGates(gates *QualityGates, response, query, gold string, sources []string, elapsed time.Duration) []GateResult {
	results := make([]GateResult, 0, 6)

	// Run correctness check.
	cr := CheckCorrectness(response, gold)
	cr.Pass = cr.Score >= gates.Correctness
	results = append(results, cr)

	// Run hallucination check.
	hr := CheckHallucination(response, sources)
	// For hallucination, the score represents 1-rate, so pass when score >= 1-threshold.
	hr.Pass = hr.Score >= (1.0 - gates.HallucinationRate)
	results = append(results, hr)

	// Run helpfulness check.
	hf := CheckHelpfulness(response, query)
	hf.Pass = hf.Score >= gates.Helpfulness
	results = append(results, hf)

	// Run coherence check.
	cc := CheckCoherence(response)
	cc.Pass = cc.Score >= gates.Coherence
	results = append(results, cc)

	// Run latency check.
	lc := CheckLatency(elapsed)
	lc.Pass = elapsed.Milliseconds() <= gates.LatencyMs
	results = append(results, lc)

	// Run failure quality check (detect if response appears to be a failure).
	isFailure := detectFailure(strings.ToLower(response))
	fq := CheckFailureQuality(response, isFailure)
	if isFailure {
		fq.Pass = fq.Score >= gates.FailureQuality
	}
	results = append(results, fq)

	return results
}

// detectFailure heuristically determines if a response is a failure/fallback.
func detectFailure(respLower string) bool {
	failureIndicators := []string{
		"i cannot", "i can't", "i'm unable", "i am unable",
		"i don't have access", "i don't have the ability",
		"as an ai", "as a language model",
		"i'm not able to", "beyond my capabilities",
	}
	for _, f := range failureIndicators {
		if strings.Contains(respLower, f) {
			return true
		}
	}
	return false
}

// --- Utility functions ---

var stopWords = map[string]bool{
	"a": true, "an": true, "the": true, "is": true, "are": true,
	"was": true, "were": true, "be": true, "been": true, "being": true,
	"have": true, "has": true, "had": true, "do": true, "does": true,
	"did": true, "will": true, "would": true, "could": true, "should": true,
	"may": true, "might": true, "shall": true, "can": true, "to": true,
	"of": true, "in": true, "for": true, "on": true, "with": true,
	"at": true, "by": true, "from": true, "as": true, "into": true,
	"through": true, "during": true, "before": true, "after": true,
	"and": true, "but": true, "or": true, "nor": true, "not": true,
	"so": true, "yet": true, "both": true, "either": true, "neither": true,
	"it": true, "its": true, "this": true, "that": true, "these": true,
	"those": true, "i": true, "you": true, "he": true, "she": true,
	"we": true, "they": true, "me": true, "him": true, "her": true,
	"us": true, "them": true, "my": true, "your": true, "his": true,
	"our": true, "their": true, "what": true, "which": true, "who": true,
	"whom": true, "how": true, "when": true, "where": true, "why": true,
	"if": true, "then": true, "than": true, "also": true, "just": true,
	"about": true, "up": true, "out": true, "no": true, "yes": true,
	"very": true, "more": true, "most": true, "other": true, "some": true,
	"such": true, "only": true, "own": true, "same": true, "each": true,
}

// extractTokens splits text into meaningful lowercase tokens, filtering stopwords
// and short words.
func extractTokens(text string) []string {
	words := strings.Fields(text)
	tokens := make([]string, 0, len(words))
	for _, w := range words {
		clean := cleanToken(w)
		if len(clean) >= 3 && !stopWords[clean] {
			tokens = append(tokens, clean)
		}
	}
	return tokens
}

// cleanToken removes punctuation from the edges of a word.
func cleanToken(w string) string {
	return strings.TrimFunc(w, func(r rune) bool {
		return !unicode.IsLetter(r) && !unicode.IsDigit(r)
	})
}

// toSet converts a string slice to a set for O(1) lookups.
func toSet(items []string) map[string]bool {
	s := make(map[string]bool, len(items))
	for _, item := range items {
		s[item] = true
	}
	return s
}

// bigrams generates adjacent token pairs.
func bigrams(tokens []string) []string {
	if len(tokens) < 2 {
		return nil
	}
	result := make([]string, 0, len(tokens)-1)
	for i := 0; i < len(tokens)-1; i++ {
		result = append(result, tokens[i]+" "+tokens[i+1])
	}
	return result
}

// splitSentences splits text into sentences using basic punctuation heuristics.
func splitSentences(text string) []string {
	var sentences []string
	var current strings.Builder

	runes := []rune(text)
	for i := 0; i < len(runes); i++ {
		current.WriteRune(runes[i])

		// End of sentence detection.
		if runes[i] == '.' || runes[i] == '!' || runes[i] == '?' {
			// Look ahead: if next char is space or end, it's likely a sentence boundary.
			// But skip abbreviations like "Dr.", "U.S.", single letters followed by period.
			if i+1 >= len(runes) || unicode.IsSpace(runes[i+1]) || unicode.IsUpper(runes[i+1]) {
				s := strings.TrimSpace(current.String())
				if len(s) > 0 {
					sentences = append(sentences, s)
				}
				current.Reset()
			}
		}
	}

	// Remaining text.
	s := strings.TrimSpace(current.String())
	if len(s) > 0 {
		sentences = append(sentences, s)
	}

	return sentences
}

// formatDetails is a convenience wrapper for fmt.Sprintf.
func formatDetails(format string, args ...interface{}) string {
	return fmt.Sprintf(format, args...)
}
