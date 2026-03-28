package cognitive

import (
	"strings"
)

// RerankerSignal is one named quality signal with its score and weight.
type RerankerSignal struct {
	Name   string
	Score  float64
	Weight float64
}

// RerankResult captures the full reranking decision.
type RerankResult struct {
	Candidates  []string
	Scores      [][]RerankerSignal // signals per candidate
	TotalScores []float64
	BestIndex   int
	BestScore   float64
}

// DeterministicReranker scores and ranks candidate responses using
// deterministic quality signals: coverage, contradiction, specificity,
// repetition, and user-goal fit.
type DeterministicReranker struct {
	Weights map[string]float64
}

// NewDeterministicReranker creates a reranker with default signal weights.
func NewDeterministicReranker() *DeterministicReranker {
	return &DeterministicReranker{
		Weights: map[string]float64{
			"coverage":      0.30,
			"contradiction": 0.20,
			"specificity":   0.18,
			"repetition":    0.15,
			"user_goal":     0.17,
		},
	}
}

// Rerank scores all candidates and selects the best one.
func (r *DeterministicReranker) Rerank(candidates []string, plan *ContentPlan, query string) *RerankResult {
	if len(candidates) == 0 {
		return &RerankResult{BestIndex: -1}
	}

	result := &RerankResult{
		Candidates:  candidates,
		Scores:      make([][]RerankerSignal, len(candidates)),
		TotalScores: make([]float64, len(candidates)),
		BestIndex:   0,
		BestScore:   -1.0,
	}

	for i, cand := range candidates {
		signals := r.scoreCandidate(cand, plan, query)
		result.Scores[i] = signals

		total := 0.0
		for _, sig := range signals {
			total += sig.Score * sig.Weight
		}
		result.TotalScores[i] = total

		if total > result.BestScore {
			result.BestScore = total
			result.BestIndex = i
		}
	}

	return result
}

// scoreCandidate computes all five signals for a single candidate.
func (r *DeterministicReranker) scoreCandidate(candidate string, plan *ContentPlan, query string) []RerankerSignal {
	return []RerankerSignal{
		{Name: "coverage", Score: scoreCoverageSignal(candidate, plan), Weight: r.Weights["coverage"]},
		{Name: "contradiction", Score: scoreContradictionSignal(candidate), Weight: r.Weights["contradiction"]},
		{Name: "specificity", Score: scoreSpecificitySignal(candidate), Weight: r.Weights["specificity"]},
		{Name: "repetition", Score: scoreRepetitionSignal(candidate), Weight: r.Weights["repetition"]},
		{Name: "user_goal", Score: scoreUserGoalFit(candidate, query), Weight: r.Weights["user_goal"]},
	}
}

// scoreCoverageSignal measures what fraction of plan claims appear in the candidate.
func scoreCoverageSignal(candidate string, plan *ContentPlan) float64 {
	if plan == nil || len(plan.Claims) == 0 {
		// No plan to measure against; give a moderate baseline.
		if plan != nil && plan.Topic != "" && containsLoose(candidate, plan.Topic) {
			return 0.7
		}
		return 0.5
	}

	lower := strings.ToLower(candidate)
	hits := 0
	for _, claim := range plan.Claims {
		if claim.Text == "" {
			continue
		}
		claimLower := strings.ToLower(claim.Text)
		// Check for full claim or key span match
		if strings.Contains(lower, claimLower) {
			hits++
			continue
		}
		// Try key span (strip grammar words from edges)
		ks := strings.ToLower(keySpan(claim.Text))
		if ks != "" && strings.Contains(lower, ks) {
			hits++
			continue
		}
		// Try content word overlap: if 60%+ of content words from the claim appear, count it
		claimWords := rerankerContentWords(claim.Text)
		if len(claimWords) > 0 {
			matchCount := 0
			for _, w := range claimWords {
				if strings.Contains(lower, w) {
					matchCount++
				}
			}
			if float64(matchCount)/float64(len(claimWords)) >= 0.6 {
				hits++
			}
		}
	}

	return float64(hits) / float64(len(plan.Claims))
}

// scoreContradictionSignal returns 1.0 if no contradictions found, lower otherwise.
// Checks sentence pairs for negation/antonymy patterns.
func scoreContradictionSignal(candidate string) float64 {
	sentences := splitSentences(candidate)
	if len(sentences) < 2 {
		return 1.0
	}

	contradictions := 0
	for i := 0; i < len(sentences); i++ {
		si := strings.ToLower(strings.TrimSpace(sentences[i]))
		if si == "" {
			continue
		}
		for j := i + 1; j < len(sentences); j++ {
			sj := strings.ToLower(strings.TrimSpace(sentences[j]))
			if sj == "" {
				continue
			}
			if detectRerankerContradiction(si, sj) {
				contradictions++
			}
		}
	}

	if contradictions == 0 {
		return 1.0
	}

	// Each contradiction reduces score significantly
	penalty := float64(contradictions) * 0.3
	score := 1.0 - penalty
	if score < 0 {
		return 0
	}
	return score
}

// detectRerankerContradiction checks if two sentences contradict each other.
func detectRerankerContradiction(a, b string) bool {
	// Pattern 1: "X is Y" vs "X is not Y"
	if strings.Contains(a, " is ") && strings.Contains(b, " is not ") {
		aWords := strings.Fields(a)
		bWords := strings.Fields(b)
		if len(aWords) >= 3 && len(bWords) >= 4 {
			// Check if same subject
			if aWords[0] == bWords[0] {
				return true
			}
		}
	}
	if strings.Contains(b, " is ") && strings.Contains(a, " is not ") {
		aWords := strings.Fields(a)
		bWords := strings.Fields(b)
		if len(aWords) >= 4 && len(bWords) >= 3 {
			if aWords[0] == bWords[0] {
				return true
			}
		}
	}

	// Pattern 2: Antonym pairs
	antonymPairs := [][2]string{
		{"good", "bad"},
		{"fast", "slow"},
		{"large", "small"},
		{"easy", "difficult"},
		{"simple", "complex"},
		{"strong", "weak"},
		{"safe", "dangerous"},
		{"efficient", "inefficient"},
		{"useful", "useless"},
		{"positive", "negative"},
	}

	for _, pair := range antonymPairs {
		if (strings.Contains(a, pair[0]) && strings.Contains(b, pair[1])) ||
			(strings.Contains(a, pair[1]) && strings.Contains(b, pair[0])) {
			// Check they're talking about the same subject
			aSubj := extractSubject(a)
			bSubj := extractSubject(b)
			if aSubj != "" && aSubj == bSubj {
				return true
			}
		}
	}

	// Pattern 3: Direct negation — same predicate with "not"
	aCleaned := strings.ReplaceAll(a, " not ", " ")
	aCleaned = strings.ReplaceAll(aCleaned, " no ", " ")
	bCleaned := strings.ReplaceAll(b, " not ", " ")
	bCleaned = strings.ReplaceAll(bCleaned, " no ", " ")
	// If removing negation makes them very similar, they likely contradict
	if aCleaned != a && strings.TrimSpace(aCleaned) == strings.TrimSpace(b) {
		return true
	}
	if bCleaned != b && strings.TrimSpace(bCleaned) == strings.TrimSpace(a) {
		return true
	}

	return false
}

// extractSubject gets the first noun-like word from a sentence.
func extractSubject(sent string) string {
	words := strings.Fields(sent)
	// Skip determiners
	skip := map[string]bool{"the": true, "a": true, "an": true, "this": true, "that": true}
	for _, w := range words {
		clean := strings.Trim(w, ".,;:!?()")
		if clean == "" {
			continue
		}
		if skip[clean] {
			continue
		}
		return clean
	}
	return ""
}

// scoreSpecificitySignal penalizes vague language. 1.0 = highly specific.
func scoreSpecificitySignal(candidate string) float64 {
	lower := strings.ToLower(candidate)
	words := strings.Fields(lower)
	if len(words) == 0 {
		return 0.5
	}

	vagueTerms := []string{
		"things", "stuff", "very", "really", "basically",
		"a lot", "kind of", "sort of", "pretty much", "quite",
		"somewhat", "various", "certain", "many", "some",
	}

	vagueCount := 0
	for _, term := range vagueTerms {
		if strings.Contains(lower, term) {
			vagueCount++
		}
	}

	// Also check for non-specific pronouns overuse
	pronouns := 0
	for _, w := range words {
		clean := strings.Trim(w, ".,;:!?()")
		switch clean {
		case "it", "they", "them", "this", "that", "these", "those":
			pronouns++
		}
	}

	pronPenalty := 0.0
	if len(words) > 0 {
		pronRatio := float64(pronouns) / float64(len(words))
		if pronRatio > 0.15 {
			pronPenalty = (pronRatio - 0.15) * 2.0
		}
	}

	vaguePenalty := float64(vagueCount) * 0.08
	totalPenalty := vaguePenalty + pronPenalty
	score := 1.0 - totalPenalty
	if score < 0 {
		return 0
	}
	if score > 1.0 {
		return 1.0
	}
	return score
}

// scoreRepetitionSignal measures sentence-level repetition. 1.0 = no repetition.
func scoreRepetitionSignal(candidate string) float64 {
	sentences := splitSentences(candidate)
	if len(sentences) < 2 {
		return 1.0
	}

	// Build trigrams for each sentence
	type trigramSet map[string]bool
	sentTrigrams := make([]trigramSet, len(sentences))
	for i, sent := range sentences {
		words := strings.Fields(strings.ToLower(strings.TrimSpace(sent)))
		tg := make(trigramSet)
		for j := 0; j+2 < len(words); j++ {
			gram := words[j] + " " + words[j+1] + " " + words[j+2]
			tg[gram] = true
		}
		sentTrigrams[i] = tg
	}

	// Check pairwise trigram overlap between nearby sentences
	overlapCount := 0
	totalPairs := 0
	for i := 0; i < len(sentTrigrams); i++ {
		// Only check within a window of 3 sentences
		end := i + 4
		if end > len(sentTrigrams) {
			end = len(sentTrigrams)
		}
		for j := i + 1; j < end; j++ {
			if len(sentTrigrams[i]) == 0 || len(sentTrigrams[j]) == 0 {
				continue
			}
			totalPairs++
			overlap := 0
			for gram := range sentTrigrams[i] {
				if sentTrigrams[j][gram] {
					overlap++
				}
			}
			minSize := len(sentTrigrams[i])
			if len(sentTrigrams[j]) < minSize {
				minSize = len(sentTrigrams[j])
			}
			if minSize > 0 && float64(overlap)/float64(minSize) > 0.4 {
				overlapCount++
			}
		}
	}

	if totalPairs == 0 {
		return 1.0
	}

	// Also check for exact duplicate sentences
	seen := make(map[string]int)
	dupes := 0
	for _, sent := range sentences {
		norm := strings.ToLower(strings.TrimSpace(sent))
		if norm == "" {
			continue
		}
		seen[norm]++
		if seen[norm] > 1 {
			dupes++
		}
	}

	overlapPenalty := float64(overlapCount) * 0.25
	dupePenalty := float64(dupes) * 0.35
	score := 1.0 - overlapPenalty - dupePenalty
	if score < 0 {
		return 0
	}
	return score
}

// scoreUserGoalFit measures how well the response matches the user's query.
func scoreUserGoalFit(candidate, query string) float64 {
	if query == "" {
		return 0.5
	}

	queryLower := strings.ToLower(query)
	candLower := strings.ToLower(candidate)

	// Extract content keywords from query
	queryWords := rerankerContentWords(query)
	if len(queryWords) == 0 {
		return 0.5
	}

	// Keyword overlap score
	hits := 0
	for _, w := range queryWords {
		if strings.Contains(candLower, w) {
			hits++
		}
	}
	keywordScore := float64(hits) / float64(len(queryWords))

	// Format match: check if the response format matches what the query implies
	formatScore := 1.0
	if strings.Contains(queryLower, "list") || strings.Contains(queryLower, "steps") {
		// User wants a list — check for numbered items or bullet-like structure
		if !strings.Contains(candidate, "1.") && !strings.Contains(candidate, "- ") && !strings.Contains(candidate, "First") {
			formatScore = 0.6
		}
	}
	if strings.Contains(queryLower, "compare") || strings.Contains(queryLower, " vs ") {
		// User wants comparison — check for both items mentioned
		parts := extractComparisonItems(query)
		if len(parts) == 2 {
			hasA := strings.Contains(candLower, strings.ToLower(parts[0]))
			hasB := strings.Contains(candLower, strings.ToLower(parts[1]))
			if !hasA || !hasB {
				formatScore = 0.5
			}
		}
	}
	if strings.Contains(queryLower, "explain") || strings.Contains(queryLower, "how") || strings.Contains(queryLower, "why") {
		// Explanation should have reasonable length
		words := strings.Fields(candidate)
		if len(words) < 15 {
			formatScore = 0.6
		}
	}

	// Weighted combination: keywords matter more than format
	return keywordScore*0.7 + formatScore*0.3
}

// rerankerContentWords extracts meaningful words from text, filtering out stop words.
func rerankerContentWords(text string) []string {
	stopWords := map[string]bool{
		"the": true, "a": true, "an": true, "is": true, "are": true,
		"was": true, "were": true, "be": true, "been": true, "being": true,
		"have": true, "has": true, "had": true, "do": true, "does": true,
		"did": true, "will": true, "would": true, "shall": true, "should": true,
		"may": true, "might": true, "must": true, "can": true, "could": true,
		"i": true, "you": true, "he": true, "she": true, "it": true,
		"we": true, "they": true, "me": true, "him": true, "her": true,
		"us": true, "them": true, "my": true, "your": true, "his": true,
		"its": true, "our": true, "their": true, "this": true, "that": true,
		"these": true, "those": true, "of": true, "to": true, "in": true,
		"for": true, "with": true, "on": true, "at": true, "by": true,
		"from": true, "and": true, "or": true, "but": true, "not": true,
		"what": true, "which": true, "who": true, "whom": true, "when": true,
		"where": true, "how": true, "why": true, "if": true, "then": true,
		"so": true, "than": true, "about": true, "up": true, "out": true,
		"no": true, "just": true, "also": true, "more": true, "some": true,
		"any": true, "each": true, "all": true, "both": true, "few": true,
		"most": true, "other": true, "into": true, "over": true, "after": true,
		"before": true, "between": true, "such": true, "through": true,
		"tell": true, "explain": true, "compare": true,
	}

	words := strings.Fields(strings.ToLower(text))
	var content []string
	for _, w := range words {
		clean := strings.Trim(w, ".,;:!?()[]{}\"'")
		if len(clean) < 3 {
			continue
		}
		if stopWords[clean] {
			continue
		}
		content = append(content, clean)
	}
	return content
}

// extractComparisonItems tries to extract two items being compared from a query.
func extractComparisonItems(query string) []string {
	lower := strings.ToLower(query)

	// "X vs Y", "X versus Y"
	for _, sep := range []string{" vs ", " versus ", " vs. "} {
		if idx := strings.Index(lower, sep); idx >= 0 {
			a := strings.TrimSpace(query[:idx])
			b := strings.TrimSpace(query[idx+len(sep):])
			// Take last word(s) of a, first word(s) of b
			aWords := strings.Fields(a)
			bWords := strings.Fields(b)
			if len(aWords) > 0 && len(bWords) > 0 {
				itemA := aWords[len(aWords)-1]
				itemB := bWords[0]
				return []string{itemA, strings.Trim(itemB, ".,;:!?()")}
			}
		}
	}

	// "difference between X and Y", "compare X and Y"
	if idx := strings.Index(lower, " between "); idx >= 0 {
		rest := query[idx+9:]
		if andIdx := strings.Index(strings.ToLower(rest), " and "); andIdx >= 0 {
			a := strings.TrimSpace(rest[:andIdx])
			b := strings.TrimSpace(rest[andIdx+5:])
			return []string{
				strings.Trim(a, ".,;:!?()"),
				strings.Trim(b, ".,;:!?()"),
			}
		}
	}

	// "compare X and Y" / "compare X with Y"
	if idx := strings.Index(lower, "compare "); idx >= 0 {
		rest := query[idx+8:]
		for _, sep := range []string{" and ", " with ", " to "} {
			if sepIdx := strings.Index(strings.ToLower(rest), sep); sepIdx >= 0 {
				a := strings.TrimSpace(rest[:sepIdx])
				b := strings.TrimSpace(rest[sepIdx+len(sep):])
				return []string{
					strings.Trim(a, ".,;:!?()"),
					strings.Trim(b, ".,;:!?()"),
				}
			}
		}
	}

	return nil
}
