package cognitive

import "strings"

// NLGScore captures deterministic quality signals for candidate responses.
type NLGScore struct {
	Coverage   float64
	Coherence  float64
	NonRepeat  float64
	Readability float64
	EntityFocus float64
	Consistency float64
	Total      float64
}

// ScoreCandidate evaluates a response candidate against a semantic plan.
func (te *ThinkingEngine) ScoreCandidate(plan *ContentPlan, text string) NLGScore {
	text = strings.TrimSpace(text)
	if text == "" {
		return NLGScore{}
	}

	coverage := scoreCoverage(plan, text)
	coherence := scoreCoherence(text)
	nonRepeat := scoreNonRepeat(text)
	readability := scoreReadability(text)
	entityFocus := scoreEntityFocus(plan, text)
	consistency := scoreConsistency(plan, text)

	// Weighted aggregate tuned for factual assistant quality.
	total := coverage*0.36 + coherence*0.16 + nonRepeat*0.14 + readability*0.10 + entityFocus*0.12 + consistency*0.12
	// Hard contradiction penalty: if consistency is low, sharply reduce final score.
	if consistency < 0.90 {
		total *= consistency
	}
	if total > 1.0 {
		total = 1.0
	}

	return NLGScore{
		Coverage:   coverage,
		Coherence:  coherence,
		NonRepeat:  nonRepeat,
		Readability: readability,
		EntityFocus: entityFocus,
		Consistency: consistency,
		Total:      total,
	}
}

func scoreCoverage(plan *ContentPlan, text string) float64 {
	if plan == nil {
		return 0.5
	}
	if len(plan.Claims) == 0 {
		if plan.Thesis == "" {
			return 0.5
		}
		if containsLoose(text, plan.Topic) {
			return 0.8
		}
		return 0.4
	}

	hits := 0
	for _, claim := range plan.Claims {
		if claim.Text == "" {
			continue
		}
		if containsLoose(text, claim.Text) || containsLoose(text, keySpan(claim.Text)) {
			hits++
		}
	}

	return float64(hits) / float64(len(plan.Claims))
}

func scoreCoherence(text string) float64 {
	lower := strings.ToLower(text)
	connectors := []string{
		"because", "however", "therefore", "for example", "in contrast",
		"in short", "first", "second", "finally", "overall",
	}

	hits := 0
	for _, c := range connectors {
		if strings.Contains(lower, c) {
			hits++
		}
	}

	// Mild baseline score plus bonus for rhetorical glue.
	score := 0.55 + 0.08*float64(hits)
	if score > 1.0 {
		return 1.0
	}
	return score
}

func scoreNonRepeat(text string) float64 {
	words := strings.Fields(strings.ToLower(text))
	if len(words) < 8 {
		return 0.7
	}

	counts := make(map[string]int)
	for _, w := range words {
		w = strings.Trim(w, ".,;:!?()[]{}\"'")
		if len(w) < 3 {
			continue
		}
		counts[w]++
	}

	repeated := 0
	for _, n := range counts {
		if n > 2 {
			repeated += n - 2
		}
	}

	penalty := float64(repeated) / float64(len(words))
	score := 1.0 - penalty*3.0
	if score < 0 {
		return 0
	}
	if score > 1.0 {
		return 1.0
	}
	return score
}

func scoreReadability(text string) float64 {
	parts := strings.FieldsFunc(text, func(r rune) bool {
		return r == '.' || r == '!' || r == '?'
	})
	if len(parts) == 0 {
		return 0.5
	}

	avgWords := 0.0
	for _, p := range parts {
		w := strings.Fields(strings.TrimSpace(p))
		avgWords += float64(len(w))
	}
	avgWords /= float64(len(parts))

	// Favor sentence length in the 10-24 word range.
	if avgWords >= 10 && avgWords <= 24 {
		return 1.0
	}
	if avgWords >= 7 && avgWords <= 30 {
		return 0.8
	}
	if avgWords >= 5 && avgWords <= 36 {
		return 0.6
	}
	return 0.4
}

func scoreEntityFocus(plan *ContentPlan, text string) float64 {
	if plan == nil || strings.TrimSpace(plan.Topic) == "" {
		return 0.7
	}

	topic := strings.ToLower(strings.TrimSpace(plan.Topic))
	parts := strings.FieldsFunc(text, func(r rune) bool {
		return r == '.' || r == '!' || r == '?'
	})
	if len(parts) == 0 {
		if strings.Contains(strings.ToLower(text), topic) {
			return 1.0
		}
		return 0.4
	}

	hits := 0
	for _, p := range parts {
		pl := strings.ToLower(strings.TrimSpace(p))
		if pl == "" {
			continue
		}
		if strings.Contains(pl, topic) || containsAnySlice(pl, []string{" it ", " this ", " they ", " these "}) {
			hits++
		}
	}

	score := float64(hits) / float64(len(parts))
	if score < 0.35 {
		return 0.35
	}
	return score
}

func scoreConsistency(plan *ContentPlan, text string) float64 {
	if strings.TrimSpace(text) == "" {
		return 0
	}
	lower := " " + strings.ToLower(text) + " "
	if plan == nil || len(plan.Claims) == 0 {
		return 0.85
	}

	penalty := 0.0
	for _, c := range plan.Claims {
		k := strings.ToLower(strings.TrimSpace(keySpan(c.Text)))
		if k == "" || len(k) < 4 {
			continue
		}
		hasPositive := strings.Contains(lower, " "+k+" ")
		hasNegative := strings.Contains(lower, " not "+k+" ") || strings.Contains(lower, " no "+k+" ")
		if hasPositive && hasNegative {
			penalty += 0.35
		}
	}

	score := 1.0 - penalty
	if score < 0 {
		return 0
	}
	return score
}

func containsLoose(text, probe string) bool {
	text = strings.ToLower(strings.TrimSpace(text))
	probe = strings.ToLower(strings.TrimSpace(probe))
	if text == "" || probe == "" {
		return false
	}
	return strings.Contains(text, probe)
}

func containsAnySlice(text string, probes []string) bool {
	for _, p := range probes {
		if strings.Contains(text, p) {
			return true
		}
	}
	return false
}

func keySpan(claim string) string {
	words := strings.Fields(strings.ToLower(claim))
	if len(words) <= 2 {
		return claim
	}
	// Drop one leading and one trailing grammatical token to improve fuzzy coverage.
	start := 0
	end := len(words)
	if isGrammarWord(words[start]) {
		start++
	}
	if end-start > 1 && isGrammarWord(words[end-1]) {
		end--
	}
	if end <= start {
		return claim
	}
	return strings.Join(words[start:end], " ")
}

func isGrammarWord(w string) bool {
	switch w {
	case "the", "a", "an", "is", "are", "was", "were", "of", "to", "in", "for", "with", "and":
		return true
	default:
		return false
	}
}