package cognitive

import (
	"fmt"
	"strings"
)

// BuildPlanCandidates produces multiple deterministic drafts from one plan.
// The first candidate is the existing baseline text to preserve behavior.
func (te *ThinkingEngine) BuildPlanCandidates(plan *ContentPlan, frame *Frame, baseline string) []string {
	baseline = strings.TrimSpace(baseline)
	if plan == nil {
		if baseline == "" {
			return nil
		}
		return []string{baseline}
	}

	var out []string
	if baseline != "" {
		out = append(out, baseline)
	}

	// Candidate A: thesis + ordered evidence + concise synthesis.
	a := te.realizePlanVariant(plan, frame, false, "overall")
	if a != "" {
		out = append(out, a)
	}

	// Candidate B: strongest-first ordering with explicit connectors.
	b := te.realizePlanVariant(plan, frame, true, "therefore")
	if b != "" {
		out = append(out, b)
	}

	// Candidate C: compact synthesis for concise readability.
	c := te.realizeCompactPlan(plan)
	if c != "" {
		out = append(out, c)
	}

	// Candidate D: corpus-backed realization from human-written exemplars.
	d := te.realizeCorpusBackedPlan(plan, frame)
	if d != "" {
		out = append(out, d)
	}

	return dedupeCandidates(out)
}

func (te *ThinkingEngine) realizeCorpusBackedPlan(plan *ContentPlan, frame *Frame) string {
	if plan == nil || te.composer == nil || te.composer.SentenceCorpus == nil {
		return ""
	}

	var parts []string
	if plan.Thesis != "" {
		parts = append(parts, strings.TrimSpace(plan.Thesis))
	}

	for _, c := range plan.Claims {
		if len(c.Evidence) == 0 {
			if c.Text != "" {
				parts = append(parts, c.Text)
			}
			continue
		}

		ev := c.Evidence[0]
		s := te.composer.SentenceCorpus.RetrieveVaried(ev.Subject, ev.Relation, ev.Object)
		if s == "" {
			s = c.Text
		}
		if s != "" {
			parts = append(parts, strings.TrimSpace(s))
		}
	}

	if plan.Counterpoint != "" {
		parts = append(parts, plan.Counterpoint)
	}

	if len(parts) == 0 {
		return ""
	}

	if frame != nil && (frame.Name == "summary" || frame.Name == "brainstorm" || frame.Name == "plan") {
		if len(parts) == 1 {
			return parts[0]
		}
		return parts[0] + "\n\n" + strings.Join(parts[1:], "\n")
	}

	return strings.Join(parts, " ")
}

// SelectBestCandidate reranks deterministic candidates with quality signals.
func (te *ThinkingEngine) SelectBestCandidate(plan *ContentPlan, candidates []string) string {
	if len(candidates) == 0 {
		return ""
	}
	best := candidates[0]
	bestScore := te.ScoreCandidate(plan, best).Total

	for i := 1; i < len(candidates); i++ {
		s := te.ScoreCandidate(plan, candidates[i]).Total
		if s > bestScore {
			best = candidates[i]
			bestScore = s
		}
	}

	return best
}

func (te *ThinkingEngine) realizePlanVariant(plan *ContentPlan, frame *Frame, strongestFirst bool, conclusionCue string) string {
	if plan == nil {
		return ""
	}

	var parts []string
	if plan.Thesis != "" {
		parts = append(parts, strings.TrimSpace(plan.Thesis))
	}

	claims := make([]PlanClaim, len(plan.Claims))
	copy(claims, plan.Claims)
	if strongestFirst && len(claims) > 1 {
		for i, j := 0, len(claims)-1; i < j; i, j = i+1, j-1 {
			claims[i], claims[j] = claims[j], claims[i]
		}
	}

	if frame != nil && (frame.Name == "brainstorm" || frame.Name == "summary" || frame.Name == "plan") {
		if len(claims) > 0 {
			var lines []string
			for i, c := range claims {
				if c.Text == "" {
					continue
				}
				lines = append(lines, toOrdinal(i+1)+" "+strings.TrimSpace(c.Text))
			}
			if len(lines) > 0 {
				parts = append(parts, strings.Join(lines, "\n"))
			}
		}
	} else {
		for idx, c := range claims {
			if c.Text == "" {
				continue
			}
			connector := "Additionally"
			if idx == 0 {
				connector = "First"
			} else if idx == len(claims)-1 {
				connector = "Finally"
			}
			parts = append(parts, connector+", "+lowerFirstSentence(c.Text))
		}
	}

	if plan.Counterpoint != "" {
		parts = append(parts, plan.Counterpoint)
	}

	if len(claims) > 0 {
		parts = append(parts, capitalizeFirst(conclusionCue)+", the strongest interpretation is to focus on the highest-impact facts first.")
	}

	if frame != nil && frame.Name == "email" {
		return strings.Join(parts, "\n\n")
	}

	return strings.Join(parts, " ")
}

func (te *ThinkingEngine) realizeCompactPlan(plan *ContentPlan) string {
	if plan == nil {
		return ""
	}

	var bits []string
	if plan.Thesis != "" {
		bits = append(bits, strings.TrimSpace(plan.Thesis))
	}

	if len(plan.Claims) > 0 {
		max := len(plan.Claims)
		if max > 3 {
			max = 3
		}
		for i := 0; i < max; i++ {
			if plan.Claims[i].Text != "" {
				bits = append(bits, plan.Claims[i].Text)
			}
		}
	}

	if len(bits) == 0 {
		return ""
	}

	return strings.Join(bits, " ")
}

func dedupeCandidates(candidates []string) []string {
	seen := make(map[string]bool)
	var out []string
	for _, c := range candidates {
		clean := strings.TrimSpace(c)
		if clean == "" {
			continue
		}
		norm := strings.ToLower(clean)
		if seen[norm] {
			continue
		}
		seen[norm] = true
		out = append(out, clean)
	}
	return out
}

func lowerFirstSentence(s string) string {
	s = strings.TrimSpace(s)
	if s == "" {
		return ""
	}
	r := []rune(s)
	r[0] = []rune(strings.ToLower(string(r[0])))[0]
	return string(r)
}

func toOrdinal(n int) string {
	return fmt.Sprintf("%d.", n)
}