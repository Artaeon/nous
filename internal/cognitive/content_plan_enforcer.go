package cognitive

import (
	"fmt"
	"strings"
)

// PlanTemplate defines the required structure for different task types.
type PlanTemplate struct {
	TaskType           ThinkTask
	RequiredParts      []string // e.g. ["thesis", "claims", "evidence", "uncertainty", "recap"]
	MinClaims          int
	MaxClaims          int
	RequireEvidence    bool
	RequireUncertainty bool
}

// EnforcedPlan extends ContentPlan with mandatory structure.
type EnforcedPlan struct {
	*ContentPlan
	Template    *PlanTemplate
	Thesis      string
	Claims      []EnforcedClaim
	Evidence    []string
	Uncertainty string   // explicit statement of what's uncertain/unknown
	Recap       string   // closing synthesis
	Complete    bool     // all required parts present
	Missing     []string // which required parts are missing
}

// EnforcedClaim extends PlanClaim with evidence tracking.
type EnforcedClaim struct {
	PlanClaim
	EvidenceText string
	Supported    bool // has backing evidence
}

// GetPlanTemplate returns the plan template for a given task type.
func GetPlanTemplate(task ThinkTask) *PlanTemplate {
	switch task {
	case TaskTeach:
		// Explain: thesis + concepts + examples + analogies + recap
		return &PlanTemplate{
			TaskType:           task,
			RequiredParts:      []string{"thesis", "claims", "evidence", "uncertainty", "recap"},
			MinClaims:          3,
			MaxClaims:          5,
			RequireEvidence:    true,
			RequireUncertainty: true,
		}
	case TaskCompare:
		// Compare: thesis + per-item claims + comparison axes + tradeoffs + uncertainty + recap
		return &PlanTemplate{
			TaskType:           task,
			RequiredParts:      []string{"thesis", "claims", "evidence", "uncertainty", "recap"},
			MinClaims:          2,
			MaxClaims:          6,
			RequireEvidence:    true,
			RequireUncertainty: true,
		}
	case TaskAnalyze:
		// Analyze: thesis + factors + evidence + implications + uncertainty
		return &PlanTemplate{
			TaskType:           task,
			RequiredParts:      []string{"thesis", "claims", "evidence", "uncertainty"},
			MinClaims:          2,
			MaxClaims:          5,
			RequireEvidence:    true,
			RequireUncertainty: true,
		}
	case TaskPlan:
		// Plan: goal + steps + dependencies + risks + recap
		return &PlanTemplate{
			TaskType:           task,
			RequiredParts:      []string{"thesis", "claims", "recap"},
			MinClaims:          2,
			MaxClaims:          6,
			RequireEvidence:    false,
			RequireUncertainty: false,
		}
	case TaskDebate:
		return &PlanTemplate{
			TaskType:           task,
			RequiredParts:      []string{"thesis", "claims", "evidence", "uncertainty"},
			MinClaims:          2,
			MaxClaims:          5,
			RequireEvidence:    true,
			RequireUncertainty: true,
		}
	case TaskAdvise:
		return &PlanTemplate{
			TaskType:           task,
			RequiredParts:      []string{"thesis", "claims", "recap"},
			MinClaims:          2,
			MaxClaims:          4,
			RequireEvidence:    false,
			RequireUncertainty: false,
		}
	default:
		// Default: thesis + claims (2+)
		return &PlanTemplate{
			TaskType:           task,
			RequiredParts:      []string{"thesis", "claims"},
			MinClaims:          2,
			MaxClaims:          5,
			RequireEvidence:    false,
			RequireUncertainty: false,
		}
	}
}

// EnforceContentPlan validates and fills required structure on a ContentPlan.
func EnforceContentPlan(plan *ContentPlan, template *PlanTemplate, topic string) *EnforcedPlan {
	if plan == nil {
		plan = &ContentPlan{Topic: topic}
	}
	if template == nil {
		template = GetPlanTemplate(TaskConverse)
	}

	ep := &EnforcedPlan{
		ContentPlan: plan,
		Template:    template,
	}

	// Fill thesis
	if plan.Thesis != "" {
		ep.Thesis = plan.Thesis
	} else if topic != "" {
		ep.Thesis = capitalizeFirst(topic) + "."
	}

	// Build enforced claims with evidence tracking
	for _, claim := range plan.Claims {
		ec := EnforcedClaim{
			PlanClaim: claim,
		}
		if len(claim.Evidence) > 0 {
			ev := claim.Evidence[0]
			ec.EvidenceText = formatEvidence(ev)
			ec.Supported = true
		}
		ep.Claims = append(ep.Claims, ec)
	}

	// Pad claims to minimum if needed
	for len(ep.Claims) < template.MinClaims && topic != "" {
		filler := EnforcedClaim{
			PlanClaim: PlanClaim{
				Text:     generateFillerClaim(topic, len(ep.Claims)),
				Priority: 10,
			},
			Supported: false,
		}
		ep.Claims = append(ep.Claims, filler)
	}

	// Trim claims to maximum
	if len(ep.Claims) > template.MaxClaims {
		ep.Claims = ep.Claims[:template.MaxClaims]
	}

	// Collect evidence strings
	for _, c := range ep.Claims {
		if c.EvidenceText != "" {
			ep.Evidence = append(ep.Evidence, c.EvidenceText)
		}
	}

	// Generate uncertainty if required
	if template.RequireUncertainty {
		ep.Uncertainty = GenerateUncertaintyStatement(topic, plan.Claims)
	}

	// Generate recap if required
	hasRecap := false
	for _, p := range template.RequiredParts {
		if p == "recap" {
			hasRecap = true
			break
		}
	}
	if hasRecap {
		ep.Recap = GenerateRecap(ep.Thesis, plan.Claims)
	}

	// Check completeness
	ep.Missing = checkMissing(ep, template)
	ep.Complete = len(ep.Missing) == 0

	return ep
}

// GenerateUncertaintyStatement generates an explicit statement about what's
// uncertain or unknown regarding a topic.
func GenerateUncertaintyStatement(topic string, claims []PlanClaim) string {
	if topic == "" {
		return "There may be aspects of this topic that remain uncertain or debated."
	}

	topic = strings.TrimSpace(topic)

	// Count how much evidence we have
	evidenceCount := 0
	for _, c := range claims {
		if len(c.Evidence) > 0 {
			evidenceCount++
		}
	}

	if evidenceCount == 0 {
		return fmt.Sprintf("It is worth noting that the available information on %s may be incomplete. "+
			"Some aspects could be more nuanced than presented here.", topic)
	}

	if evidenceCount < len(claims) {
		unsupported := len(claims) - evidenceCount
		return fmt.Sprintf("While some claims about %s are well-supported, "+
			"%d of the points presented have limited direct evidence. "+
			"Further investigation may reveal additional nuances.", topic, unsupported)
	}

	return fmt.Sprintf("Although the main facts about %s are well-established, "+
		"interpretations and real-world applications may vary depending on context.", topic)
}

// GenerateRecap generates a closing synthesis from the thesis and claims.
func GenerateRecap(thesis string, claims []PlanClaim) string {
	if thesis == "" && len(claims) == 0 {
		return "In summary, these are the key points to consider."
	}

	var parts []string

	if len(claims) > 0 {
		// Extract key subjects from claims for the recap
		subjects := make([]string, 0, 3)
		seen := make(map[string]bool)
		for _, c := range claims {
			words := rerankerContentWords(c.Text)
			for _, w := range words {
				if !seen[w] && len(subjects) < 3 {
					seen[w] = true
					subjects = append(subjects, w)
				}
			}
		}

		if len(subjects) > 0 {
			parts = append(parts, "In summary")
			if thesis != "" {
				// Use a shortened version of the thesis
				thesisShort := shortenSentence(thesis)
				parts[0] = fmt.Sprintf("In summary, %s", lowerFirstSentence(thesisShort))
			}
			parts = append(parts, fmt.Sprintf(
				"The key aspects to understand involve %s.", strings.Join(subjects, ", ")))
		} else {
			parts = append(parts, "Overall, these points provide a foundation for understanding this topic.")
		}
	} else if thesis != "" {
		parts = append(parts, fmt.Sprintf("In summary, %s", lowerFirstSentence(thesis)))
	}

	result := strings.Join(parts, " ")
	if !hasSentenceEnding(result) {
		result += "."
	}
	return result
}

// checkMissing determines which required parts are absent from the plan.
func checkMissing(ep *EnforcedPlan, template *PlanTemplate) []string {
	var missing []string

	for _, part := range template.RequiredParts {
		switch part {
		case "thesis":
			if ep.Thesis == "" {
				missing = append(missing, "thesis")
			}
		case "claims":
			if len(ep.Claims) < template.MinClaims {
				missing = append(missing, fmt.Sprintf("claims (need %d, have %d)", template.MinClaims, len(ep.Claims)))
			}
		case "evidence":
			if template.RequireEvidence && len(ep.Evidence) == 0 {
				missing = append(missing, "evidence")
			}
		case "uncertainty":
			if template.RequireUncertainty && ep.Uncertainty == "" {
				missing = append(missing, "uncertainty")
			}
		case "recap":
			if ep.Recap == "" {
				missing = append(missing, "recap")
			}
		}
	}

	return missing
}

// formatEvidence converts an edgeFact to a human-readable evidence string.
func formatEvidence(ev edgeFact) string {
	s := strings.TrimSpace(ev.Subject)
	o := strings.TrimSpace(ev.Object)
	if s == "" || o == "" {
		return ""
	}

	switch ev.Relation {
	case RelIsA:
		return fmt.Sprintf("%s is categorized as %s", s, o)
	case RelUsedFor:
		return fmt.Sprintf("%s is commonly used for %s", s, o)
	case RelHas:
		return fmt.Sprintf("%s possesses %s", s, o)
	case RelPartOf:
		return fmt.Sprintf("%s is a component of %s", s, o)
	case RelCauses:
		return fmt.Sprintf("%s leads to %s", s, o)
	case RelLocatedIn:
		return fmt.Sprintf("%s is based in %s", s, o)
	case RelFoundedIn:
		return fmt.Sprintf("%s was established in %s", s, o)
	default:
		return fmt.Sprintf("%s relates to %s", s, o)
	}
}

// generateFillerClaim produces a minimal placeholder claim when the plan is
// thin. Each pattern references the topic directly without generic filler.
func generateFillerClaim(topic string, index int) string {
	t := capitalizeFirst(topic)
	patterns := []string{
		t + " has several distinctive properties.",
		t + " serves a specific role in its domain.",
		t + " has practical applications.",
		t + " has a notable history.",
		t + " connects to related fields.",
	}
	idx := index % len(patterns)
	return patterns[idx]
}

// shortenSentence truncates a sentence at the first clause boundary if it's long.
func shortenSentence(s string) string {
	s = strings.TrimSpace(s)
	if len(s) < 60 {
		return s
	}
	// Try to cut at a comma or semicolon
	for _, sep := range []string{", ", "; "} {
		if idx := strings.Index(s, sep); idx > 20 && idx < len(s)-10 {
			result := s[:idx]
			if !hasSentenceEnding(result) {
				result += "."
			}
			return result
		}
	}
	return s
}
