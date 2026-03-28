package cognitive

import (
	"fmt"
	"sort"
	"strings"
)

// PlanClaim is one ranked claim in a deterministic response plan.
type PlanClaim struct {
	Text     string
	Evidence []edgeFact
	Priority int
}

// ContentPlan is a semantic skeleton used before surface realization.
type ContentPlan struct {
	Topic        string
	Thesis       string
	Claims       []PlanClaim
	Counterpoint string
}

// BuildContentPlan creates a deterministic thesis + ranked claim plan.
// This gives the generator a stable semantic backbone before writing prose.
func (te *ThinkingEngine) BuildContentPlan(query string, task ThinkTask, params *TaskParams) *ContentPlan {
	topic := strings.TrimSpace(params.Topic)
	if topic == "" {
		topic = strings.Join(params.Keywords, " ")
	}
	if topic == "" {
		topic = extractTopicFromQuery(query)
	}
	if topic == "" {
		topic = "this topic"
	}

	plan := &ContentPlan{Topic: topic}
	plan.Thesis = te.planThesis(topic)

	facts := te.gatherTopicFacts(topic)
	if len(facts) == 0 {
		return plan
	}

	claims := make([]PlanClaim, 0, 6)
	for _, f := range facts {
		claim := PlanClaim{
			Text:     te.factToClaim(f),
			Evidence: []edgeFact{f},
			Priority: te.relationPriority(f.Relation),
		}
		if claim.Text != "" {
			claims = append(claims, claim)
		}
	}

	// Stable deterministic ranking: priority desc, then lexical order.
	sort.SliceStable(claims, func(i, j int) bool {
		if claims[i].Priority == claims[j].Priority {
			return claims[i].Text < claims[j].Text
		}
		return claims[i].Priority > claims[j].Priority
	})

	if len(claims) > 5 {
		claims = claims[:5]
	}

	// Deduplicate near-identical claims.
	seen := make(map[string]bool)
	for _, c := range claims {
		norm := strings.ToLower(strings.TrimSpace(c.Text))
		if norm == "" || seen[norm] {
			continue
		}
		seen[norm] = true
		plan.Claims = append(plan.Claims, c)
	}

	if task == TaskDebate && len(plan.Claims) > 0 {
		plan.Counterpoint = fmt.Sprintf("A fair criticism is that %s can involve trade-offs depending on context.", topic)
	}

	return plan
}

func (te *ThinkingEngine) planThesis(topic string) string {
	if te.graph != nil {
		if desc := strings.TrimSpace(te.graph.LookupDescription(topic)); len(desc) > 20 {
			return desc
		}
	}

	facts := te.gatherTopicFacts(topic)
	for _, f := range facts {
		if f.Relation == RelIsA {
			return fmt.Sprintf("%s is a %s.", capitalizeFirst(topic), strings.TrimSpace(f.Object))
		}
	}

	// Fallback: realize whatever facts we have directly instead of generic filler.
	if len(facts) > 0 {
		eng := NewNLGEngine()
		if text := eng.Realize(topic, facts); text != "" {
			return text
		}
	}
	return capitalizeFirst(topic) + "."
}

func (te *ThinkingEngine) factToClaim(f edgeFact) string {
	subject := strings.TrimSpace(f.Subject)
	object := strings.TrimSpace(f.Object)
	if subject == "" || object == "" {
		return ""
	}

	subject = capitalizeFirst(subject)

	switch f.Relation {
	case RelIsA:
		return fmt.Sprintf("%s is a %s.", subject, object)
	case RelUsedFor:
		return fmt.Sprintf("%s is used for %s.", subject, object)
	case RelHas:
		return fmt.Sprintf("%s has %s.", subject, object)
	case RelPartOf:
		return fmt.Sprintf("%s is part of %s.", subject, object)
	case RelRelatedTo:
		return fmt.Sprintf("%s is related to %s.", subject, object)
	case RelLocatedIn:
		return fmt.Sprintf("%s is located in %s.", subject, object)
	case RelFoundedIn:
		return fmt.Sprintf("%s was founded in %s.", subject, object)
	default:
		return fmt.Sprintf("%s has a notable relation to %s.", subject, object)
	}
}

func (te *ThinkingEngine) relationPriority(rel RelType) int {
	switch rel {
	case RelIsA:
		return 100
	case RelUsedFor:
		return 90
	case RelHas:
		return 80
	case RelPartOf:
		return 70
	case RelRelatedTo:
		return 60
	case RelLocatedIn, RelFoundedIn:
		return 50
	default:
		return 30
	}
}