package cognitive

import (
	"encoding/json"
	"os"
	"sort"
	"sync"
)

// -----------------------------------------------------------------------
// Dialogue Policy Learning
//
// The SelfModel tracks performance per domain but doesn't influence
// routing decisions. This module closes that loop: it learns WHICH
// response strategy works best for WHICH type of query by tracking
// real outcomes.
//
// Strategies: "direct" (answer factually), "socratic" (guide through
// questions), "synthesis" (combine multiple angles), "explain" (teach
// the concept), "compare" (contrast options), "fallback" (hedge).
//
// The policy observes three outcomes:
//   - success:  user continues the conversation (implicit positive signal)
//   - failure:  user corrects the response or abruptly changes topic
//   - neutral:  no clear signal either way
//
// Over time the policy learns, for example, that "science:factual"
// queries do best with "direct", while "career:advice" queries
// respond better to "socratic".
// -----------------------------------------------------------------------

// DialoguePolicy learns which response strategy to use for each query type.
// It tracks outcomes: did the user continue (good), correct (bad), or leave (neutral)?
type DialoguePolicy struct {
	strategies map[string]*StrategyRecord // "domain:strategy" -> record
	mu         sync.RWMutex
}

// StrategyRecord tracks how well a strategy works for a domain.
type StrategyRecord struct {
	Domain     string  `json:"domain"`
	Strategy   string  `json:"strategy"`
	Successes  int     `json:"successes"`
	Failures   int     `json:"failures"`
	Neutrals   int     `json:"neutrals"`
	AvgQuality float64 `json:"avg_quality"`
}

// PolicyDecision recommends which strategy to use.
type PolicyDecision struct {
	Strategy     string   `json:"strategy"`
	Confidence   float64  `json:"confidence"`
	Reason       string   `json:"reason"`
	Alternatives []string `json:"alternatives"`
}

// allStrategies is the full list of known strategies.
var allStrategies = []string{"direct", "socratic", "synthesis", "explain", "compare", "fallback"}

// defaultStrategyMap provides sensible defaults when no data exists.
var defaultStrategyMap = map[string]string{
	"factual":    "direct",
	"coaching":   "socratic",
	"advice":     "socratic",
	"comparison": "compare",
	"teaching":   "explain",
	"creative":   "synthesis",
}

// NewDialoguePolicy creates a new DialoguePolicy with empty statistics.
func NewDialoguePolicy() *DialoguePolicy {
	return &DialoguePolicy{
		strategies: make(map[string]*StrategyRecord),
	}
}

// Recommend looks up the best strategy for this domain+queryType.
//   - If we have data: pick the strategy with highest success rate.
//   - If no data: use defaults (factual->direct, coaching->socratic, unknown->synthesis).
//   - Always include alternatives in case the first choice fails.
func (dp *DialoguePolicy) Recommend(domain, queryType string) *PolicyDecision {
	dp.mu.RLock()
	defer dp.mu.RUnlock()

	// Gather all strategy records for this domain.
	type scored struct {
		strategy    string
		successRate float64
		total       int
	}
	var candidates []scored

	for _, s := range allStrategies {
		key := policyKey(domain, s)
		rec, ok := dp.strategies[key]
		if !ok {
			continue
		}
		total := rec.Successes + rec.Failures + rec.Neutrals
		if total == 0 {
			continue
		}
		rate := float64(rec.Successes) / float64(total)
		candidates = append(candidates, scored{strategy: s, successRate: rate, total: total})
	}

	// No data: use defaults.
	if len(candidates) == 0 {
		return dp.defaultDecision(queryType)
	}

	// Sort by success rate descending, breaking ties by total interactions.
	sort.Slice(candidates, func(i, j int) bool {
		if candidates[i].successRate != candidates[j].successRate {
			return candidates[i].successRate > candidates[j].successRate
		}
		return candidates[i].total > candidates[j].total
	})

	best := candidates[0]

	// Confidence increases with both success rate and sample size.
	// Bayesian-ish: confidence = successRate * min(1, total/10)
	sampleFactor := float64(best.total) / 10.0
	if sampleFactor > 1.0 {
		sampleFactor = 1.0
	}
	confidence := best.successRate * sampleFactor

	// Build alternatives list from other strategies, in order.
	var alts []string
	for i := 1; i < len(candidates); i++ {
		alts = append(alts, candidates[i].strategy)
	}
	// Add any strategies we haven't seen at all as final fallbacks.
	seen := make(map[string]bool)
	for _, c := range candidates {
		seen[c.strategy] = true
	}
	for _, s := range allStrategies {
		if !seen[s] && s != best.strategy {
			alts = append(alts, s)
		}
	}

	reason := "learned"
	if best.total < 5 {
		reason = "preliminary"
	}

	return &PolicyDecision{
		Strategy:     best.strategy,
		Confidence:   clampFloat(confidence, 0, 1),
		Reason:       reason,
		Alternatives: alts,
	}
}

// RecordOutcome records the outcome of using a strategy for a domain.
// outcome must be "success", "failure", or "neutral".
func (dp *DialoguePolicy) RecordOutcome(domain, strategy string, outcome string) {
	dp.mu.Lock()
	defer dp.mu.Unlock()

	key := policyKey(domain, strategy)
	rec, ok := dp.strategies[key]
	if !ok {
		rec = &StrategyRecord{
			Domain:   domain,
			Strategy: strategy,
		}
		dp.strategies[key] = rec
	}

	switch outcome {
	case "success":
		rec.Successes++
	case "failure":
		rec.Failures++
	default:
		rec.Neutrals++
	}

	// Update running average quality (success=1.0, neutral=0.5, failure=0.0).
	total := rec.Successes + rec.Failures + rec.Neutrals
	rec.AvgQuality = (float64(rec.Successes) + 0.5*float64(rec.Neutrals)) / float64(total)
}

// GetStats returns all strategies and their performance for a domain.
// Returns nil if no data exists for the domain.
func (dp *DialoguePolicy) GetStats(domain string) map[string]*StrategyRecord {
	dp.mu.RLock()
	defer dp.mu.RUnlock()

	result := make(map[string]*StrategyRecord)
	for _, s := range allStrategies {
		key := policyKey(domain, s)
		if rec, ok := dp.strategies[key]; ok {
			// Return a copy to avoid race conditions.
			copy := *rec
			result[s] = &copy
		}
	}

	if len(result) == 0 {
		return nil
	}
	return result
}

// dialoguePolicyPersist is the JSON-serializable snapshot.
type dialoguePolicyPersist struct {
	Strategies map[string]*StrategyRecord `json:"strategies"`
}

// Save persists the dialogue policy to a JSON file.
func (dp *DialoguePolicy) Save(path string) error {
	dp.mu.RLock()
	defer dp.mu.RUnlock()

	snap := dialoguePolicyPersist{
		Strategies: dp.strategies,
	}

	data, err := json.MarshalIndent(snap, "", "  ")
	if err != nil {
		return err
	}

	return os.WriteFile(path, data, 0644)
}

// Load reads the dialogue policy from a JSON file.
func (dp *DialoguePolicy) Load(path string) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return err
	}

	var snap dialoguePolicyPersist
	if err := json.Unmarshal(data, &snap); err != nil {
		return err
	}

	dp.mu.Lock()
	defer dp.mu.Unlock()

	if snap.Strategies == nil {
		snap.Strategies = make(map[string]*StrategyRecord)
	}
	dp.strategies = snap.Strategies
	return nil
}

// defaultDecision returns a sensible default policy when no data exists.
func (dp *DialoguePolicy) defaultDecision(queryType string) *PolicyDecision {
	strategy, ok := defaultStrategyMap[queryType]
	if !ok {
		strategy = "synthesis"
	}

	// Build alternatives: all other strategies in a reasonable fallback order.
	var alts []string
	for _, s := range allStrategies {
		if s != strategy {
			alts = append(alts, s)
		}
	}

	return &PolicyDecision{
		Strategy:     strategy,
		Confidence:   0.3,
		Reason:       "default",
		Alternatives: alts,
	}
}

// policyKey builds the map key for a domain+strategy pair.
func policyKey(domain, strategy string) string {
	return domain + ":" + strategy
}
