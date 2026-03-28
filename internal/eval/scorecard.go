package eval

// Metric defines a single measurable dimension within a capability scorecard.
type Metric struct {
	Name      string
	Weight    float64
	Threshold float64 // minimum acceptable score [0,1]
}

// Scorecard describes how a single capability is evaluated.
type Scorecard struct {
	Capability  string
	Description string
	Weight      float64 // relative importance among all capabilities
	Metrics     []Metric
}

// ScorecardResult holds the evaluation outcome for one scorecard.
type ScorecardResult struct {
	Capability string
	Scores     map[string]float64 // metric name -> score
	Pass       bool
	Total      float64
}

// DefaultScorecards returns the 8 configured capability scorecards.
func DefaultScorecards() []Scorecard {
	return []Scorecard{
		{
			Capability:  "IntentRouting",
			Description: "Correctly classifying user intent into the right handler or response type",
			Weight:      1.0,
			Metrics: []Metric{
				{Name: "classification_accuracy", Weight: 0.40, Threshold: 0.85},
				{Name: "confidence_calibration", Weight: 0.20, Threshold: 0.75},
				{Name: "ambiguity_handling", Weight: 0.20, Threshold: 0.70},
				{Name: "fallback_appropriateness", Weight: 0.20, Threshold: 0.80},
			},
		},
		{
			Capability:  "FactualQA",
			Description: "Answering factual questions accurately with verifiable information",
			Weight:      1.2,
			Metrics: []Metric{
				{Name: "factual_correctness", Weight: 0.45, Threshold: 0.90},
				{Name: "source_grounding", Weight: 0.25, Threshold: 0.80},
				{Name: "completeness", Weight: 0.15, Threshold: 0.75},
				{Name: "conciseness", Weight: 0.15, Threshold: 0.70},
			},
		},
		{
			Capability:  "DeepExplain",
			Description: "Providing in-depth explanations with clear structure and progressive detail",
			Weight:      1.1,
			Metrics: []Metric{
				{Name: "depth_of_explanation", Weight: 0.30, Threshold: 0.80},
				{Name: "structural_clarity", Weight: 0.25, Threshold: 0.80},
				{Name: "accuracy", Weight: 0.25, Threshold: 0.85},
				{Name: "audience_adaptation", Weight: 0.20, Threshold: 0.75},
			},
		},
		{
			Capability:  "CompareTradeoff",
			Description: "Comparing multiple options with balanced pros, cons, and tradeoff analysis",
			Weight:      1.0,
			Metrics: []Metric{
				{Name: "balance", Weight: 0.25, Threshold: 0.80},
				{Name: "criteria_coverage", Weight: 0.25, Threshold: 0.75},
				{Name: "accuracy", Weight: 0.25, Threshold: 0.85},
				{Name: "actionable_conclusion", Weight: 0.25, Threshold: 0.70},
			},
		},
		{
			Capability:  "MultiTurnContext",
			Description: "Maintaining context, references, and coherence across multiple conversation turns",
			Weight:      1.1,
			Metrics: []Metric{
				{Name: "context_retention", Weight: 0.35, Threshold: 0.85},
				{Name: "coreference_resolution", Weight: 0.25, Threshold: 0.80},
				{Name: "topic_tracking", Weight: 0.20, Threshold: 0.80},
				{Name: "contradiction_avoidance", Weight: 0.20, Threshold: 0.85},
			},
		},
		{
			Capability:  "Planning",
			Description: "Creating structured, actionable, and realistic plans with clear steps",
			Weight:      0.9,
			Metrics: []Metric{
				{Name: "step_completeness", Weight: 0.30, Threshold: 0.80},
				{Name: "logical_ordering", Weight: 0.25, Threshold: 0.85},
				{Name: "feasibility", Weight: 0.25, Threshold: 0.75},
				{Name: "resource_awareness", Weight: 0.20, Threshold: 0.70},
			},
		},
		{
			Capability:  "ToolUseAccuracy",
			Description: "Selecting the correct tool or method and using it with proper parameters",
			Weight:      1.0,
			Metrics: []Metric{
				{Name: "tool_selection", Weight: 0.35, Threshold: 0.90},
				{Name: "parameter_accuracy", Weight: 0.30, Threshold: 0.85},
				{Name: "result_interpretation", Weight: 0.20, Threshold: 0.80},
				{Name: "error_handling", Weight: 0.15, Threshold: 0.75},
			},
		},
		{
			Capability:  "StyleControl",
			Description: "Matching the requested tone, format, verbosity, and stylistic constraints",
			Weight:      0.8,
			Metrics: []Metric{
				{Name: "tone_match", Weight: 0.30, Threshold: 0.80},
				{Name: "format_compliance", Weight: 0.30, Threshold: 0.80},
				{Name: "verbosity_control", Weight: 0.20, Threshold: 0.75},
				{Name: "consistency", Weight: 0.20, Threshold: 0.80},
			},
		},
	}
}

// Evaluate scores a response against a scorecard given per-metric scores.
// Each score in the map should be in [0, 1]. Missing metrics receive 0.
func Evaluate(card *Scorecard, scores map[string]float64) ScorecardResult {
	result := ScorecardResult{
		Capability: card.Capability,
		Scores:     make(map[string]float64, len(card.Metrics)),
		Pass:       true,
	}

	var weightedSum float64
	var totalWeight float64

	for _, m := range card.Metrics {
		score, ok := scores[m.Name]
		if !ok {
			score = 0.0
		}
		// Clamp to [0, 1].
		if score < 0 {
			score = 0
		}
		if score > 1 {
			score = 1
		}
		result.Scores[m.Name] = score

		weightedSum += score * m.Weight
		totalWeight += m.Weight

		if score < m.Threshold {
			result.Pass = false
		}
	}

	if totalWeight > 0 {
		result.Total = weightedSum / totalWeight
	}

	return result
}
