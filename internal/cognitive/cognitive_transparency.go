package cognitive

import (
	"fmt"
	"strings"
)

// -----------------------------------------------------------------------
// Cognitive transparency: reasoning traces that explain HOW a response
// was generated. Every response can reveal what knowledge was used, what
// was inferred, what is uncertain, and what reasoning path was taken.
//
// This is the difference between a black box and a glass box. Users see
// the system's thought process, not just the output.
// -----------------------------------------------------------------------

// CognitiveTransparency builds human-readable reasoning traces
// that explain HOW a response was generated.
type CognitiveTransparency struct{}

// NewCognitiveTransparency creates a new transparency engine.
func NewCognitiveTransparency() *CognitiveTransparency {
	return &CognitiveTransparency{}
}

// TransparencyTrace captures every step the system took to produce a response.
type TransparencyTrace struct {
	Steps         []TransparencyStep
	Sources       []TransparencySource
	Inferences    []TransparencyInference
	Uncertainties []string
	Confidence    float64 // overall confidence in the response
	Summary       string  // one-line summary: "Answered from knowledge graph with high confidence"
}

// TransparencyStep is one step in the reasoning process.
type TransparencyStep struct {
	Action      string  // "retrieved", "inferred", "compared", "synthesized", "fallback"
	Description string  // what happened: "Found 3 facts about quantum physics"
	Input       string  // what went in
	Output      string  // what came out
	Confidence  float64
	Duration    string // "< 1ms", "2ms", etc.
}

// TransparencySource identifies where knowledge came from.
type TransparencySource struct {
	Name       string  // "knowledge_graph", "episodic_memory", "reasoning_chain", "inference"
	FactCount  int     // how many facts from this source
	Confidence float64 // how reliable this source is
}

// TransparencyInference captures a logical inference that was made.
type TransparencyInference struct {
	Premise    string  // "Python is a programming language" + "Programming languages have syntax"
	Conclusion string  // "Python has syntax"
	Rule       string  // "transitivity", "analogy", "generalization", "decomposition"
	Confidence float64
}

// BuildTrace constructs a transparency trace from the components gathered
// during response generation.
func (ct *CognitiveTransparency) BuildTrace(
	query string,
	sources []string,
	factCount int,
	inferences []string,
	uncertainties []string,
	confidence float64,
) *TransparencyTrace {
	trace := &TransparencyTrace{
		Confidence:    clampConfidence(confidence),
		Uncertainties: uncertainties,
	}

	if trace.Uncertainties == nil {
		trace.Uncertainties = []string{}
	}

	// Build source entries and retrieval steps from source names.
	sourceFactCounts := distributeFactCount(factCount, len(sources))
	for i, src := range sources {
		count := sourceFactCounts[i]
		trace.Sources = append(trace.Sources, TransparencySource{
			Name:       src,
			FactCount:  count,
			Confidence: confidence,
		})
		desc := fmt.Sprintf("Retrieved %d facts about %q from %s", count, query, src)
		trace.Steps = append(trace.Steps, TransparencyStep{
			Action:      "retrieved",
			Description: desc,
			Input:       query,
			Output:      fmt.Sprintf("%d facts", count),
			Confidence:  confidence,
			Duration:    "< 1ms",
		})
	}

	// Build inference entries.
	for _, inf := range inferences {
		trace.Inferences = append(trace.Inferences, TransparencyInference{
			Premise:    query,
			Conclusion: inf,
			Rule:       "generalization",
			Confidence: confidence * 0.9, // inferences are slightly less certain
		})
		trace.Steps = append(trace.Steps, TransparencyStep{
			Action:      "inferred",
			Description: fmt.Sprintf("Inferred %q from related concepts", inf),
			Input:       query,
			Output:      inf,
			Confidence:  confidence * 0.9,
			Duration:    "< 1ms",
		})
	}

	// Build uncertainty steps.
	for _, u := range uncertainties {
		trace.Steps = append(trace.Steps, TransparencyStep{
			Action:      "fallback",
			Description: fmt.Sprintf("Uncertainty: %s", u),
			Input:       query,
			Output:      "",
			Confidence:  0,
			Duration:    "< 1ms",
		})
	}

	// Build summary.
	trace.Summary = buildSummary(sources, factCount, confidence)

	return trace
}

// Format produces a human-readable representation of the trace.
//
// Compact mode (verbose=false):
//
//	[Answered from knowledge graph (3 facts, confidence: high)]
//
// Verbose mode (verbose=true):
//
//	How I arrived at this:
//	1. Retrieved 3 facts about "quantum physics" from knowledge graph
//	2. Inferred "wave-particle duality" from related concepts
//	3. Uncertainty: I don't have data on recent experimental results
//
//	Sources: knowledge_graph (3 facts), inference (1 conclusion)
//	Confidence: 0.78 (moderate-high)
func (rt *TransparencyTrace) Format(verbose bool) string {
	if !verbose {
		return rt.formatCompact()
	}
	return rt.formatVerbose()
}

// formatCompact returns the single-line compact representation.
func (rt *TransparencyTrace) formatCompact() string {
	// Identify primary source.
	primarySource := "unknown"
	totalFacts := 0
	for _, s := range rt.Sources {
		if s.FactCount > totalFacts || primarySource == "unknown" {
			primarySource = s.Name
		}
		totalFacts += s.FactCount
	}
	if totalFacts == 0 && len(rt.Inferences) > 0 {
		primarySource = "inference"
	}

	label := ClassifyConfidence(rt.Confidence)
	return fmt.Sprintf("[Answered from %s (%d facts, confidence: %s)]",
		primarySource, totalFacts, label)
}

// formatVerbose returns the multi-line verbose representation.
func (rt *TransparencyTrace) formatVerbose() string {
	var b strings.Builder

	b.WriteString("How I arrived at this:\n")

	// Number the meaningful steps (retrieved, inferred, uncertainty).
	stepNum := 1
	for _, step := range rt.Steps {
		switch step.Action {
		case "retrieved":
			fmt.Fprintf(&b, "%d. %s\n", stepNum, step.Description)
			stepNum++
		case "inferred":
			fmt.Fprintf(&b, "%d. %s\n", stepNum, step.Description)
			stepNum++
		case "fallback":
			fmt.Fprintf(&b, "%d. %s\n", stepNum, step.Description)
			stepNum++
		}
	}

	// Sources line.
	if len(rt.Sources) > 0 || len(rt.Inferences) > 0 {
		b.WriteString("\nSources: ")
		var parts []string
		for _, s := range rt.Sources {
			parts = append(parts, fmt.Sprintf("%s (%d facts)", s.Name, s.FactCount))
		}
		if len(rt.Inferences) > 0 {
			noun := "conclusion"
			if len(rt.Inferences) > 1 {
				noun = "conclusions"
			}
			parts = append(parts, fmt.Sprintf("inference (%d %s)", len(rt.Inferences), noun))
		}
		b.WriteString(strings.Join(parts, ", "))
		b.WriteString("\n")
	}

	// Confidence line.
	label := ClassifyConfidence(rt.Confidence)
	fmt.Fprintf(&b, "Confidence: %.2f (%s)", rt.Confidence, label)

	return b.String()
}

// AddStep appends a step to the trace during response generation.
func (rt *TransparencyTrace) AddStep(action, description string, confidence float64) {
	rt.Steps = append(rt.Steps, TransparencyStep{
		Action:      action,
		Description: description,
		Confidence:  clampConfidence(confidence),
		Duration:    "< 1ms",
	})
}

// AddUncertainty records something the system is uncertain about.
func (rt *TransparencyTrace) AddUncertainty(what string) {
	rt.Uncertainties = append(rt.Uncertainties, what)
}

// AddInference records a logical inference.
func (rt *TransparencyTrace) AddInference(premise, conclusion, rule string, confidence float64) {
	rt.Inferences = append(rt.Inferences, TransparencyInference{
		Premise:    premise,
		Conclusion: conclusion,
		Rule:       rule,
		Confidence: clampConfidence(confidence),
	})
}

// ClassifyConfidence maps a numeric confidence score to a human label.
//
//	> 0.9  => "very high"
//	0.7-0.9 => "high"
//	0.5-0.7 => "moderate"
//	0.3-0.5 => "low"
//	< 0.3  => "very low"
func ClassifyConfidence(score float64) string {
	switch {
	case score > 0.9:
		return "very high"
	case score >= 0.7:
		return "high"
	case score >= 0.5:
		return "moderate"
	case score >= 0.3:
		return "low"
	default:
		return "very low"
	}
}

// MergeTraces combines multiple transparency traces from different
// subsystems into a single unified trace.
func MergeTraces(traces ...*TransparencyTrace) *TransparencyTrace {
	merged := &TransparencyTrace{
		Uncertainties: []string{},
	}

	if len(traces) == 0 {
		return merged
	}

	totalConf := 0.0
	confCount := 0

	for _, t := range traces {
		if t == nil {
			continue
		}
		merged.Steps = append(merged.Steps, t.Steps...)
		merged.Sources = append(merged.Sources, t.Sources...)
		merged.Inferences = append(merged.Inferences, t.Inferences...)
		merged.Uncertainties = append(merged.Uncertainties, t.Uncertainties...)

		if t.Confidence > 0 {
			totalConf += t.Confidence
			confCount++
		}
	}

	// Average confidence across traces.
	if confCount > 0 {
		merged.Confidence = totalConf / float64(confCount)
	}

	// Deduplicate sources by name, summing fact counts.
	merged.Sources = deduplicateSources(merged.Sources)

	// Build summary from merged data.
	sourceNames := make([]string, 0, len(merged.Sources))
	totalFacts := 0
	for _, s := range merged.Sources {
		sourceNames = append(sourceNames, s.Name)
		totalFacts += s.FactCount
	}
	merged.Summary = buildSummary(sourceNames, totalFacts, merged.Confidence)

	return merged
}

// ---------- Internal helpers ----------

// clampConfidence constrains a confidence value to [0, 1].
func clampConfidence(c float64) float64 {
	if c < 0 {
		return 0
	}
	if c > 1 {
		return 1
	}
	return c
}

// distributeFactCount spreads factCount across n sources.
// The first source gets any remainder.
func distributeFactCount(factCount, n int) []int {
	if n == 0 {
		return nil
	}
	counts := make([]int, n)
	base := factCount / n
	remainder := factCount % n
	for i := range counts {
		counts[i] = base
		if i < remainder {
			counts[i]++
		}
	}
	return counts
}

// deduplicateSources merges sources with the same name, summing fact
// counts and taking the max confidence.
func deduplicateSources(sources []TransparencySource) []TransparencySource {
	if len(sources) == 0 {
		return sources
	}

	seen := make(map[string]int) // name -> index in result
	var result []TransparencySource

	for _, s := range sources {
		if idx, ok := seen[s.Name]; ok {
			result[idx].FactCount += s.FactCount
			if s.Confidence > result[idx].Confidence {
				result[idx].Confidence = s.Confidence
			}
		} else {
			seen[s.Name] = len(result)
			result = append(result, s)
		}
	}
	return result
}

// buildSummary creates a one-line summary string.
func buildSummary(sources []string, factCount int, confidence float64) string {
	if len(sources) == 0 {
		return fmt.Sprintf("No sources available (confidence: %s)", ClassifyConfidence(confidence))
	}

	primary := sources[0]
	label := ClassifyConfidence(confidence)

	if factCount > 0 {
		return fmt.Sprintf("Answered from %s with %s confidence", primary, label)
	}
	return fmt.Sprintf("Answered from %s with %s confidence", primary, label)
}
