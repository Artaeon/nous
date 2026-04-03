package cognitive

import (
	"fmt"
	"math"
	"strings"
	"time"

	"github.com/artaeon/nous/internal/memory"
)

// -----------------------------------------------------------------------
// Simulation Engine — "What-If Worlds"
//
// Runs forward scenario simulations over the knowledge graph by chaining:
//   - GraphCausalReasoner: propagate causal effects step by step
//   - InnerCouncil: evaluate each intermediate state from 5 perspectives
//   - MultiHopReasoner: discover indirect connections and side effects
//   - CognitiveGraph: world state (entities + relationships)
//
// Unlike LLM-based simulators (MiroFish, OASIS) that burn API credits
// generating text, this runs entirely on graph traversal and symbolic
// reasoning — deterministic, traceable, instant.
//
// Usage:
//   sim := NewSimulationEngine(graph, causal, council, multihop)
//   result := sim.Simulate("What if renewable energy becomes cheaper than coal?", 5)
//   fmt.Println(result.Report)
// -----------------------------------------------------------------------

// SimulationEngine runs "what if" scenarios over the knowledge graph.
type SimulationEngine struct {
	Graph    *CognitiveGraph
	Causal   *GraphCausalReasoner
	Council  *InnerCouncil
	MultiHop *MultiHopReasoner
	Episodic *memory.EpisodicMemory
}

// SimulationStep is one step forward in the simulation.
type SimulationStep struct {
	Step        int             // 1-based step number
	Effects     []CausalEffect  // causal effects discovered at this step
	Connections []string        // multi-hop connections found
	Council     *CouncilDeliberation // council evaluation of this state
	Confidence  float64         // aggregate confidence for this step
	Summary     string          // natural language summary
}

// SimulationResult is the full output of a simulation run.
type SimulationResult struct {
	Scenario    string
	Steps       []SimulationStep
	Entities    []string        // all entities involved
	FinalVerdict string         // overall assessment
	Confidence  float64         // aggregate confidence (decays with steps)
	Duration    time.Duration
	Report      string          // formatted natural language report
}

// NewSimulationEngine creates a simulation engine wired to existing cognitive systems.
func NewSimulationEngine(
	graph *CognitiveGraph,
	causal *GraphCausalReasoner,
	council *InnerCouncil,
	multihop *MultiHopReasoner,
) *SimulationEngine {
	return &SimulationEngine{
		Graph:    graph,
		Causal:   causal,
		Council:  council,
		MultiHop: multihop,
	}
}

// Simulate runs a scenario forward for the given number of steps.
// Each step propagates causal effects, discovers connections, and
// consults the inner council for evaluation.
func (se *SimulationEngine) Simulate(scenario string, steps int) *SimulationResult {
	start := time.Now()

	if steps < 1 {
		steps = 1
	}
	if steps > 10 {
		steps = 10 // safety limit
	}

	result := &SimulationResult{
		Scenario: scenario,
		Steps:    make([]SimulationStep, 0, steps),
	}

	// Extract the core hypothesis from the scenario.
	hypothesis := extractHypothesis(scenario)

	// Track all entities we encounter.
	entitySet := make(map[string]bool)

	// Run the initial causal analysis.
	var allEffects []CausalEffect
	if se.Causal != nil {
		chain := se.Causal.WhatIf(hypothesis)
		if chain != nil {
			allEffects = chain.Effects
		}
	}

	// If causal analysis found nothing, try graph-based fact gathering.
	if len(allEffects) == 0 && se.Graph != nil {
		allEffects = se.gatherEffectsFromGraph(hypothesis)
	}

	// Distribute effects across simulation steps.
	// Early steps get direct effects (high confidence),
	// later steps get indirect effects (lower confidence).
	stepEffects := se.distributeEffects(allEffects, steps)

	baseConfidence := 0.95

	for i := 0; i < steps; i++ {
		stepNum := i + 1
		confidence := baseConfidence * math.Pow(0.85, float64(i)) // 15% decay per step

		effects := stepEffects[i]
		for _, e := range effects {
			entitySet[e.Entity] = true
		}

		// Discover multi-hop connections between hypothesis and effects.
		var connections []string
		if se.MultiHop != nil && len(effects) > 0 {
			for _, e := range effects[:min(len(effects), 3)] {
				conn := se.MultiHop.FindConnection(hypothesis, e.Entity)
				if conn != nil && conn.Summary != "" {
					connections = append(connections, conn.Summary)
				}
			}
		}

		// Council evaluates the current state.
		var council *CouncilDeliberation
		if se.Council != nil {
			stepContext := fmt.Sprintf("Simulating step %d of '%s'. Effects so far: %s",
				stepNum, scenario, summarizeEffects(effects))
			nlu := &NLUResult{
				Intent:   "simulate",
				Raw:      stepContext,
				Entities: map[string]string{"topic": hypothesis},
			}
			council = se.Council.Deliberate(stepContext, nlu, nil)
		}

		summary := se.composeStepSummary(stepNum, hypothesis, effects, connections, confidence)

		result.Steps = append(result.Steps, SimulationStep{
			Step:        stepNum,
			Effects:     effects,
			Connections: connections,
			Council:     council,
			Confidence:  confidence,
			Summary:     summary,
		})
	}

	// Collect entities.
	for e := range entitySet {
		result.Entities = append(result.Entities, e)
	}

	// Compute aggregate confidence.
	if len(result.Steps) > 0 {
		last := result.Steps[len(result.Steps)-1]
		result.Confidence = last.Confidence
	}

	// Generate final verdict from the last council deliberation.
	result.FinalVerdict = se.composeFinalVerdict(result)

	result.Duration = time.Since(start)
	result.Report = se.composeReport(result)

	return result
}

// SimulateRemoval runs a "what if X were removed" scenario.
func (se *SimulationEngine) SimulateRemoval(entity string) *SimulationResult {
	start := time.Now()

	result := &SimulationResult{
		Scenario: fmt.Sprintf("What if %s were removed?", entity),
	}

	if se.Causal == nil {
		result.FinalVerdict = "Cannot simulate: causal reasoner not available."
		result.Duration = time.Since(start)
		result.Report = result.FinalVerdict
		return result
	}

	chain := se.Causal.WhatIfRemoved(entity)
	if chain == nil || len(chain.Effects) == 0 {
		result.FinalVerdict = fmt.Sprintf("No significant dependencies found for %s.", entity)
		result.Confidence = 0.5
		result.Duration = time.Since(start)
		result.Report = result.FinalVerdict
		return result
	}

	// Group effects by depth into steps.
	depthMap := make(map[int][]CausalEffect)
	for _, e := range chain.Effects {
		depthMap[e.Depth] = append(depthMap[e.Depth], e)
	}

	for depth := 1; depth <= 4; depth++ {
		effects := depthMap[depth]
		if len(effects) == 0 {
			continue
		}
		confidence := 0.9 * math.Pow(0.8, float64(depth-1))
		summary := se.composeStepSummary(depth, entity, effects, nil, confidence)
		result.Steps = append(result.Steps, SimulationStep{
			Step:       depth,
			Effects:    effects,
			Confidence: confidence,
			Summary:    summary,
		})
	}

	result.Confidence = chain.Confidence
	result.FinalVerdict = se.Causal.ComposeCounterfactualAnswer(entity, chain, true)
	result.Duration = time.Since(start)
	result.Report = se.composeReport(result)

	return result
}

// -----------------------------------------------------------------------
// Internal helpers
// -----------------------------------------------------------------------

// extractHypothesis strips "what if" phrasing to get the core hypothesis.
func extractHypothesis(scenario string) string {
	lower := strings.ToLower(strings.TrimSpace(scenario))
	prefixes := []string{
		"what if ", "what would happen if ", "simulate ",
		"predict ", "what happens if ", "imagine ",
		"suppose ", "assume ", "hypothetically ",
	}
	for _, p := range prefixes {
		if strings.HasPrefix(lower, p) {
			return strings.TrimRight(strings.TrimSpace(scenario[len(p):]), "?.!")
		}
	}
	return strings.TrimRight(strings.TrimSpace(scenario), "?.!")
}

// gatherEffectsFromGraph creates synthetic effects from graph edges
// when the causal reasoner has no direct causal edges to traverse.
func (se *SimulationEngine) gatherEffectsFromGraph(topic string) []CausalEffect {
	if se.Graph == nil {
		return nil
	}

	var effects []CausalEffect
	edges := se.Graph.EdgesFrom(topic)
	for _, e := range edges {
		label := se.Graph.NodeLabel(e.To)
		if label == "" {
			label = e.To
		}
		effects = append(effects, CausalEffect{
			Entity:     label,
			Relation:   string(e.Relation),
			Depth:      1,
			Confidence: e.Confidence * 0.8,
		})
	}

	// Also check incoming edges for "what depends on this topic".
	incoming := se.Graph.EdgesTo(topic)
	for _, e := range incoming {
		label := se.Graph.NodeLabel(e.From)
		if label == "" {
			label = e.From
		}
		effects = append(effects, CausalEffect{
			Entity:     label,
			Relation:   "affected_by_" + string(e.Relation),
			Depth:      2,
			Confidence: e.Confidence * 0.6,
		})
	}

	return effects
}

// distributeEffects spreads effects across simulation steps based on depth.
func (se *SimulationEngine) distributeEffects(effects []CausalEffect, steps int) [][]CausalEffect {
	result := make([][]CausalEffect, steps)

	if len(effects) == 0 {
		return result
	}

	// Group by depth first.
	depthMap := make(map[int][]CausalEffect)
	maxDepth := 0
	for _, e := range effects {
		d := e.Depth
		if d < 1 {
			d = 1
		}
		depthMap[d] = append(depthMap[d], e)
		if d > maxDepth {
			maxDepth = d
		}
	}

	// Map depth groups to simulation steps.
	for depth, depthEffects := range depthMap {
		// Map depth 1..maxDepth to step 0..steps-1
		stepIdx := 0
		if maxDepth > 1 {
			stepIdx = int(float64(depth-1) / float64(maxDepth) * float64(steps))
		}
		if stepIdx >= steps {
			stepIdx = steps - 1
		}
		result[stepIdx] = append(result[stepIdx], depthEffects...)
	}

	// Ensure no empty steps — redistribute from neighbors.
	for i := range result {
		if len(result[i]) == 0 && i > 0 && len(result[i-1]) > 1 {
			// Borrow one effect from previous step.
			result[i] = append(result[i], result[i-1][len(result[i-1])-1])
			result[i-1] = result[i-1][:len(result[i-1])-1]
		}
	}

	return result
}

func summarizeEffects(effects []CausalEffect) string {
	if len(effects) == 0 {
		return "no direct effects found"
	}
	var parts []string
	for _, e := range effects {
		parts = append(parts, fmt.Sprintf("%s (%s)", e.Entity, e.Relation))
	}
	return strings.Join(parts, ", ")
}

func (se *SimulationEngine) composeStepSummary(step int, topic string, effects []CausalEffect, connections []string, confidence float64) string {
	confLabel := "high"
	if confidence < 0.5 {
		confLabel = "low"
	} else if confidence < 0.75 {
		confLabel = "medium"
	}

	var b strings.Builder
	fmt.Fprintf(&b, "Step %d (%s confidence): ", step, confLabel)

	if len(effects) == 0 {
		b.WriteString("No additional effects detected at this depth.")
		return b.String()
	}

	// Group effects by relation type for cleaner prose.
	relGroups := make(map[string][]string)
	for _, e := range effects {
		relGroups[e.Relation] = append(relGroups[e.Relation], e.Entity)
	}

	first := true
	for rel, entities := range relGroups {
		if !first {
			b.WriteString(" ")
		}
		first = false
		humanRel := humanizeRelation(rel)
		fmt.Fprintf(&b, "%s %s %s.", strings.Title(topic), humanRel, strings.Join(entities, ", "))
	}

	if len(connections) > 0 {
		b.WriteString(" Connections: ")
		b.WriteString(strings.Join(connections, "; "))
	}

	return b.String()
}

func humanizeRelation(rel string) string {
	switch rel {
	case "causes":
		return "causes changes in"
	case "follows":
		return "leads to"
	case "is_a":
		return "is a type of"
	case "part_of":
		return "is part of"
	case "has":
		return "has"
	case "used_for":
		return "is used for"
	case "related_to":
		return "is related to"
	case "influenced_by":
		return "is influenced by"
	case "created_by":
		return "was created by"
	case "known_for":
		return "is known for"
	default:
		if strings.HasPrefix(rel, "affected_by_") {
			return "would be affected through " + strings.TrimPrefix(rel, "affected_by_")
		}
		return rel
	}
}

func (se *SimulationEngine) composeFinalVerdict(result *SimulationResult) string {
	if len(result.Steps) == 0 {
		return "Insufficient data to simulate this scenario."
	}

	// Count total effects.
	totalEffects := 0
	for _, s := range result.Steps {
		totalEffects += len(s.Effects)
	}

	// Use the last council deliberation if available.
	var councilSynthesis string
	for i := len(result.Steps) - 1; i >= 0; i-- {
		if result.Steps[i].Council != nil && result.Steps[i].Council.Synthesis != "" {
			councilSynthesis = result.Steps[i].Council.Synthesis
			break
		}
	}

	var b strings.Builder
	fmt.Fprintf(&b, "Simulation of \"%s\" traced %d effects across %d steps. ",
		result.Scenario, totalEffects, len(result.Steps))

	confPct := int(result.Confidence * 100)
	if confPct >= 70 {
		b.WriteString(fmt.Sprintf("Overall confidence: %d%% (reliable for near-term projections). ", confPct))
	} else if confPct >= 40 {
		b.WriteString(fmt.Sprintf("Overall confidence: %d%% (moderate — later steps are speculative). ", confPct))
	} else {
		b.WriteString(fmt.Sprintf("Overall confidence: %d%% (low — treat as exploratory). ", confPct))
	}

	if councilSynthesis != "" {
		b.WriteString("Council assessment: ")
		b.WriteString(councilSynthesis)
	}

	return b.String()
}

func (se *SimulationEngine) composeReport(result *SimulationResult) string {
	var b strings.Builder

	// Header
	fmt.Fprintf(&b, "# Simulation Report: %s\n\n", result.Scenario)
	fmt.Fprintf(&b, "**Duration:** %s | **Steps:** %d | **Confidence:** %d%%\n\n",
		result.Duration.Round(time.Millisecond), len(result.Steps), int(result.Confidence*100))

	// Steps
	b.WriteString("## Causal Chain\n\n")
	for _, step := range result.Steps {
		fmt.Fprintf(&b, "### Step %d (confidence: %d%%)\n\n", step.Step, int(step.Confidence*100))
		b.WriteString(step.Summary)
		b.WriteString("\n\n")

		// Council opinions if available.
		if step.Council != nil && len(step.Council.Opinions) > 0 {
			for _, op := range step.Council.Opinions {
				if op.Assessment == "" {
					continue
				}
				label := perspectiveLabel(op.Perspective)
				fmt.Fprintf(&b, "- **%s:** %s\n", label, op.Assessment)
			}
			b.WriteString("\n")
		}
	}

	// Entities involved
	if len(result.Entities) > 0 {
		b.WriteString("## Entities Involved\n\n")
		for _, e := range result.Entities {
			fmt.Fprintf(&b, "- %s\n", e)
		}
		b.WriteString("\n")
	}

	// Final verdict
	b.WriteString("## Verdict\n\n")
	b.WriteString(result.FinalVerdict)
	b.WriteString("\n")

	return b.String()
}

func perspectiveLabel(p Perspective) string {
	switch p {
	case PerspPragmatist:
		return "Pragmatist"
	case PerspHistorian:
		return "Historian"
	case PerspEmpath:
		return "Empath"
	case PerspArchitect:
		return "Architect"
	case PerspSkeptic:
		return "Skeptic"
	default:
		return "Unknown"
	}
}

// -----------------------------------------------------------------------
// Detection helpers — used by NLU to route simulation queries.
// -----------------------------------------------------------------------

// IsSimulationQuery detects if a query is asking for a simulation.
func IsSimulationQuery(input string) bool {
	lower := strings.ToLower(strings.TrimSpace(input))
	triggers := []string{
		"what if ", "what would happen if ",
		"simulate ", "predict what ",
		"what happens if ", "what happens when ",
		"imagine if ", "suppose ",
		"hypothetically ", "scenario:",
		"what would change if ",
		"what are the consequences of ",
		"impact of ", "effect of ",
	}
	for _, t := range triggers {
		if strings.HasPrefix(lower, t) || strings.Contains(lower, t) {
			return true
		}
	}
	return false
}

// IsRemovalQuery detects "what if X were removed/eliminated/gone" patterns.
func IsRemovalQuery(input string) bool {
	lower := strings.ToLower(input)
	removalSignals := []string{
		"removed", "eliminated", "gone", "disappeared",
		"didn't exist", "never existed", "without ",
		"no longer", "ceased to exist", "abolished",
	}
	for _, s := range removalSignals {
		if strings.Contains(lower, s) {
			return true
		}
	}
	return false
}
