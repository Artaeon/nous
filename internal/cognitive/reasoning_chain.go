package cognitive

import (
	"fmt"
	"regexp"
	"strings"
)

// -----------------------------------------------------------------------
// Multi-Hop Reasoning — chain-of-thought in code, not prompts.
// Parses complex questions into a sequence of graph traversals,
// executes step by step, and returns traced, explainable answers.
// -----------------------------------------------------------------------

// ReasoningChain represents a multi-step reasoning process.
type ReasoningChain struct {
	Steps    []ReasoningStep
	Answer   string
	Trace    string  // human-readable explanation of the reasoning
	Confidence float64
}

// ReasoningStep is one step in the reasoning chain.
type ReasoningStep struct {
	Description string   // what this step does
	Query       string   // what we're looking for
	Relation    RelType  // edge type to follow (empty = any)
	Direction   string   // "forward" or "backward"
	ResultNodes []string // node IDs found
	ResultText  string   // human-readable result
	stepType    StepType // lookup, follow, compare, intersect, causal
}

// ReasoningEngine performs multi-hop reasoning over the cognitive graph.
type ReasoningEngine struct {
	Graph    *CognitiveGraph
	Semantic *SemanticEngine
	Analogy  *AnalogyEngine
}

// NewReasoningEngine creates a reasoning engine.
func NewReasoningEngine(graph *CognitiveGraph, semantic *SemanticEngine) *ReasoningEngine {
	return &ReasoningEngine{
		Graph:    graph,
		Semantic: semantic,
	}
}

// Reason answers a complex question through multi-hop graph traversal.
// Returns nil if the question can't be decomposed or answered.
func (re *ReasoningEngine) Reason(question string) *ReasoningChain {
	// 1. Try multi-hop decomposition
	plan := re.decompose(question)
	if plan == nil {
		return nil
	}

	// 2. Execute the plan step by step
	chain := &ReasoningChain{
		Steps: plan,
	}
	currentNodes := []string{} // context from previous step
	// For comparison: store results per entity
	var comparisonSets [][]string

	var traceLines []string
	for i := range chain.Steps {
		step := &chain.Steps[i]

		switch step.stepType {
		case StepCompare:
			// Compare the collected property sets
			if len(comparisonSets) >= 2 {
				shared, onlyA, onlyB := diffNodeSets(comparisonSets[0], comparisonSets[1])
				var parts []string
				if len(shared) > 0 {
					parts = append(parts, "Both share: "+strings.Join(re.nodeNames(shared), ", "))
				}
				if len(onlyA) > 0 {
					parts = append(parts, "Only first has: "+strings.Join(re.nodeNames(onlyA), ", "))
				}
				if len(onlyB) > 0 {
					parts = append(parts, "Only second has: "+strings.Join(re.nodeNames(onlyB), ", "))
				}
				step.ResultText = strings.Join(parts, ". ")
				traceLines = append(traceLines, fmt.Sprintf("Step %d: %s → %s", i+1, step.Description, step.ResultText))
			}
			continue

		case StepIntersect:
			// Find properties shared by all nodes in currentNodes
			if len(currentNodes) >= 2 {
				shared := re.intersectProperties(currentNodes)
				if len(shared) > 0 {
					names := re.nodeNames(shared)
					step.ResultText = strings.Join(names, ", ")
					traceLines = append(traceLines, fmt.Sprintf("Step %d: %s → %s", i+1, step.Description, step.ResultText))
					currentNodes = shared
				} else {
					traceLines = append(traceLines, fmt.Sprintf("Step %d: %s → no shared properties", i+1, step.Description))
				}
			}
			continue

		case StepCausal:
			// Follow causal chains (BFS on causes edges, max 4 hops)
			if len(currentNodes) > 0 {
				causalResults := re.followCausalChain(currentNodes, step.Direction, 4)
				if len(causalResults) > 0 {
					names := re.nodeNames(causalResults)
					step.ResultText = strings.Join(names, " → ")
					step.ResultNodes = causalResults
					traceLines = append(traceLines, fmt.Sprintf("Step %d: %s → %s", i+1, step.Description, step.ResultText))
					currentNodes = causalResults
				} else {
					// Fallback: follow all outgoing/incoming edges for broader results
					var fallbackResults []string
					if step.Direction == "forward" {
						fallbackResults = re.followEdges(currentNodes, "", "forward")
					} else {
						fallbackResults = re.followEdges(currentNodes, "", "backward")
					}
					if len(fallbackResults) > 0 {
						names := re.nodeNames(fallbackResults)
						step.ResultText = strings.Join(names, ", ")
						step.ResultNodes = fallbackResults
						traceLines = append(traceLines, fmt.Sprintf("Step %d: %s → %s", i+1, step.Description, step.ResultText))
						currentNodes = fallbackResults
					} else {
						traceLines = append(traceLines, fmt.Sprintf("Step %d: %s → no causal chain found", i+1, step.Description))
					}
				}
			}
			continue

		case StepAnalogy:
			// Apply entity principles to a target context
			if len(currentNodes) > 0 {
				entityID := currentNodes[0]
				entityNode := re.Graph.GetNode(entityID)
				entityLabel := entityID
				if entityNode != nil {
					entityLabel = entityNode.Label
				}
				context := step.Query

				var resultText string
				if re.Analogy != nil {
					resultText = re.Analogy.ApplyPrinciples(entityLabel, context)
				}
				if resultText == "" {
					// Fallback: gather properties from graph and compose manually
					props := re.followEdges(currentNodes, "", "forward")
					if len(props) > 0 {
						names := re.nodeNames(props)
						resultText = fmt.Sprintf("Based on %s's known attributes (%s), applied to %s.",
							entityLabel, strings.Join(names, ", "), context)
					}
				}
				if resultText != "" {
					step.ResultText = resultText
					traceLines = append(traceLines, fmt.Sprintf("Step %d: %s → %s", i+1, step.Description, step.ResultText))
				} else {
					traceLines = append(traceLines, fmt.Sprintf("Step %d: %s → no principles found", i+1, step.Description))
				}
			}
			continue

		case StepPathFind:
			// BFS shortest path between two sets of nodes
			parts := strings.SplitN(step.Query, "|", 2)
			if len(parts) == 2 {
				fromIDs := re.findNodes(strings.TrimSpace(parts[0]))
				toIDs := re.findNodes(strings.TrimSpace(parts[1]))
				if len(fromIDs) > 0 && len(toIDs) > 0 {
					path, edges := re.findPath(fromIDs, toIDs, 6)
					if len(path) > 0 {
						step.ResultText = re.describePath(path, edges)
						step.ResultNodes = path
						traceLines = append(traceLines, fmt.Sprintf("Step %d: %s → %s", i+1, step.Description, step.ResultText))
						currentNodes = path
					} else {
						traceLines = append(traceLines, fmt.Sprintf("Step %d: %s → no path found", i+1, step.Description))
					}
				} else {
					traceLines = append(traceLines, fmt.Sprintf("Step %d: %s → could not find nodes", i+1, step.Description))
				}
			}
			continue
		}

		// Default: lookup + follow (original logic)
		var startNodes []string
		if step.stepType == StepLookup || step.Query != "" {
			// Fresh lookup
			startNodes = re.findNodes(step.Query)
		} else if len(currentNodes) > 0 {
			startNodes = currentNodes
		}

		if len(startNodes) == 0 {
			traceLines = append(traceLines, fmt.Sprintf("Step %d: Could not find '%s'", i+1, step.Query))
			continue
		}

		if step.stepType == StepLookup {
			step.ResultNodes = startNodes
			names := re.nodeNames(startNodes)
			step.ResultText = strings.Join(names, ", ")
			traceLines = append(traceLines, fmt.Sprintf("Step %d: %s → %s", i+1, step.Description, step.ResultText))
			currentNodes = startNodes
			continue
		}

		// Follow edges from start nodes
		resultNodes := re.followEdges(startNodes, step.Relation, step.Direction)
		step.ResultNodes = resultNodes

		if len(resultNodes) > 0 {
			names := re.nodeNames(resultNodes)
			step.ResultText = strings.Join(names, ", ")
			traceLines = append(traceLines, fmt.Sprintf("Step %d: %s → %s", i+1, step.Description, step.ResultText))

			// For comparison: save this set
			if len(comparisonSets) < 2 {
				comparisonSets = append(comparisonSets, resultNodes)
			}
			currentNodes = resultNodes
		} else {
			traceLines = append(traceLines, fmt.Sprintf("Step %d: %s → no result", i+1, step.Description))
		}
	}

	// 3. Compose answer from results
	if len(currentNodes) > 0 || hasResults(chain.Steps) {
		chain.Answer = re.composeChainAnswer(question, chain.Steps)
		chain.Confidence = 0.7
		if chain.Answer == "" {
			names := re.nodeNames(currentNodes)
			chain.Answer = strings.Join(names, ", ")
		}
	}

	chain.Trace = strings.Join(traceLines, "\n")
	return chain
}

// -----------------------------------------------------------------------
// Question Decomposition — parse complex questions into steps
// -----------------------------------------------------------------------

// Multi-hop question patterns
var multiHopPatterns = []struct {
	re    *regexp.Regexp
	build func([]string) []ReasoningStep
}{
	// "What country is X in" → find X → follow located_in
	{regexp.MustCompile(`(?i)what (?:country|city|place|location) (?:is|are) (.+?) (?:in|from|at|located)`),
		func(m []string) []ReasoningStep {
			return []ReasoningStep{
				{Description: "Find " + m[1], Query: m[1]},
				{Description: "Find location", Relation: RelLocatedIn, Direction: "forward"},
			}
		}},

	// "Who founded/created X" → find X → follow founded_by/created_by backward
	{regexp.MustCompile(`(?i)who (?:founded|created|built|made|started) (.+)`),
		func(m []string) []ReasoningStep {
			return []ReasoningStep{
				{Description: "Find " + m[1], Query: m[1]},
				{Description: "Find founder/creator", Relation: RelFoundedBy, Direction: "forward"},
			}
		}},

	// "Where is the founder of X from" → find X → follow founded_by → follow located_in
	{regexp.MustCompile(`(?i)where (?:is|are|was) the (?:founder|creator|author) of (.+?) (?:from|located|based)`),
		func(m []string) []ReasoningStep {
			return []ReasoningStep{
				{Description: "Find " + m[1], Query: m[1]},
				{Description: "Find founder", Relation: RelFoundedBy, Direction: "forward"},
				{Description: "Find location", Relation: RelLocatedIn, Direction: "forward"},
			}
		}},

	// "What do X and Y have in common?" → find both, gather properties, compare
	{regexp.MustCompile(`(?i)what do (.+?) and (.+?) have in common`),
		func(m []string) []ReasoningStep {
			return []ReasoningStep{
				{Description: "Find " + m[1], Query: m[1], stepType: StepLookup},
				{Description: "Gather properties of " + m[1], Direction: "forward", stepType: StepFollow},
				{Description: "Find " + m[2], Query: m[2], stepType: StepLookup},
				{Description: "Gather properties of " + m[2], Direction: "forward", stepType: StepFollow},
				{Description: "Compare " + m[1] + " and " + m[2], stepType: StepCompare},
			}
		}},

	// "What does X offer/have" → find X → follow offers/has
	{regexp.MustCompile(`(?i)what (?:does|do) (.+?) (?:offer|have|provide|feature|include)`),
		func(m []string) []ReasoningStep {
			return []ReasoningStep{
				{Description: "Find " + m[1], Query: m[1]},
				{Description: "Find offerings", Relation: RelOffers, Direction: "forward"},
			}
		}},

	// "What is X used for" → find X → follow used_for
	{regexp.MustCompile(`(?i)what (?:is|are) (.+?) used for`),
		func(m []string) []ReasoningStep {
			return []ReasoningStep{
				{Description: "Find " + m[1], Query: m[1]},
				{Description: "Find uses", Relation: RelUsedFor, Direction: "forward"},
			}
		}},

	// "What companies are in X" → find X → follow located_in backward
	{regexp.MustCompile(`(?i)what (?:companies|organizations|things|entities) (?:are|is) (?:in|at|from) (.+)`),
		func(m []string) []ReasoningStep {
			return []ReasoningStep{
				{Description: "Find " + m[1], Query: m[1]},
				{Description: "Find things located there", Relation: RelLocatedIn, Direction: "backward"},
			}
		}},

	// "How are X and Y related" → find both, check edges between them
	{regexp.MustCompile(`(?i)how (?:are|is) (.+?) (?:and|&) (.+?) related`),
		func(m []string) []ReasoningStep {
			return []ReasoningStep{
				{Description: "Find " + m[1], Query: m[1]},
				{Description: "Find connection to " + m[2], Query: m[2]},
			}
		}},

	// "What kind of X is Y" / "What type of X is Y" → find Y → follow is_a
	{regexp.MustCompile(`(?i)what (?:kind|type|sort) of .+? (?:is|are) (.+)`),
		func(m []string) []ReasoningStep {
			return []ReasoningStep{
				{Description: "Find " + m[1], Query: m[1]},
				{Description: "Find type/category", Relation: RelIsA, Direction: "forward"},
			}
		}},

	// "Tell me everything about X" → find X → follow all edges
	{regexp.MustCompile(`(?i)(?:tell me )?everything (?:about|on) (.+)`),
		func(m []string) []ReasoningStep {
			return []ReasoningStep{
				{Description: "Find " + m[1], Query: m[1]},
				{Description: "Gather all knowledge", Direction: "forward"},
			}
		}},

	// "What would X say/think about Y?" → find X's principles, apply to Y
	{regexp.MustCompile(`(?i)what would (.+?) (?:say|think) about (.+)`),
		func(m []string) []ReasoningStep {
			return []ReasoningStep{
				{Description: "Find " + m[1], Query: m[1], stepType: StepLookup},
				{Description: "Apply " + m[1] + "'s principles to " + m[2], Query: m[2], stepType: StepAnalogy},
			}
		}},

	// "Why is X important?" → find X, follow causes/effects/used_for
	{regexp.MustCompile(`(?i)why (?:is|are) (.+?) important`),
		func(m []string) []ReasoningStep {
			return []ReasoningStep{
				{Description: "Find " + m[1], Query: m[1], stepType: StepLookup},
				{Description: "Find what " + m[1] + " causes or is used for", Direction: "forward", stepType: StepFollow},
			}
		}},

	// "What are the implications/consequences/effects/impact of X?"
	{regexp.MustCompile(`(?i)what (?:are|is) the (?:implications?|consequences?|effects?|impact) of (.+)`),
		func(m []string) []ReasoningStep {
			return []ReasoningStep{
				{Description: "Find " + m[1], Query: m[1], stepType: StepLookup},
				{Description: "Follow forward causal chains from " + m[1], Direction: "forward", stepType: StepCausal},
			}
		}},

	// "How is X related/connected to Y?" or "What is the relationship/connection between X and Y?"
	{regexp.MustCompile(`(?i)how (?:is|are) (.+?) (?:related|connected) to (.+)`),
		func(m []string) []ReasoningStep {
			return []ReasoningStep{
				{Description: "Find path from " + m[1] + " to " + m[2], Query: m[1] + "|" + m[2], stepType: StepPathFind},
			}
		}},
	{regexp.MustCompile(`(?i)what (?:is|are) the (?:relationship|connection) between (.+?) and (.+)`),
		func(m []string) []ReasoningStep {
			return []ReasoningStep{
				{Description: "Find path from " + m[1] + " to " + m[2], Query: m[1] + "|" + m[2], stepType: StepPathFind},
			}
		}},

	// "What if X didn't exist?" or "What if there was no X?" — counterfactual
	{regexp.MustCompile(`(?i)what if (?:there (?:was|were) no|there (?:wasn't|weren't) any?) (.+)`),
		func(m []string) []ReasoningStep {
			return []ReasoningStep{
				{Description: "Find " + m[1], Query: m[1], stepType: StepLookup},
				{Description: "Find what depends on " + m[1], Direction: "backward", stepType: StepCausal},
				{Description: "Find what " + m[1] + " causes", Direction: "forward", stepType: StepCausal},
			}
		}},
	{regexp.MustCompile(`(?i)what if (.+?) (?:didn'?t|never|hadn'?t) exist`),
		func(m []string) []ReasoningStep {
			return []ReasoningStep{
				{Description: "Find " + m[1], Query: m[1], stepType: StepLookup},
				{Description: "Find what depends on " + m[1], Direction: "backward", stepType: StepCausal},
				{Description: "Find what " + m[1] + " causes", Direction: "forward", stepType: StepCausal},
			}
		}},

	// "How did X influence Y?" → find path from X to Y through causal/related edges
	{regexp.MustCompile(`(?i)how (?:did|does|do) (.+?) influence (.+)`),
		func(m []string) []ReasoningStep {
			return []ReasoningStep{
				{Description: "Find path from " + m[1] + " to " + m[2], Query: m[1] + "|" + m[2], stepType: StepPathFind},
			}
		}},

}

func (re *ReasoningEngine) decompose(question string) []ReasoningStep {
	question = strings.TrimSpace(question)

	// Fast path: specific multi-hop patterns
	for _, p := range multiHopPatterns {
		m := p.re.FindStringSubmatch(question)
		if len(m) > 0 {
			return p.build(m)
		}
	}

	// General path: structural question analysis
	return re.decomposeGeneral(question)
}

// -----------------------------------------------------------------------
// General-Purpose Chain-of-Thought Decomposer
//
// Instead of matching fixed regex patterns, this analyzes the question's
// structure to generate a reasoning plan for ANY question type:
//   - Comparison: "How is X different from Y?"
//   - Aggregation: "What do all Greek philosophers have in common?"
//   - Causal: "Why did X happen?"
//   - Conditional: "If X, what would happen?"
//   - Multi-hop: "Where was the creator of X born?"
//   - Single-entity: "Tell me about X" (fallback)
// -----------------------------------------------------------------------

// StepType classifies what a reasoning step does.
type StepType string

const (
	StepLookup    StepType = "lookup"    // find nodes matching a query
	StepFollow    StepType = "follow"    // follow edges from found nodes
	StepCompare   StepType = "compare"   // diff two result sets
	StepIntersect StepType = "intersect" // find shared properties
	StepCausal    StepType = "causal"    // follow causal chains
	StepAnalogy   StepType = "analogy"   // map structure from one domain to another
	StepPathFind  StepType = "pathfind"  // find shortest path between two nodes
)

func (re *ReasoningEngine) decomposeGeneral(question string) []ReasoningStep {
	lower := strings.ToLower(strings.TrimRight(strings.TrimSpace(question), "?!."))

	// Comparison: "how is X different from Y", "X vs Y", "compare X and Y"
	if a, b, ok := extractComparison(lower); ok {
		return []ReasoningStep{
			{Description: "Find " + a, Query: a, stepType: StepLookup},
			{Description: "Gather properties of " + a, Direction: "forward", stepType: StepFollow},
			{Description: "Find " + b, Query: b, stepType: StepLookup},
			{Description: "Gather properties of " + b, Direction: "forward", stepType: StepFollow},
			{Description: "Compare " + a + " and " + b, stepType: StepCompare},
		}
	}

	// Aggregation: "what do all X have in common", "shared properties of X"
	if group, ok := extractAggregation(lower); ok {
		return []ReasoningStep{
			{Description: "Find " + group, Query: group, stepType: StepLookup},
			{Description: "Find members", Relation: RelIsA, Direction: "backward", stepType: StepFollow},
			{Description: "Intersect shared properties", Direction: "forward", stepType: StepIntersect},
		}
	}

	// Causal: "why does X", "what causes X", "what leads to X"
	if cause, ok := extractCausalQuery(lower); ok {
		return []ReasoningStep{
			{Description: "Find " + cause, Query: cause, stepType: StepLookup},
			{Description: "Follow causal chains", Relation: RelCauses, Direction: "backward", stepType: StepCausal},
		}
	}

	// Conditional/counterfactual: "what would happen if X", "if X then what"
	if hypothesis, ok := extractConditional(lower); ok {
		return []ReasoningStep{
			{Description: "Find " + hypothesis, Query: hypothesis, stepType: StepLookup},
			{Description: "Follow effects", Relation: RelCauses, Direction: "forward", stepType: StepCausal},
		}
	}

	// Multi-hop via possessive/prepositional chains: "the X of the Y of Z"
	if steps := extractMultiHopChain(lower); len(steps) > 0 {
		return steps
	}

	// Single-entity fallback: extract the most important noun phrase
	entity := extractPrimaryEntity(lower)
	if entity != "" {
		return []ReasoningStep{
			{Description: "Find " + entity, Query: entity, stepType: StepLookup},
			{Description: "Gather all knowledge", Direction: "forward", stepType: StepFollow},
		}
	}

	return nil
}

// extractComparison detects comparison questions and returns both targets.
func extractComparison(lower string) (string, string, bool) {
	// "how is X different from Y"
	compRe := []*regexp.Regexp{
		regexp.MustCompile(`how (?:is|are) (.+?) different from (.+)`),
		regexp.MustCompile(`(?:compare|difference between) (.+?) (?:and|&|vs\.?) (.+)`),
		regexp.MustCompile(`(.+?) (?:vs\.?|versus) (.+)`),
		regexp.MustCompile(`what(?:'s| is) the difference between (.+?) and (.+)`),
		regexp.MustCompile(`how (?:does|do) (.+?) compare (?:to|with) (.+)`),
	}
	for _, re := range compRe {
		m := re.FindStringSubmatch(lower)
		if len(m) >= 3 {
			return strings.TrimSpace(m[1]), strings.TrimSpace(m[2]), true
		}
	}
	return "", "", false
}

// extractAggregation detects "what do all X have in common" style questions.
func extractAggregation(lower string) (string, bool) {
	aggRe := []*regexp.Regexp{
		regexp.MustCompile(`what do (?:all )?(.+?) have in common`),
		regexp.MustCompile(`(?:shared|common) (?:properties|features|traits) of (.+)`),
		regexp.MustCompile(`what (?:is|are) (?:common|shared) (?:between|among) (.+)`),
		regexp.MustCompile(`how are (.+?) (?:similar|alike)`),
	}
	for _, re := range aggRe {
		m := re.FindStringSubmatch(lower)
		if len(m) >= 2 {
			return strings.TrimSpace(m[1]), true
		}
	}
	return "", false
}

// extractCausalQuery detects "why" and "what causes" questions.
func extractCausalQuery(lower string) (string, bool) {
	causalRe := []*regexp.Regexp{
		regexp.MustCompile(`why (?:does|do|did|is|are|was|were) (.+)`),
		regexp.MustCompile(`what (?:causes?|leads? to|results? in) (.+)`),
		regexp.MustCompile(`what is the (?:cause|reason) (?:of|for|behind) (.+)`),
	}
	for _, re := range causalRe {
		m := re.FindStringSubmatch(lower)
		if len(m) >= 2 {
			return strings.TrimSpace(m[1]), true
		}
	}
	return "", false
}

// extractConditional detects "what if" and hypothetical questions.
func extractConditional(lower string) (string, bool) {
	condRe := []*regexp.Regexp{
		regexp.MustCompile(`what (?:would|will|could) happen if (.+)`),
		regexp.MustCompile(`(?:suppose|imagine|assuming) (.+?)(?:,| then| what)`),
		regexp.MustCompile(`if (.+?),? what (?:would|will|could)`),
		regexp.MustCompile(`what if (.+?) (?:didn't|never|hadn't|wasn't|weren't) (.+)`),
		regexp.MustCompile(`what if (.+)`),
	}
	for _, re := range condRe {
		m := re.FindStringSubmatch(lower)
		if len(m) >= 2 {
			return strings.TrimSpace(m[1]), true
		}
	}
	return "", false
}

// extractMultiHopChain parses possessive/prepositional chains into steps.
// "where was the founder of stoicism born" → find stoicism → follow founded_by → follow located_in
func extractMultiHopChain(lower string) []ReasoningStep {
	// Detect "X of Y" chains
	ofRe := regexp.MustCompile(`the (\w+) of (.+)`)
	m := ofRe.FindStringSubmatch(lower)
	if len(m) < 3 {
		return nil
	}

	relWord := m[1]  // "founder", "creator", "capital", etc.
	target := m[2]   // "stoicism", "france", etc.

	// Map relational word to edge type
	relMap := map[string]RelType{
		"founder":  RelFoundedBy,
		"creator":  RelCreatedBy,
		"author":   RelCreatedBy,
		"location": RelLocatedIn,
		"capital":  RelLocatedIn,
		"parts":    RelPartOf,
		"parent":   RelIsA,
		"type":     RelIsA,
		"uses":     RelUsedFor,
		"purpose":  RelUsedFor,
	}

	rel, ok := relMap[relWord]
	if !ok {
		// Try with plural stripped
		singular := strings.TrimSuffix(relWord, "s")
		rel, ok = relMap[singular]
	}
	if !ok {
		return nil
	}

	steps := []ReasoningStep{
		{Description: "Find " + target, Query: target, stepType: StepLookup},
		{Description: "Follow " + relWord, Relation: rel, Direction: "forward", stepType: StepFollow},
	}

	// Check if there's a question word that implies a further hop
	// "where was the founder of X" → add location step
	if strings.HasPrefix(lower, "where") && rel != RelLocatedIn {
		steps = append(steps, ReasoningStep{
			Description: "Find location", Relation: RelLocatedIn, Direction: "forward", stepType: StepFollow,
		})
	}

	return steps
}

// extractPrimaryEntity pulls the main noun phrase from a question.
func extractPrimaryEntity(lower string) string {
	// Strip question scaffolding
	prefixes := []string{
		"tell me about ", "what is ", "what are ", "who is ", "who was ",
		"who are ", "what was ", "explain ", "describe ", "define ",
		"how does ", "how do ", "how is ", "how are ",
	}
	for _, p := range prefixes {
		if strings.HasPrefix(lower, p) {
			entity := strings.TrimSpace(lower[len(p):])
			if entity != "" {
				return entity
			}
		}
	}

	// Fall back to longest non-stop word sequence
	words := strings.Fields(lower)
	var entityWords []string
	for _, w := range words {
		if !isExtractiveStop(w) && len(w) > 2 {
			entityWords = append(entityWords, w)
		}
	}
	if len(entityWords) > 0 {
		return strings.Join(entityWords, " ")
	}
	return ""
}

// -----------------------------------------------------------------------
// Graph Traversal
// -----------------------------------------------------------------------

// findNodes finds nodes matching a query, using semantic similarity.
func (re *ReasoningEngine) findNodes(query string) []string {
	query = strings.TrimRight(strings.TrimSpace(query), "?!.")
	lower := strings.ToLower(query)

	var ids []string

	re.Graph.mu.RLock()
	defer re.Graph.mu.RUnlock()

	// Exact match
	if _, ok := re.Graph.nodes[lower]; ok {
		ids = append(ids, lower)
	}

	// Label index match
	if nodeIDs, ok := re.Graph.byLabel[lower]; ok {
		ids = append(ids, nodeIDs...)
	}

	// Substring match
	for id, node := range re.Graph.nodes {
		if strings.Contains(strings.ToLower(node.Label), lower) {
			ids = append(ids, id)
		}
	}

	// Semantic similarity match (if we have a semantic engine)
	if len(ids) == 0 && re.Semantic != nil {
		for id, node := range re.Graph.nodes {
			sim := re.Semantic.Similarity(lower, strings.ToLower(node.Label))
			if sim > 0.4 {
				ids = append(ids, id)
			}
		}
	}

	// Also try individual words from multi-word queries
	if len(ids) == 0 {
		words := strings.Fields(lower)
		for _, word := range words {
			if len(word) < 3 || isExtractiveStop(word) {
				continue
			}
			for id, node := range re.Graph.nodes {
				if strings.Contains(strings.ToLower(node.Label), word) {
					ids = append(ids, id)
				}
			}
		}
	}

	return uniqueStrings(ids)
}

// followEdges follows edges from the given nodes.
func (re *ReasoningEngine) followEdges(nodeIDs []string, rel RelType, direction string) []string {
	re.Graph.mu.RLock()
	defer re.Graph.mu.RUnlock()

	var results []string

	for _, id := range nodeIDs {
		if direction == "backward" {
			for _, edge := range re.Graph.inEdges[id] {
				if rel == "" || edge.Relation == rel {
					results = append(results, edge.From)
				}
			}
		} else {
			for _, edge := range re.Graph.outEdges[id] {
				if rel == "" || edge.Relation == rel {
					results = append(results, edge.To)
				}
			}
		}
	}

	return uniqueStrings(results)
}

// nodeNames returns human-readable labels for node IDs.
func (re *ReasoningEngine) nodeNames(ids []string) []string {
	re.Graph.mu.RLock()
	defer re.Graph.mu.RUnlock()

	var names []string
	for _, id := range ids {
		if node, ok := re.Graph.nodes[id]; ok {
			names = append(names, node.Label)
		}
	}
	return names
}

// hasResults checks if any step in the chain produced results.
func hasResults(steps []ReasoningStep) bool {
	for _, s := range steps {
		if s.ResultText != "" {
			return true
		}
	}
	return false
}

// diffNodeSets compares two sets of node IDs.
func diffNodeSets(a, b []string) (shared, onlyA, onlyB []string) {
	setA := make(map[string]bool)
	setB := make(map[string]bool)
	for _, id := range a {
		setA[id] = true
	}
	for _, id := range b {
		setB[id] = true
	}
	for id := range setA {
		if setB[id] {
			shared = append(shared, id)
		} else {
			onlyA = append(onlyA, id)
		}
	}
	for id := range setB {
		if !setA[id] {
			onlyB = append(onlyB, id)
		}
	}
	return
}

// intersectProperties finds properties (outgoing edges) shared by all given nodes.
func (re *ReasoningEngine) intersectProperties(nodeIDs []string) []string {
	if len(nodeIDs) == 0 {
		return nil
	}

	re.Graph.mu.RLock()
	defer re.Graph.mu.RUnlock()

	// Collect properties for each node
	propCounts := make(map[string]int)
	for _, id := range nodeIDs {
		seen := make(map[string]bool)
		for _, edge := range re.Graph.outEdges[id] {
			if !seen[edge.To] {
				propCounts[edge.To]++
				seen[edge.To] = true
			}
		}
	}

	// Return properties shared by at least 60% of nodes
	threshold := int(float64(len(nodeIDs)) * 0.6)
	if threshold < 2 {
		threshold = 2
	}

	var shared []string
	for prop, count := range propCounts {
		if count >= threshold {
			shared = append(shared, prop)
		}
	}
	return shared
}

// followCausalChain does BFS along causal edges up to maxDepth hops.
func (re *ReasoningEngine) followCausalChain(startNodes []string, direction string, maxDepth int) []string {
	re.Graph.mu.RLock()
	defer re.Graph.mu.RUnlock()

	visited := make(map[string]bool)
	var result []string
	current := startNodes

	for depth := 0; depth < maxDepth && len(current) > 0; depth++ {
		var next []string
		for _, id := range current {
			if visited[id] {
				continue
			}
			visited[id] = true

			var edges []*CogEdge
			if direction == "forward" {
				edges = re.Graph.outEdges[id]
			} else {
				edges = re.Graph.inEdges[id]
			}

			for _, edge := range edges {
				if edge.Relation == RelCauses || edge.Relation == RelFollows {
					target := edge.To
					if direction == "backward" {
						target = edge.From
					}
					if !visited[target] {
						next = append(next, target)
						result = append(result, target)
					}
				}
			}
		}
		current = next
	}
	return result
}

// findPath performs BFS shortest path between two sets of nodes, max depth hops.
// Returns the node IDs along the path and the connecting edges.
func (re *ReasoningEngine) findPath(fromIDs, toIDs []string, maxDepth int) ([]string, []*CogEdge) {
	re.Graph.mu.RLock()
	defer re.Graph.mu.RUnlock()

	toSet := make(map[string]bool, len(toIDs))
	for _, id := range toIDs {
		toSet[id] = true
	}

	// BFS state: track parent and connecting edge
	type bfsItem struct {
		id    string
		depth int
	}
	parent := make(map[string]string)      // child → parent node ID
	parentEdge := make(map[string]*CogEdge) // child → edge used to reach it
	visited := make(map[string]bool)

	var queue []bfsItem
	for _, id := range fromIDs {
		visited[id] = true
		parent[id] = ""
		queue = append(queue, bfsItem{id, 0})
	}

	var foundTarget string
	for len(queue) > 0 && foundTarget == "" {
		item := queue[0]
		queue = queue[1:]

		if item.depth > 0 && toSet[item.id] {
			foundTarget = item.id
			break
		}

		if item.depth >= maxDepth {
			continue
		}

		// Explore outgoing edges
		for _, edge := range re.Graph.outEdges[item.id] {
			if !visited[edge.To] {
				visited[edge.To] = true
				parent[edge.To] = item.id
				parentEdge[edge.To] = edge
				queue = append(queue, bfsItem{edge.To, item.depth + 1})
			}
		}

		// Explore incoming edges (bidirectional search)
		for _, edge := range re.Graph.inEdges[item.id] {
			if !visited[edge.From] {
				visited[edge.From] = true
				parent[edge.From] = item.id
				parentEdge[edge.From] = edge
				queue = append(queue, bfsItem{edge.From, item.depth + 1})
			}
		}
	}

	if foundTarget == "" {
		return nil, nil
	}

	// Reconstruct path
	var path []string
	var edges []*CogEdge
	for cur := foundTarget; cur != ""; cur = parent[cur] {
		path = append([]string{cur}, path...)
		if edge, ok := parentEdge[cur]; ok {
			edges = append([]*CogEdge{edge}, edges...)
		}
	}

	return path, edges
}

// describePath converts a path of node IDs and connecting edges to natural language.
func (re *ReasoningEngine) describePath(path []string, edges []*CogEdge) string {
	if len(path) == 0 {
		return ""
	}

	re.Graph.mu.RLock()
	defer re.Graph.mu.RUnlock()

	// Build "X → [rel] → Y → [rel] → Z" style description
	var parts []string
	for i, id := range path {
		name := id
		if node, ok := re.Graph.nodes[id]; ok {
			name = node.Label
		}

		if i == 0 {
			parts = append(parts, name)
			continue
		}

		if i-1 < len(edges) {
			edge := edges[i-1]
			parts = append(parts, fmt.Sprintf("[%s]", edge.Relation))
		}
		parts = append(parts, name)
	}

	arrow := strings.Join(parts, " → ")

	// Also compose a sentence form
	if len(path) >= 2 {
		firstName := path[0]
		lastName := path[len(path)-1]
		if node, ok := re.Graph.nodes[firstName]; ok {
			firstName = node.Label
		}
		if node, ok := re.Graph.nodes[lastName]; ok {
			lastName = node.Label
		}

		if len(path) == 2 && len(edges) > 0 {
			return fmt.Sprintf("%s is connected to %s via %s", firstName, lastName, edges[0].Relation)
		}
		return fmt.Sprintf("%s is connected to %s: %s", firstName, lastName, arrow)
	}

	return arrow
}

// composeChainAnswer creates a natural answer from the reasoning chain.
func (re *ReasoningEngine) composeChainAnswer(question string, steps []ReasoningStep) string {
	// Find the last step that produced results — that's the answer
	lastResult := ""
	for i := len(steps) - 1; i >= 0; i-- {
		if steps[i].ResultText != "" {
			lastResult = steps[i].ResultText
			break
		}
	}

	if lastResult == "" {
		return ""
	}

	// For multi-hop chains with results at multiple levels, show the chain
	resultCount := 0
	for _, step := range steps {
		if step.ResultText != "" {
			resultCount++
		}
	}

	if resultCount <= 1 || len(steps) <= 2 {
		return lastResult + "."
	}

	// Multi-hop: show traversal path
	var b strings.Builder
	for i, step := range steps {
		if step.ResultText == "" {
			continue
		}
		if i > 0 && b.Len() > 0 {
			b.WriteString(" → ")
		}
		b.WriteString(step.ResultText)
	}
	b.WriteString(".")
	return b.String()
}

// -----------------------------------------------------------------------
// Concept Abstraction — form general rules from specific observations
// -----------------------------------------------------------------------

// Abstraction represents a generalized rule.
type Abstraction struct {
	Rule       string   // "compiled languages tend to be fast"
	Evidence   []string // specific examples that support this
	Confidence float64
}

// AbstractionEngine discovers general patterns in the graph.
type AbstractionEngine struct {
	Graph *CognitiveGraph
}

// NewAbstractionEngine creates an abstraction engine.
func NewAbstractionEngine(graph *CognitiveGraph) *AbstractionEngine {
	return &AbstractionEngine{Graph: graph}
}

// Discover finds abstractions from the current graph state.
func (ae *AbstractionEngine) Discover() []Abstraction {
	ae.Graph.mu.RLock()
	defer ae.Graph.mu.RUnlock()

	var abstractions []Abstraction

	// Pattern: multiple entities with same is_a AND same property
	// → "things that are X tend to be Y"
	// e.g., Go is_a programming_language + Go described_as fast
	//        Rust is_a programming_language + Rust described_as fast
	//        → "programming languages tend to be fast"

	// Group nodes by their is_a type
	typeGroups := make(map[string][]string) // type → node IDs
	for _, edge := range ae.Graph.edges {
		if edge.Relation == RelIsA && !edge.Inferred {
			typeGroups[edge.To] = append(typeGroups[edge.To], edge.From)
		}
	}

	for typeID, members := range typeGroups {
		if len(members) < 2 {
			continue
		}

		typeNode := ae.Graph.nodes[typeID]
		typeLabel := typeID
		if typeNode != nil {
			typeLabel = typeNode.Label
		}

		// Find shared properties among members
		propCounts := make(map[string]int)     // property → count of members with it
		propExamples := make(map[string][]string) // property → example member labels

		for _, memberID := range members {
			for _, edge := range ae.Graph.outEdges[memberID] {
				if edge.Relation == RelDescribedAs || edge.Relation == RelHas ||
					edge.Relation == RelDomain {
					propCounts[edge.To]++
					memberNode := ae.Graph.nodes[memberID]
					if memberNode != nil {
						propExamples[edge.To] = append(propExamples[edge.To], memberNode.Label)
					}
				}
			}
		}

		// If a property is shared by 60%+ of members, it's an abstraction
		threshold := int(float64(len(members)) * 0.6)
		if threshold < 2 {
			threshold = 2
		}

		for propID, count := range propCounts {
			if count >= threshold {
				propNode := ae.Graph.nodes[propID]
				propLabel := propID
				if propNode != nil {
					propLabel = propNode.Label
				}

				rule := fmt.Sprintf("%s tend to be %s", typeLabel, propLabel)
				abstractions = append(abstractions, Abstraction{
					Rule:       rule,
					Evidence:   propExamples[propID],
					Confidence: float64(count) / float64(len(members)),
				})
			}
		}
	}

	return abstractions
}

// -----------------------------------------------------------------------
// Self-Correction — learn from user feedback
// -----------------------------------------------------------------------

// Correction represents a user-provided correction.
type Correction struct {
	WrongFact  string // what was wrong
	RightFact  string // what is correct
	NodeID     string // which node was affected
}

// ApplyCorrection updates the graph based on user feedback.
// Lowers confidence of wrong edges, adds correct edges.
func ApplyCorrection(cg *CognitiveGraph, correction Correction) {
	cg.mu.Lock()
	defer cg.mu.Unlock()

	wrongLower := strings.ToLower(correction.WrongFact)

	// Find and weaken edges matching the wrong fact
	for _, edge := range cg.edges {
		from := cg.nodes[edge.From]
		to := cg.nodes[edge.To]
		if from == nil || to == nil {
			continue
		}

		fact := strings.ToLower(from.Label + " " + string(edge.Relation) + " " + to.Label)
		if strings.Contains(fact, wrongLower) ||
			strings.Contains(wrongLower, strings.ToLower(to.Label)) {
			// Penalize this edge heavily
			edge.Confidence *= 0.2
			edge.Weight *= 0.3
		}
	}

	// Add the correct fact if provided
	if correction.RightFact != "" {
		triples := ExtractTriples(correction.RightFact)
		for _, t := range triples {
			subjType := guessNodeType(t.Subject)
			objType := guessNodeType(t.Object)
			fromID := cg.ensureNodeLocked(t.Subject, subjType, "user_correction", 0.95)
			toID := cg.ensureNodeLocked(t.Object, objType, "user_correction", 0.95)
			cg.addEdgeLocked(fromID, toID, t.Relation, "user_correction", 0.95, false)
		}
	}

	cg.modified = true
}

// DetectCorrection identifies when user input is correcting a previous answer.
// Returns a Correction if detected, nil otherwise.
func DetectCorrection(input string) *Correction {
	lower := strings.ToLower(input)

	// Patterns: "no, X is actually Y", "that's wrong, X", "incorrect, X"
	correctionPatterns := []struct {
		re      *regexp.Regexp
		wrongIdx int
		rightIdx int
	}{
		{regexp.MustCompile(`(?i)(?:no|nope|wrong|incorrect|actually),?\s*(.+?)(?:\s*[,.]?\s*(?:it'?s?|the correct|the right|actually)\s+(?:is\s+)?(.+?))?\.?$`), 1, 2},
		{regexp.MustCompile(`(?i)(?:that'?s?\s+(?:not right|wrong|incorrect))\s*[,.]?\s*(.+?)\.?$`), 1, 0},
	}

	for _, p := range correctionPatterns {
		m := p.re.FindStringSubmatch(lower)
		if len(m) > p.wrongIdx && m[p.wrongIdx] != "" {
			c := &Correction{WrongFact: strings.TrimSpace(m[p.wrongIdx])}
			if p.rightIdx > 0 && len(m) > p.rightIdx && m[p.rightIdx] != "" {
				c.RightFact = strings.TrimSpace(m[p.rightIdx])
			}
			return c
		}
	}

	// Simple negation: "X is not Y" → wrong fact
	if strings.Contains(lower, " not ") || strings.Contains(lower, " isn't ") ||
		strings.Contains(lower, " wasn't ") {
		return &Correction{WrongFact: input}
	}

	return nil
}
