package cognitive

import (
	"fmt"
	"math"
	"strings"
	"sync"
	"time"
)

// -----------------------------------------------------------------------
// Causal Reasoning Engine — discovers WHY things happen.
// Tracks temporal co-occurrence of events and builds causal hypotheses.
// Can answer "why" questions from observed patterns.
// -----------------------------------------------------------------------

// CausalEngine discovers and tracks causal relationships.
type CausalEngine struct {
	events  []CausalEvent
	links   []CausalLink
	mu      sync.RWMutex
}

// CausalEvent is a recorded event with context.
type CausalEvent struct {
	Action   string            // what happened (journal, expense, weather, mood)
	Tags     map[string]string // context (mood=stressed, amount=50, category=food)
	When     time.Time
}

// CausalLink is a discovered causal relationship.
type CausalLink struct {
	Cause       string  // event/condition that precedes
	Effect      string  // event/condition that follows
	Confidence  float64 // how confident we are (0-1)
	Occurrences int     // how many times observed
	AvgDelay    time.Duration // average time between cause and effect
	Description string  // human-readable explanation
}

// NewCausalEngine creates a causal reasoning engine.
func NewCausalEngine() *CausalEngine {
	return &CausalEngine{}
}

// RecordEvent logs an event for causal analysis.
func (ce *CausalEngine) RecordEvent(action string, tags map[string]string) {
	ce.mu.Lock()
	defer ce.mu.Unlock()

	ce.events = append(ce.events, CausalEvent{
		Action: action,
		Tags:   tags,
		When:   time.Now(),
	})

	// Keep last 5000 events
	if len(ce.events) > 5000 {
		ce.events = ce.events[len(ce.events)-5000:]
	}
}

// AnalyzeCausality discovers causal relationships from event history.
// Looks for events that consistently precede other events.
func (ce *CausalEngine) AnalyzeCausality() []CausalLink {
	ce.mu.Lock()
	defer ce.mu.Unlock()

	if len(ce.events) < 10 {
		return nil
	}

	// Count temporal co-occurrences within time windows
	type pair struct {
		cause, effect string
	}
	cooccur := make(map[pair][]time.Duration)

	// 4-hour window for causal detection
	window := 4 * time.Hour

	for i := 0; i < len(ce.events); i++ {
		for j := i + 1; j < len(ce.events); j++ {
			delay := ce.events[j].When.Sub(ce.events[i].When)
			if delay > window {
				break
			}
			if delay < 0 {
				continue
			}

			cause := ce.events[i].Action
			effect := ce.events[j].Action
			if cause == effect {
				continue
			}

			// Also check tag-level causality
			// e.g., mood=stressed → expense.category=food
			causeKey := cause
			for k, v := range ce.events[i].Tags {
				if k == "mood" || k == "weather" || k == "category" {
					causeKey = cause + ":" + k + "=" + v
				}
			}
			effectKey := effect
			for k, v := range ce.events[j].Tags {
				if k == "mood" || k == "category" || k == "amount" {
					effectKey = effect + ":" + k + "=" + v
				}
			}

			p := pair{causeKey, effectKey}
			cooccur[p] = append(cooccur[p], delay)
		}
	}

	// Find statistically significant causal links (3+ occurrences)
	var links []CausalLink
	for p, delays := range cooccur {
		if len(delays) < 3 {
			continue
		}

		// Calculate average delay
		var totalDelay time.Duration
		for _, d := range delays {
			totalDelay += d
		}
		avgDelay := totalDelay / time.Duration(len(delays))

		// Confidence based on frequency and consistency
		consistency := 1.0 - coefficientOfVariation(delays)
		if consistency < 0 {
			consistency = 0
		}
		confidence := math.Min(1.0, float64(len(delays))*0.15*consistency)

		if confidence < 0.3 {
			continue
		}

		link := CausalLink{
			Cause:       p.cause,
			Effect:      p.effect,
			Confidence:  confidence,
			Occurrences: len(delays),
			AvgDelay:    avgDelay,
			Description: formatCausalDescription(p.cause, p.effect, len(delays), avgDelay),
		}
		links = append(links, link)
	}

	ce.links = links
	return links
}

// AnswerWhy tries to answer a "why" question from causal links.
func (ce *CausalEngine) AnswerWhy(question string) string {
	ce.mu.RLock()
	defer ce.mu.RUnlock()

	if len(ce.links) == 0 {
		return ""
	}

	lower := strings.ToLower(question)

	// Find causal links that mention the effect
	var relevant []CausalLink
	for _, link := range ce.links {
		effectLower := strings.ToLower(link.Effect)
		causeLower := strings.ToLower(link.Cause)
		if strings.Contains(lower, effectLower) ||
			strings.Contains(lower, causeLower) ||
			containsAnyWord(lower, effectLower) {
			relevant = append(relevant, link)
		}
	}

	if len(relevant) == 0 {
		return ""
	}

	// Compose answer from top causal links
	var parts []string
	for i, link := range relevant {
		if i >= 3 {
			break
		}
		parts = append(parts, link.Description)
	}

	return strings.Join(parts, " Also, ")
}

// containsAnyWord checks if any word from b appears in a.
func containsAnyWord(a, b string) bool {
	words := strings.Fields(b)
	for _, w := range words {
		w = strings.Trim(w, ":=")
		if len(w) > 3 && strings.Contains(a, w) {
			return true
		}
	}
	return false
}

// coefficientOfVariation measures how consistent the delays are.
func coefficientOfVariation(delays []time.Duration) float64 {
	if len(delays) < 2 {
		return 0
	}

	var sum float64
	for _, d := range delays {
		sum += float64(d)
	}
	mean := sum / float64(len(delays))

	if mean == 0 {
		return 0
	}

	var variance float64
	for _, d := range delays {
		diff := float64(d) - mean
		variance += diff * diff
	}
	variance /= float64(len(delays))

	return math.Sqrt(variance) / mean
}

func formatCausalDescription(cause, effect string, count int, avgDelay time.Duration) string {
	causeHuman := humanizeCausalTerm(cause)
	effectHuman := humanizeCausalTerm(effect)

	delayStr := ""
	if avgDelay > time.Hour {
		hours := avgDelay.Hours()
		if hours >= 24 {
			delayStr = fmt.Sprintf("about %d days later", int(hours/24))
		} else {
			delayStr = fmt.Sprintf("about %d hours later", int(hours))
		}
	} else if avgDelay > time.Minute {
		delayStr = fmt.Sprintf("about %d minutes later", int(avgDelay.Minutes()))
	}

	if delayStr != "" {
		return fmt.Sprintf("When %s happens, %s tends to follow %s (observed %d times).",
			causeHuman, effectHuman, delayStr, count)
	}
	return fmt.Sprintf("When %s happens, %s tends to follow (observed %d times).",
		causeHuman, effectHuman, count)
}

func humanizeCausalTerm(term string) string {
	// "expense:category=food" → "food expenses"
	if strings.Contains(term, ":") {
		parts := strings.SplitN(term, ":", 2)
		action := parts[0]
		if strings.Contains(parts[1], "=") {
			kv := strings.SplitN(parts[1], "=", 2)
			return kv[1] + " " + action
		}
		return action
	}
	return term
}

// -----------------------------------------------------------------------
// Graph-Based Causal Reasoning — counterfactual "what if" analysis
// over the CognitiveGraph's causal edges.
// -----------------------------------------------------------------------

// GraphCausalReasoner performs counterfactual reasoning over the knowledge graph.
type GraphCausalReasoner struct {
	Graph *CognitiveGraph
}

// NewGraphCausalReasoner creates a graph-based causal reasoning engine.
func NewGraphCausalReasoner(graph *CognitiveGraph) *GraphCausalReasoner {
	return &GraphCausalReasoner{Graph: graph}
}

// CausalChainResult is the output of counterfactual reasoning.
type CausalChainResult struct {
	Effects    []CausalEffect // ordered chain of effects
	Trace      string         // human-readable explanation
	Confidence float64
}

// CausalEffect is one consequence in a causal chain.
type CausalEffect struct {
	Entity     string
	Relation   string  // how this effect connects
	Depth      int     // hops from the hypothesis
	Confidence float64 // decays with each hop
}

// WhatIf answers "What would happen if X?" by following causal edges forward.
func (gcr *GraphCausalReasoner) WhatIf(hypothesis string) *CausalChainResult {
	gcr.Graph.mu.RLock()
	defer gcr.Graph.mu.RUnlock()

	// Find the hypothesis node
	startIDs := gcr.findNodes(hypothesis)
	if len(startIDs) == 0 {
		return nil
	}

	// BFS forward on causes, follows, and related_to edges
	var effects []CausalEffect
	visited := make(map[string]bool)
	type queueItem struct {
		id         string
		depth      int
		confidence float64
	}
	queue := make([]queueItem, 0)

	for _, id := range startIDs {
		queue = append(queue, queueItem{id, 0, 1.0})
		visited[id] = true
	}

	for len(queue) > 0 {
		item := queue[0]
		queue = queue[1:]

		if item.depth > 4 || item.confidence < 0.1 {
			continue
		}

		for _, edge := range gcr.Graph.outEdges[item.id] {
			if visited[edge.To] {
				continue
			}
			if !isCausalRelation(edge.Relation) {
				continue
			}

			visited[edge.To] = true
			childConf := item.confidence * edge.Confidence * 0.8

			label := edge.To
			if node, ok := gcr.Graph.nodes[edge.To]; ok {
				label = node.Label
			}

			effects = append(effects, CausalEffect{
				Entity:     label,
				Relation:   string(edge.Relation),
				Depth:      item.depth + 1,
				Confidence: childConf,
			})

			queue = append(queue, queueItem{edge.To, item.depth + 1, childConf})
		}
	}

	if len(effects) == 0 {
		return nil
	}

	// Compose trace
	var trace strings.Builder
	trace.WriteString(fmt.Sprintf("If %s happens:\n", hypothesis))
	for _, e := range effects {
		indent := strings.Repeat("  ", e.Depth)
		trace.WriteString(fmt.Sprintf("%s→ %s (confidence: %.0f%%)\n", indent, e.Entity, e.Confidence*100))
	}

	return &CausalChainResult{
		Effects:    effects,
		Trace:      trace.String(),
		Confidence: effects[0].Confidence,
	}
}

// WhatIfRemoved answers "What would happen without X?" by finding
// nodes that depend solely on X (no alternate paths).
func (gcr *GraphCausalReasoner) WhatIfRemoved(entity string) *CausalChainResult {
	gcr.Graph.mu.RLock()
	defer gcr.Graph.mu.RUnlock()

	// Find the entity node
	targetIDs := gcr.findNodes(entity)
	if len(targetIDs) == 0 {
		return nil
	}

	targetSet := make(map[string]bool)
	for _, id := range targetIDs {
		targetSet[id] = true
	}

	// Find all nodes reachable from the target
	reachable := make(map[string]bool)
	var dfs func(id string, depth int)
	dfs = func(id string, depth int) {
		if depth > 5 || reachable[id] {
			return
		}
		reachable[id] = true
		for _, edge := range gcr.Graph.outEdges[id] {
			dfs(edge.To, depth+1)
		}
	}
	for _, id := range targetIDs {
		for _, edge := range gcr.Graph.outEdges[id] {
			dfs(edge.To, 0)
		}
	}

	// For each reachable node, check if there's an alternate path NOT through target
	var dependents []CausalEffect
	for nodeID := range reachable {
		if targetSet[nodeID] {
			continue
		}

		// Check if any incoming edge comes from a non-target, non-reachable-only-through-target node
		hasAlternate := false
		for _, edge := range gcr.Graph.inEdges[nodeID] {
			if !targetSet[edge.From] {
				hasAlternate = true
				break
			}
		}

		if !hasAlternate {
			label := nodeID
			if node, ok := gcr.Graph.nodes[nodeID]; ok {
				label = node.Label
			}
			dependents = append(dependents, CausalEffect{
				Entity:     label,
				Relation:   "depends on " + entity,
				Depth:      1,
				Confidence: 0.8,
			})
		}
	}

	if len(dependents) == 0 {
		return nil
	}

	entityLabel := entity
	if len(targetIDs) > 0 {
		if node, ok := gcr.Graph.nodes[targetIDs[0]]; ok {
			entityLabel = node.Label
		}
	}

	var trace strings.Builder
	trace.WriteString(fmt.Sprintf("Without %s, these would be affected:\n", entityLabel))
	for _, d := range dependents {
		trace.WriteString(fmt.Sprintf("  → %s (%s)\n", d.Entity, d.Relation))
	}

	return &CausalChainResult{
		Effects:    dependents,
		Trace:      trace.String(),
		Confidence: 0.7,
	}
}

// ComposeCounterfactualAnswer creates a natural language response from a causal result.
func (gcr *GraphCausalReasoner) ComposeCounterfactualAnswer(hypothesis string, result *CausalChainResult, isRemoval bool) string {
	if result == nil || len(result.Effects) == 0 {
		return ""
	}

	var b strings.Builder
	if isRemoval {
		b.WriteString(fmt.Sprintf("Without %s, several things would be affected. ", hypothesis))
	} else {
		b.WriteString(fmt.Sprintf("If %s were to happen, there would be consequences. ", hypothesis))
	}

	for i, e := range result.Effects {
		if i >= 5 {
			b.WriteString(fmt.Sprintf("And %d more effects.", len(result.Effects)-5))
			break
		}
		if i > 0 {
			b.WriteString(" ")
		}
		if e.Confidence > 0.6 {
			b.WriteString(fmt.Sprintf("%s would likely be affected.", e.Entity))
		} else {
			b.WriteString(fmt.Sprintf("%s could potentially be affected.", e.Entity))
		}
	}

	return b.String()
}

// findNodes finds nodes matching a query string.
func (gcr *GraphCausalReasoner) findNodes(query string) []string {
	lower := strings.ToLower(strings.TrimRight(strings.TrimSpace(query), "?!."))
	var ids []string

	// Exact match
	if _, ok := gcr.Graph.nodes[lower]; ok {
		ids = append(ids, lower)
	}

	// Label index
	if nodeIDs, ok := gcr.Graph.byLabel[lower]; ok {
		ids = append(ids, nodeIDs...)
	}

	// Substring match
	if len(ids) == 0 {
		for id, node := range gcr.Graph.nodes {
			if strings.Contains(strings.ToLower(node.Label), lower) {
				ids = append(ids, id)
			}
		}
	}

	return uniqueStrings(ids)
}

// -----------------------------------------------------------------------
// Temporal Correlation — find patterns across different data streams.
// -----------------------------------------------------------------------

// TemporalCorrelation finds correlations between event types.
type TemporalCorrelation struct {
	StreamA     string  // e.g., "mood"
	StreamB     string  // e.g., "expense"
	Correlation float64 // -1.0 to 1.0
	Description string
}

// FindCorrelations discovers temporal correlations between event streams.
func (ce *CausalEngine) FindCorrelations() []TemporalCorrelation {
	ce.mu.RLock()
	defer ce.mu.RUnlock()

	if len(ce.events) < 20 {
		return nil
	}

	// Group events by day and type
	type dayKey struct {
		day    string
		action string
	}
	daily := make(map[dayKey]int)
	days := make(map[string]bool)
	actions := make(map[string]bool)

	for _, e := range ce.events {
		day := e.When.Format("2006-01-02")
		days[day] = true
		actions[e.Action] = true
		daily[dayKey{day, e.Action}]++
	}

	if len(actions) < 2 || len(days) < 5 {
		return nil
	}

	// Calculate Pearson correlation between action pairs
	var correlations []TemporalCorrelation
	actionList := make([]string, 0, len(actions))
	for a := range actions {
		actionList = append(actionList, a)
	}
	dayList := make([]string, 0, len(days))
	for d := range days {
		dayList = append(dayList, d)
	}

	for i := 0; i < len(actionList); i++ {
		for j := i + 1; j < len(actionList); j++ {
			a, b := actionList[i], actionList[j]

			var xVals, yVals []float64
			for _, day := range dayList {
				xVals = append(xVals, float64(daily[dayKey{day, a}]))
				yVals = append(yVals, float64(daily[dayKey{day, b}]))
			}

			r := pearsonCorrelation(xVals, yVals)
			if math.Abs(r) > 0.3 {
				desc := ""
				if r > 0.5 {
					desc = fmt.Sprintf("Days with more %s tend to also have more %s.", a, b)
				} else if r < -0.5 {
					desc = fmt.Sprintf("Days with more %s tend to have less %s.", a, b)
				} else if r > 0.3 {
					desc = fmt.Sprintf("There's a slight positive correlation between %s and %s.", a, b)
				} else {
					desc = fmt.Sprintf("There's a slight negative correlation between %s and %s.", a, b)
				}

				correlations = append(correlations, TemporalCorrelation{
					StreamA:     a,
					StreamB:     b,
					Correlation: r,
					Description: desc,
				})
			}
		}
	}

	return correlations
}

// pearsonCorrelation computes Pearson's r between two sequences.
func pearsonCorrelation(x, y []float64) float64 {
	n := len(x)
	if n != len(y) || n < 3 {
		return 0
	}

	var sumX, sumY, sumXY, sumX2, sumY2 float64
	for i := 0; i < n; i++ {
		sumX += x[i]
		sumY += y[i]
		sumXY += x[i] * y[i]
		sumX2 += x[i] * x[i]
		sumY2 += y[i] * y[i]
	}

	nf := float64(n)
	numerator := nf*sumXY - sumX*sumY
	denominator := math.Sqrt((nf*sumX2 - sumX*sumX) * (nf*sumY2 - sumY*sumY))

	if denominator == 0 {
		return 0
	}
	return numerator / denominator
}

// isCausalRelation returns true for relation types that represent
// causal or consequential connections — used by the simulation engine
// and GraphCausalReasoner to traverse effect chains.
func isCausalRelation(rel RelType) bool {
	switch rel {
	case RelCauses, RelFollows, RelEnables, RelProduces,
		RelPrevents, RelRequires:
		return true
	default:
		return false
	}
}
