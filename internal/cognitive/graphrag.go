package cognitive

import (
	"fmt"
	"math"
	"sort"
	"strings"
)

// -----------------------------------------------------------------------
// GraphRAG — Graph-based Retrieval-Augmented Generation.
//
// Instead of returning a pre-written paragraph or generating text from
// nothing, GraphRAG activates a subgraph around the query topic,
// gathers ranked facts from multiple graph paths, and composes a
// coherent multi-source response with citations.
//
// Flow:
//   1. Activate: spread activation from query topics through the graph
//   2. Gather:   collect facts from activated nodes (weighted by activation)
//   3. Rank:     score facts by relevance, diversity, and source quality
//   4. Compose:  assemble facts into coherent prose with citations
//
// This is the key innovation over basic retrieval: instead of finding
// ONE matching paragraph, we synthesize from MANY graph paths —
// producing richer, more complete answers grounded in multiple sources.
// -----------------------------------------------------------------------

// GraphRAGEngine composes responses from knowledge graph subgraphs.
type GraphRAGEngine struct {
	Graph    *CognitiveGraph
	MultiHop *MultiHopReasoner
}

// GraphRAGResult is the output of a GraphRAG query.
type GraphRAGResult struct {
	Query      string
	Response   string        // composed natural language response
	Facts      []RankedFact  // all facts used, ranked
	Sources    []string      // source attributions
	Confidence float64       // aggregate confidence
	Coverage   float64       // how well the graph covers this topic (0.0-1.0)
}

// RankedFact is a fact with its relevance score and source.
type RankedFact struct {
	Text       string  // natural language fact
	Score      float64 // relevance score (0.0-1.0)
	Source     string  // where this fact came from
	Relation   string  // graph relation type
	Depth      int     // hops from query topic
}

// NewGraphRAGEngine creates a GraphRAG engine.
func NewGraphRAGEngine(graph *CognitiveGraph, multihop *MultiHopReasoner) *GraphRAGEngine {
	return &GraphRAGEngine{
		Graph:    graph,
		MultiHop: multihop,
	}
}

// Query activates the graph around a topic and composes a response.
func (g *GraphRAGEngine) Query(topic string) *GraphRAGResult {
	if g.Graph == nil {
		return &GraphRAGResult{
			Query:    topic,
			Response: "Knowledge graph not available.",
		}
	}

	result := &GraphRAGResult{
		Query: topic,
	}

	// Phase 1: Activate the subgraph around the topic.
	activated := g.activate(topic)

	// Phase 2: Gather facts from activated nodes.
	facts := g.gatherFacts(topic, activated)

	// Phase 3: Rank facts by relevance, diversity, and quality.
	ranked := g.rankFacts(facts, topic)
	result.Facts = ranked

	// Phase 4: Compose into coherent prose.
	if len(ranked) == 0 {
		result.Response = fmt.Sprintf("I don't have enough graph data about %s to compose a response.", topic)
		result.Confidence = 0.1
		return result
	}

	result.Response = g.compose(topic, ranked)
	result.Confidence = g.computeConfidence(ranked)
	result.Coverage = g.computeCoverage(topic, ranked)

	// Collect sources.
	sourceSet := make(map[string]bool)
	for _, f := range ranked {
		if f.Source != "" {
			sourceSet[f.Source] = true
		}
	}
	for s := range sourceSet {
		result.Sources = append(result.Sources, s)
	}

	return result
}

// -----------------------------------------------------------------------
// Phase 1: Activation — spread activation through the graph.
// -----------------------------------------------------------------------

type activatedNode struct {
	ID         string
	Label      string
	Activation float64
	Depth      int
}

func (g *GraphRAGEngine) activate(topic string) []activatedNode {
	if g.Graph == nil {
		return nil
	}

	var activated []activatedNode
	visited := make(map[string]bool)

	// Find seed nodes matching the topic.
	seeds := g.findSeeds(topic)
	if len(seeds) == 0 {
		return nil
	}

	// BFS with decaying activation.
	type qItem struct {
		id         string
		activation float64
		depth      int
	}
	queue := make([]qItem, 0, len(seeds))
	for _, id := range seeds {
		queue = append(queue, qItem{id, 1.0, 0})
		visited[id] = true
	}

	maxDepth := 3
	decayRate := 0.5

	for len(queue) > 0 {
		item := queue[0]
		queue = queue[1:]

		node := g.Graph.GetNode(item.id)
		if node == nil {
			continue
		}

		activated = append(activated, activatedNode{
			ID:         item.id,
			Label:      node.Label,
			Activation: item.activation,
			Depth:      item.depth,
		})

		if item.depth >= maxDepth {
			continue
		}

		// Spread to neighbors.
		nextAct := item.activation * decayRate
		if nextAct < 0.05 {
			continue
		}

		for _, edge := range g.Graph.outEdges[item.id] {
			if visited[edge.To] {
				continue
			}
			visited[edge.To] = true

			// Weight by edge confidence and relation importance.
			edgeWeight := edge.Confidence
			if edgeWeight == 0 {
				edgeWeight = 0.5
			}
			queue = append(queue, qItem{
				id:         edge.To,
				activation: nextAct * edgeWeight,
				depth:      item.depth + 1,
			})
		}

		// Also spread backward for context.
		for _, edge := range g.Graph.inEdges[item.id] {
			if visited[edge.From] {
				continue
			}
			visited[edge.From] = true
			queue = append(queue, qItem{
				id:         edge.From,
				activation: nextAct * 0.7, // backward activation slightly weaker
				depth:      item.depth + 1,
			})
		}
	}

	return activated
}

func (g *GraphRAGEngine) findSeeds(topic string) []string {
	topicLower := strings.ToLower(strings.TrimSpace(topic))
	if topicLower == "" {
		return nil
	}

	// Exact label match.
	if ids, ok := g.Graph.byLabel[topicLower]; ok && len(ids) > 0 {
		return ids
	}

	// Partial match — find labels containing the topic.
	var matches []string
	for label, ids := range g.Graph.byLabel {
		if strings.Contains(label, topicLower) || strings.Contains(topicLower, label) {
			matches = append(matches, ids...)
		}
	}

	// Word-level match for multi-word topics.
	if len(matches) == 0 {
		words := strings.Fields(topicLower)
		for _, w := range words {
			if len(w) < 4 {
				continue
			}
			if ids, ok := g.Graph.byLabel[w]; ok {
				matches = append(matches, ids...)
			}
		}
	}

	return matches
}

// -----------------------------------------------------------------------
// Phase 2: Gather facts from activated nodes.
// -----------------------------------------------------------------------

func (g *GraphRAGEngine) gatherFacts(topic string, activated []activatedNode) []RankedFact {
	var facts []RankedFact
	seen := make(map[string]bool)

	for _, node := range activated {
		// Get outgoing edges as facts.
		for _, edge := range g.Graph.outEdges[node.ID] {
			targetNode := g.Graph.GetNode(edge.To)
			if targetNode == nil {
				continue
			}

			factText := fmt.Sprintf("%s %s %s",
				node.Label, humanizeRelation(string(edge.Relation)), targetNode.Label)

			if seen[factText] {
				continue
			}
			seen[factText] = true

			facts = append(facts, RankedFact{
				Text:     factText,
				Score:    node.Activation * edge.Confidence,
				Source:   edge.Source,
				Relation: string(edge.Relation),
				Depth:    node.Depth,
			})
		}
	}

	return facts
}

// -----------------------------------------------------------------------
// Phase 3: Rank facts by relevance, diversity, and quality.
// -----------------------------------------------------------------------

func (g *GraphRAGEngine) rankFacts(facts []RankedFact, topic string) []RankedFact {
	if len(facts) == 0 {
		return nil
	}

	topicLower := strings.ToLower(topic)

	// Score each fact.
	for i := range facts {
		score := facts[i].Score

		// Boost facts that mention the topic directly.
		if strings.Contains(strings.ToLower(facts[i].Text), topicLower) {
			score *= 1.5
		}

		// Boost definitional relations.
		switch facts[i].Relation {
		case "is_a", "described_as":
			score *= 1.4
		case "known_for", "created_by", "founded_in":
			score *= 1.3
		case "has", "part_of", "used_for":
			score *= 1.2
		case "causes", "follows":
			score *= 1.1
		}

		// Penalize deep facts (less directly relevant).
		depthPenalty := math.Pow(0.8, float64(facts[i].Depth))
		score *= depthPenalty

		facts[i].Score = math.Min(1.0, score)
	}

	// Sort by score descending.
	sort.Slice(facts, func(i, j int) bool {
		return facts[i].Score > facts[j].Score
	})

	// Take top facts with diversity filtering.
	// Don't repeat the same relation type too many times.
	relCount := make(map[string]int)
	var diverse []RankedFact
	for _, f := range facts {
		if relCount[f.Relation] >= 3 {
			continue
		}
		relCount[f.Relation]++
		diverse = append(diverse, f)
		if len(diverse) >= 10 {
			break
		}
	}

	return diverse
}

// -----------------------------------------------------------------------
// Phase 4: Compose facts into coherent prose.
// -----------------------------------------------------------------------

func (g *GraphRAGEngine) compose(topic string, facts []RankedFact) string {
	if len(facts) == 0 {
		return ""
	}

	var b strings.Builder

	// Group facts by type for structured composition.
	var definitions []RankedFact
	var properties []RankedFact
	var relations []RankedFact
	var origins []RankedFact

	for _, f := range facts {
		switch f.Relation {
		case "is_a", "described_as":
			definitions = append(definitions, f)
		case "has", "part_of", "known_for":
			properties = append(properties, f)
		case "created_by", "founded_in", "founded_by", "influenced_by":
			origins = append(origins, f)
		default:
			relations = append(relations, f)
		}
	}

	// Opening: definition.
	if len(definitions) > 0 {
		b.WriteString(capFirst(definitions[0].Text))
		if !strings.HasSuffix(definitions[0].Text, ".") {
			b.WriteString(".")
		}
		for _, d := range definitions[1:] {
			b.WriteString(" ")
			b.WriteString(capFirst(d.Text))
			if !strings.HasSuffix(d.Text, ".") {
				b.WriteString(".")
			}
		}
	}

	// Origins.
	if len(origins) > 0 {
		b.WriteString(" ")
		for i, o := range origins {
			if i > 0 {
				b.WriteString(" ")
			}
			b.WriteString(capFirst(o.Text))
			if !strings.HasSuffix(o.Text, ".") {
				b.WriteString(".")
			}
		}
	}

	// Properties.
	if len(properties) > 0 {
		b.WriteString(" ")
		for i, p := range properties {
			if i > 0 {
				b.WriteString(" ")
			}
			b.WriteString(capFirst(p.Text))
			if !strings.HasSuffix(p.Text, ".") {
				b.WriteString(".")
			}
		}
	}

	// Other relations.
	if len(relations) > 0 && b.Len() < 800 {
		b.WriteString(" ")
		for i, r := range relations {
			if i >= 3 { // limit noise
				break
			}
			if i > 0 {
				b.WriteString(" ")
			}
			b.WriteString(capFirst(r.Text))
			if !strings.HasSuffix(r.Text, ".") {
				b.WriteString(".")
			}
		}
	}

	return strings.TrimSpace(b.String())
}

func capFirst(s string) string {
	if s == "" {
		return s
	}
	return strings.ToUpper(s[:1]) + s[1:]
}

// -----------------------------------------------------------------------
// Confidence and coverage scoring
// -----------------------------------------------------------------------

func (g *GraphRAGEngine) computeConfidence(facts []RankedFact) float64 {
	if len(facts) == 0 {
		return 0
	}

	// Aggregate: top score weighted by number of facts.
	topScore := facts[0].Score
	factBoost := math.Min(0.3, float64(len(facts))*0.05)

	return math.Min(0.95, topScore*0.7+factBoost)
}

func (g *GraphRAGEngine) computeCoverage(topic string, facts []RankedFact) float64 {
	// Coverage = how many distinct relation types we found.
	relTypes := make(map[string]bool)
	for _, f := range facts {
		relTypes[f.Relation] = true
	}

	// A well-covered topic has definitions, origins, properties, and relations.
	ideal := 4.0
	actual := float64(len(relTypes))
	return math.Min(1.0, actual/ideal)
}
