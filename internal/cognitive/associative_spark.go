package cognitive

import (
	"fmt"
	"math/rand"
	"strings"
	"sync"
	"time"
)

// -----------------------------------------------------------------------
// Associative Spark Engine — surfaces unexpected connections between
// concepts during conversation. This is the "I just noticed..." system
// that makes Nous feel like it's genuinely thinking, not just reacting.
// -----------------------------------------------------------------------

// AssociativeSpark represents an unexpected connection discovered in the knowledge graph.
type AssociativeSpark struct {
	Source      string   // concept from current conversation
	Target      string   // distant concept that's surprisingly connected
	Path        []string // chain of concepts connecting them
	Relations   []string // relation types along the path
	Novelty     float64  // 0.0-1.0, how unexpected this connection is
	Explanation string   // human-readable: "X connects to Y through Z"
}

// SparkEngine discovers and surfaces novel associations.
type SparkEngine struct {
	graph        *CognitiveGraph
	recentTopics []string             // rolling window of conversation topics (last 20)
	surfaced     map[string]time.Time // "source→target" → last surfaced time
	cooldownDur  time.Duration        // how long before same spark can re-fire
	minNovelty   float64              // minimum novelty score to surface
	maxPerTurn   int                  // max sparks per turn
	mu           sync.Mutex
	rng          *rand.Rand
}

// NewSparkEngine creates a spark engine attached to the given knowledge graph.
func NewSparkEngine(graph *CognitiveGraph) *SparkEngine {
	return &SparkEngine{
		graph:       graph,
		surfaced:    make(map[string]time.Time),
		cooldownDur: 24 * time.Hour,
		minNovelty:  0.6,
		maxPerTurn:  1,
		rng:         rand.New(rand.NewSource(time.Now().UnixNano())),
	}
}

// discoveredPath is a path found during BFS exploration.
type discoveredPath struct {
	nodeIDs   []string // full chain of node IDs from source to target
	relations []string // relation type at each hop
}

// maxVisited caps BFS exploration to keep it fast.
const maxVisited = 200

// hubEdgeThreshold — nodes with more edges than this are generic hubs.
const hubEdgeThreshold = 30

// -----------------------------------------------------------------------
// Core API
// -----------------------------------------------------------------------

// Ignite runs spreading exploration from the current topics and surfaces
// novel, unexpected connections. Returns nil when nothing good is found —
// silence is better than forced connections.
func (se *SparkEngine) Ignite(currentTopics []string) []AssociativeSpark {
	se.mu.Lock()
	defer se.mu.Unlock()

	now := time.Now()

	// Collect candidate sparks from all topics.
	var candidates []AssociativeSpark

	for _, topic := range currentTopics {
		id := nodeID(topic)
		if se.graph.GetNode(id) == nil {
			continue
		}

		paths := se.findPaths(id, 5)
		for _, p := range paths {
			target := p.nodeIDs[len(p.nodeIDs)-1]
			sourceLabel := se.nodeLabel(id)
			targetLabel := se.nodeLabel(target)

			// Check cooldown.
			key := sourceLabel + "→" + targetLabel
			if t, ok := se.surfaced[key]; ok && now.Sub(t) < se.cooldownDur {
				continue
			}

			novelty := se.scoreNovelty(id, target, p.nodeIDs, p.relations)
			if novelty < se.minNovelty {
				continue
			}

			// Skip paths that are too long (5+ hops usually produce
			// absurd chains like "gravity → Einstein → Germany → cheese")
			if len(p.nodeIDs) > 4 {
				continue
			}

			labels := se.pathLabels(p.nodeIDs)
			explanation := se.buildExplanation(sourceLabel, targetLabel, labels, p.relations)

			candidates = append(candidates, AssociativeSpark{
				Source:      sourceLabel,
				Target:      targetLabel,
				Path:        labels,
				Relations:   p.relations,
				Novelty:     novelty,
				Explanation: explanation,
			})
		}
	}

	if len(candidates) == 0 {
		return nil
	}

	// Sort by novelty descending.
	for i := 1; i < len(candidates); i++ {
		for j := i; j > 0 && candidates[j].Novelty > candidates[j-1].Novelty; j-- {
			candidates[j], candidates[j-1] = candidates[j-1], candidates[j]
		}
	}

	// Take up to maxPerTurn.
	limit := se.maxPerTurn
	if limit > len(candidates) {
		limit = len(candidates)
	}
	result := candidates[:limit]

	// Record that we surfaced these.
	for _, spark := range result {
		se.surfaced[spark.Source+"→"+spark.Target] = now
	}

	return result
}

// RecordTopics adds topics to the rolling window (keeps last 20).
func (se *SparkEngine) RecordTopics(topics []string) {
	se.mu.Lock()
	defer se.mu.Unlock()

	se.recentTopics = append(se.recentTopics, topics...)
	if len(se.recentTopics) > 20 {
		se.recentTopics = se.recentTopics[len(se.recentTopics)-20:]
	}
}

// RecordSurfaced marks a spark as recently surfaced for cooldown.
func (se *SparkEngine) RecordSurfaced(source, target string) {
	se.mu.Lock()
	defer se.mu.Unlock()
	se.surfaced[source+"→"+target] = time.Now()
}

// RecentTopics returns a copy of the current topic window.
func (se *SparkEngine) RecentTopics() []string {
	se.mu.Lock()
	defer se.mu.Unlock()
	out := make([]string, len(se.recentTopics))
	copy(out, se.recentTopics)
	return out
}

// -----------------------------------------------------------------------
// Path finding — BFS with depth tracking
// -----------------------------------------------------------------------

// findPaths does BFS from source, returning paths at depth 3-5.
// Skips depth 1-2 because those connections are obvious.
// Caps exploration at maxVisited nodes.
func (se *SparkEngine) findPaths(sourceID string, maxDepth int) []discoveredPath {
	se.graph.mu.RLock()
	defer se.graph.mu.RUnlock()

	type queueItem struct {
		id       string
		depth    int
		pathIDs  []string
		pathRels []string
	}

	visited := map[string]bool{sourceID: true}
	queue := []queueItem{{
		id:      sourceID,
		depth:   0,
		pathIDs: []string{sourceID},
	}}

	var results []discoveredPath
	visitCount := 1

	for len(queue) > 0 && visitCount < maxVisited {
		item := queue[0]
		queue = queue[1:]

		if item.depth >= maxDepth {
			continue
		}

		// Explore outgoing edges.
		for _, edge := range se.graph.outEdges[item.id] {
			if visited[edge.To] {
				continue
			}
			if se.graph.nodes[edge.To] == nil {
				continue
			}
			visited[edge.To] = true
			visitCount++

			newPath := make([]string, len(item.pathIDs)+1)
			copy(newPath, item.pathIDs)
			newPath[len(item.pathIDs)] = edge.To

			newRels := make([]string, len(item.pathRels)+1)
			copy(newRels, item.pathRels)
			newRels[len(item.pathRels)] = string(edge.Relation)

			nextDepth := item.depth + 1

			// Only collect paths at depth 3-5 (skip obvious 1-2 hop connections).
			if nextDepth >= 3 {
				results = append(results, discoveredPath{
					nodeIDs:   newPath,
					relations: newRels,
				})
			}

			if visitCount < maxVisited {
				queue = append(queue, queueItem{
					id:       edge.To,
					depth:    nextDepth,
					pathIDs:  newPath,
					pathRels: newRels,
				})
			}
		}

		// Also explore incoming edges (follow links backwards).
		for _, edge := range se.graph.inEdges[item.id] {
			if visited[edge.From] {
				continue
			}
			if se.graph.nodes[edge.From] == nil {
				continue
			}
			visited[edge.From] = true
			visitCount++

			newPath := make([]string, len(item.pathIDs)+1)
			copy(newPath, item.pathIDs)
			newPath[len(item.pathIDs)] = edge.From

			newRels := make([]string, len(item.pathRels)+1)
			copy(newRels, item.pathRels)
			newRels[len(item.pathRels)] = string(edge.Relation)

			nextDepth := item.depth + 1

			if nextDepth >= 3 {
				results = append(results, discoveredPath{
					nodeIDs:   newPath,
					relations: newRels,
				})
			}

			if visitCount < maxVisited {
				queue = append(queue, queueItem{
					id:       edge.From,
					depth:    nextDepth,
					pathIDs:  newPath,
					pathRels: newRels,
				})
			}
		}
	}

	return results
}

// -----------------------------------------------------------------------
// Novelty scoring
// -----------------------------------------------------------------------

// scoreNovelty rates how surprising a connection is (0.0-1.0).
func (se *SparkEngine) scoreNovelty(sourceID, targetID string, pathIDs []string, relations []string) float64 {
	score := 0.0

	// 1. Path length bonus — longer = more surprising.
	hops := len(pathIDs) - 1 // number of edges
	switch {
	case hops >= 5:
		score += 0.35
	case hops >= 4:
		score += 0.25
	case hops >= 3:
		score += 0.15
	default:
		return 0 // too short to be interesting
	}

	// 2. Cross-domain bonus.
	if se.crossDomain(sourceID, targetID) {
		score += 0.25
	}

	// 3. Relation diversity bonus — crossing relation types is more interesting
	// than chains of the same relation.
	if len(relations) > 0 {
		relSet := make(map[string]bool)
		for _, r := range relations {
			relSet[r] = true
		}
		diversity := float64(len(relSet)) / float64(len(relations))
		score += diversity * 0.2
	}

	// 4. Obscurity bonus — target nodes with low access count are more novel.
	se.graph.mu.RLock()
	targetNode := se.graph.nodes[targetID]
	se.graph.mu.RUnlock()
	if targetNode != nil {
		switch {
		case targetNode.AccessCount <= 1:
			score += 0.15
		case targetNode.AccessCount <= 5:
			score += 0.10
		case targetNode.AccessCount <= 10:
			score += 0.05
		}
	}

	// 5. Hub penalty — paths through generic hub nodes are less interesting.
	hubCount := 0
	for _, id := range pathIDs[1 : len(pathIDs)-1] { // skip source and target
		if se.isHubNode(id) {
			hubCount++
		}
	}
	score -= float64(hubCount) * 0.15

	// Clamp to [0, 1].
	if score < 0 {
		score = 0
	}
	if score > 1 {
		score = 1
	}
	return score
}

// -----------------------------------------------------------------------
// Explanation generation
// -----------------------------------------------------------------------

// relationVerb returns a human-readable verb for a relation type.
func relationVerb(rel string) string {
	switch RelType(rel) {
	case RelIsA:
		return "is a type of"
	case RelPartOf:
		return "is part of"
	case RelHas:
		return "has"
	case RelRelatedTo:
		return "is related to"
	case RelUsedFor:
		return "is used for"
	case RelCreatedBy:
		return "was created by"
	case RelLocatedIn:
		return "is located in"
	case RelFoundedIn:
		return "was founded in"
	case RelFoundedBy:
		return "was founded by"
	case RelSimilarTo:
		return "is similar to"
	case RelCauses:
		return "causes"
	case RelContradicts:
		return "contradicts"
	case RelFollows:
		return "follows"
	case RelDescribedAs:
		return "is described as"
	case RelOffers:
		return "offers"
	case RelDomain:
		return "belongs to"
	default:
		return "connects to"
	}
}

// sparkOpenings are varied sentence starters so the engine doesn't sound repetitive.
var sparkOpenings = []string{
	"Interesting —",
	"I just noticed —",
	"There's a connection here —",
	"This might be relevant —",
	"By the way —",
}

// buildExplanation creates a human-readable explanation of a spark.
func (se *SparkEngine) buildExplanation(source, target string, pathLabels []string, relations []string) string {
	opening := sparkOpenings[se.rng.Intn(len(sparkOpenings))]

	hops := len(pathLabels) - 1 // number of edges in the path

	switch {
	case hops <= 3 && hops >= 2:
		// Short path: "Interesting — [source] connects to [target] through [middle]."
		middles := pathLabels[1 : len(pathLabels)-1]
		return fmt.Sprintf("%s %s connects to %s through %s.",
			opening, source, target, strings.Join(middles, " and "))

	case hops == 4:
		// Medium path: "[source] relates to [mid1], which [rel] [mid2], which connects to [target]."
		mid1 := pathLabels[1]
		mid2 := pathLabels[2]
		verb := relationVerb(relations[2])
		return fmt.Sprintf("%s %s relates to %s, which %s %s, which connects to %s.",
			opening, source, mid1, verb, mid2, target)

	default:
		// Long path: summarize the chain.
		middles := pathLabels[1 : len(pathLabels)-1]
		chain := strings.Join(middles, " → ")
		return fmt.Sprintf("%s %s connects to %s through a chain: %s.",
			opening, source, target, chain)
	}
}

// -----------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------

// crossDomain checks if source and target belong to different domains.
// Uses RelDomain edges if present, falls back to NodeType comparison,
// then checks the Properties["domain"] field.
func (se *SparkEngine) crossDomain(sourceID, targetID string) bool {
	se.graph.mu.RLock()
	defer se.graph.mu.RUnlock()

	srcDomain := se.nodeDomain(sourceID)
	tgtDomain := se.nodeDomain(targetID)

	// If we can't determine domains, they're not cross-domain.
	if srcDomain == "" || tgtDomain == "" {
		return false
	}
	return srcDomain != tgtDomain
}

// nodeDomain determines what domain a node belongs to.
// Must be called with graph.mu held.
func (se *SparkEngine) nodeDomain(id string) string {
	// Check RelDomain edges first.
	for _, edge := range se.graph.outEdges[id] {
		if edge.Relation == RelDomain {
			if target := se.graph.nodes[edge.To]; target != nil {
				return strings.ToLower(target.Label)
			}
		}
	}

	// Check Properties["domain"].
	node := se.graph.nodes[id]
	if node == nil {
		return ""
	}
	if d, ok := node.Properties["domain"]; ok && d != "" {
		return strings.ToLower(d)
	}

	// Fall back to NodeType — coarse but better than nothing.
	return string(node.Type)
}

// isHubNode returns true if a node has many edges (generic hub).
func (se *SparkEngine) isHubNode(id string) bool {
	se.graph.mu.RLock()
	defer se.graph.mu.RUnlock()
	return len(se.graph.outEdges[id])+len(se.graph.inEdges[id]) > hubEdgeThreshold
}

// nodeLabel returns the human-readable label for a node ID.
func (se *SparkEngine) nodeLabel(id string) string {
	node := se.graph.GetNode(id)
	if node == nil {
		return id
	}
	return node.Label
}

// pathLabels converts a slice of node IDs to their labels.
func (se *SparkEngine) pathLabels(ids []string) []string {
	labels := make([]string, len(ids))
	for i, id := range ids {
		labels[i] = se.nodeLabel(id)
	}
	return labels
}
