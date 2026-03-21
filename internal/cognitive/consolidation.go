package cognitive

import (
	"fmt"
	"math"
	"sort"
	"strings"
	"time"
)

// -----------------------------------------------------------------------
// Consolidation Engine — the AI "dreams".
// Runs between sessions to discover connections, merge concepts,
// strengthen paths, and generate insights.
// -----------------------------------------------------------------------

// Insight represents a discovered connection or observation.
type Insight struct {
	Type        InsightType
	Description string
	Confidence  float64
	Nodes       []string // involved node IDs
	CreatedAt   time.Time
}

// InsightType classifies what kind of insight was discovered.
type InsightType string

const (
	InsightConnection    InsightType = "connection"    // unexpected link between concepts
	InsightCluster       InsightType = "cluster"       // group of related concepts
	InsightTrend         InsightType = "trend"         // pattern over time
	InsightContradiction InsightType = "contradiction" // conflicting facts
	InsightGap           InsightType = "gap"           // missing expected knowledge
)

// Consolidator processes the cognitive graph to discover new knowledge.
type Consolidator struct {
	graph    *CognitiveGraph
	engine   *InferenceEngine
	insights []Insight
}

// NewConsolidator creates a consolidator for a graph.
func NewConsolidator(graph *CognitiveGraph) *Consolidator {
	return &Consolidator{
		graph:  graph,
		engine: NewInferenceEngine(graph),
	}
}

// Consolidate runs all consolidation steps and returns insights.
// This is designed to be called periodically (e.g., on startup, after sessions).
func (c *Consolidator) Consolidate() []Insight {
	var insights []Insight

	// 1. Run inference — derive new facts
	inferences := c.engine.RunAll()
	for _, inf := range inferences {
		if inf.Confidence > 0.3 {
			insights = append(insights, Insight{
				Type:        InsightConnection,
				Description: inf.Reason,
				Confidence:  inf.Confidence,
				CreatedAt:   time.Now(),
			})
		}
	}

	// 2. Merge similar nodes
	merges := c.MergeSimilar()
	insights = append(insights, merges...)

	// 3. Discover clusters
	clusters := c.DiscoverClusters()
	insights = append(insights, clusters...)

	// 4. Strengthen frequently co-activated paths
	c.StrengthenPaths()

	// 5. Decay old, unused edges
	c.DecayUnused()

	// 6. Cross-domain insights
	cross := c.CrossDomainInsights()
	insights = append(insights, cross...)

	c.insights = append(c.insights, insights...)
	return insights
}

// -----------------------------------------------------------------------
// Merge Similar Nodes
// Combines nodes with the same or very similar labels.
// "programming" and "Programming" → one node.
// "Go" and "Golang" → one node (with synonym tracking).
// -----------------------------------------------------------------------

// synonyms maps known synonyms to canonical forms.
var synonyms = map[string]string{
	"golang":     "go",
	"javascript": "js",
	"typescript":  "ts",
	"python3":    "python",
	"nodejs":     "node.js",
	"react.js":   "react",
	"vue.js":     "vue",
	"ai":         "artificial intelligence",
	"ml":         "machine learning",
	"db":         "database",
}

func (c *Consolidator) MergeSimilar() []Insight {
	c.graph.mu.Lock()
	defer c.graph.mu.Unlock()

	var insights []Insight

	// Find nodes that should be merged
	mergeMap := make(map[string]string) // from → to (canonical)

	for id, node := range c.graph.nodes {
		// Check synonym table
		if canonical, ok := synonyms[strings.ToLower(node.Label)]; ok {
			canonID := nodeID(canonical)
			if canonID != id && c.graph.nodes[canonID] != nil {
				mergeMap[id] = canonID
			}
		}
	}

	// Execute merges
	for fromID, toID := range mergeMap {
		fromNode := c.graph.nodes[fromID]
		toNode := c.graph.nodes[toID]
		if fromNode == nil || toNode == nil {
			continue
		}

		// Transfer edges
		for _, edge := range c.graph.outEdges[fromID] {
			edge.From = toID
			c.graph.outEdges[toID] = append(c.graph.outEdges[toID], edge)
		}
		for _, edge := range c.graph.inEdges[fromID] {
			edge.To = toID
			c.graph.inEdges[toID] = append(c.graph.inEdges[toID], edge)
		}

		// Transfer properties
		for k, v := range fromNode.Properties {
			if _, exists := toNode.Properties[k]; !exists {
				toNode.Properties[k] = v
			}
		}

		// Boost target node confidence
		toNode.Confidence = math.Min(1.0, toNode.Confidence+0.1)
		toNode.AccessCount += fromNode.AccessCount

		// Remove source node
		delete(c.graph.nodes, fromID)
		delete(c.graph.outEdges, fromID)
		delete(c.graph.inEdges, fromID)

		insights = append(insights, Insight{
			Type:        InsightConnection,
			Description: fmt.Sprintf("Merged \"%s\" into \"%s\" (synonyms)", fromNode.Label, toNode.Label),
			Confidence:  0.9,
			Nodes:       []string{toID},
			CreatedAt:   time.Now(),
		})

		c.graph.modified = true
	}

	return insights
}

// -----------------------------------------------------------------------
// Discover Clusters — find groups of tightly connected concepts.
// -----------------------------------------------------------------------

func (c *Consolidator) DiscoverClusters() []Insight {
	c.graph.mu.RLock()
	defer c.graph.mu.RUnlock()

	var insights []Insight

	// Simple clustering: find nodes that share 3+ connections
	// Build adjacency count
	connections := make(map[string]map[string]int) // nodeA → nodeB → shared edge count

	for _, edge := range c.graph.edges {
		if connections[edge.From] == nil {
			connections[edge.From] = make(map[string]int)
		}
		connections[edge.From][edge.To]++
	}

	// Find strongly connected pairs
	type clusterSeed struct {
		a, b  string
		count int
	}
	var seeds []clusterSeed
	seen := make(map[string]bool)

	for a, neighbors := range connections {
		for b, count := range neighbors {
			if count >= 2 {
				key := a + "|" + b
				if a > b {
					key = b + "|" + a
				}
				if !seen[key] {
					seen[key] = true
					seeds = append(seeds, clusterSeed{a, b, count})
				}
			}
		}
	}

	// Group overlapping seeds into clusters
	for _, seed := range seeds {
		aNode := c.graph.nodes[seed.a]
		bNode := c.graph.nodes[seed.b]
		if aNode == nil || bNode == nil {
			continue
		}

		// Find other nodes connected to both A and B
		aNeighbors := make(map[string]bool)
		for _, e := range c.graph.outEdges[seed.a] {
			aNeighbors[e.To] = true
		}

		var cluster []string
		cluster = append(cluster, aNode.Label, bNode.Label)
		for _, e := range c.graph.outEdges[seed.b] {
			if aNeighbors[e.To] {
				if n := c.graph.nodes[e.To]; n != nil {
					cluster = append(cluster, n.Label)
				}
			}
		}

		if len(cluster) >= 3 {
			insights = append(insights, Insight{
				Type:        InsightCluster,
				Description: fmt.Sprintf("Knowledge cluster: %s", strings.Join(cluster, ", ")),
				Confidence:  0.6,
				Nodes:       []string{seed.a, seed.b},
				CreatedAt:   time.Now(),
			})
		}
	}

	return insights
}

// -----------------------------------------------------------------------
// Strengthen Paths — boost edges between frequently co-activated nodes.
// -----------------------------------------------------------------------

func (c *Consolidator) StrengthenPaths() {
	c.graph.mu.Lock()
	defer c.graph.mu.Unlock()

	for _, edge := range c.graph.edges {
		from := c.graph.nodes[edge.From]
		to := c.graph.nodes[edge.To]
		if from == nil || to == nil {
			continue
		}

		// If both nodes have been accessed frequently, strengthen the edge
		combined := from.AccessCount + to.AccessCount
		if combined > 10 {
			boost := float64(combined) * 0.001
			edge.Weight = math.Min(1.0, edge.Weight+boost)
			c.graph.modified = true
		}
	}
}

// -----------------------------------------------------------------------
// Decay Unused — weaken edges that haven't been accessed.
// -----------------------------------------------------------------------

func (c *Consolidator) DecayUnused() {
	c.graph.mu.Lock()
	defer c.graph.mu.Unlock()

	now := time.Now()
	threshold := 30 * 24 * time.Hour // 30 days

	var activeEdges []*CogEdge
	for _, edge := range c.graph.edges {
		age := now.Sub(edge.CreatedAt)
		if age > threshold && edge.Weight < 0.3 {
			// Remove very old, weak edges
			continue
		}
		if age > threshold {
			edge.Weight *= 0.95 // gentle decay
		}
		activeEdges = append(activeEdges, edge)
	}

	if len(activeEdges) < len(c.graph.edges) {
		c.graph.edges = activeEdges
		// Rebuild indexes
		c.graph.outEdges = make(map[string][]*CogEdge)
		c.graph.inEdges = make(map[string][]*CogEdge)
		for _, e := range activeEdges {
			c.graph.outEdges[e.From] = append(c.graph.outEdges[e.From], e)
			c.graph.inEdges[e.To] = append(c.graph.inEdges[e.To], e)
		}
		c.graph.modified = true
	}
}

// -----------------------------------------------------------------------
// Cross-Domain Insights — find unexpected connections across domains.
// -----------------------------------------------------------------------

func (c *Consolidator) CrossDomainInsights() []Insight {
	c.graph.mu.RLock()
	defer c.graph.mu.RUnlock()

	var insights []Insight

	// Find nodes that bridge different domains
	// A "bridge" node has edges to nodes in different domains
	domainEdges := make(map[string]map[string]bool) // nodeID → set of domains

	for _, edge := range c.graph.edges {
		if edge.Relation == RelDomain {
			if domainEdges[edge.From] == nil {
				domainEdges[edge.From] = make(map[string]bool)
			}
			domainEdges[edge.From][edge.To] = true
		}
	}

	for nodeID, domains := range domainEdges {
		if len(domains) >= 2 {
			node := c.graph.nodes[nodeID]
			if node == nil {
				continue
			}
			domainList := make([]string, 0, len(domains))
			for d := range domains {
				if n := c.graph.nodes[d]; n != nil {
					domainList = append(domainList, n.Label)
				} else {
					domainList = append(domainList, d)
				}
			}
			sort.Strings(domainList)

			insights = append(insights, Insight{
				Type:        InsightConnection,
				Description: fmt.Sprintf("\"%s\" bridges domains: %s", node.Label, strings.Join(domainList, ", ")),
				Confidence:  0.5,
				Nodes:       []string{nodeID},
				CreatedAt:   time.Now(),
			})
		}
	}

	return insights
}

// RecentInsights returns the most recent insights.
func (c *Consolidator) RecentInsights(n int) []Insight {
	if len(c.insights) <= n {
		return c.insights
	}
	return c.insights[len(c.insights)-n:]
}

