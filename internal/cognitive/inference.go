package cognitive

import (
	"fmt"
	"strings"
	"sync"
	"time"
)

// -----------------------------------------------------------------------
// Inference Engine — derives NEW knowledge from existing graph structure.
// Pure code. No LLM. The graph reasons about itself.
// -----------------------------------------------------------------------

// Inference represents a derived fact.
type Inference struct {
	Subject    string
	Relation   RelType
	Object     string
	Confidence float64
	Reason     string // human-readable explanation of how we got here
}

// InferenceEngine derives new knowledge from the cognitive graph.
type InferenceEngine struct {
	graph *CognitiveGraph
}

// NewInferenceEngine creates an inference engine for a graph.
func NewInferenceEngine(graph *CognitiveGraph) *InferenceEngine {
	return &InferenceEngine{graph: graph}
}

// RunAll runs all inference types and adds discovered facts to the graph.
// Returns the inferences discovered.
func (ie *InferenceEngine) RunAll() []Inference {
	var all []Inference
	all = append(all, ie.Transitive()...)
	all = append(all, ie.Analogical()...)
	all = append(all, ie.Contradictions()...)
	return all
}

// -----------------------------------------------------------------------
// Transitive Inference
// A is_in B, B is_in C → A is_in C
// A is_a B, B is_a C → A is_a C
// A part_of B, B part_of C → A part_of C
// -----------------------------------------------------------------------

func (ie *InferenceEngine) Transitive() []Inference {
	ie.graph.mu.Lock()
	defer ie.graph.mu.Unlock()

	var inferences []Inference

	for _, edge1 := range ie.graph.edges {
		if edge1.Inferred {
			continue
		}
		if !transitiveRels[edge1.Relation] {
			continue
		}

		// Find edges from edge1.To with the same relation type
		for _, edge2 := range ie.graph.outEdges[edge1.To] {
			if edge2.Inferred {
				continue
			}
			if edge2.Relation != edge1.Relation {
				continue
			}
			// A→B→C: check if A→C already exists
			if ie.hasEdgeLocked(edge1.From, edge2.To, edge1.Relation) {
				continue
			}

			// New inference: A → C
			fromNode := ie.graph.nodes[edge1.From]
			midNode := ie.graph.nodes[edge1.To]
			toNode := ie.graph.nodes[edge2.To]
			if fromNode == nil || midNode == nil || toNode == nil {
				continue
			}

			confidence := edge1.Confidence * edge2.Confidence * 0.8 // decay with chain length
			reason := fmt.Sprintf("%s %s %s, and %s %s %s → %s %s %s",
				fromNode.Label, edge1.Relation, midNode.Label,
				midNode.Label, edge2.Relation, toNode.Label,
				fromNode.Label, edge1.Relation, toNode.Label,
			)

			ie.graph.addEdgeLocked(edge1.From, edge2.To, edge1.Relation,
				"inference:transitive", confidence, true)

			inferences = append(inferences, Inference{
				Subject:    fromNode.Label,
				Relation:   edge1.Relation,
				Object:     toNode.Label,
				Confidence: confidence,
				Reason:     reason,
			})
		}
	}

	return inferences
}

// -----------------------------------------------------------------------
// Analogical Inference
// If A is_similar_to B, and A has_property P, then B might have P too.
// If A and B share the same is_a, they might share other properties.
// -----------------------------------------------------------------------

func (ie *InferenceEngine) Analogical() []Inference {
	ie.graph.mu.Lock()
	defer ie.graph.mu.Unlock()

	var inferences []Inference

	// Find nodes that share the same "is_a" parent
	isaGroups := make(map[string][]string) // parent → children
	for _, edge := range ie.graph.edges {
		if edge.Relation == RelIsA && !edge.Inferred {
			isaGroups[edge.To] = append(isaGroups[edge.To], edge.From)
		}
	}

	for _, siblings := range isaGroups {
		if len(siblings) < 2 {
			continue
		}

		// For each sibling pair, check for transferable properties
		for i := 0; i < len(siblings); i++ {
			for j := i + 1; j < len(siblings); j++ {
				aID := siblings[i]
				bID := siblings[j]

				// Properties of A
				for _, edge := range ie.graph.outEdges[aID] {
					if edge.Inferred || edge.Relation == RelIsA {
						continue
					}
					// Does B have this relation to the same object?
					if !ie.hasEdgeLocked(bID, edge.To, edge.Relation) {
						aNode := ie.graph.nodes[aID]
						bNode := ie.graph.nodes[bID]
						objNode := ie.graph.nodes[edge.To]
						if aNode == nil || bNode == nil || objNode == nil {
							continue
						}

						// Only infer for certain safe relation types
						if edge.Relation == RelDomain || edge.Relation == RelRelatedTo {
							confidence := edge.Confidence * 0.4 // lower confidence for analogy
							reason := fmt.Sprintf("%s and %s are both in the same category; %s %s %s → %s might also %s %s",
								aNode.Label, bNode.Label,
								aNode.Label, edge.Relation, objNode.Label,
								bNode.Label, edge.Relation, objNode.Label,
							)

							ie.graph.addEdgeLocked(bID, edge.To, edge.Relation,
								"inference:analogy", confidence, true)

							inferences = append(inferences, Inference{
								Subject:    bNode.Label,
								Relation:   edge.Relation,
								Object:     objNode.Label,
								Confidence: confidence,
								Reason:     reason,
							})
						}
					}
				}
			}
		}
	}

	return inferences
}

// -----------------------------------------------------------------------
// Contradiction Detection
// Find conflicting facts: X is_a A, X is_a B where A contradicts B.
// Or: X prefers A, X dislikes A.
// -----------------------------------------------------------------------

func (ie *InferenceEngine) Contradictions() []Inference {
	ie.graph.mu.Lock()
	defer ie.graph.mu.Unlock()

	var inferences []Inference

	for nodeID, edges := range ie.graph.outEdges {
		node := ie.graph.nodes[nodeID]
		if node == nil {
			continue
		}

		// Check prefers vs dislikes
		prefers := make(map[string]bool)
		dislikes := make(map[string]bool)
		for _, e := range edges {
			if e.Relation == RelPrefers {
				prefers[e.To] = true
			}
			if e.Relation == RelDislikes {
				dislikes[e.To] = true
			}
		}
		for objID := range prefers {
			if dislikes[objID] {
				objNode := ie.graph.nodes[objID]
				if objNode == nil {
					continue
				}

				// Mark contradiction
				ie.graph.addEdgeLocked(nodeID+"_prefers_"+objID, nodeID+"_dislikes_"+objID,
					RelContradicts, "inference:contradiction", 0.9, true)

				inferences = append(inferences, Inference{
					Subject:    node.Label,
					Relation:   RelContradicts,
					Object:     objNode.Label,
					Confidence: 0.9,
					Reason:     fmt.Sprintf("%s both prefers and dislikes %s — contradiction detected", node.Label, objNode.Label),
				})
			}
		}

		// Check location contradictions: X located_in A and X located_in B (different cities)
		var locations []string
		for _, e := range edges {
			if e.Relation == RelLocatedIn && !e.Inferred {
				locations = append(locations, e.To)
			}
		}
		if len(locations) > 1 {
			// Multiple non-inferred locations might be a contradiction
			// (unless one is a city and one is a country — that's transitive, not contradiction)
			// For now, just flag it
			locNames := make([]string, len(locations))
			for i, id := range locations {
				if n, ok := ie.graph.nodes[id]; ok {
					locNames[i] = n.Label
				}
			}
			inferences = append(inferences, Inference{
				Subject:    node.Label,
				Relation:   RelContradicts,
				Object:     strings.Join(locNames, " vs "),
				Confidence: 0.3, // low confidence — might be valid hierarchy
				Reason:     fmt.Sprintf("%s has multiple locations: %s — verify if these are hierarchical or contradictory", node.Label, strings.Join(locNames, ", ")),
			})
		}
	}

	return inferences
}

// -----------------------------------------------------------------------
// Targeted Query-Time Inference
// These methods run inference starting from specific nodes instead of the
// entire graph. Much faster for query-time use.
// -----------------------------------------------------------------------

// TransitiveFrom runs transitive inference starting ONLY from the given nodes.
// It follows transitive relations (is_a, located_in, part_of) up to 3 hops,
// decaying confidence by 0.8 per hop. Skips edges that already exist.
func (ie *InferenceEngine) TransitiveFrom(nodeIDs []string) []Inference {
	ie.graph.mu.Lock()
	defer ie.graph.mu.Unlock()

	var inferences []Inference
	type inferKey struct{ from, to string; rel RelType }
	seen := make(map[inferKey]bool)

	for _, startID := range nodeIDs {
		// BFS along transitive relations from this node
		type hop struct {
			id         string
			depth      int
			confidence float64
			chain      []string // labels along the path
		}
		queue := []hop{{id: startID, depth: 0, confidence: 1.0, chain: nil}}
		visited := map[string]bool{startID: true}

		for len(queue) > 0 {
			cur := queue[0]
			queue = queue[1:]

			if cur.depth >= 3 {
				continue
			}

			for _, edge := range ie.graph.outEdges[cur.id] {
				if !transitiveRels[edge.Relation] {
					continue
				}
				if visited[edge.To] {
					continue
				}
				visited[edge.To] = true

				nextConf := cur.confidence * edge.Confidence * 0.8
				nextChain := append(append([]string(nil), cur.chain...), cur.id)

				// For every ancestor in the chain, we can infer a transitive edge
				// from startID to this node (and from any intermediate to this node).
				// But the primary use case: startID -> edge.To
				if cur.depth > 0 || edge.To != startID {
					// We have at least a 2-hop path: startID -> ... -> cur.id -> edge.To
					// Check whether startID already connects to edge.To via this relation
					key := inferKey{startID, edge.To, edge.Relation}
					if !seen[key] && !ie.hasEdgeLocked(startID, edge.To, edge.Relation) {
						seen[key] = true

						fromNode := ie.graph.nodes[startID]
						toNode := ie.graph.nodes[edge.To]
						if fromNode != nil && toNode != nil {
							// Build a chain description
							chainLabels := make([]string, 0, len(nextChain)+1)
							for _, cid := range nextChain {
								if n := ie.graph.nodes[cid]; n != nil {
									chainLabels = append(chainLabels, n.Label)
								}
							}
							chainLabels = append(chainLabels, toNode.Label)

							reason := fmt.Sprintf("transitive %s: %s", edge.Relation,
								strings.Join(chainLabels, " → "))

							ie.graph.addEdgeLocked(startID, edge.To, edge.Relation,
								"inference:transitive_targeted", nextConf, true)

							inferences = append(inferences, Inference{
								Subject:    fromNode.Label,
								Relation:   edge.Relation,
								Object:     toNode.Label,
								Confidence: nextConf,
								Reason:     reason,
							})
						}
					}
				}

				queue = append(queue, hop{
					id:         edge.To,
					depth:      cur.depth + 1,
					confidence: nextConf,
					chain:      nextChain,
				})
			}
		}
	}

	return inferences
}

// AnalogicalFrom runs analogical inference only from the given nodes.
// It finds siblings (nodes sharing the same is_a parent) and transfers
// applicable properties (RelDomain, RelRelatedTo) with confidence 0.4.
func (ie *InferenceEngine) AnalogicalFrom(nodeIDs []string) []Inference {
	ie.graph.mu.Lock()
	defer ie.graph.mu.Unlock()

	var inferences []Inference
	type inferKey struct{ from, to string; rel RelType }
	seen := make(map[inferKey]bool)

	startSet := make(map[string]bool, len(nodeIDs))
	for _, id := range nodeIDs {
		startSet[id] = true
	}

	for _, id := range nodeIDs {
		// Find is_a parents of this node
		for _, edge := range ie.graph.outEdges[id] {
			if edge.Relation != RelIsA {
				continue
			}
			parentID := edge.To

			// Find siblings: other nodes that also is_a this parent
			for _, inEdge := range ie.graph.inEdges[parentID] {
				if inEdge.Relation != RelIsA || inEdge.From == id {
					continue
				}
				siblingID := inEdge.From

				// Transfer properties from this node to sibling
				for _, propEdge := range ie.graph.outEdges[id] {
					if propEdge.Inferred || propEdge.Relation == RelIsA {
						continue
					}
					if propEdge.Relation != RelDomain && propEdge.Relation != RelRelatedTo {
						continue
					}

					key := inferKey{siblingID, propEdge.To, propEdge.Relation}
					if seen[key] || ie.hasEdgeLocked(siblingID, propEdge.To, propEdge.Relation) {
						continue
					}
					seen[key] = true

					srcNode := ie.graph.nodes[id]
					sibNode := ie.graph.nodes[siblingID]
					objNode := ie.graph.nodes[propEdge.To]
					if srcNode == nil || sibNode == nil || objNode == nil {
						continue
					}

					confidence := propEdge.Confidence * 0.4
					reason := fmt.Sprintf("%s and %s are both %s; %s %s %s → %s might also %s %s",
						srcNode.Label, sibNode.Label, ie.graph.nodes[parentID].Label,
						srcNode.Label, propEdge.Relation, objNode.Label,
						sibNode.Label, propEdge.Relation, objNode.Label,
					)

					ie.graph.addEdgeLocked(siblingID, propEdge.To, propEdge.Relation,
						"inference:analogy_targeted", confidence, true)

					inferences = append(inferences, Inference{
						Subject:    sibNode.Label,
						Relation:   propEdge.Relation,
						Object:     objNode.Label,
						Confidence: confidence,
						Reason:     reason,
					})
				}
			}
		}
	}

	return inferences
}

// InferAt runs both TransitiveFrom and AnalogicalFrom from the given nodes
// and returns combined, deduplicated results.
func (ie *InferenceEngine) InferAt(nodeIDs []string) []Inference {
	trans := ie.TransitiveFrom(nodeIDs)
	analog := ie.AnalogicalFrom(nodeIDs)

	type inferKey struct {
		subj string
		rel  RelType
		obj  string
	}
	seen := make(map[inferKey]bool, len(trans))
	var combined []Inference

	for _, inf := range trans {
		key := inferKey{inf.Subject, inf.Relation, inf.Object}
		if !seen[key] {
			seen[key] = true
			combined = append(combined, inf)
		}
	}
	for _, inf := range analog {
		key := inferKey{inf.Subject, inf.Relation, inf.Object}
		if !seen[key] {
			seen[key] = true
			combined = append(combined, inf)
		}
	}

	return combined
}

// hasEdgeLocked checks if an edge exists (caller must hold lock).
func (ie *InferenceEngine) hasEdgeLocked(from, to string, rel RelType) bool {
	for _, e := range ie.graph.outEdges[from] {
		if e.To == to && e.Relation == rel {
			return true
		}
	}
	return false
}

// -----------------------------------------------------------------------
// Temporal Inference — pattern detection from user actions.
// -----------------------------------------------------------------------

// TemporalPattern represents a detected user behavior pattern.
type TemporalPattern struct {
	Action    string    // what the user does (e.g., "journal", "weather")
	Hour      int       // typical hour (0-23)
	DayOfWeek int       // -1 for any day, 0-6 for specific day
	Count     int       // how many times observed
	LastSeen  time.Time // last occurrence
}

// PatternDetector finds patterns in user behavior over time.
type PatternDetector struct {
	actions []userAction
	mu      sync.Mutex
}

type userAction struct {
	action string
	when   time.Time
}

// NewPatternDetector creates a new pattern detector.
func NewPatternDetector() *PatternDetector {
	return &PatternDetector{}
}

// RecordAction logs a user action with timestamp.
func (pd *PatternDetector) RecordAction(action string) {
	pd.mu.Lock()
	defer pd.mu.Unlock()
	pd.actions = append(pd.actions, userAction{action: action, when: time.Now()})

	// Keep last 1000 actions
	if len(pd.actions) > 1000 {
		pd.actions = pd.actions[len(pd.actions)-1000:]
	}
}

// DetectPatterns finds recurring behavioral patterns.
func (pd *PatternDetector) DetectPatterns() []TemporalPattern {
	pd.mu.Lock()
	defer pd.mu.Unlock()

	if len(pd.actions) < 5 {
		return nil
	}

	// Count action-hour pairs
	type hourKey struct {
		action string
		hour   int
	}
	counts := make(map[hourKey]int)
	lastSeen := make(map[hourKey]time.Time)

	for _, a := range pd.actions {
		key := hourKey{a.action, a.when.Hour()}
		counts[key]++
		if a.when.After(lastSeen[key]) {
			lastSeen[key] = a.when
		}
	}

	// A pattern requires at least 3 occurrences
	var patterns []TemporalPattern
	for key, count := range counts {
		if count >= 3 {
			patterns = append(patterns, TemporalPattern{
				Action:    key.action,
				Hour:      key.hour,
				DayOfWeek: -1,
				Count:     count,
				LastSeen:  lastSeen[key],
			})
		}
	}

	return patterns
}

// Anticipate returns actions the user is likely to take right now.
func (pd *PatternDetector) Anticipate() []string {
	patterns := pd.DetectPatterns()
	now := time.Now()
	hour := now.Hour()

	var likely []string
	for _, p := range patterns {
		if p.Hour == hour && p.Count >= 3 {
			likely = append(likely, p.Action)
		}
	}
	return likely
}
