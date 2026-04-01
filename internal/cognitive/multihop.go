package cognitive

import (
	"fmt"
	"strings"
)

// MultiHopReasoner traverses the knowledge graph to find connections
// between entities, answer comparative questions, and discover paths.
type MultiHopReasoner struct {
	Graph *CognitiveGraph
}

// NewMultiHopReasoner creates a reasoner backed by the given graph.
func NewMultiHopReasoner(graph *CognitiveGraph) *MultiHopReasoner {
	if graph == nil {
		return nil
	}
	return &MultiHopReasoner{Graph: graph}
}

// ConnectionResult describes how two entities are connected in the graph.
type ConnectionResult struct {
	EntityA     string
	EntityB     string
	Direct      []PathEdge   // direct A→B or B→A edges
	TwoHop      []TwoHopPath // A→X→B paths through an intermediate node
	SharedProps []string     // properties both entities share (has/is_a targets)
	Summary     string       // natural language summary of the connection
}

// PathEdge is a single directed edge in a connection path.
type PathEdge struct {
	From     string
	To       string
	Relation RelType
}

// TwoHopPath represents an A→X→B path through an intermediate node.
type TwoHopPath struct {
	Via   string   // intermediate node label
	EdgeAX PathEdge
	EdgeXB PathEdge
}

// FindConnection discovers how two entities are connected through the graph.
// It checks direct edges, two-hop paths, and shared properties.
func (mhr *MultiHopReasoner) FindConnection(entityA, entityB string) *ConnectionResult {
	if mhr.Graph == nil {
		return nil
	}

	idA := strings.ToLower(strings.TrimSpace(entityA))
	idB := strings.ToLower(strings.TrimSpace(entityB))
	if idA == "" || idB == "" {
		return nil
	}

	result := &ConnectionResult{
		EntityA: entityA,
		EntityB: entityB,
	}

	// 1. Direct edges: A→B and B→A
	result.Direct = mhr.findDirectEdges(idA, idB, entityA, entityB)

	// 2. Two-hop paths: A→X→B
	result.TwoHop = mhr.findTwoHopPaths(idA, idB, entityA, entityB)

	// 3. Shared properties: common "has" and "is_a" targets
	result.SharedProps = mhr.findSharedProperties(idA, idB)

	// 4. Generate summary
	result.Summary = mhr.summarize(result)

	return result
}

// findDirectEdges returns edges going directly from A to B or B to A.
func (mhr *MultiHopReasoner) findDirectEdges(idA, idB, labelA, labelB string) []PathEdge {
	var direct []PathEdge

	// A → B
	for _, edge := range mhr.Graph.EdgesFrom(idA) {
		if edge.To == idB {
			direct = append(direct, PathEdge{
				From:     labelA,
				To:       labelB,
				Relation: edge.Relation,
			})
		}
	}

	// B → A
	for _, edge := range mhr.Graph.EdgesFrom(idB) {
		if edge.To == idA {
			direct = append(direct, PathEdge{
				From:     labelB,
				To:       labelA,
				Relation: edge.Relation,
			})
		}
	}

	return direct
}

// findTwoHopPaths finds paths A→X→B where X is an intermediate node.
func (mhr *MultiHopReasoner) findTwoHopPaths(idA, idB, labelA, labelB string) []TwoHopPath {
	var paths []TwoHopPath

	// Build a set of B's neighbors (both incoming and outgoing) for fast lookup.
	bNeighbors := make(map[string]*CogEdge)       // nodeID → edge connecting to B
	bNeighborDir := make(map[string]bool)          // true = edge goes X→B, false = B→X
	for _, edge := range mhr.Graph.EdgesFrom(idB) {
		// B→X: to reach B from X we'd need X→B, so skip outgoing from B
		// unless we also check incoming to B
		_ = edge
	}
	// Edges into B: X→B
	for _, edge := range mhr.Graph.EdgesTo(idB) {
		bNeighbors[edge.From] = edge
		bNeighborDir[edge.From] = true
	}
	// Edges out of B: B→X — we can also traverse these in reverse
	for _, edge := range mhr.Graph.EdgesFrom(idB) {
		if _, exists := bNeighbors[edge.To]; !exists {
			bNeighbors[edge.To] = edge
			bNeighborDir[edge.To] = false
		}
	}

	// For each neighbor X of A, check if X is also a neighbor of B.
	seen := make(map[string]bool)

	// Outgoing from A: A→X
	for _, edgeAX := range mhr.Graph.EdgesFrom(idA) {
		x := edgeAX.To
		if x == idA || x == idB {
			continue
		}
		if edgeXB, ok := bNeighbors[x]; ok && !seen[x] {
			seen[x] = true
			xNode := mhr.Graph.GetNode(x)
			xLabel := x
			if xNode != nil {
				xLabel = xNode.Label
			}

			path := TwoHopPath{Via: xLabel}
			path.EdgeAX = PathEdge{From: labelA, To: xLabel, Relation: edgeAX.Relation}

			if bNeighborDir[x] {
				// X→B
				path.EdgeXB = PathEdge{From: xLabel, To: labelB, Relation: edgeXB.Relation}
			} else {
				// B→X, so from X's perspective it's X←B; present as B→X
				path.EdgeXB = PathEdge{From: labelB, To: xLabel, Relation: edgeXB.Relation}
			}

			paths = append(paths, path)
		}
	}

	// Incoming to A: X→A — so the path would be A←X→B
	for _, edgeXA := range mhr.Graph.EdgesTo(idA) {
		x := edgeXA.From
		if x == idA || x == idB || seen[x] {
			continue
		}
		if edgeXB, ok := bNeighbors[x]; ok {
			seen[x] = true
			xNode := mhr.Graph.GetNode(x)
			xLabel := x
			if xNode != nil {
				xLabel = xNode.Label
			}

			path := TwoHopPath{Via: xLabel}
			// X→A, presented as A←X
			path.EdgeAX = PathEdge{From: xLabel, To: labelA, Relation: edgeXA.Relation}

			if bNeighborDir[x] {
				path.EdgeXB = PathEdge{From: xLabel, To: labelB, Relation: edgeXB.Relation}
			} else {
				path.EdgeXB = PathEdge{From: labelB, To: xLabel, Relation: edgeXB.Relation}
			}

			paths = append(paths, path)
		}
	}

	// Cap at 10 to avoid explosion on densely connected graphs.
	if len(paths) > 10 {
		paths = paths[:10]
	}

	return paths
}

// findSharedProperties returns labels of nodes that both entities share
// via "has" or "is_a" edges.
func (mhr *MultiHopReasoner) findSharedProperties(idA, idB string) []string {
	propsA := mhr.collectProperties(idA)
	propsB := mhr.collectProperties(idB)

	var shared []string
	for prop := range propsA {
		if propsB[prop] {
			// Resolve to human-readable label
			node := mhr.Graph.GetNode(prop)
			if node != nil {
				shared = append(shared, node.Label)
			} else {
				shared = append(shared, prop)
			}
		}
	}
	return shared
}

// collectProperties returns the set of node IDs reachable via has/is_a edges.
func (mhr *MultiHopReasoner) collectProperties(id string) map[string]bool {
	props := make(map[string]bool)
	for _, edge := range mhr.Graph.EdgesFrom(id) {
		if edge.Relation == RelHas || edge.Relation == RelIsA {
			props[edge.To] = true
		}
	}
	return props
}

// summarize generates a natural language summary of a ConnectionResult.
func (mhr *MultiHopReasoner) summarize(cr *ConnectionResult) string {
	if cr == nil {
		return ""
	}

	var parts []string

	// Direct connections
	for _, d := range cr.Direct {
		parts = append(parts, fmt.Sprintf("%s and %s are directly connected: %s %s %s.",
			cr.EntityA, cr.EntityB, d.From, relVerb(d.Relation), d.To))
	}

	// Two-hop connections (show up to 3)
	limit := 3
	if len(cr.TwoHop) < limit {
		limit = len(cr.TwoHop)
	}
	for _, th := range cr.TwoHop[:limit] {
		parts = append(parts, fmt.Sprintf("%s and %s are connected through %s: %s %s %s, and %s %s %s.",
			cr.EntityA, cr.EntityB, th.Via,
			th.EdgeAX.From, relVerb(th.EdgeAX.Relation), th.EdgeAX.To,
			th.EdgeXB.From, relVerb(th.EdgeXB.Relation), th.EdgeXB.To))
	}

	// Shared properties
	if len(cr.SharedProps) > 0 {
		props := strings.Join(cr.SharedProps, ", ")
		parts = append(parts, fmt.Sprintf("Both %s and %s share: %s.",
			cr.EntityA, cr.EntityB, props))
	}

	if len(parts) == 0 {
		return fmt.Sprintf("I don't see a direct connection between %s and %s in my knowledge graph.",
			cr.EntityA, cr.EntityB)
	}

	return strings.Join(parts, " ")
}

// ExplainRelation generates a natural language explanation of how two
// entities are connected in the knowledge graph.
func (mhr *MultiHopReasoner) ExplainRelation(entityA, entityB string) string {
	conn := mhr.FindConnection(entityA, entityB)
	if conn == nil {
		return fmt.Sprintf("I don't see a direct connection between %s and %s in my knowledge graph.",
			entityA, entityB)
	}
	return conn.Summary
}

// relVerb converts a RelType to a human-readable verb phrase.
func relVerb(r RelType) string {
	switch r {
	case RelIsA:
		return "is a"
	case RelHas:
		return "has"
	case RelLocatedIn:
		return "is located in"
	case RelPartOf:
		return "is part of"
	case RelCreatedBy:
		return "was created by"
	case RelFoundedBy:
		return "was founded by"
	case RelFoundedIn:
		return "was founded in"
	case RelUsedFor:
		return "is used for"
	case RelOffers:
		return "offers"
	case RelRelatedTo:
		return "is related to"
	case RelSimilarTo:
		return "is similar to"
	case RelCauses:
		return "causes"
	case RelContradicts:
		return "contradicts"
	case RelFollows:
		return "follows"
	case RelPrefers:
		return "prefers"
	case RelDislikes:
		return "dislikes"
	case RelDomain:
		return "belongs to domain"
	case RelDescribedAs:
		return "is described as"
	case RelKnownFor:
		return "is known for"
	case RelInfluencedBy:
		return "was influenced by"
	case RelDerivedFrom:
		return "is derived from"
	case RelOppositeOf:
		return "is the opposite of"
	default:
		return string(r)
	}
}
