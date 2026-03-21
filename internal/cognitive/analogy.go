package cognitive

import (
	"fmt"
	"math"
	"strings"
)

// -----------------------------------------------------------------------
// Structural Analogy Engine — reasons about relationships between concepts
// by mapping relational structure across domains.
// "X is to Y as A is to ?" — solved by graph traversal, not pattern matching.
// -----------------------------------------------------------------------

// AnalogyEngine finds structural analogies within the cognitive graph.
type AnalogyEngine struct {
	Graph    *CognitiveGraph
	Semantic *SemanticEngine
}

// AnalogyResult holds the outcome of an analogy search.
type AnalogyResult struct {
	Source      string
	Target      string
	Mappings    []AnalogyMapping
	Confidence  float64
	Explanation string
}

// AnalogyMapping pairs a source node to a target node through a shared relation.
type AnalogyMapping struct {
	SourceNode string
	TargetNode string
	Relation   string
	Confidence float64
}

// NewAnalogyEngine creates an analogy engine backed by a cognitive graph
// and a semantic engine for soft matching.
func NewAnalogyEngine(graph *CognitiveGraph, semantic *SemanticEngine) *AnalogyEngine {
	return &AnalogyEngine{
		Graph:    graph,
		Semantic: semantic,
	}
}

// FindAnalogy discovers "X is to Y as A is to B" within targetDomain.
// It identifies the relation R connecting X→Y, then searches the target
// domain for pairs sharing that same relation.
func (ae *AnalogyEngine) FindAnalogy(x, y, targetDomain string) *AnalogyResult {
	xID := nodeID(x)
	yID := nodeID(y)

	// Find the relation(s) connecting X to Y.
	relations := ae.findRelations(xID, yID)
	if len(relations) == 0 {
		return nil
	}

	// Search the target domain for pairs with matching relations.
	targetNodes := ae.domainNodes(targetDomain)
	if len(targetNodes) == 0 {
		return nil
	}

	var bestMapping []AnalogyMapping
	var bestConf float64
	var bestTarget string

	for _, rel := range relations {
		for _, aID := range targetNodes {
			if aID == xID || aID == yID {
				continue
			}
			// Look for outgoing edges from aID with the same relation.
			edges := ae.Graph.EdgesFrom(aID)
			for _, edge := range edges {
				if string(edge.Relation) == rel {
					bID := edge.To
					conf := edge.Confidence * edge.Weight
					if conf > bestConf {
						aNode := ae.Graph.GetNode(aID)
						bNode := ae.Graph.GetNode(bID)
						aLabel := aID
						bLabel := bID
						if aNode != nil {
							aLabel = aNode.Label
						}
						if bNode != nil {
							bLabel = bNode.Label
						}
						bestConf = conf
						bestTarget = bLabel
						bestMapping = []AnalogyMapping{{
							SourceNode: aLabel,
							TargetNode: bLabel,
							Relation:   rel,
							Confidence: conf,
						}}
					}
				}
			}
		}
	}

	if len(bestMapping) == 0 {
		return nil
	}

	xLabel := x
	if n := ae.Graph.GetNode(xID); n != nil {
		xLabel = n.Label
	}
	yLabel := y
	if n := ae.Graph.GetNode(yID); n != nil {
		yLabel = n.Label
	}

	return &AnalogyResult{
		Source:     xLabel,
		Target:     bestTarget,
		Mappings:   bestMapping,
		Confidence: bestConf,
		Explanation: fmt.Sprintf("%s is to %s (via %s) as %s is to %s",
			xLabel, yLabel, relations[0],
			bestMapping[0].SourceNode, bestMapping[0].TargetNode),
	}
}

// CompleteAnalogy solves "X:Y :: A:?" by finding the relation from X to Y,
// then finding nodes connected to A by the same relation.
// Returns the best match and its confidence.
func (ae *AnalogyEngine) CompleteAnalogy(x, y, a string) (string, float64) {
	xID := nodeID(x)
	yID := nodeID(y)
	aID := nodeID(a)

	relations := ae.findRelations(xID, yID)
	if len(relations) == 0 {
		return "", 0
	}

	var bestLabel string
	var bestConf float64

	for _, rel := range relations {
		edges := ae.Graph.EdgesFrom(aID)
		for _, edge := range edges {
			if string(edge.Relation) == rel {
				conf := edge.Confidence * edge.Weight
				// Boost if semantic similarity to Y is high (structural parallel).
				if ae.Semantic != nil {
					targetNode := ae.Graph.GetNode(edge.To)
					yNode := ae.Graph.GetNode(yID)
					if targetNode != nil && yNode != nil {
						sim := ae.Semantic.Similarity(targetNode.Label, yNode.Label)
						conf += sim * 0.2
					}
				}
				if conf > bestConf {
					bestConf = conf
					bNode := ae.Graph.GetNode(edge.To)
					if bNode != nil {
						bestLabel = bNode.Label
					} else {
						bestLabel = edge.To
					}
				}
			}
		}
	}

	if bestConf > 1.0 {
		bestConf = 1.0
	}

	return bestLabel, bestConf
}

// MapDomains aligns two knowledge subgraphs by comparing their relational
// skeletons. Nodes in the source domain are paired with the best-matching
// nodes in the target domain based on structural similarity.
func (ae *AnalogyEngine) MapDomains(sourceDomain, targetDomain string) []AnalogyMapping {
	sourceNodes := ae.domainNodes(sourceDomain)
	targetNodes := ae.domainNodes(targetDomain)

	if len(sourceNodes) == 0 || len(targetNodes) == 0 {
		return nil
	}

	// Build skeletons for all nodes in both domains.
	sourceSkeletons := make(map[string]map[string]int, len(sourceNodes))
	for _, id := range sourceNodes {
		sourceSkeletons[id] = ae.relationalSkeleton(id)
	}
	targetSkeletons := make(map[string]map[string]int, len(targetNodes))
	for _, id := range targetNodes {
		targetSkeletons[id] = ae.relationalSkeleton(id)
	}

	// Greedy best-match alignment: for each source node, find the target
	// node with the highest structural similarity.
	used := make(map[string]bool)
	var mappings []AnalogyMapping

	for _, sID := range sourceNodes {
		var bestTarget string
		var bestSim float64

		for _, tID := range targetNodes {
			if used[tID] {
				continue
			}
			sim := ae.structuralSimilarity(sourceSkeletons[sID], targetSkeletons[tID])
			// Blend in semantic similarity if available.
			if ae.Semantic != nil {
				sNode := ae.Graph.GetNode(sID)
				tNode := ae.Graph.GetNode(tID)
				if sNode != nil && tNode != nil {
					semSim := ae.Semantic.Similarity(sNode.Label, tNode.Label)
					sim = sim*0.7 + semSim*0.3
				}
			}
			if sim > bestSim {
				bestSim = sim
				bestTarget = tID
			}
		}

		if bestTarget != "" && bestSim > 0.1 {
			used[bestTarget] = true
			sLabel := sID
			tLabel := bestTarget
			if n := ae.Graph.GetNode(sID); n != nil {
				sLabel = n.Label
			}
			if n := ae.Graph.GetNode(bestTarget); n != nil {
				tLabel = n.Label
			}
			mappings = append(mappings, AnalogyMapping{
				SourceNode: sLabel,
				TargetNode: tLabel,
				Relation:   "structural_match",
				Confidence: bestSim,
			})
		}
	}

	return mappings
}

// ApplyPrinciples extracts an entity's properties and relations from the
// graph and composes natural-language reasoning about how those principles
// apply to a given context. Useful for "What would Socrates say about X?"
func (ae *AnalogyEngine) ApplyPrinciples(entityName, context string) string {
	entityID := nodeID(entityName)
	node := ae.Graph.GetNode(entityID)
	if node == nil {
		return ""
	}

	edges := ae.Graph.EdgesFrom(entityID)
	if len(edges) == 0 {
		return ""
	}

	var principles []string
	for _, edge := range edges {
		targetNode := ae.Graph.GetNode(edge.To)
		if targetNode == nil {
			continue
		}
		switch edge.Relation {
		case RelHas, RelDescribedAs:
			principles = append(principles,
				fmt.Sprintf("Based on %s's principle of %s, applied to %s, one might consider how %s shapes our understanding.",
					node.Label, targetNode.Label, context, targetNode.Label))
		case RelIsA:
			principles = append(principles,
				fmt.Sprintf("As %s %s, their perspective on %s would be informed by that tradition.",
					articleFor(targetNode.Label), node.Label, context))
		case RelCreatedBy, RelRelatedTo, RelUsedFor:
			principles = append(principles,
				fmt.Sprintf("Given %s's connection to %s, one can draw parallels to %s.",
					node.Label, targetNode.Label, context))
		}
	}

	// Also check incoming edges for additional context.
	inEdges := ae.Graph.EdgesTo(entityID)
	for _, edge := range inEdges {
		sourceNode := ae.Graph.GetNode(edge.From)
		if sourceNode == nil {
			continue
		}
		if edge.Relation == RelCreatedBy || edge.Relation == RelFoundedBy {
			principles = append(principles,
				fmt.Sprintf("As the one behind %s, %s's approach to %s would reflect that creative drive.",
					sourceNode.Label, node.Label, context))
		}
	}

	if len(principles) == 0 {
		return ""
	}

	return strings.Join(principles, " ")
}

// -----------------------------------------------------------------------
// Internal helpers
// -----------------------------------------------------------------------

// findRelations returns the relation types connecting fromID to toID.
func (ae *AnalogyEngine) findRelations(fromID, toID string) []string {
	edges := ae.Graph.EdgesFrom(fromID)
	var rels []string
	for _, edge := range edges {
		if edge.To == toID {
			rels = append(rels, string(edge.Relation))
		}
	}
	return rels
}

// domainNodes returns all node IDs that belong to a given domain.
// A node belongs to a domain if it has a "domain" edge pointing to the
// domain, or if its label contains the domain name.
func (ae *AnalogyEngine) domainNodes(domain string) []string {
	domainLower := strings.ToLower(domain)
	domainID := nodeID(domain)

	ae.Graph.mu.RLock()
	defer ae.Graph.mu.RUnlock()

	seen := make(map[string]bool)
	var ids []string

	// Nodes with an explicit domain edge.
	for _, edge := range ae.Graph.inEdges[domainID] {
		if edge.Relation == RelDomain || edge.Relation == RelIsA ||
			edge.Relation == RelPartOf || edge.Relation == RelRelatedTo {
			if !seen[edge.From] {
				seen[edge.From] = true
				ids = append(ids, edge.From)
			}
		}
	}

	// Fallback: nodes whose label contains the domain string.
	if len(ids) == 0 {
		for id, node := range ae.Graph.nodes {
			if strings.Contains(strings.ToLower(node.Label), domainLower) {
				if !seen[id] {
					seen[id] = true
					ids = append(ids, id)
				}
			}
		}
	}

	return ids
}

// extractSubgraph performs BFS from rootID, collecting all node IDs and
// edges within maxDepth hops.
func (ae *AnalogyEngine) extractSubgraph(rootID string, maxDepth int) ([]string, []*CogEdge) {
	ae.Graph.mu.RLock()
	defer ae.Graph.mu.RUnlock()

	if _, ok := ae.Graph.nodes[rootID]; !ok {
		return nil, nil
	}

	type item struct {
		id    string
		depth int
	}

	visited := map[string]bool{rootID: true}
	queue := []item{{rootID, 0}}
	var nodeIDs []string
	var edges []*CogEdge
	edgeSeen := make(map[string]bool) // "from|to|rel"

	nodeIDs = append(nodeIDs, rootID)

	for len(queue) > 0 {
		cur := queue[0]
		queue = queue[1:]

		if cur.depth >= maxDepth {
			continue
		}

		for _, edge := range ae.Graph.outEdges[cur.id] {
			key := edge.From + "|" + edge.To + "|" + string(edge.Relation)
			if !edgeSeen[key] {
				edgeSeen[key] = true
				edges = append(edges, edge)
			}
			if !visited[edge.To] {
				visited[edge.To] = true
				nodeIDs = append(nodeIDs, edge.To)
				queue = append(queue, item{edge.To, cur.depth + 1})
			}
		}

		for _, edge := range ae.Graph.inEdges[cur.id] {
			key := edge.From + "|" + edge.To + "|" + string(edge.Relation)
			if !edgeSeen[key] {
				edgeSeen[key] = true
				edges = append(edges, edge)
			}
			if !visited[edge.From] {
				visited[edge.From] = true
				nodeIDs = append(nodeIDs, edge.From)
				queue = append(queue, item{edge.From, cur.depth + 1})
			}
		}
	}

	return nodeIDs, edges
}

// relationalSkeleton counts edge types around a node to create a
// structural fingerprint: map[relationType]count.
func (ae *AnalogyEngine) relationalSkeleton(nodeID string) map[string]int {
	skeleton := make(map[string]int)

	outEdges := ae.Graph.EdgesFrom(nodeID)
	for _, edge := range outEdges {
		skeleton["out:"+string(edge.Relation)]++
	}

	inEdges := ae.Graph.EdgesTo(nodeID)
	for _, edge := range inEdges {
		skeleton["in:"+string(edge.Relation)]++
	}

	return skeleton
}

// structuralSimilarity compares two relational skeletons using cosine
// similarity over the relation-type count vectors.
func (ae *AnalogyEngine) structuralSimilarity(a, b map[string]int) float64 {
	if len(a) == 0 || len(b) == 0 {
		return 0
	}

	// Collect all keys.
	keys := make(map[string]bool)
	for k := range a {
		keys[k] = true
	}
	for k := range b {
		keys[k] = true
	}

	var dot, normA, normB float64
	for k := range keys {
		va := float64(a[k])
		vb := float64(b[k])
		dot += va * vb
		normA += va * va
		normB += vb * vb
	}

	if normA == 0 || normB == 0 {
		return 0
	}

	return dot / (math.Sqrt(normA) * math.Sqrt(normB))
}
