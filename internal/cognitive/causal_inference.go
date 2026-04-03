package cognitive

import (
	"fmt"
	"math"
	"strconv"
	"strings"
)

// -----------------------------------------------------------------------
// Structural Causal Inference — discover causal edges from graph topology.
//
// Instead of requiring explicit "X causes Y" in text, this engine
// INFERS causal relationships from structural patterns in the knowledge
// graph. This is a novel approach — no small local system does this.
//
// Four inference strategies:
//
//   1. Temporal Ordering: If A founded_in 1905 and B founded_in 1920,
//      and they share a domain, infer A enables B (earlier discovery
//      enables later ones in the same field).
//
//   2. Dependency Chains: If A and B are both part_of C, and A is
//      described as "fundamental" or "basis", infer B requires A.
//
//   3. Inhibition from Contradiction: If A contradicts B and A has
//      causal effects, then strengthening A prevents B's effects.
//
//   4. Production Chains: If A produces X and X is required by B,
//      infer A enables B (transitive production-consumption).
//
// All inferred edges are marked with Inferred=true and lower confidence
// (0.3-0.5) to distinguish them from extracted facts.
// -----------------------------------------------------------------------

// CausalInferenceEngine discovers implicit causal edges from graph structure.
type CausalInferenceEngine struct {
	Graph *CognitiveGraph
}

// InferredEdge represents a discovered causal relationship.
type InferredEdge struct {
	From       string
	To         string
	Relation   RelType
	Confidence float64
	Reason     string // human-readable explanation of why this was inferred
}

// InferenceReport summarizes what was discovered.
type InferenceReport struct {
	Edges           []InferredEdge
	TemporalCount   int
	DependencyCount int
	InhibitionCount int
	ProductionCount int
	AddedCount      int // how many were actually new (not duplicate)
}

// NewCausalInferenceEngine creates an inference engine.
func NewCausalInferenceEngine(graph *CognitiveGraph) *CausalInferenceEngine {
	return &CausalInferenceEngine{Graph: graph}
}

// InferAll runs all four inference strategies and adds discovered edges
// to the graph. Returns a report of what was found.
func (ci *CausalInferenceEngine) InferAll() *InferenceReport {
	report := &InferenceReport{}

	// Strategy 1: Temporal ordering.
	temporal := ci.inferTemporal()
	report.TemporalCount = len(temporal)
	report.Edges = append(report.Edges, temporal...)

	// Strategy 2: Dependency chains.
	deps := ci.inferDependencyChains()
	report.DependencyCount = len(deps)
	report.Edges = append(report.Edges, deps...)

	// Strategy 3: Inhibition from contradiction.
	inhib := ci.inferInhibition()
	report.InhibitionCount = len(inhib)
	report.Edges = append(report.Edges, inhib...)

	// Strategy 4: Production chains.
	prod := ci.inferProductionChains()
	report.ProductionCount = len(prod)
	report.Edges = append(report.Edges, prod...)

	// Add all inferred edges to the graph (deduplicating).
	added := 0
	for _, e := range report.Edges {
		if ci.addInferredEdge(e) {
			added++
		}
	}
	report.AddedCount = added

	return report
}

// -----------------------------------------------------------------------
// Strategy 1: Temporal Ordering
//
// If entity A was founded_in year Y1 and entity B was founded_in Y2,
// where Y1 < Y2, and they share a domain (both is_a same parent),
// infer A enables B.
//
// Example: "General relativity (1915)" enables "Gravitational waves (2015)"
// because both are in physics and relativity came first.
// -----------------------------------------------------------------------

func (ci *CausalInferenceEngine) inferTemporal() []InferredEdge {
	if ci.Graph == nil {
		return nil
	}

	ci.Graph.mu.RLock()
	defer ci.Graph.mu.RUnlock()

	// Collect entities with founding years.
	type dated struct {
		id    string
		label string
		year  int
	}
	var entities []dated

	for id, node := range ci.Graph.nodes {
		for _, edge := range ci.Graph.outEdges[id] {
			if edge.Relation == RelFoundedIn {
				targetNode := ci.Graph.nodes[edge.To]
				if targetNode == nil {
					continue
				}
				year := extractYear(targetNode.Label)
				if year > 0 {
					entities = append(entities, dated{id, node.Label, year})
				}
			}
		}
	}

	// For each pair sharing a domain, check temporal order.
	var inferred []InferredEdge
	for i := range entities {
		for j := range entities {
			if i == j || entities[i].id == entities[j].id {
				continue
			}
			if entities[i].year >= entities[j].year {
				continue
			}
			// Check if they share a common parent (same is_a target).
			if !ci.sharesDomain(entities[i].id, entities[j].id) {
				continue
			}
			// Earlier enables later.
			gap := entities[j].year - entities[i].year
			confidence := 0.4
			if gap < 50 {
				confidence = 0.5 // closer in time = stronger relationship
			}
			if gap > 200 {
				confidence = 0.25 // very distant = weaker
			}

			inferred = append(inferred, InferredEdge{
				From:       entities[i].label,
				To:         entities[j].label,
				Relation:   RelEnables,
				Confidence: confidence,
				Reason: fmt.Sprintf("Temporal: %s (%d) precedes %s (%d) in same domain",
					entities[i].label, entities[i].year, entities[j].label, entities[j].year),
			})

			// Cap to avoid explosion.
			if len(inferred) >= 200 {
				return inferred
			}
		}
	}

	return inferred
}

// -----------------------------------------------------------------------
// Strategy 2: Dependency Chains
//
// If A and B are both part_of C, and A is described as "fundamental",
// "basic", "foundational", "essential", or "basis", infer B requires A.
//
// Example: "Arithmetic" is part_of "Mathematics" and described as
// "fundamental" → "Algebra" requires "Arithmetic".
// -----------------------------------------------------------------------

func (ci *CausalInferenceEngine) inferDependencyChains() []InferredEdge {
	if ci.Graph == nil {
		return nil
	}

	ci.Graph.mu.RLock()
	defer ci.Graph.mu.RUnlock()

	// Find nodes described as foundational.
	foundational := make(map[string]bool)
	foundationalSignals := []string{
		"fundamental", "basic", "foundational", "essential",
		"basis", "foundation", "core", "prerequisite", "elementary",
	}

	for id := range ci.Graph.nodes {
		for _, edge := range ci.Graph.outEdges[id] {
			if edge.Relation == RelDescribedAs {
				target := ci.Graph.nodes[edge.To]
				if target == nil {
					continue
				}
				descLower := strings.ToLower(target.Label)
				for _, sig := range foundationalSignals {
					if strings.Contains(descLower, sig) {
						foundational[id] = true
						break
					}
				}
			}
		}
	}

	// For each foundational node, find siblings (same parent via part_of/is_a).
	var inferred []InferredEdge
	for fID := range foundational {
		fNode := ci.Graph.nodes[fID]
		if fNode == nil {
			continue
		}

		// Find parents of the foundational node.
		for _, fEdge := range ci.Graph.outEdges[fID] {
			if fEdge.Relation != RelPartOf && fEdge.Relation != RelIsA {
				continue
			}
			parentID := fEdge.To

			// Find siblings of the foundational node.
			for _, sibEdge := range ci.Graph.inEdges[parentID] {
				if sibEdge.From == fID {
					continue
				}
				if sibEdge.Relation != RelPartOf && sibEdge.Relation != RelIsA {
					continue
				}
				sibNode := ci.Graph.nodes[sibEdge.From]
				if sibNode == nil || foundational[sibEdge.From] {
					continue // don't infer foundational requires foundational
				}

				inferred = append(inferred, InferredEdge{
					From:       sibNode.Label,
					To:         fNode.Label,
					Relation:   RelRequires,
					Confidence: 0.35,
					Reason: fmt.Sprintf("Dependency: %s and %s share parent, %s is foundational",
						sibNode.Label, fNode.Label, fNode.Label),
				})
			}
		}

		if len(inferred) >= 200 {
			break
		}
	}

	return inferred
}

// -----------------------------------------------------------------------
// Strategy 3: Inhibition from Contradiction
//
// If A contradicts B, infer that A prevents the effects of B
// and B prevents the effects of A.
//
// Example: "Creationism contradicts Evolution" → "Creationism prevents
// acceptance of evolutionary theory" (and vice versa).
// -----------------------------------------------------------------------

func (ci *CausalInferenceEngine) inferInhibition() []InferredEdge {
	if ci.Graph == nil {
		return nil
	}

	ci.Graph.mu.RLock()
	defer ci.Graph.mu.RUnlock()

	var inferred []InferredEdge

	for _, edges := range ci.Graph.outEdges {
		for _, edge := range edges {
			if edge.Relation != RelContradicts && edge.Relation != RelOppositeOf {
				continue
			}

			fromNode := ci.Graph.nodes[edge.From]
			toNode := ci.Graph.nodes[edge.To]
			if fromNode == nil || toNode == nil {
				continue
			}

			// Each side prevents the other.
			inferred = append(inferred, InferredEdge{
				From:       fromNode.Label,
				To:         toNode.Label,
				Relation:   RelPrevents,
				Confidence: 0.3,
				Reason: fmt.Sprintf("Inhibition: %s contradicts %s",
					fromNode.Label, toNode.Label),
			})
		}
	}

	return inferred
}

// -----------------------------------------------------------------------
// Strategy 4: Production Chains
//
// If A produces X and B requires X (or X is part_of B),
// infer A enables B (transitive production → consumption).
//
// Example: "Photosynthesis produces oxygen" + "Aerobic respiration
// requires oxygen" → "Photosynthesis enables aerobic respiration".
// -----------------------------------------------------------------------

func (ci *CausalInferenceEngine) inferProductionChains() []InferredEdge {
	if ci.Graph == nil {
		return nil
	}

	ci.Graph.mu.RLock()
	defer ci.Graph.mu.RUnlock()

	// Build a map of what each node produces.
	// producer → [product IDs]
	producers := make(map[string][]string) // producerID → productIDs

	for id := range ci.Graph.nodes {
		for _, edge := range ci.Graph.outEdges[id] {
			if edge.Relation == RelProduces || edge.Relation == RelCauses {
				producers[id] = append(producers[id], edge.To)
			}
		}
	}

	// For each product, find consumers (nodes that require or use it).
	var inferred []InferredEdge
	for producerID, productIDs := range producers {
		producerNode := ci.Graph.nodes[producerID]
		if producerNode == nil {
			continue
		}

		for _, productID := range productIDs {
			// Find nodes that require or depend on this product.
			for _, inEdge := range ci.Graph.inEdges[productID] {
				if inEdge.Relation != RelRequires && inEdge.Relation != RelHas {
					continue
				}
				consumerNode := ci.Graph.nodes[inEdge.From]
				if consumerNode == nil || inEdge.From == producerID {
					continue
				}

				productNode := ci.Graph.nodes[productID]
				productLabel := ""
				if productNode != nil {
					productLabel = productNode.Label
				}

				inferred = append(inferred, InferredEdge{
					From:       producerNode.Label,
					To:         consumerNode.Label,
					Relation:   RelEnables,
					Confidence: 0.35,
					Reason: fmt.Sprintf("Production chain: %s produces %s, which %s requires",
						producerNode.Label, productLabel, consumerNode.Label),
				})
			}
		}

		if len(inferred) >= 200 {
			break
		}
	}

	return inferred
}

// -----------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------

func (ci *CausalInferenceEngine) sharesDomain(idA, idB string) bool {
	// Check if A and B share a common parent via is_a or part_of.
	parentsA := make(map[string]bool)
	for _, edge := range ci.Graph.outEdges[idA] {
		if edge.Relation == RelIsA || edge.Relation == RelPartOf || edge.Relation == RelDomain {
			parentsA[edge.To] = true
		}
	}

	for _, edge := range ci.Graph.outEdges[idB] {
		if edge.Relation == RelIsA || edge.Relation == RelPartOf || edge.Relation == RelDomain {
			if parentsA[edge.To] {
				return true
			}
		}
	}

	return false
}

func (ci *CausalInferenceEngine) addInferredEdge(e InferredEdge) bool {
	if ci.Graph == nil {
		return false
	}

	// Check if this edge already exists (exact or inferred).
	ci.Graph.mu.RLock()
	fromID := nodeID(e.From)
	if edges, ok := ci.Graph.outEdges[fromID]; ok {
		toID := nodeID(e.To)
		for _, existing := range edges {
			if existing.To == toID && existing.Relation == e.Relation {
				ci.Graph.mu.RUnlock()
				return false // already exists
			}
		}
	}
	ci.Graph.mu.RUnlock()

	// Add the inferred edge.
	ci.Graph.mu.Lock()
	fromID = ci.Graph.ensureNodeLocked(e.From, NodeConcept, "causal_inference", e.Confidence)
	toID := ci.Graph.ensureNodeLocked(e.To, NodeConcept, "causal_inference", e.Confidence)
	ci.Graph.addEdgeLocked(fromID, toID, e.Relation, "causal_inference:"+e.Reason, e.Confidence, true)
	ci.Graph.mu.Unlock()

	return true
}

func extractYear(s string) int {
	s = strings.TrimSpace(s)
	// Try direct parse.
	if y, err := strconv.Atoi(s); err == nil && y > 0 && y < 3000 {
		return y
	}
	// Try extracting year from strings like "1905 CE" or "around 1920".
	for _, w := range strings.Fields(s) {
		w = strings.Trim(w, ".,;:()")
		if y, err := strconv.Atoi(w); err == nil && y > 100 && y < 3000 {
			return y
		}
	}
	return 0
}

// graphRelationWeight returns how important a relation type is for
// spreading activation in causal reasoning. Used by the simulation
// engine to weight which edges propagate effects more strongly.
func graphRelationWeight(rel RelType) float64 {
	switch rel {
	case RelCauses:
		return 1.0
	case RelEnables:
		return 0.9
	case RelProduces:
		return 0.85
	case RelPrevents:
		return 0.8
	case RelRequires:
		return 0.75
	case RelFollows:
		return 0.7
	case RelIsA:
		return 0.5
	case RelPartOf:
		return 0.4
	case RelInfluencedBy:
		return 0.6
	default:
		return math.Max(0.2, 0.3)
	}
}
