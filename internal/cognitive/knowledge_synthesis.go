package cognitive

import (
	"fmt"
	"math"
	"strings"
)

// -----------------------------------------------------------------------
// Knowledge Synthesis — reasons from gaps instead of saying "I don't know."
//
// When direct knowledge is missing, this system synthesizes qualified
// insights from adjacent knowledge. Instead of "I don't know about X",
// it says "I don't have direct knowledge about X, but based on what I
// know about Y and Z, I can reason that..."
// -----------------------------------------------------------------------

// KnowledgeSynthesizer reasons about unknown topics using adjacent knowledge.
type KnowledgeSynthesizer struct {
	graph   *CognitiveGraph
	analogy *AnalogyEngine
}

// SynthesisStrategy describes HOW the system arrived at its synthesis.
type SynthesisStrategy int

const (
	StratAnalogy       SynthesisStrategy = iota // "X is like Y because..."
	StratDecomposition                           // "X consists of parts A, B, C which I know about..."
	StratGeneralization                          // "X is a type of Y, and Y generally..."
	StratCausalChain                             // "X causes Y, Y causes Z, so X probably causes Z"
	StratContrastive                             // "X is unlike Y in these ways, suggesting..."
	StratCompositional                           // "Combining what I know about A and B..."
)

// strategyName returns a human-readable name for the strategy.
func (s SynthesisStrategy) String() string {
	switch s {
	case StratAnalogy:
		return "analogy"
	case StratDecomposition:
		return "decomposition"
	case StratGeneralization:
		return "generalization"
	case StratCausalChain:
		return "causal chain"
	case StratContrastive:
		return "contrastive reasoning"
	case StratCompositional:
		return "compositional reasoning"
	default:
		return "unknown"
	}
}

// SynthesizedKnowledge is a qualified conclusion drawn from adjacent knowledge.
type SynthesizedKnowledge struct {
	Topic      string
	Claim      string            // the synthesized insight
	Strategy   SynthesisStrategy // how we got here
	Evidence   []SynthesisEvidence
	Confidence float64 // how confident (always < direct knowledge)
	Qualifier  string  // "Based on...", "By analogy with...", "Given that..."
	Caveat     string  // explicit limitation: "This is inferred, not directly known"
}

// SynthesisEvidence is one piece of supporting knowledge.
type SynthesisEvidence struct {
	Fact      string // the known fact
	Relevance string // why this is relevant
	Source    string // where it came from
}

// SynthesisResult is the full output of a synthesis attempt.
type SynthesisResult struct {
	Topic       string
	DirectFacts int                    // how many direct facts existed
	Synthesized []SynthesizedKnowledge // synthesized claims
	OverallConf float64
	Explanation string // human-readable synthesis explanation
}

// maxSynthesisConfidence caps all synthesis confidence values. Synthesis is
// never as reliable as direct knowledge.
const maxSynthesisConfidence = 0.7

// NewKnowledgeSynthesizer creates a knowledge synthesizer backed by a
// cognitive graph and an analogy engine.
func NewKnowledgeSynthesizer(graph *CognitiveGraph, analogy *AnalogyEngine) *KnowledgeSynthesizer {
	return &KnowledgeSynthesizer{
		graph:   graph,
		analogy: analogy,
	}
}

// ShouldSynthesize returns true when direct knowledge is insufficient
// (fewer than 2 direct facts) AND the topic has some adjacent knowledge
// that could be leveraged.
func (ks *KnowledgeSynthesizer) ShouldSynthesize(topic string, directFactCount int) bool {
	if directFactCount >= 2 {
		return false
	}
	if ks.graph == nil {
		return false
	}
	// Check for adjacent knowledge: parent categories, related terms,
	// or component words that exist in the graph.
	if ks.hasAdjacentKnowledge(topic) {
		return true
	}
	return false
}

// Synthesize tries all strategies in order and collects results. It returns
// a SynthesisResult with qualified, caveated claims drawn from adjacent
// knowledge in the graph.
func (ks *KnowledgeSynthesizer) Synthesize(topic string) *SynthesisResult {
	if ks.graph == nil {
		return &SynthesisResult{
			Topic:       topic,
			Explanation: "No knowledge graph available for synthesis.",
		}
	}

	result := &SynthesisResult{
		Topic: topic,
	}

	// Count direct facts.
	result.DirectFacts = ks.countDirectFacts(topic)

	// Try each strategy in order.
	result.Synthesized = append(result.Synthesized, ks.synthesizeGeneralization(topic)...)
	result.Synthesized = append(result.Synthesized, ks.synthesizeDecomposition(topic)...)
	result.Synthesized = append(result.Synthesized, ks.synthesizeAnalogy(topic)...)
	result.Synthesized = append(result.Synthesized, ks.synthesizeCausalChain(topic)...)
	result.Synthesized = append(result.Synthesized, ks.synthesizeContrastive(topic)...)
	result.Synthesized = append(result.Synthesized, ks.synthesizeCompositional(topic)...)

	// Compute overall confidence as the average of all claims, capped.
	if len(result.Synthesized) > 0 {
		var sum float64
		for _, sk := range result.Synthesized {
			sum += sk.Confidence
		}
		result.OverallConf = capConfidence(sum / float64(len(result.Synthesized)))
	}

	result.Explanation = ks.buildExplanation(result)

	return result
}

// FormatSynthesis formats a SynthesisResult as natural, human-readable text.
func (ks *KnowledgeSynthesizer) FormatSynthesis(result *SynthesisResult) string {
	if result == nil || len(result.Synthesized) == 0 {
		return fmt.Sprintf("I don't have direct knowledge about %s, and I couldn't find enough adjacent knowledge to reason about it.", result.Topic)
	}

	var b strings.Builder

	if result.DirectFacts == 0 {
		fmt.Fprintf(&b, "I don't have direct knowledge about %s, but I can reason about it:\n\n", result.Topic)
	} else {
		fmt.Fprintf(&b, "I have limited direct knowledge about %s, but I can reason further:\n\n", result.Topic)
	}

	for i, sk := range result.Synthesized {
		if i > 0 {
			b.WriteString("\n\n")
		}
		fmt.Fprintf(&b, "%s: %s", sk.Qualifier, sk.Claim)
		if len(sk.Evidence) > 0 {
			b.WriteString(" ")
			b.WriteString(sk.Evidence[0].Fact)
		}
		fmt.Fprintf(&b, "\nConfidence: %s. %s", confidenceLevel(sk.Confidence), sk.Caveat)
	}

	return b.String()
}

// -----------------------------------------------------------------------
// Strategy implementations
// -----------------------------------------------------------------------

// synthesizeGeneralization: If X is_a Y, look up what's known about Y
// and apply it to X. "Rust is a programming language. Programming languages
// generally have syntax, compilers, and type systems."
func (ks *KnowledgeSynthesizer) synthesizeGeneralization(topic string) []SynthesizedKnowledge {
	topicID := nodeID(topic)
	edges := ks.graph.EdgesFrom(topicID)

	var results []SynthesizedKnowledge

	for _, edge := range edges {
		if edge.Relation != RelIsA {
			continue
		}
		parentNode := ks.graph.GetNode(edge.To)
		if parentNode == nil {
			continue
		}

		// Look up what's known about the parent category.
		parentEdges := ks.graph.EdgesFrom(edge.To)
		var parentProps []string
		var evidence []SynthesisEvidence

		for _, pe := range parentEdges {
			if pe.Relation == RelDescribedAs {
				continue
			}
			targetNode := ks.graph.GetNode(pe.To)
			if targetNode == nil {
				continue
			}
			fact := fmt.Sprintf("%s %s %s", parentNode.Label, pe.Relation, targetNode.Label)
			parentProps = append(parentProps, targetNode.Label)
			evidence = append(evidence, SynthesisEvidence{
				Fact:      fact,
				Relevance: fmt.Sprintf("%s is a type of %s, so this property likely applies", topic, parentNode.Label),
				Source:    "generalization from parent category",
			})
		}

		if len(parentProps) == 0 {
			continue
		}

		claim := fmt.Sprintf("%s is %s %s. %s generally %s, so %s likely does too.",
			topic, articleFor(parentNode.Label), parentNode.Label,
			capitalizeFirst(parentNode.Label),
			summarizeProperties(parentProps),
			topic,
		)

		conf := capConfidence(edge.Confidence * 0.6)

		results = append(results, SynthesizedKnowledge{
			Topic:      topic,
			Claim:      claim,
			Strategy:   StratGeneralization,
			Evidence:   evidence,
			Confidence: conf,
			Qualifier:  fmt.Sprintf("Since %s is a type of %s", topic, parentNode.Label),
			Caveat:     "This is inferred from the parent category, not directly known about " + topic + ".",
		})
	}

	return results
}

// synthesizeDecomposition: Break a compound topic into component words,
// look up each separately, and combine insights.
func (ks *KnowledgeSynthesizer) synthesizeDecomposition(topic string) []SynthesizedKnowledge {
	words := strings.Fields(topic)
	if len(words) < 2 {
		return nil
	}

	var componentFacts []struct {
		word  string
		facts []string
	}

	for _, word := range words {
		wordID := nodeID(word)
		node := ks.graph.GetNode(wordID)
		if node == nil {
			continue
		}

		edges := ks.graph.EdgesFrom(wordID)
		var facts []string
		for _, edge := range edges {
			if edge.Relation == RelDescribedAs {
				continue
			}
			targetNode := ks.graph.GetNode(edge.To)
			if targetNode == nil {
				continue
			}
			facts = append(facts, fmt.Sprintf("%s %s %s", node.Label, edge.Relation, targetNode.Label))
		}

		if len(facts) > 0 {
			componentFacts = append(componentFacts, struct {
				word  string
				facts []string
			}{word: node.Label, facts: facts})
		}
	}

	if len(componentFacts) == 0 {
		return nil
	}

	// Build the synthesis from component knowledge.
	var evidence []SynthesisEvidence
	var parts []string

	for _, cf := range componentFacts {
		parts = append(parts, cf.word)
		for _, fact := range cf.facts {
			evidence = append(evidence, SynthesisEvidence{
				Fact:      fact,
				Relevance: fmt.Sprintf("'%s' is a component of '%s'", cf.word, topic),
				Source:    "decomposition of compound topic",
			})
		}
	}

	claim := fmt.Sprintf("Breaking down '%s' into its components: I know about %s separately. Combining this knowledge suggests %s relates to the intersection of these concepts.",
		topic,
		strings.Join(parts, " and "),
		topic,
	)

	conf := capConfidence(0.4 * math.Min(1.0, float64(len(componentFacts))/float64(len(words))))

	return []SynthesizedKnowledge{{
		Topic:      topic,
		Claim:      claim,
		Strategy:   StratDecomposition,
		Evidence:   evidence,
		Confidence: conf,
		Qualifier:  fmt.Sprintf("By decomposing '%s' into %s", topic, strings.Join(parts, " and ")),
		Caveat:     "This is pieced together from component concepts, not a holistic understanding of " + topic + ".",
	}}
}

// synthesizeAnalogy: Find analogous concepts. "I don't know about X,
// but it's similar to Y which I know about..."
func (ks *KnowledgeSynthesizer) synthesizeAnalogy(topic string) []SynthesizedKnowledge {
	if ks.analogy == nil {
		return nil
	}

	topicID := nodeID(topic)

	// Find similar_to edges.
	edges := ks.graph.EdgesFrom(topicID)
	var results []SynthesizedKnowledge

	for _, edge := range edges {
		if edge.Relation != RelSimilarTo {
			continue
		}
		analogNode := ks.graph.GetNode(edge.To)
		if analogNode == nil {
			continue
		}

		// Gather what we know about the analogous concept.
		analogEdges := ks.graph.EdgesFrom(edge.To)
		var analogFacts []string
		var evidence []SynthesisEvidence

		for _, ae := range analogEdges {
			if ae.Relation == RelDescribedAs || ae.Relation == RelSimilarTo {
				continue
			}
			targetNode := ks.graph.GetNode(ae.To)
			if targetNode == nil {
				continue
			}
			fact := fmt.Sprintf("%s %s %s", analogNode.Label, ae.Relation, targetNode.Label)
			analogFacts = append(analogFacts, fact)
			evidence = append(evidence, SynthesisEvidence{
				Fact:      fact,
				Relevance: fmt.Sprintf("%s is similar to %s", topic, analogNode.Label),
				Source:    "analogy with " + analogNode.Label,
			})
		}

		if len(analogFacts) == 0 {
			continue
		}

		claim := fmt.Sprintf("%s is similar to %s. Since %s, %s may share similar characteristics.",
			topic, analogNode.Label,
			analogFacts[0],
			topic,
		)

		conf := capConfidence(edge.Confidence * 0.5)

		results = append(results, SynthesizedKnowledge{
			Topic:      topic,
			Claim:      claim,
			Strategy:   StratAnalogy,
			Evidence:   evidence,
			Confidence: conf,
			Qualifier:  fmt.Sprintf("By analogy with %s", analogNode.Label),
			Caveat:     fmt.Sprintf("This is reasoning by analogy with %s, not direct knowledge about %s.", analogNode.Label, topic),
		})
	}

	// Also try finding siblings through shared is_a parents.
	for _, edge := range edges {
		if edge.Relation != RelIsA {
			continue
		}
		parentID := edge.To
		parentNode := ks.graph.GetNode(parentID)
		if parentNode == nil {
			continue
		}

		// Find siblings: other nodes that are also is_a this parent.
		inEdges := ks.graph.EdgesTo(parentID)
		for _, ie := range inEdges {
			if ie.Relation != RelIsA || ie.From == topicID {
				continue
			}
			siblingNode := ks.graph.GetNode(ie.From)
			if siblingNode == nil {
				continue
			}

			// What do we know about this sibling?
			sibEdges := ks.graph.EdgesFrom(ie.From)
			var sibFacts []string
			var evidence []SynthesisEvidence

			for _, se := range sibEdges {
				if se.Relation == RelDescribedAs || se.Relation == RelIsA {
					continue
				}
				targetNode := ks.graph.GetNode(se.To)
				if targetNode == nil {
					continue
				}
				fact := fmt.Sprintf("%s %s %s", siblingNode.Label, se.Relation, targetNode.Label)
				sibFacts = append(sibFacts, fact)
				evidence = append(evidence, SynthesisEvidence{
					Fact:      fact,
					Relevance: fmt.Sprintf("Both %s and %s are types of %s", topic, siblingNode.Label, parentNode.Label),
					Source:    "sibling analogy via " + parentNode.Label,
				})
			}

			if len(sibFacts) == 0 {
				continue
			}

			claim := fmt.Sprintf("Both %s and %s are types of %s. Since %s, %s as a fellow %s may share some of these traits.",
				topic, siblingNode.Label, parentNode.Label,
				sibFacts[0],
				topic, parentNode.Label,
			)

			conf := capConfidence(0.35)

			results = append(results, SynthesizedKnowledge{
				Topic:      topic,
				Claim:      claim,
				Strategy:   StratAnalogy,
				Evidence:   evidence,
				Confidence: conf,
				Qualifier:  fmt.Sprintf("By analogy with sibling concept %s (both are %s)", siblingNode.Label, parentNode.Label),
				Caveat:     fmt.Sprintf("This is inferred from %s, a sibling concept, not directly known about %s.", siblingNode.Label, topic),
			})
		}
	}

	return results
}

// synthesizeCausalChain: Follow causal/relational paths. If A causes B
// and B causes C, conclude A probably causes C.
func (ks *KnowledgeSynthesizer) synthesizeCausalChain(topic string) []SynthesizedKnowledge {
	topicID := nodeID(topic)
	edges := ks.graph.EdgesFrom(topicID)

	var results []SynthesizedKnowledge

	for _, edge := range edges {
		if edge.Relation != RelCauses {
			continue
		}
		midNode := ks.graph.GetNode(edge.To)
		if midNode == nil {
			continue
		}

		// Follow the chain one more hop.
		secondEdges := ks.graph.EdgesFrom(edge.To)
		for _, se := range secondEdges {
			if se.Relation != RelCauses {
				continue
			}
			endNode := ks.graph.GetNode(se.To)
			if endNode == nil {
				continue
			}

			evidence := []SynthesisEvidence{
				{
					Fact:      fmt.Sprintf("%s causes %s", topic, midNode.Label),
					Relevance: "first link in causal chain",
					Source:    "causal inference",
				},
				{
					Fact:      fmt.Sprintf("%s causes %s", midNode.Label, endNode.Label),
					Relevance: "second link in causal chain",
					Source:    "causal inference",
				},
			}

			claim := fmt.Sprintf("%s causes %s, which in turn causes %s. Therefore, %s likely contributes to %s.",
				topic, midNode.Label, endNode.Label, topic, endNode.Label,
			)

			conf := capConfidence(edge.Confidence * se.Confidence * 0.5)

			results = append(results, SynthesizedKnowledge{
				Topic:      topic,
				Claim:      claim,
				Strategy:   StratCausalChain,
				Evidence:   evidence,
				Confidence: conf,
				Qualifier:  fmt.Sprintf("Following a causal chain through %s", midNode.Label),
				Caveat:     "This is a multi-step causal inference; intermediate effects may alter the outcome.",
			})
		}
	}

	return results
}

// synthesizeContrastive: Find what X is NOT like. "X is not Y, and Y has
// these properties, so X probably lacks them..."
func (ks *KnowledgeSynthesizer) synthesizeContrastive(topic string) []SynthesizedKnowledge {
	topicID := nodeID(topic)
	edges := ks.graph.EdgesFrom(topicID)

	var results []SynthesizedKnowledge

	for _, edge := range edges {
		if edge.Relation != RelOppositeOf {
			continue
		}
		oppositeNode := ks.graph.GetNode(edge.To)
		if oppositeNode == nil {
			continue
		}

		// Gather what we know about the opposite concept.
		oppEdges := ks.graph.EdgesFrom(edge.To)
		var oppProps []string
		var evidence []SynthesisEvidence

		for _, oe := range oppEdges {
			if oe.Relation == RelDescribedAs || oe.Relation == RelOppositeOf {
				continue
			}
			targetNode := ks.graph.GetNode(oe.To)
			if targetNode == nil {
				continue
			}
			fact := fmt.Sprintf("%s %s %s", oppositeNode.Label, oe.Relation, targetNode.Label)
			oppProps = append(oppProps, targetNode.Label)
			evidence = append(evidence, SynthesisEvidence{
				Fact:      fact,
				Relevance: fmt.Sprintf("%s is the opposite of %s, so %s likely lacks this", topic, oppositeNode.Label, topic),
				Source:    "contrastive reasoning with " + oppositeNode.Label,
			})
		}

		if len(oppProps) == 0 {
			continue
		}

		claim := fmt.Sprintf("%s is the opposite of %s. Since %s is associated with %s, %s likely differs in these respects.",
			topic, oppositeNode.Label,
			oppositeNode.Label, strings.Join(oppProps, ", "),
			topic,
		)

		conf := capConfidence(edge.Confidence * 0.45)

		results = append(results, SynthesizedKnowledge{
			Topic:      topic,
			Claim:      claim,
			Strategy:   StratContrastive,
			Evidence:   evidence,
			Confidence: conf,
			Qualifier:  fmt.Sprintf("By contrast with %s", oppositeNode.Label),
			Caveat:     fmt.Sprintf("This is inferred by contrast with %s; opposites don't always differ on every dimension.", oppositeNode.Label),
		})
	}

	return results
}

// synthesizeCompositional: Combine partial knowledge from multiple graph
// paths that mention the topic. "I know A about X from one edge and B
// about X from another..."
func (ks *KnowledgeSynthesizer) synthesizeCompositional(topic string) []SynthesizedKnowledge {
	topicID := nodeID(topic)

	// Gather outgoing facts (excluding described_as).
	outEdges := ks.graph.EdgesFrom(topicID)
	// Gather incoming facts.
	inEdges := ks.graph.EdgesTo(topicID)

	var evidence []SynthesisEvidence
	var factFragments []string

	for _, edge := range outEdges {
		if edge.Relation == RelDescribedAs {
			continue
		}
		targetNode := ks.graph.GetNode(edge.To)
		if targetNode == nil {
			continue
		}
		fact := fmt.Sprintf("%s %s %s", topic, edge.Relation, targetNode.Label)
		factFragments = append(factFragments, fact)
		evidence = append(evidence, SynthesisEvidence{
			Fact:      fact,
			Relevance: "direct partial knowledge about " + topic,
			Source:    "graph outgoing edge",
		})
	}

	for _, edge := range inEdges {
		fromNode := ks.graph.GetNode(edge.From)
		if fromNode == nil {
			continue
		}
		fact := fmt.Sprintf("%s %s %s", fromNode.Label, edge.Relation, topic)
		factFragments = append(factFragments, fact)
		evidence = append(evidence, SynthesisEvidence{
			Fact:      fact,
			Relevance: "incoming reference to " + topic,
			Source:    "graph incoming edge",
		})
	}

	// Only produce a compositional synthesis if we have at least 2 pieces
	// from different angles (outgoing + incoming, or multiple relations).
	if len(factFragments) < 2 {
		return nil
	}

	claim := fmt.Sprintf("Combining partial knowledge: %s. Together, these fragments paint a picture of %s.",
		strings.Join(factFragments, "; "),
		topic,
	)

	conf := capConfidence(0.35 + 0.05*math.Min(float64(len(factFragments)), 5.0))

	return []SynthesizedKnowledge{{
		Topic:      topic,
		Claim:      claim,
		Strategy:   StratCompositional,
		Evidence:   evidence,
		Confidence: conf,
		Qualifier:  "Combining what I know from multiple sources",
		Caveat:     "These are fragments combined; the full picture of " + topic + " may differ.",
	}}
}

// -----------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------

// countDirectFacts counts non-described_as, non-inferred outgoing edges.
func (ks *KnowledgeSynthesizer) countDirectFacts(topic string) int {
	topicID := nodeID(topic)
	edges := ks.graph.EdgesFrom(topicID)
	count := 0
	for _, edge := range edges {
		if edge.Relation != RelDescribedAs && !edge.Inferred {
			count++
		}
	}
	return count
}

// hasAdjacentKnowledge returns true if the topic has any reachable
// knowledge in the graph: parent categories, component words, similar
// concepts, or incoming references.
func (ks *KnowledgeSynthesizer) hasAdjacentKnowledge(topic string) bool {
	topicID := nodeID(topic)

	// Check for direct edges (even if fewer than 2).
	if edges := ks.graph.EdgesFrom(topicID); len(edges) > 0 {
		return true
	}
	if edges := ks.graph.EdgesTo(topicID); len(edges) > 0 {
		return true
	}

	// Check component words of compound topics.
	words := strings.Fields(topic)
	if len(words) >= 2 {
		for _, word := range words {
			if ks.graph.GetNode(nodeID(word)) != nil {
				return true
			}
		}
	}

	return false
}

// capConfidence ensures a confidence value never exceeds the synthesis cap.
func capConfidence(c float64) float64 {
	if c > maxSynthesisConfidence {
		return maxSynthesisConfidence
	}
	if c < 0 {
		return 0
	}
	return c
}

// confidenceLevel returns a human-readable confidence descriptor.
func confidenceLevel(c float64) string {
	switch {
	case c >= 0.6:
		return "moderate"
	case c >= 0.4:
		return "low-moderate"
	case c >= 0.2:
		return "low"
	default:
		return "very low"
	}
}

// summarizeProperties joins property names into a readable clause.
func summarizeProperties(props []string) string {
	if len(props) == 0 {
		return "has certain characteristics"
	}
	if len(props) == 1 {
		return "involves " + props[0]
	}
	if len(props) == 2 {
		return "involves " + props[0] + " and " + props[1]
	}
	return "involves " + strings.Join(props[:len(props)-1], ", ") + ", and " + props[len(props)-1]
}

// buildExplanation constructs a human-readable summary of the synthesis.
func (ks *KnowledgeSynthesizer) buildExplanation(result *SynthesisResult) string {
	if len(result.Synthesized) == 0 {
		return fmt.Sprintf("Could not synthesize knowledge about %s from adjacent concepts.", result.Topic)
	}

	strategies := make(map[SynthesisStrategy]int)
	for _, sk := range result.Synthesized {
		strategies[sk.Strategy]++
	}

	var parts []string
	for strat, count := range strategies {
		parts = append(parts, fmt.Sprintf("%d claim(s) via %s", count, strat.String()))
	}

	return fmt.Sprintf("Synthesized %d insight(s) about %s: %s. Overall confidence: %s.",
		len(result.Synthesized),
		result.Topic,
		strings.Join(parts, ", "),
		confidenceLevel(result.OverallConf),
	)
}
