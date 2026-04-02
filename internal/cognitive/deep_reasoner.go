package cognitive

import (
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strings"
)

// -----------------------------------------------------------------------
// Deep Reasoner — multi-step structured reasoning with explicit chains.
// Goes deeper than the existing 2-3 hop traversal by decomposing
// questions into sub-questions and chaining their answers together.
// -----------------------------------------------------------------------

// DeepReasoner performs multi-step structured reasoning with explicit
// reasoning chains. Extends the ReasoningEngine's 2-3 hop graph traversal
// to handle "why", "how", and "what if" questions through decomposition,
// sub-question answering, chaining, and synthesis.
type DeepReasoner struct {
	graph     *CognitiveGraph
	knowledge string // path to knowledge directory
}

// ReasoningChainStep is one explicit step in a reasoning chain.
type ReasoningChainStep struct {
	StepNum    int
	Premise    string  // what we know
	Reasoning  string  // what we can infer
	Conclusion string  // what we conclude
	Confidence float64
	Source     string // where the premise came from
}

// DeepReasoningResult is the output of multi-step reasoning.
type DeepReasoningResult struct {
	Question    string
	Steps       []ReasoningChainStep
	FinalAnswer string
	Confidence  float64
	Trace       string // human-readable reasoning trace
}

// NewDeepReasoner creates a DeepReasoner backed by a cognitive graph and
// knowledge text directory.
func NewDeepReasoner(graph *CognitiveGraph, knowledgeDir string) *DeepReasoner {
	return &DeepReasoner{
		graph:     graph,
		knowledge: knowledgeDir,
	}
}

// Reason performs multi-step structured reasoning to answer a question.
// Steps: decompose → answer sub-questions → chain → synthesize.
// Returns nil if the question cannot be decomposed or answered.
func (dr *DeepReasoner) Reason(question string) *DeepReasoningResult {
	question = strings.TrimSpace(question)
	if question == "" || dr.graph == nil {
		return nil
	}

	// 1. Decompose the question into sub-questions.
	subQuestions := dr.Decompose(question)
	if len(subQuestions) == 0 {
		return nil
	}

	// 2. Answer each sub-question from the knowledge graph and text.
	var steps []ReasoningChainStep
	prevConclusion := ""
	seenAnswers := make(map[string]bool) // dedup identical answers

	// Extract the original topic for relevance checking.
	origTopic := extractDeepTopic(strings.ToLower(question))
	if origTopic == "" {
		// Fallback: extract from first sub-question
		if len(subQuestions) > 0 {
			origTopic = dr.extractTopic(subQuestions[0])
		}
	}

	for i, sq := range subQuestions {
		step := ReasoningChainStep{
			StepNum: i + 1,
		}

		// Search the knowledge graph for facts.
		premise, source := dr.answerSubQuestion(sq)

		// Dedup: if this answer is identical to a previous one, skip it.
		if premise != "" {
			premiseKey := strings.ToLower(strings.TrimSpace(premise))
			if len(premiseKey) > 80 {
				premiseKey = premiseKey[:80]
			}
			if seenAnswers[premiseKey] {
				premise = ""
				source = "none"
			} else {
				seenAnswers[premiseKey] = true
			}
		}

		// Relevance check: if the answer doesn't mention any word from the
		// original topic, it's probably from a loosely related graph node.
		if premise != "" && origTopic != "" {
			premLower := strings.ToLower(premise)
			topicWords := strings.Fields(strings.ToLower(origTopic))
			relevant := false
			for _, tw := range topicWords {
				if len(tw) > 3 && strings.Contains(premLower, tw) {
					relevant = true
					break
				}
			}
			if !relevant {
				premise = ""
				source = "none"
			}
		}

		// If the previous step produced a conclusion, incorporate it.
		if prevConclusion != "" && premise != "" {
			step.Premise = prevConclusion
			step.Reasoning = fmt.Sprintf("Given that %s, and we also know that %s",
				lowerFirst(prevConclusion), lowerFirst(premise))
			step.Conclusion = dr.synthesizeStep(sq, prevConclusion, premise)
			step.Source = source
		} else if premise != "" {
			step.Premise = premise
			step.Reasoning = fmt.Sprintf("We know that %s", lowerFirst(premise))
			step.Conclusion = premise
			step.Source = source
		} else {
			// No data found for this sub-question — record but continue.
			step.Premise = sq
			step.Reasoning = fmt.Sprintf("No information found about: %s", sq)
			step.Conclusion = ""
			step.Source = "none"
		}

		// Confidence is based on whether we found real data.
		if step.Conclusion != "" {
			step.Confidence = 0.7
			if source == "graph_fact" || source == "knowledge_text" {
				step.Confidence = 0.8
			}
		} else {
			step.Confidence = 0.1
		}

		steps = append(steps, step)
		if step.Conclusion != "" {
			prevConclusion = step.Conclusion
		}
	}

	// 3. If no steps produced any conclusion, bail.
	hasConclusion := false
	for _, s := range steps {
		if s.Conclusion != "" {
			hasConclusion = true
			break
		}
	}
	if !hasConclusion {
		return nil
	}

	// 4. Synthesize a final answer from the chain.
	result := &DeepReasoningResult{
		Question: question,
		Steps:    steps,
	}

	result.FinalAnswer = dr.synthesizeFinal(question, steps)
	result.Confidence = dr.computeConfidence(steps)
	result.Trace = dr.buildTrace(steps)

	if result.FinalAnswer == "" {
		return nil
	}

	return result
}

// -----------------------------------------------------------------------
// Question Decomposition
// -----------------------------------------------------------------------

// decompositionRule maps a question pattern to a function that produces
// sub-questions from the matched groups.
type decompositionRule struct {
	pattern *regexp.Regexp
	build   func(matches []string, original string) []string
}

// decompositionRules defines how complex questions are broken down.
var decompositionRules = []decompositionRule{
	// "Why is X Y?" → extract the subject X and predicate Y properly.
	// "Why is the sky blue?" → topic = "sky", predicate = "blue"
	// "Why are cats afraid of water?" → topic = "cats"
	{
		pattern: regexp.MustCompile(`(?i)^why\s+(?:is|are|does|do|did|was|were|has|have|had|can|could|would|should|might)\s+(.+?)[\?]?$`),
		build: func(m []string, original string) []string {
			raw := strings.TrimRight(m[1], "? ")
			// Try to split into subject + predicate: "the sky blue" → "sky"
			words := strings.Fields(raw)
			topic := raw
			if len(words) >= 2 {
				// Skip articles
				start := 0
				if words[0] == "the" || words[0] == "a" || words[0] == "an" {
					start = 1
				}
				// The subject is typically the first noun phrase
				if start < len(words) {
					topic = strings.Join(words[start:], " ")
				}
			}
			return []string{
				"What is " + topic + "?",
				"What causes " + topic + "?",
			}
		},
	},
	// Simpler "Why X?" without auxiliary verb.
	{
		pattern: regexp.MustCompile(`(?i)^why\s+(.+?)[\?]?$`),
		build: func(m []string, _ string) []string {
			topic := strings.TrimRight(m[1], "? ")
			return []string{
				"What is " + topic + "?",
				"What causes " + topic + "?",
			}
		},
	},

	// "How does X affect Y?" → "What is X?" + "What is Y?" + "How are X and Y related?"
	{
		pattern: regexp.MustCompile(`(?i)^how\s+(?:does|do|did|can|could|would|might)\s+(.+?)\s+(?:affect|impact|influence|change|modify|alter)\s+(.+?)[\?]?$`),
		build: func(m []string, _ string) []string {
			x := strings.TrimRight(m[1], "? ")
			y := strings.TrimRight(m[2], "? ")
			return []string{
				"What is " + x + "?",
				"What is " + y + "?",
				"How are " + x + " and " + y + " related?",
			}
		},
	},

	// "What would happen if X?" → "What is X?" + "What depends on X?" + "What are the effects?"
	{
		pattern: regexp.MustCompile(`(?i)^what\s+(?:would|will|could|might|can)\s+happen\s+if\s+(.+?)[\?]?$`),
		build: func(m []string, _ string) []string {
			topic := strings.TrimRight(m[1], "? ")
			return []string{
				"What is " + topic + "?",
				"What depends on " + topic + "?",
				"What are the effects of " + topic + "?",
			}
		},
	},

	// "Is X better than Y?" → properties of X, properties of Y, compare.
	{
		pattern: regexp.MustCompile(`(?i)^is\s+(.+?)\s+better\s+than\s+(.+?)[\?]?$`),
		build: func(m []string, _ string) []string {
			x := strings.TrimRight(m[1], "? ")
			y := strings.TrimRight(m[2], "? ")
			return []string{
				"What are the properties of " + x + "?",
				"What are the properties of " + y + "?",
				"How do " + x + " and " + y + " compare?",
			}
		},
	},

	// "What is the relationship between X and Y?"
	{
		pattern: regexp.MustCompile(`(?i)^what\s+is\s+the\s+(?:relationship|connection|link)\s+between\s+(.+?)\s+and\s+(.+?)[\?]?$`),
		build: func(m []string, _ string) []string {
			x := strings.TrimRight(m[1], "? ")
			y := strings.TrimRight(m[2], "? ")
			return []string{
				"What is " + x + "?",
				"What is " + y + "?",
				"How are " + x + " and " + y + " related?",
			}
		},
	},

	// "How does X work?" → "What is X?" + "What are the components of X?" + "How do they interact?"
	{
		pattern: regexp.MustCompile(`(?i)^how\s+(?:does|do|did)\s+(.+?)\s+work[\?]?$`),
		build: func(m []string, _ string) []string {
			topic := strings.TrimRight(m[1], "? ")
			return []string{
				"What is " + topic + "?",
				"What are the parts of " + topic + "?",
				"How does " + topic + " function?",
			}
		},
	},
}

// Decompose splits a complex question into ordered sub-questions.
func (dr *DeepReasoner) Decompose(question string) []string {
	question = strings.TrimSpace(question)
	if question == "" {
		return nil
	}

	for _, rule := range decompositionRules {
		matches := rule.pattern.FindStringSubmatch(question)
		if matches != nil {
			return rule.build(matches, question)
		}
	}

	// Fallback for unmatched patterns: if the question looks deep
	// (contains "why", "how", causal language), decompose generically.
	lower := strings.ToLower(question)
	topic := extractDeepTopic(lower)
	if topic == "" {
		return nil
	}

	if strings.HasPrefix(lower, "why") {
		return []string{
			"What is " + topic + "?",
			"What causes " + topic + "?",
		}
	}
	if strings.HasPrefix(lower, "how") {
		return []string{
			"What is " + topic + "?",
			"How does " + topic + " work?",
		}
	}

	return nil
}

// extractDeepTopic pulls the core topic from a question for fallback decomposition.
func extractDeepTopic(question string) string {
	question = strings.TrimRight(question, "?!. ")

	// Remove common question words.
	for _, prefix := range []string{
		"why is ", "why are ", "why does ", "why do ", "why did ",
		"why was ", "why were ", "why has ", "why have ", "why had ",
		"why can ", "why could ", "why would ", "why should ", "why might ",
		"how does ", "how do ", "how did ", "how is ", "how are ",
		"how can ", "how could ", "how would ",
		"what causes ", "what caused ",
	} {
		if strings.HasPrefix(question, prefix) {
			return strings.TrimSpace(question[len(prefix):])
		}
	}

	// Strip leading why/how.
	if strings.HasPrefix(question, "why ") {
		return strings.TrimSpace(question[4:])
	}
	if strings.HasPrefix(question, "how ") {
		return strings.TrimSpace(question[4:])
	}

	return ""
}

// -----------------------------------------------------------------------
// Sub-question answering — graph + knowledge text search
// -----------------------------------------------------------------------

// answerSubQuestion searches the knowledge graph and text files for an answer.
// Returns (answer, source) where source is "graph_fact", "knowledge_text", or "".
func (dr *DeepReasoner) answerSubQuestion(question string) (string, string) {
	// Extract the topic/concept from the sub-question.
	topic := dr.extractTopic(question)
	if topic == "" {
		return "", ""
	}

	// Build a list of candidate topics: the full topic first, then individual
	// significant words as fallbacks. This handles decomposed topics like
	// "gravity exist" where the graph only has "gravity".
	candidates := []string{topic}
	words := strings.Fields(topic)
	if len(words) > 1 {
		for _, w := range words {
			w = strings.ToLower(w)
			if len(w) > 3 && w != topic {
				candidates = append(candidates, w)
			}
		}
	}

	for _, candidate := range candidates {
		answer, source := dr.lookupTopic(candidate)
		if answer != "" {
			return answer, source
		}
	}

	return "", ""
}

// lookupTopic searches for facts about a single topic string.
func (dr *DeepReasoner) lookupTopic(topic string) (string, string) {
	// Strategy 1: Look up facts in the knowledge graph.
	facts := dr.graph.LookupFacts(topic, 4)
	if len(facts) > 0 {
		return strings.Join(facts, ". "), "graph_fact"
	}

	// Strategy 2: Search graph nodes and collect edge information.
	nodes := dr.graph.FindNodes(topic)
	if len(nodes) > 0 {
		var collected []string
		for _, node := range nodes {
			// Get description if available.
			desc := dr.graph.LookupDescription(node.Label)
			if desc != "" && len(desc) >= 20 {
				collected = append(collected, desc)
				break // one good description is enough
			}

			// Get outgoing edge facts.
			edges := dr.graph.EdgesFrom(node.Label)
			for _, edge := range edges {
				if edge.Relation == RelDescribedAs {
					continue
				}
				targetNode := dr.graph.GetNode(edge.To)
				if targetNode != nil {
					fact := edgeToNaturalLanguage(node.Label, edge.Relation, targetNode.Label)
					if fact != "" {
						collected = append(collected, fact)
					}
				}
				if len(collected) >= 3 {
					break
				}
			}
			if len(collected) > 0 {
				break
			}
		}
		if len(collected) > 0 {
			return strings.Join(collected, ". "), "graph_fact"
		}
	}

	// Strategy 3: Search knowledge text files for relevant paragraphs.
	if dr.knowledge != "" {
		if paragraph := dr.searchKnowledgeText(topic); paragraph != "" {
			// Return first 2 sentences to keep steps concise.
			sentences := splitSentences(paragraph)
			if len(sentences) > 2 {
				sentences = sentences[:2]
			}
			return strings.Join(sentences, " "), "knowledge_text"
		}
	}

	return "", ""
}

// extractTopic pulls the core concept from a sub-question.
func (dr *DeepReasoner) extractTopic(question string) string {
	q := strings.ToLower(strings.TrimRight(strings.TrimSpace(question), "?!."))

	// "What is X?" → X
	for _, prefix := range []string{
		"what is ", "what are ", "what is the ",
		"what causes ", "what caused ",
		"what depends on ", "what are the effects of ",
		"what are the properties of ", "what are the parts of ",
		"how does ", "how do ", "how are ", "how is ",
		"how does ", "how did ",
	} {
		if strings.HasPrefix(q, prefix) {
			rest := strings.TrimSpace(q[len(prefix):])
			// Remove trailing phrases.
			for _, suffix := range []string{" work", " function", " related", " compare"} {
				rest = strings.TrimSuffix(rest, suffix)
			}
			return strings.TrimSpace(rest)
		}
	}

	// "How are X and Y related?" → try X
	if strings.HasPrefix(q, "how are ") {
		rest := q[8:]
		if idx := strings.Index(rest, " and "); idx >= 0 {
			return strings.TrimSpace(rest[:idx])
		}
	}

	// "How do X and Y compare?" → try X
	if strings.HasPrefix(q, "how do ") {
		rest := q[7:]
		if idx := strings.Index(rest, " and "); idx >= 0 {
			return strings.TrimSpace(rest[:idx])
		}
	}

	return q
}

// searchKnowledgeText searches .txt files in the knowledge directory for
// paragraphs mentioning the topic. Returns the best matching paragraph.
func (dr *DeepReasoner) searchKnowledgeText(topic string) string {
	if dr.knowledge == "" {
		return ""
	}

	files, err := filepath.Glob(filepath.Join(dr.knowledge, "*.txt"))
	if err != nil || len(files) == 0 {
		return ""
	}

	topicLower := strings.ToLower(topic)
	var bestParagraph string
	bestScore := 0

	for _, f := range files {
		data, err := os.ReadFile(f)
		if err != nil {
			continue
		}

		paragraphs := strings.Split(string(data), "\n\n")
		for _, p := range paragraphs {
			p = strings.TrimSpace(p)
			if len(p) < 40 {
				continue
			}
			pLower := strings.ToLower(p)

			// Score: how well does this paragraph match the topic?
			// Use whole-word matching to prevent "blue" from matching "blues".
			score := 0
			if containsWholeWord(pLower, topicLower) {
				score += 10
			}
			// Bonus for topic at start of paragraph.
			if strings.HasPrefix(pLower, topicLower+" ") || strings.HasPrefix(pLower, topicLower+",") {
				score += 20
			}
			// Bonus for matching individual words (whole-word only).
			words := strings.Fields(topicLower)
			for _, w := range words {
				if len(w) > 3 && containsWholeWord(pLower, w) {
					score += 2
				}
			}

			if score > bestScore {
				bestScore = score
				bestParagraph = p
			}
		}
	}

	if bestScore >= 10 {
		return bestParagraph
	}
	return ""
}

// splitSentences is defined in extractive.go — shared across the package.

// -----------------------------------------------------------------------
// Chain synthesis
// -----------------------------------------------------------------------

// synthesizeStep combines a previous conclusion with new evidence.
func (dr *DeepReasoner) synthesizeStep(question, prevConclusion, newEvidence string) string {
	// Try to form a concise inference.
	qLower := strings.ToLower(question)

	if strings.Contains(qLower, "cause") || strings.Contains(qLower, "why") {
		return fmt.Sprintf("Since %s, this helps explain that %s",
			lowerFirst(prevConclusion), lowerFirst(newEvidence))
	}

	if strings.Contains(qLower, "related") || strings.Contains(qLower, "connect") {
		return fmt.Sprintf("%s, and this connects to the fact that %s",
			prevConclusion, lowerFirst(newEvidence))
	}

	if strings.Contains(qLower, "effect") || strings.Contains(qLower, "depend") {
		return fmt.Sprintf("Because %s, it follows that %s",
			lowerFirst(prevConclusion), lowerFirst(newEvidence))
	}

	if strings.Contains(qLower, "compar") || strings.Contains(qLower, "propert") {
		return fmt.Sprintf("On one hand, %s. On the other hand, %s",
			lowerFirst(prevConclusion), lowerFirst(newEvidence))
	}

	// Generic chaining.
	return fmt.Sprintf("%s. Furthermore, %s",
		prevConclusion, lowerFirst(newEvidence))
}

// synthesizeFinal composes the final answer from all reasoning steps.
func (dr *DeepReasoner) synthesizeFinal(question string, steps []ReasoningChainStep) string {
	// Collect all non-empty conclusions.
	var conclusions []string
	for _, s := range steps {
		if s.Conclusion != "" {
			conclusions = append(conclusions, s.Conclusion)
		}
	}

	if len(conclusions) == 0 {
		return ""
	}

	// For single-conclusion results, just return it.
	if len(conclusions) == 1 {
		return conclusions[0]
	}

	// Use the last step's conclusion as the primary answer (it's the most
	// synthesized), preceded by a brief summary of the reasoning.
	last := conclusions[len(conclusions)-1]

	// Build a lead-in from earlier steps.
	var leadParts []string
	for _, c := range conclusions[:len(conclusions)-1] {
		// Trim to first sentence for brevity.
		if idx := strings.Index(c, ". "); idx > 0 {
			leadParts = append(leadParts, c[:idx+1])
		} else {
			leadParts = append(leadParts, c)
		}
	}

	if len(leadParts) > 0 {
		return strings.Join(leadParts, " ") + " " + last
	}

	return last
}

// computeConfidence averages confidence across steps, weighting non-empty
// conclusions higher.
func (dr *DeepReasoner) computeConfidence(steps []ReasoningChainStep) float64 {
	if len(steps) == 0 {
		return 0
	}

	total := 0.0
	weight := 0.0
	for _, s := range steps {
		w := 1.0
		if s.Conclusion != "" {
			w = 2.0
		}
		total += s.Confidence * w
		weight += w
	}

	if weight == 0 {
		return 0
	}
	return total / weight
}

// buildTrace creates a human-readable trace of the reasoning chain.
func (dr *DeepReasoner) buildTrace(steps []ReasoningChainStep) string {
	var lines []string
	for _, s := range steps {
		if s.Conclusion == "" {
			lines = append(lines, fmt.Sprintf("Step %d: %s (no information found)",
				s.StepNum, s.Reasoning))
			continue
		}

		line := fmt.Sprintf("Step %d: %s", s.StepNum, s.Reasoning)
		if s.Conclusion != s.Premise {
			line += fmt.Sprintf("\n  Therefore: %s", s.Conclusion)
		}
		lines = append(lines, line)
	}

	if len(lines) == 0 {
		return ""
	}

	return strings.Join(lines, "\n")
}

// -----------------------------------------------------------------------
// Deep question detection — used by the ActionRouter
// -----------------------------------------------------------------------

// deepQuestionPatterns match questions that benefit from deep reasoning.
var deepQuestionPatterns = []*regexp.Regexp{
	regexp.MustCompile(`(?i)^why\s`),
	regexp.MustCompile(`(?i)^how\s+(?:does|do|did|can|could|would|might)\s+.+\s+(?:affect|impact|influence|change)`),
	regexp.MustCompile(`(?i)^what\s+(?:would|will|could|might)\s+happen\s+if\s`),
	regexp.MustCompile(`(?i)^what\s+is\s+the\s+(?:relationship|connection|link)\s+between\s`),
	regexp.MustCompile(`(?i)^how\s+(?:does|do|did)\s+.+\s+work`),
	regexp.MustCompile(`(?i)^is\s+.+\s+better\s+than\s`),
}

// IsDeepQuestion returns true if the question would benefit from multi-step
// structured reasoning (why, causal, relational, hypothetical questions).
func IsDeepQuestion(question string) bool {
	question = strings.TrimSpace(question)
	for _, pat := range deepQuestionPatterns {
		if pat.MatchString(question) {
			return true
		}
	}
	return false
}

// -----------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------

// lowerFirst is defined in composer.go — shared across the package.
