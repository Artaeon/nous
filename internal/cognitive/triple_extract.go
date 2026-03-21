package cognitive

import (
	"fmt"
	"regexp"
	"strings"
)

// -----------------------------------------------------------------------
// Triple Extraction — turns natural language into graph relationships.
// Pure code, no LLM. Uses pattern matching on sentence structure.
// -----------------------------------------------------------------------

// Triple represents a subject-relation-object fact.
type Triple struct {
	Subject  string
	Relation RelType
	Object   string
}

// pattern is a compiled extraction rule.
type pattern struct {
	re      *regexp.Regexp
	rel     RelType
	subjIdx int // capture group for subject
	objIdx  int // capture group for object
}

// Compiled patterns for triple extraction.
// Order matters — more specific patterns first.
var triplePatterns = []pattern{
	// "X was founded in YEAR by Y"
	{regexp.MustCompile(`(?i)^(.+?)\s+was\s+founded\s+in\s+(\d{4})\s+by\s+(.+?)\.?$`), RelFoundedIn, 1, 2},
	// "X was founded by Y in YEAR" — subject=1, object=3 (founder)
	{regexp.MustCompile(`(?i)^(.+?)\s+was\s+founded\s+by\s+(.+?)\s+in\s+(\d{4})\.?$`), RelFoundedBy, 1, 2},
	// "X was founded by Y"
	{regexp.MustCompile(`(?i)^(.+?)\s+was\s+founded\s+by\s+(.+?)\.?$`), RelFoundedBy, 1, 2},
	// "X was founded in YEAR"
	{regexp.MustCompile(`(?i)^(.+?)\s+was\s+founded\s+in\s+(\d{4})\.?$`), RelFoundedIn, 1, 2},
	// "X was created by Y"
	{regexp.MustCompile(`(?i)^(.+?)\s+was\s+created\s+by\s+(.+?)\.?$`), RelCreatedBy, 1, 2},
	// "X is based in Y" / "X is located in Y" / "X is headquartered in Y"
	{regexp.MustCompile(`(?i)^(.+?)\s+(?:is|are)\s+(?:based|located|headquartered|situated)\s+in\s+(.+?)\.?$`), RelLocatedIn, 1, 2},
	// "X is in Y" (short form of location)
	{regexp.MustCompile(`(?i)^(.+?)\s+is\s+in\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\.?$`), RelLocatedIn, 1, 2},
	// "X is a Y" / "X is an Y" — definition
	{regexp.MustCompile(`(?i)^(.+?)\s+(?:is|are)\s+(?:a|an)\s+(.+?)\.?$`), RelIsA, 1, 2},
	// "X is Y" (described as — only when Y is an adjective-like phrase)
	{regexp.MustCompile(`(?i)^(.+?)\s+(?:is|are)\s+((?:very\s+|quite\s+|extremely\s+)?(?:fast|slow|safe|dangerous|popular|free|open[- ]source|powerful|lightweight|modern|ancient|old|new|small|large|big|simple|complex|easy|hard|beautiful|ugly|expensive|cheap|reliable|unreliable)\b.*)\.?$`), RelDescribedAs, 1, 2},
	// "X offers Y" / "X provides Y" / "X features Y"
	{regexp.MustCompile(`(?i)^(.+?)\s+(?:offers?|provides?|features?|includes?|supports?)\s+(.+?)\.?$`), RelOffers, 1, 2},
	// "X has Y"
	{regexp.MustCompile(`(?i)^(.+?)\s+(?:has|have)\s+(.+?)\.?$`), RelHas, 1, 2},
	// "X is part of Y" / "X belongs to Y"
	{regexp.MustCompile(`(?i)^(.+?)\s+(?:is\s+part\s+of|belongs?\s+to)\s+(.+?)\.?$`), RelPartOf, 1, 2},
	// "X is used for Y" / "X is designed for Y"
	{regexp.MustCompile(`(?i)^(.+?)\s+(?:is|are)\s+(?:used|designed|built|made)\s+for\s+(.+?)\.?$`), RelUsedFor, 1, 2},
	// "X is similar to Y" / "X is like Y"
	{regexp.MustCompile(`(?i)^(.+?)\s+(?:is|are)\s+(?:similar\s+to|like|comparable\s+to)\s+(.+?)\.?$`), RelSimilarTo, 1, 2},
	// "X causes Y" / "X leads to Y"
	{regexp.MustCompile(`(?i)^(.+?)\s+(?:causes?|leads?\s+to|results?\s+in)\s+(.+?)\.?$`), RelCauses, 1, 2},
}

// Additional patterns for the "founded in YEAR by PERSON" case — extracts TWO triples.
var foundedInByRe = regexp.MustCompile(`(?i)^(.+?)\s+was\s+founded\s+in\s+(\d{4})\s+by\s+(.+?)\.?$`)
var foundedByInRe = regexp.MustCompile(`(?i)^(.+?)\s+was\s+founded\s+by\s+(.+?)\s+in\s+(\d{4})\.?$`)

// Compound pattern: "X is a Y based in Z" → is_a + located_in
var isABasedInRe = regexp.MustCompile(`(?i)^(.+?)\s+(?:is|are)\s+(?:a|an)\s+(.+?)\s+(?:based|located|headquartered)\s+in\s+(.+?)\.?$`)

// ExtractTriples pulls structured relationships from a sentence.
func ExtractTriples(sentence string) []Triple {
	sentence = strings.TrimSpace(sentence)
	if sentence == "" || len(sentence) < 5 {
		return nil
	}

	var triples []Triple

	// Compound: "X is a Y based in Z" → TWO triples
	if m := isABasedInRe.FindStringSubmatch(sentence); len(m) >= 4 {
		subj := strings.TrimSpace(m[1])
		category := strings.TrimSpace(m[2])
		location := strings.TrimSpace(m[3])
		triples = append(triples,
			Triple{Subject: subj, Relation: RelIsA, Object: category},
			Triple{Subject: subj, Relation: RelLocatedIn, Object: location},
		)
		return triples
	}

	// Special case: "X was founded in YEAR by Y" → TWO triples
	if m := foundedInByRe.FindStringSubmatch(sentence); len(m) >= 4 {
		subj := strings.TrimSpace(m[1])
		year := strings.TrimSpace(m[2])
		founder := strings.TrimSpace(m[3])
		triples = append(triples,
			Triple{Subject: subj, Relation: RelFoundedIn, Object: year},
			Triple{Subject: subj, Relation: RelFoundedBy, Object: founder},
		)
		return triples
	}
	if m := foundedByInRe.FindStringSubmatch(sentence); len(m) >= 4 {
		subj := strings.TrimSpace(m[1])
		founder := strings.TrimSpace(m[2])
		year := strings.TrimSpace(m[3])
		triples = append(triples,
			Triple{Subject: subj, Relation: RelFoundedBy, Object: founder},
			Triple{Subject: subj, Relation: RelFoundedIn, Object: year},
		)
		return triples
	}

	// Try each pattern
	for _, p := range triplePatterns {
		m := p.re.FindStringSubmatch(sentence)
		if len(m) > p.objIdx {
			subj := strings.TrimSpace(m[p.subjIdx])
			obj := strings.TrimSpace(m[p.objIdx])
			if subj != "" && obj != "" && len(subj) < 100 && len(obj) < 100 {
				triples = append(triples, Triple{
					Subject:  subj,
					Relation: p.rel,
					Object:   obj,
				})
				break // first matching pattern wins
			}
		}
	}

	// Extract list items: "X offers A, B, and C" → multiple triples
	if len(triples) == 1 && (triples[0].Relation == RelOffers || triples[0].Relation == RelHas) {
		items := splitListItems(triples[0].Object)
		if len(items) > 1 {
			rel := triples[0].Relation
			subj := triples[0].Subject
			triples = nil
			for _, item := range items {
				item = strings.TrimSpace(item)
				if item != "" {
					triples = append(triples, Triple{Subject: subj, Relation: rel, Object: item})
				}
			}
		}
	}

	return triples
}

// splitListItems splits "A, B, and C" into ["A", "B", "C"].
func splitListItems(s string) []string {
	// Remove "and" / "or" at end
	s = strings.TrimSpace(s)
	s = strings.TrimSuffix(s, ".")

	// Split by comma
	parts := strings.Split(s, ",")
	if len(parts) <= 1 {
		return parts
	}

	var items []string
	for _, p := range parts {
		p = strings.TrimSpace(p)
		// Handle "and X" or "or X" in last item
		p = strings.TrimPrefix(p, "and ")
		p = strings.TrimPrefix(p, "or ")
		p = strings.TrimSpace(p)
		if p != "" {
			items = append(items, p)
		}
	}
	return items
}

// IngestToGraph extracts triples from text and adds them to the cognitive graph.
// Returns the number of relationships added.
func IngestToGraph(cg *CognitiveGraph, text, source, topic string) int {
	sentences := splitSentences(text)
	added := 0

	// Ensure the topic node exists
	if topic != "" {
		cg.EnsureNode(topic, NodeEntity)
	}

	for _, sent := range sentences {
		if isBoilerplate(sent) {
			continue
		}

		// Resolve anaphoric references ("The company", "It", "They") to topic
		if topic != "" {
			sent = resolveAnaphora(sent, topic)
		}

		triples := ExtractTriples(sent)
		for _, t := range triples {
			// Determine node types
			subjType := guessNodeType(t.Subject)
			objType := guessNodeType(t.Object)

			cg.mu.Lock()
			fromID := cg.ensureNodeLocked(t.Subject, subjType, source, 0.7)
			toID := cg.ensureNodeLocked(t.Object, objType, source, 0.6)
			cg.addEdgeLocked(fromID, toID, t.Relation, source, 0.7, false)
			cg.mu.Unlock()
			added++
		}

		// If the sentence mentions the topic but no triple was extracted,
		// add a generic "described_as" edge with the full sentence.
		if len(triples) == 0 && topic != "" && mentionsTopic(sent, topic) {
			// Store as a property of the topic node
			cg.mu.Lock()
			topicID := nodeID(topic)
			if n, ok := cg.nodes[topicID]; ok {
				key := fmt.Sprintf("fact_%d", len(n.Properties))
				n.Properties[key] = sent
			}
			cg.mu.Unlock()
		}
	}

	return added
}

// guessNodeType heuristically determines what type of node something is.
func guessNodeType(label string) NodeType {
	lower := strings.ToLower(label)

	// Years
	if len(label) == 4 && label[0] >= '1' && label[0] <= '2' {
		return NodeEvent
	}

	// Properties (adjective-like)
	adjectives := []string{"fast", "slow", "safe", "free", "open-source", "modern", "powerful"}
	for _, adj := range adjectives {
		if strings.Contains(lower, adj) {
			return NodeProperty
		}
	}

	// If it starts with uppercase, likely an entity
	if len(label) > 0 && label[0] >= 'A' && label[0] <= 'Z' {
		return NodeEntity
	}

	return NodeConcept
}

// resolveAnaphora replaces pronouns and anaphoric references with the topic.
// "The company was founded..." → "Stoicera was founded..."
func resolveAnaphora(sentence, topic string) string {
	// Only replace at sentence start (subject position)
	anaphors := []string{
		"The company", "The organization", "The startup", "The firm",
		"The project", "The platform", "The product", "The service",
		"The app", "The application", "The tool", "The system",
		"It ", "They ", "He ", "She ",
	}
	for _, a := range anaphors {
		if strings.HasPrefix(sentence, a) {
			return topic + sentence[len(a):]
		}
		lower := strings.ToLower(a)
		if strings.HasPrefix(strings.ToLower(sentence), lower) {
			return topic + sentence[len(a):]
		}
	}
	return sentence
}

// mentionsTopic checks if a sentence mentions the topic.
func mentionsTopic(sentence, topic string) bool {
	return strings.Contains(strings.ToLower(sentence), strings.ToLower(topic))
}

