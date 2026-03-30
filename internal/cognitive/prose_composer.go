package cognitive

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
)

// -----------------------------------------------------------------------
// Retrieval-Augmented Prose Composer
//
// Generates original-sounding prose by extracting CLAUSES from real
// human-written text and recombining them. Every phrase was written by
// a human; the combination is novel.
//
// Pipeline:
//   1. For each fact, retrieve the 3 best matching sentences from corpus
//   2. Extract clauses (split on commas, semicolons, relative pronouns)
//   3. Score clause relevance to the current fact
//   4. Combine the best clauses with appropriate connectors
//   5. Result: fluent prose from recombined human text
// -----------------------------------------------------------------------

// ProseComposer creates original-sounding prose from knowledge facts
// by recombining clauses extracted from human-written text.
type ProseComposer struct {
	mu          sync.RWMutex
	paragraphs  []string // cached paragraphs from knowledge files
	loaded      bool
	knowledgeDir string
}

// NewProseComposer creates a prose composer backed by knowledge text files.
func NewProseComposer(knowledgeDir string) *ProseComposer {
	return &ProseComposer{knowledgeDir: knowledgeDir}
}

// Clause is a fragment of a sentence with its source and relevance metadata.
type Clause struct {
	Text     string  // the clause text
	Source   string  // which sentence it came from
	Role     string  // "definition", "property", "origin", "usage", "detail"
	Keywords []string // content words in this clause
}

// ComposeAbout generates a multi-sentence prose passage about a topic
// using facts from the knowledge graph.
func (pc *ProseComposer) ComposeAbout(topic string, facts []edgeFact) string {
	if len(facts) == 0 {
		return ""
	}

	pc.loadParagraphs()

	// 1. For each fact, find relevant clauses from the corpus
	var allClauses []Clause
	for _, fact := range facts {
		clauses := pc.findClausesForFact(topic, fact)
		allClauses = append(allClauses, clauses...)
	}

	if len(allClauses) == 0 {
		// Fallback: direct NLG-style realization
		return pc.fallbackRealize(topic, facts)
	}

	// 2. Deduplicate clauses
	allClauses = deduplicateClauses(allClauses)

	// 3. Group by role and order: definition → origin → property → usage → detail
	groups := groupClausesByRole(allClauses)

	// 4. Build prose paragraphs from clause groups
	return buildProseFromGroups(topic, groups, facts)
}

// findClausesForFact retrieves and extracts clauses relevant to a single fact.
func (pc *ProseComposer) findClausesForFact(topic string, fact edgeFact) []Clause {
	// Find sentences mentioning the topic or fact object
	candidates := pc.findRelevantSentences(topic, fact.Object, 5)

	var clauses []Clause
	role := factRole(fact.Relation)

	for _, sent := range candidates {
		// Split sentence into clauses
		parts := extractClauses(sent)
		for _, part := range parts {
			part = strings.TrimSpace(part)
			if len(part) < 15 || len(part) > 200 {
				continue
			}
			// Score relevance to this fact
			if clauseRelevance(part, topic, fact) > 0.2 {
				clauses = append(clauses, Clause{
					Text:     part,
					Source:   sent,
					Role:     role,
					Keywords: clauseContentWords(part),
				})
			}
		}
	}

	return clauses
}

// findRelevantSentences searches the knowledge corpus for sentences
// mentioning the topic or object.
func (pc *ProseComposer) findRelevantSentences(topic, object string, maxResults int) []string {
	pc.mu.RLock()
	paras := pc.paragraphs
	pc.mu.RUnlock()

	topicLower := strings.ToLower(topic)
	objectLower := strings.ToLower(object)

	var results []string
	seen := make(map[string]bool)

	for _, para := range paras {
		paraLower := strings.ToLower(para)
		if !strings.Contains(paraLower, topicLower) && !strings.Contains(paraLower, objectLower) {
			continue
		}

		for _, sent := range splitProseSentences(para) {
			sentLower := strings.ToLower(sent)
			if seen[sentLower] || len(sent) < 30 {
				continue
			}
			if strings.Contains(sentLower, topicLower) || strings.Contains(sentLower, objectLower) {
				seen[sentLower] = true
				results = append(results, sent)
				if len(results) >= maxResults {
					return results
				}
			}
		}
	}

	return results
}

// extractClauses splits a sentence into clause-level fragments.
// Splits on commas, semicolons, relative pronouns, and conjunctions
// that introduce independent clauses.
func extractClauses(sentence string) []string {
	// First split on semicolons (strongest boundary)
	parts := strings.Split(sentence, ";")

	var clauses []string
	for _, part := range parts {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}

		// Split on clause-introducing patterns
		subClauses := splitOnClauseMarkers(part)
		clauses = append(clauses, subClauses...)
	}

	return clauses
}

// splitOnClauseMarkers splits a clause on relative pronouns and
// coordinating conjunctions that introduce substantive clauses.
func splitOnClauseMarkers(text string) []string {
	// Markers that introduce a new clause worth extracting
	markers := []string{
		", which ", ", where ", ", while ", ", whereas ",
		", including ", ", enabling ", ", making ",
		", and ", ", but ", ", yet ",
	}

	for _, m := range markers {
		idx := strings.Index(text, m)
		if idx > 20 { // don't split very short initial clauses
			first := strings.TrimSpace(text[:idx])
			rest := strings.TrimSpace(text[idx+len(m):])
			if len(first) > 15 && len(rest) > 15 {
				return []string{first, rest}
			}
		}
	}

	return []string{text}
}

// clauseRelevance scores how relevant a clause is to a fact.
func clauseRelevance(clause, topic string, fact edgeFact) float64 {
	lower := strings.ToLower(clause)
	topicLower := strings.ToLower(topic)
	objectLower := strings.ToLower(fact.Object)

	score := 0.0

	// Mentions the topic
	if strings.Contains(lower, topicLower) {
		score += 0.4
	}

	// Mentions the object
	if strings.Contains(lower, objectLower) {
		score += 0.3
	}

	// Contains relation-relevant verbs
	relVerbs := relationVerbs(fact.Relation)
	for _, v := range relVerbs {
		if strings.Contains(lower, v) {
			score += 0.2
			break
		}
	}

	// Length bonus: medium clauses (20-100 chars) are best
	if len(clause) >= 20 && len(clause) <= 100 {
		score += 0.1
	}

	return score
}

// relationVerbs returns verbs associated with a relation type.
func relationVerbs(rel RelType) []string {
	switch rel {
	case RelIsA:
		return []string{"is a", "is an", "refers to", "describes"}
	case RelCreatedBy, RelFoundedBy:
		return []string{"created", "founded", "developed", "built", "invented"}
	case RelFoundedIn:
		return []string{"founded in", "established in", "created in", "born in"}
	case RelHas, RelOffers:
		return []string{"has", "features", "includes", "offers", "provides"}
	case RelUsedFor:
		return []string{"used for", "used in", "applied to", "enables"}
	case RelPartOf:
		return []string{"part of", "component of", "belongs to"}
	case RelRelatedTo, RelSimilarTo:
		return []string{"related to", "associated with", "connected to"}
	case RelLocatedIn:
		return []string{"located in", "based in", "found in"}
	default:
		return nil
	}
}

// factRole classifies a fact's relation into a discourse role.
func factRole(rel RelType) string {
	switch rel {
	case RelIsA, RelDescribedAs, RelKnownFor:
		return "definition"
	case RelCreatedBy, RelFoundedBy, RelFoundedIn, RelDerivedFrom, RelInfluencedBy:
		return "origin"
	case RelHas, RelOffers:
		return "property"
	case RelUsedFor:
		return "usage"
	case RelLocatedIn:
		return "location"
	default:
		return "detail"
	}
}

// deduplicateClauses removes near-duplicate clauses.
func deduplicateClauses(clauses []Clause) []Clause {
	var result []Clause
	for _, c := range clauses {
		dup := false
		for _, kept := range result {
			if clauseOverlap(c.Text, kept.Text) > 0.6 {
				dup = true
				break
			}
		}
		if !dup {
			result = append(result, c)
		}
	}
	return result
}

// clauseOverlap computes word-level Jaccard similarity between clauses.
func clauseOverlap(a, b string) float64 {
	wordsA := clauseContentWords(a)
	wordsB := clauseContentWords(b)
	if len(wordsA) == 0 || len(wordsB) == 0 {
		return 0
	}
	setA := make(map[string]bool, len(wordsA))
	for _, w := range wordsA {
		setA[w] = true
	}
	inter := 0
	for _, w := range wordsB {
		if setA[w] {
			inter++
		}
	}
	union := len(setA) + len(wordsB) - inter
	if union == 0 {
		return 0
	}
	return float64(inter) / float64(union)
}

// groupClausesByRole groups and orders clauses by discourse role.
func groupClausesByRole(clauses []Clause) map[string][]Clause {
	groups := make(map[string][]Clause)
	for _, c := range clauses {
		groups[c.Role] = append(groups[c.Role], c)
	}
	return groups
}

// buildProseFromGroups assembles grouped clauses into coherent prose.
func buildProseFromGroups(topic string, groups map[string][]Clause, facts []edgeFact) string {
	var paragraphs []string

	// Order: definition → origin → property → usage → location → detail
	roleOrder := []string{"definition", "origin", "property", "usage", "location", "detail"}

	for _, role := range roleOrder {
		clauses, ok := groups[role]
		if !ok || len(clauses) == 0 {
			continue
		}

		// Take best 2-3 clauses per role
		max := 3
		if len(clauses) < max {
			max = len(clauses)
		}
		selected := clauses[:max]

		// Build a paragraph from selected clauses
		para := assembleClauseParagraph(topic, role, selected)
		if para != "" {
			paragraphs = append(paragraphs, para)
		}
	}

	// If clause retrieval produced too little, supplement with fact fallbacks
	if len(paragraphs) == 0 {
		return fallbackFromFacts(topic, facts)
	}

	return strings.Join(paragraphs, "\n\n")
}

// assembleClauseParagraph joins clauses into a coherent paragraph for a role.
func assembleClauseParagraph(topic, role string, clauses []Clause) string {
	if len(clauses) == 0 {
		return ""
	}

	var parts []string
	for _, c := range clauses {
		parts = append(parts, c.Text)
	}

	// Join with appropriate connectors based on role
	switch role {
	case "definition":
		return strings.Join(parts, ". ") + "."
	case "origin":
		if len(parts) == 1 {
			return parts[0] + "."
		}
		return parts[0] + ". " + joinWithTransition("Furthermore", parts[1:])
	case "property":
		return "Key characteristics include: " + strings.Join(parts, "; ") + "."
	case "usage":
		return "In practice, " + strings.ToLower(parts[0][:1]) + parts[0][1:] +
			joinRemaining(parts[1:]) + "."
	default:
		return strings.Join(parts, ". ") + "."
	}
}

func joinWithTransition(transition string, parts []string) string {
	if len(parts) == 0 {
		return ""
	}
	result := transition + ", " + strings.ToLower(parts[0][:1]) + parts[0][1:]
	for _, p := range parts[1:] {
		result += ". " + p
	}
	return result + "."
}

func joinRemaining(parts []string) string {
	if len(parts) == 0 {
		return ""
	}
	var b strings.Builder
	for i, p := range parts {
		if i == len(parts)-1 && len(parts) > 1 {
			b.WriteString(", and ")
		} else {
			b.WriteString(", ")
		}
		b.WriteString(strings.ToLower(p[:1]) + p[1:])
	}
	return b.String()
}

// fallbackFromFacts generates prose directly from facts when no corpus clauses match.
func fallbackFromFacts(topic string, facts []edgeFact) string {
	cap := capitalizeFirst(topic)

	var parts []string
	for _, f := range facts {
		switch f.Relation {
		case RelIsA:
			parts = append(parts, fmt.Sprintf("%s is %s", cap, addArticle(f.Object)))
		case RelHas:
			parts = append(parts, fmt.Sprintf("%s features %s", cap, f.Object))
		case RelUsedFor:
			parts = append(parts, fmt.Sprintf("it is used for %s", f.Object))
		case RelCreatedBy:
			parts = append(parts, fmt.Sprintf("it was created by %s", f.Object))
		case RelFoundedIn:
			parts = append(parts, fmt.Sprintf("it was established in %s", f.Object))
		case RelLocatedIn:
			parts = append(parts, fmt.Sprintf("it is located in %s", f.Object))
		}
		if len(parts) >= 6 {
			break
		}
	}

	if len(parts) == 0 {
		return ""
	}

	// Fuse first two facts with a relative clause
	result := parts[0]
	if len(parts) > 1 {
		result += ", and " + parts[1]
	}
	result += "."

	if len(parts) > 2 {
		result += " " + strings.Join(parts[2:], ". ") + "."
	}

	return result
}

func addArticle(s string) string {
	if s == "" {
		return s
	}
	lower := strings.ToLower(s)
	if strings.HasPrefix(lower, "a ") || strings.HasPrefix(lower, "an ") || strings.HasPrefix(lower, "the ") {
		return s
	}
	vowels := "aeiou"
	if strings.ContainsRune(vowels, rune(lower[0])) {
		return "an " + s
	}
	return "a " + s
}

// fallbackRealize generates minimal prose from facts when no corpus is available.
func (pc *ProseComposer) fallbackRealize(topic string, facts []edgeFact) string {
	return fallbackFromFacts(topic, facts)
}

// clauseContentWords extracts lowercased content words from a clause.
func clauseContentWords(text string) []string {
	words := strings.Fields(strings.ToLower(text))
	var result []string
	for _, w := range words {
		w = strings.Trim(w, ".,;:!?\"'()[]{}")
		if len(w) > 2 && !proseStopWords[w] {
			result = append(result, w)
		}
	}
	return result
}

// splitProseSentences splits text on sentence boundaries.
func splitProseSentences(text string) []string {
	var sentences []string
	remaining := strings.TrimSpace(text)
	for len(remaining) > 0 {
		best := -1
		for _, p := range []string{". ", "! ", "? "} {
			idx := strings.Index(remaining, p)
			if idx >= 0 && (best < 0 || idx < best) {
				best = idx
			}
		}
		if best < 0 {
			s := strings.TrimSpace(remaining)
			if s != "" {
				sentences = append(sentences, s)
			}
			break
		}
		s := strings.TrimSpace(remaining[:best+1])
		if s != "" {
			sentences = append(sentences, s)
		}
		remaining = remaining[best+2:]
	}
	return sentences
}

// loadParagraphs caches all paragraphs from knowledge text files.
func (pc *ProseComposer) loadParagraphs() {
	pc.mu.Lock()
	defer pc.mu.Unlock()

	if pc.loaded || pc.knowledgeDir == "" {
		return
	}
	pc.loaded = true

	files, err := filepath.Glob(filepath.Join(pc.knowledgeDir, "*.txt"))
	if err != nil || len(files) == 0 {
		return
	}

	for _, f := range files {
		data, err := os.ReadFile(f)
		if err != nil {
			continue
		}
		for _, p := range strings.Split(string(data), "\n\n") {
			p = strings.TrimSpace(p)
			if len(p) > 50 {
				pc.paragraphs = append(pc.paragraphs, p)
			}
		}
	}
}

var proseStopWords = map[string]bool{
	"the": true, "and": true, "for": true, "are": true, "but": true,
	"not": true, "you": true, "all": true, "can": true, "had": true,
	"was": true, "one": true, "our": true, "has": true, "its": true,
	"that": true, "with": true, "have": true, "this": true, "will": true,
	"from": true, "they": true, "been": true, "said": true, "which": true,
	"their": true, "there": true, "about": true, "would": true, "these": true,
	"other": true, "into": true, "more": true, "some": true, "such": true,
	"than": true, "when": true, "what": true, "also": true, "were": true,
}
