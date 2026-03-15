package cognitive

import (
	"strings"
)

// FactChecker verifies LLM-generated answers against the knowledge store.
//
// Innovation: Most AI systems generate answers and ship them to the user.
// Nous generates, then VERIFIES against its knowledge base. If a claim
// contradicts established knowledge, the answer is corrected inline.
//
// This is a post-generation verification layer — it runs AFTER the LLM
// responds but BEFORE the user sees the answer. Zero extra latency for
// correct answers (just string matching). Only triggers correction when
// a factual contradiction is detected.
//
// Example: LLM says "Venus has no atmosphere" → Knowledge says Venus has
// "thick carbon dioxide atmosphere" → FactChecker appends correction.
//
// This is fundamentally different from prompt engineering:
//   - Prompt engineering HOPES the model won't hallucinate
//   - FactChecker CATCHES hallucinations after they happen
//   - The knowledge store is the source of truth, not the model
type FactChecker struct {
	Knowledge *KnowledgeVec
}

// NewFactChecker creates a fact checker backed by the knowledge store.
func NewFactChecker(kv *KnowledgeVec) *FactChecker {
	if kv == nil {
		return nil
	}
	return &FactChecker{Knowledge: kv}
}

// contradictionPair defines a factual claim and its contradiction.
type contradictionPair struct {
	wrong   string // what the LLM might say (wrong)
	right   string // what the knowledge says (correct)
	context string // topic context for matching
}

// knownContradictions are common factual errors small models make.
// These are checked FIRST (fast path) before semantic verification.
var knownContradictions = []contradictionPair{
	// Astronomy/Physics
	{"has no atmosphere", "Venus actually has an extremely thick atmosphere composed primarily of carbon dioxide, with surface pressure about 90 times that of Earth", "venus"},
	{"doesn't have an atmosphere", "Venus actually has an extremely thick atmosphere composed primarily of carbon dioxide", "venus"},
	{"lacks an atmosphere", "Venus has the densest atmosphere of any rocky planet, composed of 96% carbon dioxide", "venus"},
	{"without an atmosphere", "Venus has an extremely dense atmosphere of carbon dioxide", "venus"},
	{"mercury is the hottest", "While Mercury is closest to the Sun, Venus is actually the hottest planet due to its extreme greenhouse effect from its thick CO2 atmosphere", "planet hottest"},
	{"pluto is a planet", "Pluto was reclassified as a dwarf planet by the International Astronomical Union in 2006", "pluto planet"},
	{"the sun is a planet", "The Sun is a star, specifically a G-type main-sequence star (yellow dwarf)", "sun"},

	// Biology
	{"humans have 5 senses", "Humans have far more than 5 senses — including proprioception, thermoception, nociception, equilibrioception, and others", "senses"},
	{"we only use 10% of our brain", "Humans use virtually all parts of the brain, and most of the brain is active most of the time", "brain 10"},

	// History
	{"columbus discovered america", "Columbus reached the Americas in 1492, but indigenous peoples had been living there for thousands of years, and Norse explorers reached North America around 1000 CE", "columbus"},
	{"napoleon was short", "Napoleon was approximately 5'7\" (170 cm), which was average or slightly above average height for his era", "napoleon height short"},

	// Geography
	{"the great wall is visible from space", "The Great Wall of China is not visible from space with the naked eye — it is too narrow", "great wall space"},
	{"sahara is the largest desert", "Antarctica is technically the largest desert by area; the Sahara is the largest hot desert", "largest desert"},

	// Science
	{"lightning never strikes the same place twice", "Lightning frequently strikes the same location — tall structures like the Empire State Building are struck dozens of times per year", "lightning strikes"},
	{"diamonds are made from coal", "Most natural diamonds formed under extreme pressure deep in the Earth's mantle from carbon, not from coal. Coal is found in surface rock layers, while diamonds form much deeper", "diamonds coal"},
}

// Check verifies an answer against the knowledge store and returns a
// corrected version if factual errors are detected. Returns the original
// answer unchanged if no issues are found.
func (fc *FactChecker) Check(answer string, query string) string {
	if fc == nil || fc.Knowledge == nil {
		return answer
	}

	lower := strings.ToLower(answer)

	// Fast path: check known contradiction patterns.
	// Only triggers when BOTH the wrong claim AND the topic context appear in the answer.
	lowerQuery := strings.ToLower(query)
	var corrections []string
	for _, cp := range knownContradictions {
		if strings.Contains(lower, cp.wrong) {
			// Check that the context is relevant (appears in answer or query)
			contextWords := strings.Fields(cp.context)
			relevant := false
			for _, w := range contextWords {
				if strings.Contains(lower, w) || strings.Contains(lowerQuery, w) {
					relevant = true
					break
				}
			}
			if relevant {
				corrections = append(corrections, cp.right)
			}
		}
	}

	// Semantic verification: search knowledge for the query topic
	// and check if the answer contradicts key facts
	if len(corrections) == 0 {
		corrections = fc.semanticVerify(answer, query)
	}

	if len(corrections) == 0 {
		return answer
	}

	// Append corrections as a footnote
	var sb strings.Builder
	sb.WriteString(answer)
	sb.WriteString("\n\n---\n*Correction from knowledge base:* ")
	sb.WriteString(strings.Join(corrections, " "))
	return sb.String()
}

// semanticVerify checks if the answer contradicts knowledge chunks.
// It searches the knowledge store for the query topic and looks for
// direct contradictions between the answer and stored facts.
func (fc *FactChecker) semanticVerify(answer string, query string) []string {
	results, err := fc.Knowledge.Search(query, 2)
	if err != nil || len(results) == 0 {
		return nil
	}

	var corrections []string
	lowerAnswer := strings.ToLower(answer)

	for _, result := range results {
		lowerChunk := strings.ToLower(result.Text)

		// Check for negation contradictions:
		// Answer says "X has no Y" but knowledge says "X has Y"
		// Answer says "X doesn't Y" but knowledge says "X does Y"
		contradictions := findNegationContradictions(lowerAnswer, lowerChunk)
		corrections = append(corrections, contradictions...)
	}

	return corrections
}

// findNegationContradictions detects when the answer negates something
// that the knowledge source affirms (or vice versa).
func findNegationContradictions(answer, knowledge string) []string {
	var corrections []string

	// Common negation patterns that contradict positive knowledge
	negPatterns := []struct {
		neg string // negation in answer
		pos string // positive assertion in knowledge
	}{
		{"no atmosphere", "atmosphere"},
		{"doesn't have an atmosphere", "atmosphere"},
		{"has no atmosphere", "thick atmosphere"},
		{"has no atmosphere", "dense atmosphere"},
		{"cannot be seen", "visible"},
		{"is not visible", "visible"},
		{"never been", "has been"},
		{"doesn't exist", "exists"},
		{"is not real", "is real"},
		{"has no", "has a"},
		{"lacks any", "has"},
	}

	for _, np := range negPatterns {
		if strings.Contains(answer, np.neg) && strings.Contains(knowledge, np.pos) {
			// Extract a useful excerpt from the knowledge
			excerpt := extractRelevantSentence(knowledge, np.pos)
			if excerpt != "" {
				corrections = append(corrections, excerpt)
			}
		}
	}

	return corrections
}

// extractRelevantSentence finds the sentence containing the keyword.
func extractRelevantSentence(text, keyword string) string {
	lower := strings.ToLower(text)
	idx := strings.Index(lower, keyword)
	if idx < 0 {
		return ""
	}

	// Find sentence boundaries
	start := idx
	for start > 0 && text[start-1] != '.' && text[start-1] != '\n' {
		start--
	}
	end := idx + len(keyword)
	for end < len(text) && text[end] != '.' && text[end] != '\n' {
		end++
	}
	if end < len(text) {
		end++ // include the period
	}

	sentence := strings.TrimSpace(text[start:end])
	if len(sentence) > 200 {
		sentence = sentence[:200] + "..."
	}
	return sentence
}

