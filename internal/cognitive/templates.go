package cognitive

import (
	"encoding/json"
	"math/rand"
	"os"
	"strings"
	"sync"
	"time"
)

// -----------------------------------------------------------------------
// Template Induction — learns sentence patterns from observed text.
//
// Instead of relying on 10 hardcoded ClausePatterns, this system learns
// new patterns by parsing ingested text and extracting reusable templates.
// Content words (nouns, adjectives, verbs) are replaced with typed slots
// that the generative engine fills at generation time.
//
// Example:
//   Input:  "The ancient philosophy shaped modern thinking."
//   Template: "The {ADJ} {NOUN} {VERB:past} {ADJ} {NOUN}."
//   Reuse:  "The remarkable language influenced global development."
//
// POS tagging is heuristic-based (no external NLP library):
//  - Lexicon words have explicit POS tags
//  - Morphological rules: -ly → adverb, -tion → noun, -ive → adjective
//  - Function words (the, a, in, on, etc.) are kept literal
// -----------------------------------------------------------------------

// SlotType represents a part-of-speech slot in a template.
type SlotType string

const (
	SlotNoun SlotType = "NOUN"
	SlotVerb SlotType = "VERB"
	SlotAdj  SlotType = "ADJ"
	SlotAdv  SlotType = "ADV"
	SlotLit  SlotType = "LIT" // literal (function word, kept as-is)
)

// TemplateSlot is a single position in a learned template.
type TemplateSlot struct {
	Type     SlotType `json:"type"`
	Literal  string   `json:"literal,omitempty"`  // for SlotLit
	Original string   `json:"original,omitempty"` // the word that was abstracted
}

// InducedTemplate is a sentence pattern learned from text.
type InducedTemplate struct {
	Slots     []TemplateSlot `json:"slots"`
	Pattern   string         `json:"pattern"`    // human-readable: "The {ADJ} {NOUN} {VERB}."
	Source    string         `json:"source"`     // origin text
	SeenCount int           `json:"seen_count"`
	Quality   float64       `json:"quality"`    // 0-1, updated by usage
	LearnedAt time.Time     `json:"learned_at"`
}

// TemplateInducer learns and manages sentence templates.
type TemplateInducer struct {
	templates   []InducedTemplate
	posHints    map[string]SlotType // word → likely POS
	stopWords   map[string]bool
	mu          sync.RWMutex
}

// NewTemplateInducer creates a template learner, seeded with POS hints
// from the generative engine's lexicon and word pools.
func NewTemplateInducer() *TemplateInducer {
	ti := &TemplateInducer{
		posHints:  make(map[string]SlotType),
		stopWords: make(map[string]bool),
	}
	ti.seedPOSHints()
	ti.seedStopWords()
	return ti
}

// Size returns the number of learned templates.
func (ti *TemplateInducer) Size() int {
	ti.mu.RLock()
	defer ti.mu.RUnlock()
	return len(ti.templates)
}

// InduceFromText extracts templates from a block of text.
// Returns the number of new templates learned.
func (ti *TemplateInducer) InduceFromText(text, source string) int {
	sentences := splitIntoSentences(text)
	learned := 0
	for _, s := range sentences {
		if tmpl := ti.induceFromSentence(s, source); tmpl != nil {
			ti.mu.Lock()
			// Check for duplicates.
			dup := false
			for i := range ti.templates {
				if ti.templates[i].Pattern == tmpl.Pattern {
					ti.templates[i].SeenCount++
					dup = true
					break
				}
			}
			if !dup {
				ti.templates = append(ti.templates, *tmpl)
				learned++
			}
			ti.mu.Unlock()
		}
	}
	return learned
}

// induceFromSentence parses a single sentence into a template.
func (ti *TemplateInducer) induceFromSentence(sentence, source string) *InducedTemplate {
	words := strings.Fields(strings.ToLower(sentence))

	// Filter: only sentences of 5-20 words.
	if len(words) < 5 || len(words) > 20 {
		return nil
	}

	var slots []TemplateSlot
	hasContentSlot := false

	for _, word := range words {
		// Clean trailing punctuation for POS detection.
		clean := strings.TrimRight(word, ".,!?;:\"')")
		pos := ti.classifyWord(clean)

		if pos == SlotLit {
			slots = append(slots, TemplateSlot{Type: SlotLit, Literal: word})
		} else {
			slots = append(slots, TemplateSlot{Type: pos, Original: clean})
			hasContentSlot = true
		}
	}

	// Must have at least 2 content slots to be a useful template.
	contentCount := 0
	for _, s := range slots {
		if s.Type != SlotLit {
			contentCount++
		}
	}
	if !hasContentSlot || contentCount < 2 {
		return nil
	}

	// Build the pattern string.
	pattern := ti.slotsToPattern(slots)

	return &InducedTemplate{
		Slots:     slots,
		Pattern:   pattern,
		Source:    source,
		SeenCount: 1,
		Quality:   0.5, // default
		LearnedAt: time.Now(),
	}
}

// slotsToPattern builds a human-readable pattern string.
func (ti *TemplateInducer) slotsToPattern(slots []TemplateSlot) string {
	var parts []string
	for _, s := range slots {
		if s.Type == SlotLit {
			parts = append(parts, s.Literal)
		} else {
			parts = append(parts, "{"+string(s.Type)+"}")
		}
	}
	return strings.Join(parts, " ")
}

// classifyWord determines the POS of a word using heuristics.
func (ti *TemplateInducer) classifyWord(word string) SlotType {
	// Check stop words and function words first.
	if ti.stopWords[word] {
		return SlotLit
	}

	// Check explicit POS hints.
	if pos, ok := ti.posHints[word]; ok {
		return pos
	}

	// Morphological heuristics.
	if strings.HasSuffix(word, "ly") && len(word) > 4 {
		return SlotAdv
	}
	if strings.HasSuffix(word, "tion") || strings.HasSuffix(word, "ness") ||
		strings.HasSuffix(word, "ment") || strings.HasSuffix(word, "ity") ||
		strings.HasSuffix(word, "ance") || strings.HasSuffix(word, "ence") {
		return SlotNoun
	}
	if strings.HasSuffix(word, "ive") || strings.HasSuffix(word, "ous") ||
		strings.HasSuffix(word, "ful") || strings.HasSuffix(word, "less") ||
		strings.HasSuffix(word, "able") || strings.HasSuffix(word, "ible") ||
		strings.HasSuffix(word, "ical") || strings.HasSuffix(word, "al") {
		return SlotAdj
	}
	if strings.HasSuffix(word, "ing") || strings.HasSuffix(word, "ed") ||
		strings.HasSuffix(word, "ize") || strings.HasSuffix(word, "ise") ||
		strings.HasSuffix(word, "ate") {
		return SlotVerb
	}

	// Default: treat as noun (most content words are nouns).
	return SlotNoun
}

// Realize fills a template's slots with provided words and returns a sentence.
func (ti *TemplateInducer) Realize(tmpl *InducedTemplate, fills map[SlotType][]string, rng *rand.Rand) string {
	// Track how many of each slot type we've used.
	slotIdx := make(map[SlotType]int)

	var parts []string
	for _, slot := range tmpl.Slots {
		if slot.Type == SlotLit {
			parts = append(parts, slot.Literal)
			continue
		}

		candidates := fills[slot.Type]
		if len(candidates) == 0 {
			// Fall back to the original word.
			parts = append(parts, slot.Original)
			continue
		}

		idx := slotIdx[slot.Type]
		if idx < len(candidates) {
			parts = append(parts, candidates[idx])
		} else {
			// Wrap around or pick random.
			parts = append(parts, candidates[rng.Intn(len(candidates))])
		}
		slotIdx[slot.Type]++
	}

	result := strings.Join(parts, " ")
	if len(result) > 0 {
		result = strings.ToUpper(result[:1]) + result[1:]
	}
	return result
}

// BestTemplates returns the n highest-quality templates.
func (ti *TemplateInducer) BestTemplates(n int) []InducedTemplate {
	ti.mu.RLock()
	defer ti.mu.RUnlock()

	if n > len(ti.templates) {
		n = len(ti.templates)
	}

	// Sort by quality * seen_count (popularity-weighted quality).
	sorted := make([]InducedTemplate, len(ti.templates))
	copy(sorted, ti.templates)

	for i := 0; i < len(sorted); i++ {
		for j := i + 1; j < len(sorted); j++ {
			scoreI := sorted[i].Quality * float64(sorted[i].SeenCount)
			scoreJ := sorted[j].Quality * float64(sorted[j].SeenCount)
			if scoreJ > scoreI {
				sorted[i], sorted[j] = sorted[j], sorted[i]
			}
		}
	}

	return sorted[:n]
}

// TemplatesWithSlots returns templates that have at least the given slot types.
func (ti *TemplateInducer) TemplatesWithSlots(required ...SlotType) []InducedTemplate {
	ti.mu.RLock()
	defer ti.mu.RUnlock()

	var result []InducedTemplate
	for _, tmpl := range ti.templates {
		if templateHasSlots(tmpl, required) {
			result = append(result, tmpl)
		}
	}
	return result
}

func templateHasSlots(tmpl InducedTemplate, required []SlotType) bool {
	slotTypes := make(map[SlotType]bool)
	for _, s := range tmpl.Slots {
		slotTypes[s.Type] = true
	}
	for _, r := range required {
		if !slotTypes[r] {
			return false
		}
	}
	return true
}

// MarkUsed updates the quality score of a template after it's been used.
// positive=true means the output was good; false means it was rejected.
func (ti *TemplateInducer) MarkUsed(pattern string, positive bool) {
	ti.mu.Lock()
	defer ti.mu.Unlock()

	for i := range ti.templates {
		if ti.templates[i].Pattern == pattern {
			if positive {
				ti.templates[i].Quality = ti.templates[i].Quality*0.9 + 0.1
			} else {
				ti.templates[i].Quality = ti.templates[i].Quality * 0.8
			}
			break
		}
	}
}

// -----------------------------------------------------------------------
// Persistence
// -----------------------------------------------------------------------

// Save persists the templates to a JSON file.
func (ti *TemplateInducer) Save(path string) error {
	ti.mu.RLock()
	defer ti.mu.RUnlock()

	data, err := json.MarshalIndent(ti.templates, "", "  ")
	if err != nil {
		return err
	}

	tmpPath := path + ".tmp"
	if err := os.WriteFile(tmpPath, data, 0644); err != nil {
		return err
	}
	return os.Rename(tmpPath, path)
}

// Load reads templates from a JSON file.
func (ti *TemplateInducer) Load(path string) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return err
	}

	ti.mu.Lock()
	defer ti.mu.Unlock()

	return json.Unmarshal(data, &ti.templates)
}

// -----------------------------------------------------------------------
// POS hints and stop words
// -----------------------------------------------------------------------

func (ti *TemplateInducer) seedPOSHints() {
	// Adjectives from the generative engine pools.
	adjs := []string{
		"remarkable", "notable", "significant", "profound", "compelling",
		"innovative", "ancient", "modern", "practical", "theoretical",
		"fundamental", "essential", "critical", "crucial", "vital",
		"elegant", "robust", "versatile", "powerful", "enduring",
		"distinctive", "creative", "artistic", "scientific", "natural",
		"resilient", "passionate", "determined", "comprehensive",
		"transformative", "revolutionary", "groundbreaking", "pioneering",
	}
	for _, w := range adjs {
		ti.posHints[w] = SlotAdj
	}

	// Nouns
	nouns := []string{
		"story", "path", "direction", "identity", "character",
		"profile", "nature", "role", "position", "place",
		"impact", "influence", "effect", "contribution",
		"strength", "elegance", "beauty", "complexity",
		"philosophy", "science", "language", "technology",
		"history", "culture", "tradition", "legacy",
		"foundation", "framework", "structure", "system",
		"world", "field", "domain", "landscape",
	}
	for _, w := range nouns {
		ti.posHints[w] = SlotNoun
	}

	// Verbs
	verbs := []string{
		"shape", "define", "create", "build", "establish",
		"influence", "transform", "reveal", "discover", "explore",
		"emerge", "evolve", "develop", "grow", "expand",
		"achieve", "deliver", "enable", "provide", "offer",
	}
	for _, w := range verbs {
		ti.posHints[w] = SlotVerb
	}
}

func (ti *TemplateInducer) seedStopWords() {
	stops := []string{
		// Determiners
		"the", "a", "an", "this", "that", "these", "those",
		"my", "your", "his", "her", "its", "our", "their",
		// Prepositions
		"in", "on", "at", "by", "for", "with", "from", "to",
		"of", "about", "into", "through", "between", "among",
		"over", "under", "above", "below", "after", "before",
		// Conjunctions
		"and", "but", "or", "nor", "yet", "so", "for",
		"because", "although", "while", "if", "when", "where",
		// Pronouns
		"i", "you", "he", "she", "it", "we", "they",
		"me", "him", "her", "us", "them",
		"who", "which", "what", "whom", "whose",
		// Auxiliaries
		"is", "are", "was", "were", "be", "been", "being",
		"has", "have", "had", "do", "does", "did",
		"will", "would", "shall", "should",
		"can", "could", "may", "might", "must",
		// Common function words
		"not", "no", "as", "than", "more", "most", "very",
		"just", "also", "too", "then", "there", "here",
		"how", "why", "all", "each", "every", "both",
		"few", "many", "much", "some", "any", "other",
	}
	for _, w := range stops {
		ti.stopWords[w] = true
	}
}
