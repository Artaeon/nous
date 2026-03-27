package cognitive

import (
	"encoding/json"
	"math/rand"
	"os"
	"strings"
	"sync"
	"unicode"
)

// -----------------------------------------------------------------------
// Absorption Engine — learns HOW things are expressed, not just WHAT.
//
// When Nous reads any text (Wikipedia, conversations, articles), the
// absorption engine extracts reusable expression patterns: rhetorical
// structure, tone, and slot-typed templates. Unlike handwritten
// templates, these are LEARNED from reading real human writing.
//
// Example source: "Democracy, while imperfect, remains the most
//                  resilient form of governance."
// Extracted:
//   Template:  "[SUBJECT], while [CONCESSION], remains the most [MODIFIER] [CATEGORY]"
//   Function:  DFEvaluates
//   Tone:      "formal"
//   Structure: "concessive_evaluation"
//
// Over time, the engine's expressive repertoire grows organically.
// Each sentence it reads teaches it a new way to say things.
// -----------------------------------------------------------------------

// AbsorbedPattern is a reusable expression pattern extracted from real text.
type AbsorbedPattern struct {
	Template   string        `json:"template"`
	Function   DiscourseFunc `json:"function"`
	Tone       string        `json:"tone"`
	Structure  string        `json:"structure"`
	Source     string        `json:"source"`
	Quality    int           `json:"quality"`
	UsageCount int           `json:"usage_count"`
	SlotTypes  []AbsorbedSlot    `json:"slot_types"`
}

// AbsorbedSlot describes what kind of content fills a template slot.
type AbsorbedSlot struct {
	Name     string `json:"name"`
	Position int    `json:"position"`
	Kind     string `json:"kind"`
}

// AbsorptionEngine reads text and extracts reusable expression patterns.
type AbsorptionEngine struct {
	mu          sync.RWMutex
	patterns    []AbsorbedPattern
	byFunction  map[DiscourseFunc][]int
	byTone      map[string][]int
	byStructure map[string][]int
	savePath    string
	rng         *rand.Rand
}

// NewAbsorptionEngine creates a new engine that persists to the given path.
func NewAbsorptionEngine(savePath string) *AbsorptionEngine {
	return &AbsorptionEngine{
		patterns:    nil,
		byFunction:  make(map[DiscourseFunc][]int),
		byTone:      make(map[string][]int),
		byStructure: make(map[string][]int),
		savePath:    savePath,
		rng:         rand.New(rand.NewSource(42)),
	}
}

// -----------------------------------------------------------------------
// Absorption — the main entry points.
// -----------------------------------------------------------------------

// Absorb processes a paragraph or article, extracting patterns from
// each sentence and storing the good ones.
func (ae *AbsorptionEngine) Absorb(text string) int {
	sentences := splitAbsorptionSentences(text)
	absorbed := 0
	for _, sent := range sentences {
		p := ae.AbsorbSentence(sent)
		if p != nil {
			absorbed++
		}
	}
	return absorbed
}

// AbsorbSentence processes a single sentence and extracts a reusable
// pattern if the sentence is high enough quality.
func (ae *AbsorptionEngine) AbsorbSentence(sent string) *AbsorbedPattern {
	sent = strings.TrimSpace(sent)
	if sent == "" {
		return nil
	}

	// Quality gate: reject fragments, questions, very short/long.
	q := scoreAbsorptionQuality(sent)
	if q < 1 {
		return nil
	}

	fn := ae.classifyFunction(sent)
	tone := ae.classifyTone(sent)
	structure := ae.classifyStructure(sent)
	tmpl, slots := ae.extractTemplate(sent, fn)

	// Reject templates that are too close to the original (no slots
	// extracted) or too abstract (more slots than words).
	if tmpl == sent || len(slots) == 0 {
		return nil
	}
	words := strings.Fields(tmpl)
	if len(slots) > len(words)/2 {
		return nil
	}

	p := AbsorbedPattern{
		Template:  tmpl,
		Function:  fn,
		Tone:      tone,
		Structure: structure,
		Source:    sent,
		Quality:   q,
		SlotTypes: slots,
	}

	ae.addPattern(p)
	return &p
}

// addPattern stores a pattern and updates all indices.
func (ae *AbsorptionEngine) addPattern(p AbsorbedPattern) {
	ae.mu.Lock()
	defer ae.mu.Unlock()
	idx := len(ae.patterns)
	ae.patterns = append(ae.patterns, p)
	ae.byFunction[p.Function] = append(ae.byFunction[p.Function], idx)
	ae.byTone[p.Tone] = append(ae.byTone[p.Tone], idx)
	ae.byStructure[p.Structure] = append(ae.byStructure[p.Structure], idx)
}

// -----------------------------------------------------------------------
// Classification — discourse function, tone, structure.
// -----------------------------------------------------------------------

// classifyFunction detects the discourse function of a sentence using
// signal words and phrases.
func (ae *AbsorptionEngine) classifyFunction(sent string) DiscourseFunc {
	lower := strings.ToLower(sent)

	// Order matters: check more specific patterns first.

	if absContainsAny(lower, absorbCausalSignals) {
		return DFExplainsWhy
	}
	if absContainsAny(lower, absorbExampleSignals) {
		return DFGivesExample
	}
	if absContainsAny(lower, absorbCompareSignals) {
		return DFCompares
	}
	if absContainsAny(lower, absorbEvalSignals) {
		return DFEvaluates
	}
	if absContainsAny(lower, absorbConsequenceSignals) {
		return DFConsequence
	}
	if absContainsAny(lower, absorbProcessSignals) {
		return DFDescribes
	}
	if absContainsAny(lower, absorbContextSignals) {
		return DFContext
	}
	if absContainsAny(lower, absorbQuantifySignals) || absContainsDigit(lower) {
		return DFQuantifies
	}
	if absContainsAny(lower, absorbDefineSignals) {
		return DFDefines
	}

	// Default: if we can't classify it, it's a description.
	return DFDescribes
}

// Signal word lists for function classification.
var (
	absorbDefineSignals = []string{
		" is a ", " is an ", " is the ", " are a ", " are an ",
		" was a ", " was an ", " refers to ", " is defined as ",
		" means ", " is called ",
	}
	absorbCausalSignals = []string{
		" because ", " due to ", " as a result of ", " since ",
		" caused by ", " owing to ", " thanks to ",
		" the reason ", " this is because ",
	}
	absorbExampleSignals = []string{
		"for example", " such as ", "for instance",
		" one example ", " examples include ",
	}
	absorbCompareSignals = []string{
		"unlike ", "compared to ", "whereas ", " in contrast",
		" on the other hand", " different from ",
		" similar to ", " differ ",
	}
	absorbEvalSignals = []string{
		" is considered ", " is known for ", " is regarded as ",
		" one of the most ", " is widely ", " is often ",
		" is commonly ", " is believed to ", " is thought to ",
		" best known ", " well known ", " widely known ",
		" most important ", " most significant ",
	}
	absorbConsequenceSignals = []string{
		" leads to ", " results in ", " therefore ",
		" consequently ", " as a result ", " this led to ",
		" this caused ", " which meant ", " which means ",
	}
	absorbProcessSignals = []string{
		" works by ", " involves ", " consists of ",
		" is made of ", " is made by ", " is produced ",
		" is formed ", " the process ", " the method ",
	}
	absorbContextSignals = []string{
		" was founded in ", " originated in ", " was established ",
		" dates back to ", " was created in ", " was built in ",
		" first appeared in ", " was introduced in ",
		" began in ", " started in ",
	}
	absorbQuantifySignals = []string{
		" million ", " billion ", " thousand ", " percent ",
		" approximately ", " roughly ", " about ",
		" km ", " miles ", " metres ", " meters ",
	}
)

// classifyTone detects the register/tone of a sentence.
func (ae *AbsorptionEngine) classifyTone(sent string) string {
	lower := strings.ToLower(sent)
	words := strings.Fields(sent)
	wordCount := len(words)
	if wordCount == 0 {
		return "casual"
	}

	// Conversational: direct address, questions.
	if strings.Contains(lower, " you ") || strings.HasPrefix(lower, "you ") {
		return "conversational"
	}
	if strings.HasSuffix(sent, "?") {
		return "conversational"
	}

	// Casual: contractions, short sentences, informal words.
	if strings.Contains(sent, "'t") || strings.Contains(sent, "'s") ||
		strings.Contains(sent, "'re") || strings.Contains(sent, "'ve") ||
		strings.Contains(sent, "'ll") || strings.Contains(sent, "'d") {
		if wordCount < 15 {
			return "casual"
		}
	}

	// Poetic: unusual patterns, imagery words.
	poeticMarkers := []string{
		" like a ", " as if ", " whisper", " shadow", " dance",
		" dream", " echo", " silence", " beneath ", " amidst ",
	}
	poeticCount := 0
	for _, m := range poeticMarkers {
		if strings.Contains(lower, m) {
			poeticCount++
		}
	}
	if poeticCount >= 2 {
		return "poetic"
	}

	// Academic: hedging, technical markers, longer sentences.
	academicMarkers := []string{
		" generally ", " typically ", " furthermore ",
		" moreover ", " nevertheless ", " however ",
		" according to ", " hypothesis ", " theoretical ",
		" empirical ", " significant", " thus ",
		" hence ", " notably ", " respectively ",
		" suggesting ", " correlation", " observed ",
		" controlled ", " experiments ",
	}
	academicCount := 0
	for _, m := range academicMarkers {
		if strings.Contains(lower, m) {
			academicCount++
		}
	}
	if academicCount >= 2 || (academicCount >= 1 && wordCount > 20) {
		return "academic"
	}

	// Formal: passive voice, longer words, no contractions.
	passiveMarkers := []string{
		" was ", " were ", " is being ", " has been ",
		" had been ", " was being ", " will be ",
	}
	hasPassive := false
	for _, m := range passiveMarkers {
		if strings.Contains(lower, m) {
			hasPassive = true
			break
		}
	}

	avgWordLen := 0
	for _, w := range words {
		avgWordLen += len(w)
	}
	avgWordLen /= wordCount

	if hasPassive && avgWordLen >= 5 {
		return "formal"
	}
	if wordCount > 15 && avgWordLen >= 5 {
		return "formal"
	}

	// Short, simple sentences default to casual.
	if wordCount <= 8 {
		return "casual"
	}

	return "formal"
}

// classifyStructure detects the syntactic pattern of a sentence.
func (ae *AbsorptionEngine) classifyStructure(sent string) string {
	lower := strings.ToLower(sent)

	// Concessive: "X, while Y, remains Z" or "although X, Y"
	if (strings.Contains(lower, ", while ") || strings.Contains(lower, "although ") ||
		strings.Contains(lower, ", despite ") || strings.Contains(lower, "even though ")) &&
		(strings.Contains(lower, ", remains ") || strings.Contains(lower, ", is ") ||
			strings.Contains(lower, ", it ")) {
		return "concessive_evaluation"
	}

	// Conditional: "if X, then Y" or "if X, Y"
	if strings.HasPrefix(lower, "if ") || strings.Contains(lower, ", if ") {
		return "conditional"
	}

	// Contrastive: "unlike X, Y" or "while X, Y does Z"
	if strings.HasPrefix(lower, "unlike ") || strings.HasPrefix(lower, "in contrast") ||
		strings.HasPrefix(lower, "whereas ") {
		return "contrastive"
	}

	// Temporal: "after X, Y" or "before X, Y" or "in YEAR"
	if strings.HasPrefix(lower, "after ") || strings.HasPrefix(lower, "before ") ||
		strings.HasPrefix(lower, "during ") || strings.HasPrefix(lower, "following ") {
		return "temporal"
	}

	// Causal chain: "X because Y" or "since X, Y"
	if strings.Contains(lower, " because ") || strings.Contains(lower, " causing ") ||
		(strings.HasPrefix(lower, "since ") && strings.Contains(lower, ", ")) {
		return "causal_chain"
	}

	// Listing: "X includes A, B, and C" or "such as A, B, and C"
	commaCount := strings.Count(sent, ",")
	if commaCount >= 2 && (strings.Contains(lower, " and ") || strings.Contains(lower, " or ")) &&
		(strings.Contains(lower, " include") || strings.Contains(lower, " such as") ||
			strings.Contains(lower, " including ")) {
		return "listing"
	}

	// Relative clause: "X, which is Y, does Z"
	if strings.Contains(lower, ", which ") || strings.Contains(lower, ", who ") ||
		strings.Contains(lower, ", where ") {
		return "relative_clause"
	}

	// Simple definition: "X is a Y"
	if absContainsAny(lower, []string{" is a ", " is an ", " was a ", " was an ", " are a "}) {
		return "simple_definition"
	}

	return "simple_statement"
}

// -----------------------------------------------------------------------
// Template Extraction — the hard part.
//
// Replace content words with typed slots while keeping function words
// and structural skeleton intact. The goal is a reusable template that
// can be filled with new content to produce novel sentences.
// -----------------------------------------------------------------------

// extractTemplate replaces content words with typed slots.
func (ae *AbsorptionEngine) extractTemplate(sent string, fn DiscourseFunc) (string, []AbsorbedSlot) {
	words := strings.Fields(sent)
	if len(words) < 4 {
		return sent, nil
	}

	var slots []AbsorbedSlot
	result := make([]string, len(words))
	copy(result, words)

	slotIndex := 0

	// Phase 1: identify proper noun spans (capitalized words not at
	// sentence start and not common title words).
	spans := findProperNounSpans(words)

	// Assign the first proper noun span as SUBJECT, subsequent as OBJECT.
	for i, span := range spans {
		slotName := "OBJECT"
		slotKind := "noun_phrase"
		if i == 0 {
			slotName = "SUBJECT"
		}

		// Replace the span in results.
		marker := "[" + slotName + "]"
		pos := positionInTemplate(result, span.start)
		slots = append(slots, AbsorbedSlot{
			Name:     slotName,
			Position: pos,
			Kind:     slotKind,
		})
		slotIndex++

		// Blank out span words, put marker at start.
		result[span.start] = marker
		for j := span.start + 1; j <= span.end && j < len(result); j++ {
			result[j] = ""
		}
	}

	// Phase 2: for definition patterns, extract CATEGORY after "is a/an".
	if fn == DFDefines || fn == DFDescribes {
		result, slots = extractCategorySlot(result, slots)
	}

	// Phase 3: extract MODIFIER (adjective before a category or noun).
	result, slots = extractModifierSlot(result, slots)

	// Phase 4: extract VERB_PHRASE after "who/that/which" + verb.
	result, slots = extractVerbPhraseSlot(result, slots)

	// Compact: remove empty strings from blanked-out spans.
	var compacted []string
	for _, w := range result {
		if w != "" {
			compacted = append(compacted, w)
		}
	}

	tmpl := strings.Join(compacted, " ")
	return tmpl, slots
}

// properNounSpan marks start..end indices of a contiguous proper noun.
type properNounSpan struct {
	start, end int
	text       string
}

// findProperNounSpans finds contiguous runs of capitalized words,
// excluding sentence-initial position and common words.
func findProperNounSpans(words []string) []properNounSpan {
	var spans []properNounSpan
	i := 1 // skip sentence-initial word
	for i < len(words) {
		if isProperNounWord(words[i]) {
			start := i
			for i < len(words) && isProperNounWord(words[i]) {
				i++
			}
			end := i - 1
			var parts []string
			for j := start; j <= end; j++ {
				parts = append(parts, words[j])
			}
			spans = append(spans, properNounSpan{
				start: start,
				end:   end,
				text:  strings.Join(parts, " "),
			})
			if len(spans) >= 3 {
				break
			}
		} else {
			i++
		}
	}
	return spans
}

// isProperNounWord returns true if a word looks like a proper noun:
// starts with uppercase, isn't a common sentence-initial word, and
// isn't a punctuation token.
func isProperNounWord(w string) bool {
	if len(w) == 0 {
		return false
	}
	// Must start uppercase.
	if !unicode.IsUpper(rune(w[0])) {
		return false
	}
	// Reject very short words that might be abbreviations or Roman numerals.
	if len(w) == 1 && w != "I" {
		return false
	}
	// Reject common words that can start sentences.
	lower := strings.ToLower(w)
	for _, skip := range commonCapitalized {
		if lower == skip {
			return false
		}
	}
	return true
}

var commonCapitalized = []string{
	"the", "a", "an", "this", "that", "these", "those",
	"it", "its", "he", "she", "they", "we", "his", "her",
	"their", "our", "my", "your", "i", "in", "on", "at",
	"by", "to", "for", "of", "with", "from", "as", "but",
	"or", "and", "not", "no", "if", "so", "yet", "also",
	"however", "although", "while", "after", "before",
	"during", "since", "until", "because", "when", "where",
	"there", "here", "many", "most", "some", "each", "every",
	"both", "all", "such", "other", "new", "old", "first",
	"last", "one", "two", "three", "four", "five",
}

// extractCategorySlot finds the noun after "is a/an/the" and replaces
// it with [CATEGORY].
func extractCategorySlot(words []string, slots []AbsorbedSlot) ([]string, []AbsorbedSlot) {
	for i := 0; i < len(words)-2; i++ {
		lower := strings.ToLower(words[i])
		next := strings.ToLower(words[i+1])
		if (lower == "is" || lower == "was" || lower == "are" || lower == "were") &&
			(next == "a" || next == "an" || next == "the") {
			// The word after the article is the category candidate.
			catIdx := i + 2
			if catIdx < len(words) && words[catIdx] != "" &&
				!strings.HasPrefix(words[catIdx], "[") {
				pos := positionInTemplate(words, catIdx)
				slots = append(slots, AbsorbedSlot{
					Name:     "CATEGORY",
					Position: pos,
					Kind:     "noun",
				})
				words[catIdx] = "[CATEGORY]"
				return words, slots
			}
		}
	}
	return words, slots
}

// extractModifierSlot finds adjectives immediately before [CATEGORY]
// or other noun slots and replaces them with [MODIFIER].
func extractModifierSlot(words []string, slots []AbsorbedSlot) ([]string, []AbsorbedSlot) {
	for i := 1; i < len(words); i++ {
		if words[i] == "[CATEGORY]" || words[i] == "[SUBJECT]" || words[i] == "[OBJECT]" {
			prev := i - 1
			if prev >= 0 && words[prev] != "" && !strings.HasPrefix(words[prev], "[") {
				lower := strings.ToLower(words[prev])
				// Skip articles and determiners.
				if lower == "a" || lower == "an" || lower == "the" ||
					lower == "this" || lower == "that" {
					continue
				}
				// Only replace if it looks like an adjective (lowercase,
				// not a preposition/conjunction).
				if unicode.IsLower(rune(words[prev][0])) && !absIsFunctionWord(lower) {
					pos := positionInTemplate(words, prev)
					slots = append(slots, AbsorbedSlot{
						Name:     "MODIFIER",
						Position: pos,
						Kind:     "adjective",
					})
					words[prev] = "[MODIFIER]"
					return words, slots
				}
			}
		}
	}
	return words, slots
}

// extractVerbPhraseSlot finds verb phrases after relative pronouns
// (who, that, which) and replaces them with [VERB_PHRASE].
func extractVerbPhraseSlot(words []string, slots []AbsorbedSlot) ([]string, []AbsorbedSlot) {
	for i := 0; i < len(words)-1; i++ {
		lower := strings.ToLower(words[i])
		if lower == "who" || lower == "that" || lower == "which" {
			// Take the next 1-3 words as the verb phrase.
			vpStart := i + 1
			vpEnd := vpStart
			for vpEnd < len(words) && vpEnd < vpStart+3 {
				w := words[vpEnd]
				if w == "" || strings.HasPrefix(w, "[") || w == "," || w == "." {
					break
				}
				vpEnd++
			}
			if vpEnd > vpStart {
				pos := positionInTemplate(words, vpStart)
				slots = append(slots, AbsorbedSlot{
					Name:     "VERB_PHRASE",
					Position: pos,
					Kind:     "verb_phrase",
				})
				words[vpStart] = "[VERB_PHRASE]"
				for j := vpStart + 1; j < vpEnd; j++ {
					words[j] = ""
				}
				return words, slots
			}
		}
	}
	return words, slots
}

// positionInTemplate estimates the character position of a word index
// within the reconstructed template string.
func positionInTemplate(words []string, idx int) int {
	pos := 0
	for i := 0; i < idx && i < len(words); i++ {
		pos += len(words[i]) + 1
	}
	return pos
}

// isFunctionWord returns true for prepositions, conjunctions, determiners,
// and other structural words that should NOT be slotted.
func absIsFunctionWord(w string) bool {
	for _, fw := range functionWords {
		if w == fw {
			return true
		}
	}
	return false
}

var functionWords = []string{
	"a", "an", "the", "this", "that", "these", "those",
	"is", "are", "was", "were", "be", "been", "being",
	"has", "have", "had", "do", "does", "did",
	"will", "would", "shall", "should", "can", "could",
	"may", "might", "must",
	"in", "on", "at", "to", "for", "of", "with", "from",
	"by", "about", "into", "through", "during", "before",
	"after", "above", "below", "between", "under", "over",
	"and", "or", "but", "nor", "so", "yet", "both",
	"not", "no", "as", "if", "than", "then", "when",
	"where", "while", "although", "because", "since",
	"until", "unless", "whether", "who", "which", "that",
	"what", "whom", "whose",
	"very", "also", "just", "only", "even", "still",
	"already", "often", "always", "never", "sometimes",
}

// -----------------------------------------------------------------------
// Quality scoring for absorption.
// -----------------------------------------------------------------------

// scoreAbsorptionQuality rates a sentence's suitability for template
// extraction. Returns 0-3 (0 = reject, 1-3 = increasing quality).
func scoreAbsorptionQuality(sent string) int {
	// Reject empty or whitespace-only.
	trimmed := strings.TrimSpace(sent)
	if trimmed == "" {
		return 0
	}

	words := strings.Fields(trimmed)
	wordCount := len(words)

	// Too short: likely fragment.
	if wordCount < 5 {
		return 0
	}
	// Too long: unwieldy template.
	if wordCount > 40 {
		return 0
	}

	// Questions are not good templates for declarative generation.
	if strings.HasSuffix(trimmed, "?") {
		return 0
	}

	// Must start with uppercase (proper sentence).
	if len(trimmed) > 0 && !unicode.IsUpper(rune(trimmed[0])) {
		return 0
	}

	// Reject wiki markup remnants.
	if strings.Contains(trimmed, "[[") || strings.Contains(trimmed, "]]") ||
		strings.Contains(trimmed, "{{") || strings.Contains(trimmed, "}}") {
		return 0
	}

	// Reject list markers and bullet points.
	if strings.HasPrefix(trimmed, "* ") || strings.HasPrefix(trimmed, "- ") ||
		strings.HasPrefix(trimmed, "# ") {
		return 0
	}

	// Score: start at 1, add points for quality signals.
	q := 1

	// Ideal length range.
	if wordCount >= 8 && wordCount <= 25 {
		q++
	}

	// Proper sentence ending.
	if strings.HasSuffix(trimmed, ".") || strings.HasSuffix(trimmed, "!") {
		q++
	}

	return q
}

// -----------------------------------------------------------------------
// Sentence splitting with abbreviation handling.
// -----------------------------------------------------------------------

// splitAbsorptionSentences splits text into sentences, handling common
// abbreviations like "Dr.", "U.S.", "e.g.", "i.e.", "etc.".
func splitAbsorptionSentences(text string) []string {
	// Protect known abbreviations by temporarily replacing their periods.
	protected := text
	for _, abbr := range knownAbbreviations {
		protected = strings.ReplaceAll(protected, abbr, strings.ReplaceAll(abbr, ".", "\x00"))
	}

	// Split on sentence-ending punctuation followed by space + uppercase
	// or end of string.
	var sentences []string
	current := strings.Builder{}
	runes := []rune(protected)

	for i := 0; i < len(runes); i++ {
		current.WriteRune(runes[i])

		if runes[i] == '.' || runes[i] == '!' || runes[i] == '?' {
			// Check if this looks like end-of-sentence:
			// at end of text, or followed by space + uppercase.
			if i+1 >= len(runes) {
				s := strings.ReplaceAll(current.String(), "\x00", ".")
				s = strings.TrimSpace(s)
				if s != "" {
					sentences = append(sentences, s)
				}
				current.Reset()
				continue
			}
			if i+2 < len(runes) && runes[i+1] == ' ' && unicode.IsUpper(runes[i+2]) {
				s := strings.ReplaceAll(current.String(), "\x00", ".")
				s = strings.TrimSpace(s)
				if s != "" {
					sentences = append(sentences, s)
				}
				current.Reset()
				continue
			}
			// Also split on newlines after sentence-enders.
			if i+1 < len(runes) && runes[i+1] == '\n' {
				s := strings.ReplaceAll(current.String(), "\x00", ".")
				s = strings.TrimSpace(s)
				if s != "" {
					sentences = append(sentences, s)
				}
				current.Reset()
				continue
			}
		}

		// Split on paragraph boundaries.
		if runes[i] == '\n' && i+1 < len(runes) && runes[i+1] == '\n' {
			s := strings.ReplaceAll(current.String(), "\x00", ".")
			s = strings.TrimSpace(s)
			if s != "" {
				sentences = append(sentences, s)
			}
			current.Reset()
		}
	}

	// Flush remainder.
	if current.Len() > 0 {
		s := strings.ReplaceAll(current.String(), "\x00", ".")
		s = strings.TrimSpace(s)
		if s != "" {
			sentences = append(sentences, s)
		}
	}

	return sentences
}

var knownAbbreviations = []string{
	"Dr.", "Mr.", "Mrs.", "Ms.", "Prof.", "Jr.", "Sr.",
	"St.", "Ave.", "Blvd.", "Dept.", "Gov.", "Gen.", "Sgt.",
	"U.S.", "U.K.", "U.N.", "E.U.",
	"e.g.", "i.e.", "etc.", "vs.", "viz.",
	"Jan.", "Feb.", "Mar.", "Apr.", "Jun.", "Jul.",
	"Aug.", "Sep.", "Sept.", "Oct.", "Nov.", "Dec.",
	"Vol.", "No.", "Fig.", "Eq.",
	"approx.", "ca.", "al.", "ed.",
}

// -----------------------------------------------------------------------
// Retrieval — find patterns matching criteria.
// -----------------------------------------------------------------------

// Retrieve finds the best pattern matching the given criteria. Falls
// back gracefully: exact match -> function+tone -> function only -> any.
func (ae *AbsorptionEngine) Retrieve(fn DiscourseFunc, tone string, structure string) *AbsorbedPattern {
	ae.mu.RLock()
	defer ae.mu.RUnlock()

	if len(ae.patterns) == 0 {
		return nil
	}

	// Tier 1: exact match on all three criteria.
	if p := ae.findMatch(fn, tone, structure); p != nil {
		return p
	}

	// Tier 2: function + tone.
	if p := ae.findMatch(fn, tone, ""); p != nil {
		return p
	}

	// Tier 3: function only.
	if p := ae.findMatchByFunction(fn); p != nil {
		return p
	}

	// Tier 4: any pattern, prefer highest quality.
	return ae.bestPattern()
}

// findMatch searches for a pattern matching all non-empty criteria,
// preferring higher quality and lower usage count.
func (ae *AbsorptionEngine) findMatch(fn DiscourseFunc, tone, structure string) *AbsorbedPattern {
	indices, ok := ae.byFunction[fn]
	if !ok || len(indices) == 0 {
		return nil
	}

	var bestIdx int = -1
	bestScore := -1

	for _, idx := range indices {
		p := &ae.patterns[idx]
		if tone != "" && p.Tone != tone {
			continue
		}
		if structure != "" && p.Structure != structure {
			continue
		}
		// Score: quality bonus minus usage penalty.
		score := p.Quality*10 - p.UsageCount
		if score > bestScore {
			bestScore = score
			bestIdx = idx
		}
	}

	if bestIdx < 0 {
		return nil
	}
	ae.patterns[bestIdx].UsageCount++
	p := ae.patterns[bestIdx]
	return &p
}

// findMatchByFunction returns the best pattern for a function.
func (ae *AbsorptionEngine) findMatchByFunction(fn DiscourseFunc) *AbsorbedPattern {
	indices, ok := ae.byFunction[fn]
	if !ok || len(indices) == 0 {
		return nil
	}

	bestIdx := indices[0]
	bestScore := ae.patterns[bestIdx].Quality*10 - ae.patterns[bestIdx].UsageCount
	for _, idx := range indices[1:] {
		score := ae.patterns[idx].Quality*10 - ae.patterns[idx].UsageCount
		if score > bestScore {
			bestScore = score
			bestIdx = idx
		}
	}

	ae.patterns[bestIdx].UsageCount++
	p := ae.patterns[bestIdx]
	return &p
}

// bestPattern returns the highest-quality pattern overall.
func (ae *AbsorptionEngine) bestPattern() *AbsorbedPattern {
	if len(ae.patterns) == 0 {
		return nil
	}
	bestIdx := 0
	bestQ := ae.patterns[0].Quality
	for i, p := range ae.patterns[1:] {
		if p.Quality > bestQ {
			bestQ = p.Quality
			bestIdx = i + 1
		}
	}
	ae.patterns[bestIdx].UsageCount++
	p := ae.patterns[bestIdx]
	return &p
}

// -----------------------------------------------------------------------
// Realization — fill slots to produce a sentence.
// -----------------------------------------------------------------------

// Realize fills a pattern's slots with actual content and produces a
// finished sentence. Handles capitalization and a/an selection.
func (ae *AbsorptionEngine) Realize(pattern *AbsorbedPattern, slots map[string]string) string {
	if pattern == nil {
		return ""
	}

	result := pattern.Template

	// Fill each slot.
	for _, st := range pattern.SlotTypes {
		marker := "[" + st.Name + "]"
		value, ok := slots[st.Name]
		if !ok {
			value = ""
		}
		result = strings.Replace(result, marker, value, 1)
	}

	// Also fill any remaining slot markers from the map that weren't
	// in SlotTypes (defensive).
	for name, value := range slots {
		marker := "[" + name + "]"
		result = strings.ReplaceAll(result, marker, value)
	}

	// Fix a/an before vowel sounds.
	result = fixArticles(result)

	// Ensure sentence starts with uppercase.
	if len(result) > 0 && unicode.IsLower(rune(result[0])) {
		runes := []rune(result)
		runes[0] = unicode.ToUpper(runes[0])
		result = string(runes)
	}

	// Clean up double spaces.
	for strings.Contains(result, "  ") {
		result = strings.ReplaceAll(result, "  ", " ")
	}

	result = strings.TrimSpace(result)

	// Ensure sentence ends with punctuation.
	if len(result) > 0 {
		last := result[len(result)-1]
		if last != '.' && last != '!' && last != '?' {
			result += "."
		}
	}

	return result
}

// fixArticles corrects "a" to "an" before vowel sounds and vice versa.
func fixArticles(s string) string {
	words := strings.Fields(s)
	for i := 0; i < len(words)-1; i++ {
		lower := strings.ToLower(words[i])
		nextLower := strings.ToLower(words[i+1])
		if len(nextLower) == 0 {
			continue
		}
		firstChar := rune(nextLower[0])
		isVowelSound := firstChar == 'a' || firstChar == 'e' ||
			firstChar == 'i' || firstChar == 'o' ||
			(firstChar == 'u' && !strings.HasPrefix(nextLower, "uni") &&
				!strings.HasPrefix(nextLower, "use") &&
				!strings.HasPrefix(nextLower, "usa"))

		if lower == "a" && isVowelSound {
			// Preserve case.
			if words[i] == "A" {
				words[i] = "An"
			} else {
				words[i] = "an"
			}
		} else if lower == "an" && !isVowelSound {
			if words[i] == "An" {
				words[i] = "A"
			} else {
				words[i] = "a"
			}
		}
	}
	return strings.Join(words, " ")
}

// -----------------------------------------------------------------------
// Persistence — JSON save/load.
// -----------------------------------------------------------------------

type absorbedEntry struct {
	Template   string   `json:"t"`
	Function   string   `json:"f"`
	Tone       string   `json:"n"`
	Structure  string   `json:"r"`
	Source     string   `json:"s"`
	Quality    int      `json:"q"`
	UsageCount int      `json:"u"`
	Slots      []absSlotEntry `json:"l"`
}

type absSlotEntry struct {
	Name     string `json:"n"`
	Position int    `json:"p"`
	Kind     string `json:"k"`
}

// Save persists all patterns to disk as JSON.
func (ae *AbsorptionEngine) Save() error {
	ae.mu.RLock()
	defer ae.mu.RUnlock()

	if ae.savePath == "" {
		return nil
	}

	entries := make([]absorbedEntry, len(ae.patterns))
	for i, p := range ae.patterns {
		var se []absSlotEntry
		for _, st := range p.SlotTypes {
			se = append(se, absSlotEntry{
				Name:     st.Name,
				Position: st.Position,
				Kind:     st.Kind,
			})
		}
		entries[i] = absorbedEntry{
			Template:   p.Template,
			Function:   p.Function.String(),
			Tone:       p.Tone,
			Structure:  p.Structure,
			Source:     p.Source,
			Quality:    p.Quality,
			UsageCount: p.UsageCount,
			Slots:      se,
		}
	}

	data, err := json.Marshal(entries)
	if err != nil {
		return err
	}
	return os.WriteFile(ae.savePath, data, 0644)
}

// Load restores patterns from disk.
func (ae *AbsorptionEngine) Load() error {
	ae.mu.Lock()
	defer ae.mu.Unlock()

	if ae.savePath == "" {
		return nil
	}

	data, err := os.ReadFile(ae.savePath)
	if err != nil {
		return err
	}

	var entries []absorbedEntry
	if err := json.Unmarshal(data, &entries); err != nil {
		return err
	}

	ae.patterns = nil
	ae.byFunction = make(map[DiscourseFunc][]int)
	ae.byTone = make(map[string][]int)
	ae.byStructure = make(map[string][]int)

	for _, e := range entries {
		var sts []AbsorbedSlot
		for _, se := range e.Slots {
			sts = append(sts, AbsorbedSlot{
				Name:     se.Name,
				Position: se.Position,
				Kind:     se.Kind,
			})
		}
		p := AbsorbedPattern{
			Template:   e.Template,
			Function:   parseDFString(e.Function),
			Tone:       e.Tone,
			Structure:  e.Structure,
			Source:     e.Source,
			Quality:    e.Quality,
			UsageCount: e.UsageCount,
			SlotTypes:  sts,
		}
		idx := len(ae.patterns)
		ae.patterns = append(ae.patterns, p)
		ae.byFunction[p.Function] = append(ae.byFunction[p.Function], idx)
		ae.byTone[p.Tone] = append(ae.byTone[p.Tone], idx)
		ae.byStructure[p.Structure] = append(ae.byStructure[p.Structure], idx)
	}

	return nil
}

// -----------------------------------------------------------------------
// Stats — monitoring and introspection.
// -----------------------------------------------------------------------

// AbsorptionStats holds counts by function, tone, and structure.
type AbsorptionStats struct {
	Total       int            `json:"total"`
	ByFunction  map[string]int `json:"by_function"`
	ByTone      map[string]int `json:"by_tone"`
	ByStructure map[string]int `json:"by_structure"`
}

// Stats returns counts of absorbed patterns by function, tone, and structure.
func (ae *AbsorptionEngine) Stats() AbsorptionStats {
	ae.mu.RLock()
	defer ae.mu.RUnlock()

	stats := AbsorptionStats{
		Total:       len(ae.patterns),
		ByFunction:  make(map[string]int),
		ByTone:      make(map[string]int),
		ByStructure: make(map[string]int),
	}

	for fn, indices := range ae.byFunction {
		stats.ByFunction[fn.String()] = len(indices)
	}
	for tone, indices := range ae.byTone {
		stats.ByTone[tone] = len(indices)
	}
	for structure, indices := range ae.byStructure {
		stats.ByStructure[structure] = len(indices)
	}

	return stats
}

// PatternCount returns the total number of stored patterns.
func (ae *AbsorptionEngine) PatternCount() int {
	ae.mu.RLock()
	defer ae.mu.RUnlock()
	return len(ae.patterns)
}

// -----------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------

// absContainsAny returns true if s contains any of the given substrings.
func absContainsAny(s string, subs []string) bool {
	for _, sub := range subs {
		if strings.Contains(s, sub) {
			return true
		}
	}
	return false
}

// absContainsDigit returns true if the string contains a digit.
func absContainsDigit(s string) bool {
	for _, r := range s {
		if unicode.IsDigit(r) {
			return true
		}
	}
	return false
}
