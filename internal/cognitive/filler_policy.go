package cognitive

import (
	"regexp"
	"strings"
)

// FillerDetector identifies and removes filler content from generated text.
type FillerDetector struct {
	Patterns   []*regexp.Regexp
	Prefixes   []string
	Hedges     []string
	VagueWords []string
}

// FillerInstance represents a single detected filler occurrence.
type FillerInstance struct {
	Text     string
	Type     string // "ai_prefix", "hedge", "vague", "meta_comment", "empty_transition"
	Start    int    // character offset
	End      int
	Severity string // "must_remove", "should_remove", "minor"
}

// NewFillerDetector creates a FillerDetector with comprehensive filler patterns.
func NewFillerDetector() *FillerDetector {
	fd := &FillerDetector{
		Prefixes: []string{
			"As an AI",
			"As a language model",
			"As an artificial intelligence",
			"I think ",
			"In my opinion, ",
			"I'd say ",
			"I believe ",
			"I would say ",
			"I feel like ",
		},
		Hedges: []string{
			"Well, ",
			"So, ",
			"You know, ",
			"I mean, ",
			"Basically, ",
			"Essentially, ",
			"Honestly, ",
			"Actually, ",
			"Look, ",
		},
		VagueWords: []string{
			"things",
			"stuff",
			"a lot",
			"very",
			"really",
			"basically",
			"actually",
			"literally",
			"sort of",
			"kind of",
		},
	}

	// AI self-reference patterns
	fd.Patterns = append(fd.Patterns,
		regexp.MustCompile(`(?i)\bAs an AI\b[^.]*[.]`),
		regexp.MustCompile(`(?i)\bAs a language model\b[^.]*[.]`),
		regexp.MustCompile(`(?i)\bAs an artificial intelligence\b[^.]*[.]`),
		regexp.MustCompile(`(?i)^I think\s`),
		regexp.MustCompile(`(?i)^In my opinion,?\s`),
		regexp.MustCompile(`(?i)^I'd say\s`),
		regexp.MustCompile(`(?i)^I believe\s`),
		regexp.MustCompile(`(?i)^I would say\s`),
	)

	// Meta-comment patterns
	fd.Patterns = append(fd.Patterns,
		regexp.MustCompile(`(?i)\bThat's a great question\b[.!]?`),
		regexp.MustCompile(`(?i)\bGood question\b[.!]?`),
		regexp.MustCompile(`(?i)\bLet me think about that\b[.!]?`),
		regexp.MustCompile(`(?i)\bThat's an? (?:interesting|excellent|wonderful) question\b[.!]?`),
	)

	// Empty transition patterns
	fd.Patterns = append(fd.Patterns,
		regexp.MustCompile(`(?i)\bNow,? let'?s move on to\b`),
		regexp.MustCompile(`(?i)\bWith that being said\b`),
		regexp.MustCompile(`(?i)\bHaving said that\b`),
		regexp.MustCompile(`(?i)\bWith that in mind\b`),
	)

	// Repetitive acknowledgment patterns
	fd.Patterns = append(fd.Patterns,
		regexp.MustCompile(`(?i)^Sure!\s*`),
		regexp.MustCompile(`(?i)^Of course!\s*`),
		regexp.MustCompile(`(?i)^Absolutely!\s*`),
		regexp.MustCompile(`(?i)^Definitely!\s*`),
		regexp.MustCompile(`(?i)^Great!\s*`),
	)

	return fd
}

// DetectFiller finds all filler in a response text.
func (fd *FillerDetector) DetectFiller(text string) []FillerInstance {
	if strings.TrimSpace(text) == "" {
		return nil
	}

	var instances []FillerInstance

	// Pre-check for short fragments that splitSentences may drop (< 6 chars).
	// This handles cases like "Sure!" or "Great!" as standalone text.
	fragments := fillerSplitFragments(text)

	// Check sentence-level patterns using both splitSentences and our fragment splitter
	sentences := splitSentences(text)
	// Merge: use fragments for short items that splitSentences misses
	allParts := mergeFragmentsAndSentences(fragments, sentences)

	charOffset := 0
	for _, sent := range allParts {
		trimmed := strings.TrimSpace(sent)
		if trimmed == "" {
			charOffset += len(sent)
			continue
		}
		start := strings.Index(text[charOffset:], trimmed)
		if start < 0 {
			// Try from beginning as fallback
			start = strings.Index(text, trimmed)
			if start < 0 {
				start = 0
			}
			charOffset = 0
		}
		absStart := charOffset + start

		// AI self-reference detection
		if fd.isAIPrefix(trimmed) {
			instances = append(instances, FillerInstance{
				Text:     trimmed,
				Type:     "ai_prefix",
				Start:    absStart,
				End:      absStart + len(trimmed),
				Severity: "must_remove",
			})
		}

		// Meta-comment detection
		if fd.isMetaComment(trimmed) {
			instances = append(instances, FillerInstance{
				Text:     trimmed,
				Type:     "meta_comment",
				Start:    absStart,
				End:      absStart + len(trimmed),
				Severity: "must_remove",
			})
		}

		// Empty transition detection
		if fd.isEmptyTransition(trimmed) {
			instances = append(instances, FillerInstance{
				Text:     trimmed,
				Type:     "empty_transition",
				Start:    absStart,
				End:      absStart + len(trimmed),
				Severity: "should_remove",
			})
		}

		// Hedge detection at sentence start
		if hedge := fd.detectHedge(trimmed); hedge != "" {
			instances = append(instances, FillerInstance{
				Text:     hedge,
				Type:     "hedge",
				Start:    absStart,
				End:      absStart + len(hedge),
				Severity: "should_remove",
			})
		}

		// Repetitive acknowledgment detection
		if fd.isRepetitiveAck(trimmed) {
			instances = append(instances, FillerInstance{
				Text:     trimmed,
				Type:     "meta_comment",
				Start:    absStart,
				End:      absStart + len(trimmed),
				Severity: "must_remove",
			})
		}

		charOffset = absStart + len(trimmed)
	}

	// Check for vague words throughout text
	lower := strings.ToLower(text)
	for _, vw := range fd.VagueWords {
		idx := 0
		for {
			pos := strings.Index(lower[idx:], vw)
			if pos < 0 {
				break
			}
			absPos := idx + pos
			// Check word boundary
			if isWordBoundary(lower, absPos, len(vw)) {
				instances = append(instances, FillerInstance{
					Text:     text[absPos : absPos+len(vw)],
					Type:     "vague",
					Start:    absPos,
					End:      absPos + len(vw),
					Severity: "minor",
				})
			}
			idx = absPos + len(vw)
			if idx >= len(lower) {
				break
			}
		}
	}

	return instances
}

// RemoveFiller strips filler from text, preserving meaning.
func (fd *FillerDetector) RemoveFiller(text string) string {
	if strings.TrimSpace(text) == "" {
		return text
	}

	sentences := splitSentences(text)
	var cleaned []string

	for _, sent := range sentences {
		trimmed := strings.TrimSpace(sent)
		if trimmed == "" {
			continue
		}

		// Skip entire sentence if it's an AI self-reference or meta-comment
		if fd.isAIPrefix(trimmed) || fd.isMetaComment(trimmed) || fd.isRepetitiveAck(trimmed) {
			continue
		}

		// Remove hedge prefixes
		result := fd.removeHedgePrefix(trimmed)

		// Remove empty transitions
		result = fd.removeEmptyTransitions(result)

		result = strings.TrimSpace(result)
		if result == "" {
			continue
		}

		// Ensure sentence starts with uppercase
		if len(result) > 0 {
			result = capitalizeFirst(result)
		}

		// Ensure sentence ends with punctuation
		if !hasSentenceEnding(result) {
			result += "."
		}

		cleaned = append(cleaned, result)
	}

	if len(cleaned) == 0 {
		return ""
	}

	return strings.Join(cleaned, " ")
}

// EnforcePolicy checks if text meets the no-filler policy.
// Returns cleaned text and whether any changes were made.
func (fd *FillerDetector) EnforcePolicy(text string, isTaskPrompt bool) (string, bool) {
	original := text
	instances := fd.DetectFiller(text)

	if len(instances) == 0 {
		return text, false
	}

	if isTaskPrompt {
		// For task prompts, aggressively remove all filler
		cleaned := fd.RemoveFiller(text)
		return cleaned, cleaned != original
	}

	// For conversational text, only remove must_remove items
	hasMustRemove := false
	for _, inst := range instances {
		if inst.Severity == "must_remove" {
			hasMustRemove = true
			break
		}
	}

	if !hasMustRemove {
		return text, false
	}

	cleaned := fd.RemoveFiller(text)
	return cleaned, cleaned != original
}

// IsStructuredUncertainty checks if text is a well-formed uncertainty statement
// (as opposed to vague hedging).
func IsStructuredUncertainty(text string) bool {
	lower := strings.ToLower(strings.TrimSpace(text))
	if lower == "" {
		return false
	}

	// Structured uncertainty indicators
	structuredPhrases := []string{
		"i don't have enough information",
		"here's what i know",
		"here is what i know",
		"what's uncertain",
		"what is uncertain",
		"what remains unclear",
		"the evidence suggests",
		"but it's unclear whether",
		"but it is unclear whether",
		"however, it's not certain",
		"however, it is not certain",
		"this is debated",
		"experts disagree on",
		"the data is limited",
		"there is limited evidence",
		"current understanding suggests",
		"it's worth noting that",
		"it is worth noting that",
		"one limitation is",
		"a key uncertainty is",
	}

	// Must contain at least one structured uncertainty phrase
	hasStructured := false
	for _, phrase := range structuredPhrases {
		if strings.Contains(lower, phrase) {
			hasStructured = true
			break
		}
	}

	if !hasStructured {
		return false
	}

	// Must NOT be primarily vague hedging
	vagueCount := 0
	vagueHedges := []string{"i think", "i believe", "maybe", "perhaps", "i guess", "probably", "might be"}
	for _, h := range vagueHedges {
		if strings.Contains(lower, h) {
			vagueCount++
		}
	}

	// If there are more vague hedges than structured markers, it's not structured
	structuredCount := 0
	for _, phrase := range structuredPhrases {
		if strings.Contains(lower, phrase) {
			structuredCount++
		}
	}

	return structuredCount >= vagueCount
}

// isAIPrefix checks if a sentence starts with an AI self-reference.
func (fd *FillerDetector) isAIPrefix(sent string) bool {
	lower := strings.ToLower(strings.TrimSpace(sent))
	aiPrefixes := []string{
		"as an ai",
		"as a language model",
		"as an artificial intelligence",
	}
	for _, p := range aiPrefixes {
		if strings.HasPrefix(lower, p) {
			return true
		}
	}
	return false
}

// isMetaComment checks if a sentence is a meta-comment about the question.
func (fd *FillerDetector) isMetaComment(sent string) bool {
	lower := strings.ToLower(strings.TrimSpace(sent))
	metaPhrases := []string{
		"that's a great question",
		"that is a great question",
		"good question",
		"let me think about that",
		"that's an interesting question",
		"that is an interesting question",
		"that's an excellent question",
		"that is an excellent question",
		"that's a wonderful question",
		"that is a wonderful question",
	}
	for _, p := range metaPhrases {
		if strings.HasPrefix(lower, p) || lower == p || lower == p+"." || lower == p+"!" {
			return true
		}
	}
	return false
}

// isRepetitiveAck checks if a sentence is a repetitive acknowledgment.
func (fd *FillerDetector) isRepetitiveAck(sent string) bool {
	trimmed := strings.TrimSpace(sent)
	acks := []string{
		"Sure!", "Of course!", "Absolutely!", "Definitely!", "Great!",
		"Sure.", "Of course.", "Absolutely.", "Definitely.", "Great.",
	}
	for _, a := range acks {
		if trimmed == a {
			return true
		}
	}
	return false
}

// isEmptyTransition checks if a sentence is an empty transition.
func (fd *FillerDetector) isEmptyTransition(sent string) bool {
	lower := strings.ToLower(strings.TrimSpace(sent))
	transitions := []string{
		"now, let's move on to",
		"now let's move on to",
		"with that being said",
		"having said that",
		"with that in mind",
	}
	for _, t := range transitions {
		if strings.HasPrefix(lower, t) {
			return true
		}
	}
	return false
}

// detectHedge returns the hedge prefix if found at the start of a sentence.
func (fd *FillerDetector) detectHedge(sent string) string {
	for _, h := range fd.Hedges {
		if len(sent) >= len(h) && strings.EqualFold(sent[:len(h)], h) {
			return sent[:len(h)]
		}
	}
	return ""
}

// removeHedgePrefix removes hedge prefixes from the beginning of a sentence.
func (fd *FillerDetector) removeHedgePrefix(sent string) string {
	result := sent
	for {
		changed := false
		for _, h := range fd.Hedges {
			if len(result) >= len(h) && strings.EqualFold(result[:len(h)], h) {
				result = strings.TrimSpace(result[len(h):])
				changed = true
				break
			}
		}
		if !changed {
			break
		}
	}
	return result
}

// removeEmptyTransitions removes empty transition phrases from text.
func (fd *FillerDetector) removeEmptyTransitions(sent string) string {
	result := sent
	transitions := []string{
		"Now, let's move on to ",
		"Now let's move on to ",
		"With that being said, ",
		"Having said that, ",
		"With that in mind, ",
	}
	lower := strings.ToLower(result)
	for _, t := range transitions {
		tl := strings.ToLower(t)
		if strings.HasPrefix(lower, tl) {
			result = strings.TrimSpace(result[len(t):])
			lower = strings.ToLower(result)
		}
	}
	return result
}

// fillerSplitFragments splits text on sentence-ending punctuation without
// a minimum length filter, so short fragments like "Sure!" are preserved.
func fillerSplitFragments(text string) []string {
	if strings.TrimSpace(text) == "" {
		return nil
	}
	var fragments []string
	var current strings.Builder
	runes := []rune(text)
	for i := 0; i < len(runes); i++ {
		current.WriteRune(runes[i])
		if runes[i] == '.' || runes[i] == '!' || runes[i] == '?' {
			if i+1 >= len(runes) || runes[i+1] == ' ' || runes[i+1] == '\n' || runes[i+1] == '\r' {
				s := strings.TrimSpace(current.String())
				if s != "" {
					fragments = append(fragments, s)
				}
				current.Reset()
			}
		}
	}
	if rem := strings.TrimSpace(current.String()); rem != "" {
		fragments = append(fragments, rem)
	}
	return fragments
}

// mergeFragmentsAndSentences returns a de-duplicated union of fragments and sentences,
// preserving order from fragments. This ensures short items dropped by splitSentences
// are still processed.
func mergeFragmentsAndSentences(fragments, sentences []string) []string {
	seen := make(map[string]bool)
	var result []string
	for _, s := range sentences {
		norm := strings.ToLower(strings.TrimSpace(s))
		if norm != "" {
			seen[norm] = true
		}
	}
	for _, f := range fragments {
		norm := strings.ToLower(strings.TrimSpace(f))
		if norm == "" {
			continue
		}
		if !seen[norm] {
			result = append(result, f)
		}
	}
	// Add all sentences
	for _, s := range sentences {
		if strings.TrimSpace(s) != "" {
			result = append(result, s)
		}
	}
	return result
}

// isWordBoundary checks if the match at position pos with length l is at word boundaries.
func isWordBoundary(text string, pos, l int) bool {
	if pos > 0 {
		c := text[pos-1]
		if isAlphaNum(c) {
			return false
		}
	}
	end := pos + l
	if end < len(text) {
		c := text[end]
		if isAlphaNum(c) {
			return false
		}
	}
	return true
}

func isAlphaNum(c byte) bool {
	return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9')
}

// hasSentenceEnding checks if text ends with sentence-ending punctuation.
func hasSentenceEnding(text string) bool {
	if text == "" {
		return false
	}
	last := text[len(text)-1]
	return last == '.' || last == '!' || last == '?'
}
