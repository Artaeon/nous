package cognitive

import (
	"math"
	"regexp"
	"sort"
	"strings"
	"unicode"
)

// ---------------------------------------------------------------------------
// Enhanced Extractive Summarizer
//
// Pure deterministic text summarization using TF-IDF sentence scoring,
// position bonuses, length penalties, and entity density.  No ML, no LLM.
// ---------------------------------------------------------------------------

// abbreviationRe matches common abbreviations so we don't split on them.
var abbreviationRe = regexp.MustCompile(
	`(?i)\b(?:Mr|Mrs|Ms|Dr|Prof|Sr|Jr|vs|etc|approx|dept|est|inc|corp|ltd|co|govt|e\.g|i\.e|U\.S|U\.K|U\.N)\.`)

// headingRe matches Markdown headings (# or ##).
var headingRe = regexp.MustCompile(`(?m)^#{1,6}\s+`)

// splitSummarySentences splits text into sentences, handling common
// abbreviations so "Dr. Smith went home." is not split at "Dr.".
func splitSummarySentences(text string) []string {
	// Normalize line breaks within paragraphs but preserve paragraph breaks.
	text = strings.TrimSpace(text)
	if text == "" {
		return nil
	}

	// Replace abbreviation dots with a placeholder to avoid false splits.
	const placeholder = "\x00"
	masked := abbreviationRe.ReplaceAllStringFunc(text, func(m string) string {
		return strings.ReplaceAll(m, ".", placeholder)
	})

	// Split on sentence-ending punctuation followed by whitespace or end-of-string.
	var sentences []string
	remaining := masked
	for len(remaining) > 0 {
		idx := -1
		for i := 0; i < len(remaining)-1; i++ {
			if (remaining[i] == '.' || remaining[i] == '!' || remaining[i] == '?') &&
				(remaining[i+1] == ' ' || remaining[i+1] == '\n' || remaining[i+1] == '\t') {
				idx = i
				break
			}
		}
		// Also check if remaining ends with sentence punctuation.
		if idx < 0 {
			last := remaining[len(remaining)-1]
			if last == '.' || last == '!' || last == '?' {
				idx = len(remaining) - 1
			}
		}

		if idx < 0 {
			s := strings.TrimSpace(remaining)
			if s != "" {
				sentences = append(sentences, strings.ReplaceAll(s, placeholder, "."))
			}
			break
		}

		s := strings.TrimSpace(remaining[:idx+1])
		if s != "" {
			sentences = append(sentences, strings.ReplaceAll(s, placeholder, "."))
		}
		remaining = remaining[idx+1:]
		// Skip leading whitespace for next sentence.
		remaining = strings.TrimLeft(remaining, " \t\n")
	}

	return sentences
}

// sentenceAfterHeading returns true if the sentence immediately follows a
// heading line in the original text.
func sentenceAfterHeading(text string, sent string) bool {
	idx := strings.Index(text, sent)
	if idx <= 0 {
		return false
	}
	// Walk backwards from idx to find the preceding line.
	before := text[:idx]
	before = strings.TrimRight(before, " \t\n")
	lastNewline := strings.LastIndex(before, "\n")
	var prevLine string
	if lastNewline >= 0 {
		prevLine = before[lastNewline+1:]
	} else {
		prevLine = before
	}
	return headingRe.MatchString(prevLine)
}

// countProperNouns counts capitalized words in a sentence, skipping the first
// word (which is capitalized due to sentence start).
func countProperNouns(sent string) int {
	words := strings.Fields(sent)
	count := 0
	for i, w := range words {
		if i == 0 {
			continue
		}
		if len(w) > 0 && unicode.IsUpper(rune(w[0])) {
			count++
		}
	}
	return count
}

// summaryScore holds a sentence with its computed score and original position.
type summaryScore struct {
	text     string
	position int
	score    float64
}

// scoreSummarySentences applies the full TF-IDF + position + length + entity
// density algorithm described in the spec.
func scoreSummarySentences(text string, sentences []string) []summaryScore {
	n := len(sentences)
	if n == 0 {
		return nil
	}

	// ---- Build per-sentence token lists (lowercased, stop-words removed) ----
	type tokenInfo struct {
		tokens []string
		wordCount int
	}
	infos := make([]tokenInfo, n)
	for i, s := range sentences {
		words := strings.Fields(s)
		infos[i].wordCount = len(words)
		// Tokenize: lowercase, strip punctuation, remove stop words.
		for _, w := range words {
			w = strings.Trim(w, ".,;:!?\"'()[]{}")
			lower := strings.ToLower(w)
			if len(lower) < 2 {
				continue
			}
			if extractiveStopWords[lower] || summarizeStopWords[lower] {
				continue
			}
			infos[i].tokens = append(infos[i].tokens, lower)
		}
	}

	// ---- Document frequency: how many sentences contain each word ----
	df := make(map[string]int)
	for _, info := range infos {
		seen := make(map[string]bool)
		for _, tok := range info.tokens {
			if !seen[tok] {
				df[tok]++
				seen[tok] = true
			}
		}
	}

	// ---- Score each sentence ----
	scored := make([]summaryScore, n)
	for i, sent := range sentences {
		info := infos[i]

		// 1. TF-IDF score
		var tfidf float64
		if len(info.tokens) > 0 {
			// Count term frequencies in this sentence.
			tf := make(map[string]int)
			for _, tok := range info.tokens {
				tf[tok]++
			}
			for tok, count := range tf {
				termFreq := float64(count) / float64(len(info.tokens))
				docFreqVal := df[tok]
				if docFreqVal == 0 {
					docFreqVal = 1
				}
				idf := math.Log(float64(n) / float64(docFreqVal))
				tfidf += termFreq * idf
			}
		}

		// 2. Position multiplier
		posMultiplier := 1.0
		if i == 0 {
			posMultiplier = 1.50 // first sentence: +50%
		} else if i == 1 {
			posMultiplier = 1.20 // second sentence: +20%
		} else if i == n-1 {
			posMultiplier = 1.30 // last sentence: +30%
		}
		if sentenceAfterHeading(text, sent) {
			posMultiplier += 0.25 // after heading: +25%
		}

		// 3. Length multiplier
		lengthMultiplier := 1.0
		wc := info.wordCount
		if wc < 5 {
			lengthMultiplier = 0.50 // very short: -50%
		} else if wc > 40 {
			lengthMultiplier = 0.80 // very long: -20%
		}
		// Optimal 10-25: no penalty (1.0)

		// 4. Entity density multiplier
		properNouns := countProperNouns(sent)
		entityBonus := float64(properNouns) * 0.10
		if entityBonus > 0.50 {
			entityBonus = 0.50
		}
		entityMultiplier := 1.0 + entityBonus

		// 5. Final score = TF-IDF * position * length * entity
		finalScore := tfidf * posMultiplier * lengthMultiplier * entityMultiplier

		scored[i] = summaryScore{
			text:     sent,
			position: i,
			score:    finalScore,
		}
	}

	return scored
}

// ExtractSummary takes arbitrary text and returns a summary composed of the
// most important sentences, preserving their original order for readability.
// Uses sentence importance scoring: TF-IDF + position + length + entity density.
// Fully deterministic, no ML.
func ExtractSummary(text string, maxSentences int) string {
	text = strings.TrimSpace(text)
	if text == "" {
		return ""
	}
	if maxSentences <= 0 {
		maxSentences = 5
	}

	sentences := splitSummarySentences(text)
	if len(sentences) == 0 {
		return ""
	}
	if len(sentences) <= maxSentences {
		return strings.Join(sentences, " ")
	}

	scored := scoreSummarySentences(text, sentences)

	// Sort by score descending to pick the top N.
	sorted := make([]summaryScore, len(scored))
	copy(sorted, scored)
	sort.Slice(sorted, func(i, j int) bool {
		return sorted[i].score > sorted[j].score
	})

	// Select top maxSentences.
	selected := sorted
	if len(selected) > maxSentences {
		selected = selected[:maxSentences]
	}

	// Re-sort by original position so the summary reads naturally.
	sort.Slice(selected, func(i, j int) bool {
		return selected[i].position < selected[j].position
	})

	parts := make([]string, len(selected))
	for i, s := range selected {
		parts[i] = s.text
	}
	return strings.Join(parts, " ")
}

// ExtractBullets returns the top sentences as bullet points.
func ExtractBullets(text string, maxBullets int) string {
	text = strings.TrimSpace(text)
	if text == "" {
		return ""
	}
	if maxBullets <= 0 {
		maxBullets = 5
	}

	sentences := splitSummarySentences(text)
	if len(sentences) == 0 {
		return ""
	}

	scored := scoreSummarySentences(text, sentences)

	// Sort by score descending.
	sorted := make([]summaryScore, len(scored))
	copy(sorted, scored)
	sort.Slice(sorted, func(i, j int) bool {
		return sorted[i].score > sorted[j].score
	})

	selected := sorted
	if len(selected) > maxBullets {
		selected = selected[:maxBullets]
	}

	// Re-sort by original position.
	sort.Slice(selected, func(i, j int) bool {
		return selected[i].position < selected[j].position
	})

	var b strings.Builder
	for i, s := range selected {
		if i > 0 {
			b.WriteByte('\n')
		}
		b.WriteString("- ")
		b.WriteString(s.text)
	}
	return b.String()
}

// ExtractOneLiner returns a single-sentence summary.
func ExtractOneLiner(text string) string {
	return ExtractSummary(text, 1)
}
