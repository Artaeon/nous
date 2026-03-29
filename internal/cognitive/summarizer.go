package cognitive

import (
	"math"
	"sort"
	"strings"
	"unicode"
)

// Summarizer extracts key sentences from text to produce concise summaries.
// Pure heuristics — no ML, no LLM. Scores sentences by position, length,
// entity density, keyword overlap, numeric presence, cue phrases, and
// redundancy, then returns the top N in original order.
type Summarizer struct{}

// NewSummarizer creates a new extractive summarizer.
func NewSummarizer() *Summarizer {
	return &Summarizer{}
}

// scoredSentence holds a sentence with its computed importance score
// and original position.
type scoredSentence struct {
	text     string
	position int
	score    float64
}

// Summarize reduces text to the maxSentences most important sentences.
// Sentences are returned in their original order, not reordered by score.
func (s *Summarizer) Summarize(text string, maxSentences int) string {
	if maxSentences <= 0 {
		maxSentences = 3
	}

	sentences := splitTextSentences(text)
	if len(sentences) <= maxSentences {
		return strings.Join(sentences, " ")
	}

	selected := s.selectTopSentences(text, sentences, maxSentences)
	return strings.Join(selected, " ")
}

// SummarizeToLength reduces text to approximately maxWords words.
func (s *Summarizer) SummarizeToLength(text string, maxWords int) string {
	if maxWords <= 0 {
		maxWords = 50
	}

	sentences := splitTextSentences(text)
	if len(sentences) == 0 {
		return ""
	}

	// Count total words; if already under limit, return as-is.
	totalWords := 0
	for _, sent := range sentences {
		totalWords += len(strings.Fields(sent))
	}
	if totalWords <= maxWords {
		return strings.Join(sentences, " ")
	}

	// Score and pick sentences until we hit the word limit.
	scored := s.scoreSentences(text, sentences)

	// Sort by score descending to pick best first.
	sort.Slice(scored, func(i, j int) bool {
		return scored[i].score > scored[j].score
	})

	// Greedily select by score, tracking words.
	selectedPositions := make(map[int]bool)
	wordCount := 0
	for _, ss := range scored {
		w := len(strings.Fields(ss.text))
		if wordCount+w > maxWords && wordCount > 0 {
			continue
		}
		selectedPositions[ss.position] = true
		wordCount += w
		if wordCount >= maxWords {
			break
		}
	}

	// Emit in original order.
	var result []string
	for i, sent := range sentences {
		if selectedPositions[i] {
			result = append(result, sent)
		}
	}
	return strings.Join(result, " ")
}

// selectTopSentences scores all sentences, applies redundancy penalty,
// and returns the top N in original order.
func (s *Summarizer) selectTopSentences(text string, sentences []string, n int) []string {
	scored := s.scoreSentences(text, sentences)

	// Sort by score descending.
	sort.Slice(scored, func(i, j int) bool {
		return scored[i].score > scored[j].score
	})

	// Greedy selection with redundancy penalty.
	var selected []scoredSentence
	for _, candidate := range scored {
		if len(selected) >= n {
			break
		}

		// Check redundancy against already-selected sentences.
		// Two checks: (1) high word overlap, or (2) same leading subject
		// with moderate overlap (catches paraphrases about the same entity).
		redundant := false
		for _, sel := range selected {
			sim := sentenceSimilarity(candidate.text, sel.text)
			if sim > 0.4 {
				redundant = true
				break
			}
			// Subject-based redundancy: if both sentences start with the
			// same subject word and share moderate similarity, they're about
			// the same thing phrased differently.
			if sim > 0.15 && sameLeadingSubject(candidate.text, sel.text) {
				redundant = true
				break
			}
		}
		if redundant {
			continue
		}

		selected = append(selected, candidate)
	}

	// Sort selected by original position for natural reading order.
	sort.Slice(selected, func(i, j int) bool {
		return selected[i].position < selected[j].position
	})

	result := make([]string, len(selected))
	for i, ss := range selected {
		result[i] = ss.text
	}
	return result
}

// scoreSentences computes importance scores for every sentence.
func (s *Summarizer) scoreSentences(text string, sentences []string) []scoredSentence {
	if len(sentences) == 0 {
		return nil
	}

	// Identify paragraph boundaries for position scoring.
	paragraphFirstSentences := paragraphStartPositions(text, sentences)

	// Title words — from the first sentence (heuristic: first sentence
	// often names the topic).
	titleWords := summarizerContentWords(sentences[0])
	titleWordSet := make(map[string]bool)
	for _, w := range titleWords {
		titleWordSet[strings.ToLower(w)] = true
	}

	scored := make([]scoredSentence, len(sentences))
	for i, sent := range sentences {
		score := 0.0

		// 1. Position score: first sentence of each paragraph gets bonus.
		if paragraphFirstSentences[i] {
			score += 2.0
		}
		// First sentence of the text gets extra bonus.
		if i == 0 {
			score += 1.0
		}
		// Early sentences generally more important.
		if i < len(sentences)/4 {
			score += 0.5
		}

		// 2. Length score: medium-length sentences preferred.
		wordCount := len(strings.Fields(sent))
		score += lengthScore(wordCount)

		// 3. Entity density: capitalized words (excluding sentence-initial).
		score += entityDensityScore(sent)

		// 4. Keyword overlap with title/first sentence.
		score += keywordOverlapScore(sent, titleWordSet)

		// 5. Number presence: sentences with numbers/dates contain key facts.
		score += numberPresenceScore(sent)

		// 6. Cue phrases.
		score += cuePhrasesScore(sent)

		scored[i] = scoredSentence{
			text:     sent,
			position: i,
			score:    score,
		}
	}

	return scored
}

// -----------------------------------------------------------------------
// Scoring functions
// -----------------------------------------------------------------------

// lengthScore prefers medium-length sentences (10-30 words).
func lengthScore(wordCount int) float64 {
	if wordCount >= 10 && wordCount <= 30 {
		return 1.0
	}
	if wordCount >= 5 && wordCount < 10 {
		return 0.5
	}
	if wordCount > 30 && wordCount <= 50 {
		return 0.5
	}
	return 0.0
}

// entityDensityScore counts capitalized words (likely entities/proper nouns)
// excluding sentence-initial position.
func entityDensityScore(sent string) float64 {
	words := strings.Fields(sent)
	if len(words) <= 1 {
		return 0.0
	}

	entities := 0
	for i, w := range words {
		if i == 0 {
			continue // skip sentence-initial word
		}
		if len(w) > 0 && unicode.IsUpper(rune(w[0])) {
			entities++
		}
	}

	if len(words) == 0 {
		return 0.0
	}
	density := float64(entities) / float64(len(words))
	return density * 3.0 // scale so a heavily-entity sentence gets ~1.0
}

// keywordOverlapScore measures how many content words in sent overlap
// with the title word set.
func keywordOverlapScore(sent string, titleWordSet map[string]bool) float64 {
	if len(titleWordSet) == 0 {
		return 0.0
	}
	words := summarizerContentWords(sent)
	overlap := 0
	for _, w := range words {
		if titleWordSet[strings.ToLower(w)] {
			overlap++
		}
	}
	return float64(overlap) / float64(len(titleWordSet)) * 2.0
}

// numberPresenceScore gives a bonus if the sentence contains numbers or dates.
func numberPresenceScore(sent string) float64 {
	score := 0.0
	for _, r := range sent {
		if unicode.IsDigit(r) {
			score = 0.5
			break
		}
	}
	return score
}

// cuePhrasesScore gives a bonus for sentences containing importance-signaling phrases.
var cuePhrases = []string{
	"important", "significant", "key", "main", "crucial",
	"essential", "fundamental", "primary", "critical", "notably",
	"in conclusion", "in summary", "overall", "therefore",
	"as a result", "consequently", "most", "major",
}

func cuePhrasesScore(sent string) float64 {
	lower := strings.ToLower(sent)
	score := 0.0
	for _, phrase := range cuePhrases {
		if strings.Contains(lower, phrase) {
			score += 0.5
		}
	}
	// Cap at 1.5 to prevent single sentences from dominating.
	if score > 1.5 {
		score = 1.5
	}
	return score
}

// sentenceSimilarity computes word-level Jaccard similarity between two sentences.
func sentenceSimilarity(a, b string) float64 {
	setA := wordSet(a)
	setB := wordSet(b)

	if len(setA) == 0 || len(setB) == 0 {
		return 0.0
	}

	intersection := 0
	for w := range setA {
		if setB[w] {
			intersection++
		}
	}

	union := len(setA) + len(setB) - intersection
	if union == 0 {
		return 0.0
	}
	return float64(intersection) / float64(union)
}

// -----------------------------------------------------------------------
// Text splitting and helpers
// -----------------------------------------------------------------------

// splitTextSentences splits text into individual sentences.
// Handles ". ", "! ", "? " as sentence boundaries.
func splitTextSentences(text string) []string {
	text = strings.TrimSpace(text)
	if text == "" {
		return nil
	}

	var sentences []string
	remaining := text

	for len(remaining) > 0 {
		// Find the earliest sentence-ending punctuation followed by a space
		// or end-of-string.
		bestIdx := -1
		for _, punct := range []string{". ", "! ", "? "} {
			idx := strings.Index(remaining, punct)
			if idx >= 0 && (bestIdx < 0 || idx < bestIdx) {
				bestIdx = idx
			}
		}

		if bestIdx < 0 {
			// No more sentence boundaries — take the rest.
			s := strings.TrimSpace(remaining)
			if s != "" {
				sentences = append(sentences, s)
			}
			break
		}

		s := strings.TrimSpace(remaining[:bestIdx+1])
		if s != "" {
			sentences = append(sentences, s)
		}
		remaining = remaining[bestIdx+2:]
	}

	return sentences
}

// paragraphStartPositions identifies which sentence indices start a paragraph.
func paragraphStartPositions(text string, sentences []string) map[int]bool {
	result := make(map[int]bool)
	if len(sentences) == 0 {
		return result
	}
	result[0] = true // first sentence is always a paragraph start

	// Split text into paragraphs (double newline), find first sentence of each.
	paragraphs := strings.Split(text, "\n\n")
	sentIdx := 0
	for _, para := range paragraphs {
		para = strings.TrimSpace(para)
		if para == "" {
			continue
		}
		// Find which sentence index corresponds to the start of this paragraph.
		for sentIdx < len(sentences) {
			if strings.HasPrefix(para, sentences[sentIdx]) ||
				strings.HasPrefix(strings.TrimSpace(para), strings.TrimSpace(sentences[sentIdx])) {
				result[sentIdx] = true
				break
			}
			sentIdx++
		}
	}

	return result
}

// summarizerContentWords returns non-stopword words from a sentence.
func summarizerContentWords(sent string) []string {
	words := strings.Fields(sent)
	var result []string
	for _, w := range words {
		w = strings.Trim(w, ".,;:!?\"'()[]{}")
		if len(w) <= 2 {
			continue
		}
		lower := strings.ToLower(w)
		if summarizeStopWords[lower] {
			continue
		}
		result = append(result, w)
	}
	return result
}

// wordSet returns a set of lowercased content words.
func wordSet(sent string) map[string]bool {
	words := summarizerContentWords(sent)
	set := make(map[string]bool, len(words))
	for _, w := range words {
		set[strings.ToLower(w)] = true
	}
	return set
}

// summarizeStopWords are common English words excluded from similarity and overlap.
var summarizeStopWords = map[string]bool{
	"the": true, "and": true, "for": true, "are": true, "but": true,
	"not": true, "you": true, "all": true, "can": true, "had": true,
	"her": true, "was": true, "one": true, "our": true, "out": true,
	"has": true, "its": true, "his": true, "how": true, "man": true,
	"new": true, "now": true, "old": true, "see": true, "way": true,
	"who": true, "did": true, "get": true, "let": true, "say": true,
	"she": true, "too": true, "use": true, "that": true, "with": true,
	"have": true, "this": true, "will": true, "your": true, "from": true,
	"they": true, "been": true, "said": true, "each": true, "which": true,
	"their": true, "there": true, "about": true, "would": true, "these": true,
	"other": true, "into": true, "more": true, "some": true, "such": true,
	"than": true, "when": true, "what": true, "also": true,
	"were": true, "then": true, "them": true,
}

// sameLeadingSubject returns true if both sentences start with the same
// subject word (first significant word), indicating they discuss the same entity.
func sameLeadingSubject(a, b string) bool {
	subA := leadingSubject(a)
	subB := leadingSubject(b)
	return subA != "" && subA == subB
}

// leadingSubject extracts the first significant word from a sentence,
// skipping articles and short words.
func leadingSubject(sent string) string {
	words := strings.Fields(sent)
	skip := map[string]bool{"the": true, "a": true, "an": true, "this": true, "that": true}
	for _, w := range words {
		w = strings.Trim(w, ".,;:!?\"'()[]{}")
		lower := strings.ToLower(w)
		if len(lower) <= 2 || skip[lower] {
			continue
		}
		return lower
	}
	return ""
}

// Ensure math is used (for potential future scoring extensions).
var _ = math.Min
