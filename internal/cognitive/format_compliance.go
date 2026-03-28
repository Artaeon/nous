package cognitive

import (
	"regexp"
	"strconv"
	"strings"
	"unicode"
)

// -----------------------------------------------------------------------
// Format Compliance — detect user format requests and reshape NLG output.
//
// The NLG engine produces good content but may ignore formatting
// instructions embedded in the query (e.g. "in 3 bullet points").
// This module detects those instructions and transforms the output
// to match the requested format.
// -----------------------------------------------------------------------

// FormatRequest captures a detected formatting instruction.
type FormatRequest struct {
	Type     string // "bullets", "numbered", "paragraph", "one_sentence", "table"
	Count    int    // 0 = unspecified, N = exactly N items
	MaxWords int    // 0 = unspecified, N = max N words
	Style    string // "brief", "detailed", "formal", "casual"
}

// FormatCompliance detects format requests and reshapes output.
type FormatCompliance struct{}

// NewFormatCompliance creates a new format compliance checker.
func NewFormatCompliance() *FormatCompliance {
	return &FormatCompliance{}
}

// -----------------------------------------------------------------------
// Detection
// -----------------------------------------------------------------------

// Compiled patterns for format detection.
var (
	// "in N bullet points" / "in N bullets" / "as N bullets" / "N bullets"
	reBulletCount = regexp.MustCompile(`(?i)\b(?:(?:in|as|with)\s+)?(\d+)\s+bullet(?:\s*point)?s?\b`)
	// "in N points" / "N key points"
	rePointCount = regexp.MustCompile(`(?i)\b(\d+)\s+(?:key\s+)?points?\b`)
	// "in one sentence"
	reOneSentence = regexp.MustCompile(`(?i)\bin\s+(?:one|a\s+single)\s+sentence\b`)
	// "as a list" / "list the"
	reListGeneric = regexp.MustCompile(`(?i)(?:\bas\s+a\s+list\b|\blist\s+the\b)`)
	// "in N words or less" / "under N words" / "keep it to N words"
	reMaxWords = regexp.MustCompile(`(?i)(?:(?:in|under|within)\s+(\d+)\s+words|(\d+)\s+words\s+or\s+(?:less|fewer)|keep\s+it\s+to\s+(\d+)\s+words)`)
	// "step by step" / "numbered steps"
	reStepByStep = regexp.MustCompile(`(?i)\b(?:step\s+by\s+step|numbered\s+steps?)\b`)
	// "as a table" / "in table format"
	reTable = regexp.MustCompile(`(?i)\b(?:as\s+a\s+table|in\s+table\s+format)\b`)
	// Style: detailed
	reDetailed = regexp.MustCompile(`(?i)\b(?:detailed|in\s+depth|comprehensive)\b`)
	// Style: brief
	reBrief = regexp.MustCompile(`(?i)\b(?:briefly|concisely|short)\b`)
)

// DetectFormat parses format requests from the query. Returns nil if no
// formatting instruction is detected.
func (fc *FormatCompliance) DetectFormat(query string) *FormatRequest {
	var req FormatRequest
	found := false

	// Check bullet count patterns.
	if m := reBulletCount.FindStringSubmatch(query); m != nil {
		n, _ := strconv.Atoi(m[1])
		req.Type = "bullets"
		req.Count = n
		found = true
	}

	// Check numbered point patterns (only if no bullet match).
	if req.Type == "" {
		if m := rePointCount.FindStringSubmatch(query); m != nil {
			n, _ := strconv.Atoi(m[1])
			req.Type = "numbered"
			req.Count = n
			found = true
		}
	}

	// "in one sentence"
	if req.Type == "" && reOneSentence.MatchString(query) {
		req.Type = "one_sentence"
		found = true
	}

	// "briefly" as type (only if no type yet set; also sets style below)
	if req.Type == "" && reBrief.MatchString(query) {
		req.Type = "one_sentence"
		found = true
	}

	// "step by step" / "numbered steps"
	if req.Type == "" && reStepByStep.MatchString(query) {
		req.Type = "numbered"
		found = true
	}

	// "as a table" / "in table format"
	if req.Type == "" && reTable.MatchString(query) {
		req.Type = "table"
		found = true
	}

	// "as a list" / "list the"
	if req.Type == "" && reListGeneric.MatchString(query) {
		req.Type = "bullets"
		found = true
	}

	// Max words.
	if m := reMaxWords.FindStringSubmatch(query); m != nil {
		for _, g := range m[1:] {
			if g != "" {
				n, _ := strconv.Atoi(g)
				req.MaxWords = n
				found = true
				break
			}
		}
	}

	// Style detection.
	if reDetailed.MatchString(query) {
		req.Style = "detailed"
		found = true
	} else if reBrief.MatchString(query) {
		req.Style = "brief"
		found = true
	}

	if !found {
		return nil
	}
	return &req
}

// -----------------------------------------------------------------------
// Reshaping
// -----------------------------------------------------------------------

// Reshape transforms text to match the format request.
func (fc *FormatCompliance) Reshape(text string, req *FormatRequest) string {
	if req == nil {
		return text
	}

	// Apply maxWords constraint regardless of type.
	result := text

	switch req.Type {
	case "bullets":
		result = fc.reshapeBullets(text, req.Count)
	case "numbered":
		result = fc.reshapeNumbered(text, req.Count)
	case "one_sentence":
		result = fc.reshapeOneSentence(text)
	case "table":
		result = fc.reshapeTable(text)
	}

	// Apply max word limit as a post-pass.
	if req.MaxWords > 0 {
		result = fc.applyMaxWords(result, req.MaxWords)
	}

	return result
}

func (fc *FormatCompliance) reshapeBullets(text string, count int) string {
	sentences := SplitIntoSentences(text)
	if len(sentences) == 0 {
		return text
	}

	var selected []string
	if count > 0 {
		selected = PickBestSentences(sentences, count)
	} else {
		selected = sentences
	}

	var b strings.Builder
	for i, s := range selected {
		if i > 0 {
			b.WriteByte('\n')
		}
		b.WriteString("- ")
		b.WriteString(strings.TrimSpace(s))
	}
	return b.String()
}

func (fc *FormatCompliance) reshapeNumbered(text string, count int) string {
	sentences := SplitIntoSentences(text)
	if len(sentences) == 0 {
		return text
	}

	var selected []string
	if count > 0 {
		selected = PickBestSentences(sentences, count)
	} else {
		selected = sentences
	}

	var b strings.Builder
	for i, s := range selected {
		if i > 0 {
			b.WriteByte('\n')
		}
		b.WriteString(strconv.Itoa(i + 1))
		b.WriteString(". ")
		b.WriteString(strings.TrimSpace(s))
	}
	return b.String()
}

func (fc *FormatCompliance) reshapeOneSentence(text string) string {
	sentences := SplitIntoSentences(text)
	if len(sentences) == 0 {
		return text
	}
	best := PickBestSentences(sentences, 1)
	if len(best) == 0 {
		return text
	}
	return strings.TrimSpace(best[0])
}

func (fc *FormatCompliance) reshapeTable(text string) string {
	sentences := SplitIntoSentences(text)
	if len(sentences) == 0 {
		return text
	}

	type kvPair struct {
		Key   string
		Value string
	}

	var pairs []kvPair
	for _, s := range sentences {
		s = strings.TrimSpace(s)
		// Try "X is Y" pattern.
		if idx := strings.Index(s, " is "); idx > 0 {
			key := strings.TrimSpace(s[:idx])
			val := strings.TrimSpace(s[idx+4:])
			if key != "" && val != "" {
				pairs = append(pairs, kvPair{Key: key, Value: val})
				continue
			}
		}
		// Try "X: Y" pattern.
		if idx := strings.Index(s, ": "); idx > 0 {
			key := strings.TrimSpace(s[:idx])
			val := strings.TrimSpace(s[idx+2:])
			if key != "" && val != "" {
				pairs = append(pairs, kvPair{Key: key, Value: val})
				continue
			}
		}
		// Try "X — Y" / "X - Y" patterns.
		for _, sep := range []string{" — ", " – ", " - "} {
			if idx := strings.Index(s, sep); idx > 0 {
				key := strings.TrimSpace(s[:idx])
				val := strings.TrimSpace(s[idx+len(sep):])
				if key != "" && val != "" {
					pairs = append(pairs, kvPair{Key: key, Value: val})
					break
				}
			}
		}
	}

	if len(pairs) == 0 {
		// Fall back to bullets if we can't extract key-value pairs.
		return fc.reshapeBullets(text, 0)
	}

	// Find max key width for alignment.
	maxKey := 0
	for _, p := range pairs {
		if len(p.Key) > maxKey {
			maxKey = len(p.Key)
		}
	}

	var b strings.Builder
	for i, p := range pairs {
		if i > 0 {
			b.WriteByte('\n')
		}
		b.WriteString(p.Key)
		// Pad to align values.
		for j := len(p.Key); j < maxKey+2; j++ {
			b.WriteByte(' ')
		}
		b.WriteString(p.Value)
	}
	return b.String()
}

func (fc *FormatCompliance) applyMaxWords(text string, maxWords int) string {
	words := strings.Fields(text)
	if len(words) <= maxWords {
		return text
	}

	// Try to cut at sentence boundaries.
	sentences := SplitIntoSentences(text)
	var result []string
	wordCount := 0
	for _, s := range sentences {
		sWords := strings.Fields(s)
		if wordCount+len(sWords) > maxWords {
			if wordCount == 0 {
				// First sentence exceeds limit: truncate at word boundary.
				truncated := strings.Join(sWords[:maxWords], " ")
				return truncated + "..."
			}
			break
		}
		result = append(result, strings.TrimSpace(s))
		wordCount += len(sWords)
	}

	if len(result) == 0 {
		// Shouldn't happen, but safety fallback.
		return strings.Join(words[:maxWords], " ") + "..."
	}

	return strings.Join(result, " ")
}

// -----------------------------------------------------------------------
// Sentence Scoring
// -----------------------------------------------------------------------

// PickBestSentences scores sentences by informativeness and returns the
// top N, preserving their original order.
func PickBestSentences(sentences []string, n int) []string {
	if n <= 0 || len(sentences) == 0 {
		return nil
	}
	if n >= len(sentences) {
		// Return all, preserving order.
		out := make([]string, len(sentences))
		copy(out, sentences)
		return out
	}

	type scored struct {
		index int
		score float64
	}

	scores := make([]scored, len(sentences))
	for i, s := range sentences {
		scores[i] = scored{index: i, score: scoreFormatSentence(s, i)}
	}

	// Selection sort for top N (small N, no need for heap).
	for i := 0; i < n; i++ {
		best := i
		for j := i + 1; j < len(scores); j++ {
			if scores[j].score > scores[best].score {
				best = j
			}
		}
		scores[i], scores[best] = scores[best], scores[i]
	}

	// Collect top N indices and sort by original position.
	topIndices := make([]int, n)
	for i := 0; i < n; i++ {
		topIndices[i] = scores[i].index
	}
	// Insertion sort on indices to preserve original order.
	for i := 1; i < len(topIndices); i++ {
		key := topIndices[i]
		j := i - 1
		for j >= 0 && topIndices[j] > key {
			topIndices[j+1] = topIndices[j]
			j--
		}
		topIndices[j+1] = key
	}

	out := make([]string, n)
	for i, idx := range topIndices {
		out[i] = sentences[idx]
	}
	return out
}

// scoreFormatSentence rates how informative a sentence is for format
// compliance selection. Separate from extractive.go's scoreSentence which
// is tuned for topic-aware extraction.
func scoreFormatSentence(s string, position int) float64 {
	s = strings.TrimSpace(s)
	words := strings.Fields(s)

	// Base: length in words (more content = higher score).
	score := float64(len(words))

	// Bonus for capitalized words (likely entities/proper nouns).
	for _, w := range words {
		if len(w) > 0 && unicode.IsUpper(rune(w[0])) && len(w) > 1 {
			score += 1.5
		}
	}

	// Bonus for numbers and dates.
	for _, w := range words {
		for _, r := range w {
			if unicode.IsDigit(r) {
				score += 1.0
				break
			}
		}
	}

	// First sentence bonus (usually the definition/introduction).
	if position == 0 {
		score += 3.0
	}

	// Penalty for connector/transition sentences.
	lower := strings.ToLower(s)
	connectors := []string{
		"however", "moreover", "furthermore", "in addition",
		"on the other hand", "in conclusion", "to summarize",
		"that said", "nevertheless", "for example",
	}
	for _, c := range connectors {
		if strings.HasPrefix(lower, c) {
			score -= 2.0
			break
		}
	}

	// Penalty for very short sentences (likely filler).
	if len(words) < 4 {
		score -= 3.0
	}

	return score
}

// -----------------------------------------------------------------------
// Sentence Splitting
// -----------------------------------------------------------------------

// SplitIntoSentences splits text on sentence boundaries: ". " followed
// by an uppercase letter, "! ", "? ", or double newlines. Handles
// common abbreviations to avoid false splits.
func SplitIntoSentences(text string) []string {
	text = strings.TrimSpace(text)
	if text == "" {
		return nil
	}

	// Common abbreviations that should not trigger a split.
	abbrevs := map[string]bool{
		"mr": true, "mrs": true, "ms": true, "dr": true,
		"prof": true, "sr": true, "jr": true,
		"st": true, "ave": true, "blvd": true,
		"vs": true, "etc": true, "approx": true,
		"inc": true, "ltd": true, "corp": true,
		"dept": true, "univ": true, "govt": true,
		"e.g": true, "i.e": true, "u.s": true,
	}

	runes := []rune(text)
	var sentences []string
	start := 0

	for i := 0; i < len(runes); i++ {
		ch := runes[i]

		// Check for double newline as sentence boundary.
		if ch == '\n' && i+1 < len(runes) && runes[i+1] == '\n' {
			seg := strings.TrimSpace(string(runes[start:i]))
			if seg != "" {
				sentences = append(sentences, seg)
			}
			start = i + 2
			continue
		}

		// Check for sentence-ending punctuation.
		if ch != '.' && ch != '!' && ch != '?' {
			continue
		}

		// Need at least a space after the punctuation.
		if i+1 >= len(runes) {
			// End of text — this is the final sentence.
			continue
		}

		nextChar := runes[i+1]

		// For '!' and '?', split if followed by space+uppercase or space.
		if ch == '!' || ch == '?' {
			if nextChar == ' ' && i+2 < len(runes) && unicode.IsUpper(runes[i+2]) {
				seg := strings.TrimSpace(string(runes[start : i+1]))
				if seg != "" {
					sentences = append(sentences, seg)
				}
				start = i + 2
			}
			continue
		}

		// ch == '.': check for abbreviation.
		if nextChar != ' ' {
			continue
		}
		// ". " found — check if it's an abbreviation.
		if i+2 < len(runes) && !unicode.IsUpper(runes[i+2]) {
			continue // Not followed by uppercase, probably not a sentence break.
		}

		// Look back for the preceding word.
		wordEnd := i
		wordStart := i - 1
		for wordStart >= 0 && runes[wordStart] != ' ' && runes[wordStart] != '\n' {
			wordStart--
		}
		wordStart++
		if wordStart < wordEnd {
			prevWord := strings.ToLower(string(runes[wordStart:wordEnd]))
			// Strip any leading/trailing dots for abbreviation check.
			prevWord = strings.Trim(prevWord, ".")
			if abbrevs[prevWord] {
				continue
			}
			// Single uppercase letter followed by dot — likely an initial.
			if len([]rune(prevWord)) == 1 && unicode.IsUpper(rune(prevWord[0])) {
				continue
			}
		}

		seg := strings.TrimSpace(string(runes[start : i+1]))
		if seg != "" {
			sentences = append(sentences, seg)
		}
		start = i + 2
	}

	// Remaining text.
	seg := strings.TrimSpace(string(runes[start:]))
	if seg != "" {
		sentences = append(sentences, seg)
	}

	return sentences
}

// -----------------------------------------------------------------------
// Noun Phrase Chunker
// -----------------------------------------------------------------------

// ExtractNounPhrase extracts the main noun phrase from a query using
// syntactic chunking. This replaces pattern-based extractMainTopic.
func ExtractNounPhrase(query string) string {
	query = strings.TrimRight(strings.TrimSpace(query), "?!.")
	tokens := strings.Fields(query)
	if len(tokens) == 0 {
		return ""
	}

	// Leading function words / question words to skip.
	skipWords := map[string]bool{
		"what": true, "how": true, "why": true, "does": true,
		"is": true, "are": true, "can": true, "give": true,
		"me": true, "an": true, "the": true, "a": true,
		"of": true, "do": true, "did": true, "was": true,
		"were": true, "will": true, "would": true, "could": true,
		"should": true, "tell": true, "explain": true, "describe": true,
		"define": true, "please": true, "about": true, "who": true,
		"compare": true, "summarize": true, "summarise": true,
		"list": true, "show": true, "has": true, "have": true,
		"had": true,
	}

	// Meta-discourse words that describe the format of the answer, not the
	// topic itself. When followed by "of X", the real topic is X.
	metaWords := map[string]bool{
		"overview": true, "summary": true, "explanation": true,
		"description": true, "definition": true, "introduction": true,
		"outline": true, "rundown": true, "breakdown": true,
	}

	// Prepositions that stop noun phrase collection (except "of", "and").
	preps := map[string]bool{
		"in": true, "on": true, "at": true, "for": true,
		"by": true, "with": true, "to": true, "from": true,
		"about": true, "into": true, "through": true,
		"during": true, "before": true, "after": true,
		"between": true, "under": true, "over": true,
		"against": true, "without": true,
	}

	// Verbs/trailing words that terminate collection.
	// Note: "work/works" and "function/functions" omitted — too commonly nouns
	// ("remote work", "cognitive function").
	verbs := map[string]bool{
		"happen": true, "happens": true,
		"operate": true, "operates": true,
		"mean": true, "means": true, "affect": true, "affects": true,
		"cause": true, "causes": true, "change": true, "changes": true,
	}

	// Phase 1: Skip leading function words.
	i := 0
	for i < len(tokens) {
		lower := strings.ToLower(tokens[i])
		if !skipWords[lower] {
			break
		}
		i++
	}

	if i >= len(tokens) {
		// Everything was a function word — return last meaningful token.
		for j := len(tokens) - 1; j >= 0; j-- {
			if !skipWords[strings.ToLower(tokens[j])] {
				return tokens[j]
			}
		}
		return strings.Join(tokens, " ")
	}

	// Phase 2: Skip meta-discourse words. If the first content word is a
	// meta-word like "overview" followed by "of", skip both to reach the
	// actual topic: "overview of operating systems" → "operating systems".
	if i < len(tokens) {
		lower := strings.ToLower(tokens[i])
		if metaWords[lower] && i+1 < len(tokens) && strings.ToLower(tokens[i+1]) == "of" {
			i += 2 // skip meta-word and "of"
			// Also skip articles after "of": "overview of the ..."
			if i < len(tokens) {
				next := strings.ToLower(tokens[i])
				if next == "the" || next == "a" || next == "an" {
					i++
				}
			}
		}
	}

	// Phase 3: Collect noun phrase tokens.
	var phrase []string
	seenNoun := false

	for i < len(tokens) {
		tok := tokens[i]
		lower := strings.ToLower(tok)

		// "and" joins compound noun phrases: "Python and Go".
		if lower == "and" && seenNoun && i+1 < len(tokens) {
			// Check if next token looks like a noun.
			nextLower := strings.ToLower(tokens[i+1])
			if isNounToken(tokens[i+1], nextLower, skipWords) {
				phrase = append(phrase, "and")
				i++
				continue
			}
			break
		}

		// Special: "X of Y" — the real topic is usually Y.
		// "pros and cons of remote work" → "remote work"
		// "benefits of meditation" → "meditation"
		if lower == "of" && i+1 < len(tokens) {
			// Restart noun phrase collection from after "of"
			phrase = nil
			seenNoun = false
			i++
			continue
		}

		// Preposition stops collection (except "of" handled above).
		if preps[lower] && seenNoun {
			break
		}

		// Verb terminates.
		if verbs[lower] && seenNoun {
			break
		}

		// Check if this is a verb-like word ending: -s after noun, -ed, -ing.
		if seenNoun && isVerbLike(lower) && !isAdjectiveSuffix(lower) {
			break
		}

		// Accept adjective-like tokens (before a noun).
		if isAdjectiveSuffix(lower) {
			phrase = append(phrase, tok)
			i++
			continue
		}

		// Accept noun-like tokens.
		if isNounToken(tok, lower, skipWords) {
			phrase = append(phrase, tok)
			seenNoun = true
			i++
			continue
		}

		// If we haven't started collecting, skip this token.
		if !seenNoun {
			i++
			continue
		}

		// Unknown token after noun — stop.
		break
	}

	if len(phrase) == 0 {
		// Fallback: return everything from position i onward.
		if i < len(tokens) {
			return strings.Join(tokens[i:], " ")
		}
		return strings.Join(tokens, " ")
	}

	return strings.Join(phrase, " ")
}

// isNounToken returns true if the token looks like a noun.
func isNounToken(tok, lower string, skipWords map[string]bool) bool {
	if skipWords[lower] {
		return false
	}
	// Capitalized words are proper nouns.
	if len(tok) > 0 && unicode.IsUpper(rune(tok[0])) {
		return true
	}
	// Lowercase words not in skip list are potential nouns.
	if len(lower) > 1 {
		return true
	}
	return false
}

// isAdjectiveSuffix returns true if the word has a common adjective suffix.
func isAdjectiveSuffix(lower string) bool {
	suffixes := []string{"-al", "-ive", "-ous", "-ic", "-ful", "-less"}
	bareSuffixes := []string{"al", "ive", "ous", "ic", "ful", "less"}
	for _, s := range suffixes {
		if strings.HasSuffix(lower, s) {
			return true
		}
	}
	for _, s := range bareSuffixes {
		if len(lower) > len(s)+2 && strings.HasSuffix(lower, s) {
			return true
		}
	}
	return false
}

// isVerbLike returns true if the word looks like a verb form (not adjective).
func isVerbLike(lower string) bool {
	// Common verb endings.
	if strings.HasSuffix(lower, "ing") && len(lower) > 5 {
		// Could be adjective (-ing before noun) or gerund.
		// Only verb-like if not an adjective suffix pattern.
		return false // Conservatively treat -ing as adjective/noun.
	}
	if strings.HasSuffix(lower, "ed") && len(lower) > 4 {
		// Could be past participle used as adjective.
		return false // Conservatively allow.
	}
	return false
}
