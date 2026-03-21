package cognitive

import (
	"math"
	"math/rand"
	"regexp"
	"sort"
	"strings"
	"unicode"
)

// -----------------------------------------------------------------------
// TextTransformer — pure-code text transformation engine.
//
// Performs paraphrase, summarize, formalize, casualize, bulletize,
// prosify, and simplify operations WITHOUT any LLM calls.
// Uses synonym maps, TF-IDF scoring, contraction expansion,
// and structural rewriting — all in pure Go.
// -----------------------------------------------------------------------

// TransformOp specifies which transformation to apply.
type TransformOp string

const (
	OpParaphrase TransformOp = "paraphrase"
	OpSummarize  TransformOp = "summarize"
	OpFormalize  TransformOp = "formalize"
	OpCasualize  TransformOp = "casualize"
	OpBulletize  TransformOp = "bulletize"
	OpProsify    TransformOp = "prosify"
	OpSimplify   TransformOp = "simplify"
)

// TextTransformer performs text transformations using pure algorithmic methods.
type TextTransformer struct {
	Embeddings *WordEmbeddings // for similarity-based operations
	rng        *rand.Rand
}

// NewTextTransformer creates a new transformer engine.
func NewTextTransformer(emb *WordEmbeddings) *TextTransformer {
	return &TextTransformer{
		Embeddings: emb,
		rng:        rand.New(rand.NewSource(42)),
	}
}

// Transform applies the specified operation to the input text.
func (t *TextTransformer) Transform(text string, operation TransformOp) string {
	text = strings.TrimSpace(text)
	if text == "" {
		return ""
	}

	switch operation {
	case OpParaphrase:
		return t.paraphrase(text)
	case OpSummarize:
		return t.summarize(text)
	case OpFormalize:
		return t.formalize(text)
	case OpCasualize:
		return t.casualize(text)
	case OpBulletize:
		return t.bulletize(text)
	case OpProsify:
		return t.prosify(text)
	case OpSimplify:
		return t.simplify(text)
	default:
		return text
	}
}

// -----------------------------------------------------------------------
// Paraphrase — synonym substitution + clause reordering
// -----------------------------------------------------------------------

func (t *TextTransformer) paraphrase(text string) string {
	sents := transformSplitSentences(text)
	if len(sents) == 0 {
		return text
	}

	var result []string
	for _, sent := range sents {
		sent = t.substituteSynonyms(sent)
		sent = t.restructureSentence(sent)
		result = append(result, sent)
	}
	return strings.Join(result, " ")
}

// substituteSynonyms replaces words with random synonyms.
func (t *TextTransformer) substituteSynonyms(sent string) string {
	words := strings.Fields(sent)
	if len(words) == 0 {
		return sent
	}

	changed := false
	for i, w := range words {
		clean := strings.ToLower(stripPunctuation(w))
		if syns, ok := transformSynonyms[clean]; ok && len(syns) > 0 {
			// ~40% chance to substitute each eligible word
			if t.rng.Float64() < 0.4 {
				replacement := syns[t.rng.Intn(len(syns))]
				// Preserve capitalization
				if len(w) > 0 && unicode.IsUpper(rune(w[0])) {
					replacement = transformCapFirst(replacement)
				}
				// Preserve trailing punctuation
				suffix := trailingPunctuation(w)
				words[i] = replacement + suffix
				changed = true
			}
		}
	}

	if !changed && len(words) > 2 {
		// Force at least one substitution
		for i, w := range words {
			clean := strings.ToLower(stripPunctuation(w))
			if syns, ok := transformSynonyms[clean]; ok && len(syns) > 0 {
				replacement := syns[t.rng.Intn(len(syns))]
				if len(w) > 0 && unicode.IsUpper(rune(w[0])) {
					replacement = transformCapFirst(replacement)
				}
				suffix := trailingPunctuation(w)
				words[i] = replacement + suffix
				break
			}
		}
	}

	return strings.Join(words, " ")
}

// restructureSentence reorders clauses or swaps active/passive where possible.
func (t *TextTransformer) restructureSentence(sent string) string {
	// Try clause reordering for sentences with commas
	if idx := strings.Index(sent, ", "); idx > 0 && idx < len(sent)-3 {
		before := strings.TrimSpace(sent[:idx])
		after := strings.TrimSpace(sent[idx+2:])

		// Only reorder if both parts are substantial
		if len(before) > 10 && len(after) > 10 && t.rng.Float64() < 0.3 {
			// Move the dependent clause
			afterCap := transformCapFirst(after)
			// Remove trailing period from 'after' part if we're putting 'before' at the end
			afterCap = strings.TrimRight(afterCap, ".")
			beforeLower := strings.ToLower(before[:1]) + before[1:]
			return afterCap + ", " + beforeLower
		}
	}
	return sent
}

// -----------------------------------------------------------------------
// Summarize — TF-IDF based extractive summarization
// -----------------------------------------------------------------------

func (t *TextTransformer) summarize(text string) string {
	sents := transformSplitSentences(text)
	if len(sents) <= 2 {
		return text
	}

	// Compute term frequencies across the whole document
	docWords := strings.Fields(strings.ToLower(text))
	tf := make(map[string]int)
	for _, w := range docWords {
		w = stripPunctuation(w)
		if w != "" && !transformStopWords[w] {
			tf[w]++
		}
	}

	// Compute sentence frequency for IDF
	sf := make(map[string]int) // word → number of sentences containing it
	for _, sent := range sents {
		seen := make(map[string]bool)
		for _, w := range strings.Fields(strings.ToLower(sent)) {
			w = stripPunctuation(w)
			if w != "" && !seen[w] {
				sf[w]++
				seen[w] = true
			}
		}
	}

	// Score each sentence
	type scoredSent struct {
		text  string
		score float64
		idx   int
	}
	var scored []scoredSent
	totalSents := float64(len(sents))

	for i, sent := range sents {
		words := strings.Fields(strings.ToLower(sent))
		var score float64
		contentWords := 0
		for _, w := range words {
			w = stripPunctuation(w)
			if w == "" || transformStopWords[w] {
				continue
			}
			contentWords++
			// TF-IDF: tf * log(N / sf)
			termFreq := float64(tf[w])
			sentFreq := float64(sf[w])
			if sentFreq == 0 {
				sentFreq = 1
			}
			idf := math.Log(totalSents / sentFreq)
			score += termFreq * idf
		}
		if contentWords > 0 {
			score /= float64(contentWords) // normalize by length
		}

		// Position bonus: first sentences are more important
		if i < 3 {
			score *= 1.0 + 0.3*(1.0-float64(i)/3.0)
		}

		scored = append(scored, scoredSent{text: sent, score: score, idx: i})
	}

	// Sort by score descending
	sort.Slice(scored, func(i, j int) bool {
		return scored[i].score > scored[j].score
	})

	// Pick top N sentences (roughly 30-40% of original)
	n := len(sents) / 3
	if n < 1 {
		n = 1
	}
	if n > 5 {
		n = 5
	}
	top := scored[:n]

	// Re-sort by original position for coherent output
	sort.Slice(top, func(i, j int) bool {
		return top[i].idx < top[j].idx
	})

	var parts []string
	for _, s := range top {
		parts = append(parts, ensureTransformPeriod(s.text))
	}
	return strings.Join(parts, " ")
}

// -----------------------------------------------------------------------
// Formalize — expand contractions, formal vocabulary, hedging
// -----------------------------------------------------------------------

func (t *TextTransformer) formalize(text string) string {
	// Expand contractions
	result := expandContractions(text)

	// Replace casual words with formal equivalents
	words := strings.Fields(result)
	for i, w := range words {
		clean := strings.ToLower(stripPunctuation(w))
		if formal, ok := casualToFormal[clean]; ok {
			suffix := trailingPunctuation(w)
			replacement := formal
			if len(w) > 0 && unicode.IsUpper(rune(w[0])) {
				replacement = transformCapFirst(replacement)
			}
			words[i] = replacement + suffix
		}
	}
	result = strings.Join(words, " ")

	// Add hedging phrases to assertions (first sentence)
	sents := transformSplitSentences(result)
	if len(sents) > 0 {
		first := sents[0]
		// Don't hedge questions or already-hedged text
		if !strings.HasSuffix(first, "?") && !startsWithHedge(first) {
			sents[0] = hedgePhrases[t.rng.Intn(len(hedgePhrases))] + " " +
				strings.ToLower(first[:1]) + first[1:]
		}
		result = strings.Join(sents, " ")
	}

	return result
}

// -----------------------------------------------------------------------
// Casualize — add contractions, simplify vocabulary
// -----------------------------------------------------------------------

func (t *TextTransformer) casualize(text string) string {
	// Contract expanded forms
	result := contractExpansions(text)

	// Replace formal words with casual equivalents
	words := strings.Fields(result)
	for i, w := range words {
		clean := strings.ToLower(stripPunctuation(w))
		if casual, ok := formalToCasual[clean]; ok {
			suffix := trailingPunctuation(w)
			replacement := casual
			if len(w) > 0 && unicode.IsUpper(rune(w[0])) {
				replacement = transformCapFirst(replacement)
			}
			words[i] = replacement + suffix
		}
	}
	result = strings.Join(words, " ")

	// Remove hedging phrases
	for _, hedge := range hedgePhrases {
		lower := strings.ToLower(result)
		hedgeLower := strings.ToLower(hedge)
		if idx := strings.Index(lower, hedgeLower); idx >= 0 {
			after := result[idx+len(hedge):]
			after = strings.TrimLeft(after, " ,")
			if len(after) > 0 {
				after = transformCapFirst(after)
			}
			result = result[:idx] + after
		}
	}

	return strings.TrimSpace(result)
}

// -----------------------------------------------------------------------
// Bulletize — split text into bullet points
// -----------------------------------------------------------------------

func (t *TextTransformer) bulletize(text string) string {
	sents := transformSplitSentences(text)
	if len(sents) == 0 {
		return text
	}

	var bullets []string
	for _, sent := range sents {
		sent = strings.TrimSpace(sent)
		if sent == "" {
			continue
		}
		// Remove trailing period for cleaner bullets
		sent = strings.TrimRight(sent, ".")
		bullets = append(bullets, "- "+sent)
	}
	return strings.Join(bullets, "\n")
}

// -----------------------------------------------------------------------
// Prosify — merge bullet points into flowing prose
// -----------------------------------------------------------------------

func (t *TextTransformer) prosify(text string) string {
	lines := strings.Split(text, "\n")
	var items []string
	for _, line := range lines {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		// Strip bullet markers: "- ", "* ", "• ", "1. ", "1) ", etc.
		line = transformBulletRe.ReplaceAllString(line, "")
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		items = append(items, ensureTransformPeriod(line))
	}

	if len(items) == 0 {
		return text
	}

	// Join with transition words
	var parts []string
	for i, item := range items {
		if i == 0 {
			parts = append(parts, item)
			continue
		}
		// Pick a transition word
		transition := transitionWords[i%len(transitionWords)]
		// Lowercase the item's first letter after the transition
		itemLower := strings.ToLower(item[:1]) + item[1:]
		parts = append(parts, transition+" "+itemLower)
	}
	return strings.Join(parts, " ")
}

// -----------------------------------------------------------------------
// Simplify — remove filler words, shorten sentences
// -----------------------------------------------------------------------

func (t *TextTransformer) simplify(text string) string {
	// Remove filler words
	words := strings.Fields(text)
	var filtered []string
	for i, w := range words {
		clean := strings.ToLower(stripPunctuation(w))
		if fillerWords[clean] {
			// If this filler starts a sentence (after period), skip it
			// but also capitalize the next word
			if i+1 < len(words) && (i == 0 || transformEndsWithPunct(words[i-1])) {
				// skip filler, capitalize next word
				continue
			}
			continue
		}
		filtered = append(filtered, w)
	}

	if len(filtered) == 0 {
		return text
	}

	result := strings.Join(filtered, " ")

	// Clean up double spaces
	for strings.Contains(result, "  ") {
		result = strings.ReplaceAll(result, "  ", " ")
	}

	// Ensure proper capitalization after sentence boundaries
	result = fixCapitalization(result)

	// Expand contractions for clarity
	result = expandContractions(result)

	return strings.TrimSpace(result)
}

// -----------------------------------------------------------------------
// Sentence splitting — handles abbreviations
// -----------------------------------------------------------------------

var transformAbbreviations = map[string]bool{
	"dr": true, "mr": true, "mrs": true, "ms": true, "prof": true,
	"sr": true, "jr": true, "st": true, "ave": true, "blvd": true,
	"vs": true, "etc": true, "inc": true, "ltd": true, "corp": true,
	"dept": true, "univ": true, "assn": true, "bros": true, "co": true,
	"no": true, "vol": true, "rev": true, "gen": true, "gov": true,
	"sgt": true, "cpl": true, "pvt": true, "capt": true, "lt": true,
	"cmdr": true, "adm": true, "fig": true, "figs": true,
}

var transformSentEndRe = regexp.MustCompile(`([.!?]+)\s+`)
var transformBulletRe = regexp.MustCompile(`^[-*•]\s+|^\d+[.)]\s+`)

func transformSplitSentences(text string) []string {
	// Normalize whitespace
	text = strings.TrimSpace(text)
	if text == "" {
		return nil
	}

	// Split on sentence boundaries, handling abbreviations
	var sentences []string
	var current strings.Builder

	// Use a simple state machine
	words := strings.Fields(text)
	for i, w := range words {
		if current.Len() > 0 {
			current.WriteString(" ")
		}
		current.WriteString(w)

		// Check if this word ends a sentence
		if transformEndsWithPunct(w) {
			// Check if it's an abbreviation
			clean := strings.ToLower(stripPunctuation(w))
			if transformAbbreviations[clean] && strings.HasSuffix(w, ".") {
				// It's an abbreviation, don't split
				continue
			}
			// Check for "U.S." style abbreviations
			if isAcronymWithDots(w) {
				continue
			}

			// It's a sentence boundary
			sent := strings.TrimSpace(current.String())
			if sent != "" {
				sentences = append(sentences, sent)
			}
			current.Reset()
		}

		// Also split on newlines that are followed by text
		_ = i
	}

	// Don't forget the last sentence
	if current.Len() > 0 {
		sent := strings.TrimSpace(current.String())
		if sent != "" {
			sentences = append(sentences, sent)
		}
	}

	return sentences
}

func isAcronymWithDots(w string) bool {
	// Matches patterns like "U.S." or "U.S.A."
	if len(w) < 3 {
		return false
	}
	dotCount := 0
	letterCount := 0
	for _, r := range w {
		if r == '.' {
			dotCount++
		} else if unicode.IsLetter(r) {
			letterCount++
		} else {
			return false
		}
	}
	return dotCount >= 2 && letterCount >= 2 && dotCount >= letterCount
}

func transformEndsWithPunct(w string) bool {
	if len(w) == 0 {
		return false
	}
	last := w[len(w)-1]
	return last == '.' || last == '!' || last == '?'
}

func stripPunctuation(w string) string {
	return strings.TrimFunc(w, func(r rune) bool {
		return !unicode.IsLetter(r) && !unicode.IsDigit(r)
	})
}

func trailingPunctuation(w string) string {
	i := len(w)
	for i > 0 {
		r := rune(w[i-1])
		if unicode.IsLetter(r) || unicode.IsDigit(r) {
			break
		}
		i--
	}
	return w[i:]
}

func transformCapFirst(s string) string {
	if s == "" {
		return s
	}
	runes := []rune(s)
	runes[0] = unicode.ToUpper(runes[0])
	return string(runes)
}

func ensureTransformPeriod(s string) string {
	s = strings.TrimSpace(s)
	if s == "" {
		return s
	}
	last := s[len(s)-1]
	if last != '.' && last != '!' && last != '?' {
		return s + "."
	}
	return s
}

func fixCapitalization(text string) string {
	runes := []rune(text)
	capitalizeNext := true
	for i, r := range runes {
		if capitalizeNext && unicode.IsLetter(r) {
			runes[i] = unicode.ToUpper(r)
			capitalizeNext = false
		}
		if r == '.' || r == '!' || r == '?' {
			capitalizeNext = true
		}
	}
	return string(runes)
}

func startsWithHedge(s string) bool {
	lower := strings.ToLower(s)
	for _, h := range hedgePhrases {
		if strings.HasPrefix(lower, strings.ToLower(h)) {
			return true
		}
	}
	return false
}

// -----------------------------------------------------------------------
// Contraction maps
// -----------------------------------------------------------------------

// expandContractions replaces contractions with their full forms.
func expandContractions(text string) string {
	for contraction, expansion := range contractionExpansions {
		// Case-insensitive replacement
		re := regexp.MustCompile(`(?i)\b` + regexp.QuoteMeta(contraction) + `\b`)
		text = re.ReplaceAllStringFunc(text, func(match string) string {
			if unicode.IsUpper(rune(match[0])) {
				return transformCapFirst(expansion)
			}
			return expansion
		})
	}
	return text
}

// contractExpansions replaces full forms with contractions.
func contractExpansions(text string) string {
	for contraction, expansion := range contractionExpansions {
		// Case-insensitive replacement of expansion → contraction
		re := regexp.MustCompile(`(?i)\b` + regexp.QuoteMeta(expansion) + `\b`)
		text = re.ReplaceAllStringFunc(text, func(match string) string {
			if unicode.IsUpper(rune(match[0])) {
				return transformCapFirst(contraction)
			}
			return contraction
		})
	}
	return text
}

var contractionExpansions = map[string]string{
	"I'm":      "I am",
	"I've":     "I have",
	"I'll":     "I will",
	"I'd":      "I would",
	"you're":   "you are",
	"you've":   "you have",
	"you'll":   "you will",
	"you'd":    "you would",
	"he's":     "he is",
	"she's":    "she is",
	"it's":     "it is",
	"we're":    "we are",
	"we've":    "we have",
	"we'll":    "we will",
	"we'd":     "we would",
	"they're":  "they are",
	"they've":  "they have",
	"they'll":  "they will",
	"they'd":   "they would",
	"that's":   "that is",
	"who's":    "who is",
	"what's":   "what is",
	"here's":   "here is",
	"there's":  "there is",
	"where's":  "where is",
	"when's":   "when is",
	"why's":    "why is",
	"how's":    "how is",
	"isn't":    "is not",
	"aren't":   "are not",
	"wasn't":   "was not",
	"weren't":  "were not",
	"hasn't":   "has not",
	"haven't":  "have not",
	"hadn't":   "had not",
	"won't":    "will not",
	"wouldn't": "would not",
	"don't":    "do not",
	"doesn't":  "does not",
	"didn't":   "did not",
	"can't":    "cannot",
	"couldn't": "could not",
	"shouldn't": "should not",
	"mightn't": "might not",
	"mustn't":  "must not",
	"let's":    "let us",
	"ain't":    "am not",
}

// -----------------------------------------------------------------------
// Formality word maps
// -----------------------------------------------------------------------

var casualToFormal = map[string]string{
	"big":       "substantial",
	"small":     "modest",
	"good":      "favorable",
	"bad":       "unfavorable",
	"nice":      "pleasant",
	"pretty":    "rather",
	"lots":      "numerous",
	"stuff":     "material",
	"things":    "items",
	"get":       "obtain",
	"got":       "obtained",
	"give":      "provide",
	"show":      "demonstrate",
	"help":      "assist",
	"try":       "attempt",
	"use":       "utilize",
	"need":      "require",
	"want":      "desire",
	"think":     "consider",
	"guess":     "estimate",
	"fix":       "rectify",
	"start":     "commence",
	"end":       "conclude",
	"begin":     "commence",
	"buy":       "purchase",
	"pick":      "select",
	"ask":       "inquire",
	"tell":      "inform",
	"find":      "locate",
	"keep":      "retain",
	"seem":      "appear",
	"look":      "examine",
	"kids":      "children",
	"guy":       "individual",
	"guys":      "individuals",
	"cool":      "acceptable",
	"okay":      "satisfactory",
	"ok":        "satisfactory",
	"yeah":      "yes",
	"yep":       "yes",
	"nope":      "no",
	"nah":       "no",
	"gonna":     "going to",
	"wanna":     "want to",
	"gotta":     "have to",
	"kinda":     "somewhat",
	"sorta":     "somewhat",
	"maybe":     "perhaps",
	"like":      "such as",
	"basically": "fundamentally",
	"actually":  "in fact",
	"really":    "truly",
	"very":      "exceedingly",
	"so":        "therefore",
	"but":       "however",
	"also":      "additionally",
	"plus":      "furthermore",
	"about":     "approximately",
	"around":    "approximately",
	"almost":    "nearly",
	"anyway":    "regardless",
	"anyways":   "regardless",
	"a lot":     "considerably",
	"lots of":   "numerous",
}

var formalToCasual = map[string]string{
	"substantial":   "big",
	"modest":        "small",
	"favorable":     "good",
	"unfavorable":   "bad",
	"pleasant":      "nice",
	"numerous":      "lots of",
	"obtain":        "get",
	"obtained":      "got",
	"provide":       "give",
	"demonstrate":   "show",
	"assist":        "help",
	"attempt":       "try",
	"utilize":       "use",
	"require":       "need",
	"desire":        "want",
	"consider":      "think",
	"estimate":      "guess",
	"rectify":       "fix",
	"commence":      "start",
	"conclude":      "end",
	"purchase":      "buy",
	"select":        "pick",
	"inquire":       "ask",
	"inform":        "tell",
	"locate":        "find",
	"retain":        "keep",
	"examine":       "look at",
	"individuals":   "people",
	"satisfactory":  "okay",
	"perhaps":       "maybe",
	"therefore":     "so",
	"however":       "but",
	"additionally":  "also",
	"furthermore":   "plus",
	"approximately": "about",
	"nearly":        "almost",
	"regardless":    "anyway",
	"considerably":  "a lot",
	"exceedingly":   "really",
	"fundamentally": "basically",
	"subsequent":    "next",
	"prior":         "before",
	"sufficient":    "enough",
	"insufficient":  "not enough",
	"acquire":       "get",
	"endeavor":      "try",
	"facilitate":    "help",
	"implement":     "do",
	"initiate":      "start",
	"terminate":     "stop",
	"transmit":      "send",
	"optimal":       "best",
	"compensate":    "pay",
	"communicate":   "talk",
	"construct":     "build",
	"diminish":      "shrink",
	"eliminate":     "remove",
	"enumerate":     "list",
	"establish":     "set up",
	"expedite":      "speed up",
	"necessitate":   "need",
	"ascertain":     "find out",
	"deliberate":    "think about",
}

// -----------------------------------------------------------------------
// Hedging phrases (for formalize)
// -----------------------------------------------------------------------

var hedgePhrases = []string{
	"It would appear that",
	"It is worth noting that",
	"One could argue that",
	"It should be noted that",
	"It is generally understood that",
	"Evidence suggests that",
	"It may be observed that",
	"It is reasonable to suggest that",
	"Research indicates that",
	"It can be contended that",
}

// -----------------------------------------------------------------------
// Transition words (for prosify)
// -----------------------------------------------------------------------

var transitionWords = []string{
	"Furthermore,",
	"In addition,",
	"Moreover,",
	"Additionally,",
	"Also,",
	"Besides,",
	"Likewise,",
	"Similarly,",
	"On the other hand,",
	"However,",
	"Nevertheless,",
	"Meanwhile,",
	"Subsequently,",
	"Consequently,",
	"As a result,",
	"In particular,",
	"Specifically,",
	"Notably,",
	"Importantly,",
	"Interestingly,",
	"In contrast,",
	"Alternatively,",
	"Rather,",
	"Indeed,",
	"Certainly,",
	"Undoubtedly,",
	"Evidently,",
	"Clearly,",
	"Significantly,",
	"Essentially,",
}

// -----------------------------------------------------------------------
// Filler words (for simplify)
// -----------------------------------------------------------------------

var fillerWords = map[string]bool{
	"actually":      true,
	"basically":     true,
	"certainly":     true,
	"clearly":       true,
	"definitely":    true,
	"essentially":   true,
	"evidently":     true,
	"extremely":     true,
	"fairly":        true,
	"frankly":       true,
	"generally":     true,
	"honestly":      true,
	"hopefully":     true,
	"indeed":        true,
	"interestingly": true,
	"just":          true,
	"largely":       true,
	"literally":     true,
	"mainly":        true,
	"merely":        true,
	"mostly":        true,
	"naturally":     true,
	"necessarily":   true,
	"normally":      true,
	"obviously":     true,
	"overall":       true,
	"particularly":  true,
	"perhaps":       true,
	"possibly":      true,
	"practically":   true,
	"presumably":    true,
	"probably":      true,
	"quite":         true,
	"rather":        true,
	"really":        true,
	"relatively":    true,
	"seemingly":     true,
	"seriously":     true,
	"significantly": true,
	"simply":        true,
	"slightly":      true,
	"somewhat":      true,
	"supposedly":    true,
	"surely":        true,
	"technically":   true,
	"typically":     true,
	"undoubtedly":   true,
	"unfortunately": true,
	"utterly":       true,
	"very":          true,
}

// -----------------------------------------------------------------------
// Stop words for TF-IDF (summarization)
// -----------------------------------------------------------------------

var transformStopWords = map[string]bool{
	"the": true, "a": true, "an": true, "is": true, "are": true,
	"was": true, "were": true, "be": true, "been": true, "being": true,
	"have": true, "has": true, "had": true, "do": true, "does": true,
	"did": true, "will": true, "would": true, "could": true, "should": true,
	"may": true, "might": true, "can": true, "shall": true, "must": true,
	"of": true, "in": true, "to": true, "for": true, "with": true,
	"on": true, "at": true, "by": true, "from": true, "up": true,
	"about": true, "into": true, "through": true, "during": true,
	"before": true, "after": true, "above": true, "below": true,
	"and": true, "but": true, "or": true, "not": true, "no": true,
	"so": true, "if": true, "then": true, "than": true, "too": true,
	"very": true, "just": true, "also": true, "that": true, "this": true,
	"these": true, "those": true, "it": true, "its": true,
	"i": true, "me": true, "my": true, "you": true, "your": true,
	"he": true, "she": true, "we": true, "they": true, "them": true,
	"their": true, "our": true, "his": true, "her": true,
	"what": true, "which": true, "who": true, "whom": true,
	"where": true, "when": true, "how": true, "why": true,
	"am": true, "as": true,
}

// -----------------------------------------------------------------------
// Synonym map — ~250 common words with synonyms, grouped loosely by register
// -----------------------------------------------------------------------

var transformSynonyms = map[string][]string{
	// Adjectives
	"good":        {"great", "fine", "excellent", "solid", "strong"},
	"bad":         {"poor", "weak", "unfortunate", "unfavorable", "negative"},
	"big":         {"large", "substantial", "significant", "considerable", "major"},
	"small":       {"little", "minor", "modest", "slight", "compact"},
	"important":   {"significant", "crucial", "essential", "vital", "critical"},
	"new":         {"novel", "fresh", "recent", "modern", "current"},
	"old":         {"ancient", "aged", "historic", "traditional", "established"},
	"fast":        {"quick", "rapid", "swift", "speedy", "prompt"},
	"slow":        {"gradual", "unhurried", "measured", "steady", "deliberate"},
	"hard":        {"difficult", "challenging", "tough", "demanding", "arduous"},
	"easy":        {"simple", "straightforward", "effortless", "uncomplicated"},
	"happy":       {"pleased", "content", "joyful", "delighted", "glad"},
	"sad":         {"unhappy", "sorrowful", "melancholy", "gloomy", "downcast"},
	"beautiful":   {"lovely", "attractive", "elegant", "stunning", "gorgeous"},
	"ugly":        {"unattractive", "unsightly", "plain", "homely"},
	"strong":      {"powerful", "robust", "sturdy", "resilient", "mighty"},
	"weak":        {"feeble", "fragile", "frail", "delicate", "vulnerable"},
	"rich":        {"wealthy", "affluent", "prosperous", "well-off"},
	"poor":        {"impoverished", "destitute", "needy", "disadvantaged"},
	"long":        {"extended", "lengthy", "prolonged", "extensive"},
	"short":       {"brief", "concise", "compact", "abbreviated"},
	"high":        {"elevated", "tall", "lofty", "towering"},
	"low":         {"minimal", "reduced", "diminished", "modest"},
	"true":        {"accurate", "correct", "valid", "genuine", "authentic"},
	"false":       {"incorrect", "wrong", "inaccurate", "untrue", "invalid"},
	"clear":       {"obvious", "evident", "apparent", "transparent", "plain"},
	"complex":     {"complicated", "intricate", "sophisticated", "elaborate"},
	"simple":      {"basic", "straightforward", "elementary", "uncomplicated"},
	"different":   {"distinct", "various", "diverse", "varied", "alternative"},
	"similar":     {"alike", "comparable", "analogous", "related", "akin"},
	"certain":     {"definite", "sure", "confident", "assured", "positive"},
	"possible":    {"feasible", "viable", "achievable", "potential", "likely"},
	"necessary":   {"essential", "required", "needed", "vital", "indispensable"},
	"useful":      {"helpful", "practical", "valuable", "beneficial", "handy"},
	"dangerous":   {"hazardous", "risky", "perilous", "unsafe", "threatening"},
	"safe":        {"secure", "protected", "sheltered", "guarded"},
	"quiet":       {"silent", "hushed", "still", "peaceful", "calm"},
	"loud":        {"noisy", "boisterous", "thunderous", "deafening"},
	"dark":        {"dim", "shadowy", "murky", "gloomy", "obscure"},
	"bright":      {"brilliant", "radiant", "luminous", "vivid", "dazzling"},
	"hot":         {"warm", "scorching", "sweltering", "blazing"},
	"cold":        {"chilly", "frigid", "freezing", "icy", "frosty"},
	"full":        {"complete", "entire", "whole", "total", "packed"},
	"empty":       {"vacant", "hollow", "bare", "void", "blank"},
	"clean":       {"spotless", "pristine", "immaculate", "pure", "tidy"},
	"dirty":       {"filthy", "grimy", "soiled", "stained", "unclean"},
	"sharp":       {"keen", "acute", "pointed", "precise", "incisive"},
	"dull":        {"boring", "tedious", "monotonous", "bland", "dreary"},
	"wide":        {"broad", "expansive", "extensive", "spacious", "vast"},
	"narrow":      {"thin", "slender", "slim", "restricted", "limited"},
	"deep":        {"profound", "thorough", "extensive", "intense"},
	"thin":        {"slender", "slim", "lean", "slight", "fine"},
	"thick":       {"dense", "heavy", "solid", "substantial"},
	"young":       {"youthful", "juvenile", "fresh", "new"},
	"ancient":     {"old", "aged", "antique", "archaic", "venerable"},
	"modern":      {"contemporary", "current", "present-day", "up-to-date"},
	"specific":    {"particular", "precise", "exact", "definite", "concrete"},
	"general":     {"broad", "overall", "universal", "widespread", "common"},
	"natural":     {"organic", "innate", "inherent", "native"},
	"common":      {"frequent", "widespread", "prevalent", "ordinary", "usual"},
	"rare":        {"uncommon", "unusual", "scarce", "exceptional", "unique"},
	"famous":      {"renowned", "celebrated", "notable", "prominent", "distinguished"},
	"popular":     {"well-known", "widespread", "fashionable", "favored"},
	"strange":     {"odd", "unusual", "peculiar", "curious", "bizarre"},
	"normal":      {"typical", "standard", "regular", "ordinary", "usual"},
	"perfect":     {"ideal", "flawless", "impeccable", "exemplary"},
	"terrible":    {"awful", "dreadful", "horrible", "atrocious", "abysmal"},
	"wonderful":   {"marvelous", "splendid", "magnificent", "superb", "fantastic"},
	"interesting": {"fascinating", "intriguing", "compelling", "engaging", "captivating"},

	// Verbs
	"say":        {"state", "declare", "mention", "express", "articulate"},
	"make":       {"create", "produce", "construct", "build", "craft"},
	"go":         {"proceed", "travel", "move", "advance", "head"},
	"take":       {"acquire", "grab", "seize", "obtain", "secure"},
	"come":       {"arrive", "approach", "reach", "appear"},
	"see":        {"observe", "notice", "perceive", "witness", "view"},
	"know":       {"understand", "comprehend", "recognize", "realize"},
	"get":        {"obtain", "acquire", "receive", "gain", "secure"},
	"give":       {"provide", "offer", "supply", "deliver", "present"},
	"find":       {"discover", "locate", "uncover", "identify", "detect"},
	"think":      {"believe", "consider", "ponder", "reflect", "contemplate"},
	"tell":       {"inform", "notify", "advise", "communicate", "relay"},
	"ask":        {"inquire", "question", "request", "query"},
	"work":       {"function", "operate", "perform", "labor", "toil"},
	"call":       {"contact", "summon", "refer to", "name", "designate"},
	"try":        {"attempt", "endeavor", "strive", "aim"},
	"need":       {"require", "demand", "necessitate"},
	"keep":       {"maintain", "preserve", "retain", "sustain", "hold"},
	"start":      {"begin", "commence", "initiate", "launch", "embark"},
	"show":       {"display", "demonstrate", "reveal", "exhibit", "present"},
	"hear":       {"listen", "perceive", "detect", "catch"},
	"run":        {"operate", "manage", "execute", "sprint", "dash"},
	"move":       {"relocate", "shift", "transfer", "transport"},
	"live":       {"reside", "dwell", "inhabit", "exist"},
	"believe":    {"think", "consider", "hold", "maintain", "trust"},
	"bring":      {"deliver", "carry", "transport", "convey"},
	"happen":     {"occur", "transpire", "take place", "arise"},
	"write":      {"compose", "author", "draft", "pen", "record"},
	"sit":        {"remain", "rest", "settle", "stay"},
	"stand":      {"rise", "endure", "withstand", "tolerate"},
	"lose":       {"misplace", "forfeit", "surrender"},
	"pay":        {"compensate", "reimburse", "remunerate"},
	"meet":       {"encounter", "confront", "assemble", "gather"},
	"play":       {"perform", "engage in", "participate"},
	"learn":      {"study", "master", "absorb", "grasp", "acquire"},
	"change":     {"alter", "modify", "transform", "adjust", "revise"},
	"help":       {"assist", "aid", "support", "facilitate"},
	"talk":       {"speak", "converse", "discuss", "communicate", "chat"},
	"turn":       {"rotate", "revolve", "spin", "shift", "pivot"},
	"hold":       {"grasp", "grip", "clutch", "possess", "retain"},
	"put":        {"place", "position", "set", "locate", "deposit"},
	"end":        {"conclude", "finish", "terminate", "complete", "cease"},
	"stop":       {"halt", "cease", "discontinue", "pause", "suspend"},
	"build":      {"construct", "erect", "assemble", "create", "establish"},
	"break":      {"shatter", "fracture", "crack", "damage", "rupture"},
	"grow":       {"expand", "increase", "develop", "flourish", "thrive"},
	"cut":        {"reduce", "trim", "slice", "diminish"},
	"kill":       {"eliminate", "destroy", "eradicate", "abolish"},
	"reach":      {"achieve", "attain", "arrive at", "accomplish"},
	"remain":     {"stay", "persist", "endure", "continue", "linger"},
	"suggest":    {"propose", "recommend", "advise", "indicate"},
	"raise":      {"lift", "elevate", "increase", "boost", "heighten"},
	"pass":       {"exceed", "surpass", "go beyond", "overtake"},
	"sell":       {"market", "trade", "vend", "retail"},
	"decide":     {"determine", "resolve", "conclude", "settle"},
	"return":     {"come back", "restore", "revert", "go back"},
	"explain":    {"clarify", "describe", "illustrate", "elucidate"},
	"hope":       {"wish", "aspire", "expect", "anticipate"},
	"develop":    {"create", "evolve", "expand", "advance", "progress"},
	"carry":      {"transport", "convey", "bear", "haul"},
	"produce":    {"generate", "create", "manufacture", "yield"},
	"eat":        {"consume", "dine on", "devour", "ingest"},
	"join":       {"connect", "unite", "merge", "combine", "link"},
	"spend":      {"expend", "invest", "allocate", "devote"},
	"choose":     {"select", "pick", "opt for", "decide on"},
	"deal":       {"handle", "manage", "cope with", "address"},
	"increase":   {"boost", "raise", "enhance", "expand", "augment"},
	"decrease":   {"reduce", "lower", "diminish", "lessen", "cut"},
	"improve":    {"enhance", "upgrade", "refine", "advance", "better"},
	"create":     {"produce", "generate", "design", "develop", "form"},
	"support":    {"assist", "back", "uphold", "sustain", "endorse"},
	"continue":   {"persist", "proceed", "carry on", "maintain", "resume"},

	// Nouns
	"idea":       {"concept", "notion", "thought", "plan", "proposal"},
	"problem":    {"issue", "challenge", "difficulty", "obstacle", "complication"},
	"answer":     {"response", "reply", "solution", "resolution"},
	"question":   {"query", "inquiry", "issue", "matter"},
	"way":        {"method", "approach", "manner", "technique", "path"},
	"part":       {"section", "portion", "segment", "component", "element"},
	"place":      {"location", "site", "area", "spot", "position"},
	"case":       {"instance", "situation", "example", "scenario"},
	"point":      {"aspect", "matter", "detail", "issue", "factor"},
	"group":      {"collection", "set", "cluster", "team", "assembly"},
	"area":       {"region", "zone", "sector", "domain", "field"},
	"result":     {"outcome", "consequence", "effect", "product"},
	"reason":     {"cause", "motive", "basis", "rationale", "grounds"},
	"fact":       {"truth", "reality", "detail", "datum", "evidence"},
	"type":       {"kind", "sort", "category", "class", "variety"},
	"level":      {"degree", "extent", "stage", "tier", "rank"},
	"effect":     {"impact", "influence", "result", "consequence"},
	"use":        {"application", "purpose", "function", "utility"},
	"power":      {"strength", "force", "authority", "influence", "might"},
	"system":     {"framework", "structure", "arrangement", "mechanism"},
	"program":    {"plan", "scheme", "initiative", "project"},
	"view":       {"perspective", "opinion", "outlook", "stance"},
	"field":      {"domain", "area", "discipline", "sector", "realm"},
	"base":       {"foundation", "basis", "ground", "core"},
	"goal":       {"objective", "target", "aim", "purpose", "ambition"},
	"study":      {"research", "investigation", "analysis", "examination"},
	"job":        {"task", "role", "position", "occupation", "duty"},
	"money":      {"funds", "capital", "resources", "finances"},
	"effort":     {"endeavor", "attempt", "exertion", "undertaking"},
	"chance":     {"opportunity", "possibility", "prospect", "likelihood"},
	"fear":       {"anxiety", "dread", "apprehension", "concern"},
	"love":       {"affection", "devotion", "passion", "fondness"},

	// Adverbs
	"quickly":    {"rapidly", "swiftly", "promptly", "speedily"},
	"slowly":     {"gradually", "steadily", "unhurriedly", "leisurely"},
	"often":      {"frequently", "regularly", "commonly", "repeatedly"},
	"usually":    {"typically", "generally", "normally", "ordinarily"},
	"always":     {"invariably", "constantly", "perpetually", "continuously"},
	"never":      {"not once", "at no time"},
	"already":    {"previously", "by now", "at this point"},
	"again":      {"once more", "anew", "afresh"},
	"soon":       {"shortly", "before long", "in due course", "presently"},
	"together":   {"collectively", "jointly", "in unison", "as one"},
	"sometimes":  {"occasionally", "now and then", "at times", "periodically"},
	"especially": {"particularly", "notably", "specifically", "chiefly"},
	"completely": {"entirely", "fully", "thoroughly", "wholly", "totally"},
	"exactly":    {"precisely", "accurately", "specifically"},
	"finally":    {"ultimately", "at last", "eventually", "in the end"},
}
