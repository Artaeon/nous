package cognitive

import (
	"regexp"
	"sort"
	"strings"
	"unicode"
)

// -----------------------------------------------------------------------
// TextTransformEngine — deterministic text transformations.
//
// Formal, casual, simple, short, bullet — pure rule-based linguistic
// transforms.  No ML, no LLM.  Instant, zero-cost writing assistance.
// -----------------------------------------------------------------------

// TextTransformEngine provides deterministic text transformations.
// No ML, no LLM — pure rule-based linguistic transforms.
type TextTransformEngine struct {
	formalMap map[string]string // casual -> formal
	casualMap map[string]string // formal -> casual
	simpleMap map[string]string // complex -> simple
}

// NewTextTransformEngine initializes the engine with comprehensive word
// replacement maps covering 60+ casual->formal, 40+ complex->simple, and
// the reverse casual map.
func NewTextTransformEngine() *TextTransformEngine {
	e := &TextTransformEngine{
		formalMap: map[string]string{
			// Greetings / slang
			"hey":  "hello",
			"hi":   "hello",
			"yo":   "hello",
			"sup":  "hello",
			"hiya": "hello",

			// Contractions (handled separately too, but useful for isolated tokens)
			"gonna":    "going to",
			"wanna":    "want to",
			"can't":    "cannot",
			"won't":    "will not",
			"don't":    "do not",
			"shouldn't": "should not",
			"couldn't": "could not",
			"wouldn't": "would not",
			"isn't":    "is not",
			"aren't":   "are not",
			"hasn't":   "has not",
			"haven't":  "have not",
			"didn't":   "did not",
			"doesn't":  "does not",
			"weren't":  "were not",
			"wasn't":   "was not",

			// Internet abbreviations
			"asap": "at your earliest convenience",
			"fyi":  "for your information",
			"btw":  "by the way",
			"imo":  "in my opinion",
			"imho": "in my humble opinion",
			"tbh":  "to be honest",
			"afaik": "as far as I know",
			"iirc": "if I recall correctly",

			// Shorthand
			"u":   "you",
			"ur":  "your",
			"r":   "are",
			"pls": "please",
			"plz": "please",
			"thx": "thank you",
			"ty":  "thank you",
			"np":  "no problem",
			"tho": "though",
			"cuz": "because",

			// Affirmatives / negatives
			"ok":    "certainly",
			"okay":  "certainly",
			"yeah":  "yes",
			"yep":   "yes",
			"yup":   "yes",
			"nah":   "no",
			"nope":  "no",
			"sure":  "certainly",

			// Degree adverbs
			"kinda":  "somewhat",
			"sorta":  "somewhat",
			"really": "very",
			"super":  "extremely",
			"pretty": "quite",

			// Informal verbs / nouns
			"gotta":  "have to",
			"lemme":  "let me",
			"gimme":  "give me",
			"stuff":  "materials",
			"things": "items",

			// Positive adjectives
			"awesome": "excellent",
			"cool":    "satisfactory",
			"great":   "excellent",
			"nice":    "pleasant",
			"amazing": "remarkable",

			// Negative adjectives
			"bad":     "unsatisfactory",
			"lousy":   "unsatisfactory",
			"crappy":  "unsatisfactory",

			// Informal address
			"dude":  "colleague",
			"bro":   "colleague",
			"guys":  "individuals",
			"guy":   "individual",
			"kids":  "children",
			"buddy": "colleague",
			"pal":   "colleague",

			// Size
			"huge": "substantial",
			"tiny": "minimal",
			"big":  "significant",

			// Quality
			"good": "satisfactory",

			// Common verbs
			"get":   "obtain",
			"got":   "obtained",
			"fix":   "resolve",
			"check": "verify",
			"send":  "transmit",
			"buy":   "purchase",
			"use":   "utilize",
			"need":  "require",
			"help":  "assist",
			"start": "commence",
			"end":   "conclude",
			"show":  "demonstrate",
			"try":   "attempt",
			"ask":   "inquire",
			"tell":  "inform",
			"find":  "locate",
			"keep":  "retain",
			"pick":  "select",
			"want":  "desire",
			"think": "consider",

			// Connectors / adverbs
			"also":      "additionally",
			"but":       "however",
			"so":        "therefore",
			"right now": "immediately",
			"soon":      "shortly",
			"later":     "subsequently",
		},

		simpleMap: map[string]string{
			// Corporate jargon
			"utilize":     "use",
			"implement":   "do",
			"facilitate":  "help",
			"demonstrate": "show",
			"endeavor":    "try",
			"commence":    "start",
			"terminate":   "end",
			"ascertain":   "find out",
			"ameliorate":  "improve",
			"leverage":    "use",
			"optimize":    "improve",
			"streamline":  "simplify",
			"synergy":     "teamwork",
			"paradigm":    "model",
			"robust":      "strong",
			"scalable":    "growable",
			"granular":    "detailed",
			"holistic":    "complete",
			"innovative":  "new",
			"disruptive":  "game-changing",
			"ecosystem":   "system",
			"stakeholder": "person involved",
			"bandwidth":   "time",
			"pivot":       "change direction",

			// Buzzphrases
			"ubiquitous": "everywhere",

			// Academic / formal
			"epistemic":      "knowledge-related",
			"ontological":    "existence-related",
			"ramification":    "effect",
			"ramifications":   "effects",
			"juxtaposition":  "contrast",
			"quintessential": "perfect example of",
			"nevertheless":   "still",
			"notwithstanding": "despite",
			"aforementioned": "previously mentioned",
			"henceforth":     "from now on",
			"wherein":        "where",
			"thereby":        "by doing this",
			"heretofore":     "until now",
			"inasmuch":       "since",
			"insofar":        "to the extent that",
			"ergo":           "so",
			"vis-a-vis":      "compared to",
			"de facto":       "in practice",
			"per se":         "by itself",
			"myriad":         "many",
			"plethora":       "lots",
			"conundrum":      "puzzle",
			"dichotomy":      "split",
		},
	}

	// Build casualMap as the reverse of formalMap.
	e.casualMap = make(map[string]string, len(e.formalMap))
	for casual, formal := range e.formalMap {
		// Only reverse single-word formals to avoid messy multi-word keys.
		if !strings.Contains(formal, " ") {
			e.casualMap[formal] = casual
		}
	}
	// Add explicit formal->casual overrides that read more naturally than
	// a blind reversal.
	for k, v := range map[string]string{
		"cannot":        "can't",
		"will not":      "won't",
		"do not":        "don't",
		"should not":    "shouldn't",
		"could not":     "couldn't",
		"would not":     "wouldn't",
		"is not":        "isn't",
		"are not":       "aren't",
		"has not":       "hasn't",
		"have not":      "haven't",
		"did not":       "didn't",
		"does not":      "doesn't",
		"were not":      "weren't",
		"was not":       "wasn't",
		"obtain":        "get",
		"resolve":       "fix",
		"verify":        "check",
		"transmit":      "send",
		"purchase":      "buy",
		"utilize":       "use",
		"require":       "need",
		"assist":        "help",
		"commence":      "start",
		"conclude":      "end",
		"demonstrate":   "show",
		"attempt":       "try",
		"inquire":       "ask",
		"inform":        "tell",
		"locate":        "find",
		"retain":        "keep",
		"select":        "pick",
		"desire":        "want",
		"consider":      "think",
		"additionally":  "also",
		"however":       "but",
		"therefore":     "so",
		"immediately":   "right now",
		"subsequently":  "later",
		"shortly":       "soon",
		"certainly":     "sure",
		"satisfactory":  "good",
		"excellent":     "great",
		"remarkable":    "amazing",
		"pleasant":      "nice",
		"substantial":   "huge",
		"significant":   "big",
		"minimal":       "tiny",
		"unsatisfactory": "bad",
		"extremely":     "super",
		"quite":         "pretty",
	} {
		e.casualMap[k] = v
	}

	return e
}

// Transform routes to the appropriate transformation by mode name.
func (t *TextTransformEngine) Transform(text string, mode string) string {
	switch strings.ToLower(mode) {
	case "formal":
		return t.Formalize(text)
	case "casual":
		return t.Casualize(text)
	case "simple":
		return t.Simplify(text)
	case "short":
		return t.Shorten(text)
	case "bullet", "bullets":
		return t.ToBullets(text)
	default:
		return text
	}
}

// -----------------------------------------------------------------------
// Formalize
// -----------------------------------------------------------------------

// Formalize makes text more formal: expands contractions, replaces casual
// words, normalizes punctuation, and ensures proper capitalization.
func (t *TextTransformEngine) Formalize(text string) string {
	text = strings.TrimSpace(text)
	if text == "" {
		return ""
	}

	// Multi-word phrase replacement (order matters: longer first).
	text = t.replacePhrasesCI(text, t.formalMap)

	// Single-word replacement.
	text = t.replaceWordsCI(text, t.formalMap)

	// Expand any remaining contractions.
	text = txExpandContractions(text)

	// Normalize punctuation: collapse "!!!" / "???" to ".", replace "!" with ".".
	text = txMultiPunctRe.ReplaceAllString(text, ".")
	text = strings.ReplaceAll(text, "!", ".")

	// Capitalize first letter of each sentence.
	text = txCapitalizeSentences(text)

	// Add period at end if missing punctuation.
	text = strings.TrimSpace(text)
	if len(text) > 0 {
		last := text[len(text)-1]
		if last != '.' && last != '?' && last != '!' {
			text += "."
		}
	}

	return text
}

// -----------------------------------------------------------------------
// Casualize
// -----------------------------------------------------------------------

// Casualize makes text more casual: contracts expansions, replaces formal
// words, and lowercases sentence starts where natural.
func (t *TextTransformEngine) Casualize(text string) string {
	text = strings.TrimSpace(text)
	if text == "" {
		return ""
	}

	// Multi-word phrase replacement (contractions like "do not" -> "don't").
	text = t.replacePhrasesCI(text, t.casualMap)

	// Single-word replacement.
	text = t.replaceWordsCI(text, t.casualMap)

	// Contract remaining expansions.
	text = txContractExpansions(text)

	// Lowercase first word of each sentence unless it's "I" or a proper noun.
	sents := txSplitSentences(text)
	for i, s := range sents {
		s = strings.TrimSpace(s)
		if s == "" {
			continue
		}
		runes := []rune(s)
		// Find first letter.
		for j, r := range runes {
			if unicode.IsLetter(r) {
				word := txFirstWord(s[j:])
				// Keep "I" capitalized, keep words that look like proper nouns
				// (we heuristically skip lowering if the second char is also upper).
				if word == "I" || (len(word) > 1 && unicode.IsUpper([]rune(word)[1])) {
					break
				}
				runes[j] = unicode.ToLower(r)
				break
			}
		}
		sents[i] = string(runes)
	}
	text = strings.Join(sents, " ")

	return strings.TrimSpace(text)
}

// -----------------------------------------------------------------------
// Simplify
// -----------------------------------------------------------------------

// Simplify makes text simpler: replaces complex words, breaks long
// sentences, removes parenthetical asides and subordinate clauses.
func (t *TextTransformEngine) Simplify(text string) string {
	text = strings.TrimSpace(text)
	if text == "" {
		return ""
	}

	// Replace complex words.
	text = t.replacePhrasesCI(text, t.simpleMap)
	text = t.replaceWordsCI(text, t.simpleMap)

	// Remove parenthetical asides: (anything inside parens).
	text = txParenRe.ReplaceAllString(text, "")

	// Remove subordinate clauses starting with certain words.
	// E.g., ", which is ...,", ", whereby ...,", ", notwithstanding ..."
	text = txSubordinateRe.ReplaceAllString(text, "")

	// Clean up double-spaces.
	text = txCollapseSpaces(text)

	// Break long sentences at conjunctions.
	sents := txSplitSentences(text)
	var result []string
	for _, s := range sents {
		s = strings.TrimSpace(s)
		if s == "" {
			continue
		}
		words := strings.Fields(s)
		if len(words) > 30 {
			result = append(result, txBreakLongSentence(s)...)
		} else {
			result = append(result, s)
		}
	}

	// Reassemble, ensuring capitalization and periods.
	var parts []string
	for _, s := range result {
		s = strings.TrimSpace(s)
		if s == "" {
			continue
		}
		s = txCapFirst(s)
		s = txEnsurePeriod(s)
		parts = append(parts, s)
	}

	return strings.Join(parts, " ")
}

// -----------------------------------------------------------------------
// Shorten
// -----------------------------------------------------------------------

// Shorten compresses text by keeping the most important 50% of sentences,
// scored by position (first/last = high), length (medium = best), and
// entity count (capitalized words).
func (t *TextTransformEngine) Shorten(text string) string {
	text = strings.TrimSpace(text)
	if text == "" {
		return ""
	}

	sents := txSplitSentences(text)
	if len(sents) <= 2 {
		return text
	}

	type scored struct {
		text  string
		score float64
		idx   int
	}
	var items []scored
	n := len(sents)

	for i, s := range sents {
		s = strings.TrimSpace(s)
		if s == "" {
			continue
		}
		words := strings.Fields(s)
		wc := len(words)
		var score float64

		// Position score: first and last sentences are important.
		if i == 0 || i == n-1 {
			score += 3.0
		} else if i == 1 || i == n-2 {
			score += 1.5
		}

		// Length score: medium-length sentences (10-25 words) are best.
		if wc >= 10 && wc <= 25 {
			score += 2.0
		} else if wc >= 5 && wc <= 35 {
			score += 1.0
		}

		// Entity count: capitalized words (excluding sentence-initial) are
		// likely entities / proper nouns -- a signal of importance.
		for j, w := range words {
			if j == 0 {
				continue
			}
			if len(w) > 0 && unicode.IsUpper(rune(w[0])) {
				score += 0.5
			}
		}

		items = append(items, scored{text: s, score: score, idx: i})
	}

	// Sort descending by score.
	sort.Slice(items, func(a, b int) bool {
		return items[a].score > items[b].score
	})

	// Keep top 50%.
	keep := len(items) / 2
	if keep < 1 {
		keep = 1
	}
	top := items[:keep]

	// Re-sort by original position for coherence.
	sort.Slice(top, func(a, b int) bool {
		return top[a].idx < top[b].idx
	})

	var parts []string
	for _, s := range top {
		parts = append(parts, txEnsurePeriod(s.text))
	}
	return strings.Join(parts, " ")
}

// -----------------------------------------------------------------------
// ToBullets
// -----------------------------------------------------------------------

// transitionStarts lists transition words that should be stripped from the
// beginning of bullet points.
var transitionStarts = []string{
	"additionally,",
	"furthermore,",
	"moreover,",
	"in addition,",
	"also,",
	"besides,",
	"likewise,",
	"similarly,",
	"however,",
	"nevertheless,",
	"on the other hand,",
	"in contrast,",
	"meanwhile,",
	"subsequently,",
	"consequently,",
	"as a result,",
	"therefore,",
	"thus,",
	"hence,",
	"accordingly,",
	"specifically,",
	"in particular,",
	"for example,",
	"for instance,",
	"notably,",
	"importantly,",
}

// ToBullets converts prose to bullet points: one per sentence, with
// transition words stripped from the start.
func (t *TextTransformEngine) ToBullets(text string) string {
	text = strings.TrimSpace(text)
	if text == "" {
		return ""
	}

	sents := txSplitSentences(text)
	var bullets []string

	for _, s := range sents {
		s = strings.TrimSpace(s)
		if s == "" {
			continue
		}

		// Remove leading transition words.
		lower := strings.ToLower(s)
		for _, tw := range transitionStarts {
			if strings.HasPrefix(lower, tw) {
				s = strings.TrimSpace(s[len(tw):])
				lower = strings.ToLower(s)
				// Capitalize after stripping.
				s = txCapFirst(s)
				break
			}
		}

		// Remove trailing period for cleaner bullets.
		s = strings.TrimRight(s, ".")

		// Collapse to one line (replace internal newlines).
		s = strings.ReplaceAll(s, "\n", " ")
		s = txCollapseSpaces(s)

		bullets = append(bullets, "- "+s)
	}

	return strings.Join(bullets, "\n")
}

// -----------------------------------------------------------------------
// Internal helpers
// -----------------------------------------------------------------------

// txMultiPunctRe matches 2+ consecutive exclamation or question marks.
var txMultiPunctRe = regexp.MustCompile(`[!?]{2,}`)

// txParenRe matches parenthetical asides: (anything).
var txParenRe = regexp.MustCompile(`\s*\([^)]*\)\s*`)

// txSubordinateRe matches subordinate clauses introduced by specific words.
var txSubordinateRe = regexp.MustCompile(`(?i),\s*(?:which|whereby|notwithstanding)\b[^,.]*[,.]?`)

// txContractionMap maps contractions to their expanded forms.
var txContractionMap = map[string]string{
	"i'm":       "I am",
	"i've":      "I have",
	"i'll":      "I will",
	"i'd":       "I would",
	"you're":    "you are",
	"you've":    "you have",
	"you'll":    "you will",
	"you'd":     "you would",
	"he's":      "he is",
	"she's":     "she is",
	"it's":      "it is",
	"we're":     "we are",
	"we've":     "we have",
	"we'll":     "we will",
	"we'd":      "we would",
	"they're":   "they are",
	"they've":   "they have",
	"they'll":   "they will",
	"they'd":    "they would",
	"that's":    "that is",
	"who's":     "who is",
	"what's":    "what is",
	"here's":    "here is",
	"there's":   "there is",
	"where's":   "where is",
	"when's":    "when is",
	"how's":     "how is",
	"isn't":     "is not",
	"aren't":    "are not",
	"wasn't":    "was not",
	"weren't":   "were not",
	"hasn't":    "has not",
	"haven't":   "have not",
	"hadn't":    "had not",
	"won't":     "will not",
	"wouldn't":  "would not",
	"don't":     "do not",
	"doesn't":   "does not",
	"didn't":    "did not",
	"can't":     "cannot",
	"couldn't":  "could not",
	"shouldn't": "should not",
	"mightn't":  "might not",
	"mustn't":   "must not",
	"let's":     "let us",
	"ain't":     "am not",
}

// txExpansionMap maps expanded forms back to contractions (for casualize).
var txExpansionMap = map[string]string{
	"I am":      "I'm",
	"I have":    "I've",
	"I will":    "I'll",
	"I would":   "I'd",
	"you are":   "you're",
	"you have":  "you've",
	"you will":  "you'll",
	"you would": "you'd",
	"he is":     "he's",
	"she is":    "she's",
	"it is":     "it's",
	"we are":    "we're",
	"we have":   "we've",
	"we will":   "we'll",
	"we would":  "we'd",
	"they are":  "they're",
	"they have": "they've",
	"they will": "they'll",
	"they would": "they'd",
	"is not":    "isn't",
	"are not":   "aren't",
	"was not":   "wasn't",
	"were not":  "weren't",
	"has not":   "hasn't",
	"have not":  "haven't",
	"had not":   "hadn't",
	"will not":  "won't",
	"would not": "wouldn't",
	"do not":    "don't",
	"does not":  "doesn't",
	"did not":   "didn't",
	"cannot":    "can't",
	"could not": "couldn't",
	"should not": "shouldn't",
	"might not": "mightn't",
	"must not":  "mustn't",
	"let us":    "let's",
}

// txExpandContractions expands all contractions in text (case-preserving).
func txExpandContractions(text string) string {
	for contraction, expansion := range txContractionMap {
		re := regexp.MustCompile(`(?i)\b` + regexp.QuoteMeta(contraction) + `\b`)
		text = re.ReplaceAllStringFunc(text, func(match string) string {
			if len(match) > 0 && unicode.IsUpper(rune(match[0])) {
				return txCapFirst(expansion)
			}
			return expansion
		})
	}
	return text
}

// txContractExpansions contracts expanded forms back to contractions.
func txContractExpansions(text string) string {
	for expansion, contraction := range txExpansionMap {
		re := regexp.MustCompile(`(?i)\b` + regexp.QuoteMeta(expansion) + `\b`)
		text = re.ReplaceAllStringFunc(text, func(match string) string {
			if len(match) > 0 && unicode.IsUpper(rune(match[0])) {
				return txCapFirst(contraction)
			}
			return contraction
		})
	}
	return text
}

// txSplitSentences splits text on sentence-ending punctuation, handling
// common abbreviations.
func txSplitSentences(text string) []string {
	text = strings.TrimSpace(text)
	if text == "" {
		return nil
	}

	var sentences []string
	var current strings.Builder
	words := strings.Fields(text)

	for _, w := range words {
		if current.Len() > 0 {
			current.WriteString(" ")
		}
		current.WriteString(w)

		if txEndsWithPunct(w) {
			clean := strings.ToLower(txStripPunct(w))
			if txAbbreviations[clean] && strings.HasSuffix(w, ".") {
				continue
			}
			sent := strings.TrimSpace(current.String())
			if sent != "" {
				sentences = append(sentences, sent)
			}
			current.Reset()
		}
	}
	if current.Len() > 0 {
		sent := strings.TrimSpace(current.String())
		if sent != "" {
			sentences = append(sentences, sent)
		}
	}
	return sentences
}

var txAbbreviations = map[string]bool{
	"dr": true, "mr": true, "mrs": true, "ms": true, "prof": true,
	"sr": true, "jr": true, "st": true, "ave": true, "blvd": true,
	"vs": true, "etc": true, "inc": true, "ltd": true, "corp": true,
	"dept": true, "univ": true, "assn": true, "co": true,
	"no": true, "vol": true, "rev": true, "gen": true, "gov": true,
	"fig": true, "e.g": true, "i.e": true,
}

func txEndsWithPunct(w string) bool {
	if len(w) == 0 {
		return false
	}
	last := w[len(w)-1]
	return last == '.' || last == '!' || last == '?'
}

func txStripPunct(w string) string {
	return strings.TrimFunc(w, func(r rune) bool {
		return !unicode.IsLetter(r) && !unicode.IsDigit(r)
	})
}

func txCapFirst(s string) string {
	if s == "" {
		return s
	}
	runes := []rune(s)
	for i, r := range runes {
		if unicode.IsLetter(r) {
			runes[i] = unicode.ToUpper(r)
			break
		}
	}
	return string(runes)
}

func txEnsurePeriod(s string) string {
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

func txCapitalizeSentences(text string) string {
	runes := []rune(text)
	capNext := true
	for i, r := range runes {
		if capNext && unicode.IsLetter(r) {
			runes[i] = unicode.ToUpper(r)
			capNext = false
		}
		if r == '.' || r == '!' || r == '?' {
			capNext = true
		}
	}
	return string(runes)
}

func txCollapseSpaces(s string) string {
	for strings.Contains(s, "  ") {
		s = strings.ReplaceAll(s, "  ", " ")
	}
	return strings.TrimSpace(s)
}

func txFirstWord(s string) string {
	for i, r := range s {
		if !unicode.IsLetter(r) && r != '\'' {
			return s[:i]
		}
	}
	return s
}

// replaceWordsCI replaces single-word matches (case-insensitive) from the
// given map, preserving original capitalization of the first character.
func (t *TextTransformEngine) replaceWordsCI(text string, m map[string]string) string {
	words := strings.Fields(text)
	for i, w := range words {
		clean := strings.ToLower(txStripPunct(w))
		if replacement, ok := m[clean]; ok {
			// Only replace if the map entry is a single word (or doesn't
			// contain spaces -- multi-word replacements are handled by
			// replacePhrasesCI).
			suffix := txTrailingPunct(w)
			rep := replacement
			if len(w) > 0 && unicode.IsUpper(rune(w[0])) {
				rep = txCapFirst(rep)
			}
			words[i] = rep + suffix
		}
	}
	return strings.Join(words, " ")
}

// replacePhrasesCI replaces multi-word phrases (case-insensitive) from the
// given map.  Longer phrases are replaced first.
func (t *TextTransformEngine) replacePhrasesCI(text string, m map[string]string) string {
	// Collect multi-word phrases sorted by length (longest first).
	type kv struct{ k, v string }
	var phrases []kv
	for k, v := range m {
		if strings.Contains(k, " ") {
			phrases = append(phrases, kv{k, v})
		}
	}
	sort.Slice(phrases, func(a, b int) bool {
		return len(phrases[a].k) > len(phrases[b].k)
	})

	for _, p := range phrases {
		re := regexp.MustCompile(`(?i)\b` + regexp.QuoteMeta(p.k) + `\b`)
		text = re.ReplaceAllStringFunc(text, func(match string) string {
			if len(match) > 0 && unicode.IsUpper(rune(match[0])) {
				return txCapFirst(p.v)
			}
			return p.v
		})
	}
	return text
}

func txTrailingPunct(w string) string {
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

// txBreakLongSentence splits a long sentence at conjunctions/connectors
// into two or more shorter sentences.
func txBreakLongSentence(s string) []string {
	// Conjunctions at which we can split.
	conj := []string{
		" and ", " but ", " or ", " yet ", " so ",
		" however ", " therefore ", " because ", " although ",
		" while ", " whereas ", " since ",
		"; ",
	}

	s = strings.TrimSpace(s)
	// Remove trailing period for splitting.
	trimmed := strings.TrimRight(s, ".")

	for _, c := range conj {
		idx := strings.Index(strings.ToLower(trimmed), c)
		if idx > 10 && idx < len(trimmed)-10 {
			left := strings.TrimSpace(trimmed[:idx])
			right := strings.TrimSpace(trimmed[idx+len(c):])
			return []string{left, right}
		}
	}

	// No good split found — return as-is.
	return []string{s}
}
