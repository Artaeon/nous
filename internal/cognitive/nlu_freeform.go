package cognitive

import (
	"regexp"
	"strings"
	"unicode"

	"github.com/artaeon/nous/internal/simd"
)

// FreeformClassifier catches natural language inputs that the pattern-based
// NLU misses. It uses question-word analysis, verb-based intent detection,
// semantic similarity to canonical examples, and improved topic extraction.
// Designed as a LAST-resort fallback — never replaces existing pattern checks.
type FreeformClassifier struct {
	Embeddings     *WordEmbeddings
	intentExamples map[string][]string // intent → canonical example phrases

	// pre-compiled regexes for topic extraction
	topicPatterns []*regexp.Regexp
	stylePatterns []*regexp.Regexp
}

// NewFreeformClassifier creates a FreeformClassifier seeded with canonical
// intent examples. If emb is nil, semantic similarity is skipped and only
// heuristic rules fire.
func NewFreeformClassifier(emb *WordEmbeddings) *FreeformClassifier {
	fc := &FreeformClassifier{
		Embeddings: emb,
		intentExamples: map[string][]string{
			"explain": {
				"tell me about X",
				"what is X",
				"explain X",
				"describe X",
				"who is X",
				"how does X work",
				"what's the deal with X",
				"break it down for me",
				"ELI5 X",
				"I'm curious about X",
				"can you explain X",
				"what do you know about X",
				"give me info on X",
				"teach me about X",
				"what makes X tick",
				"walk me through X",
				"school me on X",
				"give me the lowdown on X",
				"run me through X",
				"spill the tea on X",
				"break down X for me",
				"how exactly does X work",
				"what even is X",
				"so like what is X about",
			},
			"research": {
				"search for X",
				"look up X",
				"find information about X",
				"google X",
				"research X",
				"dig into X online",
				"find me some sources on X",
				"look into X for me",
				"what does the internet say about X",
				"investigate X",
				"find articles about X",
				"search the web for X",
				"look this up X",
				"check online for X",
				"browse for X",
				"find out about X online",
				"do some research on X",
				"pull up info on X from the web",
				"scout the web for X",
				"find recent articles on X",
			},
			"transform": {
				"rewrite this",
				"make this formal",
				"summarize this text",
				"paraphrase this",
				"rephrase this in simpler terms",
				"make this sound better",
				"clean up this writing",
				"shorten this",
				"expand on this text",
				"make this more concise",
				"reword this",
				"turn this into bullet points",
				"formalize this email",
				"make this casual",
				"simplify this paragraph",
				"dumb this down",
				"make this more professional",
				"translate this to formal English",
				"tighten up this writing",
				"make this flow better",
			},
			"creative": {
				"write me a poem",
				"create a story",
				"compose a song",
				"generate a haiku",
				"make up a joke",
				"invent a character",
				"write a limerick",
				"craft a short story",
				"come up with a pun",
				"write some fiction",
				"create a dialogue",
				"write a rap",
				"compose a letter",
				"draft a speech",
				"write me a bedtime story",
				"make up a fairy tale",
				"generate a creative writing piece",
				"write something funny",
				"pen a sonnet",
				"create a narrative",
			},
			"greeting": {
				"hi",
				"hello",
				"hey",
				"what's up",
				"howdy",
				"yo",
				"greetings",
				"good morning",
				"good afternoon",
				"good evening",
				"hiya",
				"sup",
				"hey there",
				"hi there",
				"hello there",
				"morning",
				"evening",
				"how are you",
				"how's it going",
				"what's going on",
			},
			"farewell": {
				"bye",
				"goodbye",
				"see you",
				"later",
				"thanks bye",
				"take care",
				"peace",
				"ciao",
				"adios",
				"farewell",
				"catch you later",
				"ttyl",
				"gotta go",
				"see ya",
				"talk later",
				"good night",
				"ok bye",
				"bye bye",
				"night",
				"until next time",
			},
			"meta": {
				"how do you work",
				"what can you do",
				"help",
				"what are your capabilities",
				"who are you",
				"what are you",
				"tell me about yourself",
				"what do you know how to do",
				"what features do you have",
				"show me what you can do",
				"what's your purpose",
				"how were you built",
				"are you an AI",
				"what kind of AI are you",
				"what tasks can you handle",
				"give me a list of commands",
				"how should I use you",
				"what's your name",
				"describe your abilities",
				"what are you good at",
			},
		},
	}

	// Compile topic extraction patterns — ordered most-specific first.
	fc.topicPatterns = []*regexp.Regexp{
		regexp.MustCompile(`(?i)^eli5\s+(.+)$`),
		regexp.MustCompile(`(?i)^explain\s+(?:to\s+me\s+)?(?:like\s+.+?\s+)?(.+?)(?:\s+like\s+.+)?$`),
		regexp.MustCompile(`(?i)what(?:'s| is| are)\s+the\s+deal\s+with\s+(.+?)[\s?.!]*$`),
		regexp.MustCompile(`(?i)can\s+you\s+break\s+down\s+(?:how\s+)?(.+?)(?:\s+works?)?\s*[\s?.!]*$`),
		regexp.MustCompile(`(?i)break\s+(?:it\s+)?down(?:\s*:\s*|\s+)(.+)$`),
		regexp.MustCompile(`(?i)(?:i'm|im|i am)\s+curious\s+about\s+(.+?)[\s?.!]*$`),
		regexp.MustCompile(`(?i)(?:give\s+me|gimme)\s+(?:the\s+)?(?:lowdown|rundown|scoop|skinny)\s+(?:on|about)\s+(.+?)[\s?.!]*$`),
		regexp.MustCompile(`(?i)spill\s+the\s+(?:tea|beans)\s+on\s+(.+?)[\s?.!]*$`),
		regexp.MustCompile(`(?i)school\s+me\s+on\s+(.+?)[\s?.!]*$`),
		regexp.MustCompile(`(?i)(?:run|walk)\s+me\s+through\s+(?:how\s+)?(.+?)(?:\s+works?)?\s*[\s?.!]*$`),
		regexp.MustCompile(`(?i)(?:teach|educate)\s+me\s+(?:about\s+)?(.+?)[\s?.!]*$`),
		regexp.MustCompile(`(?i)what(?:'s|\s+is)\s+(.+?)(?:\s+even)?(?:\s+about)?\s*[\s?.!]*$`),
		regexp.MustCompile(`(?i)(?:so\s+)?(?:like\s+)?what(?:'s|\s+is)\s+(.+?)(?:\s+even)?(?:\s+about)?\s*[\s?.!]*$`),
		regexp.MustCompile(`(?i)who\s+(?:even\s+)?(?:is|was)\s+(.+?)[\s?.!]*$`),
		regexp.MustCompile(`(?i)what\s+(?:makes?|causes?)\s+(.+?)[\s?.!]*$`),
		regexp.MustCompile(`(?i)how\s+does\s+(.+?)\s+work\s*(?:exactly|really)?\s*[\s?.!]*$`),
		regexp.MustCompile(`(?i)(?:yo|hey|ok|okay|so)\s+what\s+(?:do\s+you\s+know\s+about|is|are)\s+(.+?)[\s?.!]*$`),
		regexp.MustCompile(`(?i)what\s+do\s+you\s+know\s+about\s+(.+?)[\s?.!]*$`),
		regexp.MustCompile(`(?i)(?:tell|teach)\s+me\s+(?:about\s+)?(.+?)[\s?.!]*$`),
		regexp.MustCompile(`(?i)(?:give\s+me\s+)?(?:info|information)\s+(?:on|about)\s+(.+?)[\s?.!]*$`),
	}

	// Style modifier patterns — capture stylistic requests.
	fc.stylePatterns = []*regexp.Regexp{
		regexp.MustCompile(`(?i)\s+like\s+(?:i'm|im|i am)\s+(?:a\s+)?(.+?)[\s?.!]*$`),
		regexp.MustCompile(`(?i)\s+(?:in|using)\s+(simple\s+terms?|plain\s+(?:english|language)|layman(?:'s)?\s+terms?)[\s?.!]*$`),
		regexp.MustCompile(`(?i)\s+(briefly|in\s+brief|in\s+short)[\s?.!]*$`),
		regexp.MustCompile(`(?i)\s+(in\s+detail|in\s+depth|thoroughly|exhaustively)[\s?.!]*$`),
		regexp.MustCompile(`(?i)\s+(?:for|like)\s+(?:a\s+)?(\d+\s+year\s+old|child|kid|beginner|expert|scientist|professor)[\s?.!]*$`),
		regexp.MustCompile(`(?i)^eli5\s+`),
	}

	return fc
}

// Classify attempts to classify a free-form input that the pattern NLU missed.
// Returns nil if it cannot determine a clear intent.
func (fc *FreeformClassifier) Classify(input string) *NLUResult {
	lower := strings.ToLower(strings.TrimSpace(input))
	if lower == "" {
		return nil
	}

	result := &NLUResult{
		Raw:      input,
		Entities: make(map[string]string),
	}

	// ---- Strategy 1: Question-word / phrasing analysis ----
	if fc.classifyByStructure(lower, result) {
		// Enrich with topic and style extraction.
		fc.enrichEntities(input, lower, result)
		return result
	}

	// ---- Strategy 2: Verb-based intent detection ----
	if fc.classifyByVerb(lower, result) {
		fc.enrichEntities(input, lower, result)
		return result
	}

	// ---- Strategy 3: Semantic similarity to canonical examples ----
	if fc.Embeddings != nil {
		if fc.classifyBySimilarity(input, result) {
			fc.enrichEntities(input, lower, result)
			return result
		}
	}

	// ---- Strategy 4: Catch-all heuristic ----
	// If the input is mostly content words (not filler), it is probably
	// a topic/knowledge query like "isaac newton" or "quantum computing".
	// Require content words to be at least half of total words to avoid
	// matching filler-heavy inputs like "hmm well you see the thing is".
	words := contentWords(lower)
	allWords := strings.Fields(lower)
	if len(words) >= 2 && len(words)*2 >= len(allWords) {
		result.Intent = "question"
		result.Confidence = 0.50
		fc.enrichEntities(input, lower, result)
		return result
	}

	return nil
}

// -----------------------------------------------------------------------
// Strategy 1: Question structure analysis
// -----------------------------------------------------------------------

// questionLeads are phrasing patterns that almost always indicate a
// knowledge lookup. Ordered longest-first so greedy matching works.
var questionLeads = []string{
	// multi-word leads
	"can you break down",
	"could you break down",
	"can you explain",
	"could you explain",
	"can you describe",
	"could you describe",
	"can you tell me about",
	"could you tell me about",
	"would you explain",
	"would you describe",
	"would you tell me about",
	"tell me about",
	"tell me",
	"teach me about",
	"teach me",
	"school me on",
	"give me the lowdown on",
	"give me the rundown on",
	"give me info on",
	"give me information on",
	"spill the tea on",
	"run me through",
	"walk me through",
	"break it down",
	"break down",
	"fill me in on",
	"clue me in on",
	"what do you know about",
	"what's the deal with",
	"whats the deal with",
	"i'm curious about",
	"im curious about",
	"i am curious about",
	"i want to know about",
	"i want to learn about",
	"i'd like to know about",
	"id like to know about",
	// question words
	"what is",
	"what's",
	"whats",
	"what are",
	"what was",
	"what were",
	"what makes",
	"what causes",
	"who is",
	"who was",
	"who are",
	"who even is",
	"how does",
	"how do",
	"how is",
	"how come",
	"why is",
	"why are",
	"why does",
	"why do",
	"where is",
	"where are",
	"when was",
	"when is",
	"when did",
	"which is",
	"which are",
	// informal
	"so like",
	"yo what",
	"ok so what",
	"okay so what",
}

func (fc *FreeformClassifier) classifyByStructure(lower string, r *NLUResult) bool {
	stripped := strings.TrimRightFunc(lower, func(c rune) bool {
		return unicode.IsPunct(c) || unicode.IsSpace(c)
	})

	// ELI5 prefix
	if strings.HasPrefix(lower, "eli5 ") || strings.HasPrefix(lower, "eli5:") || lower == "eli5" {
		r.Intent = "explain"
		r.Confidence = 0.85
		return true
	}

	for _, lead := range questionLeads {
		if strings.HasPrefix(stripped, lead) || strings.HasPrefix(lower, lead+" ") {
			r.Intent = "explain"
			r.Confidence = 0.80
			return true
		}
	}

	// Ends with "?" and has 3+ words → knowledge question
	if strings.HasSuffix(strings.TrimSpace(lower), "?") && len(strings.Fields(lower)) >= 3 {
		r.Intent = "explain"
		r.Confidence = 0.70
		return true
	}

	return false
}

// -----------------------------------------------------------------------
// Strategy 2: Verb-based intent detection
// -----------------------------------------------------------------------

var knowledgeVerbs = []string{
	"explain", "describe", "define", "elaborate", "clarify",
	"break down", "walk through", "run through",
	"illustrate", "outline", "detail", "expound",
	"elucidate", "summarize", "recap",
}

var creativeVerbs = []string{
	"write", "create", "generate", "compose", "make up",
	"invent", "craft", "pen", "draft", "produce",
	"come up with", "dream up", "conjure",
}

var compareVerbs = []string{
	"compare", "contrast", "distinguish", "differentiate",
	"versus", "vs",
}

func (fc *FreeformClassifier) classifyByVerb(lower string, r *NLUResult) bool {
	// Check creative verbs first (they are more specific).
	for _, v := range creativeVerbs {
		if containsVerb(lower, v) {
			r.Intent = "creative"
			r.Confidence = 0.75
			return true
		}
	}

	for _, v := range compareVerbs {
		if containsVerb(lower, v) {
			r.Intent = "compare"
			r.Confidence = 0.75
			r.Entities["comparison"] = "true"
			return true
		}
	}

	for _, v := range knowledgeVerbs {
		if containsVerb(lower, v) {
			r.Intent = "explain"
			r.Confidence = 0.75
			return true
		}
	}

	return false
}

// containsVerb checks if the verb appears at a word boundary in the input.
func containsVerb(s, verb string) bool {
	idx := strings.Index(s, verb)
	if idx < 0 {
		return false
	}
	// Check left boundary.
	if idx > 0 {
		c := rune(s[idx-1])
		if unicode.IsLetter(c) || unicode.IsDigit(c) {
			return false
		}
	}
	// Check right boundary.
	end := idx + len(verb)
	if end < len(s) {
		c := rune(s[end])
		if unicode.IsLetter(c) || unicode.IsDigit(c) {
			return false
		}
	}
	return true
}

// -----------------------------------------------------------------------
// Strategy 3: Semantic similarity to canonical examples
// -----------------------------------------------------------------------

func (fc *FreeformClassifier) classifyBySimilarity(input string, r *NLUResult) bool {
	inputVec, err := fc.Embeddings.SentenceEmbed(input)
	if err != nil {
		return false
	}

	bestIntent := ""
	bestScore := 0.0

	for intent, examples := range fc.intentExamples {
		for _, ex := range examples {
			exVec, err := fc.Embeddings.SentenceEmbed(ex)
			if err != nil {
				continue
			}
			score := simd.CosineSimilarity(inputVec, exVec)
			if score > bestScore {
				bestScore = score
				bestIntent = intent
			}
		}
	}

	// Threshold: require meaningful similarity.
	if bestScore >= 0.55 && bestIntent != "" {
		r.Intent = bestIntent
		r.Confidence = bestScore * 0.9 // scale down slightly
		if r.Confidence > 0.85 {
			r.Confidence = 0.85
		}
		return true
	}

	return false
}

// -----------------------------------------------------------------------
// Entity enrichment — topic and style extraction
// -----------------------------------------------------------------------

func (fc *FreeformClassifier) enrichEntities(raw, lower string, r *NLUResult) {
	if topic := fc.ExtractTopic(raw); topic != "" {
		r.Entities["topic"] = topic
	}
	if style := fc.ExtractStyle(raw); style != "" {
		r.Entities["style"] = style
	}
}

// ExtractTopic pulls the core subject from a complex natural-language phrasing.
func (fc *FreeformClassifier) ExtractTopic(input string) string {
	cleaned := strings.TrimSpace(input)
	if cleaned == "" {
		return ""
	}

	// First strip style modifiers so they don't pollute the topic.
	for _, sp := range fc.stylePatterns {
		cleaned = sp.ReplaceAllString(cleaned, "")
	}

	// Try each topic pattern.
	for _, tp := range fc.topicPatterns {
		if m := tp.FindStringSubmatch(cleaned); len(m) > 1 {
			topic := strings.TrimSpace(m[1])
			topic = cleanFreeformTopic(topic)
			if topic != "" {
				return topic
			}
		}
	}

	// Fallback: strip common prefixes/fillers and return what remains.
	lower := strings.ToLower(cleaned)
	topic := lower

	// Strip informal prefixes.
	for _, prefix := range []string{
		"yo ", "hey ", "ok ", "okay ", "so ", "like ", "um ", "uh ",
		"well ", "hmm ", "so like ", "ok so ",
	} {
		topic = strings.TrimPrefix(topic, prefix)
	}

	// Strip question leads.
	for _, lead := range questionLeads {
		if strings.HasPrefix(topic, lead+" ") {
			topic = strings.TrimSpace(topic[len(lead):])
			break
		} else if strings.HasPrefix(topic, lead) {
			topic = strings.TrimSpace(topic[len(lead):])
			break
		}
	}

	// Strip trailing filler.
	topic = strings.TrimRight(topic, "?!. ")
	for _, suffix := range []string{
		" exactly", " really", " anyway", " even", " tho", " though",
		" about", " works", " work",
	} {
		topic = strings.TrimSuffix(topic, suffix)
	}

	topic = stripLeadingFillers(topic)
	topic = strings.TrimSpace(topic)

	if topic == "" || topic == lower {
		// Could not reduce — return the cleaned version.
		return strings.TrimRight(strings.TrimSpace(cleaned), "?!.")
	}
	return topic
}

// ExtractStyle detects style modifiers in the input (e.g. "like I'm a pirate",
// "in simple terms", "for a 5 year old").
func (fc *FreeformClassifier) ExtractStyle(input string) string {
	for _, sp := range fc.stylePatterns {
		if m := sp.FindStringSubmatch(input); len(m) > 1 {
			return strings.TrimSpace(m[1])
		}
	}

	// Check for ELI5.
	lower := strings.ToLower(input)
	if strings.HasPrefix(lower, "eli5") {
		return "for a 5 year old"
	}

	return ""
}

// -----------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------

// contentWords returns non-stopword tokens from the input.
func contentWords(lower string) []string {
	stops := map[string]bool{
		"a": true, "an": true, "the": true, "is": true, "are": true,
		"was": true, "were": true, "am": true, "be": true, "been": true,
		"i": true, "me": true, "my": true, "you": true, "your": true,
		"it": true, "its": true, "we": true, "our": true, "they": true,
		"to": true, "of": true, "in": true, "on": true, "at": true,
		"for": true, "with": true, "by": true, "from": true, "up": true,
		"do": true, "does": true, "did": true, "can": true, "could": true,
		"would": true, "should": true, "will": true, "shall": true,
		"and": true, "or": true, "but": true, "not": true, "so": true,
		"if": true, "then": true, "than": true, "that": true, "this": true,
		"what": true, "how": true, "why": true, "when": true, "where": true,
		"who": true, "which": true, "about": true, "like": true, "just": true,
		"even": true, "also": true, "very": true, "really": true, "too": true,
		"yo": true, "hey": true, "ok": true, "okay": true, "um": true,
		"uh": true, "well": true, "hmm": true, "oh": true, "ah": true,
	}

	var out []string
	for _, w := range strings.Fields(lower) {
		w = strings.Trim(w, ".,;:!?\"'()[]{}–—")
		if w != "" && !stops[w] {
			out = append(out, w)
		}
	}
	return out
}

// cleanFreeformTopic tidies up an extracted topic string.
func cleanFreeformTopic(s string) string {
	s = strings.TrimRight(s, "?!. ")
	s = stripLeadingFillers(s)
	s = strings.TrimSpace(s)
	return s
}
