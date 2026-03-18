package cognitive

import (
	"regexp"
	"strings"
	"unicode"
)

// NLUResult is defined in action.go — shared struct for NLU output.
// Fields: Intent, Action, Entities, Confidence, Raw.

// NLU is a pure-code Natural Language Understanding engine.
// Zero LLM calls. Microsecond-level. Deterministic.
type NLU struct {
	greetings    []string
	farewells    []string
	affirmatives []string
	negatives    []string
	metaPatterns []string

	// compiled regexes — built once at init
	urlRe       *regexp.Regexp
	pathRe      *regexp.Regexp
	mathRe      *regexp.Regexp
	quotedRe    *regexp.Regexp
	dateWordRe  *regexp.Regexp
	dateFormalRe *regexp.Regexp

	questionPrefixes []string
	commandVerbs     []string
	searchVerbs      []string
	fileVerbs        []string
	memoryVerbs      []string
	recallVerbs      []string
	planVerbs        []string
	explainVerbs     []string
	computeVerbs     []string

	// web-lookup signals: topics that require external/current knowledge
	currentEventWords []string
	webLookupPatterns []*regexp.Regexp
}

// NewNLU creates a new deterministic NLU engine with all pattern tables initialized.
func NewNLU() *NLU {
	n := &NLU{
		greetings: []string{
			"hi", "hello", "hey", "howdy", "hiya", "yo",
			"good morning", "good afternoon", "good evening", "good night",
			"morning", "evening", "afternoon",
			"what's up", "whats up", "sup", "greetings", "salutations",
		},
		farewells: []string{
			"bye", "goodbye", "good bye", "see ya", "see you", "later",
			"farewell", "ciao", "adios", "peace", "take care",
			"good night", "gn", "ttyl", "talk later",
		},
		affirmatives: []string{
			"yes", "yeah", "yep", "yup", "sure", "ok", "okay", "k",
			"thanks", "thank you", "thx", "ty", "great", "good", "nice",
			"awesome", "cool", "perfect", "exactly", "correct", "right",
			"got it", "understood", "makes sense", "agreed", "absolutely",
			"no", "nope", "nah", "not really", "negative",
		},
		metaPatterns: []string{
			"what can you do", "who are you", "what are you",
			"help", "how do you work", "what do you know",
			"tell me about yourself", "your capabilities",
			"what's your name", "whats your name",
		},

		urlRe:        regexp.MustCompile(`https?://[^\s<>"{}|\\^` + "`" + `\[\]]+`),
		pathRe:       regexp.MustCompile(`(?:^|[\s,])([~.]?/[a-zA-Z0-9_./-]+|[a-zA-Z0-9_./]*\.[a-zA-Z]{1,10})`),
		mathRe:       regexp.MustCompile(`\b(\d+(?:\.\d+)?)\s*([+\-*/^%])\s*(\d+(?:\.\d+)?)\b`),
		quotedRe:     regexp.MustCompile(`"([^"]+)"`),
		dateWordRe:   regexp.MustCompile(`(?i)\b(today|tomorrow|yesterday|next\s+(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday|week|month|year)|last\s+(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday|week|month|year)|this\s+(?:week|month|year|weekend))\b`),
		dateFormalRe: regexp.MustCompile(`(?i)\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2}(?:st|nd|rd|th)?(?:\s*,?\s*\d{4})?\b`),

		questionPrefixes: []string{
			"what", "who", "where", "when", "why", "how",
			"is ", "are ", "was ", "were ", "do ", "does ", "did ",
			"can ", "could ", "would ", "should ", "will ",
			"has ", "have ", "had ",
		},
		commandVerbs: []string{
			"do", "run", "execute", "start", "stop", "restart",
			"install", "deploy", "build", "compile", "test",
			"delete", "remove", "kill", "clean", "reset",
			"set", "configure", "enable", "disable", "toggle",
		},
		searchVerbs: []string{
			"search", "find", "look up", "lookup", "look for",
			"google", "bing", "duckduckgo",
		},
		fileVerbs: []string{
			"read", "open", "edit", "create", "write", "save",
			"cat", "show file", "view file", "delete file", "remove file",
			"list files", "ls", "mkdir",
		},
		memoryVerbs: []string{
			"remember", "memorize", "store", "save that",
			"my name is", "i work at", "i live in", "i like", "i prefer",
			"note that", "keep in mind",
		},
		recallVerbs: []string{
			"do you remember", "what's my", "whats my", "what is my",
			"what did i", "recall", "have i told you",
			"what do you know about me",
		},
		planVerbs: []string{
			"plan", "schedule", "remind", "reminder", "set alarm",
			"add to calendar", "create task", "todo", "to-do", "to do",
			"routine", "agenda",
		},
		explainVerbs: []string{
			"explain", "describe", "elaborate", "clarify",
			"what is", "what are", "what does",
			"how does", "how do", "how is",
			"tell me about", "teach me",
			"define", "definition of",
		},
		computeVerbs: []string{
			"calculate", "compute", "solve", "evaluate",
			"what is", // followed by math
			"convert", "how much is", "how many",
		},
		currentEventWords: []string{
			"weather", "news", "latest", "current", "today's",
			"score", "scores", "won", "winning", "lost",
			"price", "stock", "market", "trading",
			"live", "breaking", "trending", "viral",
			"election", "results",
		},
		webLookupPatterns: []*regexp.Regexp{
			regexp.MustCompile(`(?i)what(?:'s| is) the (?:weather|temperature|forecast)`),
			regexp.MustCompile(`(?i)(?:latest|recent|current|breaking|today'?s?)\s+(?:news|headlines|updates?)`),
			regexp.MustCompile(`(?i)who (?:won|is winning|lost|scored|leads?)`),
			regexp.MustCompile(`(?i)(?:stock|share)\s+price`),
			regexp.MustCompile(`(?i)what(?:'s| is)\s+(?:happening|going on)\s+(?:in|at|with)`),
			regexp.MustCompile(`(?i)how (?:much|many)\s+(?:does|is|are)\s+\w+\s+(?:cost|worth)`),
		},
	}
	return n
}

// Understand processes raw input and returns structured NLU output.
// This is PURE CODE — no LLM call, no I/O, no network.
// If Confidence < 0.5, the caller should make ONE LLM call for disambiguation.
func (n *NLU) Understand(input string) *NLUResult {
	result := &NLUResult{
		Raw:      input,
		Entities: make(map[string]string),
	}

	trimmed := strings.TrimSpace(input)
	if trimmed == "" {
		result.Intent = "unknown"
		result.Action = "respond"
		result.Confidence = 0.0
		return result
	}

	lower := strings.ToLower(trimmed)

	// Phase 1: Extract entities (always, regardless of intent)
	n.extractEntities(trimmed, lower, result)

	// Phase 2: Classify intent (order matters — most specific first)
	n.classifyIntent(trimmed, lower, result)

	// Phase 3: Map intent + entities to action
	n.mapAction(lower, result)

	return result
}

// extractEntities pulls structured data from the raw input using regex and heuristics.
func (n *NLU) extractEntities(raw, lower string, r *NLUResult) {
	// URLs
	if urls := n.urlRe.FindAllString(raw, -1); len(urls) > 0 {
		r.Entities["url"] = urls[0]
		if len(urls) > 1 {
			r.Entities["urls"] = strings.Join(urls, ",")
		}
	}

	// File paths
	if paths := n.pathRe.FindAllStringSubmatch(raw, -1); len(paths) > 0 {
		p := strings.TrimSpace(paths[0][1])
		r.Entities["path"] = p
	}

	// Math expressions
	if m := n.mathRe.FindStringSubmatch(raw); len(m) == 4 {
		r.Entities["expression"] = m[0]
	}

	// Quoted strings
	if m := n.quotedRe.FindAllStringSubmatch(raw, -1); len(m) > 0 {
		r.Entities["quoted"] = m[0][1]
		if len(m) > 1 {
			quoted := make([]string, len(m))
			for i, q := range m {
				quoted[i] = q[1]
			}
			r.Entities["all_quoted"] = strings.Join(quoted, "|")
		}
	}

	// Dates
	if m := n.dateWordRe.FindString(lower); m != "" {
		r.Entities["date"] = m
	} else if m := n.dateFormalRe.FindString(raw); m != "" {
		r.Entities["date"] = m
	}
}

// classifyIntent determines what the user wants.
func (n *NLU) classifyIntent(raw, lower string, r *NLUResult) {
	// Strip trailing punctuation for matching
	stripped := strings.TrimRightFunc(lower, func(r rune) bool {
		return unicode.IsPunct(r) || unicode.IsSpace(r)
	})

	// 1. Exact/prefix match: greetings
	for _, g := range n.greetings {
		if stripped == g || strings.HasPrefix(lower, g+" ") || strings.HasPrefix(lower, g+",") || strings.HasPrefix(lower, g+"!") {
			// Check if it's "good night" which can be farewell
			if strings.Contains(lower, "good night") {
				r.Intent = "farewell"
				r.Confidence = 0.95
				return
			}
			r.Intent = "greeting"
			r.Confidence = 0.95
			return
		}
	}

	// 2. Farewells
	for _, f := range n.farewells {
		if stripped == f || strings.HasPrefix(lower, f+" ") || strings.HasPrefix(lower, f+",") || strings.HasPrefix(lower, f+"!") {
			r.Intent = "farewell"
			r.Confidence = 0.95
			return
		}
	}

	// 3. Affirmations/negations (short responses)
	if len(strings.Fields(lower)) <= 4 {
		for _, a := range n.affirmatives {
			if stripped == a || lower == a+"!" || lower == a+"." {
				r.Intent = "affirmation"
				r.Confidence = 0.90
				return
			}
		}
	}

	// 4. Meta queries
	for _, m := range n.metaPatterns {
		if strings.Contains(lower, m) {
			r.Intent = "meta"
			r.Confidence = 0.90
			return
		}
	}

	// 5. Recall (before remember, since "do you remember" contains "remember")
	for _, v := range n.recallVerbs {
		if strings.Contains(lower, v) {
			r.Intent = "recall"
			r.Confidence = 0.85
			r.Entities["topic"] = n.extractTopic(lower, v)
			return
		}
	}

	// 6. Remember/store
	for _, v := range n.memoryVerbs {
		if strings.Contains(lower, v) {
			r.Intent = "remember"
			r.Confidence = 0.85
			r.Entities["topic"] = n.extractTopic(lower, v)
			return
		}
	}

	// 7. Plan/schedule
	for _, v := range n.planVerbs {
		if strings.Contains(lower, v) {
			r.Intent = "plan"
			r.Confidence = 0.80
			r.Entities["topic"] = n.extractTopic(lower, v)
			return
		}
	}

	// 8. File operations (before search, since "read file" should be file_op not search)
	for _, v := range n.fileVerbs {
		if matchWord(lower, v) {
			r.Intent = "file_op"
			r.Confidence = 0.85
			return
		}
	}

	// 9. URL present → fetch
	if _, hasURL := r.Entities["url"]; hasURL {
		r.Intent = "command"
		r.Action = "fetch_url"
		r.Confidence = 0.85
		return
	}

	// 10. Search verbs (before compute — explicit "search for X" wins over incidental math)
	for _, v := range n.searchVerbs {
		if strings.Contains(lower, v) {
			r.Intent = "search"
			r.Confidence = 0.85
			r.Entities["query"] = n.extractTopic(lower, v)
			return
		}
	}

	// 11. Compute: math expression present or compute verbs with numbers
	if _, hasExpr := r.Entities["expression"]; hasExpr {
		r.Intent = "compute"
		r.Confidence = 0.90
		return
	}
	for _, v := range n.computeVerbs {
		if strings.Contains(lower, v) && containsDigit(lower) {
			r.Intent = "compute"
			r.Confidence = 0.80
			return
		}
	}

	// 12. Web lookup: current events, weather, prices, scores
	for _, pat := range n.webLookupPatterns {
		if pat.MatchString(lower) {
			r.Intent = "web_lookup"
			r.Confidence = 0.90
			r.Entities["topic"] = n.extractTopicGeneral(lower)
			return
		}
	}
	for _, w := range n.currentEventWords {
		if strings.Contains(lower, w) {
			r.Intent = "web_lookup"
			r.Confidence = 0.75
			r.Entities["topic"] = n.extractTopicGeneral(lower)
			return
		}
	}

	// 13. Explain verbs
	for _, v := range n.explainVerbs {
		if strings.HasPrefix(lower, v+" ") || strings.HasPrefix(lower, v+"\t") {
			r.Intent = "explain"
			r.Confidence = 0.80
			r.Entities["topic"] = n.extractTopic(lower, v)
			return
		}
	}

	// 14. Command verbs
	for _, v := range n.commandVerbs {
		if strings.HasPrefix(lower, v+" ") || strings.HasPrefix(lower, v+"\t") {
			r.Intent = "command"
			r.Confidence = 0.80
			r.Entities["topic"] = n.extractTopic(lower, v)
			return
		}
	}

	// 15. Questions (ends with ? or starts with question word)
	if strings.HasSuffix(strings.TrimSpace(raw), "?") {
		r.Intent = "question"
		r.Confidence = 0.75
		r.Entities["topic"] = n.extractTopicGeneral(lower)
		return
	}
	for _, q := range n.questionPrefixes {
		if strings.HasPrefix(lower, q) {
			r.Intent = "question"
			r.Confidence = 0.70
			r.Entities["topic"] = n.extractTopicGeneral(lower)
			return
		}
	}

	// 16. Contains explain verbs (not prefix-only, looser match)
	for _, v := range n.explainVerbs {
		if strings.Contains(lower, v) {
			r.Intent = "explain"
			r.Confidence = 0.65
			r.Entities["topic"] = n.extractTopic(lower, v)
			return
		}
	}

	// 17. Fallback: if it's a short phrase, treat as question; if long, treat as statement
	words := strings.Fields(lower)
	if len(words) <= 3 {
		// Short input — likely a topic query
		r.Intent = "question"
		r.Confidence = 0.40
		r.Entities["topic"] = strings.Join(words, " ")
	} else {
		r.Intent = "question"
		r.Confidence = 0.35
		r.Entities["topic"] = n.extractTopicGeneral(lower)
	}
}

// mapAction maps the classified intent + entities to a concrete action.
func (n *NLU) mapAction(lower string, r *NLUResult) {
	// If action was already set (e.g., fetch_url), keep it
	if r.Action != "" {
		return
	}

	// Check for multi-step chain patterns before single-action mapping.
	if chainType := n.detectChain(lower); chainType != "" {
		r.Action = "chain"
		r.Entities["chain_type"] = chainType
		if r.Entities["topic"] == "" {
			r.Entities["topic"] = n.extractChainTopic(lower)
		}
		return
	}

	// Check for document generation patterns.
	if n.isDocGeneration(lower) {
		r.Action = "generate_doc"
		if r.Entities["topic"] == "" {
			r.Entities["topic"] = n.extractChainTopic(lower)
		}
		return
	}

	// Date questions: if a date entity is present and it's a question about time/day,
	// route to compute (the date evaluator handles these without LLM).
	if _, hasDate := r.Entities["date"]; hasDate {
		if r.Intent == "question" || r.Intent == "explain" {
			r.Action = "compute"
			r.Entities["expr"] = r.Raw
			return
		}
	}

	switch r.Intent {
	case "greeting", "farewell", "affirmation":
		r.Action = "respond"

	case "meta":
		r.Action = "respond"

	case "remember":
		r.Action = "lookup_memory" // store to memory

	case "recall":
		r.Action = "lookup_memory"

	case "plan":
		r.Action = "schedule"

	case "file_op":
		r.Action = "file_op"

	case "compute":
		r.Action = "compute"

	case "search":
		r.Action = "web_search"

	case "web_lookup":
		r.Action = "web_search"

	case "explain":
		// Explanation of a concept: try knowledge base first
		r.Action = "lookup_knowledge"

	case "question":
		// Determine if we need web, knowledge, or memory
		if n.needsWebLookup(lower) {
			r.Action = "web_search"
		} else if n.isPersonalQuestion(lower) {
			r.Action = "lookup_memory"
		} else {
			// General question — try knowledge base, caller escalates to web if not found
			r.Action = "lookup_knowledge"
		}

	case "command":
		r.Action = "llm_chat" // commands need LLM to figure out specifics

	default:
		r.Action = "llm_chat"
	}
}

// needsWebLookup returns true if the question requires external/current knowledge.
func (n *NLU) needsWebLookup(lower string) bool {
	for _, w := range n.currentEventWords {
		if strings.Contains(lower, w) {
			return true
		}
	}
	for _, pat := range n.webLookupPatterns {
		if pat.MatchString(lower) {
			return true
		}
	}
	return false
}

// isPersonalQuestion returns true if the question is about the user's stored info.
func (n *NLU) isPersonalQuestion(lower string) bool {
	personalPrefixes := []string{
		"what's my", "whats my", "what is my",
		"where do i", "what do i", "who am i",
		"what did i", "have i", "am i",
	}
	for _, p := range personalPrefixes {
		if strings.HasPrefix(lower, p) || strings.Contains(lower, p) {
			return true
		}
	}
	return false
}

// extractTopic strips the verb/trigger from input and returns the remaining subject.
func (n *NLU) extractTopic(lower, trigger string) string {
	idx := strings.Index(lower, trigger)
	if idx < 0 {
		return n.extractTopicGeneral(lower)
	}
	after := strings.TrimSpace(lower[idx+len(trigger):])
	// Strip common filler words at the start
	after = stripLeadingFillers(after)
	// Trim trailing punctuation
	after = strings.TrimRight(after, "?!.")
	return strings.TrimSpace(after)
}

// extractTopicGeneral strips common question/filler words to get the core topic.
func (n *NLU) extractTopicGeneral(lower string) string {
	// Remove question words and common fillers
	topic := lower
	for _, strip := range []string{
		"what is ", "what's ", "what are ", "what does ",
		"who is ", "who are ", "who was ", "who ",
		"where is ", "where are ",
		"when is ", "when was ", "when did ",
		"why is ", "why are ", "why does ", "why did ",
		"how does ", "how do ", "how is ", "how are ", "how can ",
		"can you ", "could you ", "would you ", "please ",
		"tell me ", "i want to know ",
		"is there ", "are there ",
	} {
		if strings.HasPrefix(topic, strip) {
			topic = topic[len(strip):]
			break
		}
	}
	topic = strings.TrimRight(topic, "?!.")
	return strings.TrimSpace(topic)
}

// stripLeadingFillers removes filler words like "about", "the", "a", "for" from the start.
func stripLeadingFillers(s string) string {
	fillers := []string{"about ", "the ", "a ", "an ", "for ", "me ", "that ", "this "}
	changed := true
	for changed {
		changed = false
		for _, f := range fillers {
			if strings.HasPrefix(s, f) {
				s = s[len(f):]
				changed = true
			}
		}
	}
	return s
}

// matchWord checks if the phrase appears in s as a word boundary match,
// not as a substring of another word. For multi-word phrases, uses Contains.
func matchWord(s, phrase string) bool {
	if strings.Contains(phrase, " ") {
		// Multi-word phrase: exact substring match is fine
		return strings.Contains(s, phrase)
	}
	// Single word: check word boundaries
	idx := 0
	for {
		pos := strings.Index(s[idx:], phrase)
		if pos < 0 {
			return false
		}
		pos += idx
		start := pos
		end := pos + len(phrase)
		leftOK := start == 0 || !isWordChar(rune(s[start-1]))
		rightOK := end >= len(s) || !isWordChar(rune(s[end]))
		if leftOK && rightOK {
			return true
		}
		idx = pos + 1
		if idx >= len(s) {
			return false
		}
	}
}

func isWordChar(r rune) bool {
	return (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') || (r >= '0' && r <= '9') || r == '_'
}

// followUpPatterns are phrases that indicate the user is referring to a previous topic.
var followUpPatterns = []string{
	"explain further", "explain more", "tell me more", "go on", "continue",
	"elaborate", "more details", "what else", "and then",
	"keep going", "go ahead",
}

// followUpExactPatterns match single-word or very short follow-ups.
var followUpExactPatterns = []string{
	"why", "how", "really", "seriously", "and",
}

// isFollowUp returns true if the input looks like a follow-up referencing a previous turn.
func isFollowUp(lower string, result *NLUResult) bool {
	stripped := strings.TrimRight(lower, "?!. ")

	// Exact match on short follow-ups
	for _, p := range followUpExactPatterns {
		if stripped == p {
			return true
		}
	}

	// Substring match on follow-up phrases
	for _, p := range followUpPatterns {
		if strings.Contains(lower, p) {
			return true
		}
	}

	// "what about X" pattern — follow-up with a new angle
	if strings.HasPrefix(lower, "what about ") || strings.HasPrefix(lower, "how about ") {
		return true
	}

	// Short input (under 4 words) starting with a question word and no clear topic
	words := strings.Fields(lower)
	if len(words) > 0 && len(words) < 4 {
		questionWords := map[string]bool{
			"what": true, "why": true, "how": true, "when": true,
			"where": true, "who": true, "which": true,
		}
		first := strings.TrimRight(words[0], "?!.,")
		if questionWords[first] {
			topic := result.Entities["topic"]
			if topic == "" || topic == stripped {
				return true
			}
		}
	}

	return false
}

// lastUserMessage returns the content of the last user message in the conversation,
// or empty string if there is no prior user message.
func lastUserMessage(conv *Conversation) string {
	if conv == nil {
		return ""
	}
	msgs := conv.Messages()
	for i := len(msgs) - 1; i >= 0; i-- {
		if msgs[i].Role == "user" {
			return msgs[i].Content
		}
	}
	return ""
}

// extractTopicFromPrior extracts the core topic from a prior user message.
func (n *NLU) extractTopicFromPrior(prior string) string {
	lower := strings.ToLower(strings.TrimSpace(prior))
	return n.extractTopicGeneral(lower)
}

// UnderstandWithContext processes input with conversation history for follow-up resolution.
// When the user says "explain further" or "tell me more", this resolves the topic
// from the previous conversation turn.
func (n *NLU) UnderstandWithContext(input string, conv *Conversation) *NLUResult {
	result := n.Understand(input)

	lower := strings.ToLower(strings.TrimSpace(input))

	if !isFollowUp(lower, result) {
		return result
	}

	prior := lastUserMessage(conv)
	if prior == "" {
		return result
	}

	// Resolve topic from the previous turn
	previousTopic := n.extractTopicFromPrior(prior)
	if previousTopic == "" {
		return result
	}

	// "what about X" / "how about X" — combine the new angle with the previous topic
	if strings.HasPrefix(lower, "what about ") {
		newAngle := strings.TrimPrefix(lower, "what about ")
		newAngle = strings.TrimRight(newAngle, "?!. ")
		result.Entities["topic"] = previousTopic + " — " + newAngle
		result.Entities["previous_topic"] = previousTopic
		result.Entities["new_angle"] = newAngle
	} else if strings.HasPrefix(lower, "how about ") {
		newAngle := strings.TrimPrefix(lower, "how about ")
		newAngle = strings.TrimRight(newAngle, "?!. ")
		result.Entities["topic"] = previousTopic + " — " + newAngle
		result.Entities["previous_topic"] = previousTopic
		result.Entities["new_angle"] = newAngle
	} else {
		// Pure follow-up — carry topic forward
		result.Entities["topic"] = previousTopic
		result.Entities["previous_topic"] = previousTopic
	}

	result.Entities["follow_up"] = "true"

	// Boost confidence — we resolved the referent
	if result.Confidence < 0.7 {
		result.Confidence = 0.7
	}

	// If intent was vague, sharpen it to explain (most follow-ups want elaboration)
	if result.Intent == "question" || result.Intent == "unknown" {
		result.Intent = "explain"
		result.Action = "lookup_knowledge"
	}

	return result
}

// containsDigit returns true if the string contains at least one digit.
func containsDigit(s string) bool {
	for _, r := range s {
		if r >= '0' && r <= '9' {
			return true
		}
	}
	return false
}

// -----------------------------------------------------------------------
// Chain detection — identifies multi-step intent patterns.
// -----------------------------------------------------------------------

// Chain detection regexes compiled once.
var (
	chainSearchAndSaveRe = regexp.MustCompile(
		`(?i)(?:search|find|look up|lookup)\s+(?:for\s+)?(.+?)\s+and\s+(?:save|write|store)\s+(?:it\s+)?(?:to\s+)?(?:a\s+)?(?:file)?`)
	chainSearchAndExplainRe = regexp.MustCompile(
		`(?i)(?:look up|lookup|search|find)\s+(?:for\s+)?(.+?)\s+and\s+(?:explain|summarize|describe)\s+(?:it)?`)
	chainResearchRe = regexp.MustCompile(
		`(?i)^(?:research|investigate|deep dive into|explore)\s+(.+)`)
	chainSummarizeFromWebRe = regexp.MustCompile(
		`(?i)(?:summarize|summarise)\s+(.+?)\s+from\s+(?:the\s+)?(?:web|internet|online)`)
)

// detectChain returns the chain_type if the input matches a multi-step pattern,
// or empty string if no chain is detected.
func (n *NLU) detectChain(lower string) string {
	// "search X and save it" / "find X and write to file"
	if chainSearchAndSaveRe.MatchString(lower) {
		return "search_and_save"
	}

	// "look up X and explain it" / "search X and summarize it"
	if chainSearchAndExplainRe.MatchString(lower) {
		return "search_and_explain"
	}

	// "research X" / "investigate X" / "deep dive into X"
	if chainResearchRe.MatchString(lower) {
		return "research_and_write"
	}

	// "summarize X from the web"
	if chainSummarizeFromWebRe.MatchString(lower) {
		return "search_and_explain"
	}

	return ""
}

// isDocGeneration returns true if the input asks for document creation.
func (n *NLU) isDocGeneration(lower string) bool {
	docPatterns := []string{
		"create a document about ",
		"create a report about ",
		"create a report on ",
		"create a document on ",
		"write a document about ",
		"write a report about ",
		"write a report on ",
		"write a document on ",
		"generate a document about ",
		"generate a report about ",
		"generate a report on ",
		"make a document about ",
		"make a report about ",
		"draft a document about ",
		"draft a report about ",
	}
	for _, p := range docPatterns {
		if strings.Contains(lower, p) {
			return true
		}
	}
	return false
}

// extractChainTopic extracts the subject/topic from a chain-type query.
func (n *NLU) extractChainTopic(lower string) string {
	// Try each chain regex to extract the topic capture group.
	if m := chainSearchAndSaveRe.FindStringSubmatch(lower); len(m) >= 2 {
		return strings.TrimSpace(m[1])
	}
	if m := chainSearchAndExplainRe.FindStringSubmatch(lower); len(m) >= 2 {
		return strings.TrimSpace(m[1])
	}
	if m := chainResearchRe.FindStringSubmatch(lower); len(m) >= 2 {
		return strings.TrimSpace(m[1])
	}
	if m := chainSummarizeFromWebRe.FindStringSubmatch(lower); len(m) >= 2 {
		return strings.TrimSpace(m[1])
	}

	// Document generation topic extraction.
	docPrefixes := []string{
		"create a document about ", "create a report about ",
		"create a report on ", "create a document on ",
		"write a document about ", "write a report about ",
		"write a report on ", "write a document on ",
		"generate a document about ", "generate a report about ",
		"generate a report on ", "make a document about ",
		"make a report about ", "draft a document about ",
		"draft a report about ",
	}
	for _, p := range docPrefixes {
		if idx := strings.Index(lower, p); idx >= 0 {
			topic := lower[idx+len(p):]
			topic = strings.TrimRight(topic, "?!. ")
			return strings.TrimSpace(topic)
		}
	}

	return n.extractTopicGeneral(lower)
}
