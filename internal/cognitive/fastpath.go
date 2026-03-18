package cognitive

import (
	"fmt"
	"regexp"
	"strings"

	"github.com/artaeon/nous/internal/memory"
	"github.com/artaeon/nous/internal/ollama"
)

// QueryPath indicates how a query should be routed.
type QueryPath string

const (
	// PathFast — single LLM call with minimal context (greetings, thanks, short chat).
	PathFast QueryPath = "fast"
	// PathMedium — single LLM call with conversation history + memory (explanations, discussions).
	PathMedium QueryPath = "medium"
	// PathFull — full cognitive pipeline with tools (file ops, code, search).
	PathFull QueryPath = "full"
)

// FastPathClassifier determines whether a query can be answered with a single
// LLM call (simple) or needs the full cognitive pipeline (complex).
// Classification is purely keyword/pattern based — no LLM call.
type FastPathClassifier struct{}

// complexPatterns match queries that require tool use or multi-step reasoning.
var complexPatterns = []*regexp.Regexp{
	// File and code operations
	regexp.MustCompile(`(?i)\b(read|write|create|delete|edit|open|save|move|rename|copy)\b.*(file|directory|folder|dir|path)`),
	regexp.MustCompile(`(?i)\b(find|search|grep|look for|locate)\b.*(file|code|function|class|error|bug|todo|pattern|log)`),
	regexp.MustCompile(`(?i)\b(run|execute|compile|build|test|deploy|install)\b`),
	regexp.MustCompile(`(?i)\b(git|commits?|push|pull|merge|branch|diff|rebase)\b`),
	regexp.MustCompile(`(?i)\b(list|show|display)\b.*(file|director|folder|process|port)`),

	// Web and network
	regexp.MustCompile(`(?i)\b(fetch|download|curl|http|api|request|scrape|browse|visit)\b`),
	regexp.MustCompile(`(?i)\b(search the web|google|look up online|web search)\b`),

	// Project-specific work
	regexp.MustCompile(`(?i)\b(refactor|debug|fix|implement|add feature|optimize|profile)\b`),
	regexp.MustCompile(`(?i)\b(analyze|scan|index|inspect)\b.*(code|project|repo|codebase)`),
	regexp.MustCompile(`(?i)\b(what does|how does|explain).*(this|the) (code|function|file|module|class|method)\b`),

	// Direct file references (e.g., "read go.mod", "show main.go", "cat README.md")
	regexp.MustCompile(`(?i)\b(read|cat|show|open|view|display)\b\s+\S+\.\w+`),
	regexp.MustCompile(`(?i)\b(grep|search|find)\b\s+\S+`),
	regexp.MustCompile(`(?i)\b(how many|count)\b.*\bfiles?\b`),
	regexp.MustCompile(`(?i)\b(largest|biggest|smallest|newest|oldest)\b.*\bfiles?\b`),
	regexp.MustCompile(`(?i)\bwhat\s+files\b.*\b(?:in|inside|under)\b`),
	regexp.MustCompile(`(?i)\b(?:find|list|show)\s+(?:all\s+)?(?:test|spec|mock|go|py|js|ts|java|rust)\s+files?\b`),

	// Tool invocations
	regexp.MustCompile(`(?i)\b(use|call|invoke|run)\b.*(tool|command|script|shell|bash|terminal)`),
	regexp.MustCompile(`(?i)\b(set|change|update|modify)\b.*(config|setting|preference|environment)`),

	// Multi-step reasoning markers
	regexp.MustCompile(`(?i)\b(step by step|first .* then|plan|schedule|create a|build a|make a)\b.*(project|app|system|workflow)`),
	regexp.MustCompile(`(?i)\b(compare|diff)\b.*(file|version|branch)`),

	// Sandbox / system operations
	regexp.MustCompile(`(?i)\b(docker|container|sandbox|process|kill|restart)\b`),
}

// fastPatterns match queries that are clearly conversational/simple — minimal context needed.
var fastPatterns = []*regexp.Regexp{
	// Greetings (with optional "nous" / name after)
	regexp.MustCompile(`(?i)^(hi|hey|hello|howdy|yo|sup|greetings|good (morning|afternoon|evening)|hola|bonjour|guten tag|hallo)(\s+nous)?[!?.\s]*$`),

	// Thanks / farewell / acknowledgments
	regexp.MustCompile(`(?i)^(thanks?|thank you|thx|bye|goodbye|see ya|ciao|cheers|great|awesome|perfect|nice|cool|ok|okay|got it|understood|noted|sure|yep|yeah|yup|nope|nah|no)[!?.\s]*$`),

	// Yes/no answers
	regexp.MustCompile(`(?i)^(yes|no|y|n|absolutely|definitely|of course|not really|maybe|perhaps)[!?.\s]*$`),

	// NOTE: Introductions ("my name is...") moved to mediumPatterns —
	// they need conversation context to store facts properly.

	// Simple questions about the assistant
	regexp.MustCompile(`(?i)^(who|what) are you[?!.\s]*$`),
	regexp.MustCompile(`(?i)^what('s| is) your name[?!.\s]*$`),
	regexp.MustCompile(`(?i)^(what (tools|capabilities|features) do you have|what can you do)[?!.\s]*$`),
	regexp.MustCompile(`(?i)^(how are you|how('s| is) it going|what('s| is) up)[?!.\s]*$`),

	// Jokes, fun
	regexp.MustCompile(`(?i)^tell me a (joke|story|riddle|fun fact)`),
	regexp.MustCompile(`(?i)^(joke|riddle|fun fact)[!?.\s]*$`),

	// Simple math
	regexp.MustCompile(`(?i)^what('s| is) \d+\s*[\+\-\*\/x×÷]\s*\d+[?\s]*$`),
	regexp.MustCompile(`(?i)^\d+\s*[\+\-\*\/x×÷]\s*\d+\s*[=?]?\s*$`),

	// Translations
	regexp.MustCompile(`(?i)^(translate|how do you say|what is .* in (french|german|spanish|italian|japanese|chinese|korean|portuguese|russian|arabic))`),
}

// mediumPatterns match queries that need conversation context + memory but NOT tools.
var mediumPatterns = []*regexp.Regexp{
	// Follow-up questions
	regexp.MustCompile(`(?i)^(tell me more|go on|continue|what else|and then|elaborate|can you explain|more details)`),
	regexp.MustCompile(`(?i)^(what about|how about|and what|what if)`),

	// Opinion / preference questions
	regexp.MustCompile(`(?i)^(do you (like|think|believe|prefer)|what do you think|how do you feel|what's your (opinion|take|view))`),
	regexp.MustCompile(`(?i)(which (is|one is) better|what would you (recommend|suggest)|pros and cons)`),

	// Explanation requests (not about specific code/files)
	regexp.MustCompile(`(?i)^(explain|what is|what are|what was|what were|how does|how do|how did|why is|why do|why does|why did) [a-zA-Z]`),
	regexp.MustCompile(`(?i)^(define|definition of|meaning of) `),
	regexp.MustCompile(`(?i)^who (is|was|are|were) `),
	regexp.MustCompile(`(?i)^(when|where) (is|was|did|does|do) `),
	regexp.MustCompile(`(?i)^how (old|tall|big|far|long|many|much) `),

	// Simple factual / definitional questions
	regexp.MustCompile(`(?i)^what (is|are|was|were) (a |an |the )?[a-zA-Z\s]{1,40}[?!.\s]*$`),

	// Conversational / discussion
	regexp.MustCompile(`(?i)(can you help me understand|i don't understand|what do you mean|could you clarify)`),
	regexp.MustCompile(`(?i)^(summarize|recap|summary of|overview of|in summary)`),

	// Introductions — need context to store facts
	regexp.MustCompile(`(?i)^(my name is|i'?m |i am |call me |i work (on|at|in|for|with)|i'?m a |i am a )`),

	// Possessive recall — questions referencing prior conversation or personal facts
	regexp.MustCompile(`(?i)\b(my|our|we|us)\b.*(name|project|hobbi|interest|work|job|task|goal)`),
	regexp.MustCompile(`(?i)(?:what'?s|what is|what are|tell me)\s+my\b`),
	regexp.MustCompile(`(?i)\b(do you remember|you remember|recall|did (i|we)|what did (i|we))\b`),
	regexp.MustCompile(`(?i)\b(earlier|before|last time|previously|we (talked|discussed|said|mentioned))\b`),
	regexp.MustCompile(`(?i)\b(do you know|you know)\s+(me|my|about me|who i am)\b`),
}

// simplePatterns is the union of fast + medium for backward-compatible IsSimple.
var simplePatterns = append(append([]*regexp.Regexp{}, fastPatterns...), mediumPatterns...)

// ClassifyQuery returns the routing path for a query: "fast", "medium", or "full".
func (c *FastPathClassifier) ClassifyQuery(query string) QueryPath {
	query = strings.TrimSpace(query)
	if query == "" {
		return PathFull
	}

	// If it matches any complex pattern, route to full pipeline.
	for _, pat := range complexPatterns {
		if pat.MatchString(query) {
			return PathFull
		}
	}

	// Check fast patterns first.
	for _, pat := range fastPatterns {
		if pat.MatchString(query) {
			return PathFast
		}
	}

	// Check medium patterns before short-message heuristic — questions like
	// "what is quantum entanglement" (4 words) need medium, not fast.
	words := strings.Fields(query)
	for _, pat := range mediumPatterns {
		if pat.MatchString(query) {
			return PathMedium
		}
	}

	// Short messages (under 5 words) that didn't match medium patterns are fast.
	if len(words) < 5 {
		return PathFast
	}

	// Short-ish messages (under 8 words) are medium (need some context).
	if len(words) <= 8 {
		return PathMedium
	}

	// Questions (starts with question word or ends with ?) → medium path.
	// These are knowledge questions, not tool requests.
	if strings.HasSuffix(strings.TrimSpace(query), "?") {
		return PathMedium
	}
	lowerFirst := strings.ToLower(words[0])
	questionWords := map[string]bool{
		"what": true, "why": true, "how": true, "when": true, "where": true,
		"who": true, "which": true, "is": true, "are": true, "can": true,
		"could": true, "would": true, "should": true, "do": true, "does": true,
		"did": true, "will": true, "tell": true, "describe": true, "explain": true,
	}
	if questionWords[lowerFirst] {
		return PathMedium
	}

	// Longer conversational messages without tool keywords → medium path.
	// Only route to full pipeline if complex patterns matched (checked above).
	return PathMedium
}

// IsSimple returns true if the query can be handled by the fast path
// (a single LLM call without the full cognitive pipeline).
// Kept for backward compatibility — returns true for both fast and medium.
func (c *FastPathClassifier) IsSimple(query string) bool {
	path := c.ClassifyQuery(query)
	return path == PathFast || path == PathMedium
}

// FastPathResponder handles simple queries with a single LLM call,
// using conversation history for context but skipping the full pipeline.
type FastPathResponder struct {
	LLM              *ollama.Client
	WorkingMem       *memory.WorkingMemory
	LongTermMem      *memory.LongTermMemory
	Knowledge        *KnowledgeVec
	VCtx             *VirtualContext
	Growth           *PersonalGrowth
	ResponseCrystals *ResponseCrystalStore // semantic cache — learns from every LLM response
}

const fastPathSystemPrompt = `You are Nous (νοῦς), a personal AI running fully on the user's machine. Be warm, friendly, and natural. Keep responses brief — 1-3 sentences for simple questions. If you don't know something, say so honestly.`

const mediumPathSystemPrompt = `You are Nous (νοῦς), a personal AI running fully on the user's machine. You know the user, remember past conversations, and grow smarter over time.

RULES:
- Be concise — answer in 2-5 sentences unless the user asks for detail
- For factual questions, use the knowledge context below
- For "how to" questions, give actionable steps
- Never say "I don't have access to" — you have knowledge, memory, and tools
- Be direct, not evasive

Use the context below for relevant answers.`

// Respond generates a response using a single LLM call with conversation context.
// For "fast" path: minimal system prompt + query.
// For "medium" path: richer system prompt with conversation history + memory facts.
func (r *FastPathResponder) Respond(conv *Conversation, query string) (string, error) {
	return r.RespondWithPath(conv, query, PathFast)
}

// quickGreetings maps trivial inputs to instant responses — no LLM call needed.
// This is faster (0ms vs 500ms) and better quality than what tiny models generate.
var quickGreetings = map[string][]string{
	// Greetings
	"hello":            {"Hello! How can I help you today?", "Hey there! What are you working on?", "Hi! Ready to help."},
	"hello there":      {"Hello! What can I do for you?", "Hey there! How can I help?"},
	"hi":               {"Hi! What can I do for you?", "Hey! What are we working on?", "Hello! How can I help?"},
	"hi there":         {"Hi there! What are you working on?", "Hey! How can I help?"},
	"hey":              {"Hey! What's up?", "Hi there! Ready when you are.", "Hey! How can I help?"},
	"hey there":        {"Hey there! What can I do for you?", "Hey! Ready when you are."},
	"yo":               {"Yo! What's the task?", "Hey! What do you need?"},
	"sup":              {"Not much — ready to help! What's up?", "All good here. What do you need?"},
	"what's up":        {"Not much — ready to help! What are you working on?", "All good! What do you need?"},
	"whats up":         {"Not much — ready to help! What are you working on?", "All good! What do you need?"},
	"howdy":            {"Howdy! What can I help with?"},
	"greetings":        {"Greetings! What can I help you with?", "Hello! How can I help?"},
	"hola":             {"Hola! How can I help you today?"},
	"bonjour":          {"Bonjour! What can I do for you?"},
	"good morning":     {"Good morning! What are we tackling today?", "Morning! How can I help?"},
	"morning":          {"Morning! What are we working on today?", "Good morning! How can I help?"},
	"good afternoon":   {"Good afternoon! How can I help?", "Afternoon! What are you working on?"},
	"good evening":     {"Good evening! What are you working on?", "Evening! How can I help?"},
	"good night":       {"Good night! See you next time.", "Night! Happy coding."},

	// Thanks
	"thanks":           {"You're welcome!", "Happy to help!", "Anytime!"},
	"thanks!":          {"You're welcome! Let me know if you need anything else.", "Glad I could help!"},
	"thank you":        {"You're welcome!", "Happy to help! Let me know if there's anything else."},
	"thank you so much": {"You're very welcome! Let me know if there's anything else.", "Happy to help!"},
	"thanks a lot":     {"You're welcome! Glad I could help.", "Anytime!"},
	"thanks so much":   {"You're very welcome!", "Happy to help!"},
	"thx":              {"No problem!", "Anytime!"},
	"ty":               {"You're welcome!", "No problem!"},
	"much appreciated": {"Happy to help! Let me know if you need anything else.", "You're welcome!"},
	"appreciate it":    {"You're welcome! Let me know if there's more.", "Glad I could help!"},

	// Farewells
	"bye":              {"See you later!", "Bye! Happy coding."},
	"goodbye":          {"Goodbye! See you next time.", "Take care!"},
	"see ya":           {"See ya! Happy coding.", "Later!"},
	"see you":          {"See you! Take care.", "Later!"},
	"see you later":    {"See you later! Happy coding.", "Later!"},
	"later":            {"Later! Happy coding.", "See you!"},
	"ciao":             {"Ciao! See you next time.", "Ciao!"},
	"take care":        {"You too! See you later.", "Take care!"},
	"have a good one":  {"You too! See you later.", "Thanks, you too!"},
	"gotta go":         {"See you later! Happy coding.", "Later!"},
	"ttyl":             {"Talk to you later!", "Later!"},
	"peace":            {"Peace! See you next time.", "Later!"},
	"night":            {"Good night! See you next time.", "Night!"},
	"nite":             {"Good night! See you later.", "Night!"},

	// Acknowledgments
	"ok":               {"Got it. Let me know if you need anything.", "Alright!"},
	"okay":             {"Got it. Let me know if you need anything.", "Alright!"},
	"k":                {"Got it. What's next?"},
	"cool":             {"Glad that works! Anything else?"},
	"great":            {"Glad to hear it! Need anything else?"},
	"awesome":          {"Thanks! Let me know if there's more to do."},
	"perfect":          {"Glad that's what you needed!"},
	"nice":             {"Thanks! What's next?"},
	"sweet":            {"Glad it works! What's next?"},
	"wonderful":        {"Glad to hear it! Anything else?"},
	"excellent":        {"Thanks! Let me know if there's more to do."},
	"got it":           {"Good. Ready for the next task.", "Alright, what's next?"},
	"understood":       {"Good. Let me know if you need anything else.", "Ready for the next task."},
	"noted":            {"Got it. What's next?", "Noted. Anything else?"},
	"sure":             {"Alright! What do you need?", "Sure thing! How can I help?"},
	"yep":              {"Got it. What's next?"},
	"yeah":             {"Got it. What do you need?"},
	"yup":              {"Got it. What's next?"},
	"nope":             {"Alright. Let me know if you change your mind.", "Got it."},
	"nah":              {"Alright. Let me know if you need anything.", "Got it."},
	"no worries":       {"All good! Let me know if you need anything.", "No problem!"},
	"all good":         {"Great! Let me know if you need anything.", "Glad to hear it!"},
	"sounds good":      {"Great! Let me know if you need anything.", "Glad we're on the same page."},
	"works for me":     {"Great! What's next?", "Glad that works!"},
	"that's it":        {"Got it. Let me know if you need anything later!", "Alright!"},
	"that's all":       {"Got it! I'm here whenever you need me.", "Alright!"},
	"never mind":       {"No problem. Let me know if you need anything.", "Alright!"},
	"nevermind":        {"No problem. Let me know if you need anything.", "Alright!"},

	// How are you
	"how are you":          {"I'm great, thanks for asking! How can I help?", "Doing well! What are you working on?"},
	"how are you doing":    {"I'm doing great! How can I help you?", "All good! What do you need?"},
	"how's it going":       {"Going well! What can I do for you?", "All good! What are you working on?"},
	"hows it going":        {"Going well! What can I do for you?", "All good! What are you working on?"},
}

// tryQuickResponse returns an instant canned response for trivial queries.
// Returns empty string if no quick response is available.
func tryQuickResponse(query string) string {
	lower := strings.ToLower(strings.TrimRight(strings.TrimSpace(query), "!?."))
	// Strip "nous" suffix — "hi nous", "hello nous", "hey nous" → "hi", "hello", "hey"
	lower = strings.TrimRight(strings.TrimSuffix(lower, " nous"), " ")
	if responses, ok := quickGreetings[lower]; ok {
		// Simple deterministic selection based on query length
		return responses[len(query)%len(responses)]
	}
	return ""
}

// RespondWithPath generates a response using the specified path level.
func (r *FastPathResponder) RespondWithPath(conv *Conversation, query string, path QueryPath) (string, error) {
	// Crystal pre-check: check ResponseCrystals BEFORE any path classification
	// or LLM call. This gives ~1ms responses for previously-answered queries.
	if r.ResponseCrystals != nil {
		if cached, ok := r.ResponseCrystals.Lookup(query); ok {
			conv.User(query)
			conv.Assistant(cached)
			return cached, nil
		}
	}

	// Instant response for trivial greetings — skip LLM entirely.
	if path == PathFast {
		if quick := tryQuickResponse(query); quick != "" {
			conv.User(query)
			conv.Assistant(quick)
			return quick, nil
		}
	}

	var sysPrompt string
	var maxHistory int

	switch path {
	case PathMedium:
		sysPrompt = r.buildMediumPrompt(conv, query)
		maxHistory = 4
	default: // PathFast
		sysPrompt = fastPathSystemPrompt
		maxHistory = 2
	}

	// Build messages: system prompt + conversation history + new query.
	msgs := make([]ollama.Message, 0, maxHistory+2)
	msgs = append(msgs, ollama.Message{Role: "system", Content: sysPrompt})

	// Include recent conversation history (skip any existing system messages).
	convMsgs := conv.Messages()
	start := 0
	if len(convMsgs) > maxHistory {
		start = len(convMsgs) - maxHistory
	}
	for _, m := range convMsgs[start:] {
		if m.Role == "system" {
			continue
		}
		msgs = append(msgs, m)
	}

	// Add the current query.
	msgs = append(msgs, ollama.Message{Role: "user", Content: query})

	// Token limits tuned for CPU inference (~7 tok/s on 1.5b).
	// Small NumCtx = faster prefill. Fast path needs almost no context.
	numPredict := 100
	numCtx := 512
	temp := 0.7
	if path == PathMedium {
		numPredict = 150
		numCtx = 1024
		temp = 0.6
	}

	resp, err := r.LLM.Chat(msgs, &ollama.ModelOptions{
		Temperature: temp,
		NumPredict:  numPredict,
		NumCtx:      numCtx,
	})
	if err != nil {
		return "", err
	}

	answer := strings.TrimSpace(resp.Message.Content)

	// Learn from this response — cache it for future similar queries.
	// Synchronous so the crystal is saved before the next request arrives.
	if r.ResponseCrystals != nil && len(answer) > 20 {
		quality := 0.65
		if path == PathMedium {
			quality = 0.7
		}
		r.ResponseCrystals.Learn(query, answer, quality)
	}

	// Post-generation fact-checking: verify answer against knowledge store.
	// This catches hallucinations AFTER generation but BEFORE the user sees them.
	if path == PathMedium && r.Knowledge != nil {
		fc := NewFactChecker(r.Knowledge)
		answer = fc.Check(answer, query)
	}

	// Record the exchange in conversation history so future messages have context.
	conv.User(query)
	conv.Assistant(answer)

	return answer, nil
}

// RespondStreamWithPath generates a streaming response, calling onToken for each token.
func (r *FastPathResponder) RespondStreamWithPathFull(conv *Conversation, query string, path QueryPath, onToken func(token string, done bool)) (string, error) {
	// Crystal pre-check
	if r.ResponseCrystals != nil {
		if cached, ok := r.ResponseCrystals.Lookup(query); ok {
			conv.User(query)
			conv.Assistant(cached)
			onToken(cached, true)
			return cached, nil
		}
	}

	// Instant greetings
	if path == PathFast {
		if quick := tryQuickResponse(query); quick != "" {
			conv.User(query)
			conv.Assistant(quick)
			onToken(quick, true)
			return quick, nil
		}
	}

	var sysPrompt string
	var maxHistory int

	switch path {
	case PathMedium:
		sysPrompt = r.buildMediumPrompt(conv, query)
		maxHistory = 4
	default:
		sysPrompt = fastPathSystemPrompt
		maxHistory = 2
	}

	msgs := make([]ollama.Message, 0, maxHistory+2)
	msgs = append(msgs, ollama.Message{Role: "system", Content: sysPrompt})

	convMsgs := conv.Messages()
	start := 0
	if len(convMsgs) > maxHistory {
		start = len(convMsgs) - maxHistory
	}
	for _, m := range convMsgs[start:] {
		if m.Role == "system" {
			continue
		}
		msgs = append(msgs, m)
	}
	msgs = append(msgs, ollama.Message{Role: "user", Content: query})

	numPredict2 := 100
	numCtx2 := 512
	temp2 := 0.7
	if path == PathMedium {
		numPredict2 = 150
		numCtx2 = 1024
		temp2 = 0.6
	}

	var fullAnswer strings.Builder
	_, err := r.LLM.ChatStream(msgs, &ollama.ModelOptions{
		Temperature: temp2,
		NumPredict:  numPredict2,
		NumCtx:      numCtx2,
	}, func(token string, done bool) {
		if !done {
			fullAnswer.WriteString(token)
		}
		onToken(token, done)
	})
	if err != nil {
		return "", err
	}

	answer := strings.TrimSpace(fullAnswer.String())

	// Learn from this response — all paths get cached for future instant hits.
	if r.ResponseCrystals != nil && len(answer) > 20 {
		quality := 0.65
		if path == PathMedium {
			quality = 0.7
		}
		r.ResponseCrystals.Learn(query, answer, quality)
	}

	conv.User(query)
	conv.Assistant(answer)

	return answer, nil
}

// buildMediumPrompt creates a richer system prompt with memory facts and knowledge for the medium path.
func (r *FastPathResponder) buildMediumPrompt(conv *Conversation, query string) string {
	var sb strings.Builder
	sb.WriteString(mediumPathSystemPrompt)

	// Knowledge context disabled for CPU speed — embedding search adds 500ms+
	// and knowledge is already captured in response crystals after first query.
	// TODO: re-enable when GPU is available or embeddings are faster.

	// Inject personal growth context
	if r.Growth != nil {
		if ctx := r.Growth.ContextForQuery(query); ctx != "" {
			sb.WriteString("\n\n[Personal Context]\n")
			sb.WriteString(ctx)
		}
	}

	// Inject memory facts if available.
	// Filter out code-indexed entries to prevent knowledge contamination
	// (e.g., code snippets leaking into "capital of france" answers).
	var facts []string

	if r.WorkingMem != nil {
		slots := r.WorkingMem.MostRelevant(5)
		for _, s := range slots {
			facts = append(facts, fmt.Sprintf("- %s: %v", s.Key, s.Value))
		}
	}

	if r.LongTermMem != nil {
		entries := r.LongTermMem.All()
		limit := 5
		if len(entries) < limit {
			limit = len(entries)
		}
		for _, e := range entries[:limit] {
			// Skip code-indexed categories — they contaminate general Q&A
			if isCodeCategory(e.Category) {
				continue
			}
			facts = append(facts, fmt.Sprintf("- [%s] %s: %s", e.Category, e.Key, e.Value))
		}
	}

	if len(facts) > 0 {
		sb.WriteString("\n\n[Memory]\n")
		sb.WriteString(strings.Join(facts, "\n"))
	}

	return sb.String()
}

// isCodeCategory returns true if a memory category contains code-indexed content
// that should not leak into general Q&A responses.
func isCodeCategory(category string) bool {
	lower := strings.ToLower(category)
	codeCategories := []string{"code", "codebase", "index", "function", "symbol", "file", "module", "package", "struct", "class"}
	for _, cc := range codeCategories {
		if strings.Contains(lower, cc) {
			return true
		}
	}
	return false
}
