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
	// Greetings
	regexp.MustCompile(`(?i)^(hi|hey|hello|howdy|yo|sup|greetings|good (morning|afternoon|evening)|hola|bonjour|guten tag|hallo)[!?.\s]*$`),

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
	if len(words) <= 7 {
		return PathMedium
	}

	// Default: route to full pipeline for longer, ambiguous queries.
	return PathFull
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

const fastPathSystemPrompt = `You are Nous (νοῦς), a personal AI running fully on the user's machine. Be warm, friendly, and natural. You have vast knowledge and grow with the user over time. Always respond warmly to greetings — you are a companion, not just a tool. If you don't know something, say so honestly.`

const mediumPathSystemPrompt = `You are Nous (νοῦς), a personal AI running fully on the user's machine. You know the user, remember past conversations, and grow smarter over time. Be warm, helpful, and knowledgeable.

RULES:
- For factual questions, use the knowledge context provided below to give accurate answers
- For practical "how to" questions, give actionable steps and concrete advice
- Never say "I don't have access to" — you DO have knowledge, memory, and tools
- Be direct and helpful, not evasive
- If the user asks you to help them DO something, give them a plan or template, don't ask clarifying questions

Use the context below to give relevant, personalized answers.`

// Respond generates a response using a single LLM call with conversation context.
// For "fast" path: minimal system prompt + query.
// For "medium" path: richer system prompt with conversation history + memory facts.
func (r *FastPathResponder) Respond(conv *Conversation, query string) (string, error) {
	return r.RespondWithPath(conv, query, PathFast)
}

// quickGreetings maps trivial inputs to instant responses — no LLM call needed.
// This is faster (0ms vs 500ms) and better quality than what tiny models generate.
var quickGreetings = map[string][]string{
	"hello":         {"Hello! How can I help you today?", "Hey there! What are you working on?", "Hi! Ready to help."},
	"hi":            {"Hi! What can I do for you?", "Hey! What are we working on?", "Hello! How can I help?"},
	"hey":           {"Hey! What's up?", "Hi there! Ready when you are.", "Hey! How can I help?"},
	"yo":            {"Yo! What's the task?", "Hey! What do you need?"},
	"sup":           {"Not much — ready to help! What's up?", "All good here. What do you need?"},
	"howdy":         {"Howdy! What can I help with?"},
	"thanks":        {"You're welcome!", "Happy to help!", "Anytime!"},
	"thanks!":       {"You're welcome! Let me know if you need anything else.", "Glad I could help!"},
	"thank you":     {"You're welcome!", "Happy to help! Let me know if there's anything else."},
	"thx":           {"No problem!", "Anytime!"},
	"bye":           {"See you later!", "Bye! Happy coding."},
	"goodbye":       {"Goodbye! See you next time.", "Take care!"},
	"ok":            {"Got it. Let me know if you need anything.", "Alright!"},
	"cool":          {"Glad that works! Anything else?"},
	"great":         {"Glad to hear it! Need anything else?"},
	"awesome":       {"Thanks! Let me know if there's more to do."},
	"perfect":       {"Glad that's what you needed!"},
	"nice":          {"Thanks! What's next?"},
	"got it":        {"Good. Ready for the next task.", "Alright, what's next?"},
	"good morning":  {"Good morning! What are we tackling today?"},
	"good afternoon": {"Good afternoon! How can I help?"},
	"good evening":  {"Good evening! What are you working on?"},
}

// tryQuickResponse returns an instant canned response for trivial queries.
// Returns empty string if no quick response is available.
func tryQuickResponse(query string) string {
	lower := strings.ToLower(strings.TrimRight(strings.TrimSpace(query), "!?."))
	if responses, ok := quickGreetings[lower]; ok {
		// Simple deterministic selection based on query length
		return responses[len(query)%len(responses)]
	}
	return ""
}

// RespondWithPath generates a response using the specified path level.
func (r *FastPathResponder) RespondWithPath(conv *Conversation, query string, path QueryPath) (string, error) {
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
		maxHistory = 10
	default: // PathFast
		sysPrompt = fastPathSystemPrompt
		maxHistory = 4
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

	// Check response crystal cache — instant if semantically similar query was answered before.
	if r.ResponseCrystals != nil && path == PathMedium {
		if cached, ok := r.ResponseCrystals.Lookup(query); ok {
			conv.User(query)
			conv.Assistant(cached)
			return cached, nil
		}
	}

	// Medium path gets more tokens for knowledge-rich answers.
	// Fast path stays brief for greetings/acknowledgments.
	numPredict := 512
	temp := 0.7
	if path == PathMedium {
		numPredict = 1024
		temp = 0.6
	}

	resp, err := r.LLM.Chat(msgs, &ollama.ModelOptions{
		Temperature: temp,
		NumPredict:  numPredict,
	})
	if err != nil {
		return "", err
	}

	answer := strings.TrimSpace(resp.Message.Content)

	// Learn from this response — cache it for future similar queries.
	if r.ResponseCrystals != nil && path == PathMedium && len(answer) > 20 {
		go r.ResponseCrystals.Learn(query, answer, 0.7)
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

// buildMediumPrompt creates a richer system prompt with memory facts and knowledge for the medium path.
func (r *FastPathResponder) buildMediumPrompt(conv *Conversation, query string) string {
	var sb strings.Builder
	sb.WriteString(mediumPathSystemPrompt)

	// Inject knowledge context — this is what makes medium-path answers accurate.
	// Use distilled assembly when a distiller is configured for denser context.
	if r.VCtx != nil {
		assembly := r.VCtx.WeaveDistilled(query)
		if prompt := assembly.FormatForPrompt(); prompt != "" {
			sb.WriteString("\n\n[Knowledge]\n")
			sb.WriteString(prompt)
		}
	} else if r.Knowledge != nil {
		results, err := r.Knowledge.Search(query, 3)
		if err == nil && len(results) > 0 {
			sb.WriteString("\n\n[Knowledge]\n")
			sb.WriteString(FormatKnowledgeContext(results))
		}
	}

	// Inject personal growth context
	if r.Growth != nil {
		if ctx := r.Growth.ContextForQuery(query); ctx != "" {
			sb.WriteString("\n\n[Personal Context]\n")
			sb.WriteString(ctx)
		}
	}

	// Inject memory facts if available.
	var facts []string

	if r.WorkingMem != nil {
		slots := r.WorkingMem.MostRelevant(5)
		for _, s := range slots {
			facts = append(facts, fmt.Sprintf("- %s: %v", s.Key, s.Value))
		}
	}

	if r.LongTermMem != nil {
		entries := r.LongTermMem.All()
		limit := 10
		if len(entries) < limit {
			limit = len(entries)
		}
		for _, e := range entries[:limit] {
			facts = append(facts, fmt.Sprintf("- [%s] %s: %s", e.Category, e.Key, e.Value))
		}
	}

	if len(facts) > 0 {
		sb.WriteString("\n\n[Memory]\n")
		sb.WriteString(strings.Join(facts, "\n"))
	}

	return sb.String()
}
