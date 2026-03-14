package cognitive

import (
	"regexp"
	"strings"

	"github.com/artaeon/nous/internal/ollama"
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
	regexp.MustCompile(`(?i)\b(git|commit|push|pull|merge|branch|diff|rebase)\b`),
	regexp.MustCompile(`(?i)\b(list|show|display)\b.*(file|director|folder|process|port)`),

	// Web and network
	regexp.MustCompile(`(?i)\b(fetch|download|curl|http|api|request|scrape|browse|visit)\b`),
	regexp.MustCompile(`(?i)\b(search the web|google|look up online|web search)\b`),

	// Project-specific work
	regexp.MustCompile(`(?i)\b(refactor|debug|fix|implement|add feature|optimize|profile)\b`),
	regexp.MustCompile(`(?i)\b(analyze|scan|index|inspect)\b.*(code|project|repo|codebase)`),
	regexp.MustCompile(`(?i)\b(what does|how does|explain).*(this|the) (code|function|file|module|class|method)\b`),

	// Tool invocations
	regexp.MustCompile(`(?i)\b(use|call|invoke|run)\b.*(tool|command|script|shell|bash|terminal)`),
	regexp.MustCompile(`(?i)\b(set|change|update|modify)\b.*(config|setting|preference|environment)`),

	// Multi-step reasoning markers
	regexp.MustCompile(`(?i)\b(step by step|first .* then|plan|schedule|create a|build a|make a)\b.*(project|app|system|workflow)`),
	regexp.MustCompile(`(?i)\b(compare|diff)\b.*(file|version|branch)`),

	// Sandbox / system operations
	regexp.MustCompile(`(?i)\b(docker|container|sandbox|process|kill|restart)\b`),
}

// simplePatterns match queries that are clearly conversational/simple.
var simplePatterns = []*regexp.Regexp{
	// Greetings
	regexp.MustCompile(`(?i)^(hi|hey|hello|howdy|yo|sup|greetings|good (morning|afternoon|evening)|hola|bonjour|guten tag|hallo)[!?.\s]*$`),

	// Thanks / farewell
	regexp.MustCompile(`(?i)^(thanks?|thank you|thx|bye|goodbye|see ya|ciao|cheers)[!?.\s]*$`),

	// Simple questions about the assistant
	regexp.MustCompile(`(?i)^(who|what) are you[?!.\s]*$`),
	regexp.MustCompile(`(?i)^what('s| is) your name[?!.\s]*$`),
	regexp.MustCompile(`(?i)^(what (tools|capabilities|features) do you have|what can you do)[?!.\s]*$`),

	// Jokes, fun
	regexp.MustCompile(`(?i)^tell me a (joke|story|riddle|fun fact)`),
	regexp.MustCompile(`(?i)^(joke|riddle|fun fact)[!?.\s]*$`),

	// Simple factual / definitional questions
	regexp.MustCompile(`(?i)^what (is|are|was|were) (a |an |the )?[a-zA-Z\s]{1,40}[?!.\s]*$`),
	regexp.MustCompile(`(?i)^(define|definition of|meaning of) `),
	regexp.MustCompile(`(?i)^who (is|was|are|were) `),
	regexp.MustCompile(`(?i)^(when|where) (is|was|did|does|do) `),
	regexp.MustCompile(`(?i)^how (old|tall|big|far|long|many|much) `),

	// Simple math
	regexp.MustCompile(`(?i)^what('s| is) \d+\s*[\+\-\*\/x×÷]\s*\d+[?\s]*$`),
	regexp.MustCompile(`(?i)^\d+\s*[\+\-\*\/x×÷]\s*\d+\s*[=?]?\s*$`),

	// Opinion / conversational
	regexp.MustCompile(`(?i)^(do you (like|think|believe|prefer)|what do you think|how do you feel|what's your (opinion|take|view))`),

	// Translations
	regexp.MustCompile(`(?i)^(translate|how do you say|what is .* in (french|german|spanish|italian|japanese|chinese|korean|portuguese|russian|arabic))`),

	// Short messages (likely conversational) — under 6 words with no tool-related terms
	regexp.MustCompile(`(?i)^[a-zA-ZäöüÄÖÜßéèêàâîôûçñ\s,!?.'-]{1,50}$`),
}

// IsSimple returns true if the query can be handled by the fast path
// (a single LLM call without the full cognitive pipeline).
func (c *FastPathClassifier) IsSimple(query string) bool {
	query = strings.TrimSpace(query)
	if query == "" {
		return false
	}

	// First check: if it matches any complex pattern, it's NOT simple.
	for _, pat := range complexPatterns {
		if pat.MatchString(query) {
			return false
		}
	}

	// Second check: if it matches a simple pattern, it IS simple.
	for _, pat := range simplePatterns {
		if pat.MatchString(query) {
			return true
		}
	}

	// Third check: short messages (under 8 words) without tool keywords are likely simple.
	words := strings.Fields(query)
	if len(words) <= 7 {
		return true
	}

	// Default: route to full pipeline for longer, ambiguous queries.
	return false
}

// FastPathResponder handles simple queries with a single LLM call,
// using conversation history for context but skipping the full pipeline.
type FastPathResponder struct {
	LLM *ollama.Client
}

const fastPathSystemPrompt = `You are Nous, a helpful AI assistant. Answer the user's message directly and concisely. Be friendly and informative. If you don't know something, say so.`

// Respond generates a response using a single LLM call with conversation context.
func (r *FastPathResponder) Respond(conv *Conversation, query string) (string, error) {
	// Build messages: system prompt + conversation history + new query.
	msgs := make([]ollama.Message, 0, len(conv.Messages())+2)
	msgs = append(msgs, ollama.Message{Role: "system", Content: fastPathSystemPrompt})

	// Include recent conversation history (skip any existing system messages).
	for _, m := range conv.Messages() {
		if m.Role == "system" {
			continue
		}
		msgs = append(msgs, m)
	}

	// Add the current query.
	msgs = append(msgs, ollama.Message{Role: "user", Content: query})

	resp, err := r.LLM.Chat(msgs, &ollama.ModelOptions{
		Temperature: 0.7,
		NumPredict:  512,
	})
	if err != nil {
		return "", err
	}

	answer := strings.TrimSpace(resp.Message.Content)

	// Record the exchange in conversation history so future messages have context.
	conv.User(query)
	conv.Assistant(answer)

	return answer, nil
}
