package cognitive

import (
	"fmt"
	"regexp"
	"strings"
)

// -----------------------------------------------------------------------
// Response Quality Gate — catches and fixes low-quality responses before
// they reach the user. This is the last line of defense.
//
// Problems it catches:
//   - Low-value acknowledgments on substantive turns ("I see.", "Noted.")
//   - Tool error leaks ("archive: 'path' argument is required")
//   - Parroting (echoing >40% of user's words back)
//   - Ignored user instructions ("ask me 2 questions", "give exactly 3 bullets")
//   - Sub-minimum-length responses on non-greeting turns
// -----------------------------------------------------------------------

// ResponseGate checks and optionally repairs a response before delivery.
type ResponseGate struct{}

// GateVerdict is the outcome of running a response through the gate.
type GateVerdict struct {
	Pass       bool
	Original   string
	Repaired   string // non-empty if the gate repaired the response
	Violations []string
}

// Check runs all quality checks on a response.
// query is the user's input, response is the candidate output.
func (rg *ResponseGate) Check(query, response, source string) *GateVerdict {
	v := &GateVerdict{Pass: true, Original: response}

	// 1. Tool error leak — never surface raw errors to user
	if isToolErrorLeak(response) {
		v.Pass = false
		v.Violations = append(v.Violations, "tool_error_leak")
		v.Repaired = repairToolError(response, query)
		return v
	}

	// 2. Low-value acknowledgment on a substantive turn
	if isSubstantiveTurn(query) && isLowValueResponse(response) {
		v.Pass = false
		v.Violations = append(v.Violations, "low_value_on_substantive")
		v.Repaired = repairLowValue(query)
		return v
	}

	// 3. Cross-domain contamination — sentences about unrelated topics
	if cleaned := removeCrossDomainSentences(query, response); cleaned != response {
		v.Pass = false
		v.Violations = append(v.Violations, "cross_domain_contamination")
		if strings.TrimSpace(cleaned) != "" {
			v.Repaired = cleaned
		}
		return v
	}

	// 4. Parroting — response echoes too much of the user's input.
	// Skip for tool sources — tools naturally echo the input
	// (calculator echoes the expression, translate echoes the source text).
	if !isToolSource(source) && isParroting(query, response) {
		v.Pass = false
		v.Violations = append(v.Violations, "parroting")
		// Don't repair parroting — let it through but flag it.
		// The caller should try a different generation strategy.
	}

	// 5. Sub-minimum length on non-trivial turns.
	// Skip for tool sources — tools produce intentionally short, factual answers
	// (e.g. "0.25*480 = 120", "Password: xK9#mP2q", "hello → bonjour").
	if isSubstantiveTurn(query) && len(strings.Fields(response)) < 8 &&
		!isAcceptableShort(response) && !isToolSource(source) {
		v.Pass = false
		v.Violations = append(v.Violations, "too_short")
	}

	return v
}

// -----------------------------------------------------------------------
// Tool Error Leak Detection
// -----------------------------------------------------------------------

var toolErrorPatterns = []*regexp.Regexp{
	regexp.MustCompile(`(?i)^\w+ error: .+`),             // "archive error: ..."
	regexp.MustCompile(`(?i)^\w+ unavailable$`),           // "archive unavailable"
	regexp.MustCompile(`(?i)^\w+ not found$`),             // "archive not found"
	regexp.MustCompile(`(?i)argument is required`),        // tool missing arg
	regexp.MustCompile(`(?i)^error: `),                    // bare error prefix
	regexp.MustCompile(`(?i)failed to (?:execute|run|call)`), // execution failure
	regexp.MustCompile(`(?i)timed out after \d+`),         // timeout
}

func isToolErrorLeak(response string) bool {
	trimmed := strings.TrimSpace(response)
	for _, re := range toolErrorPatterns {
		if re.MatchString(trimmed) {
			return true
		}
	}
	return false
}

func repairToolError(response, query string) string {
	// Extract the tool name from the error if possible
	if idx := strings.Index(response, " error:"); idx > 0 && idx < 20 {
		toolName := strings.TrimSpace(response[:idx])
		return fmt.Sprintf("I tried to use %s for that, but I need more specific information. Could you rephrase what you'd like to do?", toolName)
	}
	return "I wasn't able to complete that action. Could you provide more details about what you'd like to do?"
}

// -----------------------------------------------------------------------
// Low-Value Response Detection
// -----------------------------------------------------------------------

// lowValueExact are responses that are never acceptable on substantive turns.
var lowValueExact = map[string]bool{
	"i see":                         true,
	"okay":                          true,
	"got it":                        true,
	"right":                         true,
	"makes sense":                   true,
	"i hear you":                    true,
	"gotcha":                        true,
	"alright":                       true,
	"cool":                          true,
	"sure":                          true,
	"understood":                    true,
	"noted":                         true,
	"clear":                         true,
	"hmm":                           true,
	"interesting":                   true,
	"oh":                            true,
	"wow":                           true,
	"thank you for telling me":      true,
	"i appreciate you sharing that": true,
	"good question":                 true,
	"that's a great question":       true,
}

// lowValuePrefixes are response starts that signal low-value output.
var lowValuePrefixes = []string{
	"i appreciate you sharing",
	"thank you for telling me",
	"that's a great question",
	"good question",
	"i see what you mean",
}

func isLowValueResponse(response string) bool {
	clean := strings.ToLower(strings.TrimRight(strings.TrimSpace(response), "!?."))
	if lowValueExact[clean] {
		return true
	}
	// Check if the entire response is just a low-value prefix with nothing substantive after
	for _, prefix := range lowValuePrefixes {
		if strings.HasPrefix(clean, prefix) {
			after := strings.TrimSpace(clean[len(prefix):])
			if len(strings.Fields(after)) < 5 {
				return true // prefix + less than 5 words of substance = low value
			}
		}
	}
	return false
}

func isSubstantiveTurn(query string) bool {
	lower := strings.ToLower(strings.TrimSpace(query))
	words := strings.Fields(lower)
	// Greetings and single-word inputs are NOT substantive
	if len(words) <= 2 {
		greetings := map[string]bool{
			"hi": true, "hello": true, "hey": true, "yo": true,
			"thanks": true, "thank you": true, "bye": true, "goodbye": true,
			"ok": true, "okay": true, "yes": true, "no": true,
			"sure": true, "cool": true, "alright": true,
		}
		if greetings[lower] {
			return false
		}
	}
	// 3+ words = substantive
	return len(words) >= 3
}

func repairLowValue(query string) string {
	// Generate a response that at least engages with the query
	topic := extractMainTopic(query)
	if topic != "" {
		return fmt.Sprintf("Let me think about %s. What specifically would you like to explore?", topic)
	}
	return "That's worth exploring further. What angle are you most interested in?"
}

func isAcceptableShort(response string) bool {
	// Some short responses are fine: direct answers, confirmations after actions
	lower := strings.ToLower(strings.TrimSpace(response))
	if strings.HasPrefix(lower, "done") || strings.HasPrefix(lower, "saved") ||
		strings.HasPrefix(lower, "set ") || strings.HasPrefix(lower, "created") ||
		strings.HasPrefix(lower, "deleted") || strings.HasPrefix(lower, "updated") {
		return true
	}
	// Greetings are intentionally short — "Hey Rrl.", "Good morning."
	greetingMarkers := []string{"hey ", "hi ", "hello", "good morning", "good afternoon",
		"good evening", "morning", "afternoon", "evening", "yo ", "sup "}
	for _, g := range greetingMarkers {
		if strings.HasPrefix(lower, g) {
			return true
		}
	}
	return false
}

// isToolSource returns true if the source indicates the response came from
// a direct tool execution. Tool outputs are intentionally short and factual
// (calculator, translate, password, timer, weather, etc.) and should not
// be rejected by the length gate.
func isToolSource(source string) bool {
	switch source {
	case "calculator", "computed", "translate", "password", "timer", "weather", "codegen",
		"sysinfo", "clipboard", "notes", "todos", "bookmark", "journal",
		"habit", "expense", "convert", "reminder", "hash", "dict",
		"volume", "brightness", "app", "process", "network",
		"qrcode", "archive", "disk_usage", "calendar", "screenshot",
		"news", "find_files", "run_code", "memory", "safety",
		"planner", "honest_fallback":
		return true
	}
	return false
}

// -----------------------------------------------------------------------
// Parroting Detection
// -----------------------------------------------------------------------

func isParroting(query, response string) bool {
	qWords := gateContentWords(strings.ToLower(query))
	rWords := gateContentWords(strings.ToLower(response))
	if len(qWords) < 4 || len(rWords) < 4 {
		return false // too short to judge
	}

	qSet := make(map[string]bool, len(qWords))
	for _, w := range qWords {
		qSet[w] = true
	}

	overlap := 0
	for _, w := range rWords {
		if qSet[w] {
			overlap++
		}
	}

	// If >40% of response words are from the query, it's parroting
	ratio := float64(overlap) / float64(len(rWords))
	return ratio > 0.40
}

// gateContentWords extracts non-stop-words for parrot comparison.
func gateContentWords(text string) []string {
	stops := map[string]bool{
		"the": true, "a": true, "an": true, "is": true, "are": true,
		"was": true, "were": true, "be": true, "been": true, "being": true,
		"have": true, "has": true, "had": true, "do": true, "does": true,
		"did": true, "will": true, "would": true, "could": true, "should": true,
		"may": true, "might": true, "can": true, "shall": true,
		"i": true, "you": true, "he": true, "she": true, "it": true,
		"we": true, "they": true, "me": true, "him": true, "her": true,
		"us": true, "them": true, "my": true, "your": true, "his": true,
		"its": true, "our": true, "their": true,
		"this": true, "that": true, "these": true, "those": true,
		"in": true, "on": true, "at": true, "to": true, "for": true,
		"of": true, "with": true, "by": true, "from": true, "about": true,
		"as": true, "into": true, "through": true, "during": true,
		"and": true, "but": true, "or": true, "nor": true, "not": true,
		"so": true, "yet": true, "if": true, "then": true, "than": true,
		"what": true, "how": true, "why": true, "when": true, "where": true,
		"who": true, "which": true,
	}

	var out []string
	for _, w := range strings.Fields(text) {
		clean := strings.Trim(w, ".,!?;:'\"()-")
		if len(clean) > 2 && !stops[clean] {
			out = append(out, clean)
		}
	}
	return out
}

// -----------------------------------------------------------------------
// User Instruction Detection
// -----------------------------------------------------------------------

// UserInstruction captures an explicit instruction the user gave.
type UserInstruction struct {
	Type  string // "ask_questions", "give_n_items", "keep_under_n_words", "use_format"
	Count int    // how many (2 questions, 3 bullets, etc.)
	Value string // the instruction value ("formal tone", "bullet points")
}

var askQuestionsRe = regexp.MustCompile(`(?i)ask\s+(?:me\s+)?(\d+)\s+(?:clarifying\s+)?questions?`)
var giveNItemsRe = regexp.MustCompile(`(?i)(?:give|list|provide|name|suggest)\s+(?:me\s+)?(?:exactly\s+)?(\d+)\s+(\w+)`)
var keepUnderRe = regexp.MustCompile(`(?i)(?:keep|limit|under|max(?:imum)?)\s+(?:it\s+)?(?:to\s+)?(\d+)\s+words?`)
var inNBulletsRe = regexp.MustCompile(`(?i)in\s+(\d+)\s+bullet`)
var exactlyNRe = regexp.MustCompile(`(?i)exactly\s+(\d+)\s+(\w+)`)

// DetectInstructions parses explicit meta-instructions from user input.
func DetectInstructions(query string) []UserInstruction {
	var instructions []UserInstruction

	if m := askQuestionsRe.FindStringSubmatch(query); m != nil {
		n := parseSmallInt(m[1])
		if n > 0 && n <= 10 {
			instructions = append(instructions, UserInstruction{Type: "ask_questions", Count: n})
		}
	}

	if m := inNBulletsRe.FindStringSubmatch(query); m != nil {
		n := parseSmallInt(m[1])
		if n > 0 && n <= 20 {
			instructions = append(instructions, UserInstruction{Type: "give_n_items", Count: n, Value: "bullets"})
		}
	}

	if m := giveNItemsRe.FindStringSubmatch(query); m != nil {
		n := parseSmallInt(m[1])
		if n > 0 && n <= 50 {
			instructions = append(instructions, UserInstruction{Type: "give_n_items", Count: n, Value: m[2]})
		}
	}

	if m := exactlyNRe.FindStringSubmatch(query); m != nil {
		n := parseSmallInt(m[1])
		if n > 0 && n <= 50 {
			instructions = append(instructions, UserInstruction{Type: "give_n_items", Count: n, Value: m[2]})
		}
	}

	if m := keepUnderRe.FindStringSubmatch(query); m != nil {
		n := parseSmallInt(m[1])
		if n > 0 {
			instructions = append(instructions, UserInstruction{Type: "keep_under_n_words", Count: n})
		}
	}

	// Format instructions
	lower := strings.ToLower(query)
	if strings.Contains(lower, "bullet point") || strings.Contains(lower, "bullet list") || strings.Contains(lower, "as bullets") {
		instructions = append(instructions, UserInstruction{Type: "use_format", Value: "bullets"})
	}
	if strings.Contains(lower, "formal tone") || strings.Contains(lower, "formally") {
		instructions = append(instructions, UserInstruction{Type: "use_format", Value: "formal"})
	}
	if strings.Contains(lower, "casual tone") || strings.Contains(lower, "casually") {
		instructions = append(instructions, UserInstruction{Type: "use_format", Value: "casual"})
	}

	return instructions
}

// ValidateInstructions checks if a response satisfies detected instructions.
func ValidateInstructions(response string, instructions []UserInstruction) []string {
	var violations []string
	for _, inst := range instructions {
		switch inst.Type {
		case "ask_questions":
			qCount := strings.Count(response, "?")
			if qCount < inst.Count {
				violations = append(violations, fmt.Sprintf("asked %d questions, user wanted %d", qCount, inst.Count))
			}
		case "give_n_items":
			if inst.Value == "bullets" {
				bulletCount := countBullets(response)
				if bulletCount < inst.Count {
					violations = append(violations, fmt.Sprintf("gave %d bullets, user wanted %d", bulletCount, inst.Count))
				}
			}
		case "keep_under_n_words":
			wordCount := len(strings.Fields(response))
			if wordCount > inst.Count {
				violations = append(violations, fmt.Sprintf("used %d words, user wanted under %d", wordCount, inst.Count))
			}
		}
	}
	return violations
}

// GenerateQuestions produces N clarifying questions about a topic.
func GenerateQuestions(query string, n int) string {
	// Strip instruction suffixes before extracting topic.
	cleaned := query
	for _, suffix := range []string{
		". ask me ", ". give me ", ". keep it ",
		". please ask ", ". first ask ",
	} {
		if idx := strings.Index(strings.ToLower(cleaned), suffix); idx > 0 {
			cleaned = cleaned[:idx]
		}
	}
	topic := extractMainTopic(cleaned)
	if topic == "" {
		topic = "this"
	}

	// Question templates — these produce genuinely useful clarifying questions
	templates := []string{
		"What specific aspect of %s are you most interested in?",
		"What's your current level of familiarity with %s?",
		"Are you looking for a practical or theoretical understanding of %s?",
		"Is there a specific problem you're trying to solve with %s?",
		"What context are you working in — is this for work, learning, or personal interest?",
		"Do you want a broad overview or a deep dive into a particular area of %s?",
		"Are there any constraints or requirements I should know about?",
		"What have you already tried or considered regarding %s?",
		"Is there a specific comparison or tradeoff you're evaluating?",
		"What would a useful answer look like to you?",
	}

	var questions []string
	for i := 0; i < n && i < len(templates); i++ {
		q := templates[i]
		if strings.Contains(q, "%s") {
			q = fmt.Sprintf(q, topic)
		}
		questions = append(questions, fmt.Sprintf("%d. %s", i+1, q))
	}
	return strings.Join(questions, "\n")
}

func countBullets(text string) int {
	count := 0
	for _, line := range strings.Split(text, "\n") {
		trimmed := strings.TrimSpace(line)
		if strings.HasPrefix(trimmed, "- ") || strings.HasPrefix(trimmed, "• ") ||
			strings.HasPrefix(trimmed, "* ") ||
			(len(trimmed) > 2 && trimmed[0] >= '1' && trimmed[0] <= '9' && (trimmed[1] == '.' || trimmed[1] == ')')) {
			count++
		}
	}
	return count
}

// removeCrossDomainSentences strips sentences that contain entities clearly
// unrelated to the query topic. This is the nuclear option for cross-domain
// contamination from entity-swapping corpus retrieval.
func removeCrossDomainSentences(query, response string) string {
	queryLower := strings.ToLower(query)
	// Extract key topic words from the query
	topicWords := gateContentWords(queryLower)
	if len(topicWords) == 0 {
		return response
	}

	// Split into sentences and check each for relevance
	sentences := strings.Split(response, ". ")
	var kept []string
	for _, sent := range sentences {
		sentLower := strings.ToLower(sent)
		// Check if this sentence contains a foreign entity that's clearly
		// unrelated (proper nouns not in the query, from a different domain)
		if containsForeignEntity(sentLower, topicWords) {
			continue // drop this sentence
		}
		kept = append(kept, sent)
	}
	if len(kept) == 0 {
		return response // don't return empty
	}
	return strings.Join(kept, ". ")
}

// containsForeignEntity checks if a sentence contains proper nouns/entities
// that are clearly from a different domain than the topic.
func containsForeignEntity(sentLower string, topicWords []string) bool {
	// Known cross-domain contamination markers
	foreignEntities := []string{
		"murakami", "diogenes", "antisthenes", "parmenides", "cynicism",
		"ottoman", "suleymaniye", "byzantine", "genghis", "napoleonic",
		"buffer solutions", "pre-socratic philosopher from elea",
		"abstract mathematical models of computation",
		"prose style", "raymond carver", "vonnegut", "fitzgerald",
		"turing machine", "halting problem",
	}
	for _, fe := range foreignEntities {
		if strings.Contains(sentLower, fe) {
			// Check if it's actually relevant to the topic
			relevant := false
			for _, tw := range topicWords {
				if strings.Contains(fe, tw) || strings.Contains(tw, fe) {
					relevant = true
					break
				}
			}
			if !relevant {
				return true
			}
		}
	}
	return false
}

func parseSmallInt(s string) int {
	n := 0
	for _, c := range s {
		if c >= '0' && c <= '9' {
			n = n*10 + int(c-'0')
		}
	}
	return n
}
