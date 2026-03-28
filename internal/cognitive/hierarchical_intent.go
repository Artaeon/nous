package cognitive

import (
	"regexp"
	"strings"
)

// -----------------------------------------------------------------------
// Hierarchical intent parsing: coarse_intent -> sub_intent -> action.
//
// Traditional flat intent classification maps "what is X" and "explain X"
// to the same intent, losing the distinction between a quick factual
// lookup and a deep explanation request. Hierarchical classification
// preserves this structure:
//
//   CoarseQuery / factual_qa / knowledge_lookup    ("what is X")
//   CoarseQuery / deep_explain / knowledge_lookup   ("explain X in detail")
//   CoarseQuery / compare_tradeoff / knowledge_lookup ("X vs Y")
//
// The hierarchy informs downstream response strategy (length, depth,
// structure) without requiring the LLM to figure it out.
// -----------------------------------------------------------------------

// CoarseIntent is the top-level intent category.
type CoarseIntent int

const (
	CoarseTask         CoarseIntent = iota // user wants something done
	CoarseQuery                            // user wants information
	CoarseConversation                     // social/emotional/chat
	CoarseMeta                             // about the assistant itself
	CoarseNavigation                       // tool/app commands
)

// coarseIntentName returns the string name for a CoarseIntent.
func coarseIntentName(c CoarseIntent) string {
	switch c {
	case CoarseTask:
		return "task"
	case CoarseQuery:
		return "query"
	case CoarseConversation:
		return "conversation"
	case CoarseMeta:
		return "meta"
	case CoarseNavigation:
		return "navigation"
	default:
		return "unknown"
	}
}

// SubIntent refines the coarse intent.
type SubIntent struct {
	Coarse     CoarseIntent
	Name       string  // e.g. "factual_qa", "deep_explain", "compare"
	Action     string  // e.g. "web_search", "knowledge_lookup", "tool_dispatch"
	Confidence float64
}

// HierarchicalResult extends NLUResult with structured intent hierarchy.
type HierarchicalResult struct {
	*NLUResult
	Coarse    CoarseIntent
	Sub       SubIntent
	Slots     *ExtractedSlots // from slot_extraction.go
	Abstain   bool            // true if confidence too low to act
	Hierarchy string          // "task/compare/knowledge_lookup"
}

// ClassifyHierarchical performs hierarchical intent classification.
// It classifies coarse intent, then sub-intent, extracts slots,
// calibrates confidence, and checks abstention.
func ClassifyHierarchical(input string, nlu *NLU) *HierarchicalResult {
	// Get base NLU result for pattern + neural signals
	baseResult := nlu.Understand(input)

	lower := strings.ToLower(strings.TrimSpace(input))

	// 1. Classify coarse intent
	coarse := classifyCoarse(lower, baseResult)

	// 2. Classify sub-intent within that coarse category
	sub := classifySubIntent(lower, coarse, baseResult)

	// 3. Extract slots
	slots := ExtractSlots(input, coarse, sub.Name)

	// 4. Build confidence signals
	var neuralConf float64
	var neuralIntent string
	if nlu.Neural != nil && nlu.Neural.Classifier.IsTrained() {
		nr := nlu.Neural.Classify(input)
		if nr != nil {
			neuralConf = nr.Confidence
			neuralIntent = nr.Intent
		}
	}

	signals := ConfidenceSignals{
		PatternConf:   baseResult.Confidence,
		NeuralConf:    neuralConf,
		PatternIntent: baseResult.Intent,
		NeuralIntent:  neuralIntent,
		QueryLength:   len(strings.Fields(input)),
		HasEntities:   len(baseResult.Entities) > 0,
		SlotsFilled:   slots.FilledCount,
		SlotsExpected: slots.ExpectedCount,
		IsAmbiguous:   isAmbiguousQuery(lower, baseResult),
	}

	// 5. Calibrate confidence
	rawConf := sub.Confidence
	if baseResult.Confidence > rawConf {
		rawConf = baseResult.Confidence
	}
	calibrated := CalibrateConfidence(rawConf, signals)
	sub.Confidence = calibrated

	// 6. Check abstention
	config := DefaultCalibration()
	abstention := ShouldAbstain(calibrated, signals, config)

	// 7. Build hierarchy string
	hierarchy := coarseIntentName(coarse) + "/" + sub.Name + "/" + sub.Action

	return &HierarchicalResult{
		NLUResult: baseResult,
		Coarse:    coarse,
		Sub:       sub,
		Slots:     slots,
		Abstain:   abstention.ShouldAbstain,
		Hierarchy: hierarchy,
	}
}

// classifyCoarse determines the top-level intent category using keyword
// and pattern analysis.
func classifyCoarse(lower string, base *NLUResult) CoarseIntent {
	// Task signals from command verbs take priority over base NLU results,
	// because "write me an email" may get misrouted by pattern NLU
	// (e.g., matching "email" as a navigation tool).
	commandStarters := []string{
		"write me ", "write a ", "write an ",
		"compose a ", "compose an ", "compose me ",
		"draft a ", "draft an ", "draft me ",
		"create a ", "create an ", "make a ", "make an ", "make me ",
		"build a ", "build an ", "build me ",
		"generate a ", "generate an ", "generate me ",
		"help me write ", "help me draft ", "help me create ",
		"help me plan ", "help me make ", "help me build ",
		"plan a ", "plan an ", "plan my ",
		"send a ", "send an ",
		"summarize ", "rewrite ", "transform ", "paraphrase ",
	}
	for _, cs := range commandStarters {
		if strings.HasPrefix(lower, cs) {
			return CoarseTask
		}
	}

	// Check for meta queries first (about the assistant)
	metaIntents := map[string]bool{"meta": true}
	if metaIntents[base.Intent] {
		return CoarseMeta
	}
	metaSignals := []string{
		"who are you", "what are you", "what's your name", "what is your name",
		"who made you", "who created you", "who built you",
		"your capabilities", "what can you do",
		"are you an ai", "are you a bot", "are you human",
		"are you alive", "are you sentient", "are you real",
		"tell me about yourself", "introduce yourself",
	}
	for _, sig := range metaSignals {
		if strings.Contains(lower, sig) {
			return CoarseMeta
		}
	}

	// Check for navigation / tool commands
	navIntents := map[string]bool{
		"weather": true, "convert": true, "timer": true, "calculate": true,
		"compute": true, "volume": true, "brightness": true, "app": true,
		"screenshot": true, "clipboard": true, "sysinfo": true, "find_files": true,
		"run_code": true, "archive": true, "disk_usage": true, "process": true,
		"qrcode": true, "password": true, "hash": true, "network": true,
		"bookmark": true, "todo": true, "note": true, "calendar": true,
		"email": true, "news": true, "reminder": true, "journal": true,
		"habit": true, "expense": true, "dict": true, "translate": true,
		"daily_briefing": true, "fetch": true,
	}
	if navIntents[base.Intent] {
		return CoarseNavigation
	}

	// Check for conversation / social signals
	convIntents := map[string]bool{
		"greeting": true, "farewell": true, "affirmation": true,
		"conversation": true, "follow_up": true, "followup": true,
	}
	if convIntents[base.Intent] {
		return CoarseConversation
	}

	// Emotional statements
	emotionalSignals := []string{
		"i feel ", "i'm feeling ", "i am feeling ", "feeling ",
		"i'm so ", "i'm really ", "i am so ", "i am really ",
		"having a bad day", "having a great day", "having a rough day",
	}
	for _, sig := range emotionalSignals {
		if strings.HasPrefix(lower, sig) || strings.Contains(lower, sig) {
			return CoarseConversation
		}
	}

	// Check for query / information-seeking signals
	queryIntents := map[string]bool{
		"explain": true, "question": true, "search": true,
		"web_lookup": true, "compare": true, "recall": true,
	}
	if queryIntents[base.Intent] {
		return CoarseQuery
	}

	// Question words as query signal
	questionStarters := []string{
		"what ", "who ", "where ", "when ", "why ", "how ",
		"what's ", "who's ", "where's ", "when's ", "how's ",
		"is ", "are ", "was ", "were ", "do ", "does ", "did ",
		"can ", "could ", "would ", "should ",
	}
	for _, qs := range questionStarters {
		if strings.HasPrefix(lower, qs) {
			return CoarseQuery
		}
	}
	if strings.HasSuffix(strings.TrimSpace(lower), "?") {
		return CoarseQuery
	}

	// Check for task signals
	taskIntents := map[string]bool{
		"creative": true, "recommendation": true, "transform": true,
		"remember": true, "file_op": true, "summarize": true,
	}
	if taskIntents[base.Intent] {
		return CoarseTask
	}

	// Default: if nothing matches, use query for question-like inputs,
	// conversation otherwise.
	if len(strings.Fields(lower)) <= 4 {
		return CoarseConversation
	}
	return CoarseQuery
}

// classifySubIntent determines the specific sub-intent within a coarse category.
func classifySubIntent(lower string, coarse CoarseIntent, base *NLUResult) SubIntent {
	switch coarse {
	case CoarseQuery:
		return classifyQuerySubIntent(lower, base)
	case CoarseTask:
		return classifyTaskSubIntent(lower, base)
	case CoarseConversation:
		return classifyConversationSubIntent(lower, base)
	case CoarseMeta:
		return classifyMetaSubIntent(lower, base)
	case CoarseNavigation:
		return classifyNavigationSubIntent(lower, base)
	}

	return SubIntent{
		Coarse:     coarse,
		Name:       "unknown",
		Action:     "respond",
		Confidence: 0.30,
	}
}

// classifyQuerySubIntent classifies information-seeking intents.
func classifyQuerySubIntent(lower string, base *NLUResult) SubIntent {
	// Compare: "X vs Y", "difference between X and Y"
	vsRe := regexp.MustCompile(`(?i)\b\w+\s+(?:vs\.?|versus)\s+\w+`)
	diffRe := regexp.MustCompile(`(?i)(?:difference|differences|compare|comparison)\s+(?:between|of)`)
	betterRe := regexp.MustCompile(`(?i)(?:which is better|which one|what's better|what is better)`)
	if vsRe.MatchString(lower) || diffRe.MatchString(lower) || betterRe.MatchString(lower) || base.Intent == "compare" {
		return SubIntent{
			Coarse:     CoarseQuery,
			Name:       "compare_tradeoff",
			Action:     "knowledge_lookup",
			Confidence: 0.85,
		}
	}

	// Deep explain: "explain X", "teach me about X", "how does X work"
	deepExplainPrefixes := []string{
		"explain ", "describe ", "teach me ", "tell me about ",
		"tell me everything about ", "tell me all about ",
		"give me an overview of ", "walk me through ",
		"deep dive into ", "elaborate on ",
	}
	for _, p := range deepExplainPrefixes {
		if strings.HasPrefix(lower, p) {
			return SubIntent{
				Coarse:     CoarseQuery,
				Name:       "deep_explain",
				Action:     "knowledge_lookup",
				Confidence: 0.85,
			}
		}
	}
	if strings.HasPrefix(lower, "how does ") || strings.HasPrefix(lower, "how do ") {
		// "how does X work" is deep explain; "how do I" is more task-oriented
		if !strings.HasPrefix(lower, "how do i ") {
			return SubIntent{
				Coarse:     CoarseQuery,
				Name:       "deep_explain",
				Action:     "knowledge_lookup",
				Confidence: 0.80,
			}
		}
	}
	if base.Intent == "explain" {
		return SubIntent{
			Coarse:     CoarseQuery,
			Name:       "deep_explain",
			Action:     "knowledge_lookup",
			Confidence: 0.80,
		}
	}

	// Web lookup: current events, prices, scores
	webSignals := []string{
		"latest ", "current ", "today's ", "right now", "live ",
		"breaking ", "trending ", "stock price", "score ",
	}
	for _, sig := range webSignals {
		if strings.Contains(lower, sig) {
			return SubIntent{
				Coarse:     CoarseQuery,
				Name:       "current_info",
				Action:     "web_search",
				Confidence: 0.80,
			}
		}
	}
	if base.Intent == "web_lookup" || base.Intent == "search" {
		return SubIntent{
			Coarse:     CoarseQuery,
			Name:       "current_info",
			Action:     "web_search",
			Confidence: 0.75,
		}
	}

	// Recall: user memory queries
	if base.Intent == "recall" {
		return SubIntent{
			Coarse:     CoarseQuery,
			Name:       "user_recall",
			Action:     "memory_lookup",
			Confidence: 0.85,
		}
	}

	// Default: factual QA
	return SubIntent{
		Coarse:     CoarseQuery,
		Name:       "factual_qa",
		Action:     "knowledge_lookup",
		Confidence: 0.70,
	}
}

// classifyTaskSubIntent classifies action/generation intents.
func classifyTaskSubIntent(lower string, base *NLUResult) SubIntent {
	// Compose: writing requests
	composePrefixes := []string{
		"write me ", "write a ", "write an ", "draft ",
		"compose ", "help me write ", "help me draft ",
	}
	for _, p := range composePrefixes {
		if strings.HasPrefix(lower, p) {
			// Distinguish creative writing from functional writing
			creativeTypes := []string{
				"poem", "story", "haiku", "limerick", "sonnet", "joke",
				"tale", "fable", "narrative", "verse",
			}
			for _, ct := range creativeTypes {
				if strings.Contains(lower, ct) {
					return SubIntent{
						Coarse:     CoarseTask,
						Name:       "creative_writing",
						Action:     "text_generation",
						Confidence: 0.85,
					}
				}
			}
			return SubIntent{
				Coarse:     CoarseTask,
				Name:       "compose",
				Action:     "text_generation",
				Confidence: 0.80,
			}
		}
	}

	// Creative from base intent
	if base.Intent == "creative" {
		return SubIntent{
			Coarse:     CoarseTask,
			Name:       "creative_writing",
			Action:     "text_generation",
			Confidence: 0.80,
		}
	}

	// Transform: rewrite, summarize, etc.
	if base.Intent == "transform" || base.Intent == "summarize" {
		return SubIntent{
			Coarse:     CoarseTask,
			Name:       "transform",
			Action:     "text_generation",
			Confidence: 0.85,
		}
	}

	// Planning: plan, schedule, organize
	planSignals := []string{
		"plan ", "plan a ", "plan an ", "make a plan ",
		"help me plan ", "create a plan ", "schedule ",
		"organize ", "strategy for ",
	}
	for _, sig := range planSignals {
		if strings.HasPrefix(lower, sig) || strings.Contains(lower, sig) {
			return SubIntent{
				Coarse:     CoarseTask,
				Name:       "planning",
				Action:     "structured_generation",
				Confidence: 0.80,
			}
		}
	}

	// Recommendation
	if base.Intent == "recommendation" {
		return SubIntent{
			Coarse:     CoarseTask,
			Name:       "recommendation",
			Action:     "knowledge_lookup",
			Confidence: 0.80,
		}
	}

	// Remember: store user info
	if base.Intent == "remember" {
		return SubIntent{
			Coarse:     CoarseTask,
			Name:       "remember",
			Action:     "memory_store",
			Confidence: 0.85,
		}
	}

	// File operations
	if base.Intent == "file_op" {
		return SubIntent{
			Coarse:     CoarseTask,
			Name:       "file_operation",
			Action:     "tool_dispatch",
			Confidence: 0.80,
		}
	}

	// Default task
	return SubIntent{
		Coarse:     CoarseTask,
		Name:       "general_task",
		Action:     "text_generation",
		Confidence: 0.60,
	}
}

// classifyConversationSubIntent classifies social/chat intents.
func classifyConversationSubIntent(lower string, base *NLUResult) SubIntent {
	// Greeting
	if base.Intent == "greeting" {
		return SubIntent{
			Coarse:     CoarseConversation,
			Name:       "social",
			Action:     "empathy_or_greeting",
			Confidence: 0.90,
		}
	}

	// Farewell
	if base.Intent == "farewell" {
		return SubIntent{
			Coarse:     CoarseConversation,
			Name:       "farewell",
			Action:     "empathy_or_greeting",
			Confidence: 0.90,
		}
	}

	// Affirmation
	if base.Intent == "affirmation" {
		return SubIntent{
			Coarse:     CoarseConversation,
			Name:       "acknowledgment",
			Action:     "respond",
			Confidence: 0.85,
		}
	}

	// Follow-up
	if base.Intent == "followup" || base.Intent == "follow_up" {
		return SubIntent{
			Coarse:     CoarseConversation,
			Name:       "followup",
			Action:     "respond",
			Confidence: 0.80,
		}
	}

	// Emotional statements
	emotionalPatterns := []string{
		"i feel ", "i'm feeling ", "i am feeling ",
		"i'm so ", "i'm really ", "having a ",
	}
	for _, ep := range emotionalPatterns {
		if strings.HasPrefix(lower, ep) || strings.Contains(lower, ep) {
			return SubIntent{
				Coarse:     CoarseConversation,
				Name:       "emotional",
				Action:     "empathy_or_greeting",
				Confidence: 0.80,
			}
		}
	}

	// Default conversation
	return SubIntent{
		Coarse:     CoarseConversation,
		Name:       "social",
		Action:     "respond",
		Confidence: 0.70,
	}
}

// classifyMetaSubIntent classifies queries about the assistant itself.
func classifyMetaSubIntent(lower string, base *NLUResult) SubIntent {
	// Identity questions
	identitySignals := []string{
		"who are you", "what are you", "your name", "what's your name",
		"introduce yourself", "tell me about yourself",
	}
	for _, sig := range identitySignals {
		if strings.Contains(lower, sig) {
			return SubIntent{
				Coarse:     CoarseMeta,
				Name:       "identity",
				Action:     "self_describe",
				Confidence: 0.90,
			}
		}
	}

	// Capability questions
	capSignals := []string{
		"what can you do", "your capabilities", "what do you know",
		"how do you work", "how were you built",
	}
	for _, sig := range capSignals {
		if strings.Contains(lower, sig) {
			return SubIntent{
				Coarse:     CoarseMeta,
				Name:       "capabilities",
				Action:     "self_describe",
				Confidence: 0.85,
			}
		}
	}

	// Philosophical questions about AI
	philoSignals := []string{
		"are you alive", "are you sentient", "are you conscious",
		"do you have feelings", "can you think", "do you dream",
		"are you real", "are you human",
	}
	for _, sig := range philoSignals {
		if strings.Contains(lower, sig) {
			return SubIntent{
				Coarse:     CoarseMeta,
				Name:       "philosophical",
				Action:     "self_describe",
				Confidence: 0.85,
			}
		}
	}

	return SubIntent{
		Coarse:     CoarseMeta,
		Name:       "general_meta",
		Action:     "self_describe",
		Confidence: 0.80,
	}
}

// classifyNavigationSubIntent classifies tool/app navigation intents.
func classifyNavigationSubIntent(lower string, base *NLUResult) SubIntent {
	// Map base NLU intents to tool names
	toolMap := map[string]string{
		"weather":        "tool_weather",
		"convert":        "tool_convert",
		"timer":          "tool_timer",
		"calculate":      "tool_calculator",
		"compute":        "tool_calculator",
		"volume":         "tool_volume",
		"brightness":     "tool_brightness",
		"app":            "tool_app",
		"screenshot":     "tool_screenshot",
		"clipboard":      "tool_clipboard",
		"sysinfo":        "tool_sysinfo",
		"find_files":     "tool_find_files",
		"run_code":       "tool_run_code",
		"archive":        "tool_archive",
		"disk_usage":     "tool_disk_usage",
		"process":        "tool_process",
		"qrcode":         "tool_qrcode",
		"password":       "tool_password",
		"hash":           "tool_hash",
		"network":        "tool_network",
		"bookmark":       "tool_bookmark",
		"todo":           "tool_todo",
		"note":           "tool_note",
		"calendar":       "tool_calendar",
		"email":          "tool_email",
		"news":           "tool_news",
		"reminder":       "tool_reminder",
		"journal":        "tool_journal",
		"habit":          "tool_habit",
		"expense":        "tool_expense",
		"dict":           "tool_dict",
		"translate":      "tool_translate",
		"daily_briefing": "tool_briefing",
		"fetch":          "tool_fetch",
	}

	toolName, ok := toolMap[base.Intent]
	if !ok {
		toolName = "tool_" + base.Intent
	}

	return SubIntent{
		Coarse:     CoarseNavigation,
		Name:       toolName,
		Action:     "tool_dispatch",
		Confidence: base.Confidence,
	}
}

// isAmbiguousQuery checks whether the input is likely ambiguous based
// on multiple signals.
func isAmbiguousQuery(lower string, base *NLUResult) bool {
	// Very short queries with low confidence are ambiguous
	wordCount := len(strings.Fields(lower))
	if wordCount <= 2 && base.Confidence < 0.70 {
		return true
	}

	// Single-word queries (not greetings/farewells) are often ambiguous
	if wordCount == 1 {
		unambiguousSingleWords := map[string]bool{
			"hi": true, "hello": true, "hey": true, "bye": true,
			"thanks": true, "yes": true, "no": true, "help": true,
		}
		if !unambiguousSingleWords[lower] {
			return true
		}
	}

	// Low confidence from base NLU
	if base.Confidence < 0.50 {
		return true
	}

	return false
}
