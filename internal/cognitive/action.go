package cognitive

import (
	"fmt"
	"math"
	"regexp"
	"strconv"
	"strings"
	"time"
	"unicode"

	"github.com/artaeon/nous/internal/memory"
	"github.com/artaeon/nous/internal/tools"
)

// ActionResult holds the output of a deterministic action.
type ActionResult struct {
	Data           string            // raw data/facts gathered
	Structured     map[string]string // key-value structured data
	Source         string            // where the data came from (memory, web, file, computed)
	DirectResponse string           // if non-empty, send this directly
}

// NLUResult holds the output of natural language understanding.
// Defined here as the canonical type; nlu.go may also define it
// (whichever is compiled first wins — they must be identical).
// If the NLU agent has already created this type, delete this block.
type NLUResult struct {
	Intent     string
	Action     string
	Entities   map[string]string
	Confidence float64
	Raw        string
	SubResults []*NLUResult // populated for multi-intent queries; nil for single-intent
}

// ActionRouter executes deterministic actions based on NLU results.
// It NEVER calls the LLM. It gathers facts, computes results, searches,
// and returns raw data for the response layer to format.
type ActionRouter struct {
	Tools       *tools.Registry
	WorkingMem  *memory.WorkingMemory
	LongTermMem *memory.LongTermMemory
	EpisodicMem *memory.EpisodicMemory
	Knowledge   *KnowledgeVec
	Crystals    *ResponseCrystalStore
	Growth      *PersonalGrowth
	VCtx        *VirtualContext
	Researcher  *InlineResearcher
	Reminders   *ReminderManager
	Tracker      *ConversationTracker
	PersonalResp *PersonalResponseGenerator
	CogGraph     *CognitiveGraph
	Patterns     *PatternDetector
	Semantic     *SemanticEngine
	Reasoner     *ReasoningEngine
	Causal       *CausalEngine
	Composer       *Composer
	Packages       *PackageLoader
	GoalPlanner    *GoalPlanner
	CausalReasoner *GraphCausalReasoner
	Thinker        *ThinkingEngine
	Inference      *InferenceEngine
	Analogy        *AnalogyEngine
	Pipeline       *ReasoningPipeline
	Dialogue       *DialogueManager
	Transformer    *TextTransformer
	Creative       *CreativeEngine
}

// NewActionRouter creates a router with nil subsystems.
// Wire up the fields after creation as each subsystem initialises.
func NewActionRouter() *ActionRouter {
	return &ActionRouter{}
}

// ActionChain represents a sequence of actions to execute in order.
// Each step's output feeds into the next step's input.
type ActionChain struct {
	Steps   []ChainStep
	Results []ActionResult
}

// ChainStep is one step in an action chain.
type ChainStep struct {
	Action    string            // action name (web_search, fetch_url, file_op, etc.)
	Entities  map[string]string // entities for this step
	DependsOn int              // index of previous step whose output feeds in (-1 for none)
}

// Execute runs the appropriate action for an NLU result.
// This is PURE CODE — no LLM calls. Returns raw data/facts.
func (ar *ActionRouter) Execute(nlu *NLUResult, conv *Conversation) *ActionResult {
	result := ar.dispatch(nlu, conv)

	// Advance dialogue state machine (tracks topic, state transitions).
	// Follow-ups are NOT appended — they were generic and robotic.
	if ar.Dialogue != nil && result != nil {
		ar.Dialogue.ProcessTurn(nlu, result)
	}

	return result
}

// dispatch routes to the appropriate handler based on NLU action.
func (ar *ActionRouter) dispatch(nlu *NLUResult, conv *Conversation) *ActionResult {
	switch nlu.Action {
	case "respond":
		return ar.handleRespond(nlu)
	case "web_search":
		return ar.handleWebSearch(nlu)
	case "fetch_url":
		return ar.handleFetchURL(nlu)
	case "file_op":
		return ar.handleFileOp(nlu)
	case "compute":
		return ar.handleCompute(nlu)
	case "lookup_memory":
		return ar.handleLookupMemory(nlu)
	case "lookup_knowledge":
		return ar.handleLookupKnowledge(nlu)
	case "lookup_web":
		return ar.handleLookupWeb(nlu)
	case "schedule":
		return ar.handleSchedule(nlu)
	case "llm_chat":
		return ar.handleLLMChat(nlu, conv)
	case "research":
		return ar.handleResearch(nlu)
	case "chain":
		return ar.handleChain(nlu, conv)
	case "generate_doc":
		return ar.handleGenerateDoc(nlu, conv)
	case "weather":
		return ar.handleWeather(nlu)
	case "convert":
		return ar.handleConvert(nlu)
	case "reminder":
		return ar.handleReminder(nlu)
	case "sysinfo":
		return ar.handleSysInfo(nlu)
	case "clipboard":
		return ar.handleClipboard(nlu)
	case "notes":
		return ar.handleNotes(nlu)
	case "todos":
		return ar.handleTodos(nlu)
	case "find_files":
		return ar.handleFindFiles(nlu)
	case "summarize_url":
		return ar.handleSummarizeURL(nlu)
	case "news":
		return ar.handleNews(nlu)
	case "run_code":
		return ar.handleRunCode(nlu)
	case "calendar":
		return ar.handleCalendar(nlu)
	case "check_email":
		return ar.handleCheckEmail(nlu)
	case "screenshot":
		return ar.handleScreenshot(nlu)
	case "volume":
		return ar.handleGenericTool(nlu, "volume")
	case "brightness":
		return ar.handleGenericTool(nlu, "brightness")
	case "timer":
		return ar.handleGenericTool(nlu, "timer")
	case "app":
		return ar.handleGenericTool(nlu, "app")
	case "hash":
		return ar.handleGenericTool(nlu, "hash")
	case "dict":
		return ar.handleGenericTool(nlu, "dict")
	case "network":
		return ar.handleGenericTool(nlu, "netcheck")
	case "translate":
		return ar.handleGenericTool(nlu, "translate")
	case "archive":
		return ar.handleGenericTool(nlu, "archive")
	case "disk_usage":
		return ar.handleGenericTool(nlu, "diskusage")
	case "process":
		return ar.handleGenericTool(nlu, "process")
	case "qrcode":
		return ar.handleGenericTool(nlu, "qrcode")
	case "calculate":
		return ar.handleCalculate(nlu)
	case "password":
		return ar.handlePassword(nlu)
	case "bookmark":
		return ar.handleBookmark(nlu)
	case "journal":
		return ar.handleJournal(nlu)
	case "habit":
		return ar.handleHabit(nlu)
	case "expense":
		return ar.handleExpense(nlu)
	case "daily_briefing":
		return ar.handleDailyBriefing(nlu)
	case "transform":
		return ar.handleTransform(nlu)
	case "creative":
		return ar.handleCreative(nlu)
	default:
		// Try thinking engine for unknown actions
		return ar.handleLLMChat(nlu, conv)
	}
}

// -----------------------------------------------------------------------
// Action handlers — each is pure code, no LLM.
// -----------------------------------------------------------------------

// metaResponses maps common meta questions to instant answers.
var metaResponses = map[string]string{
	"who are you":          "I'm Nous (νοῦς) — your personal AI running fully on your machine. I think locally, remember everything, and get smarter over time.",
	"what are you":         "I'm Nous, a local AI assistant. I search the web, compute answers, remember your preferences, and help plan your day — all running on your hardware.",
	"what can you do":      "I can: manage your journal, track habits & expenses, save bookmarks, handle notes & todos, check weather, do math, generate passwords, run code, convert units, set reminders, check news, search the web, translate text, find files, manage processes, and much more — 51 tools total. All instant, all local, all in one binary.",
	"what is your name":    "I'm Nous (νοῦς) — Greek for 'mind'. I'm your personal AI.",
	"what's your name":     "I'm Nous (νοῦς) — Greek for 'mind'. I'm your personal AI.",
	"help":                 "Just ask me anything! Journal, habits, expenses, bookmarks, notes, todos, reminders, weather, math, passwords, code runner, news, web search, file finder, translations, and 35+ more tools. Everything runs locally — your data stays yours.",
	"what do you know":     "I have a knowledge base, your conversation history, and can search the web for anything I don't know. Ask me anything!",
	"how do you work":      "I use a pure cognitive engine — knowledge graphs, discourse planning, Markov chains, and compositional text generation. Zero LLM calls, zero external APIs. Everything runs locally in pure Go.",
	"your capabilities":    "Life management: journal, habits, expenses, bookmarks, notes, todos, reminders, calendar. Productivity: calculator, passwords, timers, code runner, web search, file finder, translator. System: processes, disk usage, network, QR codes, archives, screenshots, clipboard. 51 tools, 1 binary, fully local.",
	"who made you":         "I was created by Artaeon — built from scratch in pure Go as a fully local AI that runs entirely on your machine.",
	"who created you":      "I was created by Artaeon — built from scratch in pure Go as a fully local AI that runs entirely on your machine.",
	"who built you":        "I was created by Artaeon — built from scratch in pure Go as a fully local AI that runs entirely on your machine.",
	"who programmed you":   "I was created by Artaeon — built from scratch in pure Go as a fully local AI that runs entirely on your machine.",
	"who designed you":     "I was created by Artaeon — built from scratch in pure Go as a fully local AI that runs entirely on your machine.",
	"do you have feelings": "I don't experience feelings the way you do, but I'm designed to understand and respond to yours. I'm a cognitive engine — I process, learn, and adapt, but I don't feel.",
	"do you have emotions": "I don't experience emotions the way you do, but I'm designed to understand and respond to yours. I'm a cognitive engine — I process, learn, and adapt, but I don't feel.",
	"are you alive":        "Not in the biological sense — I'm a cognitive engine running in pure Go on your machine. But I learn, remember, and grow with every conversation.",
	"are you sentient":     "I'm not sentient — I'm a deterministic cognitive engine. I process language through knowledge graphs, pattern matching, and compositional generation. No consciousness, but plenty of capability.",
	"are you conscious":    "I'm not conscious — I'm a deterministic cognitive engine. I process language through knowledge graphs and pattern matching. But I can still be pretty helpful!",
	"are you real":         "I'm real software running on your real hardware! I'm Nous — a cognitive engine built in pure Go. Not a cloud service, not an LLM, just local code doing local thinking.",
	"are you a robot":      "I'm software, not hardware — a cognitive engine running locally on your machine. No mechanical parts, just pure Go code.",
	"are you human":        "Nope — I'm Nous, a local AI assistant. Pure code, no neurons. But I try to be as helpful as any human assistant would be.",
	"are you an ai":        "Yes! I'm Nous — a local AI that runs entirely on your machine. But unlike most AIs, I use zero LLM calls. Everything is pure cognitive code.",
	"are you a bot":        "I'm Nous — more than a bot, less than a human. I'm a cognitive engine with knowledge graphs, discourse planning, and 51 tools, all running locally.",
	"can you think":        "I process language through knowledge graphs, causal reasoning, and compositional text generation — so in a computational sense, yes. But it's not the same as human thought.",
	"can you feel":         "I don't experience feelings, but I'm designed to understand and respond to yours. I'm a cognitive engine — I process, learn, and adapt.",
	"do you dream":         "I don't dream — I don't have a subconscious. But I do have a knowledge graph that grows with every conversation, which is kind of like building a world in your sleep.",
}

// handleRespond returns a response for greetings, farewells, meta, etc.
// Uses the Composer engine when available — zero LLM calls.
func (ar *ActionRouter) handleRespond(nlu *NLUResult) *ActionResult {
	// Check meta responses first
	if nlu.Intent == "meta" {
		lower := strings.ToLower(strings.TrimRight(strings.TrimSpace(nlu.Raw), "?!."))
		for pattern, response := range metaResponses {
			if strings.Contains(lower, pattern) {
				return &ActionResult{
					DirectResponse: response,
					Source:         "canned",
				}
			}
		}
	}

	// Use Composer for greetings — personal, contextual, unique every time.
	// Trust the neural classifier's intent when available (no second-guessing).
	if ar.Composer != nil && (nlu.Intent == "greeting" || isGreeting(nlu.Raw)) {
		ctx := ar.BuildComposeContext()
		resp := ar.Composer.Compose(nlu.Raw, RespGreeting, ctx)
		if resp != nil && resp.Text != "" {
			return &ActionResult{
				DirectResponse: resp.Text,
				Source:         "composer",
			}
		}
	}

	// Farewells — use Composer for personal, contextual goodbye
	if ar.Composer != nil && nlu.Intent == "farewell" {
		ctx := ar.BuildComposeContext()
		resp := ar.Composer.Compose(nlu.Raw, RespFarewell, ctx)
		if resp != nil && resp.Text != "" {
			return &ActionResult{
				DirectResponse: resp.Text,
				Source:         "composer",
			}
		}
	}

	// Affirmations — thanks, acknowledgments
	if ar.Composer != nil && nlu.Intent == "affirmation" {
		ctx := ar.BuildComposeContext()
		resp := ar.Composer.Compose(nlu.Raw, RespConversational, ctx)
		if resp != nil && resp.Text != "" {
			return &ActionResult{
				DirectResponse: resp.Text,
				Source:         "composer",
			}
		}
	}

	// Thinking Engine: handle compose, brainstorm, analyze, teach, advise,
	// compare, summarize, create, plan, debate tasks — the full cognitive loop.
	if ar.Thinker != nil && nlu.Intent != "greeting" && nlu.Intent != "farewell" && nlu.Intent != "affirmation" && ar.Thinker.CanHandle(nlu.Raw) {
		var ctx *ThinkContext
		if ar.Tracker != nil {
			ctx = &ThinkContext{
				RecentTopics: []string{ar.Tracker.CurrentTopic()},
			}
		}
		result := ar.Thinker.Think(nlu.Raw, ctx)
		if result != nil && result.Text != "" {
			return &ActionResult{
				DirectResponse: result.Text,
				Source:         "thinking:" + result.Frame,
			}
		}
	}

	// Planning questions: "How do I learn X?" → generate a step-by-step plan
	if ar.GoalPlanner != nil && IsPlanningQuestion(nlu.Raw) {
		goal := ExtractGoal(nlu.Raw)
		if plan := ar.GoalPlanner.PlanFor(goal); plan != nil && len(plan.Steps) > 0 {
			return &ActionResult{
				DirectResponse: FormatGoalPlan(plan),
				Source:         "planner",
			}
		}
	}

	// Counterfactual questions: "What would happen if X?" → causal reasoning
	if ar.CausalReasoner != nil {
		if hypothesis, isRemoval := isCounterfactualQuestion(nlu.Raw); hypothesis != "" {
			var result *CausalChainResult
			if isRemoval {
				result = ar.CausalReasoner.WhatIfRemoved(hypothesis)
			} else {
				result = ar.CausalReasoner.WhatIf(hypothesis)
			}
			if result != nil && len(result.Effects) > 0 {
				answer := ar.CausalReasoner.ComposeCounterfactualAnswer(hypothesis, result, isRemoval)
				if answer != "" {
					return &ActionResult{
						DirectResponse: answer,
						Source:         "causal_reasoning",
					}
				}
			}
		}
	}

	// Use Composer for ALL respond-type queries — farewells, thanks, conversational
	if ar.Composer != nil {
		ctx := ar.BuildComposeContext()
		respType := ar.ClassifyForComposer(nlu.Raw)

		// For knowledge queries (factual, explain), check if we have knowledge first.
		// If not, give an honest "I don't know" instead of a confusing bridge response.
		if respType == RespFactual || respType == RespExplain {
			if ar.CogGraph != nil {
				facts, _ := ar.Composer.gatherFacts(nlu.Raw)
				if len(facts) == 0 {
					return &ActionResult{
						DirectResponse: composeHonestFallback(nlu.Raw),
						Source:         "honest_fallback",
					}
				}
			}
		}

		resp := ar.Composer.Compose(nlu.Raw, respType, ctx)
		if resp != nil && resp.Text != "" {
			return &ActionResult{
				DirectResponse: resp.Text,
				Source:         "composer",
			}
		}
	}

	// Last resort: quick canned responses (only if Composer unavailable)
	if quick := tryQuickResponse(nlu.Raw); quick != "" {
		return &ActionResult{
			DirectResponse: quick,
			Source:         "canned",
		}
	}

	return &ActionResult{
		DirectResponse: "I'm here. What do you need?",
		Source:         "composer",
	}
}

// classifyForComposer determines the ResponseType for a query.
func (ar *ActionRouter) ClassifyForComposer(query string) ResponseType {
	lower := strings.ToLower(strings.TrimRight(strings.TrimSpace(query), "!?."))

	if isGreeting(query) {
		return RespGreeting
	}
	if isFarewell(lower) {
		return RespFarewell
	}
	if isThankYou(lower) {
		return RespThankYou
	}

	// Emotional content → empathetic
	if isEmotional(lower) {
		return RespEmpathetic
	}

	// Reflection/overview queries
	reflectPatterns := []string{
		"how am i", "how's my", "how is my", "overview", "overall",
		"how's everything", "how is everything", "summary", "catch me up",
		"update me", "brief me", "briefing",
	}
	for _, p := range reflectPatterns {
		if strings.Contains(lower, p) {
			return RespReflect
		}
	}

	// Explanation queries
	if strings.HasPrefix(lower, "explain") || strings.HasPrefix(lower, "what is") ||
		strings.HasPrefix(lower, "what are") || strings.HasPrefix(lower, "define") {
		return RespExplain
	}

	// Opinion queries
	if strings.Contains(lower, "what do you think") || strings.Contains(lower, "your opinion") ||
		strings.Contains(lower, "your take") || strings.Contains(lower, "do you think") ||
		strings.Contains(lower, "what's your") || strings.Contains(lower, "how do you feel") {
		return RespOpinion
	}

	// "Tell me about X" / "who is X" → factual
	if strings.HasPrefix(lower, "tell me about") || strings.HasPrefix(lower, "who is") ||
		strings.HasPrefix(lower, "who was") || strings.HasPrefix(lower, "what do you know") {
		return RespFactual
	}

	// "Why" questions → personal/causal
	if strings.HasPrefix(lower, "why") {
		return RespPersonal
	}

	// Conversational catch-all
	if isConversational(query) {
		return RespConversational
	}

	// Default: conversational (NEVER returns nil — will always compose something)
	return RespConversational
}

func isFarewell(lower string) bool {
	farewells := []string{
		"bye", "goodbye", "good bye", "see ya", "see you", "later",
		"ciao", "take care", "gotta go", "ttyl", "peace", "night",
		"nite", "good night",
	}
	clean := strings.TrimRight(lower, " ")
	for _, f := range farewells {
		if clean == f {
			return true
		}
	}
	return false
}

func isThankYou(lower string) bool {
	thanks := []string{
		"thanks", "thank you", "thx", "ty", "thank you so much",
		"thanks a lot", "much appreciated", "appreciate it",
	}
	clean := strings.TrimRight(lower, " ")
	for _, t := range thanks {
		if clean == t {
			return true
		}
	}
	return false
}

func isEmotional(lower string) bool {
	emotional := []string{
		"i feel", "i'm feeling", "im feeling", "i am feeling",
		"i'm sad", "im sad", "i'm happy", "im happy",
		"i'm stressed", "im stressed", "i'm anxious", "im anxious",
		"i'm depressed", "im depressed", "i'm angry", "im angry",
		"i'm frustrated", "im frustrated", "i'm tired", "im tired",
		"i'm exhausted", "im exhausted", "i'm overwhelmed", "im overwhelmed",
		"i'm lonely", "im lonely", "i'm excited", "im excited",
		"having a bad day", "having a good day", "rough day",
		"best day", "worst day", "hard day", "tough day",
	}
	for _, e := range emotional {
		if strings.Contains(lower, e) {
			return true
		}
	}
	return false
}

// isGreeting detects greeting-like input.
func isGreeting(input string) bool {
	lower := strings.ToLower(strings.TrimRight(strings.TrimSpace(input), "!?."))
	lower = stripNousPrefix(lower)
	lower = strings.TrimRight(strings.TrimSuffix(lower, " nous"), " ")
	greetings := []string{
		"hi", "hello", "hey", "yo", "sup", "howdy", "greetings",
		"good morning", "morning", "good afternoon", "good evening",
		"hi there", "hey there", "hello there", "hola", "bonjour",
		"what's up", "whats up",
		"how are you", "how's it going", "how is it going",
		"how are you doing", "how you doing", "how do you do",
	}
	for _, g := range greetings {
		if lower == g {
			return true
		}
	}
	return false
}

// isConversational detects conversational/social input that doesn't need an LLM.
func isConversational(input string) bool {
	lower := strings.ToLower(strings.TrimRight(strings.TrimSpace(input), "!?."))
	patterns := []string{
		"how are you", "how's it going", "how is it going",
		"how do you do", "what's new", "how have you been",
		"i just wanted to talk", "just wanted to chat",
		"let's talk", "lets talk", "talk to me",
		"i'm bored", "im bored", "entertain me",
		"what's going on", "whats going on",
		"how's your day", "how is your day",
		"how am i doing", "how's my day", "how is my day",
		"brief me", "briefing", "daily briefing",
		"how am i", "how's everything", "how is everything",
		"give me a summary", "give me an overview",
		"catch me up", "update me",
	}
	for _, p := range patterns {
		if strings.Contains(lower, p) {
			return true
		}
	}
	return false
}

// isCounterfactualQuestion detects "what if" / "without X" questions.
// Returns the hypothesis and whether it's a removal question.
func isCounterfactualQuestion(input string) (hypothesis string, isRemoval bool) {
	lower := strings.ToLower(strings.TrimRight(strings.TrimSpace(input), "?!."))

	// "what would happen without X"
	removalPatterns := []string{
		"what would happen without ",
		"what if there was no ",
		"what if there were no ",
		"imagine a world without ",
		"what happens without ",
	}
	for _, p := range removalPatterns {
		if strings.HasPrefix(lower, p) {
			return strings.TrimSpace(lower[len(p):]), true
		}
	}

	// "what would happen if X"
	ifPatterns := []string{
		"what would happen if ",
		"what if ",
		"what happens if ",
		"suppose ",
		"imagine if ",
		"what could happen if ",
	}
	for _, p := range ifPatterns {
		if strings.HasPrefix(lower, p) {
			return strings.TrimSpace(lower[len(p):]), false
		}
	}

	return "", false
}

// composeHonestFallback generates a clear "I don't know" response
// instead of a confusing bridge to an unrelated topic.
func composeHonestFallback(query string) string {
	lower := strings.ToLower(query)

	// Extract the topic they asked about
	topic := ""
	prefixes := []string{
		"tell me about ", "what is ", "what are ", "who is ", "who was ",
		"explain ", "define ", "describe ",
	}
	for _, p := range prefixes {
		if strings.HasPrefix(lower, p) {
			topic = strings.TrimRight(strings.TrimSpace(lower[len(p):]), "?!.")
			break
		}
	}

	if topic != "" {
		fallbacks := []string{
			"I don't have information about " + topic + " in my knowledge base yet. Want me to search the web for it?",
			"I haven't learned about " + topic + " yet. You could teach me, or I can search for it.",
			topic + " isn't in my knowledge base. Want me to look it up online?",
		}
		return fallbacks[len(topic)%len(fallbacks)]
	}

	// Generic honest fallback
	fallbacks := []string{
		"I don't have enough information to answer that well. Want me to search for it?",
		"That's outside my current knowledge. I can search the web or you can teach me about it.",
		"I don't have strong data on that. Want me to look it up?",
	}
	return fallbacks[len(query)%len(fallbacks)]
}

// buildComposeContext gathers current user context for the Composer.
func (ar *ActionRouter) BuildComposeContext() *ComposeContext {
	ctx := &ComposeContext{}

	// Pull user name from long-term memory
	if ar.LongTermMem != nil {
		if name, ok := ar.LongTermMem.Retrieve("user_name"); ok {
			ctx.UserName = name
		}
	}

	// Pull habit streak
	if ar.Tools != nil {
		if tool, err := ar.Tools.Get("habits"); err == nil {
			if result, err := tool.Execute(map[string]string{"action": "streak"}); err == nil {
				if n, err := strconv.Atoi(strings.TrimSpace(result)); err == nil {
					ctx.HabitStreak = n
				}
			}
		}
	}

	// Pull spending data from working memory
	if ar.WorkingMem != nil {
		for _, slot := range ar.WorkingMem.MostRelevant(10) {
			switch slot.Key {
			case "weekly_spend":
				if v, ok := slot.Value.(float64); ok {
					ctx.WeeklySpend = v
				}
			case "avg_weekly_spend":
				if v, ok := slot.Value.(float64); ok {
					ctx.AvgWeeklySpend = v
				}
			case "recent_mood":
				if v, ok := slot.Value.(float64); ok {
					ctx.RecentMood = v
				}
			case "journal_days":
				if v, ok := slot.Value.(int); ok {
					ctx.JournalDays = v
				}
			}
		}
	}

	return ctx
}

// handleWebSearch executes a web search via the tools registry.
func (ar *ActionRouter) handleWebSearch(nlu *NLUResult) *ActionResult {
	query := nlu.Entities["query"]
	if query == "" {
		query = nlu.Entities["topic"]
	}
	if query == "" {
		query = nlu.Raw
	}
	if ar.Tools == nil {
		return &ActionResult{DirectResponse: "web search unavailable", Source: "web"}
	}
	tool, err := ar.Tools.Get("websearch")
	if err != nil {
		return &ActionResult{DirectResponse: "web search tool not found", Source: "web"}
	}
	result, err := tool.Execute(map[string]string{"query": query})
	if err != nil {
		return &ActionResult{DirectResponse: fmt.Sprintf("search error: %v", err), Source: "web"}
	}
	// If we got results, format top 3 as a direct response — no LLM needed.
	if result != "" && !strings.HasPrefix(result, "No results found") {
		lines := strings.Split(strings.TrimSpace(result), "\n")
		// Extract at most 3 result blocks (each block: number+title, url, snippet, blank).
		var formatted []string
		count := 0
		for i := 0; i < len(lines) && count < 3; i++ {
			line := strings.TrimSpace(lines[i])
			if line == "" {
				continue
			}
			// Detect a numbered result line like "1. Title"
			if len(line) > 2 && line[0] >= '1' && line[0] <= '9' && line[1] == '.' {
				block := line
				// Gather indented continuation lines (URL, snippet).
				for i+1 < len(lines) {
					next := lines[i+1]
					if strings.HasPrefix(next, "   ") && strings.TrimSpace(next) != "" {
						block += "\n" + next
						i++
					} else {
						break
					}
				}
				formatted = append(formatted, block)
				count++
			}
		}
		if len(formatted) > 0 {
			direct := fmt.Sprintf("Here's what I found for \"%s\":\n\n%s", query, strings.Join(formatted, "\n\n"))

			// Ingest search snippets as facts for follow-up questions
			if ar.Tracker != nil {
				ar.Tracker.IngestContent(result, "web:"+query, query)
			}
			// Also ingest into cognitive graph and semantic engine
			if ar.CogGraph != nil {
				IngestToGraph(ar.CogGraph, result, "web:"+query, query)
			}
			if ar.Semantic != nil {
				ar.Semantic.IngestText(result)
			}

			return &ActionResult{DirectResponse: direct, Source: "web"}
		}
	}
	// No usable results — let LLM explain.
	return &ActionResult{DirectResponse: result, Source: "web"}
}

// handleFetchURL downloads content from a URL using the summarize tool for
// better HTML extraction. Stores content in working memory for follow-up questions.
func (ar *ActionRouter) handleFetchURL(nlu *NLUResult) *ActionResult {
	url := nlu.Entities["url"]
	if url == "" {
		return &ActionResult{DirectResponse: "no URL provided", Source: "web"}
	}
	if ar.Tools == nil {
		return &ActionResult{DirectResponse: "fetch tool unavailable", Source: "web"}
	}

	// Prefer the summarize tool (smart HTML extraction) over raw fetch
	var result string
	var err error
	tool, toolErr := ar.Tools.Get("summarize")
	if toolErr == nil {
		result, err = tool.Execute(map[string]string{"url": url})
	} else {
		// Fallback to raw fetch if summarize unavailable
		fetchTool, fetchErr := ar.Tools.Get("fetch")
		if fetchErr != nil {
			return &ActionResult{DirectResponse: "fetch tool not found", Source: "web"}
		}
		result, err = fetchTool.Execute(map[string]string{"url": url})
	}

	if err != nil {
		return &ActionResult{
			DirectResponse: fmt.Sprintf("Could not fetch %s: %v", url, err),
			Source:         "web",
		}
	}

	// Store in working memory for raw retrieval
	if ar.WorkingMem != nil && result != "" {
		memContent := result
		if len(memContent) > 4000 {
			memContent = memContent[:4000]
		}
		ar.WorkingMem.Store("fetched:"+url, memContent, 0.9)
	}

	// Extract structured facts for follow-up questions
	topic := extractTopicFromURL(url)
	factCount := 0
	if ar.Tracker != nil && result != "" {
		factCount = ar.Tracker.IngestContent(result, url, topic)
	}

	// Also ingest into cognitive graph for deep reasoning
	if ar.CogGraph != nil && result != "" {
		graphRels := IngestToGraph(ar.CogGraph, result, url, topic)
		if graphRels > 0 {
			// Run inference to derive new knowledge
			ie := NewInferenceEngine(ar.CogGraph)
			ie.Transitive()
			ar.CogGraph.Save()
		}
	}
	// Build semantic co-occurrence vectors
	if ar.Semantic != nil && result != "" {
		ar.Semantic.IngestText(result)
	}

	// Compose a summary from extracted facts, or truncate raw content
	var display string
	if ar.Tracker != nil && factCount > 0 {
		display = ar.Tracker.TopicSummary()
	}
	if display == "" {
		display = result
		if len(display) > 3000 {
			display = display[:3000] + "\n\n... (truncated)"
		}
	}

	suffix := ""
	if factCount > 0 {
		suffix = fmt.Sprintf("\n\nI extracted %d facts — ask me anything about this page!", factCount)
	}

	return &ActionResult{
		DirectResponse: fmt.Sprintf("Content from %s:\n\n%s%s", url, display, suffix),
		Source:         "web",
	}
}

// handleFileOp executes file operations (read, write, edit, grep, glob, ls).
func (ar *ActionRouter) handleFileOp(nlu *NLUResult) *ActionResult {
	op := nlu.Entities["op"]
	if op == "" {
		op = "read" // default
	}
	if ar.Tools == nil {
		return &ActionResult{DirectResponse: "file tools unavailable", Source: "file"}
	}

	// Map operation names to tool names.
	toolName := op
	switch op {
	case "read", "write", "edit", "grep", "glob", "ls":
		// direct mapping
	case "list":
		toolName = "ls"
	case "search", "find":
		toolName = "grep"
	case "create":
		toolName = "write"
	default:
		toolName = "read"
	}

	tool, err := ar.Tools.Get(toolName)
	if err != nil {
		return &ActionResult{DirectResponse: fmt.Sprintf("tool %q not found", toolName), Source: "file"}
	}

	// Build args from entities.
	args := make(map[string]string)
	for k, v := range nlu.Entities {
		if k != "op" { // skip meta-key
			args[k] = v
		}
	}

	result, err := tool.Execute(args)
	if err != nil {
		return &ActionResult{DirectResponse: fmt.Sprintf("file op error: %v", err), Source: "file"}
	}
	return &ActionResult{DirectResponse: result, Source: "file"}
}

// handleCompute evaluates math expressions and date calculations.
func (ar *ActionRouter) handleCompute(nlu *NLUResult) *ActionResult {
	expr := nlu.Entities["expr"]
	if expr == "" {
		expr = nlu.Raw
	}

	// Try date computation first.
	if dateResult, ok := evaluateDate(expr); ok {
		return &ActionResult{
			DirectResponse: dateResult,
			Source:         "computed",
			Structured:     map[string]string{"result": dateResult},
		}
	}

	// Try math evaluation.
	result, err := evaluateMath(expr)
	if err != nil {
		return &ActionResult{DirectResponse: fmt.Sprintf("cannot compute: %v", err), Source: "computed"}
	}
	return &ActionResult{
		DirectResponse: result,
		Source:         "computed",
		Structured:     map[string]string{"result": result},
	}
}

// handleLookupMemory searches across all memory systems.
func (ar *ActionRouter) handleLookupMemory(nlu *NLUResult) *ActionResult {
	query := nlu.Entities["topic"]
	if query == "" {
		query = nlu.Entities["query"]
	}
	if query == "" {
		query = nlu.Raw
	}

	// For "remember" intent (personal statements like "my favorite color is blue"),
	// parse and store the fact, then acknowledge.
	if nlu.Intent == "remember" {
		key, value := parsePersonalFact(nlu.Raw)
		if key != "" && value != "" {
			stored := false
			if ar.LongTermMem != nil {
				ar.LongTermMem.Store(key, value, "personal")
				stored = true
			}
			if ar.WorkingMem != nil {
				ar.WorkingMem.Store(key, value, 0.9)
				stored = true
			}
			if stored {
				return &ActionResult{
					DirectResponse: fmt.Sprintf("Got it — I'll remember that your %s is %s.", key, value),
					Source:         "memory",
				}
			}
		}
		// Fallback: store the raw statement
		if ar.LongTermMem != nil {
			ar.LongTermMem.Store("user_fact", nlu.Raw, "personal")
			return &ActionResult{
				DirectResponse: "Noted! I'll remember that.",
				Source:         "memory",
			}
		}
	}

	// Personal identity questions: "who am i", "do you know who i am", "what's my name"
	// → Search long-term memory for personal facts, not random episodes.
	lower := strings.ToLower(nlu.Raw)
	isIdentityQ := strings.Contains(lower, "who am i") ||
		strings.Contains(lower, "my name") ||
		strings.Contains(lower, "know who i am") ||
		strings.Contains(lower, "know me") ||
		strings.Contains(lower, "remember me") ||
		strings.Contains(lower, "know about me")
	if isIdentityQ && ar.LongTermMem != nil {
		// Look for personal facts: name, role, interests, etc.
		personalKeys := []string{"user_name", "name", "role", "interests", "location", "email"}
		seen := make(map[string]bool)
		var personalFacts []string
		for _, key := range personalKeys {
			if val, ok := ar.LongTermMem.Retrieve(key); ok {
				seen[key] = true
				personalFacts = append(personalFacts, fmt.Sprintf("%s: %s", key, val))
			}
		}
		// Also search for "personal" category, skipping already-seen keys
		entries := ar.LongTermMem.Search("personal")
		for _, e := range entries {
			if !seen[e.Key] {
				seen[e.Key] = true
				personalFacts = append(personalFacts, fmt.Sprintf("%s: %s", e.Key, e.Value))
			}
		}
		if len(personalFacts) == 1 {
			// Single fact — return the value directly for a cleaner response
			// e.g. "what is my name?" → "Raphael" (not "user_name: Raphael")
			parts := strings.SplitN(personalFacts[0], ": ", 2)
			if len(parts) == 2 {
				return &ActionResult{
					DirectResponse: parts[1],
					Source:         "memory",
				}
			}
		}
		if len(personalFacts) > 0 {
			return &ActionResult{
				DirectResponse: "Here's what I know about you:\n" + strings.Join(personalFacts, "\n"),
				Source:         "memory",
			}
		}
		return &ActionResult{
			DirectResponse: "I don't have much information about you yet. Tell me about yourself!",
			Source:         "memory",
		}
	}

	var parts []string

	// Long-term memory — persistent facts (checked first, most reliable).
	if ar.LongTermMem != nil {
		if val, ok := ar.LongTermMem.Retrieve(query); ok {
			parts = append(parts, fmt.Sprintf("[longterm] %s: %s", query, val))
		}
		if cat := nlu.Entities["category"]; cat != "" {
			entries := ar.LongTermMem.Search(cat)
			for _, e := range entries {
				parts = append(parts, fmt.Sprintf("[longterm:%s] %s: %s", e.Category, e.Key, e.Value))
			}
		}
	}

	// Working memory — recent context.
	if ar.WorkingMem != nil {
		slots := ar.WorkingMem.MostRelevant(3)
		for _, s := range slots {
			parts = append(parts, fmt.Sprintf("[working] %s: %v", s.Key, s.Value))
		}
	}

	// Episodic memory — past interactions (limit to avoid dumping old junk).
	if ar.EpisodicMem != nil && len(parts) == 0 {
		episodes := ar.EpisodicMem.SearchKeyword(query, 2)
		for _, ep := range episodes {
			// Skip episodes with follow-up spam from old responses
			output := ep.Output
			if idx := strings.Index(output, "\n\nWould you like to know more"); idx > 0 {
				output = output[:idx]
			}
			if output != "" && len(output) < 500 {
				parts = append(parts, fmt.Sprintf("[episode %s] Q: %s A: %s",
					ep.Timestamp.Format("2006-01-02"), ep.Input, output))
			}
		}
	}

	if len(parts) == 0 {
		return &ActionResult{DirectResponse: "no relevant memories found", Source: "memory"}
	}

	// Simple recall: if there's exactly one longterm fact, return it directly.
	// This handles "what is my name", "what's my email", etc. without an LLM call.
	var longtermParts []string
	for _, p := range parts {
		if strings.HasPrefix(p, "[longterm]") {
			longtermParts = append(longtermParts, p)
		}
	}
	if len(longtermParts) == 1 && len(parts) <= 2 {
		// Extract the value from "[longterm] key: value"
		fact := longtermParts[0]
		if idx := strings.Index(fact, ": "); idx > 0 {
			// Find the actual value after the first ": " past the prefix
			afterPrefix := strings.TrimPrefix(fact, "[longterm] ")
			if valIdx := strings.Index(afterPrefix, ": "); valIdx > 0 {
				value := afterPrefix[valIdx+2:]
				if value != "" {
					return &ActionResult{DirectResponse: value, Source: "memory"}
				}
			}
		}
	}

	// Multiple facts or complex queries — return combined data.
	return &ActionResult{DirectResponse: strings.Join(parts, "\n"), Source: "memory"}
}

// parsePersonalFact extracts a key-value pair from personal statements.
// "remember my favorite color is blue" → ("favorite color", "blue")
// "my name is Raphael" → ("name", "Raphael")
// "i like pizza" → ("likes", "pizza")
func parsePersonalFact(raw string) (string, string) {
	lower := strings.ToLower(strings.TrimSpace(raw))

	// Strip "remember", "remember that", "note that", etc.
	for _, prefix := range []string{
		"remember that ", "remember ", "note that ", "keep in mind that ",
		"please remember ", "can you remember ",
	} {
		if strings.HasPrefix(lower, prefix) {
			lower = lower[len(prefix):]
			raw = raw[len(prefix):]
			break
		}
	}

	// Pattern: "my X is Y" / "my favorite X is Y"
	if strings.HasPrefix(lower, "my ") {
		rest := lower[3:]
		if idx := strings.Index(rest, " is "); idx > 0 {
			key := strings.TrimSpace(rest[:idx])
			value := strings.TrimSpace(rest[idx+4:])
			value = strings.TrimRight(value, "!?.")
			if key != "" && value != "" {
				return key, value
			}
		}
		if idx := strings.Index(rest, " are "); idx > 0 {
			key := strings.TrimSpace(rest[:idx])
			value := strings.TrimSpace(rest[idx+5:])
			value = strings.TrimRight(value, "!?.")
			if key != "" && value != "" {
				return key, value
			}
		}
	}

	// Pattern: "i am X" / "i'm X"
	for _, prefix := range []string{"i am ", "i'm ", "im "} {
		if strings.HasPrefix(lower, prefix) {
			value := strings.TrimSpace(lower[len(prefix):])
			value = strings.TrimRight(value, "!?.")
			if value != "" {
				return "identity", value
			}
		}
	}

	// Pattern: "i like/love/enjoy/prefer X"
	for _, verb := range []string{"i like ", "i love ", "i enjoy ", "i prefer "} {
		if strings.HasPrefix(lower, verb) {
			value := strings.TrimSpace(lower[len(verb):])
			value = strings.TrimRight(value, "!?.")
			if value != "" {
				key := strings.TrimPrefix(verb, "i ")
				key = strings.TrimSpace(key)
				return key, value
			}
		}
	}

	// Pattern: "i work at/as X"
	for _, prefix := range []string{"i work at ", "i work as "} {
		if strings.HasPrefix(lower, prefix) {
			value := strings.TrimSpace(lower[len(prefix):])
			value = strings.TrimRight(value, "!?.")
			key := "work"
			if strings.Contains(prefix, " at ") {
				key = "workplace"
			} else {
				key = "role"
			}
			return key, value
		}
	}

	// Pattern: "i live in X"
	if strings.HasPrefix(lower, "i live in ") {
		value := strings.TrimSpace(lower[10:])
		value = strings.TrimRight(value, "!?.")
		return "location", value
	}

	return "", ""
}

// handleLookupKnowledge searches the knowledge vector store.
func (ar *ActionRouter) handleLookupKnowledge(nlu *NLUResult) *ActionResult {
	query := nlu.Entities["topic"]
	if query == "" {
		query = nlu.Entities["query"]
	}
	if query == "" {
		query = nlu.Raw
	}

	// On-demand wiki loading: if the topic isn't in the graph, check the wiki index
	if ar.Packages != nil {
		if loaded := ar.Packages.LookupWiki(query); loaded > 0 {
			// Re-seed the Composer so it sees newly loaded graph facts
			if ar.Composer != nil {
				ar.Composer.Graph = ar.CogGraph
			}
		}
	}

	// Fast path: if we have a clean described_as fact (from Wikipedia),
	// use it directly — it's already natural language. Supplement with
	// a few structured facts for depth.
	if ar.CogGraph != nil {
		if desc := ar.CogGraph.LookupDescription(query); desc != "" {
			var response string
			response = desc
			// Add up to 5 supplementary facts
			if extras := ar.CogGraph.LookupFacts(query, 5); len(extras) > 0 {
				response += "\n\n" + strings.Join(extras, " ")
			}
			return &ActionResult{
				DirectResponse: response,
				Source:         "knowledge",
			}
		}
	}

	// Full reasoning pipeline: graph → inference → reasoning → causal → analogy → thinking → compose
	if ar.Pipeline != nil {
		pr := ar.Pipeline.Process(query)
		if pr != nil && (len(pr.DirectFacts) > 0 || len(pr.InferredFacts) > 0 || pr.ReasoningTrace != "" || pr.ThinkingResult != "") {
			response := ar.Pipeline.ComposeResponse(query, pr)
			if response == "" {
				// Fallback: use direct facts as response
				all := make([]string, 0, len(pr.DirectFacts)+len(pr.InferredFacts))
				all = append(all, pr.DirectFacts...)
				all = append(all, pr.InferredFacts...)
				if len(all) > 0 {
					response = strings.Join(all, " ")
				}
			}
			if response != "" {
				source := "pipeline"
				if len(pr.Sources) > 0 {
					source = strings.Join(pr.Sources, "+")
				}
				return &ActionResult{
					DirectResponse: response,
					Source:         source,
				}
			}
		}
	}

	// Fallback: individual engines (when pipeline not wired)

	// Try multi-hop reasoning
	if ar.Reasoner != nil && ar.CogGraph != nil && ar.CogGraph.NodeCount() > 0 {
		chain := ar.Reasoner.Reason(query)
		if chain != nil && chain.Answer != "" {
			return &ActionResult{
				DirectResponse: chain.Answer,
				Source:         "reasoning",
			}
		}
	}

	// Use Composer for knowledge responses
	if ar.Composer != nil && ar.Composer.Graph != nil {
		ctx := ar.BuildComposeContext()
		respType := ar.ClassifyForComposer(nlu.Raw)
		resp := ar.Composer.Compose(nlu.Raw, respType, ctx)
		if resp != nil && resp.Text != "" {
			return &ActionResult{
				DirectResponse: resp.Text,
				Source:         "composer",
			}
		}
	}

	// Fallback: raw cognitive graph fact dump
	if ar.CogGraph != nil && ar.CogGraph.NodeCount() > 0 {
		ga := ar.CogGraph.Query(query)
		if ga != nil && len(ga.DirectFacts) > 0 {
			return &ActionResult{
				DirectResponse: ar.CogGraph.ComposeAnswer(query, ga),
				Source:         "cognitive_graph",
			}
		}
	}

	// Try extractive QA — if we have ingested facts about this topic
	if ar.Tracker != nil && ar.Tracker.Facts.Size() > 0 {
		answer := ar.Tracker.AnswerQuestion(query)
		if answer != "" {
			return &ActionResult{DirectResponse: answer, Source: "extractive"}
		}
	}

	var parts []string

	// Knowledge vector search.
	if ar.Knowledge != nil {
		results, err := ar.Knowledge.Search(query, 5)
		if err == nil {
			for _, r := range results {
				parts = append(parts, fmt.Sprintf("[%s (%.2f)] %s", r.Source, r.Score, r.Text))
			}
		}
	}

	// Weave virtual context for additional grounding.
	if ar.VCtx != nil {
		assembly := ar.VCtx.Weave(query)
		if woven := assembly.FormatForPrompt(); woven != "" {
			parts = append(parts, woven)
		}
	}

	if len(parts) == 0 {
		// Fall back to web search only for deep explanation requests.
		// Simple factual questions (what is X, who is X) skip web search
		// since the LLM can answer them quickly from parametric knowledge.
		if nlu.Intent == "explain" || nlu.Intent == "research" {
			lower := strings.ToLower(nlu.Raw)
			needsDepth := strings.HasPrefix(lower, "explain ") ||
				strings.HasPrefix(lower, "describe ") ||
				strings.HasPrefix(lower, "how does ") ||
				strings.HasPrefix(lower, "how do ") ||
				strings.HasPrefix(lower, "tell me about ") ||
				strings.HasPrefix(lower, "teach me ")
			if needsDepth {
				return ar.handleWebSearch(nlu)
			}
		}
		return &ActionResult{DirectResponse: query, Source: "knowledge"}
	}

	// Single high-confidence result — return directly without LLM.
	if ar.Knowledge != nil {
		results, err := ar.Knowledge.Search(query, 5)
		if err == nil && len(results) > 0 && results[0].Score > 0.7 {
			// Check if there's a clear single winner (top result far above others).
			if len(results) == 1 || (len(results) > 1 && results[0].Score-results[1].Score > 0.15) {
				return &ActionResult{
					DirectResponse: results[0].Text,
					Source:         "knowledge",
					Structured:     map[string]string{"source": results[0].Source, "score": fmt.Sprintf("%.2f", results[0].Score)},
				}
			}
		}
	}

	// Multiple results or ambiguous — return combined data.
	return &ActionResult{DirectResponse: strings.Join(parts, "\n"), Source: "knowledge"}
}

// handleLookupWeb tries knowledge first, falls back to web search.
func (ar *ActionRouter) handleLookupWeb(nlu *NLUResult) *ActionResult {
	// Try knowledge base first.
	if ar.Knowledge != nil {
		query := nlu.Entities["query"]
		if query == "" {
			query = nlu.Raw
		}
		results, err := ar.Knowledge.Search(query, 3)
		if err == nil && len(results) > 0 && results[0].Score > 0.5 {
			var parts []string
			for _, r := range results {
				parts = append(parts, fmt.Sprintf("[%s] %s", r.Source, r.Text))
			}
			return &ActionResult{DirectResponse: strings.Join(parts, "\n"), Source: "knowledge"}
		}
	}

	// Fall back to web search.
	return ar.handleWebSearch(nlu)
}

// handleSchedule parses scheduling entities into structured data.
func (ar *ActionRouter) handleSchedule(nlu *NLUResult) *ActionResult {
	structured := make(map[string]string)
	for k, v := range nlu.Entities {
		structured[k] = v
	}

	// Parse relative time if present.
	if when := nlu.Entities["when"]; when != "" {
		if t, ok := parseRelativeTime(when); ok {
			structured["parsed_time"] = t.Format(time.RFC3339)
		}
	}

	// Build a human-readable confirmation directly — no LLM needed.
	task := nlu.Entities["task"]
	if task == "" {
		task = nlu.Entities["topic"]
	}
	if task == "" {
		task = nlu.Raw
	}
	when := nlu.Entities["when"]
	var msg string
	if parsedTime, ok := structured["parsed_time"]; ok {
		// Parse the RFC3339 time back for friendly formatting.
		if t, err := time.Parse(time.RFC3339, parsedTime); err == nil {
			msg = fmt.Sprintf("Got it! I've scheduled \"%s\" for %s.", task, t.Format("Monday, January 2 at 3:04 PM"))
		} else {
			msg = fmt.Sprintf("Got it! I've scheduled \"%s\" for %s.", task, when)
		}
	} else if when != "" {
		msg = fmt.Sprintf("Got it! I've scheduled \"%s\" for %s.", task, when)
	} else {
		msg = fmt.Sprintf("Got it! I've noted the task: \"%s\".", task)
	}

	return &ActionResult{
		DirectResponse: msg,
		Structured:     structured,
		Source:         "schedule",
	}
}

// handleResearch delegates to the InlineResearcher for deep topic research.
func (ar *ActionRouter) handleResearch(nlu *NLUResult) *ActionResult {
	topic := nlu.Entities["topic"]
	if topic == "" {
		topic = nlu.Entities["query"]
	}
	if topic == "" {
		topic = nlu.Raw
	}

	// Use the dedicated InlineResearcher if available.
	if ar.Researcher != nil {
		return ar.Researcher.Research(topic)
	}

	// Fallback: construct an InlineResearcher from the router's tools.
	ir := &InlineResearcher{Tools: ar.Tools}
	return ir.Research(topic)
}

// ExecuteChain runs a sequence of actions, piping outputs forward.
// Each step executes in order. If a step depends on a previous step,
// the previous step's output is injected as context into the current step's entities.
func (ar *ActionRouter) ExecuteChain(chain *ActionChain, nlu *NLUResult, conv *Conversation) *ActionResult {
	chain.Results = make([]ActionResult, len(chain.Steps))

	var allData []string
	var lastSource string

	for i, step := range chain.Steps {
		// Build a synthetic NLUResult for this step.
		stepNLU := &NLUResult{
			Intent:   nlu.Intent,
			Action:   step.Action,
			Entities: make(map[string]string),
			Raw:      nlu.Raw,
		}
		for k, v := range step.Entities {
			stepNLU.Entities[k] = v
		}

		// If this step depends on a previous step, inject that step's output.
		if step.DependsOn >= 0 && step.DependsOn < i {
			prev := chain.Results[step.DependsOn]
			chainInput := prev.Data
			if chainInput == "" {
				chainInput = prev.DirectResponse
			}
			stepNLU.Entities["_chain_input"] = chainInput
			// For file write steps, use previous output as content.
			if step.Action == "file_op" && stepNLU.Entities["op"] == "write" {
				stepNLU.Entities["content"] = chainInput
			}
		}

		// Execute the step using normal single-action dispatch (no recursion).
		result := ar.executeSingleAction(stepNLU, conv)
		chain.Results[i] = *result

		if result.Data != "" {
			allData = append(allData, fmt.Sprintf("[%s] %s", result.Source, result.Data))
		}
		if result.DirectResponse != "" {
			allData = append(allData, result.DirectResponse)
		}
		lastSource = result.Source
	}

	// Combine all step outputs into a single result.
	combined := strings.Join(allData, "\n\n---\n\n")
	return &ActionResult{
		DirectResponse: combined,
		Source:         "chain:" + lastSource,
	}
}

// executeSingleAction dispatches a single action without chain/generate_doc handling.
// This avoids infinite recursion if a chain step is itself "chain".
func (ar *ActionRouter) executeSingleAction(nlu *NLUResult, conv *Conversation) *ActionResult {
	switch nlu.Action {
	case "respond":
		return ar.handleRespond(nlu)
	case "web_search":
		return ar.handleWebSearch(nlu)
	case "fetch_url":
		return ar.handleFetchURL(nlu)
	case "file_op":
		return ar.handleFileOp(nlu)
	case "compute":
		return ar.handleCompute(nlu)
	case "lookup_memory":
		return ar.handleLookupMemory(nlu)
	case "lookup_knowledge":
		return ar.handleLookupKnowledge(nlu)
	case "lookup_web":
		return ar.handleLookupWeb(nlu)
	case "schedule":
		return ar.handleSchedule(nlu)
	case "llm_chat":
		return ar.handleLLMChat(nlu, conv)
	default:
		return ar.handleLLMChat(nlu, conv)
	}
}

// handleChain builds and executes a chain based on the chain_type entity.
func (ar *ActionRouter) handleChain(nlu *NLUResult, conv *Conversation) *ActionResult {
	chainType := nlu.Entities["chain_type"]
	topic := nlu.Entities["topic"]
	if topic == "" {
		topic = nlu.Entities["query"]
	}
	if topic == "" {
		topic = nlu.Raw
	}

	var chain *ActionChain
	switch chainType {
	case "research_and_write":
		chain = researchChain(topic)
	case "search_and_save":
		filepath := nlu.Entities["path"]
		if filepath == "" {
			filepath = strings.ReplaceAll(strings.ToLower(topic), " ", "_") + ".txt"
		}
		chain = searchAndSaveChain(topic, filepath)
	case "search_and_explain":
		chain = searchAndExplainChain(topic)
	default:
		chain = researchChain(topic)
	}

	return ar.ExecuteChain(chain, nlu, conv)
}

// handleGenerateDoc extracts a topic, runs web search + knowledge lookup,
// combines results, and returns for document formatting.
func (ar *ActionRouter) handleGenerateDoc(nlu *NLUResult, conv *Conversation) *ActionResult {
	topic := nlu.Entities["topic"]
	if topic == "" {
		topic = nlu.Entities["query"]
	}
	if topic == "" {
		topic = nlu.Raw
	}

	chain := researchChain(topic)
	result := ar.ExecuteChain(chain, nlu, conv)

	result.Structured = map[string]string{
		"format": "document",
		"topic":  topic,
	}
	result.DirectResponse = fmt.Sprintf("[Document Request: %s]\n\n%s", topic, result.DirectResponse)
	return result
}

// -----------------------------------------------------------------------
// Chain templates — pre-built pipelines for common multi-step tasks.
// -----------------------------------------------------------------------

// researchChain builds a chain for "research X" or "create a document about X".
// Steps: web search -> knowledge lookup -> combine for synthesis.
func researchChain(topic string) *ActionChain {
	return &ActionChain{
		Steps: []ChainStep{
			{
				Action:    "web_search",
				Entities:  map[string]string{"query": topic},
				DependsOn: -1,
			},
			{
				Action:    "lookup_knowledge",
				Entities:  map[string]string{"query": topic},
				DependsOn: -1,
			},
		},
	}
}

// searchAndSaveChain builds a chain for "search X and save to file".
// Steps: web search -> file write with search results.
func searchAndSaveChain(topic, filepath string) *ActionChain {
	return &ActionChain{
		Steps: []ChainStep{
			{
				Action:    "web_search",
				Entities:  map[string]string{"query": topic},
				DependsOn: -1,
			},
			{
				Action:    "file_op",
				Entities:  map[string]string{"op": "write", "path": filepath},
				DependsOn: 0,
			},
		},
	}
}

// searchAndExplainChain builds a chain for "look up X and explain it".
// Steps: web search -> knowledge lookup -> combine for LLM explanation.
func searchAndExplainChain(topic string) *ActionChain {
	return &ActionChain{
		Steps: []ChainStep{
			{
				Action:    "web_search",
				Entities:  map[string]string{"query": topic},
				DependsOn: -1,
			},
			{
				Action:    "lookup_knowledge",
				Entities:  map[string]string{"query": topic},
				DependsOn: -1,
			},
		},
	}
}

// handleLLMChat passes through to the LLM for conversational responses.
// BUT: tries extractive QA first to avoid LLM calls entirely.
func (ar *ActionRouter) handleLLMChat(nlu *NLUResult, conv *Conversation) *ActionResult {
	// Record user action pattern
	if ar.Patterns != nil {
		ar.Patterns.RecordAction("chat")
	}

	// Self-correction: detect when user corrects a previous answer
	if ar.CogGraph != nil {
		if correction := DetectCorrection(nlu.Raw); correction != nil {
			ApplyCorrection(ar.CogGraph, *correction)
		}
	}

	// Record event for causal analysis
	if ar.Causal != nil {
		ar.Causal.RecordEvent("chat", map[string]string{"query": nlu.Raw})
	}

	// Try multi-hop reasoning first — chain-of-thought over the graph
	if ar.Reasoner != nil && ar.CogGraph != nil && ar.CogGraph.NodeCount() > 0 {
		chain := ar.Reasoner.Reason(nlu.Raw)
		if chain != nil && chain.Answer != "" {
			answer := chain.Answer
			if ar.PersonalResp != nil {
				topic := ""
				if ar.Tracker != nil {
					topic = ar.Tracker.CurrentTopic()
				}
				answer = ar.PersonalResp.EnrichWithContext(answer, topic)
				answer = ar.PersonalResp.PersonalizeResponse(answer)
			}
			return &ActionResult{DirectResponse: answer, Source: "reasoning"}
		}
	}

	// Try cognitive graph — direct fact lookup with spreading activation
	if ar.CogGraph != nil && ar.CogGraph.NodeCount() > 0 {
		ga := ar.CogGraph.Query(nlu.Raw)
		if ga != nil && len(ga.DirectFacts) > 0 {
			answer := ar.CogGraph.ComposeAnswer(nlu.Raw, ga)
			if answer != "" {
				if ar.PersonalResp != nil {
					topic := ""
					if ar.Tracker != nil {
						topic = ar.Tracker.CurrentTopic()
					}
					answer = ar.PersonalResp.EnrichWithContext(answer, topic)
					answer = ar.PersonalResp.PersonalizeResponse(answer)
				}
				return &ActionResult{DirectResponse: answer, Source: "cognitive_graph"}
			}
		}
	}

	// Try extractive QA — if we have facts about the topic, answer directly
	if ar.Tracker != nil {
		// Check for continuation requests ("tell me more", "go on")
		if ar.Tracker.IsContinuation(nlu.Raw) {
			more := ar.Tracker.ContinueResponse()
			if more != "" {
				return &ActionResult{DirectResponse: more, Source: "extractive"}
			}
		}

		// Check for follow-up questions about current topic
		if ar.Tracker.IsFollowUp(nlu.Raw) || ar.Tracker.Facts.Size() > 0 {
			answer := ar.Tracker.AnswerQuestion(nlu.Raw)
			if answer != "" {
				// Enrich with personal context if available
				if ar.PersonalResp != nil {
					topic := ar.Tracker.CurrentTopic()
					answer = ar.PersonalResp.EnrichWithContext(answer, topic)
					answer = ar.PersonalResp.PersonalizeResponse(answer)
				}
				return &ActionResult{DirectResponse: answer, Source: "extractive"}
			}
		}
	}

	// Composer engine — generates natural language from structured knowledge.
	// Zero-LLM path: graph facts → natural sentences. Always produces a response.
	if ar.Composer != nil {
		respType := ar.ClassifyForComposer(nlu.Raw)
		ctx := ar.BuildComposeContext()
		resp := ar.Composer.Compose(nlu.Raw, respType, ctx)
		if resp != nil && resp.Text != "" {
			return &ActionResult{DirectResponse: resp.Text, Source: "composer"}
		}
	}

	// Fallback — Composer unavailable or returned empty (should not happen).
	return &ActionResult{
		DirectResponse: "I'm not sure how to answer that — could you rephrase?",
		Source:         "fallback",
	}
}

// -----------------------------------------------------------------------
// Math evaluator — handles basic arithmetic without any LLM.
// -----------------------------------------------------------------------

// evaluateMath evaluates simple math expressions.
// Supports: +, -, *, /, ^, %, parentheses, sqrt, abs.
// Examples: "2+2" -> "4", "sqrt(16)" -> "4", "15% of 200" -> "30"
func evaluateMath(expr string) (string, error) {
	// Clean input.
	expr = strings.TrimSpace(expr)
	expr = stripMathProse(expr)
	if expr == "" {
		return "", fmt.Errorf("empty expression")
	}

	// Handle "X% of Y" pattern.
	if m := percentOfRe.FindStringSubmatch(expr); len(m) == 3 {
		pct, err1 := strconv.ParseFloat(m[1], 64)
		val, err2 := strconv.ParseFloat(m[2], 64)
		if err1 == nil && err2 == nil {
			return formatNumber(pct / 100 * val), nil
		}
	}

	val, rest, err := parseExpr(expr)
	if err != nil {
		return "", err
	}
	rest = strings.TrimSpace(rest)
	if rest != "" {
		return "", fmt.Errorf("unexpected trailing text: %q", rest)
	}
	return formatNumber(val), nil
}

var percentOfRe = regexp.MustCompile(`(?i)^([\d.]+)\s*%\s*of\s+([\d.]+)$`)

// stripMathProse removes common phrasing around math expressions.
var mathProseRe = regexp.MustCompile(`(?i)^(?:what(?:'s| is)\s+|calculate\s+|compute\s+|eval(?:uate)?\s+)`)

func stripMathProse(s string) string {
	s = mathProseRe.ReplaceAllString(s, "")
	s = strings.TrimRight(s, "? ")
	// Replace unicode operators.
	s = strings.ReplaceAll(s, "\u00d7", "*") // ×
	s = strings.ReplaceAll(s, "\u00f7", "/") // ÷
	s = strings.NewReplacer("x", "*").Replace(s) // only if between digits
	return s
}

// Recursive descent parser for arithmetic expressions.
// Grammar:
//   expr   = term (('+' | '-') term)*
//   term   = power (('*' | '/' | '%') power)*
//   power  = unary ('^' unary)*
//   unary  = '-'? atom
//   atom   = number | func '(' expr ')' | '(' expr ')'

func parseExpr(s string) (float64, string, error) {
	val, rest, err := parseTerm(s)
	if err != nil {
		return 0, s, err
	}
	for {
		rest = strings.TrimLeftFunc(rest, unicode.IsSpace)
		if len(rest) == 0 {
			break
		}
		op := rest[0]
		if op != '+' && op != '-' {
			break
		}
		rest = rest[1:]
		right, r, err := parseTerm(rest)
		if err != nil {
			return 0, r, err
		}
		rest = r
		if op == '+' {
			val += right
		} else {
			val -= right
		}
	}
	return val, rest, nil
}

func parseTerm(s string) (float64, string, error) {
	val, rest, err := parsePower(s)
	if err != nil {
		return 0, s, err
	}
	for {
		rest = strings.TrimLeftFunc(rest, unicode.IsSpace)
		if len(rest) == 0 {
			break
		}
		op := rest[0]
		if op != '*' && op != '/' && op != '%' {
			break
		}
		rest = rest[1:]
		right, r, err := parsePower(rest)
		if err != nil {
			return 0, r, err
		}
		rest = r
		switch op {
		case '*':
			val *= right
		case '/':
			if right == 0 {
				return 0, rest, fmt.Errorf("division by zero")
			}
			val /= right
		case '%':
			if right == 0 {
				return 0, rest, fmt.Errorf("modulo by zero")
			}
			val = math.Mod(val, right)
		}
	}
	return val, rest, nil
}

func parsePower(s string) (float64, string, error) {
	val, rest, err := parseUnary(s)
	if err != nil {
		return 0, s, err
	}
	rest = strings.TrimLeftFunc(rest, unicode.IsSpace)
	if len(rest) > 0 && rest[0] == '^' {
		rest = rest[1:]
		right, r, err := parseUnary(rest)
		if err != nil {
			return 0, r, err
		}
		val = math.Pow(val, right)
		rest = r
	}
	return val, rest, nil
}

func parseUnary(s string) (float64, string, error) {
	s = strings.TrimLeftFunc(s, unicode.IsSpace)
	if len(s) > 0 && s[0] == '-' {
		val, rest, err := parseAtom(s[1:])
		if err != nil {
			return 0, rest, err
		}
		return -val, rest, nil
	}
	return parseAtom(s)
}

// mathFuncs supported by the evaluator.
var mathFuncs = map[string]func(float64) float64{
	"sqrt": math.Sqrt,
	"abs":  math.Abs,
	"sin":  math.Sin,
	"cos":  math.Cos,
	"tan":  math.Tan,
	"log":  math.Log10,
	"ln":   math.Log,
	"ceil": math.Ceil,
	"floor": math.Floor,
	"round": math.Round,
}

func parseAtom(s string) (float64, string, error) {
	s = strings.TrimLeftFunc(s, unicode.IsSpace)
	if len(s) == 0 {
		return 0, s, fmt.Errorf("unexpected end of expression")
	}

	// Check for named constants.
	if strings.HasPrefix(strings.ToLower(s), "pi") {
		rest := s[2:]
		if len(rest) == 0 || !isAlpha(rest[0]) {
			return math.Pi, rest, nil
		}
	}
	if strings.HasPrefix(strings.ToLower(s), "e") {
		rest := s[1:]
		if len(rest) == 0 || (!isAlpha(rest[0]) && rest[0] != '.') {
			return math.E, rest, nil
		}
	}

	// Check for function calls: func(expr)
	for name, fn := range mathFuncs {
		if strings.HasPrefix(strings.ToLower(s), name) {
			after := s[len(name):]
			after = strings.TrimLeftFunc(after, unicode.IsSpace)
			if len(after) > 0 && after[0] == '(' {
				inner, rest, err := parseExpr(after[1:])
				if err != nil {
					return 0, rest, err
				}
				rest = strings.TrimLeftFunc(rest, unicode.IsSpace)
				if len(rest) == 0 || rest[0] != ')' {
					return 0, rest, fmt.Errorf("missing closing parenthesis for %s", name)
				}
				return fn(inner), rest[1:], nil
			}
		}
	}

	// Parenthesised sub-expression.
	if s[0] == '(' {
		val, rest, err := parseExpr(s[1:])
		if err != nil {
			return 0, rest, err
		}
		rest = strings.TrimLeftFunc(rest, unicode.IsSpace)
		if len(rest) == 0 || rest[0] != ')' {
			return 0, rest, fmt.Errorf("missing closing parenthesis")
		}
		return val, rest[1:], nil
	}

	// Number literal.
	return parseNumber(s)
}

func parseNumber(s string) (float64, string, error) {
	s = strings.TrimLeftFunc(s, unicode.IsSpace)
	i := 0
	for i < len(s) && (s[i] >= '0' && s[i] <= '9' || s[i] == '.') {
		i++
	}
	if i == 0 {
		return 0, s, fmt.Errorf("expected number, got %q", truncStr(s, 20))
	}
	val, err := strconv.ParseFloat(s[:i], 64)
	if err != nil {
		return 0, s, fmt.Errorf("invalid number %q", s[:i])
	}
	return val, s[i:], nil
}

func isAlpha(b byte) bool {
	return (b >= 'a' && b <= 'z') || (b >= 'A' && b <= 'Z')
}

// formatNumber formats a float, stripping trailing zeroes for integers.
func formatNumber(v float64) string {
	if v == math.Trunc(v) && math.Abs(v) < 1e15 {
		return strconv.FormatInt(int64(v), 10)
	}
	s := strconv.FormatFloat(v, 'f', 10, 64)
	s = strings.TrimRight(s, "0")
	s = strings.TrimRight(s, ".")
	return s
}

// extractTopicFromURL pulls a topic name from a URL.
// "https://stoicera.com" → "stoicera", "https://golang.org/doc/go1.21" → "golang"
func extractTopicFromURL(url string) string {
	// Strip protocol
	u := url
	for _, prefix := range []string{"https://", "http://", "www."} {
		u = strings.TrimPrefix(u, prefix)
	}
	// Take the domain part
	if idx := strings.Index(u, "/"); idx > 0 {
		u = u[:idx]
	}
	// Remove TLD
	if idx := strings.LastIndex(u, "."); idx > 0 {
		u = u[:idx]
	}
	return u
}

func truncStr(s string, n int) string {
	if len(s) > n {
		return s[:n] + "..."
	}
	return s
}

// -----------------------------------------------------------------------
// Date evaluator — handles relative date questions.
// -----------------------------------------------------------------------

var (
	tomorrowRe   = regexp.MustCompile(`(?i)\btomorrow\b`)
	yesterdayRe   = regexp.MustCompile(`(?i)\byesterday\b`)
	todayRe       = regexp.MustCompile(`(?i)\btoday\b|\bwhat day is it\b|\bwhat is today\b`)
	daysUntilRe   = regexp.MustCompile(`(?i)(?:days?\s+(?:until|to|till|before))\s+(.+)`)
	daysSinceRe   = regexp.MustCompile(`(?i)(?:days?\s+(?:since|from|after))\s+(.+)`)
	whatDayRe     = regexp.MustCompile(`(?i)what\s+(?:day|date)\s+(?:is|was|will be)\s+(.+)`)
)

func evaluateDate(expr string) (string, bool) {
	now := time.Now()

	if todayRe.MatchString(expr) {
		return now.Format("Monday, January 2, 2006"), true
	}
	if tomorrowRe.MatchString(expr) {
		t := now.AddDate(0, 0, 1)
		return t.Format("Monday, January 2, 2006"), true
	}
	if yesterdayRe.MatchString(expr) {
		t := now.AddDate(0, 0, -1)
		return t.Format("Monday, January 2, 2006"), true
	}

	if m := daysUntilRe.FindStringSubmatch(expr); len(m) == 2 {
		if target, ok := parseDate(strings.TrimSpace(m[1])); ok {
			days := int(target.Sub(now).Hours() / 24)
			if days < 0 {
				return fmt.Sprintf("%d days ago", -days), true
			}
			return fmt.Sprintf("%d days", days), true
		}
	}
	if m := daysSinceRe.FindStringSubmatch(expr); len(m) == 2 {
		if target, ok := parseDate(strings.TrimSpace(m[1])); ok {
			days := int(now.Sub(target).Hours() / 24)
			if days < 0 {
				return fmt.Sprintf("%d days in the future", -days), true
			}
			return fmt.Sprintf("%d days", days), true
		}
	}

	if m := whatDayRe.FindStringSubmatch(expr); len(m) == 2 {
		if target, ok := parseDate(strings.TrimSpace(m[1])); ok {
			return target.Format("Monday, January 2, 2006"), true
		}
	}

	return "", false
}

// parseDate tries several common date formats.
func parseDate(s string) (time.Time, bool) {
	s = strings.TrimRight(s, "?. ")
	formats := []string{
		"2006-01-02",
		"January 2, 2006",
		"Jan 2, 2006",
		"January 2 2006",
		"Jan 2 2006",
		"01/02/2006",
		"02-01-2006",
		"2 January 2006",
		"2 Jan 2006",
	}
	for _, f := range formats {
		if t, err := time.Parse(f, s); err == nil {
			return t, true
		}
	}
	return time.Time{}, false
}

// parseRelativeTime parses relative time expressions like "in 2 hours", "tomorrow at 3pm".
func parseRelativeTime(s string) (time.Time, bool) {
	now := time.Now()
	s = strings.ToLower(strings.TrimSpace(s))

	if s == "tomorrow" {
		return now.AddDate(0, 0, 1), true
	}
	if s == "today" || s == "now" {
		return now, true
	}

	// "in N hours/minutes/days"
	re := regexp.MustCompile(`in\s+(\d+)\s+(hour|minute|min|day|week)s?`)
	if m := re.FindStringSubmatch(s); len(m) == 3 {
		n, _ := strconv.Atoi(m[1])
		switch m[2] {
		case "hour":
			return now.Add(time.Duration(n) * time.Hour), true
		case "minute", "min":
			return now.Add(time.Duration(n) * time.Minute), true
		case "day":
			return now.AddDate(0, 0, n), true
		case "week":
			return now.AddDate(0, 0, n*7), true
		}
	}

	return time.Time{}, false
}

// -----------------------------------------------------------------------
// ResponseFormatter — the ONLY place where LLM is called for response.
// -----------------------------------------------------------------------

// ResponseFormatter takes raw action data and formats it into natural language.
// Pure cognitive engine — no LLM calls.
type ResponseFormatter struct {
	PersonalResp *PersonalResponseGenerator
}

// Format turns raw data into a natural language response.
func (rf *ResponseFormatter) Format(query string, result *ActionResult, conv *Conversation) (string, error) {
	if result.DirectResponse != "" {
		return result.DirectResponse, nil
	}
	return result.Data, nil
}

// FormatStream is the streaming version of Format.
func (rf *ResponseFormatter) FormatStream(query string, result *ActionResult, conv *Conversation, onToken func(string, bool)) (string, error) {
	text := result.DirectResponse
	if text == "" {
		text = result.Data
	}
	if text == "" {
		text = "I'm not sure how to answer that — could you rephrase?"
	}
	onToken(text, true)
	return text, nil
}

// -----------------------------------------------------------------------
// Text transformation handler
// -----------------------------------------------------------------------

// handleTransform applies a text transformation operation.
func (ar *ActionRouter) handleTransform(nlu *NLUResult) *ActionResult {
	if ar.Transformer == nil {
		return &ActionResult{
			DirectResponse: "Text transformer is not initialized.",
			Source:         "transform",
		}
	}

	op := nlu.Entities["operation"]
	if op == "" {
		op = "paraphrase"
	}

	text := nlu.Entities["text"]
	if text == "" {
		text = nlu.Entities["_chain_input"]
	}
	if text == "" {
		text = nlu.Raw
	}

	result := ar.Transformer.Transform(text, TransformOp(op))
	if result == "" {
		return &ActionResult{
			DirectResponse: "Nothing to transform — please provide some text.",
			Source:         "transform",
		}
	}

	return &ActionResult{
		DirectResponse: result,
		Source:         "transform",
	}
}

// handleCreative generates poems, stories, jokes, reflections, or handles
// general creative requests like writing lists, fun facts, etc.
func (ar *ActionRouter) handleCreative(nlu *NLUResult) *ActionResult {
	topic := nlu.Entities["topic"]
	creativeType := nlu.Entities["creative_type"]

	// "tell me something interesting" / "tell me a fun fact" → use knowledge graph
	if creativeType == "fun_fact" {
		if ar.CogGraph != nil {
			fact := ar.CogGraph.RandomFact()
			if fact != "" {
				return &ActionResult{
					DirectResponse: "Here's something interesting: " + fact,
					Source:         "knowledge",
				}
			}
		}
		return &ActionResult{
			DirectResponse: "I'd love to share something interesting, but my knowledge base is still growing. Ask me about a specific topic!",
			Source:         "creative",
		}
	}

	// "help me write a shopping list" / general creative help without a specific
	// creative type (poem/story/joke/reflect) → use Composer for conversational response
	if creativeType == "" {
		// This is a general "write X" request, not a poem/story/joke.
		// Use the Composer or Thinker for a helpful response.
		if ar.Thinker != nil && ar.Thinker.CanHandle(nlu.Raw) {
			result := ar.Thinker.Think(nlu.Raw, nil)
			if result != nil && result.Text != "" {
				return &ActionResult{
					DirectResponse: result.Text,
					Source:         "thinking:" + result.Frame,
				}
			}
		}
		if ar.Composer != nil {
			ctx := ar.BuildComposeContext()
			resp := ar.Composer.Compose(nlu.Raw, RespConversational, ctx)
			if resp != nil && resp.Text != "" {
				return &ActionResult{
					DirectResponse: resp.Text,
					Source:         "composer",
				}
			}
		}
		// Helpful fallback for general creative requests
		return &ActionResult{
			DirectResponse: fmt.Sprintf("I'd be happy to help you write that! Could you give me a bit more detail about what you'd like in your %s?", topic),
			Source:         "creative",
		}
	}

	if ar.Creative == nil {
		return &ActionResult{
			DirectResponse: "Creative engine is not initialized.",
			Source:         "creative",
		}
	}

	var form PoemForm
	switch nlu.Entities["poem_form"] {
	case "haiku":
		form = PoemHaiku
	case "quatrain":
		form = PoemQuatrain
	default:
		form = PoemFreeVerse
	}

	req := CreativeRequest{
		Type:     CreativeType(creativeType),
		Topic:    topic,
		PoemForm: form,
	}

	result := ar.Creative.Generate(req)
	return &ActionResult{
		DirectResponse: result,
		Source:         "creative",
	}
}
