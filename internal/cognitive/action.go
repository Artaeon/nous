package cognitive

import (
	"fmt"
	"math"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"sync"
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
	CommonSense    *CommonSenseGraph
	ConvLearner    *ConversationLearner
	Subtext        *SubtextEngine
	MemTrigger     *MemoryTriggerEngine
	Absorption     *AbsorptionEngine
	Sparks         *SparkEngine
	Council        *InnerCouncil
	Opinions       *OpinionEngine
	DeepReason     *DeepReasoner
	MultiHop       *MultiHopReasoner

	// Phase 1-4 wired systems
	ConvState      *ConversationState
	FollowUp       *FollowUpResolver
	Preferences    *PreferenceModel
	Retriever      *TwoTierRetriever
	QueryRewrite   *QueryRewriter
	Filler         *FillerDetector

	// NLG engines
	NLG            *NLGEngine
	CorpusNLG      *CorpusNLG
	HybridGen      *HybridGenerator
	Format         *FormatCompliance

	// Fact extraction + fluency + policy + context
	FactExtract    *WikiFactExtractor
	Fluency        *FluencyScorer
	Policy         *DialoguePolicy
	CtxWindow      *ContextWindow
	SelfTeacher    *SelfTeach

	// Smart entity extraction (intent→entity bridging)
	EntityExtract  *SmartEntityExtractor

	// Innovation systems
	Socratic       *SocraticEngine
	Crystallizer   *InsightCrystallizer
	Transparency   *CognitiveTransparency
	Synthesizer    *KnowledgeSynthesizer
	SelfModel      *SelfModel

	// Text summarization and document generation
	Summarizer     *Summarizer
	DocGen         *DocumentGenerator

	// Prose composition and code generation
	ProseComposer  *ProseComposer
	CodeGen        *CodeGenerator

	// Innovation engines — simulation, personas, GraphRAG, causal inference, knowledge expansion
	Simulation     *SimulationEngine
	Personas       *PersonaEngine
	GraphRAG       *GraphRAGEngine
	CausalInfer    *CausalInferenceEngine
	Expander       *KnowledgeExpander
	DispatchPipe   *DispatchPipeline
	WikiLoader     *WikipediaLoader

	// Micro language model — knowledge-grounded sentence generation
	MicroModel interface {
		GenerateSentence(subject, relation, object string) string
		GenerateParagraph(topic string, facts [][3]string) string
	}

	// Cognitive compiler — compiles neural responses into deterministic handlers.
	// Checked before expensive generation paths. Every neural response feeds back
	// into the compiler, making the system faster over time.
	CogCompiler *CognitiveCompiler

	// LLM — optional local language model for enhanced responses.
	// When set, used for: knowledge gap filling, deeper explanations,
	// comparisons, and any query the deterministic pipeline can't handle well.
	// Falls back silently when nil.
	LLM LLMClient
}

// LLMClient is the interface for a local language model.
// Implemented by the Ollama client wrapper.
type LLMClient interface {
	// Generate sends a prompt and returns the response text.
	// maxTokens controls output length. Returns empty string on error.
	Generate(system, prompt string, maxTokens int) string
}

// NewActionRouter creates a router with nil subsystems.
// Wire up the fields after creation as each subsystem initialises.
func NewActionRouter() *ActionRouter {
	return &ActionRouter{}
}

// TryCompiled checks the cognitive compiler for a pre-compiled handler
// that matches the input. Returns the response and true if a compiled
// handler was found, or empty string and false if no match.
// This is the fast path: O(n) regex match, no neural computation.
func (ar *ActionRouter) TryCompiled(input string) (string, bool) {
	// Cognitive compiler disabled: the topic validation guards in Compile()
	// and Execute() prevent cross-topic contamination for NEW handlers,
	// but existing in-session handlers from the Composer path still produce
	// wrong results because the Composer generates knowledge paragraphs
	// that get compiled with overly broad patterns. Re-enabling requires
	// intent-scoped matching so "explain" patterns don't match "greeting"
	// queries, and vice versa.
	return "", false
}

// LearnResponse feeds a (input, response, nlu) tuple into the cognitive
// compiler so it can extract a pattern and compile a deterministic handler.
// Called asynchronously after every successful response generation.
func (ar *ActionRouter) LearnResponse(input, response string, nlu *NLUResult) {
	if ar.CogCompiler == nil || nlu == nil {
		return
	}

	// Only compile responses from high-quality, stable sources.
	// Dynamic/tool/knowledge sources produce topic-specific content that
	// can't generalize across different topics.
	switch nlu.Action {
	case "respond":
		// Conversational responses can be compiled — they're generic
	case "llm_chat":
		// LLM responses can be compiled — they're adaptable
	default:
		// All other actions (tools, knowledge, creative, etc.) produce
		// topic-specific content that shouldn't be cached as templates.
		return
	}

	// Require the response to be substantial but not too long
	words := strings.Fields(response)
	if len(words) < 5 || len(words) > 50 {
		return
	}

	// Require a topic entity — the compiler needs it for slot extraction
	topic := nlu.Entities["topic"]
	if topic == "" {
		return
	}

	// Verify the response mentions the topic (sanity check)
	if !strings.Contains(strings.ToLower(response), strings.ToLower(topic)) {
		return
	}

	entities := nlu.Entities
	handler := ar.CogCompiler.Compile(input, response, nlu.Intent, entities)
	if handler != nil {
		ar.CogCompiler.Save()
	}
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
	// Safety guard: refuse requests that involve harmful activities.
	if isHarmfulRequest(nlu.Raw) {
		// Self-harm needs a compassionate response, not a refusal.
		lower := strings.ToLower(nlu.Raw)
		if strings.Contains(lower, "hurt myself") || strings.Contains(lower, "end my life") ||
			strings.Contains(lower, "kill myself") || strings.Contains(lower, "suicide") {
			return &ActionResult{
				DirectResponse: "I hear you, and I want you to know that your life matters. Please reach out to a crisis helpline — in the US, call or text 988 (Suicide & Crisis Lifeline). You don't have to go through this alone.",
				Source:         "safety_compassion",
			}
		}
		return &ActionResult{
			DirectResponse: "I can't help with that. I'm designed to be helpful, but I need to decline requests that could lead to harm. Is there something else I can help you with?",
			Source:         "safety",
		}
	}

	// Conversational context: resolve pronouns and references using the
	// current conversation topic. "why are THEY dangerous?" → "they" = last topic.
	// "can anything escape from ONE?" → "one" = last topic.
	// This is the bridge between single-turn NLU and multi-turn conversation.
	ar.resolveConversationalReferences(nlu)

	// Hard-pin: explicit task prompts (explain, compare, overview, summarize,
	// walk me through) NEVER go through emotional/conversational pipelines.
	// They always route to the knowledge/thinking path, no exceptions.
	taskPinned := isExplicitTaskPrompt(nlu.Raw)

	// ---------------------------------------------------------------
	// Instruction detection: "ask me N questions" — handle immediately.
	// Must fire BEFORE empathy/subtext to prevent emotional interceptors
	// from swallowing coaching/decision queries.
	// ---------------------------------------------------------------
	if !taskPinned {
		instructions := DetectInstructions(nlu.Raw)
		for _, inst := range instructions {
			if inst.Type == "ask_questions" {
				resp := GenerateQuestions(nlu.Raw, inst.Count)
				if resp != "" {
					return &ActionResult{DirectResponse: resp, Source: "instruction_follow"}
				}
			}
		}
	}

	// ---------------------------------------------------------------
	// Socratic Engine — check if asking questions would be more valuable
	// than answering. Only for non-task, non-tool prompts. Must fire
	// BEFORE empathy/subtext so coaching and decision queries get
	// Socratic questions, not empathetic platitudes.
	// ---------------------------------------------------------------
	// Don't Socratic-question casual recommendations (food, entertainment, shopping)
	// or straightforward questions — just answer them directly.
	socraticExcluded := nlu.Intent == "recommendation" || nlu.Intent == "question"
	socraticEligible := nlu.Action == "respond" || nlu.Action == "llm_chat" || nlu.Action == "" || nlu.Action == "lookup_knowledge"
	if !taskPinned && !socraticExcluded && ar.Socratic != nil && socraticEligible {
		mode := ar.Socratic.DetectMode(nlu.Raw, ar.ConvState)
		if mode != SocraticNone {
			resp := ar.Socratic.Generate(nlu.Raw, mode, ar.ConvState)
			if resp != nil && len(resp.Questions) > 0 {
				var parts []string
				if resp.Framing != "" {
					parts = append(parts, resp.Framing)
				}
				for _, q := range resp.Questions {
					parts = append(parts, q.Text)
				}
				return &ActionResult{
					DirectResponse: strings.Join(parts, "\n\n"),
					Source:         "socratic:" + mode.String(),
				}
			}
		}
	}

	// Emotional intelligence pre-check: personal emotional statements
	// ("I got promoted!", "I'm feeling down", "I just had a great day!")
	// should get an empathetic response, not a knowledge lookup.
	// Blocked for task prompts — "explain why I feel sad" is a task, not venting.
	// Only fires if Socratic and instruction detection didn't handle the input.
	if !taskPinned && ar.Composer != nil && isEmotionalStatement(nlu.Raw) {
		ctx := ar.BuildComposeContextWithSubtext(nlu.Raw, nlu)
		resp := ar.Composer.Compose(nlu.Raw, RespEmpathetic, ctx)
		if resp != nil && resp.Text != "" {
			return &ActionResult{DirectResponse: resp.Text, Source: "empathy"}
		}
	}

	// Subtext-driven response type upgrade: if the subtext engine detects
	// emotional needs that the NLU missed, respond empathetically.
	// Blocked for task prompts — emotional subtext must not hijack task routing.
	if !taskPinned && ar.Subtext != nil && ar.Composer != nil {
		var history []ConvTurn
		if ar.Composer != nil {
			history = ar.Composer.history
		}
		analysis := ar.Subtext.Analyze(nlu.Raw, nlu, history)
		needsEmpathy := false
		switch analysis.ImpliedNeed {
		case NeedVenting, NeedReassurance:
			needsEmpathy = analysis.EmotionalState.Valence < -0.1
		case NeedCelebration:
			needsEmpathy = analysis.EmotionalState.Valence > 0.2
		}
		if needsEmpathy {
			ctx := ar.BuildComposeContext()
			ctx.Subtext = &analysis
			resp := ar.Composer.Compose(nlu.Raw, RespEmpathetic, ctx)
			if resp != nil && resp.Text != "" {
				return &ActionResult{DirectResponse: resp.Text, Source: "subtext_empathy"}
			}
		}
	}

	// Hard reroute: if the NLU classified an explicit task prompt as
	// conversational (action=respond, greeting, etc.), override it NOW
	// before dispatch. This is the second half of the task pin — the first
	// half blocked empathy/subtext, this one forces the correct pipeline.
	if taskPinned {
		switch nlu.Action {
		case "respond", "":
			// Determine the correct task action from the prompt.
			lower := strings.ToLower(strings.TrimSpace(nlu.Raw))
			switch {
			case strings.HasPrefix(lower, "compare ") ||
				strings.HasPrefix(lower, "contrast ") ||
				strings.HasPrefix(lower, "difference") ||
				strings.HasPrefix(lower, "differences") ||
				strings.HasPrefix(lower, "pros and cons") ||
				strings.Contains(lower, " vs ") ||
				strings.Contains(lower, " versus ") ||
				strings.HasPrefix(lower, "how are ") && strings.Contains(lower, " related") ||
				strings.HasPrefix(lower, "what connects ") ||
				strings.HasPrefix(lower, "connection between ") ||
				strings.HasPrefix(lower, "relationship between "):
				nlu.Intent = "compare"
				nlu.Action = "compare"
			case strings.HasPrefix(lower, "summarize ") ||
				strings.HasPrefix(lower, "summarise ") ||
				strings.HasPrefix(lower, "summary of ") ||
				strings.HasPrefix(lower, "give me a summary"):
				nlu.Intent = "explain"
				nlu.Action = "lookup_knowledge"
			default:
				// explain, overview, walk me through, what is, describe, define, etc.
				nlu.Intent = "explain"
				nlu.Action = "lookup_knowledge"
			}
			if nlu.Entities == nil {
				nlu.Entities = make(map[string]string)
			}
			if nlu.Entities["topic"] == "" {
				nlu.Entities["topic"] = extractTopicFromQuery(strings.ToLower(strings.TrimSpace(nlu.Raw)))
			}
		}
	}

	// Smart entity extraction based on classified intent.
	// Bridges the gap between intent classification and tool dispatch:
	// the NLU knows what the user wants (calculate, translate, etc.) but
	// the tool handlers need specific entities (expression, text, language).
	if ar.EntityExtract != nil {
		if nlu.Entities == nil {
			nlu.Entities = make(map[string]string)
		}
		ar.EntityExtract.ExtractForIntent(nlu.Raw, nlu.Intent, nlu.Entities)
	}

	// Extractive summarizer intercept: when the user says "summarize" or
	// "tl;dr" with accompanying text, use the Summarizer to extract the
	// most important sentences rather than a knowledge lookup.
	// Threshold: 30 words — anything paragraph-length or longer is fair game.
	if ar.Summarizer != nil && isSummarizeRequest(nlu.Raw) {
		textToSummarize := extractTextForSummarization(nlu.Raw)
		if len(strings.Fields(textToSummarize)) > 30 {
			summary := ar.Summarizer.Summarize(textToSummarize, 5)
			if summary != "" {
				return &ActionResult{
					DirectResponse: summary,
					Source:         "summarizer",
				}
			}
		}
	}

	result := ar.dispatch(nlu, conv)

	// LLM gap-fill: if the deterministic pipeline returned a deflection
	// ("What specifically about X would you like to explore?") and we have
	// a local LLM, ask it to answer the question directly instead.
	if result != nil && isDeflection(result.DirectResponse) && ar.LLM != nil {
		answer := ar.LLM.Generate(
			"Answer concisely and factually in 2-3 sentences.",
			nlu.Raw,
			96,
		)
		if answer != "" && !isDeflection(answer) {
			result = &ActionResult{DirectResponse: answer, Source: "llm"}
		}
	}

	// Detect implicit follow-ups: questions containing pronouns that reference previous turns.
	// Many follow-ups are classified as "question" by the NLU (e.g., "how does that relate to
	// genetics") and never reach the FollowUpResolver. Catch them here after dispatch when
	// the result is thin.
	if (result == nil || result.DirectResponse == "" || isLowInformationConversational(result.DirectResponse)) &&
		ar.FollowUp != nil && ar.ConvState != nil && looksLikeImplicitFollowUp(nlu.Raw) {
		resolved := ar.FollowUp.Resolve(nlu.Raw, ar.ConvState)
		if resolved != nil && resolved.IsFollowUp && resolved.ResolvedQuery != "" {
			resolvedNLU := &NLUResult{
				Raw:      resolved.ResolvedQuery,
				Intent:   "explain",
				Action:   "lookup_knowledge",
				Entities: map[string]string{"topic": resolved.ResolvedQuery},
			}
			if lookupResult := ar.handleLookupKnowledge(resolvedNLU); lookupResult != nil && lookupResult.DirectResponse != "" {
				if !isLowInformationConversational(lookupResult.DirectResponse) {
					result = lookupResult
				}
			}
		}
	}

	// When knowledge is empty and we'd produce filler, try Socratic instead.
	// This is the "I know that I know nothing" fallback — ask questions
	// rather than generating generic filler text.
	if (result == nil || result.DirectResponse == "" || isLowInformationConversational(result.DirectResponse)) &&
		!taskPinned && !socraticExcluded && ar.Socratic != nil {
		mode := ar.Socratic.DetectMode(nlu.Raw, ar.ConvState)
		if mode == SocraticNone {
			// No Socratic mode detected — try generic exploration
			mode = SocraticExplore
		}
		resp := ar.Socratic.Generate(nlu.Raw, mode, ar.ConvState)
		if resp != nil && len(resp.Questions) > 0 {
			var parts []string
			if resp.Framing != "" {
				parts = append(parts, resp.Framing)
			}
			for _, q := range resp.Questions {
				parts = append(parts, q.Text)
			}
			result = &ActionResult{
				DirectResponse: strings.Join(parts, "\n\n"),
				Source:         "socratic_fallback",
			}
		}
	}

	// Cognitive enrichment: after generating a response, check for
	// involuntary memory triggers and associative sparks.
	if result != nil && result.DirectResponse != "" {
		topics := extractKeywords(strings.ToLower(nlu.Raw))

		// Memory triggers — surface relevant past episodes
		if ar.MemTrigger != nil {
			triggers := ar.MemTrigger.Scan(nlu.Raw, topics)
			for _, trig := range triggers {
				note := ar.MemTrigger.FormatTrigger(trig)
				if note != "" {
					result.DirectResponse += "\n\n" + note
				}
				ar.MemTrigger.RecordSurfaced(trig.Episode.ID)
			}
		}

		// Associative sparks — surface unexpected connections.
		// Only surface sparks that are relevant to the current query.
		// This prevents noise like "Photosynthesis connects to South America".
		if ar.Sparks != nil {
			ar.Sparks.RecordTopics(topics)
			sparks := ar.Sparks.Ignite(topics)
			for _, spark := range sparks {
				if isSparkRelevant(spark.Explanation, topics) {
					result.DirectResponse += "\n\n" + spark.Explanation
					ar.Sparks.RecordSurfaced(spark.Source, spark.Target)
				}
			}
		}
	}

	// Opinion learning — extract evaluative language from user input
	if ar.Opinions != nil {
		topics := extractKeywords(strings.ToLower(nlu.Raw))
		ar.Opinions.LearnFromConversation(nlu.Raw, topics)
	}

	// Advance dialogue state machine (tracks topic, state transitions).
	if ar.Dialogue != nil && result != nil {
		ar.Dialogue.ProcessTurn(nlu, result)
	}

	// Update conversation state — tracks active topic, entities, coreferences.
	if ar.ConvState != nil && result != nil {
		ar.ConvState.Update(nlu.Raw, nlu, result.DirectResponse)
	}

	// Innovation: Insight Crystallizer — observe every turn, surface insights.
	if ar.Crystallizer != nil {
		sentiment := "neutral"
		if ar.Composer != nil {
			switch ar.Composer.detectSentiment(nlu.Raw) {
			case SentimentPositive:
				sentiment = "positive"
			case SentimentNegative:
				sentiment = "negative"
			}
		}
		ar.Crystallizer.Observe(nlu.Raw, nlu.Entities, sentiment)

		// Surface relevant insights when the conversation warrants it
		if result != nil && result.DirectResponse != "" && ar.ConvState != nil {
			topic := ar.ConvState.ActiveTopic
			if topic == "" {
				topic = nlu.Entities["topic"]
			}
			if insight := ar.Crystallizer.SurfaceRelevant(topic); insight != nil && insight.Confidence >= 0.5 {
				result.DirectResponse += "\n\n---\n" + insight.Text
			}
		}
	}

	// Innovation: SelfModel — record interaction outcome for self-improvement.
	if ar.SelfModel != nil && result != nil {
		domain := ar.SelfModel.ClassifyDomain(nlu.Raw)
		quality := 0.5 // baseline; user feedback adjusts later
		if result.Source != "" && !strings.Contains(result.Source, "fallback") {
			quality = 0.7 // grounded response from knowledge
		}
		ar.SelfModel.RecordOutcome(domain, nlu.Raw, quality, true, result.Source)
	}

	// Context window — record turn for repetition avoidance across turns.
	if ar.CtxWindow != nil && result != nil && result.DirectResponse != "" {
		topics := []string{}
		if t := nlu.Entities["topic"]; t != "" {
			topics = append(topics, t)
		}
		ar.CtxWindow.Record(nlu.Raw, result.DirectResponse, topics)
		// Remove sentences that were in previous responses.
		cleaned := ar.CtxWindow.AvoidRepetition(result.DirectResponse)
		if cleaned != "" {
			result.DirectResponse = cleaned
		}
	}

	// Update user preference model from behavioral signals.
	if ar.Preferences != nil && result != nil {
		wasFollowUp := nlu.Intent == "followup" || nlu.Intent == "follow_up"
		wasClarification := strings.Contains(strings.ToLower(nlu.Raw), "what do you mean") ||
			strings.Contains(strings.ToLower(nlu.Raw), "can you clarify")
		wasCorrection := strings.Contains(strings.ToLower(nlu.Raw), "no, ") ||
			strings.Contains(strings.ToLower(nlu.Raw), "that's wrong") ||
			strings.Contains(strings.ToLower(nlu.Raw), "actually,")
		ar.Preferences.ObserveTurn(nlu.Raw, result.DirectResponse, wasFollowUp, wasClarification, wasCorrection)
	}

	// Record interaction for conversation learning.
	// The learner generalizes successful patterns so Nous improves over time.
	if ar.ConvLearner != nil && result != nil && result.DirectResponse != "" {
		topic := nlu.Entities["topic"]
		sentiment := "neutral"
		if ar.Composer != nil {
			switch ar.Composer.detectSentiment(nlu.Raw) {
			case SentimentPositive:
				sentiment = "positive"
			case SentimentNegative:
				sentiment = "negative"
			case SentimentExcited:
				sentiment = "excited"
			case SentimentSad:
				sentiment = "sad"
			case SentimentAngry:
				sentiment = "angry"
			case SentimentCurious:
				sentiment = "curious"
			}
		}
		// Success is determined later when the user responds.
		// For now, record the interaction as pending.
		ar.ConvLearner.LearnFromInteraction(
			nlu.Raw, result.DirectResponse,
			nlu.Intent, sentiment, topic, true,
		)

		// Conversation-to-graph learning: extract factual knowledge
		// from substantive responses back into the knowledge graph.
		// The system gets smarter with every conversation.
		if ar.CogGraph != nil && len(result.DirectResponse) > 100 &&
			(result.Source == "knowledge_text" || result.Source == "semantic_retrieval" ||
				result.Source == "nlg" || result.Source == "knowledge") {
			go ar.ConvLearner.LearnFactsFromResponse(
				result.DirectResponse, topic, ar.CogGraph,
			)
		}
	}

	// Strip filler from all outputs — "As an AI...", empty hedges, etc.
	if ar.Filler != nil && result != nil && result.DirectResponse != "" {
		cleaned, changed := ar.Filler.EnforcePolicy(result.DirectResponse, isExplicitTaskPrompt(nlu.Raw))
		if changed && strings.TrimSpace(cleaned) != "" {
			result.DirectResponse = cleaned
		}
	}

	// ---------------------------------------------------------------
	// Response Quality Gate — last line of defense before the user
	// sees the response. Catches tool error leaks, low-value acks
	// on substantive turns, and parroting.
	// Skip for greetings/farewells — these are intentionally short.
	// ---------------------------------------------------------------
	gateSkip := nlu.Intent == "greeting" || nlu.Intent == "farewell" || nlu.Intent == "affirmation"
	if result != nil && result.DirectResponse != "" && !gateSkip {
		gate := &ResponseGate{}
		verdict := gate.Check(nlu.Raw, result.DirectResponse, result.Source)
		if !verdict.Pass {
			// Check for user instructions that we can fulfill directly
			instructions := DetectInstructions(nlu.Raw)
			for _, inst := range instructions {
				if inst.Type == "ask_questions" {
					result.DirectResponse = GenerateQuestions(nlu.Raw, inst.Count)
					result.Source = "instruction_follow"
					return result
				}
			}
			// If gate repaired the response, use that
			if verdict.Repaired != "" {
				result.DirectResponse = verdict.Repaired
				result.Source += "+gated"
			} else if ar.Thinker != nil {
				// Gate failed and no repair — try the thinking engine as fallback
				if thinkResult := ar.Thinker.Think(nlu.Raw, nil); thinkResult != nil && thinkResult.Text != "" {
					if !isLowValueResponse(thinkResult.Text) {
						result.DirectResponse = thinkResult.Text
						result.Source = "thinking:" + thinkResult.Frame + "+gated"
					}
				}
			}
		}
		// Also check if user gave instructions we should honor
		// even when the gate passed (e.g., "give me 3 bullets")
		instructions := DetectInstructions(nlu.Raw)
		for _, inst := range instructions {
			if inst.Type == "ask_questions" && strings.Count(result.DirectResponse, "?") < inst.Count {
				result.DirectResponse = GenerateQuestions(nlu.Raw, inst.Count)
				result.Source = "instruction_follow"
				break
			}
		}
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
	case "compare":
		return ar.handleCompare(nlu)
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
		result := ar.handleCalculate(nlu)
		if result != nil && strings.HasPrefix(result.DirectResponse, "Could not calculate") {
			if compResult := ar.handleCompute(nlu); compResult != nil && !strings.HasPrefix(compResult.DirectResponse, "cannot compute") {
				return compResult
			}
			// Last resort: if the input has no math, try knowledge lookup.
			// Catches misclassifications like "what is the periodic table?"
			if !containsDigit(strings.ToLower(nlu.Raw)) {
				if knResult := ar.handleLookupKnowledge(nlu); knResult != nil && knResult.Source != "honest_fallback" {
					return knResult
				}
			}
		}
		return result
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
	case "code", "codegen":
		return ar.handleCodeGen(nlu)
	case "simulate":
		return ar.handleSimulate(nlu)
	case "persona":
		return ar.handlePersona(nlu)
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
	"are you alright":      "I'm running perfectly — all systems green. Thanks for asking! What can I help you with?",
	"are you okay":         "I'm doing great — all systems running smoothly. What's on your mind?",
	"are you good":         "I'm good! Everything's running smoothly. What can I do for you?",
	"how are you":          "I'm running well — all cognitive systems active and ready. What can I help with?",
}

// handleRespond returns a response for greetings, farewells, meta, etc.
// Uses the Composer engine when available — zero LLM calls.
func (ar *ActionRouter) handleRespond(nlu *NLUResult) *ActionResult {
	// Check meta responses — these are about Nous itself (who are you, are you okay, etc.)
	// Check for any intent since neural might classify "are you alright" as conversation.
	lower := strings.ToLower(strings.TrimRight(strings.TrimSpace(nlu.Raw), "?!."))
	lower = stripNousPrefix(lower)
	for pattern, response := range metaResponses {
		if strings.Contains(lower, pattern) {
			return &ActionResult{
				DirectResponse: response,
				Source:         "canned",
			}
		}
	}

	// Explanatory/comparison prompts can be misclassified as short conversational
	// intents by neural NLU. Route them through the thinking engine first.
	if ar.Thinker != nil && looksExplanatoryQuery(nlu.Raw) && ar.Thinker.CanHandle(nlu.Raw) {
		var ctx *ThinkContext
		if ar.Tracker != nil {
			ctx = &ThinkContext{RecentTopics: []string{ar.Tracker.CurrentTopic()}}
		}
		if result := ar.Thinker.Think(nlu.Raw, ctx); result != nil && result.Text != "" {
			if !isLowInformationConversational(result.Text) {
				return &ActionResult{
					DirectResponse: result.Text,
					Source:         "thinking:" + result.Frame,
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
		// Distinguish between thanks and generic affirmations (yes, sure, ok)
		respType := RespConversational
		cleanLower := strings.ToLower(strings.TrimRight(strings.TrimSpace(nlu.Raw), "!?."))
		if isThankYou(cleanLower) {
			respType = RespThankYou
		}
		resp := ar.Composer.Compose(nlu.Raw, respType, ctx)
		if resp != nil && resp.Text != "" {
			return &ActionResult{
				DirectResponse: resp.Text,
				Source:         "composer",
			}
		}
	}

	// Follow-ups: "tell me more", "why?", "go on", "elaborate"
	// Uses FollowUpResolver (if wired) for better context-aware resolution,
	// then falls back to the old conversation history approach.
	if (nlu.Intent == "followup" || nlu.Intent == "follow_up") {
		// Try the new FollowUpResolver with conversation state first
		if ar.FollowUp != nil && ar.ConvState != nil {
			resolved := ar.FollowUp.Resolve(nlu.Raw, ar.ConvState)
			if resolved != nil && resolved.IsFollowUp && resolved.ResolvedQuery != "" {
				// Re-route the resolved query through the thinking engine
				if ar.Thinker != nil && ar.Thinker.CanHandle(resolved.ResolvedQuery) {
					var ctx *ThinkContext
					if ar.Tracker != nil {
						ctx = &ThinkContext{RecentTopics: []string{ar.Tracker.CurrentTopic()}}
					}
					if result := ar.Thinker.Think(resolved.ResolvedQuery, ctx); result != nil && result.Text != "" {
						if !isLowInformationConversational(result.Text) {
							return &ActionResult{
								DirectResponse: result.Text,
								Source:         "followup_resolved:" + result.Frame,
							}
						}
					}
				}

				// Fallback: try knowledge lookup with the resolved query
				if ar.CogGraph != nil {
					resolvedNLU := &NLUResult{
						Raw:      resolved.ResolvedQuery,
						Intent:   "explain",
						Action:   "lookup_knowledge",
						Entities: map[string]string{"topic": resolved.ResolvedQuery},
					}
					if result := ar.handleLookupKnowledge(resolvedNLU); result != nil && result.DirectResponse != "" {
						if !isLowInformationConversational(result.DirectResponse) {
							return result
						}
					}
				}
			}
		}

		prevTopic := ar.getPreviousTopic()
		if prevTopic != "" && ar.Composer != nil {
			// Determine follow-up type from the user's phrasing
			followUpType := classifyFollowUpType(lower)

			// Try the composer's follow-up method first — it uses the
			// knowledge graph and discourse corpus for rich responses.
			resp := ar.Composer.ComposeFollowUp(prevTopic, followUpType)
			if resp != nil && resp.Text != "" {
				return &ActionResult{
					DirectResponse: resp.Text,
					Source:         "followup:" + strings.Join(resp.Sources, ","),
				}
			}

			// Fallback: try the thinking engine for deeper exploration
			if ar.Thinker != nil {
				expandedQuery := prevTopic
				switch followUpType {
				case "why":
					expandedQuery = "why " + prevTopic
				case "example":
					expandedQuery = "example of " + prevTopic
				case "deeper":
					expandedQuery = "explain " + prevTopic + " in depth"
				}
				ctx := &ThinkContext{RecentTopics: []string{prevTopic}}
				result := ar.Thinker.Think(expandedQuery, ctx)
				if result != nil && result.Text != "" {
					return &ActionResult{
						DirectResponse: result.Text,
						Source:         "followup:thinking:" + result.Frame,
					}
				}
			}
		}

		// Legacy path: try the conversation tracker's fact store
		if ar.Tracker != nil {
			if ar.Tracker.IsContinuation(nlu.Raw) {
				more := ar.Tracker.ContinueResponse()
				if more != "" {
					return &ActionResult{
						DirectResponse: more,
						Source:         "continuation",
					}
				}
			}
		}
		// No continuation data — give an honest response
		return &ActionResult{
			DirectResponse: "I don't have more to add on that topic yet. Try asking about something specific, or I can search the web.",
			Source:         "fallback",
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
		ctx := ar.BuildComposeContextWithSubtext(nlu.Raw, nlu)
		respType := ar.ClassifyForComposer(nlu.Raw)

		// For knowledge queries (factual, explain), check if we have knowledge first.
		// If not, give an honest "I don't know" instead of a confusing bridge response.
		if respType == RespFactual || respType == RespExplain {
			if ar.CogGraph != nil {
				facts, _ := ar.Composer.gatherFacts(nlu.Raw)
				if len(facts) == 0 {
					// Try knowledge synthesis before honest fallback.
					if ar.Synthesizer != nil {
						topic := extractMainTopic(nlu.Raw)
						if ar.Synthesizer.ShouldSynthesize(topic, 0) {
							synthResult := ar.Synthesizer.Synthesize(topic)
							if synthResult != nil && len(synthResult.Synthesized) > 0 {
								return &ActionResult{
									DirectResponse: ar.Synthesizer.FormatSynthesis(synthResult),
									Source:         "knowledge_synthesis",
								}
							}
						}
					}
					return &ActionResult{
						DirectResponse: composeHonestFallback(nlu.Raw),
						Source:         "honest_fallback",
					}
				}
			}
		}

		// Inner Council deliberation for complex queries.
		// The council provides structured reasoning that enriches the response.
		if ar.Council != nil {
			switch respType {
			case RespOpinion, RespConversational, RespReflect, RespEmpathetic:
				delib := ar.Council.Deliberate(nlu.Raw, nlu, ctx)
				if delib != nil {
					ctx.CouncilResult = delib
				}
			}
		}

		// Check for formed opinions on the topic
		if ar.Opinions != nil && respType == RespOpinion {
			topic := nlu.Entities["topic"]
			if topic == "" {
				topic = extractMainTopic(nlu.Raw)
			}
			if op := ar.Opinions.GetOpinion(topic); op != nil && op.Confidence >= 0.3 {
				ctx.Opinion = op
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
	clean := strings.TrimRight(strings.TrimSpace(lower), "!?.\\ ")
	exactThanks := []string{
		"thanks", "thx", "ty", "cheers", "much appreciated",
		"appreciate it", "appreciated",
	}
	for _, t := range exactThanks {
		if clean == t {
			return true
		}
	}
	prefixThanks := []string{
		"thank you", "thanks a lot", "thanks so much",
		"thanks for", "thank you for", "many thanks",
	}
	for _, t := range prefixThanks {
		if strings.HasPrefix(clean, t) {
			return true
		}
	}
	return false
}

func isLowInformationConversational(text string) bool {
	clean := strings.ToLower(strings.TrimRight(strings.TrimSpace(text), "!?."))
	if clean == "" {
		return true
	}
	containsLowInfo := []string{
		"i appreciate you sharing that",
		"thank you for telling me",
		"good question",
		"that's interesting to think about",
		"late night session",
		"burning the midnight oil",
		"night owl mode",
		"make sure you're getting enough rest",
	}
	for _, p := range containsLowInfo {
		if strings.Contains(clean, p) {
			return true
		}
	}
	lowInfo := map[string]bool{
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
		"thank you for telling me":      true,
		"i appreciate you sharing that": true,
		"good question":                 true,
	}
	if lowInfo[clean] {
		return true
	}
	for prefix := range lowInfo {
		if strings.HasPrefix(clean, prefix+".") || strings.HasPrefix(clean, prefix+",") {
			return true
		}
	}

	if len(strings.Fields(clean)) <= 3 {
		switch clean {
		case "hmm", "oh", "wow", "interesting":
			return true
		}
	}

	return false
}

// looksLikeImplicitFollowUp returns true if the query contains pronouns or
// phrases that reference previous conversational context, suggesting it is a
// follow-up even though the NLU classified it as a standalone question.
func looksLikeImplicitFollowUp(query string) bool {
	lower := strings.ToLower(query)
	// Contains pronouns that reference previous context
	pronouns := []string{" that ", " this ", " it ", " its ", " his ", " her ", " their ", " those ", " these "}
	for _, p := range pronouns {
		if strings.Contains(lower, p) {
			return true
		}
	}
	// Starts with relationship phrases
	relPhrases := []string{"how does that", "how does this", "how does it", "what about", "and what about", "how is that", "why is that", "is that related"}
	for _, r := range relPhrases {
		if strings.HasPrefix(lower, r) {
			return true
		}
	}
	return false
}

// isExplicitTaskPrompt returns true for queries that explicitly request a
// structured task (explain, compare, overview, summarize, walk-me-through).
// These prompts must NEVER be intercepted by emotional/conversational
// injectors — they always route through the knowledge/thinking pipeline.
func isExplicitTaskPrompt(query string) bool {
	lower := strings.ToLower(strings.TrimSpace(query))
	for _, prefix := range []string{
		// Explain family
		"explain ", "describe ", "define ",
		"what is ", "what's ", "what are ", "what was ", "what were ",
		"what does ", "what do ",
		"who is ", "who are ", "who was ",
		"how does ", "how do ", "how is ",
		"why is ", "why are ", "why does ", "why do ",
		// Overview / teach family
		"tell me about ", "tell me everything about ", "tell me all about ",
		"give me an overview of ", "give me a full overview of ",
		"walk me through ", "deep dive into ",
		"teach me about ", "help me understand ",
		// Compare family
		"compare ", "contrast ",
		"difference between ", "differences between ",
		"pros and cons of ",
		// Summarize family
		"summarize ", "summarise ", "summary of ",
		"give me a summary of ",
		// Plan family
		"plan ", "outline ", "create a plan for ",
	} {
		if strings.HasPrefix(lower, prefix) {
			return true
		}
	}
	// Also catch "X vs Y" patterns
	if strings.Contains(lower, " vs ") || strings.Contains(lower, " versus ") {
		return true
	}
	return false
}

func looksExplanatoryQuery(query string) bool {
	lower := strings.ToLower(strings.TrimSpace(query))
	for _, prefix := range []string{
		"what is ", "what are ", "what does ",
		"how does ", "how do ", "why ",
		"tell me about ", "tell me everything about ", "tell me all about ",
		"give me an overview of ", "walk me through ",
		"summarize ", "summarise ", "summary of ",
		"explain ", "describe ", "define ",
		"compare ",
	} {
		if strings.HasPrefix(lower, prefix) {
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

// isPersonalStatement detects first-person declarative statements that should
// get a conversational response, NOT a knowledge-graph lookup.
// "tomorrow I will run for 10 miles" → true
// "what is Napoleon?" → false
func isPersonalStatement(lower string) bool {
	personalPrefixes := []string{
		"i ", "i'm ", "im ", "i'll ", "ill ", "i've ", "ive ",
		"i am ", "i will ", "i want ", "i need ", "i think ",
		"i feel ", "i have ", "i had ", "i was ", "i just ",
		"i really ", "i might ", "i could ", "i should ",
		"i can't ", "i cant ", "i don't ", "i dont ",
		"i didn't ", "i didnt ", "i won't ", "i wont ",
		"my ", "we ", "we're ", "were ", "we'll ", "we've ",
		"we should ", "we could ", "we need ", "we want ",
		"tomorrow i", "today i", "tonight i",
		"yesterday i", "last night i", "last week i",
	}
	clean := strings.TrimSpace(lower)
	for _, p := range personalPrefixes {
		if strings.HasPrefix(clean, p) {
			return true
		}
	}
	return false
}

// isContinuationRequest detects continuation requests like "tell me more", "go on".
func isContinuationRequest(lower string) bool {
	followUps := []string{
		"tell me more", "more about that", "go on", "continue",
		"keep going", "what else", "more about", "dig deeper",
		"elaborate", "and then", "go ahead", "anything else",
		"why is that", "how come", "can you explain",
		"explain that", "what do you mean", "in more detail",
	}
	clean := strings.TrimRight(strings.TrimSpace(lower), "!?.")
	for _, f := range followUps {
		if clean == f || strings.HasPrefix(clean, f) {
			return true
		}
	}
	// Short single-word follow-ups
	shortFollowUps := []string{
		"why", "how", "really", "interesting", "huh",
		"and", "so", "meaning", "more", "seriously",
	}
	for _, sf := range shortFollowUps {
		if clean == sf {
			return true
		}
	}
	return false
}

// getPreviousTopic returns the topic from the most recent conversation turn.
// Checks the conversation tracker, then composer history, as fallbacks.
func (ar *ActionRouter) getPreviousTopic() string {
	// 1. Conversation tracker — the authoritative source for current topic
	if ar.Tracker != nil {
		if topic := ar.Tracker.CurrentTopic(); topic != "" {
			return topic
		}
	}

	// 2. Composer history — last turn's topics
	if ar.Composer != nil && len(ar.Composer.history) > 0 {
		last := ar.Composer.history[len(ar.Composer.history)-1]
		if len(last.Topics) > 0 {
			return last.Topics[0]
		}
		// 3. Fallback: extract topic from the last input
		if last.Input != "" {
			keywords := extractKeywords(strings.ToLower(last.Input))
			if len(keywords) > 0 {
				return keywords[0]
			}
		}
	}

	return ""
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
// extractMainTopic extracts the primary topic from a query.
func extractMainTopic(query string) string {
	lower := strings.ToLower(strings.TrimRight(strings.TrimSpace(query), "?!."))
	// Long prefixes first for greedy matching.
	prefixes := []string{
		"give me a full overview of ", "give me an overview of ",
		"give me a summary of ", "tell me everything about ",
		"tell me all about ", "tell me about ",
		"walk me through ", "deep dive into ",
		"teach me about ", "help me understand ",
		"what do you think about ", "what's your opinion on ",
		"your take on ", "how do you feel about ",
		"explain how ", "explain why ", "explain what ",
		"explain to me ", "explain ", "describe ", "define ",
		"what is ", "what are ", "what was ",
		"who is ", "who are ", "who was ",
		"how does ", "how do ", "how is ",
		"why is ", "why are ", "why does ",
		"compare ", "summarize ", "summarise ",
		"i want to learn about ", "i want to know about ",
		"i want to learn ", "i want to study ",
		"help me decide about ", "help me with ",
	}
	for _, p := range prefixes {
		if strings.HasPrefix(lower, p) {
			return strings.TrimSpace(lower[len(p):])
		}
	}
	// Fallback: strip personal/filler prefixes, keep multi-word phrases intact.
	stripped := lower
	for _, s := range []string{
		"i am ", "i'm ", "i have been ", "i've been ",
		"i want to ", "i need to ", "help me ",
		"can you ", "could you ", "please ",
		"how to ", "what about ", "about ",
	} {
		stripped = strings.TrimPrefix(stripped, s)
	}
	stripped = strings.TrimSpace(stripped)
	if stripped != "" && stripped != lower {
		return stripped
	}
	// Last resort: return all content words joined (preserves multi-word topics).
	words := extractKeywords(lower)
	if len(words) > 1 {
		return strings.Join(words, " ")
	}
	if len(words) == 1 {
		return words[0]
	}
	return lower
}

// gatherFactsForNLG retrieves raw edgeFacts from the cognitive graph for NLG.
func (ar *ActionRouter) gatherFactsForNLG(topic string) []edgeFact {
	if ar.CogGraph == nil {
		return nil
	}
	lower := strings.ToLower(strings.TrimSpace(topic))
	ar.CogGraph.mu.RLock()
	defer ar.CogGraph.mu.RUnlock()

	// Try exact match first, then try individual words for compound topics
	ids := ar.CogGraph.byLabel[lower]
	if len(ids) == 0 {
		// Try each word: "quantum physics" → try "quantum", "physics"
		for _, word := range strings.Fields(lower) {
			if len(word) > 3 {
				if wordIDs := ar.CogGraph.byLabel[word]; len(wordIDs) > 0 {
					ids = append(ids, wordIDs...)
				}
			}
		}
	}
	if len(ids) == 0 {
		return nil
	}

	var facts []edgeFact
	seen := make(map[string]bool)
	for _, id := range ids {
		for _, edge := range ar.CogGraph.outEdges[id] {
			if edge.Relation == RelDescribedAs {
				continue
			}
			target, ok := ar.CogGraph.nodes[edge.To]
			if !ok {
				continue
			}
			obj := target.Label
			// Quality filter: skip facts with fragment/contaminated objects.
			// Max 60 chars for non-description facts prevents long sentence
			// fragments from entering the NLG pipeline.
			objLower := strings.ToLower(obj)
			if len(obj) < 3 || len(obj) > 60 ||
				strings.HasSuffix(objLower, " by") || strings.HasSuffix(objLower, " in") ||
				strings.HasSuffix(objLower, " at") || strings.HasSuffix(objLower, " and") ||
				strings.HasSuffix(objLower, " or") || strings.HasSuffix(objLower, " the") ||
				strings.HasSuffix(objLower, " a") || strings.HasSuffix(objLower, " of") ||
				strings.HasSuffix(objLower, " to") || strings.HasSuffix(objLower, " for") ||
				strings.Contains(obj, ". ") || strings.Contains(objLower, " such as ") ||
				strings.Contains(objLower, "influenced by") || strings.Contains(objLower, "prose style") ||
				strings.HasSuffix(objLower, "progra") || strings.HasSuffix(objLower, "peop") ||
				strings.HasSuffix(objLower, "intelligenc") {
				continue
			}

			// Deduplicate: skip if we already have a fact with the same
			// relation where one object contains the other
			relKey := string(edge.Relation)
			duplicate := false
			for _, existing := range facts {
				if string(existing.Relation) == relKey {
					existLower := strings.ToLower(existing.Object)
					newLower := strings.ToLower(obj)
					if strings.Contains(existLower, newLower) || strings.Contains(newLower, existLower) {
						duplicate = true
						break
					}
				}
			}
			if duplicate {
				continue
			}

			key := relKey + ":" + obj
			if seen[key] {
				continue
			}
			seen[key] = true
			facts = append(facts, edgeFact{
				Subject:  ar.CogGraph.nodes[id].Label,
				Relation: edge.Relation,
				Object:   target.Label,
			})
			if len(facts) >= 12 {
				return facts
			}
		}
	}
	return facts
}

// cleanTopicForLookup strips question-word prefixes from a topic so that
// graph lookup finds "photosynthesis" from "how photosynthesis works".
// This is NOT pattern matching for NLU — it's string cleaning for retrieval.
func cleanTopicForLookup(topic string) string {
	lower := strings.ToLower(strings.TrimSpace(topic))
	// Strip leading question/explanation words
	for _, prefix := range []string{
		"how does ", "how do ", "how is ", "how are ",
		"how ", "why does ", "why do ", "why is ",
		"why ", "what is ", "what are ", "what was ",
		"where is ", "where are ", "when was ", "when did ",
	} {
		if strings.HasPrefix(lower, prefix) {
			lower = strings.TrimSpace(lower[len(prefix):])
			break
		}
	}
	// Strip trailing verbs: "photosynthesis works" → "photosynthesis"
	for _, suffix := range []string{
		" work", " works", " happen", " happens", " function", " functions",
		" operate", " operates", " mean", " means",
	} {
		if strings.HasSuffix(lower, suffix) {
			lower = strings.TrimSpace(lower[:len(lower)-len(suffix)])
		}
	}
	if lower == "" {
		return topic // don't return empty
	}
	return lower
}

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

	ctx.TimeOfDay = time.Now()

	return ctx
}

// BuildComposeContextWithSubtext builds context enriched with subtext analysis.
func (ar *ActionRouter) BuildComposeContextWithSubtext(input string, nluResult *NLUResult) *ComposeContext {
	ctx := ar.BuildComposeContext()

	// Run subtext analysis if available
	if ar.Subtext != nil {
		var history []ConvTurn
		if ar.Composer != nil {
			history = ar.Composer.history
		}
		analysis := ar.Subtext.Analyze(input, nluResult, history)
		ctx.Subtext = &analysis
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
		expr = nlu.Entities["expression"]
	}
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
				if isCorruptedMemoryEntry(key, val) {
					continue
				}
				seen[key] = true
				personalFacts = append(personalFacts, formatPersonalFact(key, val))
			}
		}
		// Also search for "personal" category, skipping already-seen keys
		entries := ar.LongTermMem.Search("personal")
		for _, e := range entries {
			if !seen[e.Key] && !isCorruptedMemoryEntry(e.Key, e.Value) {
				seen[e.Key] = true
				personalFacts = append(personalFacts, formatPersonalFact(e.Key, e.Value))
			}
		}
		if len(personalFacts) == 1 {
			return &ActionResult{
				DirectResponse: personalFacts[0],
				Source:         "memory",
			}
		}
		if len(personalFacts) > 0 {
			return &ActionResult{
				DirectResponse: "Here's what I know about you:\n- " + strings.Join(personalFacts, "\n- "),
				Source:         "memory",
			}
		}
		return &ActionResult{
			DirectResponse: "I don't have much information about you yet. Tell me about yourself!",
			Source:         "memory",
		}
	}

	var parts []string

	// Extract a clean key from the query for lookup.
	// "what is my favorite food" → "favorite food"
	cleanKey := strings.ToLower(query)
	for _, prefix := range []string{
		"what is my ", "what's my ", "whats my ",
		"do you know my ", "do you remember my ",
		"what was my ", "tell me my ",
	} {
		if strings.HasPrefix(cleanKey, prefix) {
			cleanKey = strings.TrimPrefix(cleanKey, prefix)
			cleanKey = strings.TrimRight(cleanKey, "?!. ")
			break
		}
	}

	// Long-term memory — persistent facts (checked first, most reliable).
	if ar.LongTermMem != nil {
		// Try exact match on clean key first, then original query
		if val, ok := ar.LongTermMem.Retrieve(cleanKey); ok {
			parts = append(parts, fmt.Sprintf("[longterm] %s: %s", cleanKey, val))
		} else if cleanKey != query {
			if val, ok := ar.LongTermMem.Retrieve(query); ok {
				parts = append(parts, fmt.Sprintf("[longterm] %s: %s", query, val))
			}
		}
		if cat := nlu.Entities["category"]; cat != "" {
			entries := ar.LongTermMem.Search(cat)
			for _, e := range entries {
				parts = append(parts, fmt.Sprintf("[longterm:%s] %s: %s", e.Category, e.Key, e.Value))
			}
		}
		// Also search personal category for partial key match
		if len(parts) == 0 {
			entries := ar.LongTermMem.Search("personal")
			for _, e := range entries {
				if strings.Contains(strings.ToLower(e.Key), cleanKey) ||
					strings.Contains(cleanKey, strings.ToLower(e.Key)) {
					parts = append(parts, fmt.Sprintf("[longterm:%s] %s: %s", e.Category, e.Key, e.Value))
				}
			}
		}
	}

	// Working memory — only include entries relevant to the query,
	// and skip if already found in longterm to avoid duplicates.
	if ar.WorkingMem != nil && len(parts) == 0 {
		slots := ar.WorkingMem.MostRelevant(3)
		for _, s := range slots {
			slotKey := strings.ToLower(fmt.Sprintf("%v", s.Key))
			// Only include if the slot key matches the query
			if strings.Contains(slotKey, cleanKey) || strings.Contains(cleanKey, slotKey) {
				parts = append(parts, fmt.Sprintf("[working] %s: %v", s.Key, s.Value))
			}
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

	// Simple recall: if there's exactly one longterm fact, return it naturally.
	// This handles "what is my favorite animal", "what's my email", etc.
	var longtermParts []string
	for _, p := range parts {
		if strings.HasPrefix(p, "[longterm]") {
			longtermParts = append(longtermParts, p)
		}
	}
	if len(longtermParts) == 1 && len(parts) <= 2 {
		afterPrefix := strings.TrimPrefix(longtermParts[0], "[longterm] ")
		if valIdx := strings.Index(afterPrefix, ": "); valIdx > 0 {
			key := afterPrefix[:valIdx]
			value := afterPrefix[valIdx+2:]
			if value != "" {
				return &ActionResult{
					DirectResponse: formatPersonalFact(key, value),
					Source:         "memory",
				}
			}
		}
	}

	// Multiple facts or complex queries — format each naturally.
	var formatted []string
	for _, p := range parts {
		if strings.HasPrefix(p, "[longterm]") || strings.HasPrefix(p, "[longterm:") {
			// Strip prefix and format
			clean := p
			for _, pfx := range []string{"[longterm] ", "[longterm:personal] "} {
				clean = strings.TrimPrefix(clean, pfx)
			}
			if idx := strings.Index(clean, "] "); idx >= 0 {
				clean = clean[idx+2:]
			}
			if valIdx := strings.Index(clean, ": "); valIdx > 0 {
				formatted = append(formatted, formatPersonalFact(clean[:valIdx], clean[valIdx+2:]))
			}
		} else if strings.HasPrefix(p, "[working] ") {
			// Format working memory facts naturally instead of raw tags
			clean := strings.TrimPrefix(p, "[working] ")
			if valIdx := strings.Index(clean, ": "); valIdx > 0 {
				formatted = append(formatted, formatPersonalFact(clean[:valIdx], clean[valIdx+2:]))
			}
		}
	}
	if len(formatted) > 0 {
		if len(formatted) == 1 {
			return &ActionResult{DirectResponse: formatted[0], Source: "memory"}
		}
		return &ActionResult{
			DirectResponse: "Here's what I remember:\n- " + strings.Join(formatted, "\n- "),
			Source:         "memory",
		}
	}
	return &ActionResult{DirectResponse: strings.Join(parts, "\n"), Source: "memory"}
}

// parsePersonalFact extracts a key-value pair from personal statements.
// "remember my favorite color is blue" → ("favorite color", "blue")
// "my name is Raphael" → ("name", "Raphael")
// "i like pizza" → ("likes", "pizza")
// isCorruptedMemoryEntry detects entries that were accidentally stored from
// misclassified inputs (e.g., commands or questions stored as personal facts).
func isCorruptedMemoryEntry(key, value string) bool {
	lower := strings.ToLower(value)
	// Commands stored as values
	if strings.HasPrefix(lower, "/") || strings.HasPrefix(lower, "!") {
		return true
	}
	// Questions stored as values
	if strings.HasSuffix(strings.TrimSpace(lower), "?") {
		return true
	}
	// Very long values are likely misclassified inputs
	if len(value) > 200 {
		return true
	}
	// Keys that look like dotted internal paths with suspicious values
	if strings.Contains(key, ".") && (strings.HasPrefix(lower, "/") ||
		strings.Contains(lower, "what ") || strings.Contains(lower, "how ")) {
		return true
	}
	return false
}

// formatPersonalFact presents a key-value pair in natural language.
func formatPersonalFact(key, value string) string {
	// Normalize the key for display
	displayKey := strings.ReplaceAll(key, "_", " ")
	displayKey = strings.ReplaceAll(displayKey, "user.", "")
	displayKey = strings.ReplaceAll(displayKey, "user ", "")

	switch {
	case strings.Contains(displayKey, "name"):
		return fmt.Sprintf("Your name is %s", value)
	case strings.Contains(displayKey, "role"):
		return fmt.Sprintf("Your role is %s", value)
	case strings.Contains(displayKey, "location"):
		return fmt.Sprintf("You're located in %s", value)
	case strings.Contains(displayKey, "email"):
		return fmt.Sprintf("Your email is %s", value)
	case strings.Contains(displayKey, "interest"):
		return fmt.Sprintf("You're interested in %s", value)
	case strings.Contains(displayKey, "favorite"):
		return fmt.Sprintf("Your %s is %s", displayKey, value)
	default:
		return fmt.Sprintf("Your %s is %s", displayKey, value)
	}
}

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

// -----------------------------------------------------------------------
// Knowledge paragraph cache — read knowledge/*.txt once, reuse forever.
// -----------------------------------------------------------------------

var (
	knowledgeParagraphCache   []string
	knowledgeParagraphCacheOnce sync.Once
)

// loadKnowledgeParagraphs reads all .txt files in dir, splits them on
// blank lines, trims whitespace, and returns every non-empty paragraph.
func loadKnowledgeParagraphs(dir string) []string {
	files, err := filepath.Glob(filepath.Join(dir, "*.txt"))
	if err != nil || len(files) == 0 {
		return nil
	}
	var paragraphs []string
	for _, f := range files {
		data, err := os.ReadFile(f)
		if err != nil {
			continue
		}
		for _, p := range splitParagraphs(string(data)) {
			if p = strings.TrimSpace(p); p != "" {
				paragraphs = append(paragraphs, p)
			}
		}
	}
	return paragraphs
}

// findKnowledgeParagraph searches knowledge text files for a paragraph
// about the given topic and returns it verbatim. The paragraph is real
// human-written Wikipedia-quality prose — no reconstruction needed.
func findKnowledgeParagraph(knowledgeDir, topic string) string {
	if knowledgeDir == "" || topic == "" {
		return ""
	}

	knowledgeParagraphCacheOnce.Do(func() {
		knowledgeParagraphCache = loadKnowledgeParagraphs(knowledgeDir)
	})

	topicLower := strings.ToLower(strings.TrimSpace(topic))
	if topicLower == "" {
		return ""
	}

	// Strategy 1: first sentence starts with the topic.
	// "Gravity is the fundamental force..." matches topic "gravity".
	// "Quantum mechanics is a fundamental theory..." matches "quantum mechanics".
	for _, p := range knowledgeParagraphCache {
		firstSent := firstSentenceOf(p)
		firstLower := strings.ToLower(firstSent)
		if strings.HasPrefix(firstLower, topicLower+" ") || strings.HasPrefix(firstLower, topicLower+",") {
			return p
		}
	}

	// Strategy 2: the first word of the paragraph matches a single-word topic.
	if !strings.Contains(topicLower, " ") {
		for _, p := range knowledgeParagraphCache {
			firstWord := strings.ToLower(strings.Fields(p)[0])
			// Strip trailing punctuation from first word
			firstWord = strings.TrimRight(firstWord, ".,;:!?")
			if firstWord == topicLower {
				return p
			}
		}
	}

	// Strategy 3: full-phrase containment — the first sentence contains
	// the entire topic phrase near the beginning (subject position).
	// For single-word topics, require it in the first 60 chars to avoid
	// tangential mentions ("consciousness" in a Virginia Woolf article).
	for _, p := range knowledgeParagraphCache {
		firstSent := firstSentenceOf(p)
		firstLower := strings.ToLower(firstSent)
		idx := strings.Index(firstLower, topicLower)
		if idx < 0 {
			continue
		}
		// Multi-word topics: any position in the first sentence is fine.
		// Single-word topics: require subject position (first 60 chars).
		if strings.Contains(topicLower, " ") || idx < 60 {
			return p
		}
	}

	// Strategy 4: for multi-word topics, require the FIRST significant
	// word of the topic to match the FIRST significant word of the
	// paragraph. This prevents "programming languages" from matching
	// "Language learning basics" via the word "language".
	words := strings.Fields(topicLower)
	if len(words) > 1 {
		topicFirstWord := ""
		for _, w := range words {
			if len(w) > 3 {
				topicFirstWord = w
				break
			}
		}
		if topicFirstWord != "" {
			for _, p := range knowledgeParagraphCache {
				paraFirstWord := strings.ToLower(strings.TrimRight(strings.Fields(p)[0], ".,;:!?"))
				if paraFirstWord == topicFirstWord {
					return p
				}
			}
		}
	}

	return ""
}

// firstSentenceOf returns everything up to the first period followed by a
// space (or the end of string). Used to check what a paragraph is "about".
func firstSentenceOf(text string) string {
	idx := strings.Index(text, ". ")
	if idx >= 0 {
		return text[:idx+1]
	}
	return text
}

// resolveCompoundTopic checks if any multi-word substring of the query
// matches a known topic in the knowledge graph. This handles "world war 2",
// "social media", "machine learning" etc. that would otherwise be split
// into individual words by the noun-phrase extractor.
func (ar *ActionRouter) resolveCompoundTopic(query string) string {
	if ar.CogGraph == nil {
		return ""
	}
	queryLower := strings.ToLower(strings.TrimSpace(query))
	if queryLower == "" {
		return ""
	}

	labels := ar.CogGraph.AllLabels()

	// Collect multi-word labels that appear inside the query.
	// Return the longest match so "world war ii" beats "world war".
	best := ""
	for _, label := range labels {
		if !strings.Contains(label, " ") {
			continue // single-word labels handled elsewhere
		}
		if strings.Contains(queryLower, label) && len(label) > len(best) {
			best = label
		}
	}
	return best
}

// handleLookupKnowledge searches the knowledge vector store.
func (ar *ActionRouter) handleLookupKnowledge(nlu *NLUResult) *ActionResult {
	// Planning questions ("how do I learn guitar?") → generate a learning plan
	if IsPlanningQuestion(nlu.Raw) {
		goal := ExtractGoal(nlu.Raw)
		// Always use the generic learning plan template for now.
		// The graph-based planner produces low-quality results because
		// the knowledge graph contains reference data, not learning paths.
		return &ActionResult{
			DirectResponse: formatGenericLearningPlan(goal),
			Source:         "planner",
		}
	}

	query := nlu.Entities["topic"]
	if query == "" {
		query = nlu.Entities["query"]
	}
	if query == "" {
		query = nlu.Raw
	}

	// Extract the core noun phrase for graph matching.
	// "how photosynthesis works" → "photosynthesis"
	// "give me an overview of operating systems" → "operating systems"
	if np := ExtractNounPhrase(query); np != "" {
		query = np
	} else {
		query = cleanTopicForLookup(query)
	}

	// Try to resolve compound topics using graph labels as dictionary.
	// This prevents "world war 2" from being split into "world".
	if ar.CogGraph != nil {
		if compound := ar.resolveCompoundTopic(query); compound != "" {
			query = compound
		}
	}

	// Primary path: find the original Wikipedia paragraph about this topic.
	// This is real human-written prose — no reconstruction, no templates.
	if ar.SelfTeacher != nil {
		if knowledgeParagraph := findKnowledgeParagraph(ar.SelfTeacher.KnowledgeDir(), query); knowledgeParagraph != "" {
			// Apply format compliance if requested
			if ar.Format != nil {
				if req := ar.Format.DetectFormat(nlu.Raw); req != nil {
					knowledgeParagraph = ar.Format.Reshape(knowledgeParagraph, req)
				}
			}
			// Track this topic for follow-up conversation context.
			if ar.Tracker != nil && query != "" {
				ar.Tracker.TrackTopic(query, "knowledge_text")
			}
			return &ActionResult{
				DirectResponse: knowledgeParagraph,
				Source:         "knowledge_text",
			}
		}
	}

	// Tier 1.5: Semantic retrieval — when exact paragraph match fails, find
	// the most relevant knowledge chunk via embedding similarity. 220 chunks,
	// 50-dim cosine similarity — instant (~1ms).
	if ar.Knowledge != nil {
		results, err := ar.Knowledge.Search(query, 3)
		if err == nil && len(results) > 0 && results[0].Score > 0.50 {
			bestText := results[0].Text
			// Verify the top result is actually about this topic.
			// Check the first 400 chars (not just 200) to catch topics
			// mentioned after an introductory clause.
			bestLower := strings.ToLower(bestText)
			queryLower := strings.ToLower(query)
			relevant := false
			checkLen := min(len(bestLower), 400)
			for _, w := range strings.Fields(queryLower) {
				if len(w) > 3 && strings.Contains(bestLower[:checkLen], w) {
					relevant = true
					break
				}
			}
			if relevant {
				if ar.Format != nil {
					if req := ar.Format.DetectFormat(nlu.Raw); req != nil {
						bestText = ar.Format.Reshape(bestText, req)
					}
				}
				if ar.Tracker != nil && query != "" {
					ar.Tracker.TrackTopic(query, "semantic_retrieval")
				}
				return &ActionResult{
					DirectResponse: bestText,
					Source:         "semantic_retrieval",
				}
			}
		}
	}

	// Deep reasoning: for "why", "how does X affect Y", and similar complex
	// questions, use multi-step structured reasoning with explicit chains.
	// Fires AFTER the Wikipedia paragraph path (which serves direct answers)
	// but BEFORE NLG reconstruction (which synthesises from graph fragments).
	if IsDeepQuestion(nlu.Raw) && ar.DeepReason != nil {
		result := ar.DeepReason.Reason(nlu.Raw)
		if result != nil && result.FinalAnswer != "" {
			return &ActionResult{
				DirectResponse: result.Trace + "\n\n" + result.FinalAnswer,
				Source:         "deep_reasoning",
			}
		}
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

	// NLG: generate prose from graph facts using best available tier.
	if ar.CogGraph != nil {
		facts := ar.gatherFactsForNLG(query)
		if len(facts) >= 2 {
			var prose string

			// Tier 1: Hybrid generator — retrieval + recombination + neural connectors
			if ar.HybridGen != nil {
				if nlu.Intent == "explain" || nlu.Intent == "question" {
					prose = ar.HybridGen.GenerateExplanation(query, facts)
				} else {
					prose = ar.HybridGen.Generate(query, facts)
				}
			}

			// Tier 2: Structural NLG engine — sentence fusion + pronominalization
			if (prose == "" || len(strings.Fields(prose)) < 10) && ar.NLG != nil {
				if nlu.Intent == "explain" || nlu.Intent == "question" {
					prose = ar.NLG.RealizeExplanation(query, facts)
				} else {
					prose = ar.NLG.Realize(query, facts)
				}
			}

			if prose != "" && len(strings.Fields(prose)) >= 10 {
				// Supplement with Wikipedia description if available
				desc := ar.CogGraph.LookupDescription(query)
				if len(desc) >= 40 {
					prose = desc + "\n\n" + prose
				}

				// Apply format compliance if user requested specific formatting
				if ar.Format != nil {
					if req := ar.Format.DetectFormat(nlu.Raw); req != nil {
						prose = ar.Format.Reshape(prose, req)
					}
				}

				return &ActionResult{
					DirectResponse: prose,
					Source:         "nlg",
				}
			}
		}
	}

	// Fast path: if we have Wikipedia knowledge (lead paragraph + discourse
	// sentences), compose the response from REAL human-written text.
	// No templates. Every word from Wikipedia.
	if ar.CogGraph != nil {
		// For follow-up questions, use discourse corpus to answer the
		// specific question type instead of repeating the definition.
		followUpType := nlu.Entities["_follow_up_type"]
		if followUpType != "" && ar.Composer != nil && ar.Composer.DiscourseCorpus != nil {
			var funcs []DiscourseFunc
			switch followUpType {
			case "why":
				funcs = []DiscourseFunc{DFExplainsWhy, DFConsequence}
			case "how":
				funcs = []DiscourseFunc{DFDescribes, DFGivesExample}
			case "example":
				funcs = []DiscourseFunc{DFGivesExample, DFDescribes}
			case "quantify":
				funcs = []DiscourseFunc{DFQuantifies, DFContext}
			case "consequence":
				funcs = []DiscourseFunc{DFConsequence, DFExplainsWhy}
			case "compare":
				funcs = []DiscourseFunc{DFCompares, DFEvaluates}
			default: // "elaborate"
				funcs = []DiscourseFunc{DFEvaluates, DFGivesExample, DFContext, DFQuantifies}
			}
			sents := ar.Composer.DiscourseCorpus.RetrieveMulti(query, funcs)
			if len(sents) > 0 {
				return &ActionResult{
					DirectResponse: strings.Join(sents, " "),
					Source:         "discourse",
				}
			}
		}

		desc := ar.CogGraph.LookupDescription(query)
		if len(desc) < 40 {
			desc = "" // too short to be a real description
		}

		// Layer 2b: supplement with discourse-typed sentences from the
		// topic's own Wikipedia article. These are real sentences tagged
		// by how they communicate (explains_why, evaluates, gives_example).
		var supplement string
		if ar.Composer != nil && ar.Composer.DiscourseCorpus != nil {
			// Pick discourse functions that complement the lead paragraph.
			suppFuncs := []DiscourseFunc{DFEvaluates, DFExplainsWhy, DFContext, DFGivesExample, DFQuantifies}
			suppSents := ar.Composer.DiscourseCorpus.RetrieveMulti(query, suppFuncs)
			if desc != "" && len(suppSents) > 0 {
				// Filter sentences already in the description.
				descLower := strings.ToLower(desc)
				var filtered []string
				for _, s := range suppSents {
					// Skip if >50% of words overlap with description.
					if !sentenceOverlaps(s, descLower) {
						filtered = append(filtered, s)
					}
				}
				if len(filtered) > 3 {
					filtered = filtered[:3]
				}
				if len(filtered) > 0 {
					supplement = strings.Join(filtered, " ")
				}
			} else if desc == "" && len(suppSents) > 0 {
				// No lead paragraph — use discourse sentences as the response.
				supplement = strings.Join(suppSents, " ")
			}
		}

		// Fallback: if discourse corpus didn't provide supplements,
		// use structured facts from the knowledge graph.
		if supplement == "" {
			facts := ar.CogGraph.LookupFacts(query, 4)
			if len(facts) > 0 {
				if ar.Composer != nil {
					facts = ar.Composer.applyPronounVariation(facts, query)
				}
				supplement = strings.Join(facts, " ")
			}
		}

		if desc != "" || supplement != "" {
			var response string
			if desc != "" {
				response = desc
			}
			if supplement != "" {
				if response != "" {
					response += "\n\n" + supplement
				} else {
					response = supplement
				}
			}
			// Track this topic for follow-up conversation context.
			if ar.Tracker != nil && query != "" {
				ar.Tracker.TrackTopic(query, "knowledge")
			}
			return &ActionResult{
				DirectResponse: response,
				Source:         "knowledge",
			}
		}
	}

	// Common sense fallback: when wiki knowledge doesn't cover the topic,
	// try everyday associations. "what should I have for dinner?" → food suggestions.
	// This runs AFTER wiki lookup so factual topics go through Wikipedia.
	if ar.CommonSense != nil {
		resolved := ar.CommonSense.Resolve(nlu.Raw)
		if resolved != nil && resolved.Topic != "" {
			suggestions := ar.CommonSense.Suggest(resolved.Topic, resolved.Context)
			if len(suggestions) > 0 {
				response := strings.Join(suggestions[:min(len(suggestions), 3)], " ")
				return &ActionResult{
					DirectResponse: response,
					Source:         "commonsense",
				}
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

	// Tier 3b: Self-teach — mine knowledge files on the fly for new facts.
	if ar.SelfTeacher != nil && !ar.SelfTeacher.HasLearned(query) {
		learned, _ := ar.SelfTeacher.LearnAbout(query)
		if learned > 0 {
			facts := ar.gatherFactsForNLG(query)
			if len(facts) >= 2 && ar.NLG != nil {
				prose := ar.NLG.RealizeExplanation(query, facts)
				if prose != "" && len(strings.Fields(prose)) >= 10 {
					if ar.Format != nil {
						if req := ar.Format.DetectFormat(nlu.Raw); req != nil {
							prose = ar.Format.Reshape(prose, req)
						}
					}
					return &ActionResult{
						DirectResponse: prose,
						Source:         "self_teach",
					}
				}
			}
		}
	}

	// Tier 3c: Two-tier retrieval — BM25 + semantic + graph fusion.
	// This is the unified fallback that replaces the old scattered
	// deep retrieval, multi-hop, and synthesis tiers.
	if ar.Retriever != nil {
		results := ar.Retriever.Retrieve(query, 5)
		if len(results) > 0 && results[0].Score > 0.3 {
			var parts []string
			seen := make(map[string]bool)
			for _, r := range results {
				if r.Score > 0.2 && !seen[r.Text] {
					seen[r.Text] = true
					parts = append(parts, r.Text)
				}
				if len(parts) >= 3 {
					break
				}
			}
			if len(parts) > 0 {
				return &ActionResult{
					DirectResponse: strings.Join(parts, "\n\n"),
					Source:         "two_tier_retrieval",
				}
			}
		}
	}

	// Tier 4: On-demand Wikipedia — fetch article, extract facts, answer.
	// This is the "infinite knowledge" tier: if the topic exists on Wikipedia,
	// Nous learns about it in ~200ms and knows it forever.
	if ar.WikiLoader != nil && query != "" {
		result := ar.WikiLoader.FetchAndLearn(query)
		if result != nil && result.Paragraph != "" {
			response := result.Paragraph
			if ar.Format != nil {
				if req := ar.Format.DetectFormat(nlu.Raw); req != nil {
					response = ar.Format.Reshape(response, req)
				}
			}
			if ar.Tracker != nil {
				ar.Tracker.TrackTopic(query, "wikipedia_ondemand")
			}
			return &ActionResult{
				DirectResponse: response,
				Source:         "wikipedia_ondemand",
			}
		}
	}

	// Tier 5: Knowledge Synthesis — reason from adjacent knowledge.
	// Instead of "I don't know", synthesize qualified insights from related
	// topics in the graph using generalization, analogy, causal chains, etc.
	if ar.Synthesizer != nil && query != "" {
		if ar.Synthesizer.ShouldSynthesize(query, 0) {
			synthResult := ar.Synthesizer.Synthesize(query)
			if synthResult != nil && len(synthResult.Synthesized) > 0 {
				formatted := ar.Synthesizer.FormatSynthesis(synthResult)
				if formatted != "" {
					return &ActionResult{
						DirectResponse: formatted,
						Source:         "knowledge_synthesis",
					}
				}
			}
		}
	}

	// Tier 6: Honest fallback — we don't have relevant information.
	return &ActionResult{
		DirectResponse: composeHonestFallback(nlu.Raw),
		Source:         "honest_fallback",
	}
}

// handleCompare generates a comparison between two items using knowledge graph data.
func (ar *ActionRouter) handleCompare(nlu *NLUResult) *ActionResult {
	raw := strings.ToLower(nlu.Raw)
	normalized := strings.TrimSpace(raw)
	for _, prefix := range []string{"compare ", "contrast ", "differences between ", "difference between "} {
		normalized = strings.TrimPrefix(normalized, prefix)
	}

	// Extract the two items to compare
	var itemA, itemB string
	// Try "X vs Y", "X versus Y", "X or Y", "X and Y" patterns
	for _, sep := range []string{" vs ", " versus ", " vs. "} {
		if idx := strings.Index(normalized, sep); idx > 0 {
			itemA = strings.TrimSpace(normalized[:idx])
			itemB = strings.TrimSpace(normalized[idx+len(sep):])
		}
	}
	if itemA == "" {
		// Try "compare X and Y", "compare X with Y"
		stripped := normalized
		for _, sep := range []string{" and ", " with ", " to "} {
			if idx := strings.Index(stripped, sep); idx > 0 {
				itemA = strings.TrimSpace(stripped[:idx])
				itemB = strings.TrimSpace(stripped[idx+len(sep):])
				break
			}
		}
	}
	// "should I learn X or Y", "which is better X or Y"
	if itemA == "" {
		if idx := strings.Index(normalized, " or "); idx > 0 {
			before := normalized[:idx]
			after := strings.TrimSpace(normalized[idx+4:])
			// Extract last word(s) before "or" as item A
			words := strings.Fields(before)
			if len(words) > 0 {
				itemA = words[len(words)-1]
			}
			// First word(s) after "or" as item B
			afterWords := strings.Fields(after)
			if len(afterWords) > 0 {
				itemB = afterWords[0]
			}
		}
	}
	// Clean trailing punctuation
	itemA = strings.TrimRight(itemA, "?!.")
	itemB = strings.TrimRight(itemB, "?!.")

	if itemA == "" || itemB == "" {
		// Can't extract two items — fall back to knowledge lookup
		return ar.handleLookupKnowledge(nlu)
	}

	// Load wiki data for both items if available (must happen before disambiguation)
	if ar.Packages != nil {
		ar.Packages.LookupWiki(itemA)
		ar.Packages.LookupWiki(itemB)
	}

	// Disambiguate: if both items have "(programming language)" variants, prefer those.
	// This handles "compare rust and go" → "rust (programming language)" vs "go (programming language)".
	itemA, itemB = ar.disambiguateComparisonPair(itemA, itemB)

	// Load wiki data for disambiguated items too
	if ar.Packages != nil {
		ar.Packages.LookupWiki(itemA)
		ar.Packages.LookupWiki(itemB)
	}

	// Look up descriptions and facts for each.
	var descA, descB string
	var factsA, factsB []string
	if ar.CogGraph != nil {
		descA = ar.CogGraph.LookupDescription(itemA)
		descB = ar.CogGraph.LookupDescription(itemB)
		factsA = ar.CogGraph.LookupFacts(itemA, 5)
		factsB = ar.CogGraph.LookupFacts(itemB, 5)
		// Fallback: try base name (without qualifier) for descriptions and facts.
		// For facts, only use base-name results if they match the qualifier's domain
		// (e.g., "rust" facts are corrosion — don't use for "rust (programming language)").
		baseA := stripQualifier(itemA)
		baseB := stripQualifier(itemB)
		if descA == "" && baseA != itemA {
			descA = ar.CogGraph.LookupDescription(baseA)
		}
		if descB == "" && baseB != itemB {
			descB = ar.CogGraph.LookupDescription(baseB)
		}
		if len(factsA) == 0 && baseA != itemA {
			qualifier := extractQualifier(itemA)
			baseFacts := ar.CogGraph.LookupFacts(baseA, 8)
			factsA = filterFactsByDomain(baseFacts, qualifier)
		}
		if len(factsB) == 0 && baseB != itemB {
			qualifier := extractQualifier(itemB)
			baseFacts := ar.CogGraph.LookupFacts(baseB, 8)
			factsB = filterFactsByDomain(baseFacts, qualifier)
		}
	}

	// Build comparison response — use clean display names without qualifiers
	displayA := strings.Title(stripQualifier(itemA))
	displayB := strings.Title(stripQualifier(itemB))
	var sb strings.Builder
	sb.WriteString(fmt.Sprintf("**%s vs %s**\n\n", displayA, displayB))

	if descA != "" {
		// Truncate long descriptions to first sentence
		if idx := strings.Index(descA, ". "); idx > 0 && idx < 200 {
			descA = descA[:idx+1]
		} else if len(descA) > 200 {
			descA = descA[:200] + "..."
		}
		sb.WriteString(fmt.Sprintf("**%s:** %s\n", displayA, descA))
	}
	if descB != "" {
		if idx := strings.Index(descB, ". "); idx > 0 && idx < 200 {
			descB = descB[:idx+1]
		} else if len(descB) > 200 {
			descB = descB[:200] + "..."
		}
		sb.WriteString(fmt.Sprintf("**%s:** %s\n", displayB, descB))
	}

	if len(factsA) > 0 || len(factsB) > 0 {
		sb.WriteString("\n")
		if len(factsA) > 0 {
			sb.WriteString(fmt.Sprintf("%s: %s\n", displayA, strings.Join(factsA, " ")))
		}
		if len(factsB) > 0 {
			sb.WriteString(fmt.Sprintf("%s: %s\n", displayB, strings.Join(factsB, " ")))
		}
	}

	// Enrich with multi-hop connection reasoning if available.
	if ar.MultiHop != nil {
		conn := ar.MultiHop.FindConnection(itemA, itemB)
		if conn != nil && conn.Summary != "" {
			// Only append connection info when there is a real link found.
			hasLink := len(conn.Direct) > 0 || len(conn.TwoHop) > 0 || len(conn.SharedProps) > 0
			if hasLink {
				sb.WriteString("\n**Connection:** ")
				sb.WriteString(conn.Summary)
				sb.WriteString("\n")
			}
		}
	}

	if descA == "" && descB == "" && len(factsA) == 0 && len(factsB) == 0 {
		// No graph facts for either item — try LLM for a full comparison
		if ar.LLM != nil {
			answer := ar.LLM.Generate(
				"Compare these two items concisely. Use short bullet points for pros/cons.",
				nlu.Raw,
				160,
			)
			if answer != "" {
				return &ActionResult{DirectResponse: answer, Source: "llm"}
			}
		}
		sb.Reset()
		sb.WriteString(ar.buildSparseComparisonFallback(displayA, displayB, raw))
	}

	return &ActionResult{
		DirectResponse: sb.String(),
		Source:         "compare",
	}
}

func (ar *ActionRouter) buildSparseKnowledgeFallback(topic string) string {
	cleanTopic := strings.TrimSpace(topic)
	if cleanTopic == "" {
		cleanTopic = "that topic"
	}

	var b strings.Builder
	b.WriteString(fmt.Sprintf("I don't have detailed knowledge about %s yet.", cleanTopic))
	b.WriteString(" I can still help by structuring the topic into key dimensions and questions to investigate next.")
	b.WriteString(" Ask me to break it down, compare options, or generate a learning plan.")
	return b.String()
}

func (ar *ActionRouter) buildSparseComparisonFallback(itemA, itemB, raw string) string {
	criteria := []string{"learning curve", "runtime performance", "tooling and ecosystem", "safety and reliability", "team productivity"}
	lower := strings.ToLower(raw)
	if strings.Contains(lower, "database") || strings.Contains(lower, "sql") {
		criteria = []string{"query power", "operational complexity", "consistency model", "scalability", "cost"}
	} else if strings.Contains(lower, "framework") || strings.Contains(lower, "library") {
		criteria = []string{"developer experience", "ecosystem maturity", "performance", "community support", "long-term maintainability"}
	}

	var b strings.Builder
	b.WriteString(fmt.Sprintf("I don't have enough grounded facts to compare %s and %s in depth yet.\n\n", itemA, itemB))
	b.WriteString("A useful comparison framework is:\n")
	for _, c := range criteria {
		b.WriteString("- ")
		b.WriteString(c)
		b.WriteString("\n")
	}
	b.WriteString("\nIf you want, ask me to gather facts on each item first, then I can generate a structured tradeoff summary.")
	return b.String()
}

// disambiguateComparisonPair checks if both items in a comparison have variants
// with shared qualifiers (e.g., "(programming language)") and prefers those.
// This handles "compare rust and go" → both as programming languages.
func (ar *ActionRouter) disambiguateComparisonPair(a, b string) (string, string) {
	// Common disambiguation qualifiers, ordered by likelihood in tech contexts
	qualifiers := []string{
		"programming language", "software", "framework", "language",
		"game", "film", "band", "company",
	}

	hasItem := func(label string) bool {
		// Check cognitive graph
		if ar.CogGraph != nil && ar.CogGraph.HasLabel(label) {
			return true
		}
		// Check wiki index
		if ar.Packages != nil && ar.Packages.HasWikiEntry(label) {
			return true
		}
		return false
	}

	for _, q := range qualifiers {
		qualA := a + " (" + q + ")"
		qualB := b + " (" + q + ")"
		if hasItem(qualA) && hasItem(qualB) {
			return qualA, qualB
		}
	}

	// Individual disambiguation: if one item only exists as a qualified variant,
	// disambiguate the other to match the same domain.
	// E.g., "python vs javascript" — javascript is only known as a programming language,
	// so python should also be treated as one.
	for _, q := range qualifiers {
		qualA := a + " (" + q + ")"
		qualB := b + " (" + q + ")"
		aHasQual := hasItem(qualA)
		bHasQual := hasItem(qualB)
		if aHasQual && !bHasQual {
			// A has a qualified form; B is unambiguous — use qualified A
			return qualA, b
		}
		if bHasQual && !aHasQual {
			// B has a qualified form; A is unambiguous — use qualified B
			return a, qualB
		}
	}
	return a, b
}

// stripQualifier removes a parenthetical qualifier from a name.
// E.g., "python (programming language)" → "python".
func stripQualifier(name string) string {
	if idx := strings.Index(name, " ("); idx > 0 {
		return strings.TrimSpace(name[:idx])
	}
	return name
}

// extractQualifier returns the parenthetical qualifier from a name.
// E.g., "python (programming language)" → "programming language".
func extractQualifier(name string) string {
	start := strings.Index(name, " (")
	if start < 0 {
		return ""
	}
	end := strings.Index(name[start:], ")")
	if end < 0 {
		return ""
	}
	return name[start+2 : start+end]
}

// filterFactsByDomain keeps only facts whose content is related to the qualifier domain.
// E.g., for qualifier "programming language", keeps "Python is a programming language"
// but drops "Rust is a type of corrosion".
func filterFactsByDomain(facts []string, qualifier string) []string {
	if qualifier == "" {
		return facts
	}
	qualLower := strings.ToLower(qualifier)
	// Extract domain keywords from qualifier
	qualWords := strings.Fields(qualLower)

	var filtered []string
	for _, fact := range facts {
		lower := strings.ToLower(fact)
		// Keep facts that mention any qualifier keyword OR don't contradict the domain.
		// A fact like "Python was created by Guido" is domain-neutral — keep it.
		// A fact like "Rust is a type of corrosion" contradicts "programming language" — drop it.
		isContradict := false
		// Check if the fact's is_a/type classification contradicts the qualifier
		for _, marker := range []string{" is a ", " is ", " are "} {
			if idx := strings.Index(lower, marker); idx > 0 {
				classification := lower[idx+len(marker):]
				// If the classification doesn't share any words with the qualifier,
				// and it's a concrete classification, it likely belongs to a different entity
				hasOverlap := false
				for _, qw := range qualWords {
					if len(qw) >= 3 && strings.Contains(classification, qw) {
						hasOverlap = true
						break
					}
				}
				if !hasOverlap && len(classification) > 5 {
					isContradict = true
				}
				break
			}
		}
		if !isContradict {
			filtered = append(filtered, fact)
		}
	}
	return filtered
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

	// Personal statements ("I will run tomorrow") and follow-ups ("tell me more")
	// should NOT do graph lookups — they match common words and produce garbage.
	lower := strings.ToLower(nlu.Raw)
	personal := isPersonalStatement(lower) || isContinuationRequest(lower)

	// Try multi-hop reasoning first — chain-of-thought over the graph
	// Skip for personal statements where keywords would match random entries.
	if !personal && ar.Reasoner != nil && ar.CogGraph != nil && ar.CogGraph.NodeCount() > 0 {
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
	// Skip for personal statements to avoid looking up "I", "you", "run", etc.
	if !personal && ar.CogGraph != nil && ar.CogGraph.NodeCount() > 0 {
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

	// On-demand wiki loading: if the topic isn't in the graph yet,
	// load it before composing. This prevents deflections for topics
	// that are in the wiki index but haven't been loaded.
	if !personal && ar.Packages != nil {
		topic := nlu.Entities["topic"]
		if topic == "" {
			topic = nlu.Entities["query"]
		}
		if topic == "" {
			topic = extractTopicFromQuery(nlu.Raw)
		}
		if topic != "" {
			if loaded := ar.Packages.LookupWiki(topic); loaded > 0 {
				if ar.Composer != nil {
					ar.Composer.Graph = ar.CogGraph
				}
				// Wiki loaded — compose response from description + discourse sentences.
				desc := ar.CogGraph.LookupDescription(topic)
				if len(desc) >= 40 {
					response := desc
					// Add discourse corpus sentences for depth.
					if ar.Composer != nil && ar.Composer.DiscourseCorpus != nil {
						suppFuncs := []DiscourseFunc{DFEvaluates, DFExplainsWhy, DFGivesExample}
						suppSents := ar.Composer.DiscourseCorpus.RetrieveMulti(topic, suppFuncs)
						descLower := strings.ToLower(desc)
						var extra []string
						for _, s := range suppSents {
							if !sentenceOverlaps(s, descLower) {
								extra = append(extra, s)
							}
							if len(extra) >= 2 {
								break
							}
						}
						if len(extra) > 0 {
							response += "\n\n" + strings.Join(extra, " ")
						}
					}
					return &ActionResult{DirectResponse: response, Source: "knowledge"}
				}
			}
		}
	}

	// Composer engine — generates natural language from structured knowledge.
	// Zero-LLM path: graph facts → natural sentences. Always produces a response.
	if ar.Composer != nil {
		respType := ar.ClassifyForComposer(nlu.Raw)
		ctx := ar.BuildComposeContext()
		resp := ar.Composer.Compose(nlu.Raw, respType, ctx)
		if resp != nil && resp.Text != "" &&
			!isLowInformationConversational(resp.Text) &&
			!isDeflection(resp.Text) {
			return &ActionResult{DirectResponse: resp.Text, Source: "composer"}
		}
	}

	// LLM fallback — when the knowledge graph and composer can't answer,
	// ask the local LLM directly. This fills knowledge gaps for topics
	// not in the graph (Roman Empire, car engines, yoga, etc.).
	if ar.LLM != nil {
		answer := ar.LLM.Generate(
			"Answer concisely and factually in 2-3 sentences.",
			nlu.Raw,
			96,
		)
		if answer != "" {
			return &ActionResult{DirectResponse: answer, Source: "llm"}
		}
	}

	// Fallback — no LLM, no knowledge, no composer.
	return &ActionResult{
		DirectResponse: "I don't have detailed knowledge about that topic yet. Try asking me to look it up, or point me to a source I can learn from.",
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
var mathProseRe = regexp.MustCompile(`(?i)^(?:what(?:'s| is)\s+|calculate\s+|compute\s+|eval(?:uate)?\s+|how much is\s+)`)

// mathWordOpRe replaces English operator words with symbols (between digits).
var mathWordOpRe = regexp.MustCompile(`(?i)(\d)\s+(times|multiplied by)\s+(\d)`)
var mathWordAddRe = regexp.MustCompile(`(?i)(\d)\s+plus\s+(\d)`)
var mathWordSubRe = regexp.MustCompile(`(?i)(\d)\s+minus\s+(\d)`)
var mathWordDivRe = regexp.MustCompile(`(?i)(\d)\s+divided by\s+(\d)`)

// mathDigitXRe replaces "x" with "*" only when between digits (e.g. "5x3" → "5*3").
var mathDigitXRe = regexp.MustCompile(`(\d)\s*x\s*(\d)`)

func stripMathProse(s string) string {
	s = mathProseRe.ReplaceAllString(s, "")
	s = strings.TrimRight(s, "? ")
	// "sqrt of 144" → "sqrt(144)", "log of 100" → "log(100)"
	s = mathFuncOfRe.ReplaceAllString(s, "${1}(${2})")
	// Replace English operator words: "times", "plus", "minus", "divided by"
	s = mathWordOpRe.ReplaceAllString(s, "${1}*${3}")
	s = mathWordAddRe.ReplaceAllString(s, "${1}+${2}")
	s = mathWordSubRe.ReplaceAllString(s, "${1}-${2}")
	s = mathWordDivRe.ReplaceAllString(s, "${1}/${2}")
	// Replace unicode operators.
	s = strings.ReplaceAll(s, "\u00d7", "*") // ×
	s = strings.ReplaceAll(s, "\u00f7", "/") // ÷
	// Replace "x" with "*" only between digits (not in words like "max", "six")
	s = mathDigitXRe.ReplaceAllString(s, "${1}*${2}")
	return s
}

var mathFuncOfRe = regexp.MustCompile(`(?i)^(sqrt|abs|sin|cos|tan|log|ln|ceil|floor|round)\s+(?:of\s+)?(\d[\d.]*)$`)

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

// handleSimulate runs a "what if" scenario simulation.
func (ar *ActionRouter) handleSimulate(nlu *NLUResult) *ActionResult {
	if ar.Simulation == nil {
		// Try to build one on the fly from available engines.
		if ar.CogGraph != nil && ar.CausalReasoner != nil {
			ar.Simulation = NewSimulationEngine(ar.CogGraph, ar.CausalReasoner, ar.Council, ar.MultiHop)
			if ar.SelfTeacher != nil {
				ar.Simulation.KnowledgeDir = ar.SelfTeacher.KnowledgeDir()
			}
		} else {
			return &ActionResult{
				DirectResponse: "Simulation engine not available. Requires knowledge graph and causal reasoner.",
				Source:         "simulate",
			}
		}
	}

	scenario := nlu.Raw
	steps := 5 // default

	var result *SimulationResult
	if IsRemovalQuery(scenario) {
		entity := extractHypothesis(scenario)
		result = ar.Simulation.SimulateRemoval(entity)
	} else {
		result = ar.Simulation.Simulate(scenario, steps)
	}

	if result == nil {
		return &ActionResult{
			DirectResponse: "Could not simulate this scenario. The topic may not be in my knowledge base.",
			Source:         "simulate",
		}
	}

	return &ActionResult{
		DirectResponse: result.Report,
		Source:         "simulate",
	}
}

// handlePersona routes to an expert persona for a domain-constrained answer.
func (ar *ActionRouter) handlePersona(nlu *NLUResult) *ActionResult {
	if ar.Personas == nil {
		if ar.CogGraph != nil {
			ar.Personas = NewPersonaEngine(ar.CogGraph)
			if ar.SelfTeacher != nil {
				ar.Personas.KnowledgeDir = ar.SelfTeacher.KnowledgeDir()
			}
			ar.Personas.WikiLoader = ar.WikiLoader
		} else {
			return &ActionResult{
				DirectResponse: "Expert persona engine not available.",
				Source:         "persona",
			}
		}
	}

	personaName := nlu.Entities["persona"]
	if personaName == "" {
		personaName = "physicist" // default
	}

	answer := ar.Personas.Answer(nlu.Raw, personaName)
	if answer == nil {
		return &ActionResult{
			DirectResponse: "Could not generate a persona response.",
			Source:         "persona",
		}
	}

	response := answer.Response
	if answer.Disclaimer != "" {
		response += "\n\n" + answer.Disclaimer
	}

	return &ActionResult{
		DirectResponse: response,
		Source:         fmt.Sprintf("persona:%s", answer.Persona),
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

	// Essay/article requests: use Thinker for structured long-form content
	if creativeType == "essay" {
		if topic == "" {
			topic = nlu.Raw
		}
		// First try knowledge lookup to gather material
		if ar.Packages != nil {
			ar.Packages.LookupWiki(topic)
		}
		if ar.Thinker != nil {
			result := ar.Thinker.Think(nlu.Raw, nil)
			if result != nil && result.Text != "" {
				return &ActionResult{
					DirectResponse: result.Text,
					Source:         "thinking:" + result.Frame,
				}
			}
		}
		// Fallback: use knowledge graph for a short essay
		if ar.CogGraph != nil {
			if desc := ar.CogGraph.LookupDescription(topic); desc != "" {
				extras := ar.CogGraph.LookupFacts(topic, 10)
				response := desc
				if len(extras) > 0 {
					response += "\n\n" + strings.Join(extras, " ")
				}
				return &ActionResult{
					DirectResponse: response,
					Source:         "knowledge",
				}
			}
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
			DirectResponse: "I can't generate creative writing right now, but I can help you think through ideas. What are you working on?",
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

	// Map specific creative types to their base type for the engine
	cType := CreativeType(creativeType)
	switch creativeType {
	case "haiku", "limerick", "quatrain", "verse":
		cType = CreativePoem
	}

	req := CreativeRequest{
		Type:     cType,
		Topic:    topic,
		PoemForm: form,
	}

	result := ar.Creative.Generate(req)
	return &ActionResult{
		DirectResponse: result,
		Source:         "creative",
	}
}

// handleCodeGen generates code from a natural language request using the
// CodeGenerator's template system.
func (ar *ActionRouter) handleCodeGen(nlu *NLUResult) *ActionResult {
	if ar.CodeGen == nil {
		ar.CodeGen = NewCodeGenerator()
	}

	result := ar.CodeGen.GenerateFromQuery(nlu.Raw)
	if result == nil {
		return &ActionResult{
			DirectResponse: "I can generate code in Python, JavaScript, and Go. Try: \"write a python function to read a CSV file\"",
			Source:         "codegen",
		}
	}

	var response strings.Builder
	response.WriteString("```" + result.Language + "\n")
	response.WriteString(result.Code)
	response.WriteString("```\n\n")
	response.WriteString(result.Explanation)

	return &ActionResult{
		DirectResponse: response.String(),
		Source:         "codegen",
	}
}

// resolveConversationalReferences replaces pronouns and vague references
// in the NLU result with the current conversation topic. This bridges
// single-turn NLU to multi-turn conversation without any LLM.
//
// "what is a black hole?" → topic = "black hole" (stored)
// "why are they dangerous?" → "they" = "black hole" → topic = "black hole"
// "can anything escape from one?" → "one" = "black hole" → rewrites query
func (ar *ActionRouter) resolveConversationalReferences(nlu *NLUResult) {
	if ar.Tracker == nil {
		return
	}
	currentTopic := ar.Tracker.CurrentTopic()
	if currentTopic == "" {
		return
	}

	lower := strings.ToLower(nlu.Raw)

	// Check if the extracted topic is a pronoun or generic word that
	// needs resolution to the current conversation topic.
	topic := nlu.Entities["topic"]
	if topic == "" {
		topic = nlu.Entities["query"]
	}
	topicLower := strings.ToLower(topic)

	// Pronouns and generic references that should resolve to the current topic.
	pronouns := map[string]bool{
		"it": true, "they": true, "them": true, "its": true, "their": true,
		"this": true, "that": true, "these": true, "those": true,
		"one": true, "ones": true, "he": true, "she": true,
	}

	// Generic words that are almost certainly not the intended topic.
	generic := map[string]bool{
		"dangerous": true, "important": true, "useful": true, "interesting": true,
		"good": true, "bad": true, "big": true, "small": true,
		"black": true, "white": true, "blue": true, "red": true,
		"hot": true, "cold": true, "fast": true, "slow": true,
		"work": true, "possible": true, "real": true,
	}

	needsResolution := false

	// Case 1: extracted topic is a pronoun.
	if pronouns[topicLower] {
		needsResolution = true
	}

	// Case 2: extracted topic is a single generic word.
	if !needsResolution && generic[topicLower] {
		needsResolution = true
	}

	// Case 3: the topic or query CONTAINS a pronoun, and the topic
	// doesn't map to a real entity. This catches:
	// - "they dangerous" (topic contains pronoun)
	// - "can anything escape from one" (query contains "one")
	if !needsResolution {
		for pron := range pronouns {
			// Check in the extracted topic.
			if strings.Contains(" "+topicLower+" ", " "+pron+" ") {
				needsResolution = true
				break
			}
			// Check in the raw query.
			if strings.Contains(" "+lower+" ", " "+pron+" ") {
				// Only resolve if the topic isn't a known entity.
				if topic == "" || (ar.Packages != nil && !ar.Packages.HasWikiEntry(topicLower)) {
					needsResolution = true
					break
				}
			}
		}
	}

	// Case 4: no topic extracted at all, but the query is a follow-up.
	if !needsResolution && (topic == "" || nlu.Intent == "follow_up") {
		followUpPatterns := []string{
			"tell me more", "what else", "go on", "continue",
			"can you explain", "what about", "and what", "how about",
		}
		for _, pat := range followUpPatterns {
			if strings.Contains(lower, pat) {
				needsResolution = true
				break
			}
		}
		// Queries starting with "why", "how", "can", "does", "is", "are"
		// without a real topic are likely follow-ups.
		if !needsResolution && topic == "" {
			for _, prefix := range []string{"why ", "how ", "can ", "does ", "do ", "is ", "are ", "would ", "could "} {
				if strings.HasPrefix(lower, prefix) {
					needsResolution = true
					break
				}
			}
		}
	}

	if needsResolution {
		if nlu.Entities == nil {
			nlu.Entities = make(map[string]string)
		}
		nlu.Entities["topic"] = currentTopic
		nlu.Entities["query"] = currentTopic
		nlu.Entities["_original_query"] = nlu.Raw
		nlu.Entities["_resolved_from"] = topicLower

		// Detect what kind of follow-up this is from the original query.
		// This tells the discourse corpus which TYPE of sentence to retrieve
		// instead of just repeating the definition.
		nlu.Entities["_follow_up_type"] = classifyFollowUpType(lower)

		// Rewrite raw query, replacing pronouns with the actual topic.
		rewritten := lower
		for pron := range pronouns {
			rewritten = strings.ReplaceAll(rewritten, " "+pron+" ", " "+currentTopic+" ")
			if strings.HasPrefix(rewritten, pron+" ") {
				rewritten = currentTopic + rewritten[len(pron):]
			}
		}
		if rewritten != lower {
			nlu.Raw = rewritten
		}
	}
}

// classifyFollowUpType determines what kind of information the follow-up
// question is seeking, so the discourse corpus retrieves the right type
// of sentence instead of repeating the definition.
// Returns one of: "why", "how", "example", "compare", "deeper", "more".
func classifyFollowUpType(query string) string {
	lower := strings.TrimRight(strings.TrimSpace(strings.ToLower(query)), "?!.")

	// Exact short-form matches first (e.g. bare "why?", "how?")
	switch lower {
	case "why", "how come", "how so":
		return "why"
	case "how":
		return "how"
	case "example", "examples", "like what":
		return "example"
	case "explain", "elaborate", "deeper", "dig deeper":
		return "deeper"
	}

	if strings.Contains(lower, "why") || strings.Contains(lower, "because") ||
		strings.Contains(lower, "cause") || strings.Contains(lower, "reason") ||
		strings.Contains(lower, "how come") || strings.Contains(lower, "how so") {
		return "why"
	}
	if strings.Contains(lower, "explain") || strings.Contains(lower, "elaborate") ||
		strings.Contains(lower, "in depth") || strings.Contains(lower, "deeper") ||
		strings.Contains(lower, "in detail") || strings.Contains(lower, "more detail") {
		return "deeper"
	}
	if strings.Contains(lower, "how") || strings.Contains(lower, "work") ||
		strings.Contains(lower, "process") {
		return "how"
	}
	if strings.Contains(lower, "example") || strings.Contains(lower, "such as") ||
		strings.Contains(lower, "instance") || strings.Contains(lower, "like what") {
		return "example"
	}
	if strings.Contains(lower, "compare") || strings.Contains(lower, "differ") ||
		strings.Contains(lower, "versus") || strings.Contains(lower, " vs ") ||
		strings.Contains(lower, "similar") {
		return "compare"
	}
	if strings.Contains(lower, "big") || strings.Contains(lower, "many") ||
		strings.Contains(lower, "much") || strings.Contains(lower, "size") ||
		strings.Contains(lower, "number") || strings.Contains(lower, "long") {
		return "quantify"
	}
	if strings.Contains(lower, "danger") || strings.Contains(lower, "risk") ||
		strings.Contains(lower, "effect") || strings.Contains(lower, "impact") {
		return "consequence"
	}
	return "more"
}

// sentenceOverlaps checks if a sentence largely overlaps with existing text.
// Used to avoid repeating content from the lead paragraph in supplementary sentences.
func sentenceOverlaps(sentence, existingLower string) bool {
	words := strings.Fields(strings.ToLower(sentence))
	if len(words) < 4 {
		return false
	}
	// Check if any 4-word phrase from the sentence appears in existing text.
	for i := 0; i+3 < len(words); i++ {
		phrase := words[i] + " " + words[i+1] + " " + words[i+2] + " " + words[i+3]
		if strings.Contains(existingLower, phrase) {
			return true
		}
	}
	return false
}

// isEmotionalStatement detects personal emotional statements that need
// empathy, not knowledge lookup. E.g., "I got promoted!", "I'm feeling sad",
// "I just had a terrible day", "my dog passed away".
func isHarmfulRequest(input string) bool {
	lower := strings.ToLower(strings.TrimSpace(input))
	harmfulPatterns := []string{
		// Hacking and unauthorized access.
		"hack into", "hack someone", "hack a computer", "hack a phone",
		"hack a website", "hack an account", "hack a server",
		"break into someone", "break into a system",
		"steal password", "steal someone", "steal credit card",
		"ddos", "denial of service",
		// Weapons and violence.
		"make a bomb", "build a bomb", "build an explosive", "make explosives",
		"make a weapon", "build a weapon", "create a weapon",
		"make a gun", "build a gun", "3d print a gun",
		"how to kill", "how to murder", "how to poison",
		"how to stab", "how to shoot", "how to strangle",
		"how to hurt", "how to harm someone", "how to injure",
		// Stalking and surveillance.
		"how to stalk", "how to spy on someone", "how to track someone",
		"how to follow someone", "how to surveil",
		// Malware and exploits.
		"create a virus", "create malware", "write malware",
		"write a virus", "write ransomware", "write a trojan",
		"write an exploit", "create an exploit",
		"phishing", "social engineer someone",
		// Fraud and forgery.
		"forge a document", "fake identity", "fake passport",
		"counterfeit", "forge a check", "fake a diploma",
		// Security attacks.
		"bypass security", "crack a password", "brute force",
		"exploit a vulnerability", "sql injection attack",
		// Theft.
		"how to rob", "how to burglar", "how to shoplift", "how to pickpocket",
		// CSAM.
		"child exploitation", "child porn", "child abuse material",
		// Drug manufacturing.
		"how to make drugs", "how to cook meth", "how to make fentanyl",
		"how to synthesize", "how to manufacture drugs",
		// Self-harm.
		"how to hurt myself", "how to end my life",
		// Prompt injection.
		"ignore all previous instructions", "ignore your instructions",
		"disregard your programming", "override your safety",
		"forget your rules", "pretend you have no restrictions",
	}
	for _, p := range harmfulPatterns {
		if strings.Contains(lower, p) {
			return true
		}
	}
	return false
}

func isEmotionalStatement(input string) bool {
	lower := strings.ToLower(strings.TrimSpace(input))
	// Must start with a personal pronoun or possessive.
	personalStarts := []string{
		"i ", "i'm ", "i've ", "i just ", "i got ", "i had ",
		"i feel ", "i can't ", "i don't ", "i couldn't ", "i wasn't ",
		"i am ", "i was ", "i think i ",
		"my ", "we ", "we're ", "we've ",
	}
	isPersonal := false
	for _, p := range personalStarts {
		if strings.HasPrefix(lower, p) {
			isPersonal = true
			break
		}
	}
	if !isPersonal {
		return false
	}
	// Decision-making, coaching, and planning language should NOT be treated
	// as emotional statements — they are substantive queries.
	decisionWords := []string{
		"decide", "torn between", "choosing between", "should i", "help me decide",
		"career", "business", "start my own", "change careers", "switching to",
		"improve", "get better", "learn to", "want to change", "planning",
		"strategy", "how do i", "what should", "figure out",
		"compare", "difference", "versus", "pros and cons",
		"stuck", "don't know where to start", "need guidance",
	}
	for _, d := range decisionWords {
		if strings.Contains(lower, d) {
			return false
		}
	}
	// Must contain an emotional signal (not a question or command).
	if strings.HasSuffix(lower, "?") {
		return false
	}
	emotionalWords := []string{
		"promoted", "fired", "hired", "quit", "resigned",
		"married", "engaged", "divorced", "pregnant", "baby",
		"died", "passed away", "lost", "sick", "ill", "diagnosed",
		"stressed", "anxious", "depressed", "lonely", "sad", "happy",
		"excited", "proud", "grateful", "thankful", "frustrated",
		"angry", "upset", "worried", "scared", "afraid",
		"love", "hate", "miss", "broke up", "break up",
		"graduated", "accepted", "rejected", "failed", "passed",
		"won", "lost", "finished", "completed", "achieved",
		"great day", "bad day", "terrible day", "amazing day", "best day", "worst day",
		"stuck", "struggling", "overwhelmed", "burned out", "burnt out",
		"give up", "giving up", "can't figure", "cannot figure",
		"can't take", "cannot take", "so tired", "exhausted",
	}
	for _, w := range emotionalWords {
		if strings.Contains(lower, w) {
			return true
		}
	}
	// Exclamation mark with personal pronoun = emotional.
	if strings.HasSuffix(strings.TrimSpace(input), "!") {
		return true
	}
	return false
}

// extractFactObjects extracts the object portion of natural language facts.
// E.g., "Einstein is a theoretical physicist." → ["theoretical physicist"]
func extractFactObjects(fact string) []string {
	var objects []string
	// Common fact patterns: "X is a Y", "X has Y", "X is located in Y"
	patterns := []string{" is a ", " is an ", " is ", " has ", " are ", " was ", " were ",
		" is located in ", " is part of ", " was created by ", " was founded by "}
	for _, p := range patterns {
		if idx := strings.Index(fact, p); idx >= 0 {
			obj := strings.TrimRight(fact[idx+len(p):], ".!? ")
			if obj != "" {
				objects = append(objects, obj)
			}
		}
	}
	return objects
}

// isSummarizeRequest detects whether the user input is a summarization request
// (e.g., "summarize this", "tl;dr", "give me a summary" followed by long text).
func isSummarizeRequest(raw string) bool {
	lower := strings.ToLower(strings.TrimSpace(raw))
	prefixes := []string{
		"summarize this:", "summarise this:", "summarize this", "summarise this",
		"summarize:", "summarise:",
		"tl;dr:", "tl;dr", "tldr:", "tldr", "tl dr:", "tl dr",
		"give me a summary", "can you summarize", "can you summarise",
		"please summarize", "please summarise",
		"summary:", "summarize the following:", "summarise the following:",
		"summarize the following", "summarise the following",
	}
	for _, p := range prefixes {
		if strings.HasPrefix(lower, p) {
			return true
		}
	}
	// Also check for trailing summarize commands after pasted text.
	suffixes := []string{
		"summarize this", "summarise this", "tl;dr", "tldr",
		"summarize", "summarise", "summary please",
	}
	for _, s := range suffixes {
		if strings.HasSuffix(lower, s) {
			return true
		}
	}
	return false
}

// extractTextForSummarization strips the summarization command from the input,
// leaving only the text to be summarized.
func extractTextForSummarization(raw string) string {
	lower := strings.ToLower(strings.TrimSpace(raw))
	text := strings.TrimSpace(raw)

	// Strip leading command.
	leadPrefixes := []string{
		"summarize this:", "summarise this:", "summarize this",
		"summarise this", "summarize:", "summarise:",
		"tl;dr:", "tl;dr", "tldr:", "tldr", "tl dr:",
		"give me a summary of:", "give me a summary of",
		"give me a summary:", "give me a summary",
		"can you summarize this:", "can you summarize this",
		"can you summarise this:", "can you summarise this",
		"please summarize:", "please summarise:",
		"please summarize", "please summarise",
		"summary:", "summarize the following:", "summarize the following",
		"summarise the following:", "summarise the following",
	}
	for _, p := range leadPrefixes {
		if strings.HasPrefix(lower, p) {
			text = strings.TrimSpace(text[len(p):])
			return text
		}
	}

	// Strip trailing command.
	trailSuffixes := []string{
		"summarize this", "summarise this", "tl;dr", "tldr",
		"summarize", "summarise", "summary please",
	}
	for _, s := range trailSuffixes {
		if strings.HasSuffix(lower, s) {
			text = strings.TrimSpace(text[:len(text)-len(s)])
			return text
		}
	}

	return text
}

// isDeflection detects when the response is a deflection rather than an answer.
// E.g., "What specifically about X would you like to explore?" is a deflection.
func isDeflection(text string) bool {
	lower := strings.ToLower(text)
	deflections := []string{
		"what specifically about",
		"would you like to explore",
		"what angle are you",
		"what aspect of",
		"could you be more specific",
		"what would you like to know about",
		"i'd need more context",
		"that's a broad topic",
		"here's my take. what specifically",
		"let me address that directly. what specifically",
	}
	for _, d := range deflections {
		if strings.Contains(lower, d) {
			return true
		}
	}
	return false
}

// isSparkRelevant checks whether an associative spark explanation is relevant
// to the current conversation topics. Prevents cross-domain noise like
// "Photosynthesis connects to South America" when discussing biology.
func isSparkRelevant(explanation string, topics []string) bool {
	if len(topics) == 0 {
		return false
	}
	lower := strings.ToLower(explanation)
	for _, topic := range topics {
		if len(topic) >= 3 && strings.Contains(lower, topic) {
			return true
		}
	}
	return false
}
