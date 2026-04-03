package cognitive

import (
	"fmt"
	"math/rand"
	"regexp"
	"strings"
	"time"
	"unicode"
)

// -----------------------------------------------------------------------
// Cognitive Response Composer — generates natural language from structured
// knowledge WITHOUT any LLM. Every response is unique through:
//
//   - Sentence Transforms: active/passive, topicalization, cleft, appositive
//   - Anti-Repetition Memory: tracks recent phrases, never repeats in session
//   - Fact Reordering: shuffles emphasis each time
//   - Tone Modulation: casual/warm/direct randomly per response
//   - Rhythm Engine: mixes short punchy + long flowing sentences
//   - Smart References: "the company", "the language" not just "it"
//   - Parenthetical Asides: natural insertions in flowing prose
//   - Discourse Planning: connectors, paragraph, mixed, contrastive
// -----------------------------------------------------------------------

// ResponseType classifies what kind of response to generate.
type ResponseType int

const (
	RespFactual        ResponseType = iota // answering a factual question
	RespPersonal                           // personal insight/reflection
	RespBriefing                           // daily briefing / status
	RespAcknowledge                        // acknowledging an action (expense logged, etc)
	RespExplain                            // explaining a concept
	RespReflect                            // reflecting on patterns/trends
	RespGreeting                           // time-aware greeting
	RespUncertain                          // when we don't know enough
	RespConversational                     // open-ended dialogue
	RespEmpathetic                         // emotional support response
	RespOpinion                            // opinion/thought on a topic
	RespFarewell                           // goodbye
	RespThankYou                           // responding to thanks
)

// Tone sets the emotional register for a response.
type Tone int

const (
	ToneNeutral Tone = iota
	ToneCasual
	ToneWarm
	ToneDirect
)

// ComposedResponse is the output of the composer with traceability.
type ComposedResponse struct {
	Text    string   // the natural language response
	Sources []string // which data sources contributed
	Type    ResponseType
}

// Sentiment represents detected emotional tone of user input.
type Sentiment int

const (
	SentimentNeutral  Sentiment = iota
	SentimentPositive
	SentimentNegative
	SentimentExcited
	SentimentSad
	SentimentAngry
	SentimentCurious
)

// ConvTurn records one turn of conversation for context.
type ConvTurn struct {
	Input    string
	Response string
	Topics   []string
}

// Composer generates natural language responses from structured knowledge.
type Composer struct {
	Graph    *CognitiveGraph
	Semantic *SemanticEngine
	Causal   *CausalEngine
	Patterns *PatternDetector
	rng      *rand.Rand
	recent   map[string]int // anti-repetition: phrase → recency counter
	turnID   int            // increments each Compose call
	history  []ConvTurn     // last N conversation turns for context

	// Self-Improving Phrase Pools — tracks which phrases resonate with the user.
	// Every phrase gets a score: positive engagement raises it, negative lowers it.
	// Higher-scored phrases are selected more often, evolving the engine's voice.
	phraseScores   map[string]float64 // phrase → engagement score (default 1.0)
	lastPhrases    []string           // phrases used in most recent response
	emotionalMem   map[string]float64 // topic → cumulative sentiment (-1.0 to +1.0)
	sessionPhrases map[string]int     // phrases used this session for diversity tracking

	// Generative Sentence Planner — grammar-rule-based sentence builder.
	// Used as an alternate realization strategy alongside templates.
	Generative *GenerativeEngine

	// Discourse Planner — RST-based rhetorical structure planning.
	// Plans section order and transitions before text generation.
	Discourse *DiscoursePlanner

	// TextGen — GRU-based neural text generation model.
	// When loaded, used as an alternative to template-based sentence generation.
	TextGen *TextGenModel

	// SentenceCorpus — Layer 2: retrieval-based sentence generation.
	// Retrieves real human-written sentences from Wikipedia and adapts them
	// by swapping entities. The corpus IS the model.
	SentenceCorpus *SentenceCorpus

	// DiscourseCorpus — Layer 2b: sentences indexed by discourse function.
	// Retrieves sentences by HOW they communicate (defines, explains_why,
	// compares, evaluates) for assembling multi-sentence responses.
	DiscourseCorpus *DiscourseCorpus

	// Absorption — learns expression patterns from text.
	// When set, provides an additional realization strategy.
	Absorption *AbsorptionEngine
}

// NewComposer creates a response composer wired to the cognitive systems.
func NewComposer(graph *CognitiveGraph, semantic *SemanticEngine, causal *CausalEngine, patterns *PatternDetector) *Composer {
	gen := NewGenerativeEngine()

	// Initialize NLG subsystems — embeddings, Markov, templates.
	// These are optional layers that enhance generation quality.
	embeddings := NewWordEmbeddings(50)
	embeddings.SeedFromTaxonomy(conceptCategory)
	embeddings.SeedPoolWords()
	gen.SetEmbeddings(embeddings)

	markov := NewMarkovModel()
	gen.SetMarkov(markov)

	templates := NewTemplateInducer()
	gen.SetTemplates(templates)

	// Build embeddings from semantic co-occurrence if available.
	if semantic != nil {
		semantic.mu.RLock()
		if len(semantic.cooccurrence) > 0 {
			embeddings.BuildFromCooccurrence(semantic.cooccurrence)
		}
		semantic.mu.RUnlock()
	}

	rng := rand.New(rand.NewSource(time.Now().UnixNano()))

	return &Composer{
		Graph:          graph,
		Semantic:       semantic,
		Causal:         causal,
		Patterns:       patterns,
		Generative:     gen,
		Discourse:      NewDiscoursePlanner(rng),
		rng:            rng,
		recent:         make(map[string]int),
		phraseScores:   make(map[string]float64),
		lastPhrases:    nil,
		emotionalMem:   make(map[string]float64),
		sessionPhrases: make(map[string]int),
	}
}

// LoadTextGen loads a trained GRU text generation model from disk.
// If the file doesn't exist, the composer continues using templates only.
func (c *Composer) LoadTextGen(path string) error {
	model := NewTextGenModel(DefaultTextGenConfig())
	if err := model.Load(path); err != nil {
		return err
	}
	c.TextGen = model
	return nil
}

// RecordTurn saves a conversation turn for contextual follow-ups.
// Also auto-detects engagement from the new input and updates phrase scores
// from the previous response — this is how the engine learns and evolves.
func (c *Composer) RecordTurn(input, response string) {
	// Self-improvement: the user's NEW input tells us how they felt about
	// our LAST response. Detect engagement and update phrase scores.
	if len(c.lastPhrases) > 0 {
		engagement := c.DetectEngagement(input)
		c.ObserveEngagement(engagement)
	}

	// Update emotional memory: track sentiment per topic over time
	topics := extractKeywords(strings.ToLower(input))
	sentiment := c.detectSentiment(input)
	sentVal := sentimentToFloat(sentiment)
	for _, topic := range topics {
		old := c.emotionalMem[topic]
		// Exponential moving average: slowly shift toward current sentiment
		c.emotionalMem[topic] = old*0.7 + sentVal*0.3
	}

	c.history = append(c.history, ConvTurn{
		Input:    input,
		Response: response,
		Topics:   topics,
	})
	// Keep last 10 turns
	if len(c.history) > 10 {
		c.history = c.history[len(c.history)-10:]
	}

	// Feed generated text to Markov and template induction for self-improvement.
	if response != "" && c.Generative != nil {
		if c.Generative.markov != nil {
			c.Generative.markov.Train(response)
		}
		if c.Generative.templates != nil {
			c.Generative.templates.InduceFromText(response, "self")
		}
	}
}

// IngestContent feeds text to all NLG subsystems for learning.
// Call this whenever content is ingested (fetched pages, corpus text, etc.).
func (c *Composer) IngestContent(text string) {
	if c.Generative == nil || text == "" {
		return
	}
	if c.Generative.markov != nil {
		c.Generative.markov.Train(text)
	}
	if c.Generative.templates != nil {
		c.Generative.templates.InduceFromText(text, "ingested")
	}
	// Rebuild embeddings from updated co-occurrence data.
	if c.Semantic != nil && c.Generative.embeddings != nil {
		c.Semantic.mu.RLock()
		if len(c.Semantic.cooccurrence) > 50 {
			c.Generative.embeddings.BuildFromCooccurrence(c.Semantic.cooccurrence)
		}
		c.Semantic.mu.RUnlock()
	}

	// Absorb expression patterns — learn HOW things are said, not just facts.
	if c.Absorption != nil {
		c.Absorption.Absorb(text)
	}
}

// sentimentToFloat converts a Sentiment enum to a float for emotional memory.
func sentimentToFloat(s Sentiment) float64 {
	switch s {
	case SentimentPositive:
		return 0.6
	case SentimentExcited:
		return 1.0
	case SentimentCurious:
		return 0.3
	case SentimentNeutral:
		return 0.0
	case SentimentNegative:
		return -0.5
	case SentimentSad:
		return -0.7
	case SentimentAngry:
		return -1.0
	}
	return 0.0
}

// EmotionalTone returns the remembered emotional tone for a topic.
// Positive = user tends to be happy about this topic.
// Negative = user tends to be stressed/sad about this topic.
// Zero = neutral or unknown.
func (c *Composer) EmotionalTone(topic string) float64 {
	return c.emotionalMem[strings.ToLower(topic)]
}

// Compose generates a response to a query using only structured knowledge.
// NEVER returns nil — always produces a meaningful response.
// The engine evolves over time: phrases that get positive engagement from
// the user are selected more often (self-improving phrase pools).
func (c *Composer) Compose(query string, respType ResponseType, context *ComposeContext) *ComposedResponse {
	c.turnID++
	c.lastPhrases = nil // reset phrase tracking for this response

	// Decay old entries from anti-repetition memory
	for k, turn := range c.recent {
		if c.turnID-turn > 8 {
			delete(c.recent, k)
		}
	}

	var resp *ComposedResponse
	switch respType {
	case RespGreeting:
		resp = c.composeGreeting(context)
	case RespAcknowledge:
		resp = c.composeAcknowledge(query, context)
	case RespFactual:
		resp = c.composeFactual(query)
	case RespPersonal:
		resp = c.composePersonal(query, context)
	case RespReflect:
		resp = c.composeReflection(query, context)
	case RespBriefing:
		resp = c.composeBriefing(context)
	case RespUncertain:
		resp = c.composeUncertain(query)
	case RespConversational:
		resp = c.composeConversational(query, context)
	case RespEmpathetic:
		resp = c.composeEmpathetic(query, context)
	case RespOpinion:
		resp = c.composeOpinion(query, context)
	case RespFarewell:
		resp = c.composeFarewell(context)
	case RespThankYou:
		resp = c.composeThankYou(context)
	case RespExplain:
		resp = c.composeExplainWithContext(query, context)
	default:
		resp = c.composeConversational(query, context)
	}

	// NEVER return nil — if the specific composer returned nothing,
	// fall through: try conversational, then uncertain, then catch-all.
	if resp == nil || resp.Text == "" {
		resp = c.composeConversational(query, context)
	}
	if resp == nil || resp.Text == "" {
		resp = c.composeUncertain(query)
	}
	return resp
}

// ComposeContext provides situational context for response generation.
type ComposeContext struct {
	UserName       string
	TimeOfDay      time.Time
	RecentMood     float64 // 1-5, 0 = unknown
	RecentTopics   []string
	HabitStreak    int
	WeeklySpend    float64
	AvgWeeklySpend float64
	JournalDays    int // days since last journal entry
	ConvTurns      int // how many turns in current conversation

	// Cognitive context — populated by the new cognitive engines.
	Subtext          *SubtextAnalysis  // what the user really means (Phase 1)
	TriggeredMemories []MemoryTrigger  // involuntary episodic recall (Phase 2)
	Sparks           []AssociativeSpark // unexpected knowledge connections (Phase 4)
	CouncilResult    *CouncilDeliberation // inner council deliberation (Phase 5)
	Opinion          *Opinion            // formed opinion on topic (Phase 6)
}

// edgeFact holds a typed fact extracted from the graph for composition.
type edgeFact struct {
	Subject  string
	Relation RelType
	Object   string
	Inferred bool
}

// -----------------------------------------------------------------------
// Self-Improving Weighted Selection
// -----------------------------------------------------------------------

// phraseScore returns the engagement score for a phrase (default 1.0).
func (c *Composer) phraseScore(phrase string) float64 {
	if score, ok := c.phraseScores[phrase]; ok {
		return score
	}
	return 1.0
}

// pick selects a phrase using weighted random selection based on engagement
// scores, while avoiding recently used phrases. Higher-scored phrases
// (ones that historically got positive engagement) are chosen more often.
func (c *Composer) pick(options []string) string {
	if len(options) == 0 {
		return ""
	}
	if len(options) == 1 {
		chosen := options[0]
		c.recent[chosen] = c.turnID
		c.lastPhrases = append(c.lastPhrases, chosen)
		c.sessionPhrases[chosen]++
		return chosen
	}

	// Build candidate list excluding recently used phrases
	var candidates []string
	for _, opt := range options {
		if _, used := c.recent[opt]; !used {
			candidates = append(candidates, opt)
		}
	}
	if len(candidates) == 0 {
		candidates = options
	}

	// Weighted random selection: phrases with higher engagement scores
	// get proportionally more chance of being selected.
	var totalWeight float64
	weights := make([]float64, len(candidates))
	for i, cand := range candidates {
		w := c.phraseScore(cand)
		if w < 0.1 {
			w = 0.1 // floor: never completely eliminate a phrase
		}
		// Boost phrases not yet used this session for diversity
		if c.sessionPhrases[cand] == 0 {
			w *= 1.3
		}
		weights[i] = w
		totalWeight += w
	}

	// Weighted random pick
	r := c.rng.Float64() * totalWeight
	var cumulative float64
	chosen := candidates[len(candidates)-1] // fallback
	for i, w := range weights {
		cumulative += w
		if r <= cumulative {
			chosen = candidates[i]
			break
		}
	}

	c.recent[chosen] = c.turnID
	c.lastPhrases = append(c.lastPhrases, chosen)
	c.sessionPhrases[chosen]++
	return chosen
}

// ObserveEngagement updates phrase scores based on user engagement signals.
// Call this after each user response with the detected engagement level.
//
// Engagement signals:
//   +1.0 = user said thanks, expressed appreciation, continued on topic
//   +0.5 = user continued conversation normally
//   -0.5 = user rephrased/asked again (our response wasn't helpful)
//   -1.0 = user expressed frustration or immediately changed topic
//
// This is the heart of the self-improving engine: phrases that resonate
// with the user naturally rise to the top over time.
func (c *Composer) ObserveEngagement(score float64) {
	for _, phrase := range c.lastPhrases {
		current := c.phraseScore(phrase)
		// Exponential moving average: 80% old score + 20% new signal
		// This makes the system learn gradually, not overreact to single events
		newScore := current*0.8 + (1.0+score)*0.2
		// Clamp to [0.1, 3.0] — never eliminate, never dominate too much
		if newScore < 0.1 {
			newScore = 0.1
		}
		if newScore > 3.0 {
			newScore = 3.0
		}
		c.phraseScores[phrase] = newScore
	}
	c.lastPhrases = nil // reset for next response
}

// DetectEngagement automatically determines engagement from the user's
// follow-up message. No manual scoring needed — the engine reads signals.
func (c *Composer) DetectEngagement(followUp string) float64 {
	lower := strings.ToLower(followUp)
	sentiment := c.detectSentiment(followUp)

	// Strong positive signals
	if sentiment == SentimentPositive || sentiment == SentimentExcited {
		return 1.0
	}
	for _, w := range []string{"thanks", "thank you", "perfect", "exactly",
		"great", "awesome", "nice", "love it", "spot on", "that's right"} {
		if strings.Contains(lower, w) {
			return 1.0
		}
	}

	// Mild positive: user continues on the same topic
	if len(c.history) > 0 {
		lastTopics := c.history[len(c.history)-1].Topics
		newTopics := extractKeywords(lower)
		for _, nt := range newTopics {
			for _, lt := range lastTopics {
				if nt == lt {
					return 0.5 // topic continuity = mild positive
				}
			}
		}
	}

	// Negative signals
	for _, w := range []string{"no ", "not what", "wrong", "incorrect",
		"that's not", "i meant", "actually", "try again", "rephrase"} {
		if strings.Contains(lower, w) {
			return -0.5
		}
	}
	if sentiment == SentimentAngry || sentiment == SentimentNegative {
		return -1.0
	}

	// Neutral: normal conversation flow
	return 0.3
}

// PhraseStats returns the top N highest and lowest scored phrases.
// Useful for debugging and understanding how the engine is evolving.
func (c *Composer) PhraseStats(n int) (top []string, bottom []string) {
	type scored struct {
		phrase string
		score  float64
	}
	var all []scored
	for phrase, score := range c.phraseScores {
		all = append(all, scored{phrase, score})
	}

	// Sort descending by score
	for i := 0; i < len(all); i++ {
		for j := i + 1; j < len(all); j++ {
			if all[j].score > all[i].score {
				all[i], all[j] = all[j], all[i]
			}
		}
	}

	for i := 0; i < n && i < len(all); i++ {
		top = append(top, fmt.Sprintf("%.2f %s", all[i].score, all[i].phrase))
	}
	for i := len(all) - 1; i >= 0 && len(bottom) < n; i-- {
		if all[i].score < 1.0 {
			bottom = append(bottom, fmt.Sprintf("%.2f %s", all[i].score, all[i].phrase))
		}
	}
	return
}

// pickToned selects a phrase appropriate for the current tone.
func (c *Composer) pickToned(neutral, casual, warm, direct []string, tone Tone) string {
	switch tone {
	case ToneCasual:
		if len(casual) > 0 {
			return c.pick(casual)
		}
	case ToneWarm:
		if len(warm) > 0 {
			return c.pick(warm)
		}
	case ToneDirect:
		if len(direct) > 0 {
			return c.pick(direct)
		}
	}
	return c.pick(neutral)
}

// randomTone picks a tone for this response, influenced by emotional memory.
// If the user has been positive recently → lean warm/casual.
// If the user has been negative recently → lean warm (empathetic).
// Otherwise → truly random across all 4 tones.
func (c *Composer) randomTone() Tone {
	// Check recent emotional state from conversation history
	if len(c.history) > 0 {
		recentTopics := c.history[len(c.history)-1].Topics
		var totalEmotion float64
		var emotionCount int
		for _, topic := range recentTopics {
			if e, ok := c.emotionalMem[topic]; ok {
				totalEmotion += e
				emotionCount++
			}
		}
		if emotionCount > 0 {
			avgEmotion := totalEmotion / float64(emotionCount)
			if avgEmotion > 0.3 {
				// User is happy about this topic → casual or warm
				if c.rng.Float64() < 0.6 {
					return ToneCasual
				}
				return ToneWarm
			}
			if avgEmotion < -0.3 {
				// User is stressed about this topic → warm (empathetic)
				if c.rng.Float64() < 0.7 {
					return ToneWarm
				}
				return ToneNeutral
			}
		}
	}
	// No emotional signal → random
	return Tone(c.rng.Intn(4))
}

// -----------------------------------------------------------------------
// Greeting Composer
// -----------------------------------------------------------------------

func (c *Composer) composeGreeting(ctx *ComposeContext) *ComposedResponse {
	var parts []string
	var sources []string
	tone := c.randomTone()

	hour := time.Now().Hour()
	greeting := c.pick(morningGreetings)
	if hour >= 12 && hour < 17 {
		greeting = c.pick(afternoonGreetings)
	} else if hour >= 17 {
		greeting = c.pick(eveningGreetings)
	}

	if ctx != nil && ctx.UserName != "" && looksLikeProperName(ctx.UserName) {
		// Vary how the name is placed
		switch c.rng.Intn(3) {
		case 0:
			greeting += " " + ctx.UserName
		case 1:
			greeting += ", " + ctx.UserName
		case 2:
			greeting = ctx.UserName + " — " + lowerFirst(greeting)
		}
	}
	parts = append(parts, greeting+".")

	if ctx != nil {
		// Shuffle which contextual facts we mention and in what order
		type contextFact struct {
			text   string
			source string
		}
		var contextFacts []contextFact

		if ctx.JournalDays > 2 {
			contextFacts = append(contextFacts, contextFact{
				text:   fmt.Sprintf(c.pick(journalGapPhrases), ctx.JournalDays),
				source: "journal_pattern",
			})
		}
		if ctx.RecentMood > 0 && ctx.RecentMood < 2.5 {
			contextFacts = append(contextFacts, contextFact{
				text:   c.pickToned(lowMoodPhrases, lowMoodCasual, lowMoodWarm, lowMoodDirect, tone),
				source: "mood_tracking",
			})
		} else if ctx.RecentMood >= 4.0 {
			contextFacts = append(contextFacts, contextFact{
				text:   c.pickToned(highMoodPhrases, highMoodCasual, highMoodWarm, highMoodDirect, tone),
				source: "mood_tracking",
			})
		}
		if ctx.HabitStreak >= 5 {
			contextFacts = append(contextFacts, contextFact{
				text:   fmt.Sprintf(c.pick(habitStreakPhrases), ctx.HabitStreak),
				source: "habit_tracking",
			})
		}
		if ctx.WeeklySpend > 0 && ctx.AvgWeeklySpend > 0 {
			ratio := ctx.WeeklySpend / ctx.AvgWeeklySpend
			if ratio < 0.8 {
				contextFacts = append(contextFacts, contextFact{
					text:   c.pick(underBudgetPhrases),
					source: "expense_tracking",
				})
			} else if ratio > 1.2 {
				contextFacts = append(contextFacts, contextFact{
					text:   c.pick(overBudgetPhrases),
					source: "expense_tracking",
				})
			}
		}

		// Shuffle order for uniqueness
		c.rng.Shuffle(len(contextFacts), func(i, j int) {
			contextFacts[i], contextFacts[j] = contextFacts[j], contextFacts[i]
		})
		for _, cf := range contextFacts {
			parts = append(parts, cf.text)
			sources = append(sources, cf.source)
		}
	}

	return &ComposedResponse{
		Text:    strings.Join(parts, " "),
		Sources: sources,
		Type:    RespGreeting,
	}
}

// -----------------------------------------------------------------------
// Factual Composer — sentence fusion + structural transforms
// -----------------------------------------------------------------------

func (c *Composer) composeFactual(query string) *ComposedResponse {
	if c.Graph == nil || c.Graph.NodeCount() == 0 {
		return nil
	}

	// Fast path: if a clean described_as fact exists, use it directly.
	// These are human-written descriptions (typically from Wikipedia)
	// and don't need template wrapping.
	cleanQuery := strings.TrimRight(strings.ToLower(strings.TrimSpace(query)), "?!.")
	queryTopic := cleanQuery
	if t := c.extractTopic(query); t != "" {
		queryTopic = strings.ToLower(t)
	}
	if desc := c.Graph.LookupDescription(queryTopic); desc != "" {
		text := desc
		if extras := c.Graph.LookupFacts(queryTopic, 4); len(extras) > 0 {
			varied := c.applyPronounVariation(extras, queryTopic)
			text += "\n\n" + strings.Join(varied, " ")
		}
		return &ComposedResponse{
			Text:    text,
			Sources: []string{"knowledge"},
			Type:    RespFactual,
		}
	}

	facts, sources := c.gatherFacts(query)

	// Determine topic from facts or query
	topic := cleanQuery
	if len(facts) > 0 {
		// Relevance guard: verify facts are about the queried topic.
		ftLower := strings.ToLower(facts[0].Subject)
		qtLower := strings.ToLower(queryTopic)
		if qtLower != ftLower &&
			!strings.Contains(qtLower, ftLower) &&
			!strings.Contains(ftLower, qtLower) {
			return nil // facts are unrelated to query
		}
		topic = facts[0].Subject
	}

	// Clean factual output: description + structured realization.
	var parts []string
	desc := c.Graph.LookupDescription(topic)
	if len(desc) >= 40 {
		parts = append(parts, desc)
	}
	// Filter out facts already covered by the description to avoid duplication.
	if desc != "" {
		facts = filterRedundantFacts(facts, desc)
	}
	if len(facts) > 0 {
		// For rich topics (4+ facts), use discourse planner for paragraph structure
		if c.Discourse != nil && len(facts) >= 4 {
			plan := c.Discourse.PlanFromFacts(topic, facts, RespFactual)
			if plan != nil && len(plan.Sections) > 1 {
				text := c.realizePlan(plan)
				if text != "" {
					parts = append(parts, text)
				}
			}
		}
		// Fallback or few facts: flat structured realization
		if len(parts) <= 1 {
			factText := c.structuredRealization(facts)
			if factText != "" {
				parts = append(parts, factText)
			}
		}
	}
	if len(parts) == 0 {
		return nil
	}
	text := cleanSentences(strings.Join(parts, "\n\n"))
	return &ComposedResponse{
		Text:    text,
		Sources: uniqueStrings(sources),
		Type:    RespFactual,
	}
}

// cleanSentences post-processes composed text to remove broken fragments,
// fix pronoun capitalization issues, and eliminate duplicate sentences.
func cleanSentences(text string) string {
	// Split on paragraph boundaries to preserve structure.
	paragraphs := strings.Split(text, "\n\n")
	var cleanParagraphs []string

	// Collect all sentences across paragraphs for substring dedup.
	var allSentences []string

	for _, para := range paragraphs {
		sentences := splitCleanSentences(para)
		for _, s := range sentences {
			s = strings.TrimSpace(s)
			if s == "" {
				continue
			}
			allSentences = append(allSentences, s)
		}
	}

	for _, para := range paragraphs {
		sentences := splitCleanSentences(para)
		var kept []string
		for _, s := range sentences {
			s = strings.TrimSpace(s)
			if s == "" {
				continue
			}
			// Remove short fragments (likely broken).
			if len(s) < 15 {
				continue
			}
			// Fix mid-sentence pronoun capitalization: "of It" → "of it"
			s = fixPronounCase(s)
			// Remove sentences that are substrings of other sentences.
			if isSentenceSubstring(s, allSentences) {
				continue
			}
			kept = append(kept, s)
		}
		if len(kept) > 0 {
			p := strings.Join(kept, " ")
			// Collapse multiple spaces.
			p = collapseCleanSpaces(p)
			cleanParagraphs = append(cleanParagraphs, p)
		}
	}
	if len(cleanParagraphs) == 0 {
		return text // Don't destroy the response if cleaning is too aggressive
	}
	return strings.Join(cleanParagraphs, "\n\n")
}

// splitCleanSentences splits text on sentence-ending punctuation while
// keeping the terminator attached to each sentence.
func splitCleanSentences(text string) []string {
	var sentences []string
	start := 0
	for i := 0; i < len(text); i++ {
		if text[i] == '.' || text[i] == '!' || text[i] == '?' {
			// Look ahead to see if this is truly a sentence boundary
			// (followed by space+uppercase, end of string, or newline).
			end := i + 1
			if end >= len(text) || (end < len(text) && (text[end] == ' ' || text[end] == '\n')) {
				sentences = append(sentences, strings.TrimSpace(text[start:end]))
				start = end
			}
		}
	}
	if start < len(text) {
		remainder := strings.TrimSpace(text[start:])
		if remainder != "" {
			sentences = append(sentences, remainder)
		}
	}
	return sentences
}

// fixPronounCase fixes common mid-sentence pronoun capitalization errors
// like "of It", "use of It", "for It".
var pronounCaseRe = regexp.MustCompile(`\b(of|for|about|with|from|by|to|use of|in) (It|Its)\b`)

func fixPronounCase(s string) string {
	return pronounCaseRe.ReplaceAllStringFunc(s, func(m string) string {
		return strings.Replace(m, " It", " it", 1)
	})
}

// isSentenceSubstring returns true if sentence is a proper substring of
// any other sentence in the list (not equal, just strictly contained).
func isSentenceSubstring(sentence string, all []string) bool {
	lower := strings.ToLower(sentence)
	for _, other := range all {
		otherLower := strings.ToLower(other)
		if len(otherLower) > len(lower) && strings.Contains(otherLower, lower) {
			return true
		}
	}
	return false
}

// collapseCleanSpaces replaces runs of whitespace with a single space and
// fixes double-period punctuation.
func collapseCleanSpaces(s string) string {
	// Collapse multiple spaces.
	for strings.Contains(s, "  ") {
		s = strings.ReplaceAll(s, "  ", " ")
	}
	// Fix double periods.
	s = strings.ReplaceAll(s, "..", ".")
	return s
}

// filterRedundantFacts removes facts whose object is already mentioned
// in a pre-written description, avoiding duplicated information.
func filterRedundantFacts(facts []edgeFact, description string) []edgeFact {
	lower := strings.ToLower(description)
	var filtered []edgeFact
	for _, f := range facts {
		objLower := strings.ToLower(f.Object)
		// Skip if the object is already mentioned in the description
		if strings.Contains(lower, objLower) {
			continue
		}
		filtered = append(filtered, f)
	}
	return filtered
}

// gatherFacts extracts typed facts from the graph for a query.
func (c *Composer) gatherFacts(query string) ([]edgeFact, []string) {
	lower := strings.ToLower(strings.TrimRight(query, "?!."))
	keywords := extractKeywords(lower)

	// extractKeywords filters short words like "Go", "AI", "C" — but those
	// are valid topics. Use extractTopic as fallback for short-word queries.
	topicStr := ""
	if topic := c.extractTopic(query); topic != "" {
		topicStr = strings.ToLower(topic)
		found := false
		for _, kw := range keywords {
			if kw == topicStr {
				found = true
				break
			}
		}
		if !found {
			keywords = append([]string{topicStr}, keywords...)
		}
	}

	var facts []edgeFact
	var sources []string
	seen := make(map[string]bool)

	c.Graph.mu.RLock()
	defer c.Graph.mu.RUnlock()

	for _, kw := range keywords {
		id := nodeID(kw)
		node := c.Graph.nodes[id]
		if node == nil {
			if ids, ok := c.Graph.byLabel[strings.ToLower(kw)]; ok && len(ids) > 0 {
				id = ids[0]
				node = c.Graph.nodes[id]
			}
		}
		if node == nil {
			continue
		}

		// Relevance guard: if we have a multi-word topic like "meaning of life"
		// but only matched a single generic keyword like "life", skip it.
		// Single-keyword matches on generic words produce unrelated content.
		if topicStr != "" && kw != topicStr {
			nodeLower := strings.ToLower(node.Label)
			// The matched node must contain the keyword AND be related to
			// the overall topic. A node labeled "life" matching keyword "life"
			// from query "meaning of life" is too loose.
			if !strings.Contains(nodeLower, topicStr) && !strings.Contains(topicStr, nodeLower) {
				// Check if the node label shares at least 2 significant words
				// with the topic to allow reasonable partial matches.
				topicWords := strings.Fields(topicStr)
				if len(topicWords) > 1 {
					matchCount := 0
					for _, tw := range topicWords {
						if len(tw) > 3 && strings.Contains(nodeLower, tw) {
							matchCount++
						}
					}
					if matchCount < 2 {
						continue // too loosely related
					}
				}
			}
		}

		for _, edge := range c.Graph.outEdges[id] {
			// Skip described_as — handled separately via LookupDescription
			// to avoid duplicating the wiki description inside fact templates.
			if edge.Relation == RelDescribedAs {
				continue
			}
			to := c.Graph.nodes[edge.To]
			if to == nil {
				continue
			}
			key := node.Label + "|" + string(edge.Relation) + "|" + to.Label
			if seen[key] {
				continue
			}
			seen[key] = true

			facts = append(facts, edgeFact{
				Subject:  node.Label,
				Relation: edge.Relation,
				Object:   to.Label,
				Inferred: edge.Inferred,
			})
			if edge.Inferred {
				sources = append(sources, "inference")
			} else {
				sources = append(sources, "knowledge_graph")
			}
		}

		if len(facts) > 0 {
			break
		}
	}

	return facts, sources
}

// realizeFacts converts facts to natural language using multiple strategies.
func (c *Composer) realizeFacts(facts []edgeFact) string {
	if len(facts) == 0 {
		return ""
	}

	// Filter out fragment objects before realization
	var clean []edgeFact
	for _, f := range facts {
		if !isFragmentObject(f.Object) {
			clean = append(clean, f)
		}
	}
	if len(clean) == 0 {
		return ""
	}

	// Shuffle facts for different emphasis each time
	shuffled := make([]edgeFact, len(clean))
	copy(shuffled, clean)
	c.rng.Shuffle(len(shuffled), func(i, j int) {
		shuffled[i], shuffled[j] = shuffled[j], shuffled[i]
	})

	// Strategy selection — 5 template-based strategies.
	// (Generative engine disabled for factual output — quality too low.)
	strategy := c.rng.Intn(5)

	switch {
	case len(shuffled) >= 3 && strategy == 0:
		return c.fusedRealization(shuffled)
	case len(shuffled) >= 2 && strategy == 1:
		return c.flowingRealization(shuffled)
	case len(shuffled) >= 2 && strategy == 2:
		return c.appositiveRealization(shuffled)
	case len(shuffled) >= 2 && strategy == 3:
		return c.topicalizedRealization(shuffled)
	default:
		return c.structuredRealization(shuffled)
	}
}

// fusedRealization combines multiple facts into complex sentences.
// "Stoicera is a philosophy company based in Vienna, founded by Raphael."
func (c *Composer) fusedRealization(facts []edgeFact) string {
	if len(facts) == 0 {
		return ""
	}

	subject := facts[0].Subject
	var fusedParts []string
	var remaining []edgeFact

	for _, f := range facts {
		if f.Subject == subject {
			fusedParts = append(fusedParts, c.factFragment(f))
		} else {
			remaining = append(remaining, f)
		}
	}

	var b strings.Builder
	b.WriteString(subject)
	if len(fusedParts) == 1 {
		b.WriteString(" " + fusedParts[0] + ".")
	} else if len(fusedParts) == 2 {
		// Vary the conjunction
		conj := c.pick([]string{", ", " and ", " — "})
		b.WriteString(" " + fusedParts[0] + conj + fusedParts[1] + ".")
	} else {
		for i, part := range fusedParts {
			if i == 0 {
				b.WriteString(" " + part)
			} else if i == len(fusedParts)-1 {
				lastConj := c.pick([]string{", and also ", ". Additionally, it ", ". It also "})
				b.WriteString(lastConj + part)
			} else {
				b.WriteString(", " + part)
			}
		}
		b.WriteString(".")
	}

	for _, f := range remaining {
		sentence := c.transformSentence(f)
		if sentence != "" {
			b.WriteString(" " + sentence)
		}
	}

	return b.String()
}

// factFragment returns a sentence fragment for fusion (no subject, no period).
func (c *Composer) factFragment(f edgeFact) string {
	if isFragmentObject(f.Object) {
		return ""
	}
	switch f.Relation {
	case RelIsA:
		return c.pick([]string{
			"is a " + f.Object,
			"is a type of " + f.Object,
			"is classified as a " + f.Object,
			"is what you'd call a " + f.Object,
			"counts as a " + f.Object,
			"falls into the " + f.Object + " category",
		})
	case RelLocatedIn:
		return c.pick([]string{
			"is based in " + f.Object,
			"is located in " + f.Object,
			"is headquartered in " + f.Object,
			"operates out of " + f.Object,
			"has roots in " + f.Object,
			"calls " + f.Object + " home",
		})
	case RelFoundedBy:
		return c.pick([]string{
			"was founded by " + f.Object,
			"was created by " + f.Object,
			"was started by " + f.Object,
			"is the brainchild of " + f.Object,
			"was brought to life by " + f.Object,
			"was built from the ground up by " + f.Object,
		})
	case RelFoundedIn:
		if looksLikePersonName(f.Subject) {
			return c.pick([]string{
				"was born in " + f.Object,
				"came into the world in " + f.Object,
			})
		}
		return c.pick([]string{
			"was founded in " + f.Object,
			"was established in " + f.Object,
			"was started in " + f.Object,
			"has been around since " + f.Object,
			"dates back to " + f.Object,
		})
	case RelOffers:
		return c.pick([]string{
			"offering " + f.Object,
			"providing " + f.Object,
			"featuring " + f.Object,
			"built around " + f.Object,
			"centered on " + f.Object,
		})
	case RelHas:
		return c.pick([]string{
			"with " + f.Object,
			"featuring " + f.Object,
			"including " + f.Object,
			"equipped with " + f.Object,
			"sporting " + f.Object,
		})
	case RelDescribedAs:
		return c.pick([]string{
			"known for being " + f.Object,
			"described as " + f.Object,
			"characterized as " + f.Object,
			"widely regarded as " + f.Object,
			"recognized for being " + f.Object,
		})
	case RelUsedFor:
		return c.pick([]string{
			"used for " + f.Object,
			"applied to " + f.Object,
			"serving as " + f.Object,
			"designed for " + f.Object,
			"geared toward " + f.Object,
			"built for " + f.Object,
		})
	case RelRelatedTo:
		return c.pick([]string{
			"connected to " + f.Object,
			"linked to " + f.Object,
			"related to " + f.Object,
			"tied to " + f.Object,
			"with connections to " + f.Object,
		})
	case RelDomain:
		return "in the " + f.Object + " domain"
	case RelPartOf:
		return c.pick([]string{
			"part of " + f.Object,
			"a piece of " + f.Object,
			"belonging to " + f.Object,
		})
	case RelPrefers:
		return c.pick([]string{
			"with a preference for " + f.Object,
			"favoring " + f.Object,
			"leaning toward " + f.Object,
		})
	case RelCauses:
		return c.pick([]string{
			"which leads to " + f.Object,
			"resulting in " + f.Object,
			"driving " + f.Object,
		})
	default:
		return "associated with " + f.Object
	}
}

// appositiveRealization: "Stoicera, a philosophy company, is based in Vienna."
func (c *Composer) appositiveRealization(facts []edgeFact) string {
	if len(facts) < 2 {
		return c.structuredRealization(facts)
	}

	subject := facts[0].Subject
	// Find an is_a fact for the appositive, use the rest normally
	var appositiveFact *edgeFact
	var remaining []edgeFact

	for i := range facts {
		if facts[i].Relation == RelIsA && facts[i].Subject == subject && appositiveFact == nil {
			appositiveFact = &facts[i]
		} else {
			remaining = append(remaining, facts[i])
		}
	}

	if appositiveFact == nil {
		// No is_a fact, fall back to structured
		return c.structuredRealization(facts)
	}

	// Build: "Subject, a <type>, <rest>"
	var b strings.Builder
	apposition := c.pick([]string{
		"a " + appositiveFact.Object,
		"which is a " + appositiveFact.Object,
		"a notable " + appositiveFact.Object,
	})

	if len(remaining) > 0 {
		// "Stoicera, a philosophy company, is based in Vienna."
		first := remaining[0]
		verb := c.relationVerb(first.Relation, first.Object, subject)
		b.WriteString(subject + ", " + apposition + ", " + verb + ".")

		// Handle rest with pronouns
		for i := 1; i < len(remaining); i++ {
			ref := c.smartRef(subject, i+1)
			sentence := c.edgeToSentence(ref, remaining[i].Relation, remaining[i].Object, remaining[i].Inferred)
			if sentence != "" {
				b.WriteString(" " + sentence)
			}
		}
	} else {
		b.WriteString(subject + " is " + apposition + ".")
	}

	return b.String()
}

// topicalizedRealization: "Based in Vienna, Stoicera is a philosophy company."
func (c *Composer) topicalizedRealization(facts []edgeFact) string {
	if len(facts) < 2 {
		return c.structuredRealization(facts)
	}

	subject := facts[0].Subject

	// Pick one fact to topicalize (front), use another as main clause
	topicIdx := c.rng.Intn(len(facts))
	topicFact := facts[topicIdx]
	topicPhrase := c.topicPhrase(topicFact)

	var b strings.Builder
	b.WriteString(topicPhrase + ", " + subject)

	// Add the other facts
	first := true
	for i, f := range facts {
		if i == topicIdx {
			continue
		}
		if first {
			b.WriteString(" " + c.factFragment(f) + ".")
			first = false
		} else {
			ref := c.smartRef(subject, i+1)
			sentence := c.edgeToSentence(ref, f.Relation, f.Object, f.Inferred)
			if sentence != "" {
				b.WriteString(" " + sentence)
			}
		}
	}

	return b.String()
}

// topicPhrase converts a fact to a fronted/topicalized phrase.
func (c *Composer) topicPhrase(f edgeFact) string {
	switch f.Relation {
	case RelLocatedIn:
		return c.pick([]string{
			"Based in " + f.Object,
			"Operating out of " + f.Object,
			"Rooted in " + f.Object,
			"Hailing from " + f.Object,
			"With its home in " + f.Object,
		})
	case RelFoundedBy:
		return c.pick([]string{
			"Founded by " + f.Object,
			"The creation of " + f.Object,
			"Built by " + f.Object,
			"Started by " + f.Object,
			"Brought into being by " + f.Object,
		})
	case RelFoundedIn:
		if looksLikePersonName(f.Subject) {
			return c.pick([]string{
				"Born in " + f.Object,
			})
		}
		return c.pick([]string{
			"Established in " + f.Object,
			"Dating back to " + f.Object,
			"Around since " + f.Object,
		})
	case RelIsA:
		return c.pick([]string{
			"As a " + f.Object,
			"Being a " + f.Object,
			"A " + f.Object + " at heart",
			"Firmly a " + f.Object,
		})
	case RelDescribedAs:
		return c.pick([]string{
			"Known for being " + f.Object,
			"Widely considered " + f.Object,
			"Recognized as " + f.Object,
		})
	case RelUsedFor:
		return c.pick([]string{
			"Designed for " + f.Object,
			"Built for " + f.Object,
			"Created to serve " + f.Object,
		})
	default:
		return capitalizeFirst(c.factFragment(f))
	}
}

// relationVerb returns a verb phrase for a relation (used in appositive construction).
func (c *Composer) relationVerb(rel RelType, object string, subject string) string {
	switch rel {
	case RelLocatedIn:
		return c.pick([]string{
			"is based in " + object,
			"operates from " + object,
			"calls " + object + " home",
			"has its roots in " + object,
		})
	case RelFoundedBy:
		return c.pick([]string{
			"was founded by " + object,
			"was created by " + object,
			"owes its existence to " + object,
		})
	case RelFoundedIn:
		if isPersonReference(subject) {
			return "was born in " + object
		}
		return c.pick([]string{
			"was established in " + object,
			"has been around since " + object,
		})
	case RelUsedFor:
		return c.pick([]string{
			"is used for " + object,
			"is designed for " + object,
		})
	default:
		return "is " + c.factFragment(edgeFact{Relation: rel, Object: object})
	}
}

// flowingRealization uses discourse connectors with pronoun substitution.
func (c *Composer) flowingRealization(facts []edgeFact) string {
	if len(facts) == 0 {
		return ""
	}

	var b strings.Builder
	subject := facts[0].Subject
	mentionCount := 0

	for i, f := range facts {
		if i == 0 {
			b.WriteString(c.transformSentence(f))
			mentionCount++
		} else {
			displaySubj := f.Subject
			if f.Subject == subject {
				mentionCount++
				displaySubj = c.smartRef(subject, mentionCount)
			}

			connector := c.pick(allConnectors)
			sentence := c.edgeToSentence(displaySubj, f.Relation, f.Object, f.Inferred)
			if sentence != "" {
				b.WriteString(" " + connector + " " + lowerFirst(sentence))
			}
		}
	}

	return b.String()
}

// structuredRealization: one fact per sentence with varied templates.
// taggedSentence pairs a rendered sentence with its source relation type
// and the original subject/object for sentence fusion.
type taggedSentence struct {
	Text     string
	Relation RelType
	Subject  string // original subject from the edgeFact
	Object   string // original object from the edgeFact
}

func (c *Composer) structuredRealization(facts []edgeFact) string {
	if len(facts) == 0 {
		return ""
	}

	// Deduplicate facts by (relation, lower(object)) to avoid repeating
	// the same information with slightly different phrasing.
	seen := make(map[string]bool)
	var dedupFacts []edgeFact
	for _, f := range facts {
		key := string(f.Relation) + ":" + strings.ToLower(f.Object)
		if seen[key] {
			continue
		}
		seen[key] = true
		dedupFacts = append(dedupFacts, f)
	}
	facts = dedupFacts

	facts = c.varyOpening(facts)

	var tagged []taggedSentence
	subject := facts[0].Subject
	mentionCount := 0

	for _, f := range facts {
		// Skip fragment objects that produce nonsense sentences
		if isFragmentObject(f.Object) {
			continue
		}
		displaySubj := f.Subject
		if strings.EqualFold(f.Subject, subject) {
			mentionCount++
			if mentionCount > 1 {
				displaySubj = c.smartRef(subject, mentionCount)
			}
		}
		sentence := c.edgeToSentence(displaySubj, f.Relation, f.Object, f.Inferred)
		if sentence != "" {
			tagged = append(tagged, taggedSentence{
				Text:     sentence,
				Relation: f.Relation,
				Subject:  f.Subject,
				Object:   f.Object,
			})
		}
	}

	tagged = c.fuseSentences(tagged)
	return c.combineTaggedWithFlow(tagged)
}

// varyOpening occasionally reorders facts so that the response does not
// always begin with "X is a Y." (RelIsA). About 40% of the time it will
// lead with the most specific or interesting fact instead — a feature
// (RelHas), a purpose (RelUsedFor), or an origin fact (RelFoundedBy,
// RelCreatedBy). The identity fact is not removed, just repositioned.
func (c *Composer) varyOpening(facts []edgeFact) []edgeFact {
	if len(facts) < 3 {
		return facts // too few to meaningfully reorder
	}
	// Only reorder when the first fact is an identity fact.
	if facts[0].Relation != RelIsA {
		return facts
	}
	// 60% of the time, keep the default identity-first ordering.
	if c.rng.Intn(5) < 3 {
		return facts
	}

	// Find a more interesting lead fact: prefer has/used_for/founded_by/created_by.
	interestingRels := map[RelType]bool{
		RelHas:       true,
		RelUsedFor:   true,
		RelFoundedBy: true,
		RelCreatedBy: true,
		RelKnownFor:  true,
	}
	leadIdx := -1
	for i := 1; i < len(facts); i++ {
		if interestingRels[facts[i].Relation] {
			leadIdx = i
			break
		}
	}
	if leadIdx < 0 {
		return facts // nothing interesting to promote
	}

	// Move the interesting fact to position 0, shift identity to position 1.
	reordered := make([]edgeFact, 0, len(facts))
	reordered = append(reordered, facts[leadIdx])
	for i, f := range facts {
		if i != leadIdx {
			reordered = append(reordered, f)
		}
	}
	return reordered
}

// fuseSentences combines consecutive same-subject, same-relation facts
// into compound sentences joined by "and" or relative clauses. This avoids
// the mechanical one-fact-per-sentence cadence.
//
// Before: "Python has readable syntax. Python has an extensive standard library."
// After:  "Python has readable syntax and an extensive standard library."
//
// Before: "DNA is a molecule. DNA is used for encoding genetic information."
// After:  "DNA is a molecule used for encoding genetic information."
func (c *Composer) fuseSentences(tagged []taggedSentence) []taggedSentence {
	if len(tagged) <= 1 {
		return tagged
	}

	var result []taggedSentence
	i := 0
	for i < len(tagged) {
		// Try to fuse a run of consecutive same-subject, same-relation facts.
		j := i + 1
		for j < len(tagged) && j-i < 3 &&
			strings.EqualFold(tagged[j].Subject, tagged[i].Subject) &&
			tagged[j].Relation == tagged[i].Relation {
			j++
		}

		if j-i >= 2 {
			// We have 2-3 facts to fuse.
			fused := c.fuseRun(tagged[i:j])
			result = append(result, fused)
		} else if j-i == 1 && i+1 < len(tagged) &&
			strings.EqualFold(tagged[i+1].Subject, tagged[i].Subject) &&
			tagged[i].Relation == RelIsA && tagged[i+1].Relation == RelUsedFor {
			// Fuse "X is a Y" + "X is used for Z" → "X is a Y used for Z"
			fused := c.fuseIsAWithUsedFor(tagged[i], tagged[i+1])
			result = append(result, fused)
			i = i + 2
			continue
		} else {
			result = append(result, tagged[i])
		}
		i = j
	}
	return result
}

// fuseRun combines 2-3 facts with the same subject and relation into one
// compound sentence using "and".
//
// Strategy: take the first sentence, strip its period, then append the
// objects of the subsequent facts joined by "and".
func (c *Composer) fuseRun(run []taggedSentence) taggedSentence {
	if len(run) < 2 {
		return run[0]
	}

	base := strings.TrimRight(run[0].Text, ".")
	objects := make([]string, 0, len(run)-1)
	for _, ts := range run[1:] {
		obj := strings.TrimSpace(ts.Object)
		if obj != "" {
			objects = append(objects, obj)
		}
	}

	if len(objects) == 0 {
		return run[0]
	}

	var tail string
	if len(objects) == 1 {
		tail = " and " + objects[0]
	} else {
		tail = ", " + strings.Join(objects[:len(objects)-1], ", ") +
			", and " + objects[len(objects)-1]
	}

	return taggedSentence{
		Text:     base + tail + ".",
		Relation: run[0].Relation,
		Subject:  run[0].Subject,
		Object:   run[0].Object,
	}
}

// fuseIsAWithUsedFor combines "X is a Y." + "X is used for Z." into
// "X is a Y used for Z." using a reduced relative clause.
func (c *Composer) fuseIsAWithUsedFor(identity, purpose taggedSentence) taggedSentence {
	base := strings.TrimRight(identity.Text, ".")
	obj := strings.TrimSpace(purpose.Object)
	if obj == "" {
		return identity
	}
	return taggedSentence{
		Text:     base + " used for " + obj + ".",
		Relation: identity.Relation,
		Subject:  identity.Subject,
		Object:   identity.Object,
	}
}

// combineTaggedWithFlow joins sentences using relation-aware connectors.
func (c *Composer) combineTaggedWithFlow(tagged []taggedSentence) string {
	if len(tagged) == 0 {
		return ""
	}
	if len(tagged) == 1 {
		return tagged[0].Text
	}

	// Anti-repetition: track connectors used within this response so
	// the same non-empty connector is not repeated within a 3-sentence window.
	recentConnectors := make([]string, 0, len(tagged))

	var b strings.Builder
	b.WriteString(tagged[0].Text)
	for i := 1; i < len(tagged); i++ {
		connector := c.connectorBetweenAvoid(
			tagged[i-1].Relation, tagged[i].Relation, recentConnectors,
		)
		recentConnectors = append(recentConnectors, connector)
		if connector == "" {
			b.WriteString(" " + tagged[i].Text)
		} else {
			b.WriteString(" " + connector + " " + connectorLowerFirst(tagged[i].Text))
		}
	}
	return b.String()
}

// connectorBetweenAvoid picks a connector like connectorBetween but avoids
// any non-empty connector that appeared in the last 3 entries of recent.
func (c *Composer) connectorBetweenAvoid(prev, cur RelType, recent []string) string {
	const maxRetries = 4
	avoid := make(map[string]bool)
	start := len(recent) - 3
	if start < 0 {
		start = 0
	}
	for _, r := range recent[start:] {
		if r != "" {
			avoid[r] = true
		}
	}

	for attempt := 0; attempt < maxRetries; attempt++ {
		candidate := c.connectorBetween(prev, cur)
		if candidate == "" || !avoid[candidate] {
			return candidate
		}
	}
	// After retries, fall back to no connector for natural flow.
	return ""
}

// realizePlan renders a discourse plan into paragraphed text.
// Each non-empty section becomes a paragraph, connected by its discourse connector.
func (c *Composer) realizePlan(plan *DiscoursePlan) string {
	if plan == nil || len(plan.Sections) == 0 {
		return ""
	}

	var paragraphs []string
	for _, section := range plan.Sections {
		if len(section.Facts) == 0 {
			continue // Skip hook/close sections with no facts
		}
		text := c.structuredRealization(section.Facts)
		if text == "" {
			continue
		}
		// Prepend the section's discourse connector if present
		if section.Connector != "" {
			text = section.Connector + " " + connectorLowerFirst(text)
		}
		paragraphs = append(paragraphs, text)
	}
	if len(paragraphs) == 0 {
		return ""
	}
	return strings.Join(paragraphs, "\n\n")
}

// generativeRealization uses the grammar-rule-based GenerativeEngine
// to compose sentences from first principles instead of templates.
func (c *Composer) generativeRealization(facts []edgeFact) string {
	if c.Generative == nil || len(facts) == 0 {
		return c.structuredRealization(facts)
	}
	return c.Generative.GenerateFromFacts(facts)
}

// -----------------------------------------------------------------------
// Sentence Transforms — make each sentence structurally unique
// -----------------------------------------------------------------------

// transformSentence applies a random structural transform to a fact.
func (c *Composer) transformSentence(f edgeFact) string {
	transform := c.rng.Intn(4)
	switch transform {
	case 0:
		// Active voice (default)
		return c.edgeToSentence(f.Subject, f.Relation, f.Object, f.Inferred)
	case 1:
		// Passive voice: "Vienna is where Stoicera is based."
		return c.passiveSentence(f)
	case 2:
		// Cleft: "It's Raphael who founded Stoicera."
		return c.cleftSentence(f)
	case 3:
		// Emphatic/dash: "Stoicera — a philosophy company based in Vienna."
		return c.emphaticSentence(f)
	}
	return c.edgeToSentence(f.Subject, f.Relation, f.Object, f.Inferred)
}

// passiveSentence flips subject/object for a different perspective.
func (c *Composer) passiveSentence(f edgeFact) string {
	switch f.Relation {
	case RelLocatedIn:
		return c.pick([]string{
			f.Object + " is where " + f.Subject + " is based.",
			f.Object + " is home to " + f.Subject + ".",
			"In " + f.Object + ", you'll find " + f.Subject + ".",
		})
	case RelFoundedBy:
		return c.pick([]string{
			f.Object + " is the one who founded " + f.Subject + ".",
			f.Object + " is behind " + f.Subject + ".",
			f.Object + " brought " + f.Subject + " to life.",
		})
	case RelIsA:
		return c.pick([]string{
			"When it comes to " + f.Object + ", " + f.Subject + " is a notable example.",
			"In the world of " + f.Object + ", there's " + f.Subject + ".",
			"Among " + simplePlural(f.Object) + ", " + f.Subject + " stands out.",
		})
	case RelUsedFor:
		return c.pick([]string{
			f.Object + " is what " + f.Subject + " is designed for.",
			"For " + f.Object + ", " + f.Subject + " is a solid choice.",
			"If you need " + f.Object + ", " + f.Subject + " delivers.",
		})
	case RelCreatedBy:
		return c.pick([]string{
			f.Object + " created " + f.Subject + ".",
			f.Object + " is the mind behind " + f.Subject + ".",
		})
	default:
		// Fall back to active
		return c.edgeToSentence(f.Subject, f.Relation, f.Object, f.Inferred)
	}
}

// cleftSentence creates emphasis: "It's X who/that Y."
func (c *Composer) cleftSentence(f edgeFact) string {
	switch f.Relation {
	case RelFoundedBy:
		return c.pick([]string{
			"It's " + f.Object + " who founded " + f.Subject + ".",
			"It was " + f.Object + " who started " + f.Subject + ".",
			"It was none other than " + f.Object + " who built " + f.Subject + ".",
		})
	case RelLocatedIn:
		return c.pick([]string{
			"It's in " + f.Object + " that " + f.Subject + " is based.",
			"It's " + f.Object + " where " + f.Subject + " calls home.",
		})
	case RelIsA:
		return c.pick([]string{
			"What " + f.Subject + " really is, is a " + f.Object + ".",
			"At its core, " + f.Subject + " is a " + f.Object + ".",
			"First and foremost, " + f.Subject + " is a " + f.Object + ".",
		})
	default:
		return c.edgeToSentence(f.Subject, f.Relation, f.Object, f.Inferred)
	}
}

// emphaticSentence uses dashes and colons for emphasis.
func (c *Composer) emphaticSentence(f edgeFact) string {
	fragment := c.factFragment(f)
	return c.pick([]string{
		f.Subject + " — " + fragment + ".",
		f.Subject + ": " + fragment + ".",
		"Here's the thing about " + f.Subject + " — " + fragment + ".",
		"One thing to know: " + f.Subject + " is " + fragment + ".",
	})
}

// -----------------------------------------------------------------------
// Smart References — context-aware alternatives to "it"
// -----------------------------------------------------------------------

// smartRef returns a contextually appropriate reference for a subject.
func (c *Composer) smartRef(subject string, mentionCount int) string {
	// Always capitalize the subject for sentence-initial position.
	subject = capitalizeFirst(subject)

	if mentionCount <= 1 {
		return subject
	}

	lower := strings.ToLower(subject)

	// Detect entity type for smarter references
	isCompany := containsAny(lower, "inc", "corp", "llc", "ltd", "gmbh", "company", "co")
	isPerson := looksLikePersonName(subject)
	isLanguage := containsAny(lower, "script", "lang") ||
		isIn(lower, "go", "rust", "python", "java", "ruby", "swift", "kotlin", "c", "c++", "javascript", "typescript")

	isPlural := isLikelyPlural(subject)

	if mentionCount == 2 {
		// Second mention: pronoun or short ref
		if isPerson {
			// Use last name (more natural: "Einstein" not "Albert")
			parts := strings.Fields(subject)
			if len(parts) > 1 {
				return parts[len(parts)-1]
			}
			return genderPronoun(detectGender(subject))
		}
		if isCompany {
			return c.pick([]string{"It", "The company", "The team"})
		}
		if isLanguage {
			return c.pick([]string{"It", "The language"})
		}
		if isPlural {
			return "They"
		}
		return "It"
	}

	// Third+ mention: vary more
	if isPerson {
		pronoun := genderPronoun(detectGender(subject))
		parts := strings.Fields(subject)
		if len(parts) > 1 {
			return c.pick([]string{parts[len(parts)-1], pronoun, pronoun})
		}
		return pronoun
	}
	if isCompany {
		return c.pick([]string{"It", "The company", "They", "The organization"})
	}
	if isLanguage {
		return c.pick([]string{"It", "The language", "It"})
	}
	if isPlural {
		return c.pick([]string{"They", "They", subject})
	}
	return c.pick([]string{"It", "It", subject})
}

// applyPronounVariation replaces repeated subject names with pronouns in a
// list of fact sentences. The first sentence keeps the full subject; subsequent
// ones use "It"/"They" etc.
func (c *Composer) applyPronounVariation(sentences []string, topic string) []string {
	if len(sentences) <= 1 {
		return sentences
	}
	// Detect the actual subject from the first sentence. The topic from NLU
	// may be partial (e.g., "einstein" for "Albert Einstein"), so we find the
	// longest common prefix that repeats across sentences.
	subject := detectRepeatedSubject(sentences)
	if subject == "" {
		subject = capitalizeFirst(topic)
	}
	result := make([]string, len(sentences))
	result[0] = sentences[0]
	for i := 1; i < len(sentences); i++ {
		s := sentences[i]
		ref := c.smartRef(subject, i+1)
		if strings.HasPrefix(s, subject+" ") {
			s = ref + s[len(subject):]
			// Fix verb agreement when pronoun changes number.
			// E.g., "They" replacing singular subject needs "has"→"have", "is"→"are".
			if isLikelyPlural(ref) && !isLikelyPlural(subject) {
				s = fixSubjectVerbAgreement(s, ref)
			}
		}
		result[i] = s
	}
	return result
}

// detectRepeatedSubject finds the common subject prefix across sentences.
// E.g., ["Albert Einstein is...", "Albert Einstein has..."] → "Albert Einstein"
func detectRepeatedSubject(sentences []string) string {
	if len(sentences) < 2 {
		return ""
	}
	// Try progressively longer prefixes from the first sentence,
	// stopping before verb words.
	words := strings.Fields(sentences[0])
	verbs := map[string]bool{
		"is": true, "are": true, "was": true, "were": true,
		"has": true, "have": true, "had": true,
		"does": true, "do": true, "did": true,
		"can": true, "could": true, "will": true, "would": true,
		"causes": true, "leads": true, "offers": true,
		"includes": true, "features": true, "relates": true,
	}
	best := ""
	for n := 1; n <= len(words) && n <= 5; n++ {
		word := strings.ToLower(words[n-1])
		// Stop if this word is a verb — it's not part of the subject.
		if verbs[word] {
			break
		}
		candidate := strings.Join(words[:n], " ")
		matches := 0
		for _, s := range sentences[1:] {
			if strings.HasPrefix(s, candidate+" ") {
				matches++
			}
		}
		if matches >= len(sentences[1:])/2 {
			best = candidate
		} else {
			break
		}
	}
	return best
}

// -----------------------------------------------------------------------
// Acknowledge Composer
// -----------------------------------------------------------------------

func (c *Composer) composeAcknowledge(action string, ctx *ComposeContext) *ComposedResponse {
	lower := strings.ToLower(action)
	var parts []string
	var sources []string
	tone := c.randomTone()

	switch {
	case strings.Contains(lower, "expense") || strings.Contains(lower, "spent"):
		parts = append(parts, c.pickToned(
			expenseAckPhrases, expenseAckCasual, expenseAckWarm, expenseAckDirect, tone))
		if ctx != nil && ctx.WeeklySpend > 0 && ctx.AvgWeeklySpend > 0 {
			pct := (ctx.WeeklySpend / ctx.AvgWeeklySpend) * 100
			if pct < 90 {
				parts = append(parts, fmt.Sprintf(c.pick(spendTrendPhrases), pct))
			} else if pct > 110 {
				parts = append(parts, fmt.Sprintf(c.pick(spendTrendOver), pct))
			}
			sources = append(sources, "expense_tracking")
		}
	case strings.Contains(lower, "journal"):
		parts = append(parts, c.pickToned(
			journalAckPhrases, journalAckCasual, journalAckWarm, journalAckDirect, tone))
		sources = append(sources, "journal")
	case strings.Contains(lower, "habit"):
		parts = append(parts, c.pickToned(
			habitAckPhrases, habitAckCasual, habitAckWarm, habitAckDirect, tone))
		if ctx != nil && ctx.HabitStreak > 0 {
			parts = append(parts, fmt.Sprintf(c.pick(streakPhrases), ctx.HabitStreak))
			sources = append(sources, "habit_tracking")
		}
	case strings.Contains(lower, "reminder"):
		parts = append(parts, c.pick(reminderAckPhrases))
	default:
		parts = append(parts, "Done.")
	}

	return &ComposedResponse{
		Text:    strings.Join(parts, " "),
		Sources: sources,
		Type:    RespAcknowledge,
	}
}

// -----------------------------------------------------------------------
// Personal Composer
// -----------------------------------------------------------------------

func (c *Composer) composePersonal(query string, ctx *ComposeContext) *ComposedResponse {
	var parts []string
	var sources []string

	if c.Causal != nil {
		answer := c.Causal.AnswerWhy(query)
		if answer != "" {
			parts = append(parts, answer)
			sources = append(sources, "causal_analysis")
		}
	}

	if c.Graph != nil && c.Graph.NodeCount() > 0 {
		facts, edgeSources := c.gatherFacts(query)
		if len(facts) > 0 {
			parts = append(parts, c.realizeFacts(facts))
			sources = append(sources, edgeSources...)
		}
	}

	if len(parts) == 0 {
		return nil
	}

	return &ComposedResponse{
		Text:    strings.Join(parts, " "),
		Sources: uniqueStrings(sources),
		Type:    RespPersonal,
	}
}

// -----------------------------------------------------------------------
// Reflection Composer
// -----------------------------------------------------------------------

func (c *Composer) composeReflection(query string, ctx *ComposeContext) *ComposedResponse {
	var parts []string
	var sources []string
	lower := strings.ToLower(query)
	tone := c.randomTone()

	isOverview := strings.Contains(lower, "how am i") ||
		strings.Contains(lower, "how's my") ||
		strings.Contains(lower, "overview") ||
		strings.Contains(lower, "overall")

	if isOverview && ctx != nil {
		// Collect reflection segments then shuffle their order
		type segment struct {
			text   string
			source string
		}
		var segments []segment

		if ctx.WeeklySpend > 0 {
			if ctx.AvgWeeklySpend > 0 {
				diff := ctx.WeeklySpend - ctx.AvgWeeklySpend
				if diff < 0 {
					segments = append(segments, segment{
						text:   fmt.Sprintf(c.pick(spendSummaryUnder), ctx.WeeklySpend, -diff),
						source: "expense_tracking",
					})
				} else {
					segments = append(segments, segment{
						text:   fmt.Sprintf(c.pick(spendSummaryOver), ctx.WeeklySpend, diff),
						source: "expense_tracking",
					})
				}
			} else {
				segments = append(segments, segment{
					text:   fmt.Sprintf("You've spent €%.0f this week.", ctx.WeeklySpend),
					source: "expense_tracking",
				})
			}
		}

		if ctx.RecentMood > 0 {
			moodWord := "neutral"
			switch {
			case ctx.RecentMood >= 4.0:
				moodWord = c.pick(moodWordsGreat)
			case ctx.RecentMood >= 3.0:
				moodWord = c.pick(moodWordsOkay)
			case ctx.RecentMood >= 2.0:
				moodWord = c.pick(moodWordsLow)
			default:
				moodWord = c.pick(moodWordsTough)
			}
			segments = append(segments, segment{
				text: fmt.Sprintf(
					c.pickToned(moodSummaryPhrases, moodSummaryCasual, moodSummaryWarm, moodSummaryDirect, tone),
					moodWord, ctx.RecentMood),
				source: "mood_tracking",
			})
		}

		if ctx.HabitStreak > 0 {
			segments = append(segments, segment{
				text:   fmt.Sprintf(c.pick(habitSummaryPhrases), ctx.HabitStreak),
				source: "habit_tracking",
			})
		}

		if ctx.JournalDays > 0 {
			if ctx.JournalDays == 1 {
				segments = append(segments, segment{
					text: c.pick(journalRecentPhrases),
					source: "journal",
				})
			} else if ctx.JournalDays > 3 {
				segments = append(segments, segment{
					text:   fmt.Sprintf(c.pick(journalGapPhrases), ctx.JournalDays),
					source: "journal",
				})
			}
		}

		// Shuffle segment order for uniqueness
		c.rng.Shuffle(len(segments), func(i, j int) {
			segments[i], segments[j] = segments[j], segments[i]
		})
		for _, seg := range segments {
			parts = append(parts, seg.text)
			sources = append(sources, seg.source)
		}
	}

	if c.Causal != nil {
		correlations := c.Causal.FindCorrelations()
		for _, corr := range correlations {
			if corr.Correlation > 0.5 || corr.Correlation < -0.5 {
				parts = append(parts, fmt.Sprintf(
					c.pick(correlationPhrases), corr.StreamA, corr.StreamB))
				sources = append(sources, "causal_analysis")
				break
			}
		}
	}

	if len(parts) == 0 {
		return nil
	}

	text := c.combineWithFlow(parts)
	return &ComposedResponse{
		Text:    text,
		Sources: uniqueStrings(sources),
		Type:    RespReflect,
	}
}

// -----------------------------------------------------------------------
// Briefing Composer
// -----------------------------------------------------------------------

func (c *Composer) composeBriefing(ctx *ComposeContext) *ComposedResponse {
	var parts []string
	var sources []string
	tone := c.randomTone()

	hour := time.Now().Hour()
	if hour < 12 {
		if ctx != nil && ctx.UserName != "" && looksLikeProperName(ctx.UserName) {
			parts = append(parts, fmt.Sprintf(c.pick(morningBriefingOpeners), ctx.UserName))
		} else {
			parts = append(parts, c.pick(briefingOpenersGeneric))
		}
	} else {
		parts = append(parts, c.pickToned(
			briefingOpenersGeneric, briefingOpenersCasual, briefingOpenersWarm, briefingOpenersDirect, tone))
	}

	if ctx != nil {
		// Build briefing items then optionally shuffle
		type briefItem struct {
			text   string
			source string
		}
		var items []briefItem

		if ctx.WeeklySpend > 0 {
			items = append(items, briefItem{
				text:   fmt.Sprintf(c.pick(briefSpendPhrases), ctx.WeeklySpend),
				source: "expense_tracking",
			})
		}
		if ctx.HabitStreak > 0 {
			items = append(items, briefItem{
				text:   fmt.Sprintf(c.pick(briefHabitPhrases), ctx.HabitStreak),
				source: "habit_tracking",
			})
		}
		if ctx.RecentMood > 0 {
			items = append(items, briefItem{
				text:   fmt.Sprintf(c.pick(briefMoodPhrases), ctx.RecentMood),
				source: "mood_tracking",
			})
		}
		if ctx.JournalDays > 2 {
			items = append(items, briefItem{
				text:   fmt.Sprintf(c.pick(journalGapPhrases), ctx.JournalDays),
				source: "journal",
			})
		}

		// Occasionally shuffle briefing order
		if c.rng.Float64() < 0.4 {
			c.rng.Shuffle(len(items), func(i, j int) {
				items[i], items[j] = items[j], items[i]
			})
		}

		for _, item := range items {
			parts = append(parts, item.text)
			sources = append(sources, item.source)
		}
	}

	if len(parts) <= 1 {
		parts = append(parts, "I don't have much data yet. Use me more and I'll build a richer picture.")
	}

	return &ComposedResponse{
		Text:    strings.Join(parts, "\n\n"),
		Sources: sources,
		Type:    RespBriefing,
	}
}

// -----------------------------------------------------------------------
// Explain Composer
// -----------------------------------------------------------------------

func (c *Composer) composeExplain(query string) *ComposedResponse {
	return c.composeExplainWithContext(query, nil)
}

func (c *Composer) composeExplainWithContext(query string, ctx *ComposeContext) *ComposedResponse {
	if c.Graph == nil {
		return nil
	}

	facts, sources := c.gatherFacts(query)

	// Determine topic from facts or query
	queryTopic := c.extractTopic(query)
	topic := queryTopic
	if len(facts) > 0 {
		topic = facts[0].Subject
	}

	// Relevance guard: if the facts are about a completely different topic
	// than what was asked, don't use them. This prevents "meaning of life"
	// returning facts about "photosynthesis" via loose keyword matching.
	if queryTopic != "" && topic != "" && len(facts) > 0 {
		qtLower := strings.ToLower(queryTopic)
		ftLower := strings.ToLower(topic)
		if qtLower != ftLower &&
			!strings.Contains(qtLower, ftLower) &&
			!strings.Contains(ftLower, qtLower) {
			// Facts are about a different topic — discard them
			facts = nil
			topic = queryTopic
		}
	}

	// Build a clean explanation from description + structured facts.
	var parts []string

	// If the inner council produced a synthesis, use it as a framing statement.
	if ctx != nil && ctx.CouncilResult != nil && ctx.CouncilResult.Synthesis != "" {
		parts = append(parts, ctx.CouncilResult.Synthesis)
		sources = append(sources, "inner_council")
	}

	if desc := c.Graph.LookupDescription(topic); len(desc) >= 40 {
		parts = append(parts, desc)
	}
	if len(facts) > 0 {
		factText := c.structuredRealization(facts)
		if factText != "" {
			parts = append(parts, factText)
		}
	}

	// If knowledge graph data is thin (0-1 facts), try the discourse corpus.
	if len(facts) <= 1 && c.DiscourseCorpus != nil && topic != "" {
		composed := c.DiscourseCorpus.ComposeResponse(topic, "explain")
		if composed != "" {
			parts = append(parts, composed)
			sources = append(sources, "discourse_corpus")
		}
	}

	if len(parts) == 0 {
		return nil
	}
	return &ComposedResponse{
		Text:    strings.Join(parts, "\n\n"),
		Sources: uniqueStrings(sources),
		Type:    RespExplain,
	}
}

// -----------------------------------------------------------------------
// Follow-Up Composer — expands on a previous topic based on follow-up type.
// Called when the user says "tell me more", "why?", "examples?", etc.
// -----------------------------------------------------------------------

// ComposeFollowUp generates a response that expands on a previously discussed
// topic. The followUpType controls what kind of expansion:
//   - "more":    additional facts beyond what was already shown
//   - "why":     causal explanations (uses DFExplainsWhy, DFConsequence)
//   - "example": concrete examples (uses DFGivesExample)
//   - "compare": comparative information (uses DFCompares, DFEvaluates)
//   - "deeper":  in-depth explanation (uses thinking engine path)
func (c *Composer) ComposeFollowUp(originalTopic string, followUpType string) *ComposedResponse {
	if originalTopic == "" {
		return nil
	}

	var parts []string
	var sources []string

	// Lead phrase based on follow-up type
	leadPhrases := map[string][]string{
		"more": {
			"Here's more about " + originalTopic + ".",
			"There's more to " + originalTopic + ".",
			"Expanding on " + originalTopic + ":",
			"Additionally, regarding " + originalTopic + ":",
		},
		"why": {
			"Here's why:",
			"The reason behind this:",
			"To understand why:",
		},
		"example": {
			"Here's an example:",
			"For instance:",
			"To illustrate:",
		},
		"compare": {
			"For comparison:",
			"Looking at how it compares:",
		},
		"deeper": {
			"Let me explain " + originalTopic + " in more depth.",
			"Going deeper into " + originalTopic + ":",
			"To elaborate on " + originalTopic + ":",
		},
	}

	phrases, ok := leadPhrases[followUpType]
	if !ok {
		phrases = leadPhrases["more"]
	}
	lead := c.pick(phrases)

	// Strategy 1: Discourse corpus — retrieve sentences by discourse function
	if c.DiscourseCorpus != nil {
		var queryType string
		switch followUpType {
		case "why":
			queryType = "why"
		case "example":
			queryType = "example"
		case "compare":
			queryType = "compare"
		case "deeper":
			queryType = "explain"
		default:
			queryType = "explain"
		}
		composed := c.DiscourseCorpus.ComposeResponse(originalTopic, queryType)
		if composed != "" {
			parts = append(parts, composed)
			sources = append(sources, "discourse_corpus")
		}
	}

	// Strategy 2: Knowledge graph — gather more facts with a broader search.
	// For "more" follow-ups, look for facts we haven't surfaced yet.
	if c.Graph != nil {
		facts, graphSources := c.gatherFollowUpFacts(originalTopic)
		if len(facts) > 0 {
			factText := c.structuredRealization(facts)
			if factText != "" {
				parts = append(parts, factText)
				sources = append(sources, graphSources...)
			}
		}

		// For "why" follow-ups, also try causal reasoning
		if followUpType == "why" && c.Causal != nil {
			if answer := c.Causal.AnswerWhy("why " + originalTopic); answer != "" {
				parts = append(parts, answer)
				sources = append(sources, "causal_engine")
			}
		}

		// Description as a fallback for "deeper" if we haven't found much
		if followUpType == "deeper" && len(parts) == 0 {
			if desc := c.Graph.LookupDescription(originalTopic); len(desc) >= 40 {
				parts = append(parts, desc)
				sources = append(sources, "knowledge_graph")
			}
		}
	}

	if len(parts) == 0 {
		return nil
	}

	text := lead + " " + strings.Join(parts, "\n\n")
	return &ComposedResponse{
		Text:    strings.TrimSpace(text),
		Sources: uniqueStrings(sources),
		Type:    RespConversational,
	}
}

// gatherFollowUpFacts retrieves facts about a topic with a higher limit
// than the normal gatherFacts, and tries to surface facts that weren't
// shown in the initial response.
func (c *Composer) gatherFollowUpFacts(topic string) ([]edgeFact, []string) {
	if c.Graph == nil {
		return nil, nil
	}

	var facts []edgeFact
	var sources []string
	seen := make(map[string]bool)

	c.Graph.mu.RLock()
	defer c.Graph.mu.RUnlock()

	// Look up the topic node
	id := nodeID(strings.ToLower(topic))
	node := c.Graph.nodes[id]
	if node == nil {
		if ids, ok := c.Graph.byLabel[strings.ToLower(topic)]; ok && len(ids) > 0 {
			id = ids[0]
			node = c.Graph.nodes[id]
		}
	}
	if node == nil {
		return nil, nil
	}

	// Gather outgoing edges — higher limit (10 instead of typical 4-5)
	for _, edge := range c.Graph.outEdges[id] {
		if edge.Relation == RelDescribedAs {
			continue
		}
		to := c.Graph.nodes[edge.To]
		if to == nil {
			continue
		}
		key := node.Label + "|" + string(edge.Relation) + "|" + to.Label
		if seen[key] {
			continue
		}
		seen[key] = true
		facts = append(facts, edgeFact{
			Subject:  node.Label,
			Relation: edge.Relation,
			Object:   to.Label,
			Inferred: edge.Inferred,
		})
		if edge.Inferred {
			sources = append(sources, "inference")
		} else {
			sources = append(sources, "knowledge_graph")
		}
	}

	// Also gather incoming edges for richer follow-up context
	for _, edge := range c.Graph.inEdges[id] {
		if edge.Relation == RelDescribedAs {
			continue
		}
		from := c.Graph.nodes[edge.From]
		if from == nil {
			continue
		}
		key := from.Label + "|" + string(edge.Relation) + "|" + node.Label
		if seen[key] {
			continue
		}
		seen[key] = true
		facts = append(facts, edgeFact{
			Subject:  from.Label,
			Relation: edge.Relation,
			Object:   node.Label,
			Inferred: edge.Inferred,
		})
		sources = append(sources, "knowledge_graph")
	}

	// Cap at 10 facts for a follow-up (generous but not overwhelming)
	if len(facts) > 10 {
		facts = facts[:10]
	}

	return facts, sources
}

// -----------------------------------------------------------------------
// Uncertain Composer
// -----------------------------------------------------------------------

func (c *Composer) composeUncertain(query string) *ComposedResponse {
	// Try to bridge to something we DO know
	if c.Graph != nil && c.Graph.NodeCount() > 0 {
		bridge := c.findBridgeTopic(query)
		if bridge != "" {
			return &ComposedResponse{
				Text:    c.pick(uncertainBridgePhrases) + " " + bridge,
				Sources: []string{"knowledge_graph"},
				Type:    RespUncertain,
			}
		}
	}
	return &ComposedResponse{
		Text: c.pick(uncertainPhrases),
		Type: RespUncertain,
	}
}

// -----------------------------------------------------------------------
// Conversational Composer — the 4-stage pipeline:
//   Acknowledge → Bridge → Contribute → Engage
// Handles open-ended dialogue without any LLM.
// -----------------------------------------------------------------------

func (c *Composer) composeConversational(query string, ctx *ComposeContext) *ComposedResponse {
	lowerQuery := strings.ToLower(query)
	isContinuation := isContinuationRequest(lowerQuery)

	// Continuation requests: instead of skipping knowledge, expand on the
	// previous topic using conversation history.
	if isContinuation && len(c.history) > 0 {
		last := c.history[len(c.history)-1]
		prevTopic := ""
		if len(last.Topics) > 0 {
			prevTopic = last.Topics[0]
		}
		if prevTopic != "" {
			followUpType := classifyFollowUpType(lowerQuery)
			resp := c.ComposeFollowUp(prevTopic, followUpType)
			if resp != nil && resp.Text != "" {
				return resp
			}
		}
	}

	// Personal statements ("I will run tomorrow", "we should try that")
	// should NEVER do a knowledge lookup — they need a conversational
	// response, not a fact dump about "run" or "tell".
	skipKnowledge := isPersonalStatement(lowerQuery) || isContinuation

	// If we have substantial knowledge about the query topic, USE it.
	// Use clean fact sentences — no decorative prose, no hooks/closers.
	if !skipKnowledge && c.Graph != nil {
		queryTopic := c.extractTopic(query)
		facts, sources := c.gatherFacts(query)

		// Relevance guard: verify that gathered facts are actually about
		// the queried topic, not just a loose keyword match.
		if len(facts) > 0 && queryTopic != "" {
			qtLower := strings.ToLower(queryTopic)
			ftLower := strings.ToLower(facts[0].Subject)
			if qtLower != ftLower &&
				!strings.Contains(qtLower, ftLower) &&
				!strings.Contains(ftLower, qtLower) {
				facts = nil // discard unrelated facts
			}
		}

		if len(facts) >= 2 {
			// Lead with description if available, then structured facts
			topic := facts[0].Subject
			var parts []string
			if desc := c.Graph.LookupDescription(topic); len(desc) >= 40 {
				parts = append(parts, desc)
			}
			factText := c.structuredRealization(facts)
			if factText != "" {
				parts = append(parts, factText)
			}
			if len(parts) > 0 {
				return &ComposedResponse{
					Text:    strings.Join(parts, "\n\n"),
					Sources: uniqueStrings(sources),
					Type:    RespConversational,
				}
			}
		} else if len(facts) == 1 {
			// Single fact — compose a brief sentence
			sent := c.edgeToSentence(facts[0].Subject, facts[0].Relation, facts[0].Object, facts[0].Inferred)
			followUp := c.pick([]string{
				"Want to know more?",
				"I can tell you more about that.",
				"There's more to it if you're curious.",
			})
			return &ComposedResponse{
				Text:    sent + " " + followUp,
				Sources: uniqueStrings(sources),
				Type:    RespConversational,
			}
		}
	}

	// Detect sentiment for appropriate response framing.
	// If the inner council recommended a tone, use it instead of random.
	var tone Tone
	if ctx != nil && ctx.CouncilResult != nil && ctx.CouncilResult.ResponseTone != "" {
		tone = councilToneToComposerTone(ctx.CouncilResult.ResponseTone)
	} else {
		tone = c.randomTone()
	}
	sentiment := c.detectSentiment(query)

	// Positive news ("I got promoted!", "I just finished!") → celebrate.
	// Don't deflect with "what got you thinking about promoted?"
	if sentiment == SentimentPositive || sentiment == SentimentExcited {
		resp := c.composeEmpathetic(query, ctx)
		if resp != nil && resp.Text != "" {
			return resp
		}
	}

	// Personal statements / follow-ups / no knowledge → conversational pipeline
	var parts []string
	var sources []string

	// Stage 1: ACKNOWLEDGE — show you heard them
	ack := c.acknowledgeInput(query, sentiment, tone)
	if ack != "" {
		parts = append(parts, ack)
	}

	// Stage 2: BRIDGE — connect to something relevant
	// Skip for personal statements — knowledge bridges produce garbage
	if !skipKnowledge {
		bridge := c.bridgeToKnowledge(query, ctx)
		if bridge != "" {
			parts = append(parts, bridge)
			sources = append(sources, "knowledge_graph")
		}
	}

	// Stage 3: CONTRIBUTE — add value from what we know
	contribution := c.contributeInsight(query, ctx)
	if contribution != "" {
		parts = append(parts, contribution)
		sources = append(sources, "insight")
	}

	// Stage 3b: If the council produced a synthesis, use it as substantive content.
	if ctx != nil && ctx.CouncilResult != nil && ctx.CouncilResult.Synthesis != "" {
		parts = append(parts, ctx.CouncilResult.Synthesis)
		sources = append(sources, "inner_council")
	}

	// Stage 4: If response is still thin (just an acknowledgment with no
	// substance), try the council's clarifying question, then discourse
	// corpus, then knowledge graph — instead of deflecting.
	if len(parts) <= 1 && !skipKnowledge {
		// If the council thinks we should ask a clarifying question, do that.
		if ctx != nil && ctx.CouncilResult != nil && ctx.CouncilResult.ShouldAsk && ctx.CouncilResult.AskWhat != "" {
			parts = append(parts, ctx.CouncilResult.AskWhat)
			sources = append(sources, "inner_council")
			return &ComposedResponse{
				Text:    strings.Join(parts, " "),
				Sources: sources,
				Type:    RespConversational,
			}
		}
	}
	if len(parts) <= 1 && !skipKnowledge {
		topic := c.extractTopic(query)
		if topic != "" && len(topic) > 2 {
			// Try discourse corpus first — retrieves real sentences about the topic.
			if c.DiscourseCorpus != nil {
				queryType := classifyQueryType(query)
				composed := c.DiscourseCorpus.ComposeResponse(topic, queryType)
				if composed != "" {
					parts = append(parts, composed)
					sources = append(sources, "discourse_corpus")
				}
			}
			// If discourse corpus didn't help, try knowledge graph.
			if len(parts) <= 1 {
				if desc := c.Graph.LookupDescription(topic); len(desc) >= 40 {
					parts = append(parts, desc)
					sources = append(sources, "knowledge")
				} else if facts := c.Graph.LookupFacts(topic, 2); len(facts) > 0 {
					parts = append(parts, strings.Join(facts, " "))
					sources = append(sources, "knowledge")
				}
			}
		}
	}

	// If response is empty or just a thin acknowledgment, give an honest
	// "I don't know" instead of a generic deflection question.
	if len(parts) == 0 {
		topic := c.extractTopic(query)
		if topic != "" && len(topic) > 2 {
			return &ComposedResponse{
				Text:    fmt.Sprintf("I don't have much information about %s in my knowledge base yet. You can teach me, or I can search the web if available.", topic),
				Sources: []string{"honest_fallback"},
				Type:    RespConversational,
			}
		}
		return nil
	}
	if len(parts) == 1 {
		topic := c.extractTopic(query)
		if topic != "" {
			followUps := []string{
				fmt.Sprintf("Want me to dig deeper into %s?", topic),
				fmt.Sprintf("I can tell you more about %s if you're curious.", topic),
				fmt.Sprintf("There's more to %s — just ask.", topic),
				fmt.Sprintf("Shall I elaborate on %s?", topic),
				fmt.Sprintf("Ask me anything else about %s.", topic),
			}
			parts = append(parts, followUps[len(topic)%len(followUps)])
		}
	}

	return &ComposedResponse{
		Text:    strings.Join(parts, " "),
		Sources: sources,
		Type:    RespConversational,
	}
}

// acknowledgeInput produces an acknowledgment based on sentiment and content.
func (c *Composer) acknowledgeInput(query string, sentiment Sentiment, tone Tone) string {
	lower := strings.ToLower(query)

	// Detect specific content to acknowledge
	if strings.Contains(lower, "just wanted to talk") || strings.Contains(lower, "let's talk") ||
		strings.Contains(lower, "lets talk") || strings.Contains(lower, "talk to me") {
		return c.pickToned(talkAckNeutral, talkAckCasual, talkAckWarm, talkAckDirect, tone)
	}

	if strings.Contains(lower, "bored") || strings.Contains(lower, "nothing to do") {
		return c.pick(boredAckPhrases)
	}

	switch sentiment {
	case SentimentPositive, SentimentExcited:
		return c.pickToned(positiveAckNeutral, positiveAckCasual, positiveAckWarm, positiveAckDirect, tone)
	case SentimentNegative, SentimentSad:
		return c.pickToned(negativeAckNeutral, negativeAckCasual, negativeAckWarm, negativeAckDirect, tone)
	case SentimentAngry:
		return c.pick(angryAckPhrases)
	case SentimentCurious:
		return c.pick(curiousAckPhrases)
	default:
		return c.pickToned(neutralAckNeutral, neutralAckCasual, neutralAckWarm, neutralAckDirect, tone)
	}
}

// bridgeToKnowledge finds the nearest relevant knowledge to bridge to.
func (c *Composer) bridgeToKnowledge(query string, ctx *ComposeContext) string {
	if c.Graph == nil || c.Graph.NodeCount() == 0 {
		return ""
	}

	// Extract topics from query and try to find graph connections
	facts, _ := c.gatherFacts(query)
	if len(facts) > 0 {
		// We found something relevant — compose a bridge
		fact := facts[c.rng.Intn(len(facts))]
		return c.pick(bridgePhrases) + " " + lowerFirst(c.edgeToSentence(fact.Subject, fact.Relation, fact.Object, fact.Inferred))
	}

	// Try to bridge from conversation history — but ONLY if the current
	// query is actually related to the previous topic. Otherwise we get
	// confusing responses like "Going back to tell — anything else there?"
	if len(c.history) > 0 {
		lastTurn := c.history[len(c.history)-1]
		if len(lastTurn.Topics) > 0 {
			topic := lastTurn.Topics[0]
			// Only bridge if the query contains or overlaps with the topic
			queryWords := strings.Fields(strings.ToLower(query))
			topicLower := strings.ToLower(topic)
			related := false
			for _, w := range queryWords {
				if len(w) > 3 && strings.Contains(topicLower, w) {
					related = true
					break
				}
				if strings.Contains(strings.ToLower(query), topicLower) {
					related = true
					break
				}
			}
			if related {
				return fmt.Sprintf(c.pick(historyBridgePhrases), topic)
			}
		}
	}

	return ""
}

// contributeInsight adds value from context or patterns.
func (c *Composer) contributeInsight(query string, ctx *ComposeContext) string {
	// Check causal patterns
	if c.Causal != nil {
		answer := c.Causal.AnswerWhy(query)
		if answer != "" {
			return answer
		}
	}

	// Offer contextual insights based on time/mood/habits
	if ctx != nil {
		var insights []string
		hour := time.Now().Hour()
		if hour >= 22 || hour < 5 {
			insights = append(insights, c.pick(lateNightInsights))
		}
		if ctx.RecentMood > 0 && ctx.RecentMood >= 4.0 {
			insights = append(insights, c.pick(goodMoodInsights))
		}
		if ctx.HabitStreak > 7 {
			insights = append(insights, fmt.Sprintf(c.pick(streakInsights), ctx.HabitStreak))
		}
		if len(insights) > 0 {
			return insights[c.rng.Intn(len(insights))]
		}
	}

	return ""
}

// engageFollowUp generates a contextual follow-up to keep conversation alive.
func (c *Composer) engageFollowUp(query string, sentiment Sentiment, ctx *ComposeContext) string {
	lower := strings.ToLower(query)

	// If they're sharing something personal, ask about it
	if strings.Contains(lower, "i feel") || strings.Contains(lower, "i think") ||
		strings.Contains(lower, "i've been") || strings.Contains(lower, "ive been") {
		return c.pick(personalFollowUps)
	}

	// If they mentioned a substantive topic (not just function words), ask about it.
	// Only use keywords that are actual content words, not artifacts like "tell", "about".
	trivialWords := map[string]bool{
		"tell": true, "about": true, "know": true, "think": true,
		"what": true, "more": true, "also": true, "really": true,
		"just": true, "very": true, "much": true, "some": true,
		"good": true, "well": true, "like": true, "want": true,
		"need": true, "have": true, "been": true, "i'm": true,
		"going": true, "doing": true, "having": true, "getting": true,
		"name": true, "hello": true, "thanks": true, "thank": true,
		"please": true, "yeah": true, "okay": true, "sure": true,
		"favorite": true, "love": true, "enjoy": true, "had": true,
		"i've": true, "let's": true, "right": true, "make": true,
	}
	keywords := extractKeywords(lower)
	for _, kw := range keywords {
		if len(kw) > 3 && !trivialWords[kw] {
			return fmt.Sprintf(c.pick(topicFollowUps), kw)
		}
	}

	// Generic engagement
	return c.pick(genericEngagement)
}

// -----------------------------------------------------------------------
// Empathetic Composer — responds to emotional content
// -----------------------------------------------------------------------

func (c *Composer) composeEmpathetic(query string, ctx *ComposeContext) *ComposedResponse {
	sentiment := c.detectSentiment(query)
	tone := ToneWarm // Always warm for empathy
	var parts []string

	// Try to construct a specific response from the user's actual words
	// before falling back to generic phrase pools.
	specifics := extractSpecifics(query)
	if len(specifics) > 0 {
		specific := c.composeSpecificEmpathy(specifics, sentiment)
		if specific != "" {
			parts = append(parts, specific)
		}
	}

	// Use subtext for richer empathy when available
	if len(parts) == 0 && ctx != nil && ctx.Subtext != nil {
		sub := ctx.Subtext
		switch sub.ImpliedNeed {
		case NeedCelebration:
			parts = append(parts, c.pick(empatheticHappyPhrases))
		case NeedVenting:
			parts = append(parts, c.pick([]string{
				"I hear you. That sounds genuinely frustrating.",
				"That's rough. You don't have to have it figured out right now.",
				"I can tell this is weighing on you.",
				"That would frustrate anyone. Give yourself some credit for sticking with it.",
			}))
		case NeedReassurance:
			parts = append(parts, c.pick([]string{
				"You're doing better than you think.",
				"The fact that you care enough to worry about this says something good.",
				"It's okay to not have all the answers yet.",
				"You've handled harder things than this before.",
			}))
		case NeedValidation:
			parts = append(parts, c.pick([]string{
				"That makes sense to me.",
				"I think your instinct is right here.",
				"Trust yourself on this one.",
			}))
		default:
			// Fall through to sentiment-based
		}
	}

	// Sentiment-based fallback (or if subtext didn't produce anything)
	if len(parts) == 0 {
		switch sentiment {
		case SentimentSad, SentimentNegative:
			parts = append(parts, c.pickToned(empatheticSadNeutral, empatheticSadCasual, empatheticSadWarm, empatheticSadDirect, tone))
			if ctx != nil && ctx.RecentMood > 0 && ctx.RecentMood < 3.0 {
				parts = append(parts, c.pick(moodAwareSadPhrases))
			}
		case SentimentAngry:
			parts = append(parts, c.pick(empatheticAngryPhrases))
		case SentimentExcited, SentimentPositive:
			parts = append(parts, c.pick(empatheticHappyPhrases))
		default:
			parts = append(parts, c.pick(empatheticNeutralPhrases))
		}
	}

	// Offer something actionable — but avoid near-duplicate of what we just said
	action := c.pick(empatheticActions)
	if !phraseTooSimilar(parts[0], action) {
		parts = append(parts, action)
	}

	return &ComposedResponse{
		Text:    strings.Join(parts, " "),
		Sources: []string{"empathy"},
		Type:    RespEmpathetic,
	}
}

// phraseTooSimilar checks if two phrases share too many words (prevents
// "I'm here for you. Whatever you need — I'm here." duplication).
func phraseTooSimilar(a, b string) bool {
	wordsA := strings.Fields(strings.ToLower(a))
	wordsB := strings.Fields(strings.ToLower(b))
	if len(wordsA) == 0 || len(wordsB) == 0 {
		return false
	}
	setA := make(map[string]bool, len(wordsA))
	for _, w := range wordsA {
		setA[w] = true
	}
	overlap := 0
	for _, w := range wordsB {
		if setA[w] {
			overlap++
		}
	}
	// If more than 40% of words overlap, it's too similar
	shorter := len(wordsA)
	if len(wordsB) < shorter {
		shorter = len(wordsB)
	}
	return float64(overlap)/float64(shorter) > 0.4
}

// -----------------------------------------------------------------------
// Opinion Composer — synthesizes viewpoints from graph knowledge
// -----------------------------------------------------------------------

func (c *Composer) composeOpinion(query string, ctx *ComposeContext) *ComposedResponse {
	var parts []string
	var sources []string

	// If the council thinks we should ask a clarifying question, do that instead.
	if ctx != nil && ctx.CouncilResult != nil && ctx.CouncilResult.ShouldAsk && ctx.CouncilResult.AskWhat != "" {
		return &ComposedResponse{
			Text:    ctx.CouncilResult.AskWhat,
			Sources: []string{"inner_council"},
			Type:    RespOpinion,
		}
	}

	// Choose opener based on council's recommended tone.
	if ctx != nil && ctx.CouncilResult != nil && ctx.CouncilResult.ResponseTone != "" {
		switch ctx.CouncilResult.ResponseTone {
		case "cautious":
			parts = append(parts, c.pick([]string{
				"I'm not entirely sure, but...",
				"I've been going back and forth on this, but...",
				"This is tricky, but here's where I land —",
			}))
		case "enthusiastic":
			parts = append(parts, c.pick([]string{
				"I actually have a strong take on this —",
				"Oh, I've thought about this a lot —",
				"This is one I feel strongly about —",
			}))
		case "direct":
			parts = append(parts, c.pick([]string{
				"Here's how I see it.",
				"Straightforwardly —",
				"My take is simple.",
			}))
		case "empathetic":
			parts = append(parts, c.pick([]string{
				"I can see why this matters to you.",
				"This is one of those questions where the answer depends on who you are.",
				"There's no wrong way to feel about this, but here's my read —",
			}))
		default:
			parts = append(parts, c.pick(opinionOpeners))
		}
	} else {
		// Default: start with a generic thinking phrase
		parts = append(parts, c.pick(opinionOpeners))
	}

	// If the opinion engine has formed a confident opinion, use its summary
	// as the core statement rather than synthesizing from raw facts.
	opinionUsed := false
	if ctx != nil && ctx.Opinion != nil && ctx.Opinion.Confidence >= 0.3 && ctx.Opinion.Summary != "" {
		parts = append(parts, ctx.Opinion.Summary)
		sources = append(sources, "opinion_engine")
		opinionUsed = true
	}

	// If we didn't get an opinion summary, try graph facts.
	if !opinionUsed && c.Graph != nil && c.Graph.NodeCount() > 0 {
		facts, factSources := c.gatherFacts(query)
		if len(facts) > 0 {
			parts = append(parts, c.synthesizeOpinion(facts))
			sources = append(sources, factSources...)
		}
	}

	// Add the council's synthesis as additional perspective,
	// but filter out internal system language that shouldn't be user-facing.
	if ctx != nil && ctx.CouncilResult != nil && ctx.CouncilResult.Synthesis != "" {
		synth := ctx.CouncilResult.Synthesis
		// Skip if the synthesis contains internal assessment language.
		skipPhrases := []string{"no facts found", "cannot provide", "knowledge gap", "not well covered"}
		shouldSkip := false
		synthLower := strings.ToLower(synth)
		for _, sp := range skipPhrases {
			if strings.Contains(synthLower, sp) {
				shouldSkip = true
				break
			}
		}
		if !shouldSkip {
			parts = append(parts, synth)
			sources = append(sources, "inner_council")
		}
	}

	if len(parts) <= 1 {
		// No facts or opinion to draw on — construct a reasoned response
		// from the question structure itself rather than giving a generic dodge.
		topic := c.extractTopic(query)
		lower := strings.ToLower(query)

		// Detect specific opinion patterns and give structured answers.
		if topic != "" && (strings.Contains(lower, "replace") || strings.Contains(lower, "take over")) {
			parts = append(parts, fmt.Sprintf(
				"I think %s will transform many fields, but replacement is rarely total. "+
					"History shows new tools augment human capability more often than they replace it entirely. "+
					"The real question is which specific tasks change, and how people adapt.",
				topic))
		} else if topic != "" && (strings.Contains(lower, "good") || strings.Contains(lower, "bad") || strings.Contains(lower, "dangerous")) {
			parts = append(parts, fmt.Sprintf(
				"Like most powerful things, %s has both potential and risk. "+
					"The outcome depends on how it's developed, regulated, and used. "+
					"I lean toward cautious optimism — the benefits are real, but so are the risks.",
				topic))
		} else if topic != "" && (strings.Contains(lower, "future") || strings.Contains(lower, "will")) {
			parts = append(parts, fmt.Sprintf(
				"Predicting the future of %s is hard, but the trajectory is clear: "+
					"the technology is advancing fast, adoption is accelerating, and "+
					"the consequences — both good and bad — will be significant.",
				topic))
		} else if topic != "" {
			parts = append(parts, fmt.Sprintf(c.pick(genericOpinions), topic))
		} else {
			parts = append(parts, c.pick(thoughtfulGeneric))
		}
	}

	// End with engagement
	parts = append(parts, c.pick(opinionClosers))

	return &ComposedResponse{
		Text:    strings.Join(parts, " "),
		Sources: sources,
		Type:    RespOpinion,
	}
}

// synthesizeOpinion builds an opinion from graph facts.
func (c *Composer) synthesizeOpinion(facts []edgeFact) string {
	if len(facts) == 0 {
		return ""
	}

	// 30% chance: use generative engine for a creative take
	if c.Generative != nil && c.rng.Float64() < 0.3 {
		topic := facts[0].Subject
		text := c.Generative.ComposeCreativeText(topic, facts)
		if text != "" {
			return text
		}
	}

	subject := facts[0].Subject
	var observations []string

	for _, f := range facts {
		switch f.Relation {
		case RelIsA:
			observations = append(observations, fmt.Sprintf(c.pick(opinionIsA), subject, f.Object))
		case RelUsedFor:
			observations = append(observations, fmt.Sprintf(c.pick(opinionUsedFor), subject, f.Object))
		case RelDescribedAs:
			observations = append(observations, fmt.Sprintf(c.pick(opinionDescribed), subject, f.Object))
		default:
			observations = append(observations, lowerFirst(c.edgeToSentence(f.Subject, f.Relation, f.Object, f.Inferred)))
		}
	}

	if len(observations) > 2 {
		observations = observations[:2]
	}
	return strings.Join(observations, " ")
}

// -----------------------------------------------------------------------
// Farewell Composer
// -----------------------------------------------------------------------

func (c *Composer) composeFarewell(ctx *ComposeContext) *ComposedResponse {
	var parts []string

	fare := c.pick(farewellPhrases)
	if ctx != nil && ctx.UserName != "" && looksLikeProperName(ctx.UserName) {
		fare = fmt.Sprintf(c.pick(farewellNamePhrases), ctx.UserName)
	}
	parts = append(parts, fare)

	// Add a contextual parting note
	if ctx != nil && ctx.HabitStreak > 3 {
		parts = append(parts, fmt.Sprintf(c.pick(farewellStreakPhrases), ctx.HabitStreak))
	}

	return &ComposedResponse{
		Text:    strings.Join(parts, " "),
		Sources: []string{"farewell"},
		Type:    RespFarewell,
	}
}

// -----------------------------------------------------------------------
// Thank You Composer
// -----------------------------------------------------------------------

func (c *Composer) composeThankYou(ctx *ComposeContext) *ComposedResponse {
	return &ComposedResponse{
		Text:    c.pick(thankYouResponses),
		Sources: []string{"social"},
		Type:    RespThankYou,
	}
}

// -----------------------------------------------------------------------
// Sentiment Detection — keyword-based, zero ML
// -----------------------------------------------------------------------

func (c *Composer) detectSentiment(query string) Sentiment {
	lower := strings.ToLower(query)

	// Check for strong signals first
	for _, w := range angryWords {
		if strings.Contains(lower, w) {
			return SentimentAngry
		}
	}
	for _, w := range sadWords {
		if strings.Contains(lower, w) {
			return SentimentSad
		}
	}
	for _, w := range excitedWords {
		if strings.Contains(lower, w) {
			return SentimentExcited
		}
	}

	// Count positive vs negative signals
	posCount := 0
	negCount := 0
	for _, w := range positiveWords {
		if strings.Contains(lower, w) {
			posCount++
		}
	}
	for _, w := range negativeWords {
		if strings.Contains(lower, w) {
			negCount++
		}
	}

	// Check for curiosity
	if strings.Contains(lower, "?") || strings.HasPrefix(lower, "what") ||
		strings.HasPrefix(lower, "how") || strings.HasPrefix(lower, "why") ||
		strings.HasPrefix(lower, "can") || strings.HasPrefix(lower, "do you") {
		return SentimentCurious
	}

	if posCount > negCount {
		return SentimentPositive
	}
	if negCount > posCount {
		return SentimentNegative
	}
	return SentimentNeutral
}

// findBridgeTopic finds the closest known topic to bridge from an unknown query.
func (c *Composer) findBridgeTopic(query string) string {
	keywords := extractKeywords(strings.ToLower(query))
	if c.Graph == nil {
		return ""
	}

	c.Graph.mu.RLock()
	defer c.Graph.mu.RUnlock()

	// Try partial label matches
	for _, kw := range keywords {
		if len(kw) < 3 {
			continue
		}
		for label, ids := range c.Graph.byLabel {
			if strings.Contains(label, kw) && len(ids) > 0 {
				node := c.Graph.nodes[ids[0]]
				if node == nil {
					continue
				}
				// Found a nearby topic — get one fact about it
				edges := c.Graph.outEdges[ids[0]]
				if len(edges) > 0 {
					edge := edges[c.rng.Intn(len(edges))]
					to := c.Graph.nodes[edge.To]
					if to != nil {
						return c.edgeToSentenceUnlocked(node.Label, edge.Relation, to.Label, edge.Inferred)
					}
				}
			}
		}
	}
	return ""
}

// edgeToSentenceUnlocked is like edgeToSentence but assumes graph lock is held.
func (c *Composer) edgeToSentenceUnlocked(subject string, rel RelType, object string, inferred bool) string {
	templates, ok := relationTemplates[rel]
	if rel == RelFoundedIn && isPersonReference(subject) {
		templates = []string{"%s was born in %s."}
		ok = true
	}
	if !ok || len(templates) == 0 {
		return ""
	}
	sentence := fmt.Sprintf(c.pick(templates), subject, object)
	if inferred {
		sentence = c.pick(inferredPrefixes) + sentence
	}
	return sentence
}

// -----------------------------------------------------------------------
// Sentence Realization — 6+ templates per relation type
// -----------------------------------------------------------------------

func (c *Composer) edgeToSentence(subject string, rel RelType, object string, inferred bool) string {
	if isFragmentObject(object) {
		return ""
	}

	// Try sentence corpus retrieval first (Layer 2) — real human-written
	// sentences adapted by entity swapping. Highest quality output.
	if c.SentenceCorpus != nil {
		retrieved := c.SentenceCorpus.RetrieveVaried(subject, rel, object)
		if retrieved != "" && isAcceptableSentence(retrieved, subject, object) {
			if inferred {
				retrieved = c.pick(inferredPrefixes) + retrieved
			}
			return retrieved
		}
	}

	// Try GRU neural generation (if model is loaded)
	if c.TextGen != nil {
		neural := c.TextGen.Generate(subject, rel, object, 0.5)
		if isAcceptableSentence(neural, subject, object) {
			if inferred {
				neural = c.pick(inferredPrefixes) + neural
			}
			return neural
		}
	}

	// Try absorbed expression patterns (Layer 3) — patterns learned from
	// reading real text, adapted with slot filling. Produces varied,
	// natural sentences without templates.
	if c.Absorption != nil {
		fn := relationToDiscourseFunc(rel)
		pattern := c.Absorption.Retrieve(fn, "", "")
		if pattern != nil {
			slots := map[string]string{
				"SUBJECT":  subject,
				"OBJECT":   object,
				"CATEGORY": object,
			}
			realized := c.Absorption.Realize(pattern, slots)
			if realized != "" && isAcceptableSentence(realized, subject, object) {
				if inferred {
					realized = c.pick(inferredPrefixes) + realized
				}
				return realized
			}
		}
	}

	templates, ok := relationTemplates[rel]
	if rel == RelFoundedIn && isPersonReference(subject) {
		templates = []string{"%s was born in %s."}
		ok = true
	}
	if !ok || len(templates) == 0 {
		return ""
	}

	// Capitalize subject for sentence-initial position.
	subject = capitalizeFirst(subject)

	sentence := fmt.Sprintf(c.pick(templates), subject, object)

	// Fix subject-verb agreement for plural subjects
	if isLikelyPlural(subject) {
		sentence = fixSubjectVerbAgreement(sentence, subject)
	}

	// Fix article agreement: "a acceleration" → "an acceleration"
	sentence = fixArticleAgreement(sentence)

	// Fix "one of the" + singular: "one of the language" → "one of the languages"
	sentence = fixOneOfThePlural(sentence)

	if inferred {
		sentence = c.pick(inferredPrefixes) + sentence
	}

	return sentence
}

// relationToDiscourseFunc maps knowledge graph relation types to discourse
// functions for absorption pattern retrieval.
func relationToDiscourseFunc(rel RelType) DiscourseFunc {
	switch rel {
	case RelIsA:
		return DFDefines
	case RelFoundedIn, RelLocatedIn:
		return DFContext
	case RelCreatedBy, RelFoundedBy:
		return DFContext
	case RelDescribedAs:
		return DFEvaluates
	case RelHas, RelPartOf:
		return DFDescribes
	case RelUsedFor:
		return DFDescribes
	case RelCauses:
		return DFConsequence
	case RelSimilarTo, RelContradicts:
		return DFCompares
	default:
		return DFDescribes
	}
}

// isAcceptableSentence checks if a neural-generated sentence meets minimum
// quality standards. Rejects garbage, repetitions, or sentences that don't
// mention the subject or object at all.
func isAcceptableSentence(s, subject, object string) bool {
	if len(s) < 10 || len(s) > 300 {
		return false
	}
	// Must end with punctuation
	if !strings.HasSuffix(s, ".") && !strings.HasSuffix(s, "!") && !strings.HasSuffix(s, "?") {
		return false
	}
	lower := strings.ToLower(s)
	// Must reference at least the subject
	if !strings.Contains(lower, strings.ToLower(subject)) {
		return false
	}
	// Reject if it contains repeated character sequences (degenerate output)
	for i := 0; i+6 < len(s); i++ {
		chunk := s[i : i+3]
		if strings.Count(s[i:], chunk) > 4 {
			return false
		}
	}
	return true
}

// -----------------------------------------------------------------------
// Discourse Flow — multiple strategies for combining sentences
// -----------------------------------------------------------------------

func (c *Composer) combineWithFlow(sentences []string) string {
	if len(sentences) == 0 {
		return ""
	}
	if len(sentences) == 1 {
		return sentences[0]
	}

	strategy := c.rng.Intn(4)
	switch strategy {
	case 0:
		return c.flowConnectors(sentences)
	case 1:
		return c.flowParagraph(sentences)
	case 2:
		return c.flowMixed(sentences)
	case 3:
		return c.flowRhythmic(sentences)
	default:
		return c.flowConnectors(sentences)
	}
}

// flowConnectors: classic connector-based flow.
func (c *Composer) flowConnectors(sentences []string) string {
	var b strings.Builder
	b.WriteString(sentences[0])
	for i := 1; i < len(sentences); i++ {
		connector := c.pick(allConnectors)
		if connector == "" {
			b.WriteString(" " + sentences[i])
		} else {
			b.WriteString(" " + connector + " " + connectorLowerFirst(sentences[i]))
		}
	}
	return b.String()
}

// flowParagraph: separate sentences, no connectors (cleaner, more direct).
func (c *Composer) flowParagraph(sentences []string) string {
	return strings.Join(sentences, " ")
}

// flowMixed: alternates between connected and standalone sentences.
func (c *Composer) flowMixed(sentences []string) string {
	var b strings.Builder
	b.WriteString(sentences[0])
	for i := 1; i < len(sentences); i++ {
		if i%2 == 1 {
			connector := c.pick(allConnectors)
			if connector == "" {
				b.WriteString(" " + sentences[i])
			} else {
				b.WriteString(" " + connector + " " + connectorLowerFirst(sentences[i]))
			}
		} else {
			b.WriteString(" " + sentences[i])
		}
	}
	return b.String()
}

// flowRhythmic: short→long→short pattern for natural reading rhythm.
func (c *Composer) flowRhythmic(sentences []string) string {
	var b strings.Builder
	for i, s := range sentences {
		if i > 0 {
			if i%2 == 0 {
				b.WriteString(" " + s)
			} else {
				b.WriteString(" " + c.pick(allConnectors) + " " + connectorLowerFirst(s))
			}
		} else {
			b.WriteString(s)
		}
	}
	return b.String()
}

// -----------------------------------------------------------------------
// Phrase Libraries — deep variation pools with tone variants
// -----------------------------------------------------------------------

var relationTemplates = map[RelType][]string{
	RelIsA: {
		"%s is a %s.",
		"%s is a type of %s.",
		"%s is considered a %s.",
		"%s represents a %s.",
		"%s functions as a %s.",
		"%s falls into the category of %s.",
		"%s can best be understood as a %s.",
		"At its core, %s is a %s.",
		"In essence, %s is a %s.",
	},
	RelLocatedIn: {
		"%s is based in %s.",
		"%s is located in %s.",
		"%s is in %s.",
		"%s is headquartered in %s.",
		"%s can be found in %s.",
		"%s sits in %s.",
		"%s calls %s home.",
	},
	RelFoundedBy: {
		"%s was founded by %s.",
		"%s was created by %s.",
		"%s was started by %s.",
		"%s was built by %s.",
		"%s was established by %s.",
		"%s owes its existence to %s.",
		"The mind behind %s was %s.",
		"It was %s who created %s.",
	},
	RelFoundedIn: {
		"%s was founded in %s.",
		"%s was established in %s.",
		"%s started in %s.",
		"%s began in %s.",
		"%s dates back to %s.",
		"%s traces its origins to %s.",
		"%s first appeared in %s.",
		"%s has been around since %s.",
	},
	RelOffers: {
		"%s offers %s.",
		"%s provides %s.",
		"%s features %s.",
		"%s includes %s.",
		"%s brings %s to the table.",
		"%s comes with %s.",
	},
	RelHas: {
		"%s has %s.",
		"%s includes %s.",
		"%s features %s.",
		"%s possesses %s.",
		"%s is characterized by %s.",
		"One defining trait of %s is %s.",
		"A key aspect of %s is %s.",
		"%s is known for having %s.",
	},
	RelUsedFor: {
		"%s is used for %s.",
		"%s is applied to %s.",
		"%s is designed for %s.",
		"%s serves %s.",
		"%s plays a role in %s.",
		"%s finds application in %s.",
		"People turn to %s for %s.",
		"The main use of %s is %s.",
	},
	RelCreatedBy: {
		"%s was created by %s.",
		"%s was built by %s.",
		"%s was made by %s.",
		"%s was developed by %s.",
		"%s was authored by %s.",
		"%s came from %s.",
		"The creator of %s is %s.",
		"%s is the work of %s.",
	},
	RelDomain: {
		"%s is in the field of %s.",
		"%s relates to %s.",
		"%s belongs to the field of %s.",
		"%s falls within %s.",
		"%s is a subject within %s.",
	},
	RelDescribedAs: {
		"%s is %s.",
		"%s is described as %s.",
		"%s is known as %s.",
		"%s is recognized as %s.",
		"%s is often characterized as %s.",
	},
	RelPartOf: {
		"%s is part of %s.",
		"%s belongs to %s.",
		"%s fits within %s.",
		"%s forms a component of %s.",
		"%s is integral to %s.",
	},
	RelPrefers: {
		"%s prefers %s.",
		"%s favors %s.",
		"%s tends to choose %s.",
		"%s gravitates toward %s.",
	},
	RelDislikes: {
		"%s dislikes %s.",
		"%s avoids %s.",
		"%s isn't fond of %s.",
		"%s tends to steer clear of %s.",
	},
	RelContradicts: {
		"%s contradicts %s.",
		"%s is at odds with %s.",
		"%s runs counter to %s.",
		"%s conflicts with %s.",
		"%s stands in contrast to %s.",
	},
	RelCauses: {
		"%s leads to %s.",
		"%s causes %s.",
		"%s contributes to %s.",
		"%s gives rise to %s.",
		"%s can result in %s.",
	},
	RelRelatedTo: {
		"%s is connected to %s.",
		"%s relates to %s.",
		"%s is linked to %s.",
		"%s has ties to %s.",
		"%s intersects with %s.",
		"There is a close relationship between %s and %s.",
	},
	RelKnownFor: {
		"%s is known for %s.",
		"%s is famous for %s.",
		"%s is recognized for %s.",
		"%s made its mark through %s.",
		"What sets %s apart is %s.",
	},
	RelInfluencedBy: {
		"%s was influenced by %s.",
		"%s drew inspiration from %s.",
		"%s was shaped by %s.",
		"The ideas behind %s trace back to %s.",
	},
}

// fixArticleAgreement corrects article-noun agreement in both directions:
//   "a acceleration" → "an acceleration"
//   "an programming" → "a programming"
var articleToAnRe = regexp.MustCompile(`\b(a) ([aeiouAEIOU]\w*)`)
var articleToARe = regexp.MustCompile(`\b(an) ([^aeiouAEIOU\s]\w*)`)

func fixArticleAgreement(sentence string) string {
	// Fix "a" → "an" before vowel sounds
	sentence = articleToAnRe.ReplaceAllStringFunc(sentence, func(match string) string {
		parts := articleToAnRe.FindStringSubmatch(match)
		if len(parts) < 3 {
			return match
		}
		word := strings.ToLower(parts[2])
		// Exceptions: vowel letter but consonant sound
		for _, ex := range []string{"uni", "use", "uti", "ubi"} {
			if strings.HasPrefix(word, ex) {
				return match
			}
		}
		if parts[1] == "A" {
			return "An " + parts[2]
		}
		return "an " + parts[2]
	})
	// Fix "an" → "a" before consonant sounds
	sentence = articleToARe.ReplaceAllStringFunc(sentence, func(match string) string {
		parts := articleToARe.FindStringSubmatch(match)
		if len(parts) < 3 {
			return match
		}
		word := strings.ToLower(parts[2])
		// Exceptions: consonant letter but vowel sound ("an hour", "an honor")
		for _, ex := range []string{"hour", "honor", "honest", "heir"} {
			if strings.HasPrefix(word, ex) {
				return match
			}
		}
		if parts[1] == "An" {
			return "A " + parts[2]
		}
		return "a " + parts[2]
	})
	return sentence
}

// fixOneOfThePlural fixes "one of the X" where X should be plural.
// E.g., "is one of the programming language." → "...programming languages."
// Handles multi-word noun phrases: takes the LAST word and pluralizes it.
var oneOfTheRe = regexp.MustCompile(`one of the ([^.]+)\.`)

func fixOneOfThePlural(sentence string) string {
	return oneOfTheRe.ReplaceAllStringFunc(sentence, func(match string) string {
		parts := oneOfTheRe.FindStringSubmatch(match)
		if len(parts) < 2 {
			return match
		}
		phrase := strings.TrimSpace(parts[1])
		words := strings.Fields(phrase)
		if len(words) == 0 {
			return match
		}
		lastWord := words[len(words)-1]
		// Already plural
		if strings.HasSuffix(lastWord, "s") {
			return match
		}
		// Simple pluralization of last word
		plural := simplePlural(lastWord)
		words[len(words)-1] = plural
		return "one of the " + strings.Join(words, " ") + "."
	})
}


// isFragmentObject detects fact objects that are sentence fragments rather than
// proper noun phrases. E.g., "from the atmosphere" or "which are used in" are
// fragments that produce nonsense when plugged into templates like "%s has %s."
func isFragmentObject(obj string) bool {
	if len(obj) < 3 {
		return true
	}
	lower := strings.ToLower(strings.TrimSpace(obj))
	// Starts with a preposition/conjunction/relative pronoun — likely a fragment
	fragmentPrefixes := []string{
		"from ", "with ", "in ", "on ", "at ", "by ", "for ", "to ",
		"which ", "that ", "who ", "whom ", "whose ", "where ", "when ",
		"of ", "into ", "onto ", "upon ", "through ", "during ",
		"and ", "or ", "but ", "nor ", "yet ",
		"than ", "as ", "like ",
	}
	for _, p := range fragmentPrefixes {
		if strings.HasPrefix(lower, p) {
			return true
		}
	}
	// Contains wiki markup artifacts
	if strings.Contains(obj, "]]") || strings.Contains(obj, "[[") {
		return true
	}
	// Extremely long objects are likely raw wiki paragraph extracts, not facts.
	// Allow up to 600 chars for Layer 1 lead paragraphs (capped at ~500).
	if len(obj) > 600 {
		return true
	}
	return false
}

// isLikelyPlural detects plural subjects for verb agreement.
// Handles common patterns: "black holes", "programming languages", "the Beatles".
func isLikelyPlural(subject string) bool {
	lower := strings.ToLower(strings.TrimSpace(subject))
	// Check for known singular patterns
	if strings.HasSuffix(lower, "ss") || strings.HasSuffix(lower, "us") ||
		strings.HasSuffix(lower, "is") || strings.HasSuffix(lower, "ics") {
		return false // "physics", "stoicism", "canvas", "analysis", etc.
	}
	// Pronouns
	switch lower {
	case "they", "we", "these", "those":
		return true
	case "it", "he", "she", "this", "that":
		return false
	}
	// Most English plurals end in "s" (but not "ss", "us", etc.)
	lastWord := lower
	if idx := strings.LastIndex(lower, " "); idx >= 0 {
		lastWord = lower[idx+1:]
	}
	// Proper nouns ending in common Greek/Latin suffixes are singular
	// Socrates, Hercules, Heracles, Diogenes, Descartes, Pythagoras, etc.
	if strings.HasSuffix(lastWord, "tes") || strings.HasSuffix(lastWord, "les") ||
		strings.HasSuffix(lastWord, "nes") || strings.HasSuffix(lastWord, "res") ||
		strings.HasSuffix(lastWord, "des") {
		if len(subject) > 0 && subject[0] >= 'A' && subject[0] <= 'Z' {
			return false
		}
	}

	if strings.HasSuffix(lastWord, "s") && !strings.HasSuffix(lastWord, "ss") &&
		!strings.HasSuffix(lastWord, "us") && !strings.HasSuffix(lastWord, "is") {
		return true
	}
	return false
}

// fixSubjectVerbAgreement corrects 3rd-person singular verbs to plural form
// when the subject is plural. E.g., "black holes has X" → "black holes have X".
func fixSubjectVerbAgreement(sentence string, subject string) string {
	// Only fix verbs that immediately follow the subject
	prefix := subject + " "
	if !strings.HasPrefix(sentence, prefix) {
		return sentence
	}
	rest := sentence[len(prefix):]

	// Map singular → plural verb forms
	fixes := []struct{ singular, plural string }{
		{"has ", "have "},
		{"was ", "were "},
		{"is a ", "are a "},
		{"is an ", "are an "},
		{"is the ", "are the "},
		{"is in ", "are in "},
		{"is based ", "are based "},
		{"is located ", "are located "},
		{"is classified ", "are classified "},
		{"is considered ", "are considered "},
		{"is described ", "are described "},
		{"is known ", "are known "},
		{"is characterized ", "are characterized "},
		{"is recognized ", "are recognized "},
		{"is connected ", "are connected "},
		{"is related ", "are related "},
		{"is linked ", "are linked "},
		{"is used ", "are used "},
		{"is designed ", "are designed "},
		{"is built ", "are built "},
		{"is geared ", "are geared "},
		{"is equipped ", "are equipped "},
		{"is applied ", "are applied "},
		{"is part ", "are part "},
		{"is embedded ", "are embedded "},
		{"is planted ", "are planted "},
		{"is best described ", "are best described "},
		{"is %s.", "are %s."},
		{"includes ", "include "},
		{"comes with ", "come with "},
		{"features ", "feature "},
		{"encompasses ", "encompass "},
		{"offers ", "offer "},
		{"provides ", "provide "},
		{"delivers ", "deliver "},
		{"brings ", "bring "},
		{"makes ", "make "},
		{"enables ", "enable "},
		{"helps ", "help "},
		{"serves ", "serve "},
		{"falls ", "fall "},
		{"qualifies ", "qualify "},
		{"fits ", "fit "},
		{"sits ", "sit "},
		{"belongs ", "belong "},
		{"stands ", "stand "},
		{"calls ", "call "},
		{"operates ", "operate "},
		{"prefers ", "prefer "},
		{"favors ", "favor "},
		{"leans ", "lean "},
		{"gravitates ", "gravitate "},
		{"tends ", "tend "},
		{"dislikes ", "dislike "},
		{"avoids ", "avoid "},
		{"contradicts ", "contradict "},
		{"leads ", "lead "},
		{"causes ", "cause "},
		{"contributes ", "contribute "},
		{"drives ", "drive "},
		{"triggers ", "trigger "},
		{"sets ", "set "},
		{"relates ", "relate "},
		{"intersects ", "intersect "},
	}

	for _, fix := range fixes {
		if strings.HasPrefix(rest, fix.singular) {
			return prefix + fix.plural + rest[len(fix.singular):]
		}
	}
	return sentence
}

var inferredPrefixes = []string{
	"Based on what I know, ", "From what I've gathered, ",
	"By inference, ", "Connecting the dots, ",
	"From the available information, ", "Piecing things together, ",
	"Reading between the lines, ", "If I connect the dots, ",
	"Based on what I've seen, ",
}

var allConnectors = []string{
	"",  // no connector — just continue naturally
	"",  // doubled to increase probability of clean flow
	"In addition,",
	"What stands out is that",
	"It is worth mentioning that",
	"Interestingly,",
	"On a related note,",
	"Looking deeper,",
	"Equally important,",
	"Also,",
	"Moreover,",
}

var contrastConnectors = []string{
	"That said,",
	"However,",
	"At the same time,",
	"Meanwhile,",
	"On the other hand,",
	"Then again,",
	"Even so,",
	"Still,",
	"In contrast,",
}

// semanticConnectors maps (previous relation, current relation) pairs to
// contextually appropriate discourse connectors. All connectors MUST end
// with a comma — they are prepended to the next sentence with the first
// letter lowercased. Full-sentence connectors are avoided because they
// duplicate verbs from the templated sentence.
var semanticConnectors = map[RelType]map[RelType][]string{
	// After identity → explaining properties/location/origin
	RelIsA: {
		RelLocatedIn:   {"", "Geographically,", "In terms of location,", "As for where it sits,"},
		RelFoundedIn:   {"", "Historically,", "Looking back,", "Going back in time,"},
		RelFoundedBy:   {"", "In terms of its creation,", "Behind the scenes,", "As for who started it,"},
		RelHas:         {"", "Among its characteristics,", "One thing that stands out:", "It comes with"},
		RelUsedFor:     {"", "In practice,", "When put to use,", "On the practical side,"},
		RelDescribedAs: {"", "More specifically,", "In particular,", "To be more precise,"},
	},
	// After origin → properties/purpose
	RelFoundedIn: {
		RelFoundedBy: {"", "Regarding its creation,", "As for who was behind it,"},
		RelHas:       {"", "Among its features,", "It brings with it", "One notable aspect:"},
		RelUsedFor:   {"", "In practice,", "When put to use,"},
		RelIsA:       {"", "At its core,", "Fundamentally,"},
	},
	RelFoundedBy: {
		RelFoundedIn: {"", "Historically,", "In terms of timing,", "Looking at the timeline,"},
		RelHas:       {"", "Among its features,", "It offers", "One notable aspect:"},
		RelUsedFor:   {"", "In practice,", "When put to use,"},
	},
	// After features → more features or purpose
	RelHas: {
		RelHas:     {"", "", "It also has", "Another aspect is", "On top of that,"},
		RelUsedFor: {"", "In practice,", "When put to use,", "This makes it useful for"},
		RelOffers:  {"", "It also provides", "Along with that,"},
	},
	// After purpose → related concepts
	RelUsedFor: {
		RelUsedFor:   {"", "", "It also serves", "Another use is"},
		RelRelatedTo: {"", "On a related note,", "Connected to this,", "This ties into"},
		RelHas:       {"", "It also features", "Alongside that,"},
	},
}

// connectorBetween picks a semantically appropriate connector between two
// consecutive facts based on their relation types. Falls back to generic
// additive connectors if no specific mapping exists.
func (c *Composer) connectorBetween(prev, cur RelType) string {
	if m, ok := semanticConnectors[prev]; ok {
		if conns, ok := m[cur]; ok {
			return c.pick(conns)
		}
	}
	// Same relation type → sometimes no connector, sometimes explicit parallel
	if prev == cur {
		return c.pick([]string{"", "", "Similarly,", "Likewise,", "Along the same lines,", "In much the same way,"})
	}
	// Contrast relations
	if cur == RelContradicts {
		return c.pick(contrastConnectors)
	}
	return c.pick(allConnectors)
}

var morningGreetings = []string{
	"Good morning", "Morning", "Hey, good morning",
	"Rise and shine", "Top of the morning",
	"Early bird catches the worm — good morning",
}
var afternoonGreetings = []string{
	"Good afternoon", "Hey", "Afternoon", "Hey there",
	"Yo", "What's up",
}
var eveningGreetings = []string{
	"Good evening", "Evening", "Hey", "Hey there",
	"Winding down? Good evening",
}

var morningBriefingOpeners = []string{
	"Good morning %s. Here's your day.",
	"Morning %s. Let me catch you up.",
	"Hey %s. Here's where things stand.",
	"Good morning %s. Here's what I've got for you.",
	"Morning %s. Quick rundown for you.",
	"%s — here's your morning snapshot.",
}

var briefingOpenersGeneric = []string{
	"Here's where things stand.",
	"Quick update on your day.",
	"Here's your current status.",
	"Let me catch you up.",
	"Here's what I've got.",
	"Quick snapshot for you.",
	"Here's the rundown.",
}
var briefingOpenersCasual = []string{
	"Alright, here's the deal.",
	"So here's where things are at.",
	"Quick check-in for you.",
}
var briefingOpenersWarm = []string{
	"Hope you're doing well. Here's a quick look at your day.",
	"Let me walk you through where things stand.",
	"Here's your update — nice and easy.",
}
var briefingOpenersDirect = []string{
	"Status update.",
	"Your numbers.",
	"The rundown.",
}

var briefSpendPhrases = []string{
	"Spending this week: €%.0f.",
	"You've spent €%.0f so far this week.",
	"Weekly spending sits at €%.0f.",
	"€%.0f spent this week.",
	"This week's spend: €%.0f.",
}

var briefHabitPhrases = []string{
	"Habit streak: %d days.",
	"You're on a %d-day habit streak.",
	"Habits: %d days running.",
	"%d consecutive days of habits.",
	"Streak watch: %d days and counting.",
}

var briefMoodPhrases = []string{
	"Recent mood: %.1f/5.",
	"Mood average: %.1f out of 5.",
	"You've been sitting at a %.1f/5 mood-wise.",
	"Emotional pulse: %.1f/5.",
}

var journalGapPhrases = []string{
	"It's been %d days since your last journal entry — your best weeks tend to start with writing.",
	"You haven't journaled in %d days. Even a quick entry might help.",
	"No journal entries for %d days. Your mood tends to track higher when you write regularly.",
	"%d days without a journal entry. Writing helps you process — might be worth a few minutes.",
	"Your journal's been quiet for %d days. Past data shows your best clarity comes after writing.",
	"Journal gap: %d days. That pen's been waiting for you.",
	"%d days since you last wrote. Even a few lines can shift your perspective.",
}

var journalRecentPhrases = []string{
	"You journaled yesterday.",
	"Last journal entry was yesterday — that's recent.",
	"You wrote in your journal yesterday.",
	"Journal's fresh — you wrote yesterday.",
	"Yesterday's journal entry is still recent.",
}

var lowMoodPhrases = []string{
	"Your recent mood has been lower than usual. Take it easy today.",
	"Things have felt heavy lately based on your mood tracking.",
	"Your mood scores have dipped recently — be gentle with yourself.",
	"Mood has been trending down. No pressure, just wanted to flag it.",
	"Your recent mood suggests a tougher stretch. That's okay.",
}
var lowMoodCasual = []string{
	"Mood's been a bit off lately. No biggie, just flagging it.",
	"You haven't been feeling great based on your numbers. Take it easy.",
}
var lowMoodWarm = []string{
	"I've noticed your mood has been lower recently. I hope today's a bit brighter.",
	"Your mood tracking shows a dip — take care of yourself today.",
}
var lowMoodDirect = []string{
	"Mood is down. Worth paying attention to.",
	"Your mood scores are below baseline.",
}

var highMoodPhrases = []string{
	"Your mood has been strong lately — keep that momentum going.",
	"You've been feeling good recently. Nice.",
	"Mood scores are up. Whatever you're doing is working.",
	"Things seem to be going well — your mood reflects it.",
	"Your recent mood is notably positive. Ride the wave.",
}
var highMoodCasual = []string{
	"You've been vibing lately. Mood's looking good.",
	"Feeling good, huh? Numbers back it up.",
}
var highMoodWarm = []string{
	"It's great to see your mood so positive lately.",
	"Your mood has been shining — keep doing what you're doing.",
}
var highMoodDirect = []string{
	"Mood is strong. Keep going.",
	"You're in a good stretch emotionally.",
}

var habitStreakPhrases = []string{
	"You're on a %d-day habit streak — solid consistency.",
	"%d-day habit streak going. Don't break the chain.",
	"Nice — %d days of consistent habits.",
	"%d days running on your habits. That takes discipline.",
	"Habit streak: %d days. You're building real momentum.",
	"%d days strong. That's not luck, that's effort.",
	"Your habits have held for %d days straight. Impressive.",
}

var underBudgetPhrases = []string{
	"You're tracking under your usual spending this week.",
	"Spending is below average — good discipline.",
	"Below your typical weekly spend so far.",
	"Running lean on spending this week. Well managed.",
	"Budget-wise, you're doing better than usual.",
}

var overBudgetPhrases = []string{
	"Spending is running higher than usual this week.",
	"You're above your typical weekly average on spending.",
	"This week's spending is trending above your norm.",
	"Heads up — spending is above your baseline.",
}

var expenseAckPhrases = []string{
	"Recorded.", "Got it.", "Logged.", "Noted.", "Tracked.", "Added.",
}
var expenseAckCasual = []string{
	"Done deal.", "On the books.", "Sorted.", "Cool, logged.",
}
var expenseAckWarm = []string{
	"Got it, thanks for logging that.", "Noted — keeping your records tidy.",
}
var expenseAckDirect = []string{
	"Logged.", "Added.", "Done.",
}

var spendTrendPhrases = []string{
	"You're at %.0f%% of your typical weekly spending so far.",
	"That puts you at %.0f%% of your usual weekly budget.",
	"You've used about %.0f%% of your average weekly spend.",
	"Roughly %.0f%% of your weekly average is gone.",
}

var spendTrendOver = []string{
	"You're at %.0f%% of your typical weekly spending.",
	"That puts you at %.0f%% of your weekly average — a bit over.",
	"You've hit %.0f%% of your usual weekly budget already.",
}

var spendSummaryUnder = []string{
	"You've spent €%.0f this week, which is €%.0f below your average.",
	"€%.0f spent this week — that's €%.0f under your usual.",
	"This week you've spent €%.0f, running €%.0f below average.",
	"Your spending sits at €%.0f — €%.0f less than your weekly norm.",
	"€%.0f this week. That's €%.0f below what you'd normally spend.",
}

var spendSummaryOver = []string{
	"You've spent €%.0f this week, €%.0f above your usual.",
	"€%.0f this week — that's €%.0f over your average.",
	"Spending sits at €%.0f, which is €%.0f above your norm.",
	"This week's spending: €%.0f, running €%.0f hot.",
	"€%.0f spent — that's €%.0f more than your typical week.",
}

var moodWordsGreat = []string{"great", "really good", "positive", "strong", "elevated", "high", "excellent", "upbeat"}
var moodWordsOkay = []string{"decent", "okay", "steady", "alright", "stable", "moderate", "solid", "balanced"}
var moodWordsLow = []string{"low", "rough", "below your usual", "subdued", "dipping", "off", "muted"}
var moodWordsTough = []string{"tough", "really difficult", "challenging", "heavy", "strained", "hard"}

var moodSummaryPhrases = []string{
	"Your mood has been %s — averaging %.1f out of 5.",
	"Mood-wise, things have been %s, sitting at %.1f/5.",
	"You've been feeling %s lately — %.1f out of 5 on average.",
	"Your emotional state has been %s, around %.1f/5.",
	"On the mood front, you're at %s — %.1f/5 to be exact.",
}
var moodSummaryCasual = []string{
	"Mood's been %s — %.1f out of 5.",
	"You're running %s mood-wise. That's a %.1f/5.",
	"Feeling %s? Your numbers say %.1f/5.",
}
var moodSummaryWarm = []string{
	"Your mood has been %s recently, sitting at %.1f out of 5.",
	"I've been tracking your mood as %s — %.1f/5 on average.",
}
var moodSummaryDirect = []string{
	"Mood: %s. Score: %.1f/5.",
	"Emotional state: %s (%.1f/5).",
}

var habitSummaryPhrases = []string{
	"You're on a %d-day habit streak.",
	"Habits: %d days running without a break.",
	"%d consecutive days of habit completion.",
	"Your habit streak stands at %d days.",
	"That's %d days of habits in a row. Not bad at all.",
	"%d-day streak on your habits — building real consistency.",
}

var journalAckPhrases = []string{
	"Saved to your journal.", "Entry recorded.", "Written.",
	"Journal entry saved.", "Got it — written to your journal.",
}
var journalAckCasual = []string{
	"Written down.", "In the journal.", "Jotted.", "Locked in.",
}
var journalAckWarm = []string{
	"Beautifully captured. It's in your journal.",
	"Saved. Glad you took the time to write.",
}
var journalAckDirect = []string{
	"Saved.", "Written.", "Done.",
}

var habitAckPhrases = []string{
	"Habit logged.", "Tracked.", "Done.", "Habit recorded.", "Checked off.",
}
var habitAckCasual = []string{
	"Another one down.", "Boom, logged.", "On it.", "Check.",
}
var habitAckWarm = []string{
	"Great job, habit tracked.", "Logged — keep it up.", "Nice one, recorded.",
}
var habitAckDirect = []string{
	"Done.", "Logged.", "Tracked.",
}

var streakPhrases = []string{
	"That's %d days in a row now.",
	"Day %d of your streak.",
	"%d consecutive days.",
	"You're at %d and counting.",
	"%d days running. Strong.",
	"Streak: %d. Don't stop now.",
}

var reminderAckPhrases = []string{
	"Reminder set.", "I'll remind you.", "Set.", "You'll be reminded.",
	"Noted — I'll nudge you.", "Consider it set.",
	"You'll hear from me.",
}

var correlationPhrases = []string{
	"I've noticed a pattern: %s tends to correlate with %s.",
	"There seems to be a connection between %s and %s.",
	"Your data suggests %s and %s are linked.",
	"Looking at your patterns, %s and %s tend to move together.",
	"An interesting trend: %s appears connected to %s in your data.",
	"Here's something: %s and %s seem to track together.",
	"Pattern I've spotted — %s and %s influence each other.",
}

var uncertainPhrases = []string{
	"I don't have enough information about that yet. Ask me after I've learned more, or try a web search.",
	"My knowledge graph doesn't cover that topic well. Want me to research it?",
	"I'm not sure about this one — I'd rather be honest than guess.",
	"That's outside what I currently know. I could search the web for you.",
	"I don't have strong data on that. Want me to look it up?",
	"Honestly, I'd be guessing. Let me search for real information instead.",
	"Haven't built up enough knowledge there yet. Want me to dig into it?",
	"That's a gap in what I know. Let me fix that — should I search?",
}

var followUpQuestions = []string{
	"Want me to dig deeper into any of this?",
	"Anything specific you'd like to explore further?",
	"Should I research more about this?",
	"Want me to find more details?",
	"Curious about any particular aspect?",
	"Shall I keep going?",
	"Want the full picture, or is that enough?",
	"Anything else you're curious about?",
}

// -----------------------------------------------------------------------
// Conversational Phrase Libraries
// -----------------------------------------------------------------------

// Sentiment word lists
var positiveWords = []string{
	"good", "great", "awesome", "love", "happy", "nice", "wonderful",
	"fantastic", "excellent", "amazing", "perfect", "beautiful",
	"glad", "pleased", "enjoy", "fun", "excited", "grateful",
	"promoted", "graduated", "achieved", "succeeded", "won", "hired",
	"accepted", "finished", "completed", "proud",
}
var negativeWords = []string{
	"bad", "terrible", "hate", "awful", "horrible", "worst",
	"disappointed", "annoying", "frustrating", "boring", "tired",
	"exhausted", "overwhelmed", "stressed", "struggling",
}
var excitedWords = []string{
	"amazing", "incredible", "wow", "omg", "insane", "unbelievable",
	"mind-blowing", "blown away", "so excited", "can't wait",
}
var sadWords = []string{
	"sad", "depressed", "lonely", "heartbroken", "miss", "lost",
	"crying", "hurting", "empty", "hopeless", "grief",
}
var angryWords = []string{
	"angry", "furious", "pissed", "rage", "livid", "outraged",
	"infuriating", "fed up", "sick of", "so mad",
}

// Talk acknowledgments
var talkAckNeutral = []string{
	"What's on your mind?",
	"What would you like to explore?",
	"What are you thinking about?",
	"Where should we start?",
	"What's the topic?",
}
var talkAckCasual = []string{
	"What's on your mind?",
	"What are you curious about?",
	"What should we dig into?",
}
var talkAckWarm = []string{
	"I'd like to hear what's on your mind.",
	"What are you thinking about? Take your time.",
	"What would you like to talk through?",
}
var talkAckDirect = []string{
	"Go ahead.",
	"What's the topic?",
	"What do you need?",
}

// Boredom acknowledgments
var boredAckPhrases = []string{
	"Boredom usually means you're ready for a challenge. What sounds interesting right now?",
	"Let's find something worth your attention. What are you in the mood for?",
	"That restless feeling is useful — it means you want to engage with something. What direction?",
	"Good — boredom is a signal. What's something you've been meaning to think about?",
	"Let's put that energy somewhere. What topic would grab you right now?",
}

// Positive sentiment acknowledgments
var positiveAckNeutral = []string{
	"That's great — what made the difference?",
	"Good to hear. What are you thinking about next?",
	"Nice — tell me more about how that happened.",
	"Sounds like things are moving. What's the next step?",
}
var positiveAckCasual = []string{
	"Love that — what's the plan from here?",
	"Let's go! What are you building on next?",
	"That's solid. How did you pull it off?",
}
var positiveAckWarm = []string{
	"That's wonderful — I'd love to hear more about it.",
	"I'm really glad to hear that. What does this open up for you?",
	"That's the kind of thing worth celebrating. What comes next?",
}
var positiveAckDirect = []string{
	"Good. What's next?", "Solid result. Where does that leave you?",
	"That works. What are you building toward?",
}

// Negative sentiment acknowledgments
var negativeAckNeutral = []string{
	"That sounds difficult. What's the main challenge right now?",
	"I can see why that's hard. What would help most?",
	"That's a tough spot. What are your options?",
	"I hear you — that's not easy. What's the biggest obstacle?",
}
var negativeAckCasual = []string{
	"Yeah, that's rough. What's the part that's getting to you most?",
	"Ugh, I hear you. What would make the biggest difference right now?",
	"That's frustrating. What have you tried so far?",
}
var negativeAckWarm = []string{
	"I'm sorry you're dealing with that. What would be most helpful to talk through?",
	"That sounds really hard. I want to help if I can — what's weighing on you most?",
	"I can see why that's weighing on you. Where would you like to start?",
}
var negativeAckDirect = []string{
	"That's a real problem. What's the core issue?",
	"Understood. What are you trying to solve?",
	"Clear. What would a good outcome look like?",
}

// Angry acknowledgments
var angryAckPhrases = []string{
	"That's a legitimate frustration — let's work through it.",
	"I hear you. Let's figure out what can actually change here.",
	"That would frustrate anyone. What's the core issue?",
	"Understood — let's focus on what you can do about it.",
}

// Curious acknowledgments
var curiousAckPhrases = []string{
	"Here's what I can tell you about that.",
	"Let me share what I know.",
	"That's a topic I can help with.",
	"Let me pull together what I know on this.",
	"Here's my understanding.",
}

// Neutral acknowledgments
var neutralAckNeutral = []string{
	"That's worth thinking about.",
	"Let me consider that.",
	"There's a lot to unpack there.",
	"That raises some interesting points.",
	"Here's how I'd think about that.",
	"Let me work through that with you.",
}
var neutralAckCasual = []string{
	"That's a solid question to dig into.",
	"Let's think through this.",
	"Here's what comes to mind.",
	"Good topic — let me think about that.",
}
var neutralAckWarm = []string{
	"That's something I'd like to help with.",
	"Let's explore that together.",
	"I want to make sure I give you something useful here.",
	"That deserves a thoughtful response.",
}
var neutralAckDirect = []string{
	"Let me address that directly.",
	"Here's what I think.",
	"Let me cut to what matters here.",
	"Straightforward question — here's my take.",
}

// Bridge phrases — connecting to knowledge
var bridgePhrases = []string{
	"Speaking of which,", "On a related note,", "That reminds me —",
	"Now that you mention it,", "You know what's interesting?",
	"That connects to something I know:", "Actually,",
	"Come to think of it,", "Here's something relevant:",
}

// History bridge phrases
var historyBridgePhrases = []string{
	"We were talking about %s earlier — want to pick that back up?",
	"Earlier you mentioned %s. Still on your mind?",
	"By the way, you brought up %s before. Anything more on that?",
	"Going back to %s — anything else there?",
}

// Late night insights
var lateNightInsights = []string{
	"It's late — sometimes the best thinking happens at night, but so does overthinking. Be careful which one it is.",
	"Late night session? Make sure you're getting enough rest.",
	"Night owl mode. Just remember — sleep is when your brain consolidates everything.",
	"Burning the midnight oil. Tomorrow-you will appreciate some rest.",
}

// Good mood insights
var goodMoodInsights = []string{
	"You've been in a great headspace lately. That usually means you're doing something right.",
	"Your mood data shows you're on a good run. Whatever you're doing, keep doing it.",
	"Things seem to be clicking for you right now.",
}

// Streak insights
var streakInsights = []string{
	"Your %d-day streak shows real dedication. That kind of consistency compounds.",
	"You've been at it for %d days straight. That's character, not luck.",
	"%d days of habits — you're past the point where most people quit.",
}

// Personal follow-ups
var personalFollowUps = []string{
	"Tell me more about that.",
	"What made you feel that way?",
	"How long has that been going on?",
	"What triggered that?",
	"Want to unpack that a bit?",
	"I'm curious — what's behind that?",
	"How are you processing that?",
}

// Topic follow-ups
var topicFollowUps = []string{
	"What's your take on %s?",
	"Is there something specific about %s you're thinking about?",
	"Want me to look into %s more?",
	"Anything particular about %s that's on your mind?",
	"What got you thinking about %s?",
}

// Generic engagement
var genericEngagement = []string{
	"What's on your mind?",
	"What are you thinking about?",
	"Anything you want to explore?",
	"What's been on your radar lately?",
	"What would be most useful to talk about?",
	"I'm here — what do you need?",
}

// Empathetic phrases — sad
var empatheticSadNeutral = []string{
	"That's a heavy thing to carry.", "I hear you — that's not easy.",
	"That sounds really hard.", "I'm sorry you're dealing with that.",
}
var empatheticSadCasual = []string{
	"Man, that's rough.", "That's heavy. I'm here though.",
}
var empatheticSadWarm = []string{
	"I'm really sorry you're going through this.",
	"That must weigh on you. I wish I could do more.",
	"Take your time with it. There's no rush to feel better.",
	"That's tough. Stress is real — I hear you.",
	"Sorry to hear that. You don't have to figure it all out right now.",
}
var empatheticSadDirect = []string{
	"That's hard. What do you need right now?",
	"I hear you. What would help?",
}

// Empathetic phrases — angry
var empatheticAngryPhrases = []string{
	"Your frustration is completely valid. What happened?",
	"That would make anyone angry. Want to vent?",
	"I get it — that's infuriating.",
	"Anger like that usually means something important got violated.",
}

// Empathetic phrases — happy
var empatheticHappyPhrases = []string{
	"That's amazing! I love hearing that.",
	"Hell yes. You deserve that.",
	"That's genuinely great news.",
	"Your energy is contagious right now.",
	"That's what it's all about.",
}

// Empathetic phrases — neutral
var empatheticNeutralPhrases = []string{
	"I'm here for whatever you need.",
	"Thanks for sharing that with me.",
	"I'm listening. Keep going if you want.",
}

// Empathetic actions — universally appropriate (no "take a breath" which
// is wrong for celebration contexts, no stress-specific phrases).
var empatheticActions = []string{
	"Want to write about it in your journal? Sometimes that helps.",
	"Would it help to talk it through?",
	"I can check in with you later if you'd like.",
	"I'm not going anywhere.",
	"Take your time — no rush.",
	"Want to talk about what's on your mind, or would a quick distraction help?",
	"Sometimes just naming it helps — what's the biggest thing weighing on you?",
	"Would it help to make a quick to-do list to organize things, or do you just need to vent?",
	"One step at a time. What's the first thing on your mind?",
	"You don't have to have it all figured out right now.",
	"I'm listening. Keep going if you want.",
	"That sounds rough. Want to unpack it or just vent?",
}

// Mood-aware sad phrases
var moodAwareSadPhrases = []string{
	"Your mood tracking confirms this has been a rough stretch. You're allowed to not be okay.",
	"I've noticed your mood has been lower lately. That tracks with what you're telling me.",
	"Your data shows this isn't just today — it's been building. Take it seriously.",
}

// Opinion phrases
var opinionOpeners = []string{
	"Here's how I see it.", "My take on this:",
	"If I'm being honest,", "The way I understand it,",
	"From what I've observed,", "Thinking about this —",
	"Here's what I think.", "So here's my perspective.",
}

var opinionIsA = []string{
	"The fact that %s is a %s says a lot about its nature.",
	"Being a %s, %s has certain inherent strengths.",
	"%s being a %s is significant — it defines its entire approach.",
}

var opinionUsedFor = []string{
	"%s being used for %s tells you what it really excels at.",
	"The %s use case for %s is where it really shines.",
	"When it comes to %s, %s is purpose-built for it.",
}

var opinionDescribed = []string{
	"People describe %s as %s, and from what I know, that's accurate.",
	"%s being characterized as %s is well-earned.",
	"The reputation of %s as %s holds up.",
}

var genericOpinions = []string{
	"%s is one of those things where context really matters. What angle are you coming from?",
	"There's a lot to say about %s. It depends on what matters most to you.",
	"%s — that's a layered topic. My honest take depends on what you're trying to achieve.",
	"I think %s is worth understanding deeply before forming a strong opinion.",
}

var thoughtfulGeneric = []string{
	"That's the kind of question that deserves more than a quick answer.",
	"I think the answer depends on what you value most.",
	"There's no single right answer there — it depends on context.",
	"That's genuinely complex. I'd want to think about it from multiple angles.",
}

var opinionClosers = []string{
	"But I'm curious what you think.",
	"What's your take?",
	"Does that resonate with you?",
	"Where do you stand on it?",
	"Am I off base?",
	"That's just how I see it — you might see it differently.",
}

// Farewell phrases
var farewellPhrases = []string{
	"See you later.", "Take care.", "Until next time.",
	"Catch you later.", "I'll be here when you're back.",
	"Peace.", "Later.", "Take it easy.",
}
var farewellNamePhrases = []string{
	"See you later, %s.", "Take care, %s.",
	"Until next time, %s.", "Catch you later, %s.",
	"I'll be here when you're back, %s.",
}
var farewellStreakPhrases = []string{
	"Don't forget — you're on a %d-day streak. Keep it going.",
	"Remember, %d-day streak. Tomorrow matters.",
	"Your %d-day streak is counting on you. See you.",
}

// Thank you responses
var thankYouResponses = []string{
	"Happy to help — that's what I'm here for.",
	"Glad I could be useful.",
	"Anytime.",
	"You're welcome — let me know if you need anything else.",
	"No problem at all.",
	"Of course, happy to help.",
	"Anytime — seriously.",
	"My pleasure.",
	"Don't mention it. I'm here whenever you need me.",
}

// Uncertain bridge phrases — when we don't know but can connect to something
var uncertainBridgePhrases = []string{
	"I don't have a direct answer for that, but here's something related:",
	"That's not in my knowledge yet, though I do know this:",
	"I can't answer that directly, but on a related note:",
	"I'm not sure about that specifically, but here's what I do know:",
}

// -----------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------

func lowerFirst(s string) string {
	if s == "" {
		return s
	}
	if len(s) > 1 && s[1] >= 'A' && s[1] <= 'Z' {
		return s
	}
	return strings.ToLower(s[:1]) + s[1:]
}

// looksLikeProperName returns true if the name looks like a real human name
// rather than a unix login (e.g. "rrl", "admin", "root"). A proper name is
// capitalised, may contain spaces, and doesn't look like a short all-lowercase
// system identifier.
func looksLikeProperName(name string) bool {
	name = strings.TrimSpace(name)
	if name == "" {
		return false
	}
	// Names with spaces are almost certainly real names ("Raphael Lugmayr").
	if strings.Contains(name, " ") {
		return true
	}
	// All-lowercase short tokens look like unix usernames (rrl, admin, root).
	if name == strings.ToLower(name) {
		return false
	}
	// Single capitalised word ≥ 2 chars is plausible ("Raphael", "Alice").
	if len(name) >= 2 && name[0] >= 'A' && name[0] <= 'Z' {
		return true
	}
	return false
}

// connectorLowerFirst lowercases the first letter of a sentence that follows
// a discourse connector, but only for pronouns and generic references.
// Proper nouns (Linux, Python, Albert Einstein) stay capitalized.
func connectorLowerFirst(s string) string {
	if s == "" {
		return s
	}
	// Lowercase if it starts with a pronoun/generic reference
	pronounStarts := []string{
		"It ", "They ", "The ", "He ", "She ", "We ",
		"Its ", "Their ", "His ", "Her ", "Our ",
	}
	for _, p := range pronounStarts {
		if strings.HasPrefix(s, p) {
			return strings.ToLower(s[:1]) + s[1:]
		}
	}
	// Otherwise keep original case (likely a proper noun)
	return s
}

func capitalizeFirst(s string) string {
	if s == "" {
		return s
	}
	// Preserve casing for known acronyms and all-caps terms
	if isAcronym(s) {
		return strings.ToUpper(s)
	}
	runes := []rune(s)
	runes[0] = unicode.ToUpper(runes[0])
	return string(runes)
}

// isAcronym returns true if the string should be displayed in all-caps.
func isAcronym(s string) bool {
	upper := strings.ToUpper(s)
	known := map[string]bool{
		"DNA": true, "RNA": true, "AI": true, "MRI": true, "HTTP": true,
		"SQL": true, "HTML": true, "CSS": true, "API": true, "URL": true,
		"CPU": true, "GPU": true, "RAM": true, "SSD": true, "USB": true,
		"LED": true, "LCD": true, "NASA": true, "NATO": true, "CERN": true,
		"GDP": true, "CEO": true, "CTO": true, "PhD": true, "MIT": true,
		"UCLA": true, "IBM": true, "BMW": true, "NLP": true, "ML": true,
		"IoT": true, "VR": true, "AR": true, "5G": true, "TCP": true,
		"IP": true, "OS": true, "UI": true, "UX": true, "PR": true,
	}
	return known[upper]
}

// NodeLabel returns the display label for a node ID.
func (cg *CognitiveGraph) NodeLabel(id string) string {
	cg.mu.RLock()
	defer cg.mu.RUnlock()
	if n, ok := cg.nodes[id]; ok {
		return n.Label
	}
	return ""
}

func isIn(s string, options ...string) bool {
	for _, opt := range options {
		if s == opt {
			return true
		}
	}
	return false
}

// extractTopic pulls the main topic from a query, smarter than extractKeywords.
// Handles short words like "Go", "AI", "ML" that keywords would filter out.
func (c *Composer) extractTopic(query string) string {
	lower := strings.ToLower(query)

	// Strip common question prefixes to isolate the topic.
	// Long prefixes first for greedy matching.
	prefixes := []string{
		"give me a full overview of ", "give me an overview of ",
		"give me a summary of ", "tell me everything about ",
		"tell me all about ", "tell me about ",
		"walk me through ", "deep dive into ",
		"teach me about ", "help me understand ",
		"what do you think about ", "what's your opinion on ",
		"your take on ", "do you think ", "how do you feel about ",
		"what about ",
		"explain how ", "explain why ", "explain what ",
		"explain to me ", "explain ", "describe ", "define ",
		"what is ", "what are ", "what was ",
		"who is ", "who are ", "who was ",
		"how does ", "how do ", "how is ",
		"why is ", "why are ", "why does ",
		"compare ", "summarize ", "summarise ",
	}
	for _, p := range prefixes {
		if strings.HasPrefix(lower, p) {
			topic := strings.TrimSpace(query[len(p):])
			topic = strings.TrimRight(topic, "?!.")
			if topic != "" {
				return topic
			}
		}
	}

	// Fall back to keywords but include short words if capitalized in original
	words := strings.Fields(query)
	for _, w := range words {
		clean := strings.Trim(w, "?!.,;:")
		if len(clean) == 0 {
			continue
		}
		// Keep capitalized short words (Go, AI, ML, C++)
		// But skip pronouns/stop words even when capitalized (I, We, My)
		lowerClean := strings.ToLower(clean)
		if clean[0] >= 'A' && clean[0] <= 'Z' && !isStopWord(lowerClean) &&
			!extractiveStopWords[lowerClean] && !contentStopWords[lowerClean] {
			return clean
		}
	}

	// Last resort: extractKeywords
	keywords := extractKeywords(lower)
	if len(keywords) > 0 {
		return keywords[0]
	}
	return ""
}

// simplePlural adds a basic English plural suffix.
func simplePlural(word string) string {
	if word == "" {
		return word
	}
	lower := strings.ToLower(word)
	// words ending in y (preceded by consonant) → ies
	if strings.HasSuffix(lower, "y") && len(lower) > 1 {
		prev := lower[len(lower)-2]
		if prev != 'a' && prev != 'e' && prev != 'i' && prev != 'o' && prev != 'u' {
			return word[:len(word)-1] + "ies"
		}
	}
	// words ending in s, sh, ch, x, z → es
	for _, suf := range []string{"s", "sh", "ch", "x", "z"} {
		if strings.HasSuffix(lower, suf) {
			return word + "es"
		}
	}
	return word + "s"
}

func looksLikePersonName(s string) bool {
	words := strings.Fields(s)
	if len(words) == 0 || len(words) > 4 {
		return false
	}
	// Single capitalised word: check if it's a known surname or person entity.
	// Names like "Shakespeare", "Beethoven", "Napoleon" are single-word person refs.
	if len(words) == 1 {
		if len(s) >= 3 && s[0] >= 'A' && s[0] <= 'Z' && !strings.Contains(s, " ") {
			// Known single-word person names (famous surnames used alone)
			knownPersons := map[string]bool{
				"shakespeare": true, "beethoven": true, "napoleon": true,
				"aristotle": true, "plato": true, "socrates": true,
				"darwin": true, "newton": true, "galileo": true,
				"copernicus": true, "hippocrates": true, "archimedes": true,
				"confucius": true, "cleopatra": true, "michelangelo": true,
				"rembrandt": true, "voltaire": true, "mozart": true,
				"bach": true, "chopin": true, "picasso": true,
				"tesla": true, "edison": true, "gandhi": true,
				"caesar": true, "homer": true, "virgil": true,
			}
			if knownPersons[strings.ToLower(s)] {
				return true
			}
		}
		return false
	}
	if len(words) < 2 {
		return false
	}
	for _, w := range words {
		if len(w) == 0 || w[0] < 'A' || w[0] > 'Z' {
			return false
		}
	}
	// Reject known non-person patterns: events, places, concepts
	lower := strings.ToLower(s)
	nonPersonWords := map[string]bool{
		"war": true, "battle": true, "revolution": true,
		"empire": true, "kingdom": true, "republic": true,
		"ocean": true, "sea": true, "river": true, "lake": true,
		"mountain": true, "island": true, "desert": true,
		"north": true, "south": true, "east": true, "west": true,
		"central": true, "new": true, "old": true,
		"world": true, "united": true, "great": true, "holy": true,
		"black": true, "white": true, "red": true, "blue": true,
		"age": true, "era": true, "period": true, "century": true,
		"mount": true, "cape": true, "fort": true, "port": true,
		"city": true, "state": true, "park": true, "tower": true,
	}
	for _, w := range strings.Fields(lower) {
		if nonPersonWords[w] {
			return false
		}
	}
	// Reject Roman numerals without a known person first name
	// ("World War II" → not a person, but "Henry VIII" → person)
	last := words[len(words)-1]
	if isRomanNumeral(last) {
		first := strings.ToLower(words[0])
		if !maleFirstNames[first] && !femaleFirstNames[first] {
			return false
		}
	}
	return true
}

// isRomanNumeral returns true for strings like "I", "II", "III", "IV", "V", etc.
func isRomanNumeral(s string) bool {
	if len(s) == 0 || len(s) > 5 {
		return false
	}
	for _, c := range s {
		if c != 'I' && c != 'V' && c != 'X' && c != 'L' && c != 'C' && c != 'D' && c != 'M' {
			return false
		}
	}
	return true
}

// isPersonReference returns true if the subject is a person name or a personal pronoun
// (covers all gendered and neutral pronoun forms).
func isPersonReference(s string) bool {
	lower := strings.ToLower(strings.TrimSpace(s))
	switch lower {
	case "he", "she", "him", "her", "his", "hers", "they", "them", "their":
		return true
	}
	return looksLikePersonName(s)
}

// Gender represents grammatical gender for pronoun selection.
type Gender int

const (
	GenderUnknown Gender = iota
	GenderMale
	GenderFemale
)

// Common female first names for gender detection.
var femaleFirstNames = map[string]bool{
	"marie": true, "maria": true, "mary": true, "ada": true, "elizabeth": true,
	"victoria": true, "catherine": true, "margaret": true, "jane": true, "anne": true,
	"anna": true, "emily": true, "charlotte": true, "rosa": true, "rosalind": true,
	"florence": true, "amelia": true, "cleopatra": true, "frida": true, "indira": true,
	"joan": true, "julia": true, "helen": true, "sophia": true, "alice": true,
	"angela": true, "diana": true, "eva": true, "grace": true, "harriet": true,
	"ida": true, "irene": true, "katharine": true, "katherine": true, "kate": true,
	"lise": true, "louise": true, "lucy": true, "martha": true, "nancy": true,
	"nefertiti": true, "olga": true, "rachel": true, "ruth": true, "sarah": true,
	"simone": true, "susan": true, "sylvia": true, "hedy": true, "hypatia": true,
	"dorothy": true, "emmy": true, "eleanor": true, "sojourner": true, "malala": true,
	"oprah": true, "venus": true, "serena": true, "valentina": true, "wangari": true,
	"aung": true, "benazir": true, "golda": true, "hildegard": true, "marie-curie": true,
	"queen": true, "empress": true, "mrs": true, "ms": true, "madame": true,
	"sister": true, "mother": true, "lady": true, "dame": true, "countess": true,
	"duchess": true, "princess": true, "baroness": true,
}

// Common male first names/titles for gender detection.
var maleFirstNames = map[string]bool{
	"king": true, "emperor": true, "mr": true, "sir": true, "lord": true,
	"prince": true, "duke": true, "baron": true, "count": true, "father": true,
	"pope": true, "saint": true, "brother": true,
}

// detectGender infers grammatical gender from a person's name.
func detectGender(name string) Gender {
	parts := strings.Fields(name)
	if len(parts) == 0 {
		return GenderUnknown
	}
	first := strings.ToLower(parts[0])
	first = strings.TrimSuffix(first, ".")
	if femaleFirstNames[first] {
		return GenderFemale
	}
	if maleFirstNames[first] {
		return GenderMale
	}
	// Default to male for unknown person names (statistical majority in
	// encyclopedic text). Use "they" only for truly unknown entities.
	return GenderMale
}

// genderPronoun returns the appropriate pronoun form for a gender.
func genderPronoun(g Gender) string {
	switch g {
	case GenderFemale:
		return "She"
	case GenderMale:
		return "He"
	default:
		return "They"
	}
}

// -----------------------------------------------------------------------
// Council & Opinion Integration Helpers
// -----------------------------------------------------------------------

// councilToneToComposerTone maps CouncilDeliberation.ResponseTone strings
// to the Composer's Tone enum values.
func councilToneToComposerTone(councilTone string) Tone {
	switch councilTone {
	case "direct":
		return ToneDirect
	case "empathetic":
		return ToneWarm
	case "cautious":
		return ToneNeutral
	case "enthusiastic":
		return ToneCasual
	case "analytical":
		return ToneNeutral
	default:
		return ToneNeutral
	}
}

// extractSpecifics pulls concrete details from user input that can be
// referenced in empathetic responses instead of using generic phrases.
// Returns a map with keys: "duration", "subject", "action".
func extractSpecifics(input string) map[string]string {
	result := make(map[string]string)
	lower := strings.ToLower(input)

	// Duration mentions: "3 days", "a week", "two months", etc.
	durationRe := regexp.MustCompile(`(?i)(\d+\s+(?:day|days|week|weeks|month|months|hour|hours|year|years|minute|minutes))|(?:a|an|one|two|three|four|five|six|seven|eight|nine|ten)\s+(?:day|days|week|weeks|month|months|hour|hours|year|years)`)
	if m := durationRe.FindString(lower); m != "" {
		result["duration"] = strings.TrimSpace(m)
	}

	// Subject nouns: first significant noun after "about", "on", "with", "my"
	subjectRe := regexp.MustCompile(`(?:about|on|with|my)\s+(?:the\s+)?(\w{3,})`)
	if m := subjectRe.FindStringSubmatch(lower); len(m) > 1 {
		word := m[1]
		// Filter out function words
		skipWords := map[string]bool{
			"this": true, "that": true, "the": true, "some": true,
			"what": true, "how": true, "very": true, "really": true,
			"just": true, "it": true, "them": true, "things": true,
		}
		if !skipWords[word] {
			result["subject"] = word
		}
	}

	// Action verbs: gerunds that indicate ongoing effort
	actionRe := regexp.MustCompile(`(?:been|keep|always|still)\s+(\w+ing)`)
	if m := actionRe.FindStringSubmatch(lower); len(m) > 1 {
		result["action"] = m[1]
	}

	return result
}

// composeSpecificEmpathy builds a response that references concrete details
// the user mentioned, making empathy feel personal rather than generic.
func (c *Composer) composeSpecificEmpathy(specifics map[string]string, sentiment Sentiment) string {
	duration := specifics["duration"]
	subject := specifics["subject"]
	action := specifics["action"]

	// Duration + subject: "Three days on a bug is genuinely draining."
	if duration != "" && subject != "" {
		templates := []string{
			"%s on %s is genuinely draining.",
			"%s dealing with %s — that takes real patience.",
			"After %s of %s, anyone would feel that way.",
		}
		if sentiment == SentimentPositive || sentiment == SentimentExcited {
			templates = []string{
				"%s on %s and still going strong — that's real dedication.",
				"%s with %s — you've clearly put in the work.",
			}
		}
		d := capitalizeFirst(duration)
		return fmt.Sprintf(c.pick(templates), d, subject)
	}

	// Duration alone: "Three days is a long time to sit with that."
	if duration != "" {
		templates := []string{
			"%s is a long time to sit with that.",
			"After %s, it's natural to feel worn down.",
			"%s — that's not nothing.",
		}
		if sentiment == SentimentPositive || sentiment == SentimentExcited {
			templates = []string{
				"%s of effort — and it paid off.",
				"%s well spent, by the sound of it.",
			}
		}
		return fmt.Sprintf(c.pick(templates), capitalizeFirst(duration))
	}

	// Subject + action: "Working on a project that matters is hard."
	if subject != "" && action != "" {
		templates := []string{
			"%s on %s — that takes more out of you than people realize.",
			"Still %s on %s says something about your commitment.",
			"The fact that you're still %s on %s matters.",
		}
		if sentiment == SentimentPositive || sentiment == SentimentExcited {
			templates = []string{
				"%s on %s and it's paying off — that's great to hear.",
				"All that %s on %s is clearly worth it.",
			}
		}
		return fmt.Sprintf(c.pick(templates), capitalizeFirst(action), subject)
	}

	// Subject alone: empathetic acknowledgment of what they're dealing with.
	if subject != "" {
		templates := []string{
			"I'm sorry you're going through that with %s.",
			"Dealing with %s is harder than it sounds.",
			"The %s situation sounds genuinely tough.",
		}
		if sentiment == SentimentPositive || sentiment == SentimentExcited {
			templates = []string{
				"Sounds like %s is going well — you should feel good about that.",
				"Good things happening with %s — that's worth celebrating.",
			}
		}
		return fmt.Sprintf(c.pick(templates), capitalizeFirst(subject))
	}

	// Action alone: "Working through that takes real stamina."
	if action != "" {
		templates := []string{
			"%s through that takes real stamina.",
			"The fact that you keep %s says a lot.",
		}
		return fmt.Sprintf(c.pick(templates), capitalizeFirst(action))
	}

	return ""
}

