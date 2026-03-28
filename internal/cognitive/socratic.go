package cognitive

import (
	"fmt"
	"regexp"
	"strings"
)

// -----------------------------------------------------------------------
// Socratic Dialogue Engine — a thinking partner, not an encyclopedia.
//
// Most AI systems try to answer every question. This engine detects when
// asking questions would be MORE valuable than answering. It implements
// five Socratic modes:
//
//   - Explore:    user is mapping unfamiliar territory — help them see the shape
//   - Decide:     user is choosing between options — help them clarify criteria
//   - Coach:      user wants to grow — help them discover their own insight
//   - Challenge:  user's reasoning has gaps — gently surface them
//   - Deepen:     user is surface-level on a deep topic — draw them in
//
// The default is SocraticNone. Socratic mode activates only when probing
// questions would genuinely advance the user's thinking beyond what a
// direct answer could.
// -----------------------------------------------------------------------

// SocraticMode describes WHY Socratic approach is appropriate.
type SocraticMode int

const (
	SocraticNone      SocraticMode = iota // just answer directly
	SocraticExplore                       // user is exploring, help them map the space
	SocraticDecide                        // user is deciding, help them clarify criteria
	SocraticCoach                         // user wants coaching, help them think through it
	SocraticChallenge                     // user's assumptions need gentle testing
	SocraticDeepen                        // user is surface-level, help them go deeper
)

func (m SocraticMode) String() string {
	switch m {
	case SocraticNone:
		return "none"
	case SocraticExplore:
		return "explore"
	case SocraticDecide:
		return "decide"
	case SocraticCoach:
		return "coach"
	case SocraticChallenge:
		return "challenge"
	case SocraticDeepen:
		return "deepen"
	default:
		return "unknown"
	}
}

// SocraticQuestion is a question designed to advance the user's thinking.
type SocraticQuestion struct {
	Text     string // the actual question text (topic-injected)
	Purpose  string // what this question is trying to reveal
	Depth    int    // 0=surface, 1=intermediate, 2=deep
	Category string // "clarify", "probe_assumptions", "explore_implications", "evaluate_evidence", "perspective_shift"
}

// SocraticResponse is the engine's output — questions instead of answers.
type SocraticResponse struct {
	Mode      SocraticMode
	Questions []SocraticQuestion
	Framing   string // optional intro sentence before questions
	Rationale string // why Socratic mode was chosen (for transparency)
}

// SocraticEngine detects when questions are more valuable than answers
// and generates probing questions that help the user think.
type SocraticEngine struct {
	questionBank    map[string][]questionTemplate // category → templates
	depthPatterns   []*regexp.Regexp              // detect when user needs depth
	decisionSignals []string                      // signals user is deciding
	coachingSignals []string                      // signals user wants coaching
	exploreSignals  []string                      // signals user is exploring
}

// questionTemplate is an internal template with a %s placeholder for topic injection.
type questionTemplate struct {
	Template string // contains %s for topic/entity injection
	Purpose  string
	Depth    int
}

// NewSocraticEngine creates a Socratic engine with a full question bank.
func NewSocraticEngine() *SocraticEngine {
	se := &SocraticEngine{
		questionBank: buildQuestionBank(),
		depthPatterns: []*regexp.Regexp{
			regexp.MustCompile(`(?i)^(what(?:'s| is| are) )\w+\??$`),                        // very short "what is X?"
			regexp.MustCompile(`(?i)^(what about|how about|what of) \w+\??$`),                // "what about life?"
			regexp.MustCompile(`(?i)^(tell me about|thoughts on|views on) \w[\w\s]{0,15}\??$`), // short open-ended
		},
		decisionSignals: []string{
			"should i",
			"which is better",
			"i'm torn between",
			"im torn between",
			"help me decide",
			"i can't choose",
			"i cant choose",
			"which one should",
			"should i go with",
			"which should i pick",
			"what would you choose",
			"i don't know which",
			"i dont know which",
			"between x and y",
			"or should i",
			"weighing my options",
			"pros and cons",
			"trade-offs between",
			"tradeoffs between",
		},
		coachingSignals: []string{
			"help me think about",
			"i'm struggling with",
			"im struggling with",
			"i want to improve",
			"how do i get better at",
			"i need to figure out",
			"i'm stuck on",
			"im stuck on",
			"i keep failing at",
			"how can i grow",
			"i want to learn how to",
			"i'm not sure how to approach",
			"im not sure how to approach",
			"help me work through",
			"help me understand my",
			"i need guidance on",
			"how should i think about",
			"i feel stuck",
			"feeling stuck",
			"stuck in my",
			"dont know what direction",
			"don't know what direction",
			"dont know where to start",
			"don't know where to start",
			"need direction",
			"lost in my career",
		},
		exploreSignals: []string{
			"i'm curious about",
			"im curious about",
			"what do you think about",
			"tell me your perspective on",
			"what's your take on",
			"whats your take on",
			"i've been wondering about",
			"ive been wondering about",
			"i want to explore",
			"what are the different ways to think about",
			"how do people think about",
			"what are the perspectives on",
		},
	}
	return se
}

// DetectMode analyzes whether a Socratic approach would be more valuable
// than a direct answer. Returns SocraticNone for most queries — Socratic
// mode is the exception, not the rule.
func (se *SocraticEngine) DetectMode(query string, state *ConversationState) SocraticMode {
	lower := strings.ToLower(strings.TrimSpace(query))

	// Fast exit: empty or very short factual queries should never be Socratic.
	if len(lower) < 3 {
		return SocraticNone
	}

	// Fast exit: direct factual questions ("what is 2+2", "who is X", "when did Y")
	if isDirectFactual(lower) {
		return SocraticNone
	}

	// Fast exit: commands and requests ("set a timer", "remind me", "search for")
	if isCommand(lower) {
		return SocraticNone
	}

	// Check decision signals first — highest confidence signal.
	for _, sig := range se.decisionSignals {
		if strings.Contains(lower, sig) {
			return SocraticDecide
		}
	}

	// Check coaching signals — user explicitly wants help thinking.
	for _, sig := range se.coachingSignals {
		if strings.Contains(lower, sig) {
			return SocraticCoach
		}
	}

	// Check exploration signals — user is curious, not asking for facts.
	for _, sig := range se.exploreSignals {
		if strings.Contains(lower, sig) {
			return SocraticExplore
		}
	}

	// Challenge detection: user is stuck in a loop (same topic 3+ turns).
	if state != nil && se.detectLoop(state) {
		return SocraticChallenge
	}

	// Challenge detection: strong unsupported claims.
	if containsStrongClaim(lower) {
		return SocraticChallenge
	}

	// Deepening detection: very short query on a big/abstract topic.
	if se.detectShallowBigTopic(lower) {
		return SocraticDeepen
	}

	return SocraticNone
}

// Generate produces 2-3 probing questions tailored to the detected mode
// and the user's actual topic. Questions are never generic — they reference
// the topic, entities, and context from the conversation state.
func (se *SocraticEngine) Generate(query string, mode SocraticMode, state *ConversationState) *SocraticResponse {
	if mode == SocraticNone {
		return &SocraticResponse{Mode: SocraticNone}
	}

	topic := extractTopic(query, state)
	entity := extractSocraticEntity(query, state)

	resp := &SocraticResponse{
		Mode:      mode,
		Questions: make([]SocraticQuestion, 0, 3),
	}

	switch mode {
	case SocraticExplore:
		resp.Framing = fmt.Sprintf("Let me help you explore %s more deeply.", topic)
		resp.Rationale = "You seem to be exploring this space — questions will help you map it better than a summary would."
		resp.Questions = se.pickQuestions(topic, entity, []string{"clarify", "perspective_shift", "explore_implications"}, 0)

	case SocraticDecide:
		resp.Framing = "Before I give you my take, let me help you clarify what matters most to you."
		resp.Rationale = "You're making a decision — the right answer depends on your values and constraints, which questions can surface."
		resp.Questions = se.pickQuestions(topic, entity, []string{"clarify", "explore_implications", "evaluate_evidence"}, 0)

	case SocraticCoach:
		resp.Framing = fmt.Sprintf("Let's think through %s together.", topic)
		resp.Rationale = "You're working through something — discovering your own answer will be more lasting than receiving mine."
		resp.Questions = se.pickQuestions(topic, entity, []string{"clarify", "probe_assumptions", "explore_implications"}, 1)

	case SocraticChallenge:
		resp.Framing = "I want to make sure we're building on solid ground."
		resp.Rationale = "There may be assumptions worth examining before going further."
		resp.Questions = se.pickQuestions(topic, entity, []string{"probe_assumptions", "evaluate_evidence", "perspective_shift"}, 1)

	case SocraticDeepen:
		resp.Framing = fmt.Sprintf("That's a rich topic. Let me help you find your angle on %s.", topic)
		resp.Rationale = "This topic has many dimensions — questions will help you find the aspect that matters most to you."
		resp.Questions = se.pickQuestions(topic, entity, []string{"clarify", "probe_assumptions", "perspective_shift"}, 0)
	}

	return resp
}

// ProgressiveDeepen generates questions that get deeper as the conversation
// progresses on the same topic. Early turns use surface-level clarifying
// questions; later turns probe assumptions and implications.
func (se *SocraticEngine) ProgressiveDeepen(turnCount int, topic string) []SocraticQuestion {
	entity := topic // use topic as entity for injection

	var depth int
	var categories []string

	switch {
	case turnCount <= 2:
		// Surface: help user articulate what they mean
		depth = 0
		categories = []string{"clarify"}
	case turnCount <= 4:
		// Intermediate: probe what's underneath
		depth = 1
		categories = []string{"probe_assumptions", "evaluate_evidence"}
	default:
		// Deep: implications, perspectives, synthesis
		depth = 2
		categories = []string{"explore_implications", "perspective_shift"}
	}

	return se.pickQuestions(topic, entity, categories, depth)
}

// ---------------------------------------------------------------------------
// Internal detection helpers
// ---------------------------------------------------------------------------

// isDirectFactual returns true for queries that just want a fact.
func isDirectFactual(lower string) bool {
	factualPrefixes := []string{
		"what is the capital of",
		"what is the population of",
		"who is the president",
		"who wrote",
		"who invented",
		"when was",
		"when did",
		"where is",
		"where was",
		"how many",
		"how much does",
		"how tall is",
		"how old is",
		"what year",
		"what time",
		"what date",
		"define ",
		"translate ",
		"convert ",
		"calculate ",
		"what is the formula",
		"what does the acronym",
	}
	for _, p := range factualPrefixes {
		if strings.HasPrefix(lower, p) {
			return true
		}
	}

	// Simple arithmetic or factual patterns
	factualPatterns := []*regexp.Regexp{
		regexp.MustCompile(`^what(?:'s| is) \d+[\s+\-*/x×]\s*\d+`), // "what is 2+2"
		regexp.MustCompile(`^\d+[\s]*[+\-*/x×]\s*\d+`),             // "2+2"
		regexp.MustCompile(`^(?:what|who|when|where|how many|how much)`), // starts with factual wh-word
	}

	// Phrases that look like wh-questions but are actually exploratory/opinon-seeking.
	opinionPhrases := []string{
		"what do you think", "what are your thoughts",
		"what's your take", "whats your take",
		"what would you", "what should i",
		"how do i get better", "how should i",
		"how do people think", "what are the different",
		"what are the perspectives",
	}
	for _, op := range opinionPhrases {
		if strings.HasPrefix(lower, op) {
			return false
		}
	}

	// Only the first two patterns are strong factual signals.
	// The third pattern (wh-questions) only applies for SHORT queries
	// that are clearly factual.
	for i, pat := range factualPatterns {
		if pat.MatchString(lower) {
			if i < 2 {
				return true
			}
			// For wh-questions, only flag as factual if short and doesn't
			// contain exploration/decision/coaching language or abstract topics.
			if len(lower) < 40 && !containsAbstractTopic(lower) {
				return true
			}
		}
	}

	return false
}

// isCommand returns true for action requests that should be executed, not discussed.
func isCommand(lower string) bool {
	commandPrefixes := []string{
		"set a ", "set the ", "create a ", "make a ", "open ",
		"search for ", "find ", "remind me", "show me",
		"play ", "stop ", "pause ", "turn on ", "turn off ",
		"send ", "call ", "text ", "email ",
		"add ", "remove ", "delete ", "save ",
	}
	for _, p := range commandPrefixes {
		if strings.HasPrefix(lower, p) {
			return true
		}
	}
	return false
}

// containsAbstractTopic returns true if the query touches abstract/philosophical territory.
func containsAbstractTopic(lower string) bool {
	abstract := []string{
		"life", "death", "meaning", "purpose", "happiness", "success",
		"love", "freedom", "justice", "truth", "beauty", "morality",
		"consciousness", "existence", "reality", "wisdom", "virtue",
		"ethics", "philosophy", "values", "fulfillment", "identity",
	}
	for _, a := range abstract {
		if strings.Contains(lower, a) {
			return true
		}
	}
	return false
}

// containsStrongClaim detects unsupported absolute statements.
func containsStrongClaim(lower string) bool {
	claimMarkers := []string{
		"obviously ", "clearly ", "everyone knows",
		"it's obvious that", "there's no way",
		"always ", "never ", " is the best",
		" is the worst", "nobody ", "everybody ",
		"the only way", "the only reason",
		"without a doubt", "undeniably",
	}
	for _, m := range claimMarkers {
		if strings.Contains(lower, m) {
			// But not if it's a question about the claim.
			if strings.Contains(lower, "?") || strings.HasPrefix(lower, "is it true") {
				return false
			}
			return true
		}
	}
	return false
}

// detectLoop checks if the conversation has been on the same topic for 3+ turns
// without meaningful progress (indicated by clarification or correction count).
func (se *SocraticEngine) detectLoop(state *ConversationState) bool {
	if state.TurnCount < 3 {
		return false
	}

	// Count how many of the top stack entries are the same topic
	if len(state.TopicStack) < 3 {
		return false
	}

	topic := strings.ToLower(state.ActiveTopic)
	if topic == "" {
		return false
	}

	sameCount := 0
	for i := 0; i < len(state.TopicStack) && i < 5; i++ {
		if strings.ToLower(state.TopicStack[i]) == topic {
			sameCount++
		}
	}

	return sameCount >= 3
}

// detectShallowBigTopic returns true when the user asks a very short question
// about a deep/abstract topic that would benefit from scoping.
func (se *SocraticEngine) detectShallowBigTopic(lower string) bool {
	// Must be short (under ~50 chars) to count as "shallow"
	if len(lower) > 50 {
		return false
	}

	// Must match a depth pattern
	for _, pat := range se.depthPatterns {
		if pat.MatchString(lower) {
			if containsAbstractTopic(lower) {
				return true
			}
		}
	}
	return false
}

// ---------------------------------------------------------------------------
// Topic and entity extraction
// ---------------------------------------------------------------------------

// extractTopic pulls the conversational topic from the query and state.
func extractTopic(query string, state *ConversationState) string {
	// Priority 1: extract from the CURRENT query — always prefer this over state.
	// State topics can be stale from previous turns.
	lower := strings.ToLower(strings.TrimSpace(query))
	stripPrefixes := []string{
		"should i ", "help me decide ", "help me think about ",
		"i'm struggling with ", "im struggling with ",
		"i'm curious about ", "im curious about ",
		"what do you think about ", "tell me about ",
		"i want to explore ", "thoughts on ",
		"what about ", "how about ",
		"i'm torn between ", "im torn between ",
		"help me work through ",
	}
	for _, p := range stripPrefixes {
		if strings.HasPrefix(lower, p) {
			topic := strings.TrimSpace(query[len(p):])
			topic = strings.TrimRight(topic, "?.!")
			if topic != "" {
				return topic
			}
		}
	}

	// Priority 2: use the full query (trimmed) as topic
	topic := strings.TrimRight(strings.TrimSpace(query), "?.!")
	if len(topic) > 80 {
		topic = topic[:80]
	}
	if topic != "" {
		return topic
	}

	// Priority 3 (fallback): state's active topic
	if state != nil && state.ActiveTopic != "" {
		return state.ActiveTopic
	}
	return "this"
}

// extractSocraticEntity gets the most relevant entity from state or query.
func extractSocraticEntity(query string, state *ConversationState) string {
	// Priority 1: extract from the current query
	lower := strings.ToLower(strings.TrimSpace(query))
	// Strip common prefixes to find the core entity
	for _, p := range []string{
		"help me decide about ", "help me decide between ",
		"should i ", "i'm torn between ", "im torn between ",
		"i am torn between ", "help me think about ",
		"i want to ", "i feel stuck in ", "i feel stuck on ",
		"i'm struggling with ", "im struggling with ",
		"i need help with ", "help me with ",
		"i dont know what to do about ", "i don't know what to do about ",
	} {
		if strings.HasPrefix(lower, p) {
			entity := strings.TrimSpace(query[len(p):])
			entity = strings.TrimRight(entity, "?.!")
			if entity != "" {
				return entity
			}
		}
	}

	// Priority 2: extract the main topic using the global topic extractor
	topic := extractMainTopic(query)
	if topic != "" && len(topic) < 80 {
		return topic
	}

	// Priority 3: use the query itself (shortened)
	cleaned := strings.TrimRight(strings.TrimSpace(query), "?.!")
	if len(cleaned) > 60 {
		cleaned = cleaned[:60]
	}
	if cleaned != "" {
		return cleaned
	}

	// Priority 3: fall back to state
	if state != nil && len(state.MentionedEntities) > 0 {
		if v, ok := state.MentionedEntities["topic"]; ok {
			return v
		}
	}
	return "this"
}

// ---------------------------------------------------------------------------
// Question bank construction — 15+ templates per category (75+ total)
// ---------------------------------------------------------------------------

func buildQuestionBank() map[string][]questionTemplate {
	bank := make(map[string][]questionTemplate, 5)

	// ---- CLARIFY (depth 0: surface) ----
	bank["clarify"] = []questionTemplate{
		{`When you say "%s," what specifically do you mean?`, "surface the user's precise definition", 0},
		{`What aspect of %s matters most to you right now?`, "narrow the scope to what's relevant", 0},
		{`What would a good outcome look like for you regarding %s?`, "clarify success criteria", 0},
		{`Are you asking about %s in general, or a specific situation you're facing?`, "distinguish abstract from concrete", 0},
		{`What prompted your interest in %s at this moment?`, "reveal the underlying motivation", 0},
		{`Is there a particular part of %s that feels unclear or confusing?`, "find the friction point", 0},
		{`When you picture %s going well, what does that look like?`, "make the ideal concrete", 0},
		{`Who else is affected by %s in your situation?`, "reveal stakeholders and scope", 0},
		{`What have you already tried or considered regarding %s?`, "avoid repeating known ground", 0},
		{`What would change for you if you understood %s better?`, "surface the stakes", 0},
		{`Is %s the core issue, or is it a symptom of something deeper?`, "test whether we're at the right level", 0},
		{`What timeframe are you thinking about for %s?`, "add temporal constraints", 0},
		{`How would you explain %s to someone unfamiliar with it?`, "force articulation of understanding", 0},
		{`What's the one thing about %s you wish you knew right now?`, "find the highest-value unknown", 0},
		{`On a scale of urgency, how pressing is %s for you?`, "gauge priority and emotional weight", 0},
		{`What do you already know about %s that you're confident in?`, "establish the known foundation", 0},
	}

	// ---- PROBE ASSUMPTIONS (depth 1: intermediate) ----
	bank["probe_assumptions"] = []questionTemplate{
		{`What are you assuming about %s that might not be true?`, "surface hidden assumptions", 1},
		{`What would have to be true for your view of %s to be wrong?`, "identify falsification criteria", 1},
		{`Where did your belief about %s originally come from?`, "trace the source of a belief", 1},
		{`Is it possible that %s works differently than you expect?`, "open space for alternative models", 1},
		{`What are you taking for granted about %s?`, "reveal the unexamined baseline", 1},
		{`If someone disagreed with your approach to %s, what would their strongest argument be?`, "steelman the opposition", 1},
		{`Are there conditions under which %s would NOT apply?`, "find boundary conditions", 1},
		{`What past experience is shaping how you see %s?`, "separate experience from current reality", 1},
		{`Is your view of %s based on evidence, intuition, or something you were told?`, "categorize the epistemic basis", 1},
		{`What would surprise you most about %s if you investigated further?`, "prime for discovery", 1},
		{`Are you assuming %s has to be either/or, or could it be both?`, "challenge false dichotomies", 1},
		{`What would change about your thinking on %s if you had more time?`, "separate urgency from truth", 1},
		{`Who benefits from the current way you think about %s?`, "reveal structural incentives", 1},
		{`What's the most common misconception about %s?`, "use meta-knowledge to check own beliefs", 1},
		{`Have you considered that your framing of %s might be limiting your options?`, "question the question itself", 1},
		{`What would a beginner notice about %s that an expert might overlook?`, "leverage fresh-eyes perspective", 1},
	}

	// ---- EXPLORE IMPLICATIONS (depth 2: deep) ----
	bank["explore_implications"] = []questionTemplate{
		{`If you chose this path with %s, what would that mean for you a year from now?`, "project long-term consequences", 2},
		{`What's the second-order effect of %s that people usually miss?`, "think beyond immediate impact", 2},
		{`If %s succeeds, what new problems does that create?`, "anticipate success costs", 2},
		{`What does %s make possible that wasn't possible before?`, "find positive externalities", 2},
		{`What does %s make impossible or harder?`, "find negative externalities", 2},
		{`How would your decision about %s look if you couldn't reverse it?`, "test commitment level", 2},
		{`Who would be most affected if you changed your approach to %s?`, "map the impact radius", 2},
		{`What's the worst realistic outcome of %s, and could you live with it?`, "calibrate risk tolerance", 2},
		{`If you had to teach someone your approach to %s, what would they struggle with?`, "find implicit complexity", 2},
		{`What does your approach to %s reveal about what you value?`, "connect decision to values", 2},
		{`How does %s connect to other things you care about?`, "find systemic relationships", 2},
		{`What would you need to give up to fully commit to %s?`, "surface trade-off costs", 2},
		{`If %s turned out to be wrong, what would you do next?`, "build resilience into thinking", 2},
		{`What precedent does your choice about %s set for future decisions?`, "consider pattern effects", 2},
		{`In what ways is %s simpler than you're making it, and in what ways is it more complex?`, "calibrate complexity perception", 2},
		{`What would you advise a friend who was in your exact position with %s?`, "gain distance from the problem", 2},
	}

	// ---- EVALUATE EVIDENCE (depth 1: intermediate) ----
	bank["evaluate_evidence"] = []questionTemplate{
		{`What evidence do you have that %s is true?`, "force evidence review", 1},
		{`How confident are you in your understanding of %s, on a scale of 1-10?`, "calibrate confidence", 1},
		{`Have you seen %s work in practice, or is this theoretical?`, "distinguish theory from experience", 1},
		{`What data would change your mind about %s?`, "define the update trigger", 1},
		{`Are you basing your view of %s on a single example or a pattern?`, "check sample size", 1},
		{`What's the strongest piece of evidence against your position on %s?`, "steelman the counterevidence", 1},
		{`How recent is your information about %s?`, "check for stale data", 1},
		{`Could your evidence about %s be explained by something else entirely?`, "consider alternative explanations", 1},
		{`If you had to bet real money on %s, would you?`, "force skin-in-the-game calibration", 1},
		{`Who would you trust most to give you the truth about %s?`, "identify credible sources", 1},
		{`Is your view of %s based on what usually happens, or what happened to you?`, "separate base rates from anecdotes", 1},
		{`What would you need to see to feel confident about %s?`, "define evidence requirements", 1},
		{`Are there experiments you could run to test your beliefs about %s?`, "move from opinion to empiricism", 1},
		{`How much of your certainty about %s comes from repetition vs. evidence?`, "check for availability bias", 1},
		{`What's the most important thing you DON'T know about %s?`, "map the unknown unknowns", 1},
	}

	// ---- PERSPECTIVE SHIFT (depth 2: deep) ----
	bank["perspective_shift"] = []questionTemplate{
		{`How would someone who disagrees with you see %s?`, "force empathetic counter-modeling", 2},
		{`What would someone from a completely different background think about %s?`, "break cultural assumptions", 2},
		{`If you had to argue the opposite side of %s, what would you say?`, "strengthen through opposition", 2},
		{`How would your future self look back on your current thinking about %s?`, "use temporal distance for clarity", 2},
		{`What would a child's perspective on %s reveal?`, "strip away accumulated complexity", 2},
		{`If you removed emotion from the equation, how would %s look different?`, "separate rational from emotional", 2},
		{`What would someone who has already solved %s tell you?`, "borrow from imagined expertise", 2},
		{`How would this look if %s were someone else's problem, not yours?`, "gain emotional distance", 2},
		{`What historical parallel exists for %s, and how did it play out?`, "learn from analogous situations", 2},
		{`If you zoomed out ten years, how important is %s really?`, "test significance through temporal zoom", 2},
		{`What would you think about %s if you had unlimited resources?`, "remove constraint bias", 2},
		{`How would the person you most admire approach %s?`, "borrow from role models", 2},
		{`What's the version of %s that scares you, and why?`, "surface hidden fears shaping judgment", 2},
		{`If you were advising a competitor on %s, what would you honestly tell them?`, "bypass self-serving bias", 2},
		{`What about %s are you avoiding thinking about?`, "surface deliberate blind spots", 2},
		{`If %s were a problem in a completely different domain, what solutions would be obvious?`, "use cross-domain transfer", 2},
	}

	return bank
}

// ---------------------------------------------------------------------------
// Question selection and injection
// ---------------------------------------------------------------------------

// pickQuestions selects 2-3 questions from the specified categories, injects
// the topic/entity, and returns them. It picks one question from each category
// to ensure diversity of questioning angle.
func (se *SocraticEngine) pickQuestions(topic, entity string, categories []string, minDepth int) []SocraticQuestion {
	questions := make([]SocraticQuestion, 0, 3)

	// Use a deterministic but varied selection: hash-based index from topic.
	seed := topicHash(topic)

	for i, cat := range categories {
		if len(questions) >= 3 {
			break
		}

		templates, ok := se.questionBank[cat]
		if !ok || len(templates) == 0 {
			continue
		}

		// Filter templates by minimum depth
		eligible := make([]questionTemplate, 0, len(templates))
		for _, t := range templates {
			if t.Depth >= minDepth {
				eligible = append(eligible, t)
			}
		}
		if len(eligible) == 0 {
			// Fall back to all templates in this category
			eligible = templates
		}

		// Select using (seed + category index) to vary across categories
		idx := (seed + i) % len(eligible)
		tmpl := eligible[idx]

		// Inject topic — use entity if it's more specific
		inject := topic
		if entity != "" && entity != topic && len(entity) < len(topic) {
			inject = entity
		}

		questions = append(questions, SocraticQuestion{
			Text:     fmt.Sprintf(tmpl.Template, inject),
			Purpose:  tmpl.Purpose,
			Depth:    tmpl.Depth,
			Category: cat,
		})
	}

	return questions
}

// topicHash produces a simple deterministic hash from a string.
// Not cryptographic — just for varied-but-reproducible selection.
func topicHash(s string) int {
	h := 0
	for _, c := range s {
		h = h*31 + int(c)
	}
	if h < 0 {
		h = -h
	}
	return h
}
