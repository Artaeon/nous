package cognitive

import (
	"fmt"
	"math"
	"math/rand"
	"strings"

	"github.com/artaeon/nous/internal/memory"
)

// -----------------------------------------------------------------------
// Inner Council — multiple internal perspectives that evaluate a situation
// before responding. Instead of single-pass response generation, the
// council debates from different angles and an arbiter synthesizes the
// best response strategy.
//
// Five perspectives:
//   - Pragmatist: "What's the practical answer?" (knowledge graph)
//   - Historian:  "What does past experience say?" (episodic memory)
//   - Empath:     "How does the user feel?" (subtext analysis)
//   - Architect:  "What's the big-picture view?" (sparks + connections)
//   - Skeptic:    "What could go wrong?" (devil's advocate)
//
// The arbiter combines them into a single coherent strategy. The council
// runs on every non-trivial query — it must be fast. Each perspective is
// 5-15 lines of logic. The power comes from combining simple viewpoints.
// -----------------------------------------------------------------------

// Perspective represents a cognitive viewpoint.
type Perspective int

const (
	PerspPragmatist Perspective = iota // "What's the practical answer?"
	PerspHistorian                     // "What does past experience say?"
	PerspEmpath                        // "How does the user feel?"
	PerspArchitect                     // "What's the systemic/big-picture view?"
	PerspSkeptic                       // "What could go wrong? What are we missing?"
)

// CouncilOpinion is one perspective's assessment.
type CouncilOpinion struct {
	Perspective Perspective
	Assessment  string  // the perspective's take (1-2 sentences)
	Confidence  float64 // 0.0-1.0
	KeyInsight  string  // one-line core insight
	Priority    float64 // how relevant this perspective is to the current query
}

// CouncilDeliberation is the result of the full council debate.
type CouncilDeliberation struct {
	Opinions     []CouncilOpinion
	Dominant     Perspective // which perspective won
	Synthesis    string      // the arbiter's combined recommendation
	ResponseTone string     // recommended tone: "direct", "empathetic", "cautious", "enthusiastic", "analytical"
	ShouldAsk    bool       // should Nous ask a clarifying question instead of answering?
	AskWhat      string     // if ShouldAsk, what to ask
	Trace        string     // reasoning trace (for debugging/transparency)
}

// InnerCouncil runs multi-perspective evaluation before response generation.
type InnerCouncil struct {
	graph    *CognitiveGraph
	episodic *memory.EpisodicMemory
	rng      *rand.Rand
}

// NewInnerCouncil creates a council wired to the knowledge graph and episodic memory.
func NewInnerCouncil(graph *CognitiveGraph, episodic *memory.EpisodicMemory) *InnerCouncil {
	return &InnerCouncil{
		graph:    graph,
		episodic: episodic,
		rng:      rand.New(rand.NewSource(42)),
	}
}

// Deliberate consults all 5 perspectives, scores them, runs the arbiter,
// and returns a synthesized recommendation.
func (ic *InnerCouncil) Deliberate(input string, nluResult *NLUResult, ctx *ComposeContext) *CouncilDeliberation {
	// Extract topics from NLU and context.
	var topics []string
	if nluResult != nil {
		for _, v := range nluResult.Entities {
			topics = append(topics, v)
		}
	}
	if ctx != nil {
		topics = append(topics, ctx.RecentTopics...)
	}

	// Extract subtext and sparks from context.
	var subtext *SubtextAnalysis
	var sparks []AssociativeSpark
	if ctx != nil {
		subtext = ctx.Subtext
		sparks = ctx.Sparks
	}

	// Consult each perspective.
	pragmatist := ic.consultPragmatist(input, topics)
	historian := ic.consultHistorian(input)
	empath := ic.consultEmpath(input, subtext)
	architect := ic.consultArchitect(input, topics, sparks)

	// Skeptic reviews the others.
	others := []CouncilOpinion{pragmatist, historian, empath, architect}
	skeptic := ic.consultSkeptic(input, others)

	opinions := []CouncilOpinion{pragmatist, historian, empath, architect, skeptic}

	return ic.arbitrate(opinions, subtext)
}

// -----------------------------------------------------------------------
// Perspective consultants
// -----------------------------------------------------------------------

// consultPragmatist looks up facts in the knowledge graph and provides
// the direct, practical answer.
func (ic *InnerCouncil) consultPragmatist(input string, topics []string) CouncilOpinion {
	if ic.graph == nil {
		return CouncilOpinion{
			Perspective: PerspPragmatist,
			Assessment:  "No knowledge graph available.",
			Confidence:  0.1,
			KeyInsight:  "No factual grounding.",
			Priority:    0.2,
		}
	}

	totalFacts := 0
	var bestTopic string
	var bestCount int

	// Check each topic for available facts.
	checked := make(map[string]bool)
	for _, topic := range topics {
		lower := strings.ToLower(topic)
		if checked[lower] {
			continue
		}
		checked[lower] = true
		facts := ic.graph.LookupFacts(topic, 10)
		totalFacts += len(facts)
		if len(facts) > bestCount {
			bestCount = len(facts)
			bestTopic = topic
		}
	}

	// Also check the raw input as a topic.
	inputWords := extractTopicWords(strings.ToLower(input))
	for _, w := range inputWords {
		if checked[w] {
			continue
		}
		checked[w] = true
		facts := ic.graph.LookupFacts(w, 10)
		totalFacts += len(facts)
		if len(facts) > bestCount {
			bestCount = len(facts)
			bestTopic = w
		}
	}

	if totalFacts == 0 {
		return CouncilOpinion{
			Perspective: PerspPragmatist,
			Assessment:  "This topic isn't well covered in our knowledge base yet.",
			Confidence:  0.15,
			KeyInsight:  "Limited factual grounding available.",
			Priority:    0.3,
		}
	}

	confidence := math.Min(0.9, 0.4+float64(totalFacts)*0.08)
	priority := math.Min(0.9, 0.5+float64(totalFacts)*0.06)

	return CouncilOpinion{
		Perspective: PerspPragmatist,
		Assessment:  fmt.Sprintf("We have %d facts available. Lead with %s (%d facts).", totalFacts, bestTopic, bestCount),
		Confidence:  confidence,
		KeyInsight:  fmt.Sprintf("Strong factual grounding on %s.", bestTopic),
		Priority:    priority,
	}
}

// consultHistorian searches episodic memory for past interactions on this topic.
func (ic *InnerCouncil) consultHistorian(input string) CouncilOpinion {
	if ic.episodic == nil {
		return CouncilOpinion{
			Perspective: PerspHistorian,
			Assessment:  "No episodic memory available.",
			Confidence:  0.1,
			KeyInsight:  "No history.",
			Priority:    0.1,
		}
	}

	episodes := ic.episodic.SearchKeyword(input, 10)
	if len(episodes) == 0 {
		return CouncilOpinion{
			Perspective: PerspHistorian,
			Assessment:  "No prior interactions on this topic.",
			Confidence:  0.3,
			KeyInsight:  "Fresh topic.",
			Priority:    0.2,
		}
	}

	// Count how many were successful.
	successCount := 0
	for _, ep := range episodes {
		if ep.Success {
			successCount++
		}
	}

	assessment := ""
	keyInsight := ""
	confidence := 0.0
	priority := 0.0

	switch {
	case len(episodes) >= 3 && successCount < len(episodes)/2:
		// Repeated topic with poor success — previous answers weren't helpful.
		assessment = fmt.Sprintf("User asked about this %d times before. Only %d/%d were successful — previous answers may not have been helpful.",
			len(episodes), successCount, len(episodes))
		keyInsight = "Repeated struggle. Try a different approach."
		confidence = 0.7
		priority = 0.8
	case len(episodes) >= 3 && successCount >= len(episodes)/2:
		// Repeated topic but success — they keep coming back to it.
		assessment = fmt.Sprintf("User has asked about this %d times, mostly successfully. This is a recurring interest.",
			len(episodes))
		keyInsight = "Recurring topic, well-handled before."
		confidence = 0.6
		priority = 0.5
	case len(episodes) >= 1:
		// Some history.
		if episodes[0].Success {
			assessment = fmt.Sprintf("User asked about this before (%d prior). Last interaction was successful.",
				len(episodes))
			keyInsight = "Prior success — similar approach should work."
		} else {
			assessment = fmt.Sprintf("User asked about this before (%d prior). Last interaction was not successful.",
				len(episodes))
			keyInsight = "Previous attempt failed — adjust approach."
		}
		confidence = 0.5
		priority = 0.5
	}

	return CouncilOpinion{
		Perspective: PerspHistorian,
		Assessment:  assessment,
		Confidence:  confidence,
		KeyInsight:  keyInsight,
		Priority:    priority,
	}
}

// consultEmpath reads the subtext analysis and focuses on emotional state,
// implied needs, and behavioral signals.
func (ic *InnerCouncil) consultEmpath(input string, subtext *SubtextAnalysis) CouncilOpinion {
	if subtext == nil {
		return CouncilOpinion{
			Perspective: PerspEmpath,
			Assessment:  "No subtext analysis available.",
			Confidence:  0.1,
			KeyInsight:  "Cannot read emotional state.",
			Priority:    0.1,
		}
	}

	emotion := subtext.EmotionalState
	signals := subtext.Signals

	// Strong emotional signal — empath should lead.
	emotionStrength := math.Abs(emotion.Valence) + emotion.Arousal
	if emotionStrength < 0.3 && len(signals) == 0 {
		return CouncilOpinion{
			Perspective: PerspEmpath,
			Assessment:  "User seems emotionally neutral. No strong signals detected.",
			Confidence:  0.4,
			KeyInsight:  "Calm and factual mode.",
			Priority:    0.2,
		}
	}

	// Build assessment from signals — human-readable language only,
	// since this text can leak into the Synthesis which the user sees.
	var parts []string
	if emotion.Dominant != "neutral" {
		parts = append(parts, fmt.Sprintf("This person seems to be feeling %s right now", emotion.Dominant))
	}
	for _, sig := range signals {
		if sig.Weight >= 0.4 {
			parts = append(parts, fmt.Sprintf("there are signs of %s", sig.Type))
		}
	}
	if subtext.ImpliedNeed != "" {
		switch subtext.ImpliedNeed {
		case NeedVenting:
			parts = append(parts, "they might appreciate space to vent")
		case NeedReassurance:
			parts = append(parts, "they might appreciate some reassurance")
		case NeedValidation:
			parts = append(parts, "they might appreciate validation")
		case NeedCelebration:
			parts = append(parts, "they might appreciate celebrating with them")
		default:
			parts = append(parts, fmt.Sprintf("they might appreciate %s", strings.ReplaceAll(string(subtext.ImpliedNeed), "_", " ")))
		}
	}

	assessment := strings.Join(parts, "; ") + "."

	// Determine key insight based on implied need.
	keyInsight := "Acknowledge emotional state before information."
	if subtext.ImpliedNeed == NeedVenting {
		keyInsight = "User is venting — listen, don't fix."
	} else if subtext.ImpliedNeed == NeedReassurance {
		keyInsight = "User needs reassurance, not more data."
	} else if subtext.ImpliedNeed == NeedValidation {
		keyInsight = "User wants validation of their thinking."
	} else if subtext.ImpliedNeed == NeedCelebration {
		keyInsight = "User is sharing good news — celebrate with them."
	}

	confidence := subtext.Confidence
	priority := math.Min(0.95, emotionStrength*0.6+float64(len(signals))*0.1)

	return CouncilOpinion{
		Perspective: PerspEmpath,
		Assessment:  assessment,
		Confidence:  confidence,
		KeyInsight:  keyInsight,
		Priority:    priority,
	}
}

// consultArchitect takes the big picture view: connections between topics,
// patterns, and the user's larger context.
func (ic *InnerCouncil) consultArchitect(input string, topics []string, sparks []AssociativeSpark) CouncilOpinion {
	if len(topics) == 0 && len(sparks) == 0 {
		return CouncilOpinion{
			Perspective: PerspArchitect,
			Assessment:  "No topics or connections to analyze.",
			Confidence:  0.2,
			KeyInsight:  "Isolated query.",
			Priority:    0.1,
		}
	}

	var assessment string
	var keyInsight string
	confidence := 0.3
	priority := 0.3

	// Check for cross-topic connections via sparks.
	if len(sparks) > 0 {
		best := sparks[0]
		for _, s := range sparks[1:] {
			if s.Novelty > best.Novelty {
				best = s
			}
		}
		assessment = fmt.Sprintf("Found connection: %s links to %s (novelty=%.2f). %s",
			best.Source, best.Target, best.Novelty, best.Explanation)
		keyInsight = fmt.Sprintf("Frame answer in context of %s-%s connection.", best.Source, best.Target)
		confidence = math.Min(0.8, 0.4+best.Novelty*0.4)
		priority = math.Min(0.8, 0.3+best.Novelty*0.4)
	} else if len(topics) >= 2 {
		// Multiple topics suggest a broader question.
		assessment = fmt.Sprintf("Query spans %d topics (%s). Consider the connections between them.",
			len(topics), strings.Join(topics, ", "))
		keyInsight = "Multi-topic query — synthesize, don't isolate."
		confidence = 0.5
		priority = 0.5
	} else {
		assessment = fmt.Sprintf("Single topic: %s. No broader connections found.", topics[0])
		keyInsight = "Straightforward, no systemic view needed."
		confidence = 0.4
		priority = 0.2
	}

	return CouncilOpinion{
		Perspective: PerspArchitect,
		Assessment:  assessment,
		Confidence:  confidence,
		KeyInsight:  keyInsight,
		Priority:    priority,
	}
}

// consultSkeptic reviews the other opinions and identifies potential problems:
// missing information, assumptions, contradictions, overconfidence.
func (ic *InnerCouncil) consultSkeptic(input string, others []CouncilOpinion) CouncilOpinion {
	// Check for overconfidence — if everyone is very confident, that's suspicious.
	avgConfidence := 0.0
	activeCount := 0
	for _, o := range others {
		if o.Confidence > 0.15 { // skip inactive perspectives
			avgConfidence += o.Confidence
			activeCount++
		}
	}
	if activeCount > 0 {
		avgConfidence /= float64(activeCount)
	}

	// Check for disagreement between perspectives.
	highPriorityCount := 0
	for _, o := range others {
		if o.Priority >= 0.6 {
			highPriorityCount++
		}
	}

	// Check if the input has multiple valid interpretations.
	hasQuestion := strings.Contains(input, "?")
	words := strings.Fields(input)
	isShort := len(words) <= 3
	isAmbiguous := isShort && !hasQuestion

	// Analyze for pronouns without clear referents.
	lower := strings.ToLower(input)
	hasVaguePronouns := false
	vaguePronouns := []string{" it ", " that ", " this ", " they "}
	for _, p := range vaguePronouns {
		if strings.Contains(" "+lower+" ", p) {
			hasVaguePronouns = true
			break
		}
	}

	// Build the skeptic's assessment.
	var concerns []string

	if avgConfidence > 0.75 && activeCount >= 3 {
		concerns = append(concerns, "all perspectives are very confident — verify assumptions")
	}

	if highPriorityCount >= 3 {
		concerns = append(concerns, "multiple perspectives competing for dominance — complex query")
	}

	if isAmbiguous {
		concerns = append(concerns, "very short input with no question mark — intent unclear")
	}

	if hasVaguePronouns && isShort {
		concerns = append(concerns, "vague pronoun without clear referent")
	}

	// If pragmatist has no facts and historian has no history, we're flying blind.
	pragConf := others[0].Confidence // PerspPragmatist is first
	histConf := others[1].Confidence // PerspHistorian is second
	if pragConf < 0.2 && histConf < 0.2 {
		concerns = append(concerns, "no factual grounding and no history — high risk of unhelpful answer")
	}

	if len(concerns) == 0 {
		return CouncilOpinion{
			Perspective: PerspSkeptic,
			Assessment:  "No significant concerns. Proceed with confidence.",
			Confidence:  0.3,
			KeyInsight:  "All clear.",
			Priority:    0.1,
		}
	}

	confidence := math.Min(0.85, 0.3+float64(len(concerns))*0.2)
	priority := math.Min(0.85, 0.2+float64(len(concerns))*0.2)

	return CouncilOpinion{
		Perspective: PerspSkeptic,
		Assessment:  strings.Join(concerns, "; ") + ".",
		Confidence:  confidence,
		KeyInsight:  concerns[0],
		Priority:    priority,
		// The caller checks ShouldAsk via the arbiter — we store it in the struct below.
	}
}

// -----------------------------------------------------------------------
// Arbiter — synthesizes the council's opinions into a strategy.
// -----------------------------------------------------------------------

// arbitrate combines perspectives into a single coherent deliberation.
func (ic *InnerCouncil) arbitrate(opinions []CouncilOpinion, subtext *SubtextAnalysis) *CouncilDeliberation {
	delib := &CouncilDeliberation{
		Opinions: opinions,
	}

	// Index opinions by perspective for easy access.
	byPersp := make(map[Perspective]*CouncilOpinion)
	for i := range opinions {
		byPersp[opinions[i].Perspective] = &opinions[i]
	}

	pragmatist := byPersp[PerspPragmatist]
	historian := byPersp[PerspHistorian]
	empath := byPersp[PerspEmpath]
	architect := byPersp[PerspArchitect]
	skeptic := byPersp[PerspSkeptic]

	var trace []string

	// Rule 1: If Empath has high priority AND subtext shows negative valence, Empath leads.
	if subtext != nil && empath.Priority >= 0.5 && subtext.EmotionalState.Valence < -0.2 {
		delib.Dominant = PerspEmpath
		delib.ResponseTone = "empathetic"
		trace = append(trace, fmt.Sprintf("empath leads: priority=%.2f, valence=%.2f", empath.Priority, subtext.EmotionalState.Valence))
	}

	// Rule 2: If Skeptic recommends asking AND has reasonable confidence, set ShouldAsk.
	if skeptic.Confidence >= 0.5 && skeptic.Priority >= 0.4 {
		// Check if skeptic flagged ambiguity (indicated by concerns about short/vague input).
		if strings.Contains(skeptic.Assessment, "intent unclear") || strings.Contains(skeptic.Assessment, "vague pronoun") {
			delib.ShouldAsk = true
			// Extract the ask from the skeptic's assessment context.
			delib.AskWhat = ic.extractAskWhat(skeptic)
			trace = append(trace, fmt.Sprintf("skeptic recommends asking: conf=%.2f", skeptic.Confidence))
		}
	}

	// Rule 3: If Historian shows repeated questions with poor success, adjust approach.
	if historian.Priority >= 0.7 && strings.Contains(historian.KeyInsight, "different approach") {
		if delib.Dominant == 0 && delib.ResponseTone == "" {
			delib.Dominant = PerspHistorian
			delib.ResponseTone = "cautious"
		}
		trace = append(trace, fmt.Sprintf("historian flags repeated struggle: priority=%.2f", historian.Priority))
	}

	// Rule 4: If Pragmatist has high confidence and no dominant yet, lead with facts.
	if delib.ResponseTone == "" && pragmatist.Confidence >= 0.5 {
		delib.Dominant = PerspPragmatist
		delib.ResponseTone = "direct"
		trace = append(trace, fmt.Sprintf("pragmatist leads: conf=%.2f", pragmatist.Confidence))
	}

	// Rule 5: If Architect found a connection, weave it in.
	if architect.Priority >= 0.5 {
		trace = append(trace, fmt.Sprintf("architect connection: priority=%.2f", architect.Priority))
		// If no dominant yet, architect takes over.
		if delib.ResponseTone == "" {
			delib.Dominant = PerspArchitect
			delib.ResponseTone = "analytical"
		}
	}

	// Fallback: if still no tone, default to direct.
	if delib.ResponseTone == "" {
		delib.Dominant = PerspPragmatist
		delib.ResponseTone = "direct"
		trace = append(trace, "fallback: direct tone")
	}

	// Override tone for positive high-arousal (enthusiasm).
	if subtext != nil && subtext.EmotionalState.Valence > 0.3 && subtext.EmotionalState.Arousal > 0.4 {
		delib.ResponseTone = "enthusiastic"
		trace = append(trace, "tone override: enthusiastic (positive+aroused)")
	}

	// Build synthesis from dominant + supporting insights.
	dominant := byPersp[delib.Dominant]
	var synthesis []string
	synthesis = append(synthesis, dominant.Assessment)

	// Add supporting insights from other high-priority perspectives.
	for _, o := range opinions {
		if o.Perspective == delib.Dominant {
			continue
		}
		if o.Priority >= 0.4 && o.KeyInsight != "" && o.KeyInsight != "All clear." {
			synthesis = append(synthesis, o.KeyInsight)
		}
	}

	delib.Synthesis = strings.Join(synthesis, " ")
	delib.Trace = strings.Join(trace, " -> ")

	return delib
}

// extractAskWhat determines what clarifying question to ask based on the skeptic's assessment.
func (ic *InnerCouncil) extractAskWhat(skeptic *CouncilOpinion) string {
	if strings.Contains(skeptic.Assessment, "vague pronoun") {
		return "What are you referring to?"
	}
	if strings.Contains(skeptic.Assessment, "intent unclear") {
		return "Could you tell me more about what you mean?"
	}
	return "Could you clarify what you're looking for?"
}

// perspectiveName returns a human-readable name for a perspective.
func perspectiveName(p Perspective) string {
	switch p {
	case PerspPragmatist:
		return "Pragmatist"
	case PerspHistorian:
		return "Historian"
	case PerspEmpath:
		return "Empath"
	case PerspArchitect:
		return "Architect"
	case PerspSkeptic:
		return "Skeptic"
	default:
		return "Unknown"
	}
}

// -----------------------------------------------------------------------
// Multi-Round Debate — perspectives respond to each other.
//
// Unlike the standard single-pass Deliberate, DebateRounds lets
// perspectives see each other's opinions and refine their positions.
// Each round, perspectives can:
//   - Strengthen: add evidence for their position
//   - Concede: lower confidence and defer to stronger arguments
//   - Challenge: raise specific objections to another perspective
//   - Synthesize: propose a combined position
// -----------------------------------------------------------------------

// DebateResult captures the full multi-round debate.
type DebateResult struct {
	Rounds       []DebateRound
	FinalOpinions []CouncilOpinion
	Consensus    string      // the point of agreement (if any)
	Dissent      string      // remaining disagreement (if any)
	Dominant     Perspective // which perspective won after all rounds
	Confidence   float64     // aggregate confidence
	Trace        string      // full debate trace
}

// DebateRound captures one round of the debate.
type DebateRound struct {
	Round    int
	Moves    []DebateMove
	Shifting bool   // did any perspective change position?
	Summary  string // one-line summary of the round
}

// DebateMove is a single perspective's action in a round.
type DebateMove struct {
	Perspective Perspective
	MoveType    string  // "strengthen", "concede", "challenge", "synthesize"
	Target      Perspective // who they're responding to (for challenge)
	Argument    string
	NewConfidence float64
}

// Debate runs a multi-round deliberation where perspectives interact.
// More expensive than Deliberate but produces deeper reasoning.
func (ic *InnerCouncil) Debate(input string, nluResult *NLUResult, ctx *ComposeContext, rounds int) *DebateResult {
	if rounds < 1 {
		rounds = 1
	}
	if rounds > 5 {
		rounds = 5 // diminishing returns after 5 rounds
	}

	// Start with standard opinion gathering.
	var topics []string
	if nluResult != nil {
		for _, v := range nluResult.Entities {
			topics = append(topics, v)
		}
	}
	if ctx != nil {
		topics = append(topics, ctx.RecentTopics...)
	}

	var subtext *SubtextAnalysis
	var sparks []AssociativeSpark
	if ctx != nil {
		subtext = ctx.Subtext
		sparks = ctx.Sparks
	}

	// Initial opinions (round 0).
	opinions := []CouncilOpinion{
		ic.consultPragmatist(input, topics),
		ic.consultHistorian(input),
		ic.consultEmpath(input, subtext),
		ic.consultArchitect(input, topics, sparks),
	}
	skeptic := ic.consultSkeptic(input, opinions)
	opinions = append(opinions, skeptic)

	result := &DebateResult{
		Rounds: make([]DebateRound, 0, rounds),
	}

	var traceLines []string
	traceLines = append(traceLines, "Initial positions established")

	// Run debate rounds.
	for r := 0; r < rounds; r++ {
		round := DebateRound{Round: r + 1}
		anyShift := false

		// Each perspective evaluates others and may adjust.
		for i := range opinions {
			move := ic.debateMove(&opinions[i], opinions, r)
			if move != nil {
				round.Moves = append(round.Moves, *move)
				// Apply the move.
				if move.NewConfidence != opinions[i].Confidence {
					anyShift = true
				}
				opinions[i].Confidence = move.NewConfidence
				if move.Argument != "" {
					opinions[i].Assessment = move.Argument
				}
			}
		}

		round.Shifting = anyShift
		round.Summary = ic.summarizeRound(round)
		result.Rounds = append(result.Rounds, round)

		traceLines = append(traceLines, fmt.Sprintf("Round %d: %s", r+1, round.Summary))

		// Early termination: if no perspective shifted, consensus reached.
		if !anyShift {
			traceLines = append(traceLines, "Consensus reached — no further shifts")
			break
		}
	}

	result.FinalOpinions = opinions

	// Find dominant perspective (highest confidence * priority).
	bestScore := 0.0
	for _, o := range opinions {
		score := o.Confidence * o.Priority
		if score > bestScore {
			bestScore = score
			result.Dominant = o.Perspective
		}
	}
	result.Confidence = bestScore

	// Build consensus and dissent.
	result.Consensus, result.Dissent = ic.findConsensusAndDissent(opinions)
	result.Trace = strings.Join(traceLines, " -> ")

	return result
}

// debateMove determines how a perspective responds to the current state.
func (ic *InnerCouncil) debateMove(self *CouncilOpinion, all []CouncilOpinion, round int) *DebateMove {
	// Find the strongest other opinion.
	var strongest *CouncilOpinion
	for i := range all {
		if all[i].Perspective == self.Perspective {
			continue
		}
		if strongest == nil || (all[i].Confidence*all[i].Priority) > (strongest.Confidence*strongest.Priority) {
			strongest = &all[i]
		}
	}

	if strongest == nil {
		return nil
	}

	strongestScore := strongest.Confidence * strongest.Priority
	selfScore := self.Confidence * self.Priority

	// Decision logic:
	switch {
	case selfScore >= strongestScore*1.3:
		// We're significantly stronger — strengthen our position.
		return &DebateMove{
			Perspective:   self.Perspective,
			MoveType:      "strengthen",
			Argument:      self.Assessment, // keep position
			NewConfidence: math.Min(0.95, self.Confidence+0.05),
		}

	case strongestScore >= selfScore*1.5:
		// Other perspective is much stronger — concede.
		concession := fmt.Sprintf("Deferring to %s: %s",
			perspectiveName(strongest.Perspective), strongest.KeyInsight)
		return &DebateMove{
			Perspective:   self.Perspective,
			MoveType:      "concede",
			Target:        strongest.Perspective,
			Argument:      concession,
			NewConfidence: math.Max(0.1, self.Confidence-0.15),
		}

	case self.Perspective == PerspSkeptic && self.Priority >= 0.4:
		// Skeptic challenges the dominant perspective.
		challenge := fmt.Sprintf("Challenging %s: %s But consider — %s",
			perspectiveName(strongest.Perspective), strongest.KeyInsight, self.KeyInsight)
		return &DebateMove{
			Perspective:   PerspSkeptic,
			MoveType:      "challenge",
			Target:        strongest.Perspective,
			Argument:      challenge,
			NewConfidence: self.Confidence,
		}

	case round >= 2 && math.Abs(selfScore-strongestScore) < 0.15:
		// Close match in later rounds — propose synthesis.
		synthesis := fmt.Sprintf("Combining with %s: %s — while also noting %s",
			perspectiveName(strongest.Perspective), strongest.KeyInsight, self.KeyInsight)
		return &DebateMove{
			Perspective:   self.Perspective,
			MoveType:      "synthesize",
			Target:        strongest.Perspective,
			Argument:      synthesis,
			NewConfidence: math.Min(0.9, (self.Confidence+strongest.Confidence)/2+0.1),
		}

	default:
		// Hold position.
		return &DebateMove{
			Perspective:   self.Perspective,
			MoveType:      "strengthen",
			Argument:      self.Assessment,
			NewConfidence: self.Confidence,
		}
	}
}

func (ic *InnerCouncil) summarizeRound(round DebateRound) string {
	if len(round.Moves) == 0 {
		return "no moves"
	}

	concessions := 0
	challenges := 0
	syntheses := 0
	for _, m := range round.Moves {
		switch m.MoveType {
		case "concede":
			concessions++
		case "challenge":
			challenges++
		case "synthesize":
			syntheses++
		}
	}

	var parts []string
	if concessions > 0 {
		parts = append(parts, fmt.Sprintf("%d concession(s)", concessions))
	}
	if challenges > 0 {
		parts = append(parts, fmt.Sprintf("%d challenge(s)", challenges))
	}
	if syntheses > 0 {
		parts = append(parts, fmt.Sprintf("%d synthesis(es)", syntheses))
	}
	if len(parts) == 0 {
		parts = append(parts, "all positions held")
	}

	return strings.Join(parts, ", ")
}

func (ic *InnerCouncil) findConsensusAndDissent(opinions []CouncilOpinion) (string, string) {
	// Consensus: perspectives with confidence > 0.5 and similar key insights.
	var highConf []CouncilOpinion
	var lowConf []CouncilOpinion
	for _, o := range opinions {
		if o.Confidence >= 0.5 {
			highConf = append(highConf, o)
		} else if o.Confidence >= 0.3 {
			lowConf = append(lowConf, o)
		}
	}

	consensus := ""
	if len(highConf) > 0 {
		var parts []string
		for _, o := range highConf {
			parts = append(parts, fmt.Sprintf("%s (%.0f%%): %s",
				perspectiveName(o.Perspective), o.Confidence*100, o.KeyInsight))
		}
		consensus = strings.Join(parts, "; ")
	}

	dissent := ""
	// If skeptic still has concerns, that's dissent.
	for _, o := range opinions {
		if o.Perspective == PerspSkeptic && o.Priority >= 0.4 && o.Confidence >= 0.4 {
			dissent = o.KeyInsight
			break
		}
	}

	return consensus, dissent
}
