package cognitive

import (
	"math"
	"regexp"
	"strings"
	"time"
	"unicode"

	"github.com/artaeon/nous/internal/memory"
)

// -----------------------------------------------------------------------
// Subtext Engine — reads what the user MEANS, not just what they say.
//
// Most systems respond to the literal words. Nous reads beneath them.
// When someone says "how's my project going?" at 11pm for the third
// time this week, they don't want a status update — they want
// reassurance. When someone says "this stupid bug" with no question,
// they're venting, not asking for a fix (yet).
//
// The engine cross-references:
//   - Linguistic signals: hedging, brevity, repetition, intensity
//   - Emotional cues: valence (positive/negative), arousal (calm/agitated)
//   - Temporal patterns: time-of-day, day-of-week, historical behavior
//   - Episodic memory: what has the user asked before? were they satisfied?
//
// Output: a SubtextAnalysis that tells the response layer what the user
// actually needs, so Nous can respond to the person, not just the query.
// -----------------------------------------------------------------------

// Implied need constants — what the user actually needs from Nous.
const (
	NeedPracticalHelp  = "practical_help"
	NeedReassurance    = "reassurance"
	NeedValidation     = "validation"
	NeedVenting        = "venting"
	NeedInformation    = "information"
	NeedGuidance       = "guidance"
	NeedConnection     = "connection"     // just wants to chat
	NeedCelebration    = "celebration"    // wants to share good news
	NeedAccountability = "accountability" // wants to be held to a commitment
)

// SubtextAnalysis is the result of reading between the lines.
type SubtextAnalysis struct {
	LiteralIntent   string             // what they said (from NLU)
	ImpliedNeed     string             // what they likely need
	EmotionalState  EmotionalState     // detected emotional context
	Urgency         float64            // 0.0-1.0
	Signals         []BehavioralSignal // detected behavioral patterns
	TemporalContext TemporalContext    // time-based patterns
	Confidence      float64            // how confident we are in this reading
}

// EmotionalState captures the affective dimensions of an utterance.
type EmotionalState struct {
	Valence  float64 // -1.0 (negative) to +1.0 (positive)
	Arousal  float64 // 0.0 (calm) to 1.0 (agitated/excited)
	Dominant string  // primary emotion label
}

// BehavioralSignal is a single detected pattern in the user's language.
type BehavioralSignal struct {
	Type     string  // "hedging", "repetition", "brevity", "verbosity", "topic_avoidance", "seeking_validation", "venting"
	Evidence string  // what triggered this detection
	Weight   float64 // 0.0-1.0
}

// TemporalContext captures time-of-day and weekly patterns.
type TemporalContext struct {
	TimeOfDay   string // "early_morning", "morning", "afternoon", "evening", "late_night"
	IsWeekend   bool
	UserPattern string // detected pattern like "user usually vents in evenings"
}

// SubtextEngine reads the emotional and situational context beneath words.
type SubtextEngine struct {
	EpisodicMem *memory.EpisodicMemory
	now         func() time.Time // injectable clock for testing
}

// NewSubtextEngine creates a subtext engine wired to episodic memory.
func NewSubtextEngine(em *memory.EpisodicMemory) *SubtextEngine {
	return &SubtextEngine{
		EpisodicMem: em,
		now:         time.Now,
	}
}

// Analyze performs a full subtext reading of the user's input.
// It combines linguistic, emotional, temporal, and episodic signals
// to determine what the user actually needs.
func (se *SubtextEngine) Analyze(input string, nlu *NLUResult, history []ConvTurn) SubtextAnalysis {
	var signals []BehavioralSignal

	// Detect behavioral signals
	if sig, ok := se.detectHedging(input); ok {
		signals = append(signals, sig)
	}
	if sig, ok := se.detectBrevity(input, history); ok {
		signals = append(signals, sig)
	}
	if sig, ok := se.detectVerbosity(input, history); ok {
		signals = append(signals, sig)
	}
	if sig, ok := se.detectRepetition(input, nlu); ok {
		signals = append(signals, sig)
	}
	if sig, ok := se.detectValidationSeeking(input); ok {
		signals = append(signals, sig)
	}
	if sig, ok := se.detectVenting(input); ok {
		signals = append(signals, sig)
	}

	urgency := se.detectUrgency(input)
	emotion := se.detectEmotionalState(input)
	temporal := se.matchTemporalPatterns(input, nlu)

	// Late night + high urgency shifts toward anxiety
	if temporal.TimeOfDay == "late_night" && urgency > 0.5 {
		if emotion.Arousal < 0.8 {
			emotion.Arousal = math.Min(1.0, emotion.Arousal+0.2)
		}
		if emotion.Dominant == "neutral" {
			emotion.Dominant = "anxious"
		}
	}

	intent := ""
	if nlu != nil {
		intent = nlu.Intent
	}

	need := se.inferImpliedNeed(intent, emotion, signals, temporal)
	confidence := se.computeConfidence(signals, emotion)

	return SubtextAnalysis{
		LiteralIntent:   intent,
		ImpliedNeed:     need,
		EmotionalState:  emotion,
		Urgency:         urgency,
		Signals:         signals,
		TemporalContext: temporal,
		Confidence:      confidence,
	}
}

// -----------------------------------------------------------------------
// Signal detectors
// -----------------------------------------------------------------------

// subtextHedgePhrases are multi-word hedges checked first; subtextHedgeWords are single-word fallbacks.
var subtextHedgePhrases = []string{
	"i think maybe", "not sure if", "sort of", "kind of",
	"i don't know if", "i was wondering if", "it might be",
	"i suppose", "possibly maybe", "maybe i should",
}

var subtextHedgeWords = []string{
	"maybe", "perhaps", "possibly", "somewhat", "apparently",
	"i guess", "i think",
}

func (se *SubtextEngine) detectHedging(input string) (BehavioralSignal, bool) {
	lower := strings.ToLower(input)
	words := strings.Fields(lower)
	if len(words) == 0 {
		return BehavioralSignal{}, false
	}

	var matches []string

	// Check multi-word phrases first
	for _, phrase := range subtextHedgePhrases {
		if strings.Contains(lower, phrase) {
			matches = append(matches, phrase)
		}
	}

	// Check single-word hedges
	for _, hedge := range subtextHedgeWords {
		if strings.Contains(lower, hedge) {
			// Avoid double-counting phrases that contain this word
			alreadyCounted := false
			for _, m := range matches {
				if strings.Contains(m, hedge) {
					alreadyCounted = true
					break
				}
			}
			if !alreadyCounted {
				matches = append(matches, hedge)
			}
		}
	}

	if len(matches) == 0 {
		return BehavioralSignal{}, false
	}

	// Weight: one hedge is mild (0.3), two is moderate (0.6), three+ is heavy (0.85)
	weight := math.Min(0.85, float64(len(matches))*0.3)

	return BehavioralSignal{
		Type:     "hedging",
		Evidence: strings.Join(matches, ", "),
		Weight:   weight,
	}, true
}

var (
	capsWordRe      = regexp.MustCompile(`\b[A-Z]{2,}\b`)
	repeatedPunctRe = regexp.MustCompile(`[!?]{2,}`)
)

func (se *SubtextEngine) detectUrgency(input string) float64 {
	urgency := 0.0

	// Exclamation marks
	excl := strings.Count(input, "!")
	if excl >= 3 {
		urgency += 0.4
	} else if excl >= 1 {
		urgency += 0.2
	}

	// ALL CAPS words (ignore single-letter words and common acronyms)
	capsWords := capsWordRe.FindAllString(input, -1)
	significantCaps := 0
	for _, w := range capsWords {
		if len(w) >= 2 && !isCommonAcronym(w) {
			significantCaps++
		}
	}
	if significantCaps >= 3 {
		urgency += 0.35
	} else if significantCaps >= 1 {
		urgency += 0.15
	}

	// Repeated punctuation (!!! or ???)
	if repeatedPunctRe.MatchString(input) {
		urgency += 0.15
	}

	// Short imperative sentences (under 5 words, no question mark)
	sentences := subtextSplitSentences(input)
	imperatives := 0
	for _, s := range sentences {
		s = strings.TrimSpace(s)
		words := strings.Fields(s)
		if len(words) > 0 && len(words) <= 4 && !strings.HasSuffix(s, "?") {
			imperatives++
		}
	}
	if imperatives > 0 {
		urgency += math.Min(0.2, float64(imperatives)*0.1)
	}

	// Urgency words
	lower := strings.ToLower(input)
	urgencyWords := []string{"urgent", "asap", "immediately", "right now", "help me", "emergency", "critical"}
	for _, w := range urgencyWords {
		if strings.Contains(lower, w) {
			urgency += 0.25
			break
		}
	}

	return math.Min(1.0, urgency)
}

func (se *SubtextEngine) detectBrevity(input string, history []ConvTurn) (BehavioralSignal, bool) {
	if len(history) < 3 {
		return BehavioralSignal{}, false
	}

	// Check if messages are getting shorter over the last few turns
	recent := history
	if len(recent) > 5 {
		recent = recent[len(recent)-5:]
	}

	lengths := make([]int, len(recent))
	for i, turn := range recent {
		lengths[i] = len(strings.Fields(turn.Input))
	}

	// Current input length
	currentLen := len(strings.Fields(input))

	// Calculate trend: are messages shrinking?
	if len(lengths) < 2 {
		return BehavioralSignal{}, false
	}

	avgEarly := 0.0
	for _, l := range lengths[:len(lengths)/2] {
		avgEarly += float64(l)
	}
	avgEarly /= float64(len(lengths) / 2)

	avgLate := float64(currentLen)
	for _, l := range lengths[len(lengths)/2:] {
		avgLate += float64(l)
	}
	avgLate /= float64(len(lengths)-len(lengths)/2) + 1

	// Significant shortening: late average is less than half of early
	if avgEarly > 4 && avgLate < avgEarly*0.5 {
		weight := math.Min(0.8, (avgEarly-avgLate)/avgEarly)
		return BehavioralSignal{
			Type:     "brevity",
			Evidence: "messages getting shorter over conversation",
			Weight:   weight,
		}, true
	}

	return BehavioralSignal{}, false
}

func (se *SubtextEngine) detectVerbosity(input string, history []ConvTurn) (BehavioralSignal, bool) {
	words := strings.Fields(input)
	if len(words) < 40 {
		return BehavioralSignal{}, false
	}

	// Very long message relative to conversation norm
	if len(history) == 0 {
		return BehavioralSignal{
			Type:     "verbosity",
			Evidence: "unusually long message",
			Weight:   0.4,
		}, true
	}

	totalWords := 0
	for _, t := range history {
		totalWords += len(strings.Fields(t.Input))
	}
	avgWords := float64(totalWords) / float64(len(history))

	if float64(len(words)) > avgWords*2.5 && avgWords > 5 {
		weight := math.Min(0.8, (float64(len(words))-avgWords)/(avgWords*3))
		return BehavioralSignal{
			Type:     "verbosity",
			Evidence: "message significantly longer than average",
			Weight:   weight,
		}, true
	}

	return BehavioralSignal{}, false
}

func (se *SubtextEngine) detectRepetition(input string, nlu *NLUResult) (BehavioralSignal, bool) {
	if se.EpisodicMem == nil || nlu == nil {
		return BehavioralSignal{}, false
	}

	// Search episodic memory for similar past queries
	episodes := se.EpisodicMem.Search(input, 10)
	if len(episodes) == 0 {
		return BehavioralSignal{}, false
	}

	// Count how many times a similar topic appeared in the last 50 episodes
	lower := strings.ToLower(input)
	topicWords := extractTopicWords(lower)
	if len(topicWords) == 0 {
		return BehavioralSignal{}, false
	}

	matchCount := 0
	for _, ep := range episodes {
		epLower := strings.ToLower(ep.Input)
		overlap := 0
		for _, tw := range topicWords {
			if strings.Contains(epLower, tw) {
				overlap++
			}
		}
		if float64(overlap) >= float64(len(topicWords))*0.5 {
			matchCount++
		}
	}

	if matchCount < 2 {
		return BehavioralSignal{}, false
	}

	weight := math.Min(0.9, float64(matchCount)*0.25)
	return BehavioralSignal{
		Type:     "repetition",
		Evidence: strings.Join(topicWords, ", "),
		Weight:   weight,
	}, true
}

func (se *SubtextEngine) detectValidationSeeking(input string) (BehavioralSignal, bool) {
	lower := strings.ToLower(input)

	validationPhrases := []string{
		"what do you think",
		"do you think i should",
		"is it a good idea",
		"would it be okay",
		"does that make sense",
		"am i right",
		"is that weird",
		"would you agree",
		"right?",
	}

	for _, phrase := range validationPhrases {
		if strings.Contains(lower, phrase) {
			return BehavioralSignal{
				Type:     "seeking_validation",
				Evidence: phrase,
				Weight:   0.6,
			}, true
		}
	}

	return BehavioralSignal{}, false
}

func (se *SubtextEngine) detectVenting(input string) (BehavioralSignal, bool) {
	lower := strings.ToLower(input)

	// Frustration markers
	ventingPhrases := []string{
		"this stupid", "so annoying", "i hate", "i can't believe",
		"sick of", "tired of", "fed up", "ugh", "argh",
		"drives me crazy", "what the hell", "ridiculous",
		"waste of time", "so frustrating", "i'm done with",
	}

	matches := 0
	var evidence []string
	for _, phrase := range ventingPhrases {
		if strings.Contains(lower, phrase) {
			matches++
			evidence = append(evidence, phrase)
		}
	}

	if matches == 0 {
		return BehavioralSignal{}, false
	}

	// Check if there's no question — pure venting has no ask
	hasQuestion := strings.Contains(input, "?")
	weight := math.Min(0.9, float64(matches)*0.3)
	if !hasQuestion {
		weight = math.Min(0.95, weight+0.2)
	}

	return BehavioralSignal{
		Type:     "venting",
		Evidence: strings.Join(evidence, ", "),
		Weight:   weight,
	}, true
}

// -----------------------------------------------------------------------
// Emotional state detection
// -----------------------------------------------------------------------

var subtextPositiveWords = map[string]float64{
	"great": 0.6, "awesome": 0.7, "love": 0.7, "amazing": 0.8,
	"wonderful": 0.7, "happy": 0.6, "excited": 0.7, "good": 0.4,
	"nice": 0.3, "fantastic": 0.8, "perfect": 0.6, "beautiful": 0.5,
	"grateful": 0.6, "thanks": 0.3, "thank": 0.3, "excellent": 0.7,
	"brilliant": 0.7, "proud": 0.6, "promoted": 0.7, "succeeded": 0.7,
	"won": 0.6, "yay": 0.7, "finally": 0.3, "celebrate": 0.6,
}

var subtextNegativeWords = map[string]float64{
	"bad": 0.4, "terrible": 0.7, "awful": 0.7, "hate": 0.7,
	"stupid": 0.5, "annoying": 0.5, "frustrated": 0.6, "angry": 0.7,
	"sad": 0.6, "depressed": 0.8, "worried": 0.5, "anxious": 0.6,
	"scared": 0.6, "ugly": 0.4, "broken": 0.4, "failed": 0.6,
	"failing": 0.5, "stuck": 0.4, "lost": 0.4, "confused": 0.4,
	"hopeless": 0.8, "overwhelming": 0.6, "stressed": 0.6,
	"exhausted": 0.5, "miserable": 0.8, "sucks": 0.5,
}

var subtextArousalWords = map[string]float64{
	"urgent": 0.7, "immediately": 0.6, "asap": 0.7, "now": 0.3,
	"hurry": 0.6, "emergency": 0.9, "critical": 0.7, "please": 0.2,
	"help": 0.4, "desperate": 0.8, "can't wait": 0.5,
	"excited": 0.6, "amazing": 0.5, "incredible": 0.5,
}

func (se *SubtextEngine) detectEmotionalState(input string) EmotionalState {
	lower := strings.ToLower(input)
	words := subtextTokenize(lower)

	// Calculate valence
	posScore := 0.0
	negScore := 0.0
	for _, w := range words {
		if v, ok := subtextPositiveWords[w]; ok {
			posScore += v
		}
		if v, ok := subtextNegativeWords[w]; ok {
			negScore += v
		}
	}

	valence := 0.0
	total := posScore + negScore
	if total > 0 {
		valence = (posScore - negScore) / total
	}
	valence = math.Max(-1.0, math.Min(1.0, valence))

	// Calculate arousal
	arousal := 0.0
	for _, w := range words {
		if v, ok := subtextArousalWords[w]; ok {
			arousal += v
		}
	}

	// Caps and punctuation boost arousal
	capsWords := capsWordRe.FindAllString(input, -1)
	arousal += float64(len(capsWords)) * 0.1

	excl := strings.Count(input, "!")
	arousal += float64(excl) * 0.1

	if repeatedPunctRe.MatchString(input) {
		arousal += 0.15
	}

	arousal = math.Min(1.0, arousal)

	// Determine dominant emotion
	dominant := classifyDominant(valence, arousal, words)

	return EmotionalState{
		Valence:  valence,
		Arousal:  arousal,
		Dominant: dominant,
	}
}

func classifyDominant(valence, arousal float64, words []string) string {
	// Check specific emotion words first
	wordSet := make(map[string]bool, len(words))
	for _, w := range words {
		wordSet[w] = true
	}

	if wordSet["angry"] || wordSet["furious"] || wordSet["livid"] {
		return "angry"
	}
	if wordSet["sad"] || wordSet["depressed"] || wordSet["miserable"] || wordSet["heartbroken"] {
		return "sad"
	}
	if wordSet["anxious"] || wordSet["worried"] || wordSet["nervous"] || wordSet["scared"] {
		return "anxious"
	}
	if wordSet["grateful"] || wordSet["thankful"] {
		return "grateful"
	}
	if wordSet["curious"] || wordSet["wondering"] || wordSet["interested"] {
		return "curious"
	}

	// Fall back to valence/arousal quadrant
	if valence > 0.3 && arousal > 0.4 {
		return "excited"
	}
	if valence > 0.2 {
		return "grateful" // mild positive
	}
	if valence < -0.3 && arousal > 0.5 {
		return "frustrated"
	}
	if valence < -0.3 && arousal <= 0.5 {
		return "sad"
	}
	if valence < -0.1 && arousal > 0.5 {
		return "anxious"
	}
	if arousal > 0.6 {
		return "excited"
	}

	return "neutral"
}

// -----------------------------------------------------------------------
// Temporal pattern matching
// -----------------------------------------------------------------------

func (se *SubtextEngine) matchTemporalPatterns(input string, nlu *NLUResult) TemporalContext {
	now := se.now()
	hour := now.Hour()
	weekday := now.Weekday()

	tc := TemporalContext{
		TimeOfDay: classifyTimeOfDay(hour),
		IsWeekend: weekday == time.Saturday || weekday == time.Sunday,
	}

	// Look for patterns in episodic memory
	if se.EpisodicMem == nil || nlu == nil {
		return tc
	}

	episodes := se.EpisodicMem.Search(input, 20)
	if len(episodes) < 3 {
		return tc
	}

	// Count what time of day similar queries happen
	timeBuckets := map[string]int{
		"early_morning": 0,
		"morning":       0,
		"afternoon":     0,
		"evening":       0,
		"late_night":    0,
	}

	for _, ep := range episodes {
		bucket := classifyTimeOfDay(ep.Timestamp.Hour())
		timeBuckets[bucket]++
	}

	// Find dominant time bucket
	maxCount := 0
	maxBucket := ""
	for bucket, count := range timeBuckets {
		if count > maxCount {
			maxCount = count
			maxBucket = bucket
		}
	}

	if maxCount >= 3 {
		topic := "this topic"
		if nlu.Intent != "" {
			topic = nlu.Intent
		}
		tc.UserPattern = "user often discusses " + topic + " in the " + strings.ReplaceAll(maxBucket, "_", " ")
	}

	return tc
}

func classifyTimeOfDay(hour int) string {
	switch {
	case hour < 6:
		return "late_night"
	case hour < 9:
		return "early_morning"
	case hour < 12:
		return "morning"
	case hour < 17:
		return "afternoon"
	case hour < 21:
		return "evening"
	default:
		return "late_night"
	}
}

// -----------------------------------------------------------------------
// Implied need inference
// -----------------------------------------------------------------------

func (se *SubtextEngine) inferImpliedNeed(intent string, emotion EmotionalState, signals []BehavioralSignal, temporal TemporalContext) string {
	signalMap := make(map[string]float64)
	for _, sig := range signals {
		signalMap[sig.Type] = sig.Weight
	}

	// Celebration: strong positive emotion
	if emotion.Valence > 0.5 && emotion.Arousal > 0.3 {
		return NeedCelebration
	}

	// Venting: negative emotion + venting signal + no direct question
	if ventW, ok := signalMap["venting"]; ok && ventW > 0.4 {
		return NeedVenting
	}

	// Validation: seeking validation signal + hedging
	if valW, ok := signalMap["seeking_validation"]; ok && valW > 0.3 {
		if _, hedging := signalMap["hedging"]; hedging {
			return NeedValidation
		}
		return NeedValidation
	}

	// Reassurance: repetition + late night or negative emotion
	if repW, ok := signalMap["repetition"]; ok && repW > 0.3 {
		if temporal.TimeOfDay == "late_night" || temporal.TimeOfDay == "evening" || emotion.Valence < -0.1 {
			return NeedReassurance
		}
	}

	// Anxiety-driven reassurance: anxious + asking about status
	if emotion.Dominant == "anxious" || emotion.Dominant == "frustrated" {
		if subtextContainsAny(intent, []string{"status", "check", "how", "progress"}) {
			return NeedReassurance
		}
	}

	// Connection: casual greetings, small talk, verbosity without a clear question
	if subtextContainsAny(intent, []string{"greeting", "chat", "smalltalk"}) {
		return NeedConnection
	}
	if verbW, ok := signalMap["verbosity"]; ok && verbW > 0.5 && emotion.Valence > -0.2 {
		return NeedConnection
	}

	// Accountability: commitment language
	lower := strings.ToLower(intent)
	if subtextContainsAny(lower, []string{"promise", "commit", "goal", "plan", "accountability"}) {
		return NeedAccountability
	}

	// Brevity + negative = frustration, provide guidance not lecture
	if brevW, ok := signalMap["brevity"]; ok && brevW > 0.4 && emotion.Valence < 0 {
		return NeedGuidance
	}

	// Hedging alone (without validation-seeking) suggests they want guidance
	if hedgeW, ok := signalMap["hedging"]; ok && hedgeW > 0.4 {
		return NeedGuidance
	}

	// Explicit help/how-to intent → practical help
	if subtextContainsAny(intent, []string{"help", "fix", "how", "solve", "create", "build", "make"}) {
		return NeedPracticalHelp
	}

	// Questions about facts → information
	if subtextContainsAny(intent, []string{"what", "who", "where", "when", "define", "explain", "tell"}) {
		return NeedInformation
	}

	// Default: practical help (safe fallback)
	return NeedPracticalHelp
}

// -----------------------------------------------------------------------
// Confidence scoring
// -----------------------------------------------------------------------

func (se *SubtextEngine) computeConfidence(signals []BehavioralSignal, emotion EmotionalState) float64 {
	if len(signals) == 0 && emotion.Dominant == "neutral" {
		return 0.3 // low confidence when we have no signals
	}

	conf := 0.4 // base confidence

	// More signals = more data = higher confidence
	for _, sig := range signals {
		conf += sig.Weight * 0.15
	}

	// Strong emotion is a clear signal
	if math.Abs(emotion.Valence) > 0.5 {
		conf += 0.15
	}
	if emotion.Arousal > 0.5 {
		conf += 0.1
	}

	return math.Min(0.95, conf)
}

// -----------------------------------------------------------------------
// Utility functions
// -----------------------------------------------------------------------

var commonAcronyms = map[string]bool{
	"AI": true, "API": true, "URL": true, "HTTP": true, "HTML": true,
	"CSS": true, "SQL": true, "JSON": true, "XML": true, "PDF": true,
	"ID": true, "OK": true, "US": true, "UK": true, "EU": true,
	"PR": true, "CI": true, "CD": true, "UI": true, "UX": true,
	"OS": true, "CPU": true, "GPU": true, "RAM": true, "SSD": true,
	"CEO": true, "CTO": true, "VP": true, "PM": true, "HR": true,
	"TODO": true, "FYI": true, "ASAP": true, "ETA": true, "FAQ": true,
}

func isCommonAcronym(word string) bool {
	return commonAcronyms[word]
}

// subtextTokenize splits input into lowercase words, stripping punctuation but keeping all words.
func subtextTokenize(input string) []string {
	var words []string
	for _, w := range strings.Fields(input) {
		cleaned := strings.TrimFunc(w, func(r rune) bool {
			return unicode.IsPunct(r) || unicode.IsSymbol(r)
		})
		if cleaned != "" {
			words = append(words, strings.ToLower(cleaned))
		}
	}
	return words
}

// subtextSplitSentences roughly splits on sentence boundaries.
func subtextSplitSentences(input string) []string {
	var sentences []string
	start := 0
	for i, r := range input {
		if r == '.' || r == '!' || r == '?' {
			s := strings.TrimSpace(input[start : i+1])
			if s != "" {
				sentences = append(sentences, s)
			}
			start = i + 1
		}
	}
	// Trailing fragment
	if start < len(input) {
		s := strings.TrimSpace(input[start:])
		if s != "" {
			sentences = append(sentences, s)
		}
	}
	return sentences
}

// extractTopicWords pulls content words (non-stopwords) from input.
func extractTopicWords(input string) []string {
	stopwords := map[string]bool{
		"i": true, "me": true, "my": true, "we": true, "you": true,
		"the": true, "a": true, "an": true, "is": true, "are": true,
		"was": true, "were": true, "be": true, "been": true, "being": true,
		"have": true, "has": true, "had": true, "do": true, "does": true,
		"did": true, "will": true, "would": true, "could": true, "should": true,
		"can": true, "to": true, "of": true, "in": true, "for": true,
		"on": true, "with": true, "at": true, "by": true, "from": true,
		"it": true, "its": true, "this": true, "that": true, "these": true,
		"and": true, "or": true, "but": true, "not": true, "so": true,
		"if": true, "then": true, "than": true, "what": true, "how": true,
		"about": true, "just": true, "very": true, "really": true,
	}

	var topics []string
	for _, w := range subtextTokenize(input) {
		if len(w) >= 3 && !stopwords[w] {
			topics = append(topics, w)
		}
	}
	return topics
}

func subtextContainsAny(s string, substrs []string) bool {
	lower := strings.ToLower(s)
	for _, sub := range substrs {
		if strings.Contains(lower, sub) {
			return true
		}
	}
	return false
}
