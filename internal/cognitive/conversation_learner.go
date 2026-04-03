package cognitive

import (
	"encoding/json"
	"os"
	"sort"
	"strings"
	"sync"
	"time"
)

// -----------------------------------------------------------------------
// Conversation Learner — Nous learns response patterns from its own
// conversations instead of from static corpora.
//
// Every successful interaction teaches a new pattern. Every failure
// reduces confidence in a pattern. Over time, Nous builds a library
// of proven response strategies.
//
// This COMPLEMENTS ResponseCrystalStore (which caches exact responses)
// by learning GENERALIZED patterns that transfer across topics:
//
//   Crystal:  "What is Go?" → exact cached response
//   Learner:  "What is [TOPIC]?" + positive sentiment → "define then evaluate" pattern
//
// The crystal is a cache. The learner is a skill.
// -----------------------------------------------------------------------

// ResponsePattern is a generalized response pattern extracted from a
// successful conversation. Unlike LearnedPattern in learning_engine.go
// (which tracks sentence templates absorbed from user input), this
// tracks Nous's own response strategies that worked.
type ResponsePattern struct {
	InputPattern string  `json:"input_pattern"` // generalized: "I just [ACHIEVEMENT] at work!"
	Response     string  `json:"response"`      // the response that worked
	Intent       string  `json:"intent"`        // detected intent
	Sentiment    string  `json:"sentiment"`      // detected sentiment
	Topic        string  `json:"topic"`          // topic area
	SuccessCount int     `json:"success_count"`  // times this pattern worked
	FailCount    int     `json:"fail_count"`     // times it failed
	Quality      float64 `json:"quality"`        // success / (success + fail)
	LastUsed     time.Time `json:"last_used"`
	CreatedAt    time.Time `json:"created_at"`
}

// ConversationLearner mines successful interaction patterns and builds
// a response pattern library that improves conversation quality over time.
type ConversationLearner struct {
	patterns    []ResponsePattern
	byIntent    map[string][]int // intent → pattern indices
	bySentiment map[string][]int // sentiment → pattern indices
	byTopic     map[string][]int // topic → pattern indices
	mu          sync.RWMutex
	savePath    string
}

// NewConversationLearner creates a learner, optionally loading from disk.
func NewConversationLearner(savePath string) *ConversationLearner {
	cl := &ConversationLearner{
		byIntent:    make(map[string][]int),
		bySentiment: make(map[string][]int),
		byTopic:     make(map[string][]int),
		savePath:    savePath,
	}
	if savePath != "" {
		_ = cl.Load(savePath)
	}
	return cl
}

// -----------------------------------------------------------------------
// Core Learning — records interaction outcomes
// -----------------------------------------------------------------------

// LearnFromInteraction records a conversation outcome. If success is true,
// the response pattern is reinforced. If false, similar patterns lose
// confidence. This is the primary learning signal.
func (cl *ConversationLearner) LearnFromInteraction(input, response, intent, sentiment, topic string, success bool) {
	cl.mu.Lock()
	defer cl.mu.Unlock()

	generalized := GeneralizeInput(input)

	// Search for an existing pattern with the same generalized input and intent.
	idx := cl.findExactPattern(generalized, intent)

	if idx >= 0 {
		// Reinforce or penalize the existing pattern.
		if success {
			cl.patterns[idx].SuccessCount++
		} else {
			cl.patterns[idx].FailCount++
		}
		cl.patterns[idx].Quality = patternQuality(
			cl.patterns[idx].SuccessCount,
			cl.patterns[idx].FailCount,
		)
		cl.patterns[idx].LastUsed = time.Now()
		// If the new response is better (success on a previously failing
		// pattern), update the stored response.
		if success && cl.patterns[idx].Quality > 0.5 {
			cl.patterns[idx].Response = response
		}
		return
	}

	// No existing match — create a new pattern if this was a success.
	if !success {
		return // don't create patterns from failures
	}

	now := time.Now()
	newIdx := len(cl.patterns)
	cl.patterns = append(cl.patterns, ResponsePattern{
		InputPattern: generalized,
		Response:     response,
		Intent:       intent,
		Sentiment:    sentiment,
		Topic:        topic,
		SuccessCount: 1,
		FailCount:    0,
		Quality:      1.0,
		LastUsed:     now,
		CreatedAt:    now,
	})

	// Update indices.
	if intent != "" {
		cl.byIntent[intent] = append(cl.byIntent[intent], newIdx)
	}
	if sentiment != "" {
		cl.bySentiment[sentiment] = append(cl.bySentiment[sentiment], newIdx)
	}
	if topic != "" {
		cl.byTopic[strings.ToLower(topic)] = append(cl.byTopic[strings.ToLower(topic)], newIdx)
	}
}

// findExactPattern returns the index of a pattern with matching
// generalized input and intent, or -1 if not found.
func (cl *ConversationLearner) findExactPattern(generalized, intent string) int {
	for i, p := range cl.patterns {
		if p.InputPattern == generalized && p.Intent == intent {
			return i
		}
	}
	return -1
}

// patternQuality calculates success ratio with a Bayesian prior.
// With zero data, defaults to 0.5 (neutral). The prior prevents a
// single success from reading as 1.0.
func patternQuality(success, fail int) float64 {
	// Add a prior of 1 success and 1 failure (Laplace smoothing).
	return float64(success+1) / float64(success+fail+2)
}

// -----------------------------------------------------------------------
// Input Generalization — converts specific inputs into reusable patterns
// -----------------------------------------------------------------------

// GeneralizeInput converts a specific input into a generalized pattern
// by replacing content words with slot markers. The structure and
// function words are preserved so patterns can match new inputs.
//
//	"I just got promoted at work!" → "I just [ACHIEVEMENT] at work!"
//	"what should I have for dinner?" → "what should I have for [MEAL]?"
//	"recommend me a good book" → "recommend me a good [MEDIA]"
func GeneralizeInput(input string) string {
	words := strings.Fields(input)
	if len(words) == 0 {
		return input
	}

	var result []string
	for i, word := range words {
		clean := strings.Trim(strings.ToLower(word), ".,!?;:'\"")
		// Preserve trailing punctuation.
		suffix := ""
		if len(word) > 0 {
			last := word[len(word)-1]
			if last == '.' || last == '!' || last == '?' || last == ',' {
				suffix = string(last)
			}
		}

		if slot := classifyWord(clean, i, words); slot != "" {
			result = append(result, slot+suffix)
		} else {
			result = append(result, word)
		}
	}

	return strings.Join(result, " ")
}

// classifyWord checks if a word should be replaced with a slot marker.
// Returns the slot name or empty string if the word should stay literal.
func classifyWord(word string, pos int, context []string) string {
	// Never replace the first word (usually structural) or function words.
	if pos == 0 || generalizeSkip[word] {
		return ""
	}

	// Check each category in priority order.
	if emotionWords[word] {
		return "[EMOTION]"
	}
	if achievementWords[word] {
		return "[ACHIEVEMENT]"
	}
	if activityWords[word] {
		return "[ACTIVITY]"
	}
	if mealWords[word] {
		return "[MEAL]"
	}
	if mediaWords[word] {
		return "[MEDIA]"
	}
	if placeWords[word] {
		return "[PLACE]"
	}
	if timeWords[word] {
		return "[TIME]"
	}

	// Capitalized words that aren't sentence-initial → proper noun.
	if pos > 0 && len(word) > 0 && word[0] >= 'A' && word[0] <= 'Z' {
		clean := strings.Trim(strings.ToLower(word), ".,!?;:'\"")
		if !generalizeSkip[clean] {
			return "[NAME]"
		}
	}

	return ""
}

// Word category maps for generalization.
var (
	generalizeSkip = buildSet(
		"i", "you", "he", "she", "it", "we", "they", "me", "my", "your",
		"the", "a", "an", "is", "are", "was", "were", "am", "be", "been",
		"have", "has", "had", "do", "does", "did", "will", "would", "could",
		"should", "can", "may", "might", "must", "shall",
		"in", "on", "at", "to", "for", "of", "with", "by", "from", "about",
		"into", "through", "during", "before", "after", "above", "below",
		"and", "or", "but", "nor", "not", "no", "so", "yet", "if", "then",
		"than", "as", "this", "that", "these", "those", "what", "which",
		"who", "how", "when", "where", "why", "just", "really", "very",
		"much", "also", "too", "some", "any", "all", "each", "every",
		"tell", "know", "think", "want", "need", "like", "got", "get",
		"make", "go", "come", "take", "give", "say", "said", "let",
		"going", "should", "recommend", "suggest", "good", "me",
	)

	emotionWords = buildSet(
		"happy", "sad", "angry", "anxious", "excited", "nervous",
		"frustrated", "depressed", "thrilled", "worried", "stressed",
		"overwhelmed", "grateful", "lonely", "scared", "hopeful",
		"disappointed", "proud", "ashamed", "jealous", "confused",
		"bored", "content", "nostalgic", "euphoric", "miserable",
	)

	achievementWords = buildSet(
		"promoted", "graduated", "married", "engaged", "hired",
		"accepted", "published", "elected", "awarded", "certified",
		"finished", "completed", "accomplished", "succeeded",
		"won", "earned", "achieved", "passed", "qualified",
		"promotion", "graduation", "marriage", "engagement",
	)

	activityWords = buildSet(
		"running", "cooking", "reading", "swimming", "hiking",
		"painting", "coding", "writing", "singing", "dancing",
		"gardening", "fishing", "cycling", "climbing", "surfing",
		"traveling", "volunteering", "studying", "practicing",
		"exercising", "meditating", "shopping", "camping",
	)

	mealWords = buildSet(
		"breakfast", "lunch", "dinner", "supper", "brunch",
		"snack", "dessert", "meal",
	)

	mediaWords = buildSet(
		"book", "movie", "film", "song", "album", "show",
		"series", "podcast", "game", "documentary", "novel",
		"anime", "manga", "comic", "article", "video",
	)

	placeWords = buildSet(
		"home", "school", "office", "work", "hospital",
		"park", "restaurant", "gym", "beach", "library",
		"church", "store", "mall", "airport", "station",
	)

	timeWords = buildSet(
		"today", "yesterday", "tomorrow", "morning", "evening",
		"afternoon", "night", "weekend", "monday", "tuesday",
		"wednesday", "thursday", "friday", "saturday", "sunday",
	)
)

func buildSet(words ...string) map[string]bool {
	m := make(map[string]bool, len(words))
	for _, w := range words {
		m[w] = true
	}
	return m
}

// -----------------------------------------------------------------------
// Pattern Matching — finds the best learned pattern for a new input
// -----------------------------------------------------------------------

// FindPattern finds the best matching learned pattern for a new input.
// Uses intent matching, sentiment matching, input pattern similarity,
// and quality score to rank candidates.
func (cl *ConversationLearner) FindPattern(input, intent, sentiment string) *ResponsePattern {
	cl.mu.RLock()
	defer cl.mu.RUnlock()

	if len(cl.patterns) == 0 {
		return nil
	}

	generalized := GeneralizeInput(input)

	var bestIdx int
	var bestScore float64
	bestIdx = -1

	for i, p := range cl.patterns {
		score := cl.scoreMatch(generalized, intent, sentiment, &p)
		if score > bestScore {
			bestScore = score
			bestIdx = i
		}
	}

	if bestIdx < 0 || bestScore < 0.2 {
		return nil
	}

	// Return a copy to avoid races.
	result := cl.patterns[bestIdx]
	return &result
}

// scoreMatch computes how well a pattern matches a query. Returns 0-1.
func (cl *ConversationLearner) scoreMatch(generalized, intent, sentiment string, p *ResponsePattern) float64 {
	var score float64

	// Intent match (40% weight) — exact match required.
	if intent != "" && p.Intent == intent {
		score += 0.40
	}

	// Sentiment match (15% weight).
	if sentiment != "" && p.Sentiment == sentiment {
		score += 0.15
	}

	// Input pattern similarity (30% weight) — fuzzy match on generalized forms.
	sim := patternSimilarity(generalized, p.InputPattern)
	score += sim * 0.30

	// Quality score (15% weight) — prefer proven patterns.
	score += p.Quality * 0.15

	return score
}

// patternSimilarity computes similarity between two generalized input
// patterns. Uses token-level Jaccard similarity, treating slot markers
// as wildcards that match each other.
func patternSimilarity(a, b string) float64 {
	tokensA := strings.Fields(strings.ToLower(a))
	tokensB := strings.Fields(strings.ToLower(b))

	if len(tokensA) == 0 || len(tokensB) == 0 {
		return 0
	}

	setA := make(map[string]bool, len(tokensA))
	for _, t := range tokensA {
		setA[normalizeToken(t)] = true
	}
	setB := make(map[string]bool, len(tokensB))
	for _, t := range tokensB {
		setB[normalizeToken(t)] = true
	}

	// Count intersection.
	intersection := 0
	for t := range setA {
		if setB[t] {
			intersection++
		}
	}

	// Jaccard similarity.
	union := len(setA) + len(setB) - intersection
	if union == 0 {
		return 0
	}
	return float64(intersection) / float64(union)
}

// normalizeToken normalizes a token for comparison. Slot markers like
// [EMOTION] and [ACHIEVEMENT] are all treated as "[SLOT]" so they match
// each other.
func normalizeToken(t string) string {
	t = strings.Trim(t, ".,!?;:'\"")
	if len(t) > 2 && t[0] == '[' && t[len(t)-1] == ']' {
		return "[SLOT]"
	}
	return strings.ToLower(t)
}

// -----------------------------------------------------------------------
// Response Adaptation — takes a learned response and adapts it
// -----------------------------------------------------------------------

// AdaptResponse takes a learned response pattern and adapts it to a
// new input context. Substitutes topic-specific words from the original
// with the equivalent from the new input.
//
// Pattern response: "Congratulations on your promotion! That's a huge achievement."
// New input: "I just got married!"
// Adapted: "Congratulations on getting married! That's a huge achievement."
func AdaptResponse(pattern *ResponsePattern, newInput string) string {
	if pattern == nil {
		return ""
	}

	response := pattern.Response

	// Extract the specific content words from the original and new inputs.
	origSlots := extractSlotValues(pattern.InputPattern, pattern.Topic)
	newSlots := extractResponseContentWords(newInput)

	if len(origSlots) == 0 || len(newSlots) == 0 {
		return response
	}

	// Replace original content words with new ones in the response.
	for _, orig := range origSlots {
		if orig == "" {
			continue
		}
		for _, repl := range newSlots {
			if repl == "" || strings.EqualFold(orig, repl) {
				continue
			}
			// Case-preserving replacement: if the original is capitalized
			// in the response, capitalize the replacement too.
			if strings.Contains(response, orig) {
				response = strings.Replace(response, orig, repl, 1)
				break
			}
			lower := strings.ToLower(orig)
			if strings.Contains(response, lower) {
				response = strings.Replace(response, lower, strings.ToLower(repl), 1)
				break
			}
		}
	}

	return response
}

// extractSlotValues pulls out the specific words that were generalized
// (achievements, emotions, etc.) from the original interaction context.
func extractSlotValues(pattern, topic string) []string {
	var values []string
	if topic != "" {
		values = append(values, topic)
	}

	// Any word in the original input that would have been slotted is a
	// value we might want to swap.
	words := strings.Fields(pattern)
	for _, w := range words {
		clean := strings.Trim(w, ".,!?;:'\"")
		lower := strings.ToLower(clean)
		if achievementWords[lower] || emotionWords[lower] ||
			activityWords[lower] || mediaWords[lower] {
			values = append(values, clean)
		}
	}
	return values
}

// extractResponseContentWords pulls meaningful content words from an input
// for use in response adaptation.
func extractResponseContentWords(input string) []string {
	var content []string
	words := strings.Fields(input)
	for _, w := range words {
		clean := strings.Trim(w, ".,!?;:'\"")
		lower := strings.ToLower(clean)
		if generalizeSkip[lower] || len(clean) < 3 {
			continue
		}
		content = append(content, clean)
	}
	return content
}

// -----------------------------------------------------------------------
// Consolidation — merges, decays, and prunes patterns
// -----------------------------------------------------------------------

// Consolidate runs maintenance on the pattern library:
//   - Merges similar patterns (same intent + sentiment + similar input)
//   - Decays old patterns that haven't been used recently
//   - Prunes low-quality patterns (quality < 0.3)
func (cl *ConversationLearner) Consolidate() {
	cl.mu.Lock()
	defer cl.mu.Unlock()

	// 1. Merge similar patterns.
	cl.mergeSimilar()

	// 2. Decay old patterns — reduce quality of patterns unused for 30+ days.
	now := time.Now()
	decayThreshold := now.Add(-30 * 24 * time.Hour)
	for i := range cl.patterns {
		if cl.patterns[i].LastUsed.Before(decayThreshold) {
			// Decay: add a virtual failure.
			cl.patterns[i].FailCount++
			cl.patterns[i].Quality = patternQuality(
				cl.patterns[i].SuccessCount,
				cl.patterns[i].FailCount,
			)
		}
	}

	// 3. Prune low-quality patterns.
	var kept []ResponsePattern
	for _, p := range cl.patterns {
		if p.Quality >= 0.3 || p.SuccessCount >= 5 {
			kept = append(kept, p)
		}
	}
	cl.patterns = kept

	// 4. Rebuild indices.
	cl.rebuildIndices()
}

// mergeSimilar finds patterns with the same intent and sentiment whose
// generalized inputs are highly similar, and merges them. The higher-
// quality pattern absorbs the other's success/fail counts.
func (cl *ConversationLearner) mergeSimilar() {
	merged := make(map[int]bool) // indices already merged away

	for i := 0; i < len(cl.patterns); i++ {
		if merged[i] {
			continue
		}
		for j := i + 1; j < len(cl.patterns); j++ {
			if merged[j] {
				continue
			}
			a := &cl.patterns[i]
			b := &cl.patterns[j]

			if a.Intent != b.Intent || a.Sentiment != b.Sentiment {
				continue
			}

			sim := patternSimilarity(a.InputPattern, b.InputPattern)
			if sim < 0.7 {
				continue
			}

			// Merge b into a (keep the higher-quality one).
			if b.Quality > a.Quality {
				a.Response = b.Response
				a.InputPattern = b.InputPattern
			}
			a.SuccessCount += b.SuccessCount
			a.FailCount += b.FailCount
			a.Quality = patternQuality(a.SuccessCount, a.FailCount)
			if b.LastUsed.After(a.LastUsed) {
				a.LastUsed = b.LastUsed
			}
			merged[j] = true
		}
	}

	if len(merged) == 0 {
		return
	}

	// Remove merged patterns.
	var kept []ResponsePattern
	for i, p := range cl.patterns {
		if !merged[i] {
			kept = append(kept, p)
		}
	}
	cl.patterns = kept
}

// rebuildIndices reconstructs the intent/sentiment/topic index maps
// from the current pattern list.
func (cl *ConversationLearner) rebuildIndices() {
	cl.byIntent = make(map[string][]int)
	cl.bySentiment = make(map[string][]int)
	cl.byTopic = make(map[string][]int)

	for i, p := range cl.patterns {
		if p.Intent != "" {
			cl.byIntent[p.Intent] = append(cl.byIntent[p.Intent], i)
		}
		if p.Sentiment != "" {
			cl.bySentiment[p.Sentiment] = append(cl.bySentiment[p.Sentiment], i)
		}
		if p.Topic != "" {
			cl.byTopic[strings.ToLower(p.Topic)] = append(cl.byTopic[strings.ToLower(p.Topic)], i)
		}
	}
}

// -----------------------------------------------------------------------
// Success Detection — heuristics for whether an interaction succeeded
// -----------------------------------------------------------------------

// InteractionOutcome represents the detected outcome of an interaction.
type InteractionOutcome int

const (
	OutcomeNeutral       InteractionOutcome = iota // session ended normally
	OutcomeSuccess                                 // user continued engagement
	OutcomeStrongSuccess                           // user expressed satisfaction
	OutcomeFailure                                 // user expressed dissatisfaction
	OutcomeMildFailure                             // user changed topic immediately
)

// DetectOutcome examines the user's follow-up message to determine
// whether the previous response was successful.
func DetectOutcome(followUp string) InteractionOutcome {
	if followUp == "" {
		return OutcomeNeutral
	}

	lower := strings.ToLower(strings.TrimSpace(followUp))

	// Strong success signals.
	for _, s := range strongSuccessSignals {
		if strings.Contains(lower, s) {
			return OutcomeStrongSuccess
		}
	}

	// Failure signals.
	for _, s := range failureSignals {
		if strings.Contains(lower, s) {
			return OutcomeFailure
		}
	}

	// Mild failure: very short topic change (under 4 words, no connection).
	words := strings.Fields(lower)
	if len(words) <= 3 {
		// Short messages that aren't affirmations are mild failures.
		for _, s := range []string{"ok", "okay", "sure", "fine", "whatever", "anyway"} {
			if lower == s {
				return OutcomeMildFailure
			}
		}
	}

	// Default: user continued the conversation → success.
	return OutcomeSuccess
}

// IsSuccess returns true if the outcome represents a successful interaction.
func (o InteractionOutcome) IsSuccess() bool {
	return o == OutcomeSuccess || o == OutcomeStrongSuccess
}

var strongSuccessSignals = []string{
	"thanks", "thank you", "exactly", "perfect", "great answer",
	"that's right", "that's correct", "good point", "well said",
	"helpful", "makes sense", "i see", "got it", "understood",
	"interesting", "fascinating", "tell me more", "go on",
	"nice", "awesome", "brilliant", "love it",
}

var failureSignals = []string{
	"no", "wrong", "that's not right", "that's incorrect",
	"not what i asked", "not what i meant", "you're wrong",
	"that doesn't make sense", "try again", "bad answer",
	"unhelpful", "useless", "off topic", "irrelevant",
	"i said", "i meant", "what i actually",
}

// -----------------------------------------------------------------------
// Persistence — save/load the pattern library as JSON
// -----------------------------------------------------------------------

// Save persists the pattern library to a JSON file.
func (cl *ConversationLearner) Save(path string) error {
	cl.mu.RLock()
	defer cl.mu.RUnlock()

	data, err := json.MarshalIndent(cl.patterns, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0644)
}

// Load reads the pattern library from a JSON file.
func (cl *ConversationLearner) Load(path string) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return err
	}

	var patterns []ResponsePattern
	if err := json.Unmarshal(data, &patterns); err != nil {
		return err
	}

	cl.mu.Lock()
	defer cl.mu.Unlock()

	cl.patterns = patterns
	cl.rebuildIndices()
	return nil
}

// -----------------------------------------------------------------------
// Seeding — bootstrap from discourse corpus or built-in patterns
// -----------------------------------------------------------------------

// SeedFromCorpus bootstraps the learner with initial patterns derived
// from a DiscourseCorpus. Converts discourse sentences into response
// patterns for common query types.
func (cl *ConversationLearner) SeedFromCorpus(corpus *DiscourseCorpus) {
	if corpus == nil {
		return
	}

	cl.mu.Lock()
	defer cl.mu.Unlock()

	now := time.Now()
	corpus.mu.RLock()
	defer corpus.mu.RUnlock()

	// Map discourse functions to conversational intent/pattern pairs.
	type seedRule struct {
		fn     DiscourseFunc
		intent string
		input  string // generalized input pattern
	}
	rules := []seedRule{
		{DFDefines, "question", "what is [TOPIC]"},
		{DFEvaluates, "opinion", "what do you think about [TOPIC]"},
		{DFExplainsWhy, "question", "why does [TOPIC] happen"},
		{DFDescribes, "question", "how does [TOPIC] work"},
		{DFCompares, "question", "how does [TOPIC] compare"},
		{DFGivesExample, "question", "give me an example of [TOPIC]"},
		{DFContext, "question", "when was [TOPIC] created"},
		{DFConsequence, "question", "what happens because of [TOPIC]"},
	}

	added := 0
	for _, rule := range rules {
		sents, ok := corpus.sentences[rule.fn]
		if !ok {
			continue
		}
		// Take up to 3 high-quality sentences per function.
		count := 0
		for _, ds := range sents {
			if ds.Quality < 2 || count >= 3 {
				continue
			}
			// Check for duplicate.
			duplicate := false
			for _, p := range cl.patterns {
				if p.Response == ds.Sentence {
					duplicate = true
					break
				}
			}
			if duplicate {
				continue
			}

			cl.patterns = append(cl.patterns, ResponsePattern{
				InputPattern: rule.input,
				Response:     ds.Sentence,
				Intent:       rule.intent,
				Sentiment:    "neutral",
				Topic:        ds.Topic,
				SuccessCount: 1,
				FailCount:    0,
				Quality:      0.6, // seeded patterns start moderate
				LastUsed:     now,
				CreatedAt:    now,
			})
			count++
			added++
		}
	}

	if added > 0 {
		cl.rebuildIndices()
	}
}

// SeedDefaults populates the learner with built-in patterns for common
// conversational situations. These provide a baseline before any real
// interactions have occurred.
func (cl *ConversationLearner) SeedDefaults() {
	cl.mu.Lock()
	defer cl.mu.Unlock()

	now := time.Now()

	seeds := []struct {
		input     string
		response  string
		intent    string
		sentiment string
		topic     string
	}{
		// Greetings.
		{"hello", "Hey! What's on your mind?", "greeting", "positive", "greeting"},
		{"hi there", "Hi! How can I help you today?", "greeting", "positive", "greeting"},
		{"good [TIME]", "Good to see you! What can I do for you?", "greeting", "positive", "greeting"},

		// Achievements / positive news.
		{"I just [ACHIEVEMENT] at work", "That's wonderful news! You should be really proud of yourself.", "sharing", "positive", "achievement"},
		{"I got [ACHIEVEMENT]", "Congratulations! That's a huge milestone. How are you feeling about it?", "sharing", "positive", "achievement"},
		{"I [ACHIEVEMENT] [TIME]", "That's great to hear! Tell me more about how it went.", "sharing", "positive", "achievement"},

		// Negative emotions.
		{"I'm feeling [EMOTION]", "I hear you. It's completely valid to feel that way. Do you want to talk about what's going on?", "sharing", "negative", "emotion"},
		{"I feel [EMOTION] about", "That sounds tough. What's been happening?", "sharing", "negative", "emotion"},
		{"I'm so [EMOTION]", "I'm sorry you're going through that. Is there something specific that triggered this?", "sharing", "negative", "emotion"},

		// Positive emotions.
		{"I'm [EMOTION] about", "That's great! What's making you feel that way?", "sharing", "positive", "emotion"},

		// Recommendations.
		{"recommend me a good [MEDIA]", "What kind of mood are you in? That'll help me suggest something you'll actually enjoy.", "request", "neutral", "recommendation"},
		{"suggest a [MEDIA]", "Sure! What genres or styles do you usually enjoy?", "request", "neutral", "recommendation"},
		{"what [MEDIA] should I", "It depends on what you're looking for. Are you in the mood for something light or something more serious?", "request", "neutral", "recommendation"},

		// Advice.
		{"what should I do about", "Let's think through this together. What are the options you're considering?", "question", "neutral", "advice"},
		{"I don't know what to do", "That's a frustrating place to be. Can you walk me through the situation?", "sharing", "negative", "advice"},
		{"should I [ACTIVITY]", "That depends on a few things. What's drawing you toward it, and what's holding you back?", "question", "neutral", "advice"},

		// Knowledge questions.
		{"what is [TOPIC]", "Let me share what I know about that.", "question", "neutral", "knowledge"},
		{"tell me about [TOPIC]", "Sure, here's what I can tell you.", "question", "neutral", "knowledge"},
		{"why does [TOPIC]", "That's a great question. Here's the reasoning behind it.", "question", "neutral", "knowledge"},
		{"how does [TOPIC] work", "Good question. Let me break it down.", "question", "neutral", "knowledge"},

		// Conversation management.
		{"tell me more", "Sure, let me expand on that.", "followup", "neutral", "continuation"},
		{"what else", "Here's what else I can add.", "followup", "neutral", "continuation"},
		{"can you explain that", "Of course. Let me put it differently.", "followup", "neutral", "clarification"},

		// Meal / daily life.
		{"what should I have for [MEAL]", "What kind of food are you in the mood for? Something quick or something more involved?", "request", "neutral", "food"},

		// Farewells.
		{"goodbye", "Take care! It was good talking with you.", "farewell", "positive", "farewell"},
		{"bye", "See you later!", "farewell", "positive", "farewell"},
		{"thanks for your help", "You're welcome! Happy I could help.", "farewell", "positive", "farewell"},

		// Meta / about Nous.
		{"who are you", "I'm Nous, a local AI assistant. I learn from our conversations and get better over time.", "meta", "neutral", "identity"},
		{"what can you do", "I can answer questions, have conversations, help you think through problems, and learn from our interactions.", "meta", "neutral", "capabilities"},
	}

	for _, s := range seeds {
		// Skip duplicates.
		duplicate := false
		for _, p := range cl.patterns {
			if p.InputPattern == s.input && p.Intent == s.intent {
				duplicate = true
				break
			}
		}
		if duplicate {
			continue
		}

		cl.patterns = append(cl.patterns, ResponsePattern{
			InputPattern: s.input,
			Response:     s.response,
			Intent:       s.intent,
			Sentiment:    s.sentiment,
			Topic:        s.topic,
			SuccessCount: 2, // seeds start with a small prior
			FailCount:    0,
			Quality:      0.75,
			LastUsed:     now,
			CreatedAt:    now,
		})
	}

	cl.rebuildIndices()
}

// -----------------------------------------------------------------------
// Query / Stats
// -----------------------------------------------------------------------

// PatternCount returns the total number of learned patterns.
func (cl *ConversationLearner) PatternCount() int {
	cl.mu.RLock()
	defer cl.mu.RUnlock()
	return len(cl.patterns)
}

// Patterns returns a copy of all learned patterns, sorted by quality
// descending.
func (cl *ConversationLearner) Patterns() []ResponsePattern {
	cl.mu.RLock()
	defer cl.mu.RUnlock()

	result := make([]ResponsePattern, len(cl.patterns))
	copy(result, cl.patterns)
	sort.Slice(result, func(i, j int) bool {
		return result[i].Quality > result[j].Quality
	})
	return result
}

// TopPatterns returns the N highest-quality patterns.
func (cl *ConversationLearner) TopPatterns(n int) []ResponsePattern {
	all := cl.Patterns()
	if n > len(all) {
		n = len(all)
	}
	return all[:n]
}

// PatternsByIntent returns all patterns matching a specific intent.
func (cl *ConversationLearner) PatternsByIntent(intent string) []ResponsePattern {
	cl.mu.RLock()
	defer cl.mu.RUnlock()

	indices, ok := cl.byIntent[intent]
	if !ok {
		return nil
	}

	result := make([]ResponsePattern, 0, len(indices))
	for _, idx := range indices {
		if idx < len(cl.patterns) {
			result = append(result, cl.patterns[idx])
		}
	}
	return result
}

// -----------------------------------------------------------------------
// Conversation-to-Graph Learning — self-growing intelligence.
//
// Every substantive response Nous generates contains factual claims.
// This method extracts those claims back into the knowledge graph,
// making the system smarter with every conversation.
//
// Example: Nous generates "Gravity is the fundamental force of
// attraction between all objects with mass or energy." → extracts
// "gravity" is_a "fundamental force", gravity has "mass", etc.
//
// Conversation-derived facts are tagged with lower confidence (0.4)
// and source "conversation:{topic}" for provenance tracking.
// -----------------------------------------------------------------------

// LearnFactsFromResponse extracts factual knowledge from a response
// and ingests it into the knowledge graph. Called after every
// substantive response (>100 chars) from a knowledge source.
func (cl *ConversationLearner) LearnFactsFromResponse(
	response, topic string,
	graph *CognitiveGraph,
) int {
	if len(response) < 100 || graph == nil || topic == "" {
		return 0
	}

	// Extract simple facts from the response text.
	facts := extractConversationFacts(response, topic)
	added := 0

	for _, fact := range facts {
		// Add to graph with lower confidence (conversation-derived).
		graph.AddEdge(fact.from, fact.to, fact.rel, "conversation:"+topic)
		added++
	}

	return added
}

// extractConversationFacts mines typed relationships from response text.
func extractConversationFacts(text, topic string) []sentenceFact {
	var facts []sentenceFact
	lower := strings.ToLower(text)
	topicLower := strings.ToLower(topic)

	// Split into sentences and process each.
	remaining := text
	for {
		idx := strings.Index(remaining, ". ")
		if idx < 0 {
			break
		}
		sentence := strings.TrimSpace(remaining[:idx+1])
		remaining = remaining[idx+2:]

		if len(sentence) < 20 {
			continue
		}

		sentLower := strings.ToLower(sentence)

		// "X is a Y" / "X is the Y" pattern.
		if strings.Contains(sentLower, topicLower+" is a ") || strings.Contains(sentLower, topicLower+" is the ") {
			after := ""
			if idx := strings.Index(sentLower, topicLower+" is a "); idx >= 0 {
				after = sentence[idx+len(topic)+5:]
			} else if idx := strings.Index(sentLower, topicLower+" is the "); idx >= 0 {
				after = sentence[idx+len(topic)+7:]
			}
			if after != "" {
				obj := extractFirstPhrase(after)
				if obj != "" && len(obj) > 3 {
					facts = append(facts, sentenceFact{topic, obj, RelIsA})
				}
			}
		}

		// Causal patterns in conversation text.
		causalPatterns := map[string]RelType{
			" causes ": RelCauses, " enables ": RelEnables,
			" prevents ": RelPrevents, " requires ": RelRequires,
			" produces ": RelProduces,
		}
		for pattern, rel := range causalPatterns {
			if strings.Contains(lower, pattern) {
				// Find the object after the verb.
				if idx := strings.Index(sentLower, pattern); idx >= 0 {
					obj := extractFirstPhrase(sentence[idx+len(pattern):])
					if obj != "" && len(obj) > 3 {
						facts = append(facts, sentenceFact{topic, obj, rel})
					}
				}
			}
		}
	}

	// Deduplicate.
	seen := make(map[string]bool)
	var unique []sentenceFact
	for _, f := range facts {
		key := f.from + "|" + string(f.rel) + "|" + f.to
		if !seen[key] {
			seen[key] = true
			unique = append(unique, f)
		}
	}

	return unique
}

// extractFirstPhrase extracts text up to the first major delimiter.
func extractFirstPhrase(text string) string {
	text = strings.TrimSpace(text)
	// Cut at comma, period, semicolon, or "that"/"which"/"and".
	delimiters := []string{",", ".", ";", " that ", " which ", " and "}
	minIdx := len(text)
	for _, d := range delimiters {
		if idx := strings.Index(text, d); idx >= 0 && idx < minIdx {
			minIdx = idx
		}
	}
	result := strings.TrimSpace(text[:minIdx])
	// Cap at 8 words.
	words := strings.Fields(result)
	if len(words) > 8 {
		result = strings.Join(words[:8], " ")
	}
	return result
}
