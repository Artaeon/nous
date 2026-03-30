package cognitive

import (
	"fmt"
	"math"
	"sort"
	"strings"
	"sync"
	"time"
)

// InsightCrystallizer watches conversation patterns over time and
// discovers emergent insights — connections, recurring themes, and
// unresolved tensions that the user may not have noticed.
//
// Innovation: Most AI systems are stateless within a conversation and
// amnesic across conversations. InsightCrystallizer accumulates
// observations over time, then applies pattern detection to surface
// things the user never asked about but probably needs to hear.
//
// It is a thinking partner that says "You've mentioned career change
// 5 times this week, always connected to creativity. Your core tension
// might be between security and creative expression."
//
// Detection categories:
//   - Recurring themes — topics mentioned 3+ times
//   - Cross-connections — two unrelated topics that share keywords
//   - Unresolved tensions — conflicting sentiments about the same topic
//   - Trends — topics increasing or decreasing in frequency
//   - Blind spots — topics asked about but never acted on
type InsightCrystallizer struct {
	mu           sync.RWMutex
	themes       []ThemeAccumulator
	connections  []CrossConnection
	tensions     []UnresolvedTension
	observations []Observation
	maxObs       int // max observations to keep (500)

	// Surfacing state
	lastSurfacedAt int // turn index when we last surfaced an insight
	lastObsCount   int // observation count at last surfacing
	crystallized   []CrystallizedInsight
}

// Observation is one atomic piece of conversational data.
type Observation struct {
	Topic     string
	Sentiment string // positive, negative, neutral, conflicted
	Keywords  []string
	Timestamp time.Time
	TurnIndex int
}

// ThemeAccumulator tracks how often a theme appears and in what contexts.
type ThemeAccumulator struct {
	Theme     string
	Count     int
	Contexts  []string // what the user was discussing when this theme appeared
	Sentiment string   // dominant sentiment when this theme comes up
	FirstSeen time.Time
	LastSeen  time.Time
	Trend     string // "increasing", "decreasing", "stable"
}

// CrossConnection is a discovered link between two seemingly unrelated topics.
type CrossConnection struct {
	TopicA      string
	TopicB      string
	SharedWords []string // words that appear in both contexts
	Strength    float64  // 0-1, how strong the connection is
	Insight     string   // human-readable insight about the connection
}

// UnresolvedTension is a detected conflict in the user's thinking.
type UnresolvedTension struct {
	SideA     string // one perspective the user expressed
	SideB     string // conflicting perspective
	Topic     string // what the tension is about
	SeenCount int    // how many times this tension appeared
	Insight   string // articulation of the tension
}

// CrystallizedInsight is a ready-to-surface insight.
type CrystallizedInsight struct {
	Type       string   // "recurring_theme", "cross_connection", "tension", "trend", "blind_spot"
	Text       string   // human-readable insight
	Confidence float64  // 0-1
	Evidence   []string // what observations support this
	Actionable bool     // is there something the user could do about it
	Suggestion string   // optional suggestion
}

// NewInsightCrystallizer creates a new crystallizer ready to accumulate
// observations and discover patterns.
func NewInsightCrystallizer() *InsightCrystallizer {
	return &InsightCrystallizer{
		maxObs: 500,
	}
}

// ---------------------------------------------------------------------------
// Observation Recording
// ---------------------------------------------------------------------------

// Observe records a new observation from a conversation turn.
// query is the raw user input, entities maps entity types to values
// (e.g. "topic" -> "career change"), and sentiment is the detected
// sentiment of the turn.
func (ic *InsightCrystallizer) Observe(query string, entities map[string]string, sentiment string) {
	ic.mu.Lock()
	defer ic.mu.Unlock()

	topic := ""
	if t, ok := entities["topic"]; ok {
		topic = strings.ToLower(strings.TrimSpace(t))
	}

	keywords := extractInsightKeywords(query)

	turnIdx := 0
	if len(ic.observations) > 0 {
		turnIdx = ic.observations[len(ic.observations)-1].TurnIndex + 1
	}

	obs := Observation{
		Topic:     topic,
		Sentiment: normalizeSentiment(sentiment),
		Keywords:  keywords,
		Timestamp: time.Now(),
		TurnIndex: turnIdx,
	}

	ic.observations = append(ic.observations, obs)

	// Evict oldest if over capacity.
	if len(ic.observations) > ic.maxObs {
		ic.observations = ic.observations[len(ic.observations)-ic.maxObs:]
	}

	// Update theme accumulators for this observation.
	ic.accumulateTheme(obs)
}

// ObserveAt is like Observe but allows specifying a timestamp (for testing).
func (ic *InsightCrystallizer) ObserveAt(query string, entities map[string]string, sentiment string, ts time.Time) {
	ic.mu.Lock()
	defer ic.mu.Unlock()

	topic := ""
	if t, ok := entities["topic"]; ok {
		topic = strings.ToLower(strings.TrimSpace(t))
	}

	keywords := extractInsightKeywords(query)

	turnIdx := 0
	if len(ic.observations) > 0 {
		turnIdx = ic.observations[len(ic.observations)-1].TurnIndex + 1
	}

	obs := Observation{
		Topic:     topic,
		Sentiment: normalizeSentiment(sentiment),
		Keywords:  keywords,
		Timestamp: ts,
		TurnIndex: turnIdx,
	}

	ic.observations = append(ic.observations, obs)
	if len(ic.observations) > ic.maxObs {
		ic.observations = ic.observations[len(ic.observations)-ic.maxObs:]
	}
	ic.accumulateTheme(obs)
}

// accumulateTheme updates or creates a ThemeAccumulator for the observation.
// Must be called under write lock.
func (ic *InsightCrystallizer) accumulateTheme(obs Observation) {
	if obs.Topic == "" {
		return
	}
	for i := range ic.themes {
		if ic.themes[i].Theme == obs.Topic {
			ic.themes[i].Count++
			ic.themes[i].LastSeen = obs.Timestamp
			if len(ic.themes[i].Contexts) < 20 {
				ctx := strings.Join(obs.Keywords, " ")
				if ctx != "" {
					ic.themes[i].Contexts = append(ic.themes[i].Contexts, ctx)
				}
			}
			ic.themes[i].Sentiment = ic.dominantSentimentForTopic(obs.Topic)
			return
		}
	}
	// New theme.
	ctx := strings.Join(obs.Keywords, " ")
	var contexts []string
	if ctx != "" {
		contexts = []string{ctx}
	}
	ic.themes = append(ic.themes, ThemeAccumulator{
		Theme:     obs.Topic,
		Count:     1,
		Contexts:  contexts,
		Sentiment: obs.Sentiment,
		FirstSeen: obs.Timestamp,
		LastSeen:  obs.Timestamp,
		Trend:     "stable",
	})
}

// ---------------------------------------------------------------------------
// Crystallization — the core insight detection engine
// ---------------------------------------------------------------------------

// Crystallize analyzes accumulated observations and produces insights.
// It detects recurring themes, cross-connections, unresolved tensions,
// trends, and blind spots.
func (ic *InsightCrystallizer) Crystallize() []CrystallizedInsight {
	ic.mu.Lock()
	defer ic.mu.Unlock()

	var insights []CrystallizedInsight

	insights = append(insights, ic.detectRecurringThemes()...)
	insights = append(insights, ic.detectCrossConnections()...)
	insights = append(insights, ic.detectTensions()...)
	insights = append(insights, ic.detectTrends()...)
	insights = append(insights, ic.detectBlindSpots()...)

	// Sort by confidence descending.
	sort.Slice(insights, func(i, j int) bool {
		return insights[i].Confidence > insights[j].Confidence
	})

	ic.crystallized = insights
	return insights
}

// detectRecurringThemes finds topics mentioned 3+ times.
func (ic *InsightCrystallizer) detectRecurringThemes() []CrystallizedInsight {
	var out []CrystallizedInsight

	for _, th := range ic.themes {
		if th.Count < 3 {
			continue
		}

		evidence := make([]string, 0, th.Count)
		for _, obs := range ic.observations {
			if obs.Topic == th.Theme {
				evidence = append(evidence, fmt.Sprintf("turn %d: discussed %s (%s)",
					obs.TurnIndex, th.Theme, obs.Sentiment))
			}
		}

		sentimentNote := ""
		switch th.Sentiment {
		case "positive":
			sentimentNote = fmt.Sprintf(", and it seems to energize you")
		case "negative":
			sentimentNote = fmt.Sprintf(", often with some concern")
		case "conflicted":
			sentimentNote = fmt.Sprintf(", with mixed feelings")
		}

		text := fmt.Sprintf("I've noticed you keep coming back to %s — you've mentioned it %d times across our conversations%s. It seems important to you.",
			th.Theme, th.Count, sentimentNote)

		confidence := math.Min(1.0, float64(th.Count)/10.0+0.3)

		out = append(out, CrystallizedInsight{
			Type:       "recurring_theme",
			Text:       text,
			Confidence: confidence,
			Evidence:   evidence,
			Actionable: th.Count >= 5,
			Suggestion: ic.themeSuggestion(th),
		})
	}
	return out
}

// detectCrossConnections finds two topics that share keywords but were
// discussed separately.
func (ic *InsightCrystallizer) detectCrossConnections() []CrystallizedInsight {
	var out []CrystallizedInsight

	// Build keyword sets per topic.
	topicKeywords := make(map[string]map[string]int) // topic -> keyword -> count
	for _, obs := range ic.observations {
		if obs.Topic == "" {
			continue
		}
		if _, ok := topicKeywords[obs.Topic]; !ok {
			topicKeywords[obs.Topic] = make(map[string]int)
		}
		for _, kw := range obs.Keywords {
			topicKeywords[obs.Topic][kw]++
		}
	}

	// Find pairs with shared keywords.
	topics := make([]string, 0, len(topicKeywords))
	for t := range topicKeywords {
		topics = append(topics, t)
	}
	sort.Strings(topics)

	seen := make(map[string]bool)
	for i := 0; i < len(topics); i++ {
		for j := i + 1; j < len(topics); j++ {
			a, b := topics[i], topics[j]
			key := a + "|" + b
			if seen[key] {
				continue
			}
			seen[key] = true

			shared := findSharedKeywords(topicKeywords[a], topicKeywords[b])
			if len(shared) == 0 {
				continue
			}

			strength := math.Min(1.0, float64(len(shared))/5.0)
			if strength < 0.5 {
				continue
			}

			sharedStr := strings.Join(shared, ", ")
			insight := fmt.Sprintf("Your interest in %s and %s might be connected — both involve %s.",
				a, b, sharedStr)

			evidence := []string{
				fmt.Sprintf("%s keywords: %s", a, joinTopKeywords(topicKeywords[a], 5)),
				fmt.Sprintf("%s keywords: %s", b, joinTopKeywords(topicKeywords[b], 5)),
				fmt.Sprintf("shared: %s", sharedStr),
			}

			conn := CrossConnection{
				TopicA:      a,
				TopicB:      b,
				SharedWords: shared,
				Strength:    strength,
				Insight:     insight,
			}
			ic.connections = append(ic.connections, conn)

			out = append(out, CrystallizedInsight{
				Type:       "cross_connection",
				Text:       insight,
				Confidence: strength,
				Evidence:   evidence,
				Actionable: true,
				Suggestion: fmt.Sprintf("You might explore the intersection of %s and %s — there could be something interesting there.", a, b),
			})
		}
	}
	return out
}

// detectTensions finds topics where the user expressed conflicting sentiments.
func (ic *InsightCrystallizer) detectTensions() []CrystallizedInsight {
	var out []CrystallizedInsight

	// Group observations by topic and sentiment.
	type topicSentiments struct {
		positive []Observation
		negative []Observation
	}
	byTopic := make(map[string]*topicSentiments)

	for _, obs := range ic.observations {
		if obs.Topic == "" {
			continue
		}
		ts, ok := byTopic[obs.Topic]
		if !ok {
			ts = &topicSentiments{}
			byTopic[obs.Topic] = ts
		}
		switch obs.Sentiment {
		case "positive":
			ts.positive = append(ts.positive, obs)
		case "negative":
			ts.negative = append(ts.negative, obs)
		}
	}

	for topic, ts := range byTopic {
		if len(ts.positive) == 0 || len(ts.negative) == 0 {
			continue
		}

		seenCount := len(ts.positive) + len(ts.negative)

		// Build evidence from both sides.
		evidence := make([]string, 0, seenCount)
		for _, p := range ts.positive {
			evidence = append(evidence, fmt.Sprintf("turn %d: positive about %s (%s)",
				p.TurnIndex, topic, strings.Join(p.Keywords, ", ")))
		}
		for _, n := range ts.negative {
			evidence = append(evidence, fmt.Sprintf("turn %d: negative about %s (%s)",
				n.TurnIndex, topic, strings.Join(n.Keywords, ", ")))
		}

		sideA := fmt.Sprintf("positive feelings about %s (mentioned %d times)", topic, len(ts.positive))
		sideB := fmt.Sprintf("concerns about %s (mentioned %d times)", topic, len(ts.negative))

		insightText := fmt.Sprintf(
			"I've noticed you seem conflicted about %s — sometimes you're excited about it (in %d conversations) and sometimes concerned (in %d conversations). This tension might be worth exploring.",
			topic, len(ts.positive), len(ts.negative))

		tension := UnresolvedTension{
			SideA:     sideA,
			SideB:     sideB,
			Topic:     topic,
			SeenCount: seenCount,
			Insight:   insightText,
		}
		ic.tensions = append(ic.tensions, tension)

		confidence := math.Min(1.0, float64(seenCount)/8.0+0.2)

		out = append(out, CrystallizedInsight{
			Type:       "tension",
			Text:       insightText,
			Confidence: confidence,
			Evidence:   evidence,
			Actionable: true,
			Suggestion: fmt.Sprintf("It might help to articulate what specifically excites you about %s and what concerns you — seeing both sides clearly can help you decide.", topic),
		})
	}
	return out
}

// detectTrends finds topics increasing or decreasing in frequency.
func (ic *InsightCrystallizer) detectTrends() []CrystallizedInsight {
	var out []CrystallizedInsight

	if len(ic.observations) < 6 {
		return nil
	}

	// Split observations into two halves by turn index and compare
	// topic frequency in each half.
	mid := len(ic.observations) / 2
	firstHalf := ic.observations[:mid]
	secondHalf := ic.observations[mid:]

	firstCounts := countTopics(firstHalf)
	secondCounts := countTopics(secondHalf)

	// Find topics that are increasing.
	allTopics := make(map[string]bool)
	for t := range firstCounts {
		allTopics[t] = true
	}
	for t := range secondCounts {
		allTopics[t] = true
	}

	for topic := range allTopics {
		first := firstCounts[topic]
		second := secondCounts[topic]
		total := first + second
		if total < 3 {
			continue
		}

		var trend string
		var text string

		if second > first && second >= 2 {
			trend = "increasing"
			text = fmt.Sprintf("I've noticed you've been thinking about %s more and more lately — it came up %d times recently compared to %d times earlier.", topic, second, first)
			// Update theme trend.
			for i := range ic.themes {
				if ic.themes[i].Theme == topic {
					ic.themes[i].Trend = "increasing"
				}
			}
		} else if first > second && first >= 2 {
			trend = "decreasing"
			text = fmt.Sprintf("You used to bring up %s more often (%d times earlier vs %d recently). Your focus may be shifting.", topic, first, second)
			for i := range ic.themes {
				if ic.themes[i].Theme == topic {
					ic.themes[i].Trend = "decreasing"
				}
			}
		} else {
			continue
		}

		confidence := math.Min(1.0, float64(total)/10.0+0.2)

		out = append(out, CrystallizedInsight{
			Type:       "trend",
			Text:       text,
			Confidence: confidence,
			Evidence: []string{
				fmt.Sprintf("first half: %s mentioned %d times", topic, first),
				fmt.Sprintf("second half: %s mentioned %d times", topic, second),
				fmt.Sprintf("trend: %s", trend),
			},
			Actionable: trend == "increasing",
		})
	}
	return out
}

// detectBlindSpots finds topics the user asks about repeatedly but never
// discusses concrete next steps for.
func (ic *InsightCrystallizer) detectBlindSpots() []CrystallizedInsight {
	var out []CrystallizedInsight

	// Action words that suggest concrete next steps.
	actionWords := map[string]bool{
		"plan": true, "start": true, "begin": true, "try": true,
		"schedule": true, "book": true, "apply": true, "build": true,
		"create": true, "launch": true, "implement": true, "buy": true,
		"sign": true, "register": true, "commit": true, "decide": true,
		"choose": true, "deadline": true, "goal": true, "step": true,
		"action": true, "do": true, "make": true, "write": true,
	}

	// For each topic with 3+ mentions, check if any observation has action keywords.
	for _, th := range ic.themes {
		if th.Count < 3 {
			continue
		}
		hasAction := false
		for _, obs := range ic.observations {
			if obs.Topic != th.Theme {
				continue
			}
			for _, kw := range obs.Keywords {
				if actionWords[kw] {
					hasAction = true
					break
				}
			}
			if hasAction {
				break
			}
		}
		if hasAction {
			continue
		}

		text := fmt.Sprintf("I've noticed you ask about %s frequently (%d times) but haven't discussed concrete next steps. If this matters to you, it might help to think about what action you could take.",
			th.Theme, th.Count)

		out = append(out, CrystallizedInsight{
			Type:       "blind_spot",
			Text:       text,
			Confidence: math.Min(1.0, float64(th.Count)/8.0+0.2),
			Evidence: []string{
				fmt.Sprintf("%s mentioned %d times", th.Theme, th.Count),
				"no action-oriented keywords detected in related observations",
			},
			Actionable: true,
			Suggestion: fmt.Sprintf("What would a first small step toward %s look like for you?", th.Theme),
		})
	}
	return out
}

// ---------------------------------------------------------------------------
// Surfacing — deciding when and what to show
// ---------------------------------------------------------------------------

// ShouldSurface returns true if it's a good time to surface an insight.
// It avoids being noisy — only surfacing when meaningful conditions are met.
func (ic *InsightCrystallizer) ShouldSurface(turnCount int) bool {
	ic.mu.RLock()
	defer ic.mu.RUnlock()

	obsCount := len(ic.observations)

	// Need at least some observations.
	if obsCount < 3 {
		return false
	}

	// Enough new observations since last surfacing.
	newObs := obsCount - ic.lastObsCount
	if newObs >= 10 {
		return true
	}

	// Strong pattern just detected: re-crystallize and check.
	for _, insight := range ic.crystallized {
		if insight.Confidence >= 0.7 {
			return true
		}
	}

	return false
}

// SurfaceRelevant returns an insight relevant to the current topic, if any.
// Returns nil if nothing relevant is crystallized.
func (ic *InsightCrystallizer) SurfaceRelevant(currentTopic string) *CrystallizedInsight {
	ic.mu.Lock()
	defer ic.mu.Unlock()

	currentTopic = strings.ToLower(strings.TrimSpace(currentTopic))
	if currentTopic == "" {
		return nil
	}

	// Crystallize if we haven't yet.
	if len(ic.crystallized) == 0 {
		ic.crystallized = ic.crystallizeUnlocked()
	}

	var best *CrystallizedInsight
	bestScore := 0.0

	for i := range ic.crystallized {
		insight := &ic.crystallized[i]
		score := ic.relevanceScore(insight, currentTopic)
		if score > bestScore {
			bestScore = score
			best = insight
		}
	}

	if best == nil || bestScore < 0.3 || best.Confidence < 0.5 {
		return nil
	}

	// Mark that we surfaced.
	if len(ic.observations) > 0 {
		ic.lastSurfacedAt = ic.observations[len(ic.observations)-1].TurnIndex
	}
	ic.lastObsCount = len(ic.observations)

	return best
}

// crystallizeUnlocked is like Crystallize but without locking (caller holds lock).
func (ic *InsightCrystallizer) crystallizeUnlocked() []CrystallizedInsight {
	var insights []CrystallizedInsight
	insights = append(insights, ic.detectRecurringThemes()...)
	insights = append(insights, ic.detectCrossConnections()...)
	insights = append(insights, ic.detectTensions()...)
	insights = append(insights, ic.detectTrends()...)
	insights = append(insights, ic.detectBlindSpots()...)

	sort.Slice(insights, func(i, j int) bool {
		return insights[i].Confidence > insights[j].Confidence
	})
	return insights
}

// relevanceScore computes how relevant an insight is to the current topic.
func (ic *InsightCrystallizer) relevanceScore(insight *CrystallizedInsight, topic string) float64 {
	text := strings.ToLower(insight.Text)

	// Direct topic mention in the insight text.
	if strings.Contains(text, topic) {
		return insight.Confidence
	}

	// Check evidence lines.
	for _, ev := range insight.Evidence {
		if strings.Contains(strings.ToLower(ev), topic) {
			return insight.Confidence * 0.8
		}
	}

	// Check suggestion.
	if strings.Contains(strings.ToLower(insight.Suggestion), topic) {
		return insight.Confidence * 0.6
	}

	return 0
}

// ---------------------------------------------------------------------------
// Stats and accessors
// ---------------------------------------------------------------------------

// ObservationCount returns the number of recorded observations.
func (ic *InsightCrystallizer) ObservationCount() int {
	ic.mu.RLock()
	defer ic.mu.RUnlock()
	return len(ic.observations)
}

// Themes returns a copy of the current theme accumulators.
func (ic *InsightCrystallizer) Themes() []ThemeAccumulator {
	ic.mu.RLock()
	defer ic.mu.RUnlock()
	out := make([]ThemeAccumulator, len(ic.themes))
	copy(out, ic.themes)
	return out
}

// Tensions returns a copy of detected tensions.
func (ic *InsightCrystallizer) Tensions() []UnresolvedTension {
	ic.mu.RLock()
	defer ic.mu.RUnlock()
	out := make([]UnresolvedTension, len(ic.tensions))
	copy(out, ic.tensions)
	return out
}

// Connections returns a copy of detected cross-connections.
func (ic *InsightCrystallizer) Connections() []CrossConnection {
	ic.mu.RLock()
	defer ic.mu.RUnlock()
	out := make([]CrossConnection, len(ic.connections))
	copy(out, ic.connections)
	return out
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// insightStopWords are common words excluded from keyword extraction.
var insightStopWords = map[string]bool{
	"the": true, "a": true, "an": true, "is": true, "are": true,
	"was": true, "were": true, "be": true, "been": true, "being": true,
	"have": true, "has": true, "had": true, "do": true, "does": true,
	"did": true, "will": true, "would": true, "could": true, "should": true,
	"may": true, "might": true, "shall": true, "can": true,
	"and": true, "or": true, "but": true, "if": true, "then": true,
	"of": true, "in": true, "on": true, "at": true, "to": true,
	"for": true, "with": true, "from": true, "by": true, "about": true,
	"this": true, "that": true, "these": true, "those": true,
	"it": true, "its": true, "i": true, "me": true, "my": true,
	"you": true, "your": true, "we": true, "our": true,
	"what": true, "which": true, "who": true, "how": true, "where": true,
	"when": true, "why": true, "all": true, "each": true, "every": true,
	"tell": true, "know": true, "help": true, "want": true, "need": true,
	"please": true, "thanks": true, "thank": true, "hi": true, "hello": true,
	"hey": true, "there": true, "just": true, "also": true, "very": true,
	"so": true, "not": true, "no": true, "yes": true, "up": true,
	"out": true, "some": true, "more": true, "much": true, "like": true,
	"really": true, "get": true, "got": true, "think": true, "thing": true,
	"things": true, "going": true, "go": true, "see": true, "look": true,
	"into": true, "them": true, "they": true, "their": true,
	"too": true, "any": true, "only": true, "than": true, "own": true,
	"most": true, "other": true, "over": true, "such": true, "even": true,
	"back": true, "well": true, "way": true, "new": true, "still": true,
	"now": true, "come": true, "after": true, "since": true, "through": true,
	"between": true, "because": true, "here": true, "before": true,
	"take": true, "many": true, "same": true, "both": true, "something": true,
	"him": true, "her": true, "she": true, "he": true, "his": true,
	"am": true, "im": true, "ive": true, "dont": true, "cant": true,
}

// extractInsightKeywords pulls meaningful words from a query.
func extractInsightKeywords(query string) []string {
	words := strings.Fields(strings.ToLower(query))
	var keywords []string
	seen := make(map[string]bool)
	for _, w := range words {
		// Strip punctuation from edges.
		w = strings.Trim(w, ".,!?;:'\"()-[]{}/*")
		if len(w) < 2 {
			continue
		}
		if insightStopWords[w] {
			continue
		}
		if seen[w] {
			continue
		}
		seen[w] = true
		keywords = append(keywords, w)
	}
	return keywords
}

// normalizeSentiment normalizes a sentiment string to one of the canonical values.
func normalizeSentiment(s string) string {
	s = strings.ToLower(strings.TrimSpace(s))
	switch s {
	case "positive", "negative", "neutral", "conflicted":
		return s
	case "pos":
		return "positive"
	case "neg":
		return "negative"
	default:
		return "neutral"
	}
}

// dominantSentimentForTopic computes the most common sentiment for a topic.
// Must be called under lock.
func (ic *InsightCrystallizer) dominantSentimentForTopic(topic string) string {
	counts := map[string]int{}
	for _, obs := range ic.observations {
		if obs.Topic == topic {
			counts[obs.Sentiment]++
		}
	}
	// If we have both positive and negative, it's conflicted.
	if counts["positive"] > 0 && counts["negative"] > 0 {
		return "conflicted"
	}
	best := "neutral"
	bestCount := 0
	for s, c := range counts {
		if c > bestCount {
			best = s
			bestCount = c
		}
	}
	return best
}

// findSharedKeywords returns keywords that appear in both maps.
func findSharedKeywords(a, b map[string]int) []string {
	var shared []string
	for kw := range a {
		if _, ok := b[kw]; ok {
			// Skip date/number-only connections — they create noise
			// like "blockchain connects to Bitcoin" through "2009"
			if isDateOrNumberKeyword(kw) {
				continue
			}
			shared = append(shared, kw)
		}
	}
	sort.Strings(shared)
	return shared
}

// isDateOrNumberKeyword returns true if a keyword is just a date or number.
func isDateOrNumberKeyword(kw string) bool {
	// Pure numbers: "2009", "100", "1956"
	allDigit := true
	for _, r := range kw {
		if r < '0' || r > '9' {
			allDigit = false
			break
		}
	}
	if allDigit && len(kw) > 0 {
		return true
	}
	// Month names, ordinals
	months := map[string]bool{
		"january": true, "february": true, "march": true, "april": true,
		"may": true, "june": true, "july": true, "august": true,
		"september": true, "october": true, "november": true, "december": true,
	}
	return months[strings.ToLower(kw)]
}

// joinTopKeywords returns the top-N keywords from a frequency map as a string.
func joinTopKeywords(m map[string]int, n int) string {
	type kv struct {
		k string
		v int
	}
	var pairs []kv
	for k, v := range m {
		pairs = append(pairs, kv{k, v})
	}
	sort.Slice(pairs, func(i, j int) bool { return pairs[i].v > pairs[j].v })
	var out []string
	for i, p := range pairs {
		if i >= n {
			break
		}
		out = append(out, p.k)
	}
	return strings.Join(out, ", ")
}

// countTopics counts topic occurrences in a slice of observations.
func countTopics(obs []Observation) map[string]int {
	counts := make(map[string]int)
	for _, o := range obs {
		if o.Topic != "" {
			counts[o.Topic]++
		}
	}
	return counts
}

// themeSuggestion produces a suggestion for a recurring theme.
func (ic *InsightCrystallizer) themeSuggestion(th ThemeAccumulator) string {
	if th.Count >= 5 {
		return fmt.Sprintf("Since %s keeps coming up, would it help to explore it more deeply or set a specific goal around it?", th.Theme)
	}
	return ""
}
