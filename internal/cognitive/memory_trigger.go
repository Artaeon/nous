package cognitive

import (
	"fmt"
	"math/rand"
	"strings"
	"sync"
	"time"

	"github.com/artaeon/nous/internal/memory"
	"github.com/artaeon/nous/internal/simd"
)

// -----------------------------------------------------------------------
// Memory Trigger Engine — involuntary episodic recall
//
// Every conversation turn, the engine scans episodic memory for relevant
// past interactions and surfaces them when confidence is high enough.
// This is the AI equivalent of "that reminds me..." — unprompted,
// associative recall that makes conversation feel alive.
//
// Four scan types, ranked by power:
//   1. Semantic — embedding cosine similarity (best, requires embedFn)
//   2. Topic   — keyword overlap on episode tags (fast fallback)
//   3. Temporal — cyclical patterns (same time last week/month)
//   4. Pattern — repeated questions (user might be stuck)
// -----------------------------------------------------------------------

// MemoryTrigger is a single surfaced memory with context about why
// it was triggered and how relevant it is.
type MemoryTrigger struct {
	Episode     memory.Episode
	Similarity  float64       // 0.0–1.0
	TriggerType string        // "semantic", "topic", "temporal", "pattern"
	Relevance   string        // human-readable why this surfaced
	Age         time.Duration // how old the memory is
}

// MemoryTriggerEngine scans episodic memory every turn for involuntary recall.
type MemoryTriggerEngine struct {
	episodic   *memory.EpisodicMemory
	embedFn    memory.EmbedFunc
	threshold  float64              // minimum similarity to trigger
	cooldown   map[string]time.Time // episode ID → last surfaced time
	maxPerTurn int                  // max memories to surface per turn
	minAge     time.Duration        // minimum age to surface (don't echo recent)
	mu         sync.Mutex
}

// NewMemoryTriggerEngine creates a trigger engine with sane defaults.
func NewMemoryTriggerEngine(episodic *memory.EpisodicMemory, embedFn memory.EmbedFunc) *MemoryTriggerEngine {
	return &MemoryTriggerEngine{
		episodic:   episodic,
		embedFn:    embedFn,
		threshold:  0.60,
		cooldown:   make(map[string]time.Time),
		maxPerTurn: 1,
		minAge:     time.Hour,
	}
}

// Scan is the main entry point — called every conversation turn.
// It runs all scan types, deduplicates, ranks by relevance, respects
// cooldown and limits, and returns at most maxPerTurn triggers.
func (mte *MemoryTriggerEngine) Scan(input string, currentTopics []string) []MemoryTrigger {
	if mte.episodic == nil || mte.episodic.Size() == 0 {
		return nil
	}

	var all []MemoryTrigger

	// Run all scan types and collect candidates
	all = append(all, mte.semanticScan(input)...)
	all = append(all, mte.topicScan(currentTopics)...)
	all = append(all, mte.temporalScan()...)
	all = append(all, mte.patternScan(input)...)

	// Deduplicate by episode ID, keeping the highest-similarity entry
	all = deduplicateTriggers(all)

	// Filter through shouldSurface (cooldown, age, threshold)
	var valid []MemoryTrigger
	for _, t := range all {
		if mte.shouldSurface(t) {
			valid = append(valid, t)
		}
	}

	// Sort by similarity descending
	for i := 0; i < len(valid); i++ {
		for j := i + 1; j < len(valid); j++ {
			if valid[j].Similarity > valid[i].Similarity {
				valid[i], valid[j] = valid[j], valid[i]
			}
		}
	}

	// Limit to maxPerTurn
	if len(valid) > mte.maxPerTurn {
		valid = valid[:mte.maxPerTurn]
	}

	return valid
}

// semanticScan embeds the current input and finds episodic memories with
// high cosine similarity. This is the most powerful scan — it catches
// conceptual connections that keyword matching would miss.
func (mte *MemoryTriggerEngine) semanticScan(input string) []MemoryTrigger {
	if mte.embedFn == nil || input == "" {
		return nil
	}

	vec, err := mte.embedFn(input)
	if err != nil || len(vec) == 0 {
		return nil
	}

	episodes := mte.episodic.SearchSemantic(vec, 5)
	now := time.Now()
	var triggers []MemoryTrigger

	for _, ep := range episodes {
		if len(ep.Embedding) == 0 {
			continue
		}
		sim := simd.CosineSimilarity(vec, ep.Embedding)
		if sim < mte.threshold {
			continue
		}
		age := now.Sub(ep.Timestamp)
		triggers = append(triggers, MemoryTrigger{
			Episode:     ep,
			Similarity:  sim,
			TriggerType: "semantic",
			Relevance:   fmt.Sprintf("semantically similar to current input (%.0f%%)", sim*100),
			Age:         age,
		})
	}

	return triggers
}

// topicScan matches current conversation topics against episode tags.
// Fast keyword-based fallback when embeddings aren't available.
func (mte *MemoryTriggerEngine) topicScan(topics []string) []MemoryTrigger {
	if len(topics) == 0 {
		return nil
	}

	// Build a combined query from topics for keyword search
	query := strings.Join(topics, " ")
	episodes := mte.episodic.SearchKeyword(query, 10)
	now := time.Now()

	topicSet := make(map[string]bool, len(topics))
	for _, t := range topics {
		topicSet[strings.ToLower(t)] = true
	}

	var triggers []MemoryTrigger
	for _, ep := range episodes {
		// Count how many current topics match episode tags
		matchCount := 0
		var matchedTopics []string
		for _, tag := range ep.Tags {
			if topicSet[strings.ToLower(tag)] {
				matchCount++
				matchedTopics = append(matchedTopics, tag)
			}
		}
		if matchCount == 0 {
			continue
		}

		// Similarity is the fraction of current topics that matched
		sim := float64(matchCount) / float64(len(topics))
		// Boost slightly for episodes with many matching tags
		if matchCount > 1 {
			sim += 0.05 * float64(matchCount-1)
		}
		if sim > 1.0 {
			sim = 1.0
		}

		age := now.Sub(ep.Timestamp)
		triggers = append(triggers, MemoryTrigger{
			Episode:     ep,
			Similarity:  sim,
			TriggerType: "topic",
			Relevance:   fmt.Sprintf("shares topics: %s", strings.Join(matchedTopics, ", ")),
			Age:         age,
		})
	}

	return triggers
}

// temporalScan finds memories from cyclical time patterns — same time
// last week, same day last month. People have routines: Monday standups,
// Friday reviews, Sunday planning sessions. Noticing these patterns
// makes the AI feel like it actually remembers your life.
func (mte *MemoryTriggerEngine) temporalScan() []MemoryTrigger {
	now := time.Now()

	// Retrieve a large window of episodes to scan for temporal matches.
	// We need enough history to find weekly/monthly patterns.
	episodes := mte.episodic.Recent(mte.episodic.Size())
	if len(episodes) == 0 {
		return nil
	}

	var triggers []MemoryTrigger

	for _, ep := range episodes {
		age := now.Sub(ep.Timestamp)

		// Skip too-recent episodes
		if age < mte.minAge {
			continue
		}

		sim, label := mte.temporalMatch(now, ep.Timestamp)
		if sim == 0 {
			continue
		}

		triggers = append(triggers, MemoryTrigger{
			Episode:     ep,
			Similarity:  sim,
			TriggerType: "temporal",
			Relevance:   label,
			Age:         age,
		})
	}

	// Only keep the best temporal match
	if len(triggers) > 1 {
		best := 0
		for i := 1; i < len(triggers); i++ {
			if triggers[i].Similarity > triggers[best].Similarity {
				best = i
			}
		}
		triggers = []MemoryTrigger{triggers[best]}
	}

	return triggers
}

// temporalMatch checks if an episode timestamp matches a cyclical pattern
// relative to now. Returns a similarity score and human-readable label.
func (mte *MemoryTriggerEngine) temporalMatch(now, then time.Time) (float64, string) {
	daysSince := now.Sub(then).Hours() / 24

	// Weekly pattern: 7 days ago, ±1 day
	if daysSince >= 6 && daysSince <= 8 {
		// Check time-of-day match: ±2 hours
		hourDiff := absInt(now.Hour() - then.Hour())
		if hourDiff <= 2 || hourDiff >= 22 { // handle wrap-around
			return 0.65, "around this time last week"
		}
	}

	// Two weeks ago
	if daysSince >= 13 && daysSince <= 15 {
		hourDiff := absInt(now.Hour() - then.Hour())
		if hourDiff <= 2 || hourDiff >= 22 {
			return 0.60, "around this time two weeks ago"
		}
	}

	// Monthly pattern: 28-31 days ago, ±1 day
	if daysSince >= 27 && daysSince <= 32 {
		hourDiff := absInt(now.Hour() - then.Hour())
		if hourDiff <= 2 || hourDiff >= 22 {
			return 0.62, "around this time last month"
		}
	}

	return 0, ""
}

// patternScan detects repeated questions — when the user keeps asking
// about the same topic, they might be stuck. This is different from
// topicScan: it specifically looks for REPEATED queries with similar
// phrasing, not just shared tags.
func (mte *MemoryTriggerEngine) patternScan(input string) []MemoryTrigger {
	if input == "" {
		return nil
	}

	// Normalize input for comparison
	normalized := strings.ToLower(strings.TrimSpace(input))
	if len(normalized) < 5 {
		return nil
	}

	// Search for episodes with similar input text
	episodes := mte.episodic.SearchKeyword(input, 20)
	now := time.Now()

	// Count how many times similar queries appear
	var repeatEpisodes []memory.Episode
	for _, ep := range episodes {
		epNorm := strings.ToLower(strings.TrimSpace(ep.Input))

		// Check for substantial overlap in words
		overlap := wordOverlap(normalized, epNorm)
		if overlap >= 0.5 {
			repeatEpisodes = append(repeatEpisodes, ep)
		}
	}

	// Need at least 2 previous occurrences to call it a pattern
	if len(repeatEpisodes) < 2 {
		return nil
	}

	// Use the most recent repeat as the trigger episode
	mostRecent := repeatEpisodes[0]
	for _, ep := range repeatEpisodes[1:] {
		if ep.Timestamp.After(mostRecent.Timestamp) {
			mostRecent = ep
		}
	}

	age := now.Sub(mostRecent.Timestamp)
	count := len(repeatEpisodes)

	// Higher similarity for more repeats
	sim := 0.60 + 0.05*float64(count)
	if sim > 0.95 {
		sim = 0.95
	}

	return []MemoryTrigger{{
		Episode:     mostRecent,
		Similarity:  sim,
		TriggerType: "pattern",
		Relevance:   fmt.Sprintf("asked %d times before", count),
		Age:         age,
	}}
}

// shouldSurface checks whether a trigger should actually be shown:
// cooldown (same episode can't surface within 24 hours), minimum age
// (don't echo what just happened), and minimum similarity threshold.
func (mte *MemoryTriggerEngine) shouldSurface(trigger MemoryTrigger) bool {
	// Minimum similarity threshold
	if trigger.Similarity < mte.threshold {
		return false
	}

	// Minimum age — don't echo recent interactions
	if trigger.Age < mte.minAge {
		return false
	}

	// Cooldown — same episode can't resurface within 24 hours
	mte.mu.Lock()
	lastSurfaced, exists := mte.cooldown[trigger.Episode.ID]
	mte.mu.Unlock()

	if exists && time.Since(lastSurfaced) < 24*time.Hour {
		return false
	}

	return true
}

// RecordSurfaced marks an episode as recently surfaced for cooldown tracking.
func (mte *MemoryTriggerEngine) RecordSurfaced(episodeID string) {
	mte.mu.Lock()
	mte.cooldown[episodeID] = time.Now()
	mte.mu.Unlock()

	// Prune old cooldown entries to prevent unbounded growth
	mte.pruneCooldown()
}

// pruneCooldown removes cooldown entries older than 48 hours.
func (mte *MemoryTriggerEngine) pruneCooldown() {
	mte.mu.Lock()
	defer mte.mu.Unlock()

	cutoff := time.Now().Add(-48 * time.Hour)
	for id, t := range mte.cooldown {
		if t.Before(cutoff) {
			delete(mte.cooldown, id)
		}
	}
}

// FormatTrigger generates natural phrasing for the triggered memory.
// Each trigger type has a pool of phrasings to avoid sounding robotic.
func (mte *MemoryTriggerEngine) FormatTrigger(trigger MemoryTrigger) string {
	age := formatAge(trigger.Age)
	topic := extractTopicSummary(trigger.Episode)
	quote := truncate(trigger.Episode.Input, 80)

	switch trigger.TriggerType {
	case "semantic":
		return pickRandom(semanticPhrases, age, topic, quote)
	case "topic":
		return pickRandom(topicPhrases, age, topic, quote)
	case "temporal":
		return pickRandom(temporalPhrases, age, topic)
	case "pattern":
		return pickRandom(patternPhrases, topic, age, trigger.Relevance)
	default:
		return fmt.Sprintf("I recall something related — %s you were discussing %s.", age, topic)
	}
}

// --- phrase pools ---

var semanticPhrases = []string{
	"That reminds me — %s you mentioned %s. You said: \"%s\"",
	"This connects to something from %s — you were talking about %s. \"%s\"",
	"I'm reminded of a conversation from %s about %s. You asked: \"%s\"",
	"Something similar came up %s when you were exploring %s. \"%s\"",
	"This echoes what you said %s about %s: \"%s\"",
}

var topicPhrases = []string{
	"You've talked about %s before — %s you said: \"%s\"",
	"This topic came up %s. You mentioned %s: \"%s\"",
	"I remember %s discussing %s with you. You said: \"%s\"",
	"We've been here before — %s you were asking about %s: \"%s\"",
}

var temporalPhrases = []string{
	"Around this time %s, you were working on %s.",
	"If I recall, %s you were focused on %s.",
	"Thinking back to %s — you were into %s.",
}

var patternPhrases = []string{
	"You've asked about %s a few times now — last was %s. %s",
	"This keeps coming up — you've asked about %s before, %s. %s",
	"I've noticed a pattern — you come back to %s regularly, last %s. %s",
	"You keep coming back to %s — last time was %s. %s",
}

// --- helpers ---

// deduplicateTriggers keeps only the highest-similarity trigger per episode ID.
func deduplicateTriggers(triggers []MemoryTrigger) []MemoryTrigger {
	best := make(map[string]MemoryTrigger)
	for _, t := range triggers {
		existing, ok := best[t.Episode.ID]
		if !ok || t.Similarity > existing.Similarity {
			best[t.Episode.ID] = t
		}
	}

	result := make([]MemoryTrigger, 0, len(best))
	for _, t := range best {
		result = append(result, t)
	}
	return result
}

// wordOverlap returns the fraction of words in a that also appear in b.
func wordOverlap(a, b string) float64 {
	wordsA := strings.Fields(a)
	if len(wordsA) == 0 {
		return 0
	}

	setB := make(map[string]bool, len(strings.Fields(b)))
	for _, w := range strings.Fields(b) {
		setB[w] = true
	}

	matches := 0
	for _, w := range wordsA {
		if setB[w] {
			matches++
		}
	}
	return float64(matches) / float64(len(wordsA))
}

// formatAge converts a duration to a human-readable age string.
func formatAge(d time.Duration) string {
	hours := d.Hours()
	switch {
	case hours < 2:
		return "about an hour ago"
	case hours < 24:
		return fmt.Sprintf("about %d hours ago", int(hours))
	case hours < 48:
		return "yesterday"
	case hours < 24*7:
		return fmt.Sprintf("%d days ago", int(hours/24))
	case hours < 24*14:
		return "last week"
	case hours < 24*30:
		return fmt.Sprintf("%d weeks ago", int(hours/(24*7)))
	case hours < 24*60:
		return "last month"
	default:
		return fmt.Sprintf("%d months ago", int(hours/(24*30)))
	}
}

// extractTopicSummary pulls a short topic description from an episode.
func extractTopicSummary(ep memory.Episode) string {
	// Filter out stopwords from tags to get meaningful topic words
	if len(ep.Tags) > 0 {
		stops := map[string]bool{
			"what": true, "how": true, "does": true, "is": true, "are": true,
			"the": true, "a": true, "an": true, "do": true, "can": true,
			"about": true, "tell": true, "me": true, "you": true, "your": true,
			"think": true, "know": true, "work": true, "explain": true,
			"should": true, "would": true, "could": true, "will": true,
		}
		var meaningful []string
		for _, tag := range ep.Tags {
			if !stops[strings.ToLower(tag)] && len(tag) > 2 {
				meaningful = append(meaningful, tag)
			}
		}
		if len(meaningful) > 0 {
			limit := len(meaningful)
			if limit > 3 {
				limit = 3
			}
			return strings.Join(meaningful[:limit], ", ")
		}
	}
	return truncate(ep.Input, 40)
}

// truncate shortens a string to maxLen, adding "..." if needed.
func truncate(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	// Cut at last space before maxLen to avoid mid-word truncation
	cut := strings.LastIndex(s[:maxLen], " ")
	if cut < maxLen/2 {
		cut = maxLen
	}
	return s[:cut] + "..."
}

// pickRandom selects a random phrase from the pool and formats it.
func pickRandom(pool []string, args ...any) string {
	if len(pool) == 0 {
		return ""
	}
	template := pool[rand.Intn(len(pool))]
	return fmt.Sprintf(template, args...)
}

// absInt returns the absolute value of an int.
func absInt(n int) int {
	if n < 0 {
		return -n
	}
	return n
}
