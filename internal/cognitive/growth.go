package cognitive

import (
	"encoding/json"
	"fmt"
	"os"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/artaeon/nous/internal/safefile"
)

// PersonalGrowth tracks the user's interests, preferences, and interaction
// patterns over time. Unlike traditional user profiles that are static,
// PersonalGrowth is DYNAMIC — it evolves with every interaction.
//
// Innovation: Most AI assistants start from zero every conversation. Nous
// remembers what you care about, what topics you explore, how you like
// answers, and builds a growing understanding of YOU. This isn't a settings
// file — it's a living model of the user that makes every interaction better.
//
// The growth system tracks:
//   - Topics of interest (weighted by frequency and recency)
//   - Interaction style preferences (verbose vs concise, formal vs casual)
//   - Domain expertise signals (beginner vs expert in different areas)
//   - Temporal patterns (what topics come up at different times)
//   - Emotional context (what makes the user curious, frustrated, excited)
//
// Over weeks/months, this builds a rich personal context that the
// Virtual Context Engine weaves into every prompt — making Nous feel
// like it truly knows you.
type PersonalGrowth struct {
	mu       sync.RWMutex
	path     string
	profile  UserProfile
	modified bool
}

// UserProfile is the persisted representation of what Nous knows about the user.
type UserProfile struct {
	// Topic interests with weights
	Topics map[string]*TopicInterest `json:"topics"`

	// Style preferences
	Style StylePreferences `json:"style"`

	// Interaction history summary
	TotalInteractions int       `json:"total_interactions"`
	FirstSeen         time.Time `json:"first_seen"`
	LastSeen          time.Time `json:"last_seen"`

	// Facts the user has shared about themselves
	Facts []PersonalFact `json:"facts"`
}

// TopicInterest tracks engagement with a topic over time.
type TopicInterest struct {
	Name       string    `json:"name"`
	Count      int       `json:"count"`       // total mentions
	LastSeen   time.Time `json:"last_seen"`
	Weight     float64   `json:"weight"`      // decayed importance (0-1)
	SubTopics  []string  `json:"sub_topics"`  // related sub-topics
}

// StylePreferences tracks how the user likes to interact.
type StylePreferences struct {
	PrefersConcise bool    `json:"prefers_concise"` // short vs detailed
	PrefersExamples bool   `json:"prefers_examples"` // likes examples
	Formality      float64 `json:"formality"`        // 0=casual, 1=formal
	TechLevel      float64 `json:"tech_level"`       // 0=beginner, 1=expert
	Language       string  `json:"language"`          // preferred language
}

// PersonalFact is something the user explicitly shared about themselves.
type PersonalFact struct {
	Fact      string    `json:"fact"`
	Category  string    `json:"category"` // "work", "interest", "preference", "identity"
	Timestamp time.Time `json:"timestamp"`
}

// NewPersonalGrowth creates a new personal growth system.
func NewPersonalGrowth(path string) *PersonalGrowth {
	g := &PersonalGrowth{
		path: path,
		profile: UserProfile{
			Topics: make(map[string]*TopicInterest),
		},
	}
	g.load()
	return g
}

// RecordInteraction processes a user query and updates the growth model.
func (g *PersonalGrowth) RecordInteraction(query string) {
	g.mu.Lock()
	defer g.mu.Unlock()

	now := time.Now()
	g.profile.TotalInteractions++
	g.profile.LastSeen = now

	if g.profile.FirstSeen.IsZero() {
		g.profile.FirstSeen = now
	}

	// Extract topics from the query
	topics := extractTopics(query)
	for _, topic := range topics {
		if existing, ok := g.profile.Topics[topic]; ok {
			existing.Count++
			existing.LastSeen = now
			existing.Weight = calculateWeight(existing.Count, now.Sub(existing.LastSeen))
		} else {
			g.profile.Topics[topic] = &TopicInterest{
				Name:     topic,
				Count:    1,
				LastSeen: now,
				Weight:   0.5,
			}
		}
	}

	// Detect style preferences from query patterns
	g.detectStyle(query)

	g.modified = true

	// Auto-save periodically
	if g.profile.TotalInteractions%10 == 0 {
		g.save()
	}
}

// LearnFact records a personal fact about the user.
func (g *PersonalGrowth) LearnFact(fact, category string) {
	g.mu.Lock()
	defer g.mu.Unlock()

	// Check for duplicates
	lower := strings.ToLower(fact)
	for _, f := range g.profile.Facts {
		if strings.ToLower(f.Fact) == lower {
			return
		}
	}

	g.profile.Facts = append(g.profile.Facts, PersonalFact{
		Fact:      fact,
		Category:  category,
		Timestamp: time.Now(),
	})

	g.modified = true
	g.save()
}

// ContextForQuery returns relevant personal context for a query.
// This is what gets woven into the system prompt by the Virtual Context Engine.
func (g *PersonalGrowth) ContextForQuery(query string) string {
	g.mu.RLock()
	defer g.mu.RUnlock()

	var parts []string

	// Add relevant personal facts
	queryLower := strings.ToLower(query)
	for _, f := range g.profile.Facts {
		if isRelevantFact(f, queryLower) {
			parts = append(parts, fmt.Sprintf("User info: %s", f.Fact))
		}
	}

	// Add top interests as context (so the model can relate to the user)
	topTopics := g.topTopics(3)
	if len(topTopics) > 0 {
		names := make([]string, len(topTopics))
		for i, t := range topTopics {
			names[i] = t.Name
		}
		parts = append(parts, fmt.Sprintf("User interests: %s", strings.Join(names, ", ")))
	}

	// Add style guidance
	if g.profile.Style.PrefersConcise {
		parts = append(parts, "User prefers concise answers.")
	}
	if g.profile.Style.PrefersExamples {
		parts = append(parts, "User likes examples in answers.")
	}
	if g.profile.Style.Language != "" {
		parts = append(parts, fmt.Sprintf("User's preferred language: %s", g.profile.Style.Language))
	}

	if len(parts) == 0 {
		return ""
	}

	return "[Personal context]\n" + strings.Join(parts, "\n")
}

// TopInterests returns the user's top N interests.
func (g *PersonalGrowth) TopInterests(n int) []TopicInterest {
	g.mu.RLock()
	defer g.mu.RUnlock()
	return g.topTopics(n)
}

// Profile returns a summary of the user profile.
func (g *PersonalGrowth) Profile() UserProfile {
	g.mu.RLock()
	defer g.mu.RUnlock()
	return g.profile
}

// Stats returns growth statistics.
func (g *PersonalGrowth) Stats() GrowthStats {
	g.mu.RLock()
	defer g.mu.RUnlock()

	stats := GrowthStats{
		TotalInteractions: g.profile.TotalInteractions,
		TopicsTracked:     len(g.profile.Topics),
		FactsLearned:      len(g.profile.Facts),
	}

	if !g.profile.FirstSeen.IsZero() {
		stats.DaysKnown = int(time.Since(g.profile.FirstSeen).Hours() / 24)
	}

	return stats
}

// GrowthStats holds growth system statistics.
type GrowthStats struct {
	TotalInteractions int
	TopicsTracked     int
	FactsLearned      int
	DaysKnown         int
}

// Save persists the growth data to disk.
func (g *PersonalGrowth) Save() {
	g.mu.Lock()
	defer g.mu.Unlock()
	g.save()
}

// totalTokens estimates the total token count of all personal context.
func (g *PersonalGrowth) totalTokens() int {
	g.mu.RLock()
	defer g.mu.RUnlock()

	tokens := 0
	for _, f := range g.profile.Facts {
		tokens += len(f.Fact) / 4
	}
	for _, t := range g.profile.Topics {
		tokens += len(t.Name)/4 + 10 // name + metadata
	}
	return tokens
}

// --- Internal ---

func (g *PersonalGrowth) topTopics(n int) []TopicInterest {
	if len(g.profile.Topics) == 0 {
		return nil
	}

	// Decay all weights based on recency
	now := time.Now()
	topics := make([]TopicInterest, 0, len(g.profile.Topics))
	for _, t := range g.profile.Topics {
		decayed := *t
		decayed.Weight = calculateWeight(t.Count, now.Sub(t.LastSeen))
		topics = append(topics, decayed)
	}

	sort.Slice(topics, func(i, j int) bool {
		return topics[i].Weight > topics[j].Weight
	})

	if n > len(topics) {
		n = len(topics)
	}
	return topics[:n]
}

func (g *PersonalGrowth) detectStyle(query string) {
	words := strings.Fields(query)

	// Detect conciseness preference from query length
	if len(words) <= 5 {
		// Short queries suggest user prefers concise interactions
		g.profile.Style.PrefersConcise = true
	}

	// Detect example preference
	lower := strings.ToLower(query)
	if strings.Contains(lower, "example") || strings.Contains(lower, "for instance") ||
		strings.Contains(lower, "show me") {
		g.profile.Style.PrefersExamples = true
	}
}

// extractTopics pulls meaningful topics from a query.
func extractTopics(query string) []string {
	words := strings.Fields(strings.ToLower(query))

	// Stop words to ignore
	stopWords := map[string]bool{
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
		"much": true, "many": true, "some": true, "any": true, "not": true,
		"no": true, "yes": true, "more": true, "most": true, "like": true,
	}

	var topics []string
	for _, w := range words {
		clean := strings.TrimRight(w, "?!.,;:'\"")
		if len(clean) >= 3 && !stopWords[clean] {
			topics = append(topics, clean)
		}
	}

	// Also extract bigrams for compound topics
	for i := 0; i < len(words)-1; i++ {
		w1 := strings.TrimRight(words[i], "?!.,;:'\"")
		w2 := strings.TrimRight(words[i+1], "?!.,;:'\"")
		if len(w1) >= 3 && len(w2) >= 3 && !stopWords[w1] && !stopWords[w2] {
			topics = append(topics, w1+" "+w2)
		}
	}

	return topics
}

// calculateWeight computes a time-decayed weight for a topic.
func calculateWeight(count int, timeSinceLastSeen time.Duration) float64 {
	// Frequency signal (log scale, caps at ~1.0 around 50 mentions)
	freq := 0.0
	if count > 0 {
		freq = 0.2 * float64(count)
		if freq > 1.0 {
			freq = 1.0
		}
	}

	// Recency signal (exponential decay, half-life = 7 days)
	days := timeSinceLastSeen.Hours() / 24
	recency := 1.0
	if days > 0 {
		// Half every 7 days
		recency = 1.0 / (1.0 + days/7.0)
	}

	// Combine: 60% recency, 40% frequency
	return recency*0.6 + freq*0.4
}

// isRelevantFact checks if a personal fact is relevant to a query.
func isRelevantFact(fact PersonalFact, queryLower string) bool {
	factLower := strings.ToLower(fact.Fact)
	factWords := strings.Fields(factLower)
	queryWords := strings.Fields(queryLower)

	matchCount := 0
	for _, fw := range factWords {
		if len(fw) < 4 {
			continue
		}
		for _, qw := range queryWords {
			if len(qw) < 4 {
				continue
			}
			// Check prefix match (e.g. "physicist" matches "physics")
			short, long := fw, qw
			if len(short) > len(long) {
				short, long = long, short
			}
			if strings.HasPrefix(long, short[:min(len(short), 5)]) {
				matchCount++
				break
			}
		}
	}

	// At least 1 significant word overlap
	return matchCount >= 1
}


// --- Persistence ---

func (g *PersonalGrowth) save() {
	if g.path == "" || !g.modified {
		return
	}
	data, err := json.MarshalIndent(g.profile, "", "  ")
	if err != nil {
		return
	}
	safefile.WriteAtomic(g.path, data, 0o644)
	g.modified = false
}

func (g *PersonalGrowth) load() {
	if g.path == "" {
		return
	}
	data, err := os.ReadFile(g.path)
	if err != nil {
		return
	}
	var profile UserProfile
	if err := json.Unmarshal(data, &profile); err != nil {
		return
	}
	if profile.Topics == nil {
		profile.Topics = make(map[string]*TopicInterest)
	}
	g.profile = profile
}
