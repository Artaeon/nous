package cognitive

import (
	"strings"
	"testing"
	"time"

	"github.com/artaeon/nous/internal/memory"
)

// --- test helpers ---

// newTestEpisodicMemory creates an in-memory episodic store populated
// with the given episodes. embedFn is optional.
func newTestEpisodicMemory(episodes []memory.Episode, embedFn memory.EmbedFunc) *memory.EpisodicMemory {
	em := memory.NewEpisodicMemory("", embedFn)
	for _, ep := range episodes {
		em.Record(ep)
	}
	return em
}

func makeEpisode(id string, input string, tags []string, ts time.Time) memory.Episode {
	return memory.Episode{
		ID:        id,
		Timestamp: ts,
		Input:     input,
		Intent:    "test",
		Output:    "test response",
		Success:   true,
		Tags:      tags,
	}
}

// -----------------------------------------------------------------------
// Topic matching
// -----------------------------------------------------------------------

func TestTopicScan_MatchesSharedTags(t *testing.T) {
	now := time.Now()
	episodes := []memory.Episode{
		makeEpisode("ep-old", "how do goroutines work", []string{"goroutines", "concurrency", "golang"}, now.Add(-48*time.Hour)),
		makeEpisode("ep-unrelated", "what is the weather today", []string{"weather", "today"}, now.Add(-24*time.Hour)),
	}
	em := newTestEpisodicMemory(episodes, nil)
	mte := NewMemoryTriggerEngine(em, nil)

	triggers := mte.topicScan([]string{"goroutines", "concurrency"})
	if len(triggers) == 0 {
		t.Fatal("expected at least one topic trigger")
	}

	found := false
	for _, tr := range triggers {
		if tr.Episode.ID == "ep-old" || strings.Contains(tr.Relevance, "goroutines") {
			found = true
			break
		}
	}
	if !found {
		t.Error("expected trigger for goroutines episode")
	}
}

func TestTopicScan_NoMatchOnUnrelatedTopics(t *testing.T) {
	now := time.Now()
	episodes := []memory.Episode{
		makeEpisode("ep1", "explain monads", []string{"monads", "haskell", "functional"}, now.Add(-72*time.Hour)),
	}
	em := newTestEpisodicMemory(episodes, nil)
	mte := NewMemoryTriggerEngine(em, nil)

	triggers := mte.topicScan([]string{"cooking", "recipes"})
	if len(triggers) != 0 {
		t.Errorf("expected no triggers for unrelated topics, got %d", len(triggers))
	}
}

// -----------------------------------------------------------------------
// Temporal scanning
// -----------------------------------------------------------------------

func TestTemporalScan_OneWeekAgo(t *testing.T) {
	now := time.Now()
	// Episode from exactly 7 days ago at the same hour
	weekAgo := now.Add(-7 * 24 * time.Hour)

	episodes := []memory.Episode{
		makeEpisode("ep-weekly", "monday standup notes", []string{"standup", "monday"}, weekAgo),
		makeEpisode("ep-recent", "something recent", []string{"recent"}, now.Add(-30*time.Minute)),
	}
	em := newTestEpisodicMemory(episodes, nil)
	mte := NewMemoryTriggerEngine(em, nil)

	triggers := mte.temporalScan()

	found := false
	for _, tr := range triggers {
		if tr.TriggerType == "temporal" && strings.Contains(tr.Relevance, "week") {
			found = true
		}
	}
	if !found {
		t.Error("expected temporal trigger for 'this time last week'")
	}
}

func TestTemporalScan_OneMonthAgo(t *testing.T) {
	now := time.Now()
	monthAgo := now.Add(-30 * 24 * time.Hour)

	episodes := []memory.Episode{
		makeEpisode("ep-monthly", "monthly review", []string{"review", "monthly"}, monthAgo),
	}
	em := newTestEpisodicMemory(episodes, nil)
	mte := NewMemoryTriggerEngine(em, nil)

	triggers := mte.temporalScan()

	found := false
	for _, tr := range triggers {
		if tr.TriggerType == "temporal" && strings.Contains(tr.Relevance, "month") {
			found = true
		}
	}
	if !found {
		t.Error("expected temporal trigger for 'this time last month'")
	}
}

func TestTemporalScan_NoMatchForArbitraryTime(t *testing.T) {
	now := time.Now()
	// 3 days ago — doesn't match any cyclical pattern
	episodes := []memory.Episode{
		makeEpisode("ep-3d", "random chat", []string{"chat"}, now.Add(-3*24*time.Hour)),
	}
	em := newTestEpisodicMemory(episodes, nil)
	mte := NewMemoryTriggerEngine(em, nil)

	triggers := mte.temporalScan()
	if len(triggers) != 0 {
		t.Errorf("expected no temporal triggers for 3-day-old episode, got %d", len(triggers))
	}
}

// -----------------------------------------------------------------------
// Cooldown logic
// -----------------------------------------------------------------------

func TestCooldown_BlocksRecentlySurfaced(t *testing.T) {
	mte := NewMemoryTriggerEngine(nil, nil)

	trigger := MemoryTrigger{
		Episode:     memory.Episode{ID: "ep-cooldown"},
		Similarity:  0.85,
		TriggerType: "semantic",
		Age:         48 * time.Hour,
	}

	// Should surface before recording
	if !mte.shouldSurface(trigger) {
		t.Fatal("trigger should surface before cooldown is set")
	}

	// Record it as surfaced
	mte.RecordSurfaced("ep-cooldown")

	// Should NOT surface now — within 24h cooldown
	if mte.shouldSurface(trigger) {
		t.Fatal("trigger should be blocked by cooldown")
	}
}

func TestCooldown_AllowsAfter24Hours(t *testing.T) {
	mte := NewMemoryTriggerEngine(nil, nil)

	// Manually set cooldown to 25 hours ago
	mte.mu.Lock()
	mte.cooldown["ep-old-cooldown"] = time.Now().Add(-25 * time.Hour)
	mte.mu.Unlock()

	trigger := MemoryTrigger{
		Episode:     memory.Episode{ID: "ep-old-cooldown"},
		Similarity:  0.85,
		TriggerType: "semantic",
		Age:         48 * time.Hour,
	}

	if !mte.shouldSurface(trigger) {
		t.Fatal("trigger should surface after 24h cooldown expires")
	}
}

func TestShouldSurface_RejectsLowSimilarity(t *testing.T) {
	mte := NewMemoryTriggerEngine(nil, nil)

	trigger := MemoryTrigger{
		Episode:     memory.Episode{ID: "ep-low"},
		Similarity:  0.40,
		TriggerType: "topic",
		Age:         48 * time.Hour,
	}

	if mte.shouldSurface(trigger) {
		t.Fatal("trigger with similarity below threshold should not surface")
	}
}

func TestShouldSurface_RejectsTooRecent(t *testing.T) {
	mte := NewMemoryTriggerEngine(nil, nil)

	trigger := MemoryTrigger{
		Episode:     memory.Episode{ID: "ep-new"},
		Similarity:  0.90,
		TriggerType: "semantic",
		Age:         30 * time.Minute, // less than 1h minAge
	}

	if mte.shouldSurface(trigger) {
		t.Fatal("trigger for very recent episode should not surface")
	}
}

// -----------------------------------------------------------------------
// FormatTrigger — output variety
// -----------------------------------------------------------------------

func TestFormatTrigger_SemanticPhrasing(t *testing.T) {
	mte := NewMemoryTriggerEngine(nil, nil)
	trigger := MemoryTrigger{
		Episode: memory.Episode{
			ID:    "ep-fmt",
			Input: "how does garbage collection work in Go",
			Tags:  []string{"garbage", "collection", "golang"},
		},
		Similarity:  0.85,
		TriggerType: "semantic",
		Age:         72 * time.Hour,
	}

	output := mte.FormatTrigger(trigger)
	if output == "" {
		t.Fatal("FormatTrigger returned empty string")
	}
	// Should contain a time reference and some topic content
	if !strings.Contains(output, "ago") && !strings.Contains(output, "days") {
		t.Errorf("expected time reference in output: %q", output)
	}
}

func TestFormatTrigger_TopicPhrasing(t *testing.T) {
	mte := NewMemoryTriggerEngine(nil, nil)
	trigger := MemoryTrigger{
		Episode: memory.Episode{
			ID:    "ep-topic",
			Input: "explain channels",
			Tags:  []string{"channels", "concurrency"},
		},
		Similarity:  0.70,
		TriggerType: "topic",
		Age:         24 * time.Hour,
	}

	output := mte.FormatTrigger(trigger)
	if output == "" {
		t.Fatal("FormatTrigger returned empty for topic trigger")
	}
}

func TestFormatTrigger_TemporalPhrasing(t *testing.T) {
	mte := NewMemoryTriggerEngine(nil, nil)
	trigger := MemoryTrigger{
		Episode: memory.Episode{
			ID:    "ep-temporal",
			Input: "weekly review",
			Tags:  []string{"review", "weekly"},
		},
		Similarity:  0.65,
		TriggerType: "temporal",
		Relevance:   "around this time last week",
		Age:         7 * 24 * time.Hour,
	}

	output := mte.FormatTrigger(trigger)
	if output == "" {
		t.Fatal("FormatTrigger returned empty for temporal trigger")
	}
}

func TestFormatTrigger_PatternPhrasing(t *testing.T) {
	mte := NewMemoryTriggerEngine(nil, nil)
	trigger := MemoryTrigger{
		Episode: memory.Episode{
			ID:    "ep-pattern",
			Input: "how do I debug memory leaks",
			Tags:  []string{"debug", "memory", "leaks"},
		},
		Similarity:  0.75,
		TriggerType: "pattern",
		Relevance:   "asked 3 times before",
		Age:         5 * 24 * time.Hour,
	}

	output := mte.FormatTrigger(trigger)
	if output == "" {
		t.Fatal("FormatTrigger returned empty for pattern trigger")
	}
}

func TestFormatTrigger_Variety(t *testing.T) {
	mte := NewMemoryTriggerEngine(nil, nil)
	trigger := MemoryTrigger{
		Episode: memory.Episode{
			ID:    "ep-variety",
			Input: "explain interfaces in Go",
			Tags:  []string{"interfaces", "golang"},
		},
		Similarity:  0.80,
		TriggerType: "semantic",
		Age:         48 * time.Hour,
	}

	// Generate many outputs and check we get at least 2 distinct phrasings
	seen := make(map[string]bool)
	for i := 0; i < 50; i++ {
		out := mte.FormatTrigger(trigger)
		seen[out] = true
	}
	if len(seen) < 2 {
		t.Errorf("expected variety in FormatTrigger output, got %d unique phrasings", len(seen))
	}
}

// -----------------------------------------------------------------------
// Pattern detection — repeated queries
// -----------------------------------------------------------------------

func TestPatternScan_DetectsRepeatedQueries(t *testing.T) {
	now := time.Now()
	episodes := []memory.Episode{
		makeEpisode("ep-r1", "how do I fix segfault errors", []string{"segfault", "errors", "debug"}, now.Add(-72*time.Hour)),
		makeEpisode("ep-r2", "how do I fix segfault errors", []string{"segfault", "errors", "debug"}, now.Add(-48*time.Hour)),
		makeEpisode("ep-r3", "how do I fix segfault errors", []string{"segfault", "errors", "debug"}, now.Add(-24*time.Hour)),
	}
	em := newTestEpisodicMemory(episodes, nil)
	mte := NewMemoryTriggerEngine(em, nil)

	triggers := mte.patternScan("how do I fix segfault errors")
	if len(triggers) == 0 {
		t.Fatal("expected pattern trigger for repeated query")
	}
	if triggers[0].TriggerType != "pattern" {
		t.Errorf("expected trigger type 'pattern', got %q", triggers[0].TriggerType)
	}
	if !strings.Contains(triggers[0].Relevance, "times") {
		t.Errorf("expected relevance to mention repetition count: %q", triggers[0].Relevance)
	}
}

func TestPatternScan_NoPatternForSingleOccurrence(t *testing.T) {
	now := time.Now()
	episodes := []memory.Episode{
		makeEpisode("ep-once", "explain monads in haskell", []string{"monads", "haskell"}, now.Add(-48*time.Hour)),
	}
	em := newTestEpisodicMemory(episodes, nil)
	mte := NewMemoryTriggerEngine(em, nil)

	triggers := mte.patternScan("explain monads in haskell")
	if len(triggers) != 0 {
		t.Errorf("expected no pattern trigger for single occurrence, got %d", len(triggers))
	}
}

// -----------------------------------------------------------------------
// Deduplication and ranking
// -----------------------------------------------------------------------

func TestDeduplicateTriggers(t *testing.T) {
	triggers := []MemoryTrigger{
		{Episode: memory.Episode{ID: "ep1"}, Similarity: 0.70, TriggerType: "topic"},
		{Episode: memory.Episode{ID: "ep1"}, Similarity: 0.85, TriggerType: "semantic"},
		{Episode: memory.Episode{ID: "ep2"}, Similarity: 0.60, TriggerType: "topic"},
	}

	deduped := deduplicateTriggers(triggers)
	if len(deduped) != 2 {
		t.Fatalf("expected 2 deduplicated triggers, got %d", len(deduped))
	}

	// ep1 should have the higher similarity (semantic)
	for _, tr := range deduped {
		if tr.Episode.ID == "ep1" && tr.Similarity != 0.85 {
			t.Errorf("expected ep1 to keep highest similarity 0.85, got %.2f", tr.Similarity)
		}
	}
}

func TestScan_RanksAndLimits(t *testing.T) {
	now := time.Now()
	episodes := []memory.Episode{
		makeEpisode("ep-a", "building a compiler in rust", []string{"compiler", "rust", "building"}, now.Add(-48*time.Hour)),
		makeEpisode("ep-b", "rust borrow checker explained", []string{"rust", "borrow", "checker"}, now.Add(-72*time.Hour)),
		makeEpisode("ep-c", "cooking pasta at home", []string{"cooking", "pasta", "home"}, now.Add(-96*time.Hour)),
	}
	em := newTestEpisodicMemory(episodes, nil)
	mte := NewMemoryTriggerEngine(em, nil)
	mte.maxPerTurn = 1

	triggers := mte.Scan("tell me about rust programming", []string{"rust", "programming"})
	if len(triggers) > 1 {
		t.Errorf("expected at most 1 trigger (maxPerTurn=1), got %d", len(triggers))
	}
}

func TestScan_EmptyInput(t *testing.T) {
	em := newTestEpisodicMemory(nil, nil)
	mte := NewMemoryTriggerEngine(em, nil)

	triggers := mte.Scan("", nil)
	if len(triggers) != 0 {
		t.Errorf("expected no triggers for empty input, got %d", len(triggers))
	}
}

func TestScan_NilEpisodicMemory(t *testing.T) {
	mte := NewMemoryTriggerEngine(nil, nil)
	triggers := mte.Scan("hello", []string{"greeting"})
	if len(triggers) != 0 {
		t.Errorf("expected no triggers with nil episodic memory, got %d", len(triggers))
	}
}

// -----------------------------------------------------------------------
// Helper function tests
// -----------------------------------------------------------------------

func TestWordOverlap(t *testing.T) {
	tests := []struct {
		a, b string
		min  float64
	}{
		{"hello world", "hello world", 1.0},
		{"hello world", "goodbye world", 0.4},
		{"a b c d", "a b c d", 1.0},
		{"apple banana cherry", "date elderberry fig", 0.0},
		{"", "something", 0.0},
	}
	for _, tc := range tests {
		got := wordOverlap(tc.a, tc.b)
		if got < tc.min {
			t.Errorf("wordOverlap(%q, %q) = %.2f, want >= %.2f", tc.a, tc.b, got, tc.min)
		}
	}
}

func TestFormatAge(t *testing.T) {
	tests := []struct {
		d    time.Duration
		want string
	}{
		{90 * time.Minute, "about an hour ago"},
		{5 * time.Hour, "about 5 hours ago"},
		{36 * time.Hour, "yesterday"},
		{4 * 24 * time.Hour, "4 days ago"},
		{10 * 24 * time.Hour, "last week"},
		{21 * 24 * time.Hour, "3 weeks ago"},
		{45 * 24 * time.Hour, "last month"},
		{90 * 24 * time.Hour, "3 months ago"},
	}
	for _, tc := range tests {
		got := formatAge(tc.d)
		if got != tc.want {
			t.Errorf("formatAge(%v) = %q, want %q", tc.d, got, tc.want)
		}
	}
}

func TestTruncate(t *testing.T) {
	short := "hello"
	if truncate(short, 80) != short {
		t.Error("truncate should not modify short strings")
	}

	long := "this is a much longer string that should be truncated at a word boundary"
	result := truncate(long, 30)
	if len(result) > 35 { // some slack for "..."
		t.Errorf("truncated string too long: %q (len %d)", result, len(result))
	}
	if !strings.HasSuffix(result, "...") {
		t.Errorf("expected truncated string to end with '...': %q", result)
	}
}
