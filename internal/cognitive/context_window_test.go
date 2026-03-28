package cognitive

import (
	"strings"
	"testing"
)

// -----------------------------------------------------------------------
// Context Window Tests
// -----------------------------------------------------------------------

func TestContextWindow_Record(t *testing.T) {
	cw := NewContextWindow(3)

	cw.Record("hello", "Hi there!", []string{"greeting"})
	cw.Record("tell me about Go", "Go is a compiled language.", []string{"Go", "programming"})

	if prev := cw.PreviousResponse(); prev != "Go is a compiled language." {
		t.Errorf("PreviousResponse = %q, want Go response", prev)
	}

	// Verify oldest turn is dropped when exceeding maxTurns.
	cw.Record("third", "Third response.", []string{"C"})
	cw.Record("fourth", "Fourth response.", []string{"D"})

	if prev := cw.PreviousResponse(); prev != "Fourth response." {
		t.Errorf("after overflow PreviousResponse = %q, want 'Fourth response.'", prev)
	}

	// The window should now contain turns 2, 3, 4 (oldest dropped).
	if cw.WasMentioned("greeting") {
		t.Error("oldest turn should have been evicted, but 'greeting' still found")
	}
}

func TestContextWindow_AvoidRepetition(t *testing.T) {
	cw := NewContextWindow(5)

	cw.Record("what is python?", "Python is a programming language. It was created by Guido van Rossum.", []string{"Python"})

	// Candidate contains one duplicate sentence and one new one.
	candidate := "Python is a programming language. It supports multiple paradigms."
	result := cw.AvoidRepetition(candidate)

	if strings.Contains(result, "Python is a programming language.") {
		t.Errorf("duplicate sentence was not removed: %s", result)
	}
	if !strings.Contains(result, "It supports multiple paradigms.") {
		t.Errorf("new sentence was incorrectly removed: %s", result)
	}

	// If ALL sentences are duplicates, candidate should be returned unchanged.
	allDup := "Python is a programming language."
	if got := cw.AvoidRepetition(allDup); got != allDup {
		t.Errorf("all-duplicate candidate should be returned unchanged, got: %s", got)
	}
}

func TestContextWindow_WasMentioned(t *testing.T) {
	cw := NewContextWindow(5)

	cw.Record("hi", "hello", []string{"Gravity", "Physics"})

	if !cw.WasMentioned("gravity") {
		t.Error("expected 'gravity' to be mentioned (case-insensitive)")
	}
	if !cw.WasMentioned("PHYSICS") {
		t.Error("expected 'PHYSICS' to be mentioned (case-insensitive)")
	}
	if cw.WasMentioned("chemistry") {
		t.Error("'chemistry' should not be mentioned")
	}
}

func TestContextWindow_BuildSummary(t *testing.T) {
	cw := NewContextWindow(5)

	// Empty window produces no summary.
	if s := cw.BuildContextSummary(); s != "" {
		t.Errorf("empty window should give empty summary, got: %s", s)
	}

	cw.Record("q1", "a1", []string{"Gravity"})
	s := cw.BuildContextSummary()
	if !strings.Contains(s, "Gravity") {
		t.Errorf("summary should mention Gravity: %s", s)
	}

	cw.Record("q2", "a2", []string{"Optics"})
	s = cw.BuildContextSummary()
	if !strings.Contains(s, "and") {
		t.Errorf("two-topic summary should use 'and': %s", s)
	}

	cw.Record("q3", "a3", []string{"Thermodynamics"})
	s = cw.BuildContextSummary()
	if !strings.Contains(s, ", and ") {
		t.Errorf("three-topic summary should use Oxford comma: %s", s)
	}
	if !strings.HasPrefix(s, "We've been discussing") {
		t.Errorf("summary should start with 'We've been discussing': %s", s)
	}
}

func TestContextWindow_RecentTopicsDedup(t *testing.T) {
	cw := NewContextWindow(5)

	cw.Record("q1", "a1", []string{"Go", "concurrency"})
	cw.Record("q2", "a2", []string{"Go", "channels"}) // "Go" appears again

	topics := cw.RecentTopics()

	// "Go" should appear only once.
	count := 0
	for _, tp := range topics {
		if strings.EqualFold(tp, "Go") {
			count++
		}
	}
	if count != 1 {
		t.Errorf("expected 'Go' once in deduped topics, got %d times in %v", count, topics)
	}

	// Most-recent-first: the first topic should be from the latest turn.
	if len(topics) == 0 || (!strings.EqualFold(topics[0], "Go") && !strings.EqualFold(topics[0], "channels")) {
		t.Errorf("most-recent topic should come first, got %v", topics)
	}
}

func TestContextWindow_DefaultMaxTurns(t *testing.T) {
	cw := NewContextWindow(0) // should default to 5
	for i := 0; i < 10; i++ {
		cw.Record("q", "a", nil)
	}
	cw.mu.RLock()
	n := len(cw.turns)
	cw.mu.RUnlock()
	if n != 5 {
		t.Errorf("default maxTurns should be 5, window has %d turns", n)
	}
}
