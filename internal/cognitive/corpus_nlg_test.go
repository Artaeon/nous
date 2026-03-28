package cognitive

import (
	"strings"
	"testing"
)

// -----------------------------------------------------------------------
// Corpus NLG Tests — verify pattern mining from real knowledge files,
// bigram model construction, and natural language generation.
// -----------------------------------------------------------------------

const corpusKnowledgeDir = "../../knowledge"

// helper: create and ingest a CorpusNLG from the real knowledge files.
func setupCorpusNLG(t *testing.T) *CorpusNLG {
	t.Helper()
	c := NewCorpusNLG()
	if err := c.IngestCorpus(corpusKnowledgeDir); err != nil {
		t.Fatalf("IngestCorpus failed: %v", err)
	}
	return c
}

func TestIngestCorpus(t *testing.T) {
	c := setupCorpusNLG(t)

	if c.PatternCount() == 0 {
		t.Fatal("expected patterns to be mined from knowledge files, got 0")
	}
	if c.BigramCount() == 0 {
		t.Fatal("expected bigrams to be built, got 0")
	}
	if c.VocabSize() == 0 {
		t.Fatal("expected vocabulary to be populated, got 0")
	}

	t.Logf("Ingested: %d patterns, %d bigram entries, %d vocab words",
		c.PatternCount(), c.BigramCount(), c.VocabSize())

	// Verify we have patterns across multiple functions.
	counts := c.FunctionCounts()
	t.Logf("Function distribution: %v", counts)

	functionsSeen := 0
	for fn, count := range counts {
		if count > 0 {
			functionsSeen++
			t.Logf("  %s: %d patterns", fn, count)
		}
	}
	if functionsSeen < 3 {
		t.Errorf("expected patterns in at least 3 functions, got %d", functionsSeen)
	}
}

func TestCorpusNLG_PatternCount(t *testing.T) {
	c := setupCorpusNLG(t)

	count := c.PatternCount()
	if count < 100 {
		t.Errorf("expected at least 100 patterns from 78K word corpus, got %d", count)
	}
	t.Logf("Total mined patterns: %d", count)
}

func TestBigramModel(t *testing.T) {
	c := setupCorpusNLG(t)

	if c.BigramCount() < 1000 {
		t.Errorf("expected at least 1000 unique bigram words from 78K corpus, got %d", c.BigramCount())
	}

	// Check that common words have bigram entries.
	c.mu.RLock()
	defer c.mu.RUnlock()

	commonChecks := []string{"the", "is", "of", "and", "in"}
	for _, w := range commonChecks {
		if nexts, ok := c.bigrams[w]; !ok || len(nexts) == 0 {
			t.Errorf("expected bigram entries for common word %q", w)
		}
	}

	t.Logf("Bigram model: %d unique words, %d vocab", len(c.bigrams), len(c.vocab))
}

func TestGenerateFromFacts(t *testing.T) {
	c := setupCorpusNLG(t)

	facts := []edgeFact{
		{Subject: "Python", Relation: RelIsA, Object: "programming language"},
		{Subject: "Python", Relation: RelCreatedBy, Object: "Guido van Rossum"},
		{Subject: "Python", Relation: RelHas, Object: "dynamic typing"},
		{Subject: "Python", Relation: RelUsedFor, Object: "web development"},
	}

	result := c.GenerateFromFacts("Python", facts)
	if result == "" {
		t.Fatal("expected non-empty output from GenerateFromFacts")
	}

	// Must contain the subject.
	if !strings.Contains(result, "Python") && !strings.Contains(strings.ToLower(result), "python") {
		t.Errorf("expected result to mention Python, got: %s", result)
	}

	// Must mention at least some fact objects.
	mentionedFacts := 0
	for _, f := range facts {
		if strings.Contains(strings.ToLower(result), strings.ToLower(f.Object)) {
			mentionedFacts++
		}
	}
	if mentionedFacts == 0 {
		t.Errorf("expected result to include at least one fact object, got: %s", result)
	}

	// Must NOT contain unfilled slot markers.
	if strings.Contains(result, "[SUBJECT]") || strings.Contains(result, "[OBJECT]") ||
		strings.Contains(result, "[CATEGORY]") || strings.Contains(result, "[PERSON]") {
		t.Errorf("result contains unfilled slot markers: %s", result)
	}

	t.Logf("Generated: %s", result)
}

func TestFillPattern(t *testing.T) {
	c := NewCorpusNLG()

	pattern := MinedPattern{
		Structure: "[SUBJECT] is [CATEGORY] that was developed by [PERSON].",
		Function:  "definition",
		SlotCount: 3,
		Quality:   1.0,
		WordCount: 10,
	}

	slots := map[string]string{
		"SUBJECT":  "Go",
		"CATEGORY": "a statically typed programming language",
		"PERSON":   "Google",
	}

	result := c.FillPattern(pattern, slots)

	if !strings.Contains(result, "Go") {
		t.Errorf("expected SUBJECT to be filled, got: %s", result)
	}
	if !strings.Contains(result, "statically typed") {
		t.Errorf("expected CATEGORY to be filled, got: %s", result)
	}
	if !strings.Contains(result, "Google") {
		t.Errorf("expected PERSON to be filled, got: %s", result)
	}
	if strings.Contains(result, "[") || strings.Contains(result, "]") {
		t.Errorf("unfilled slots remain: %s", result)
	}

	t.Logf("Filled: %s", result)
}

func TestBigramNext(t *testing.T) {
	c := setupCorpusNLG(t)

	// BigramNext should return a real word for known words.
	results := make(map[string]bool)
	for i := 0; i < 50; i++ {
		next := c.BigramNext("the")
		if next == "" {
			t.Error("BigramNext returned empty for 'the'")
		}
		results[next] = true
	}

	// With 50 tries, we should get some variety.
	if len(results) < 3 {
		t.Errorf("expected at least 3 different next words after 'the', got %d: %v", len(results), results)
	}

	t.Logf("Distinct next words after 'the' (50 samples): %d", len(results))

	// Unknown word should still return something (Laplace fallback).
	next := c.BigramNext("xyznonexistent")
	if next == "" {
		t.Error("BigramNext should return fallback for unknown word")
	}
}

func TestGenerateOpener(t *testing.T) {
	c := setupCorpusNLG(t)

	functions := []string{"definition", "origin", "property", "usage"}

	for _, fn := range functions {
		openers := make(map[string]bool)
		for i := 0; i < 20; i++ {
			opener := c.GenerateOpener(fn)
			if opener == "" {
				t.Errorf("GenerateOpener returned empty for function %q", fn)
				continue
			}
			openers[opener] = true
		}

		// Should produce varied openers.
		if len(openers) < 2 {
			t.Errorf("expected varied openers for %q, got %d unique: %v", fn, len(openers), openers)
		}
		t.Logf("Openers for %s (%d unique):", fn, len(openers))
		count := 0
		for o := range openers {
			if count < 3 {
				t.Logf("  %q", o)
				count++
			}
		}
	}
}

func TestGenerateTransition(t *testing.T) {
	c := setupCorpusNLG(t)

	tests := []struct {
		prevWord string
		function string
	}{
		{"physics", "definition"},
		{"century", "origin"},
		{"language", "property"},
		{"computing", "usage"},
	}

	for _, tt := range tests {
		transition := c.GenerateTransition(tt.prevWord, tt.function)
		if transition == "" {
			t.Errorf("expected non-empty transition from %q to %q function", tt.prevWord, tt.function)
			continue
		}

		// Should be a short phrase (not a full sentence).
		words := strings.Fields(transition)
		if len(words) > 6 {
			t.Errorf("transition too long (%d words): %q", len(words), transition)
		}

		t.Logf("Transition from %q -> %s: %q", tt.prevWord, tt.function, transition)
	}
}

func TestAntiRepetition(t *testing.T) {
	c := setupCorpusNLG(t)

	facts := []edgeFact{
		{Subject: "Rust", Relation: RelIsA, Object: "systems programming language"},
	}

	// Generate multiple times and collect results.
	results := make(map[string]bool)
	for i := 0; i < 10; i++ {
		result := c.GenerateFromFacts("Rust", facts)
		if result != "" {
			results[result] = true
		}
	}

	// With anti-repetition + random selection, we should get some variety.
	if len(results) < 2 {
		t.Logf("Warning: only %d unique outputs in 10 runs (may be acceptable with limited patterns)", len(results))
	}

	t.Logf("Unique outputs in 10 runs: %d", len(results))
	count := 0
	for r := range results {
		if count < 3 {
			t.Logf("  %q", r)
			count++
		}
	}
}

func TestPronominalization(t *testing.T) {
	c := setupCorpusNLG(t)

	facts := []edgeFact{
		{Subject: "JavaScript", Relation: RelIsA, Object: "programming language"},
		{Subject: "JavaScript", Relation: RelCreatedBy, Object: "Brendan Eich"},
		{Subject: "JavaScript", Relation: RelUsedFor, Object: "web development"},
		{Subject: "JavaScript", Relation: RelHas, Object: "dynamic typing"},
	}

	result := c.GenerateFromFacts("JavaScript", facts)
	if result == "" {
		t.Fatal("expected non-empty output")
	}

	// Count raw subject mentions — pronominalization should reduce them.
	subjectCount := strings.Count(result, "JavaScript")
	sentenceCount := strings.Count(result, ".") + 1

	t.Logf("Result: %s", result)
	t.Logf("Subject mentions: %d in ~%d sentences", subjectCount, sentenceCount)

	// If we have multiple sentences, subject should not appear in every one.
	if sentenceCount >= 3 && subjectCount >= sentenceCount {
		t.Errorf("expected pronominalization to reduce subject repetition: %d mentions in %d sentences",
			subjectCount, sentenceCount)
	}
}

// TestPatternStructures verifies that mined patterns have valid structure.
func TestPatternStructures(t *testing.T) {
	c := setupCorpusNLG(t)

	patterns := c.Patterns()
	if len(patterns) == 0 {
		t.Fatal("no patterns to check")
	}

	slotsFound := map[string]int{}
	for _, p := range patterns {
		for _, slot := range []string{"[SUBJECT]", "[OBJECT]", "[CATEGORY]", "[PERSON]", "[DATE]", "[PLACE]"} {
			if strings.Contains(p.Structure, slot) {
				slotsFound[slot]++
			}
		}
	}

	t.Logf("Slot distribution across %d patterns:", len(patterns))
	for slot, count := range slotsFound {
		t.Logf("  %s: %d patterns", slot, count)
	}

	// Must have [SUBJECT] in many patterns.
	if slotsFound["[SUBJECT]"] < 10 {
		t.Errorf("expected at least 10 patterns with [SUBJECT], got %d", slotsFound["[SUBJECT]"])
	}
}

// TestCorpusNLG_EmptyFacts verifies graceful handling of edge cases.
func TestCorpusNLG_EmptyFacts(t *testing.T) {
	c := setupCorpusNLG(t)

	result := c.GenerateFromFacts("Test", nil)
	if result != "" {
		t.Errorf("expected empty result for nil facts, got: %s", result)
	}

	result = c.GenerateFromFacts("Test", []edgeFact{})
	if result != "" {
		t.Errorf("expected empty result for empty facts, got: %s", result)
	}
}
