package cognitive

import (
	"strings"
	"testing"
)

// -----------------------------------------------------------------------
// NLG Engine Tests
// -----------------------------------------------------------------------

func TestNLGEngine_SingleFactRealization(t *testing.T) {
	eng := NewNLGEngine()

	facts := []edgeFact{
		{Subject: "Python", Relation: RelIsA, Object: "programming language"},
	}

	result := eng.Realize("Python", facts)
	if result == "" {
		t.Fatal("expected non-empty output for single IsA fact")
	}

	// Must contain the subject and the classification
	if !strings.Contains(result, "Python") {
		t.Errorf("expected result to contain 'Python', got: %s", result)
	}
	if !strings.Contains(result, "programming language") {
		t.Errorf("expected result to contain 'programming language', got: %s", result)
	}

	// Must NOT be a raw template like "Python is a %s"
	if strings.Contains(result, "%s") || strings.Contains(result, "%d") {
		t.Errorf("result contains format verbs — not real NLG: %s", result)
	}

	t.Logf("Single fact: %s", result)
}

func TestNLGEngine_MultiFactPropertyFusion(t *testing.T) {
	eng := NewNLGEngine()

	facts := []edgeFact{
		{Subject: "Python", Relation: RelHas, Object: "readable syntax"},
		{Subject: "Python", Relation: RelHas, Object: "dynamic typing"},
		{Subject: "Python", Relation: RelHas, Object: "extensive libraries"},
	}

	result := eng.Realize("Python", facts)
	if result == "" {
		t.Fatal("expected non-empty output")
	}

	// Properties must be combined with commas and "and", not as separate
	// "Python has X. Python has Y." sentences.
	if strings.Count(result, "Python") > 2 {
		t.Errorf("too many subject repetitions in fused output — properties should aggregate: %s", result)
	}

	// Must contain all three properties
	for _, prop := range []string{"readable syntax", "dynamic typing", "extensive libraries"} {
		if !strings.Contains(result, prop) {
			t.Errorf("missing property %q in output: %s", prop, result)
		}
	}

	// Oxford comma: should contain ", and"
	if !strings.Contains(result, ", and ") {
		t.Errorf("expected Oxford comma coordination in list of 3, got: %s", result)
	}

	t.Logf("Property fusion: %s", result)
}

func TestNLGEngine_OriginFusion(t *testing.T) {
	eng := NewNLGEngine()

	facts := []edgeFact{
		{Subject: "Python", Relation: RelCreatedBy, Object: "Guido van Rossum"},
		{Subject: "Python", Relation: RelFoundedIn, Object: "1991"},
	}

	result := eng.Realize("Python", facts)
	if result == "" {
		t.Fatal("expected non-empty output for origin facts")
	}

	// Creator and date should be in ONE sentence, not two separate ones.
	if !strings.Contains(result, "Guido van Rossum") {
		t.Errorf("expected creator mention, got: %s", result)
	}
	if !strings.Contains(result, "1991") {
		t.Errorf("expected date mention, got: %s", result)
	}

	// Should use participial form "Created by ... in ..." rather than
	// two separate "was created by" + "was founded in" template sentences.
	if strings.Contains(result, "Python was created") && strings.Contains(result, "Python was founded") {
		t.Errorf("origin facts not fused — still two separate template sentences: %s", result)
	}

	t.Logf("Origin fusion: %s", result)
}

func TestNLGEngine_Pronominalization(t *testing.T) {
	eng := NewNLGEngine()

	// Use a full set of facts that will produce text with multiple subject mentions
	facts := []edgeFact{
		{Subject: "Python", Relation: RelIsA, Object: "programming language"},
		{Subject: "Python", Relation: RelCreatedBy, Object: "Guido van Rossum"},
		{Subject: "Python", Relation: RelFoundedIn, Object: "1991"},
		{Subject: "Python", Relation: RelHas, Object: "readable syntax"},
		{Subject: "Python", Relation: RelHas, Object: "dynamic typing"},
		{Subject: "Python", Relation: RelUsedFor, Object: "data science"},
	}

	result := eng.Realize("Python", facts)
	if result == "" {
		t.Fatal("expected non-empty output")
	}

	// Count how many times "Python" appears (exact whole-word)
	pythonCount := nlgCountSubject(result, "Python")
	totalSentences := strings.Count(result, ".")

	// With 6+ facts generating multiple sentences, the subject should NOT
	// appear in every single sentence. Pronominalization must reduce it.
	if totalSentences >= 3 && pythonCount >= totalSentences {
		t.Errorf("pronominalization not working: 'Python' appears %d times in %d sentences: %s",
			pythonCount, totalSentences, result)
	}

	// Should contain at least one pronoun or definite description
	hasRef := strings.Contains(strings.ToLower(result), " it ") ||
		strings.Contains(strings.ToLower(result), " it'") ||
		strings.Contains(strings.ToLower(result), "the language") ||
		strings.Contains(strings.ToLower(result), "the programming")
	if totalSentences >= 3 && !hasRef {
		t.Errorf("expected pronoun or definite description in multi-sentence text: %s", result)
	}

	t.Logf("Pronominalized: %s", result)
}

func TestNLGEngine_GroupOrdering(t *testing.T) {
	eng := NewNLGEngine()

	// Feed facts in REVERSE order: usage, properties, origin, identity
	facts := []edgeFact{
		{Subject: "Go", Relation: RelUsedFor, Object: "systems programming"},
		{Subject: "Go", Relation: RelHas, Object: "goroutines"},
		{Subject: "Go", Relation: RelCreatedBy, Object: "Google"},
		{Subject: "Go", Relation: RelIsA, Object: "programming language"},
	}

	result := eng.Realize("Go", facts)
	if result == "" {
		t.Fatal("expected non-empty output")
	}

	// Identity ("is a programming language") must come BEFORE origin/properties/usage.
	idIdx := strings.Index(result, "programming language")
	propIdx := strings.Index(result, "goroutines")
	usageIdx := strings.Index(result, "systems programming")

	if idIdx < 0 {
		t.Fatalf("missing identity info in output: %s", result)
	}

	if propIdx >= 0 && propIdx < idIdx {
		t.Errorf("properties appeared before identity — group ordering broken: %s", result)
	}
	if usageIdx >= 0 && usageIdx < idIdx {
		t.Errorf("usage appeared before identity — group ordering broken: %s", result)
	}

	t.Logf("Ordered output: %s", result)
}

func TestNLGEngine_ComparisonGeneration(t *testing.T) {
	eng := NewNLGEngine()

	factsA := []edgeFact{
		{Subject: "Python", Relation: RelIsA, Object: "programming language"},
		{Subject: "Python", Relation: RelHas, Object: "dynamic typing"},
		{Subject: "Python", Relation: RelUsedFor, Object: "data science"},
		{Subject: "Python", Relation: RelUsedFor, Object: "scripting"},
	}

	factsB := []edgeFact{
		{Subject: "Go", Relation: RelIsA, Object: "programming language"},
		{Subject: "Go", Relation: RelHas, Object: "static typing"},
		{Subject: "Go", Relation: RelUsedFor, Object: "systems programming"},
		{Subject: "Go", Relation: RelUsedFor, Object: "data science"},
	}

	result := eng.RealizeComparison("Python", "Go", factsA, factsB)
	if result == "" {
		t.Fatal("expected non-empty comparison output")
	}

	// Must mention both subjects
	if !strings.Contains(result, "Python") {
		t.Errorf("comparison missing first subject 'Python': %s", result)
	}
	if !strings.Contains(result, "Go") {
		t.Errorf("comparison missing second subject 'Go': %s", result)
	}

	// Should mention shared property "data science"
	if !strings.Contains(result, "data science") {
		t.Errorf("comparison should mention shared property 'data science': %s", result)
	}

	// Should mention "Both" for shared properties
	if !strings.Contains(result, "Both") {
		t.Errorf("comparison should use 'Both' for shared properties: %s", result)
	}

	t.Logf("Comparison: %s", result)
}

func TestNLGEngine_ExplanationGeneration(t *testing.T) {
	eng := NewNLGEngine()

	facts := []edgeFact{
		{Subject: "machine learning", Relation: RelIsA, Object: "branch of artificial intelligence"},
		{Subject: "machine learning", Relation: RelDescribedAs, Object: "data-driven"},
		{Subject: "machine learning", Relation: RelUsedFor, Object: "pattern recognition"},
		{Subject: "machine learning", Relation: RelUsedFor, Object: "predictive analytics"},
		{Subject: "machine learning", Relation: RelDerivedFrom, Object: "statistical learning theory"},
		{Subject: "machine learning", Relation: RelContradicts, Object: "rule-based expert systems"},
	}

	result := eng.RealizeExplanation("machine learning", facts)
	if result == "" {
		t.Fatal("expected non-empty explanation output")
	}

	// Definition should appear
	if !strings.Contains(result, "artificial intelligence") {
		t.Errorf("explanation missing definition: %s", result)
	}

	// Should have both example uses
	if !strings.Contains(result, "pattern recognition") {
		t.Errorf("explanation missing usage example: %s", result)
	}

	// Should have the caveat about contradicting approach
	if !strings.Contains(result, "rule-based expert systems") {
		t.Errorf("explanation missing caveat: %s", result)
	}

	t.Logf("Explanation: %s", result)
}

func TestNLGEngine_EmptyFacts(t *testing.T) {
	eng := NewNLGEngine()

	// All three entry points should return empty string for no facts
	if result := eng.Realize("Python", nil); result != "" {
		t.Errorf("Realize with nil facts should return empty, got: %s", result)
	}
	if result := eng.Realize("Python", []edgeFact{}); result != "" {
		t.Errorf("Realize with empty facts should return empty, got: %s", result)
	}
	if result := eng.RealizeComparison("A", "B", nil, nil); result != "" {
		t.Errorf("RealizeComparison with nil facts should return empty, got: %s", result)
	}
	if result := eng.RealizeExplanation("X", nil); result != "" {
		t.Errorf("RealizeExplanation with nil facts should return empty, got: %s", result)
	}
}

func TestNLGEngine_SingleWordSubject(t *testing.T) {
	eng := NewNLGEngine()

	facts := []edgeFact{
		{Subject: "Go", Relation: RelIsA, Object: "programming language"},
		{Subject: "Go", Relation: RelHas, Object: "goroutines"},
		{Subject: "Go", Relation: RelUsedFor, Object: "cloud services"},
	}

	result := eng.Realize("Go", facts)
	if result == "" {
		t.Fatal("expected non-empty output for single-word subject")
	}

	// "Go" is a tricky subject because it's also a common English word.
	// The engine should still produce valid prose.
	if !strings.Contains(result, "programming language") {
		t.Errorf("missing classification for 'Go': %s", result)
	}

	t.Logf("Single-word subject: %s", result)
}

func TestNLGEngine_MultiWordSubject(t *testing.T) {
	eng := NewNLGEngine()

	facts := []edgeFact{
		{Subject: "machine learning", Relation: RelIsA, Object: "branch of artificial intelligence"},
		{Subject: "machine learning", Relation: RelUsedFor, Object: "prediction"},
	}

	result := eng.Realize("machine learning", facts)
	if result == "" {
		t.Fatal("expected non-empty output for multi-word subject")
	}

	// First letter should be capitalized
	if !strings.HasPrefix(result, "M") {
		t.Errorf("multi-word subject should be capitalized at start: %s", result)
	}

	if !strings.Contains(result, "artificial intelligence") {
		t.Errorf("missing classification: %s", result)
	}

	t.Logf("Multi-word subject: %s", result)
}

func TestNLGEngine_UsageFusion(t *testing.T) {
	eng := NewNLGEngine()

	facts := []edgeFact{
		{Subject: "Python", Relation: RelUsedFor, Object: "data science"},
		{Subject: "Python", Relation: RelUsedFor, Object: "web development"},
		{Subject: "Python", Relation: RelUsedFor, Object: "automation"},
	}

	result := eng.Realize("Python", facts)
	if result == "" {
		t.Fatal("expected non-empty output")
	}

	// All three uses should be in the same sentence, not repeated
	if strings.Count(result, ".") > 1 {
		t.Errorf("usage facts should fuse into one sentence, got multiple: %s", result)
	}

	// Must contain all three usages
	for _, use := range []string{"data science", "web development", "automation"} {
		if !strings.Contains(result, use) {
			t.Errorf("missing usage %q in output: %s", use, result)
		}
	}

	// Should use coordinated list with "and"
	if !strings.Contains(result, ", and ") {
		t.Errorf("expected Oxford comma in coordinated usage list: %s", result)
	}

	t.Logf("Usage fusion: %s", result)
}

func TestNLGEngine_FullPipelineProse(t *testing.T) {
	eng := NewNLGEngine()

	facts := []edgeFact{
		{Subject: "Python", Relation: RelIsA, Object: "programming language"},
		{Subject: "Python", Relation: RelDescribedAs, Object: "versatile"},
		{Subject: "Python", Relation: RelKnownFor, Object: "its readable syntax"},
		{Subject: "Python", Relation: RelCreatedBy, Object: "Guido van Rossum"},
		{Subject: "Python", Relation: RelFoundedIn, Object: "1991"},
		{Subject: "Python", Relation: RelHas, Object: "dynamic typing"},
		{Subject: "Python", Relation: RelHas, Object: "automatic memory management"},
		{Subject: "Python", Relation: RelHas, Object: "extensive standard library"},
		{Subject: "Python", Relation: RelUsedFor, Object: "data science"},
		{Subject: "Python", Relation: RelUsedFor, Object: "web development"},
		{Subject: "Python", Relation: RelUsedFor, Object: "automation"},
		{Subject: "Python", Relation: RelRelatedTo, Object: "Ruby"},
		{Subject: "Python", Relation: RelRelatedTo, Object: "JavaScript"},
	}

	result := eng.Realize("Python", facts)
	if result == "" {
		t.Fatal("expected non-empty output for full pipeline")
	}

	// Prose quality checks:
	// 1. Should be multiple sentences (we have 13 facts)
	sentCount := strings.Count(result, ".")
	if sentCount < 3 {
		t.Errorf("expected at least 3 sentences from 13 facts, got %d: %s", sentCount, result)
	}

	// 2. No format verbs
	if strings.Contains(result, "%s") || strings.Contains(result, "%d") {
		t.Errorf("result contains format verbs: %s", result)
	}

	// 3. Should contain all the key information
	for _, expected := range []string{
		"programming language",
		"Guido van Rossum",
		"1991",
		"data science",
	} {
		if !strings.Contains(result, expected) {
			t.Errorf("missing key info %q in full pipeline output: %s", expected, result)
		}
	}

	// 4. Should NOT repeat "Python" in every sentence
	pythonCount := nlgCountSubject(result, "Python")
	if pythonCount > sentCount {
		t.Errorf("'Python' appears %d times in %d sentences — insufficient pronominalization: %s",
			pythonCount, sentCount, result)
	}

	t.Logf("Full pipeline:\n%s", result)
}

func TestNLGEngine_LocationFusion(t *testing.T) {
	eng := NewNLGEngine()

	facts := []edgeFact{
		{Subject: "Google", Relation: RelIsA, Object: "technology company"},
		{Subject: "Google", Relation: RelLocatedIn, Object: "Mountain View, California"},
		{Subject: "Google", Relation: RelFoundedBy, Object: "Larry Page and Sergey Brin"},
	}

	result := eng.Realize("Google", facts)
	if result == "" {
		t.Fatal("expected non-empty output")
	}

	if !strings.Contains(result, "Mountain View") {
		t.Errorf("missing location in output: %s", result)
	}
	if !strings.Contains(result, "technology company") {
		t.Errorf("missing identity in output: %s", result)
	}

	t.Logf("Location: %s", result)
}

func TestNLGEngine_PersonPronominalization(t *testing.T) {
	eng := NewNLGEngine()

	facts := []edgeFact{
		{Subject: "Marie Curie", Relation: RelIsA, Object: "physicist"},
		{Subject: "Marie Curie", Relation: RelKnownFor, Object: "research on radioactivity"},
		{Subject: "Marie Curie", Relation: RelHas, Object: "two Nobel Prizes"},
		{Subject: "Marie Curie", Relation: RelLocatedIn, Object: "Paris"},
	}

	result := eng.Realize("Marie Curie", facts)
	if result == "" {
		t.Fatal("expected non-empty output")
	}

	// For a person, pronominalization should use "she" (Marie is female)
	lowerResult := strings.ToLower(result)
	hasShe := strings.Contains(lowerResult, " she ") || strings.Contains(lowerResult, "she ")
	hasPhysicist := strings.Contains(lowerResult, "the physicist")

	// With 4+ facts, we expect some form of pronoun/description
	curieCount := nlgCountSubject(result, "Marie Curie")
	if curieCount >= 4 {
		t.Errorf("too many full-name repetitions for a person: %s", result)
	}

	if !hasShe && !hasPhysicist && curieCount > 2 {
		t.Errorf("expected female pronoun 'she' or description 'the physicist' for Marie Curie: %s", result)
	}

	t.Logf("Person pronominalization: %s", result)
}

// -----------------------------------------------------------------------
// Helper function tests
// -----------------------------------------------------------------------

func TestJoinCoordinated(t *testing.T) {
	tests := []struct {
		items    []string
		expected string
	}{
		{nil, ""},
		{[]string{}, ""},
		{[]string{"alpha"}, "alpha"},
		{[]string{"alpha", "beta"}, "alpha and beta"},
		{[]string{"alpha", "beta", "gamma"}, "alpha, beta, and gamma"},
		{[]string{"a", "b", "c", "d"}, "a, b, c, and d"},
	}

	for _, tt := range tests {
		result := joinCoordinated(tt.items)
		if result != tt.expected {
			t.Errorf("joinCoordinated(%v) = %q, want %q", tt.items, result, tt.expected)
		}
	}
}

func TestIsProperNoun(t *testing.T) {
	if !isProperNoun("Python") {
		t.Error("expected 'Python' to be a proper noun")
	}
	if isProperNoun("language") {
		t.Error("expected 'language' to not be a proper noun")
	}
	if isProperNoun("") {
		t.Error("expected empty string to not be a proper noun")
	}
}

func TestNlgFindSubject(t *testing.T) {
	// Whole-word matching
	text := "Go is a language. Google was founded by Go creators."
	idx := nlgFindSubject(text, "Go")
	if idx != 0 {
		t.Errorf("expected first 'Go' at 0, got %d", idx)
	}

	// "Go" inside "Google" should NOT match
	remaining := text[idx+2:]
	idx2 := nlgFindSubject(remaining, "Go")
	// Should find the standalone "Go" before "creators", not inside "Google"
	if idx2 < 0 {
		t.Error("expected to find standalone 'Go' after 'Google'")
	} else if strings.HasPrefix(remaining[idx2:], "Google") {
		t.Errorf("matched inside 'Google' — word boundary check failed")
	}
}
