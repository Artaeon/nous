package cognitive

import (
	"strings"
	"testing"
)

// -----------------------------------------------------------------------
// Test helpers — build minimal test infrastructure
// -----------------------------------------------------------------------

// newTestCorpus creates a SentenceCorpus seeded with exemplars for testing.
func newTestCorpus() *SentenceCorpus {
	sc := NewSentenceCorpus()
	sc.Add(SentenceExemplar{
		Sentence: "Python is a high-level programming language.",
		Subject:  "Python",
		Object:   "high-level programming language",
		Relation: RelIsA,
	})
	sc.Add(SentenceExemplar{
		Sentence: "Ruby was created by Yukihiro Matsumoto.",
		Subject:  "Ruby",
		Object:   "Yukihiro Matsumoto",
		Relation: RelCreatedBy,
	})
	sc.Add(SentenceExemplar{
		Sentence: "Java is used for building enterprise applications.",
		Subject:  "Java",
		Object:   "building enterprise applications",
		Relation: RelUsedFor,
	})
	sc.Add(SentenceExemplar{
		Sentence: "Haskell has a strong type system.",
		Subject:  "Haskell",
		Object:   "a strong type system",
		Relation: RelHas,
	})
	sc.Add(SentenceExemplar{
		Sentence: "Perl was founded in 1987.",
		Subject:  "Perl",
		Object:   "1987",
		Relation: RelFoundedIn,
	})
	sc.Add(SentenceExemplar{
		Sentence: "Scala is similar to Kotlin.",
		Subject:  "Scala",
		Object:   "Kotlin",
		Relation: RelSimilarTo,
	})
	return sc
}

// newTestHybridGenerator creates a HybridGenerator with a seeded corpus
// and no GRU or fluency scorer (testing retrieval + structural paths).
func newTestHybridGenerator() *HybridGenerator {
	return NewHybridGenerator(newTestCorpus(), nil, nil)
}

// newTestHybridGeneratorWithFluency creates a generator with a fluency scorer.
func newTestHybridGeneratorWithFluency() *HybridGenerator {
	fs := NewFluencyScorer()
	return NewHybridGenerator(newTestCorpus(), nil, fs)
}

// -----------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------

func TestHybridGenerate_WithCorpus(t *testing.T) {
	hg := newTestHybridGenerator()

	facts := []edgeFact{
		{Subject: "Go", Relation: RelIsA, Object: "programming language"},
		{Subject: "Go", Relation: RelCreatedBy, Object: "Google"},
	}

	result := hg.Generate("Go", facts)

	if result == "" {
		t.Fatal("expected non-empty output from Generate with corpus")
	}

	// The output should contain the subject somewhere.
	if !strings.Contains(result, "Go") && !strings.Contains(result, "It") {
		t.Errorf("output should mention the subject: %q", result)
	}

	// Output should be multi-sentence (we provided two facts).
	if !strings.Contains(result, ".") {
		t.Errorf("output should contain at least one period: %q", result)
	}

	t.Logf("Generated with corpus: %s", result)
}

func TestHybridGenerate_WithoutCorpus(t *testing.T) {
	// No corpus, no GRU — pure structural fallback.
	hg := NewHybridGenerator(nil, nil, nil)

	facts := []edgeFact{
		{Subject: "Rust", Relation: RelIsA, Object: "programming language"},
		{Subject: "Rust", Relation: RelCreatedBy, Object: "Mozilla"},
		{Subject: "Rust", Relation: RelUsedFor, Object: "systems programming"},
	}

	result := hg.Generate("Rust", facts)

	if result == "" {
		t.Fatal("expected non-empty output from structural fallback")
	}

	// Structural fallback should produce grammatically correct sentences.
	if !strings.Contains(result, "programming language") {
		t.Errorf("structural output should contain the object: %q", result)
	}

	// Check that structural sentences use the correct patterns.
	if !strings.Contains(result, "is a") && !strings.Contains(result, "is an") {
		t.Errorf("IsA fact should produce 'is a/an' sentence: %q", result)
	}

	t.Logf("Generated without corpus: %s", result)
}

func TestHybridGenerate_Pronominalization(t *testing.T) {
	hg := NewHybridGenerator(nil, nil, nil)

	facts := []edgeFact{
		{Subject: "Python", Relation: RelIsA, Object: "programming language"},
		{Subject: "Python", Relation: RelCreatedBy, Object: "Guido van Rossum"},
		{Subject: "Python", Relation: RelUsedFor, Object: "data science"},
		{Subject: "Python", Relation: RelHas, Object: "dynamic typing"},
	}

	result := hg.Generate("Python", facts)

	if result == "" {
		t.Fatal("expected non-empty output")
	}

	// The full name "Python" should appear (first mention).
	if !strings.Contains(result, "Python") {
		t.Errorf("first mention should use full name: %q", result)
	}

	// A pronoun "It" should appear (subsequent mentions).
	if !strings.Contains(result, "It ") && !strings.Contains(result, "The programming language") {
		t.Errorf("subsequent mentions should be pronominalized: %q", result)
	}

	// Count how many times the full name appears — should be reduced.
	fullNameCount := strings.Count(result, "Python")
	if fullNameCount >= 4 {
		t.Errorf("expected fewer than 4 full name occurrences, got %d: %q", fullNameCount, result)
	}

	t.Logf("Pronominalized output: %s", result)
}

func TestHybridGenerate_MinimalConnectors(t *testing.T) {
	hg := newTestHybridGenerator()

	facts := []edgeFact{
		{Subject: "Go", Relation: RelIsA, Object: "programming language"},
		{Subject: "Go", Relation: RelCreatedBy, Object: "Google"},
		{Subject: "Go", Relation: RelHas, Object: "garbage collection"},
		{Subject: "Go", Relation: RelUsedFor, Object: "cloud services"},
		{Subject: "Go", Relation: RelRelatedTo, Object: "C"},
	}

	result := hg.Generate("Go", facts)

	// Connectors should be minimal — no verbose transitions.
	verboseConnectors := []string{
		"Furthermore,",
		"Additionally,",
		"Moreover,",
		"In addition,",
		"On the other hand,",
		"It is worth noting that",
	}

	for _, vc := range verboseConnectors {
		if strings.Contains(result, vc) {
			t.Errorf("output should not contain verbose connector %q: %q", vc, result)
		}
	}

	// The allowed connectors are short: "In practice," "Practically," "However," "That said,"
	t.Logf("Minimal connectors output: %s", result)
}

func TestBuildSimpleSentence(t *testing.T) {
	tests := []struct {
		subject  string
		rel      RelType
		object   string
		contains string
	}{
		{"Go", RelIsA, "programming language", "Go is a programming language."},
		{"Rust", RelCreatedBy, "Mozilla", "Rust was created by Mozilla."},
		{"Python", RelUsedFor, "data science", "Python is used for data science."},
		{"Java", RelHas, "garbage collection", "Java has garbage collection."},
		{"Vienna", RelLocatedIn, "Austria", "Vienna is located in Austria."},
		{"Go", RelFoundedBy, "Rob Pike", "Go was founded by Rob Pike."},
		{"Linux", RelFoundedIn, "1991", "Linux was founded in 1991."},
		{"Git", RelPartOf, "version control", "Git is part of version control."},
		{"AWS", RelOffers, "cloud computing", "AWS offers cloud computing."},
		{"Go", RelRelatedTo, "C", "Go is related to C."},
		{"Go", RelSimilarTo, "Rust", "Go is similar to Rust."},
		{"Stress", RelCauses, "anxiety", "Stress causes anxiety."},
		{"Peace", RelContradicts, "war", "Peace contradicts war."},
		{"Lunch", RelFollows, "breakfast", "Lunch follows breakfast."},
		{"User", RelPrefers, "simplicity", "User prefers simplicity."},
		{"User", RelDislikes, "complexity", "User dislikes complexity."},
		{"Go", RelDomain, "programming", "Go belongs to the domain of programming."},
		{"Einstein", RelKnownFor, "relativity", "Einstein is known for relativity."},
		{"Go", RelInfluencedBy, "C", "Go was influenced by C."},
		{"Go", RelDerivedFrom, "C", "Go is derived from C."},
		{"Love", RelOppositeOf, "hate", "Love is the opposite of hate."},
		{"Go", RelDescribedAs, "fast and simple", "Go is fast and simple."},
	}

	for _, tc := range tests {
		t.Run(string(tc.rel), func(t *testing.T) {
			result := buildSimpleSentence(tc.subject, tc.rel, tc.object)
			if result != tc.contains {
				t.Errorf("buildSimpleSentence(%q, %q, %q) = %q, want %q",
					tc.subject, tc.rel, tc.object, result, tc.contains)
			}
			// Verify grammatical structure: ends with period.
			if !strings.HasSuffix(result, ".") {
				t.Errorf("sentence should end with period: %q", result)
			}
			// Starts with capital letter.
			if result[0] < 'A' || result[0] > 'Z' {
				t.Errorf("sentence should start with capital: %q", result)
			}
		})
	}
}

func TestHybridExplanation(t *testing.T) {
	hg := NewHybridGenerator(nil, nil, nil)

	facts := []edgeFact{
		{Subject: "Stoicism", Relation: RelUsedFor, Object: "emotional resilience"},
		{Subject: "Stoicism", Relation: RelIsA, Object: "philosophy"},
		{Subject: "Stoicism", Relation: RelFoundedIn, Object: "300 BC"},
		{Subject: "Stoicism", Relation: RelContradicts, Object: "hedonism"},
	}

	result := hg.GenerateExplanation("Stoicism", facts)

	if result == "" {
		t.Fatal("expected non-empty explanation output")
	}

	// Definition should come first in the text.
	defIdx := strings.Index(result, "philosophy")
	usageIdx := strings.Index(result, "emotional resilience")
	caveatIdx := strings.Index(result, "hedonism")

	if defIdx < 0 {
		t.Errorf("explanation should contain the definition: %q", result)
	}
	if usageIdx < 0 {
		t.Errorf("explanation should contain usage: %q", result)
	}

	// Definition should appear before usage.
	if defIdx >= 0 && usageIdx >= 0 && defIdx > usageIdx {
		t.Errorf("definition (idx %d) should come before usage (idx %d): %q",
			defIdx, usageIdx, result)
	}

	// Caveats should come after usage.
	if usageIdx >= 0 && caveatIdx >= 0 && caveatIdx < usageIdx {
		t.Errorf("caveats (idx %d) should come after usage (idx %d): %q",
			caveatIdx, usageIdx, result)
	}

	t.Logf("Explanation output: %s", result)
}

func TestHybridComparison(t *testing.T) {
	hg := NewHybridGenerator(nil, nil, nil)

	factsA := []edgeFact{
		{Subject: "Go", Relation: RelIsA, Object: "programming language"},
		{Subject: "Go", Relation: RelCreatedBy, Object: "Google"},
		{Subject: "Go", Relation: RelUsedFor, Object: "cloud services"},
	}

	factsB := []edgeFact{
		{Subject: "Rust", Relation: RelIsA, Object: "programming language"},
		{Subject: "Rust", Relation: RelCreatedBy, Object: "Mozilla"},
		{Subject: "Rust", Relation: RelUsedFor, Object: "systems programming"},
	}

	result := hg.GenerateComparison("Go", "Rust", factsA, factsB)

	if result == "" {
		t.Fatal("expected non-empty comparison output")
	}

	// Both subjects should be mentioned.
	if !strings.Contains(result, "Go") {
		t.Errorf("comparison should mention subject A: %q", result)
	}
	if !strings.Contains(result, "Rust") {
		t.Errorf("comparison should mention subject B: %q", result)
	}

	// Shared property should be mentioned.
	if !strings.Contains(result, "programming language") {
		t.Errorf("comparison should mention shared property: %q", result)
	}

	// "Both" should appear for shared facts.
	if !strings.Contains(result, "Both") {
		t.Errorf("comparison should use 'Both' for shared facts: %q", result)
	}

	t.Logf("Comparison output: %s", result)
}

func TestHybridGenerate_EmptyFacts(t *testing.T) {
	hg := newTestHybridGenerator()

	result := hg.Generate("Go", nil)
	if result != "" {
		t.Errorf("empty facts should produce empty output, got: %q", result)
	}

	result = hg.Generate("Go", []edgeFact{})
	if result != "" {
		t.Errorf("zero-length facts should produce empty output, got: %q", result)
	}
}

func TestHybridComparison_EmptyFacts(t *testing.T) {
	hg := NewHybridGenerator(nil, nil, nil)

	result := hg.GenerateComparison("Go", "Rust", nil, nil)
	if result != "" {
		t.Errorf("empty comparison should produce empty output, got: %q", result)
	}
}

func TestPronominalize_NoCategory(t *testing.T) {
	// When no IsA fact provides a category, all replacements use "It".
	text := "Python is great. Python was created early. Python is fast."
	result := pronominalize(text, "Python", "")

	if !strings.Contains(result, "Python") {
		t.Errorf("first occurrence should keep full name: %q", result)
	}

	// Should contain "It" for subsequent mentions.
	if !strings.Contains(result, "It") {
		t.Errorf("should pronominalize with 'It': %q", result)
	}

	t.Logf("No-category pronominalization: %s", result)
}

func TestPronominalize_WithCategory(t *testing.T) {
	text := "Python is a language. Python was created by Guido. Python is popular. Python is fast."
	result := pronominalize(text, "Python", "language")

	// Should contain "The language" for the definite description.
	if !strings.Contains(result, "The language") {
		t.Errorf("should use definite description 'The language': %q", result)
	}

	t.Logf("With-category pronominalization: %s", result)
}

func TestDiscourseOrdering(t *testing.T) {
	facts := []edgeFact{
		{Subject: "Go", Relation: RelUsedFor, Object: "web servers"},
		{Subject: "Go", Relation: RelIsA, Object: "programming language"},
		{Subject: "Go", Relation: RelCreatedBy, Object: "Google"},
		{Subject: "Go", Relation: RelHas, Object: "goroutines"},
		{Subject: "Go", Relation: RelContradicts, Object: "complexity"},
	}

	groups := planDiscourseOrder(facts)

	if len(groups) == 0 {
		t.Fatal("expected discourse groups")
	}

	// First group should be definition.
	if groups[0].role != "definition" {
		t.Errorf("first group should be 'definition', got %q", groups[0].role)
	}

	// Last group should be caveat.
	last := groups[len(groups)-1]
	if last.role != "caveat" {
		t.Errorf("last group should be 'caveat', got %q", last.role)
	}

	// Check that all facts are accounted for.
	totalFacts := 0
	for _, g := range groups {
		totalFacts += len(g.facts)
	}
	if totalFacts != len(facts) {
		t.Errorf("expected %d total facts across groups, got %d", len(facts), totalFacts)
	}
}

func TestSplitIntoSentences(t *testing.T) {
	text := "Go is great. It was created by Google. The language is fast."
	sentences := hybridSplitSentences(text)

	if len(sentences) != 3 {
		t.Errorf("expected 3 sentences, got %d: %v", len(sentences), sentences)
	}

	if sentences[0] != "Go is great." {
		t.Errorf("first sentence: %q", sentences[0])
	}
}

func TestCleanupWhitespace(t *testing.T) {
	tests := []struct {
		input, want string
	}{
		{"hello  world", "hello world"},
		{"  spaces  everywhere  ", "spaces everywhere"},
		{"single space", "single space"},
		{"", ""},
	}

	for _, tc := range tests {
		got := cleanupWhitespace(tc.input)
		if got != tc.want {
			t.Errorf("cleanupWhitespace(%q) = %q, want %q", tc.input, got, tc.want)
		}
	}
}

func TestRetrieveOrGenerate_TierFallthrough(t *testing.T) {
	// With corpus that has no matching exemplar for this relation,
	// should fall through to structural.
	corpus := NewSentenceCorpus()
	// Add exemplar for a different relation than what we'll query.
	corpus.Add(SentenceExemplar{
		Sentence: "Python is a language.",
		Subject:  "Python",
		Object:   "language",
		Relation: RelIsA,
	})

	hg := NewHybridGenerator(corpus, nil, nil)

	// Query a relation not in the corpus.
	fact := edgeFact{Subject: "Go", Relation: RelCauses, Object: "joy"}
	result := hg.retrieveOrGenerate("Go", fact)

	if result == "" {
		t.Fatal("expected structural fallback")
	}

	if !strings.Contains(result, "Go") || !strings.Contains(result, "joy") {
		t.Errorf("structural fallback should contain subject and object: %q", result)
	}
}

func TestGenerateConnector(t *testing.T) {
	hg := newTestHybridGenerator()

	// First group never gets a connector.
	c := hg.generateConnector("", "definition")
	if c != "" {
		t.Errorf("first group should have no connector, got %q", c)
	}

	// Definition -> property should be empty.
	c = hg.generateConnector("definition", "property")
	if c != "" {
		t.Errorf("definition->property should be empty, got %q", c)
	}

	// Anything -> caveat should produce a short connector.
	c = hg.generateConnector("property", "caveat")
	if c == "" {
		t.Log("caveat connector was empty (acceptable)")
	} else if len(c) > 20 {
		t.Errorf("connector should be short (<=20 chars), got %q (%d chars)", c, len(c))
	}
}

// -----------------------------------------------------------------------
// Sentence fusion tests
// -----------------------------------------------------------------------

func TestFuseSameVerbSentences_Has(t *testing.T) {
	input := "It has wave-particle duality. It has superposition. It has entanglement."
	want := "It has wave-particle duality, superposition, and entanglement."
	got := fuseSameVerbSentences(input)
	if got != want {
		t.Errorf("fuseSameVerbSentences:\n  got:  %q\n  want: %q", got, want)
	}
}

func TestFuseSameVerbSentences_RelatedTo(t *testing.T) {
	input := "It is related to atomic structure. It is related to modern technologies."
	want := "It is related to atomic structure and modern technologies."
	got := fuseSameVerbSentences(input)
	if got != want {
		t.Errorf("fuseSameVerbSentences:\n  got:  %q\n  want: %q", got, want)
	}
}

func TestFuseSameVerbSentences_NoFusionForDifferentVerbs(t *testing.T) {
	input := "Python is a programming language. Python was created by Guido van Rossum."
	got := fuseSameVerbSentences(input)
	// No consecutive same-prefix sentences — should be unchanged.
	if got != input {
		t.Errorf("should not fuse different verbs:\n  got:  %q\n  want: %q", got, input)
	}
}

func TestFuseSameVerbSentences_MixedGroups(t *testing.T) {
	input := "Go is a language. It has concurrency. It has garbage collection. It was created by Google."
	got := fuseSameVerbSentences(input)
	// "It has" group fuses; others stay.
	if !strings.Contains(got, "concurrency") || !strings.Contains(got, "garbage collection") {
		t.Errorf("fused output should contain both objects: %q", got)
	}
	if !strings.Contains(got, "and garbage collection") {
		t.Errorf("fused output should use 'and' for two items: %q", got)
	}
	if !strings.Contains(got, "Google") {
		t.Errorf("non-fused sentence should be preserved: %q", got)
	}
}

func TestFuseSameVerbSentences_SingleSentence(t *testing.T) {
	input := "It has concurrency."
	got := fuseSameVerbSentences(input)
	if got != input {
		t.Errorf("single sentence should be unchanged: got %q", got)
	}
}

func TestFuseSameVerbSentences_Empty(t *testing.T) {
	got := fuseSameVerbSentences("")
	if got != "" {
		t.Errorf("empty input should return empty: got %q", got)
	}
}
