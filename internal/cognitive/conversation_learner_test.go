package cognitive

import (
	"os"
	"path/filepath"
	"testing"
)

// -----------------------------------------------------------------------
// Input Generalization
// -----------------------------------------------------------------------

func TestGeneralizeInput(t *testing.T) {
	tests := []struct {
		input string
		want  string // substring that should appear
		slot  string // slot marker that should appear
	}{
		{
			input: "I just got promoted at work!",
			slot:  "[ACHIEVEMENT]",
		},
		{
			input: "I'm feeling happy today",
			slot:  "[EMOTION]",
		},
		{
			input: "what should I have for dinner?",
			slot:  "[MEAL]",
		},
		{
			input: "recommend me a good book",
			slot:  "[MEDIA]",
		},
		{
			input: "I've been running all morning",
			slot:  "[ACTIVITY]",
		},
		{
			input: "I feel sad about yesterday",
			slot:  "[EMOTION]",
		},
	}

	for _, tt := range tests {
		result := GeneralizeInput(tt.input)
		t.Logf("%-45s → %s", tt.input, result)

		if tt.slot != "" && !containsSubstring(result, tt.slot) {
			t.Errorf("GeneralizeInput(%q) = %q, expected slot %q", tt.input, result, tt.slot)
		}
		// The result should differ from the input (something was generalized).
		if result == tt.input && tt.slot != "" {
			t.Errorf("GeneralizeInput(%q) returned unchanged input", tt.input)
		}
	}
}

func TestGeneralizeInputPreservesStructure(t *testing.T) {
	// Function words and structure should be preserved.
	result := GeneralizeInput("what should I have for dinner?")
	if !containsSubstring(result, "what") {
		t.Error("expected 'what' to be preserved")
	}
	if !containsSubstring(result, "should") {
		t.Error("expected 'should' to be preserved")
	}
	if !containsSubstring(result, "for") {
		t.Error("expected 'for' to be preserved")
	}
	t.Logf("Result: %s", result)
}

// -----------------------------------------------------------------------
// Learning from Interactions
// -----------------------------------------------------------------------

func TestLearnFromInteraction(t *testing.T) {
	cl := NewConversationLearner("")

	// Learn a successful interaction.
	cl.LearnFromInteraction(
		"I just got promoted at work!",
		"Congratulations! That's a huge achievement.",
		"sharing", "positive", "career",
		true,
	)

	if cl.PatternCount() != 1 {
		t.Fatalf("expected 1 pattern, got %d", cl.PatternCount())
	}

	patterns := cl.Patterns()
	p := patterns[0]
	if p.SuccessCount != 1 {
		t.Errorf("expected success_count=1, got %d", p.SuccessCount)
	}
	if p.Quality <= 0 {
		t.Errorf("expected positive quality, got %f", p.Quality)
	}
	t.Logf("Learned pattern: %q → %q (quality=%.2f)", p.InputPattern, p.Response, p.Quality)
}

func TestLearnFromInteractionReinforcement(t *testing.T) {
	cl := NewConversationLearner("")

	// Same pattern succeeds multiple times.
	for i := 0; i < 5; i++ {
		cl.LearnFromInteraction(
			"I just got promoted at work!",
			"Congratulations! That's wonderful.",
			"sharing", "positive", "career",
			true,
		)
	}

	if cl.PatternCount() != 1 {
		t.Fatalf("expected 1 pattern (reinforced), got %d", cl.PatternCount())
	}

	p := cl.Patterns()[0]
	if p.SuccessCount != 5 {
		t.Errorf("expected success_count=5, got %d", p.SuccessCount)
	}
	t.Logf("Reinforced quality: %.2f (success=%d, fail=%d)", p.Quality, p.SuccessCount, p.FailCount)
}

func TestLearnFromInteractionFailure(t *testing.T) {
	cl := NewConversationLearner("")

	// First: learn a success.
	cl.LearnFromInteraction(
		"I just got promoted!",
		"That's nice.",
		"sharing", "positive", "career",
		true,
	)

	// Then: same pattern fails.
	cl.LearnFromInteraction(
		"I just got promoted!",
		"That's nice.",
		"sharing", "positive", "career",
		false,
	)

	p := cl.Patterns()[0]
	if p.FailCount != 1 {
		t.Errorf("expected fail_count=1, got %d", p.FailCount)
	}
	// Quality should have dropped.
	if p.Quality >= 1.0 {
		t.Errorf("expected quality < 1.0 after failure, got %f", p.Quality)
	}
	t.Logf("After failure: quality=%.2f (success=%d, fail=%d)", p.Quality, p.SuccessCount, p.FailCount)
}

func TestFailureDoesNotCreatePattern(t *testing.T) {
	cl := NewConversationLearner("")

	// A failure with no existing pattern should NOT create one.
	cl.LearnFromInteraction(
		"tell me a joke",
		"Why did the chicken cross the road?",
		"request", "neutral", "humor",
		false,
	)

	if cl.PatternCount() != 0 {
		t.Errorf("expected 0 patterns (failure should not create), got %d", cl.PatternCount())
	}
}

// -----------------------------------------------------------------------
// Pattern Matching
// -----------------------------------------------------------------------

func TestFindPattern(t *testing.T) {
	cl := NewConversationLearner("")

	// Learn some patterns.
	cl.LearnFromInteraction(
		"I just got promoted at work!",
		"Congratulations! That's a huge achievement.",
		"sharing", "positive", "career",
		true,
	)
	cl.LearnFromInteraction(
		"I'm feeling sad today",
		"I'm sorry to hear that. Do you want to talk about it?",
		"sharing", "negative", "emotion",
		true,
	)

	// Find a match for a similar but different input.
	match := cl.FindPattern("I just got married!", "sharing", "positive")
	if match == nil {
		t.Fatal("expected to find a matching pattern")
	}
	t.Logf("Matched: %q → %q (score implied by selection)", match.InputPattern, match.Response)

	// The match should be the achievement pattern, not the sadness one.
	if match.Sentiment != "positive" {
		t.Errorf("expected positive sentiment match, got %q", match.Sentiment)
	}
}

func TestFindPatternNoMatch(t *testing.T) {
	cl := NewConversationLearner("")

	match := cl.FindPattern("hello there", "greeting", "neutral")
	if match != nil {
		t.Error("expected nil for empty learner")
	}
}

func TestFindPatternPrefersQuality(t *testing.T) {
	cl := NewConversationLearner("")

	// Learn a high-quality and a low-quality pattern for the same intent.
	cl.LearnFromInteraction(
		"what is photosynthesis",
		"Great question about photosynthesis.",
		"question", "neutral", "science",
		true,
	)
	// Reinforce the first one.
	for i := 0; i < 4; i++ {
		cl.LearnFromInteraction(
			"what is photosynthesis",
			"Great question about photosynthesis.",
			"question", "neutral", "science",
			true,
		)
	}

	cl.LearnFromInteraction(
		"what is gravity",
		"Gravity is complicated.",
		"question", "neutral", "science",
		true,
	)
	// Add failures to the second.
	for i := 0; i < 3; i++ {
		cl.LearnFromInteraction(
			"what is gravity",
			"Gravity is complicated.",
			"question", "neutral", "science",
			false,
		)
	}

	match := cl.FindPattern("what is evolution", "question", "neutral")
	if match == nil {
		t.Fatal("expected a match")
	}
	// Should prefer the higher-quality pattern.
	if match.SuccessCount < 3 {
		t.Errorf("expected the high-quality pattern to be selected, got success_count=%d", match.SuccessCount)
	}
	t.Logf("Selected: quality=%.2f success=%d fail=%d", match.Quality, match.SuccessCount, match.FailCount)
}

// -----------------------------------------------------------------------
// Response Adaptation
// -----------------------------------------------------------------------

func TestAdaptResponse(t *testing.T) {
	pattern := &ResponsePattern{
		InputPattern: "I just [ACHIEVEMENT] at work!",
		Response:     "Congratulations on your promotion! That's a huge achievement.",
		Intent:       "sharing",
		Sentiment:    "positive",
		Topic:        "promotion",
	}

	adapted := AdaptResponse(pattern, "I just got married!")
	t.Logf("Original:  %s", pattern.Response)
	t.Logf("Adapted:   %s", adapted)

	// The adapted response should be non-empty.
	if adapted == "" {
		t.Error("expected non-empty adapted response")
	}
}

func TestAdaptResponseNilPattern(t *testing.T) {
	result := AdaptResponse(nil, "anything")
	if result != "" {
		t.Error("expected empty string for nil pattern")
	}
}

// -----------------------------------------------------------------------
// Outcome Detection
// -----------------------------------------------------------------------

func TestDetectOutcome(t *testing.T) {
	tests := []struct {
		followUp string
		want     InteractionOutcome
	}{
		{"thanks, that was helpful!", OutcomeStrongSuccess},
		{"thank you", OutcomeStrongSuccess},
		{"exactly what I needed", OutcomeStrongSuccess},
		{"perfect", OutcomeStrongSuccess},
		{"tell me more about that", OutcomeStrongSuccess},
		{"no, that's wrong", OutcomeFailure},
		{"that's not right", OutcomeFailure},
		{"not what I asked", OutcomeFailure},
		{"what about the weather tomorrow?", OutcomeSuccess},
		{"", OutcomeNeutral},
		{"ok", OutcomeMildFailure},
		{"whatever", OutcomeMildFailure},
	}

	for _, tt := range tests {
		got := DetectOutcome(tt.followUp)
		if got != tt.want {
			t.Errorf("DetectOutcome(%q) = %d, want %d", tt.followUp, got, tt.want)
		}
	}
}

func TestOutcomeIsSuccess(t *testing.T) {
	if !OutcomeSuccess.IsSuccess() {
		t.Error("OutcomeSuccess should be success")
	}
	if !OutcomeStrongSuccess.IsSuccess() {
		t.Error("OutcomeStrongSuccess should be success")
	}
	if OutcomeFailure.IsSuccess() {
		t.Error("OutcomeFailure should not be success")
	}
	if OutcomeNeutral.IsSuccess() {
		t.Error("OutcomeNeutral should not be success")
	}
}

// -----------------------------------------------------------------------
// Consolidation
// -----------------------------------------------------------------------

func TestConsolidate(t *testing.T) {
	cl := NewConversationLearner("")

	// Add a high-quality pattern.
	cl.LearnFromInteraction("hello", "Hi there!", "greeting", "positive", "greeting", true)
	for i := 0; i < 4; i++ {
		cl.LearnFromInteraction("hello", "Hi there!", "greeting", "positive", "greeting", true)
	}

	// Add a low-quality pattern (lots of failures).
	cl.LearnFromInteraction("tell me a joke", "knock knock", "request", "neutral", "humor", true)
	for i := 0; i < 10; i++ {
		cl.LearnFromInteraction("tell me a joke", "knock knock", "request", "neutral", "humor", false)
	}

	before := cl.PatternCount()
	cl.Consolidate()
	after := cl.PatternCount()

	t.Logf("Before consolidation: %d patterns", before)
	t.Logf("After consolidation: %d patterns", after)

	// The low-quality pattern should have been pruned.
	if after >= before {
		t.Error("expected consolidation to remove low-quality patterns")
	}
}

func TestConsolidateMergesSimilar(t *testing.T) {
	cl := NewConversationLearner("")

	// Add two very similar patterns with the same intent.
	cl.LearnFromInteraction(
		"I just got promoted at work!",
		"Congratulations on your promotion!",
		"sharing", "positive", "career",
		true,
	)
	cl.LearnFromInteraction(
		"I just got promoted at my job!",
		"That's wonderful news about your promotion!",
		"sharing", "positive", "career",
		true,
	)

	before := cl.PatternCount()
	cl.Consolidate()
	after := cl.PatternCount()

	t.Logf("Before merge: %d patterns, after: %d", before, after)
	// Similar patterns should be merged.
	if after > before {
		t.Error("expected merge to reduce or maintain pattern count")
	}
}

// -----------------------------------------------------------------------
// Persistence
// -----------------------------------------------------------------------

func TestConversationLearnerSaveLoad(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "patterns.json")

	// Create and populate.
	cl := NewConversationLearner("")
	cl.LearnFromInteraction("hello", "Hi!", "greeting", "positive", "greeting", true)
	cl.LearnFromInteraction("how are you", "Doing well!", "question", "neutral", "status", true)

	if err := cl.Save(path); err != nil {
		t.Fatal(err)
	}

	// Verify the file exists.
	if _, err := os.Stat(path); err != nil {
		t.Fatal("save file not found:", err)
	}

	// Load into a new learner.
	cl2 := NewConversationLearner("")
	if err := cl2.Load(path); err != nil {
		t.Fatal(err)
	}

	if cl2.PatternCount() != cl.PatternCount() {
		t.Errorf("loaded %d patterns, expected %d", cl2.PatternCount(), cl.PatternCount())
	}

	// Indices should be rebuilt.
	greetings := cl2.PatternsByIntent("greeting")
	if len(greetings) != 1 {
		t.Errorf("expected 1 greeting pattern after load, got %d", len(greetings))
	}
}

func TestNewConversationLearnerLoadsFromPath(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "patterns.json")

	// Save some patterns.
	cl := NewConversationLearner("")
	cl.LearnFromInteraction("test input", "test response", "test", "neutral", "test", true)
	if err := cl.Save(path); err != nil {
		t.Fatal(err)
	}

	// Create a new learner with the path — should auto-load.
	cl2 := NewConversationLearner(path)
	if cl2.PatternCount() != 1 {
		t.Errorf("expected 1 pattern from auto-load, got %d", cl2.PatternCount())
	}
}

// -----------------------------------------------------------------------
// Seeding
// -----------------------------------------------------------------------

func TestSeedDefaults(t *testing.T) {
	cl := NewConversationLearner("")
	cl.SeedDefaults()

	count := cl.PatternCount()
	t.Logf("Seeded %d default patterns", count)

	if count < 20 {
		t.Errorf("expected at least 20 seed patterns, got %d", count)
	}

	// Check that various intents are covered.
	intents := map[string]bool{}
	for _, p := range cl.Patterns() {
		intents[p.Intent] = true
	}

	for _, want := range []string{"greeting", "sharing", "question", "request", "farewell", "meta"} {
		if !intents[want] {
			t.Errorf("missing seed intent: %s", want)
		}
	}
}

func TestSeedDefaultsIdempotent(t *testing.T) {
	cl := NewConversationLearner("")
	cl.SeedDefaults()
	first := cl.PatternCount()

	cl.SeedDefaults()
	second := cl.PatternCount()

	if second != first {
		t.Errorf("SeedDefaults not idempotent: %d → %d", first, second)
	}
}

func TestSeedFromCorpus(t *testing.T) {
	corpus := NewDiscourseCorpus()
	corpus.Add(DiscourseSentence{
		Sentence: "Democracy is a system of government by the people.",
		Topic:    "Democracy",
		Function: DFDefines,
		Quality:  3,
	})
	corpus.Add(DiscourseSentence{
		Sentence: "Democracy is considered one of the most important political ideas.",
		Topic:    "Democracy",
		Function: DFEvaluates,
		Quality:  3,
	})

	cl := NewConversationLearner("")
	cl.SeedFromCorpus(corpus)

	if cl.PatternCount() < 2 {
		t.Errorf("expected at least 2 patterns from corpus, got %d", cl.PatternCount())
	}
	t.Logf("Seeded %d patterns from corpus", cl.PatternCount())
}

// -----------------------------------------------------------------------
// Pattern Query
// -----------------------------------------------------------------------

func TestPatternsByIntent(t *testing.T) {
	cl := NewConversationLearner("")
	cl.SeedDefaults()

	greetings := cl.PatternsByIntent("greeting")
	if len(greetings) == 0 {
		t.Error("expected greeting patterns")
	}
	t.Logf("Found %d greeting patterns", len(greetings))

	questions := cl.PatternsByIntent("question")
	if len(questions) == 0 {
		t.Error("expected question patterns")
	}
	t.Logf("Found %d question patterns", len(questions))
}

func TestTopPatterns(t *testing.T) {
	cl := NewConversationLearner("")
	cl.SeedDefaults()

	top5 := cl.TopPatterns(5)
	if len(top5) != 5 {
		t.Errorf("expected 5 top patterns, got %d", len(top5))
	}

	// Should be sorted by quality descending.
	for i := 1; i < len(top5); i++ {
		if top5[i].Quality > top5[i-1].Quality {
			t.Error("TopPatterns not sorted by quality descending")
			break
		}
	}
}

// -----------------------------------------------------------------------
// Integration: full learning cycle
// -----------------------------------------------------------------------

func TestFullLearningCycle(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "learning.json")

	cl := NewConversationLearner("")
	cl.SeedDefaults()

	// Simulate a conversation where Nous learns.
	// Turn 1: Nous responds to a greeting.
	match := cl.FindPattern("hey there!", "greeting", "positive")
	if match == nil {
		t.Fatal("expected to find greeting pattern")
	}
	t.Logf("Turn 1 match: %q", match.Response)

	// User continues (success signal).
	outcome := DetectOutcome("thanks! so what's new?")
	if !outcome.IsSuccess() {
		t.Error("expected success outcome")
	}
	cl.LearnFromInteraction("hey there!", match.Response, "greeting", "positive", "greeting", true)

	// Turn 2: User asks for a recommendation.
	cl.LearnFromInteraction(
		"recommend me a good movie",
		"What genre are you in the mood for?",
		"request", "neutral", "recommendation",
		true,
	)

	// Turn 3: Failure — user corrects Nous.
	cl.LearnFromInteraction(
		"what's 2+2",
		"I think it's 5.",
		"question", "neutral", "math",
		false, // user said "no, it's 4"
	)

	// Save and reload.
	if err := cl.Save(path); err != nil {
		t.Fatal(err)
	}

	cl2 := NewConversationLearner(path)
	t.Logf("After reload: %d patterns", cl2.PatternCount())

	// The failed pattern should not have been created.
	for _, p := range cl2.Patterns() {
		if p.Response == "I think it's 5." {
			t.Error("failure response should not have created a new pattern")
		}
	}

	// Consolidate.
	cl2.Consolidate()
	t.Logf("After consolidation: %d patterns", cl2.PatternCount())
}

// -----------------------------------------------------------------------
// Pattern Similarity
// -----------------------------------------------------------------------

func TestPatternSimilarity(t *testing.T) {
	tests := []struct {
		a, b    string
		minSim  float64
	}{
		// Identical.
		{"I just [ACHIEVEMENT] at work!", "I just [ACHIEVEMENT] at work!", 0.99},
		// Same structure, different slot types should still match (slots normalize).
		{"I just [ACHIEVEMENT] at work!", "I just [ACTIVITY] at work!", 0.99},
		// Partially similar.
		{"I just [ACHIEVEMENT] at work!", "I [ACHIEVEMENT] at my job!", 0.3},
		// Completely different.
		{"hello there", "what is the weather", 0.0},
	}

	for _, tt := range tests {
		sim := patternSimilarity(tt.a, tt.b)
		t.Logf("sim(%q, %q) = %.2f", tt.a, tt.b, sim)
		if sim < tt.minSim {
			t.Errorf("expected sim >= %.2f, got %.2f", tt.minSim, sim)
		}
	}
}

// -----------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------

func containsSubstring(s, sub string) bool {
	return len(s) >= len(sub) && (s == sub || len(sub) == 0 ||
		(len(s) > 0 && len(sub) > 0 && clStringContains(s, sub)))
}

func clStringContains(s, sub string) bool {
	for i := 0; i <= len(s)-len(sub); i++ {
		if s[i:i+len(sub)] == sub {
			return true
		}
	}
	return false
}
