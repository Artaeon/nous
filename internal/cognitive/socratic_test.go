package cognitive

import (
	"strings"
	"testing"
)

// ---------------------------------------------------------------------------
// TestDetectMode — verify each mode triggers on appropriate inputs
// ---------------------------------------------------------------------------

func TestDetectModeDecision(t *testing.T) {
	se := NewSocraticEngine()
	state := NewConversationState()

	cases := []struct {
		query string
		want  SocraticMode
	}{
		{"should I learn Go or Rust?", SocraticDecide},
		{"which is better, React or Vue?", SocraticDecide},
		{"I'm torn between staying and leaving", SocraticDecide},
		{"help me decide on a career path", SocraticDecide},
		{"I can't choose between these two offers", SocraticDecide},
		{"weighing my options for the move", SocraticDecide},
	}

	for _, tc := range cases {
		got := se.DetectMode(tc.query, state)
		if got != tc.want {
			t.Errorf("DetectMode(%q) = %v, want %v", tc.query, got, tc.want)
		}
	}
}

func TestDetectModeCoaching(t *testing.T) {
	se := NewSocraticEngine()
	state := NewConversationState()

	cases := []struct {
		query string
		want  SocraticMode
	}{
		{"help me think about my leadership style", SocraticCoach},
		{"I'm struggling with time management", SocraticCoach},
		{"how do I get better at public speaking?", SocraticCoach},
		{"I need to figure out my priorities", SocraticCoach},
		{"I'm stuck on this design problem", SocraticCoach},
		{"help me work through this conflict", SocraticCoach},
	}

	for _, tc := range cases {
		got := se.DetectMode(tc.query, state)
		if got != tc.want {
			t.Errorf("DetectMode(%q) = %v, want %v", tc.query, got, tc.want)
		}
	}
}

func TestDetectModeExplore(t *testing.T) {
	se := NewSocraticEngine()
	state := NewConversationState()

	cases := []struct {
		query string
		want  SocraticMode
	}{
		{"I'm curious about how memory works", SocraticExplore},
		{"what do you think about remote work?", SocraticExplore},
		{"tell me your perspective on minimalism", SocraticExplore},
		{"I want to explore different approaches to meditation", SocraticExplore},
	}

	for _, tc := range cases {
		got := se.DetectMode(tc.query, state)
		if got != tc.want {
			t.Errorf("DetectMode(%q) = %v, want %v", tc.query, got, tc.want)
		}
	}
}

func TestDetectModeChallenge(t *testing.T) {
	se := NewSocraticEngine()

	// Strong unsupported claims should trigger challenge
	claims := []string{
		"obviously Python is the best language",
		"everyone knows that college is a waste of time",
		"the only way to succeed is to work 80 hours a week",
	}
	state := NewConversationState()
	for _, q := range claims {
		got := se.DetectMode(q, state)
		if got != SocraticChallenge {
			t.Errorf("DetectMode(%q) = %v, want SocraticChallenge", q, got)
		}
	}

	// Loop detection: same topic 3+ turns
	loopState := NewConversationState()
	loopState.ActiveTopic = "productivity"
	loopState.TurnCount = 5
	loopState.TopicStack = []string{"productivity", "productivity", "productivity", "productivity"}

	got := se.DetectMode("tell me more about productivity", loopState)
	if got != SocraticChallenge {
		t.Errorf("DetectMode with loop state = %v, want SocraticChallenge", got)
	}
}

func TestDetectModeDeepen(t *testing.T) {
	se := NewSocraticEngine()
	state := NewConversationState()

	cases := []struct {
		query string
		want  SocraticMode
	}{
		{"what about life?", SocraticDeepen},
		{"what is success?", SocraticDeepen},
		{"tell me about happiness", SocraticDeepen},
		{"thoughts on love", SocraticDeepen},
	}

	for _, tc := range cases {
		got := se.DetectMode(tc.query, state)
		if got != tc.want {
			t.Errorf("DetectMode(%q) = %v, want %v", tc.query, got, tc.want)
		}
	}
}

// ---------------------------------------------------------------------------
// TestSocraticNoneForSimpleQueries — factual/command queries stay direct
// ---------------------------------------------------------------------------

func TestSocraticNoneForSimpleQueries(t *testing.T) {
	se := NewSocraticEngine()
	state := NewConversationState()

	directQueries := []string{
		"what is 2+2",
		"who is the president of France",
		"when was the Eiffel Tower built",
		"where is Tokyo",
		"how many planets are there",
		"define photosynthesis",
		"translate hello to Spanish",
		"set a timer for 5 minutes",
		"remind me to call mom",
		"search for Italian restaurants",
		"hi",
		"",
		"ok",
		"what is the capital of Japan",
		"calculate 15% of 200",
	}

	for _, q := range directQueries {
		got := se.DetectMode(q, state)
		if got != SocraticNone {
			t.Errorf("DetectMode(%q) = %v, want SocraticNone — simple queries should not be Socratic", q, got)
		}
	}
}

func TestDetectModeNoneForMediumFactual(t *testing.T) {
	se := NewSocraticEngine()
	state := NewConversationState()

	// These are questions but they are factual, not philosophical
	factual := []string{
		"what is the boiling point of water",
		"who wrote Romeo and Juliet",
		"how tall is Mount Everest",
		"when did World War 2 end",
		"where was Einstein born",
	}
	for _, q := range factual {
		got := se.DetectMode(q, state)
		if got != SocraticNone {
			t.Errorf("DetectMode(%q) = %v, want SocraticNone", q, got)
		}
	}
}

// ---------------------------------------------------------------------------
// TestGenerate — verify question generation for each mode
// ---------------------------------------------------------------------------

func TestGenerateExplore(t *testing.T) {
	se := NewSocraticEngine()
	state := NewConversationState()
	state.ActiveTopic = "remote work"

	resp := se.Generate("I'm curious about remote work", SocraticExplore, state)

	if resp.Mode != SocraticExplore {
		t.Errorf("expected mode SocraticExplore, got %v", resp.Mode)
	}
	if len(resp.Questions) < 2 || len(resp.Questions) > 3 {
		t.Errorf("expected 2-3 questions, got %d", len(resp.Questions))
	}
	if resp.Framing == "" {
		t.Error("expected non-empty framing for explore mode")
	}
	if resp.Rationale == "" {
		t.Error("expected non-empty rationale")
	}
}

func TestGenerateDecide(t *testing.T) {
	se := NewSocraticEngine()
	state := NewConversationState()
	state.ActiveTopic = "Go vs Rust"
	state.MentionedEntities = map[string]string{"topic": "Go vs Rust"}

	resp := se.Generate("should I learn Go or Rust?", SocraticDecide, state)

	if resp.Mode != SocraticDecide {
		t.Errorf("expected mode SocraticDecide, got %v", resp.Mode)
	}
	if len(resp.Questions) < 2 {
		t.Errorf("expected at least 2 questions, got %d", len(resp.Questions))
	}
}

func TestGenerateCoach(t *testing.T) {
	se := NewSocraticEngine()
	state := NewConversationState()
	state.ActiveTopic = "leadership"

	resp := se.Generate("help me think about my leadership style", SocraticCoach, state)

	if resp.Mode != SocraticCoach {
		t.Errorf("expected mode SocraticCoach, got %v", resp.Mode)
	}
	if len(resp.Questions) < 2 {
		t.Errorf("expected at least 2 questions, got %d", len(resp.Questions))
	}
	if resp.Framing == "" {
		t.Error("expected non-empty framing for coach mode")
	}
}

func TestGenerateChallenge(t *testing.T) {
	se := NewSocraticEngine()
	state := NewConversationState()
	state.ActiveTopic = "Python"

	resp := se.Generate("obviously Python is the best language", SocraticChallenge, state)

	if resp.Mode != SocraticChallenge {
		t.Errorf("expected mode SocraticChallenge, got %v", resp.Mode)
	}
	if len(resp.Questions) < 2 {
		t.Errorf("expected at least 2 questions, got %d", len(resp.Questions))
	}
	// Challenge questions should include assumption-probing
	hasAssumption := false
	for _, q := range resp.Questions {
		if q.Category == "probe_assumptions" {
			hasAssumption = true
		}
	}
	if !hasAssumption {
		t.Error("challenge mode should include at least one probe_assumptions question")
	}
}

func TestGenerateDeepen(t *testing.T) {
	se := NewSocraticEngine()
	state := NewConversationState()
	state.ActiveTopic = "success"

	resp := se.Generate("what is success?", SocraticDeepen, state)

	if resp.Mode != SocraticDeepen {
		t.Errorf("expected mode SocraticDeepen, got %v", resp.Mode)
	}
	if len(resp.Questions) < 2 {
		t.Errorf("expected at least 2 questions, got %d", len(resp.Questions))
	}
}

func TestGenerateNoneReturnsEmpty(t *testing.T) {
	se := NewSocraticEngine()
	state := NewConversationState()

	resp := se.Generate("what is 2+2", SocraticNone, state)

	if resp.Mode != SocraticNone {
		t.Errorf("expected SocraticNone, got %v", resp.Mode)
	}
	if len(resp.Questions) != 0 {
		t.Errorf("expected 0 questions for SocraticNone, got %d", len(resp.Questions))
	}
}

// ---------------------------------------------------------------------------
// TestProgressiveDeepen — depth increases with turn count
// ---------------------------------------------------------------------------

func TestProgressiveDeepen(t *testing.T) {
	se := NewSocraticEngine()

	// Early turns: surface-level clarifying questions
	earlyQs := se.ProgressiveDeepen(1, "career change")
	if len(earlyQs) == 0 {
		t.Fatal("expected questions for turn 1")
	}
	for _, q := range earlyQs {
		if q.Category != "clarify" {
			t.Errorf("turn 1 question should be 'clarify', got %q", q.Category)
		}
		if q.Depth > 0 {
			t.Errorf("turn 1 question depth should be 0 (surface), got %d", q.Depth)
		}
	}

	// Middle turns: assumption-probing
	midQs := se.ProgressiveDeepen(3, "career change")
	if len(midQs) == 0 {
		t.Fatal("expected questions for turn 3")
	}
	allIntermediate := true
	for _, q := range midQs {
		if q.Category != "probe_assumptions" && q.Category != "evaluate_evidence" {
			allIntermediate = false
		}
	}
	if !allIntermediate {
		t.Error("turn 3-4 questions should be probe_assumptions or evaluate_evidence")
	}

	// Deep turns: implications and perspectives
	deepQs := se.ProgressiveDeepen(6, "career change")
	if len(deepQs) == 0 {
		t.Fatal("expected questions for turn 6")
	}
	allDeep := true
	for _, q := range deepQs {
		if q.Category != "explore_implications" && q.Category != "perspective_shift" {
			allDeep = false
		}
	}
	if !allDeep {
		t.Error("turn 5+ questions should be explore_implications or perspective_shift")
	}

	// Verify depth increases
	maxEarlyDepth := 0
	for _, q := range earlyQs {
		if q.Depth > maxEarlyDepth {
			maxEarlyDepth = q.Depth
		}
	}
	minDeepDepth := 100
	for _, q := range deepQs {
		if q.Depth < minDeepDepth {
			minDeepDepth = q.Depth
		}
	}
	if minDeepDepth <= maxEarlyDepth {
		t.Errorf("deep questions (min depth %d) should be deeper than early questions (max depth %d)",
			minDeepDepth, maxEarlyDepth)
	}
}

// ---------------------------------------------------------------------------
// TestQuestionsAreTopicSpecific — verify topic injection
// ---------------------------------------------------------------------------

func TestQuestionsAreTopicSpecific(t *testing.T) {
	se := NewSocraticEngine()

	topics := []struct {
		topic string
		mode  SocraticMode
	}{
		{"machine learning", SocraticExplore},
		{"career change", SocraticDecide},
		{"public speaking", SocraticCoach},
		{"remote work", SocraticChallenge},
		{"happiness", SocraticDeepen},
	}

	for _, tc := range topics {
		state := NewConversationState()
		state.ActiveTopic = tc.topic

		resp := se.Generate("tell me about "+tc.topic, tc.mode, state)

		for _, q := range resp.Questions {
			if !strings.Contains(strings.ToLower(q.Text), strings.ToLower(tc.topic)) {
				t.Errorf("question %q does not contain topic %q — questions must be topic-specific",
					q.Text, tc.topic)
			}
		}
	}
}

// ---------------------------------------------------------------------------
// TestQuestionBankCompleteness — at least 15 templates per category
// ---------------------------------------------------------------------------

func TestQuestionBankCompleteness(t *testing.T) {
	bank := buildQuestionBank()

	required := []string{
		"clarify",
		"probe_assumptions",
		"explore_implications",
		"evaluate_evidence",
		"perspective_shift",
	}

	totalTemplates := 0
	for _, cat := range required {
		templates, ok := bank[cat]
		if !ok {
			t.Errorf("missing question category %q", cat)
			continue
		}
		if len(templates) < 15 {
			t.Errorf("category %q has %d templates, want at least 15", cat, len(templates))
		}
		totalTemplates += len(templates)
	}

	if totalTemplates < 75 {
		t.Errorf("total templates = %d, want at least 75", totalTemplates)
	}
}

// ---------------------------------------------------------------------------
// TestSocraticModeString — verify String() method
// ---------------------------------------------------------------------------

func TestSocraticModeString(t *testing.T) {
	modes := []struct {
		mode SocraticMode
		want string
	}{
		{SocraticNone, "none"},
		{SocraticExplore, "explore"},
		{SocraticDecide, "decide"},
		{SocraticCoach, "coach"},
		{SocraticChallenge, "challenge"},
		{SocraticDeepen, "deepen"},
	}
	for _, tc := range modes {
		if got := tc.mode.String(); got != tc.want {
			t.Errorf("%d.String() = %q, want %q", tc.mode, got, tc.want)
		}
	}
}

// ---------------------------------------------------------------------------
// TestExtractTopic — topic extraction from various inputs
// ---------------------------------------------------------------------------

func TestExtractTopic(t *testing.T) {
	cases := []struct {
		query string
		state *ConversationState
		want  string
	}{
		{"should I learn Go?", nil, "learn Go"},
		{"help me think about career change", nil, "career change"},
		{"I'm curious about quantum computing", nil, "quantum computing"},
		{"random thing", &ConversationState{ActiveTopic: "cooking"}, "cooking"},
	}

	for _, tc := range cases {
		got := extractTopic(tc.query, tc.state)
		if got != tc.want {
			t.Errorf("extractTopic(%q) = %q, want %q", tc.query, got, tc.want)
		}
	}
}

// ---------------------------------------------------------------------------
// TestQuestionCategoriesAreCorrect — each question reports the right category
// ---------------------------------------------------------------------------

func TestQuestionCategoriesAreCorrect(t *testing.T) {
	se := NewSocraticEngine()
	state := NewConversationState()
	state.ActiveTopic = "testing"

	modeCategories := map[SocraticMode][]string{
		SocraticExplore:   {"clarify", "perspective_shift", "explore_implications"},
		SocraticDecide:    {"clarify", "explore_implications", "evaluate_evidence"},
		SocraticCoach:     {"clarify", "probe_assumptions", "explore_implications"},
		SocraticChallenge: {"probe_assumptions", "evaluate_evidence", "perspective_shift"},
		SocraticDeepen:    {"clarify", "probe_assumptions", "perspective_shift"},
	}

	for mode, expectedCats := range modeCategories {
		resp := se.Generate("testing question", mode, state)

		for _, q := range resp.Questions {
			found := false
			for _, cat := range expectedCats {
				if q.Category == cat {
					found = true
					break
				}
			}
			if !found {
				t.Errorf("mode %v: question category %q not in expected categories %v",
					mode, q.Category, expectedCats)
			}
		}
	}
}

// ---------------------------------------------------------------------------
// TestClaimDetectionDoesNotTriggerOnQuestions
// ---------------------------------------------------------------------------

func TestClaimDetectionDoesNotTriggerOnQuestions(t *testing.T) {
	se := NewSocraticEngine()
	state := NewConversationState()

	// Asking about strong claims should NOT be challenged
	questions := []string{
		"is it true that everyone knows about this?",
		"obviously this is wrong?",
	}

	for _, q := range questions {
		got := se.DetectMode(q, state)
		if got == SocraticChallenge {
			t.Errorf("DetectMode(%q) = SocraticChallenge, but questions about claims should not be challenged", q)
		}
	}
}

// ---------------------------------------------------------------------------
// TestNilStateHandling — engine works without conversation state
// ---------------------------------------------------------------------------

func TestNilStateHandling(t *testing.T) {
	se := NewSocraticEngine()

	// DetectMode should work with nil state
	mode := se.DetectMode("should I learn Go or Rust?", nil)
	if mode != SocraticDecide {
		t.Errorf("DetectMode with nil state = %v, want SocraticDecide", mode)
	}

	// Generate should work with nil state
	resp := se.Generate("should I learn Go or Rust?", SocraticDecide, nil)
	if resp.Mode != SocraticDecide {
		t.Errorf("Generate with nil state: mode = %v, want SocraticDecide", resp.Mode)
	}
	if len(resp.Questions) < 2 {
		t.Errorf("Generate with nil state: got %d questions, want at least 2", len(resp.Questions))
	}
}
