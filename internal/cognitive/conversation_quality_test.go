package cognitive

import (
	"fmt"
	"strings"
	"testing"
)

// -----------------------------------------------------------------------
// Conversation Quality & Uniqueness Tests
// These simulate real multi-turn conversations and measure how well
// the Composer engine performs without any LLM.
// -----------------------------------------------------------------------

// setupFullEngine creates a Composer + LearningEngine with preloaded knowledge.
func setupFullEngine() (*Composer, *LearningEngine, *ActionRouter) {
	graph := NewCognitiveGraph("")
	semantic := NewSemanticEngine()
	causal := NewCausalEngine()
	patterns := NewPatternDetector()
	composer := NewComposer(graph, semantic, causal, patterns)
	learning := NewLearningEngine(graph, composer, "")
	ar := NewActionRouter()
	ar.CogGraph = graph
	ar.Composer = composer

	// Preload knowledge — simulate a user who has been talking to Nous
	facts := []string{
		"Stoicera is a philosophy company.",
		"Stoicera was founded by Raphael.",
		"Stoicera is based in Vienna.",
		"Go is a programming language.",
		"Go was created by Google.",
		"Go is used for backend development.",
		"Raphael is a software engineer.",
		"Raphael lives in Vienna.",
		"Rust is a systems programming language.",
		"Python is used for data science.",
		"Vienna is located in Austria.",
		"Marcus Aurelius is a Stoic philosopher.",
		"Stoicism is a philosophy.",
		"Stoicism is used for inner peace.",
	}
	for _, f := range facts {
		learning.LearnFromConversation(f)
	}

	return composer, learning, ar
}

func TestConversationGreetingFlow(t *testing.T) {
	composer, _, _ := setupFullEngine()
	ctx := &ComposeContext{
		UserName:    "Raphael",
		HabitStreak: 5,
	}

	// Simulate a greeting conversation
	greetings := []string{"hi", "hey there", "good morning", "what's up", "hello"}
	responses := make(map[string]bool)

	for _, g := range greetings {
		resp := composer.Compose(g, RespGreeting, ctx)
		if resp == nil || resp.Text == "" {
			t.Errorf("no response for greeting %q", g)
			continue
		}
		responses[resp.Text] = true
		composer.RecordTurn(g, resp.Text)
		t.Logf("  User: %s", g)
		t.Logf("  Nous: %s", resp.Text)
		t.Log("  ---")
	}

	uniqueRatio := float64(len(responses)) / float64(len(greetings)) * 100
	t.Logf("  Uniqueness: %d/%d (%.0f%%)", len(responses), len(greetings), uniqueRatio)
	if len(responses) < 3 {
		t.Errorf("too few unique greetings: %d out of %d", len(responses), len(greetings))
	}
}

func TestConversationMultiTurn(t *testing.T) {
	composer, learning, _ := setupFullEngine()
	ctx := &ComposeContext{
		UserName:    "Raphael",
		HabitStreak: 7,
		WeeklySpend:   45.0,
		RecentMood:     4.2,
	}

	// Simulate a realistic multi-turn conversation
	turns := []struct {
		input    string
		respType ResponseType
	}{
		{"hey Nous", RespGreeting},
		{"how am I doing this week?", RespBriefing},
		{"that's great, thanks!", RespThankYou},
		{"what do you know about Go?", RespFactual},
		{"what do you think about programming?", RespOpinion},
		{"I've been feeling stressed about work", RespEmpathetic},
		{"tell me about Stoicera", RespFactual},
		{"thanks for the chat", RespThankYou},
		{"goodbye!", RespFarewell},
	}

	t.Log("=== Multi-Turn Conversation ===")
	allResponses := make([]string, 0, len(turns))
	for _, turn := range turns {
		resp := composer.Compose(turn.input, turn.respType, ctx)
		if resp == nil || resp.Text == "" {
			t.Errorf("no response for %q", turn.input)
			continue
		}

		composer.RecordTurn(turn.input, resp.Text)
		learning.LearnFromConversation(turn.input)
		allResponses = append(allResponses, resp.Text)

		t.Logf("  User: %s", turn.input)
		t.Logf("  Nous: %s", resp.Text)
		t.Log("  ---")
	}

	// Check no two responses are identical
	seen := make(map[string]bool)
	dupes := 0
	for _, r := range allResponses {
		if seen[r] {
			dupes++
		}
		seen[r] = true
	}
	if dupes > 0 {
		t.Errorf("found %d duplicate responses in conversation", dupes)
	}

	t.Logf("  Total turns: %d, Unique responses: %d, Duplicates: %d",
		len(allResponses), len(seen), dupes)
}

func TestConversationSameQuestionVariation(t *testing.T) {
	composer, _, _ := setupFullEngine()
	ctx := &ComposeContext{UserName: "Raphael"}

	// Ask the same question 20 times — measure how many unique answers we get
	question := "tell me about Stoicera"
	responses := make(map[string]bool)

	t.Log("=== Same Question 20 Times ===")
	for i := 0; i < 20; i++ {
		resp := composer.Compose(question, RespFactual, ctx)
		if resp == nil || resp.Text == "" {
			t.Error("nil response")
			continue
		}
		responses[resp.Text] = true
		// Don't record turn — we want to test raw variation
	}

	uniqueRatio := float64(len(responses)) / 20.0 * 100
	t.Logf("  Unique responses: %d/20 (%.0f%%)", len(responses), uniqueRatio)

	// Show a sample of responses
	count := 0
	for r := range responses {
		if count < 5 {
			t.Logf("  Sample %d: %s", count+1, r)
		}
		count++
	}

	if len(responses) < 5 {
		t.Errorf("expected at least 5 unique responses for same question, got %d", len(responses))
	}
}

func TestConversationLearningInAction(t *testing.T) {
	composer, learning, _ := setupFullEngine()
	ctx := &ComposeContext{UserName: "Raphael"}

	t.Log("=== Learning From Conversation ===")

	// Teach Nous new facts through conversation
	teachings := []string{
		"Did you know Kubernetes was created by Google?",
		"Docker is a containerization platform.",
		"I love building distributed systems.",
		"My favorite editor is Neovim.",
		"I live in Vienna.",
		"Let me teach you: PostgreSQL is a relational database.",
	}

	totalLearned := 0
	for _, teach := range teachings {
		n := learning.LearnFromConversation(teach)
		totalLearned += n

		// Generate a response
		respType := ClassifyForComposer(teach)
		resp := composer.Compose(teach, respType, ctx)
		composer.RecordTurn(teach, resp.Text)

		t.Logf("  User: %s", teach)
		t.Logf("  Nous: %s", resp.Text)
		t.Logf("  [Learned %d new facts]", n)
		t.Log("  ---")
	}

	t.Logf("  Total facts learned: %d", totalLearned)
	if totalLearned < 3 {
		t.Errorf("should learn at least 3 facts, learned %d", totalLearned)
	}

	// Check learning report
	report := learning.FormatLearningReport()
	t.Logf("\n  %s", strings.ReplaceAll(report, "\n", "\n  "))

	stats := learning.Stats()
	if stats.TotalFacts == 0 {
		t.Error("should have accumulated facts")
	}
}

func TestConversationEngagementEvolution(t *testing.T) {
	composer, _, _ := setupFullEngine()
	ctx := &ComposeContext{UserName: "Raphael"}

	t.Log("=== Self-Improving Through Engagement ===")

	// Simulate 10 turns with varying engagement
	interactions := []struct {
		query    string
		respType ResponseType
		followUp string // simulates user response
	}{
		{"hello", RespGreeting, "thanks, that's nice!"},
		{"hello", RespGreeting, "perfect"},
		{"hello", RespGreeting, "no that's weird"},
		{"hello", RespGreeting, "great!"},
		{"hello", RespGreeting, "awesome"},
		{"hello", RespGreeting, "that's not right"},
		{"hello", RespGreeting, "love it"},
		{"hello", RespGreeting, "excellent!"},
		{"hello", RespGreeting, "thank you"},
		{"hello", RespGreeting, "wonderful"},
	}

	for _, inter := range interactions {
		resp := composer.Compose(inter.query, inter.respType, ctx)
		// Simulate engagement from follow-up
		composer.RecordTurn(inter.followUp, "")
		t.Logf("  Nous: %-60s → User: %s", resp.Text, inter.followUp)
	}

	top, bottom := composer.PhraseStats(5)
	t.Log("\n  Top-scoring phrases (user liked these):")
	for _, p := range top {
		t.Logf("    %s", p)
	}
	t.Log("  Bottom-scoring phrases (user didn't like):")
	for _, p := range bottom {
		t.Logf("    %s", p)
	}

	if len(top) == 0 {
		t.Error("should have differentiated phrase scores after engagement")
	}
}

func TestConversationEmotionalAdaptation(t *testing.T) {
	composer, _, _ := setupFullEngine()
	ctx := &ComposeContext{UserName: "Raphael"}

	t.Log("=== Emotional Memory Across Turns ===")

	// Send messages with different emotional tones about different topics
	emotionalInputs := []struct {
		input   string
		topic   string
	}{
		{"I love working on Nous, it makes me so happy!", "nous"},
		{"work deadlines are killing me, so stressed", "deadlines"},
		{"programming is the most exciting thing ever!", "programming"},
		{"taxes are terrible and frustrating", "taxes"},
		{"building Nous is absolutely thrilling!", "nous"},
	}

	for _, ei := range emotionalInputs {
		resp := composer.Compose(ei.input, RespEmpathetic, ctx)
		composer.RecordTurn(ei.input, resp.Text)
		tone := composer.EmotionalTone(ei.topic)

		t.Logf("  User: %s", ei.input)
		t.Logf("  Nous: %s", resp.Text)
		t.Logf("  [Emotional memory for %q: %.2f]", ei.topic, tone)
		t.Log("  ---")
	}

	// Verify emotional differentiation
	nousEmotion := composer.EmotionalTone("nous")
	taxesEmotion := composer.EmotionalTone("taxes")
	t.Logf("  Final: 'nous'=%.2f (should be positive), 'taxes'=%.2f (should be negative)",
		nousEmotion, taxesEmotion)
}

func TestConversationResponseTypes(t *testing.T) {
	composer, _, _ := setupFullEngine()
	ctx := &ComposeContext{
		UserName:     "Raphael",
		HabitStreak:  12,
		WeeklySpend:    67.0,
		RecentMood:      3.8,
		JournalDays:  2,
	}

	t.Log("=== All 13 Response Types ===")

	types := []struct {
		name     string
		query    string
		respType ResponseType
	}{
		{"Factual", "tell me about Go", RespFactual},
		{"Personal", "how am I doing?", RespPersonal},
		{"Briefing", "give me my briefing", RespBriefing},
		{"Acknowledge", "logged 15 euros for lunch", RespAcknowledge},
		{"Explain", "explain Stoicism to me", RespExplain},
		{"Reflect", "any patterns in my behavior?", RespReflect},
		{"Greeting", "good morning!", RespGreeting},
		{"Uncertain", "what's the meaning of quantum chromodynamics?", RespUncertain},
		{"Conversational", "so I was thinking about life", RespConversational},
		{"Empathetic", "I'm feeling really down today", RespEmpathetic},
		{"Opinion", "what do you think about Rust?", RespOpinion},
		{"Farewell", "goodbye for now", RespFarewell},
		{"ThankYou", "thanks so much!", RespThankYou},
	}

	allPassed := true
	for _, tt := range types {
		resp := composer.Compose(tt.query, tt.respType, ctx)
		if resp == nil || resp.Text == "" {
			t.Errorf("  FAIL %s: no response", tt.name)
			allPassed = false
			continue
		}
		composer.RecordTurn(tt.query, resp.Text)
		t.Logf("  [%s] User: %s", tt.name, tt.query)
		t.Logf("  [%s] Nous: %s", tt.name, resp.Text)
		t.Log("  ---")
	}

	if allPassed {
		t.Log("  All 13 response types produced responses.")
	}
}

func TestConversationFullSession(t *testing.T) {
	composer, learning, _ := setupFullEngine()
	ctx := &ComposeContext{
		UserName:    "Raphael",
		HabitStreak: 7,
		WeeklySpend:   52.0,
		RecentMood:     4.0,
		JournalDays: 1,
	}

	t.Log("╔══════════════════════════════════════════╗")
	t.Log("║   Full Conversation Session Simulation   ║")
	t.Log("╚══════════════════════════════════════════╝")
	t.Log("")

	session := []struct {
		input    string
		respType ResponseType
	}{
		// Natural conversation flow
		{"good morning Nous!", RespGreeting},
		{"how am I doing this week?", RespBriefing},
		{"that's helpful, thanks", RespThankYou},
		{"what do you know about Go?", RespFactual},
		{"I've been learning Rust lately, it's exciting!", RespConversational},
		{"did you know Rust was created by Mozilla?", RespConversational},
		{"what do you think about systems programming?", RespOpinion},
		{"I'm a bit stressed about a deadline", RespEmpathetic},
		{"tell me something about Stoicera", RespFactual},
		{"any patterns you've noticed in my behavior?", RespReflect},
		{"that's really insightful", RespThankYou},
		{"alright, I need to get back to work", RespFarewell},
	}

	uniqueResponses := make(map[string]bool)
	for i, turn := range session {
		resp := composer.Compose(turn.input, turn.respType, ctx)
		if resp == nil || resp.Text == "" {
			t.Errorf("Turn %d: no response for %q", i+1, turn.input)
			continue
		}

		uniqueResponses[resp.Text] = true
		composer.RecordTurn(turn.input, resp.Text)
		learning.LearnFromConversation(turn.input)

		t.Logf("  Turn %d", i+1)
		t.Logf("  Raphael: %s", turn.input)
		t.Logf("  Nous:    %s", resp.Text)
		t.Log("")
	}

	// Final stats
	t.Log("═══ Session Summary ═══")
	t.Logf("  Total turns: %d", len(session))
	t.Logf("  Unique responses: %d/%d (%.0f%%)",
		len(uniqueResponses), len(session),
		float64(len(uniqueResponses))/float64(len(session))*100)

	stats := learning.Stats()
	t.Logf("  Facts in knowledge graph: %d", stats.TotalFacts)
	t.Logf("  Patterns absorbed: %d", stats.PatternsLearned)
	t.Logf("  Top interests: %s", strings.Join(stats.TopTopics, ", "))

	top, _ := composer.PhraseStats(3)
	if len(top) > 0 {
		t.Log("  Top phrases by engagement:")
		for _, p := range top {
			t.Logf("    %s", p)
		}
	}

	// Quality checks
	if len(uniqueResponses) < len(session)-1 {
		t.Errorf("too many duplicate responses: %d unique out of %d", len(uniqueResponses), len(session))
	}
	if stats.TotalFacts < 5 {
		t.Errorf("should have accumulated knowledge, got %d facts", stats.TotalFacts)
	}

	fmt.Println() // clean output separator
}

// ClassifyForComposer is a helper that maps queries to response types.
// (Exported version is on ActionRouter, this is a standalone for testing.)
func ClassifyForComposer(query string) ResponseType {
	lower := strings.ToLower(query)

	if isGreeting(lower) {
		return RespGreeting
	}
	if isFarewell(lower) {
		return RespFarewell
	}
	if isThankYou(lower) {
		return RespThankYou
	}
	if isEmotional(lower) {
		return RespEmpathetic
	}

	// Teaching / fact sharing
	for _, sig := range []string{"did you know", "fun fact", "let me teach"} {
		if strings.Contains(lower, sig) {
			return RespConversational
		}
	}

	// Opinion
	if strings.Contains(lower, "think about") || strings.Contains(lower, "opinion") {
		return RespOpinion
	}

	// Factual
	if strings.Contains(lower, "tell me about") || strings.Contains(lower, "what is") {
		return RespFactual
	}

	return RespConversational
}
