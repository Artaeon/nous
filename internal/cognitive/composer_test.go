package cognitive

import (
	"strings"
	"testing"
)

func TestComposerGreetingMorning(t *testing.T) {
	c := NewComposer(nil, nil, nil, nil)
	ctx := &ComposeContext{
		UserName:    "Raphael",
		HabitStreak: 7,
		JournalDays: 4,
	}

	resp := c.Compose("good morning", RespGreeting, ctx)
	if resp == nil {
		t.Fatal("expected response, got nil")
	}

	t.Logf("Greeting: %s", resp.Text)

	// Should mention the user's name
	if !strings.Contains(resp.Text, "Raphael") {
		t.Error("should mention user name")
	}
	// Should mention habit streak (>= 5)
	if !strings.Contains(resp.Text, "7") {
		t.Error("should mention habit streak")
	}
	// Should mention journal gap (> 2 days)
	if !strings.Contains(resp.Text, "4 day") {
		t.Error("should mention journal gap")
	}
}

func TestComposerFactual(t *testing.T) {
	cg := NewCognitiveGraph("")
	cg.EnsureNode("Stoicera", NodeEntity)
	cg.EnsureNode("philosophy company", NodeConcept)
	cg.EnsureNode("Vienna", NodeEntity)
	cg.EnsureNode("Raphael Lugmayr", NodeEntity)
	cg.AddEdge("stoicera", "philosophy company", RelIsA, "test")
	cg.AddEdge("stoicera", "vienna", RelLocatedIn, "test")
	cg.AddEdge("stoicera", "raphael lugmayr", RelFoundedBy, "test")

	c := NewComposer(cg, nil, nil, nil)
	resp := c.Compose("tell me about Stoicera", RespFactual, nil)
	if resp == nil {
		t.Fatal("expected response, got nil")
	}

	t.Logf("Factual: %s", resp.Text)

	// Should mention all three facts
	text := strings.ToLower(resp.Text)
	if !strings.Contains(text, "stoicera") {
		t.Error("should mention Stoicera")
	}
	if !strings.Contains(text, "vienna") {
		t.Error("should mention Vienna")
	}
	if !strings.Contains(text, "raphael") {
		t.Error("should mention Raphael")
	}
}

func TestComposerFactualVariation(t *testing.T) {
	cg := NewCognitiveGraph("")
	cg.EnsureNode("Rust", NodeConcept)
	cg.EnsureNode("programming language", NodeConcept)
	cg.AddEdge("rust", "programming language", RelIsA, "test")

	c := NewComposer(cg, nil, nil, nil)

	// Generate 20 responses — they should not all be identical
	responses := make(map[string]bool)
	for i := 0; i < 20; i++ {
		resp := c.Compose("what is Rust", RespFactual, nil)
		if resp != nil {
			responses[resp.Text] = true
		}
	}

	t.Logf("Got %d unique responses out of 20", len(responses))
	if len(responses) < 2 {
		t.Error("expected at least 2 variations in 20 attempts")
	}
}

func TestComposerAcknowledgeExpense(t *testing.T) {
	c := NewComposer(nil, nil, nil, nil)
	ctx := &ComposeContext{
		WeeklySpend:    47,
		AvgWeeklySpend: 60,
	}

	resp := c.Compose("spent 12 euros on lunch", RespAcknowledge, ctx)
	if resp == nil {
		t.Fatal("expected response, got nil")
	}

	t.Logf("Expense ack: %s", resp.Text)

	// Should acknowledge and mention spending trend
	if len(resp.Text) < 5 {
		t.Error("response too short")
	}
}

func TestComposerReflection(t *testing.T) {
	c := NewComposer(nil, nil, nil, nil)
	ctx := &ComposeContext{
		WeeklySpend:    47,
		AvgWeeklySpend: 60,
		RecentMood:     3.8,
		HabitStreak:    12,
		JournalDays:    1,
	}

	resp := c.Compose("how am I doing overall", RespReflect, ctx)
	if resp == nil {
		t.Fatal("expected response, got nil")
	}

	t.Logf("Reflection: %s", resp.Text)

	// Should contain spending info
	if !strings.Contains(resp.Text, "47") {
		t.Error("should mention spending amount")
	}
	// Should mention mood
	if !strings.Contains(resp.Text, "3.8") {
		t.Error("should mention mood score")
	}
	// Should mention habits
	if !strings.Contains(resp.Text, "12") {
		t.Error("should mention habit streak")
	}
}

func TestComposerBriefing(t *testing.T) {
	c := NewComposer(nil, nil, nil, nil)
	ctx := &ComposeContext{
		UserName:       "Raphael",
		WeeklySpend:    35,
		HabitStreak:    5,
		RecentMood:     4.2,
		JournalDays:    3,
	}

	resp := c.Compose("", RespBriefing, ctx)
	if resp == nil {
		t.Fatal("expected response, got nil")
	}

	t.Logf("Briefing:\n%s", resp.Text)

	if !strings.Contains(resp.Text, "35") {
		t.Error("should mention spending")
	}
	if !(strings.Contains(resp.Text, "5") && (strings.Contains(resp.Text, "streak") ||
		strings.Contains(resp.Text, "habit") || strings.Contains(resp.Text, "days") ||
		strings.Contains(resp.Text, "consecutive"))) {
		t.Error("should mention habit streak")
	}
}

func TestComposerUncertain(t *testing.T) {
	c := NewComposer(nil, nil, nil, nil)
	resp := c.Compose("what is quantum chromodynamics", RespUncertain, nil)
	if resp == nil {
		t.Fatal("expected response, got nil")
	}

	t.Logf("Uncertain: %s", resp.Text)

	if len(resp.Text) < 20 {
		t.Error("uncertain response too short")
	}
}

func TestComposerExplain(t *testing.T) {
	cg := NewCognitiveGraph("")
	cg.EnsureNode("Stoicism", NodeConcept)
	cg.EnsureNode("philosophy", NodeConcept)
	cg.EnsureNode("Marcus Aurelius", NodeEntity)
	cg.EnsureNode("inner peace", NodeConcept)
	cg.AddEdge("stoicism", "philosophy", RelIsA, "test")
	cg.AddEdge("stoicism", "marcus aurelius", RelRelatedTo, "test")
	cg.AddEdge("stoicism", "inner peace", RelUsedFor, "test")

	c := NewComposer(cg, nil, nil, nil)
	resp := c.Compose("explain stoicism", RespExplain, nil)
	if resp == nil {
		t.Fatal("expected response, got nil")
	}

	t.Logf("Explain: %s", resp.Text)

	text := strings.ToLower(resp.Text)
	if !strings.Contains(text, "stoicism") {
		t.Error("should mention stoicism")
	}
	if !strings.Contains(text, "philosophy") {
		t.Error("should mention philosophy")
	}
}

func TestComposerEdgeToSentence(t *testing.T) {
	c := NewComposer(nil, nil, nil, nil)

	tests := []struct {
		subj, obj string
		rel       RelType
		inferred  bool
		wantSubj  bool
		wantObj   bool
	}{
		{"Go", "programming language", RelIsA, false, true, true},
		{"Stoicera", "Vienna", RelLocatedIn, false, true, true},
		{"Stoicera", "Raphael", RelFoundedBy, true, true, true},
	}

	for _, tt := range tests {
		sentence := c.edgeToSentence(tt.subj, tt.rel, tt.obj, tt.inferred)
		t.Logf("  %s -[%s]-> %s (inferred=%v): %s", tt.subj, tt.rel, tt.obj, tt.inferred, sentence)
		if sentence == "" {
			t.Errorf("expected sentence for %s -> %s", tt.subj, tt.obj)
			continue
		}
		if tt.wantSubj && !strings.Contains(sentence, tt.subj) {
			t.Errorf("sentence should contain subject %q", tt.subj)
		}
		if tt.wantObj && !strings.Contains(sentence, tt.obj) {
			t.Errorf("sentence should contain object %q", tt.obj)
		}
		inferMarkers := []string{
			"infer", "based", "connected", "connecting", "gathered",
			"piecing", "available information", "reading between",
			"dots", "seen",
		}
		hasInferMarker := false
		for _, marker := range inferMarkers {
			if strings.Contains(strings.ToLower(sentence), marker) {
				hasInferMarker = true
				break
			}
		}
		if tt.inferred && !hasInferMarker {
			t.Errorf("inferred fact should be marked")
		}
	}
}

func TestComposerCombineWithFlow(t *testing.T) {
	c := NewComposer(nil, nil, nil, nil)

	sentences := []string{
		"Stoicera is a philosophy company.",
		"Stoicera is based in Vienna.",
		"Stoicera was founded by Raphael.",
	}

	// Run multiple times — different flow strategies should produce variation
	hasConnector := false
	for i := 0; i < 20; i++ {
		result := c.combineWithFlow(sentences)
		if i == 0 {
			t.Logf("Combined: %s", result)
		}

		// Should contain all facts
		if !strings.Contains(result, "philosophy") {
			t.Error("missing philosophy")
		}
		if !strings.Contains(result, "Vienna") {
			t.Error("missing Vienna")
		}
		if !strings.Contains(result, "Raphael") {
			t.Error("missing Raphael")
		}

		for _, conn := range allConnectors {
			if strings.Contains(result, strings.TrimSpace(conn)) {
				hasConnector = true
			}
		}
	}
	if !hasConnector {
		t.Error("should use discourse connectors in at least some variations")
	}
}

func TestLowerFirst(t *testing.T) {
	tests := []struct {
		in, want string
	}{
		{"Hello", "hello"},
		{"", ""},
		{"ABC", "ABC"}, // acronym preserved
		{"Go is great", "go is great"},
	}
	for _, tt := range tests {
		got := lowerFirst(tt.in)
		if got != tt.want {
			t.Errorf("lowerFirst(%q) = %q, want %q", tt.in, got, tt.want)
		}
	}
}

func TestComposerConversational(t *testing.T) {
	cg := NewCognitiveGraph("")
	c := NewComposer(cg, nil, nil, nil)

	ctx := &ComposeContext{UserName: "Raphael"}
	resp := c.Compose("I just wanted to talk to you", RespConversational, ctx)
	if resp == nil || resp.Text == "" {
		t.Fatal("conversational response should not be empty")
	}
	t.Logf("Talk: %s", resp.Text)
}

func TestComposerEmpathetic(t *testing.T) {
	c := NewComposer(NewCognitiveGraph(""), nil, nil, nil)

	resp := c.Compose("I'm feeling really sad today", RespEmpathetic, nil)
	if resp == nil || resp.Text == "" {
		t.Fatal("empathetic response should not be empty")
	}
	t.Logf("Empathetic: %s", resp.Text)
}

func TestComposerOpinion(t *testing.T) {
	cg := NewCognitiveGraph("")
	cg.AddEdge("Go", "programming language", RelIsA, "test")
	cg.AddEdge("Go", "fast", RelDescribedAs, "test")
	c := NewComposer(cg, nil, nil, nil)

	resp := c.Compose("what do you think about Go", RespOpinion, nil)
	if resp == nil || resp.Text == "" {
		t.Fatal("opinion response should not be empty")
	}
	t.Logf("Opinion: %s", resp.Text)
	if !strings.Contains(resp.Text, "Go") {
		t.Error("opinion should reference the topic")
	}
}

func TestComposerFarewell(t *testing.T) {
	c := NewComposer(NewCognitiveGraph(""), nil, nil, nil)
	ctx := &ComposeContext{UserName: "Raphael", HabitStreak: 5}

	resp := c.Compose("bye", RespFarewell, ctx)
	if resp == nil || resp.Text == "" {
		t.Fatal("farewell should not be empty")
	}
	t.Logf("Farewell: %s", resp.Text)
	if !strings.Contains(resp.Text, "Raphael") {
		t.Error("farewell should use name")
	}
}

func TestComposerThankYou(t *testing.T) {
	c := NewComposer(NewCognitiveGraph(""), nil, nil, nil)

	resp := c.Compose("thanks", RespThankYou, nil)
	if resp == nil || resp.Text == "" {
		t.Fatal("thank you response should not be empty")
	}
	t.Logf("Thanks: %s", resp.Text)
}

func TestComposerSentiment(t *testing.T) {
	c := NewComposer(NewCognitiveGraph(""), nil, nil, nil)

	tests := []struct {
		input string
		want  Sentiment
	}{
		{"I love this!", SentimentPositive},
		{"this is terrible", SentimentNegative},
		{"I'm so excited!", SentimentExcited},
		{"I'm really sad", SentimentSad},
		{"I'm furious", SentimentAngry},
		{"what is that?", SentimentCurious},
		{"okay", SentimentNeutral},
	}

	for _, tt := range tests {
		got := c.detectSentiment(tt.input)
		if got != tt.want {
			t.Errorf("detectSentiment(%q) = %d, want %d", tt.input, got, tt.want)
		}
	}
}

func TestComposerNeverReturnsNil(t *testing.T) {
	c := NewComposer(NewCognitiveGraph(""), nil, nil, nil)

	// Even for completely unknown inputs, Compose should NEVER return nil
	inputs := []string{
		"asdfghjkl", "tell me something random",
		"what's the meaning of life", "do you dream?",
		"12345",
	}
	for _, input := range inputs {
		resp := c.Compose(input, RespConversational, nil)
		if resp == nil || resp.Text == "" {
			t.Errorf("Compose(%q) returned nil/empty — should NEVER happen", input)
		}
	}
}

func TestComposerVariationAcrossTypes(t *testing.T) {
	c := NewComposer(NewCognitiveGraph(""), nil, nil, nil)
	ctx := &ComposeContext{UserName: "Raphael"}

	// Each type should produce output and they should all be different
	types := []ResponseType{
		RespGreeting, RespConversational, RespEmpathetic,
		RespFarewell, RespThankYou, RespUncertain,
	}
	responses := make(map[string]bool)
	for _, rt := range types {
		resp := c.Compose("hello", rt, ctx)
		if resp == nil || resp.Text == "" {
			t.Errorf("type %d returned empty", rt)
			continue
		}
		responses[resp.Text] = true
		t.Logf("Type %d: %s", rt, resp.Text)
	}
	if len(responses) < 4 {
		t.Errorf("expected at least 4 unique responses across types, got %d", len(responses))
	}
}

// -----------------------------------------------------------------------
// Self-Improving Phrase Pool Tests
// -----------------------------------------------------------------------

func TestComposerEngagementDetection(t *testing.T) {
	c := NewComposer(NewCognitiveGraph(""), nil, nil, nil)

	tests := []struct {
		input string
		min   float64
		max   float64
	}{
		{"thanks, that was great!", 0.5, 1.5},
		{"perfect, exactly what I needed", 0.5, 1.5},
		{"no that's wrong", -1.5, 0.0},
		{"I meant something else", -1.5, 0.0},
		{"okay", 0.0, 0.5},
	}

	for _, tt := range tests {
		score := c.DetectEngagement(tt.input)
		if score < tt.min || score > tt.max {
			t.Errorf("DetectEngagement(%q) = %.1f, want [%.1f, %.1f]",
				tt.input, score, tt.min, tt.max)
		}
	}
}

func TestComposerPhraseScoresEvolve(t *testing.T) {
	c := NewComposer(NewCognitiveGraph(""), nil, nil, nil)
	ctx := &ComposeContext{UserName: "Raphael"}

	// Generate several responses to build up phrase history
	for i := 0; i < 5; i++ {
		c.Compose("hello", RespGreeting, ctx)
		// Simulate positive engagement
		c.ObserveEngagement(1.0)
	}

	// Phrases used during positive engagement should have scores > 1.0
	boosted := 0
	for _, score := range c.phraseScores {
		if score > 1.0 {
			boosted++
		}
	}
	if boosted == 0 {
		t.Error("expected some phrases to be boosted after positive engagement")
	}
	t.Logf("Boosted %d phrases after 5 positive interactions", boosted)
}

func TestComposerPhraseScoresDecay(t *testing.T) {
	c := NewComposer(NewCognitiveGraph(""), nil, nil, nil)
	ctx := &ComposeContext{UserName: "Raphael"}

	// Generate responses and simulate negative engagement
	for i := 0; i < 5; i++ {
		c.Compose("hello", RespGreeting, ctx)
		c.ObserveEngagement(-1.0)
	}

	// Some phrases should have scores < 1.0
	decayed := 0
	for _, score := range c.phraseScores {
		if score < 1.0 {
			decayed++
		}
	}
	if decayed == 0 {
		t.Error("expected some phrases to decay after negative engagement")
	}
	t.Logf("Decayed %d phrases after 5 negative interactions", decayed)
}

func TestComposerEmotionalMemory(t *testing.T) {
	c := NewComposer(NewCognitiveGraph(""), nil, nil, nil)

	// Record turns with negative sentiment — "terrible" and "deadline" are
	// long enough to survive extractKeywords (>= 3 chars, not stop words).
	c.RecordTurn("the deadline is terrible and I hate everything about it", "I understand.")
	c.RecordTurn("this deadline makes me furious", "That sounds rough.")

	// Check which topics were stored
	hasNegative := false
	for topic, tone := range c.emotionalMem {
		t.Logf("Emotional tone for %q: %.2f", topic, tone)
		if tone < 0 {
			hasNegative = true
		}
	}
	if !hasNegative {
		t.Error("should have negative emotional tone for at least one topic")
	}

	// Now record positive sentiment
	c.RecordTurn("I love cooking, it makes me so happy!", "That's wonderful!")
	c.RecordTurn("cooking is amazing and exciting!", "Great to hear!")

	hasPositive := false
	for topic, tone := range c.emotionalMem {
		if tone > 0 {
			hasPositive = true
			t.Logf("Positive topic: %q = %.2f", topic, tone)
		}
	}
	if !hasPositive {
		t.Error("should have positive emotional tone for at least one topic")
	}
}

func TestComposerAutoEngagement(t *testing.T) {
	c := NewComposer(NewCognitiveGraph(""), nil, nil, nil)
	ctx := &ComposeContext{UserName: "Raphael"}

	// First response: generates phrases
	c.Compose("hello", RespGreeting, ctx)

	// Record turn with positive input — should auto-detect and boost
	c.RecordTurn("thanks that's so nice!", "You're welcome!")

	// Now check that scores were updated
	updated := false
	for _, score := range c.phraseScores {
		if score != 1.0 {
			updated = true
			break
		}
	}
	if !updated {
		t.Error("RecordTurn should auto-detect engagement and update scores")
	}
}

func TestComposerPhraseStats(t *testing.T) {
	c := NewComposer(NewCognitiveGraph(""), nil, nil, nil)
	ctx := &ComposeContext{UserName: "Raphael"}

	// Generate and score several interactions
	for i := 0; i < 10; i++ {
		c.Compose("hello", RespGreeting, ctx)
		if i%2 == 0 {
			c.ObserveEngagement(1.0)
		} else {
			c.ObserveEngagement(-0.5)
		}
	}

	top, bottom := c.PhraseStats(3)
	t.Logf("Top phrases: %v", top)
	t.Logf("Bottom phrases: %v", bottom)

	if len(top) == 0 {
		t.Error("should have top-scored phrases after interactions")
	}
}

func TestComposerWeightedSelection(t *testing.T) {
	c := NewComposer(NewCognitiveGraph(""), nil, nil, nil)

	// Manually set scores: "alpha" is strongly preferred
	c.phraseScores["alpha"] = 3.0
	c.phraseScores["beta"] = 0.1
	c.phraseScores["gamma"] = 0.1

	options := []string{"alpha", "beta", "gamma"}
	counts := make(map[string]int)
	for i := 0; i < 100; i++ {
		// Reset recent to allow re-picking
		c.recent = make(map[string]int)
		c.sessionPhrases = make(map[string]int)
		chosen := c.pick(options)
		counts[chosen]++
	}

	t.Logf("Selection distribution: %v", counts)
	// "alpha" (score 3.0) should be chosen more than "beta" (score 0.1)
	if counts["alpha"] <= counts["beta"] {
		t.Errorf("higher-scored 'alpha' (%d) should be chosen more than 'beta' (%d)",
			counts["alpha"], counts["beta"])
	}
}

func TestSentimentToFloat(t *testing.T) {
	tests := []struct {
		sentiment Sentiment
		wantSign  int // 1 = positive, -1 = negative, 0 = zero
	}{
		{SentimentPositive, 1},
		{SentimentExcited, 1},
		{SentimentCurious, 1},
		{SentimentNeutral, 0},
		{SentimentNegative, -1},
		{SentimentSad, -1},
		{SentimentAngry, -1},
	}
	for _, tt := range tests {
		got := sentimentToFloat(tt.sentiment)
		switch tt.wantSign {
		case 1:
			if got <= 0 {
				t.Errorf("sentimentToFloat(%d) = %.1f, want positive", tt.sentiment, got)
			}
		case -1:
			if got >= 0 {
				t.Errorf("sentimentToFloat(%d) = %.1f, want negative", tt.sentiment, got)
			}
		case 0:
			if got != 0 {
				t.Errorf("sentimentToFloat(%d) = %.1f, want zero", tt.sentiment, got)
			}
		}
	}
}
