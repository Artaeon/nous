package cognitive

import (
	"strings"
	"testing"
	"time"
)

func TestObserve(t *testing.T) {
	ic := NewInsightCrystallizer()

	ic.Observe("tell me about quantum physics", map[string]string{"topic": "quantum physics"}, "positive")
	ic.Observe("how does machine learning work?", map[string]string{"topic": "machine learning"}, "neutral")

	if ic.ObservationCount() != 2 {
		t.Fatalf("observation count = %d, want 2", ic.ObservationCount())
	}

	// Verify observations are recorded with correct fields.
	ic.mu.RLock()
	obs := ic.observations[0]
	ic.mu.RUnlock()

	if obs.Topic != "quantum physics" {
		t.Errorf("topic = %q, want %q", obs.Topic, "quantum physics")
	}
	if obs.Sentiment != "positive" {
		t.Errorf("sentiment = %q, want %q", obs.Sentiment, "positive")
	}
	if len(obs.Keywords) == 0 {
		t.Error("keywords should not be empty")
	}
	if obs.TurnIndex != 0 {
		t.Errorf("first turn index = %d, want 0", obs.TurnIndex)
	}

	// Second observation should have turn index 1.
	ic.mu.RLock()
	obs2 := ic.observations[1]
	ic.mu.RUnlock()
	if obs2.TurnIndex != 1 {
		t.Errorf("second turn index = %d, want 1", obs2.TurnIndex)
	}
}

func TestObserveEviction(t *testing.T) {
	ic := NewInsightCrystallizer()
	ic.maxObs = 5

	for i := 0; i < 10; i++ {
		ic.Observe("test query", map[string]string{"topic": "test"}, "neutral")
	}

	if ic.ObservationCount() != 5 {
		t.Errorf("after eviction, count = %d, want 5", ic.ObservationCount())
	}
}

func TestRecurringThemes(t *testing.T) {
	ic := NewInsightCrystallizer()

	// Mention "career change" 4 times.
	for i := 0; i < 4; i++ {
		ic.Observe("I've been thinking about a career change", map[string]string{"topic": "career change"}, "positive")
	}
	// Mention "cooking" once — should NOT be a recurring theme.
	ic.Observe("what's a good pasta recipe?", map[string]string{"topic": "cooking"}, "neutral")

	insights := ic.Crystallize()

	var recurring []CrystallizedInsight
	for _, ins := range insights {
		if ins.Type == "recurring_theme" {
			recurring = append(recurring, ins)
		}
	}

	if len(recurring) == 0 {
		t.Fatal("expected at least one recurring theme insight")
	}

	found := false
	for _, r := range recurring {
		if strings.Contains(r.Text, "career change") {
			found = true
			if !strings.Contains(r.Text, "4 times") {
				t.Errorf("insight should mention count; got %q", r.Text)
			}
			if len(r.Evidence) < 4 {
				t.Errorf("evidence items = %d, want >= 4", len(r.Evidence))
			}
			break
		}
	}
	if !found {
		t.Error("no recurring theme insight found for 'career change'")
	}

	// "cooking" should NOT appear as recurring.
	for _, r := range recurring {
		if strings.Contains(r.Text, "cooking") {
			t.Error("cooking should not be a recurring theme (only mentioned once)")
		}
	}
}

func TestCrossConnections(t *testing.T) {
	ic := NewInsightCrystallizer()

	// Two seemingly unrelated topics that share multiple keywords.
	// Need 3+ shared keywords to exceed the cross-connection threshold.
	ic.Observe("painting involves creativity expression and design thinking", map[string]string{"topic": "painting"}, "positive")
	ic.Observe("visual art requires creativity innovation and design skills", map[string]string{"topic": "painting"}, "positive")
	ic.Observe("painting uses pattern recognition and abstract thinking", map[string]string{"topic": "painting"}, "positive")
	ic.Observe("software engineering requires creativity and design thinking", map[string]string{"topic": "software"}, "positive")
	ic.Observe("coding involves pattern recognition innovation and abstract design", map[string]string{"topic": "software"}, "positive")
	ic.Observe("software development uses creative problem solving and design", map[string]string{"topic": "software"}, "positive")

	insights := ic.Crystallize()

	var connections []CrystallizedInsight
	for _, ins := range insights {
		if ins.Type == "cross_connection" {
			connections = append(connections, ins)
		}
	}

	if len(connections) == 0 {
		t.Fatal("expected at least one cross-connection insight")
	}

	found := false
	for _, c := range connections {
		if (strings.Contains(c.Text, "painting") && strings.Contains(c.Text, "software")) ||
			(strings.Contains(c.Text, "software") && strings.Contains(c.Text, "painting")) {
			found = true
			if !strings.Contains(strings.ToLower(c.Text), "creativity") {
				t.Errorf("cross-connection should mention shared keyword 'creativity'; got %q", c.Text)
			}
			break
		}
	}
	if !found {
		t.Error("no cross-connection found between painting and software")
	}
}

func TestUnresolvedTensions(t *testing.T) {
	ic := NewInsightCrystallizer()

	// Positive feelings about freelancing in some turns.
	ic.Observe("freelancing gives me so much freedom", map[string]string{"topic": "freelancing"}, "positive")
	ic.Observe("I love the flexibility of freelancing", map[string]string{"topic": "freelancing"}, "positive")
	// Negative feelings in others.
	ic.Observe("freelancing income is so unpredictable", map[string]string{"topic": "freelancing"}, "negative")
	ic.Observe("the uncertainty of freelancing is stressful", map[string]string{"topic": "freelancing"}, "negative")

	insights := ic.Crystallize()

	var tensions []CrystallizedInsight
	for _, ins := range insights {
		if ins.Type == "tension" {
			tensions = append(tensions, ins)
		}
	}

	if len(tensions) == 0 {
		t.Fatal("expected at least one tension insight")
	}

	found := false
	for _, ten := range tensions {
		if strings.Contains(ten.Text, "freelancing") {
			found = true
			if !strings.Contains(ten.Text, "conflicted") {
				t.Errorf("tension insight should use word 'conflicted'; got %q", ten.Text)
			}
			if !strings.Contains(ten.Text, "excited") || !strings.Contains(ten.Text, "concerned") {
				t.Errorf("tension should mention both sides; got %q", ten.Text)
			}
			if len(ten.Evidence) < 4 {
				t.Errorf("evidence = %d, want >= 4", len(ten.Evidence))
			}
			break
		}
	}
	if !found {
		t.Error("no tension insight found for 'freelancing'")
	}
}

func TestShouldSurface(t *testing.T) {
	ic := NewInsightCrystallizer()

	// With no observations, should not surface.
	if ic.ShouldSurface(0) {
		t.Error("should not surface with 0 observations")
	}

	// With just 2 observations, still no.
	ic.Observe("hello", map[string]string{"topic": "greeting"}, "neutral")
	ic.Observe("hi again", map[string]string{"topic": "greeting"}, "neutral")
	if ic.ShouldSurface(2) {
		t.Error("should not surface with only 2 observations")
	}

	// After 10+ new observations, should surface.
	for i := 0; i < 11; i++ {
		ic.Observe("thinking about career", map[string]string{"topic": "career"}, "positive")
	}
	if !ic.ShouldSurface(13) {
		t.Error("should surface after 10+ new observations")
	}
}

func TestShouldSurfaceHighConfidence(t *testing.T) {
	ic := NewInsightCrystallizer()

	// Build up enough data for a high-confidence insight.
	for i := 0; i < 8; i++ {
		ic.Observe("career is on my mind", map[string]string{"topic": "career"}, "positive")
	}

	// Crystallize to populate the crystallized field.
	_ = ic.Crystallize()

	// Reset lastObsCount to simulate that we already surfaced.
	ic.mu.Lock()
	ic.lastObsCount = len(ic.observations)
	ic.mu.Unlock()

	// Even without 10 new obs, high-confidence insights should trigger.
	ic.Observe("career again", map[string]string{"topic": "career"}, "positive")
	if !ic.ShouldSurface(9) {
		t.Error("should surface when high-confidence insight exists")
	}
}

func TestSurfaceRelevant(t *testing.T) {
	ic := NewInsightCrystallizer()

	// Build a recurring theme.
	for i := 0; i < 5; i++ {
		ic.Observe("I keep thinking about writing", map[string]string{"topic": "writing"}, "positive")
	}
	ic.Observe("gardening is relaxing", map[string]string{"topic": "gardening"}, "positive")

	// Crystallize insights.
	_ = ic.Crystallize()

	// Query about writing should surface an insight.
	insight := ic.SurfaceRelevant("writing")
	if insight == nil {
		t.Fatal("expected a relevant insight for 'writing'")
	}
	if !strings.Contains(insight.Text, "writing") {
		t.Errorf("surfaced insight should mention writing; got %q", insight.Text)
	}

	// Query about an unrelated topic should return nil.
	unrelated := ic.SurfaceRelevant("astrophysics")
	if unrelated != nil {
		t.Errorf("expected nil for unrelated topic, got %q", unrelated.Text)
	}
}

func TestSurfaceRelevantEmpty(t *testing.T) {
	ic := NewInsightCrystallizer()

	// No observations, no insights.
	insight := ic.SurfaceRelevant("anything")
	if insight != nil {
		t.Error("expected nil insight with no observations")
	}

	// Empty topic should return nil.
	ic.Observe("test", map[string]string{"topic": "test"}, "neutral")
	insight = ic.SurfaceRelevant("")
	if insight != nil {
		t.Error("expected nil insight for empty topic")
	}
}

func TestInsightTextIsSpecific(t *testing.T) {
	ic := NewInsightCrystallizer()

	// Build varied observations.
	for i := 0; i < 4; i++ {
		ic.Observe("machine learning fascinates me", map[string]string{"topic": "machine learning"}, "positive")
	}
	for i := 0; i < 3; i++ {
		ic.Observe("I worry about climate change impacts", map[string]string{"topic": "climate change"}, "negative")
	}

	insights := ic.Crystallize()

	if len(insights) == 0 {
		t.Fatal("expected at least one insight")
	}

	for _, ins := range insights {
		// Every insight must reference at least one actual topic.
		containsTopic := strings.Contains(ins.Text, "machine learning") ||
			strings.Contains(ins.Text, "climate change")
		if !containsTopic {
			t.Errorf("insight text is too generic — does not reference actual topics: %q", ins.Text)
		}

		// Insights should be framed as observations.
		if strings.HasPrefix(ins.Text, "You should") {
			t.Errorf("insight should not be judgmental; got %q", ins.Text)
		}

		// Evidence should not be empty.
		if len(ins.Evidence) == 0 {
			t.Errorf("insight has no evidence: %q", ins.Text)
		}

		// Confidence should be in valid range.
		if ins.Confidence < 0 || ins.Confidence > 1 {
			t.Errorf("confidence = %f, want [0,1]", ins.Confidence)
		}
	}
}

func TestTrends(t *testing.T) {
	ic := NewInsightCrystallizer()

	base := time.Now().Add(-24 * time.Hour)

	// First half: mention "cooking" 1 time, "fitness" 3 times.
	ic.ObserveAt("cooking recipe", map[string]string{"topic": "cooking"}, "neutral", base)
	ic.ObserveAt("fitness routine", map[string]string{"topic": "fitness"}, "positive", base.Add(1*time.Minute))
	ic.ObserveAt("fitness goals", map[string]string{"topic": "fitness"}, "positive", base.Add(2*time.Minute))
	ic.ObserveAt("fitness plan", map[string]string{"topic": "fitness"}, "positive", base.Add(3*time.Minute))

	// Second half: mention "cooking" 3 times, "fitness" 1 time.
	ic.ObserveAt("cooking ideas", map[string]string{"topic": "cooking"}, "positive", base.Add(10*time.Minute))
	ic.ObserveAt("cooking class", map[string]string{"topic": "cooking"}, "positive", base.Add(11*time.Minute))
	ic.ObserveAt("cooking techniques", map[string]string{"topic": "cooking"}, "positive", base.Add(12*time.Minute))
	ic.ObserveAt("fitness check", map[string]string{"topic": "fitness"}, "neutral", base.Add(13*time.Minute))

	insights := ic.Crystallize()

	var trends []CrystallizedInsight
	for _, ins := range insights {
		if ins.Type == "trend" {
			trends = append(trends, ins)
		}
	}

	if len(trends) == 0 {
		t.Fatal("expected at least one trend insight")
	}

	// Cooking should be increasing, fitness decreasing.
	foundCookingInc := false
	foundFitnessDec := false
	for _, tr := range trends {
		if strings.Contains(tr.Text, "cooking") && strings.Contains(tr.Text, "more") {
			foundCookingInc = true
		}
		if strings.Contains(tr.Text, "fitness") && (strings.Contains(tr.Text, "shifting") || strings.Contains(tr.Text, "used to")) {
			foundFitnessDec = true
		}
	}

	if !foundCookingInc {
		t.Error("expected cooking to be detected as increasing trend")
	}
	if !foundFitnessDec {
		t.Error("expected fitness to be detected as decreasing trend")
	}
}

func TestBlindSpots(t *testing.T) {
	ic := NewInsightCrystallizer()

	// User asks about meditation 4 times but never uses action words.
	ic.Observe("what are the benefits of meditation?", map[string]string{"topic": "meditation"}, "neutral")
	ic.Observe("how does meditation affect the brain?", map[string]string{"topic": "meditation"}, "neutral")
	ic.Observe("is meditation effective for stress?", map[string]string{"topic": "meditation"}, "neutral")
	ic.Observe("meditation and focus connection", map[string]string{"topic": "meditation"}, "neutral")

	insights := ic.Crystallize()

	var blindSpots []CrystallizedInsight
	for _, ins := range insights {
		if ins.Type == "blind_spot" {
			blindSpots = append(blindSpots, ins)
		}
	}

	if len(blindSpots) == 0 {
		t.Fatal("expected at least one blind spot insight")
	}

	found := false
	for _, bs := range blindSpots {
		if strings.Contains(bs.Text, "meditation") {
			found = true
			if !strings.Contains(bs.Text, "next steps") {
				t.Errorf("blind spot should mention 'next steps'; got %q", bs.Text)
			}
			if !bs.Actionable {
				t.Error("blind spot should be actionable")
			}
			break
		}
	}
	if !found {
		t.Error("no blind spot detected for meditation")
	}
}

func TestBlindSpotNotTriggeredWithAction(t *testing.T) {
	ic := NewInsightCrystallizer()

	// User discusses "exercise" with action keywords.
	ic.Observe("I want to start exercising regularly", map[string]string{"topic": "exercise"}, "positive")
	ic.Observe("my plan for exercise this week", map[string]string{"topic": "exercise"}, "positive")
	ic.Observe("I'll begin exercise tomorrow morning", map[string]string{"topic": "exercise"}, "positive")

	insights := ic.Crystallize()

	for _, ins := range insights {
		if ins.Type == "blind_spot" && strings.Contains(ins.Text, "exercise") {
			t.Error("exercise should NOT be a blind spot — user used action words")
		}
	}
}

func TestSentimentNormalization(t *testing.T) {
	tests := []struct {
		input string
		want  string
	}{
		{"positive", "positive"},
		{"Positive", "positive"},
		{"NEGATIVE", "negative"},
		{"neg", "negative"},
		{"pos", "positive"},
		{"neutral", "neutral"},
		{"conflicted", "conflicted"},
		{"unknown", "neutral"},
		{"", "neutral"},
	}
	for _, tt := range tests {
		got := normalizeSentiment(tt.input)
		if got != tt.want {
			t.Errorf("normalizeSentiment(%q) = %q, want %q", tt.input, got, tt.want)
		}
	}
}

func TestKeywordExtraction(t *testing.T) {
	kws := extractInsightKeywords("Tell me about quantum physics and machine learning")
	if len(kws) == 0 {
		t.Fatal("expected keywords from query")
	}

	// Should include substantive words, not stop words.
	kwSet := make(map[string]bool)
	for _, k := range kws {
		kwSet[k] = true
	}

	if !kwSet["quantum"] {
		t.Error("expected 'quantum' in keywords")
	}
	if !kwSet["physics"] {
		t.Error("expected 'physics' in keywords")
	}
	if !kwSet["machine"] {
		t.Error("expected 'machine' in keywords")
	}
	if !kwSet["learning"] {
		t.Error("expected 'learning' in keywords")
	}

	// Stop words should be excluded.
	if kwSet["tell"] {
		t.Error("'tell' should be filtered out")
	}
	if kwSet["me"] {
		t.Error("'me' should be filtered out")
	}
	if kwSet["about"] {
		t.Error("'about' should be filtered out")
	}
	if kwSet["and"] {
		t.Error("'and' should be filtered out")
	}
}

func TestConcurrentObserve(t *testing.T) {
	ic := NewInsightCrystallizer()
	done := make(chan struct{})

	// Concurrent writes.
	for i := 0; i < 10; i++ {
		go func(n int) {
			defer func() { done <- struct{}{} }()
			for j := 0; j < 20; j++ {
				ic.Observe("concurrent test query", map[string]string{"topic": "concurrency"}, "neutral")
			}
		}(i)
	}

	for i := 0; i < 10; i++ {
		<-done
	}

	if ic.ObservationCount() != 200 {
		t.Errorf("observation count = %d, want 200", ic.ObservationCount())
	}
}
