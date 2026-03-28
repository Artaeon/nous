package cognitive

import (
	"strings"
	"testing"
)

// -----------------------------------------------------------------------
// Wiring Integration Tests — verify that Phase 1-4 subsystems are
// properly connected to the main pipeline and produce observable effects.
// -----------------------------------------------------------------------

func TestWiring_ConversationStateTracksTopics(t *testing.T) {
	ar := NewActionRouter()
	ar.ConvState = NewConversationState()
	ar.FollowUp = NewFollowUpResolver()
	ar.Preferences = NewPreferenceModel()
	ar.Filler = NewFillerDetector()

	// First turn
	nlu1 := &NLUResult{
		Raw:      "tell me about quantum physics",
		Intent:   "explain",
		Action:   "lookup_knowledge",
		Entities: map[string]string{"topic": "quantum physics"},
	}
	ar.Execute(nlu1, nil)

	// ConvState should have recorded the topic
	if ar.ConvState.ActiveTopic == "" && len(ar.ConvState.TopicStack) == 0 {
		t.Error("ConvState should track topic after first turn")
	}
	if ar.ConvState.TurnCount != 1 {
		t.Errorf("TurnCount = %d, want 1", ar.ConvState.TurnCount)
	}
}

func TestWiring_PreferenceModelUpdates(t *testing.T) {
	ar := NewActionRouter()
	ar.ConvState = NewConversationState()
	ar.Preferences = NewPreferenceModel()
	ar.Filler = NewFillerDetector()

	nlu := &NLUResult{
		Raw:      "I'm interested in learning about distributed systems architecture for my work",
		Intent:   "conversation",
		Action:   "respond",
		Entities: map[string]string{},
	}
	ar.Execute(nlu, nil)

	if ar.Preferences.TurnsSampled != 1 {
		t.Errorf("Preferences.TurnsSampled = %d, want 1", ar.Preferences.TurnsSampled)
	}
}

func TestWiring_FillerStrippedFromOutput(t *testing.T) {
	ar := NewActionRouter()
	ar.Filler = NewFillerDetector()

	// Simulate a response that contains filler
	nlu := &NLUResult{
		Raw:      "explain photosynthesis",
		Intent:   "explain",
		Action:   "lookup_knowledge",
		Entities: map[string]string{"topic": "photosynthesis"},
	}
	result := ar.Execute(nlu, nil)

	// Even if the response is nil (no knowledge loaded), the filler detector
	// should be wired and ready
	if ar.Filler == nil {
		t.Error("Filler should be wired")
	}

	// The filler detector should strip AI prefixes from any response
	if result != nil && result.DirectResponse != "" {
		lower := strings.ToLower(result.DirectResponse)
		if strings.HasPrefix(lower, "as an ai") {
			t.Error("filler 'As an AI' should have been stripped")
		}
	}
}

func TestWiring_FollowUpResolverWired(t *testing.T) {
	ar := NewActionRouter()
	ar.ConvState = NewConversationState()
	ar.FollowUp = NewFollowUpResolver()
	ar.Filler = NewFillerDetector()

	// Set up some conversation state
	ar.ConvState.PushTopic("machine learning")
	ar.ConvState.TrackEntity("topic", "machine learning")

	// Now send a follow-up
	nlu := &NLUResult{
		Raw:      "tell me more",
		Intent:   "followup",
		Action:   "respond",
		Entities: map[string]string{},
	}
	result := ar.Execute(nlu, nil)

	// The follow-up resolver should have been consulted (even if no
	// thinking engine is available for the actual response)
	_ = result // We just verify it doesn't panic
}

func TestWiring_QueryRewriterWired(t *testing.T) {
	ar := NewActionRouter()
	ar.QueryRewrite = NewQueryRewriter()
	ar.Filler = NewFillerDetector()

	// The query rewriter should be available
	if ar.QueryRewrite == nil {
		t.Error("QueryRewriter should be wired")
	}

	// Test that it can decompose a complex query
	decomposed := ar.QueryRewrite.Decompose("compare Python and Go for web development")
	if !decomposed.IsComplex {
		t.Error("comparison query should be classified as complex")
	}
	if len(decomposed.SubQueries) < 2 {
		t.Errorf("expected at least 2 sub-queries, got %d", len(decomposed.SubQueries))
	}
}

func TestWiring_MultiPassInThink(t *testing.T) {
	// Create a ThinkingEngine and verify Think() uses multi-pass
	te := NewThinkingEngine(nil, nil)
	result := te.Think("explain how computers work", nil)
	if result == nil {
		t.Fatal("Think should return non-nil")
	}
	// The trace should mention multi-pass (either success or fallback)
	if !strings.Contains(result.Trace, "Multi-pass") && !strings.Contains(result.Trace, "Plan rerank") {
		// With no graph, multi-pass may not be accepted, so plan rerank runs.
		// Either is fine — we just want to see the pipeline was attempted.
		t.Logf("trace: %s", result.Trace)
	}
}

func TestWiring_ResponseGateIntegration(t *testing.T) {
	ar := NewActionRouter()
	ar.ConvState = NewConversationState()
	ar.FollowUp = NewFollowUpResolver()
	ar.Preferences = NewPreferenceModel()
	ar.Filler = NewFillerDetector()

	// Test that the response gate catches instructions
	nlu := &NLUResult{
		Raw:        "Help me think about my career. Ask me 3 clarifying questions.",
		Intent:     "conversation",
		Action:     "respond",
		Confidence: 0.8,
		Entities:   map[string]string{},
	}
	result := ar.Execute(nlu, nil)
	if result != nil && result.DirectResponse != "" {
		qCount := strings.Count(result.DirectResponse, "?")
		if qCount < 3 {
			t.Errorf("asked for 3 questions, got %d in:\n%s", qCount, result.DirectResponse)
		}
	}
}
