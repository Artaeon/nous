package cognitive

import (
	"strings"
	"testing"
	"time"

	"github.com/artaeon/nous/internal/memory"
)

func buildCouncilTestGraph() *CognitiveGraph {
	cg := NewCognitiveGraph("")
	cg.AddEdge("Go", "programming language", RelIsA, "test")
	cg.AddEdge("Go", "Google", RelCreatedBy, "test")
	cg.AddEdge("Go", "concurrency", RelHas, "test")
	cg.AddEdge("Go", "static typing", RelHas, "test")
	cg.AddEdge("Rust", "programming language", RelIsA, "test")
	cg.AddEdge("Rust", "memory safety", RelHas, "test")
	return cg
}

func buildCouncilEpisodic() *memory.EpisodicMemory {
	em := memory.NewEpisodicMemory("", nil)
	em.Record(memory.Episode{
		Timestamp: time.Now().Add(-14 * 24 * time.Hour),
		Input:     "tell me about Go",
		Intent:    "question",
		Output:    "Go is a programming language created by Google.",
		Success:   true,
		Tags:      []string{"go", "programming"},
	})
	em.Record(memory.Episode{
		Timestamp: time.Now().Add(-7 * 24 * time.Hour),
		Input:     "how does Go handle concurrency",
		Intent:    "question",
		Output:    "Go uses goroutines and channels.",
		Success:   true,
		Tags:      []string{"go", "concurrency"},
	})
	em.Record(memory.Episode{
		Timestamp: time.Now().Add(-1 * 24 * time.Hour),
		Input:     "Go concurrency patterns",
		Intent:    "question",
		Output:    "Common patterns include fan-out/fan-in...",
		Success:   false,
		Tags:      []string{"go", "concurrency"},
	})
	return em
}

// -----------------------------------------------------------------------
// Pragmatist tests
// -----------------------------------------------------------------------

func TestPragmatist_WithFacts(t *testing.T) {
	ic := NewInnerCouncil(buildCouncilTestGraph(), nil)
	op := ic.consultPragmatist("tell me about Go", []string{"Go"})

	if op.Perspective != PerspPragmatist {
		t.Fatalf("expected PerspPragmatist, got %d", op.Perspective)
	}
	if op.Confidence < 0.4 {
		t.Errorf("expected moderate confidence with facts, got %.2f", op.Confidence)
	}
	if !strings.Contains(op.Assessment, "facts") {
		t.Errorf("expected assessment to mention facts, got: %s", op.Assessment)
	}
}

func TestPragmatist_NoFacts(t *testing.T) {
	ic := NewInnerCouncil(buildCouncilTestGraph(), nil)
	op := ic.consultPragmatist("tell me about quantum computing", []string{"quantum computing"})

	if op.Confidence > 0.3 {
		t.Errorf("expected low confidence without facts, got %.2f", op.Confidence)
	}
}

func TestPragmatist_NilGraph(t *testing.T) {
	ic := NewInnerCouncil(nil, nil)
	op := ic.consultPragmatist("anything", nil)

	if op.Confidence > 0.2 {
		t.Errorf("expected very low confidence with nil graph, got %.2f", op.Confidence)
	}
}

// -----------------------------------------------------------------------
// Historian tests
// -----------------------------------------------------------------------

func TestHistorian_WithEpisodes(t *testing.T) {
	ic := NewInnerCouncil(nil, buildCouncilEpisodic())
	op := ic.consultHistorian("tell me about Go")

	if op.Perspective != PerspHistorian {
		t.Fatalf("expected PerspHistorian, got %d", op.Perspective)
	}
	if op.Confidence < 0.3 {
		t.Errorf("expected some confidence with history, got %.2f", op.Confidence)
	}
	if strings.Contains(op.Assessment, "No prior") {
		t.Errorf("expected to find prior interactions, got: %s", op.Assessment)
	}
}

func TestHistorian_NilEpisodic(t *testing.T) {
	ic := NewInnerCouncil(nil, nil)
	op := ic.consultHistorian("anything")

	if op.Confidence > 0.2 {
		t.Errorf("expected very low confidence with nil episodic, got %.2f", op.Confidence)
	}
	if !strings.Contains(op.Assessment, "No episodic") {
		t.Errorf("expected nil episodic message, got: %s", op.Assessment)
	}
}

func TestHistorian_NoMatches(t *testing.T) {
	ic := NewInnerCouncil(nil, buildCouncilEpisodic())
	op := ic.consultHistorian("kubernetes deployment")

	if !strings.Contains(op.Assessment, "No prior") {
		t.Errorf("expected no prior interactions, got: %s", op.Assessment)
	}
}

// -----------------------------------------------------------------------
// Empath tests
// -----------------------------------------------------------------------

func TestEmpath_WithSubtext(t *testing.T) {
	subtext := &SubtextAnalysis{
		EmotionalState: EmotionalState{
			Valence:  -0.6,
			Arousal:  0.7,
			Dominant: "frustrated",
		},
		ImpliedNeed: NeedVenting,
		Signals: []BehavioralSignal{
			{Type: "venting", Evidence: "so frustrating", Weight: 0.7},
		},
		Confidence: 0.8,
	}

	ic := NewInnerCouncil(nil, nil)
	op := ic.consultEmpath("this stupid bug is driving me crazy", subtext)

	if op.Perspective != PerspEmpath {
		t.Fatalf("expected PerspEmpath, got %d", op.Perspective)
	}
	if op.Priority < 0.5 {
		t.Errorf("expected high priority for frustrated user, got %.2f", op.Priority)
	}
	if !strings.Contains(op.KeyInsight, "venting") {
		t.Errorf("expected venting insight, got: %s", op.KeyInsight)
	}
}

func TestEmpath_NilSubtext(t *testing.T) {
	ic := NewInnerCouncil(nil, nil)
	op := ic.consultEmpath("anything", nil)

	if op.Priority > 0.2 {
		t.Errorf("expected low priority with nil subtext, got %.2f", op.Priority)
	}
}

func TestEmpath_NeutralEmotion(t *testing.T) {
	subtext := &SubtextAnalysis{
		EmotionalState: EmotionalState{
			Valence:  0.0,
			Arousal:  0.1,
			Dominant: "neutral",
		},
		ImpliedNeed: NeedInformation,
		Confidence:  0.4,
	}

	ic := NewInnerCouncil(nil, nil)
	op := ic.consultEmpath("what is Go", subtext)

	if op.Priority > 0.3 {
		t.Errorf("expected low priority for neutral emotion, got %.2f", op.Priority)
	}
}

// -----------------------------------------------------------------------
// Architect tests
// -----------------------------------------------------------------------

func TestArchitect_WithSparks(t *testing.T) {
	sparks := []AssociativeSpark{
		{
			Source:      "Go",
			Target:      "Rust",
			Novelty:     0.7,
			Explanation: "Both are systems languages with different memory models.",
		},
	}

	ic := NewInnerCouncil(nil, nil)
	op := ic.consultArchitect("Go vs Rust", []string{"Go", "Rust"}, sparks)

	if op.Perspective != PerspArchitect {
		t.Fatalf("expected PerspArchitect, got %d", op.Perspective)
	}
	if !strings.Contains(op.Assessment, "Go") || !strings.Contains(op.Assessment, "Rust") {
		t.Errorf("expected spark subjects in assessment, got: %s", op.Assessment)
	}
}

func TestArchitect_NoTopicsNoSparks(t *testing.T) {
	ic := NewInnerCouncil(nil, nil)
	op := ic.consultArchitect("hi", nil, nil)

	if op.Priority > 0.2 {
		t.Errorf("expected low priority with no topics, got %.2f", op.Priority)
	}
}

// -----------------------------------------------------------------------
// Skeptic tests
// -----------------------------------------------------------------------

func TestSkeptic_AmbiguousInput(t *testing.T) {
	others := []CouncilOpinion{
		{Perspective: PerspPragmatist, Confidence: 0.2, Priority: 0.3},
		{Perspective: PerspHistorian, Confidence: 0.1, Priority: 0.2},
		{Perspective: PerspEmpath, Confidence: 0.3, Priority: 0.2},
		{Perspective: PerspArchitect, Confidence: 0.2, Priority: 0.1},
	}

	ic := NewInnerCouncil(nil, nil)
	op := ic.consultSkeptic("it", others)

	if !strings.Contains(op.Assessment, "intent unclear") && !strings.Contains(op.Assessment, "vague pronoun") {
		t.Errorf("expected ambiguity concern for short vague input, got: %s", op.Assessment)
	}
	if op.Priority < 0.3 {
		t.Errorf("expected non-trivial priority for ambiguous input, got %.2f", op.Priority)
	}
}

func TestSkeptic_NoConcerns(t *testing.T) {
	others := []CouncilOpinion{
		{Perspective: PerspPragmatist, Confidence: 0.6, Priority: 0.5},
		{Perspective: PerspHistorian, Confidence: 0.5, Priority: 0.3},
		{Perspective: PerspEmpath, Confidence: 0.4, Priority: 0.2},
		{Perspective: PerspArchitect, Confidence: 0.4, Priority: 0.2},
	}

	ic := NewInnerCouncil(nil, nil)
	op := ic.consultSkeptic("tell me about Go programming language and its concurrency model", others)

	if op.Priority > 0.2 {
		t.Errorf("expected low priority when no concerns, got %.2f", op.Priority)
	}
	if !strings.Contains(op.Assessment, "No significant concerns") {
		t.Errorf("expected all-clear, got: %s", op.Assessment)
	}
}

// -----------------------------------------------------------------------
// Arbiter tests
// -----------------------------------------------------------------------

func TestArbiter_EmpathWinsWhenEmotional(t *testing.T) {
	subtext := &SubtextAnalysis{
		EmotionalState: EmotionalState{
			Valence:  -0.6,
			Arousal:  0.7,
			Dominant: "frustrated",
		},
		ImpliedNeed: NeedVenting,
		Confidence:  0.8,
	}

	opinions := []CouncilOpinion{
		{Perspective: PerspPragmatist, Confidence: 0.6, Priority: 0.5, Assessment: "Facts available.", KeyInsight: "Lead with facts."},
		{Perspective: PerspHistorian, Confidence: 0.3, Priority: 0.2, Assessment: "Some history.", KeyInsight: "Prior interactions found."},
		{Perspective: PerspEmpath, Confidence: 0.8, Priority: 0.7, Assessment: "User is frustrated.", KeyInsight: "Listen first."},
		{Perspective: PerspArchitect, Confidence: 0.3, Priority: 0.2, Assessment: "Single topic.", KeyInsight: "No systemic view."},
		{Perspective: PerspSkeptic, Confidence: 0.2, Priority: 0.1, Assessment: "No concerns.", KeyInsight: "All clear."},
	}

	ic := NewInnerCouncil(nil, nil)
	delib := ic.arbitrate(opinions, subtext)

	if delib.Dominant != PerspEmpath {
		t.Errorf("expected Empath to dominate, got %s", perspectiveName(delib.Dominant))
	}
	if delib.ResponseTone != "empathetic" {
		t.Errorf("expected empathetic tone, got %s", delib.ResponseTone)
	}
}

func TestArbiter_PragmatistWinsWhenFactual(t *testing.T) {
	opinions := []CouncilOpinion{
		{Perspective: PerspPragmatist, Confidence: 0.7, Priority: 0.7, Assessment: "5 facts about Go.", KeyInsight: "Strong grounding."},
		{Perspective: PerspHistorian, Confidence: 0.3, Priority: 0.2, Assessment: "Fresh topic.", KeyInsight: "No history."},
		{Perspective: PerspEmpath, Confidence: 0.3, Priority: 0.2, Assessment: "Neutral emotion.", KeyInsight: "Calm mode."},
		{Perspective: PerspArchitect, Confidence: 0.3, Priority: 0.2, Assessment: "Single topic.", KeyInsight: "Straightforward."},
		{Perspective: PerspSkeptic, Confidence: 0.2, Priority: 0.1, Assessment: "No concerns.", KeyInsight: "All clear."},
	}

	ic := NewInnerCouncil(nil, nil)
	delib := ic.arbitrate(opinions, nil)

	if delib.Dominant != PerspPragmatist {
		t.Errorf("expected Pragmatist to dominate, got %s", perspectiveName(delib.Dominant))
	}
	if delib.ResponseTone != "direct" {
		t.Errorf("expected direct tone, got %s", delib.ResponseTone)
	}
}

func TestArbiter_SkepticTriggersAsk(t *testing.T) {
	opinions := []CouncilOpinion{
		{Perspective: PerspPragmatist, Confidence: 0.2, Priority: 0.3, Assessment: "No facts.", KeyInsight: "Knowledge gap."},
		{Perspective: PerspHistorian, Confidence: 0.1, Priority: 0.1, Assessment: "No history.", KeyInsight: "Fresh."},
		{Perspective: PerspEmpath, Confidence: 0.2, Priority: 0.1, Assessment: "Neutral.", KeyInsight: "Calm."},
		{Perspective: PerspArchitect, Confidence: 0.2, Priority: 0.1, Assessment: "Isolated.", KeyInsight: "No connections."},
		{Perspective: PerspSkeptic, Confidence: 0.6, Priority: 0.5, Assessment: "very short input with no question mark — intent unclear; no factual grounding and no history.", KeyInsight: "intent unclear"},
	}

	ic := NewInnerCouncil(nil, nil)
	delib := ic.arbitrate(opinions, nil)

	if !delib.ShouldAsk {
		t.Error("expected ShouldAsk=true when skeptic flags ambiguity")
	}
	if delib.AskWhat == "" {
		t.Error("expected AskWhat to be populated")
	}
}

func TestArbiter_HistorianRepeatedStruggle(t *testing.T) {
	opinions := []CouncilOpinion{
		{Perspective: PerspPragmatist, Confidence: 0.4, Priority: 0.4, Assessment: "Some facts.", KeyInsight: "Partial grounding."},
		{Perspective: PerspHistorian, Confidence: 0.7, Priority: 0.8, Assessment: "Asked 3 times, mostly failed.", KeyInsight: "Repeated struggle. Try a different approach."},
		{Perspective: PerspEmpath, Confidence: 0.3, Priority: 0.2, Assessment: "Neutral.", KeyInsight: "Calm."},
		{Perspective: PerspArchitect, Confidence: 0.3, Priority: 0.2, Assessment: "Single topic.", KeyInsight: "Simple."},
		{Perspective: PerspSkeptic, Confidence: 0.3, Priority: 0.2, Assessment: "No concerns.", KeyInsight: "All clear."},
	}

	ic := NewInnerCouncil(nil, nil)
	delib := ic.arbitrate(opinions, nil)

	if delib.Dominant != PerspHistorian {
		t.Errorf("expected Historian to dominate for repeated struggle, got %s", perspectiveName(delib.Dominant))
	}
	if delib.ResponseTone != "cautious" {
		t.Errorf("expected cautious tone, got %s", delib.ResponseTone)
	}
}

func TestArbiter_EnthusiasticTone(t *testing.T) {
	subtext := &SubtextAnalysis{
		EmotionalState: EmotionalState{
			Valence:  0.7,
			Arousal:  0.6,
			Dominant: "excited",
		},
		ImpliedNeed: NeedCelebration,
		Confidence:  0.7,
	}

	opinions := []CouncilOpinion{
		{Perspective: PerspPragmatist, Confidence: 0.5, Priority: 0.4, Assessment: "Facts available.", KeyInsight: "Grounded."},
		{Perspective: PerspHistorian, Confidence: 0.3, Priority: 0.2, Assessment: "Some history.", KeyInsight: "Known topic."},
		{Perspective: PerspEmpath, Confidence: 0.7, Priority: 0.6, Assessment: "User is excited.", KeyInsight: "Celebrate with them."},
		{Perspective: PerspArchitect, Confidence: 0.3, Priority: 0.2, Assessment: "Single topic.", KeyInsight: "Simple."},
		{Perspective: PerspSkeptic, Confidence: 0.2, Priority: 0.1, Assessment: "No concerns.", KeyInsight: "All clear."},
	}

	ic := NewInnerCouncil(nil, nil)
	delib := ic.arbitrate(opinions, subtext)

	if delib.ResponseTone != "enthusiastic" {
		t.Errorf("expected enthusiastic tone for positive+aroused, got %s", delib.ResponseTone)
	}
}

// -----------------------------------------------------------------------
// Full deliberation integration
// -----------------------------------------------------------------------

func TestDeliberate_FullIntegration(t *testing.T) {
	graph := buildCouncilTestGraph()
	episodic := buildCouncilEpisodic()
	ic := NewInnerCouncil(graph, episodic)

	nlu := &NLUResult{
		Intent:   "question",
		Entities: map[string]string{"topic": "Go"},
		Raw:      "tell me about Go",
	}

	ctx := &ComposeContext{
		UserName:     "Raphael",
		TimeOfDay:    time.Now(),
		RecentTopics: []string{"Go"},
		Subtext: &SubtextAnalysis{
			EmotionalState: EmotionalState{
				Valence:  0.1,
				Arousal:  0.2,
				Dominant: "neutral",
			},
			ImpliedNeed: NeedInformation,
			Confidence:  0.5,
		},
	}

	delib := ic.Deliberate("tell me about Go", nlu, ctx)

	if delib == nil {
		t.Fatal("expected non-nil deliberation")
	}
	if len(delib.Opinions) != 5 {
		t.Errorf("expected 5 opinions, got %d", len(delib.Opinions))
	}
	if delib.Synthesis == "" {
		t.Error("expected non-empty synthesis")
	}
	if delib.ResponseTone == "" {
		t.Error("expected non-empty response tone")
	}
	if delib.Trace == "" {
		t.Error("expected non-empty trace")
	}

	// With good facts and neutral emotion, pragmatist should lead.
	if delib.Dominant != PerspPragmatist {
		t.Errorf("expected Pragmatist for factual neutral query, got %s", perspectiveName(delib.Dominant))
	}
}

func TestDeliberate_NilSubtext(t *testing.T) {
	graph := buildCouncilTestGraph()
	ic := NewInnerCouncil(graph, nil)

	nlu := &NLUResult{
		Intent:   "question",
		Entities: map[string]string{"topic": "Go"},
		Raw:      "tell me about Go",
	}

	ctx := &ComposeContext{
		RecentTopics: []string{"Go"},
		// No Subtext field — nil
	}

	delib := ic.Deliberate("tell me about Go", nlu, ctx)

	if delib == nil {
		t.Fatal("expected non-nil deliberation")
	}
	// Empath should have low priority with nil subtext.
	for _, op := range delib.Opinions {
		if op.Perspective == PerspEmpath && op.Priority > 0.2 {
			t.Errorf("expected low empath priority with nil subtext, got %.2f", op.Priority)
		}
	}
}

func TestDeliberate_NilEpisodicMemory(t *testing.T) {
	ic := NewInnerCouncil(buildCouncilTestGraph(), nil)

	delib := ic.Deliberate("tell me about Go", nil, nil)

	if delib == nil {
		t.Fatal("expected non-nil deliberation")
	}
	// Historian should have very low confidence.
	for _, op := range delib.Opinions {
		if op.Perspective == PerspHistorian && op.Confidence > 0.2 {
			t.Errorf("expected low historian confidence with nil episodic, got %.2f", op.Confidence)
		}
	}
}

func TestDeliberate_NilContext(t *testing.T) {
	ic := NewInnerCouncil(buildCouncilTestGraph(), nil)
	delib := ic.Deliberate("hello", nil, nil)

	if delib == nil {
		t.Fatal("expected non-nil deliberation with nil context")
	}
	if len(delib.Opinions) != 5 {
		t.Errorf("expected 5 opinions, got %d", len(delib.Opinions))
	}
}

// -----------------------------------------------------------------------
// Response tone selection
// -----------------------------------------------------------------------

func TestResponseTone_DirectForFactual(t *testing.T) {
	ic := NewInnerCouncil(buildCouncilTestGraph(), nil)
	ctx := &ComposeContext{RecentTopics: []string{"Go"}}
	nlu := &NLUResult{Entities: map[string]string{"topic": "Go"}}

	delib := ic.Deliberate("what is Go", nlu, ctx)
	if delib.ResponseTone != "direct" {
		t.Errorf("expected direct tone for factual query, got %s", delib.ResponseTone)
	}
}

func TestResponseTone_EmpatheticForFrustrated(t *testing.T) {
	ic := NewInnerCouncil(nil, nil)
	ctx := &ComposeContext{
		Subtext: &SubtextAnalysis{
			EmotionalState: EmotionalState{
				Valence:  -0.7,
				Arousal:  0.8,
				Dominant: "frustrated",
			},
			ImpliedNeed: NeedVenting,
			Signals: []BehavioralSignal{
				{Type: "venting", Weight: 0.8},
			},
			Confidence: 0.8,
		},
	}

	delib := ic.Deliberate("this stupid thing never works", nil, ctx)
	if delib.ResponseTone != "empathetic" {
		t.Errorf("expected empathetic tone for frustrated user, got %s", delib.ResponseTone)
	}
}

// -----------------------------------------------------------------------
// Trace generation
// -----------------------------------------------------------------------

func TestTrace_ContainsReasoningSteps(t *testing.T) {
	ic := NewInnerCouncil(buildCouncilTestGraph(), nil)
	ctx := &ComposeContext{RecentTopics: []string{"Go"}}
	nlu := &NLUResult{Entities: map[string]string{"topic": "Go"}}

	delib := ic.Deliberate("tell me about Go", nlu, ctx)

	if delib.Trace == "" {
		t.Error("expected non-empty trace")
	}
	// Trace should mention which perspective leads.
	if !strings.Contains(delib.Trace, "leads") && !strings.Contains(delib.Trace, "fallback") {
		t.Errorf("expected trace to record dominant perspective, got: %s", delib.Trace)
	}
}

// -----------------------------------------------------------------------
// perspectiveName
// -----------------------------------------------------------------------

func TestPerspectiveName(t *testing.T) {
	tests := []struct {
		p    Perspective
		want string
	}{
		{PerspPragmatist, "Pragmatist"},
		{PerspHistorian, "Historian"},
		{PerspEmpath, "Empath"},
		{PerspArchitect, "Architect"},
		{PerspSkeptic, "Skeptic"},
		{Perspective(99), "Unknown"},
	}

	for _, tt := range tests {
		got := perspectiveName(tt.p)
		if got != tt.want {
			t.Errorf("perspectiveName(%d) = %q, want %q", tt.p, got, tt.want)
		}
	}
}
