package cognitive

import (
	"os"
	"path/filepath"
	"testing"
	"time"
)

// -----------------------------------------------------------------------
// Opinion Formation Engine — tests
// -----------------------------------------------------------------------

func newTestOpinionEngine(t *testing.T) *OpinionEngine {
	t.Helper()
	dir := t.TempDir()
	return NewOpinionEngine(filepath.Join(dir, "opinions.json"))
}

func TestAccumulateEvidencePositive(t *testing.T) {
	oe := newTestOpinionEngine(t)

	oe.AccumulateEvidence("Go", Evidence{
		Type:    EvidenceConversation,
		Source:  "test",
		Content: "Go is fast and reliable",
		Valence: 0.8,
		Weight:  0.7,
	})

	op := oe.GetOpinion("Go")
	if op == nil {
		t.Fatal("expected opinion, got nil")
	}
	if op.Stance <= 0 {
		t.Errorf("expected positive stance, got %f", op.Stance)
	}
	if op.ForCount != 1 {
		t.Errorf("expected ForCount=1, got %d", op.ForCount)
	}
}

func TestAccumulateEvidenceNegative(t *testing.T) {
	oe := newTestOpinionEngine(t)

	oe.AccumulateEvidence("PHP", Evidence{
		Type:    EvidenceConversation,
		Source:  "test",
		Content: "PHP has inconsistent APIs",
		Valence: -0.7,
		Weight:  0.6,
	})

	op := oe.GetOpinion("PHP")
	if op == nil {
		t.Fatal("expected opinion, got nil")
	}
	if op.Stance >= 0 {
		t.Errorf("expected negative stance, got %f", op.Stance)
	}
	if op.AgainstCount != 1 {
		t.Errorf("expected AgainstCount=1, got %d", op.AgainstCount)
	}
}

func TestAccumulateEvidenceMixed(t *testing.T) {
	oe := newTestOpinionEngine(t)

	oe.AccumulateEvidence("JavaScript", Evidence{
		Type: EvidenceConversation, Source: "test",
		Content: "JS has a huge ecosystem", Valence: 0.7, Weight: 0.6,
	})
	oe.AccumulateEvidence("JavaScript", Evidence{
		Type: EvidenceConversation, Source: "test",
		Content: "JS type coercion is confusing", Valence: -0.7, Weight: 0.6,
	})
	oe.AccumulateEvidence("JavaScript", Evidence{
		Type: EvidenceConversation, Source: "test",
		Content: "async/await is nice", Valence: 0.5, Weight: 0.5,
	})
	oe.AccumulateEvidence("JavaScript", Evidence{
		Type: EvidenceConversation, Source: "test",
		Content: "callback hell was awful", Valence: -0.6, Weight: 0.5,
	})

	op := oe.GetOpinion("JavaScript")
	if op == nil {
		t.Fatal("expected opinion, got nil")
	}
	t.Logf("JavaScript: stance=%.2f confidence=%.2f position=%s", op.Stance, op.Confidence, op.Position)

	// Stance should be near zero with balanced evidence.
	if op.Stance > 0.5 || op.Stance < -0.5 {
		t.Errorf("expected stance near zero for mixed evidence, got %f", op.Stance)
	}
	if op.ForCount == 0 || op.AgainstCount == 0 {
		t.Errorf("expected both for and against counts >0, got for=%d against=%d", op.ForCount, op.AgainstCount)
	}
}

func TestStanceComputation(t *testing.T) {
	oe := newTestOpinionEngine(t)

	// Add three positive, one negative — stance should be positive.
	for _, v := range []float64{0.8, 0.7, 0.6, -0.3} {
		oe.AccumulateEvidence("Rust", Evidence{
			Type: EvidenceConversation, Source: "test",
			Content: "test", Valence: v, Weight: 0.5,
		})
	}

	op := oe.GetOpinion("Rust")
	if op == nil {
		t.Fatal("expected opinion, got nil")
	}
	if op.Stance <= 0.2 {
		t.Errorf("expected positive stance > 0.2, got %f", op.Stance)
	}
}

func TestConfidenceGrowth(t *testing.T) {
	oe := newTestOpinionEngine(t)

	// One piece of evidence — confidence should be low.
	oe.AccumulateEvidence("Docker", Evidence{
		Type: EvidenceConversation, Source: "test",
		Content: "Docker simplifies deployment", Valence: 0.7, Weight: 0.6,
	})
	op1 := oe.GetOpinion("Docker")
	if op1 == nil {
		t.Fatal("expected opinion")
	}
	if op1.Confidence > 0.3 {
		t.Errorf("single evidence should have confidence <= 0.3, got %f", op1.Confidence)
	}

	// Add more evidence — confidence should grow.
	for i := 0; i < 5; i++ {
		oe.AccumulateEvidence("Docker", Evidence{
			Type: EvidenceConversation, Source: "test",
			Content: "Docker is useful", Valence: 0.6, Weight: 0.7,
		})
	}
	op2 := oe.GetOpinion("Docker")
	if op2.Confidence <= op1.Confidence {
		t.Errorf("confidence should grow with evidence: was %f, now %f", op1.Confidence, op2.Confidence)
	}
	t.Logf("Docker confidence: 1 ev=%.2f, 6 ev=%.2f", op1.Confidence, op2.Confidence)
}

func TestPositionClassification(t *testing.T) {
	tests := []struct {
		name    string
		valences []float64
		want    string
	}{
		{"strongly positive", []float64{0.8, 0.7, 0.9}, "positive"},
		{"strongly negative", []float64{-0.8, -0.7, -0.9}, "negative"},
		{"mixed", []float64{0.8, -0.7, 0.6, -0.8}, "mixed"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			oe := newTestOpinionEngine(t)
			for _, v := range tt.valences {
				oe.AccumulateEvidence("topic", Evidence{
					Type: EvidenceConversation, Source: "test",
					Content: "test", Valence: v, Weight: 0.6,
				})
			}
			op := oe.GetOpinion("topic")
			if op == nil {
				t.Fatal("expected opinion")
			}
			if op.Position != tt.want {
				t.Errorf("expected position %q, got %q (stance=%.2f, for=%d, against=%d)",
					tt.want, op.Position, op.Stance, op.ForCount, op.AgainstCount)
			}
		})
	}
}

func TestPositionUncertain(t *testing.T) {
	oe := newTestOpinionEngine(t)
	// Single weak evidence — should be uncertain.
	oe.AccumulateEvidence("obscure-thing", Evidence{
		Type: EvidenceObservation, Source: "test",
		Content: "mentioned once", Valence: 0.1, Weight: 0.2,
	})
	op := oe.GetOpinion("obscure-thing")
	if op == nil {
		t.Fatal("expected opinion")
	}
	if op.Position != "uncertain" {
		t.Errorf("expected uncertain, got %q (confidence=%.2f)", op.Position, op.Confidence)
	}
}

func TestLearnFromConversationGoIsGreat(t *testing.T) {
	oe := newTestOpinionEngine(t)
	oe.LearnFromConversation("Go is great", nil)

	op := oe.GetOpinion("Go")
	if op == nil {
		t.Fatal("expected opinion on Go")
	}
	if op.Stance <= 0 {
		t.Errorf("expected positive stance for 'Go is great', got %f", op.Stance)
	}
	t.Logf("Go: stance=%.2f position=%s summary=%q", op.Stance, op.Position, op.Summary)
}

func TestLearnFromConversationIHateBugs(t *testing.T) {
	oe := newTestOpinionEngine(t)
	oe.LearnFromConversation("I hate bugs", nil)

	op := oe.GetOpinion("bugs")
	if op == nil {
		t.Fatal("expected opinion on bugs")
	}
	if op.Stance >= 0 {
		t.Errorf("expected negative stance for 'I hate bugs', got %f", op.Stance)
	}
	t.Logf("bugs: stance=%.2f position=%s", op.Stance, op.Position)
}

func TestLearnFromConversationWorksWell(t *testing.T) {
	oe := newTestOpinionEngine(t)
	oe.LearnFromConversation("Python works well", nil)

	op := oe.GetOpinion("python")
	if op == nil {
		t.Fatal("expected opinion on python")
	}
	if op.Stance <= 0 {
		t.Errorf("expected positive stance for 'Python works well', got %f", op.Stance)
	}
}

func TestLearnFromConversationComparative(t *testing.T) {
	oe := newTestOpinionEngine(t)
	oe.LearnFromConversation("Go is better than Java", nil)

	opGo := oe.GetOpinion("Go")
	opJava := oe.GetOpinion("Java")

	if opGo == nil {
		t.Fatal("expected opinion on Go")
	}
	if opJava == nil {
		t.Fatal("expected opinion on Java")
	}
	if opGo.Stance <= opJava.Stance {
		t.Errorf("Go stance (%.2f) should be higher than Java stance (%.2f)", opGo.Stance, opJava.Stance)
	}
}

func TestArticulateOpinionVariety(t *testing.T) {
	tests := []struct {
		name     string
		position string
		stance   float64
		conf     float64
		nev      int
		wantSub  string // substring that should appear
	}{
		{"high-conf positive", "positive", 0.8, 0.8, 12, "genuinely good"},
		{"low-conf positive", "positive", 0.4, 0.4, 3, "seems solid"},
		{"high-conf negative", "negative", -0.8, 0.8, 10, "has real issues"},
		{"low-conf negative", "negative", -0.4, 0.4, 3, "leaning negative"},
		{"uncertain", "uncertain", 0.0, 0.1, 1, "don't have a strong opinion"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			op := &Opinion{
				Topic:      "testing",
				Position:   tt.position,
				Stance:     tt.stance,
				Confidence: tt.conf,
			}
			for i := 0; i < tt.nev; i++ {
				op.Evidence = append(op.Evidence, Evidence{
					Type: EvidenceConversation, Source: "test",
					Content: "test evidence " + string(rune('a'+i)),
					Valence: tt.stance, Weight: 0.5,
				})
			}
			summary := ArticulateOpinion(op)
			t.Logf("%s: %q", tt.name, summary)
			if summary == "" {
				t.Error("expected non-empty summary")
			}
			if tt.wantSub != "" && !containsLower(summary, tt.wantSub) {
				t.Errorf("expected summary to contain %q, got %q", tt.wantSub, summary)
			}
		})
	}
}

func TestArticulateMixed(t *testing.T) {
	op := &Opinion{
		Topic:    "frameworks",
		Position: "mixed",
		Stance:   0.05,
		Confidence: 0.5,
		Evidence: []Evidence{
			{Type: EvidenceConversation, Content: "saves time", Valence: 0.7, Weight: 0.6},
			{Type: EvidenceConversation, Content: "adds complexity", Valence: -0.6, Weight: 0.6},
		},
	}
	summary := ArticulateOpinion(op)
	t.Logf("mixed: %q", summary)
	if !containsLower(summary, "mixed bag") {
		t.Errorf("expected 'mixed bag' in summary, got %q", summary)
	}
}

func TestChallengeOpinionShiftsStance(t *testing.T) {
	oe := newTestOpinionEngine(t)

	// Build a positive opinion.
	for i := 0; i < 4; i++ {
		oe.AccumulateEvidence("Kubernetes", Evidence{
			Type: EvidenceConversation, Source: "test",
			Content: "K8s is powerful", Valence: 0.8, Weight: 0.7,
		})
	}

	before := oe.GetOpinion("Kubernetes")
	if before == nil {
		t.Fatal("expected opinion")
	}
	stanceBefore := before.Stance

	// Challenge it.
	oe.ChallengeOpinion("Kubernetes", "Kubernetes is overcomplicated for small teams")

	after := oe.GetOpinion("Kubernetes")
	if after.Stance >= stanceBefore {
		t.Errorf("expected stance to decrease after challenge: before=%.2f after=%.2f", stanceBefore, after.Stance)
	}
	t.Logf("K8s stance: before=%.2f after=%.2f", stanceBefore, after.Stance)
}

func TestEvidenceDecay(t *testing.T) {
	oe := newTestOpinionEngine(t)

	old := time.Now().Add(-90 * 24 * time.Hour) // 3 months ago
	oe.AccumulateEvidence("OldTech", Evidence{
		Type: EvidenceConversation, Source: "test",
		Content: "was good back then", Valence: 0.8, Weight: 0.8,
		Timestamp: old,
	})

	before := oe.GetOpinion("OldTech")
	weightBefore := before.Evidence[0].Weight

	oe.DecayOldEvidence()

	after := oe.GetOpinion("OldTech")
	weightAfter := after.Evidence[0].Weight

	if weightAfter >= weightBefore {
		t.Errorf("expected weight to decrease after decay: before=%.3f after=%.3f", weightBefore, weightAfter)
	}
	t.Logf("Weight decay: %.3f → %.3f", weightBefore, weightAfter)
}

func TestEvidenceDecayVeryOld(t *testing.T) {
	oe := newTestOpinionEngine(t)

	veryOld := time.Now().Add(-200 * 24 * time.Hour) // ~7 months
	oe.AccumulateEvidence("AncientTech", Evidence{
		Type: EvidenceConversation, Source: "test",
		Content: "ancient evidence", Valence: 0.8, Weight: 1.0,
		Timestamp: veryOld,
	})

	oe.DecayOldEvidence()

	op := oe.GetOpinion("AncientTech")
	if op.Evidence[0].Weight > 0.55 {
		t.Errorf("very old evidence should be halved: got weight=%.3f", op.Evidence[0].Weight)
	}
}

func TestTopOpinionsRanking(t *testing.T) {
	oe := newTestOpinionEngine(t)

	// Create opinions with different confidence levels.
	topics := []struct {
		name string
		n    int // number of evidence pieces
	}{
		{"low", 1},
		{"medium", 4},
		{"high", 8},
	}

	for _, tt := range topics {
		for i := 0; i < tt.n; i++ {
			oe.AccumulateEvidence(tt.name, Evidence{
				Type: EvidenceConversation, Source: "test",
				Content: "evidence", Valence: 0.6, Weight: 0.6,
			})
		}
	}

	top := oe.TopOpinions(2)
	if len(top) != 2 {
		t.Fatalf("expected 2 top opinions, got %d", len(top))
	}
	// First should be highest confidence.
	if top[0].Confidence < top[1].Confidence {
		t.Errorf("top opinions not sorted by confidence: %f < %f", top[0].Confidence, top[1].Confidence)
	}
	t.Logf("Top 2: %s (%.2f), %s (%.2f)", top[0].Topic, top[0].Confidence, top[1].Topic, top[1].Confidence)
}

func TestSaveLoadRoundTrip(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "opinions.json")

	oe := NewOpinionEngine(path)
	oe.AccumulateEvidence("Go", Evidence{
		Type: EvidenceConversation, Source: "test",
		Content: "Go is great", Valence: 0.8, Weight: 0.7,
	})
	oe.AccumulateEvidence("Python", Evidence{
		Type: EvidenceConversation, Source: "test",
		Content: "Python is versatile", Valence: 0.6, Weight: 0.5,
	})

	if err := oe.Save(); err != nil {
		t.Fatalf("Save failed: %v", err)
	}

	// Verify file exists.
	if _, err := os.Stat(path); err != nil {
		t.Fatalf("save file not found: %v", err)
	}

	// Load into a new engine.
	oe2 := NewOpinionEngine(path)
	opGo := oe2.GetOpinion("Go")
	if opGo == nil {
		t.Fatal("expected Go opinion after load")
	}
	if opGo.Stance <= 0 {
		t.Errorf("expected positive stance after load, got %f", opGo.Stance)
	}

	opPy := oe2.GetOpinion("Python")
	if opPy == nil {
		t.Fatal("expected Python opinion after load")
	}
	t.Logf("Loaded: Go stance=%.2f, Python stance=%.2f", opGo.Stance, opPy.Stance)
}

func TestTopicNormalization(t *testing.T) {
	oe := newTestOpinionEngine(t)

	oe.AccumulateEvidence("  Go  ", Evidence{
		Type: EvidenceConversation, Source: "test",
		Content: "test", Valence: 0.5, Weight: 0.5,
	})

	// Should find with different casing and spacing.
	if op := oe.GetOpinion("go"); op == nil {
		t.Error("expected to find opinion with lowercase")
	}
	if op := oe.GetOpinion(" GO "); op == nil {
		t.Error("expected to find opinion with uppercase + spaces")
	}
}

func TestTopicNormalizationPlurals(t *testing.T) {
	oe := newTestOpinionEngine(t)

	oe.AccumulateEvidence("containers", Evidence{
		Type: EvidenceConversation, Source: "test",
		Content: "test", Valence: 0.5, Weight: 0.5,
	})

	// "containers" should normalize to "container".
	if op := oe.GetOpinion("container"); op == nil {
		t.Error("expected plural normalization: 'containers' → 'container'")
	}
}

func TestEvidencePruning(t *testing.T) {
	oe := newTestOpinionEngine(t)

	// Add more than maxEvidencePerOpinion pieces.
	for i := 0; i < 30; i++ {
		oe.AccumulateEvidence("crowded", Evidence{
			Type: EvidenceConversation, Source: "test",
			Content: "evidence piece",
			Valence: float64(i%2)*2 - 1, // alternating +1 / -1
			Weight:  float64(i) / 30.0,
		})
	}

	op := oe.GetOpinion("crowded")
	if op == nil {
		t.Fatal("expected opinion")
	}
	if len(op.Evidence) > maxEvidencePerOpinion {
		t.Errorf("evidence should be capped at %d, got %d", maxEvidencePerOpinion, len(op.Evidence))
	}
}

func TestFormCountIncrement(t *testing.T) {
	oe := newTestOpinionEngine(t)

	for i := 0; i < 3; i++ {
		oe.AccumulateEvidence("topic", Evidence{
			Type: EvidenceConversation, Source: "test",
			Content: "evidence", Valence: 0.5, Weight: 0.5,
		})
	}

	op := oe.GetOpinion("topic")
	if op.FormCount != 3 {
		t.Errorf("expected FormCount=3, got %d", op.FormCount)
	}
}

func TestGetOpinionReturnsNilForUnknown(t *testing.T) {
	oe := newTestOpinionEngine(t)
	if op := oe.GetOpinion("nonexistent"); op != nil {
		t.Errorf("expected nil for unknown topic, got %+v", op)
	}
}

