package cognitive

import (
	"testing"
)

const fluencyKnowledgeDir = "../../knowledge"

func setupFluencyScorer(t *testing.T) *FluencyScorer {
	t.Helper()
	fs := NewFluencyScorer()
	if err := fs.LoadCorpus(fluencyKnowledgeDir); err != nil {
		t.Fatalf("LoadCorpus failed: %v", err)
	}
	return fs
}

func TestFluencyScorer_LoadCorpus(t *testing.T) {
	fs := setupFluencyScorer(t)

	if !fs.CorpusLoaded() {
		t.Fatal("expected corpus to be loaded")
	}
	if fs.VocabSize() == 0 {
		t.Fatal("expected vocabulary to be populated, got 0")
	}

	t.Logf("Loaded corpus: %d vocab words", fs.VocabSize())
}

func TestFluencyScorer_Score(t *testing.T) {
	fs := setupFluencyScorer(t)

	// A natural sentence should score higher than garbled text.
	natural := "The theory of evolution describes how species change over time through natural selection."
	garbled := "Evolution banana quickly the however from seventeen purple."

	naturalScore := fs.Score(natural)
	garbledScore := fs.Score(garbled)

	t.Logf("natural score: %.4f", naturalScore)
	t.Logf("garbled score: %.4f", garbledScore)

	if naturalScore <= garbledScore {
		t.Errorf("natural text (%.4f) should score higher than garbled (%.4f)", naturalScore, garbledScore)
	}

	// Score should be in [0, 1] range.
	if naturalScore < 0 || naturalScore > 1 {
		t.Errorf("natural score %.4f out of [0,1] range", naturalScore)
	}
	if garbledScore < 0 || garbledScore > 1 {
		t.Errorf("garbled score %.4f out of [0,1] range", garbledScore)
	}
}

func TestFluencyScorer_Score_EdgeCases(t *testing.T) {
	fs := setupFluencyScorer(t)

	// Single word should return 0 (no pairs).
	if s := fs.Score("hello"); s != 0.0 {
		t.Errorf("single word score = %.4f, want 0.0", s)
	}

	// Empty string should return 0.
	if s := fs.Score(""); s != 0.0 {
		t.Errorf("empty string score = %.4f, want 0.0", s)
	}
}

func TestFluencyScorer_ScoreBest(t *testing.T) {
	fs := setupFluencyScorer(t)

	candidates := []string{
		"Purple the banana from quickly however seventeen evolution.",
		"The development of modern science has transformed our understanding of the natural world.",
		"Zzz qqq xxx bbb nnn mmm ppp.",
	}

	idx, score := fs.ScoreBest(candidates)

	t.Logf("Best candidate: index=%d, score=%.4f", idx, score)

	if idx != 1 {
		t.Errorf("expected index 1 (natural sentence), got %d", idx)
	}
	if score <= 0 {
		t.Errorf("best score should be positive, got %.4f", score)
	}
}

func TestFluencyScorer_ScoreBest_Empty(t *testing.T) {
	fs := NewFluencyScorer()

	idx, score := fs.ScoreBest(nil)
	if idx != -1 || score != 0 {
		t.Errorf("empty candidates: idx=%d score=%.4f, want -1, 0", idx, score)
	}
}

func TestFluencyScorer_SuggestNextWord(t *testing.T) {
	fs := setupFluencyScorer(t)

	// "the" is extremely common; it should prefer frequent followers.
	candidates := []string{"of", "xyznotaword", "qqq"}
	best := fs.SuggestNextWord("the", candidates)

	t.Logf("SuggestNextWord('the', %v) = %q", candidates, best)

	// "of" is a very common follower of "the" in English text.
	if best != "of" {
		t.Errorf("expected 'of' after 'the', got %q", best)
	}
}

func TestFluencyScorer_SuggestNextWord_NoCandidates(t *testing.T) {
	fs := setupFluencyScorer(t)

	result := fs.SuggestNextWord("the", nil)
	if result != "" {
		t.Errorf("expected empty string for nil candidates, got %q", result)
	}
}

func TestFluencyScorer_NoCorpus(t *testing.T) {
	fs := NewFluencyScorer()

	// Should not panic, just return 0 scores.
	score := fs.Score("This is a test sentence.")
	if score < 0 || score > 1 {
		t.Errorf("score without corpus = %.4f, should be in [0,1]", score)
	}
}
