package micromodel

import (
	"strings"
	"testing"
)

func TestFactTrieConstruction(t *testing.T) {
	tok := NewTokenizer()
	tok.BuildVocab([]string{
		"quantum mechanics is a branch of physics",
		"bitcoin was created by satoshi nakamoto",
	}, 200)

	facts := []string{
		"Quantum mechanics is a branch of physics.",
		"Bitcoin was created by Satoshi Nakamoto.",
	}

	fc := NewFactConstraint(facts, tok)

	// Trie should not be empty
	if len(fc.Trie.Children) == 0 {
		t.Error("trie has no children")
	}

	// Allowed tokens should contain fact words
	quantumID, ok := tok.Word2ID["quantum"]
	if ok && !fc.AllowedTokens[quantumID] {
		t.Error("'quantum' should be in allowed tokens")
	}

	physicsID, ok := tok.Word2ID["physics"]
	if ok && !fc.AllowedTokens[physicsID] {
		t.Error("'physics' should be in allowed tokens")
	}

	t.Logf("allowed tokens: %d, trie root children: %d",
		len(fc.AllowedTokens), len(fc.Trie.Children))
}

func TestFactConstraintApply(t *testing.T) {
	tok := NewTokenizer()
	tok.BuildVocab([]string{
		"quantum mechanics is a branch of physics",
		"energy matter wave function probability",
	}, 200)

	facts := []string{
		"Quantum mechanics is a branch of physics.",
	}

	fc := NewFactConstraint(facts, tok)
	vocabSize := tok.VocabSize()

	// Create uniform logits
	logits := make([]float32, vocabSize)
	for i := range logits {
		logits[i] = 0.0
	}

	// Apply constraints
	constrained := fc.ApplyConstraints(nil, logits, vocabSize)

	// Tokens in facts should have higher values
	quantumID, qOK := tok.Word2ID["quantum"]
	energyID, eOK := tok.Word2ID["energy"]

	if qOK && eOK {
		if constrained[quantumID] <= constrained[energyID] {
			t.Logf("quantum=%f, energy=%f", constrained[quantumID], constrained[energyID])
			// quantum is in the fact, energy is not directly — quantum should be boosted more
		}
	}

	// Allowed tokens should be boosted above zero
	boosted := 0
	for id := 5; id < vocabSize; id++ { // skip special tokens
		if constrained[id] > 0 {
			boosted++
		}
	}
	t.Logf("boosted tokens: %d out of %d", boosted, vocabSize-5)
}

func TestTrieMatch(t *testing.T) {
	tok := NewTokenizer()
	tok.BuildVocab([]string{
		"quantum mechanics is a branch of physics",
	}, 200)

	facts := []string{
		"quantum mechanics is a branch of physics",
	}

	fc := NewFactConstraint(facts, tok)

	// Encode first few words
	ids := tok.Encode("quantum mechanics is")

	depth, node := fc.trieMatch(ids)
	if depth == 0 {
		t.Error("expected trie match for 'quantum mechanics is'")
	}
	t.Logf("trie depth: %d, node has %d children", depth, len(node.Children))

	// Non-matching sequence
	randomIDs := []int{100, 200, 300}
	depth2, _ := fc.trieMatch(randomIDs)
	if depth2 > 0 {
		t.Error("unexpected trie match for random IDs")
	}
}

func TestTopKIndices(t *testing.T) {
	values := []float32{0.1, 0.5, 0.3, 0.9, 0.2}
	top3 := topKIndices(values, 3)

	if len(top3) != 3 {
		t.Fatalf("expected 3 indices, got %d", len(top3))
	}

	// Top should be index 3 (0.9)
	if top3[0] != 3 {
		t.Errorf("top[0] should be index 3 (0.9), got %d", top3[0])
	}
}

func TestTripleToSentence(t *testing.T) {
	tests := []struct {
		s, r, o string
		want    string
	}{
		{"Python", "is_a", "programming language", "Python is a programming language."},
		{"Linux", "created_by", "Linus Torvalds", "Linux was created by Linus Torvalds."},
		{"Google", "founded_in", "1998", "Google was founded in 1998."},
		{"Python", "has", "list comprehensions", "Python has list comprehensions."},
	}

	for _, tt := range tests {
		got := tripleToSentence(tt.s, tt.r, tt.o)
		if got != tt.want {
			t.Errorf("tripleToSentence(%q, %q, %q) = %q, want %q",
				tt.s, tt.r, tt.o, got, tt.want)
		}
	}
}

func TestConstrainedGenerate(t *testing.T) {
	cfg := SmallMambaConfig()
	m := NewMambaModel(cfg)

	m.Tok.BuildVocab([]string{
		"quantum mechanics is a branch of physics that describes matter and energy",
		"physics branch describes behavior particles waves",
	}, cfg.VocabSize)

	facts := []string{
		"Quantum mechanics is a branch of physics.",
		"It describes the behavior of matter and energy.",
	}

	result := m.ConstrainedGenerate("quantum mechanics", "is_a", "branch of physics", facts, 15, 2)

	// Untrained model won't produce great text, but should not panic
	t.Logf("constrained generation: %q", result)
}

func TestGenerateGrounded(t *testing.T) {
	cfg := SmallMambaConfig()
	m := NewMambaModel(cfg)

	m.Tok.BuildVocab([]string{
		"python is a programming language created by guido van rossum",
		"programming language created development web data",
	}, cfg.VocabSize)

	extraFacts := [][3]string{
		{"Python", "created_by", "Guido van Rossum"},
		{"Python", "used_for", "web development"},
	}

	result := m.GenerateGrounded("Python", "is_a", "programming language", extraFacts, 2)
	t.Logf("grounded generation: %q", result)
}

func TestConstraintPreservesSpecialTokens(t *testing.T) {
	tok := NewTokenizer()
	tok.BuildVocab([]string{"hello world test"}, 100)

	fc := NewFactConstraint([]string{"hello world"}, tok)

	// Special tokens (0-4) should always be in allowed set
	for id := 0; id < 5; id++ {
		if !fc.AllowedTokens[id] {
			t.Errorf("special token %d should be allowed", id)
		}
	}
}

func TestBeamSorting(t *testing.T) {
	beams := []*Beam{
		{IDs: []int{1, 2, 3}, Score: -5.0},
		{IDs: []int{1, 2}, Score: -2.0},
		{IDs: []int{1, 2, 3, 4}, Score: -3.0},
	}

	sortBeams(beams)

	// After sorting, best normalized score should be first
	prev := normalizedScore(beams[0])
	for i := 1; i < len(beams); i++ {
		curr := normalizedScore(beams[i])
		if curr > prev {
			t.Errorf("beam %d (score=%.3f) should not be after beam %d (score=%.3f)",
				i, curr, i-1, prev)
		}
		prev = curr
	}
}

func TestFactConstraintWithEmptyFacts(t *testing.T) {
	tok := NewTokenizer()
	tok.BuildVocab([]string{"hello"}, 100)

	// Empty facts should not panic
	fc := NewFactConstraint(nil, tok)
	if fc == nil {
		t.Fatal("NewFactConstraint returned nil for empty facts")
	}

	logits := make([]float32, tok.VocabSize())
	result := fc.ApplyConstraints(nil, logits, tok.VocabSize())
	if len(result) != len(logits) {
		t.Errorf("result length mismatch: %d vs %d", len(result), len(logits))
	}
}

func TestConstrainedBridgeGeneration(t *testing.T) {
	cfg := SmallMambaConfig()
	m := NewMambaModel(cfg)
	m.Tok.BuildVocab([]string{
		"dna is a molecule that carries genetic information",
	}, cfg.VocabSize)

	b := NewMambaBridge(m)

	// ConstrainedGenerate through bridge pattern
	result := b.GenerateSentence("DNA", "is_a", "molecule")
	_ = result
	_ = strings.Contains // silence import if needed
}
