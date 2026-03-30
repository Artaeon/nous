package micromodel

import (
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestTokenizer_EncodeDecode(t *testing.T) {
	tok := NewTokenizer()
	tok.BuildVocab([]string{
		"Quantum mechanics is a branch of physics.",
		"Bitcoin was created by Satoshi Nakamoto.",
		"Python is a programming language.",
	}, 100)

	text := "quantum mechanics is a branch of physics"
	ids := tok.Encode(text)
	if len(ids) == 0 {
		t.Fatal("Encode returned empty")
	}

	decoded := tok.Decode(ids)
	if !strings.Contains(strings.ToLower(decoded), "quantum") {
		t.Errorf("Decode should contain 'quantum', got %q", decoded)
	}
	t.Logf("Encoded %q → %v → %q", text, ids, decoded)
}

func TestTokenizer_EncodeTriple(t *testing.T) {
	tok := NewTokenizer()
	tok.BuildVocab([]string{"quantum mechanics", "is_a", "branch of physics"}, 50)

	ids := tok.EncodeTriple("quantum mechanics", "is_a", "branch of physics")
	if len(ids) < 5 { // at least bos + 1 + sep + 1 + sep + 1 + eos
		t.Errorf("EncodeTriple too short: %v", ids)
	}
	if ids[0] != BosID {
		t.Errorf("first token should be BOS, got %d", ids[0])
	}
	if ids[len(ids)-1] != EosID {
		t.Errorf("last token should be EOS, got %d", ids[len(ids)-1])
	}
	t.Logf("Triple encoded: %v", ids)
}

func TestTokenizer_SpecialTokens(t *testing.T) {
	tok := NewTokenizer()
	if tok.VocabSize() != 5 {
		t.Errorf("empty tokenizer should have 5 special tokens, got %d", tok.VocabSize())
	}
	if tok.Word2ID["<pad>"] != PadID {
		t.Error("pad ID mismatch")
	}
	if tok.Word2ID["<bos>"] != BosID {
		t.Error("bos ID mismatch")
	}
}

func TestAttention_Basic(t *testing.T) {
	// Test that attention produces output of correct shape
	dim := 8
	numHeads := 2
	seqLen := 3

	Q := make([]float32, seqLen*dim)
	K := make([]float32, seqLen*dim)
	V := make([]float32, seqLen*dim)
	Wo := make([]float32, dim*dim)

	// Initialize with small random values
	for i := range Q {
		Q[i] = float32(i) * 0.1
		K[i] = float32(i) * 0.1
		V[i] = float32(i) * 0.1
	}
	for i := range Wo {
		Wo[i] = 0.1
	}

	out := MultiHeadAttention(Q, K, V, seqLen, seqLen, dim, numHeads, Wo, false)
	if len(out) != seqLen*dim {
		t.Errorf("attention output should be %d, got %d", seqLen*dim, len(out))
	}

	// Check no NaN
	for i, v := range out {
		if math.IsNaN(float64(v)) {
			t.Fatalf("NaN at position %d", i)
		}
	}
}

func TestAttention_CausalMask(t *testing.T) {
	// With causal masking, first position should only attend to itself
	dim := 4
	numHeads := 1
	seqLen := 3

	Q := make([]float32, seqLen*dim)
	K := make([]float32, seqLen*dim)
	V := make([]float32, seqLen*dim)
	Wo := make([]float32, dim*dim)

	for i := range V {
		V[i] = float32(i)
	}
	for i := range Wo {
		Wo[i] = 0.1
	}
	// Q and K set so all positions attend equally (without mask)
	for i := range Q {
		Q[i] = 1.0
		K[i] = 1.0
	}

	causal := MultiHeadAttention(Q, K, V, seqLen, seqLen, dim, numHeads, Wo, true)
	noncausal := MultiHeadAttention(Q, K, V, seqLen, seqLen, dim, numHeads, Wo, false)

	// Results should differ (causal prevents attending to future)
	same := true
	for i := range causal {
		if causal[i] != noncausal[i] {
			same = false
			break
		}
	}
	if same {
		t.Error("causal and non-causal attention should produce different results")
	}
}

func TestForward_Shape(t *testing.T) {
	cfg := SmallConfig()
	cfg.VocabSize = 50
	m := NewMicroModel(cfg)

	// Build a tiny vocab
	m.Tok.BuildVocab([]string{"hello world foo bar baz"}, 50)
	m.Config.VocabSize = m.Tok.VocabSize()

	// Reinitialize with correct vocab size
	m = NewMicroModel(m.Config)
	m.Tok.BuildVocab([]string{"hello world foo bar baz"}, 50)

	encIDs := []int{BosID, 5, 6, EosID}
	decIDs := []int{BosID, 5}

	probs := m.Forward(encIDs, decIDs)
	expectedLen := len(decIDs) * m.Config.VocabSize
	if len(probs) != expectedLen {
		t.Fatalf("Forward output should be %d, got %d", expectedLen, len(probs))
	}

	// Check probabilities sum to ~1 for each position
	for pos := 0; pos < len(decIDs); pos++ {
		var sum float32
		for v := 0; v < m.Config.VocabSize; v++ {
			p := probs[pos*m.Config.VocabSize+v]
			if math.IsNaN(float64(p)) {
				t.Fatalf("NaN in probs at pos=%d, v=%d", pos, v)
			}
			sum += p
		}
		if math.Abs(float64(sum-1.0)) > 0.01 {
			t.Errorf("probs at pos %d sum to %f, want ~1.0", pos, sum)
		}
	}
}

func TestGenerate_ProducesTokens(t *testing.T) {
	texts := []string{
		"quantum mechanics is a branch of physics",
		"bitcoin was created by satoshi nakamoto",
		"python is a programming language",
		"the sun is a star in our solar system",
	}
	cfg := SmallConfig()
	cfg.VocabSize = 100

	tok := NewTokenizer()
	tok.BuildVocab(texts, 100)
	cfg.VocabSize = tok.VocabSize()

	m := NewMicroModel(cfg)
	m.Tok = tok

	result := m.Generate("quantum mechanics", "is_a", "branch of physics", 20, 0.8)
	t.Logf("Generated (untrained): %q", result)

	// Should produce SOMETHING (even if nonsensical before training)
	if len(result) == 0 {
		t.Error("Generate should produce non-empty output")
	}
}

func TestTraining_LossDecreases(t *testing.T) {
	cfg := SmallConfig()
	cfg.VocabSize = 100
	m := NewMicroModel(cfg)

	examples := []TrainingExample{
		{Input: "python <sep> is_a <sep> programming language", Target: "Python is a programming language."},
		{Input: "bitcoin <sep> created_by <sep> satoshi nakamoto", Target: "Bitcoin was created by Satoshi Nakamoto."},
		{Input: "google <sep> founded_in <sep> 1998", Target: "Google was founded in 1998."},
		{Input: "dna <sep> is_a <sep> molecule", Target: "DNA is a molecule."},
	}

	// Build vocab from examples
	var texts []string
	for _, ex := range examples {
		texts = append(texts, ex.Input, ex.Target)
	}
	m.Tok.BuildVocab(texts, 100)
	m.Config.VocabSize = m.Tok.VocabSize()
	m = NewMicroModel(m.Config) // reinitialize with correct vocab
	m.Tok.BuildVocab(texts, 100)

	// Compute initial loss
	var initialLoss float32
	for _, ex := range examples {
		encIDs := m.Tok.Encode(ex.Input)
		decInput := append([]int{BosID}, m.Tok.Encode(ex.Target)...)
		decTarget := append(m.Tok.Encode(ex.Target), EosID)
		for len(decTarget) < len(decInput) {
			decTarget = append(decTarget, PadID)
		}
		probs := m.Forward(encIDs, decInput)
		initialLoss += crossEntropyLoss(probs, m.Config.VocabSize, decTarget)
	}
	initialLoss /= float32(len(examples))

	// Train for a few epochs
	result := m.Train(examples, 5, 0.01)

	t.Logf("Initial loss: %.4f", initialLoss)
	t.Logf("Final loss: %.4f (after %d epochs, %s)", result.FinalLoss, result.Epochs, result.Duration)

	if result.FinalLoss >= float64(initialLoss) {
		t.Errorf("loss should decrease: initial=%.4f, final=%.4f", initialLoss, result.FinalLoss)
	}
}

func TestSaveLoad_RoundTrip(t *testing.T) {
	cfg := SmallConfig()
	cfg.VocabSize = 50
	m := NewMicroModel(cfg)
	m.Tok.BuildVocab([]string{"hello world test"}, 50)

	dir := t.TempDir()
	path := filepath.Join(dir, "model.bin")

	if err := m.Save(path); err != nil {
		t.Fatalf("Save: %v", err)
	}

	info, _ := os.Stat(path)
	t.Logf("Model file: %d bytes", info.Size())

	m2 := &MicroModel{}
	if err := m2.Load(path); err != nil {
		t.Fatalf("Load: %v", err)
	}

	if m2.Config.EmbedDim != m.Config.EmbedDim {
		t.Errorf("EmbedDim mismatch: %d vs %d", m2.Config.EmbedDim, m.Config.EmbedDim)
	}
	if m2.Config.NumLayers != m.Config.NumLayers {
		t.Errorf("NumLayers mismatch: %d vs %d", m2.Config.NumLayers, m.Config.NumLayers)
	}
	if len(m2.TokenEmbed) != len(m.TokenEmbed) {
		t.Errorf("TokenEmbed length mismatch: %d vs %d", len(m2.TokenEmbed), len(m.TokenEmbed))
	}
	if m2.Tok.VocabSize() != m.Tok.VocabSize() {
		t.Errorf("VocabSize mismatch: %d vs %d", m2.Tok.VocabSize(), m.Tok.VocabSize())
	}

	// Weights should be identical
	for i := range m.TokenEmbed {
		if m.TokenEmbed[i] != m2.TokenEmbed[i] {
			t.Errorf("TokenEmbed[%d] mismatch: %f vs %f", i, m.TokenEmbed[i], m2.TokenEmbed[i])
			break
		}
	}
}

func TestParamCount(t *testing.T) {
	cfg := DefaultConfig()
	m := NewMicroModel(cfg)
	count := m.ParamCount()
	t.Logf("Default config: %d parameters (%.1f MB at float32)", count, float64(count)*4/1024/1024)

	small := SmallConfig()
	ms := NewMicroModel(small)
	t.Logf("Small config: %d parameters (%.1f KB at float32)", ms.ParamCount(), float64(ms.ParamCount())*4/1024)
}

func TestTemplatePairs(t *testing.T) {
	pairs := templatePairs()
	if len(pairs) < 50 {
		t.Errorf("expected at least 50 template pairs, got %d", len(pairs))
	}
	t.Logf("Generated %d template training pairs", len(pairs))

	// Check a few
	for _, p := range pairs[:3] {
		t.Logf("  Input: %s", p.Input)
		t.Logf("  Target: %s", p.Target)
	}
}
