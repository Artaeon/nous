package micromodel

import (
	"math"
	"os"
	"path/filepath"
	"testing"
)

func TestSelectiveScan(t *testing.T) {
	seqLen := 4
	dInner := 2
	dState := 2

	x := []float32{1, 0, 0, 1, 1, 1, 0.5, 0.5}
	dt := []float32{0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1}
	A := []float32{-1, -2, -1, -2} // (dInner=2, dState=2)
	B := []float32{1, 1, 0, 1, 1, 0, 1, 1}
	C := []float32{1, 0, 0, 1, 1, 1, 0.5, 0.5}
	D := []float32{1, 1}

	y := SelectiveScan(x, seqLen, dInner, dt, A, dState, B, C, D)

	if len(y) != seqLen*dInner {
		t.Fatalf("expected output length %d, got %d", seqLen*dInner, len(y))
	}

	// Basic sanity: output should not be all zeros
	var sum float32
	for _, v := range y {
		sum += v * v
	}
	if sum < 1e-6 {
		t.Error("output is all zeros")
	}
}

func TestSelectiveScanBackward(t *testing.T) {
	seqLen := 3
	dInner := 2
	dState := 2

	x := []float32{1, 0.5, 0.3, 0.8, 0.5, 0.2}
	dt := []float32{0.1, 0.1, 0.1, 0.1, 0.1, 0.1}
	A := []float32{-1, -1, -1, -1}
	B := []float32{1, 0.5, 0.5, 1, 0.8, 0.3}
	C := []float32{0.5, 1, 1, 0.5, 0.7, 0.7}
	D := []float32{1, 1}
	dy := []float32{1, 1, 1, 1, 1, 1}

	dx, ddt, dA, dB, dC, dD := SelectiveScanBackward(dy, x, seqLen, dInner, dt, A, dState, B, C, D)

	// Check shapes
	if len(dx) != seqLen*dInner {
		t.Errorf("dx length: expected %d, got %d", seqLen*dInner, len(dx))
	}
	if len(ddt) != seqLen*dInner {
		t.Errorf("ddt length: expected %d, got %d", seqLen*dInner, len(ddt))
	}
	if len(dA) != dInner*dState {
		t.Errorf("dA length: expected %d, got %d", dInner*dState, len(dA))
	}
	if len(dB) != seqLen*dState {
		t.Errorf("dB length: expected %d, got %d", seqLen*dState, len(dB))
	}
	if len(dC) != seqLen*dState {
		t.Errorf("dC length: expected %d, got %d", seqLen*dState, len(dC))
	}
	if len(dD) != dInner {
		t.Errorf("dD length: expected %d, got %d", dInner, len(dD))
	}

	// Gradients should not all be zero
	var sumDx float32
	for _, v := range dx {
		sumDx += v * v
	}
	if sumDx < 1e-10 {
		t.Error("dx gradients are all zero")
	}
}

func TestSiLU(t *testing.T) {
	x := []float32{-2, -1, 0, 1, 2}
	y := silu(x)

	// SiLU(0) = 0
	if math.Abs(float64(y[2])) > 1e-6 {
		t.Errorf("silu(0) = %f, expected 0", y[2])
	}

	// SiLU is approximately identity for large x
	if y[4] < 1.5 {
		t.Errorf("silu(2) = %f, expected ~1.76", y[4])
	}

	// SiLU(-x) is small
	if y[0] > -0.1 {
		t.Errorf("silu(-2) = %f, expected ~-0.27", y[0])
	}
}

func TestConv1D(t *testing.T) {
	seqLen := 4
	dInner := 2
	kSize := 3

	x := []float32{
		1, 0,
		0, 1,
		1, 1,
		0, 0,
	}
	weight := []float32{
		1, 0.5, 0.25, // channel 0 kernel
		1, 0.5, 0.25, // channel 1 kernel
	}
	bias := []float32{0, 0}

	y := conv1D(x, seqLen, dInner, weight, bias, kSize)

	// Position 0: only k=0 contributes (causal, no history)
	// y[0,0] = x[0,0] * w[0,0] = 1 * 1 = 1
	if math.Abs(float64(y[0]-1.0)) > 1e-6 {
		t.Errorf("conv1d[0,0] = %f, expected 1.0", y[0])
	}

	// Position 1: k=0 and k=1 contribute
	// y[1,0] = x[1,0]*w[0,0] + x[0,0]*w[0,1] = 0*1 + 1*0.5 = 0.5
	if math.Abs(float64(y[2]-0.5)) > 1e-6 {
		t.Errorf("conv1d[1,0] = %f, expected 0.5", y[2])
	}
}

func TestConv1DBackward(t *testing.T) {
	seqLen := 3
	dInner := 2
	kSize := 2

	x := []float32{1, 0, 0, 1, 1, 1}
	weight := []float32{1, 0.5, 1, 0.5}
	dy := []float32{1, 1, 1, 1, 1, 1}

	dx, dw, db := conv1DBackward(dy, x, seqLen, dInner, weight, kSize)

	if len(dx) != seqLen*dInner {
		t.Errorf("dx length: expected %d, got %d", seqLen*dInner, len(dx))
	}
	if len(dw) != dInner*kSize {
		t.Errorf("dw length: expected %d, got %d", dInner*kSize, len(dw))
	}
	if len(db) != dInner {
		t.Errorf("db length: expected %d, got %d", dInner, len(db))
	}

	// Bias gradient should equal sum of dy per channel
	// db[0] = dy[0,0] + dy[1,0] + dy[2,0] = 3
	if math.Abs(float64(db[0]-3.0)) > 1e-6 {
		t.Errorf("db[0] = %f, expected 3.0", db[0])
	}
}

func TestMambaModelCreateAndParamCount(t *testing.T) {
	cfg := SmallMambaConfig()
	m := NewMambaModel(cfg)

	if m == nil {
		t.Fatal("NewMambaModel returned nil")
	}

	params := m.ParamCount()
	if params <= 0 {
		t.Fatalf("param count should be positive, got %d", params)
	}
	t.Logf("Small Mamba model: %d parameters", params)

	// Check layer count
	if len(m.Layers) != cfg.NumLayers {
		t.Errorf("expected %d layers, got %d", cfg.NumLayers, len(m.Layers))
	}

	// Check embedding dimensions
	if len(m.TokenEmbed) != cfg.VocabSize*cfg.ModelDim {
		t.Errorf("token embed size: expected %d, got %d",
			cfg.VocabSize*cfg.ModelDim, len(m.TokenEmbed))
	}
}

func TestMambaForward(t *testing.T) {
	cfg := SmallMambaConfig()
	m := NewMambaModel(cfg)

	inputIDs := []int{BosID, 5, 6, SepID, 7, SepID, 8, EosID}
	logits := m.Forward(inputIDs)

	expectedLen := len(inputIDs) * cfg.VocabSize
	if len(logits) != expectedLen {
		t.Fatalf("logits length: expected %d, got %d", expectedLen, len(logits))
	}

	// Logits should not be all zeros or NaN
	var hasNonZero bool
	for _, v := range logits {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatal("logits contain NaN or Inf")
		}
		if v != 0 {
			hasNonZero = true
		}
	}
	if !hasNonZero {
		t.Error("all logits are zero")
	}
}

func TestMambaStatefulGeneration(t *testing.T) {
	cfg := SmallMambaConfig()
	m := NewMambaModel(cfg)

	// Build minimal vocab
	m.Tok.BuildVocab([]string{
		"quantum mechanics is a branch of physics",
		"bitcoin was created by satoshi nakamoto",
	}, cfg.VocabSize)

	state := m.NewMambaState()

	// Process a few tokens
	for _, id := range []int{BosID, 5, SepID, 7, SepID} {
		logits := m.StepForward(id, state)
		if len(logits) != cfg.VocabSize {
			t.Fatalf("step logits: expected %d, got %d", cfg.VocabSize, len(logits))
		}
	}

	// State should be populated
	for i, h := range state.H {
		var sum float32
		for _, v := range h {
			sum += v * v
		}
		if sum < 1e-10 {
			t.Errorf("layer %d state is all zeros after processing tokens", i)
		}
	}
}

func TestMambaGenerate(t *testing.T) {
	cfg := SmallMambaConfig()
	m := NewMambaModel(cfg)

	m.Tok.BuildVocab([]string{
		"quantum mechanics is a branch of physics that describes behavior of matter and energy",
		"bitcoin is a cryptocurrency created by satoshi nakamoto in 2008",
		"python is a programming language used for web development and data science",
	}, cfg.VocabSize)

	// Generate should produce something (even if nonsensical for untrained model)
	result := m.Generate("quantum mechanics", "is_a", "branch of physics", 15, 0.8)

	// Should return some text (untrained model generates random tokens)
	if len(result) == 0 {
		t.Log("generated empty text (expected for untrained model)")
	} else {
		t.Logf("generated: %q", result)
	}
}

func TestMambaTrainStep(t *testing.T) {
	cfg := SmallMambaConfig()
	m := NewMambaModel(cfg)

	m.Tok.BuildVocab([]string{
		"quantum mechanics <sep> is_a <sep> branch of physics",
		"Quantum mechanics is a branch of physics.",
	}, cfg.VocabSize)

	ex := TrainingExample{
		Input:  "quantum mechanics <sep> is_a <sep> branch of physics",
		Target: "Quantum mechanics is a branch of physics.",
	}

	loss1 := m.trainStep(ex, 0.001)
	loss2 := m.trainStep(ex, 0.001)

	if math.IsNaN(float64(loss1)) || math.IsInf(float64(loss1), 0) {
		t.Fatalf("loss1 is NaN or Inf: %f", loss1)
	}
	if math.IsNaN(float64(loss2)) || math.IsInf(float64(loss2), 0) {
		t.Fatalf("loss2 is NaN or Inf: %f", loss2)
	}

	t.Logf("loss1=%.4f, loss2=%.4f", loss1, loss2)
}

func TestMambaSaveLoad(t *testing.T) {
	cfg := SmallMambaConfig()
	m := NewMambaModel(cfg)

	m.Tok.BuildVocab([]string{
		"hello world test vocabulary for save load",
	}, cfg.VocabSize)

	dir := t.TempDir()
	path := filepath.Join(dir, "test_mamba.bin")

	// Save
	if err := m.Save(path); err != nil {
		t.Fatalf("Save failed: %v", err)
	}

	// Check file exists
	info, err := os.Stat(path)
	if err != nil {
		t.Fatalf("model file not found: %v", err)
	}
	t.Logf("model file size: %d bytes", info.Size())

	// Load
	loaded, err := LoadMambaModel(path)
	if err != nil {
		t.Fatalf("LoadMambaModel failed: %v", err)
	}

	// Check config matches
	if loaded.Config.ModelDim != cfg.ModelDim {
		t.Errorf("ModelDim: expected %d, got %d", cfg.ModelDim, loaded.Config.ModelDim)
	}
	if loaded.Config.NumLayers != cfg.NumLayers {
		t.Errorf("NumLayers: expected %d, got %d", cfg.NumLayers, loaded.Config.NumLayers)
	}
	if loaded.Config.StateDim != cfg.StateDim {
		t.Errorf("StateDim: expected %d, got %d", cfg.StateDim, loaded.Config.StateDim)
	}

	// Check vocab
	if loaded.Tok.VocabSize() != m.Tok.VocabSize() {
		t.Errorf("vocab size: expected %d, got %d", m.Tok.VocabSize(), loaded.Tok.VocabSize())
	}

	// Check weights match
	for i := range m.TokenEmbed {
		if math.Abs(float64(m.TokenEmbed[i]-loaded.TokenEmbed[i])) > 1e-6 {
			t.Errorf("TokenEmbed[%d]: expected %f, got %f", i, m.TokenEmbed[i], loaded.TokenEmbed[i])
			break
		}
	}

	// Generate with loaded model should not crash
	loaded.Generate("test", "is_a", "word", 10, 0.7)
}

func TestMambaBridge(t *testing.T) {
	cfg := SmallMambaConfig()
	m := NewMambaModel(cfg)

	m.Tok.BuildVocab([]string{
		"python is a programming language",
	}, cfg.VocabSize)

	b := NewMambaBridge(m)

	if !b.IsMamba() {
		t.Error("expected IsMamba() to be true")
	}

	// Should not panic
	result := b.GenerateSentence("python", "is_a", "programming language")
	t.Logf("bridge generated: %q", result)

	// Paragraph generation
	facts := [][3]string{
		{"python", "is_a", "programming language"},
		{"python", "created_by", "Guido van Rossum"},
	}
	para := b.GenerateParagraph("python", facts)
	t.Logf("bridge paragraph: %q", para)
}

func TestMambaDefaultConfig(t *testing.T) {
	cfg := DefaultMambaConfig()

	if cfg.DtRank() != 16 {
		t.Errorf("DtRank: expected 16, got %d", cfg.DtRank())
	}
	if cfg.InnerDim() != 512 {
		t.Errorf("InnerDim: expected 512, got %d", cfg.InnerDim())
	}

	m := NewMambaModel(cfg)
	params := m.ParamCount()
	t.Logf("Default Mamba model: %d parameters (%.1f MB)", params, float64(params)*4/1024/1024)

	// Should be roughly 7-10M parameters
	if params < 5_000_000 || params > 15_000_000 {
		t.Errorf("unexpected param count: %d (expected 5-15M)", params)
	}
}
