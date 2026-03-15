package cognitive

import (
	"math"
	"os"
	"path/filepath"
	"testing"
)

// --- Neural Cortex Tests ---

func TestNeuralCortexCreation(t *testing.T) {
	labels := []string{"grep", "read", "write", "none"}
	nc := NewNeuralCortex(4, 3, labels, "")
	if nc == nil {
		t.Fatal("should not return nil")
	}
	if nc.InputSize != 4 {
		t.Errorf("input size = %d, want 4", nc.InputSize)
	}
	if nc.OutputSize != 4 {
		t.Errorf("output size = %d, want 4", nc.OutputSize)
	}
}

func TestNeuralCortexPredict(t *testing.T) {
	labels := []string{"grep", "read", "write"}
	nc := NewNeuralCortex(4, 3, labels, "")

	input := []float64{0.1, 0.2, 0.3, 0.4}
	pred := nc.Predict(input)

	if pred.Label == "" {
		t.Error("prediction should have a label")
	}
	if pred.Confidence <= 0 {
		t.Error("confidence should be positive")
	}
	if len(pred.AllScores) != 3 {
		t.Errorf("should have scores for all 3 labels, got %d", len(pred.AllScores))
	}

	// Softmax outputs should sum to ~1.0
	sum := 0.0
	for _, score := range pred.AllScores {
		sum += score
	}
	if math.Abs(sum-1.0) > 0.01 {
		t.Errorf("softmax scores should sum to 1.0, got %f", sum)
	}
}

func TestNeuralCortexPredictWrongInputSize(t *testing.T) {
	nc := NewNeuralCortex(4, 3, []string{"a", "b"}, "")
	pred := nc.Predict([]float64{1.0}) // wrong size
	if pred.Label != "" {
		t.Error("wrong input size should return empty prediction")
	}
}

func TestNeuralCortexTrain(t *testing.T) {
	labels := []string{"grep", "read", "write"}
	nc := NewNeuralCortex(4, 3, labels, "")

	input := []float64{1.0, 0.0, 0.0, 0.0}

	// Train 100 times on "grep"
	for i := 0; i < 100; i++ {
		nc.Train(input, "grep")
	}

	pred := nc.Predict(input)
	if pred.Label != "grep" {
		t.Errorf("after 100 training steps, should predict 'grep', got %q", pred.Label)
	}
	if pred.Confidence < 0.5 {
		t.Errorf("confidence should be >0.5, got %f", pred.Confidence)
	}
	if nc.TrainCount != 100 {
		t.Errorf("train count = %d, want 100", nc.TrainCount)
	}
}

func TestNeuralCortexTrainMultiplePatterns(t *testing.T) {
	labels := []string{"grep", "read", "write"}
	nc := NewNeuralCortex(4, 3, labels, "")

	grepInput := []float64{1.0, 0.0, 0.0, 0.0}
	readInput := []float64{0.0, 1.0, 0.0, 0.0}
	writeInput := []float64{0.0, 0.0, 1.0, 0.0}

	// Train on each pattern (500 iterations for reliable convergence)
	for i := 0; i < 500; i++ {
		nc.Train(grepInput, "grep")
		nc.Train(readInput, "read")
		nc.Train(writeInput, "write")
	}

	// Test each pattern
	if p := nc.Predict(grepInput); p.Label != "grep" {
		t.Errorf("should predict grep, got %q (conf %.2f)", p.Label, p.Confidence)
	}
	if p := nc.Predict(readInput); p.Label != "read" {
		t.Errorf("should predict read, got %q (conf %.2f)", p.Label, p.Confidence)
	}
	if p := nc.Predict(writeInput); p.Label != "write" {
		t.Errorf("should predict write, got %q (conf %.2f)", p.Label, p.Confidence)
	}
}

func TestNeuralCortexSaveLoad(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "cortex.json")

	labels := []string{"grep", "read", "write"}
	nc := NewNeuralCortex(4, 3, labels, path)

	input := []float64{1.0, 0.0, 0.0, 0.0}
	for i := 0; i < 50; i++ {
		nc.Train(input, "grep")
	}

	// Save
	if err := nc.Save(); err != nil {
		t.Fatalf("save failed: %v", err)
	}

	// Load into new cortex
	nc2 := NewNeuralCortex(4, 3, labels, path)
	if nc2.TrainCount != 50 {
		t.Errorf("loaded train count = %d, want 50", nc2.TrainCount)
	}

	// Should predict same as original
	pred1 := nc.Predict(input)
	pred2 := nc2.Predict(input)
	if pred1.Label != pred2.Label {
		t.Errorf("loaded cortex predicts %q, original predicts %q", pred2.Label, pred1.Label)
	}
}

func TestNeuralCortexStats(t *testing.T) {
	labels := []string{"a", "b", "c"}
	nc := NewNeuralCortex(10, 5, labels, "")

	trainCount, paramCount := nc.Stats()
	if trainCount != 0 {
		t.Error("initial train count should be 0")
	}
	// 10*5 + 5 + 5*3 + 3 = 50 + 5 + 15 + 3 = 73
	if paramCount != 73 {
		t.Errorf("param count = %d, want 73", paramCount)
	}
}

func TestNeuralCortexLoadNonexistent(t *testing.T) {
	nc := NewNeuralCortex(4, 3, []string{"a"}, "/nonexistent/path.json")
	// Should not crash, just initialize with random weights
	pred := nc.Predict([]float64{1.0, 0.0, 0.0, 0.0})
	if pred.Label == "" {
		t.Error("should still work with fresh weights")
	}
}

// --- Math Tests ---

func TestRelu(t *testing.T) {
	if relu(5.0) != 5.0 {
		t.Error("relu(5) should be 5")
	}
	if relu(-3.0) != 0 {
		t.Error("relu(-3) should be 0")
	}
	if relu(0) != 0 {
		t.Error("relu(0) should be 0")
	}
}

func TestSoftmax(t *testing.T) {
	logits := []float64{1.0, 2.0, 3.0}
	result := softmax(logits)

	// Should sum to 1
	sum := 0.0
	for _, v := range result {
		sum += v
	}
	if math.Abs(sum-1.0) > 1e-6 {
		t.Errorf("softmax should sum to 1, got %f", sum)
	}

	// Last element should be highest
	if result[2] <= result[1] || result[1] <= result[0] {
		t.Error("softmax should preserve ordering")
	}
}

func TestSoftmaxNumericalStability(t *testing.T) {
	// Large values that would overflow naive exp()
	logits := []float64{1000.0, 1001.0, 1002.0}
	result := softmax(logits)

	sum := 0.0
	for _, v := range result {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			t.Fatal("softmax produced NaN/Inf")
		}
		sum += v
	}
	if math.Abs(sum-1.0) > 1e-6 {
		t.Errorf("softmax should sum to 1 even with large values, got %f", sum)
	}
}

// --- Benchmark ---

func BenchmarkCortexPredict(b *testing.B) {
	labels := []string{"grep", "read", "write", "ls", "tree", "glob", "git", "edit", "none"}
	nc := NewNeuralCortex(768, 128, labels, "")
	input := make([]float64, 768)
	for i := range input {
		input[i] = float64(i) * 0.001
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		nc.Predict(input)
	}
}

func BenchmarkCortexTrain(b *testing.B) {
	labels := []string{"grep", "read", "write", "none"}
	nc := NewNeuralCortex(768, 128, labels, "")
	input := make([]float64, 768)
	for i := range input {
		input[i] = float64(i) * 0.001
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		nc.Train(input, "grep")
	}
}

func TestMain(m *testing.M) {
	os.Exit(m.Run())
}
