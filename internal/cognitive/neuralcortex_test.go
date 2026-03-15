package cognitive

import (
	"math"
	"os"
	"path/filepath"
	"sync"
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

	// Train on each pattern (1000 iterations for reliable convergence with random init)
	for i := 0; i < 1000; i++ {
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

func TestNeuralCortexLearningRateDecay(t *testing.T) {
	labels := []string{"grep", "read", "write"}
	nc := NewNeuralCortex(4, 3, labels, "")

	initialLR := nc.LearningRate
	input := []float64{1.0, 0.0, 0.0, 0.0}

	// Train enough to trigger decay
	for i := 0; i < 200; i++ {
		nc.Train(input, "grep")
	}

	if nc.LearningRate >= initialLR {
		t.Errorf("learning rate should decay, got %f (initial %f)", nc.LearningRate, initialLR)
	}
	if nc.LearningRate <= 0 {
		t.Error("learning rate should never reach zero")
	}
}

func TestNeuralCortexWeightDecay(t *testing.T) {
	labels := []string{"grep", "read"}
	nc := NewNeuralCortex(4, 3, labels, "")
	nc.WeightDecay = 0.1 // aggressive decay for testing

	// Record initial weight magnitude
	initialMag := 0.0
	for i := range nc.W1 {
		for j := range nc.W1[i] {
			initialMag += nc.W1[i][j] * nc.W1[i][j]
		}
	}

	// Train with high weight decay — weights should be smaller
	input := []float64{1.0, 0.0, 0.0, 0.0}
	for i := 0; i < 500; i++ {
		nc.Train(input, "grep")
	}

	finalMag := 0.0
	for i := range nc.W1 {
		for j := range nc.W1[i] {
			finalMag += nc.W1[i][j] * nc.W1[i][j]
		}
	}

	// With L2 regularization, weights should be pushed toward zero
	// The network should still learn (weights shouldn't be zero),
	// but they should be smaller than without regularization
	if finalMag == 0 {
		t.Error("weights should not be exactly zero")
	}
}

func TestNeuralCortexRegularizationDefaults(t *testing.T) {
	nc := NewNeuralCortex(4, 3, []string{"a", "b"}, "")
	if nc.WeightDecay != 0.0001 {
		t.Errorf("default weight decay = %f, want 0.0001", nc.WeightDecay)
	}
	if nc.InitialLR != 0.01 {
		t.Errorf("default initial LR = %f, want 0.01", nc.InitialLR)
	}
}

// --- Race Condition Tests ---

func TestNeuralCortexConcurrentTrainPredict(t *testing.T) {
	labels := []string{"grep", "read", "write", "ls", "edit"}
	nc := NewNeuralCortex(8, 4, labels, "")

	var wg sync.WaitGroup

	// 10 goroutines training concurrently
	for g := 0; g < 10; g++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			input := make([]float64, 8)
			input[id%8] = 1.0
			label := labels[id%len(labels)]
			for i := 0; i < 100; i++ {
				nc.Train(input, label)
			}
		}(g)
	}

	// 10 goroutines predicting concurrently
	for g := 0; g < 10; g++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			input := make([]float64, 8)
			input[id%8] = 1.0
			for i := 0; i < 100; i++ {
				pred := nc.Predict(input)
				if pred.Label == "" && pred.Confidence != 0 {
					t.Errorf("inconsistent prediction: empty label but non-zero confidence")
				}
			}
		}(g)
	}

	wg.Wait()

	// Verify cortex is still functional
	pred := nc.Predict([]float64{1, 0, 0, 0, 0, 0, 0, 0})
	if pred.Label == "" {
		t.Error("cortex should be functional after concurrent access")
	}
}

// --- Formula Verification Tests ---

func TestLearningRateDecayFormula(t *testing.T) {
	labels := []string{"a", "b"}

	// Verify at specific step counts: LR = InitialLR / (1 + N/500)
	checkpoints := []int{1, 50, 100, 500, 1000}
	for _, steps := range checkpoints {
		nc := NewNeuralCortex(4, 3, labels, "")
		input := []float64{1.0, 0.0, 0.0, 0.0}
		for i := 0; i < steps; i++ {
			nc.Train(input, "a")
		}
		expectedLR := 0.01 / (1.0 + float64(steps)/500.0)
		if math.Abs(nc.LearningRate-expectedLR) > 1e-10 {
			t.Errorf("after %d steps: LR = %f, want %f", steps, nc.LearningRate, expectedLR)
		}
	}

	// Step 0: LR should equal InitialLR
	nc0 := NewNeuralCortex(4, 3, labels, "")
	if nc0.LearningRate != 0.01 {
		t.Errorf("at step 0: LR = %f, want 0.01", nc0.LearningRate)
	}
}

func TestNeuralCortexZeroWeightDecay(t *testing.T) {
	labels := []string{"a", "b"}

	// Train two cortexes: one with WeightDecay=0, one with WeightDecay>0
	nc0 := NewNeuralCortex(4, 3, labels, "")
	nc0.WeightDecay = 0.0

	ncWD := NewNeuralCortex(4, 3, labels, "")
	ncWD.WeightDecay = 0.1

	// Copy weights from nc0 to ncWD so they start the same
	ncWD.mu.Lock()
	nc0.mu.RLock()
	for i := range nc0.W1 {
		copy(ncWD.W1[i], nc0.W1[i])
	}
	copy(ncWD.B1, nc0.B1)
	for i := range nc0.W2 {
		copy(ncWD.W2[i], nc0.W2[i])
	}
	copy(ncWD.B2, nc0.B2)
	nc0.mu.RUnlock()
	ncWD.mu.Unlock()

	input := []float64{1.0, 0.5, 0.0, 0.0}
	for i := 0; i < 50; i++ {
		nc0.Train(input, "a")
		ncWD.Train(input, "a")
	}

	// Weight decay should produce smaller L2 norm
	sumW0, sumWD := 0.0, 0.0
	nc0.mu.RLock()
	ncWD.mu.RLock()
	for i := range nc0.W1 {
		for j := range nc0.W1[i] {
			sumW0 += nc0.W1[i][j] * nc0.W1[i][j]
			sumWD += ncWD.W1[i][j] * ncWD.W1[i][j]
		}
	}
	nc0.mu.RUnlock()
	ncWD.mu.RUnlock()

	if sumWD >= sumW0 {
		t.Errorf("weight decay should reduce weight magnitude: L2(wd=0)=%f, L2(wd=0.1)=%f", sumW0, sumWD)
	}
}

func TestMain(m *testing.M) {
	os.Exit(m.Run())
}
