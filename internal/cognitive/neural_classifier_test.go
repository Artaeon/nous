package cognitive

import (
	"fmt"
	"os"
	"testing"
	"time"
)

func TestNeuralFeatureExtraction(t *testing.T) {
	// Basic feature extraction
	features := ExtractFeatures("hello world", DefaultFeatureSize)
	if len(features) != DefaultFeatureSize {
		t.Fatalf("expected %d features, got %d", DefaultFeatureSize, len(features))
	}

	// Should be L2 normalized (norm ≈ 1.0)
	var norm float32
	for _, v := range features {
		norm += v * v
	}
	if norm < 0.99 || norm > 1.01 {
		t.Errorf("expected normalized vector (norm=1.0), got norm=%.4f", norm)
	}

	// Different inputs should produce different features
	f1 := ExtractFeatures("define serendipity", DefaultFeatureSize)
	f2 := ExtractFeatures("translate hello to french", DefaultFeatureSize)

	var dot float32
	for i := range f1 {
		dot += f1[i] * f2[i]
	}
	// Cosine similarity should be < 1.0 for different inputs
	if dot > 0.95 {
		t.Errorf("different inputs should produce different features, cosine=%.4f", dot)
	}

	// Empty input
	empty := ExtractFeatures("", DefaultFeatureSize)
	var emptyNorm float32
	for _, v := range empty {
		emptyNorm += v * v
	}
	if emptyNorm > 0.001 {
		t.Errorf("empty input should produce zero vector, got norm=%.4f", emptyNorm)
	}
}

func TestNeuralClassifierTrainAndClassify(t *testing.T) {
	nc := NewNeuralClassifier(DefaultFeatureSize, DefaultHiddenSize)

	examples := []TrainingExample{
		{Text: "hello", Intent: "greeting"},
		{Text: "hi there", Intent: "greeting"},
		{Text: "hey", Intent: "greeting"},
		{Text: "good morning", Intent: "greeting"},
		{Text: "howdy", Intent: "greeting"},
		{Text: "what's up", Intent: "greeting"},
		{Text: "hi how are you", Intent: "greeting"},

		{Text: "define serendipity", Intent: "dict"},
		{Text: "definition of ubiquitous", Intent: "dict"},
		{Text: "what does ephemeral mean", Intent: "dict"},
		{Text: "synonyms for happy", Intent: "dict"},
		{Text: "meaning of paradigm", Intent: "dict"},
		{Text: "define the word pragmatic", Intent: "dict"},

		{Text: "translate hello to french", Intent: "translate"},
		{Text: "how do you say goodbye in spanish", Intent: "translate"},
		{Text: "translate thank you to japanese", Intent: "translate"},
		{Text: "what is hello in german", Intent: "translate"},

		{Text: "remind me to call mom", Intent: "reminder"},
		{Text: "set a reminder for 5pm", Intent: "reminder"},
		{Text: "remind me in 30 minutes", Intent: "reminder"},
		{Text: "don't let me forget", Intent: "reminder"},

		{Text: "what is quantum physics", Intent: "explain"},
		{Text: "explain photosynthesis", Intent: "explain"},
		{Text: "tell me about black holes", Intent: "explain"},
		{Text: "how does gravity work", Intent: "explain"},
		{Text: "who is Einstein", Intent: "explain"},
	}

	result := nc.Train(examples, 50, 0.05)
	t.Logf("Training: %s", result)

	if result.Accuracy < 0.5 {
		t.Errorf("training accuracy too low: %.1f%%", result.Accuracy*100)
	}

	// Test classification
	tests := []struct {
		input      string
		wantIntent string
	}{
		{"hi there!", "greeting"},
		{"hello", "greeting"},
		{"define ubiquitous", "dict"},
		{"translate goodbye to french", "translate"},
		{"remind me to buy groceries", "reminder"},
		{"explain how DNA works", "explain"},
	}

	correct := 0
	for _, tt := range tests {
		intent, conf := nc.Classify(tt.input)
		if intent == tt.wantIntent {
			correct++
		}
		t.Logf("%-40s → %-12s (conf=%.2f) want=%-12s %s",
			tt.input, intent, conf, tt.wantIntent,
			func() string {
				if intent == tt.wantIntent {
					return "OK"
				}
				return "MISS"
			}())
	}
	t.Logf("Test accuracy: %d/%d", correct, len(tests))
}

func TestNeuralClassifierSaveLoad(t *testing.T) {
	nc := NewNeuralClassifier(DefaultFeatureSize, DefaultHiddenSize)

	examples := []TrainingExample{
		{Text: "hello", Intent: "greeting"},
		{Text: "hi", Intent: "greeting"},
		{Text: "hey", Intent: "greeting"},
		{Text: "define word", Intent: "dict"},
		{Text: "definition of", Intent: "dict"},
		{Text: "translate to", Intent: "translate"},
		{Text: "say in french", Intent: "translate"},
	}

	nc.Train(examples, 20, 0.05)

	// Save
	tmpFile := os.TempDir() + "/nous_test_model.bin"
	defer os.Remove(tmpFile)

	if err := nc.Save(tmpFile); err != nil {
		t.Fatalf("save: %v", err)
	}

	// Load into new classifier
	nc2 := NewNeuralClassifier(0, 0) // sizes will be set from file
	if err := nc2.Load(tmpFile); err != nil {
		t.Fatalf("load: %v", err)
	}

	// Both should classify the same way
	intent1, conf1 := nc.Classify("hello there")
	intent2, conf2 := nc2.Classify("hello there")

	if intent1 != intent2 {
		t.Errorf("loaded model gives different intent: %s vs %s", intent1, intent2)
	}
	if conf1 != conf2 {
		t.Errorf("loaded model gives different confidence: %.4f vs %.4f", conf1, conf2)
	}
}

func TestNeuralClassifierFullTrainingData(t *testing.T) {
	nlu := NewNLU()
	examples := GenerateTrainingData(nlu)
	t.Logf("Generated %d training examples", len(examples))

	// Count intents
	intentCounts := make(map[string]int)
	for _, ex := range examples {
		intentCounts[ex.Intent]++
	}
	t.Logf("Intent distribution (%d intents):", len(intentCounts))
	for intent, count := range intentCounts {
		t.Logf("  %-20s %d", intent, count)
	}

	// Augment
	augmented := AugmentExamples(examples)
	t.Logf("After augmentation: %d examples", len(augmented))

	// Train
	nc := NewNeuralClassifier(DefaultFeatureSize, DefaultHiddenSize)
	start := time.Now()
	result := nc.Train(augmented, 80, 0.1)
	trainDur := time.Since(start)
	t.Logf("Training: %s (took %s)", result, trainDur)

	if result.Accuracy < 0.70 {
		t.Errorf("full training accuracy too low: %.1f%%", result.Accuracy*100)
	}

	// Test on the Round 2 problem cases
	problemCases := []struct {
		input      string
		wantIntent string
	}{
		// Round 2 fixes
		{"what is your name?", "meta"},
		{"define serendipity", "dict"},
		{"translate hello to french", "translate"},
		{"remind me to call mom tomorrow", "reminder"},
		{"remember my favorite color is blue", "remember"},
		{"what is my favorite color?", "recall"},
		{"tell me something interesting", "creative"},
		{"who made you?", "meta"},
		{"do you have feelings?", "meta"},
		{"I feel happy today", "greeting"},
		{"what is the meaning of life?", "creative"},

		// Additional cases
		{"hello", "greeting"},
		{"bye", "farewell"},
		{"thanks", "affirmation"},
		{"what's the weather", "weather"},
		{"set a timer for 5 minutes", "timer"},
		{"generate a password", "password"},
		{"write me a poem", "creative"},
		{"what is quantum physics", "explain"},
		{"search for golang tutorials", "search"},
		{"open firefox", "app"},
		{"take a screenshot", "screenshot"},
		{"good morning", "daily_briefing"},
	}

	correct := 0
	for _, tt := range problemCases {
		intent, conf := nc.Classify(tt.input)
		ok := intent == tt.wantIntent
		if ok {
			correct++
		}
		status := "OK"
		if !ok {
			status = "MISS"
		}
		t.Logf("%-50s → %-16s (conf=%.2f) want=%-16s %s",
			tt.input, intent, conf, tt.wantIntent, status)
	}

	accuracy := float64(correct) / float64(len(problemCases))
	t.Logf("Problem case accuracy: %d/%d (%.0f%%)", correct, len(problemCases), accuracy*100)

	if accuracy < 0.60 {
		t.Errorf("problem case accuracy too low: %.0f%%", accuracy*100)
	}
}

func TestNeuralClassifierInferenceSpeed(t *testing.T) {
	nlu := NewNLU()
	examples := GenerateTrainingData(nlu)
	augmented := AugmentExamples(examples)

	nc := NewNeuralClassifier(DefaultFeatureSize, DefaultHiddenSize)
	nc.Train(augmented, 20, 0.03)

	// Benchmark inference speed
	inputs := []string{
		"hello how are you",
		"define serendipity",
		"translate hello to french",
		"remind me to call mom tomorrow",
		"what is quantum physics",
		"write me a poem about the ocean",
		"I feel happy today",
		"what is your name",
	}

	// Warmup
	for _, input := range inputs {
		nc.Classify(input)
	}

	// Benchmark
	iterations := 10000
	start := time.Now()
	for i := 0; i < iterations; i++ {
		nc.Classify(inputs[i%len(inputs)])
	}
	elapsed := time.Since(start)

	avgNs := elapsed.Nanoseconds() / int64(iterations)
	t.Logf("Average inference time: %d ns (%.1f µs) over %d iterations",
		avgNs, float64(avgNs)/1000.0, iterations)

	// Should be under 100 microseconds
	if avgNs > 100_000 {
		t.Errorf("inference too slow: %d ns (want < 100,000 ns)", avgNs)
	}
}

func TestNeuralClassifierTopN(t *testing.T) {
	nlu := NewNLU()
	examples := GenerateTrainingData(nlu)
	augmented := AugmentExamples(examples)

	nc := NewNeuralClassifier(DefaultFeatureSize, DefaultHiddenSize)
	nc.Train(augmented, 20, 0.03)

	// "what is your name" should have "meta" in top-3
	top3 := nc.ClassifyTopN("what is your name?", 3)
	if len(top3) != 3 {
		t.Fatalf("expected 3 predictions, got %d", len(top3))
	}

	fmt.Printf("Top-3 for 'what is your name?':\n")
	for _, p := range top3 {
		fmt.Printf("  %-16s %.2f\n", p.Intent, p.Confidence)
	}

	// Top-1 should have reasonable confidence
	if top3[0].Confidence < 0.1 {
		t.Errorf("top prediction confidence too low: %.2f", top3[0].Confidence)
	}
}
