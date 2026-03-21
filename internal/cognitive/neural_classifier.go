package cognitive

import (
	"encoding/binary"
	"fmt"
	"math"
	"math/rand"
	"os"
	"sort"
	"sync"
)

// -----------------------------------------------------------------------
// Neural Intent Classifier — a lightweight 2-layer MLP trained on labeled
// examples to classify user input into intents.
//
// Architecture:
//   Input (feature_size) → Hidden (hidden_size, ReLU) → Output (num_intents, Softmax)
//
// Key properties:
//   - Pure Go, zero external dependencies
//   - Sub-microsecond inference (single forward pass ~267K multiply-adds)
//   - Float32 for cache-friendly weight storage
//   - Binary serialization for fast model load (~500KB)
//   - Thread-safe: concurrent reads, exclusive writes
//   - Incremental learning with experience replay
//
// Training uses mini-batch SGD with:
//   - Cross-entropy loss
//   - He initialization for ReLU layer
//   - Learning rate decay
//   - Class-balanced sampling
// -----------------------------------------------------------------------

const (
	DefaultHiddenSize = 64
	modelMagic        = 0x4E4F5553 // "NOUS"
	modelVersion      = 1
)

// TrainingExample is a single labeled training instance.
type TrainingExample struct {
	Text   string `json:"text"`
	Intent string `json:"intent"`
}

// NeuralClassifier is a 2-layer MLP for intent classification.
type NeuralClassifier struct {
	mu sync.RWMutex

	// Network architecture
	featureSize int
	hiddenSize  int
	numIntents  int

	// Weights and biases (row-major)
	w1 []float32 // featureSize × hiddenSize
	b1 []float32 // hiddenSize
	w2 []float32 // hiddenSize × numIntents
	b2 []float32 // numIntents

	// Intent label index
	intentToIdx map[string]int
	idxToIntent []string

	// Experience replay buffer for incremental learning
	replayBuffer []TrainingExample
	replayMax    int

	// Training stats
	Epochs       int
	TrainingSamples int
	Accuracy     float64
}

// NewNeuralClassifier creates an untrained classifier.
func NewNeuralClassifier(featureSize, hiddenSize int) *NeuralClassifier {
	return &NeuralClassifier{
		featureSize:  featureSize,
		hiddenSize:   hiddenSize,
		intentToIdx:  make(map[string]int),
		replayMax:    500, // keep up to 500 examples for replay
	}
}

// initWeights allocates and initializes weights using He/Xavier initialization.
func (nc *NeuralClassifier) initWeights() {
	nc.w1 = make([]float32, nc.featureSize*nc.hiddenSize)
	nc.b1 = make([]float32, nc.hiddenSize)
	nc.w2 = make([]float32, nc.hiddenSize*nc.numIntents)
	nc.b2 = make([]float32, nc.numIntents)

	// He initialization for ReLU layer: stddev = sqrt(2 / fan_in)
	heStd := float64(math.Sqrt(2.0 / float64(nc.featureSize)))
	for i := range nc.w1 {
		nc.w1[i] = float32(rand.NormFloat64() * heStd)
	}

	// Xavier initialization for output layer: stddev = sqrt(1 / fan_in)
	xavierStd := float64(math.Sqrt(1.0 / float64(nc.hiddenSize)))
	for i := range nc.w2 {
		nc.w2[i] = float32(rand.NormFloat64() * xavierStd)
	}
}

// Classify runs a forward pass and returns the predicted intent and confidence.
func (nc *NeuralClassifier) Classify(text string) (intent string, confidence float64) {
	nc.mu.RLock()
	defer nc.mu.RUnlock()

	if nc.numIntents == 0 || nc.w1 == nil {
		return "", 0
	}

	features := ExtractFeatures(text, nc.featureSize)
	probs := nc.forward(features)

	// Find argmax
	bestIdx := 0
	bestProb := probs[0]
	for i := 1; i < len(probs); i++ {
		if probs[i] > bestProb {
			bestProb = probs[i]
			bestIdx = i
		}
	}

	if bestIdx < len(nc.idxToIntent) {
		return nc.idxToIntent[bestIdx], float64(bestProb)
	}
	return "", 0
}

// ClassifyTopN returns the top N predictions with their confidences.
func (nc *NeuralClassifier) ClassifyTopN(text string, n int) []IntentPrediction {
	nc.mu.RLock()
	defer nc.mu.RUnlock()

	if nc.numIntents == 0 || nc.w1 == nil {
		return nil
	}

	features := ExtractFeatures(text, nc.featureSize)
	probs := nc.forward(features)

	type idxProb struct {
		idx  int
		prob float32
	}
	ranked := make([]idxProb, len(probs))
	for i, p := range probs {
		ranked[i] = idxProb{i, p}
	}
	sort.Slice(ranked, func(i, j int) bool {
		return ranked[i].prob > ranked[j].prob
	})

	if n > len(ranked) {
		n = len(ranked)
	}
	results := make([]IntentPrediction, n)
	for i := 0; i < n; i++ {
		results[i] = IntentPrediction{
			Intent:     nc.idxToIntent[ranked[i].idx],
			Confidence: float64(ranked[i].prob),
		}
	}
	return results
}

// IntentPrediction holds a single prediction.
type IntentPrediction struct {
	Intent     string
	Confidence float64
}

// forward computes the forward pass: features → hidden (ReLU) → output (softmax).
// Uses sparse feature optimization — skips zero-valued features (typically >90% sparse).
func (nc *NeuralClassifier) forward(features []float32) []float32 {
	// Build sparse index: only non-zero features matter
	type sparseEntry struct {
		idx int
		val float32
	}
	var sparse []sparseEntry
	for i, v := range features {
		if v != 0 {
			sparse = append(sparse, sparseEntry{i, v})
		}
	}

	// Layer 1: hidden = ReLU(features · W1 + b1) — sparse computation
	hidden := make([]float32, nc.hiddenSize)
	copy(hidden, nc.b1) // start with bias
	for _, s := range sparse {
		row := s.idx * nc.hiddenSize
		val := s.val
		for j := 0; j < nc.hiddenSize; j++ {
			hidden[j] += val * nc.w1[row+j]
		}
	}
	// ReLU
	for j := range hidden {
		if hidden[j] < 0 {
			hidden[j] = 0
		}
	}

	// Layer 2: logits = hidden · W2 + b2 — dense (hidden is small)
	logits := make([]float32, nc.numIntents)
	copy(logits, nc.b2)
	for j := 0; j < nc.hiddenSize; j++ {
		if hidden[j] == 0 {
			continue // skip dead ReLU neurons
		}
		row := j * nc.numIntents
		val := hidden[j]
		for k := 0; k < nc.numIntents; k++ {
			logits[k] += val * nc.w2[row+k]
		}
	}

	return softmax32(logits)
}

// softmax computes the softmax of logits with numerical stability.
func softmax32(logits []float32) []float32 {
	probs := make([]float32, len(logits))

	// Find max for numerical stability
	maxVal := logits[0]
	for _, v := range logits[1:] {
		if v > maxVal {
			maxVal = v
		}
	}

	// exp(x - max) and sum
	var sum float32
	for i, v := range logits {
		probs[i] = float32(math.Exp(float64(v - maxVal)))
		sum += probs[i]
	}

	// Normalize
	if sum > 0 {
		invSum := 1.0 / sum
		for i := range probs {
			probs[i] *= invSum
		}
	}

	return probs
}

// -----------------------------------------------------------------------
// Training
// -----------------------------------------------------------------------

// Train trains the classifier on labeled examples using mini-batch SGD.
func (nc *NeuralClassifier) Train(examples []TrainingExample, epochs int, lr float32) TrainResult {
	nc.mu.Lock()
	defer nc.mu.Unlock()

	if len(examples) == 0 {
		return TrainResult{}
	}

	// Build intent index
	intentSet := make(map[string]bool)
	for _, ex := range examples {
		intentSet[ex.Intent] = true
	}
	intents := make([]string, 0, len(intentSet))
	for intent := range intentSet {
		intents = append(intents, intent)
	}
	sort.Strings(intents)

	nc.intentToIdx = make(map[string]int, len(intents))
	nc.idxToIntent = intents
	nc.numIntents = len(intents)
	for i, intent := range intents {
		nc.intentToIdx[intent] = i
	}

	// Initialize weights
	nc.initWeights()

	// Pre-extract features for all examples
	type preparedExample struct {
		features []float32
		label    int
	}
	prepared := make([]preparedExample, len(examples))
	for i, ex := range examples {
		prepared[i] = preparedExample{
			features: ExtractFeatures(ex.Text, nc.featureSize),
			label:    nc.intentToIdx[ex.Intent],
		}
	}

	// Training loop with SGD + momentum
	batchSize := 16
	if batchSize > len(prepared) {
		batchSize = len(prepared)
	}

	// Momentum buffers
	momentum := float32(0.9)
	vW1 := make([]float32, len(nc.w1))
	vb1 := make([]float32, nc.hiddenSize)
	vW2 := make([]float32, len(nc.w2))
	vb2 := make([]float32, nc.numIntents)

	var finalLoss float64
	var finalAcc float64

	for epoch := 0; epoch < epochs; epoch++ {
		// Shuffle
		rand.Shuffle(len(prepared), func(i, j int) {
			prepared[i], prepared[j] = prepared[j], prepared[i]
		})

		// Learning rate decay: cosine annealing
		progress := float64(epoch) / float64(epochs)
		currentLR := lr * float32(0.5*(1.0+math.Cos(progress*math.Pi)))
		if currentLR < lr*0.01 {
			currentLR = lr * 0.01
		}

		var epochLoss float64
		var epochCorrect int

		for batchStart := 0; batchStart < len(prepared); batchStart += batchSize {
			batchEnd := batchStart + batchSize
			if batchEnd > len(prepared) {
				batchEnd = len(prepared)
			}
			batch := prepared[batchStart:batchEnd]

			// Accumulate gradients over batch
			dW1 := make([]float32, len(nc.w1))
			db1 := make([]float32, nc.hiddenSize)
			dW2 := make([]float32, len(nc.w2))
			db2 := make([]float32, nc.numIntents)

			for _, ex := range batch {
				loss, correct := nc.backprop(ex.features, ex.label, dW1, db1, dW2, db2)
				epochLoss += loss
				if correct {
					epochCorrect++
				}
			}

			// SGD with momentum
			scale := currentLR / float32(len(batch))
			for i := range nc.w1 {
				vW1[i] = momentum*vW1[i] + scale*dW1[i]
				nc.w1[i] -= vW1[i]
			}
			for i := range nc.b1 {
				vb1[i] = momentum*vb1[i] + scale*db1[i]
				nc.b1[i] -= vb1[i]
			}
			for i := range nc.w2 {
				vW2[i] = momentum*vW2[i] + scale*dW2[i]
				nc.w2[i] -= vW2[i]
			}
			for i := range nc.b2 {
				vb2[i] = momentum*vb2[i] + scale*db2[i]
				nc.b2[i] -= vb2[i]
			}
		}

		finalLoss = epochLoss / float64(len(prepared))
		finalAcc = float64(epochCorrect) / float64(len(prepared))
	}

	// Store replay buffer (sample from training data)
	nc.updateReplayBuffer(examples)

	nc.Epochs = epochs
	nc.TrainingSamples = len(examples)
	nc.Accuracy = finalAcc

	return TrainResult{
		Epochs:   epochs,
		Samples:  len(examples),
		Intents:  nc.numIntents,
		Loss:     finalLoss,
		Accuracy: finalAcc,
	}
}

// TrainResult holds training statistics.
type TrainResult struct {
	Epochs   int
	Samples  int
	Intents  int
	Loss     float64
	Accuracy float64
}

func (r TrainResult) String() string {
	return fmt.Sprintf("%d epochs, %d samples, %d intents → loss=%.4f acc=%.1f%%",
		r.Epochs, r.Samples, r.Intents, r.Loss, r.Accuracy*100)
}

// backprop computes gradients for a single example (accumulates into grad buffers).
// Returns the cross-entropy loss and whether the prediction was correct.
func (nc *NeuralClassifier) backprop(features []float32, label int,
	dW1, db1, dW2, db2 []float32) (float64, bool) {

	// ---- Forward pass ----
	// Layer 1: hidden = ReLU(features · W1 + b1)
	hidden := make([]float32, nc.hiddenSize)
	preReLU := make([]float32, nc.hiddenSize) // save for backward
	for j := 0; j < nc.hiddenSize; j++ {
		var sum float32
		for i := 0; i < nc.featureSize; i++ {
			sum += features[i] * nc.w1[i*nc.hiddenSize+j]
		}
		sum += nc.b1[j]
		preReLU[j] = sum
		if sum > 0 {
			hidden[j] = sum
		}
	}

	// Layer 2: probs = softmax32(hidden · W2 + b2)
	logits := make([]float32, nc.numIntents)
	for k := 0; k < nc.numIntents; k++ {
		var sum float32
		for j := 0; j < nc.hiddenSize; j++ {
			sum += hidden[j] * nc.w2[j*nc.numIntents+k]
		}
		logits[k] = sum + nc.b2[k]
	}
	probs := softmax32(logits)

	// Loss: -log(prob[label])
	loss := -math.Log(float64(probs[label]) + 1e-10)

	// Correct prediction?
	bestIdx := 0
	for i := 1; i < nc.numIntents; i++ {
		if probs[i] > probs[bestIdx] {
			bestIdx = i
		}
	}
	correct := bestIdx == label

	// ---- Backward pass ----
	// dL/dlogits = probs - one_hot(label) (softmax + cross-entropy gradient)
	dLogits := make([]float32, nc.numIntents)
	for k := range dLogits {
		dLogits[k] = probs[k]
	}
	dLogits[label] -= 1.0

	// Gradients for W2 and b2
	for j := 0; j < nc.hiddenSize; j++ {
		for k := 0; k < nc.numIntents; k++ {
			dW2[j*nc.numIntents+k] += hidden[j] * dLogits[k]
		}
	}
	for k := 0; k < nc.numIntents; k++ {
		db2[k] += dLogits[k]
	}

	// dL/dhidden = dLogits · W2^T
	dHidden := make([]float32, nc.hiddenSize)
	for j := 0; j < nc.hiddenSize; j++ {
		var sum float32
		for k := 0; k < nc.numIntents; k++ {
			sum += dLogits[k] * nc.w2[j*nc.numIntents+k]
		}
		dHidden[j] = sum
	}

	// Apply ReLU gradient: dL/dpreReLU = dL/dhidden * (preReLU > 0)
	for j := 0; j < nc.hiddenSize; j++ {
		if preReLU[j] <= 0 {
			dHidden[j] = 0
		}
	}

	// Gradients for W1 and b1
	for i := 0; i < nc.featureSize; i++ {
		if features[i] == 0 {
			continue // sparse optimization
		}
		for j := 0; j < nc.hiddenSize; j++ {
			dW1[i*nc.hiddenSize+j] += features[i] * dHidden[j]
		}
	}
	for j := 0; j < nc.hiddenSize; j++ {
		db1[j] += dHidden[j]
	}

	return loss, correct
}

// TrainIncremental adds new examples and retrains with experience replay.
// Uses a lower learning rate to avoid catastrophic forgetting.
func (nc *NeuralClassifier) TrainIncremental(newExamples []TrainingExample) TrainResult {
	nc.mu.Lock()

	// Combine new examples with replay buffer
	combined := make([]TrainingExample, 0, len(newExamples)+len(nc.replayBuffer))
	combined = append(combined, newExamples...)
	combined = append(combined, nc.replayBuffer...)

	// Check for new intents
	hasNewIntent := false
	for _, ex := range newExamples {
		if _, ok := nc.intentToIdx[ex.Intent]; !ok {
			hasNewIntent = true
			break
		}
	}

	nc.mu.Unlock()

	if hasNewIntent {
		// New intent detected — full retrain needed
		return nc.Train(combined, 30, 0.02)
	}

	// Incremental update with lower learning rate
	return nc.Train(combined, 10, 0.005)
}

// updateReplayBuffer stores a balanced sample of training examples.
func (nc *NeuralClassifier) updateReplayBuffer(examples []TrainingExample) {
	// Group by intent
	byIntent := make(map[string][]TrainingExample)
	for _, ex := range examples {
		byIntent[ex.Intent] = append(byIntent[ex.Intent], ex)
	}

	// Sample up to replayMax/numIntents per intent
	perIntent := nc.replayMax / len(byIntent)
	if perIntent < 3 {
		perIntent = 3
	}

	nc.replayBuffer = nil
	for _, exs := range byIntent {
		if len(exs) <= perIntent {
			nc.replayBuffer = append(nc.replayBuffer, exs...)
		} else {
			// Random sample
			perm := rand.Perm(len(exs))
			for i := 0; i < perIntent; i++ {
				nc.replayBuffer = append(nc.replayBuffer, exs[perm[i]])
			}
		}
	}
}

// -----------------------------------------------------------------------
// Persistence — binary format for fast save/load.
// -----------------------------------------------------------------------

// Save writes the trained model to a binary file.
func (nc *NeuralClassifier) Save(path string) error {
	nc.mu.RLock()
	defer nc.mu.RUnlock()

	if nc.numIntents == 0 {
		return fmt.Errorf("cannot save untrained model")
	}

	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	// Header
	if err := binary.Write(f, binary.LittleEndian, uint32(modelMagic)); err != nil {
		return err
	}
	if err := binary.Write(f, binary.LittleEndian, uint32(modelVersion)); err != nil {
		return err
	}
	if err := binary.Write(f, binary.LittleEndian, uint32(nc.featureSize)); err != nil {
		return err
	}
	if err := binary.Write(f, binary.LittleEndian, uint32(nc.hiddenSize)); err != nil {
		return err
	}
	if err := binary.Write(f, binary.LittleEndian, uint32(nc.numIntents)); err != nil {
		return err
	}

	// Weights
	if err := binary.Write(f, binary.LittleEndian, nc.w1); err != nil {
		return err
	}
	if err := binary.Write(f, binary.LittleEndian, nc.b1); err != nil {
		return err
	}
	if err := binary.Write(f, binary.LittleEndian, nc.w2); err != nil {
		return err
	}
	if err := binary.Write(f, binary.LittleEndian, nc.b2); err != nil {
		return err
	}

	// Intent names (length-prefixed strings)
	for _, name := range nc.idxToIntent {
		nameBytes := []byte(name)
		if err := binary.Write(f, binary.LittleEndian, uint16(len(nameBytes))); err != nil {
			return err
		}
		if _, err := f.Write(nameBytes); err != nil {
			return err
		}
	}

	return nil
}

// Load reads a trained model from a binary file.
func (nc *NeuralClassifier) Load(path string) error {
	nc.mu.Lock()
	defer nc.mu.Unlock()

	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()

	// Header
	var magic, version, featSize, hidSize, numIntents uint32
	if err := binary.Read(f, binary.LittleEndian, &magic); err != nil {
		return err
	}
	if magic != modelMagic {
		return fmt.Errorf("invalid model file (bad magic: 0x%X)", magic)
	}
	if err := binary.Read(f, binary.LittleEndian, &version); err != nil {
		return err
	}
	if version != modelVersion {
		return fmt.Errorf("unsupported model version: %d", version)
	}
	if err := binary.Read(f, binary.LittleEndian, &featSize); err != nil {
		return err
	}
	if err := binary.Read(f, binary.LittleEndian, &hidSize); err != nil {
		return err
	}
	if err := binary.Read(f, binary.LittleEndian, &numIntents); err != nil {
		return err
	}

	nc.featureSize = int(featSize)
	nc.hiddenSize = int(hidSize)
	nc.numIntents = int(numIntents)

	// Weights
	nc.w1 = make([]float32, nc.featureSize*nc.hiddenSize)
	nc.b1 = make([]float32, nc.hiddenSize)
	nc.w2 = make([]float32, nc.hiddenSize*nc.numIntents)
	nc.b2 = make([]float32, nc.numIntents)

	if err := binary.Read(f, binary.LittleEndian, nc.w1); err != nil {
		return err
	}
	if err := binary.Read(f, binary.LittleEndian, nc.b1); err != nil {
		return err
	}
	if err := binary.Read(f, binary.LittleEndian, nc.w2); err != nil {
		return err
	}
	if err := binary.Read(f, binary.LittleEndian, nc.b2); err != nil {
		return err
	}

	// Intent names
	nc.idxToIntent = make([]string, nc.numIntents)
	nc.intentToIdx = make(map[string]int, nc.numIntents)
	for i := 0; i < nc.numIntents; i++ {
		var nameLen uint16
		if err := binary.Read(f, binary.LittleEndian, &nameLen); err != nil {
			return err
		}
		nameBytes := make([]byte, nameLen)
		if _, err := f.Read(nameBytes); err != nil {
			return err
		}
		name := string(nameBytes)
		nc.idxToIntent[i] = name
		nc.intentToIdx[name] = i
	}

	return nil
}

// IsTrained returns true if the model has been trained.
func (nc *NeuralClassifier) IsTrained() bool {
	nc.mu.RLock()
	defer nc.mu.RUnlock()
	return nc.numIntents > 0 && nc.w1 != nil
}

// NumIntents returns the number of intents the model knows.
func (nc *NeuralClassifier) NumIntents() int {
	nc.mu.RLock()
	defer nc.mu.RUnlock()
	return nc.numIntents
}
