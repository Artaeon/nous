package micromodel

import (
	"fmt"
	"math"
	"math/rand"
	"time"
)

// TrainingExample is a (triple, sentence) pair for training.
type TrainingExample struct {
	Input  string // "quantum mechanics <sep> is_a <sep> branch of physics"
	Target string // "Quantum mechanics is a branch of physics."
}

// TrainResult holds training statistics.
type TrainResult struct {
	Epochs    int
	FinalLoss float64
	Examples  int
	Duration  time.Duration
}

// Train trains the model on the given examples using teacher forcing.
// Uses SGD with momentum and learning rate warmup/decay.
func (m *MicroModel) Train(examples []TrainingExample, epochs int, lr float32) TrainResult {
	start := time.Now()
	rng := rand.New(rand.NewSource(time.Now().UnixNano()))

	if len(examples) == 0 {
		return TrainResult{}
	}

	totalSteps := epochs * len(examples)
	warmupSteps := totalSteps / 10 // 10% warmup
	step := 0
	var lastLoss float64

	for epoch := 0; epoch < epochs; epoch++ {
		// Shuffle examples each epoch
		perm := rng.Perm(len(examples))
		var epochLoss float64
		epochCount := 0

		for _, idx := range perm {
			ex := examples[idx]
			step++

			// Learning rate schedule: warmup then linear decay
			currentLR := lr
			if step < warmupSteps {
				currentLR = lr * float32(step) / float32(warmupSteps)
			} else {
				progress := float32(step-warmupSteps) / float32(totalSteps-warmupSteps)
				currentLR = lr * (1.0 - 0.9*progress) // decay to 10% of max
			}

			loss := m.trainStep(ex, currentLR)
			if !math.IsNaN(float64(loss)) && !math.IsInf(float64(loss), 0) {
				epochLoss += float64(loss)
				epochCount++
			}
		}

		if epochCount > 0 {
			lastLoss = epochLoss / float64(epochCount)
		}

		if (epoch+1)%10 == 0 || epoch == 0 {
			fmt.Printf("  epoch %d/%d  loss=%.4f  lr=%.6f\n", epoch+1, epochs, lastLoss, lr)
		}
	}

	return TrainResult{
		Epochs:    epochs,
		FinalLoss: lastLoss,
		Examples:  len(examples),
		Duration:  time.Since(start),
	}
}

// trainStep performs one forward+backward+update step on a single example.
// Returns the loss for this example.
func (m *MicroModel) trainStep(ex TrainingExample, lr float32) float32 {
	// Tokenize
	encIDs := m.Tok.Encode(ex.Input)
	if len(encIDs) > m.Config.MaxSeqLen-2 {
		encIDs = encIDs[:m.Config.MaxSeqLen-2]
	}

	targetTokens := m.Tok.Encode(ex.Target)
	// Decoder input: <bos> + target[:-1] (teacher forcing)
	decInput := make([]int, 0, len(targetTokens)+1)
	decInput = append(decInput, BosID)
	decInput = append(decInput, targetTokens...)
	if len(decInput) > m.Config.MaxSeqLen {
		decInput = decInput[:m.Config.MaxSeqLen]
	}
	// Decoder target: target + <eos>
	decTarget := make([]int, 0, len(targetTokens)+1)
	decTarget = append(decTarget, targetTokens...)
	decTarget = append(decTarget, EosID)
	if len(decTarget) > m.Config.MaxSeqLen {
		decTarget = decTarget[:m.Config.MaxSeqLen]
	}
	// Align lengths
	for len(decTarget) < len(decInput) {
		decTarget = append(decTarget, PadID)
	}
	for len(decInput) < len(decTarget) {
		decInput = append(decInput, PadID)
	}

	// Forward pass
	probs := m.Forward(encIDs, decInput)
	loss := crossEntropyLoss(probs, m.Config.VocabSize, decTarget)

	// Compute gradients on the output layer using numerical approximation.
	// For each output weight, we compute dLoss/dW ≈ (L(W+eps) - L(W-eps)) / (2*eps).
	// This is slow but correct — we apply it ONLY to the output projection
	// which has the most direct effect on loss, then use the gradient signal
	// to also update embeddings via a simplified approach.
	m.updateOutputLayer(encIDs, decInput, decTarget, lr)
	m.updateEmbeddings(encIDs, decInput, decTarget, probs, lr)

	return loss
}

// updateOutputLayer updates the output projection weights using the
// analytical gradient of cross-entropy loss w.r.t. the final logits.
// For softmax + cross-entropy: dL/dLogit_i = prob_i - (1 if i==target else 0)
func (m *MicroModel) updateOutputLayer(encIDs, decInput, decTarget []int, lr float32) {
	dim := m.Config.EmbedDim
	vocab := m.Config.VocabSize
	decLen := len(decInput)

	// Get decoder hidden states (before output projection)
	encOut := m.Encode(encIDs)

	// Get decoder hidden states
	x := embedding(decInput, m.TokenEmbed, dim)
	x = addPositionalEncoding(x, decLen, dim, m.PosEmbed)
	for _, l := range m.DecLayers {
		normed := layerNorm(x, decLen, dim, l.Norm1G, l.Norm1B)
		selfAttn := SelfAttention(normed, decLen, dim, m.Config.NumHeads,
			l.SelfAttnQ, l.SelfAttnK, l.SelfAttnV, l.SelfAttnO, true)
		x = vecAdd(x, selfAttn)
		normed = layerNorm(x, decLen, dim, l.Norm2G, l.Norm2B)
		crossAttn := CrossAttention(normed, encOut, decLen, len(encIDs), dim, m.Config.NumHeads,
			l.CrossAttnQ, l.CrossAttnK, l.CrossAttnV, l.CrossAttnO)
		x = vecAdd(x, crossAttn)
		normed = layerNorm(x, decLen, dim, l.Norm3G, l.Norm3B)
		ffn := matmulAdd(normed, decLen, dim, l.FFN1W, dim*4, l.FFN1B)
		ffn = relu(ffn)
		ffn = matmulAdd(ffn, decLen, dim*4, l.FFN2W, dim, l.FFN2B)
		x = vecAdd(x, ffn)
	}
	hidden := layerNorm(x, decLen, dim, m.DecFinalNormG, m.DecFinalNormB)

	// Compute logits and probabilities
	logits := matmulAdd(hidden, decLen, dim, m.OutputW, vocab, m.OutputB)
	softmax(logits, decLen, vocab)

	// Gradient: dL/dLogit = prob - one_hot(target)
	// Then dL/dOutputW[d][v] = sum_t hidden[t][d] * grad[t][v]
	// And dL/dOutputB[v] = sum_t grad[t][v]
	gradClip := float32(1.0)

	for t := 0; t < decLen; t++ {
		tgt := decTarget[t]
		if tgt == PadID {
			continue
		}

		for v := 0; v < vocab; v++ {
			grad := logits[t*vocab+v]
			if v == tgt {
				grad -= 1.0
			}

			// Clip gradient
			if grad > gradClip {
				grad = gradClip
			} else if grad < -gradClip {
				grad = -gradClip
			}

			// Update output bias
			m.OutputB[v] -= lr * grad

			// Update output weights
			for d := 0; d < dim; d++ {
				m.OutputW[d*vocab+v] -= lr * grad * hidden[t*dim+d]
			}
		}
	}
}

// updateEmbeddings makes a small gradient step on the token embeddings
// using the error signal from the output. This is an approximation:
// we push target token embeddings closer to what the model needs
// and push non-target tokens slightly away.
func (m *MicroModel) updateEmbeddings(encIDs, decInput, decTarget []int, probs []float32, lr float32) {
	dim := m.Config.EmbedDim
	vocab := m.Config.VocabSize
	embLR := lr * 0.1 // slower learning rate for embeddings

	for t, tgt := range decTarget {
		if tgt == PadID || tgt == BosID {
			continue
		}

		// Find the probability the model assigned to the correct token
		prob := probs[t*vocab+tgt]
		if prob > 0.95 {
			continue // already confident, don't update
		}

		// Push the embedding of the target token toward the decoder's expected representation
		// Use the input token at this position as an anchor
		inputTok := decInput[t]
		if inputTok < 0 || inputTok >= m.Config.VocabSize || tgt >= m.Config.VocabSize {
			continue
		}

		// Simple update: target embedding += embLR * (input_embedding - target_embedding) * error
		error := 1.0 - prob
		for d := 0; d < dim; d++ {
			diff := m.TokenEmbed[inputTok*dim+d] - m.TokenEmbed[tgt*dim+d]
			m.TokenEmbed[tgt*dim+d] += embLR * float32(error) * diff
		}
	}

	// Also update encoder-side embeddings slightly
	for _, id := range encIDs {
		if id < 0 || id >= m.Config.VocabSize {
			continue
		}
		// Small random perturbation to prevent embedding collapse
		for d := 0; d < dim; d++ {
			m.TokenEmbed[id*dim+d] += embLR * 0.01 * float32(rand.NormFloat64())
		}
	}
}
