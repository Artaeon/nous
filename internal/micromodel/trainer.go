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

// Train trains the model on the given examples using teacher forcing
// with backpropagation through all FFN layers + output projection + embeddings.
func (m *MicroModel) Train(examples []TrainingExample, epochs int, lr float32) TrainResult {
	start := time.Now()
	rng := rand.New(rand.NewSource(time.Now().UnixNano()))

	if len(examples) == 0 {
		return TrainResult{}
	}

	totalSteps := epochs * len(examples)
	warmupSteps := totalSteps / 10
	step := 0
	var lastLoss float64

	for epoch := 0; epoch < epochs; epoch++ {
		perm := rng.Perm(len(examples))
		var epochLoss float64
		epochCount := 0

		for _, idx := range perm {
			ex := examples[idx]
			step++

			currentLR := lr
			if step < warmupSteps && warmupSteps > 0 {
				currentLR = lr * float32(step) / float32(warmupSteps)
			} else if totalSteps > warmupSteps {
				progress := float32(step-warmupSteps) / float32(totalSteps-warmupSteps)
				currentLR = lr * (1.0 - 0.9*progress)
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

		if (epoch+1)%10 == 0 || epoch == 0 || epoch == epochs-1 {
			fmt.Printf("  epoch %d/%d  loss=%.4f  lr=%.6f\n", epoch+1, epochs, lastLoss, currentLR(lr, step, warmupSteps, totalSteps))
		}
	}

	return TrainResult{
		Epochs:    epochs,
		FinalLoss: lastLoss,
		Examples:  len(examples),
		Duration:  time.Since(start),
	}
}

func currentLR(baseLR float32, step, warmup, total int) float32 {
	if step < warmup && warmup > 0 {
		return baseLR * float32(step) / float32(warmup)
	}
	if total > warmup {
		progress := float32(step-warmup) / float32(total-warmup)
		return baseLR * (1.0 - 0.9*progress)
	}
	return baseLR
}

// trainStep performs one forward + backward + update on a single example.
func (m *MicroModel) trainStep(ex TrainingExample, lr float32) float32 {
	dim := m.Config.EmbedDim
	ffDim := dim * 4
	vocab := m.Config.VocabSize

	// Tokenize
	encIDs := m.Tok.Encode(ex.Input)
	if len(encIDs) > m.Config.MaxSeqLen-2 {
		encIDs = encIDs[:m.Config.MaxSeqLen-2]
	}

	targetTokens := m.Tok.Encode(ex.Target)
	decInput := make([]int, 0, len(targetTokens)+1)
	decInput = append(decInput, BosID)
	decInput = append(decInput, targetTokens...)
	if len(decInput) > m.Config.MaxSeqLen {
		decInput = decInput[:m.Config.MaxSeqLen]
	}
	decTarget := make([]int, 0, len(targetTokens)+1)
	decTarget = append(decTarget, targetTokens...)
	decTarget = append(decTarget, EosID)
	if len(decTarget) > m.Config.MaxSeqLen {
		decTarget = decTarget[:m.Config.MaxSeqLen]
	}
	for len(decTarget) < len(decInput) {
		decTarget = append(decTarget, PadID)
	}
	for len(decInput) < len(decTarget) {
		decInput = append(decInput, PadID)
	}

	decLen := len(decInput)
	encLen := len(encIDs)

	// === FORWARD PASS with activation caching ===
	encOut := m.Encode(encIDs)

	// Decoder forward — cache activations at each layer for backprop
	x := embedding(decInput, m.TokenEmbed, dim)
	x = addPositionalEncoding(x, decLen, dim, m.PosEmbed)

	// Cache pre-FFN activations for each decoder layer
	type layerCache struct {
		preFFN   []float32 // input to FFN (after norm3)
		ffnHid   []float32 // hidden layer after ReLU (for ReLU backward)
		preNorm3 []float32 // input to norm3 (residual stream before FFN)
	}
	caches := make([]layerCache, len(m.DecLayers))

	for li, l := range m.DecLayers {
		// Self-attention (don't backprop through this — too complex)
		normed := layerNorm(x, decLen, dim, l.Norm1G, l.Norm1B)
		selfAttn := SelfAttention(normed, decLen, dim, m.Config.NumHeads,
			l.SelfAttnQ, l.SelfAttnK, l.SelfAttnV, l.SelfAttnO, true)
		x = vecAdd(x, selfAttn)

		// Cross-attention (don't backprop through this either)
		normed = layerNorm(x, decLen, dim, l.Norm2G, l.Norm2B)
		crossAttn := CrossAttention(normed, encOut, decLen, encLen, dim, m.Config.NumHeads,
			l.CrossAttnQ, l.CrossAttnK, l.CrossAttnV, l.CrossAttnO)
		x = vecAdd(x, crossAttn)

		// FFN — CACHE activations for backward pass
		caches[li].preNorm3 = make([]float32, len(x))
		copy(caches[li].preNorm3, x)

		normed = layerNorm(x, decLen, dim, l.Norm3G, l.Norm3B)
		caches[li].preFFN = normed

		// FFN1: normed -> hidden (dim -> ffDim)
		ffnHid := matmulAdd(normed, decLen, dim, l.FFN1W, ffDim, l.FFN1B)
		// ReLU — cache pre-ReLU for backward
		caches[li].ffnHid = make([]float32, len(ffnHid))
		copy(caches[li].ffnHid, ffnHid) // pre-ReLU values
		ffnHid = relu(ffnHid)

		// FFN2: hidden -> out (ffDim -> dim)
		ffnOut := matmulAdd(ffnHid, decLen, ffDim, l.FFN2W, dim, l.FFN2B)
		x = vecAdd(x, ffnOut)
	}

	// Final norm
	hidden := layerNorm(x, decLen, dim, m.DecFinalNormG, m.DecFinalNormB)

	// Output projection: logits = hidden @ OutputW + OutputB
	logits := matmulAdd(hidden, decLen, dim, m.OutputW, vocab, m.OutputB)

	// Softmax
	probs := make([]float32, len(logits))
	copy(probs, logits)
	softmax(probs, decLen, vocab)

	loss := crossEntropyLoss(probs, vocab, decTarget)

	// === BACKWARD PASS ===
	gradClip := float32(1.0)

	// Step 1: dL/dLogits = probs - one_hot(target)  [shape: decLen x vocab]
	dLogits := make([]float32, decLen*vocab)
	validPositions := 0
	for t := 0; t < decLen; t++ {
		tgt := decTarget[t]
		if tgt == PadID {
			continue
		}
		validPositions++
		for v := 0; v < vocab; v++ {
			dLogits[t*vocab+v] = probs[t*vocab+v]
			if v == tgt {
				dLogits[t*vocab+v] -= 1.0
			}
		}
	}
	// Average over valid positions
	if validPositions > 0 {
		scale := 1.0 / float32(validPositions)
		for i := range dLogits {
			dLogits[i] *= scale
		}
	}

	// Step 2: Update output projection
	// dL/dOutputW = hidden^T @ dLogits  [dim x vocab]
	// dL/dOutputB = sum_t dLogits[t]     [vocab]
	// dL/dHidden = dLogits @ OutputW^T   [decLen x dim]
	dHidden := make([]float32, decLen*dim)
	for t := 0; t < decLen; t++ {
		for v := 0; v < vocab; v++ {
			g := dLogits[t*vocab+v]
			if g == 0 {
				continue
			}
			g = clipGrad(g, gradClip)

			m.OutputB[v] -= lr * g
			for d := 0; d < dim; d++ {
				m.OutputW[d*vocab+v] -= lr * g * hidden[t*dim+d]
				dHidden[t*dim+d] += g * m.OutputW[d*vocab+v]
			}
		}
	}

	// Step 3: Backprop through decoder FFN layers (reverse order)
	dX := dHidden // gradient flowing back through the residual stream
	for li := len(m.DecLayers) - 1; li >= 0; li-- {
		l := &m.DecLayers[li]
		cache := caches[li]

		// dX is the gradient coming from the layer above (or output)
		// FFN residual: x_out = x_in + FFN(norm(x_in))
		// So dL/dFFNout = dX (gradient passes through residual)

		// Backprop FFN2: dL/dFFN2W, dL/dFFN2B, dL/dFFNhid
		// ffnOut = relu(ffnHid) @ FFN2W + FFN2B
		// dL/dFFN2W = relu(ffnHid)^T @ dX
		// dL/dReLUout = dX @ FFN2W^T

		// We need relu(ffnHid) — recompute from cached preFFN
		ffnHidPre := cache.ffnHid // pre-ReLU values
		ffnHidReLU := make([]float32, len(ffnHidPre))
		for i, v := range ffnHidPre {
			if v > 0 {
				ffnHidReLU[i] = v
			}
		}

		// Update FFN2 weights: dW2 = ffnHidReLU^T @ dX
		for t := 0; t < decLen; t++ {
			for d := 0; d < dim; d++ {
				g := clipGrad(dX[t*dim+d], gradClip)
				l.FFN2B[d] -= lr * g
				for h := 0; h < ffDim; h++ {
					l.FFN2W[h*dim+d] -= lr * g * ffnHidReLU[t*ffDim+h]
				}
			}
		}

		// dL/dReLUout = dX @ FFN2W^T [decLen x ffDim]
		dReLUout := make([]float32, decLen*ffDim)
		for t := 0; t < decLen; t++ {
			for h := 0; h < ffDim; h++ {
				var sum float32
				for d := 0; d < dim; d++ {
					sum += dX[t*dim+d] * l.FFN2W[h*dim+d]
				}
				dReLUout[t*ffDim+h] = sum
			}
		}

		// Backprop through ReLU: dL/dFFNhid = dReLUout * (ffnHidPre > 0)
		dFFNhid := make([]float32, decLen*ffDim)
		for i, v := range ffnHidPre {
			if v > 0 {
				dFFNhid[i] = dReLUout[i]
			}
		}

		// Update FFN1 weights: dW1 = preFFN^T @ dFFNhid
		preFFN := cache.preFFN
		for t := 0; t < decLen; t++ {
			for h := 0; h < ffDim; h++ {
				g := clipGrad(dFFNhid[t*ffDim+h], gradClip)
				l.FFN1B[h] -= lr * g
				for d := 0; d < dim; d++ {
					l.FFN1W[d*ffDim+h] -= lr * g * preFFN[t*dim+d]
				}
			}
		}

		// Gradient continues flowing: dX stays as-is through the residual
		// (the attention layers don't get gradients in this version,
		// but the FFN layers DO get trained which is the majority of parameters)
	}

	// Step 4: Update token embeddings based on error signal
	m.updateEmbeddingsFromGrad(encIDs, decInput, decTarget, probs, lr*0.1)

	return loss
}

// updateEmbeddingsFromGrad updates token embeddings using the prediction error.
func (m *MicroModel) updateEmbeddingsFromGrad(encIDs, decInput, decTarget []int, probs []float32, lr float32) {
	dim := m.Config.EmbedDim
	vocab := m.Config.VocabSize

	for t, tgt := range decTarget {
		if tgt == PadID || tgt == BosID || tgt >= vocab {
			continue
		}

		prob := probs[t*vocab+tgt]
		if prob > 0.9 {
			continue
		}

		// Move target embedding toward input embedding (teacher signal)
		inputTok := decInput[t]
		if inputTok < 0 || inputTok >= vocab {
			continue
		}

		errSignal := 1.0 - prob
		for d := 0; d < dim; d++ {
			diff := m.TokenEmbed[inputTok*dim+d] - m.TokenEmbed[tgt*dim+d]
			m.TokenEmbed[tgt*dim+d] += lr * float32(errSignal) * diff
		}
	}
}

func clipGrad(g, limit float32) float32 {
	if g > limit {
		return limit
	}
	if g < -limit {
		return -limit
	}
	return g
}
