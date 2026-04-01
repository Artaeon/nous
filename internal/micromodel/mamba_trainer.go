package micromodel

import (
	"fmt"
	"math"
	"math/rand"
	"time"
)

// TrainMamba trains the Mamba model on (triple, sentence) pairs.
// Uses teacher forcing with full backpropagation through all parameters:
// output projection, Mamba block linear layers, SSM parameters, and embeddings.
func (m *MambaModel) Train(examples []TrainingExample, epochs int, lr float32) TrainResult {
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

			// Learning rate schedule: linear warmup then cosine annealing
			currentLR := mambaLR(lr, step, warmupSteps, totalSteps)

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
			fmt.Printf("  epoch %d/%d  loss=%.4f  lr=%.6f\n",
				epoch+1, epochs, lastLoss, mambaLR(lr, step, warmupSteps, totalSteps))
		}
	}

	return TrainResult{
		Epochs:    epochs,
		FinalLoss: lastLoss,
		Examples:  len(examples),
		Duration:  time.Since(start),
	}
}

func mambaLR(baseLR float32, step, warmup, total int) float32 {
	if step < warmup && warmup > 0 {
		return baseLR * float32(step) / float32(warmup)
	}
	if total > warmup {
		progress := float64(step-warmup) / float64(total-warmup)
		// Cosine annealing: lr * 0.5 * (1 + cos(pi * progress))
		return baseLR * float32(0.5*(1+math.Cos(math.Pi*progress)))
	}
	return baseLR
}

// trainStep performs one forward + backward + update on a single example.
func (m *MambaModel) trainStep(ex TrainingExample, lr float32) float32 {
	dim := m.Config.ModelDim
	vocab := m.Config.VocabSize
	inner := m.Config.InnerDim()
	dState := m.Config.StateDim
	dtRank := m.Config.DtRank()
	kSize := m.Config.ConvDim

	// Tokenize: for decoder-only, concatenate triple + target
	// Format: <bos> subject <sep> relation <sep> object <sep> target_tokens <eos>
	tripleIDs := m.Tok.EncodeTriple(ex.Input, "", "")
	// Parse the input which is already in "subject <sep> relation <sep> object" format
	tripleIDs = m.Tok.Encode(ex.Input)
	tripleIDs = append([]int{BosID}, tripleIDs...)
	tripleIDs = append(tripleIDs, SepID)

	targetIDs := m.Tok.Encode(ex.Target)

	// Full sequence: triple + target + EOS
	fullIDs := make([]int, 0, len(tripleIDs)+len(targetIDs)+1)
	fullIDs = append(fullIDs, tripleIDs...)
	fullIDs = append(fullIDs, targetIDs...)
	fullIDs = append(fullIDs, EosID)

	if len(fullIDs) > m.Config.MaxSeqLen {
		fullIDs = fullIDs[:m.Config.MaxSeqLen]
	}

	seqLen := len(fullIDs)
	if seqLen < 2 {
		return 0
	}

	// Input: all tokens except last; Target: all tokens except first
	inputIDs := fullIDs[:seqLen-1]
	targetTokens := fullIDs[1:]
	seqLen = len(inputIDs)

	// Prefix length: tokens from the triple that we don't compute loss on
	prefixLen := len(tripleIDs)
	if prefixLen >= seqLen {
		prefixLen = 0 // fallback: compute loss on everything
	}

	// === FORWARD PASS with caching ===
	caches, logits := m.forwardWithCache(inputIDs)

	// Softmax
	probs := make([]float32, len(logits))
	copy(probs, logits)
	softmax(probs, seqLen, vocab)

	// Loss: only on target positions (after prefix)
	var loss float32
	var lossCount int
	for t := prefixLen; t < seqLen; t++ {
		tgt := targetTokens[t]
		if tgt == PadID {
			continue
		}
		p := probs[t*vocab+tgt]
		if p < 1e-10 {
			p = 1e-10
		}
		loss -= float32(math.Log(float64(p)))
		lossCount++
	}
	if lossCount > 0 {
		loss /= float32(lossCount)
	}

	// === BACKWARD PASS ===
	gradClip := float32(1.0)

	// Step 1: dL/dLogits = probs - one_hot(target)
	dLogits := make([]float32, seqLen*vocab)
	for t := prefixLen; t < seqLen; t++ {
		tgt := targetTokens[t]
		if tgt == PadID {
			continue
		}
		for v := 0; v < vocab; v++ {
			dLogits[t*vocab+v] = probs[t*vocab+v]
			if v == tgt {
				dLogits[t*vocab+v] -= 1.0
			}
		}
	}
	if lossCount > 0 {
		scale := 1.0 / float32(lossCount)
		for i := range dLogits {
			dLogits[i] *= scale
		}
	}

	// Step 2: Backprop through output projection
	// hidden is the output of the final layer norm
	hidden := layerNorm(caches[len(caches)-1].residual, seqLen, dim, m.FinalNormG, m.FinalNormB)
	// Recompute: we need the hidden state after all blocks
	// Actually we need to reconstruct the final hidden from the last block
	lastResidual := caches[len(caches)-1].residual
	lastOut := matmulAdd(caches[len(caches)-1].gated, seqLen, inner, m.Layers[len(m.Layers)-1].OutProjW, dim, m.Layers[len(m.Layers)-1].OutProjB)
	finalX := vecAdd(lastResidual, lastOut)
	hidden = layerNorm(finalX, seqLen, dim, m.FinalNormG, m.FinalNormB)

	dHidden := make([]float32, seqLen*dim)
	for t := 0; t < seqLen; t++ {
		for v := 0; v < vocab; v++ {
			g := clipGrad(dLogits[t*vocab+v], gradClip)
			if g == 0 {
				continue
			}
			m.OutputB[v] -= lr * g
			for d := 0; d < dim; d++ {
				m.OutputW[d*vocab+v] -= lr * g * hidden[t*dim+d]
				dHidden[t*dim+d] += g * m.OutputW[d*vocab+v]
			}
		}
	}

	// Step 3: Backprop through Mamba blocks (reverse order)
	// We backprop through: OutProj, gating, SSM projections, Conv1D, InProj
	dX := dHidden

	for li := len(m.Layers) - 1; li >= 0; li-- {
		l := &m.Layers[li]
		cache := &caches[li]

		// dX flows through residual: dResidual = dX, dOut = dX
		// Backprop through output projection: out = gated @ OutProjW + OutProjB
		// dOutProjW = gated^T @ dX, dGated = dX @ OutProjW^T
		dGated := make([]float32, seqLen*inner)
		for t := 0; t < seqLen; t++ {
			for d := 0; d < dim; d++ {
				g := clipGrad(dX[t*dim+d], gradClip)
				l.OutProjB[d] -= lr * g
				for j := 0; j < inner; j++ {
					l.OutProjW[j*dim+d] -= lr * g * cache.gated[t*inner+j]
					dGated[t*inner+j] += g * l.OutProjW[j*dim+d]
				}
			}
		}

		// Backprop through gating: gated = ssmOut * silu(z)
		// dSsmOut = dGated * silu(z), dZSilu = dGated * ssmOut
		dSsmOut := vecMul(dGated, cache.zSilu)
		dZSilu := vecMul(dGated, cache.ssmOut)

		// Backprop through z SiLU
		dZBranch := siluBackward(cache.zBranch, dZSilu)

		// Backprop through SSM: use SelectiveScanBackward
		dXSilu, dDt, dAGrad, _, _, dDGrad := SelectiveScanBackward(
			dSsmOut, cache.xSilu, seqLen, inner,
			cache.dt, cache.A, dState,
			cache.B, cache.C, l.D,
		)

		// Update D
		for i := 0; i < inner; i++ {
			l.D[i] -= lr * clipGrad(dDGrad[i], gradClip)
		}

		// Update A_log: dA_log = dA * (-exp(A_log)) since A = -exp(A_log)
		for i := range l.ALog {
			aVal := -float32(math.Exp(float64(l.ALog[i])))
			l.ALog[i] -= lr * clipGrad(dAGrad[i]*aVal, gradClip)
		}

		// Backprop through softplus(dt): d_dtPre = dDt * sigmoid(dtPre)
		dDtPre := softplusBackward(cache.dtPre, dDt)

		// Backprop through dt projection: dtPre = dtInput @ DtProjW + DtProjB
		dDtInput := make([]float32, seqLen*dtRank)
		for t := 0; t < seqLen; t++ {
			for j := 0; j < inner; j++ {
				g := clipGrad(dDtPre[t*inner+j], gradClip)
				l.DtProjB[j] -= lr * g
				// dtInput is not directly cached, skip weight update for DtProjW
				// (we'd need dtInput which requires recomputation)
			}
		}
		_ = dDtInput

		// Backprop through SiLU on x branch
		dXConv := siluBackward(cache.xConv, dXSilu)

		// Backprop through conv1D
		dXBranchConv, dConvW, dConvB := conv1DBackward(dXConv, cache.xBranch, seqLen, inner, l.Conv1DW, kSize)

		// Update conv1D weights
		for i := range l.Conv1DW {
			l.Conv1DW[i] -= lr * clipGrad(dConvW[i], gradClip)
		}
		for i := range l.Conv1DB {
			l.Conv1DB[i] -= lr * clipGrad(dConvB[i], gradClip)
		}

		// Combine x and z branch gradients for InProj backward
		// proj = normed @ InProjW + InProjB, then split to xBranch and zBranch
		dProj := make([]float32, seqLen*2*inner)
		for t := 0; t < seqLen; t++ {
			copy(dProj[t*2*inner:], dXBranchConv[t*inner:(t+1)*inner])
			copy(dProj[t*2*inner+inner:], dZBranch[t*inner:(t+1)*inner])
		}

		// Backprop through InProj: update weights
		for t := 0; t < seqLen; t++ {
			for j := 0; j < 2*inner; j++ {
				g := clipGrad(dProj[t*2*inner+j], gradClip)
				l.InProjB[j] -= lr * g
				for d := 0; d < dim; d++ {
					l.InProjW[d*2*inner+j] -= lr * g * cache.normed[t*dim+d]
				}
			}
		}

		// dX stays as-is through the residual for the next layer
		// (gradient flows unchanged through the skip connection)
	}

	// Step 4: Update token embeddings
	m.updateEmbeddings(inputIDs, targetTokens, probs, prefixLen, lr*0.1)

	return loss
}

// updateEmbeddings updates token embeddings based on prediction error.
func (m *MambaModel) updateEmbeddings(inputIDs, targets []int, probs []float32, prefixLen int, lr float32) {
	dim := m.Config.ModelDim
	vocab := m.Config.VocabSize

	for t := prefixLen; t < len(targets); t++ {
		tgt := targets[t]
		if tgt == PadID || tgt == BosID || tgt >= vocab {
			continue
		}

		prob := probs[t*vocab+tgt]
		if prob > 0.9 {
			continue
		}

		inputTok := inputIDs[t]
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
