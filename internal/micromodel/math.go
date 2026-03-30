package micromodel

import "math"

// -----------------------------------------------------------------------
// Core tensor operations in float32.
// All tensors are flat []float32 slices in row-major order.
// A matrix of shape (rows, cols) is stored as rows*cols contiguous floats.
// -----------------------------------------------------------------------

// matmul computes C = A * B where A is (m, k) and B is (k, n).
// Result C is (m, n).
func matmul(A []float32, m, k int, B []float32, n int) []float32 {
	C := make([]float32, m*n)
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			var sum float32
			for p := 0; p < k; p++ {
				sum += A[i*k+p] * B[p*n+j]
			}
			C[i*n+j] = sum
		}
	}
	return C
}

// matmulAdd computes C = A * B + bias where bias is length n (broadcast over rows).
func matmulAdd(A []float32, m, k int, B []float32, n int, bias []float32) []float32 {
	C := matmul(A, m, k, B, n)
	if bias != nil {
		for i := 0; i < m; i++ {
			for j := 0; j < n; j++ {
				C[i*n+j] += bias[j]
			}
		}
	}
	return C
}

// matmulT computes C = A * B^T where A is (m, k) and B is (n, k).
// Result C is (m, n).
func matmulT(A []float32, m, k int, B []float32, n int) []float32 {
	C := make([]float32, m*n)
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			var sum float32
			for p := 0; p < k; p++ {
				sum += A[i*k+p] * B[j*k+p]
			}
			C[i*n+j] = sum
		}
	}
	return C
}

// vecAdd adds two vectors element-wise: out = a + b.
func vecAdd(a, b []float32) []float32 {
	out := make([]float32, len(a))
	for i := range a {
		out[i] = a[i] + b[i]
	}
	return out
}

// softmax computes softmax over the last dimension of a (seqLen, vocabOrSeqLen) tensor.
// Operates in-place on each row.
func softmax(logits []float32, rows, cols int) {
	for i := 0; i < rows; i++ {
		row := logits[i*cols : (i+1)*cols]

		// Find max for numerical stability
		maxVal := row[0]
		for _, v := range row[1:] {
			if v > maxVal {
				maxVal = v
			}
		}

		// Exp and sum
		var sum float32
		for j := range row {
			row[j] = float32(math.Exp(float64(row[j] - maxVal)))
			sum += row[j]
		}

		// Normalize
		if sum > 0 {
			inv := 1.0 / sum
			for j := range row {
				row[j] *= inv
			}
		}
	}
}

// layerNorm applies layer normalization to each row of x (seqLen, dim).
// gamma and beta are learnable parameters of length dim.
func layerNorm(x []float32, seqLen, dim int, gamma, beta []float32) []float32 {
	out := make([]float32, len(x))
	const eps = 1e-5

	for i := 0; i < seqLen; i++ {
		row := x[i*dim : (i+1)*dim]

		// Compute mean
		var mean float32
		for _, v := range row {
			mean += v
		}
		mean /= float32(dim)

		// Compute variance
		var variance float32
		for _, v := range row {
			d := v - mean
			variance += d * d
		}
		variance /= float32(dim)

		// Normalize, scale, shift
		invStd := float32(1.0 / math.Sqrt(float64(variance+eps)))
		for j := 0; j < dim; j++ {
			normalized := (row[j] - mean) * invStd
			out[i*dim+j] = normalized*gamma[j] + beta[j]
		}
	}
	return out
}

// relu applies ReLU activation element-wise.
func relu(x []float32) []float32 {
	out := make([]float32, len(x))
	for i, v := range x {
		if v > 0 {
			out[i] = v
		}
	}
	return out
}

// embedding looks up token embeddings. ids is a sequence of token IDs.
// Returns (len(ids), dim) flattened.
func embedding(ids []int, table []float32, dim int) []float32 {
	out := make([]float32, len(ids)*dim)
	for i, id := range ids {
		if id >= 0 && id*dim+dim <= len(table) {
			copy(out[i*dim:], table[id*dim:(id+1)*dim])
		}
	}
	return out
}

// addPositionalEncoding adds learned positional embeddings.
func addPositionalEncoding(x []float32, seqLen, dim int, posTable []float32) []float32 {
	out := make([]float32, len(x))
	copy(out, x)
	for i := 0; i < seqLen; i++ {
		if i*dim+dim <= len(posTable) {
			for j := 0; j < dim; j++ {
				out[i*dim+j] += posTable[i*dim+j]
			}
		}
	}
	return out
}

// crossEntropyLoss computes the average cross-entropy loss.
// probs is (seqLen, vocabSize), targets is (seqLen) with target token IDs.
// Only counts positions where target != PadID.
func crossEntropyLoss(probs []float32, vocabSize int, targets []int) float32 {
	var loss float32
	var count int
	for i, tgt := range targets {
		if tgt == PadID {
			continue
		}
		p := probs[i*vocabSize+tgt]
		if p < 1e-10 {
			p = 1e-10
		}
		loss -= float32(math.Log(float64(p)))
		count++
	}
	if count == 0 {
		return 0
	}
	return loss / float32(count)
}

// sampleToken samples a token ID from a probability distribution
// using temperature scaling.
func sampleToken(probs []float32, temperature float32, rng func() float64) int {
	if temperature <= 0 {
		// Greedy
		best := 0
		for i, p := range probs {
			if p > probs[best] {
				best = i
			}
		}
		return best
	}

	// Apply temperature
	scaled := make([]float32, len(probs))
	copy(scaled, probs)
	if temperature != 1.0 {
		// Re-compute from logits: take log, divide by temp, re-softmax
		for i := range scaled {
			if scaled[i] < 1e-10 {
				scaled[i] = 1e-10
			}
			scaled[i] = float32(math.Log(float64(scaled[i]))) / temperature
		}
		softmax(scaled, 1, len(scaled))
	}

	// Sample from distribution
	r := float32(rng())
	var cumsum float32
	for i, p := range scaled {
		cumsum += p
		if r <= cumsum {
			return i
		}
	}
	return len(scaled) - 1
}

// initWeight fills a slice with Xavier uniform initialization.
func initWeight(w []float32, fanIn, fanOut int, rng func() float64) {
	limit := float64(math.Sqrt(6.0 / float64(fanIn+fanOut)))
	for i := range w {
		w[i] = float32((rng()*2 - 1) * limit)
	}
}

// initZero fills a slice with zeros.
func initZero(w []float32) {
	for i := range w {
		w[i] = 0
	}
}

// initOnes fills a slice with ones (for LayerNorm gamma).
func initOnes(w []float32) {
	for i := range w {
		w[i] = 1
	}
}
