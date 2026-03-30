package micromodel

import "math"

// -----------------------------------------------------------------------
// Multi-Head Attention
//
// Implements scaled dot-product attention with optional causal masking.
// All operations in float32 on flat slices.
// -----------------------------------------------------------------------

// MultiHeadAttention computes multi-head attention.
//
// Q, K, V are (seqLen, dim) — already projected by Wq, Wk, Wv.
// Wo is (dim, dim) — output projection.
// numHeads: number of attention heads.
// causal: if true, apply causal mask (decoder self-attention).
//
// Returns (seqLen, dim).
func MultiHeadAttention(Q, K, V []float32, seqLen, kvLen, dim, numHeads int, Wo []float32, causal bool) []float32 {
	headDim := dim / numHeads
	scale := float32(1.0 / math.Sqrt(float64(headDim)))

	// Output accumulator
	out := make([]float32, seqLen*dim)

	// Process each head independently
	for h := 0; h < numHeads; h++ {
		offset := h * headDim

		// Extract head slices: Q_h[i] = Q[i*dim + offset : offset+headDim]
		// Compute attention scores: scores[i][j] = Q_h[i] . K_h[j] * scale
		scores := make([]float32, seqLen*kvLen)
		for i := 0; i < seqLen; i++ {
			for j := 0; j < kvLen; j++ {
				var dot float32
				for d := 0; d < headDim; d++ {
					dot += Q[i*dim+offset+d] * K[j*dim+offset+d]
				}
				scores[i*kvLen+j] = dot * scale
			}
		}

		// Apply causal mask: set future positions to -inf
		if causal {
			for i := 0; i < seqLen; i++ {
				for j := i + 1; j < kvLen; j++ {
					scores[i*kvLen+j] = -1e9
				}
			}
		}

		// Softmax over each row
		softmax(scores, seqLen, kvLen)

		// Weighted sum: attn_out[i] = sum_j scores[i][j] * V_h[j]
		for i := 0; i < seqLen; i++ {
			for d := 0; d < headDim; d++ {
				var sum float32
				for j := 0; j < kvLen; j++ {
					sum += scores[i*kvLen+j] * V[j*dim+offset+d]
				}
				out[i*dim+offset+d] = sum
			}
		}
	}

	// Output projection: result = out * Wo
	result := matmul(out, seqLen, dim, Wo, dim)
	return result
}

// SelfAttention computes self-attention where Q=K=V come from the same source.
// x is (seqLen, dim). Wq, Wk, Wv, Wo are (dim, dim).
func SelfAttention(x []float32, seqLen, dim, numHeads int, Wq, Wk, Wv, Wo []float32, causal bool) []float32 {
	Q := matmul(x, seqLen, dim, Wq, dim)
	K := matmul(x, seqLen, dim, Wk, dim)
	V := matmul(x, seqLen, dim, Wv, dim)
	return MultiHeadAttention(Q, K, V, seqLen, seqLen, dim, numHeads, Wo, causal)
}

// CrossAttention computes attention where Q comes from the decoder
// and K, V come from the encoder.
// decX is (decLen, dim), encOut is (encLen, dim).
func CrossAttention(decX, encOut []float32, decLen, encLen, dim, numHeads int, Wq, Wk, Wv, Wo []float32) []float32 {
	Q := matmul(decX, decLen, dim, Wq, dim)
	K := matmul(encOut, encLen, dim, Wk, dim)
	V := matmul(encOut, encLen, dim, Wv, dim)
	return MultiHeadAttention(Q, K, V, decLen, encLen, dim, numHeads, Wo, false)
}
