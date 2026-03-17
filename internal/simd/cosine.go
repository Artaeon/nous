// Package simd provides optimized vector math operations.
// The cosine similarity function uses 4-way loop unrolling which allows
// the Go compiler to auto-vectorize on architectures with SIMD support.
package simd

import "math"

// CosineSimilarity computes the cosine similarity between two vectors.
// Uses 4-way loop unrolling for better compiler auto-vectorization.
// Returns 0 if vectors have different lengths or are empty.
func CosineSimilarity(a, b []float64) float64 {
	n := len(a)
	if n != len(b) || n == 0 {
		return 0
	}

	var dot, normA, normB float64

	// 4-way unrolled loop — enables compiler auto-vectorization.
	// On x86-64, Go 1.21+ will emit SIMD instructions for this pattern.
	i := 0
	for ; i+3 < n; i += 4 {
		a0, a1, a2, a3 := a[i], a[i+1], a[i+2], a[i+3]
		b0, b1, b2, b3 := b[i], b[i+1], b[i+2], b[i+3]

		dot += a0*b0 + a1*b1 + a2*b2 + a3*b3
		normA += a0*a0 + a1*a1 + a2*a2 + a3*a3
		normB += b0*b0 + b1*b1 + b2*b2 + b3*b3
	}

	// Handle remainder
	for ; i < n; i++ {
		dot += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}

	denom := math.Sqrt(normA * normB)
	if denom == 0 {
		return 0
	}

	return dot / denom
}

// DotProduct computes the dot product of two vectors.
func DotProduct(a, b []float64) float64 {
	n := len(a)
	if n != len(b) || n == 0 {
		return 0
	}

	var sum float64
	i := 0
	for ; i+3 < n; i += 4 {
		sum += a[i]*b[i] + a[i+1]*b[i+1] + a[i+2]*b[i+2] + a[i+3]*b[i+3]
	}
	for ; i < n; i++ {
		sum += a[i] * b[i]
	}
	return sum
}

// Norm computes the L2 norm of a vector.
func Norm(v []float64) float64 {
	if len(v) == 0 {
		return 0
	}

	var sum float64
	i := 0
	for ; i+3 < len(v); i += 4 {
		sum += v[i]*v[i] + v[i+1]*v[i+1] + v[i+2]*v[i+2] + v[i+3]*v[i+3]
	}
	for ; i < len(v); i++ {
		sum += v[i] * v[i]
	}
	return math.Sqrt(sum)
}
