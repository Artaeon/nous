package simd

import (
	"math"
	"testing"
)

func TestCosineSimilarityIdentical(t *testing.T) {
	a := []float64{1, 2, 3, 4, 5}
	got := CosineSimilarity(a, a)
	if math.Abs(got-1.0) > 1e-10 {
		t.Errorf("identical vectors: got %f, want 1.0", got)
	}
}

func TestCosineSimilarityOrthogonal(t *testing.T) {
	a := []float64{1, 0, 0, 0}
	b := []float64{0, 1, 0, 0}
	got := CosineSimilarity(a, b)
	if math.Abs(got) > 1e-10 {
		t.Errorf("orthogonal vectors: got %f, want 0.0", got)
	}
}

func TestCosineSimilarityOpposite(t *testing.T) {
	a := []float64{1, 2, 3, 4}
	b := []float64{-1, -2, -3, -4}
	got := CosineSimilarity(a, b)
	if math.Abs(got-(-1.0)) > 1e-10 {
		t.Errorf("opposite vectors: got %f, want -1.0", got)
	}
}

func TestCosineSimilarityDifferentLength(t *testing.T) {
	a := []float64{1, 2, 3}
	b := []float64{1, 2}
	got := CosineSimilarity(a, b)
	if got != 0 {
		t.Errorf("different lengths: got %f, want 0", got)
	}
}

func TestCosineSimilarityEmpty(t *testing.T) {
	got := CosineSimilarity(nil, nil)
	if got != 0 {
		t.Errorf("empty: got %f, want 0", got)
	}
}

func TestCosineSimilarityZeroVector(t *testing.T) {
	a := []float64{0, 0, 0, 0}
	b := []float64{1, 2, 3, 4}
	got := CosineSimilarity(a, b)
	if got != 0 {
		t.Errorf("zero vector: got %f, want 0", got)
	}
}

func TestCosineSimilaritySmallVectors(t *testing.T) {
	// Test with size < 4 to exercise remainder loop
	a := []float64{3, 4}
	b := []float64{4, 3}
	got := CosineSimilarity(a, b)
	expected := 24.0 / 25.0 // (3*4 + 4*3) / (5 * 5)
	if math.Abs(got-expected) > 1e-10 {
		t.Errorf("small vectors: got %f, want %f", got, expected)
	}
}

func TestCosineSimilarityLargeVector(t *testing.T) {
	// 768-dimensional vector (typical embedding size)
	n := 768
	a := make([]float64, n)
	b := make([]float64, n)
	for i := range a {
		a[i] = float64(i) * 0.01
		b[i] = float64(n-i) * 0.01
	}
	got := CosineSimilarity(a, b)
	if math.IsNaN(got) || math.IsInf(got, 0) {
		t.Errorf("large vector: got %f, should be finite", got)
	}
}

func TestCosineSimilarityMatchesNaive(t *testing.T) {
	// Compare optimized against naive implementation
	a := []float64{0.1, 0.5, 0.3, 0.7, 0.2, 0.9, 0.4}
	b := []float64{0.4, 0.2, 0.8, 0.1, 0.6, 0.3, 0.5}

	// Naive
	var dot, na, nb float64
	for i := range a {
		dot += a[i] * b[i]
		na += a[i] * a[i]
		nb += b[i] * b[i]
	}
	expected := dot / (math.Sqrt(na) * math.Sqrt(nb))

	got := CosineSimilarity(a, b)
	if math.Abs(got-expected) > 1e-10 {
		t.Errorf("mismatch: got %f, want %f", got, expected)
	}
}

func TestDotProduct(t *testing.T) {
	a := []float64{1, 2, 3, 4, 5}
	b := []float64{5, 4, 3, 2, 1}
	got := DotProduct(a, b)
	expected := 1.0*5 + 2*4 + 3*3 + 4*2 + 5*1 // = 35
	if math.Abs(got-expected) > 1e-10 {
		t.Errorf("DotProduct = %f, want %f", got, expected)
	}
}

func TestDotProductEmpty(t *testing.T) {
	if got := DotProduct(nil, nil); got != 0 {
		t.Errorf("empty DotProduct = %f, want 0", got)
	}
}

func TestNorm(t *testing.T) {
	v := []float64{3, 4}
	got := Norm(v)
	if math.Abs(got-5.0) > 1e-10 {
		t.Errorf("Norm([3,4]) = %f, want 5.0", got)
	}
}

func TestNormEmpty(t *testing.T) {
	if got := Norm(nil); got != 0 {
		t.Errorf("empty Norm = %f, want 0", got)
	}
}

func TestNormLarge(t *testing.T) {
	v := make([]float64, 768)
	for i := range v {
		v[i] = 1.0
	}
	got := Norm(v)
	expected := math.Sqrt(768)
	if math.Abs(got-expected) > 1e-10 {
		t.Errorf("Norm = %f, want %f", got, expected)
	}
}

// Benchmarks

func BenchmarkCosineSimilarity768(b *testing.B) {
	a := make([]float64, 768)
	bv := make([]float64, 768)
	for i := range a {
		a[i] = float64(i) * 0.001
		bv[i] = float64(768-i) * 0.001
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		CosineSimilarity(a, bv)
	}
}

func BenchmarkCosineSimilarityNaive768(b *testing.B) {
	a := make([]float64, 768)
	bv := make([]float64, 768)
	for i := range a {
		a[i] = float64(i) * 0.001
		bv[i] = float64(768-i) * 0.001
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		naiveCosineSimilarity(a, bv)
	}
}

func naiveCosineSimilarity(a, b []float64) float64 {
	var dot, na, nb float64
	for i := range a {
		dot += a[i] * b[i]
		na += a[i] * a[i]
		nb += b[i] * b[i]
	}
	d := math.Sqrt(na) * math.Sqrt(nb)
	if d == 0 {
		return 0
	}
	return dot / d
}
