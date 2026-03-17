package cognitive

import (
	"math"
	"path/filepath"
	"testing"
)

func TestMmapKnowledgeStoreAddAndSearch(t *testing.T) {
	store := NewMmapKnowledgeStore(filepath.Join(t.TempDir(), "test"))

	// Add chunks with simple embeddings
	store.Add("Go is a programming language", "test", []float64{1, 0, 0, 0})
	store.Add("Python is a programming language", "test", []float64{0.9, 0.1, 0, 0})
	store.Add("Cooking is a skill", "test", []float64{0, 0, 1, 0})

	if store.Size() != 3 {
		t.Fatalf("Size = %d, want 3", store.Size())
	}
	if store.Dim() != 4 {
		t.Fatalf("Dim = %d, want 4", store.Dim())
	}

	// Search for "Go" — should be closest to [1,0,0,0]
	results := store.Search([]float64{1, 0, 0, 0}, 2)
	if len(results) != 2 {
		t.Fatalf("got %d results, want 2", len(results))
	}
	if results[0].Text != "Go is a programming language" {
		t.Errorf("top result = %q, want Go", results[0].Text)
	}
	if results[0].Score < 0.99 {
		t.Errorf("top score = %f, want ~1.0", results[0].Score)
	}
}

func TestMmapKnowledgeStoreEmpty(t *testing.T) {
	store := NewMmapKnowledgeStore(filepath.Join(t.TempDir(), "empty"))

	results := store.Search([]float64{1, 0, 0}, 5)
	if len(results) != 0 {
		t.Errorf("empty store should return no results, got %d", len(results))
	}
}

func TestMmapKnowledgeStoreEmptyEmbedding(t *testing.T) {
	store := NewMmapKnowledgeStore(filepath.Join(t.TempDir(), "test"))
	store.Add("text", "source", nil) // empty embedding
	if store.Size() != 0 {
		t.Errorf("should not add chunk with empty embedding")
	}
}

func TestMmapKnowledgeStoreDimensionMismatch(t *testing.T) {
	store := NewMmapKnowledgeStore(filepath.Join(t.TempDir(), "test"))
	store.Add("first", "source", []float64{1, 0, 0})
	store.Add("second", "source", []float64{1, 0}) // wrong dimension

	if store.Size() != 1 {
		t.Errorf("should reject mismatched dimensions, Size = %d", store.Size())
	}
}

func TestMmapKnowledgeStoreSearchDimensionMismatch(t *testing.T) {
	store := NewMmapKnowledgeStore(filepath.Join(t.TempDir(), "test"))
	store.Add("text", "source", []float64{1, 0, 0})

	results := store.Search([]float64{1, 0}, 5) // wrong query dim
	if len(results) != 0 {
		t.Errorf("should return no results for mismatched query dim, got %d", len(results))
	}
}

func TestMmapKnowledgeStoreTopKLimit(t *testing.T) {
	store := NewMmapKnowledgeStore(filepath.Join(t.TempDir(), "test"))
	store.Add("a", "s", []float64{1, 0})
	store.Add("b", "s", []float64{0, 1})

	results := store.Search([]float64{1, 0}, 10) // ask for 10, only 2 exist
	if len(results) != 2 {
		t.Errorf("should return min(topK, size) = 2, got %d", len(results))
	}
}

func TestMmapKnowledgeStorePersistence(t *testing.T) {
	dir := t.TempDir()
	basePath := filepath.Join(dir, "knowledge")

	// Create and save
	store := NewMmapKnowledgeStore(basePath)
	store.Add("Go language", "wiki", []float64{1, 0, 0, 0})
	store.Add("Python language", "wiki", []float64{0, 1, 0, 0})
	if err := store.Save(); err != nil {
		t.Fatalf("Save failed: %v", err)
	}

	// Load into new instance
	store2 := NewMmapKnowledgeStore(basePath)
	if store2.Size() != 2 {
		t.Fatalf("loaded Size = %d, want 2", store2.Size())
	}
	if store2.Dim() != 4 {
		t.Fatalf("loaded Dim = %d, want 4", store2.Dim())
	}

	// Verify search still works
	results := store2.Search([]float64{1, 0, 0, 0}, 1)
	if len(results) != 1 {
		t.Fatalf("loaded search returned %d results", len(results))
	}
	if results[0].Text != "Go language" {
		t.Errorf("loaded top result = %q, want 'Go language'", results[0].Text)
	}
}

func TestMmapKnowledgeStoreMemoryUsage(t *testing.T) {
	store := NewMmapKnowledgeStore(filepath.Join(t.TempDir(), "test"))

	// 100 chunks × 768 dims = 100 * 768 * 8 bytes = ~600KB
	dim := 768
	embed := make([]float64, dim)
	for i := range embed {
		embed[i] = 0.01
	}
	for i := 0; i < 100; i++ {
		store.Add("chunk", "src", embed)
	}

	mb := store.MemoryUsageMB()
	expected := float64(100*dim*8) / (1024 * 1024)
	if math.Abs(mb-expected) > 0.01 {
		t.Errorf("MemoryUsageMB = %f, want ~%f", mb, expected)
	}
}

func TestMmapKnowledgeStoreSearchOrdering(t *testing.T) {
	store := NewMmapKnowledgeStore(filepath.Join(t.TempDir(), "test"))
	store.Add("exact match", "s", []float64{1, 0, 0})
	store.Add("partial match", "s", []float64{0.7, 0.7, 0})
	store.Add("no match", "s", []float64{0, 0, 1})

	results := store.Search([]float64{1, 0, 0}, 3)
	if len(results) != 3 {
		t.Fatalf("got %d results, want 3", len(results))
	}

	// Verify ordering: exact > partial > no match
	if results[0].Text != "exact match" {
		t.Errorf("first = %q, want 'exact match'", results[0].Text)
	}
	if results[1].Text != "partial match" {
		t.Errorf("second = %q, want 'partial match'", results[1].Text)
	}
	if results[2].Text != "no match" {
		t.Errorf("third = %q, want 'no match'", results[2].Text)
	}

	// Verify scores are monotonically decreasing
	for i := 1; i < len(results); i++ {
		if results[i].Score > results[i-1].Score {
			t.Errorf("results not sorted: [%d].Score=%f > [%d].Score=%f",
				i, results[i].Score, i-1, results[i-1].Score)
		}
	}
}

func BenchmarkMmapSearch1000(b *testing.B) {
	store := NewMmapKnowledgeStore(filepath.Join(b.TempDir(), "bench"))
	dim := 768
	embed := make([]float64, dim)
	for i := range embed {
		embed[i] = float64(i) * 0.001
	}
	for i := 0; i < 1000; i++ {
		store.Add("chunk", "src", embed)
	}

	query := make([]float64, dim)
	for i := range query {
		query[i] = float64(dim-i) * 0.001
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		store.Search(query, 5)
	}
}
