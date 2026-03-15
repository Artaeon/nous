package cognitive

import (
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// --- Knowledge Vec Tests ---

// mockEmbed returns a simple deterministic embedding based on the text.
func mockEmbed(text string) ([]float64, error) {
	// Create a simple embedding from character frequencies
	vec := make([]float64, 8) // small dim for testing
	for _, c := range text {
		idx := int(c) % 8
		vec[idx] += 1.0
	}
	// Normalize
	norm := 0.0
	for _, v := range vec {
		norm += v * v
	}
	norm = math.Sqrt(norm)
	if norm > 0 {
		for i := range vec {
			vec[i] /= norm
		}
	}
	return vec, nil
}

func TestKnowledgeVecCreation(t *testing.T) {
	kv := NewKnowledgeVec(mockEmbed, "")
	if kv == nil {
		t.Fatal("should not return nil")
	}
	if kv.Size() != 0 {
		t.Error("new store should be empty")
	}
}

func TestKnowledgeVecAddChunk(t *testing.T) {
	kv := NewKnowledgeVec(mockEmbed, "")

	err := kv.AddChunk("Quantum physics studies subatomic particles.", "wiki")
	if err != nil {
		t.Fatalf("AddChunk failed: %v", err)
	}

	if kv.Size() != 1 {
		t.Errorf("size = %d, want 1", kv.Size())
	}
}

func TestKnowledgeVecAddEmptyChunk(t *testing.T) {
	kv := NewKnowledgeVec(mockEmbed, "")
	kv.AddChunk("", "test")
	if kv.Size() != 0 {
		t.Error("empty chunk should not be added")
	}
}

func TestKnowledgeVecSearch(t *testing.T) {
	kv := NewKnowledgeVec(mockEmbed, "")

	kv.AddChunk("Quantum physics studies subatomic particles and wave functions.", "physics")
	kv.AddChunk("Machine learning uses neural networks for pattern recognition.", "cs")
	kv.AddChunk("The Roman Empire lasted from 27 BC to 476 AD.", "history")

	results, err := kv.Search("quantum mechanics and particles", 2)
	if err != nil {
		t.Fatalf("search failed: %v", err)
	}

	if len(results) != 2 {
		t.Fatalf("should return 2 results, got %d", len(results))
	}

	// First result should be the most relevant
	if results[0].Score < results[1].Score {
		t.Error("results should be sorted by relevance")
	}
}

func TestKnowledgeVecSearchEmpty(t *testing.T) {
	kv := NewKnowledgeVec(mockEmbed, "")
	results, err := kv.Search("anything", 5)
	if err != nil {
		t.Fatal("search on empty store should not error")
	}
	if len(results) != 0 {
		t.Error("empty store should return no results")
	}
}

func TestKnowledgeVecSearchTopK(t *testing.T) {
	kv := NewKnowledgeVec(mockEmbed, "")
	kv.AddChunk("fact one", "src1")
	kv.AddChunk("fact two", "src2")

	results, _ := kv.Search("fact", 10)
	if len(results) != 2 {
		t.Errorf("should return min(topK, size) results, got %d", len(results))
	}
}

func TestKnowledgeVecIngestText(t *testing.T) {
	kv := NewKnowledgeVec(mockEmbed, "")

	text := "First paragraph about quantum physics.\n\nSecond paragraph about machine learning.\n\nThird paragraph about history and the Roman Empire."
	added, err := kv.IngestText(text, "test")
	if err != nil {
		t.Fatalf("IngestText failed: %v", err)
	}
	if added != 3 {
		t.Errorf("should add 3 chunks, got %d", added)
	}
	if kv.Size() != 3 {
		t.Errorf("size = %d, want 3", kv.Size())
	}
}

func TestKnowledgeVecIngestTextSkipsShort(t *testing.T) {
	kv := NewKnowledgeVec(mockEmbed, "")
	text := "hi\n\nThis is a longer paragraph with enough text.\n\nok"
	added, _ := kv.IngestText(text, "test")
	if added != 1 {
		t.Errorf("should skip short paragraphs, added %d", added)
	}
}

func TestKnowledgeVecIngestFile(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "knowledge.txt")

	// Content needs enough text to form at least one chunk (Ingest flushes remaining at end)
	content := "First paragraph about physics and quantum mechanics and the study of subatomic particles and wave functions and energy levels.\n\nSecond paragraph about computer science and algorithms and data structures and computational complexity theory.\n\nThird paragraph about biology and the evolution of species and natural selection and genetics and molecular biology."
	os.WriteFile(path, []byte(content), 0644)

	kv := NewKnowledgeVec(mockEmbed, "")
	added, err := kv.Ingest(path)
	if err != nil {
		t.Fatalf("Ingest failed: %v", err)
	}
	if added < 1 {
		t.Errorf("should ingest at least 1 chunk, got %d", added)
	}
}

func TestKnowledgeVecIngestMissingFile(t *testing.T) {
	kv := NewKnowledgeVec(mockEmbed, "")
	_, err := kv.Ingest("/nonexistent/file.txt")
	if err == nil {
		t.Error("should error on missing file")
	}
}

func TestKnowledgeVecSaveLoad(t *testing.T) {
	dir := t.TempDir()
	storePath := filepath.Join(dir, "knowledge.json")

	kv := NewKnowledgeVec(mockEmbed, storePath)
	kv.AddChunk("Quantum physics studies particles.", "physics")
	kv.AddChunk("Machine learning uses neural networks.", "cs")
	kv.Save()

	// Load into new store
	kv2 := NewKnowledgeVec(mockEmbed, storePath)
	if kv2.Size() != 2 {
		t.Errorf("loaded size = %d, want 2", kv2.Size())
	}

	// Search should work on loaded data
	results, _ := kv2.Search("quantum", 1)
	if len(results) != 1 {
		t.Error("search should work after loading")
	}
}

func TestKnowledgeVecFormatContext(t *testing.T) {
	results := []KnowledgeResult{
		{Text: "Quantum physics studies particles.", Source: "wiki", Score: 0.9},
		{Text: "Einstein developed relativity.", Source: "wiki", Score: 0.7},
	}

	ctx := FormatKnowledgeContext(results)
	if !strings.Contains(ctx, "Quantum") {
		t.Error("context should include first result")
	}
	if !strings.Contains(ctx, "wiki") {
		t.Error("context should include source")
	}
}

func TestKnowledgeVecFormatContextEmpty(t *testing.T) {
	ctx := FormatKnowledgeContext(nil)
	if ctx != "" {
		t.Error("empty results should return empty context")
	}
}

func TestKnowledgeVecFormatContextLowScore(t *testing.T) {
	results := []KnowledgeResult{
		{Text: "irrelevant", Source: "src", Score: 0.1},
	}
	ctx := FormatKnowledgeContext(results)
	if strings.Contains(ctx, "irrelevant") {
		t.Error("low-score results should be filtered")
	}
}

func TestKnowledgeVecSearchByVector(t *testing.T) {
	kv := NewKnowledgeVec(mockEmbed, "")

	vec1 := []float64{1, 0, 0, 0, 0, 0, 0, 0}
	vec2 := []float64{0, 1, 0, 0, 0, 0, 0, 0}

	kv.AddChunkWithEmbedding("chunk one", "src1", vec1)
	kv.AddChunkWithEmbedding("chunk two", "src2", vec2)

	// Search with vector close to vec1
	results := kv.SearchByVector([]float64{0.9, 0.1, 0, 0, 0, 0, 0, 0}, 1)
	if len(results) != 1 {
		t.Fatal("should return 1 result")
	}
	if results[0].Text != "chunk one" {
		t.Errorf("should find chunk one, got %q", results[0].Text)
	}
}

// --- Cosine Similarity Tests ---

func TestCosineSim(t *testing.T) {
	a := []float64{1, 0, 0}
	b := []float64{1, 0, 0}
	if math.Abs(cosineSim(a, b)-1.0) > 1e-6 {
		t.Error("identical vectors should have similarity 1.0")
	}

	c := []float64{0, 1, 0}
	if math.Abs(cosineSim(a, c)) > 1e-6 {
		t.Error("orthogonal vectors should have similarity 0.0")
	}

	d := []float64{-1, 0, 0}
	if math.Abs(cosineSim(a, d)-(-1.0)) > 1e-6 {
		t.Error("opposite vectors should have similarity -1.0")
	}
}

func TestCosineSimDifferentLength(t *testing.T) {
	a := []float64{1, 0}
	b := []float64{1, 0, 0}
	if cosineSim(a, b) != 0 {
		t.Error("different length vectors should return 0")
	}
}

func TestCosineSimEmpty(t *testing.T) {
	if cosineSim(nil, nil) != 0 {
		t.Error("nil vectors should return 0")
	}
}

// --- Benchmark ---

func BenchmarkKnowledgeSearch(b *testing.B) {
	kv := NewKnowledgeVec(mockEmbed, "")
	for i := 0; i < 100; i++ {
		kv.AddChunk(strings.Repeat("knowledge chunk ", 10), "test")
	}

	queryVec, _ := mockEmbed("test query")

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		kv.SearchByVector(queryVec, 5)
	}
}
