package cognitive

import (
	"math"
	"strings"
	"testing"
)

// mockEmbedFn creates a deterministic embedding function for testing.
// Maps known words to specific vector positions so we can control similarity.
func mockEmbedFn() EmbedFunc {
	wordVectors := map[string][]float64{
		"search":   {1, 0, 0, 0, 0},
		"find":     {0.9, 0.1, 0, 0, 0},
		"read":     {0, 1, 0, 0, 0},
		"show":     {0, 0.9, 0.1, 0, 0},
		"write":    {0, 0, 1, 0, 0},
		"create":   {0, 0, 0.9, 0.1, 0},
		"list":     {0, 0, 0, 1, 0},
		"files":    {0, 0, 0, 0.8, 0.2},
		"grep":     {0.95, 0.05, 0, 0, 0},
		"main.go":  {0, 0.8, 0, 0, 0.2},
		"auth.go":  {0, 0.7, 0.1, 0, 0.2},
		"pipeline": {0.5, 0.3, 0, 0, 0.2},
	}

	return func(text string) ([]float64, error) {
		// Average word vectors for multi-word inputs
		words := strings.Fields(strings.ToLower(text))
		result := make([]float64, 5)
		count := 0

		for _, w := range words {
			// Strip common separators
			w = strings.Trim(w, ":.,'\"")
			if vec, ok := wordVectors[w]; ok {
				for i := range result {
					result[i] += vec[i]
				}
				count++
			}
		}

		if count > 0 {
			for i := range result {
				result[i] /= float64(count)
			}
		} else {
			// Unknown words get a random-ish vector
			for i := range result {
				result[i] = 0.1
			}
		}

		return result, nil
	}
}

// --- Creation Tests ---

func TestEmbedGrounderCreation(t *testing.T) {
	eg := NewEmbedGrounder(nil)
	if eg == nil {
		t.Fatal("NewEmbedGrounder should not return nil")
	}
	if eg.ToolCount() != 0 {
		t.Error("new grounder should have 0 tools")
	}
	if eg.FileCount() != 0 {
		t.Error("new grounder should have 0 files")
	}
	if eg.HistoryCount() != 0 {
		t.Error("new grounder should have 0 history")
	}
}

func TestEmbedGrounderNilEmbedFn(t *testing.T) {
	eg := NewEmbedGrounder(nil)

	// All operations should be no-ops with nil embedFn
	if err := eg.IndexTool("grep", "search files"); err != nil {
		t.Errorf("IndexTool with nil embedFn should not error: %v", err)
	}
	if err := eg.IndexFile("main.go", "entry point"); err != nil {
		t.Errorf("IndexFile with nil embedFn should not error: %v", err)
	}
	eg.RecordSuccess("query", "grep", nil) // should not panic

	result, err := eg.Ground("test")
	if err != nil {
		t.Errorf("Ground with nil embedFn should not error: %v", err)
	}
	if result == nil {
		t.Error("should return empty result, not nil")
	}
}

// --- Index Tests ---

func TestIndexTool(t *testing.T) {
	eg := NewEmbedGrounder(mockEmbedFn())

	if err := eg.IndexTool("grep", "search file contents"); err != nil {
		t.Fatalf("IndexTool error: %v", err)
	}

	if eg.ToolCount() != 1 {
		t.Errorf("tool count = %d, want 1", eg.ToolCount())
	}
}

func TestIndexMultipleTools(t *testing.T) {
	eg := NewEmbedGrounder(mockEmbedFn())

	tools := map[string]string{
		"grep": "search file contents",
		"read": "read a file",
		"ls":   "list files in directory",
	}

	for name, desc := range tools {
		if err := eg.IndexTool(name, desc); err != nil {
			t.Fatalf("IndexTool(%s) error: %v", name, err)
		}
	}

	if eg.ToolCount() != 3 {
		t.Errorf("tool count = %d, want 3", eg.ToolCount())
	}
}

func TestIndexFile(t *testing.T) {
	eg := NewEmbedGrounder(mockEmbedFn())

	if err := eg.IndexFile("main.go", "entry point"); err != nil {
		t.Fatalf("IndexFile error: %v", err)
	}

	if eg.FileCount() != 1 {
		t.Errorf("file count = %d, want 1", eg.FileCount())
	}
}

func TestIndexFileNoSummary(t *testing.T) {
	eg := NewEmbedGrounder(mockEmbedFn())

	if err := eg.IndexFile("main.go", ""); err != nil {
		t.Fatalf("IndexFile error: %v", err)
	}

	if eg.FileCount() != 1 {
		t.Errorf("file count = %d, want 1", eg.FileCount())
	}
}

// --- History Tests ---

func TestRecordSuccess(t *testing.T) {
	eg := NewEmbedGrounder(mockEmbedFn())

	eg.RecordSuccess("search for Pipeline", "grep", map[string]string{"pattern": "Pipeline"})

	if eg.HistoryCount() != 1 {
		t.Errorf("history count = %d, want 1", eg.HistoryCount())
	}
}

func TestRecordSuccessCapacity(t *testing.T) {
	eg := NewEmbedGrounder(mockEmbedFn())

	// Record 600 entries — should be capped at 500
	for i := 0; i < 600; i++ {
		eg.RecordSuccess("query", "grep", nil)
	}

	if eg.HistoryCount() > 500 {
		t.Errorf("history should be capped at 500, got %d", eg.HistoryCount())
	}
}

// --- Semantic Matching Tests ---

func TestRecommendTool(t *testing.T) {
	eg := NewEmbedGrounder(mockEmbedFn())
	eg.IndexTool("grep", "search find files")
	eg.IndexTool("read", "read show file")
	eg.IndexTool("ls", "list files directory")

	tool, score, err := eg.RecommendTool("search for something")
	if err != nil {
		t.Fatalf("RecommendTool error: %v", err)
	}
	if tool != "grep" {
		t.Errorf("expected grep for search query, got %q (score: %f)", tool, score)
	}
	if score <= 0 {
		t.Error("score should be positive")
	}
}

func TestRecommendToolNil(t *testing.T) {
	eg := NewEmbedGrounder(nil)
	tool, score, err := eg.RecommendTool("search")
	if err != nil {
		t.Errorf("should not error with nil embedFn: %v", err)
	}
	if tool != "" {
		t.Errorf("should return empty tool with nil embedFn, got %q", tool)
	}
	if score != 0 {
		t.Errorf("should return 0 score with nil embedFn, got %f", score)
	}
}

func TestResolveFile(t *testing.T) {
	eg := NewEmbedGrounder(mockEmbedFn())
	eg.IndexFile("main.go", "main entry point")
	eg.IndexFile("auth.go", "authentication module")

	path, score, err := eg.ResolveFile("read main.go")
	if err != nil {
		t.Fatalf("ResolveFile error: %v", err)
	}
	if score <= 0 {
		t.Error("should find a matching file")
	}
	_ = path // path depends on embedding quality
}

func TestResolveFileNil(t *testing.T) {
	eg := NewEmbedGrounder(nil)
	path, score, err := eg.ResolveFile("main.go")
	if err != nil {
		t.Errorf("should not error: %v", err)
	}
	if path != "" || score != 0 {
		t.Error("nil embedFn should return empty results")
	}
}

func TestFindSimilar(t *testing.T) {
	eg := NewEmbedGrounder(mockEmbedFn())

	eg.RecordSuccess("search for Pipeline", "grep", map[string]string{"pattern": "Pipeline"})
	eg.RecordSuccess("find the auth module", "grep", map[string]string{"pattern": "auth"})

	matches, err := eg.FindSimilar("search for something", 5)
	if err != nil {
		t.Fatalf("FindSimilar error: %v", err)
	}

	// Should find at least one similar past query
	if len(matches) == 0 {
		t.Error("should find similar past interactions")
	}
}

func TestFindSimilarNil(t *testing.T) {
	eg := NewEmbedGrounder(nil)
	matches, err := eg.FindSimilar("test", 5)
	if err != nil {
		t.Errorf("should not error: %v", err)
	}
	if len(matches) != 0 {
		t.Error("nil embedFn should return empty results")
	}
}

// --- Ground Tests ---

func TestGround(t *testing.T) {
	eg := NewEmbedGrounder(mockEmbedFn())
	eg.IndexTool("grep", "search find files")
	eg.IndexTool("read", "read show file")
	eg.IndexFile("main.go", "main entry point")
	eg.RecordSuccess("search for Pipeline", "grep", map[string]string{"pattern": "Pipeline"})

	result, err := eg.Ground("find Pipeline")
	if err != nil {
		t.Fatalf("Ground error: %v", err)
	}
	if result == nil {
		t.Fatal("result should not be nil")
	}
	if result.Tool == "" {
		t.Error("should recommend a tool")
	}
}

func TestGroundNilEmbedFn(t *testing.T) {
	eg := NewEmbedGrounder(nil)
	result, err := eg.Ground("test")
	if err != nil {
		t.Errorf("should not error: %v", err)
	}
	if result == nil {
		t.Fatal("should return empty result, not nil")
	}
}

// --- FormatGroundingContext Tests ---

func TestFormatGroundingContextFull(t *testing.T) {
	result := &GroundedResult{
		Tool:      "grep",
		ToolScore: 0.85,
		ResolvedArgs: map[string]string{
			"path": "main.go",
		},
		SimilarPast: []PastMatch{
			{Query: "find Pipeline", Tool: "grep", Args: map[string]string{"pattern": "Pipeline"}, Similarity: 0.9},
		},
	}

	ctx := result.FormatGroundingContext()
	if !strings.Contains(ctx, "grep") {
		t.Error("should mention recommended tool")
	}
	if !strings.Contains(ctx, "main.go") {
		t.Error("should mention resolved file path")
	}
	if !strings.Contains(ctx, "Pipeline") {
		t.Error("should mention similar past query")
	}
}

func TestFormatGroundingContextEmpty(t *testing.T) {
	result := &GroundedResult{}
	ctx := result.FormatGroundingContext()
	if ctx != "" {
		t.Errorf("empty result should produce empty context, got %q", ctx)
	}
}

func TestFormatGroundingContextNil(t *testing.T) {
	var result *GroundedResult
	ctx := result.FormatGroundingContext()
	if ctx != "" {
		t.Error("nil result should produce empty context")
	}
}

func TestFormatGroundingContextLowScore(t *testing.T) {
	result := &GroundedResult{
		Tool:      "grep",
		ToolScore: 0.3, // below 0.5 threshold
	}
	ctx := result.FormatGroundingContext()
	if strings.Contains(ctx, "grep") {
		t.Error("low-score tool should not be included in context")
	}
}

// --- Cosine Similarity Tests ---

func TestEmbedCosineSimilarityIdentical(t *testing.T) {
	a := []float64{1, 2, 3}
	sim := embedCosineSimilarity(a, a)
	if math.Abs(sim-1.0) > 0.001 {
		t.Errorf("identical vectors should have similarity 1.0, got %f", sim)
	}
}

func TestEmbedCosineSimilarityOrthogonal(t *testing.T) {
	a := []float64{1, 0, 0}
	b := []float64{0, 1, 0}
	sim := embedCosineSimilarity(a, b)
	if math.Abs(sim) > 0.001 {
		t.Errorf("orthogonal vectors should have similarity 0, got %f", sim)
	}
}

func TestEmbedCosineSimilarityOpposite(t *testing.T) {
	a := []float64{1, 0}
	b := []float64{-1, 0}
	sim := embedCosineSimilarity(a, b)
	if math.Abs(sim-(-1.0)) > 0.001 {
		t.Errorf("opposite vectors should have similarity -1, got %f", sim)
	}
}

func TestEmbedCosineSimilarityEmpty(t *testing.T) {
	if embedCosineSimilarity(nil, nil) != 0 {
		t.Error("empty vectors should return 0")
	}
	if embedCosineSimilarity([]float64{1}, []float64{1, 2}) != 0 {
		t.Error("different length vectors should return 0")
	}
}

func TestEmbedCosineSimilarityZero(t *testing.T) {
	a := []float64{0, 0, 0}
	b := []float64{1, 2, 3}
	sim := embedCosineSimilarity(a, b)
	if sim != 0 {
		t.Errorf("zero vector should return 0 similarity, got %f", sim)
	}
}

// --- Benchmark ---

func BenchmarkEmbedCosineSimilarity(b *testing.B) {
	a := make([]float64, 768) // nomic-embed-text dimension
	c := make([]float64, 768)
	for i := range a {
		a[i] = float64(i) * 0.01
		c[i] = float64(768-i) * 0.01
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		embedCosineSimilarity(a, c)
	}
}

func BenchmarkGround(b *testing.B) {
	eg := NewEmbedGrounder(mockEmbedFn())
	for i := 0; i < 10; i++ {
		eg.IndexTool("tool"+string(rune('a'+i)), "description")
	}
	for i := 0; i < 50; i++ {
		eg.IndexFile("file"+string(rune('a'+i))+".go", "summary")
	}
	for i := 0; i < 100; i++ {
		eg.RecordSuccess("query", "grep", nil)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		eg.Ground("search for Pipeline")
	}
}
