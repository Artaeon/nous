package memory

import (
	"math"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestEpisodicMemoryRecord(t *testing.T) {
	em := NewEpisodicMemory("", nil)

	em.Record(Episode{
		Timestamp: time.Now(),
		Input:     "show me the reasoner code",
		Output:    "Here is the reasoner...",
		ToolsUsed: []string{"grep", "read"},
		Success:   true,
	})

	if em.Size() != 1 {
		t.Errorf("expected 1 episode, got %d", em.Size())
	}
}

func TestEpisodicMemoryRecent(t *testing.T) {
	em := NewEpisodicMemory("", nil)

	for i := 0; i < 5; i++ {
		em.Record(Episode{
			Timestamp: time.Now().Add(time.Duration(i) * time.Second),
			Input:     "query " + string(rune('A'+i)),
			Output:    "answer",
			Success:   true,
		})
	}

	recent := em.Recent(3)
	if len(recent) != 3 {
		t.Fatalf("expected 3 recent, got %d", len(recent))
	}

	// Should be newest first
	if recent[0].Input != "query E" {
		t.Errorf("most recent should be 'query E', got %q", recent[0].Input)
	}
}

func TestEpisodicMemoryRecentMoreThanAvailable(t *testing.T) {
	em := NewEpisodicMemory("", nil)
	em.Record(Episode{Timestamp: time.Now(), Input: "only one"})

	recent := em.Recent(10)
	if len(recent) != 1 {
		t.Errorf("expected 1, got %d", len(recent))
	}
}

func TestEpisodicMemorySearchKeyword(t *testing.T) {
	em := NewEpisodicMemory("", nil)

	em.Record(Episode{
		Timestamp: time.Now(),
		Input:     "how does the reasoner work",
		Output:    "The reasoner uses a pipeline...",
		Success:   true,
	})
	em.Record(Episode{
		Timestamp: time.Now(),
		Input:     "list all files",
		Output:    "main.go README.md",
		Success:   true,
	})

	results := em.SearchKeyword("reasoner pipeline", 5)
	if len(results) == 0 {
		t.Fatal("expected results for 'reasoner pipeline'")
	}
	if results[0].Input != "how does the reasoner work" {
		t.Errorf("best match should be reasoner episode, got %q", results[0].Input)
	}
}

func TestEpisodicMemorySearchKeywordNoResults(t *testing.T) {
	em := NewEpisodicMemory("", nil)
	em.Record(Episode{Timestamp: time.Now(), Input: "hello"})

	results := em.SearchKeyword("kubernetes deployment yaml", 5)
	if len(results) != 0 {
		t.Errorf("expected no results, got %d", len(results))
	}
}

func TestEpisodicMemorySearchSemantic(t *testing.T) {
	em := NewEpisodicMemory("", nil)

	// Manually set embeddings
	em.mu.Lock()
	em.episodes = append(em.episodes, Episode{
		ID:        "ep1",
		Input:     "show reasoner code",
		Embedding: []float64{1.0, 0.0, 0.0},
	})
	em.episodes = append(em.episodes, Episode{
		ID:        "ep2",
		Input:     "list directory files",
		Embedding: []float64{0.0, 1.0, 0.0},
	})
	em.mu.Unlock()

	// Query vector close to ep1
	results := em.SearchSemantic([]float64{0.9, 0.1, 0.0}, 5)
	if len(results) == 0 {
		t.Fatal("expected semantic results")
	}
	if results[0].ID != "ep1" {
		t.Errorf("best match should be ep1, got %s", results[0].ID)
	}
}

func TestEpisodicMemorySearchSemanticEmpty(t *testing.T) {
	em := NewEpisodicMemory("", nil)

	results := em.SearchSemantic([]float64{1.0, 0.0}, 5)
	if len(results) != 0 {
		t.Error("empty memory should return no results")
	}
}

func TestEpisodicMemorySearchSemanticNilVector(t *testing.T) {
	em := NewEpisodicMemory("", nil)
	results := em.SearchSemantic(nil, 5)
	if len(results) != 0 {
		t.Error("nil vector should return no results")
	}
}

func TestEpisodicMemorySuccessRate(t *testing.T) {
	em := NewEpisodicMemory("", nil)

	em.Record(Episode{Timestamp: time.Now(), Success: true})
	em.Record(Episode{Timestamp: time.Now(), Success: true})
	em.Record(Episode{Timestamp: time.Now(), Success: false})

	rate := em.SuccessRate()
	expected := 2.0 / 3.0
	if math.Abs(rate-expected) > 0.01 {
		t.Errorf("success rate = %f, want ~%f", rate, expected)
	}
}

func TestEpisodicMemorySuccessRateEmpty(t *testing.T) {
	em := NewEpisodicMemory("", nil)
	if em.SuccessRate() != 0 {
		t.Error("empty memory should have 0 success rate")
	}
}

func TestEpisodicMemoryToolUsageStats(t *testing.T) {
	em := NewEpisodicMemory("", nil)

	em.Record(Episode{Timestamp: time.Now(), ToolsUsed: []string{"read", "grep"}})
	em.Record(Episode{Timestamp: time.Now(), ToolsUsed: []string{"read", "ls"}})

	stats := em.ToolUsageStats()
	if stats["read"] != 2 {
		t.Errorf("read usage = %d, want 2", stats["read"])
	}
	if stats["grep"] != 1 {
		t.Errorf("grep usage = %d, want 1", stats["grep"])
	}
}

func TestEpisodicMemoryPersistence(t *testing.T) {
	dir := t.TempDir()

	em1 := NewEpisodicMemory(dir, nil)
	em1.Record(Episode{
		Timestamp: time.Now(),
		Input:     "test persistence",
		Output:    "it works",
		Success:   true,
	})
	em1.Save()

	// Verify file exists
	if _, err := os.Stat(filepath.Join(dir, "episodes.json")); err != nil {
		t.Fatalf("episodes.json not found: %v", err)
	}

	// Load from disk
	em2 := NewEpisodicMemory(dir, nil)
	if em2.Size() != 1 {
		t.Errorf("expected 1 episode after reload, got %d", em2.Size())
	}
}

func TestEpisodicMemoryPruning(t *testing.T) {
	em := NewEpisodicMemory("", nil)
	em.maxSize = 5

	for i := 0; i < 10; i++ {
		em.Record(Episode{
			Timestamp: time.Now().Add(time.Duration(i) * time.Second),
			Input:     "query",
		})
	}

	if em.Size() > 5 {
		t.Errorf("should prune to maxSize 5, got %d", em.Size())
	}
}

func TestEpisodicMemoryTags(t *testing.T) {
	em := NewEpisodicMemory("", nil)
	em.Record(Episode{
		Timestamp: time.Now(),
		Input:     "show me the reasoner code in Go",
	})

	em.mu.RLock()
	defer em.mu.RUnlock()
	tags := em.episodes[0].Tags
	if len(tags) == 0 {
		t.Fatal("expected auto-extracted tags")
	}

	// Should contain "show", "reasoner", "code" but not "me", "the", "in"
	found := make(map[string]bool)
	for _, tag := range tags {
		found[tag] = true
	}
	if !found["reasoner"] {
		t.Errorf("expected 'reasoner' in tags, got: %v", tags)
	}
	if found["the"] || found["me"] {
		t.Errorf("stop words should be filtered, got: %v", tags)
	}
}

func TestCosineSimilarity(t *testing.T) {
	// Identical vectors = 1.0
	sim := cosineSimilarity([]float64{1, 0, 0}, []float64{1, 0, 0})
	if math.Abs(sim-1.0) > 0.001 {
		t.Errorf("identical vectors: sim = %f, want 1.0", sim)
	}

	// Orthogonal vectors = 0.0
	sim = cosineSimilarity([]float64{1, 0, 0}, []float64{0, 1, 0})
	if math.Abs(sim) > 0.001 {
		t.Errorf("orthogonal vectors: sim = %f, want 0.0", sim)
	}

	// Opposite vectors = -1.0
	sim = cosineSimilarity([]float64{1, 0}, []float64{-1, 0})
	if math.Abs(sim-(-1.0)) > 0.001 {
		t.Errorf("opposite vectors: sim = %f, want -1.0", sim)
	}

	// Different lengths = 0
	sim = cosineSimilarity([]float64{1, 0}, []float64{1, 0, 0})
	if sim != 0 {
		t.Errorf("different lengths: sim = %f, want 0", sim)
	}

	// Empty = 0
	sim = cosineSimilarity(nil, nil)
	if sim != 0 {
		t.Errorf("nil vectors: sim = %f, want 0", sim)
	}
}

func TestExtractTags(t *testing.T) {
	tags := extractTags("How can I read the main.go file?")
	found := make(map[string]bool)
	for _, tag := range tags {
		found[tag] = true
	}

	if found["how"] || found["can"] || found["the"] {
		t.Errorf("stop words should be filtered: %v", tags)
	}
	if !found["read"] || !found["file"] {
		t.Errorf("expected 'read' and 'file' in tags: %v", tags)
	}
}

func TestExtractTagsLimit(t *testing.T) {
	tags := extractTags("one two three four five six seven eight nine ten eleven twelve")
	if len(tags) > 8 {
		t.Errorf("tags should be capped at 8, got %d", len(tags))
	}
}

func TestEpisodicMemoryHybridSearch(t *testing.T) {
	em := NewEpisodicMemory("", nil)

	// Without embeddings, Search should fall back to keyword
	em.Record(Episode{
		Timestamp: time.Now(),
		Input:     "find the bug in reasoner",
		Output:    "Found it!",
	})

	results := em.Search("reasoner bug", 5)
	if len(results) == 0 {
		t.Error("hybrid search should fall back to keyword")
	}
}
