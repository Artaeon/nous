package memory

import (
	"sync/atomic"
	"testing"
	"time"
)

func TestNewWorkingMemory(t *testing.T) {
	wm := NewWorkingMemory(10)
	if wm == nil {
		t.Fatal("NewWorkingMemory returned nil")
	}
	if wm.capacity != 10 {
		t.Errorf("expected capacity 10, got %d", wm.capacity)
	}
	if wm.Size() != 0 {
		t.Errorf("expected size 0, got %d", wm.Size())
	}
}

func TestStoreAndRetrieve(t *testing.T) {
	wm := NewWorkingMemory(10)

	wm.Store("language", "Go", 0.8)

	val, ok := wm.Retrieve("language")
	if !ok {
		t.Fatal("expected to retrieve stored item")
	}
	if val != "Go" {
		t.Errorf("expected 'Go', got %v", val)
	}
}

func TestRetrieveNonExistent(t *testing.T) {
	wm := NewWorkingMemory(10)

	_, ok := wm.Retrieve("nonexistent")
	if ok {
		t.Error("expected ok=false for nonexistent key")
	}
}

func TestRetrieveBoostsRelevance(t *testing.T) {
	wm := NewWorkingMemory(10)
	wm.Store("key", "value", 0.5)

	// Access the item to boost relevance
	wm.Retrieve("key")

	// Check that the slot's relevance was boosted
	wm.mu.RLock()
	slot := wm.slots["key"]
	wm.mu.RUnlock()

	expected := 0.6 // 0.5 + 0.1 access boost
	if slot.Relevance != expected {
		t.Errorf("expected relevance %.1f after access, got %.1f", expected, slot.Relevance)
	}
}

func TestRetrieveRelevanceCappedAtOne(t *testing.T) {
	wm := NewWorkingMemory(10)
	wm.Store("key", "value", 0.95)

	wm.Retrieve("key") // 0.95 + 0.1 should be capped at 1.0

	wm.mu.RLock()
	slot := wm.slots["key"]
	wm.mu.RUnlock()

	if slot.Relevance != 1.0 {
		t.Errorf("expected relevance capped at 1.0, got %f", slot.Relevance)
	}
}

func TestStoreOverwrite(t *testing.T) {
	wm := NewWorkingMemory(10)
	wm.Store("key", "v1", 0.5)
	wm.Store("key", "v2", 0.9)

	val, _ := wm.Retrieve("key")
	if val != "v2" {
		t.Errorf("expected 'v2', got %v", val)
	}
}

func TestSize(t *testing.T) {
	wm := NewWorkingMemory(10)

	wm.Store("a", 1, 0.5)
	wm.Store("b", 2, 0.6)
	wm.Store("c", 3, 0.7)

	if wm.Size() != 3 {
		t.Errorf("expected size 3, got %d", wm.Size())
	}
}

func TestCapacityEviction(t *testing.T) {
	wm := NewWorkingMemory(3)

	wm.Store("a", 1, 0.3)
	wm.Store("b", 2, 0.9)
	wm.Store("c", 3, 0.7)

	if wm.Size() != 3 {
		t.Fatalf("expected size 3, got %d", wm.Size())
	}

	// Adding a 4th item should evict the least relevant one ("a" at 0.3)
	wm.Store("d", 4, 0.8)

	if wm.Size() != 3 {
		t.Errorf("expected size 3 after eviction, got %d", wm.Size())
	}

	// "a" should have been evicted as it had the lowest relevance
	_, ok := wm.Retrieve("a")
	if ok {
		t.Error("expected 'a' to be evicted (lowest relevance)")
	}

	// Others should still exist
	for _, key := range []string{"b", "c", "d"} {
		if _, ok := wm.Retrieve(key); !ok {
			t.Errorf("expected key %q to still exist", key)
		}
	}
}

func TestCapacityOfOne(t *testing.T) {
	wm := NewWorkingMemory(1)

	wm.Store("a", 1, 0.5)
	wm.Store("b", 2, 0.9)

	if wm.Size() != 1 {
		t.Errorf("expected size 1, got %d", wm.Size())
	}

	// "b" should remain (higher relevance)
	val, ok := wm.Retrieve("b")
	if !ok {
		t.Error("expected 'b' to exist")
	}
	if val != 2 {
		t.Errorf("expected value 2, got %v", val)
	}
}

func TestMostRelevant(t *testing.T) {
	wm := NewWorkingMemory(10)

	wm.Store("low", "L", 0.2)
	wm.Store("mid", "M", 0.5)
	wm.Store("high", "H", 0.9)

	top := wm.MostRelevant(2)
	if len(top) != 2 {
		t.Fatalf("expected 2 results, got %d", len(top))
	}

	if top[0].Key != "high" {
		t.Errorf("expected first result key='high', got %q", top[0].Key)
	}
	if top[1].Key != "mid" {
		t.Errorf("expected second result key='mid', got %q", top[1].Key)
	}
}

func TestMostRelevantMoreThanAvailable(t *testing.T) {
	wm := NewWorkingMemory(10)
	wm.Store("only", "one", 0.5)

	top := wm.MostRelevant(5)
	if len(top) != 1 {
		t.Errorf("expected 1 result, got %d", len(top))
	}
}

func TestMostRelevantEmpty(t *testing.T) {
	wm := NewWorkingMemory(10)
	top := wm.MostRelevant(5)
	if len(top) != 0 {
		t.Errorf("expected 0 results, got %d", len(top))
	}
}

func TestRelevanceDecayOverTime(t *testing.T) {
	wm := NewWorkingMemory(10)

	// Store an item with very low relevance so that decay pushes it to <= 0
	wm.Store("ephemeral", "temp", 0.001)

	// Manipulate the AccessedAt time to simulate passage of time
	wm.mu.Lock()
	slot := wm.slots["ephemeral"]
	slot.AccessedAt = time.Now().Add(-10 * time.Minute)
	wm.mu.Unlock()

	// With decayRate=0.01 per second and 600 seconds elapsed,
	// effective score = 0.001 - (0.01 * 600) = -5.999, which is < 0
	// So MostRelevant should not include this item
	top := wm.MostRelevant(10)
	if len(top) != 0 {
		t.Errorf("expected decayed item to be excluded from MostRelevant, got %d items", len(top))
	}
}

func TestSetEmbedFunc(t *testing.T) {
	wm := NewWorkingMemory(10)

	var called atomic.Bool
	wm.SetEmbedFunc(func(text string) ([]float64, error) {
		called.Store(true)
		return []float64{1.0, 0.0, 0.0}, nil
	})

	wm.Store("test", "value", 0.8)

	// Give async goroutine time to run
	time.Sleep(50 * time.Millisecond)

	if !called.Load() {
		t.Error("expected embed function to be called on Store")
	}

	// Verify embedding was stored
	wm.mu.RLock()
	slot := wm.slots["test"]
	wm.mu.RUnlock()

	if len(slot.Embedding) != 3 {
		t.Errorf("expected 3-dim embedding, got %d", len(slot.Embedding))
	}
}

func TestSemanticSearchWithEmbeddings(t *testing.T) {
	wm := NewWorkingMemory(10)

	// Store items with pre-set embeddings (bypass async)
	wm.mu.Lock()
	wm.slots["go code"] = &Slot{
		Key: "go code", Value: "wrote a Go function",
		Relevance: 0.8, AccessedAt: time.Now(),
		Embedding: []float64{1.0, 0.0, 0.0}, // "code" direction
	}
	wm.slots["dinner"] = &Slot{
		Key: "dinner", Value: "had pasta for dinner",
		Relevance: 0.8, AccessedAt: time.Now(),
		Embedding: []float64{0.0, 1.0, 0.0}, // "food" direction
	}
	wm.slots["testing"] = &Slot{
		Key: "testing", Value: "ran unit tests",
		Relevance: 0.8, AccessedAt: time.Now(),
		Embedding: []float64{0.9, 0.1, 0.0}, // similar to "code"
	}
	wm.mu.Unlock()

	// Search for something similar to "code" direction
	query := []float64{1.0, 0.0, 0.0}
	results := wm.SemanticSearch(query, 2)

	if len(results) != 2 {
		t.Fatalf("expected 2 results, got %d", len(results))
	}

	// "go code" should be most similar (exact match)
	if results[0].Key != "go code" {
		t.Errorf("expected first result to be 'go code', got %q", results[0].Key)
	}

	// "testing" should be second (0.9 cosine similarity)
	if results[1].Key != "testing" {
		t.Errorf("expected second result to be 'testing', got %q", results[1].Key)
	}
}

func TestSemanticSearchFallsBackWhenNoEmbeddings(t *testing.T) {
	wm := NewWorkingMemory(10)

	wm.Store("a", "val", 0.9)
	wm.Store("b", "val", 0.5)

	// Search with empty query vector — should fall back to decay-based
	results := wm.SemanticSearch(nil, 2)
	if len(results) != 2 {
		t.Fatalf("expected 2 results, got %d", len(results))
	}
	if results[0].Key != "a" {
		t.Errorf("expected first result to be 'a' (highest relevance), got %q", results[0].Key)
	}
}

func TestSemanticSearchFallsBackWhenFewEmbeddings(t *testing.T) {
	wm := NewWorkingMemory(10)

	// 3 items without embeddings, 1 with — should fall back
	wm.Store("a", "val", 0.9)
	wm.Store("b", "val", 0.8)
	wm.Store("c", "val", 0.7)

	wm.mu.Lock()
	wm.slots["d"] = &Slot{
		Key: "d", Value: "val",
		Relevance: 0.6, AccessedAt: time.Now(),
		Embedding: []float64{1.0, 0.0},
	}
	wm.mu.Unlock()

	results := wm.SemanticSearch([]float64{1.0, 0.0}, 4)
	if len(results) != 4 {
		t.Fatalf("expected 4 results (fallback), got %d", len(results))
	}
	// Should be ordered by relevance (decay-based fallback)
	if results[0].Key != "a" {
		t.Errorf("expected first result to be 'a', got %q", results[0].Key)
	}
}

func TestCosineSim(t *testing.T) {
	// Identical vectors → 1.0
	sim := cosineSim([]float64{1, 0, 0}, []float64{1, 0, 0})
	if sim < 0.99 {
		t.Errorf("expected ~1.0, got %f", sim)
	}

	// Orthogonal vectors → 0.0
	sim = cosineSim([]float64{1, 0, 0}, []float64{0, 1, 0})
	if sim > 0.01 {
		t.Errorf("expected ~0.0, got %f", sim)
	}

	// Empty vectors → 0.0
	sim = cosineSim(nil, nil)
	if sim != 0 {
		t.Errorf("expected 0.0 for nil, got %f", sim)
	}
}

func TestMinHelper(t *testing.T) {
	tests := []struct {
		a, b, expected float64
	}{
		{1.0, 2.0, 1.0},
		{2.0, 1.0, 1.0},
		{1.0, 1.0, 1.0},
		{-1.0, 0.0, -1.0},
	}

	for _, tt := range tests {
		got := min(tt.a, tt.b)
		if got != tt.expected {
			t.Errorf("min(%f, %f) = %f, want %f", tt.a, tt.b, got, tt.expected)
		}
	}
}
