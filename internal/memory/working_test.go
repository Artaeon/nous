package memory

import (
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
