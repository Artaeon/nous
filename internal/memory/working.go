package memory

import (
	"sync"
	"time"
)

// Slot represents an item in working memory with a decay timer.
type Slot struct {
	Key       string
	Value     interface{}
	Relevance float64
	CreatedAt time.Time
	AccessedAt time.Time
}

// WorkingMemory is a capacity-limited, decay-based short-term store.
// Items lose relevance over time and are evicted when capacity is exceeded.
type WorkingMemory struct {
	mu       sync.RWMutex
	slots    map[string]*Slot
	capacity int
	decayRate float64 // relevance decay per second
}

func NewWorkingMemory(capacity int) *WorkingMemory {
	return &WorkingMemory{
		slots:     make(map[string]*Slot),
		capacity:  capacity,
		decayRate: 0.01,
	}
}

// Store adds or updates an item in working memory.
func (wm *WorkingMemory) Store(key string, value interface{}, relevance float64) {
	wm.mu.Lock()
	defer wm.mu.Unlock()

	wm.slots[key] = &Slot{
		Key:        key,
		Value:      value,
		Relevance:  relevance,
		CreatedAt:  time.Now(),
		AccessedAt: time.Now(),
	}

	wm.evictIfNeeded()
}

// Retrieve gets an item and boosts its relevance (recency effect).
func (wm *WorkingMemory) Retrieve(key string) (interface{}, bool) {
	wm.mu.Lock()
	defer wm.mu.Unlock()

	slot, ok := wm.slots[key]
	if !ok {
		return nil, false
	}

	slot.AccessedAt = time.Now()
	slot.Relevance = min(1.0, slot.Relevance+0.1) // Access boost

	return slot.Value, true
}

// MostRelevant returns the top-n items by current relevance.
func (wm *WorkingMemory) MostRelevant(n int) []Slot {
	wm.mu.RLock()
	defer wm.mu.RUnlock()

	// Apply decay
	now := time.Now()
	type scored struct {
		slot  Slot
		score float64
	}

	var items []scored
	for _, s := range wm.slots {
		elapsed := now.Sub(s.AccessedAt).Seconds()
		score := s.Relevance - (wm.decayRate * elapsed)
		if score > 0 {
			items = append(items, scored{slot: *s, score: score})
		}
	}

	// Sort by score descending (simple insertion sort for small N)
	for i := 1; i < len(items); i++ {
		for j := i; j > 0 && items[j].score > items[j-1].score; j-- {
			items[j], items[j-1] = items[j-1], items[j]
		}
	}

	if n > len(items) {
		n = len(items)
	}

	result := make([]Slot, n)
	for i := 0; i < n; i++ {
		result[i] = items[i].slot
	}
	return result
}

// Size returns the current number of items.
func (wm *WorkingMemory) Size() int {
	wm.mu.RLock()
	defer wm.mu.RUnlock()
	return len(wm.slots)
}

func (wm *WorkingMemory) evictIfNeeded() {
	if len(wm.slots) <= wm.capacity {
		return
	}

	// Find the least relevant item
	var minKey string
	minScore := 2.0

	now := time.Now()
	for key, s := range wm.slots {
		elapsed := now.Sub(s.AccessedAt).Seconds()
		score := s.Relevance - (wm.decayRate * elapsed)
		if score < minScore {
			minScore = score
			minKey = key
		}
	}

	if minKey != "" {
		delete(wm.slots, minKey)
	}
}

func min(a, b float64) float64 {
	if a < b {
		return a
	}
	return b
}
