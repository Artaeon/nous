package memory

import (
	"math"
	"sync"
	"time"
)

// Slot represents an item in working memory with a decay timer.
type Slot struct {
	Key        string
	Value      interface{}
	Relevance  float64
	CreatedAt  time.Time
	AccessedAt time.Time
	Embedding  []float64 // semantic vector for similarity search
}

// WorkingMemory is a capacity-limited, decay-based short-term store.
// Items lose relevance over time and are evicted when capacity is exceeded.
// Supports both recency-based and semantic similarity retrieval.
type WorkingMemory struct {
	mu        sync.RWMutex
	slots     map[string]*Slot
	capacity  int
	decayRate float64   // relevance decay per second
	embedFn   EmbedFunc // optional: computes embeddings for semantic search
}

func NewWorkingMemory(capacity int) *WorkingMemory {
	return &WorkingMemory{
		slots:     make(map[string]*Slot),
		capacity:  capacity,
		decayRate: 0.01,
	}
}

// SetEmbedFunc configures the embedding function for semantic retrieval.
// If not set, SemanticSearch falls back to MostRelevant.
func (wm *WorkingMemory) SetEmbedFunc(fn EmbedFunc) {
	wm.mu.Lock()
	defer wm.mu.Unlock()
	wm.embedFn = fn
}

// Store adds or updates an item in working memory.
// Embeddings are computed asynchronously if an embed function is configured.
func (wm *WorkingMemory) Store(key string, value interface{}, relevance float64) {
	wm.mu.Lock()
	slot := &Slot{
		Key:        key,
		Value:      value,
		Relevance:  relevance,
		CreatedAt:  time.Now(),
		AccessedAt: time.Now(),
	}
	wm.slots[key] = slot
	wm.evictIfNeeded()
	embedFn := wm.embedFn
	wm.mu.Unlock()

	// Compute embedding asynchronously to avoid blocking the caller
	if embedFn != nil {
		go func() {
			text := key
			if s, ok := value.(string); ok {
				text = key + " " + s
			}
			if len(text) > 300 {
				text = text[:300]
			}
			vec, err := embedFn(text)
			if err != nil || len(vec) == 0 {
				return
			}
			wm.mu.Lock()
			if s, ok := wm.slots[key]; ok {
				s.Embedding = vec
			}
			wm.mu.Unlock()
		}()
	}
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

// MostRelevant returns the top-n items by current relevance (decay-adjusted).
func (wm *WorkingMemory) MostRelevant(n int) []Slot {
	wm.mu.RLock()
	defer wm.mu.RUnlock()

	return wm.topByDecay(n)
}

// SemanticSearch returns the top-n items most similar to the query embedding.
// Combines semantic similarity (70%) with recency/relevance (30%) for ranking.
// Falls back to MostRelevant if no embeddings are available.
func (wm *WorkingMemory) SemanticSearch(queryVec []float64, n int) []Slot {
	wm.mu.RLock()
	defer wm.mu.RUnlock()

	if len(queryVec) == 0 {
		return wm.topByDecay(n)
	}

	now := time.Now()
	type scored struct {
		slot  Slot
		score float64
	}

	var withEmbed []scored
	var withoutEmbed int

	for _, s := range wm.slots {
		elapsed := now.Sub(s.AccessedAt).Seconds()
		decayScore := s.Relevance - (wm.decayRate * elapsed)
		if decayScore <= 0 {
			continue
		}

		if len(s.Embedding) > 0 {
			sim := cosineSim(queryVec, s.Embedding)
			// Blend: 70% semantic similarity, 30% recency/relevance
			combined := sim*0.7 + decayScore*0.3
			withEmbed = append(withEmbed, scored{slot: *s, score: combined})
		} else {
			withoutEmbed++
		}
	}

	// If fewer than half the slots have embeddings, fall back to decay-based
	if len(withEmbed) == 0 || withoutEmbed > len(withEmbed)*2 {
		return wm.topByDecay(n)
	}

	// Sort by combined score descending
	for i := 1; i < len(withEmbed); i++ {
		for j := i; j > 0 && withEmbed[j].score > withEmbed[j-1].score; j-- {
			withEmbed[j], withEmbed[j-1] = withEmbed[j-1], withEmbed[j]
		}
	}

	if n > len(withEmbed) {
		n = len(withEmbed)
	}

	result := make([]Slot, n)
	for i := 0; i < n; i++ {
		result[i] = withEmbed[i].slot
	}
	return result
}

// Size returns the current number of items.
func (wm *WorkingMemory) Size() int {
	wm.mu.RLock()
	defer wm.mu.RUnlock()
	return len(wm.slots)
}

// topByDecay returns the top-n items by decay-adjusted relevance.
// Must be called with mu held.
func (wm *WorkingMemory) topByDecay(n int) []Slot {
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

func (wm *WorkingMemory) evictIfNeeded() {
	if len(wm.slots) <= wm.capacity {
		return
	}

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

// cosineSim computes cosine similarity between two vectors.
func cosineSim(a, b []float64) float64 {
	if len(a) != len(b) || len(a) == 0 {
		return 0
	}

	var dot, normA, normB float64
	for i := range a {
		dot += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}

	denom := math.Sqrt(normA) * math.Sqrt(normB)
	if denom == 0 {
		return 0
	}

	return dot / denom
}

func min(a, b float64) float64 {
	if a < b {
		return a
	}
	return b
}
