package cognitive

import (
	"crypto/sha256"
	"sync"
	"time"
)

// EmbedCache is an LRU cache for embedding vectors with TTL expiration.
// Eliminates redundant embedding calls (500ms-2s each) for the same or
// repeated text within a reasoning cycle.
type EmbedCache struct {
	mu       sync.RWMutex
	entries  map[[32]byte]*embedEntry
	order    [][32]byte // LRU order (most recent at end)
	maxSize  int
	ttl      time.Duration
	hits     int64
	misses   int64
}

type embedEntry struct {
	vec       []float64
	createdAt time.Time
}

// NewEmbedCache creates a new embedding cache.
func NewEmbedCache(maxSize int, ttl time.Duration) *EmbedCache {
	return &EmbedCache{
		entries: make(map[[32]byte]*embedEntry, maxSize),
		maxSize: maxSize,
		ttl:     ttl,
	}
}

// Get returns a cached embedding vector for the given text, or nil if not found/expired.
func (ec *EmbedCache) Get(text string) []float64 {
	key := sha256.Sum256([]byte(text))

	ec.mu.RLock()
	entry, ok := ec.entries[key]
	ec.mu.RUnlock()

	if !ok {
		ec.mu.Lock()
		ec.misses++
		ec.mu.Unlock()
		return nil
	}

	if time.Since(entry.createdAt) > ec.ttl {
		ec.mu.Lock()
		delete(ec.entries, key)
		ec.misses++
		ec.mu.Unlock()
		return nil
	}

	ec.mu.Lock()
	ec.hits++
	// Move to end of LRU order
	for i, k := range ec.order {
		if k == key {
			ec.order = append(ec.order[:i], ec.order[i+1:]...)
			ec.order = append(ec.order, key)
			break
		}
	}
	ec.mu.Unlock()

	return entry.vec
}

// Put stores an embedding vector in the cache.
func (ec *EmbedCache) Put(text string, vec []float64) {
	if len(vec) == 0 {
		return
	}

	key := sha256.Sum256([]byte(text))

	ec.mu.Lock()
	defer ec.mu.Unlock()

	// Evict LRU if at capacity
	if len(ec.entries) >= ec.maxSize {
		if len(ec.order) > 0 {
			oldest := ec.order[0]
			ec.order = ec.order[1:]
			delete(ec.entries, oldest)
		}
	}

	ec.entries[key] = &embedEntry{
		vec:       vec,
		createdAt: time.Now(),
	}
	ec.order = append(ec.order, key)
}

// Stats returns cache hit/miss statistics.
func (ec *EmbedCache) Stats() (hits, misses int64) {
	ec.mu.RLock()
	defer ec.mu.RUnlock()
	return ec.hits, ec.misses
}

// Size returns the number of cached entries.
func (ec *EmbedCache) Size() int {
	ec.mu.RLock()
	defer ec.mu.RUnlock()
	return len(ec.entries)
}
