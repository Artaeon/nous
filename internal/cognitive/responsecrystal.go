package cognitive

import (
	"encoding/json"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/artaeon/nous/internal/safefile"
)

// ResponseCrystal stores a high-quality LLM response for future instant reuse.
// When a similar query arrives, the crystal serves the cached response (0ms)
// instead of calling the LLM again (20-40s on CPU).
type ResponseCrystal struct {
	Query     string    `json:"query"`
	Response  string    `json:"response"`
	Embedding []float64 `json:"embedding"`
	Quality   float64   `json:"quality"`
	Uses      int       `json:"uses"`
	CreatedAt time.Time `json:"created_at"`
	LastUsed  time.Time `json:"last_used"`
}

// ResponseCrystalStore is a semantic cache that learns from every LLM response.
// Over time, it "compiles" conversations into instant deterministic answers.
type ResponseCrystalStore struct {
	mu         sync.RWMutex
	crystals   []ResponseCrystal
	embedFunc  func(string) ([]float64, error)
	embedCache *EmbedCache // optional LRU cache for embedding vectors
	path       string
	threshold  float64 // similarity threshold for cache hit (default 0.82)
	maxSize    int     // maximum crystals to store (default 500)
}

// NewResponseCrystalStore creates a store that learns from LLM responses.
func NewResponseCrystalStore(embedFunc func(string) ([]float64, error), storePath string) *ResponseCrystalStore {
	s := &ResponseCrystalStore{
		embedFunc: embedFunc,
		path:      filepath.Join(storePath, "response_crystals.json"),
		threshold: 0.78,
		maxSize:   500,
	}
	s.load()
	return s
}

// SetEmbedCache attaches an EmbedCache to avoid redundant embedding calls.
// When set, Lookup and Learn check the cache before calling embedFunc.
func (s *ResponseCrystalStore) SetEmbedCache(cache *EmbedCache) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.embedCache = cache
}

// cachedEmbed returns an embedding, checking the cache first if available.
func (s *ResponseCrystalStore) cachedEmbed(text string) ([]float64, error) {
	if s.embedCache != nil {
		if vec := s.embedCache.Get(text); vec != nil {
			return vec, nil
		}
	}
	vec, err := s.embedFunc(text)
	if err != nil {
		return nil, err
	}
	if s.embedCache != nil && len(vec) > 0 {
		s.embedCache.Put(text, vec)
	}
	return vec, nil
}

// Lookup finds a cached response for a query using semantic similarity.
// Returns the response and true if a high-quality match is found.
// Fast path: exact text match (~0ms) before falling back to embedding similarity.
func (s *ResponseCrystalStore) Lookup(query string) (string, bool) {
	if len(query) < 10 {
		return "", false
	}

	// Fast path: exact text match — no embedding call needed (~0ms)
	normalized := strings.ToLower(strings.TrimSpace(query))
	s.mu.Lock()
	for i, c := range s.crystals {
		if strings.ToLower(strings.TrimSpace(c.Query)) == normalized {
			s.crystals[i].Uses++
			s.crystals[i].LastUsed = time.Now()
			resp := s.crystals[i].Response
			s.mu.Unlock()
			return resp, true
		}
	}
	s.mu.Unlock()

	// Slow path: semantic similarity via embedding
	if s.embedFunc == nil {
		return "", false
	}

	queryVec, err := s.cachedEmbed(query)
	if err != nil || len(queryVec) == 0 {
		return "", false
	}

	// Use full write lock to avoid RUnlock→Lock race window
	s.mu.Lock()
	defer s.mu.Unlock()

	bestScore := 0.0
	bestIdx := -1

	for i, c := range s.crystals {
		if len(c.Embedding) == 0 {
			continue
		}
		score := cosineSim(queryVec, c.Embedding)
		if score > bestScore {
			bestScore = score
			bestIdx = i
		}
	}

	if bestIdx < 0 || bestScore < s.threshold {
		return "", false
	}

	// Safe to update under the same write lock — no race window
	s.crystals[bestIdx].Uses++
	s.crystals[bestIdx].LastUsed = time.Now()

	return s.crystals[bestIdx].Response, true
}

// Learn stores a high-quality LLM response for future reuse.
// Only stores responses with quality >= 0.6 and length >= 20 chars.
// The embedding call and save are synchronous to prevent data loss
// when the server is killed before async goroutines complete.
func (s *ResponseCrystalStore) Learn(query, response string, quality float64) {
	if s.embedFunc == nil || quality < 0.6 || len(response) < 20 || len(query) < 10 {
		return
	}

	// Don't cache very long responses (tool dumps, raw data that changes)
	if len(response) > 5000 {
		return
	}

	queryVec, err := s.cachedEmbed(query)
	if err != nil || len(queryVec) == 0 {
		return
	}

	s.mu.Lock()

	// Check for existing similar crystal — update instead of duplicate
	for i, c := range s.crystals {
		if len(c.Embedding) > 0 && cosineSim(queryVec, c.Embedding) > 0.90 {
			// Update existing crystal if new response is higher quality
			if quality > c.Quality {
				s.crystals[i].Response = response
				s.crystals[i].Quality = quality
				s.crystals[i].LastUsed = time.Now()
			}
			s.mu.Unlock()
			return
		}
	}

	// Add new crystal
	s.crystals = append(s.crystals, ResponseCrystal{
		Query:     query,
		Response:  response,
		Embedding: queryVec,
		Quality:   quality,
		CreatedAt: time.Now(),
		LastUsed:  time.Now(),
	})

	// Prune if over capacity — remove lowest quality + least used
	if len(s.crystals) > s.maxSize {
		s.prune()
	}
	s.mu.Unlock()

	// Synchronous save — ensures crystals persist even if server is killed
	s.save()
}

// Size returns the number of cached crystals.
func (s *ResponseCrystalStore) Size() int {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return len(s.crystals)
}

// Stats returns cache statistics.
func (s *ResponseCrystalStore) Stats() (size int, totalHits int) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	for _, c := range s.crystals {
		totalHits += c.Uses
	}
	return len(s.crystals), totalHits
}

// Flush persists all crystals to disk. Call on graceful shutdown.
func (s *ResponseCrystalStore) Flush() {
	s.save()
}

func (s *ResponseCrystalStore) prune() {
	// Score = quality * 0.5 + recency * 0.3 + usage * 0.2
	type scored struct {
		idx   int
		score float64
	}
	var scores []scored
	now := time.Now()
	for i, c := range s.crystals {
		age := now.Sub(c.LastUsed).Hours() / 24.0 / 30.0 // months
		recency := 1.0 / (1.0 + age)
		usage := float64(c.Uses) / 100.0
		if usage > 1.0 {
			usage = 1.0
		}
		sc := c.Quality*0.5 + recency*0.3 + usage*0.2
		scores = append(scores, scored{i, sc})
	}
	sort.Slice(scores, func(i, j int) bool { return scores[i].score > scores[j].score })

	keep := make([]ResponseCrystal, 0, s.maxSize)
	for i := 0; i < s.maxSize && i < len(scores); i++ {
		keep = append(keep, s.crystals[scores[i].idx])
	}
	s.crystals = keep
}

func (s *ResponseCrystalStore) save() {
	s.mu.RLock()
	data, err := json.Marshal(s.crystals)
	s.mu.RUnlock()
	if err != nil {
		return
	}
	safefile.WriteAtomicWithBackup(s.path, data, 0644, 3)
}

func (s *ResponseCrystalStore) load() {
	data, err := os.ReadFile(s.path)
	if err != nil {
		return
	}
	var crystals []ResponseCrystal
	if err := json.Unmarshal(data, &crystals); err != nil {
		return
	}
	s.crystals = crystals
}

// cosineSim is defined in knowledgevec.go (shared across cognitive package)
