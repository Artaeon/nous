package memory

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"
)

// Episode is a single interaction stored with full context.
// This is the building block of Nous's autobiographical memory —
// every conversation, every tool call, every outcome, forever.
type Episode struct {
	ID        string    `json:"id"`
	Timestamp time.Time `json:"timestamp"`
	Input     string    `json:"input"`      // what the user said
	Intent    string    `json:"intent"`     // perceived intent
	Output    string    `json:"output"`     // what Nous answered
	ToolsUsed []string  `json:"tools_used"` // tools invoked during this episode
	Success   bool      `json:"success"`    // whether it ended well
	Duration  int64     `json:"duration_ms"`
	Tags      []string  `json:"tags"`       // auto-extracted topic tags

	// Embedding vector for semantic search (computed lazily)
	Embedding []float64 `json:"embedding,omitempty"`
}

// EmbedFunc is a function that generates embedding vectors.
// Injected from the Ollama client to keep memory package dependency-free.
type EmbedFunc func(text string) ([]float64, error)

// EpisodicMemory stores every interaction for total recall.
// It supports both keyword and semantic (embedding-based) search.
type EpisodicMemory struct {
	mu        sync.RWMutex
	episodes  []Episode
	storePath string
	maxSize   int
	embedFn   EmbedFunc
}

// NewEpisodicMemory creates or loads an episodic memory store.
func NewEpisodicMemory(storePath string, embedFn EmbedFunc) *EpisodicMemory {
	em := &EpisodicMemory{
		storePath: storePath,
		maxSize:   10000, // 10K episodes (~5MB on disk)
		embedFn:   embedFn,
	}
	em.load()
	return em
}

// Record stores a new episode. Embeddings are computed asynchronously.
func (em *EpisodicMemory) Record(ep Episode) {
	ep.ID = episodeID(ep.Timestamp)
	ep.Tags = extractTags(ep.Input)

	em.mu.Lock()
	em.episodes = append(em.episodes, ep)

	// Prune oldest if over capacity
	if len(em.episodes) > em.maxSize {
		em.episodes = em.episodes[len(em.episodes)-em.maxSize:]
	}
	em.mu.Unlock()

	// Compute embedding asynchronously
	if em.embedFn != nil {
		go func() {
			text := ep.Input + " " + ep.Output
			if len(text) > 500 {
				text = text[:500]
			}
			vec, err := em.embedFn(text)
			if err != nil {
				return
			}
			em.mu.Lock()
			for i := range em.episodes {
				if em.episodes[i].ID == ep.ID {
					em.episodes[i].Embedding = vec
					break
				}
			}
			em.mu.Unlock()
		}()
	}

	// Auto-save periodically (every 10 episodes)
	em.mu.RLock()
	count := len(em.episodes)
	em.mu.RUnlock()
	if count%10 == 0 {
		go em.Save()
	}
}

// SearchKeyword returns episodes matching keyword query (most recent first).
func (em *EpisodicMemory) SearchKeyword(query string, limit int) []Episode {
	em.mu.RLock()
	defer em.mu.RUnlock()

	words := strings.Fields(strings.ToLower(query))
	if len(words) == 0 {
		return nil
	}

	type scored struct {
		ep    Episode
		score int
	}

	var matches []scored

	for _, ep := range em.episodes {
		text := strings.ToLower(ep.Input + " " + ep.Output + " " + strings.Join(ep.Tags, " "))
		score := 0
		for _, w := range words {
			if len(w) < 2 {
				continue
			}
			if strings.Contains(text, w) {
				score++
			}
		}
		if score > 0 {
			matches = append(matches, scored{ep: ep, score: score})
		}
	}

	// Sort by score desc, then recency
	for i := 0; i < len(matches); i++ {
		for j := i + 1; j < len(matches); j++ {
			if matches[j].score > matches[i].score ||
				(matches[j].score == matches[i].score &&
					matches[j].ep.Timestamp.After(matches[i].ep.Timestamp)) {
				matches[i], matches[j] = matches[j], matches[i]
			}
		}
	}

	if len(matches) > limit {
		matches = matches[:limit]
	}

	results := make([]Episode, len(matches))
	for i, m := range matches {
		results[i] = m.ep
	}
	return results
}

// SearchSemantic returns episodes most similar to the query embedding.
// This is the real ML-powered search — cosine similarity over embedding vectors.
func (em *EpisodicMemory) SearchSemantic(queryVec []float64, limit int) []Episode {
	em.mu.RLock()
	defer em.mu.RUnlock()

	if len(queryVec) == 0 {
		return nil
	}

	type scored struct {
		ep    Episode
		score float64
	}

	var matches []scored

	for _, ep := range em.episodes {
		if len(ep.Embedding) == 0 {
			continue
		}
		sim := cosineSimilarity(queryVec, ep.Embedding)
		if sim > 0.3 { // minimum similarity threshold
			matches = append(matches, scored{ep: ep, score: sim})
		}
	}

	// Sort by similarity descending
	for i := 0; i < len(matches); i++ {
		for j := i + 1; j < len(matches); j++ {
			if matches[j].score > matches[i].score {
				matches[i], matches[j] = matches[j], matches[i]
			}
		}
	}

	if len(matches) > limit {
		matches = matches[:limit]
	}

	results := make([]Episode, len(matches))
	for i, m := range matches {
		results[i] = m.ep
	}
	return results
}

// Search performs hybrid search: semantic if embeddings available, keyword fallback.
func (em *EpisodicMemory) Search(query string, limit int) []Episode {
	// Try semantic search first
	if em.embedFn != nil {
		vec, err := em.embedFn(query)
		if err == nil && len(vec) > 0 {
			results := em.SearchSemantic(vec, limit)
			if len(results) > 0 {
				return results
			}
		}
	}

	// Fallback to keyword search
	return em.SearchKeyword(query, limit)
}

// Recent returns the N most recent episodes.
func (em *EpisodicMemory) Recent(n int) []Episode {
	em.mu.RLock()
	defer em.mu.RUnlock()

	if n > len(em.episodes) {
		n = len(em.episodes)
	}

	// Return most recent (end of slice)
	result := make([]Episode, n)
	copy(result, em.episodes[len(em.episodes)-n:])

	// Reverse to get newest first
	for i, j := 0, len(result)-1; i < j; i, j = i+1, j-1 {
		result[i], result[j] = result[j], result[i]
	}

	return result
}

// Size returns total episodes stored.
func (em *EpisodicMemory) Size() int {
	em.mu.RLock()
	defer em.mu.RUnlock()
	return len(em.episodes)
}

// SuccessRate returns the ratio of successful episodes.
func (em *EpisodicMemory) SuccessRate() float64 {
	em.mu.RLock()
	defer em.mu.RUnlock()

	if len(em.episodes) == 0 {
		return 0
	}

	successes := 0
	for _, ep := range em.episodes {
		if ep.Success {
			successes++
		}
	}
	return float64(successes) / float64(len(em.episodes))
}

// ToolUsageStats returns how many times each tool has been used across all episodes.
func (em *EpisodicMemory) ToolUsageStats() map[string]int {
	em.mu.RLock()
	defer em.mu.RUnlock()

	stats := make(map[string]int)
	for _, ep := range em.episodes {
		for _, tool := range ep.ToolsUsed {
			stats[tool]++
		}
	}
	return stats
}

// Save persists episodes to disk.
func (em *EpisodicMemory) Save() error {
	em.mu.RLock()
	data, err := json.Marshal(em.episodes)
	em.mu.RUnlock()
	if err != nil {
		return err
	}

	if err := os.MkdirAll(em.storePath, 0755); err != nil {
		return err
	}

	return os.WriteFile(filepath.Join(em.storePath, "episodes.json"), data, 0644)
}

func (em *EpisodicMemory) load() {
	if em.storePath == "" {
		return
	}
	data, err := os.ReadFile(filepath.Join(em.storePath, "episodes.json"))
	if err != nil {
		return
	}
	em.mu.Lock()
	defer em.mu.Unlock()
	_ = json.Unmarshal(data, &em.episodes)
}

// --- helpers ---

func episodeID(t time.Time) string {
	return t.Format("20060102-150405.000")
}

func extractTags(input string) []string {
	words := strings.Fields(strings.ToLower(input))
	var tags []string
	seen := make(map[string]bool)

	stopWords := map[string]bool{
		"the": true, "and": true, "for": true, "that": true, "this": true,
		"with": true, "from": true, "are": true, "was": true, "have": true,
		"what": true, "how": true, "can": true, "you": true, "please": true,
	}

	for _, w := range words {
		w = strings.Trim(w, ".,!?;:'\"()[]")
		if len(w) >= 3 && !stopWords[w] && !seen[w] {
			seen[w] = true
			tags = append(tags, w)
		}
	}

	if len(tags) > 8 {
		tags = tags[:8]
	}
	return tags
}

// cosineSimilarity computes the cosine similarity between two vectors.
// Returns 0.0-1.0 (higher = more similar).
func cosineSimilarity(a, b []float64) float64 {
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
