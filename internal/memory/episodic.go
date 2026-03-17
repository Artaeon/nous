package memory

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/artaeon/nous/internal/safefile"
	"github.com/artaeon/nous/internal/simd"
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
		sim := simd.CosineSimilarity(queryVec, ep.Embedding)
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

// TopicFrequency returns the most frequently discussed topics across all episodes,
// enabling the system to understand what the user cares about.
func (em *EpisodicMemory) TopicFrequency(limit int) []TagCount {
	em.mu.RLock()
	defer em.mu.RUnlock()

	freq := make(map[string]int)
	for _, ep := range em.episodes {
		for _, tag := range ep.Tags {
			freq[tag]++
		}
	}

	var counts []TagCount
	for tag, count := range freq {
		if count >= 2 { // only topics mentioned 2+ times
			counts = append(counts, TagCount{Tag: tag, Count: count})
		}
	}

	// Sort by count descending
	for i := 0; i < len(counts); i++ {
		for j := i + 1; j < len(counts); j++ {
			if counts[j].Count > counts[i].Count {
				counts[i], counts[j] = counts[j], counts[i]
			}
		}
	}

	if len(counts) > limit {
		counts = counts[:limit]
	}
	return counts
}

// TagCount holds a topic tag and its occurrence count.
type TagCount struct {
	Tag   string
	Count int
}

// RelatedEpisodes finds episodes that share tags with the given episode.
// This enables "you also asked about X" style connections.
func (em *EpisodicMemory) RelatedEpisodes(epID string, limit int) []Episode {
	em.mu.RLock()
	defer em.mu.RUnlock()

	// Find the source episode
	var sourceTags map[string]bool
	for _, ep := range em.episodes {
		if ep.ID == epID {
			sourceTags = make(map[string]bool, len(ep.Tags))
			for _, t := range ep.Tags {
				sourceTags[t] = true
			}
			break
		}
	}
	if sourceTags == nil {
		return nil
	}

	type scored struct {
		ep    Episode
		score int
	}
	var matches []scored
	for _, ep := range em.episodes {
		if ep.ID == epID {
			continue
		}
		score := 0
		for _, t := range ep.Tags {
			if sourceTags[t] {
				score++
			}
		}
		if score > 0 {
			matches = append(matches, scored{ep: ep, score: score})
		}
	}

	// Sort by overlap score desc
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

	return safefile.WriteAtomicWithBackup(filepath.Join(em.storePath, "episodes.json"), data, 0644, safefile.MaxBackups)
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

// SuccessPattern represents a recurring successful tool sequence mined from episodes.
type SuccessPattern struct {
	Tools     []string // ordered tool sequence
	Count     int      // how many episodes used this sequence successfully
	AvgDurMs  int64    // average duration in ms
	Keywords  []string // common keywords across matching episodes
}

// SuccessPatterns mines episodic memory for recurring successful tool sequences.
// Returns patterns sorted by frequency — the most common successful sequences first.
func (em *EpisodicMemory) SuccessPatterns(minOccurrences int) []SuccessPattern {
	em.mu.RLock()
	defer em.mu.RUnlock()

	if minOccurrences < 2 {
		minOccurrences = 2
	}

	// Count tool sequence occurrences across successful episodes
	type seqStats struct {
		count    int
		totalDur int64
		allTags  map[string]int
	}
	seqMap := make(map[string]*seqStats)

	for _, ep := range em.episodes {
		if !ep.Success || len(ep.ToolsUsed) == 0 {
			continue
		}
		key := strings.Join(ep.ToolsUsed, "→")
		ss, ok := seqMap[key]
		if !ok {
			ss = &seqStats{allTags: make(map[string]int)}
			seqMap[key] = ss
		}
		ss.count++
		ss.totalDur += ep.Duration
		for _, tag := range ep.Tags {
			ss.allTags[tag]++
		}
	}

	// Filter by minimum occurrences and build result
	var patterns []SuccessPattern
	for key, ss := range seqMap {
		if ss.count < minOccurrences {
			continue
		}
		tools := strings.Split(key, "→")
		avgDur := ss.totalDur / int64(ss.count)

		// Extract top keywords (appearing in >50% of episodes)
		var keywords []string
		threshold := ss.count / 2
		if threshold < 1 {
			threshold = 1
		}
		for tag, cnt := range ss.allTags {
			if cnt >= threshold {
				keywords = append(keywords, tag)
			}
		}

		patterns = append(patterns, SuccessPattern{
			Tools:    tools,
			Count:    ss.count,
			AvgDurMs: avgDur,
			Keywords: keywords,
		})
	}

	// Sort by count descending
	for i := 0; i < len(patterns); i++ {
		for j := i + 1; j < len(patterns); j++ {
			if patterns[j].Count > patterns[i].Count {
				patterns[i], patterns[j] = patterns[j], patterns[i]
			}
		}
	}

	return patterns
}

// SuccessfulToolEpisodes returns successful episodes that used a specific tool.
func (em *EpisodicMemory) SuccessfulToolEpisodes(tool string, limit int) []Episode {
	em.mu.RLock()
	defer em.mu.RUnlock()

	var results []Episode
	// Iterate backwards (most recent first)
	for i := len(em.episodes) - 1; i >= 0 && len(results) < limit; i-- {
		ep := em.episodes[i]
		if !ep.Success {
			continue
		}
		for _, t := range ep.ToolsUsed {
			if t == tool {
				results = append(results, ep)
				break
			}
		}
	}
	return results
}

// FailurePatterns returns the most common failure scenarios for learning.
func (em *EpisodicMemory) FailurePatterns(limit int) []SuccessPattern {
	em.mu.RLock()
	defer em.mu.RUnlock()

	seqMap := make(map[string]*struct {
		count    int
		totalDur int64
		allTags  map[string]int
	})

	for _, ep := range em.episodes {
		if ep.Success || len(ep.ToolsUsed) == 0 {
			continue
		}
		key := strings.Join(ep.ToolsUsed, "→")
		ss, ok := seqMap[key]
		if !ok {
			ss = &struct {
				count    int
				totalDur int64
				allTags  map[string]int
			}{allTags: make(map[string]int)}
			seqMap[key] = ss
		}
		ss.count++
		ss.totalDur += ep.Duration
		for _, tag := range ep.Tags {
			ss.allTags[tag]++
		}
	}

	var patterns []SuccessPattern
	for key, ss := range seqMap {
		tools := strings.Split(key, "→")
		avgDur := int64(0)
		if ss.count > 0 {
			avgDur = ss.totalDur / int64(ss.count)
		}
		patterns = append(patterns, SuccessPattern{
			Tools:    tools,
			Count:    ss.count,
			AvgDurMs: avgDur,
		})
	}

	// Sort by count descending
	for i := 0; i < len(patterns); i++ {
		for j := i + 1; j < len(patterns); j++ {
			if patterns[j].Count > patterns[i].Count {
				patterns[i], patterns[j] = patterns[j], patterns[i]
			}
		}
	}

	if len(patterns) > limit {
		patterns = patterns[:limit]
	}
	return patterns
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

// cosineSimilarity delegates to the shared SIMD-optimized implementation.
func cosineSimilarity(a, b []float64) float64 {
	return simd.CosineSimilarity(a, b)
}
