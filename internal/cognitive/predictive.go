package cognitive

import (
	"strings"
	"sync"
	"time"
)

// PredictiveWarmer pre-warms context caches based on predicted next queries.
// It runs in the background after each interaction, speculatively fetching
// embeddings and context for likely follow-up queries.
//
// How it works:
//   - After each query, analyze patterns to predict what the user might ask next
//   - Pre-compute embeddings for predicted queries (fills EmbedCache)
//   - Pre-weave context for predicted queries (fills VirtualContext caches)
//   - When the predicted query actually arrives, response is near-instant
type PredictiveWarmer struct {
	mu         sync.Mutex
	vctx       *VirtualContext
	embedCache *EmbedCache
	embedFunc  func(text string) []float64

	// Query history for pattern prediction
	history     []queryRecord
	maxHistory  int
	predictions []string // most recent predictions

	// Stats
	totalPredictions int
	hits             int // predictions that matched actual queries
}

type queryRecord struct {
	query string
	at    time.Time
}

// NewPredictiveWarmer creates a warmer that pre-computes context.
func NewPredictiveWarmer(vctx *VirtualContext, embedCache *EmbedCache) *PredictiveWarmer {
	return &PredictiveWarmer{
		vctx:       vctx,
		embedCache: embedCache,
		maxHistory: 50,
	}
}

// SetEmbedFunc sets the embedding function for pre-warming.
func (pw *PredictiveWarmer) SetEmbedFunc(fn func(string) []float64) {
	pw.mu.Lock()
	defer pw.mu.Unlock()
	pw.embedFunc = fn
}

// AfterQuery should be called after each user query completes.
// It predicts follow-up queries and pre-warms caches in the background.
func (pw *PredictiveWarmer) AfterQuery(query string) {
	pw.mu.Lock()
	pw.history = append(pw.history, queryRecord{query: query, at: time.Now()})
	if len(pw.history) > pw.maxHistory {
		pw.history = pw.history[1:]
	}
	pw.mu.Unlock()

	// Predict and warm in background
	predicted := pw.predict(query)
	if len(predicted) == 0 {
		return
	}

	pw.mu.Lock()
	pw.predictions = predicted
	pw.totalPredictions += len(predicted)
	pw.mu.Unlock()

	go pw.warmPredictions(predicted)
}

// CheckHit records whether a query matched a prediction.
func (pw *PredictiveWarmer) CheckHit(query string) bool {
	pw.mu.Lock()
	defer pw.mu.Unlock()

	lower := strings.ToLower(query)
	for _, pred := range pw.predictions {
		if strings.Contains(lower, strings.ToLower(pred)) ||
			strings.Contains(strings.ToLower(pred), lower) {
			pw.hits++
			return true
		}
	}
	return false
}

// Stats returns prediction statistics.
func (pw *PredictiveWarmer) Stats() (total, hits int) {
	pw.mu.Lock()
	defer pw.mu.Unlock()
	return pw.totalPredictions, pw.hits
}

// predict generates likely follow-up queries based on the current query
// and conversation history.
func (pw *PredictiveWarmer) predict(query string) []string {
	var predictions []string

	lower := strings.ToLower(query)

	// Pattern 1: If user asked about a file, predict they'll ask about related files
	if containsFileRef(lower) {
		predictions = append(predictions, extractFileRef(lower)+" test")
		predictions = append(predictions, "explain "+extractFileRef(lower))
	}

	// Pattern 2: If user asked "what is X", predict "how does X work"
	if strings.HasPrefix(lower, "what is ") || strings.HasPrefix(lower, "what are ") {
		topic := strings.TrimPrefix(strings.TrimPrefix(lower, "what is "), "what are ")
		topic = strings.TrimRight(topic, "?. ")
		if topic != "" {
			predictions = append(predictions, "how does "+topic+" work")
			predictions = append(predictions, "example of "+topic)
		}
	}

	// Pattern 3: If user asked "how to X", predict "show me X example"
	if strings.HasPrefix(lower, "how to ") || strings.HasPrefix(lower, "how do i ") {
		topic := strings.TrimPrefix(strings.TrimPrefix(lower, "how to "), "how do i ")
		topic = strings.TrimRight(topic, "?. ")
		if topic != "" {
			predictions = append(predictions, topic+" example")
		}
	}

	// Pattern 4: Repeated topic detection from history
	pw.mu.Lock()
	topics := pw.extractRepeatedTopics()
	pw.mu.Unlock()
	for _, topic := range topics {
		if !strings.Contains(lower, strings.ToLower(topic)) {
			predictions = append(predictions, "more about "+topic)
		}
	}

	// Limit predictions
	if len(predictions) > 5 {
		predictions = predictions[:5]
	}

	return predictions
}

// warmPredictions pre-computes embeddings and context for predicted queries.
func (pw *PredictiveWarmer) warmPredictions(predictions []string) {
	pw.mu.Lock()
	embedFn := pw.embedFunc
	pw.mu.Unlock()

	for _, pred := range predictions {
		// Pre-warm embedding cache
		if embedFn != nil && pw.embedCache != nil {
			if cached := pw.embedCache.Get(pred); cached == nil {
				vec := embedFn(pred)
				if len(vec) > 0 {
					pw.embedCache.Put(pred, vec)
				}
			}
		}

		// Pre-warm virtual context
		if pw.vctx != nil {
			pw.vctx.Weave(pred)
		}
	}
}

// extractRepeatedTopics finds topics that appear in multiple recent queries.
func (pw *PredictiveWarmer) extractRepeatedTopics() []string {
	if len(pw.history) < 2 {
		return nil
	}

	// Extract significant words (>4 chars) from recent queries
	wordCounts := make(map[string]int)
	recent := pw.history
	if len(recent) > 10 {
		recent = recent[len(recent)-10:]
	}
	for _, qr := range recent {
		seen := make(map[string]bool)
		for _, word := range strings.Fields(strings.ToLower(qr.query)) {
			word = strings.Trim(word, "?.,!;:'\"")
			if len(word) > 4 && !isStopWord(word) && !seen[word] {
				wordCounts[word]++
				seen[word] = true
			}
		}
	}

	var topics []string
	for word, count := range wordCounts {
		if count >= 2 {
			topics = append(topics, word)
		}
	}
	if len(topics) > 3 {
		topics = topics[:3]
	}
	return topics
}

// containsFileRef checks if a query references a file.
func containsFileRef(s string) bool {
	for _, ext := range []string{".go", ".py", ".js", ".ts", ".rs", ".java", ".md", ".json", ".yaml", ".yml", ".toml"} {
		if strings.Contains(s, ext) {
			return true
		}
	}
	return false
}

// extractFileRef extracts the first file reference from a query.
func extractFileRef(s string) string {
	for _, word := range strings.Fields(s) {
		if containsFileRef(word) {
			return word
		}
	}
	return ""
}

// isStopWord returns true for common English stop words.
func isStopWord(word string) bool {
	stops := map[string]bool{
		"about": true, "could": true, "would": true, "should": true,
		"their": true, "there": true, "these": true, "those": true,
		"which": true, "where": true, "while": true, "being": true,
		"doing": true, "having": true, "other": true, "every": true,
		"after": true, "before": true, "between": true, "under": true,
		"above": true, "below": true, "again": true, "further": true,
	}
	return stops[word]
}
