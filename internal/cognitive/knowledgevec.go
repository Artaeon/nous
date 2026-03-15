package cognitive

import (
	"bufio"
	"encoding/json"
	"math"
	"os"
	"sort"
	"strings"
	"sync"
)

// KnowledgeVec is a vector knowledge store that gives Nous access to
// unlimited knowledge — just like a human uses an encyclopedia.
//
// Innovation: The model doesn't need to KNOW quantum physics from its
// weights. It needs to SEARCH for it. KnowledgeVec stores text chunks
// with their embeddings and retrieves the most relevant ones for any
// query. Combined with the Exocortex, this means Nous can answer
// "what is quantum physics" by looking it up, synthesizing the answer,
// and NEVER hallucinating because the facts come from the store.
//
// This is deeper than standard RAG:
//   - Integrated with the Exocortex (tool-level, not prompt-level)
//   - Knowledge feeds the Model Compiler (frequent lookups get baked in)
//   - Neural Cortex learns WHEN to search (not every query needs it)
//   - Verification Oracle checks answers against retrieved knowledge
//
// Usage:
//   kv.Ingest("path/to/knowledge.txt")  // one-time
//   results := kv.Search("quantum physics", 3)  // instant
//   // results[0].Text = "Quantum physics is the study of..."
type KnowledgeVec struct {
	mu        sync.RWMutex
	chunks    []KnowledgeChunk
	embedFunc func(string) ([]float64, error)
	path      string

	// Stats
	SearchCount int `json:"search_count"`
	HitCount    int `json:"hit_count"`
}

// KnowledgeChunk is one piece of knowledge with its embedding.
type KnowledgeChunk struct {
	Text      string    `json:"text"`
	Source    string    `json:"source"`
	Embedding []float64 `json:"embedding"`
}

// KnowledgeResult is a search result with relevance score.
type KnowledgeResult struct {
	Text   string
	Source string
	Score  float64
}

// NewKnowledgeVec creates a new knowledge vector store.
func NewKnowledgeVec(embedFunc func(string) ([]float64, error), path string) *KnowledgeVec {
	kv := &KnowledgeVec{
		embedFunc: embedFunc,
		path:      path,
	}

	// Try loading existing store
	if path != "" {
		kv.Load()
	}

	return kv
}

// AddChunk adds a single knowledge chunk with embedding.
func (kv *KnowledgeVec) AddChunk(text, source string) error {
	if strings.TrimSpace(text) == "" {
		return nil
	}

	embedding, err := kv.embedFunc(text)
	if err != nil {
		return err
	}

	kv.mu.Lock()
	kv.chunks = append(kv.chunks, KnowledgeChunk{
		Text:      text,
		Source:    source,
		Embedding: embedding,
	})
	kv.mu.Unlock()

	return nil
}

// AddChunkWithEmbedding adds a pre-embedded chunk (for bulk loading).
func (kv *KnowledgeVec) AddChunkWithEmbedding(text, source string, embedding []float64) {
	kv.mu.Lock()
	kv.chunks = append(kv.chunks, KnowledgeChunk{
		Text:      text,
		Source:    source,
		Embedding: embedding,
	})
	kv.mu.Unlock()
}

// Search finds the most relevant knowledge chunks for a query.
func (kv *KnowledgeVec) Search(query string, topK int) ([]KnowledgeResult, error) {
	if kv.embedFunc == nil {
		return nil, nil
	}

	queryVec, err := kv.embedFunc(query)
	if err != nil {
		return nil, err
	}

	return kv.SearchByVector(queryVec, topK), nil
}

// SearchByVector finds the most relevant chunks using a pre-computed query vector.
func (kv *KnowledgeVec) SearchByVector(queryVec []float64, topK int) []KnowledgeResult {
	kv.mu.RLock()
	defer kv.mu.RUnlock()

	if len(kv.chunks) == 0 {
		return nil
	}

	kv.SearchCount++

	type scored struct {
		idx   int
		score float64
	}

	scores := make([]scored, 0, len(kv.chunks))
	for i, chunk := range kv.chunks {
		if len(chunk.Embedding) == 0 {
			continue
		}
		sim := cosineSim(queryVec, chunk.Embedding)
		scores = append(scores, scored{idx: i, score: sim})
	}

	sort.Slice(scores, func(i, j int) bool {
		return scores[i].score > scores[j].score
	})

	if topK > len(scores) {
		topK = len(scores)
	}

	results := make([]KnowledgeResult, topK)
	for i := 0; i < topK; i++ {
		chunk := kv.chunks[scores[i].idx]
		results[i] = KnowledgeResult{
			Text:   chunk.Text,
			Source: chunk.Source,
			Score:  scores[i].score,
		}
	}

	if len(results) > 0 && results[0].Score > 0.5 {
		kv.HitCount++
	}

	return results
}

// Ingest reads a text file and splits it into chunks for knowledge storage.
func (kv *KnowledgeVec) Ingest(filePath string) (int, error) {
	f, err := os.Open(filePath)
	if err != nil {
		return 0, err
	}
	defer f.Close()

	// Read and split into chunks
	var chunks []string
	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 1024*1024), 1024*1024)

	var current strings.Builder
	wordCount := 0

	for scanner.Scan() {
		line := scanner.Text()

		// Split on empty lines (paragraph boundaries) or when chunk is large enough
		if strings.TrimSpace(line) == "" && wordCount > 50 {
			if text := strings.TrimSpace(current.String()); text != "" {
				chunks = append(chunks, text)
			}
			current.Reset()
			wordCount = 0
			continue
		}

		// Also split on markdown headers
		if strings.HasPrefix(line, "# ") && wordCount > 20 {
			if text := strings.TrimSpace(current.String()); text != "" {
				chunks = append(chunks, text)
			}
			current.Reset()
			wordCount = 0
		}

		if current.Len() > 0 {
			current.WriteString("\n")
		}
		current.WriteString(line)
		wordCount += len(strings.Fields(line))

		// Hard limit per chunk
		if wordCount > 300 {
			if text := strings.TrimSpace(current.String()); text != "" {
				chunks = append(chunks, text)
			}
			current.Reset()
			wordCount = 0
		}
	}

	// Flush remaining
	if text := strings.TrimSpace(current.String()); text != "" {
		chunks = append(chunks, text)
	}

	if err := scanner.Err(); err != nil {
		return 0, err
	}

	// Embed and store each chunk
	added := 0
	for _, chunk := range chunks {
		if err := kv.AddChunk(chunk, filePath); err != nil {
			continue // skip chunks that fail to embed
		}
		added++
	}

	// Auto-save after ingestion
	if kv.path != "" {
		kv.Save()
	}

	return added, nil
}

// IngestText ingests raw text directly (not from file).
func (kv *KnowledgeVec) IngestText(text, source string) (int, error) {
	paragraphs := strings.Split(text, "\n\n")
	added := 0
	for _, p := range paragraphs {
		p = strings.TrimSpace(p)
		if len(p) < 20 {
			continue // skip very short chunks
		}
		if err := kv.AddChunk(p, source); err != nil {
			continue
		}
		added++
	}
	return added, nil
}

// FormatContext formats search results as context for the LLM.
func FormatKnowledgeContext(results []KnowledgeResult) string {
	if len(results) == 0 {
		return ""
	}

	var sb strings.Builder
	sb.WriteString("Relevant knowledge:\n")
	for i, r := range results {
		if r.Score < 0.3 {
			continue // skip low-relevance results
		}
		sb.WriteString(strings.Repeat("-", 40))
		sb.WriteString("\n")
		if r.Source != "" {
			sb.WriteString("[")
			sb.WriteString(r.Source)
			sb.WriteString("]\n")
		}
		sb.WriteString(r.Text)
		sb.WriteString("\n")
		if i >= 2 {
			break // max 3 chunks
		}
	}
	return sb.String()
}

// Size returns the number of stored chunks.
func (kv *KnowledgeVec) Size() int {
	kv.mu.RLock()
	defer kv.mu.RUnlock()
	return len(kv.chunks)
}

// Save persists the knowledge store to disk.
func (kv *KnowledgeVec) Save() error {
	kv.mu.RLock()
	defer kv.mu.RUnlock()

	if kv.path == "" {
		return nil
	}

	data := struct {
		Chunks      []KnowledgeChunk `json:"chunks"`
		SearchCount int              `json:"search_count"`
		HitCount    int              `json:"hit_count"`
	}{
		Chunks:      kv.chunks,
		SearchCount: kv.SearchCount,
		HitCount:    kv.HitCount,
	}

	b, err := json.Marshal(data)
	if err != nil {
		return err
	}
	return os.WriteFile(kv.path, b, 0644)
}

// Load restores the knowledge store from disk.
func (kv *KnowledgeVec) Load() error {
	if kv.path == "" {
		return nil
	}

	b, err := os.ReadFile(kv.path)
	if err != nil {
		return err
	}

	var data struct {
		Chunks      []KnowledgeChunk `json:"chunks"`
		SearchCount int              `json:"search_count"`
		HitCount    int              `json:"hit_count"`
	}

	if err := json.Unmarshal(b, &data); err != nil {
		return err
	}

	kv.mu.Lock()
	kv.chunks = data.Chunks
	kv.SearchCount = data.SearchCount
	kv.HitCount = data.HitCount
	kv.mu.Unlock()

	return nil
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
