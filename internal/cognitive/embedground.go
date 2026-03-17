package cognitive

import (
	"strings"
	"sync"
	"time"

	"github.com/artaeon/nous/internal/simd"
)

// EmbedGrounder uses embedding vectors as a semantic secondary brain.
// Instead of relying solely on the LLM to understand what the user means,
// it uses a separate embedding model (nomic-embed-text, ~274MB) to compute
// semantic similarity between queries and known concepts.
//
// Innovation: The LLM generates text; embeddings compute meaning. By using
// the embedding model as an oracle for semantic questions, the LLM is freed
// from tasks it's bad at (resolving ambiguous references, finding similar
// past tool calls) and can focus on what it's good at (natural language).
//
// Use cases:
//   - Tool argument resolution: "that file" → find semantically closest file path
//   - Tool selection: which tool description is closest to the query?
//   - Past interaction recall: find similar past queries that succeeded
//   - Concept grounding: "the authentication module" → auth.go, middleware.go
type EmbedGrounder struct {
	embedFn   EmbedFunc
	toolIndex map[string][]float64 // tool name → description embedding
	fileIndex map[string][]float64 // file path → content embedding
	histIndex []historyEntry       // past successful interactions
	mu        sync.RWMutex
}

// EmbedFunc generates an embedding vector for text.
type EmbedFunc func(text string) ([]float64, error)

// historyEntry stores a past successful interaction with its embedding.
type historyEntry struct {
	Query     string
	Tool      string
	Args      map[string]string
	Embedding []float64
	Timestamp time.Time
}

// GroundedResult holds the result of embedding-based grounding.
type GroundedResult struct {
	// Recommended tool based on semantic similarity
	Tool       string
	ToolScore  float64

	// Resolved arguments from semantic matching
	ResolvedArgs map[string]string

	// Similar past interactions
	SimilarPast []PastMatch
}

// PastMatch is a previous interaction with semantic similarity.
type PastMatch struct {
	Query      string
	Tool       string
	Args       map[string]string
	Similarity float64
}

// NewEmbedGrounder creates a new embedding-based grounder.
func NewEmbedGrounder(embedFn EmbedFunc) *EmbedGrounder {
	return &EmbedGrounder{
		embedFn:   embedFn,
		toolIndex: make(map[string][]float64),
		fileIndex: make(map[string][]float64),
	}
}

// IndexTool computes and stores the embedding for a tool's description.
// Call this once per tool at startup.
func (eg *EmbedGrounder) IndexTool(name, description string) error {
	if eg.embedFn == nil {
		return nil
	}

	vec, err := eg.embedFn(name + ": " + description)
	if err != nil {
		return err
	}

	eg.mu.Lock()
	eg.toolIndex[name] = vec
	eg.mu.Unlock()
	return nil
}

// IndexFile computes and stores the embedding for a file path.
// Uses the file path and a brief content summary as embedding input.
func (eg *EmbedGrounder) IndexFile(path, summary string) error {
	if eg.embedFn == nil {
		return nil
	}

	text := path
	if summary != "" {
		text += ": " + summary
	}

	vec, err := eg.embedFn(text)
	if err != nil {
		return err
	}

	eg.mu.Lock()
	eg.fileIndex[path] = vec
	eg.mu.Unlock()
	return nil
}

// RecordSuccess stores a successful interaction for future reference.
func (eg *EmbedGrounder) RecordSuccess(query, tool string, args map[string]string) {
	if eg.embedFn == nil {
		return
	}

	vec, err := eg.embedFn(query)
	if err != nil {
		return
	}

	eg.mu.Lock()
	eg.histIndex = append(eg.histIndex, historyEntry{
		Query:     query,
		Tool:      tool,
		Args:      args,
		Embedding: vec,
		Timestamp: time.Now(),
	})

	// Keep at most 500 entries
	if len(eg.histIndex) > 500 {
		eg.histIndex = eg.histIndex[len(eg.histIndex)-500:]
	}
	eg.mu.Unlock()
}

// Ground performs semantic grounding for a query.
// Returns tool recommendations, resolved arguments, and similar past interactions.
func (eg *EmbedGrounder) Ground(query string) (*GroundedResult, error) {
	if eg.embedFn == nil {
		return &GroundedResult{}, nil
	}

	queryVec, err := eg.embedFn(query)
	if err != nil {
		return nil, err
	}

	result := &GroundedResult{
		ResolvedArgs: make(map[string]string),
	}

	eg.mu.RLock()
	defer eg.mu.RUnlock()

	// Find best matching tool
	result.Tool, result.ToolScore = eg.findBestTool(queryVec)

	// Find similar past interactions
	result.SimilarPast = eg.findSimilarPast(queryVec, 3)

	// Resolve file path arguments from semantic similarity
	if path, score := eg.findBestFile(queryVec); score > 0.5 {
		result.ResolvedArgs["path"] = path
	}

	return result, nil
}

// ResolveFile finds the most semantically similar file path to the query.
func (eg *EmbedGrounder) ResolveFile(query string) (string, float64, error) {
	if eg.embedFn == nil {
		return "", 0, nil
	}

	queryVec, err := eg.embedFn(query)
	if err != nil {
		return "", 0, err
	}

	eg.mu.RLock()
	defer eg.mu.RUnlock()

	path, score := eg.findBestFile(queryVec)
	return path, score, nil
}

// RecommendTool finds the tool whose description is most semantically
// similar to the query.
func (eg *EmbedGrounder) RecommendTool(query string) (string, float64, error) {
	if eg.embedFn == nil {
		return "", 0, nil
	}

	queryVec, err := eg.embedFn(query)
	if err != nil {
		return "", 0, err
	}

	eg.mu.RLock()
	defer eg.mu.RUnlock()

	tool, score := eg.findBestTool(queryVec)
	return tool, score, nil
}

// FindSimilar returns past interactions semantically similar to the query.
func (eg *EmbedGrounder) FindSimilar(query string, limit int) ([]PastMatch, error) {
	if eg.embedFn == nil {
		return nil, nil
	}

	queryVec, err := eg.embedFn(query)
	if err != nil {
		return nil, err
	}

	eg.mu.RLock()
	defer eg.mu.RUnlock()

	return eg.findSimilarPast(queryVec, limit), nil
}

// ToolCount returns the number of indexed tools.
func (eg *EmbedGrounder) ToolCount() int {
	eg.mu.RLock()
	defer eg.mu.RUnlock()
	return len(eg.toolIndex)
}

// FileCount returns the number of indexed files.
func (eg *EmbedGrounder) FileCount() int {
	eg.mu.RLock()
	defer eg.mu.RUnlock()
	return len(eg.fileIndex)
}

// HistoryCount returns the number of recorded interactions.
func (eg *EmbedGrounder) HistoryCount() int {
	eg.mu.RLock()
	defer eg.mu.RUnlock()
	return len(eg.histIndex)
}

// --- Internal Methods ---

// findBestTool returns the tool with highest cosine similarity to the query.
// Caller must hold eg.mu.RLock().
func (eg *EmbedGrounder) findBestTool(queryVec []float64) (string, float64) {
	bestTool := ""
	bestScore := 0.0

	for name, toolVec := range eg.toolIndex {
		sim := embedCosineSimilarity(queryVec, toolVec)
		if sim > bestScore {
			bestScore = sim
			bestTool = name
		}
	}

	return bestTool, bestScore
}

// findBestFile returns the file path with highest cosine similarity.
// Caller must hold eg.mu.RLock().
func (eg *EmbedGrounder) findBestFile(queryVec []float64) (string, float64) {
	bestPath := ""
	bestScore := 0.0

	for path, fileVec := range eg.fileIndex {
		sim := embedCosineSimilarity(queryVec, fileVec)
		if sim > bestScore {
			bestScore = sim
			bestPath = path
		}
	}

	return bestPath, bestScore
}

// findSimilarPast returns past interactions sorted by similarity.
// Caller must hold eg.mu.RLock().
func (eg *EmbedGrounder) findSimilarPast(queryVec []float64, limit int) []PastMatch {
	type scored struct {
		entry historyEntry
		sim   float64
	}

	var matches []scored
	for _, h := range eg.histIndex {
		if len(h.Embedding) == 0 {
			continue
		}
		sim := embedCosineSimilarity(queryVec, h.Embedding)
		if sim > 0.3 { // minimum relevance threshold
			matches = append(matches, scored{entry: h, sim: sim})
		}
	}

	// Sort by similarity descending
	for i := 1; i < len(matches); i++ {
		for j := i; j > 0 && matches[j].sim > matches[j-1].sim; j-- {
			matches[j], matches[j-1] = matches[j-1], matches[j]
		}
	}

	if len(matches) > limit {
		matches = matches[:limit]
	}

	results := make([]PastMatch, len(matches))
	for i, m := range matches {
		results[i] = PastMatch{
			Query:      m.entry.Query,
			Tool:       m.entry.Tool,
			Args:       m.entry.Args,
			Similarity: m.sim,
		}
	}
	return results
}

// FormatGroundingContext formats grounding results as context for the LLM.
func (gr *GroundedResult) FormatGroundingContext() string {
	if gr == nil {
		return ""
	}

	var b strings.Builder

	if gr.Tool != "" && gr.ToolScore > 0.5 {
		b.WriteString("[Semantic hint: best tool match is '")
		b.WriteString(gr.Tool)
		b.WriteString("']\n")
	}

	if path, ok := gr.ResolvedArgs["path"]; ok {
		b.WriteString("[Semantic hint: likely file is '")
		b.WriteString(path)
		b.WriteString("']\n")
	}

	if len(gr.SimilarPast) > 0 {
		b.WriteString("[Similar past queries:\n")
		for _, pm := range gr.SimilarPast {
			b.WriteString("  - \"")
			b.WriteString(pm.Query)
			b.WriteString("\" → ")
			b.WriteString(pm.Tool)
			if len(pm.Args) > 0 {
				b.WriteString("(")
				b.WriteString(formatArgs(pm.Args))
				b.WriteString(")")
			}
			b.WriteString("\n")
		}
		b.WriteString("]\n")
	}

	return b.String()
}

// embedCosineSimilarity computes cosine similarity between two vectors.
func embedCosineSimilarity(a, b []float64) float64 {
	return simd.CosineSimilarity(a, b)
}
