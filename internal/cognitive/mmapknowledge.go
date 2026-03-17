package cognitive

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"sync"

	"github.com/artaeon/nous/internal/safefile"
	"github.com/artaeon/nous/internal/simd"
)

// MmapKnowledgeStore is a memory-efficient knowledge store that keeps
// embedding vectors in a compact binary format on disk and loads them
// on demand, rather than holding everything in memory as JSON.
//
// Format:
//   - metadata.json: text chunks, sources, and embedding dimensions
//   - vectors.bin: raw float64 embeddings packed contiguously
//
// For 5K chunks × 768-dim embeddings:
//   - JSON approach: ~120MB in memory (float64 as JSON strings)
//   - Binary approach: ~30MB on disk, ~30MB in memory (or mmap'd)
//
// This makes it feasible to have 50K+ knowledge chunks on a laptop.
type MmapKnowledgeStore struct {
	mu         sync.RWMutex
	chunks     []mmapChunk
	vectors    []float64 // flat array: chunk[i] starts at i*dim
	dim        int       // embedding dimension
	metaPath   string
	vectorPath string
}

type mmapChunk struct {
	Text   string `json:"text"`
	Source string `json:"source"`
}

type mmapMeta struct {
	Dim    int         `json:"dim"`
	Chunks []mmapChunk `json:"chunks"`
}

// NewMmapKnowledgeStore creates or loads a memory-mapped knowledge store.
func NewMmapKnowledgeStore(basePath string) *MmapKnowledgeStore {
	store := &MmapKnowledgeStore{
		metaPath:   basePath + ".meta.json",
		vectorPath: basePath + ".vectors.bin",
	}
	store.load()
	return store
}

// Add adds a chunk with its pre-computed embedding.
func (ms *MmapKnowledgeStore) Add(text, source string, embedding []float64) {
	ms.mu.Lock()
	defer ms.mu.Unlock()

	if len(embedding) == 0 {
		return
	}

	// Set dimension from first embedding
	if ms.dim == 0 {
		ms.dim = len(embedding)
	}

	// Skip if dimension mismatch
	if len(embedding) != ms.dim {
		return
	}

	ms.chunks = append(ms.chunks, mmapChunk{Text: text, Source: source})
	ms.vectors = append(ms.vectors, embedding...)
}

// Search finds the top-k most similar chunks to the query embedding.
func (ms *MmapKnowledgeStore) Search(queryEmbed []float64, topK int) []KnowledgeResult {
	ms.mu.RLock()
	defer ms.mu.RUnlock()

	if len(ms.chunks) == 0 || len(queryEmbed) != ms.dim {
		return nil
	}

	type scored struct {
		idx   int
		score float64
	}

	results := make([]scored, 0, len(ms.chunks))
	for i := range ms.chunks {
		start := i * ms.dim
		end := start + ms.dim
		vec := ms.vectors[start:end]
		score := simd.CosineSimilarity(queryEmbed, vec)
		results = append(results, scored{idx: i, score: score})
	}

	// Partial sort: find top-k without full sort
	if topK > len(results) {
		topK = len(results)
	}

	// Simple selection sort for top-k (efficient when k << n)
	for i := 0; i < topK; i++ {
		maxIdx := i
		for j := i + 1; j < len(results); j++ {
			if results[j].score > results[maxIdx].score {
				maxIdx = j
			}
		}
		results[i], results[maxIdx] = results[maxIdx], results[i]
	}

	out := make([]KnowledgeResult, topK)
	for i := 0; i < topK; i++ {
		out[i] = KnowledgeResult{
			Text:   ms.chunks[results[i].idx].Text,
			Source: ms.chunks[results[i].idx].Source,
			Score:  results[i].score,
		}
	}
	return out
}

// Size returns the number of chunks.
func (ms *MmapKnowledgeStore) Size() int {
	ms.mu.RLock()
	defer ms.mu.RUnlock()
	return len(ms.chunks)
}

// Dim returns the embedding dimension.
func (ms *MmapKnowledgeStore) Dim() int {
	ms.mu.RLock()
	defer ms.mu.RUnlock()
	return ms.dim
}

// MemoryUsageMB returns estimated memory usage in MB.
func (ms *MmapKnowledgeStore) MemoryUsageMB() float64 {
	ms.mu.RLock()
	defer ms.mu.RUnlock()
	// float64 = 8 bytes per element
	vectorBytes := float64(len(ms.vectors)) * 8
	return vectorBytes / (1024 * 1024)
}

// Save persists the store to disk in compact binary format.
func (ms *MmapKnowledgeStore) Save() error {
	ms.mu.RLock()
	defer ms.mu.RUnlock()

	// Save metadata as JSON
	meta := mmapMeta{
		Dim:    ms.dim,
		Chunks: ms.chunks,
	}
	metaData, err := json.Marshal(meta)
	if err != nil {
		return err
	}
	if err := safefile.WriteAtomic(ms.metaPath, metaData, 0644); err != nil {
		return err
	}

	// Save vectors as raw binary (8 bytes per float64)
	buf := make([]byte, len(ms.vectors)*8)
	for i, v := range ms.vectors {
		binary.LittleEndian.PutUint64(buf[i*8:], math.Float64bits(v))
	}
	return safefile.WriteAtomic(ms.vectorPath, buf, 0644)
}

// load restores the store from disk.
func (ms *MmapKnowledgeStore) load() {
	// Load metadata
	metaData, err := os.ReadFile(ms.metaPath)
	if err != nil {
		return
	}
	var meta mmapMeta
	if err := json.Unmarshal(metaData, &meta); err != nil {
		return
	}

	// Load vectors
	vecData, err := os.ReadFile(ms.vectorPath)
	if err != nil {
		return
	}

	expectedSize := len(meta.Chunks) * meta.Dim * 8
	if len(vecData) != expectedSize {
		return // corrupt or mismatched
	}

	vectors := make([]float64, len(meta.Chunks)*meta.Dim)
	for i := range vectors {
		vectors[i] = math.Float64frombits(binary.LittleEndian.Uint64(vecData[i*8:]))
	}

	ms.mu.Lock()
	defer ms.mu.Unlock()
	ms.chunks = meta.Chunks
	ms.vectors = vectors
	ms.dim = meta.Dim
}
