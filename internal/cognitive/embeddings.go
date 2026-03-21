package cognitive

import (
	"bufio"
	"encoding/binary"
	"encoding/json"
	"hash/fnv"
	"math"
	"math/rand"
	"os"
	"sort"
	"strconv"
	"strings"
	"sync"

	"github.com/artaeon/nous/internal/simd"
)

// -----------------------------------------------------------------------
// Word Embeddings — semantic word vectors for context-aware generation.
//
// Instead of picking random words from pools ("innovative" for Stoicism),
// the embedding engine selects semantically appropriate words by measuring
// vector distance between candidates and the current context.
//
// Three layers, each optional:
//  1. Taxonomy-seeded vectors — built from conceptCategory in semantic.go
//  2. Co-occurrence projection — learned from ingested text (random projection)
//  3. GloVe import — load pre-trained vectors for best quality
//
// When no embeddings are loaded, the generative engine falls back to its
// existing random-pool selection. Every word pool call is wrapped in
// pickSemantic() which degrades gracefully.
// -----------------------------------------------------------------------

// WordEmbeddings maps words to dense real-valued vectors.
type WordEmbeddings struct {
	vectors map[string][]float64
	dim     int
	mu      sync.RWMutex
}

// NewWordEmbeddings creates an embedding space with the given dimensionality.
func NewWordEmbeddings(dim int) *WordEmbeddings {
	return &WordEmbeddings{
		vectors: make(map[string][]float64),
		dim:     dim,
	}
}

// Dim returns the vector dimensionality.
func (we *WordEmbeddings) Dim() int {
	return we.dim
}

// Size returns the number of words with vectors.
func (we *WordEmbeddings) Size() int {
	we.mu.RLock()
	defer we.mu.RUnlock()
	return len(we.vectors)
}

// SetVector sets the embedding vector for a word.
func (we *WordEmbeddings) SetVector(word string, vec []float64) {
	we.mu.Lock()
	defer we.mu.Unlock()
	we.vectors[strings.ToLower(word)] = vec
}

// Vector returns the embedding vector for a word (nil if unknown).
func (we *WordEmbeddings) Vector(word string) []float64 {
	we.mu.RLock()
	defer we.mu.RUnlock()
	return we.vectors[strings.ToLower(word)]
}

// HasVector returns true if the word has an embedding.
func (we *WordEmbeddings) HasVector(word string) bool {
	we.mu.RLock()
	defer we.mu.RUnlock()
	_, ok := we.vectors[strings.ToLower(word)]
	return ok
}

// Similarity returns the cosine similarity between two words.
// Returns 0 if either word is unknown.
func (we *WordEmbeddings) Similarity(a, b string) float64 {
	we.mu.RLock()
	defer we.mu.RUnlock()

	va := we.vectors[strings.ToLower(a)]
	vb := we.vectors[strings.ToLower(b)]
	if va == nil || vb == nil {
		return 0
	}
	return simd.CosineSimilarity(va, vb)
}

// contextVector builds a centroid vector from multiple context words.
// Unknown words are skipped.
func (we *WordEmbeddings) contextVector(words []string) []float64 {
	centroid := make([]float64, we.dim)
	count := 0
	for _, w := range words {
		v := we.vectors[strings.ToLower(w)]
		if v == nil {
			continue
		}
		for i := range centroid {
			centroid[i] += v[i]
		}
		count++
	}
	if count == 0 {
		return nil
	}
	for i := range centroid {
		centroid[i] /= float64(count)
	}
	return centroid
}

// rankedCandidate pairs a word with its similarity score.
type rankedCandidate struct {
	Word  string
	Score float64
}

// KNearestFrom finds the k candidates most similar to the target word.
// Returns up to k words sorted by descending similarity.
func (we *WordEmbeddings) KNearestFrom(target string, candidates []string, k int) []string {
	we.mu.RLock()
	defer we.mu.RUnlock()

	tv := we.vectors[strings.ToLower(target)]
	if tv == nil {
		return nil
	}

	var ranked []rankedCandidate
	for _, c := range candidates {
		cv := we.vectors[strings.ToLower(c)]
		if cv == nil {
			continue
		}
		score := simd.CosineSimilarity(tv, cv)
		ranked = append(ranked, rankedCandidate{Word: c, Score: score})
	}

	sort.Slice(ranked, func(i, j int) bool {
		return ranked[i].Score > ranked[j].Score
	})

	if k > len(ranked) {
		k = len(ranked)
	}
	result := make([]string, k)
	for i := 0; i < k; i++ {
		result[i] = ranked[i].Word
	}
	return result
}

// KNearestFromContext finds the k candidates most similar to a context
// (multiple words averaged into a centroid vector).
func (we *WordEmbeddings) KNearestFromContext(context []string, candidates []string, k int) []string {
	we.mu.RLock()
	defer we.mu.RUnlock()

	cv := we.contextVector(context)
	if cv == nil {
		return nil
	}

	var ranked []rankedCandidate
	for _, c := range candidates {
		wv := we.vectors[strings.ToLower(c)]
		if wv == nil {
			continue
		}
		score := simd.CosineSimilarity(cv, wv)
		ranked = append(ranked, rankedCandidate{Word: c, Score: score})
	}

	sort.Slice(ranked, func(i, j int) bool {
		return ranked[i].Score > ranked[j].Score
	})

	if k > len(ranked) {
		k = len(ranked)
	}
	result := make([]string, k)
	for i := 0; i < k; i++ {
		result[i] = ranked[i].Word
	}
	return result
}

// -----------------------------------------------------------------------
// Layer 1: Taxonomy-seeded vectors
// -----------------------------------------------------------------------

// SeedFromTaxonomy builds initial vectors from a category → words mapping.
// Words sharing categories get similar vectors. This provides a cold-start
// semantic space without any external data.
func (we *WordEmbeddings) SeedFromTaxonomy(categories map[string][]string) {
	we.mu.Lock()
	defer we.mu.Unlock()

	// Collect all unique categories and assign each a random direction.
	catSet := make(map[string]bool)
	for _, cats := range categories {
		for _, c := range cats {
			catSet[c] = true
		}
	}
	catList := make([]string, 0, len(catSet))
	for c := range catSet {
		catList = append(catList, c)
	}
	sort.Strings(catList)

	// Each category gets a random unit vector (seeded for determinism).
	rng := rand.New(rand.NewSource(42))
	catVectors := make(map[string][]float64, len(catList))
	for _, c := range catList {
		v := make([]float64, we.dim)
		for i := range v {
			v[i] = rng.NormFloat64()
		}
		normalize(v)
		catVectors[c] = v
	}

	// Each word's vector is the sum of its category vectors, normalized.
	for word, cats := range categories {
		v := make([]float64, we.dim)
		for _, c := range cats {
			cv := catVectors[c]
			for i := range v {
				v[i] += cv[i]
			}
		}
		normalize(v)
		we.vectors[strings.ToLower(word)] = v
	}
}

// -----------------------------------------------------------------------
// Layer 2: Co-occurrence projection via random indexing
// -----------------------------------------------------------------------

// BuildFromCooccurrence constructs dense vectors from sparse co-occurrence
// counts using random projection (Johnson-Lindenstrauss). This preserves
// cosine similarity structure without needing SVD.
func (we *WordEmbeddings) BuildFromCooccurrence(cooc map[string]map[string]int) {
	we.mu.Lock()
	defer we.mu.Unlock()

	if len(cooc) == 0 {
		return
	}

	// Build vocabulary index.
	vocab := make(map[string]int)
	vocabList := make([]string, 0, len(cooc))
	for w := range cooc {
		if _, ok := vocab[w]; !ok {
			vocab[w] = len(vocabList)
			vocabList = append(vocabList, w)
		}
		for cw := range cooc[w] {
			if _, ok := vocab[cw]; !ok {
				vocab[cw] = len(vocabList)
				vocabList = append(vocabList, cw)
			}
		}
	}

	vocabSize := len(vocabList)
	if vocabSize == 0 {
		return
	}

	// Generate sparse random projection matrix.
	// Each vocab word gets a random vector in the target dim space.
	// We use sparse random projection: +1, 0, -1 with probabilities 1/6, 4/6, 1/6.
	rng := rand.New(rand.NewSource(1337))
	proj := make([][]float64, vocabSize)
	for i := range proj {
		v := make([]float64, we.dim)
		for j := range v {
			r := rng.Float64()
			if r < 1.0/6.0 {
				v[j] = 1.0
			} else if r < 2.0/6.0 {
				v[j] = -1.0
			}
			// else 0 (sparse)
		}
		proj[i] = v
	}

	// Project each word's co-occurrence vector to the dense space.
	for word, context := range cooc {
		dense := make([]float64, we.dim)
		for cw, count := range context {
			if idx, ok := vocab[cw]; ok {
				weight := math.Log1p(float64(count)) // log-scaled
				for d := 0; d < we.dim; d++ {
					dense[d] += weight * proj[idx][d]
				}
			}
		}
		normalize(dense)

		// If we already have a taxonomy-seeded vector, blend them.
		key := strings.ToLower(word)
		if existing, ok := we.vectors[key]; ok {
			for i := range dense {
				dense[i] = 0.3*existing[i] + 0.7*dense[i]
			}
			normalize(dense)
		}
		we.vectors[key] = dense
	}
}

// -----------------------------------------------------------------------
// Layer 3: GloVe/word2vec text format loader
// -----------------------------------------------------------------------

// LoadGloVe loads word vectors from a GloVe-format text file.
// If vocabFilter is non-nil, only loads vectors for words in the filter.
// Format: "word 0.123 -0.456 0.789 ..."
func (we *WordEmbeddings) LoadGloVe(path string, vocabFilter map[string]bool) (int, error) {
	f, err := os.Open(path)
	if err != nil {
		return 0, err
	}
	defer f.Close()

	we.mu.Lock()
	defer we.mu.Unlock()

	loaded := 0
	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 1024*1024), 1024*1024) // 1MB line buffer

	for scanner.Scan() {
		line := scanner.Text()
		fields := strings.Fields(line)
		if len(fields) < 2 {
			continue
		}

		word := strings.ToLower(fields[0])

		// Apply vocab filter if provided.
		if vocabFilter != nil && !vocabFilter[word] {
			continue
		}

		// Parse vector values.
		dim := len(fields) - 1
		if we.dim == 0 {
			we.dim = dim
		}
		if dim != we.dim {
			continue // skip mismatched dimensions
		}

		vec := make([]float64, dim)
		valid := true
		for i := 0; i < dim; i++ {
			v, err := strconv.ParseFloat(fields[i+1], 64)
			if err != nil {
				valid = false
				break
			}
			vec[i] = v
		}
		if !valid {
			continue
		}

		we.vectors[word] = vec
		loaded++
	}

	return loaded, scanner.Err()
}

// -----------------------------------------------------------------------
// Persistence
// -----------------------------------------------------------------------

type embeddingFile struct {
	Dim     int                    `json:"dim"`
	Vectors map[string][]float64   `json:"vectors"`
}

// Save persists the embedding space to a JSON file.
func (we *WordEmbeddings) Save(path string) error {
	we.mu.RLock()
	defer we.mu.RUnlock()

	data, err := json.Marshal(embeddingFile{
		Dim:     we.dim,
		Vectors: we.vectors,
	})
	if err != nil {
		return err
	}

	tmpPath := path + ".tmp"
	if err := os.WriteFile(tmpPath, data, 0644); err != nil {
		return err
	}
	return os.Rename(tmpPath, path)
}

// Load reads the embedding space from a JSON file.
func (we *WordEmbeddings) Load(path string) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return err
	}

	var ef embeddingFile
	if err := json.Unmarshal(data, &ef); err != nil {
		return err
	}

	we.mu.Lock()
	defer we.mu.Unlock()

	we.dim = ef.Dim
	for w, v := range ef.Vectors {
		we.vectors[strings.ToLower(w)] = v
	}
	return nil
}

// -----------------------------------------------------------------------
// Word pool seeding — builds vectors for all words in the generative pools
// -----------------------------------------------------------------------

// SeedPoolWords creates approximate vectors for words in the generative
// engine's word pools by clustering them semantically. Words that appear
// in similar pools get similar vectors.
func (we *WordEmbeddings) SeedPoolWords() {
	we.mu.Lock()
	defer we.mu.Unlock()

	// Define semantic clusters: groups of words with shared meaning.
	// Each cluster gets a base direction, and member words get vectors
	// near that direction with small random perturbations.
	clusters := map[string][]string{
		// Positive evaluation
		"positive": {
			"remarkable", "notable", "significant", "profound", "compelling",
			"impressive", "outstanding", "excellent", "magnificent", "splendid",
			"essential", "fundamental", "important", "crucial", "critical",
			"innovative", "groundbreaking", "revolutionary", "pioneering",
			"valuable", "worthy", "meaningful", "substantive",
		},
		// Scale/magnitude
		"magnitude": {
			"vast", "immense", "enormous", "substantial", "considerable",
			"extensive", "comprehensive", "sweeping", "broad",
			"deep", "profound", "intense", "powerful",
		},
		// Temporal/historical
		"temporal": {
			"ancient", "historic", "enduring", "timeless", "eternal",
			"classical", "traditional", "venerable", "established",
			"modern", "contemporary", "current", "recent", "emerging",
		},
		// Intellectual/philosophical
		"intellectual": {
			"philosophical", "theoretical", "conceptual", "abstract",
			"logical", "rational", "analytical", "systematic",
			"wisdom", "insight", "understanding", "knowledge",
			"thought", "idea", "concept", "principle",
		},
		// Emotional/human
		"emotional": {
			"passionate", "inspiring", "moving", "touching",
			"resilient", "determined", "steadfast", "courageous",
			"gentle", "compassionate", "empathetic", "kind",
		},
		// Physical/concrete
		"physical": {
			"tangible", "concrete", "practical", "measurable",
			"structural", "architectural", "engineered", "built",
			"solid", "robust", "sturdy", "durable",
		},
		// Abstract quality nouns
		"quality_noun": {
			"elegance", "simplicity", "complexity", "beauty",
			"strength", "clarity", "precision", "depth",
			"nature", "character", "identity", "essence",
			"legacy", "heritage", "tradition", "influence",
		},
		// Impact/effect nouns
		"impact_noun": {
			"impact", "influence", "effect", "consequence",
			"contribution", "achievement", "accomplishment",
			"footprint", "mark", "imprint", "stamp",
			"presence", "weight", "significance", "relevance",
		},
		// Structural nouns
		"structure_noun": {
			"story", "path", "direction", "trajectory",
			"foundation", "framework", "structure", "architecture",
			"role", "position", "place", "standing",
			"profile", "reputation", "stature",
		},
		// Creative/artistic
		"creative": {
			"artistic", "creative", "expressive", "imaginative",
			"poetic", "lyrical", "musical", "theatrical",
			"vivid", "colorful", "dramatic", "striking",
		},
		// Scientific/technical
		"scientific": {
			"scientific", "empirical", "experimental", "quantitative",
			"technical", "specialized", "precise", "rigorous",
			"molecular", "atomic", "cellular", "genetic",
		},
		// Natural/organic
		"natural": {
			"natural", "organic", "ecological", "environmental",
			"evolutionary", "biological", "geological",
			"elemental", "primal", "raw", "pure",
		},
	}

	rng := rand.New(rand.NewSource(7919)) // deterministic

	// Generate a base direction for each cluster.
	clusterVecs := make(map[string][]float64)
	for name := range clusters {
		v := make([]float64, we.dim)
		for i := range v {
			v[i] = rng.NormFloat64()
		}
		normalize(v)
		clusterVecs[name] = v
	}

	// Assign vectors to words: cluster base + small perturbation.
	for name, words := range clusters {
		base := clusterVecs[name]
		for _, word := range words {
			key := strings.ToLower(word)
			if _, exists := we.vectors[key]; exists {
				continue // don't overwrite loaded vectors
			}
			v := make([]float64, we.dim)
			for i := range v {
				v[i] = base[i] + rng.NormFloat64()*0.15
			}
			normalize(v)
			we.vectors[key] = v
		}
	}

	// Words appearing in multiple clusters get blended vectors.
	// Track which clusters each word belongs to.
	wordClusters := make(map[string][]string)
	for name, words := range clusters {
		for _, w := range words {
			wordClusters[strings.ToLower(w)] = append(wordClusters[strings.ToLower(w)], name)
		}
	}
	for word, clNames := range wordClusters {
		if len(clNames) <= 1 {
			continue
		}
		v := make([]float64, we.dim)
		for _, cn := range clNames {
			cv := clusterVecs[cn]
			for i := range v {
				v[i] += cv[i]
			}
		}
		for i := range v {
			v[i] += rng.NormFloat64() * 0.1
		}
		normalize(v)
		we.vectors[word] = v
	}
}

// -----------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------

func normalize(v []float64) {
	norm := simd.Norm(v)
	if norm == 0 {
		return
	}
	for i := range v {
		v[i] /= norm
	}
}

// -----------------------------------------------------------------------
// Sentence embedding — average word vectors into a single vector
// -----------------------------------------------------------------------

// SentenceEmbed produces a dense vector for arbitrary text by averaging the
// word vectors of its constituent tokens. When none of the words have known
// vectors, it falls back to a deterministic hash-based vector so that
// identical inputs always yield identical embeddings.
func (we *WordEmbeddings) SentenceEmbed(text string) ([]float64, error) {
	we.mu.RLock()
	defer we.mu.RUnlock()

	words := strings.Fields(strings.ToLower(text))

	centroid := make([]float64, we.dim)
	count := 0
	for _, w := range words {
		// Strip common punctuation so "hello," matches "hello".
		w = strings.Trim(w, ".,;:!?\"'()[]{}–—")
		if w == "" {
			continue
		}
		v, ok := we.vectors[w]
		if !ok {
			continue
		}
		for i := range centroid {
			centroid[i] += v[i]
		}
		count++
	}

	if count > 0 {
		for i := range centroid {
			centroid[i] /= float64(count)
		}
		normalize(centroid)
		return centroid, nil
	}

	// Fallback: deterministic hash-based vector for completely unknown text.
	// Use FNV-64a seeded from the full input to produce reproducible values.
	h := fnv.New64a()
	h.Write([]byte(text))
	seed := h.Sum64()

	for i := 0; i < we.dim; i++ {
		// Derive a per-dimension hash by mixing seed with the dimension index.
		var buf [8]byte
		binary.LittleEndian.PutUint64(buf[:], seed+uint64(i)*2654435761)
		dh := fnv.New64a()
		dh.Write(buf[:])
		bits := dh.Sum64()
		// Map to [-1, 1] range.
		centroid[i] = float64(int64(bits)) / float64(math.MaxInt64)
	}
	normalize(centroid)
	return centroid, nil
}

// MakeEmbedFunc creates a memory.EmbedFunc from a WordEmbeddings instance.
// This replaces the ollama embedding function with a fully local, instant alternative.
func MakeEmbedFunc(we *WordEmbeddings) func(text string) ([]float64, error) {
	return we.SentenceEmbed
}
