package cognitive

import (
	"hash/fnv"
	"math"
	"strings"
	"unicode"
)

// -----------------------------------------------------------------------
// Neural Feature Extraction — converts text to fixed-size feature vectors
// using the hashing trick (à la fastText / Vowpal Wabbit).
//
// Features extracted:
//   - Character n-grams (3-grams and 4-grams with boundary markers)
//   - Word unigrams (whole words)
//   - Word shape features (capitalization, punctuation, digits)
//
// All features are hashed to a fixed-size vector using FNV-32, avoiding
// the need for an explicit vocabulary. This means:
//   - Zero vocabulary management
//   - Handles unseen words naturally (via subword n-grams)
//   - Fixed memory regardless of vocabulary size
//   - Microsecond-level extraction
// -----------------------------------------------------------------------

const (
	// DefaultFeatureSize is the dimensionality of the feature vector.
	// 2048 gives low collision rates for ~38 intents.
	DefaultFeatureSize = 2048
)

// ExtractFeatures converts text to a fixed-size float32 feature vector
// using character n-grams and word unigrams hashed to a fixed space.
// The returned vector is L2-normalized.
func ExtractFeatures(text string, size int) []float32 {
	features := make([]float32, size)

	lower := strings.ToLower(strings.TrimSpace(text))
	if lower == "" {
		return features
	}

	words := tokenizeForFeatures(lower)

	// 1. Character n-grams (3-grams and 4-grams with boundary markers)
	// Boundary markers help distinguish word-initial/final patterns.
	// "<hel", "hel", "ell", "llo", "lo>" captures position info.
	padded := "<" + lower + ">"
	runes := []rune(padded)

	// Character 3-grams
	for i := 0; i+3 <= len(runes); i++ {
		ngram := string(runes[i : i+3])
		idx := hashFeature(ngram, "c3") % uint32(size)
		features[idx] += 1.0
	}

	// Character 4-grams
	for i := 0; i+4 <= len(runes); i++ {
		ngram := string(runes[i : i+4])
		idx := hashFeature(ngram, "c4") % uint32(size)
		features[idx] += 1.0
	}

	// 2. Word unigrams
	for _, w := range words {
		idx := hashFeature(w, "w1") % uint32(size)
		features[idx] += 1.0
	}

	// 3. Word shape features (structural signals)
	addShapeFeatures(text, words, features, size)

	// L2 normalize
	normalizeL2(features)

	return features
}

// tokenize splits text into words, stripping punctuation.
func tokenizeForFeatures(text string) []string {
	var words []string
	for _, w := range strings.Fields(text) {
		cleaned := strings.TrimFunc(w, func(r rune) bool {
			return unicode.IsPunct(r) || unicode.IsSymbol(r)
		})
		if cleaned != "" {
			words = append(words, cleaned)
		}
	}
	return words
}

// hashFeature computes a FNV-32 hash for a feature string with a namespace prefix.
// The namespace prevents collisions between different feature types.
func hashFeature(feature, namespace string) uint32 {
	h := fnv.New32a()
	h.Write([]byte(namespace))
	h.Write([]byte{0}) // separator
	h.Write([]byte(feature))
	return h.Sum32()
}

// addShapeFeatures adds structural signals to the feature vector.
func addShapeFeatures(original string, words []string, features []float32, size int) {
	// Question mark present
	if strings.HasSuffix(strings.TrimSpace(original), "?") {
		idx := hashFeature("HAS_QUESTION_MARK", "shape") % uint32(size)
		features[idx] += 2.0
	}

	// Exclamation mark
	if strings.Contains(original, "!") {
		idx := hashFeature("HAS_EXCLAMATION", "shape") % uint32(size)
		features[idx] += 1.5
	}

	// Word count buckets
	nw := len(words)
	var bucket string
	switch {
	case nw <= 1:
		bucket = "LEN_1"
	case nw <= 3:
		bucket = "LEN_SHORT"
	case nw <= 7:
		bucket = "LEN_MEDIUM"
	default:
		bucket = "LEN_LONG"
	}
	idx := hashFeature(bucket, "shape") % uint32(size)
	features[idx] += 1.0

	// Contains digits
	for _, r := range original {
		if unicode.IsDigit(r) {
			idx := hashFeature("HAS_DIGIT", "shape") % uint32(size)
			features[idx] += 1.0
			break
		}
	}

	// First word (important for intent: "define", "explain", "translate", etc.)
	if len(words) > 0 {
		idx := hashFeature(words[0], "first") % uint32(size)
		features[idx] += 2.0 // higher weight for first word
	}

	// Contains URL
	if strings.Contains(original, "http://") || strings.Contains(original, "https://") {
		idx := hashFeature("HAS_URL", "shape") % uint32(size)
		features[idx] += 3.0
	}

	// Contains file path
	if strings.Contains(original, "/") || strings.Contains(original, "~") {
		idx := hashFeature("HAS_PATH", "shape") % uint32(size)
		features[idx] += 1.5
	}
}

// normalizeL2 normalizes a float32 vector to unit length.
func normalizeL2(v []float32) {
	var sum float32
	i := 0
	// 4-way unrolled accumulation
	for ; i+3 < len(v); i += 4 {
		sum += v[i]*v[i] + v[i+1]*v[i+1] + v[i+2]*v[i+2] + v[i+3]*v[i+3]
	}
	for ; i < len(v); i++ {
		sum += v[i] * v[i]
	}

	norm := float32(math.Sqrt(float64(sum)))
	if norm == 0 {
		return
	}

	i = 0
	invNorm := 1.0 / norm
	for ; i+3 < len(v); i += 4 {
		v[i] *= invNorm
		v[i+1] *= invNorm
		v[i+2] *= invNorm
		v[i+3] *= invNorm
	}
	for ; i < len(v); i++ {
		v[i] *= invNorm
	}
}
