package cognitive

import (
	"math"
	"strings"
	"sync"
)

// -----------------------------------------------------------------------
// Semantic Similarity Engine — makes Nous understand meaning, not just words.
// No ML model. Builds semantic understanding from co-occurrence patterns
// in ingested content + a lightweight concept taxonomy.
// -----------------------------------------------------------------------

// SemanticEngine provides semantic similarity between concepts.
// Three layers:
//  1. Concept taxonomy — hand-crafted category hierarchy (instant)
//  2. Co-occurrence vectors — learned from ingested text (builds up over time)
//  3. Character n-gram similarity — morphological fallback (always works)
type SemanticEngine struct {
	// Co-occurrence matrix: word → {context_word → count}
	cooccurrence map[string]map[string]int
	// Document frequency: word → how many documents it appeared in
	docFreq map[string]int
	docCount int

	mu sync.RWMutex
}

// NewSemanticEngine creates a semantic engine.
func NewSemanticEngine() *SemanticEngine {
	return &SemanticEngine{
		cooccurrence: make(map[string]map[string]int),
		docFreq:      make(map[string]int),
	}
}

// Similarity returns semantic similarity between two terms (0.0–1.0).
// Checks taxonomy first, then co-occurrence, then character n-grams.
func (se *SemanticEngine) Similarity(a, b string) float64 {
	a = strings.ToLower(strings.TrimSpace(a))
	b = strings.ToLower(strings.TrimSpace(b))

	if a == b {
		return 1.0
	}

	// Layer 1: Concept taxonomy (instant, high confidence)
	if score := taxonomySimilarity(a, b); score > 0 {
		return score
	}

	// Layer 2: Co-occurrence similarity (learned)
	if score := se.cooccurrenceSimilarity(a, b); score > 0.1 {
		return score
	}

	// Layer 3: Character n-gram similarity (morphological fallback)
	return ngramSimilarity(a, b)
}

// SimilarTerms returns terms most similar to the query.
func (se *SemanticEngine) SimilarTerms(query string, candidates []string, threshold float64) []string {
	query = strings.ToLower(query)
	var results []string
	for _, c := range candidates {
		if se.Similarity(query, c) >= threshold {
			results = append(results, c)
		}
	}
	return results
}

// IngestText builds co-occurrence vectors from text.
// Call this whenever content is ingested.
func (se *SemanticEngine) IngestText(text string) {
	se.mu.Lock()
	defer se.mu.Unlock()

	words := tokenizeForSemantic(text)
	if len(words) < 2 {
		return
	}

	se.docCount++
	seen := make(map[string]bool)

	// Window-based co-occurrence (window size 5)
	window := 5
	for i, word := range words {
		if !seen[word] {
			seen[word] = true
			se.docFreq[word]++
		}

		if se.cooccurrence[word] == nil {
			se.cooccurrence[word] = make(map[string]int)
		}

		start := i - window
		if start < 0 {
			start = 0
		}
		end := i + window + 1
		if end > len(words) {
			end = len(words)
		}

		for j := start; j < end; j++ {
			if j == i {
				continue
			}
			se.cooccurrence[word][words[j]]++
		}
	}
}

// cooccurrenceSimilarity computes cosine similarity between co-occurrence vectors.
func (se *SemanticEngine) cooccurrenceSimilarity(a, b string) float64 {
	se.mu.RLock()
	defer se.mu.RUnlock()

	vecA := se.cooccurrence[a]
	vecB := se.cooccurrence[b]
	if len(vecA) == 0 || len(vecB) == 0 {
		return 0
	}

	// Cosine similarity
	var dotProduct, normA, normB float64
	for word, countA := range vecA {
		if countB, ok := vecB[word]; ok {
			// TF-IDF weighting
			idfA := float64(countA) * se.idf(word)
			idfB := float64(countB) * se.idf(word)
			dotProduct += idfA * idfB
		}
		normA += float64(countA * countA)
	}
	for _, countB := range vecB {
		normB += float64(countB * countB)
	}

	if normA == 0 || normB == 0 {
		return 0
	}

	return dotProduct / (math.Sqrt(normA) * math.Sqrt(normB))
}

func (se *SemanticEngine) idf(word string) float64 {
	df := se.docFreq[word]
	if df == 0 || se.docCount == 0 {
		return 1.0
	}
	return math.Log(float64(se.docCount)/float64(df)) + 1.0
}

// -----------------------------------------------------------------------
// Layer 1: Concept Taxonomy
// A lightweight, hand-crafted hierarchy of concepts.
// "car" and "automobile" share category "vehicle" → similar.
// -----------------------------------------------------------------------

// conceptCategory maps terms to their semantic category.
var conceptCategory = map[string][]string{
	// Vehicles
	"car": {"vehicle", "transport"}, "automobile": {"vehicle", "transport"},
	"truck": {"vehicle", "transport"}, "van": {"vehicle", "transport"},
	"bus": {"vehicle", "transport"}, "motorcycle": {"vehicle", "transport"},
	"bike": {"vehicle", "transport"}, "bicycle": {"vehicle", "transport"},
	"plane": {"vehicle", "transport", "aviation"}, "airplane": {"vehicle", "transport", "aviation"},
	"aircraft": {"vehicle", "transport", "aviation"},
	"ship": {"vehicle", "transport", "maritime"}, "boat": {"vehicle", "transport", "maritime"},
	"train": {"vehicle", "transport", "rail"},

	// Programming languages
	"go": {"programming_language", "technology"}, "golang": {"programming_language", "technology"},
	"rust": {"programming_language", "technology"}, "python": {"programming_language", "technology"},
	"javascript": {"programming_language", "technology"}, "js": {"programming_language", "technology"},
	"typescript": {"programming_language", "technology"}, "ts": {"programming_language", "technology"},
	"java": {"programming_language", "technology"}, "c++": {"programming_language", "technology"},
	"c": {"programming_language", "technology"}, "ruby": {"programming_language", "technology"},
	"swift": {"programming_language", "technology"}, "kotlin": {"programming_language", "technology"},
	"php": {"programming_language", "technology"},

	// Tech concepts
	"ai": {"technology", "ai_ml"}, "artificial intelligence": {"technology", "ai_ml"},
	"machine learning": {"technology", "ai_ml"}, "ml": {"technology", "ai_ml"},
	"deep learning": {"technology", "ai_ml"}, "neural network": {"technology", "ai_ml"},
	"llm": {"technology", "ai_ml"}, "large language model": {"technology", "ai_ml"},
	"database": {"technology", "data"}, "db": {"technology", "data"},
	"sql": {"technology", "data"}, "nosql": {"technology", "data"},
	"api": {"technology", "web"}, "rest": {"technology", "web"},
	"server": {"technology", "infrastructure"}, "cloud": {"technology", "infrastructure"},
	"container": {"technology", "infrastructure"}, "docker": {"technology", "infrastructure"},
	"kubernetes": {"technology", "infrastructure"}, "k8s": {"technology", "infrastructure"},

	// Emotions
	"happy": {"emotion", "positive"}, "joy": {"emotion", "positive"},
	"excited": {"emotion", "positive"}, "glad": {"emotion", "positive"},
	"sad": {"emotion", "negative"}, "unhappy": {"emotion", "negative"},
	"depressed": {"emotion", "negative"}, "anxious": {"emotion", "negative"},
	"stressed": {"emotion", "negative"}, "angry": {"emotion", "negative"},
	"frustrated": {"emotion", "negative"}, "worried": {"emotion", "negative"},
	"calm": {"emotion", "positive"}, "relaxed": {"emotion", "positive"},
	"peaceful": {"emotion", "positive"},

	// Time
	"morning": {"time", "day_part"}, "afternoon": {"time", "day_part"},
	"evening": {"time", "day_part"}, "night": {"time", "day_part"},
	"today": {"time", "relative"}, "tomorrow": {"time", "relative"},
	"yesterday": {"time", "relative"}, "monday": {"time", "weekday"},
	"tuesday": {"time", "weekday"}, "wednesday": {"time", "weekday"},
	"thursday": {"time", "weekday"}, "friday": {"time", "weekday"},
	"saturday": {"time", "weekend"}, "sunday": {"time", "weekend"},

	// Food
	"coffee": {"food", "beverage"}, "tea": {"food", "beverage"},
	"water": {"food", "beverage"}, "juice": {"food", "beverage"},
	"lunch": {"food", "meal"}, "dinner": {"food", "meal"},
	"breakfast": {"food", "meal"}, "snack": {"food", "meal"},
	"pizza": {"food"}, "burger": {"food"}, "salad": {"food"},
	"groceries": {"food", "shopping"},

	// Geography
	"city": {"geography", "place"}, "country": {"geography", "place"},
	"continent": {"geography", "place"}, "state": {"geography", "place"},
	"town": {"geography", "place"}, "village": {"geography", "place"},
	"europe": {"geography", "continent"}, "asia": {"geography", "continent"},
	"america": {"geography", "continent"}, "africa": {"geography", "continent"},

	// Philosophy
	"philosophy": {"philosophy", "academic"}, "stoicism": {"philosophy"},
	"stoic": {"philosophy"}, "ethics": {"philosophy"},
	"wisdom": {"philosophy"}, "meditation": {"philosophy", "wellness"},
	"mindfulness": {"philosophy", "wellness"},

	// Health/Wellness
	"exercise": {"health", "wellness"}, "gym": {"health", "wellness"},
	"workout": {"health", "wellness"}, "running": {"health", "wellness"},
	"yoga": {"health", "wellness"}, "diet": {"health", "wellness"},
	"sleep": {"health", "wellness"}, "nap": {"health", "wellness"},
}

// taxonomySimilarity checks if two terms share semantic categories.
func taxonomySimilarity(a, b string) float64 {
	catsA, okA := conceptCategory[a]
	catsB, okB := conceptCategory[b]
	if !okA || !okB {
		return 0
	}

	// Jaccard similarity over category sets
	shared := 0
	setB := make(map[string]bool)
	for _, c := range catsB {
		setB[c] = true
	}
	for _, c := range catsA {
		if setB[c] {
			shared++
		}
	}

	total := len(catsA) + len(catsB) - shared
	if total == 0 {
		return 0
	}
	return float64(shared) / float64(total)
}

// -----------------------------------------------------------------------
// Layer 3: Character N-gram Similarity
// Morphological similarity — catches plurals, verb forms, typos.
// "program" ≈ "programming" ≈ "programmer"
// -----------------------------------------------------------------------

// ngramSimilarity computes character trigram Jaccard similarity.
func ngramSimilarity(a, b string) float64 {
	ngramsA := charNgrams(a, 3)
	ngramsB := charNgrams(b, 3)

	if len(ngramsA) == 0 || len(ngramsB) == 0 {
		return 0
	}

	// Jaccard similarity
	intersection := 0
	for ng := range ngramsA {
		if ngramsB[ng] {
			intersection++
		}
	}

	union := len(ngramsA) + len(ngramsB) - intersection
	if union == 0 {
		return 0
	}
	return float64(intersection) / float64(union)
}

// charNgrams generates character n-grams with boundary markers.
func charNgrams(s string, n int) map[string]bool {
	s = "#" + strings.ToLower(s) + "#"
	grams := make(map[string]bool)
	for i := 0; i <= len(s)-n; i++ {
		grams[s[i:i+n]] = true
	}
	return grams
}

// tokenizeForSemantic splits text into lowercase words, filtering noise.
func tokenizeForSemantic(text string) []string {
	text = strings.ToLower(text)
	fields := strings.FieldsFunc(text, func(r rune) bool {
		return !((r >= 'a' && r <= 'z') || (r >= '0' && r <= '9') || r == '-' || r == '+' || r == '\'')
	})

	var words []string
	for _, w := range fields {
		if len(w) > 1 && !isExtractiveStop(w) {
			words = append(words, w)
		}
	}
	return words
}
