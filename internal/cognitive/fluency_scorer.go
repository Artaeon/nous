package cognitive

import (
	"math"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"unicode"
)

// -----------------------------------------------------------------------
// Bigram Fluency Scorer
//
// The corpus NLG's bigram model failed as a GENERATOR — random walks over
// bigram transitions produce incoherent text. But bigrams are excellent as
// a SCORER: given a candidate sentence, we can evaluate how natural it
// sounds by checking how often its word sequences appear in real text.
//
// This flips the role of the bigram model from generation to evaluation:
// instead of sampling P(next|current) to produce text, we compute the
// average log-probability of the observed transitions. Sentences whose
// word pairs appear frequently in the corpus score high; garbled or
// unnatural text scores low.
//
// Uses Laplace smoothing so unseen bigrams get a small non-zero
// probability rather than blowing up to -inf.
// -----------------------------------------------------------------------

// FluencyScorer uses corpus statistics to score how natural a sentence sounds.
// Instead of generating FROM bigrams (which produces incoherent text),
// it scores EXISTING candidates by their bigram probability.
type FluencyScorer struct {
	bigrams      map[string]map[string]int // word -> next -> count
	totals       map[string]int            // word -> total transitions
	vocab        map[string]bool
	corpusLoaded bool
	mu           sync.RWMutex
}

// NewFluencyScorer creates a new FluencyScorer with empty statistics.
func NewFluencyScorer() *FluencyScorer {
	return &FluencyScorer{
		bigrams: make(map[string]map[string]int),
		totals:  make(map[string]int),
		vocab:   make(map[string]bool),
	}
}

// LoadCorpus reads all .txt files from the given directory and builds
// bigram statistics from their contents.
func (fs *FluencyScorer) LoadCorpus(dir string) error {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return err
	}

	var allText strings.Builder
	for _, e := range entries {
		if e.IsDir() || !strings.HasSuffix(e.Name(), ".txt") {
			continue
		}
		data, err := os.ReadFile(filepath.Join(dir, e.Name()))
		if err != nil {
			continue
		}
		allText.WriteString(string(data))
		allText.WriteString(" ")
	}

	corpus := allText.String()
	if len(corpus) == 0 {
		return nil
	}

	fs.mu.Lock()
	defer fs.mu.Unlock()

	words := fluencyTokenize(corpus)
	for i := 0; i < len(words)-1; i++ {
		w := strings.ToLower(words[i])
		next := strings.ToLower(words[i+1])
		fs.vocab[w] = true
		fs.vocab[next] = true

		if fs.bigrams[w] == nil {
			fs.bigrams[w] = make(map[string]int)
		}
		fs.bigrams[w][next]++
		fs.totals[w]++
	}

	fs.corpusLoaded = true
	return nil
}

// CorpusLoaded reports whether a corpus has been loaded.
func (fs *FluencyScorer) CorpusLoaded() bool {
	fs.mu.RLock()
	defer fs.mu.RUnlock()
	return fs.corpusLoaded
}

// VocabSize returns the number of distinct words in the loaded corpus.
func (fs *FluencyScorer) VocabSize() int {
	fs.mu.RLock()
	defer fs.mu.RUnlock()
	return len(fs.vocab)
}

// Score evaluates a sentence by average log-probability of its bigram
// transitions. Higher score = more fluent.
//
// Algorithm:
//  1. Tokenize into words (lowercase)
//  2. For each consecutive pair (w1, w2), compute P(w2|w1) with Laplace smoothing
//  3. Score = average log2(P) across all pairs
//  4. Normalize to 0-1 range (typical range -15 to -5 maps to 0 to 1)
func (fs *FluencyScorer) Score(sentence string) float64 {
	fs.mu.RLock()
	defer fs.mu.RUnlock()

	words := fluencyTokenize(sentence)
	if len(words) < 2 {
		return 0.0
	}

	vocabSize := len(fs.vocab)
	if vocabSize == 0 {
		vocabSize = 1 // avoid division by zero when no corpus is loaded
	}

	var totalLogProb float64
	pairs := 0

	for i := 0; i < len(words)-1; i++ {
		w1 := strings.ToLower(words[i])
		w2 := strings.ToLower(words[i+1])

		count := 0
		if next, ok := fs.bigrams[w1]; ok {
			count = next[w2]
		}
		total := fs.totals[w1]

		// Laplace smoothing: P = (count + 1) / (total + |vocab|)
		prob := float64(count+1) / float64(total+vocabSize)
		totalLogProb += math.Log2(prob)
		pairs++
	}

	if pairs == 0 {
		return 0.0
	}

	avgLogProb := totalLogProb / float64(pairs)

	// Normalize: map typical range [-15, -5] to [0, 1].
	// Values outside this range are clamped.
	const lo = -15.0
	const hi = -5.0
	normalized := (avgLogProb - lo) / (hi - lo)
	return clampFloat(normalized, 0, 1)
}

// ScoreBest returns the index and score of the most fluent candidate.
// Returns (-1, 0) if candidates is empty.
func (fs *FluencyScorer) ScoreBest(candidates []string) (int, float64) {
	if len(candidates) == 0 {
		return -1, 0
	}

	bestIdx := 0
	bestScore := fs.Score(candidates[0])

	for i := 1; i < len(candidates); i++ {
		s := fs.Score(candidates[i])
		if s > bestScore {
			bestScore = s
			bestIdx = i
		}
	}

	return bestIdx, bestScore
}

// SuggestNextWord picks the most probable next word from a list of
// candidates given a context word. Returns empty string if candidates
// is empty or the context word has no bigram data.
func (fs *FluencyScorer) SuggestNextWord(context string, candidates []string) string {
	if len(candidates) == 0 {
		return ""
	}

	fs.mu.RLock()
	defer fs.mu.RUnlock()

	ctx := strings.ToLower(context)
	next, ok := fs.bigrams[ctx]

	bestWord := candidates[0]
	bestCount := -1

	if ok {
		for _, c := range candidates {
			cl := strings.ToLower(c)
			if cnt, found := next[cl]; found && cnt > bestCount {
				bestCount = cnt
				bestWord = c
			}
		}
	}

	return bestWord
}

// fluencyTokenize splits text into words, stripping punctuation edges
// but keeping all words (no stop-word filtering — needed for bigrams).
func fluencyTokenize(text string) []string {
	raw := strings.Fields(text)
	words := make([]string, 0, len(raw))
	for _, w := range raw {
		w = strings.TrimFunc(w, func(r rune) bool {
			return !unicode.IsLetter(r) && !unicode.IsDigit(r) && r != '-' && r != '\''
		})
		if len(w) > 0 {
			words = append(words, w)
		}
	}
	return words
}
