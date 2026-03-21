package cognitive

import (
	"encoding/json"
	"math/rand"
	"os"
	"strings"
	"sync"
	"unicode"
)

// -----------------------------------------------------------------------
// Markov Chain — learns natural word transitions from text.
//
// Instead of assembling sentences from small hardcoded pools, the Markov
// model learns how words actually flow together by observing real text.
// It builds transition probabilities from trigrams (three-word sequences)
// with bigram fallback for sparse contexts.
//
// Training sources:
//  1. Ingested web content (via SemanticEngine.IngestText)
//  2. The engine's own generated text (self-improvement loop)
//  3. Optional corpus file (~/.nous/corpus.txt)
//
// The model generates sentence fragments (5-15 words), not full articles.
// The generative engine still controls structure; Markov provides natural
// word flow within that structure.
// -----------------------------------------------------------------------

// MarkovModel is a trigram language model with bigram fallback.
type MarkovModel struct {
	// trigrams: (w1, w2) → possible next words with counts
	trigrams map[[2]string][]markovNext
	// bigrams: w1 → possible next words with counts
	bigrams map[string][]markovNext
	// sentence starters
	starters []markovStarter
	// total training tokens
	totalTokens int

	mu sync.RWMutex
}

type markovNext struct {
	Word  string `json:"w"`
	Count int    `json:"c"`
}

type markovStarter struct {
	Words [2]string `json:"ws"`
	Count int       `json:"c"`
}

// NewMarkovModel creates an empty Markov chain.
func NewMarkovModel() *MarkovModel {
	return &MarkovModel{
		trigrams: make(map[[2]string][]markovNext),
		bigrams:  make(map[string][]markovNext),
	}
}

// Size returns the number of unique trigram contexts.
func (m *MarkovModel) Size() int {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return len(m.trigrams)
}

// TotalTokens returns how many tokens have been trained on.
func (m *MarkovModel) TotalTokens() int {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.totalTokens
}

// Train processes raw text into trigram/bigram statistics.
func (m *MarkovModel) Train(text string) {
	sentences := splitIntoSentences(text)
	for _, s := range sentences {
		m.trainSentence(s)
	}
}

// TrainSentences trains on pre-split sentences.
func (m *MarkovModel) TrainSentences(sentences []string) {
	for _, s := range sentences {
		m.trainSentence(s)
	}
}

func (m *MarkovModel) trainSentence(sentence string) {
	words := tokenizeForMarkov(sentence)
	if len(words) < 3 {
		return
	}

	m.mu.Lock()
	defer m.mu.Unlock()

	m.totalTokens += len(words)

	// Record sentence starter
	starter := [2]string{words[0], words[1]}
	found := false
	for i := range m.starters {
		if m.starters[i].Words == starter {
			m.starters[i].Count++
			found = true
			break
		}
	}
	if !found {
		m.starters = append(m.starters, markovStarter{Words: starter, Count: 1})
	}

	// Build bigrams
	for i := 0; i < len(words)-1; i++ {
		m.addBigram(words[i], words[i+1])
	}

	// Build trigrams
	for i := 0; i < len(words)-2; i++ {
		key := [2]string{words[i], words[i+1]}
		m.addTrigram(key, words[i+2])
	}
}

func (m *MarkovModel) addBigram(w1, w2 string) {
	nexts := m.bigrams[w1]
	for i := range nexts {
		if nexts[i].Word == w2 {
			nexts[i].Count++
			m.bigrams[w1] = nexts
			return
		}
	}
	m.bigrams[w1] = append(nexts, markovNext{Word: w2, Count: 1})
}

func (m *MarkovModel) addTrigram(key [2]string, w3 string) {
	nexts := m.trigrams[key]
	for i := range nexts {
		if nexts[i].Word == w3 {
			nexts[i].Count++
			m.trigrams[key] = nexts
			return
		}
	}
	m.trigrams[key] = append(nexts, markovNext{Word: w3, Count: 1})
}

// Generate produces text of up to maxWords, starting from a random starter.
func (m *MarkovModel) Generate(maxWords int, rng *rand.Rand) string {
	m.mu.RLock()
	defer m.mu.RUnlock()

	if len(m.starters) == 0 {
		return ""
	}

	// Pick a weighted random starter.
	starter := m.pickStarter(rng)
	return m.generate(starter, maxWords, rng)
}

// GenerateFrom produces text seeded with a specific word.
// Finds a starter containing the seed, or falls back to random.
func (m *MarkovModel) GenerateFrom(seed string, maxWords int, rng *rand.Rand) string {
	m.mu.RLock()
	defer m.mu.RUnlock()

	if len(m.starters) == 0 {
		return ""
	}

	seedLower := strings.ToLower(seed)

	// Try to find a starter containing the seed word.
	var candidates []markovStarter
	for _, s := range m.starters {
		if strings.ToLower(s.Words[0]) == seedLower ||
			strings.ToLower(s.Words[1]) == seedLower {
			candidates = append(candidates, s)
		}
	}

	var starter [2]string
	if len(candidates) > 0 {
		// Pick weighted random from matching starters.
		total := 0
		for _, c := range candidates {
			total += c.Count
		}
		r := rng.Intn(total)
		cumulative := 0
		for _, c := range candidates {
			cumulative += c.Count
			if r < cumulative {
				starter = c.Words
				break
			}
		}
	} else {
		// No matching starter — try to generate from bigrams.
		if nexts, ok := m.bigrams[seedLower]; ok && len(nexts) > 0 {
			next := m.pickWeighted(nexts, rng)
			starter = [2]string{seedLower, next}
		} else {
			starter = m.pickStarter(rng)
		}
	}

	return m.generate(starter, maxWords, rng)
}

// GenerateFragment produces a sentence fragment of minWords to maxWords.
func (m *MarkovModel) GenerateFragment(seed string, minWords, maxWords int, rng *rand.Rand) string {
	text := m.GenerateFrom(seed, maxWords, rng)
	words := strings.Fields(text)
	if len(words) < minWords {
		return "" // too short, discard
	}
	if len(words) > maxWords {
		words = words[:maxWords]
	}
	return strings.Join(words, " ")
}

func (m *MarkovModel) generate(starter [2]string, maxWords int, rng *rand.Rand) string {
	words := []string{starter[0], starter[1]}

	for len(words) < maxWords {
		key := [2]string{words[len(words)-2], words[len(words)-1]}

		// Try trigram first.
		if nexts, ok := m.trigrams[key]; ok && len(nexts) > 0 {
			next := m.pickWeighted(nexts, rng)
			words = append(words, next)

			// Stop at sentence-ending punctuation.
			if endsWithPunctuation(next) {
				break
			}
			continue
		}

		// Fall back to bigram.
		lastWord := words[len(words)-1]
		if nexts, ok := m.bigrams[lastWord]; ok && len(nexts) > 0 {
			next := m.pickWeighted(nexts, rng)
			words = append(words, next)
			if endsWithPunctuation(next) {
				break
			}
			continue
		}

		break // dead end
	}

	return strings.Join(words, " ")
}

func (m *MarkovModel) pickStarter(rng *rand.Rand) [2]string {
	total := 0
	for _, s := range m.starters {
		total += s.Count
	}
	if total == 0 {
		return m.starters[0].Words
	}
	r := rng.Intn(total)
	cumulative := 0
	for _, s := range m.starters {
		cumulative += s.Count
		if r < cumulative {
			return s.Words
		}
	}
	return m.starters[0].Words
}

func (m *MarkovModel) pickWeighted(nexts []markovNext, rng *rand.Rand) string {
	total := 0
	for _, n := range nexts {
		total += n.Count
	}
	if total == 0 {
		return nexts[0].Word
	}
	r := rng.Intn(total)
	cumulative := 0
	for _, n := range nexts {
		cumulative += n.Count
		if r < cumulative {
			return n.Word
		}
	}
	return nexts[0].Word
}

// -----------------------------------------------------------------------
// Persistence
// -----------------------------------------------------------------------

type markovFile struct {
	Trigrams    map[string][]markovNext `json:"trigrams"`
	Bigrams    map[string][]markovNext `json:"bigrams"`
	Starters   []markovStarter         `json:"starters"`
	TotalTokens int                    `json:"total_tokens"`
}

// Save persists the Markov model to a JSON file.
func (m *MarkovModel) Save(path string) error {
	m.mu.RLock()
	defer m.mu.RUnlock()

	// Convert [2]string keys to "w1|w2" for JSON.
	trigramMap := make(map[string][]markovNext, len(m.trigrams))
	for key, nexts := range m.trigrams {
		trigramMap[key[0]+"|"+key[1]] = nexts
	}

	data, err := json.Marshal(markovFile{
		Trigrams:    trigramMap,
		Bigrams:     m.bigrams,
		Starters:    m.starters,
		TotalTokens: m.totalTokens,
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

// Load reads the Markov model from a JSON file.
func (m *MarkovModel) Load(path string) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return err
	}

	var mf markovFile
	if err := json.Unmarshal(data, &mf); err != nil {
		return err
	}

	m.mu.Lock()
	defer m.mu.Unlock()

	// Convert "w1|w2" keys back to [2]string.
	m.trigrams = make(map[[2]string][]markovNext, len(mf.Trigrams))
	for key, nexts := range mf.Trigrams {
		parts := strings.SplitN(key, "|", 2)
		if len(parts) == 2 {
			m.trigrams[[2]string{parts[0], parts[1]}] = nexts
		}
	}
	m.bigrams = mf.Bigrams
	if m.bigrams == nil {
		m.bigrams = make(map[string][]markovNext)
	}
	m.starters = mf.Starters
	m.totalTokens = mf.TotalTokens
	return nil
}

// -----------------------------------------------------------------------
// Tokenization
// -----------------------------------------------------------------------

// tokenizeForMarkov splits text into lowercase tokens, preserving punctuation
// as separate tokens where useful.
func tokenizeForMarkov(text string) []string {
	// Lowercase and split on whitespace.
	text = strings.ToLower(text)
	rawWords := strings.Fields(text)
	var tokens []string

	for _, w := range rawWords {
		// Strip leading punctuation
		w = strings.TrimLeftFunc(w, func(r rune) bool {
			return r == '"' || r == '\'' || r == '(' || r == '['
		})
		if w == "" {
			continue
		}
		tokens = append(tokens, w)
	}

	return tokens
}

// splitIntoSentences splits text into individual sentences.
func splitIntoSentences(text string) []string {
	var sentences []string
	var current strings.Builder

	for _, r := range text {
		current.WriteRune(r)
		if r == '.' || r == '!' || r == '?' {
			s := strings.TrimSpace(current.String())
			if len(strings.Fields(s)) >= 3 { // only sentences with 3+ words
				sentences = append(sentences, s)
			}
			current.Reset()
		}
	}

	// Remaining text without sentence-ending punctuation.
	if s := strings.TrimSpace(current.String()); len(strings.Fields(s)) >= 3 {
		sentences = append(sentences, s)
	}

	return sentences
}

func endsWithPunctuation(word string) bool {
	if word == "" {
		return false
	}
	last := rune(word[len(word)-1])
	return last == '.' || last == '!' || last == '?' ||
		unicode.IsPunct(last)
}
