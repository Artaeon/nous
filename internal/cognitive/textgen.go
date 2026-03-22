package cognitive

import (
	"encoding/binary"
	"fmt"
	"math"
	"math/rand"
	"os"
	"strings"
	"sync"
)

// -----------------------------------------------------------------------
// Text Generation Model — Word-level conditioned GRU language model.
//
// Architecture: GRU conditioned on (subject, relation, object) triples.
// Uses word-level tokenization for efficient training on small datasets.
//
// Key insight: With ~350 training sentences (avg ~8 words each), word-level
// tokens give sequences of 5-15 steps instead of 30-60 characters. This
// makes full BPTT trivially fast and the model learns meaningful patterns.
//
// Vocab is built from training data + special tokens (<pad>, <eos>, <unk>).
// Subject/object words are injected via the conditioning vector (hashed
// n-grams) rather than needing to appear in the vocabulary.
//
// Parameter budget (default):
//   vocabSize=1024, embedDim=64, hiddenDim=128, condDim=32
//   - Embedding: 1024 × 64 = 65K
//   - GRU: 3 × (64+32+128) × 128 = 86K
//   - Output: 128 × 1024 + 1024 = 132K
//   - RelEmbed: 20 × 32 = 640
//   - Total: ~284K params × 4 bytes ≈ 1.1MB
// -----------------------------------------------------------------------

// TextGenConfig holds model hyperparameters.
type TextGenConfig struct {
	VocabSize int // built from training data
	EmbedDim  int // word embedding dimension
	HiddenDim int // GRU hidden state size
	CondDim   int // conditioning vector dimension
	MaxLen    int // maximum output sequence length (words)
	NumRels   int // number of relation types
}

// DefaultTextGenConfig returns a balanced configuration.
func DefaultTextGenConfig() TextGenConfig {
	return TextGenConfig{
		VocabSize: 1024,
		EmbedDim:  64,
		HiddenDim: 128,
		CondDim:   32,
		MaxLen:    30,
		NumRels:   20,
	}
}

// SmallTextGenConfig returns a lighter configuration for testing.
func SmallTextGenConfig() TextGenConfig {
	return TextGenConfig{
		VocabSize: 512,
		EmbedDim:  32,
		HiddenDim: 64,
		CondDim:   32,
		MaxLen:    20,
		NumRels:   20,
	}
}

const (
	tokPad = 0
	tokEOS = 1
	tokUNK = 2
)

// TextGenModel is a conditioned GRU language model for generating natural
// language sentences from knowledge graph triples.
type TextGenModel struct {
	Config TextGenConfig

	// Word-level vocabulary
	Word2ID map[string]int // word → token ID
	ID2Word []string       // token ID → word

	// Embedding: vocabSize × embedDim
	Embed []float32

	// GRU parameters (input = embedDim + condDim, hidden = hiddenDim)
	Wr, Br []float32 // Reset gate
	Wz, Bz []float32 // Update gate
	Wn, Bn []float32 // New gate

	// Output projection: hiddenDim → vocabSize
	Wout []float32
	Bout []float32

	// Relation embeddings for conditioning
	RelEmbed []float32 // numRels × condDim

	trainStep int
	rng       *rand.Rand
	mu        sync.RWMutex
}

// NewTextGenModel creates a new randomly initialized model.
func NewTextGenModel(cfg TextGenConfig) *TextGenModel {
	rng := rand.New(rand.NewSource(42))
	inputDim := cfg.EmbedDim + cfg.CondDim + cfg.HiddenDim

	m := &TextGenModel{
		Config:   cfg,
		Word2ID:  make(map[string]int),
		ID2Word:  make([]string, 3), // pad, eos, unk
		rng:      rng,
		Embed:    make([]float32, cfg.VocabSize*cfg.EmbedDim),
		RelEmbed: make([]float32, cfg.NumRels*cfg.CondDim),
		Wr:       make([]float32, inputDim*cfg.HiddenDim),
		Br:       make([]float32, cfg.HiddenDim),
		Wz:       make([]float32, inputDim*cfg.HiddenDim),
		Bz:       make([]float32, cfg.HiddenDim),
		Wn:       make([]float32, inputDim*cfg.HiddenDim),
		Bn:       make([]float32, cfg.HiddenDim),
		Wout:     make([]float32, cfg.HiddenDim*cfg.VocabSize),
		Bout:     make([]float32, cfg.VocabSize),
	}

	// Initialize special tokens
	m.ID2Word[tokPad] = "<pad>"
	m.ID2Word[tokEOS] = "<eos>"
	m.ID2Word[tokUNK] = "<unk>"
	m.Word2ID["<pad>"] = tokPad
	m.Word2ID["<eos>"] = tokEOS
	m.Word2ID["<unk>"] = tokUNK

	// Xavier initialization
	scale := float32(math.Sqrt(2.0 / float64(cfg.EmbedDim)))
	for i := range m.Embed {
		m.Embed[i] = float32(rng.NormFloat64()) * scale * 0.1
	}
	for i := range m.RelEmbed {
		m.RelEmbed[i] = float32(rng.NormFloat64()) * scale * 0.1
	}

	gruScale := float32(math.Sqrt(2.0 / float64(inputDim)))
	initWeights(m.Wr, gruScale, rng)
	initWeights(m.Wz, gruScale, rng)
	initWeights(m.Wn, gruScale, rng)

	outScale := float32(math.Sqrt(2.0 / float64(cfg.HiddenDim+cfg.VocabSize)))
	initWeights(m.Wout, outScale, rng)

	return m
}

func initWeights(w []float32, scale float32, rng *rand.Rand) {
	for i := range w {
		w[i] = float32(rng.NormFloat64()) * scale
	}
}

// -----------------------------------------------------------------------
// Tokenization — word-level with frequency-based vocabulary.
// -----------------------------------------------------------------------

// tokenize splits text into lowercase word tokens.
func tokenizeWords(text string) []string {
	text = strings.ToLower(text)
	// Replace punctuation with space-separated tokens
	for _, p := range []string{".", ",", "!", "?", ";", ":", "(", ")"} {
		text = strings.ReplaceAll(text, p, " "+p+" ")
	}
	fields := strings.Fields(text)
	result := make([]string, 0, len(fields))
	for _, f := range fields {
		if f != "" {
			result = append(result, f)
		}
	}
	return result
}

// BuildVocab constructs the vocabulary from training examples.
func (m *TextGenModel) BuildVocab(examples []TextGenExample) {
	freq := make(map[string]int)
	for _, ex := range examples {
		for _, w := range tokenizeWords(ex.Target) {
			freq[w]++
		}
	}

	// Sort by frequency, take top vocabSize-3 words
	type wf struct {
		word string
		freq int
	}
	var sorted []wf
	for w, f := range freq {
		sorted = append(sorted, wf{w, f})
	}
	// Simple insertion sort (small vocab)
	for i := 1; i < len(sorted); i++ {
		j := i
		for j > 0 && sorted[j].freq > sorted[j-1].freq {
			sorted[j], sorted[j-1] = sorted[j-1], sorted[j]
			j--
		}
	}

	maxVocab := m.Config.VocabSize - 3
	if len(sorted) < maxVocab {
		maxVocab = len(sorted)
	}

	m.ID2Word = make([]string, 3+maxVocab)
	m.ID2Word[tokPad] = "<pad>"
	m.ID2Word[tokEOS] = "<eos>"
	m.ID2Word[tokUNK] = "<unk>"
	m.Word2ID = make(map[string]int, 3+maxVocab)
	m.Word2ID["<pad>"] = tokPad
	m.Word2ID["<eos>"] = tokEOS
	m.Word2ID["<unk>"] = tokUNK

	for i := 0; i < maxVocab; i++ {
		id := i + 3
		m.ID2Word[id] = sorted[i].word
		m.Word2ID[sorted[i].word] = id
	}

	// Update config to match actual vocab size
	m.Config.VocabSize = len(m.ID2Word)

	// Reallocate output layer and embeddings to match actual vocab
	inputDim := m.Config.EmbedDim + m.Config.CondDim + m.Config.HiddenDim
	m.Embed = make([]float32, m.Config.VocabSize*m.Config.EmbedDim)
	m.Wout = make([]float32, m.Config.HiddenDim*m.Config.VocabSize)
	m.Bout = make([]float32, m.Config.VocabSize)

	// Re-initialize
	rng := m.rng
	scale := float32(math.Sqrt(2.0 / float64(m.Config.EmbedDim)))
	for i := range m.Embed {
		m.Embed[i] = float32(rng.NormFloat64()) * scale * 0.1
	}
	outScale := float32(math.Sqrt(2.0 / float64(m.Config.HiddenDim+m.Config.VocabSize)))
	initWeights(m.Wout, outScale, rng)
	_ = inputDim
}

// encode converts text to token IDs.
func (m *TextGenModel) encode(text string) []int {
	words := tokenizeWords(text)
	ids := make([]int, len(words)+1) // +1 for EOS
	for i, w := range words {
		id, ok := m.Word2ID[w]
		if !ok {
			id = tokUNK
		}
		ids[i] = id
	}
	ids[len(words)] = tokEOS
	return ids
}

// decode converts token IDs back to text.
func (m *TextGenModel) decode(ids []int) string {
	var parts []string
	for _, id := range ids {
		if id == tokEOS || id == tokPad {
			break
		}
		if id >= 0 && id < len(m.ID2Word) {
			parts = append(parts, m.ID2Word[id])
		}
	}
	text := strings.Join(parts, " ")
	// Clean up punctuation spacing
	for _, p := range []string{" .", " ,", " !", " ?", " ;", " :"} {
		text = strings.ReplaceAll(text, p, p[1:])
	}
	return text
}

// -----------------------------------------------------------------------
// Conditioning — convert a triple into a fixed-size condition vector.
// -----------------------------------------------------------------------

var relIndex = map[RelType]int{
	RelIsA: 0, RelLocatedIn: 1, RelPartOf: 2, RelCreatedBy: 3,
	RelFoundedBy: 4, RelFoundedIn: 5, RelHas: 6, RelOffers: 7,
	RelUsedFor: 8, RelRelatedTo: 9, RelSimilarTo: 10, RelCauses: 11,
	RelContradicts: 12, RelFollows: 13, RelPrefers: 14, RelDislikes: 15,
	RelDomain: 16, RelDescribedAs: 17,
}

func (m *TextGenModel) conditionVector(subj string, rel RelType, obj string) []float32 {
	cond := make([]float32, m.Config.CondDim)

	idx, ok := relIndex[rel]
	if !ok {
		idx = 0
	}
	if idx < m.Config.NumRels {
		start := idx * m.Config.CondDim
		end := start + m.Config.CondDim
		if end <= len(m.RelEmbed) {
			copy(cond, m.RelEmbed[start:end])
		}
	}

	hashAndAdd(cond, subj, 0x5375626A)
	hashAndAdd(cond, obj, 0x4F626A65)

	return cond
}

func hashAndAdd(vec []float32, text string, seed uint32) {
	text = strings.ToLower(text)
	dim := len(vec)
	for i := 0; i < len(text)-2; i++ {
		trigram := text[i : i+3]
		h := fnv32(trigram, seed)
		idx := int(h) % dim
		if idx < 0 {
			idx += dim
		}
		vec[idx] += 0.1
	}
}

func fnv32(s string, seed uint32) uint32 {
	h := seed ^ 0x811c9dc5
	for i := 0; i < len(s); i++ {
		h ^= uint32(s[i])
		h *= 0x01000193
	}
	return h
}

// -----------------------------------------------------------------------
// GRU Forward Pass
// -----------------------------------------------------------------------

func (m *TextGenModel) gruStep(x, cond, h []float32) []float32 {
	cfg := m.Config
	inputDim := cfg.EmbedDim + cfg.CondDim + cfg.HiddenDim

	concat := make([]float32, inputDim)
	copy(concat[0:], x)
	copy(concat[cfg.EmbedDim:], cond)
	copy(concat[cfg.EmbedDim+cfg.CondDim:], h)

	r := matVecMul(m.Wr, concat, cfg.HiddenDim, inputDim)
	vecAdd(r, m.Br)
	sigmoid(r)

	z := matVecMul(m.Wz, concat, cfg.HiddenDim, inputDim)
	vecAdd(z, m.Bz)
	sigmoid(z)

	concatN := make([]float32, inputDim)
	copy(concatN[0:], x)
	copy(concatN[cfg.EmbedDim:], cond)
	for i := 0; i < cfg.HiddenDim; i++ {
		concatN[cfg.EmbedDim+cfg.CondDim+i] = r[i] * h[i]
	}

	n := matVecMul(m.Wn, concatN, cfg.HiddenDim, inputDim)
	vecAdd(n, m.Bn)
	tanhVec(n)

	hNew := make([]float32, cfg.HiddenDim)
	for i := 0; i < cfg.HiddenDim; i++ {
		hNew[i] = (1-z[i])*n[i] + z[i]*h[i]
	}
	return hNew
}

// -----------------------------------------------------------------------
// Generation
// -----------------------------------------------------------------------

// Generate produces a natural language sentence from a knowledge triple.
func (m *TextGenModel) Generate(subj string, rel RelType, obj string, temp float32) string {
	m.mu.RLock()
	defer m.mu.RUnlock()

	cfg := m.Config
	cond := m.conditionVector(subj, rel, obj)
	h := make([]float32, cfg.HiddenDim)

	var generated []int

	// Feed subject words as prompt
	subjWords := tokenizeWords(subj)
	for _, w := range subjWords {
		id, ok := m.Word2ID[w]
		if !ok {
			id = tokUNK
		}
		embed := m.lookupEmbed(id)
		h = m.gruStep(embed, cond, h)
		generated = append(generated, id)
	}

	// Generate autoregressively
	prevToken := generated[len(generated)-1]
	for i := 0; i < cfg.MaxLen-len(subjWords); i++ {
		embed := m.lookupEmbed(prevToken)
		h = m.gruStep(embed, cond, h)

		logits := matVecMul(m.Wout, h, cfg.VocabSize, cfg.HiddenDim)
		vecAdd(logits, m.Bout)

		var nextToken int
		if temp > 0 {
			nextToken = sampleFromLogits(logits, temp, m.rng)
		} else {
			nextToken = argmax(logits)
		}

		if nextToken == tokEOS || nextToken == tokPad {
			break
		}

		generated = append(generated, nextToken)
		prevToken = nextToken
	}

	return m.decode(generated)
}

func (m *TextGenModel) lookupEmbed(tokenID int) []float32 {
	start := tokenID * m.Config.EmbedDim
	end := start + m.Config.EmbedDim
	if end > len(m.Embed) {
		return make([]float32, m.Config.EmbedDim)
	}
	result := make([]float32, m.Config.EmbedDim)
	copy(result, m.Embed[start:end])
	return result
}

// -----------------------------------------------------------------------
// Training — Full BPTT (sequences are short with word-level tokens)
// -----------------------------------------------------------------------

// TextGenExample is a training pair: triple → target sentence.
type TextGenExample struct {
	Subject  string
	Relation RelType
	Object   string
	Target   string
}

// Train trains the model on examples using full BPTT with mini-batching.
func (m *TextGenModel) Train(examples []TextGenExample, epochs int, lr float32) TextGenTrainResult {
	m.mu.Lock()
	defer m.mu.Unlock()

	cfg := m.Config
	inputDim := cfg.EmbedDim + cfg.CondDim + cfg.HiddenDim
	totalLoss := float64(0)
	totalTokens := 0

	const batchSize = 32

	// Pre-encode all examples
	type encodedExample struct {
		tokens []int
		cond   []float32
	}
	encoded := make([]encodedExample, 0, len(examples))
	for _, ex := range examples {
		tokens := m.encode(ex.Target)
		if len(tokens) < 2 {
			continue
		}
		encoded = append(encoded, encodedExample{
			tokens: tokens,
			cond:   m.conditionVector(ex.Subject, ex.Relation, ex.Object),
		})
	}

	// Pre-allocate gradient buffers
	dWr := make([]float32, len(m.Wr))
	dWz := make([]float32, len(m.Wz))
	dWn := make([]float32, len(m.Wn))
	dBr := make([]float32, cfg.HiddenDim)
	dBz := make([]float32, cfg.HiddenDim)
	dBn := make([]float32, cfg.HiddenDim)
	dWout := make([]float32, len(m.Wout))
	dBout := make([]float32, cfg.VocabSize)
	dEmbed := make([]float32, len(m.Embed))

	clearGrads := func() {
		for i := range dWr {
			dWr[i] = 0
		}
		for i := range dWz {
			dWz[i] = 0
		}
		for i := range dWn {
			dWn[i] = 0
		}
		for i := range dBr {
			dBr[i] = 0
		}
		for i := range dBz {
			dBz[i] = 0
		}
		for i := range dBn {
			dBn[i] = 0
		}
		for i := range dWout {
			dWout[i] = 0
		}
		for i := range dBout {
			dBout[i] = 0
		}
		for i := range dEmbed {
			dEmbed[i] = 0
		}
	}

	applyGrads := func(count int) {
		// Warm-up for first 10% of steps, then constant LR
		totalSteps := float32(tgMaxInt(1, epochs*(len(encoded)/batchSize+1)))
		warmupSteps := totalSteps * 0.1
		currentLR := lr
		if float32(m.trainStep) < warmupSteps {
			currentLR = lr * float32(m.trainStep) / warmupSteps
		}
		scale := -currentLR / float32(count)

		gradNorm := gradientNorm(dWr, dWz, dWn, dWout)
		if gradNorm > 5.0 {
			scale *= 5.0 / gradNorm
		}

		applyGrad(m.Wr, dWr, scale)
		applyGrad(m.Wz, dWz, scale)
		applyGrad(m.Wn, dWn, scale)
		applyGrad(m.Br, dBr, scale)
		applyGrad(m.Bz, dBz, scale)
		applyGrad(m.Bn, dBn, scale)
		applyGrad(m.Wout, dWout, scale)
		applyGrad(m.Bout, dBout, scale)
		applyGrad(m.Embed, dEmbed, scale)
		clearGrads()
		m.trainStep++
	}

	for epoch := 0; epoch < epochs; epoch++ {
		epochLoss := float64(0)
		epochTokens := 0
		batchCount := 0

		perm := m.rng.Perm(len(encoded))
		clearGrads()

		for _, pi := range perm {
			ex := encoded[pi]
			tokens := ex.tokens
			seqLen := len(tokens)

			// Forward pass: store all hidden states
			hs := make([][]float32, seqLen)
			hs[0] = make([]float32, cfg.HiddenDim)
			for t := 0; t < seqLen-1; t++ {
				embed := m.lookupEmbed(tokens[t])
				hs[t+1] = m.gruStep(embed, ex.cond, hs[t])
			}

			// Full BPTT backward pass
			dh := make([]float32, cfg.HiddenDim)

			for t := seqLen - 2; t >= 0; t-- {
				targetID := tokens[t+1]

				// Output logits and loss
				logits := matVecMul(m.Wout, hs[t+1], cfg.VocabSize, cfg.HiddenDim)
				vecAdd(logits, m.Bout)
				probs := softmaxF32(logits)
				epochLoss += -float64(math.Log(float64(probs[targetID]) + 1e-10))
				epochTokens++

				// dL/d(logits)
				dLogits := make([]float32, cfg.VocabSize)
				copy(dLogits, probs)
				dLogits[targetID] -= 1.0

				// Gradient for Wout
				outerAddGradOut(dWout, dLogits, hs[t+1], cfg.VocabSize, cfg.HiddenDim)
				vecAdd(dBout, dLogits)

				// dh from output
				dhOut := matVecMulInputGrad(m.Wout, dLogits, cfg.HiddenDim, cfg.VocabSize)
				for i := range dhOut {
					dhOut[i] += dh[i]
				}

				// GRU backward
				embed := m.lookupEmbed(tokens[t])
				dh = m.gruBackward(embed, ex.cond, hs[t], hs[t+1], dhOut,
					dWr, dBr, dWz, dBz, dWn, dBn, dEmbed, tokens[t], inputDim)
			}

			batchCount++
			if batchCount%batchSize == 0 {
				applyGrads(batchSize)
			}
		}

		if batchCount%batchSize != 0 {
			applyGrads(batchCount % batchSize)
		}

		totalLoss += epochLoss
		totalTokens += epochTokens

		if epoch%10 == 0 || epoch == epochs-1 {
			avgLoss := epochLoss / float64(tgMaxInt(1, epochTokens))
			fmt.Printf("  epoch %d/%d  loss=%.4f  tokens=%d\n", epoch+1, epochs, avgLoss, epochTokens)
		}
	}

	return TextGenTrainResult{
		FinalLoss:   totalLoss / float64(tgMaxInt(1, totalTokens)),
		TotalTokens: totalTokens,
		Epochs:      epochs,
	}
}

// TextGenTrainResult holds training metrics.
type TextGenTrainResult struct {
	FinalLoss   float64
	TotalTokens int
	Epochs      int
}

// gruBackward computes gradients for one GRU timestep.
func (m *TextGenModel) gruBackward(
	x, cond, hPrev, hNew, dhNew []float32,
	dWr, dBr, dWz, dBz, dWn, dBn, dEmbed []float32,
	tokenID, inputDim int,
) []float32 {
	cfg := m.Config

	// Recompute forward gates
	concat := make([]float32, inputDim)
	copy(concat[0:], x)
	copy(concat[cfg.EmbedDim:], cond)
	copy(concat[cfg.EmbedDim+cfg.CondDim:], hPrev)

	r := matVecMul(m.Wr, concat, cfg.HiddenDim, inputDim)
	vecAdd(r, m.Br)
	sigmoid(r)

	z := matVecMul(m.Wz, concat, cfg.HiddenDim, inputDim)
	vecAdd(z, m.Bz)
	sigmoid(z)

	concatN := make([]float32, inputDim)
	copy(concatN[0:], x)
	copy(concatN[cfg.EmbedDim:], cond)
	for i := 0; i < cfg.HiddenDim; i++ {
		concatN[cfg.EmbedDim+cfg.CondDim+i] = r[i] * hPrev[i]
	}

	n := matVecMul(m.Wn, concatN, cfg.HiddenDim, inputDim)
	vecAdd(n, m.Bn)
	tanhVec(n)

	// Gradients through h' = (1-z)⊙n + z⊙h_prev
	dz := make([]float32, cfg.HiddenDim)
	dn := make([]float32, cfg.HiddenDim)
	dhPrev := make([]float32, cfg.HiddenDim)
	for i := range dz {
		dz[i] = dhNew[i] * (hPrev[i] - n[i])
		dn[i] = dhNew[i] * (1 - z[i])
		dhPrev[i] = dhNew[i] * z[i]
	}

	// Through activations
	dnPre := make([]float32, cfg.HiddenDim)
	dzPre := make([]float32, cfg.HiddenDim)
	for i := range dnPre {
		dnPre[i] = dn[i] * (1 - n[i]*n[i])
		dzPre[i] = dz[i] * z[i] * (1 - z[i])
	}

	// Weight gradients
	outerAddGrad(dWn, dnPre, concatN, cfg.HiddenDim, inputDim)
	vecAdd(dBn, dnPre)
	outerAddGrad(dWz, dzPre, concat, cfg.HiddenDim, inputDim)
	vecAdd(dBz, dzPre)

	// Gradient through r⊙h in n gate — vectorized
	dConcatN := matVecMulInputGrad(m.Wn, dnPre, inputDim, cfg.HiddenDim)
	dr := make([]float32, cfg.HiddenDim)
	hiddenOffset := cfg.EmbedDim + cfg.CondDim
	for i := 0; i < cfg.HiddenDim; i++ {
		dr[i] = dConcatN[hiddenOffset+i] * hPrev[i]
		dhPrev[i] += dConcatN[hiddenOffset+i] * r[i]
	}

	drPre := make([]float32, cfg.HiddenDim)
	for i := range drPre {
		drPre[i] = dr[i] * r[i] * (1 - r[i])
	}

	outerAddGrad(dWr, drPre, concat, cfg.HiddenDim, inputDim)
	vecAdd(dBr, drPre)

	// Propagate through concat to hPrev
	dConcatZ := matVecMulInputGrad(m.Wz, dzPre, inputDim, cfg.HiddenDim)
	dConcatR := matVecMulInputGrad(m.Wr, drPre, inputDim, cfg.HiddenDim)
	for i := 0; i < cfg.HiddenDim; i++ {
		dhPrev[i] += dConcatZ[hiddenOffset+i] + dConcatR[hiddenOffset+i]
	}

	// Embedding gradient
	if tokenID >= 0 && tokenID*cfg.EmbedDim+cfg.EmbedDim <= len(dEmbed) {
		for i := 0; i < cfg.EmbedDim; i++ {
			dEmbed[tokenID*cfg.EmbedDim+i] += dConcatR[i] + dConcatZ[i] + dConcatN[i]
		}
	}

	return dhPrev
}

// -----------------------------------------------------------------------
// Training Data Generation
// -----------------------------------------------------------------------

// GenerateTextGenTrainingData creates training examples from a cognitive graph.
func GenerateTextGenTrainingData(graph *CognitiveGraph) []TextGenExample {
	if graph == nil {
		return nil
	}
	graph.mu.RLock()
	defer graph.mu.RUnlock()

	var examples []TextGenExample

	for _, edge := range graph.edges {
		if edge.Inferred {
			continue
		}
		from := graph.nodes[edge.From]
		to := graph.nodes[edge.To]
		if from == nil || to == nil {
			continue
		}

		subj := from.Label
		obj := to.Label
		rel := edge.Relation

		if rel == RelDescribedAs {
			continue
		}

		sentence := edgeToNaturalLanguage(subj, rel, obj)
		if sentence == "" || len(sentence) < 10 {
			continue
		}

		examples = append(examples, TextGenExample{
			Subject:  subj,
			Relation: rel,
			Object:   obj,
			Target:   sentence,
		})
	}

	return examples
}

// -----------------------------------------------------------------------
// Model Persistence — binary format with vocabulary.
// -----------------------------------------------------------------------

const textgenMagic = 0x4E475452
const textgenVersion = 2 // v2 = word-level

func (m *TextGenModel) Save(path string) error {
	m.mu.RLock()
	defer m.mu.RUnlock()

	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	// Header
	binary.Write(f, binary.LittleEndian, uint32(textgenMagic))
	binary.Write(f, binary.LittleEndian, uint32(textgenVersion))
	binary.Write(f, binary.LittleEndian, uint32(m.Config.VocabSize))
	binary.Write(f, binary.LittleEndian, uint32(m.Config.EmbedDim))
	binary.Write(f, binary.LittleEndian, uint32(m.Config.HiddenDim))
	binary.Write(f, binary.LittleEndian, uint32(m.Config.CondDim))
	binary.Write(f, binary.LittleEndian, uint32(m.Config.MaxLen))
	binary.Write(f, binary.LittleEndian, uint32(m.Config.NumRels))

	// Vocabulary
	binary.Write(f, binary.LittleEndian, uint32(len(m.ID2Word)))
	for _, w := range m.ID2Word {
		wb := []byte(w)
		binary.Write(f, binary.LittleEndian, uint16(len(wb)))
		f.Write(wb)
	}

	// Weights
	binary.Write(f, binary.LittleEndian, m.Embed)
	binary.Write(f, binary.LittleEndian, m.RelEmbed)
	binary.Write(f, binary.LittleEndian, m.Wr)
	binary.Write(f, binary.LittleEndian, m.Br)
	binary.Write(f, binary.LittleEndian, m.Wz)
	binary.Write(f, binary.LittleEndian, m.Bz)
	binary.Write(f, binary.LittleEndian, m.Wn)
	binary.Write(f, binary.LittleEndian, m.Bn)
	binary.Write(f, binary.LittleEndian, m.Wout)
	binary.Write(f, binary.LittleEndian, m.Bout)

	return nil
}

func (m *TextGenModel) Load(path string) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()

	var magic, version uint32
	binary.Read(f, binary.LittleEndian, &magic)
	binary.Read(f, binary.LittleEndian, &version)
	if magic != textgenMagic {
		return fmt.Errorf("invalid textgen model file (bad magic)")
	}
	if version != textgenVersion {
		return fmt.Errorf("unsupported textgen model version %d (want %d)", version, textgenVersion)
	}

	var vocabSize, embedDim, hiddenDim, condDim, maxLen, numRels uint32
	binary.Read(f, binary.LittleEndian, &vocabSize)
	binary.Read(f, binary.LittleEndian, &embedDim)
	binary.Read(f, binary.LittleEndian, &hiddenDim)
	binary.Read(f, binary.LittleEndian, &condDim)
	binary.Read(f, binary.LittleEndian, &maxLen)
	binary.Read(f, binary.LittleEndian, &numRels)

	m.Config = TextGenConfig{
		VocabSize: int(vocabSize),
		EmbedDim:  int(embedDim),
		HiddenDim: int(hiddenDim),
		CondDim:   int(condDim),
		MaxLen:    int(maxLen),
		NumRels:   int(numRels),
	}

	// Vocabulary
	var vocabLen uint32
	binary.Read(f, binary.LittleEndian, &vocabLen)
	m.ID2Word = make([]string, vocabLen)
	m.Word2ID = make(map[string]int, vocabLen)
	for i := uint32(0); i < vocabLen; i++ {
		var wLen uint16
		binary.Read(f, binary.LittleEndian, &wLen)
		wb := make([]byte, wLen)
		f.Read(wb)
		m.ID2Word[i] = string(wb)
		m.Word2ID[string(wb)] = int(i)
	}

	inputDim := int(embedDim) + int(condDim) + int(hiddenDim)

	m.Embed = make([]float32, vocabSize*embedDim)
	m.RelEmbed = make([]float32, numRels*condDim)
	m.Wr = make([]float32, inputDim*int(hiddenDim))
	m.Br = make([]float32, hiddenDim)
	m.Wz = make([]float32, inputDim*int(hiddenDim))
	m.Bz = make([]float32, hiddenDim)
	m.Wn = make([]float32, inputDim*int(hiddenDim))
	m.Bn = make([]float32, hiddenDim)
	m.Wout = make([]float32, hiddenDim*vocabSize)
	m.Bout = make([]float32, vocabSize)

	binary.Read(f, binary.LittleEndian, m.Embed)
	binary.Read(f, binary.LittleEndian, m.RelEmbed)
	binary.Read(f, binary.LittleEndian, m.Wr)
	binary.Read(f, binary.LittleEndian, m.Br)
	binary.Read(f, binary.LittleEndian, m.Wz)
	binary.Read(f, binary.LittleEndian, m.Bz)
	binary.Read(f, binary.LittleEndian, m.Wn)
	binary.Read(f, binary.LittleEndian, m.Bn)
	binary.Read(f, binary.LittleEndian, m.Wout)
	binary.Read(f, binary.LittleEndian, m.Bout)

	m.rng = rand.New(rand.NewSource(42))
	return nil
}

// -----------------------------------------------------------------------
// Math utilities
// -----------------------------------------------------------------------

func matVecMul(w, v []float32, rows, cols int) []float32 {
	out := make([]float32, rows)
	for i := 0; i < rows; i++ {
		sum := float32(0)
		base := i * cols
		j := 0
		for ; j+3 < cols; j += 4 {
			sum += w[base+j]*v[j] + w[base+j+1]*v[j+1] +
				w[base+j+2]*v[j+2] + w[base+j+3]*v[j+3]
		}
		for ; j < cols; j++ {
			sum += w[base+j] * v[j]
		}
		out[i] = sum
	}
	return out
}

func matVecMulInputGrad(w, dOut []float32, inputDim, hiddenDim int) []float32 {
	dInput := make([]float32, inputDim)
	for h := 0; h < hiddenDim; h++ {
		if dOut[h] == 0 {
			continue
		}
		base := h * inputDim
		d := dOut[h]
		for i := 0; i < inputDim; i++ {
			dInput[i] += w[base+i] * d
		}
	}
	return dInput
}

func vecAdd(a, b []float32) {
	for i := range a {
		if i < len(b) {
			a[i] += b[i]
		}
	}
}

func sigmoid(v []float32) {
	for i, x := range v {
		if x > 15 {
			v[i] = 1.0
		} else if x < -15 {
			v[i] = 0.0
		} else {
			v[i] = 1.0 / (1.0 + float32(math.Exp(-float64(x))))
		}
	}
}

func tanhVec(v []float32) {
	for i, x := range v {
		v[i] = float32(math.Tanh(float64(x)))
	}
}

func softmaxF32(logits []float32) []float32 {
	maxVal := logits[0]
	for _, v := range logits[1:] {
		if v > maxVal {
			maxVal = v
		}
	}
	probs := make([]float32, len(logits))
	sum := float32(0)
	for i, v := range logits {
		probs[i] = float32(math.Exp(float64(v - maxVal)))
		sum += probs[i]
	}
	for i := range probs {
		probs[i] /= sum
	}
	return probs
}

func sampleFromLogits(logits []float32, temp float32, rng *rand.Rand) int {
	scaled := make([]float32, len(logits))
	for i, v := range logits {
		scaled[i] = v / temp
	}
	probs := softmaxF32(scaled)

	r := rng.Float32()
	cumsum := float32(0)
	for i, p := range probs {
		cumsum += p
		if r < cumsum {
			return i
		}
	}
	return len(probs) - 1
}

func argmax(v []float32) int {
	best := 0
	for i := 1; i < len(v); i++ {
		if v[i] > v[best] {
			best = i
		}
	}
	return best
}

// outerAddGradOut accumulates dWout[v*hiddenDim+h] += dLogits[v] * hidden[h]
// matching matVecMul(Wout, h, vocabSize, hiddenDim) where Wout is [vocabSize × hiddenDim].
func outerAddGradOut(dW, dLogits, h []float32, vocabSize, hiddenDim int) {
	for v := 0; v < vocabSize; v++ {
		if dLogits[v] == 0 {
			continue
		}
		base := v * hiddenDim
		dl := dLogits[v]
		for j := 0; j < hiddenDim; j++ {
			dW[base+j] += dl * h[j]
		}
	}
}

func outerAddGrad(dW, dOut, input []float32, hiddenDim, inputDim int) {
	for h := 0; h < hiddenDim; h++ {
		if dOut[h] == 0 {
			continue
		}
		base := h * inputDim
		for i := 0; i < inputDim; i++ {
			dW[base+i] += dOut[h] * input[i]
		}
	}
}

func applyGrad(w, grad []float32, scale float32) {
	for i := range w {
		w[i] += scale * grad[i]
	}
}

func gradientNorm(grads ...[]float32) float32 {
	sum := float64(0)
	for _, g := range grads {
		for _, v := range g {
			sum += float64(v) * float64(v)
		}
	}
	return float32(math.Sqrt(sum))
}

func tgMaxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}
