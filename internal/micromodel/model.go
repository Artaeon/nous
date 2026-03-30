package micromodel

import (
	"encoding/binary"
	"fmt"
	"io"
	"math/rand"
	"os"
	"time"
)

// Config defines the micro model architecture.
type Config struct {
	VocabSize int // built from training data
	EmbedDim  int // 256
	NumLayers int // 4
	NumHeads  int // 4
	MaxSeqLen int // 64
}

// DefaultConfig returns the standard 8M-parameter configuration.
func DefaultConfig() Config {
	return Config{
		VocabSize: 8000,
		EmbedDim:  256,
		NumLayers: 4,
		NumHeads:  4,
		MaxSeqLen: 64,
	}
}

// SmallConfig returns a smaller config for fast testing.
func SmallConfig() Config {
	return Config{
		VocabSize: 512,
		EmbedDim:  64,
		NumLayers: 2,
		NumHeads:  2,
		MaxSeqLen: 32,
	}
}

// EncoderLayer holds weights for one encoder transformer layer.
type EncoderLayer struct {
	SelfAttnQ []float32 // dim x dim
	SelfAttnK []float32
	SelfAttnV []float32
	SelfAttnO []float32
	FFN1W     []float32 // dim x (dim*4)
	FFN1B     []float32 // dim*4
	FFN2W     []float32 // (dim*4) x dim
	FFN2B     []float32 // dim
	Norm1G    []float32 // dim (gamma)
	Norm1B    []float32 // dim (beta)
	Norm2G    []float32
	Norm2B    []float32
}

// DecoderLayer holds weights for one decoder transformer layer.
type DecoderLayer struct {
	SelfAttnQ  []float32
	SelfAttnK  []float32
	SelfAttnV  []float32
	SelfAttnO  []float32
	CrossAttnQ []float32
	CrossAttnK []float32
	CrossAttnV []float32
	CrossAttnO []float32
	FFN1W      []float32
	FFN1B      []float32
	FFN2W      []float32
	FFN2B      []float32
	Norm1G     []float32
	Norm1B     []float32
	Norm2G     []float32
	Norm2B     []float32
	Norm3G     []float32
	Norm3B     []float32
}

// MicroModel is a small encoder-decoder transformer for knowledge-grounded
// sentence generation. Pure Go, zero dependencies.
type MicroModel struct {
	Config Config

	// Embeddings
	TokenEmbed []float32 // VocabSize x EmbedDim
	PosEmbed   []float32 // MaxSeqLen x EmbedDim

	// Encoder and decoder layers
	EncLayers []EncoderLayer
	DecLayers []DecoderLayer

	// Final output projection: EmbedDim → VocabSize
	OutputW []float32
	OutputB []float32

	// Final layer norms
	EncFinalNormG []float32
	EncFinalNormB []float32
	DecFinalNormG []float32
	DecFinalNormB []float32

	// Tokenizer
	Tok *Tokenizer

	rng *rand.Rand
}

// NewMicroModel creates and randomly initializes a model.
func NewMicroModel(cfg Config) *MicroModel {
	rng := rand.New(rand.NewSource(time.Now().UnixNano()))
	m := &MicroModel{
		Config: cfg,
		Tok:    NewTokenizer(),
		rng:    rng,
	}
	rf := rng.Float64

	dim := cfg.EmbedDim
	ffDim := dim * 4

	// Embeddings
	m.TokenEmbed = make([]float32, cfg.VocabSize*dim)
	initWeight(m.TokenEmbed, cfg.VocabSize, dim, rf)
	m.PosEmbed = make([]float32, cfg.MaxSeqLen*dim)
	initWeight(m.PosEmbed, cfg.MaxSeqLen, dim, rf)

	// Encoder layers
	m.EncLayers = make([]EncoderLayer, cfg.NumLayers)
	for i := range m.EncLayers {
		l := &m.EncLayers[i]
		l.SelfAttnQ = make([]float32, dim*dim)
		l.SelfAttnK = make([]float32, dim*dim)
		l.SelfAttnV = make([]float32, dim*dim)
		l.SelfAttnO = make([]float32, dim*dim)
		initWeight(l.SelfAttnQ, dim, dim, rf)
		initWeight(l.SelfAttnK, dim, dim, rf)
		initWeight(l.SelfAttnV, dim, dim, rf)
		initWeight(l.SelfAttnO, dim, dim, rf)

		l.FFN1W = make([]float32, dim*ffDim)
		l.FFN1B = make([]float32, ffDim)
		l.FFN2W = make([]float32, ffDim*dim)
		l.FFN2B = make([]float32, dim)
		initWeight(l.FFN1W, dim, ffDim, rf)
		initZero(l.FFN1B)
		initWeight(l.FFN2W, ffDim, dim, rf)
		initZero(l.FFN2B)

		l.Norm1G = make([]float32, dim)
		l.Norm1B = make([]float32, dim)
		l.Norm2G = make([]float32, dim)
		l.Norm2B = make([]float32, dim)
		initOnes(l.Norm1G)
		initZero(l.Norm1B)
		initOnes(l.Norm2G)
		initZero(l.Norm2B)
	}

	// Decoder layers
	m.DecLayers = make([]DecoderLayer, cfg.NumLayers)
	for i := range m.DecLayers {
		l := &m.DecLayers[i]
		// Self-attention
		l.SelfAttnQ = make([]float32, dim*dim)
		l.SelfAttnK = make([]float32, dim*dim)
		l.SelfAttnV = make([]float32, dim*dim)
		l.SelfAttnO = make([]float32, dim*dim)
		initWeight(l.SelfAttnQ, dim, dim, rf)
		initWeight(l.SelfAttnK, dim, dim, rf)
		initWeight(l.SelfAttnV, dim, dim, rf)
		initWeight(l.SelfAttnO, dim, dim, rf)
		// Cross-attention
		l.CrossAttnQ = make([]float32, dim*dim)
		l.CrossAttnK = make([]float32, dim*dim)
		l.CrossAttnV = make([]float32, dim*dim)
		l.CrossAttnO = make([]float32, dim*dim)
		initWeight(l.CrossAttnQ, dim, dim, rf)
		initWeight(l.CrossAttnK, dim, dim, rf)
		initWeight(l.CrossAttnV, dim, dim, rf)
		initWeight(l.CrossAttnO, dim, dim, rf)
		// FFN
		l.FFN1W = make([]float32, dim*ffDim)
		l.FFN1B = make([]float32, ffDim)
		l.FFN2W = make([]float32, ffDim*dim)
		l.FFN2B = make([]float32, dim)
		initWeight(l.FFN1W, dim, ffDim, rf)
		initZero(l.FFN1B)
		initWeight(l.FFN2W, ffDim, dim, rf)
		initZero(l.FFN2B)
		// Layer norms
		l.Norm1G = make([]float32, dim)
		l.Norm1B = make([]float32, dim)
		l.Norm2G = make([]float32, dim)
		l.Norm2B = make([]float32, dim)
		l.Norm3G = make([]float32, dim)
		l.Norm3B = make([]float32, dim)
		initOnes(l.Norm1G)
		initZero(l.Norm1B)
		initOnes(l.Norm2G)
		initZero(l.Norm2B)
		initOnes(l.Norm3G)
		initZero(l.Norm3B)
	}

	// Output projection
	m.OutputW = make([]float32, dim*cfg.VocabSize)
	m.OutputB = make([]float32, cfg.VocabSize)
	initWeight(m.OutputW, dim, cfg.VocabSize, rf)
	initZero(m.OutputB)

	// Final norms
	m.EncFinalNormG = make([]float32, dim)
	m.EncFinalNormB = make([]float32, dim)
	m.DecFinalNormG = make([]float32, dim)
	m.DecFinalNormB = make([]float32, dim)
	initOnes(m.EncFinalNormG)
	initZero(m.EncFinalNormB)
	initOnes(m.DecFinalNormG)
	initZero(m.DecFinalNormB)

	return m
}

// Encode runs the encoder on a token sequence.
// Returns encoder output: (seqLen, dim).
func (m *MicroModel) Encode(inputIDs []int) []float32 {
	seqLen := len(inputIDs)
	dim := m.Config.EmbedDim

	// Token + positional embeddings
	x := embedding(inputIDs, m.TokenEmbed, dim)
	x = addPositionalEncoding(x, seqLen, dim, m.PosEmbed)

	// Encoder layers
	for _, l := range m.EncLayers {
		// Self-attention with residual + norm
		normed := layerNorm(x, seqLen, dim, l.Norm1G, l.Norm1B)
		attn := SelfAttention(normed, seqLen, dim, m.Config.NumHeads,
			l.SelfAttnQ, l.SelfAttnK, l.SelfAttnV, l.SelfAttnO, false)
		x = vecAdd(x, attn)

		// FFN with residual + norm
		normed = layerNorm(x, seqLen, dim, l.Norm2G, l.Norm2B)
		ffn := matmulAdd(normed, seqLen, dim, l.FFN1W, dim*4, l.FFN1B)
		ffn = relu(ffn)
		ffn = matmulAdd(ffn, seqLen, dim*4, l.FFN2W, dim, l.FFN2B)
		x = vecAdd(x, ffn)
	}

	// Final layer norm
	x = layerNorm(x, seqLen, dim, m.EncFinalNormG, m.EncFinalNormB)
	return x
}

// Decode runs the decoder for one step (or all steps).
// decIDs is the decoder input sequence, encOut is the encoder output.
// Returns logits: (decLen, VocabSize).
func (m *MicroModel) Decode(decIDs []int, encOut []float32, encLen int) []float32 {
	decLen := len(decIDs)
	dim := m.Config.EmbedDim

	x := embedding(decIDs, m.TokenEmbed, dim)
	x = addPositionalEncoding(x, decLen, dim, m.PosEmbed)

	for _, l := range m.DecLayers {
		// Masked self-attention
		normed := layerNorm(x, decLen, dim, l.Norm1G, l.Norm1B)
		selfAttn := SelfAttention(normed, decLen, dim, m.Config.NumHeads,
			l.SelfAttnQ, l.SelfAttnK, l.SelfAttnV, l.SelfAttnO, true)
		x = vecAdd(x, selfAttn)

		// Cross-attention to encoder
		normed = layerNorm(x, decLen, dim, l.Norm2G, l.Norm2B)
		crossAttn := CrossAttention(normed, encOut, decLen, encLen, dim, m.Config.NumHeads,
			l.CrossAttnQ, l.CrossAttnK, l.CrossAttnV, l.CrossAttnO)
		x = vecAdd(x, crossAttn)

		// FFN
		normed = layerNorm(x, decLen, dim, l.Norm3G, l.Norm3B)
		ffn := matmulAdd(normed, decLen, dim, l.FFN1W, dim*4, l.FFN1B)
		ffn = relu(ffn)
		ffn = matmulAdd(ffn, decLen, dim*4, l.FFN2W, dim, l.FFN2B)
		x = vecAdd(x, ffn)
	}

	// Final norm
	x = layerNorm(x, decLen, dim, m.DecFinalNormG, m.DecFinalNormB)

	// Project to vocabulary logits
	logits := matmulAdd(x, decLen, dim, m.OutputW, m.Config.VocabSize, m.OutputB)
	return logits
}

// Forward runs a full forward pass for training.
// Returns softmax probabilities: (decLen, VocabSize).
func (m *MicroModel) Forward(encIDs, decIDs []int) []float32 {
	encOut := m.Encode(encIDs)
	logits := m.Decode(decIDs, encOut, len(encIDs))
	// Apply softmax to get probabilities
	softmax(logits, len(decIDs), m.Config.VocabSize)
	return logits
}

// Generate produces text auto-regressively from a knowledge triple.
func (m *MicroModel) Generate(subject, relation, object string, maxLen int, temperature float32) string {
	if maxLen <= 0 {
		maxLen = m.Config.MaxSeqLen
	}

	// Encode the triple
	encIDs := m.Tok.EncodeTriple(subject, relation, object)
	if len(encIDs) > m.Config.MaxSeqLen {
		encIDs = encIDs[:m.Config.MaxSeqLen]
	}
	encOut := m.Encode(encIDs)

	// Auto-regressive decoding
	decIDs := []int{BosID}
	rngFunc := func() float64 { return m.rng.Float64() }

	for i := 0; i < maxLen-1; i++ {
		logits := m.Decode(decIDs, encOut, len(encIDs))

		// Get logits for last position
		lastPos := len(decIDs) - 1
		lastLogits := logits[lastPos*m.Config.VocabSize : (lastPos+1)*m.Config.VocabSize]

		// Softmax
		probs := make([]float32, len(lastLogits))
		copy(probs, lastLogits)
		softmax(probs, 1, len(probs))

		// Suppress special tokens in early positions to prevent
		// degenerate generation (untrained model often produces EOS first)
		if i < 3 {
			probs[EosID] = 0
			probs[PadID] = 0
			probs[BosID] = 0
			// Re-normalize
			var sum float32
			for _, p := range probs {
				sum += p
			}
			if sum > 0 {
				for j := range probs {
					probs[j] /= sum
				}
			}
		}

		// Sample
		nextID := sampleToken(probs, temperature, rngFunc)

		if nextID == EosID || nextID == PadID {
			break
		}

		decIDs = append(decIDs, nextID)
	}

	return m.Tok.Decode(decIDs)
}

// ParamCount returns the total number of trainable parameters.
func (m *MicroModel) ParamCount() int {
	dim := m.Config.EmbedDim
	ffDim := dim * 4
	layers := m.Config.NumLayers
	vocab := m.Config.VocabSize
	maxSeq := m.Config.MaxSeqLen

	embedParams := vocab*dim + maxSeq*dim
	encLayerParams := 4*dim*dim + dim*ffDim + ffDim + ffDim*dim + dim + 4*dim // attn + ffn + norms
	decLayerParams := 8*dim*dim + dim*ffDim + ffDim + ffDim*dim + dim + 6*dim // self+cross attn + ffn + norms
	outputParams := dim*vocab + vocab
	normParams := 4 * dim

	return embedParams + layers*(encLayerParams+decLayerParams) + outputParams + normParams
}

// -----------------------------------------------------------------------
// Save / Load — binary format
// -----------------------------------------------------------------------

const modelMagic uint32 = 0x4E4F5553 // "NOUS"
const modelVersion uint32 = 1

// Save writes the model to a binary file.
func (m *MicroModel) Save(path string) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	// Header
	binary.Write(f, binary.LittleEndian, modelMagic)
	binary.Write(f, binary.LittleEndian, modelVersion)

	// Config
	binary.Write(f, binary.LittleEndian, int32(m.Config.VocabSize))
	binary.Write(f, binary.LittleEndian, int32(m.Config.EmbedDim))
	binary.Write(f, binary.LittleEndian, int32(m.Config.NumLayers))
	binary.Write(f, binary.LittleEndian, int32(m.Config.NumHeads))
	binary.Write(f, binary.LittleEndian, int32(m.Config.MaxSeqLen))

	// Vocabulary
	binary.Write(f, binary.LittleEndian, int32(len(m.Tok.ID2Word)))
	for _, word := range m.Tok.ID2Word {
		binary.Write(f, binary.LittleEndian, int32(len(word)))
		f.Write([]byte(word))
	}

	// All weight tensors
	writeFloats := func(data []float32) {
		binary.Write(f, binary.LittleEndian, int32(len(data)))
		binary.Write(f, binary.LittleEndian, data)
	}

	writeFloats(m.TokenEmbed)
	writeFloats(m.PosEmbed)

	for i := range m.EncLayers {
		l := &m.EncLayers[i]
		writeFloats(l.SelfAttnQ)
		writeFloats(l.SelfAttnK)
		writeFloats(l.SelfAttnV)
		writeFloats(l.SelfAttnO)
		writeFloats(l.FFN1W)
		writeFloats(l.FFN1B)
		writeFloats(l.FFN2W)
		writeFloats(l.FFN2B)
		writeFloats(l.Norm1G)
		writeFloats(l.Norm1B)
		writeFloats(l.Norm2G)
		writeFloats(l.Norm2B)
	}

	for i := range m.DecLayers {
		l := &m.DecLayers[i]
		writeFloats(l.SelfAttnQ)
		writeFloats(l.SelfAttnK)
		writeFloats(l.SelfAttnV)
		writeFloats(l.SelfAttnO)
		writeFloats(l.CrossAttnQ)
		writeFloats(l.CrossAttnK)
		writeFloats(l.CrossAttnV)
		writeFloats(l.CrossAttnO)
		writeFloats(l.FFN1W)
		writeFloats(l.FFN1B)
		writeFloats(l.FFN2W)
		writeFloats(l.FFN2B)
		writeFloats(l.Norm1G)
		writeFloats(l.Norm1B)
		writeFloats(l.Norm2G)
		writeFloats(l.Norm2B)
		writeFloats(l.Norm3G)
		writeFloats(l.Norm3B)
	}

	writeFloats(m.OutputW)
	writeFloats(m.OutputB)
	writeFloats(m.EncFinalNormG)
	writeFloats(m.EncFinalNormB)
	writeFloats(m.DecFinalNormG)
	writeFloats(m.DecFinalNormB)

	return nil
}

// Load reads a model from a binary file.
func (m *MicroModel) Load(path string) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()

	var magic, version uint32
	binary.Read(f, binary.LittleEndian, &magic)
	binary.Read(f, binary.LittleEndian, &version)
	if magic != modelMagic {
		return fmt.Errorf("invalid model file (bad magic)")
	}

	var vs, ed, nl, nh, ms int32
	binary.Read(f, binary.LittleEndian, &vs)
	binary.Read(f, binary.LittleEndian, &ed)
	binary.Read(f, binary.LittleEndian, &nl)
	binary.Read(f, binary.LittleEndian, &nh)
	binary.Read(f, binary.LittleEndian, &ms)
	m.Config = Config{VocabSize: int(vs), EmbedDim: int(ed), NumLayers: int(nl), NumHeads: int(nh), MaxSeqLen: int(ms)}

	// Vocabulary
	var vocabLen int32
	binary.Read(f, binary.LittleEndian, &vocabLen)
	m.Tok = NewTokenizer()
	m.Tok.ID2Word = make([]string, vocabLen)
	m.Tok.Word2ID = make(map[string]int, vocabLen)
	for i := 0; i < int(vocabLen); i++ {
		var wl int32
		binary.Read(f, binary.LittleEndian, &wl)
		buf := make([]byte, wl)
		io.ReadFull(f, buf)
		word := string(buf)
		m.Tok.ID2Word[i] = word
		m.Tok.Word2ID[word] = i
	}

	readFloats := func() []float32 {
		var n int32
		binary.Read(f, binary.LittleEndian, &n)
		data := make([]float32, n)
		binary.Read(f, binary.LittleEndian, data)
		return data
	}

	m.TokenEmbed = readFloats()
	m.PosEmbed = readFloats()

	m.EncLayers = make([]EncoderLayer, m.Config.NumLayers)
	for i := range m.EncLayers {
		l := &m.EncLayers[i]
		l.SelfAttnQ = readFloats()
		l.SelfAttnK = readFloats()
		l.SelfAttnV = readFloats()
		l.SelfAttnO = readFloats()
		l.FFN1W = readFloats()
		l.FFN1B = readFloats()
		l.FFN2W = readFloats()
		l.FFN2B = readFloats()
		l.Norm1G = readFloats()
		l.Norm1B = readFloats()
		l.Norm2G = readFloats()
		l.Norm2B = readFloats()
	}

	m.DecLayers = make([]DecoderLayer, m.Config.NumLayers)
	for i := range m.DecLayers {
		l := &m.DecLayers[i]
		l.SelfAttnQ = readFloats()
		l.SelfAttnK = readFloats()
		l.SelfAttnV = readFloats()
		l.SelfAttnO = readFloats()
		l.CrossAttnQ = readFloats()
		l.CrossAttnK = readFloats()
		l.CrossAttnV = readFloats()
		l.CrossAttnO = readFloats()
		l.FFN1W = readFloats()
		l.FFN1B = readFloats()
		l.FFN2W = readFloats()
		l.FFN2B = readFloats()
		l.Norm1G = readFloats()
		l.Norm1B = readFloats()
		l.Norm2G = readFloats()
		l.Norm2B = readFloats()
		l.Norm3G = readFloats()
		l.Norm3B = readFloats()
	}

	m.OutputW = readFloats()
	m.OutputB = readFloats()
	m.EncFinalNormG = readFloats()
	m.EncFinalNormB = readFloats()
	m.DecFinalNormG = readFloats()
	m.DecFinalNormB = readFloats()

	if m.rng == nil {
		m.rng = rand.New(rand.NewSource(time.Now().UnixNano()))
	}

	return nil
}
