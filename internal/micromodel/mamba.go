package micromodel

import (
	"encoding/binary"
	"fmt"
	"math"
	"math/rand"
	"os"
	"time"
)

// -----------------------------------------------------------------------
// Mamba Model — Linear-time sequence modeling via selective state spaces.
//
// Architecture per block:
//   x ─→ LayerNorm ─→ InProj ─→ split ─→ xBranch ─→ Conv1D ─→ SiLU ─→ SSM ─→ gate ─→ OutProj ─→ + residual
//                                     └→ zBranch ──────────────────────→ SiLU ─→ ↑
//
// Key advantage over transformer: O(n) inference vs O(n²), no KV cache needed.
// The hidden state is a fixed-size (dInner × dState) matrix that gets updated
// per token, making autoregressive generation constant-time per step.
// -----------------------------------------------------------------------

// MambaConfig defines the Mamba model architecture.
type MambaConfig struct {
	VocabSize int // built from training data (default: 8000)
	ModelDim  int // d_model: main hidden dimension (default: 256)
	StateDim  int // d_state: SSM state dimension (default: 16)
	ConvDim   int // d_conv: causal convolution kernel size (default: 4)
	Expand    int // expansion factor for inner dimension (default: 2)
	NumLayers int // number of Mamba blocks (default: 8)
	MaxSeqLen int // maximum sequence length (default: 128)
}

// DefaultMambaConfig returns the standard ~8M parameter configuration.
func DefaultMambaConfig() MambaConfig {
	return MambaConfig{
		VocabSize: 8000,
		ModelDim:  256,
		StateDim:  16,
		ConvDim:   4,
		Expand:    2,
		NumLayers: 8,
		MaxSeqLen: 128,
	}
}

// SmallMambaConfig returns a smaller config for fast testing.
func SmallMambaConfig() MambaConfig {
	return MambaConfig{
		VocabSize: 512,
		ModelDim:  64,
		StateDim:  8,
		ConvDim:   4,
		Expand:    2,
		NumLayers: 2,
		MaxSeqLen: 32,
	}
}

// DtRank returns the rank of the dt projection (ceil(ModelDim / 16)).
func (c MambaConfig) DtRank() int {
	return (c.ModelDim + 15) / 16
}

// InnerDim returns the expanded inner dimension (ModelDim * Expand).
func (c MambaConfig) InnerDim() int {
	return c.ModelDim * c.Expand
}

// MambaBlock holds weights for one Mamba block.
type MambaBlock struct {
	// Pre-block layer norm
	NormG []float32 // (ModelDim)
	NormB []float32 // (ModelDim)

	// Input projection: ModelDim → 2*InnerDim (x branch + z gate branch)
	InProjW []float32 // (ModelDim, 2*InnerDim)
	InProjB []float32 // (2*InnerDim)

	// Causal 1D depthwise convolution on x branch
	Conv1DW []float32 // (InnerDim, ConvDim)
	Conv1DB []float32 // (InnerDim)

	// SSM parameter projection: InnerDim → DtRank + 2*StateDim
	XProjW []float32 // (InnerDim, DtRank + 2*StateDim)

	// Dt projection: DtRank → InnerDim (with bias, then softplus)
	DtProjW []float32 // (DtRank, InnerDim)
	DtProjB []float32 // (InnerDim)

	// SSM state matrix (stored as log for numerical stability)
	ALog []float32 // (InnerDim, StateDim)

	// Skip connection parameter
	D []float32 // (InnerDim)

	// Output projection: InnerDim → ModelDim
	OutProjW []float32 // (InnerDim, ModelDim)
	OutProjB []float32 // (ModelDim)
}

// MambaModel is a Mamba-based language model for knowledge-grounded generation.
// Decoder-only: the knowledge triple is the prefix, generation follows.
type MambaModel struct {
	Config MambaConfig

	// Token embeddings
	TokenEmbed []float32 // (VocabSize, ModelDim)

	// Mamba blocks
	Layers []MambaBlock

	// Final layer norm
	FinalNormG []float32 // (ModelDim)
	FinalNormB []float32 // (ModelDim)

	// Output projection to vocabulary
	OutputW []float32 // (ModelDim, VocabSize)
	OutputB []float32 // (VocabSize)

	// Tokenizer (shared with transformer model)
	Tok *Tokenizer

	rng *rand.Rand
}

// NewMambaModel creates and randomly initializes a Mamba model.
func NewMambaModel(cfg MambaConfig) *MambaModel {
	rng := rand.New(rand.NewSource(time.Now().UnixNano()))
	m := &MambaModel{
		Config: cfg,
		Tok:    NewTokenizer(),
		rng:    rng,
	}
	rf := rng.Float64

	dim := cfg.ModelDim
	inner := cfg.InnerDim()
	dState := cfg.StateDim
	dtRank := cfg.DtRank()
	kSize := cfg.ConvDim

	// Token embeddings
	m.TokenEmbed = make([]float32, cfg.VocabSize*dim)
	initWeight(m.TokenEmbed, cfg.VocabSize, dim, rf)

	// Mamba blocks
	m.Layers = make([]MambaBlock, cfg.NumLayers)
	for i := range m.Layers {
		l := &m.Layers[i]

		// Layer norm
		l.NormG = make([]float32, dim)
		l.NormB = make([]float32, dim)
		initOnes(l.NormG)
		initZero(l.NormB)

		// Input projection
		l.InProjW = make([]float32, dim*2*inner)
		l.InProjB = make([]float32, 2*inner)
		initWeight(l.InProjW, dim, 2*inner, rf)
		initZero(l.InProjB)

		// Conv1D
		l.Conv1DW = make([]float32, inner*kSize)
		l.Conv1DB = make([]float32, inner)
		initWeight(l.Conv1DW, inner, kSize, rf)
		initZero(l.Conv1DB)

		// SSM parameter projection
		xProjOut := dtRank + 2*dState
		l.XProjW = make([]float32, inner*xProjOut)
		initWeight(l.XProjW, inner, xProjOut, rf)

		// Dt projection
		l.DtProjW = make([]float32, dtRank*inner)
		l.DtProjB = make([]float32, inner)
		initWeight(l.DtProjW, dtRank, inner, rf)
		// Initialize dt bias to be in a reasonable range after softplus
		for j := range l.DtProjB {
			l.DtProjB[j] = float32(rf()*0.5 + 0.5) // [0.5, 1.0]
		}

		// A_log: initialize to create a range of timescales
		l.ALog = make([]float32, inner*dState)
		for j := 0; j < inner; j++ {
			for s := 0; s < dState; s++ {
				// Log-spaced initialization as in the Mamba paper
				l.ALog[j*dState+s] = float32(math.Log(float64(s + 1)))
			}
		}

		// D (skip connection): initialize to ones
		l.D = make([]float32, inner)
		initOnes(l.D)

		// Output projection
		l.OutProjW = make([]float32, inner*dim)
		l.OutProjB = make([]float32, dim)
		initWeight(l.OutProjW, inner, dim, rf)
		initZero(l.OutProjB)
	}

	// Final norm
	m.FinalNormG = make([]float32, dim)
	m.FinalNormB = make([]float32, dim)
	initOnes(m.FinalNormG)
	initZero(m.FinalNormB)

	// Output projection
	m.OutputW = make([]float32, dim*cfg.VocabSize)
	m.OutputB = make([]float32, cfg.VocabSize)
	initWeight(m.OutputW, dim, cfg.VocabSize, rf)
	initZero(m.OutputB)

	return m
}

// ParamCount returns the total number of trainable parameters.
func (m *MambaModel) ParamCount() int {
	cfg := m.Config
	dim := cfg.ModelDim
	inner := cfg.InnerDim()
	dState := cfg.StateDim
	dtRank := cfg.DtRank()
	kSize := cfg.ConvDim

	embedParams := cfg.VocabSize * dim

	blockParams := 0
	// Norm
	blockParams += 2 * dim
	// InProj
	blockParams += dim*2*inner + 2*inner
	// Conv1D
	blockParams += inner*kSize + inner
	// XProj
	blockParams += inner * (dtRank + 2*dState)
	// DtProj
	blockParams += dtRank*inner + inner
	// ALog + D
	blockParams += inner*dState + inner
	// OutProj
	blockParams += inner*dim + dim

	totalBlockParams := cfg.NumLayers * blockParams
	finalNormParams := 2 * dim
	outputParams := dim*cfg.VocabSize + cfg.VocabSize

	return embedParams + totalBlockParams + finalNormParams + outputParams
}

// -----------------------------------------------------------------------
// Forward pass
// -----------------------------------------------------------------------

// MambaBlockCache holds intermediate values for one block's forward pass
// needed for backpropagation.
type MambaBlockCache struct {
	normed   []float32 // after layer norm
	xBranch  []float32 // x branch before conv1d
	xConv    []float32 // after conv1d, before SiLU
	xSilu    []float32 // after SiLU
	zBranch  []float32 // z branch (gate)
	zSilu    []float32 // z branch after SiLU
	dt       []float32 // discretization step after softplus
	dtPre    []float32 // dt before softplus
	A        []float32 // -exp(ALog)
	B        []float32 // input-dependent B
	C        []float32 // input-dependent C
	ssmOut   []float32 // SSM output
	gated    []float32 // after gating
	residual []float32 // input to block (for residual)
}

// Forward runs the full Mamba model forward pass.
// inputIDs is the token sequence.
// Returns logits: (seqLen, VocabSize).
func (m *MambaModel) Forward(inputIDs []int) []float32 {
	_, logits := m.forwardWithCache(inputIDs)
	return logits
}

// forwardWithCache runs forward pass and returns per-block caches for training.
func (m *MambaModel) forwardWithCache(inputIDs []int) ([]MambaBlockCache, []float32) {
	seqLen := len(inputIDs)
	dim := m.Config.ModelDim
	inner := m.Config.InnerDim()
	dState := m.Config.StateDim
	dtRank := m.Config.DtRank()
	kSize := m.Config.ConvDim

	// Token embeddings
	x := embedding(inputIDs, m.TokenEmbed, dim)

	caches := make([]MambaBlockCache, len(m.Layers))

	for li, l := range m.Layers {
		cache := &caches[li]
		cache.residual = make([]float32, len(x))
		copy(cache.residual, x)

		// Layer norm
		normed := layerNorm(x, seqLen, dim, l.NormG, l.NormB)
		cache.normed = normed

		// Input projection → (seqLen, 2*inner)
		proj := matmulAdd(normed, seqLen, dim, l.InProjW, 2*inner, l.InProjB)

		// Split into x branch and z branch
		xBranch := make([]float32, seqLen*inner)
		zBranch := make([]float32, seqLen*inner)
		for t := 0; t < seqLen; t++ {
			copy(xBranch[t*inner:], proj[t*2*inner:t*2*inner+inner])
			copy(zBranch[t*inner:], proj[t*2*inner+inner:(t+1)*2*inner])
		}
		cache.xBranch = make([]float32, len(xBranch))
		copy(cache.xBranch, xBranch)
		cache.zBranch = zBranch

		// Causal 1D convolution on x branch
		xConv := conv1D(xBranch, seqLen, inner, l.Conv1DW, l.Conv1DB, kSize)
		cache.xConv = xConv

		// SiLU activation
		xSilu := silu(xConv)
		cache.xSilu = xSilu

		// Project to SSM parameters: (seqLen, inner) → (seqLen, dtRank + 2*dState)
		xProjOut := dtRank + 2*dState
		ssmParams := matmul(xSilu, seqLen, inner, l.XProjW, xProjOut)

		// Split into dt_input, B, C
		dtInput := make([]float32, seqLen*dtRank)
		B := make([]float32, seqLen*dState)
		C := make([]float32, seqLen*dState)
		for t := 0; t < seqLen; t++ {
			row := ssmParams[t*xProjOut:]
			copy(dtInput[t*dtRank:], row[:dtRank])
			copy(B[t*dState:], row[dtRank:dtRank+dState])
			copy(C[t*dState:], row[dtRank+dState:dtRank+2*dState])
		}
		cache.B = B
		cache.C = C

		// Dt projection: (seqLen, dtRank) → (seqLen, inner)
		dtPre := matmulAdd(dtInput, seqLen, dtRank, l.DtProjW, inner, l.DtProjB)
		cache.dtPre = dtPre

		// Softplus to ensure positive dt
		dt := softplus(dtPre)
		cache.dt = dt

		// Reconstruct A from log (negative for stability)
		A := make([]float32, len(l.ALog))
		for j := range A {
			A[j] = -float32(math.Exp(float64(l.ALog[j])))
		}
		cache.A = A

		// Core SSM: selective scan
		ssmOut := SelectiveScan(xSilu, seqLen, inner, dt, A, dState, B, C, l.D)
		cache.ssmOut = ssmOut

		// Gate with z branch
		zSilu := silu(zBranch)
		cache.zSilu = zSilu
		gated := vecMul(ssmOut, zSilu)
		cache.gated = gated

		// Output projection: (seqLen, inner) → (seqLen, dim)
		out := matmulAdd(gated, seqLen, inner, l.OutProjW, dim, l.OutProjB)

		// Residual connection
		x = vecAdd(cache.residual, out)
	}

	// Final layer norm
	x = layerNorm(x, seqLen, dim, m.FinalNormG, m.FinalNormB)

	// Output projection to vocabulary logits
	logits := matmulAdd(x, seqLen, dim, m.OutputW, m.Config.VocabSize, m.OutputB)

	return caches, logits
}

// -----------------------------------------------------------------------
// Generation (inference)
// -----------------------------------------------------------------------

// MambaState holds the recurrent state for autoregressive generation.
// This makes generation O(1) per token instead of O(n).
type MambaState struct {
	// Per-layer SSM hidden states: (dInner, dState)
	H [][]float32
	// Per-layer conv1d buffer: last (kSize-1) inputs
	ConvBuf [][]float32
}

// NewMambaState creates a fresh state for generation.
func (m *MambaModel) NewMambaState() *MambaState {
	inner := m.Config.InnerDim()
	dState := m.Config.StateDim
	kSize := m.Config.ConvDim

	s := &MambaState{
		H:       make([][]float32, len(m.Layers)),
		ConvBuf: make([][]float32, len(m.Layers)),
	}
	for i := range m.Layers {
		s.H[i] = make([]float32, inner*dState)
		s.ConvBuf[i] = make([]float32, (kSize-1)*inner)
	}
	return s
}

// StepForward runs one token through the model, updating state in-place.
// Returns logits for the single token: (VocabSize).
func (m *MambaModel) StepForward(tokenID int, state *MambaState) []float32 {
	dim := m.Config.ModelDim
	inner := m.Config.InnerDim()
	dState := m.Config.StateDim
	dtRank := m.Config.DtRank()
	kSize := m.Config.ConvDim

	// Token embedding for single token
	x := make([]float32, dim)
	if tokenID >= 0 && tokenID*dim+dim <= len(m.TokenEmbed) {
		copy(x, m.TokenEmbed[tokenID*dim:(tokenID+1)*dim])
	}

	for li, l := range m.Layers {
		residual := make([]float32, dim)
		copy(residual, x)

		// Layer norm (single vector)
		normed := layerNorm(x, 1, dim, l.NormG, l.NormB)

		// Input projection → (2*inner)
		proj := matmulAdd(normed, 1, dim, l.InProjW, 2*inner, l.InProjB)
		xBranch := proj[:inner]
		zBranch := proj[inner:]

		// Conv1D with state buffer
		// Shift buffer left, append new input
		buf := state.ConvBuf[li]
		bufLen := kSize - 1
		// Shift: drop oldest, append new
		copy(buf, buf[inner:])
		copy(buf[(bufLen-1)*inner:], xBranch)

		// Apply conv: sum over kernel positions
		xConv := make([]float32, inner)
		for ch := 0; ch < inner; ch++ {
			var sum float32
			// k=0 is current position (xBranch)
			sum += xBranch[ch] * l.Conv1DW[ch*kSize+0]
			// k=1..kSize-1 are past positions from buffer
			for k := 1; k < kSize; k++ {
				bufIdx := bufLen - k
				if bufIdx >= 0 {
					sum += buf[bufIdx*inner+ch] * l.Conv1DW[ch*kSize+k]
				}
			}
			xConv[ch] = sum + l.Conv1DB[ch]
		}

		// SiLU
		xAct := silu(xConv)

		// Project to SSM parameters
		xProjOut := dtRank + 2*dState
		ssmParams := matmul(xAct, 1, inner, l.XProjW, xProjOut)

		dtInput := ssmParams[:dtRank]
		B := ssmParams[dtRank : dtRank+dState]
		C := ssmParams[dtRank+dState:]

		// Dt projection + softplus
		dtVal := matmulAdd(dtInput, 1, dtRank, l.DtProjW, inner, l.DtProjB)
		dtVal = softplus(dtVal)

		// SSM step: update state and compute output
		h := state.H[li]
		ySSM := make([]float32, inner)
		for i := 0; i < inner; i++ {
			dt := dtVal[i]
			xv := xAct[i]
			var yVal float32
			for s := 0; s < dState; s++ {
				aBar := float32(math.Exp(float64(-dt * float32(math.Exp(float64(l.ALog[i*dState+s]))))))
				bBar := dt * B[s]
				h[i*dState+s] = aBar*h[i*dState+s] + bBar*xv
				yVal += C[s] * h[i*dState+s]
			}
			ySSM[i] = yVal + l.D[i]*xv
		}

		// Gate with z branch
		zAct := silu(zBranch)
		gated := vecMul(ySSM, zAct)

		// Output projection
		out := matmulAdd(gated, 1, inner, l.OutProjW, dim, l.OutProjB)

		// Residual
		x = vecAdd(residual, out)
	}

	// Final norm
	x = layerNorm(x, 1, dim, m.FinalNormG, m.FinalNormB)

	// Output projection
	logits := matmulAdd(x, 1, dim, m.OutputW, m.Config.VocabSize, m.OutputB)

	return logits
}

// Generate produces text auto-regressively from a knowledge triple.
// Uses stateful inference: O(1) per token, O(n) total.
func (m *MambaModel) Generate(subject, relation, object string, maxLen int, temperature float32) string {
	if maxLen <= 0 {
		maxLen = m.Config.MaxSeqLen
	}

	// Encode the triple as prefix: <bos> subject <sep> relation <sep> object <sep>
	prefixIDs := m.Tok.EncodeTriple(subject, relation, object)
	if len(prefixIDs) > m.Config.MaxSeqLen/2 {
		prefixIDs = prefixIDs[:m.Config.MaxSeqLen/2]
	}

	state := m.NewMambaState()
	rngFunc := func() float64 { return m.rng.Float64() }

	// Process prefix tokens (building up state)
	for _, id := range prefixIDs {
		m.StepForward(id, state)
	}

	// Auto-regressive generation
	var genIDs []int
	for i := 0; i < maxLen; i++ {
		var logits []float32
		if len(genIDs) == 0 {
			// First generated token: use last prefix token's output
			logits = m.StepForward(prefixIDs[len(prefixIDs)-1], state)
		} else {
			logits = m.StepForward(genIDs[len(genIDs)-1], state)
		}

		// Softmax
		probs := make([]float32, len(logits))
		copy(probs, logits)
		softmax(probs, 1, len(probs))

		// Suppress special tokens early
		if i < 3 {
			probs[EosID] = 0
			probs[PadID] = 0
			probs[BosID] = 0
			probs[SepID] = 0
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

		nextID := sampleToken(probs, temperature, rngFunc)
		if nextID == EosID || nextID == PadID {
			break
		}
		genIDs = append(genIDs, nextID)
	}

	return m.Tok.Decode(genIDs)
}

// -----------------------------------------------------------------------
// Save / Load — binary format (version 2 for Mamba)
// -----------------------------------------------------------------------

const mambaModelMagic uint32 = 0x4D414D42   // "MAMB"
const mambaModelVersion uint32 = 1

// Save writes the Mamba model to a binary file.
func (m *MambaModel) Save(path string) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	// Header
	binary.Write(f, binary.LittleEndian, mambaModelMagic)
	binary.Write(f, binary.LittleEndian, mambaModelVersion)

	// Config
	binary.Write(f, binary.LittleEndian, int32(m.Config.VocabSize))
	binary.Write(f, binary.LittleEndian, int32(m.Config.ModelDim))
	binary.Write(f, binary.LittleEndian, int32(m.Config.StateDim))
	binary.Write(f, binary.LittleEndian, int32(m.Config.ConvDim))
	binary.Write(f, binary.LittleEndian, int32(m.Config.Expand))
	binary.Write(f, binary.LittleEndian, int32(m.Config.NumLayers))
	binary.Write(f, binary.LittleEndian, int32(m.Config.MaxSeqLen))

	// Vocabulary
	binary.Write(f, binary.LittleEndian, int32(len(m.Tok.ID2Word)))
	for _, word := range m.Tok.ID2Word {
		binary.Write(f, binary.LittleEndian, int32(len(word)))
		f.Write([]byte(word))
	}

	writeFloats := func(data []float32) {
		binary.Write(f, binary.LittleEndian, int32(len(data)))
		binary.Write(f, binary.LittleEndian, data)
	}

	// Token embeddings
	writeFloats(m.TokenEmbed)

	// Mamba blocks
	for i := range m.Layers {
		l := &m.Layers[i]
		writeFloats(l.NormG)
		writeFloats(l.NormB)
		writeFloats(l.InProjW)
		writeFloats(l.InProjB)
		writeFloats(l.Conv1DW)
		writeFloats(l.Conv1DB)
		writeFloats(l.XProjW)
		writeFloats(l.DtProjW)
		writeFloats(l.DtProjB)
		writeFloats(l.ALog)
		writeFloats(l.D)
		writeFloats(l.OutProjW)
		writeFloats(l.OutProjB)
	}

	// Final norm + output
	writeFloats(m.FinalNormG)
	writeFloats(m.FinalNormB)
	writeFloats(m.OutputW)
	writeFloats(m.OutputB)

	return nil
}

// LoadMambaModel reads a Mamba model from a binary file.
func LoadMambaModel(path string) (*MambaModel, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var magic, version uint32
	binary.Read(f, binary.LittleEndian, &magic)
	binary.Read(f, binary.LittleEndian, &version)
	if magic != mambaModelMagic {
		return nil, fmt.Errorf("invalid Mamba model file (bad magic: %x)", magic)
	}

	var vs, md, sd, cd, ex, nl, ms int32
	binary.Read(f, binary.LittleEndian, &vs)
	binary.Read(f, binary.LittleEndian, &md)
	binary.Read(f, binary.LittleEndian, &sd)
	binary.Read(f, binary.LittleEndian, &cd)
	binary.Read(f, binary.LittleEndian, &ex)
	binary.Read(f, binary.LittleEndian, &nl)
	binary.Read(f, binary.LittleEndian, &ms)

	m := &MambaModel{
		Config: MambaConfig{
			VocabSize: int(vs),
			ModelDim:  int(md),
			StateDim:  int(sd),
			ConvDim:   int(cd),
			Expand:    int(ex),
			NumLayers: int(nl),
			MaxSeqLen: int(ms),
		},
		rng: rand.New(rand.NewSource(time.Now().UnixNano())),
	}

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
		f.Read(buf)
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

	// Token embeddings
	m.TokenEmbed = readFloats()

	// Mamba blocks
	m.Layers = make([]MambaBlock, m.Config.NumLayers)
	for i := range m.Layers {
		l := &m.Layers[i]
		l.NormG = readFloats()
		l.NormB = readFloats()
		l.InProjW = readFloats()
		l.InProjB = readFloats()
		l.Conv1DW = readFloats()
		l.Conv1DB = readFloats()
		l.XProjW = readFloats()
		l.DtProjW = readFloats()
		l.DtProjB = readFloats()
		l.ALog = readFloats()
		l.D = readFloats()
		l.OutProjW = readFloats()
		l.OutProjB = readFloats()
	}

	// Final norm + output
	m.FinalNormG = readFloats()
	m.FinalNormB = readFloats()
	m.OutputW = readFloats()
	m.OutputB = readFloats()

	return m, nil
}
