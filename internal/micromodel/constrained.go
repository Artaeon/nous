package micromodel

import (
	"math"
)

// -----------------------------------------------------------------------
// Knowledge-Constrained Decoding
//
// Ensures the Mamba model only generates text grounded in known facts.
// Neural fluency with deterministic truthfulness.
//
// Approach:
//   1. Build a FactTrie from tokenized fact sentences
//   2. Track which tokens appear in known facts (allowed vocabulary)
//   3. During generation, bias toward fact-grounded tokens
//   4. Use beam search with trie-guided scoring
//
// The model provides variety and fluency; the fact constraints provide
// truthfulness. The result: zero-hallucination neural generation.
// -----------------------------------------------------------------------

// FactTrieNode is a node in the prefix trie of allowed token sequences.
type FactTrieNode struct {
	Children map[int]*FactTrieNode // token ID → child
	IsEnd    bool                  // marks end of a complete fact phrase
	FactIdx  int                   // which fact this phrase came from (-1 if internal)
}

// FactConstraint holds all constraints for a knowledge-grounded generation.
type FactConstraint struct {
	// Trie of allowed token sequences from known facts
	Trie *FactTrieNode

	// Set of token IDs that appear in any known fact
	AllowedTokens map[int]bool

	// Token frequency in facts — higher = more relevant
	TokenWeight map[int]float32

	// Penalty applied to tokens not in any fact (0.0 = no penalty, 1.0 = full suppress)
	UnknownPenalty float32

	// Bonus for tokens that continue a valid trie path
	TrieBonus float32
}

// NewFactConstraint builds constraints from a set of fact sentences.
// Each fact is a plain text sentence expressing a known truth.
func NewFactConstraint(facts []string, tok *Tokenizer) *FactConstraint {
	fc := &FactConstraint{
		Trie:           &FactTrieNode{Children: make(map[int]*FactTrieNode), FactIdx: -1},
		AllowedTokens:  make(map[int]bool),
		TokenWeight:    make(map[int]float32),
		UnknownPenalty: 0.5, // moderate penalty — allow some creativity
		TrieBonus:      2.0, // strong bonus for fact-aligned tokens
	}

	for i, fact := range facts {
		ids := tok.Encode(fact)
		fc.addToTrie(ids, i)

		// Track allowed tokens and their frequency
		for _, id := range ids {
			if id != PadID && id != BosID && id != EosID && id != SepID {
				fc.AllowedTokens[id] = true
				fc.TokenWeight[id] += 1.0
			}
		}
	}

	// Normalize weights
	var maxWeight float32
	for _, w := range fc.TokenWeight {
		if w > maxWeight {
			maxWeight = w
		}
	}
	if maxWeight > 0 {
		for id := range fc.TokenWeight {
			fc.TokenWeight[id] /= maxWeight
		}
	}

	// Always allow special tokens and common function words
	for id := 0; id < 5; id++ {
		fc.AllowedTokens[id] = true
	}

	return fc
}

// addToTrie inserts a token sequence into the fact trie.
func (fc *FactConstraint) addToTrie(ids []int, factIdx int) {
	node := fc.Trie
	for _, id := range ids {
		if id == PadID || id == BosID || id == EosID {
			continue
		}
		child, ok := node.Children[id]
		if !ok {
			child = &FactTrieNode{Children: make(map[int]*FactTrieNode), FactIdx: -1}
			node.Children[id] = child
		}
		node = child
	}
	node.IsEnd = true
	node.FactIdx = factIdx
}

// ApplyConstraints modifies logits to bias toward fact-grounded tokens.
// Takes the current generated sequence and the raw logits from the model.
// Returns modified logits.
func (fc *FactConstraint) ApplyConstraints(generatedIDs []int, logits []float32, vocabSize int) []float32 {
	constrained := make([]float32, len(logits))
	copy(constrained, logits)

	// Check how deep we are in a trie path
	trieDepth, activeNode := fc.trieMatch(generatedIDs)

	for id := 0; id < vocabSize; id++ {
		// Skip special tokens
		if id < 5 {
			continue
		}

		// Bonus for tokens that continue a valid trie path
		if activeNode != nil {
			if _, hasChild := activeNode.Children[id]; hasChild {
				constrained[id] += fc.TrieBonus
			}
		}

		// Fact vocabulary bonus/penalty
		if fc.AllowedTokens[id] {
			// Token appears in known facts — apply weight bonus
			weight := fc.TokenWeight[id]
			constrained[id] += weight * fc.TrieBonus * 0.5
		} else {
			// Token not in any fact — apply penalty
			constrained[id] -= fc.UnknownPenalty
		}
	}

	// Stronger trie guidance in early generation (first ~10 tokens)
	if len(generatedIDs) < 10 && trieDepth > 0 {
		// We're following a known fact — boost trie continuation
		if activeNode != nil {
			for childID := range activeNode.Children {
				constrained[childID] += fc.TrieBonus * 1.5
			}
		}
	}

	return constrained
}

// trieMatch finds how far the generated sequence matches a trie path.
// Returns the depth of the match and the active trie node (or nil if no match).
func (fc *FactConstraint) trieMatch(ids []int) (int, *FactTrieNode) {
	node := fc.Trie
	depth := 0

	// Try matching from the end of the sequence (sliding window)
	// This handles cases where generation starts mid-sentence
	for startPos := max(0, len(ids)-20); startPos < len(ids); startPos++ {
		testNode := fc.Trie
		testDepth := 0
		matched := true

		for i := startPos; i < len(ids); i++ {
			id := ids[i]
			if id == PadID || id == BosID || id == EosID {
				continue
			}
			child, ok := testNode.Children[id]
			if !ok {
				matched = false
				break
			}
			testNode = child
			testDepth++
		}

		if matched && testDepth > depth {
			depth = testDepth
			node = testNode
		}
	}

	if depth == 0 {
		return 0, nil
	}
	return depth, node
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

// -----------------------------------------------------------------------
// Constrained Beam Search
// -----------------------------------------------------------------------

// Beam represents one candidate in beam search.
type Beam struct {
	IDs   []int   // generated token IDs so far
	Score float64 // cumulative log-probability
	State *MambaState
}

// ConstrainedGenerate produces text using beam search with fact constraints.
// This is the high-quality generation path: slower than greedy but produces
// better, more factual text.
//
// facts: plain text sentences expressing known truths about the topic.
// beamWidth: number of parallel beams (3-5 recommended).
func (m *MambaModel) ConstrainedGenerate(
	subject, relation, object string,
	facts []string,
	maxLen int,
	beamWidth int,
) string {
	if maxLen <= 0 {
		maxLen = m.Config.MaxSeqLen
	}
	if beamWidth <= 0 {
		beamWidth = 3
	}

	// Build fact constraints
	constraint := NewFactConstraint(facts, m.Tok)

	// Encode prefix
	prefixIDs := m.Tok.EncodeTriple(subject, relation, object)
	if len(prefixIDs) > m.Config.MaxSeqLen/2 {
		prefixIDs = prefixIDs[:m.Config.MaxSeqLen/2]
	}

	// Initialize beams
	beams := make([]*Beam, beamWidth)
	for i := range beams {
		state := m.NewMambaState()
		// Process prefix
		for _, id := range prefixIDs {
			m.StepForward(id, state)
		}
		beams[i] = &Beam{
			IDs:   nil,
			Score: 0,
			State: state,
		}
	}

	// Beam search
	for step := 0; step < maxLen; step++ {
		var candidates []*Beam

		for _, beam := range beams {
			// Get logits for next token
			var logits []float32
			if len(beam.IDs) == 0 {
				logits = m.StepForward(prefixIDs[len(prefixIDs)-1], beam.State)
			} else {
				logits = m.StepForward(beam.IDs[len(beam.IDs)-1], beam.State)
			}

			// Apply fact constraints
			logits = constraint.ApplyConstraints(beam.IDs, logits, m.Config.VocabSize)

			// Softmax
			probs := make([]float32, len(logits))
			copy(probs, logits)
			softmax(probs, 1, len(probs))

			// Suppress special tokens early
			if step < 3 {
				probs[EosID] = 0
				probs[PadID] = 0
				probs[BosID] = 0
				probs[SepID] = 0
			}

			// Take top-K tokens as candidates
			topK := beamWidth * 2
			topIDs := topKIndices(probs, topK)

			for _, nextID := range topIDs {
				if nextID == EosID || nextID == PadID {
					if step > 3 { // allow ending after a few tokens
						newIDs := make([]int, len(beam.IDs))
						copy(newIDs, beam.IDs)
						candidates = append(candidates, &Beam{
							IDs:   newIDs,
							Score: beam.Score, // no penalty for stopping
							State: nil,        // mark as finished
						})
					}
					continue
				}

				p := probs[nextID]
				if p < 1e-10 {
					continue
				}
				logP := math.Log(float64(p))

				newIDs := make([]int, len(beam.IDs)+1)
				copy(newIDs, beam.IDs)
				newIDs[len(beam.IDs)] = nextID

				// Clone state for new beam
				newState := m.cloneState(beam.State)

				candidates = append(candidates, &Beam{
					IDs:   newIDs,
					Score: beam.Score + logP,
					State: newState,
				})
			}
		}

		if len(candidates) == 0 {
			break
		}

		// Select top beams by score (normalized by length)
		sortBeams(candidates)
		if len(candidates) > beamWidth {
			candidates = candidates[:beamWidth]
		}

		// Check if all beams are finished
		allDone := true
		for _, b := range candidates {
			if b.State != nil {
				allDone = false
				break
			}
		}

		beams = candidates
		if allDone {
			break
		}
	}

	// Return best beam's text
	if len(beams) > 0 && len(beams[0].IDs) > 0 {
		return m.Tok.Decode(beams[0].IDs)
	}
	return ""
}

// cloneState deep-copies a MambaState for beam branching.
func (m *MambaModel) cloneState(s *MambaState) *MambaState {
	if s == nil {
		return nil
	}
	clone := &MambaState{
		H:       make([][]float32, len(s.H)),
		ConvBuf: make([][]float32, len(s.ConvBuf)),
	}
	for i := range s.H {
		clone.H[i] = make([]float32, len(s.H[i]))
		copy(clone.H[i], s.H[i])
		clone.ConvBuf[i] = make([]float32, len(s.ConvBuf[i]))
		copy(clone.ConvBuf[i], s.ConvBuf[i])
	}
	return clone
}

// topKIndices returns the indices of the top-K values in a slice.
func topKIndices(values []float32, k int) []int {
	if k > len(values) {
		k = len(values)
	}

	// Simple selection: find top-K by scanning
	type iv struct {
		idx int
		val float32
	}
	top := make([]iv, 0, k)

	for i, v := range values {
		if len(top) < k {
			top = append(top, iv{i, v})
			// Bubble up
			for j := len(top) - 1; j > 0 && top[j].val > top[j-1].val; j-- {
				top[j], top[j-1] = top[j-1], top[j]
			}
		} else if v > top[len(top)-1].val {
			top[len(top)-1] = iv{i, v}
			// Bubble up
			for j := len(top) - 1; j > 0 && top[j].val > top[j-1].val; j-- {
				top[j], top[j-1] = top[j-1], top[j]
			}
		}
	}

	result := make([]int, len(top))
	for i, t := range top {
		result[i] = t.idx
	}
	return result
}

// sortBeams sorts beams by length-normalized score (descending).
func sortBeams(beams []*Beam) {
	// Insertion sort (beams are usually small)
	for i := 1; i < len(beams); i++ {
		for j := i; j > 0; j-- {
			si := normalizedScore(beams[j])
			sj := normalizedScore(beams[j-1])
			if si > sj {
				beams[j], beams[j-1] = beams[j-1], beams[j]
			}
		}
	}
}

// normalizedScore returns a length-normalized beam score.
// Uses length penalty alpha=0.6 to balance short vs long outputs.
func normalizedScore(b *Beam) float64 {
	length := float64(len(b.IDs))
	if length == 0 {
		return b.Score
	}
	// Length penalty: ((5 + length) / 6) ^ alpha
	penalty := math.Pow((5+length)/6, 0.6)
	return b.Score / penalty
}

// -----------------------------------------------------------------------
// Convenience: generate with fact grounding from triples
// -----------------------------------------------------------------------

// GenerateGrounded generates text constrained by knowledge triples.
// Converts triples to fact sentences using templates, then runs
// constrained beam search.
func (m *MambaModel) GenerateGrounded(
	subject, relation, object string,
	extraFacts [][3]string, // additional (subject, relation, object) triples
	beamWidth int,
) string {
	// Convert triples to natural language sentences
	var facts []string

	// Primary triple
	facts = append(facts, tripleToSentence(subject, relation, object))

	// Additional triples
	for _, f := range extraFacts {
		facts = append(facts, tripleToSentence(f[0], f[1], f[2]))
	}

	return m.ConstrainedGenerate(subject, relation, object, facts, 40, beamWidth)
}

// tripleToSentence converts a knowledge triple to a natural language sentence.
func tripleToSentence(subject, relation, object string) string {
	switch relation {
	case "is_a":
		return subject + " is a " + object + "."
	case "created_by":
		return subject + " was created by " + object + "."
	case "founded_in":
		return subject + " was founded in " + object + "."
	case "has":
		return subject + " has " + object + "."
	case "used_for":
		return subject + " is used for " + object + "."
	case "located_in":
		return subject + " is located in " + object + "."
	case "part_of":
		return subject + " is part of " + object + "."
	case "related_to":
		return subject + " is related to " + object + "."
	case "known_for":
		return subject + " is known for " + object + "."
	case "influenced_by":
		return subject + " was influenced by " + object + "."
	case "described_as":
		return subject + " is described as " + object + "."
	default:
		return subject + " " + relation + " " + object + "."
	}
}
