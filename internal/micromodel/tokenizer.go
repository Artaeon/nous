package micromodel

import (
	"sort"
	"strings"
	"unicode"
)

// Special token IDs.
const (
	PadID = 0
	BosID = 1
	EosID = 2
	UnkID = 3
	SepID = 4
)

// Tokenizer converts text to integer token sequences and back.
type Tokenizer struct {
	Word2ID map[string]int
	ID2Word []string
}

// NewTokenizer creates an empty tokenizer with special tokens.
func NewTokenizer() *Tokenizer {
	t := &Tokenizer{
		Word2ID: map[string]int{
			"<pad>": PadID,
			"<bos>": BosID,
			"<eos>": EosID,
			"<unk>": UnkID,
			"<sep>": SepID,
		},
		ID2Word: []string{"<pad>", "<bos>", "<eos>", "<unk>", "<sep>"},
	}
	return t
}

// BuildVocab constructs the vocabulary from training texts.
// Keeps the top maxVocab words by frequency.
func (t *Tokenizer) BuildVocab(texts []string, maxVocab int) {
	freq := make(map[string]int)
	for _, text := range texts {
		for _, tok := range tokenize(text) {
			freq[tok]++
		}
	}

	// Sort by frequency descending
	type wf struct {
		word string
		freq int
	}
	var pairs []wf
	for w, f := range freq {
		if _, special := t.Word2ID[w]; !special {
			pairs = append(pairs, wf{w, f})
		}
	}
	sort.Slice(pairs, func(i, j int) bool { return pairs[i].freq > pairs[j].freq })

	// Keep top maxVocab - len(specials)
	limit := maxVocab - len(t.ID2Word)
	if limit > len(pairs) {
		limit = len(pairs)
	}

	for i := 0; i < limit; i++ {
		id := len(t.ID2Word)
		t.Word2ID[pairs[i].word] = id
		t.ID2Word = append(t.ID2Word, pairs[i].word)
	}
}

// VocabSize returns the total vocabulary size.
func (t *Tokenizer) VocabSize() int { return len(t.ID2Word) }

// Encode converts text to a sequence of token IDs.
func (t *Tokenizer) Encode(text string) []int {
	tokens := tokenize(text)
	ids := make([]int, len(tokens))
	for i, tok := range tokens {
		if id, ok := t.Word2ID[tok]; ok {
			ids[i] = id
		} else {
			ids[i] = UnkID
		}
	}
	return ids
}

// Decode converts token IDs back to text.
func (t *Tokenizer) Decode(ids []int) string {
	var words []string
	for _, id := range ids {
		if id == PadID || id == BosID || id == EosID {
			continue
		}
		if id >= 0 && id < len(t.ID2Word) {
			words = append(words, t.ID2Word[id])
		}
	}
	return detokenize(words)
}

// EncodeTriple encodes a (subject, relation, object) triple as a token sequence:
// <bos> subject_tokens <sep> relation_tokens <sep> object_tokens <eos>
func (t *Tokenizer) EncodeTriple(subject, relation, object string) []int {
	var ids []int
	ids = append(ids, BosID)
	ids = append(ids, t.Encode(subject)...)
	ids = append(ids, SepID)
	ids = append(ids, t.Encode(relation)...)
	ids = append(ids, SepID)
	ids = append(ids, t.Encode(object)...)
	ids = append(ids, EosID)
	return ids
}

// tokenize splits text into lowercase word and punctuation tokens.
func tokenize(text string) []string {
	text = strings.ToLower(strings.TrimSpace(text))
	var tokens []string
	var current strings.Builder

	for _, r := range text {
		if unicode.IsLetter(r) || unicode.IsDigit(r) || r == '\'' || r == '-' {
			current.WriteRune(r)
		} else if unicode.IsPunct(r) || r == '(' || r == ')' {
			if current.Len() > 0 {
				tokens = append(tokens, current.String())
				current.Reset()
			}
			tokens = append(tokens, string(r))
		} else if unicode.IsSpace(r) {
			if current.Len() > 0 {
				tokens = append(tokens, current.String())
				current.Reset()
			}
		}
	}
	if current.Len() > 0 {
		tokens = append(tokens, current.String())
	}
	return tokens
}

// detokenize joins tokens back into readable text.
func detokenize(tokens []string) string {
	if len(tokens) == 0 {
		return ""
	}
	var b strings.Builder
	for i, tok := range tokens {
		if i > 0 && !isPunct(tok) && !isPunct(tokens[i-1]) {
			b.WriteByte(' ')
		}
		if i == 0 {
			// Capitalize first token
			b.WriteString(strings.ToUpper(tok[:1]) + tok[1:])
		} else {
			b.WriteString(tok)
		}
	}
	return b.String()
}

func isPunct(s string) bool {
	if len(s) != 1 {
		return false
	}
	return unicode.IsPunct(rune(s[0])) || s == "(" || s == ")"
}
