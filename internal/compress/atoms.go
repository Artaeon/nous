package compress

import (
	"fmt"
	"strings"

	"github.com/artaeon/nous/internal/ollama"
)

// Atom is a compressed unit of context — a dense summary of a past interaction
// that can be quickly injected into future prompts without carrying the full
// conversation history.
type Atom struct {
	Trigger  string  // What kind of input this atom is relevant to
	Content  string  // The compressed knowledge
	Weight   float64 // How important/reliable this atom is
}

// Compressor manages context atoms — compressing conversation history
// into dense, reusable knowledge fragments.
type Compressor struct {
	llm   *ollama.Client
	atoms []Atom
}

func NewCompressor(llm *ollama.Client) *Compressor {
	return &Compressor{
		llm: llm,
	}
}

// Compress takes a conversation fragment and distills it into an atom.
func (c *Compressor) Compress(userInput, assistantResponse string) (*Atom, error) {
	prompt := fmt.Sprintf(`Compress this interaction into a single dense knowledge statement.
The statement should capture the key insight that would be useful in future similar interactions.

User: %s
Assistant: %s

Respond with exactly:
TRIGGER: <what kind of input this is relevant to, 2-3 words>
KNOWLEDGE: <the compressed insight, one sentence>`, userInput, assistantResponse)

	resp, err := c.llm.Chat([]ollama.Message{
		{Role: "system", Content: "You are a knowledge compressor. Distill interactions into dense, reusable insights."},
		{Role: "user", Content: prompt},
	}, &ollama.ModelOptions{
		Temperature: 0.1,
		NumPredict:  100,
	})
	if err != nil {
		return nil, fmt.Errorf("compress: %w", err)
	}

	atom := parseAtom(resp.Message.Content)
	c.atoms = append(c.atoms, atom)
	return &atom, nil
}

// Relevant returns atoms that match the given input context.
// Uses simple keyword overlap — future versions will use embeddings.
func (c *Compressor) Relevant(input string, maxAtoms int) []Atom {
	if len(c.atoms) == 0 {
		return nil
	}

	type scored struct {
		atom  Atom
		score float64
	}

	inputWords := strings.Fields(strings.ToLower(input))
	var results []scored

	for _, a := range c.atoms {
		triggerWords := strings.Fields(strings.ToLower(a.Trigger))
		score := overlapScore(inputWords, triggerWords) * a.Weight
		if score > 0 {
			results = append(results, scored{atom: a, score: score})
		}
	}

	// Sort by score descending
	for i := 1; i < len(results); i++ {
		for j := i; j > 0 && results[j].score > results[j-1].score; j-- {
			results[j], results[j-1] = results[j-1], results[j]
		}
	}

	if maxAtoms > len(results) {
		maxAtoms = len(results)
	}

	out := make([]Atom, maxAtoms)
	for i := 0; i < maxAtoms; i++ {
		out[i] = results[i].atom
	}
	return out
}

// Count returns the number of stored atoms.
func (c *Compressor) Count() int {
	return len(c.atoms)
}

func parseAtom(response string) Atom {
	atom := Atom{Weight: 0.5}

	for _, line := range strings.Split(response, "\n") {
		line = strings.TrimSpace(line)
		upper := strings.ToUpper(line)

		if strings.HasPrefix(upper, "TRIGGER:") {
			atom.Trigger = strings.TrimSpace(line[8:])
		} else if strings.HasPrefix(upper, "KNOWLEDGE:") {
			atom.Content = strings.TrimSpace(line[10:])
		}
	}

	if atom.Content == "" {
		atom.Content = response
		atom.Trigger = "general"
	}

	return atom
}

func overlapScore(a, b []string) float64 {
	if len(b) == 0 {
		return 0
	}

	matches := 0
	for _, wa := range a {
		for _, wb := range b {
			if wa == wb {
				matches++
				break
			}
		}
	}

	return float64(matches) / float64(len(b))
}
