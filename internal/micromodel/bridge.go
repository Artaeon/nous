package micromodel

import (
	"strings"
)

// Generator is the interface that both MicroModel (transformer) and
// MambaModel implement for text generation from knowledge triples.
type Generator interface {
	Generate(subject, relation, object string, maxLen int, temperature float32) string
}

// Bridge connects the micro model to Nous's cognitive systems.
// Supports both transformer and Mamba backends transparently.
type Bridge struct {
	Model     *MicroModel // transformer backend (legacy)
	Mamba     *MambaModel // Mamba backend (preferred)
	generator Generator   // unified interface
}

// LoadBridge loads a trained model from disk and wraps it in a Bridge.
// Detects the model type from the file magic bytes automatically.
func LoadBridge(path string) (*Bridge, error) {
	// Try Mamba first (preferred)
	mamba, err := LoadMambaModel(path)
	if err == nil {
		return &Bridge{Mamba: mamba, generator: mamba}, nil
	}

	// Fall back to transformer
	m := &MicroModel{}
	if err := m.Load(path); err != nil {
		return nil, err
	}
	return &Bridge{Model: m, generator: m}, nil
}

// NewMambaBridge wraps a MambaModel in a Bridge.
func NewMambaBridge(m *MambaModel) *Bridge {
	return &Bridge{Mamba: m, generator: m}
}

// NewTransformerBridge wraps a MicroModel in a Bridge.
func NewTransformerBridge(m *MicroModel) *Bridge {
	return &Bridge{Model: m, generator: m}
}

// IsMamba returns true if the bridge is using the Mamba backend.
func (b *Bridge) IsMamba() bool {
	return b != nil && b.Mamba != nil
}

// GenerateSentence takes a knowledge triple and returns a fluent sentence.
func (b *Bridge) GenerateSentence(subject, relation, object string) string {
	if b == nil || b.generator == nil {
		return ""
	}
	return b.generator.Generate(subject, relation, object, 40, 0.7)
}

// GenerateConstrained generates a single sentence using knowledge-constrained
// beam search. Only available with the Mamba backend. Falls back to standard
// generation for transformer models.
func (b *Bridge) GenerateConstrained(subject, relation, object string, extraFacts [][3]string) string {
	if b == nil {
		return ""
	}
	// Use constrained generation with Mamba
	if b.Mamba != nil {
		return b.Mamba.GenerateGrounded(subject, relation, object, extraFacts, 3)
	}
	// Fallback: standard generation
	if b.generator != nil {
		return b.generator.Generate(subject, relation, object, 40, 0.7)
	}
	return ""
}

// GenerateParagraph takes multiple fact triples and returns a coherent paragraph.
// Uses constrained generation with Mamba, standard generation with transformer.
func (b *Bridge) GenerateParagraph(topic string, facts [][3]string) string {
	if b == nil || b.generator == nil || len(facts) == 0 {
		return ""
	}

	// Generate a sentence for each fact
	var sentences []string
	seen := make(map[string]bool)

	// Order: is_a first, then created_by/founded_in, then has/used_for, then rest
	ordered := orderFacts(facts)

	for _, fact := range ordered {
		var sent string
		if b.Mamba != nil {
			// Mamba: use constrained generation grounded in all facts
			sent = b.Mamba.GenerateGrounded(fact[0], fact[1], fact[2], ordered, 3)
		} else {
			sent = b.generator.Generate(fact[0], fact[1], fact[2], 40, 0.7)
		}
		sent = strings.TrimSpace(sent)
		if sent == "" || len(sent) < 10 {
			continue
		}
		// Ensure it ends with period
		if !strings.HasSuffix(sent, ".") && !strings.HasSuffix(sent, "!") && !strings.HasSuffix(sent, "?") {
			sent += "."
		}
		// Dedup
		lower := strings.ToLower(sent)
		if seen[lower] {
			continue
		}
		seen[lower] = true
		sentences = append(sentences, sent)
	}

	if len(sentences) == 0 {
		return ""
	}

	// Pronominalize: replace topic name with pronoun after first mention
	result := pronominalizeParagraph(topic, sentences)
	return result
}

// orderFacts sorts facts by relation type priority.
func orderFacts(facts [][3]string) [][3]string {
	priority := map[string]int{
		"is_a":          0,
		"described_as":  0,
		"known_for":     1,
		"created_by":    2,
		"founded_by":    2,
		"founded_in":    2,
		"located_in":    3,
		"has":           4,
		"offers":        4,
		"used_for":      5,
		"part_of":       6,
		"related_to":    7,
		"influenced_by": 7,
	}

	ordered := make([][3]string, len(facts))
	copy(ordered, facts)

	// Simple insertion sort by priority
	for i := 1; i < len(ordered); i++ {
		for j := i; j > 0; j-- {
			pa := factPriority(ordered[j][1], priority)
			pb := factPriority(ordered[j-1][1], priority)
			if pa < pb {
				ordered[j], ordered[j-1] = ordered[j-1], ordered[j]
			}
		}
	}

	return ordered
}

func factPriority(rel string, m map[string]int) int {
	if p, ok := m[rel]; ok {
		return p
	}
	return 8
}

// pronominalizeParagraph replaces the topic name with a pronoun
// in sentences after the first mention.
func pronominalizeParagraph(topic string, sentences []string) string {
	if len(sentences) == 0 {
		return ""
	}

	var result []string
	topicLower := strings.ToLower(topic)

	for i, sent := range sentences {
		if i >= 2 && strings.HasPrefix(strings.ToLower(sent), topicLower) {
			// Replace leading topic with "It"
			sent = "It" + sent[len(topic):]
		}
		result = append(result, sent)
	}

	return strings.Join(result, " ")
}
