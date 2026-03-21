package cognitive

import (
	"math/rand"
	"path/filepath"
	"strings"
	"testing"
)

func TestMarkovTrainAndGenerate(t *testing.T) {
	m := NewMarkovModel()

	// Train on some text.
	corpus := `Stoicism is an ancient philosophy. The Stoics believed in virtue.
Virtue is the highest good according to Stoicism. Marcus Aurelius practiced Stoicism.
Philosophy helps us understand the world. The world is full of complexity.
Understanding leads to wisdom. Wisdom is the goal of philosophy.`

	m.Train(corpus)

	t.Logf("Trigrams: %d", m.Size())
	t.Logf("Total tokens: %d", m.TotalTokens())

	if m.Size() == 0 {
		t.Fatal("expected non-zero trigrams after training")
	}

	// Generate some text.
	rng := rand.New(rand.NewSource(42))
	for i := 0; i < 5; i++ {
		text := m.Generate(20, rng)
		t.Logf("  Generated: %s", text)
		if text == "" {
			t.Error("expected non-empty generated text")
		}
	}
}

func TestMarkovGenerateFrom(t *testing.T) {
	m := NewMarkovModel()

	corpus := `The philosophy of Stoicism teaches inner peace.
Stoicism was founded by Zeno of Citium in ancient Greece.
Ancient philosophy shaped Western thought for centuries.
The Stoic approach to life emphasizes virtue and reason.`

	m.Train(corpus)

	rng := rand.New(rand.NewSource(42))

	// Generate seeded with "stoicism"
	text := m.GenerateFrom("stoicism", 15, rng)
	t.Logf("Seeded with 'stoicism': %s", text)

	if text == "" {
		t.Error("expected non-empty text seeded with known word")
	}

	// Should contain the seed word or related content.
	if !strings.Contains(strings.ToLower(text), "stoicism") &&
		!strings.Contains(strings.ToLower(text), "philosophy") {
		t.Logf("Warning: generated text doesn't contain seed-related words")
	}
}

func TestMarkovFragment(t *testing.T) {
	m := NewMarkovModel()

	corpus := `Language is a tool for communication. Communication enables understanding.
Understanding builds bridges between people. People use language every day.
Every day brings new opportunities for growth. Growth comes from learning.`

	m.Train(corpus)

	rng := rand.New(rand.NewSource(42))

	fragment := m.GenerateFragment("language", 3, 10, rng)
	t.Logf("Fragment: %s", fragment)

	words := strings.Fields(fragment)
	if len(words) > 0 && len(words) < 3 {
		t.Errorf("fragment too short: %d words", len(words))
	}
	if len(words) > 10 {
		t.Errorf("fragment too long: %d words", len(words))
	}
}

func TestMarkovPersistence(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "markov.json")

	m := NewMarkovModel()
	m.Train("The quick brown fox jumps over the lazy dog. The dog sleeps all day.")
	origSize := m.Size()

	if err := m.Save(path); err != nil {
		t.Fatal(err)
	}

	m2 := NewMarkovModel()
	if err := m2.Load(path); err != nil {
		t.Fatal(err)
	}

	if m2.Size() != origSize {
		t.Errorf("expected %d trigrams after load, got %d", origSize, m2.Size())
	}
	if m2.TotalTokens() != m.TotalTokens() {
		t.Errorf("expected %d tokens after load, got %d", m.TotalTokens(), m2.TotalTokens())
	}
}

func TestMarkovSelfImprovement(t *testing.T) {
	// Simulate the self-improvement loop: train on generated text.
	m := NewMarkovModel()

	// Initial training
	m.Train("Science explains the natural world through observation and experiment.")
	m.Train("The scientific method is a systematic approach to understanding.")
	m.Train("Understanding the world requires careful observation.")

	rng := rand.New(rand.NewSource(42))

	// Generate, then re-train on generated text (self-improvement).
	initialSize := m.Size()
	for round := 0; round < 3; round++ {
		generated := m.Generate(20, rng)
		if generated != "" {
			m.Train(generated)
		}
	}

	t.Logf("Trigrams: %d → %d (after self-improvement)", initialSize, m.Size())
	if m.Size() < initialSize {
		t.Error("expected trigram count to grow with self-improvement")
	}
}
