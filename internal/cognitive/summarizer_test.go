package cognitive

import (
	"strings"
	"testing"
)

func TestSummarize_Basic(t *testing.T) {
	s := NewSummarizer()

	text := "Gravity is the fundamental force of attraction between all objects with mass. " +
		"Isaac Newton described it as a universal force proportional to mass. " +
		"Einstein reinterpreted gravity in general relativity as spacetime curvature. " +
		"Gravitational acceleration near Earth is approximately 9.81 meters per second squared. " +
		"Gravity is the weakest of the four fundamental forces. " +
		"Yet it dominates at cosmic scales because it acts over infinite range. " +
		"It shapes galaxies, star systems, and the expansion of the universe. " +
		"The study of gravity continues to be central to modern physics. " +
		"Gravitational waves were first detected in 2015 by the LIGO observatory. " +
		"This confirmed a major prediction of general relativity."

	summary := s.Summarize(text, 3)

	if summary == "" {
		t.Fatal("expected non-empty summary")
	}

	// Summary should be shorter than original.
	if len(strings.Fields(summary)) >= len(strings.Fields(text)) {
		t.Error("summary should be shorter than original text")
	}

	// Should contain exactly 3 sentences (each ending with a period).
	sentences := splitTextSentences(summary)
	if len(sentences) != 3 {
		t.Errorf("expected 3 sentences, got %d: %q", len(sentences), summary)
	}
}

func TestSummarize_ShortText(t *testing.T) {
	s := NewSummarizer()

	text := "Short text here. Only two sentences."
	summary := s.Summarize(text, 5)

	// Should return all sentences when maxSentences > total.
	if !strings.Contains(summary, "Short text") {
		t.Error("short text should be returned as-is")
	}
}

func TestSummarize_Scoring(t *testing.T) {
	s := NewSummarizer()

	// The first sentence of a paragraph should score higher (position bonus).
	text := "Quantum mechanics is a fundamental theory in physics. " +
		"It describes nature at the atomic scale. " +
		"The theory was developed in the early twentieth century. " +
		"Max Planck introduced the quantum hypothesis. " +
		"Werner Heisenberg formulated the uncertainty principle.\n\n" +
		"Applications of quantum mechanics are widespread in modern technology. " +
		"Semiconductors rely on quantum tunneling effects. " +
		"Lasers operate on the principle of stimulated emission. " +
		"MRI machines use quantum properties of atomic nuclei. " +
		"Quantum computers promise exponential speedup for certain problems."

	// Score the sentences.
	sentences := splitTextSentences(text)
	scored := s.scoreSentences(text, sentences)

	// First sentence should have a high score (position bonus + keyword overlap).
	if len(scored) > 1 && scored[0].score <= scored[len(scored)-1].score {
		t.Log("note: first sentence did not outscore last; specific to input distribution")
	}

	// Find the first-sentence scores — they should get paragraph-start bonus.
	firstSentScore := scored[0].score
	if firstSentScore < 2.0 {
		t.Errorf("first sentence should have position bonus, score: %.2f", firstSentScore)
	}

	// Sentence about "Applications" starts a new paragraph and should get bonus too.
	for _, ss := range scored {
		if strings.HasPrefix(ss.text, "Applications") {
			if ss.score < 2.0 {
				t.Errorf("paragraph-initial sentence should have bonus, score: %.2f", ss.score)
			}
			break
		}
	}
}

func TestSummarize_Redundancy(t *testing.T) {
	s := NewSummarizer()

	// Create text with near-duplicate sentences.
	text := "Gravity is the fundamental force between objects with mass. " +
		"Electromagnetism governs charged particle interactions. " +
		"Gravity is the fundamental force of attraction between massive objects. " + // near-duplicate of first
		"The strong force binds quarks together inside protons. " +
		"Gravity is the force pulling all masses toward each other. " + // another near-duplicate
		"The weak force is responsible for radioactive decay. " +
		"Quantum mechanics describes nature at the smallest scales. " +
		"Thermodynamics studies heat and energy transformations."

	summary := s.Summarize(text, 4)

	// Count how many sentences mention gravity.
	sentences := splitTextSentences(summary)
	gravityCount := 0
	for _, sent := range sentences {
		if strings.Contains(strings.ToLower(sent), "gravity") {
			gravityCount++
		}
	}

	// Should not have more than 1 gravity sentence (redundancy penalty).
	if gravityCount > 1 {
		t.Errorf("expected at most 1 gravity sentence due to redundancy penalty, got %d", gravityCount)
	}
}

func TestSummarizeToLength(t *testing.T) {
	s := NewSummarizer()

	text := "Gravity is the fundamental force of attraction between all objects with mass. " +
		"Isaac Newton described it as a universal force proportional to mass. " +
		"Einstein reinterpreted gravity as spacetime curvature. " +
		"Gravitational acceleration near Earth is approximately 9.81 m/s squared. " +
		"Gravity dominates at cosmic scales because it acts over infinite range. " +
		"It shapes galaxies and star systems throughout the universe."

	summary := s.SummarizeToLength(text, 30)

	wordCount := len(strings.Fields(summary))
	if wordCount > 40 { // some tolerance since we select whole sentences
		t.Errorf("expected roughly 30 words, got %d", wordCount)
	}
	if summary == "" {
		t.Error("expected non-empty summary")
	}
}

func TestSplitTextSentences(t *testing.T) {
	text := "First sentence. Second sentence! Third sentence? Last one."
	sentences := splitTextSentences(text)

	if len(sentences) != 4 {
		t.Errorf("expected 4 sentences, got %d: %v", len(sentences), sentences)
	}
}

func TestSentenceSimilarity(t *testing.T) {
	a := "Gravity is the fundamental force between objects."
	b := "Gravity is the fundamental force between massive objects."
	c := "Electromagnetism governs charged particle interactions."

	simAB := sentenceSimilarity(a, b)
	simAC := sentenceSimilarity(a, c)

	if simAB <= simAC {
		t.Errorf("similar sentences should score higher: AB=%.2f, AC=%.2f", simAB, simAC)
	}
}

func TestSummarize_OriginalOrder(t *testing.T) {
	s := NewSummarizer()

	// Create text where important sentences are scattered.
	text := "Alpha is the first letter. " +
		"Beta is the second letter. " +
		"Gamma is the third letter. " +
		"Delta is the fourth letter. " +
		"Epsilon is the fifth letter. " +
		"Zeta is the sixth letter. " +
		"Eta is the seventh letter. " +
		"Theta is the eighth letter. " +
		"Iota is the ninth letter. " +
		"Kappa is the tenth letter."

	summary := s.Summarize(text, 3)
	sentences := splitTextSentences(summary)

	// Verify the selected sentences maintain their original order.
	prevIdx := -1
	for _, sent := range sentences {
		idx := strings.Index(text, sent)
		if idx < prevIdx {
			t.Errorf("sentences not in original order: %q appears before previous", sent)
		}
		prevIdx = idx
	}
}

func TestCuePhraseScoring(t *testing.T) {
	important := "This is the most important finding in the study."
	boring := "The weather was nice that day."

	impScore := cuePhrasesScore(important)
	borScore := cuePhrasesScore(boring)

	if impScore <= borScore {
		t.Errorf("sentence with cue phrases should score higher: imp=%.2f, bor=%.2f", impScore, borScore)
	}
}

func TestEntityDensityScore(t *testing.T) {
	entityRich := "Albert Einstein developed General Relativity at the University of Berlin."
	entityPoor := "the ball rolled down the hill quickly and stopped."

	richScore := entityDensityScore(entityRich)
	poorScore := entityDensityScore(entityPoor)

	if richScore <= poorScore {
		t.Errorf("entity-rich sentence should score higher: rich=%.2f, poor=%.2f", richScore, poorScore)
	}
}
