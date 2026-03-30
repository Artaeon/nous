package cognitive

import (
	"strings"
	"testing"
)

// -----------------------------------------------------------------------
// Format Detection Tests
// -----------------------------------------------------------------------

func TestDetectFormat_Bullets(t *testing.T) {
	fc := NewFormatCompliance()

	tests := []struct {
		query string
		count int
	}{
		{"summarize stoicism in 3 bullet points", 3},
		{"explain gravity in 5 bullets", 5},
		{"as 4 bullet points, describe DNA", 4},
		{"give me 2 bullets on machine learning", 2},
	}

	for _, tt := range tests {
		req := fc.DetectFormat(tt.query)
		if req == nil {
			t.Errorf("DetectFormat(%q) = nil, want bullets with count=%d", tt.query, tt.count)
			continue
		}
		if req.Type != "bullets" {
			t.Errorf("DetectFormat(%q).Type = %q, want %q", tt.query, req.Type, "bullets")
		}
		if req.Count != tt.count {
			t.Errorf("DetectFormat(%q).Count = %d, want %d", tt.query, req.Count, tt.count)
		}
	}
}

func TestDetectFormat_Numbered(t *testing.T) {
	fc := NewFormatCompliance()

	tests := []struct {
		query string
		count int
	}{
		{"give me 5 key points about gravity", 5},
		{"list 3 points on evolution", 3},
		{"explain this step by step", 0},
		{"give me numbered steps for cooking pasta", 0},
	}

	for _, tt := range tests {
		req := fc.DetectFormat(tt.query)
		if req == nil {
			t.Errorf("DetectFormat(%q) = nil, want numbered", tt.query)
			continue
		}
		if req.Type != "numbered" {
			t.Errorf("DetectFormat(%q).Type = %q, want %q", tt.query, req.Type, "numbered")
		}
		if req.Count != tt.count {
			t.Errorf("DetectFormat(%q).Count = %d, want %d", tt.query, req.Count, tt.count)
		}
	}
}

func TestDetectFormat_OneSentence(t *testing.T) {
	fc := NewFormatCompliance()

	tests := []string{
		"explain quantum physics in one sentence",
		"describe DNA in a single sentence",
		"briefly explain photosynthesis",
	}

	for _, query := range tests {
		req := fc.DetectFormat(query)
		if req == nil {
			t.Errorf("DetectFormat(%q) = nil, want one_sentence", query)
			continue
		}
		if req.Type != "one_sentence" {
			t.Errorf("DetectFormat(%q).Type = %q, want %q", query, req.Type, "one_sentence")
		}
	}
}

func TestDetectFormat_MaxWords(t *testing.T) {
	fc := NewFormatCompliance()

	tests := []struct {
		query    string
		maxWords int
	}{
		{"explain gravity under 50 words", 50},
		{"describe DNA in 30 words or less", 30},
		{"tell me about evolution, keep it to 100 words", 100},
	}

	for _, tt := range tests {
		req := fc.DetectFormat(tt.query)
		if req == nil {
			t.Errorf("DetectFormat(%q) = nil, want maxWords=%d", tt.query, tt.maxWords)
			continue
		}
		if req.MaxWords != tt.maxWords {
			t.Errorf("DetectFormat(%q).MaxWords = %d, want %d", tt.query, req.MaxWords, tt.maxWords)
		}
	}
}

func TestDetectFormat_None(t *testing.T) {
	fc := NewFormatCompliance()

	queries := []string{
		"what is quantum physics",
		"tell me about the Roman Empire",
		"how does photosynthesis work",
		"explain machine learning",
	}

	for _, query := range queries {
		req := fc.DetectFormat(query)
		if req != nil {
			t.Errorf("DetectFormat(%q) = %+v, want nil", query, req)
		}
	}
}

// -----------------------------------------------------------------------
// Reshape Tests
// -----------------------------------------------------------------------

var testProse = "Stoicism is an ancient Greek philosophy founded in Athens. " +
	"It teaches the development of self-control and fortitude. " +
	"Marcus Aurelius was one of the most famous Stoic philosophers. " +
	"The Stoics believed that virtue is the highest good. " +
	"Epictetus taught that we should focus only on what we can control."

func TestReshape_Bullets(t *testing.T) {
	fc := NewFormatCompliance()
	req := &FormatRequest{Type: "bullets", Count: 3}
	result := fc.Reshape(testProse, req)

	lines := strings.Split(result, "\n")
	if len(lines) != 3 {
		t.Errorf("expected 3 bullet lines, got %d: %q", len(lines), result)
	}
	for i, line := range lines {
		if !strings.HasPrefix(line, "- ") {
			t.Errorf("line %d does not start with '- ': %q", i, line)
		}
	}
	t.Logf("Bullets:\n%s", result)
}

func TestReshape_Numbered(t *testing.T) {
	fc := NewFormatCompliance()
	req := &FormatRequest{Type: "numbered", Count: 3}
	result := fc.Reshape(testProse, req)

	lines := strings.Split(result, "\n")
	if len(lines) != 3 {
		t.Errorf("expected 3 numbered lines, got %d: %q", len(lines), result)
	}
	for i, line := range lines {
		expected := strings.Repeat("", 0) // just to use i
		_ = expected
		prefix := string(rune('1'+i)) + ". "
		if !strings.HasPrefix(line, prefix) {
			t.Errorf("line %d does not start with %q: %q", i, prefix, line)
		}
	}
	t.Logf("Numbered:\n%s", result)
}

func TestReshape_OneSentence(t *testing.T) {
	fc := NewFormatCompliance()
	req := &FormatRequest{Type: "one_sentence"}
	result := fc.Reshape(testProse, req)

	// Should be a single sentence (no period-space-uppercase in the middle).
	sentences := SplitIntoSentences(result)
	if len(sentences) != 1 {
		t.Errorf("expected 1 sentence, got %d: %q", len(sentences), result)
	}
	if result == "" {
		t.Error("result is empty")
	}
	t.Logf("One sentence: %s", result)
}

func TestReshape_MaxWords(t *testing.T) {
	fc := NewFormatCompliance()
	req := &FormatRequest{MaxWords: 15}
	result := fc.Reshape(testProse, req)

	words := strings.Fields(result)
	// Allow for "..." at the end.
	cleanResult := strings.TrimSuffix(result, "...")
	cleanWords := strings.Fields(cleanResult)

	if len(cleanWords) > 15 {
		t.Errorf("expected <= 15 words, got %d: %q", len(words), result)
	}
	if result == "" {
		t.Error("result is empty")
	}
	t.Logf("MaxWords (15): %s", result)
}

// -----------------------------------------------------------------------
// Sentence Scoring Test
// -----------------------------------------------------------------------

func TestPickBestSentences(t *testing.T) {
	sentences := []string{
		"However, this is a transition.",
		"Quantum mechanics was developed in the early 20th century by physicists including Max Planck and Niels Bohr.",
		"It is interesting.",
		"The theory describes the behavior of matter and energy at the atomic and subatomic levels.",
		"Ok.",
	}

	best := PickBestSentences(sentences, 2)
	if len(best) != 2 {
		t.Fatalf("expected 2 sentences, got %d", len(best))
	}

	// The two long, entity-rich sentences should be picked.
	for _, s := range best {
		if s == "Ok." || s == "It is interesting." || strings.HasPrefix(s, "However") {
			t.Errorf("picked low-quality sentence: %q", s)
		}
	}

	// Should preserve original order: the Quantum sentence comes first.
	if !strings.HasPrefix(best[0], "Quantum") {
		t.Errorf("expected first pick to start with 'Quantum', got %q", best[0])
	}

	t.Logf("Best 2: %v", best)
}

// -----------------------------------------------------------------------
// Noun Phrase Extraction Tests
// -----------------------------------------------------------------------

func TestExtractNounPhrase(t *testing.T) {
	tests := []struct {
		query    string
		expected string
	}{
		{"give me an overview of operating systems", "operating systems"},
		{"explain how photosynthesis works", "photosynthesis works"},
		{"what is quantum physics", "quantum physics"},
		{"compare Python and Go for web development", "Python and Go"},
		{"tell me about the Roman Empire", "Roman Empire"},
		{"how does machine learning work", "machine learning work"},
		{"what are the benefits of meditation", "meditation"},
		{"who was Albert Einstein", "Albert Einstein"},
	}

	for _, tt := range tests {
		result := ExtractNounPhrase(tt.query)
		if result != tt.expected {
			t.Errorf("ExtractNounPhrase(%q) = %q, want %q", tt.query, result, tt.expected)
		}
	}
}

func TestExtractNounPhrase_Simple(t *testing.T) {
	tests := []struct {
		query    string
		expected string
	}{
		{"DNA", "DNA"},
		{"Socrates", "Socrates"},
	}

	for _, tt := range tests {
		result := ExtractNounPhrase(tt.query)
		if result != tt.expected {
			t.Errorf("ExtractNounPhrase(%q) = %q, want %q", tt.query, result, tt.expected)
		}
	}
}
