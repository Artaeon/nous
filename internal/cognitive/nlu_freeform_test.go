package cognitive

import (
	"testing"
)

func TestFreeformClassifier_KnowledgeQueries(t *testing.T) {
	emb := NewWordEmbeddings(64)
	emb.SeedPoolWords()
	fc := NewFreeformClassifier(emb)

	tests := []struct {
		input string
		topic string // expected substring in extracted topic
	}{
		{"Explain quantum entanglement like I'm a pirate", "quantum entanglement"},
		{"what's the deal with gravity?", "gravity"},
		{"can you break down how photosynthesis works?", "photosynthesis"},
		{"ELI5 the French Revolution", "french revolution"},
		{"I'm curious about black holes", "black holes"},
		{"yo what do you know about dogs", "dogs"},
		{"give me the lowdown on jazz", "jazz"},
		{"how does DNA replication work exactly?", "dna replication"},
		{"who even is Nikola Tesla?", "nikola tesla"},
		{"what makes the ocean salty?", "ocean salty"},
		{"so like what's philosophy even about?", "philosophy"},
		{"break it down: machine learning", "machine learning"},
		{"school me on ancient Rome", "ancient rome"},
		{"spill the tea on quantum computing", "quantum computing"},
		{"run me through how vaccines work", "vaccines"},
	}

	for _, tt := range tests {
		t.Run(tt.input, func(t *testing.T) {
			result := fc.Classify(tt.input)
			if result == nil {
				t.Fatalf("Classify(%q) returned nil, expected lookup_knowledge", tt.input)
			}
			if result.Intent != "explain" {
				t.Errorf("Classify(%q).Intent = %q, want lookup_knowledge", tt.input, result.Intent)
			}
			if result.Confidence < 0.40 {
				t.Errorf("Classify(%q).Confidence = %.2f, want >= 0.40", tt.input, result.Confidence)
			}
			topic, ok := result.Entities["topic"]
			if !ok || topic == "" {
				t.Errorf("Classify(%q) missing topic entity", tt.input)
			} else {
				lowerTopic := toLower(topic)
				lowerWant := toLower(tt.topic)
				if !containsStr(lowerTopic, lowerWant) {
					t.Errorf("Classify(%q) topic = %q, want it to contain %q", tt.input, topic, tt.topic)
				}
			}
		})
	}
}

func TestFreeformClassifier_StyleExtraction(t *testing.T) {
	emb := NewWordEmbeddings(64)
	emb.SeedPoolWords()
	fc := NewFreeformClassifier(emb)

	tests := []struct {
		input string
		style string
	}{
		{"Explain quantum entanglement like I'm a pirate", "pirate"},
		{"ELI5 the French Revolution", "for a 5 year old"},
		{"explain gravity in simple terms", "simple terms"},
		{"describe relativity in detail", "in detail"},
		{"what is calculus for a 5 year old", "5 year old"},
	}

	for _, tt := range tests {
		t.Run(tt.input, func(t *testing.T) {
			style := fc.ExtractStyle(tt.input)
			if !containsStr(toLower(style), toLower(tt.style)) {
				t.Errorf("ExtractStyle(%q) = %q, want it to contain %q", tt.input, style, tt.style)
			}
		})
	}
}

func TestFreeformClassifier_CreativeIntent(t *testing.T) {
	emb := NewWordEmbeddings(64)
	emb.SeedPoolWords()
	fc := NewFreeformClassifier(emb)

	tests := []string{
		"write me a poem about the moon",
		"create a story about a dragon",
		"compose a haiku",
		"generate a joke",
		"craft a limerick about cats",
	}

	for _, input := range tests {
		t.Run(input, func(t *testing.T) {
			result := fc.Classify(input)
			if result == nil {
				t.Fatalf("Classify(%q) returned nil", input)
			}
			if result.Intent != "creative" {
				t.Errorf("Classify(%q).Intent = %q, want creative", input, result.Intent)
			}
		})
	}
}

func TestFreeformClassifier_TopicExtraction(t *testing.T) {
	emb := NewWordEmbeddings(64)
	emb.SeedPoolWords()
	fc := NewFreeformClassifier(emb)

	tests := []struct {
		input string
		want  string
	}{
		{"what's the deal with gravity?", "gravity"},
		{"ELI5 the French Revolution", "french revolution"},
		{"can you break down how photosynthesis works?", "photosynthesis"},
		{"I'm curious about black holes", "black holes"},
		{"break it down: machine learning", "machine learning"},
		{"school me on ancient Rome", "ancient rome"},
		{"spill the tea on quantum computing", "quantum computing"},
	}

	for _, tt := range tests {
		t.Run(tt.input, func(t *testing.T) {
			got := toLower(fc.ExtractTopic(tt.input))
			if !containsStr(got, toLower(tt.want)) {
				t.Errorf("ExtractTopic(%q) = %q, want it to contain %q", tt.input, got, tt.want)
			}
		})
	}
}

func TestFreeformClassifier_NilEmbeddings(t *testing.T) {
	fc := NewFreeformClassifier(nil)

	// Should still work via heuristic rules, just skip similarity.
	result := fc.Classify("what's the deal with gravity?")
	if result == nil {
		t.Fatal("Classify returned nil with nil embeddings")
	}
	if result.Intent != "explain" {
		t.Errorf("Intent = %q, want lookup_knowledge", result.Intent)
	}
}

func TestFreeformClassifier_IntegrationWithNLU(t *testing.T) {
	nlu := NewNLU()

	// These inputs should now route through the freeform fallback
	// and get classified as question/lookup_knowledge rather than
	// falling through to a low-confidence generic "question".
	tests := []struct {
		input     string
		wantTopic string
	}{
		{"Explain quantum entanglement like I'm a pirate", "quantum entanglement"},
		{"what's the deal with gravity?", "gravity"},
		{"ELI5 the French Revolution", "french revolution"},
		{"I'm curious about black holes", "black holes"},
		{"school me on ancient Rome", "ancient rome"},
		{"spill the tea on quantum computing", "quantum computing"},
	}

	for _, tt := range tests {
		t.Run(tt.input, func(t *testing.T) {
			result := nlu.Understand(tt.input)
			if result == nil {
				t.Fatal("Understand returned nil")
			}
			// Should have a reasonable intent — not unknown.
			if result.Intent == "unknown" || result.Intent == "" {
				t.Errorf("Understand(%q).Intent = %q, want a real intent", tt.input, result.Intent)
			}
			if result.Confidence < 0.35 {
				t.Errorf("Understand(%q).Confidence = %.2f, too low", tt.input, result.Confidence)
			}
		})
	}
}

// helpers

func toLower(s string) string {
	return stringToLower(s)
}

func containsStr(s, substr string) bool {
	return stringContains(s, substr)
}

// Wrappers to avoid importing strings in test (already in package).
func stringToLower(s string) string {
	out := make([]byte, len(s))
	for i := 0; i < len(s); i++ {
		c := s[i]
		if c >= 'A' && c <= 'Z' {
			c += 'a' - 'A'
		}
		out[i] = c
	}
	return string(out)
}

func stringContains(s, substr string) bool {
	return len(substr) == 0 || findSubstring(s, substr) >= 0
}

func findSubstring(s, sub string) int {
	if len(sub) > len(s) {
		return -1
	}
	for i := 0; i <= len(s)-len(sub); i++ {
		if s[i:i+len(sub)] == sub {
			return i
		}
	}
	return -1
}
