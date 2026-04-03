package cognitive

import (
	"strings"
	"testing"
)

func TestPersonaEngine_Answer(t *testing.T) {
	graph := newTestGraph()
	pe := NewPersonaEngine(graph)

	// Physicist answering about gravity.
	answer := pe.Answer("What is gravity?", "physicist")
	if answer == nil {
		t.Fatal("expected non-nil answer")
	}
	if answer.Persona != "physicist" {
		t.Fatalf("expected physicist persona, got %q", answer.Persona)
	}
	if answer.Response == "" {
		t.Fatal("expected non-empty response")
	}
	if !strings.Contains(strings.ToLower(answer.Response), "physicist") {
		t.Fatalf("expected response to mention physicist perspective: %q", answer.Response)
	}
}

func TestPersonaEngine_UnknownPersona(t *testing.T) {
	graph := newTestGraph()
	pe := NewPersonaEngine(graph)

	answer := pe.Answer("test query", "astrologer")
	if answer == nil {
		t.Fatal("expected non-nil answer")
	}
	if !strings.Contains(answer.Response, "Unknown persona") {
		t.Fatalf("expected unknown persona message, got: %q", answer.Response)
	}
}

func TestPersonaEngine_ListPersonas(t *testing.T) {
	graph := newTestGraph()
	pe := NewPersonaEngine(graph)

	personas := pe.ListPersonas()
	if len(personas) < 5 {
		t.Fatalf("expected at least 5 personas, got %d", len(personas))
	}

	// Check specific personas exist.
	personaSet := make(map[string]bool)
	for _, p := range personas {
		personaSet[p] = true
	}
	for _, expected := range []string{"physicist", "historian", "economist", "philosopher", "biologist"} {
		if !personaSet[expected] {
			t.Errorf("expected persona %q in list", expected)
		}
	}
}

func TestPersonaEngine_RegisterCustom(t *testing.T) {
	graph := newTestGraph()
	pe := NewPersonaEngine(graph)

	pe.RegisterPersona(&ExpertPersona{
		Name:        "astronomer",
		DisplayName: "Astronomer",
		Description: "Space expert",
		Domains:     []string{"astronomy", "star", "planet", "galaxy"},
		Relations:   []RelType{RelIsA, RelRelatedTo},
		FrameVerbs:  []string{"can be observed in the cosmos"},
	})

	answer := pe.Answer("What is a star?", "astronomer")
	if answer == nil {
		t.Fatal("expected non-nil answer")
	}
	if answer.Persona != "astronomer" {
		t.Fatalf("expected astronomer persona, got %q", answer.Persona)
	}
}

func TestPersonaEngine_NilGraph(t *testing.T) {
	pe := NewPersonaEngine(nil)
	answer := pe.Answer("test", "physicist")
	if answer == nil {
		t.Fatal("expected non-nil answer")
	}
	if answer.Confidence != 0 {
		t.Fatalf("expected zero confidence with nil graph, got %f", answer.Confidence)
	}
}

func TestIsExpertPersonaQuery(t *testing.T) {
	tests := []struct {
		input    string
		isPersona bool
		name     string
	}{
		{"As a physicist, explain gravity", true, "physicist"},
		{"Ask the historian about Rome", true, "historian"},
		{"From a philosopher's perspective on ethics", true, "philosopher"},
		{"What is gravity?", false, ""},
		{"Hello", false, ""},
		{"As a friend, how are you?", false, ""}, // "friend" not in persona list
	}

	for _, tt := range tests {
		got, name := IsExpertPersonaQuery(tt.input)
		if got != tt.isPersona {
			t.Errorf("IsExpertPersonaQuery(%q) = %v, want %v", tt.input, got, tt.isPersona)
		}
		if got && name != tt.name {
			t.Errorf("IsExpertPersonaQuery(%q) name = %q, want %q", tt.input, name, tt.name)
		}
	}
}

func TestExtractExpertTopic(t *testing.T) {
	tests := []struct {
		input    string
		expected string
	}{
		{"As a physicist, what is gravity?", "gravity"},
		{"explain dark matter", "dark matter"},
		{"tell me about evolution", "evolution"},
		{"quantum mechanics", "quantum mechanics"},
	}

	for _, tt := range tests {
		got := extractExpertTopic(tt.input)
		if got != tt.expected {
			t.Errorf("extractExpertTopic(%q) = %q, want %q", tt.input, got, tt.expected)
		}
	}
}
