package cognitive

import (
	"strings"
	"testing"
)

func TestFactCheckerNilSafe(t *testing.T) {
	var fc *FactChecker
	answer := fc.Check("hello world", "test")
	if answer != "hello world" {
		t.Fatalf("nil FactChecker should return answer unchanged, got %q", answer)
	}
}

func TestFactCheckerNoKnowledge(t *testing.T) {
	fc := &FactChecker{Knowledge: nil}
	answer := fc.Check("hello world", "test")
	if answer != "hello world" {
		t.Fatalf("nil knowledge should return answer unchanged, got %q", answer)
	}
}

func TestFactCheckerCatchesVenusError(t *testing.T) {
	fc := &FactChecker{Knowledge: &KnowledgeVec{}}
	answer := "Venus is hot because it has no atmosphere, meaning heat escapes quickly."
	checked := fc.Check(answer, "venus atmosphere")

	if !strings.Contains(checked, "Correction") {
		t.Fatalf("should catch Venus atmosphere error, got %q", checked)
	}
	if !strings.Contains(checked, "thick atmosphere") || !strings.Contains(checked, "carbon dioxide") {
		t.Fatalf("correction should mention thick CO2 atmosphere, got %q", checked)
	}
}

func TestFactCheckerPassesCorrectAnswer(t *testing.T) {
	fc := &FactChecker{Knowledge: &KnowledgeVec{}}
	answer := "Venus has the thickest atmosphere of any rocky planet, composed primarily of carbon dioxide."
	checked := fc.Check(answer, "venus")

	if checked != answer {
		t.Fatalf("correct answer should pass unchanged, got %q", checked)
	}
}

func TestFactCheckerCatchesMercuryHottest(t *testing.T) {
	fc := &FactChecker{Knowledge: &KnowledgeVec{}}
	answer := "Mercury is the hottest planet because it is closest to the Sun."
	checked := fc.Check(answer, "hottest planet")

	if !strings.Contains(checked, "Correction") {
		t.Fatalf("should catch Mercury-hottest error, got %q", checked)
	}
	if !strings.Contains(checked, "Venus") {
		t.Fatalf("correction should mention Venus, got %q", checked)
	}
}

func TestFactCheckerCatchesPlutoError(t *testing.T) {
	fc := &FactChecker{Knowledge: &KnowledgeVec{}}
	answer := "Pluto is a planet in our solar system."
	checked := fc.Check(answer, "pluto")

	if !strings.Contains(checked, "Correction") {
		t.Fatalf("should catch Pluto error, got %q", checked)
	}
	if !strings.Contains(checked, "dwarf planet") {
		t.Fatalf("correction should mention dwarf planet, got %q", checked)
	}
}

func TestKnownContradictionsHaveRequiredFields(t *testing.T) {
	for i, cp := range knownContradictions {
		if cp.wrong == "" {
			t.Errorf("contradiction %d has empty 'wrong' field", i)
		}
		if cp.right == "" {
			t.Errorf("contradiction %d has empty 'right' field", i)
		}
		if cp.context == "" {
			t.Errorf("contradiction %d has empty 'context' field", i)
		}
	}
}

func TestExtractRelevantSentence(t *testing.T) {
	text := "The Sun is a star. Venus has a thick atmosphere. Mars is red."
	got := extractRelevantSentence(text, "atmosphere")
	if !strings.Contains(got, "Venus") || !strings.Contains(got, "atmosphere") {
		t.Fatalf("expected sentence about Venus atmosphere, got %q", got)
	}
}

func TestFindNegationContradictions(t *testing.T) {
	answer := "venus has no atmosphere so heat escapes quickly"
	knowledge := "venus has a thick atmosphere composed of carbon dioxide with surface pressure 90 times earth"

	corrections := findNegationContradictions(answer, knowledge)
	if len(corrections) == 0 {
		t.Fatal("should detect negation contradiction about atmosphere")
	}
}
