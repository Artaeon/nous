package cognitive

import (
	"strings"
	"testing"
)

func TestPatternCount(t *testing.T) {
	g := NewGenerativeEngine()
	count := len(g.patterns)
	t.Logf("Total pattern count: %d", count)
	if count < 200 {
		t.Errorf("expected at least 200 patterns, got %d", count)
	}
}

func TestSyntacticPatterns(t *testing.T) {
	g := NewGenerativeEngine()
	subj, verb, obj := "Stoicism", "represent", "a school of philosophy"

	syntacticPrefixes := []string{
		"compound-", "subordinate-", "participial-", "appositive-",
		"inverted-",
	}

	for _, p := range g.patterns {
		isSyntactic := false
		for _, prefix := range syntacticPrefixes {
			if strings.HasPrefix(p.Name, prefix) {
				isSyntactic = true
				break
			}
		}
		if !isSyntactic {
			continue
		}

		t.Run(p.Name, func(t *testing.T) {
			result := p.Build(g, subj, verb, obj, TensePresent)
			if result == "" {
				t.Errorf("pattern %q produced empty output", p.Name)
			}
			if len(result) < 10 {
				t.Errorf("pattern %q produced suspiciously short output: %q", p.Name, result)
			}
			t.Logf("%s: %s", p.Name, result)
		})
	}
}

func TestDomainPatterns(t *testing.T) {
	g := NewGenerativeEngine()

	tests := []struct {
		domain string
		prefix string
		subj   string
		verb   string
		obj    string
	}{
		{"person", "person-", "Einstein", "advance", "theoretical physics"},
		{"place", "place-", "Vienna", "offer", "rich cultural heritage"},
		{"event", "event-", "The Revolution", "transform", "the political landscape"},
		{"concept", "concept-", "Stoicism", "represent", "a school of philosophy"},
	}

	for _, tc := range tests {
		for _, p := range g.patterns {
			if !strings.HasPrefix(p.Name, tc.prefix) {
				continue
			}
			t.Run(p.Name, func(t *testing.T) {
				if p.DomainFilter != tc.domain {
					t.Errorf("pattern %q has DomainFilter %q, expected %q", p.Name, p.DomainFilter, tc.domain)
				}
				result := p.Build(g, tc.subj, tc.verb, tc.obj, TensePresent)
				if result == "" {
					t.Errorf("pattern %q produced empty output", p.Name)
				}
				t.Logf("%s: %s", p.Name, result)
			})
		}
	}
}

func TestTonePatterns(t *testing.T) {
	g := NewGenerativeEngine()
	subj, verb, obj := "Stoicism", "represent", "a school of philosophy"

	tonePrefixes := []string{
		"formal-", "casual-", "narrative-", "reflective-",
	}

	for _, p := range g.patterns {
		isTone := false
		for _, prefix := range tonePrefixes {
			if strings.HasPrefix(p.Name, prefix) {
				isTone = true
				break
			}
		}
		if !isTone {
			continue
		}

		t.Run(p.Name, func(t *testing.T) {
			result := p.Build(g, subj, verb, obj, TensePresent)
			if result == "" {
				t.Errorf("pattern %q produced empty output", p.Name)
			}
			t.Logf("%s: %s", p.Name, result)
		})
	}
}

func TestRhetoricalPatterns(t *testing.T) {
	g := NewGenerativeEngine()
	subj, verb, obj := "Stoicism", "represent", "a school of philosophy"

	rhetPrefixes := []string{
		"analogy-", "enum-", "example-", "definition-", "concession-",
	}

	for _, p := range g.patterns {
		isRhet := false
		for _, prefix := range rhetPrefixes {
			if strings.HasPrefix(p.Name, prefix) {
				isRhet = true
				break
			}
		}
		if !isRhet {
			continue
		}

		t.Run(p.Name, func(t *testing.T) {
			result := p.Build(g, subj, verb, obj, TensePresent)
			if result == "" {
				t.Errorf("pattern %q produced empty output", p.Name)
			}
			t.Logf("%s: %s", p.Name, result)
		})
	}
}

func TestDomainFilterBoost(t *testing.T) {
	g := NewGenerativeEngine()
	g.topicCategory = "philosopher" // isPerson → "person" domain

	// Run many pattern picks and count how often person-domain patterns appear
	personCount := 0
	totalRuns := 1000
	for i := 0; i < totalRuns; i++ {
		p := g.pickPatternFor(RelIsA)
		if p.DomainFilter == "person" {
			personCount++
		}
	}

	// Person-domain patterns should appear more often with a person topic
	// than they would by pure weight alone. Just verify they appear at all.
	t.Logf("Person-domain patterns picked %d/%d times with philosopher topic", personCount, totalRuns)
	if personCount == 0 {
		t.Errorf("expected person-domain patterns to be picked at least once with philosopher topic")
	}

	// Now test with a non-person topic
	g.topicCategory = "language" // concept domain
	personCountConcept := 0
	for i := 0; i < totalRuns; i++ {
		p := g.pickPatternFor(RelIsA)
		if p.DomainFilter == "person" {
			personCountConcept++
		}
	}

	t.Logf("Person-domain patterns picked %d/%d times with language topic", personCountConcept, totalRuns)
	// With a concept topic, person patterns should be picked less often
	if personCount <= personCountConcept {
		t.Errorf("expected person patterns to be picked more often with philosopher (%d) than language (%d)",
			personCount, personCountConcept)
	}
}

func TestNoPatternDuplicateNames(t *testing.T) {
	g := NewGenerativeEngine()
	seen := make(map[string]bool)
	for _, p := range g.patterns {
		if seen[p.Name] {
			t.Errorf("duplicate pattern name: %q", p.Name)
		}
		seen[p.Name] = true
	}
	t.Logf("Checked %d patterns, all names unique", len(g.patterns))
}
