package cognitive

import (
	"testing"
)

func TestTemplateMiner_ProcessSentence(t *testing.T) {
	tm := NewTemplateMiner()

	tests := []struct {
		sentence string
		subject  string
		wantRel  RelType
		wantTmpl string
		wantOK   bool
	}{
		{
			"Go is a programming language developed by Google.",
			"Go",
			RelIsA, "%s is a %s.", true,
		},
		{
			"The Beatles were an English rock band formed in Liverpool.",
			"The Beatles",
			RelIsA, "%s were %s.", true,
		},
		{
			"Python is an interpreted, high-level programming language.",
			"Python",
			RelIsA, "%s is an %s.", true,
		},
		{
			"Rust was developed by Mozilla Research.",
			"Rust",
			RelCreatedBy, "%s was developed by %s.", true,
		},
		{
			"Apple Inc. was founded by Steve Jobs.",
			"Apple Inc.",
			RelFoundedBy, "%s was founded by %s.", true,
		},
		{
			"Tokyo is located in Japan.",
			"Tokyo",
			RelLocatedIn, "%s is located in %s.", true,
		},
		{
			"CERN was established in 1954.",
			"CERN",
			RelFoundedIn, "%s was established in %s.", true,
		},
		{
			"Linux was created by Linus Torvalds.",
			"Linux",
			RelCreatedBy, "%s was created by %s.", true,
		},
		{
			"Photosynthesis is the process used by plants.",
			"Photosynthesis",
			RelIsA, "%s is the %s.", true,
		},
		// Should reject
		{"Too short.", "Too", "", "", false},
		{"It is a thing.", "It", "", "", false},
		{"Contains [[wiki markup]] artifacts.", "Contains", "", "", false},
	}

	for _, tt := range tests {
		rel, tmpl, ok := tm.ProcessSentence(tt.sentence, tt.subject)
		if ok != tt.wantOK {
			t.Errorf("ProcessSentence(%q, %q): ok=%v, want %v", tt.sentence, tt.subject, ok, tt.wantOK)
			continue
		}
		if !ok {
			continue
		}
		if rel != tt.wantRel {
			t.Errorf("ProcessSentence(%q, %q): rel=%s, want %s", tt.sentence, tt.subject, rel, tt.wantRel)
		}
		if tmpl != tt.wantTmpl {
			t.Errorf("ProcessSentence(%q, %q): tmpl=%q, want %q", tt.sentence, tt.subject, tmpl, tt.wantTmpl)
		}
	}
}

func TestTemplateMiner_Export(t *testing.T) {
	tm := NewTemplateMiner()

	// Add some templates
	for i := 0; i < 5; i++ {
		tm.AddTemplate(RelIsA, "%s is a %s.")
	}
	for i := 0; i < 2; i++ {
		tm.AddTemplate(RelIsA, "%s was a %s.")
	}
	tm.AddTemplate(RelCreatedBy, "%s was made by %s.")

	// Export with min-freq 3
	result := tm.Export(3)

	isATemplates := result.Templates["is_a"]
	if len(isATemplates) != 1 {
		t.Errorf("Expected 1 is_a template (freq>=3), got %d", len(isATemplates))
	}
	if len(isATemplates) > 0 && isATemplates[0].Freq != 5 {
		t.Errorf("Expected freq=5 for is_a template, got %d", isATemplates[0].Freq)
	}

	// created_by should be filtered out (freq=1 < 3)
	if _, ok := result.Templates["created_by"]; ok {
		t.Error("Expected created_by to be filtered out (freq < 3)")
	}
}

func TestValidateTemplate(t *testing.T) {
	tests := []struct {
		tmpl string
		want bool
	}{
		{"%s is a %s.", true},
		{"%s was created by %s.", true},
		{"%s is one of the %s.", true},
		{"bad template", false},           // no %s
		{"%s only one slot.", false},       // only 1 %s
		{"not starting with %s %s.", false}, // doesn't start with %s
		{"%s has a Capital Word in %s.", false}, // leaked proper noun
	}

	for _, tt := range tests {
		got := validateTemplate(tt.tmpl)
		if got != tt.want {
			t.Errorf("validateTemplate(%q) = %v, want %v", tt.tmpl, got, tt.want)
		}
	}
}
