package cognitive

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestDocumentGenerate_Overview(t *testing.T) {
	// Set up a graph with test data.
	graph := NewCognitiveGraph("")
	graph.EnsureNode("Gravity", NodeConcept)
	graph.EnsureNode("Isaac Newton", NodeEntity)
	graph.EnsureNode("Albert Einstein", NodeEntity)
	graph.EnsureNode("General relativity", NodeConcept)
	graph.EnsureNode("Planetary orbits", NodeConcept)

	graph.AddEdge("Gravity", "Isaac Newton", RelCreatedBy, "test")
	graph.AddEdge("Gravity", "Albert Einstein", RelRelatedTo, "test")
	graph.AddEdge("Gravity", "General relativity", RelRelatedTo, "test")
	graph.AddEdge("Gravity", "Planetary orbits", RelUsedFor, "test")
	graph.AddEdge("Gravity", "a fundamental force of attraction between all objects", RelDescribedAs, "test")

	// Create a temp knowledge dir with a test paragraph.
	tmpDir := t.TempDir()
	content := "Gravity is the fundamental force of attraction between all objects with mass or energy. " +
		"Described by Isaac Newton as a universal force proportional to mass and inversely proportional " +
		"to the square of distance, gravity governs planetary orbits, tides, and the large-scale structure " +
		"of the universe. Albert Einstein reinterpreted gravity in general relativity as the curvature of " +
		"spacetime caused by mass and energy, rather than a force acting at a distance.\n\n" +
		"General relativity is Einstein's theory of gravitation published in 1915. It describes gravity " +
		"as the curvature of spacetime caused by mass and energy."
	os.WriteFile(filepath.Join(tmpDir, "science.txt"), []byte(content), 0644)

	dg := NewDocumentGenerator(graph, tmpDir)
	doc := dg.Generate("gravity", "overview")

	if doc == nil {
		t.Fatal("expected non-nil document")
	}

	if doc.Title == "" {
		t.Error("expected non-empty title")
	}

	if !strings.Contains(doc.Title, "Gravity") && !strings.Contains(doc.Title, "gravity") {
		t.Errorf("title should mention gravity, got: %s", doc.Title)
	}

	// Should have multiple sections.
	if len(doc.Sections) < 2 {
		t.Errorf("expected at least 2 sections, got %d", len(doc.Sections))
	}

	// Check that required section headings are present.
	headings := make(map[string]bool)
	for _, sec := range doc.Sections {
		headings[sec.Heading] = true
		if sec.Content == "" {
			t.Errorf("section %q has empty content", sec.Heading)
		}
	}

	if !headings["Introduction"] {
		t.Error("missing Introduction section")
	}

	// Word count should be populated.
	if doc.WordCount == 0 && len(doc.Sections) > 0 {
		t.Error("word count should be > 0")
	}

	// Test Markdown formatting.
	md := FormatAsMarkdown(doc)
	if !strings.Contains(md, "# ") {
		t.Error("Markdown output should contain heading markers")
	}
	if !strings.Contains(md, "## Introduction") {
		t.Error("Markdown output should contain Introduction heading")
	}

	// Test plain text formatting.
	pt := FormatAsPlainText(doc)
	if !strings.Contains(pt, "===") || !strings.Contains(pt, "---") {
		t.Error("plain text output should contain underlines")
	}
}

func TestDocumentGenerate_Guide(t *testing.T) {
	graph := NewCognitiveGraph("")
	graph.EnsureNode("Programming", NodeConcept)
	graph.EnsureNode("Variables", NodeConcept)
	graph.EnsureNode("Functions", NodeConcept)
	graph.EnsureNode("Data structures", NodeConcept)
	graph.EnsureNode("Software development", NodeConcept)

	graph.AddEdge("Programming", "Software development", RelIsA, "test")
	graph.AddEdge("Programming", "Variables", RelHas, "test")
	graph.AddEdge("Programming", "Functions", RelHas, "test")
	graph.AddEdge("Programming", "Data structures", RelUsedFor, "test")
	graph.AddEdge("Programming", "the process of creating instructions for computers", RelDescribedAs, "test")

	tmpDir := t.TempDir()
	content := "Programming is the process of creating sets of instructions that tell a computer how to perform a task. " +
		"It involves writing code in various programming languages, each designed for different purposes. " +
		"Modern programming encompasses web development, mobile apps, data science, and artificial intelligence.\n\n" +
		"Variables are named storage locations in computer memory that hold data values during program execution."
	os.WriteFile(filepath.Join(tmpDir, "tech.txt"), []byte(content), 0644)

	dg := NewDocumentGenerator(graph, tmpDir)
	doc := dg.Generate("programming", "guide")

	if doc == nil {
		t.Fatal("expected non-nil document")
	}

	if !strings.Contains(doc.Title, "Guide") {
		t.Errorf("guide title should say 'Guide', got: %s", doc.Title)
	}

	// Should have guide-specific sections.
	headings := make(map[string]bool)
	for _, sec := range doc.Sections {
		headings[sec.Heading] = true
	}

	if !headings["Overview"] {
		t.Error("guide should have Overview section")
	}

	// At least 2 sections expected (Overview + at least one of Prerequisites/Steps/Tips/Next Steps).
	if len(doc.Sections) < 2 {
		t.Errorf("expected at least 2 sections for guide, got %d", len(doc.Sections))
	}

	// Test that word count is computed.
	if doc.WordCount == 0 && len(doc.Sections) > 0 {
		t.Error("word count should be > 0 when sections have content")
	}
}

func TestDocumentGenerate_Report(t *testing.T) {
	graph := NewCognitiveGraph("")
	graph.EnsureNode("Climate change", NodeConcept)
	graph.EnsureNode("Greenhouse gases", NodeConcept)
	graph.EnsureNode("Carbon dioxide", NodeConcept)

	graph.AddEdge("Climate change", "Greenhouse gases", RelRelatedTo, "test")
	graph.AddEdge("Climate change", "Carbon dioxide", RelCauses, "test")
	graph.AddEdge("Climate change", "long-term shifts in temperatures and weather patterns", RelDescribedAs, "test")

	tmpDir := t.TempDir()
	content := "Climate change refers to long-term shifts in temperatures and weather patterns. " +
		"These shifts may be natural, such as through variations in the solar cycle. " +
		"Since the 1800s, human activities have been the main driver of climate change."
	os.WriteFile(filepath.Join(tmpDir, "env.txt"), []byte(content), 0644)

	dg := NewDocumentGenerator(graph, tmpDir)
	doc := dg.Generate("climate change", "report")

	if doc == nil {
		t.Fatal("expected non-nil document")
	}

	if !strings.Contains(doc.Title, "Report") {
		t.Errorf("report title should say 'Report', got: %s", doc.Title)
	}

	headings := make(map[string]bool)
	for _, sec := range doc.Sections {
		headings[sec.Heading] = true
	}

	if !headings["Executive Summary"] && !headings["Background"] {
		t.Error("report should have Executive Summary or Background section")
	}
}

func TestDocumentGenerate_Essay(t *testing.T) {
	graph := NewCognitiveGraph("")
	graph.EnsureNode("Democracy", NodeConcept)
	graph.EnsureNode("Freedom", NodeConcept)
	graph.EnsureNode("Authoritarianism", NodeConcept)

	graph.AddEdge("Democracy", "Freedom", RelRelatedTo, "test")
	graph.AddEdge("Democracy", "Authoritarianism", RelContradicts, "test")
	graph.AddEdge("Democracy", "a system of government by the people", RelDescribedAs, "test")

	tmpDir := t.TempDir()
	content := "Democracy is a system of government in which the citizens exercise power directly or elect representatives. " +
		"It originated in ancient Athens around the 5th century BCE. " +
		"Modern democracies typically feature regular elections, rule of law, and protection of individual rights."
	os.WriteFile(filepath.Join(tmpDir, "gov.txt"), []byte(content), 0644)

	dg := NewDocumentGenerator(graph, tmpDir)
	doc := dg.Generate("democracy", "essay")

	if doc == nil {
		t.Fatal("expected non-nil document")
	}

	headings := make(map[string]bool)
	for _, sec := range doc.Sections {
		headings[sec.Heading] = true
	}

	if !headings["Thesis"] {
		t.Error("essay should have Thesis section")
	}
}

func TestFormatAsMarkdown_Nil(t *testing.T) {
	result := FormatAsMarkdown(nil)
	if result != "" {
		t.Error("expected empty string for nil document")
	}
}

func TestFormatAsPlainText_Nil(t *testing.T) {
	result := FormatAsPlainText(nil)
	if result != "" {
		t.Error("expected empty string for nil document")
	}
}

func TestFirstSentencesOf(t *testing.T) {
	text := "First sentence. Second sentence. Third sentence."
	got := firstSentencesOf(text, 2)
	if !strings.HasPrefix(got, "First sentence.") {
		t.Errorf("expected to start with first sentence, got: %s", got)
	}
	if !strings.Contains(got, "Second sentence.") {
		t.Errorf("expected to contain second sentence, got: %s", got)
	}
	if strings.Contains(got, "Third") {
		t.Errorf("should not contain third sentence, got: %s", got)
	}
}

func TestCountDocWords(t *testing.T) {
	doc := &GeneratedDocument{
		Sections: []DocumentSection{
			{Heading: "A", Content: "one two three"},
			{Heading: "B", Content: "four five"},
		},
	}
	doc.WordCount = countDocWords(doc)
	if doc.WordCount != 5 {
		t.Errorf("expected 5 words, got %d", doc.WordCount)
	}
}
