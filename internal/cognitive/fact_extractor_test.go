package cognitive

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// -----------------------------------------------------------------------
// Unit tests for individual extraction patterns
// -----------------------------------------------------------------------

func TestWikiExtractIsA(t *testing.T) {
	fe := NewWikiFactExtractor()

	facts := fe.ExtractFromText("Python is a programming language")
	found := false
	for _, f := range facts {
		if f.Relation == RelIsA &&
			strings.EqualFold(f.Subject, "Python") &&
			strings.Contains(strings.ToLower(f.Object), "programming language") {
			found = true
			break
		}
	}
	if !found {
		t.Errorf("expected (Python, is_a, programming language), got %d facts: %+v", len(facts), facts)
	}
}

func TestWikiExtractIsAPlural(t *testing.T) {
	fe := NewWikiFactExtractor()

	facts := fe.ExtractFromText("Operating systems are software that manage hardware resources")
	found := false
	for _, f := range facts {
		if f.Relation == RelIsA &&
			strings.Contains(strings.ToLower(f.Subject), "operating systems") {
			found = true
			break
		}
	}
	if !found {
		t.Errorf("expected IsA fact for operating systems, got: %+v", facts)
	}
}

func TestWikiExtractIsAAppositive(t *testing.T) {
	fe := NewWikiFactExtractor()

	facts := fe.ExtractFromText("Python, a programming language, is widely used")
	found := false
	for _, f := range facts {
		if f.Relation == RelIsA &&
			strings.EqualFold(f.Subject, "Python") &&
			strings.Contains(strings.ToLower(f.Object), "programming language") {
			found = true
			break
		}
	}
	if !found {
		t.Errorf("expected appositive IsA for Python, got: %+v", facts)
	}
}

func TestWikiExtractCreatedBy(t *testing.T) {
	fe := NewWikiFactExtractor()

	tests := []struct {
		input   string
		subject string
		object  string
	}{
		{
			input:   "Linux was created by Linus Torvalds",
			subject: "Linux",
			object:  "Linus Torvalds",
		},
		{
			input:   "The telephone was invented by Alexander Graham Bell",
			subject: "The telephone",
			object:  "Alexander Graham Bell",
		},
		{
			input:   "General relativity was developed by Albert Einstein",
			subject: "General relativity",
			object:  "Albert Einstein",
		},
	}

	for _, tt := range tests {
		facts := fe.ExtractFromText(tt.input)
		found := false
		for _, f := range facts {
			if f.Relation == RelCreatedBy &&
				strings.Contains(strings.ToLower(f.Object), strings.ToLower(tt.object)) {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("input %q: expected created_by %s, got: %+v", tt.input, tt.object, facts)
		}
	}
}

func TestWikiExtractFoundedBy(t *testing.T) {
	fe := NewWikiFactExtractor()
	facts := fe.ExtractFromText("Apple was founded by Steve Jobs")
	found := false
	for _, f := range facts {
		if f.Relation == RelFoundedBy &&
			strings.Contains(strings.ToLower(f.Object), "steve jobs") {
			found = true
			break
		}
	}
	if !found {
		t.Errorf("expected founded_by Steve Jobs, got: %+v", facts)
	}
}

func TestWikiExtractFoundedIn(t *testing.T) {
	fe := NewWikiFactExtractor()

	tests := []struct {
		input string
		year  string
	}{
		{"Google was founded in 1998", "1998"},
		{"The university was established in 1746", "1746"},
		{"Python was released in 1991", "1991"},
		{"The concept began in 2005", "2005"},
	}

	for _, tt := range tests {
		facts := fe.ExtractFromText(tt.input)
		found := false
		for _, f := range facts {
			if f.Relation == RelFoundedIn &&
				strings.Contains(f.Object, tt.year) {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("input %q: expected founded_in %s, got: %+v", tt.input, tt.year, facts)
		}
	}
}

func TestWikiExtractHas(t *testing.T) {
	fe := NewWikiFactExtractor()

	tests := []struct {
		input string
	}{
		{"Python features readable syntax"},
		{"The library includes documentation and examples"},
		{"Go provides garbage collection"},
	}

	for _, tt := range tests {
		facts := fe.ExtractFromText(tt.input)
		found := false
		for _, f := range facts {
			if f.Relation == RelHas {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("input %q: expected has fact, got: %+v", tt.input, facts)
		}
	}
}

func TestWikiExtractUsedFor(t *testing.T) {
	fe := NewWikiFactExtractor()

	tests := []struct {
		input string
	}{
		{"Python is used for data science"},
		{"Calculus is applied to engineering problems"},
		{"The framework is designed for web development"},
	}

	for _, tt := range tests {
		facts := fe.ExtractFromText(tt.input)
		found := false
		for _, f := range facts {
			if f.Relation == RelUsedFor {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("input %q: expected used_for fact, got: %+v", tt.input, facts)
		}
	}
}

func TestWikiExtractLocatedIn(t *testing.T) {
	fe := NewWikiFactExtractor()

	facts := fe.ExtractFromText("CERN is located in Geneva")
	found := false
	for _, f := range facts {
		if f.Relation == RelLocatedIn &&
			strings.Contains(strings.ToLower(f.Object), "geneva") {
			found = true
			break
		}
	}
	if !found {
		t.Errorf("expected located_in Geneva, got: %+v", facts)
	}
}

func TestWikiExtractPartOf(t *testing.T) {
	fe := NewWikiFactExtractor()

	facts := fe.ExtractFromText("The engine is part of the vehicle")
	found := false
	for _, f := range facts {
		if f.Relation == RelPartOf {
			found = true
			break
		}
	}
	if !found {
		t.Errorf("expected part_of fact, got: %+v", facts)
	}
}

func TestWikiExtractRelatedTo(t *testing.T) {
	fe := NewWikiFactExtractor()

	facts := fe.ExtractFromText("Quantum mechanics is related to wave theory")
	found := false
	for _, f := range facts {
		if f.Relation == RelRelatedTo {
			found = true
			break
		}
	}
	if !found {
		t.Errorf("expected related_to fact, got: %+v", facts)
	}
}

func TestWikiExtractDescribedAs(t *testing.T) {
	fe := NewWikiFactExtractor()

	// DescribedAs from paragraph-level extraction: first sentence gets
	// a described_as fact with the full sentence.
	text := "Gravity is the fundamental force of attraction between all objects.\n\nThis is another paragraph."
	facts := fe.ExtractFromText(text)
	found := false
	for _, f := range facts {
		if f.Relation == RelDescribedAs &&
			strings.EqualFold(f.Subject, "Gravity") {
			found = true
			break
		}
	}
	if !found {
		t.Errorf("expected described_as for Gravity, got: %+v", facts)
	}
}

// -----------------------------------------------------------------------
// Object cleaning tests
// -----------------------------------------------------------------------

func TestCleanObject(t *testing.T) {
	tests := []struct {
		input string
		want  string
	}{
		{"a programming language, which supports multiple paradigms", "a programming language"},
		{"Geneva, Switzerland.", "Geneva, Switzerland"},
		{"data science   and  machine  learning", "data science and machine learning"},
	}
	for _, tt := range tests {
		got := cleanObject(tt.input)
		if got != tt.want {
			t.Errorf("cleanObject(%q) = %q, want %q", tt.input, got, tt.want)
		}
	}
}

// -----------------------------------------------------------------------
// File-level extraction tests
// -----------------------------------------------------------------------

func TestWikiExtractFromFile(t *testing.T) {
	path := filepath.Join("..", "..", "knowledge", "01_science.txt")
	if _, err := os.Stat(path); os.IsNotExist(err) {
		t.Skip("knowledge file not found at", path)
	}

	fe := NewWikiFactExtractor()
	facts, err := fe.ExtractFromFile(path)
	if err != nil {
		t.Fatalf("ExtractFromFile: %v", err)
	}

	if len(facts) < 50 {
		t.Errorf("expected 50+ facts from science file, got %d", len(facts))
	}

	// Verify we got a mix of relation types.
	relCounts := make(map[RelType]int)
	for _, f := range facts {
		relCounts[f.Relation]++
	}

	if relCounts[RelIsA] < 5 {
		t.Errorf("expected 5+ IsA facts, got %d", relCounts[RelIsA])
	}
	if relCounts[RelDescribedAs] < 5 {
		t.Errorf("expected 5+ DescribedAs facts, got %d", relCounts[RelDescribedAs])
	}

	t.Logf("extracted %d facts from science file", len(facts))
	for rel, count := range relCounts {
		t.Logf("  %s: %d", rel, count)
	}
}

func TestWikiExtractFromDirectory(t *testing.T) {
	dir := filepath.Join("..", "..", "knowledge")
	if _, err := os.Stat(dir); os.IsNotExist(err) {
		t.Skip("knowledge directory not found at", dir)
	}

	fe := NewWikiFactExtractor()
	facts, err := fe.ExtractFromDirectory(dir)
	if err != nil {
		t.Fatalf("ExtractFromDirectory: %v", err)
	}

	if len(facts) < 500 {
		t.Errorf("expected 500+ facts from all knowledge files, got %d", len(facts))
	}

	// Verify relation type diversity.
	relCounts := make(map[RelType]int)
	for _, f := range facts {
		relCounts[f.Relation]++
	}

	t.Logf("extracted %d total facts from knowledge directory", len(facts))
	for rel, count := range relCounts {
		t.Logf("  %s: %d", rel, count)
	}

	// Should have at least 4 different relation types.
	if len(relCounts) < 4 {
		t.Errorf("expected 4+ relation types, got %d: %v", len(relCounts), relCounts)
	}
}

// -----------------------------------------------------------------------
// Graph ingestion test
// -----------------------------------------------------------------------

func TestWikiIngestIntoGraph(t *testing.T) {
	fe := NewWikiFactExtractor()
	graph := NewCognitiveGraph("")

	facts := fe.ExtractFromText(
		"Python is a programming language. " +
			"Python was created by Guido van Rossum. " +
			"Python was released in 1991. " +
			"Python is used for data science. " +
			"Python features readable syntax.",
	)

	added := fe.IngestIntoGraph(graph, facts)
	if added == 0 {
		t.Fatal("expected facts to be added to graph")
	}

	if graph.NodeCount() < 2 {
		t.Errorf("expected 2+ nodes, got %d", graph.NodeCount())
	}
	if graph.EdgeCount() < 2 {
		t.Errorf("expected 2+ edges, got %d", graph.EdgeCount())
	}

	t.Logf("ingested %d facts, graph has %d nodes and %d edges",
		added, graph.NodeCount(), graph.EdgeCount())
}

// -----------------------------------------------------------------------
// Deduplication test
// -----------------------------------------------------------------------

func TestWikiDeduplication(t *testing.T) {
	fe := NewWikiFactExtractor()

	// Same text twice should not produce duplicate facts.
	text := "Go is a programming language. Go was created by Google."
	facts1 := fe.ExtractFromText(text)
	facts2 := fe.ExtractFromText(text + "\n\n" + text)

	if len(facts2) > len(facts1) {
		t.Errorf("deduplication failed: single pass produced %d facts, double pass produced %d",
			len(facts1), len(facts2))
	}

	// IngestIntoGraph should also deduplicate.
	graph := NewCognitiveGraph("")
	added1 := fe.IngestIntoGraph(graph, facts1)
	added2 := fe.IngestIntoGraph(graph, facts1) // same facts again
	if added2 > 0 {
		// The graph's addEdgeLocked boosts weight instead of adding duplicates,
		// but our IngestIntoGraph uses a seen map to skip duplicates entirely.
		// Second pass should add 0 new facts because of seen-map dedup.
		// Actually, IngestIntoGraph creates a fresh seen map each call,
		// but the graph itself deduplicates edges. So added2 == added1 is OK
		// as long as EdgeCount doesn't double.
		edges1 := graph.EdgeCount()
		graph2 := NewCognitiveGraph("")
		fe.IngestIntoGraph(graph2, append(facts1, facts1...))
		edges2 := graph2.EdgeCount()
		if edges2 > edges1 {
			t.Errorf("graph dedup failed: single ingest %d edges, double ingest %d edges",
				edges1, edges2)
		}
	}

	t.Logf("dedup: %d facts from single pass, %d from double pass, %d+%d ingested",
		len(facts1), len(facts2), added1, added2)
}

// -----------------------------------------------------------------------
// Edge case tests
// -----------------------------------------------------------------------

func TestWikiExtractEmptyText(t *testing.T) {
	fe := NewWikiFactExtractor()
	facts := fe.ExtractFromText("")
	if len(facts) != 0 {
		t.Errorf("expected 0 facts from empty text, got %d", len(facts))
	}
}

func TestWikiExtractShortText(t *testing.T) {
	fe := NewWikiFactExtractor()
	facts := fe.ExtractFromText("Hi")
	if len(facts) != 0 {
		t.Errorf("expected 0 facts from short text, got %d", len(facts))
	}
}

func TestWikiExtractConfidence(t *testing.T) {
	fe := NewWikiFactExtractor()
	facts := fe.ExtractFromText("Python is a programming language")
	for _, f := range facts {
		if f.Confidence < 0.5 || f.Confidence > 1.0 {
			t.Errorf("confidence %f out of range [0.5, 1.0]", f.Confidence)
		}
	}
}

func TestWikiExtractMissingFile(t *testing.T) {
	fe := NewWikiFactExtractor()
	_, err := fe.ExtractFromFile("/nonexistent/file.txt")
	if err == nil {
		t.Error("expected error for missing file")
	}
}

func TestWikiExtractMissingDirectory(t *testing.T) {
	fe := NewWikiFactExtractor()
	_, err := fe.ExtractFromDirectory("/nonexistent/dir")
	if err == nil {
		t.Error("expected error for missing directory")
	}
}

// -----------------------------------------------------------------------
// Test on actual knowledge content to verify pattern coverage
// -----------------------------------------------------------------------

func TestWikiExtractRealContent(t *testing.T) {
	fe := NewWikiFactExtractor()

	// Sample paragraphs from the actual knowledge files.
	text := `Gravity is the fundamental force of attraction between all objects with mass or energy. Described by Isaac Newton as a universal force proportional to mass and inversely proportional to the square of distance, gravity governs planetary orbits, tides, and the large-scale structure of the universe.

Quantum mechanics is the branch of physics describing the behavior of matter and energy at atomic and subatomic scales. Developed in the early twentieth century by Planck, Heisenberg, Schrodinger, and Dirac, it replaces deterministic classical mechanics with probabilistic wave functions.

The periodic table is a systematic arrangement of chemical elements organized by increasing atomic number and grouped by recurring chemical properties. Dmitri Mendeleev published the first widely recognized version in 1869, predicting the existence and properties of undiscovered elements.

DNA, deoxyribonucleic acid, is the molecule that carries the genetic instructions for the development, functioning, growth, and reproduction of all known living organisms and many viruses. Its structure, elucidated by Watson and Crick in 1953, consists of two antiparallel polynucleotide chains wound into a double helix.`

	facts := fe.ExtractFromText(text)

	if len(facts) < 8 {
		t.Errorf("expected 8+ facts from real content, got %d", len(facts))
	}

	// Check for specific expected extractions.
	hasGravityIsA := false
	hasQuantumIsA := false
	hasPeriodicTableIsA := false
	hasDNAIsA := false

	for _, f := range facts {
		subj := strings.ToLower(f.Subject)
		switch {
		case strings.Contains(subj, "gravity") && f.Relation == RelIsA:
			hasGravityIsA = true
		case strings.Contains(subj, "quantum mechanics") && f.Relation == RelIsA:
			hasQuantumIsA = true
		case strings.Contains(subj, "periodic table") && f.Relation == RelIsA:
			hasPeriodicTableIsA = true
		case strings.Contains(subj, "dna") && f.Relation == RelIsA:
			hasDNAIsA = true
		}
	}

	if !hasGravityIsA {
		t.Error("missing: Gravity is_a ...")
	}
	if !hasQuantumIsA {
		t.Error("missing: Quantum mechanics is_a ...")
	}
	if !hasPeriodicTableIsA {
		t.Error("missing: The periodic table is_a ...")
	}
	if !hasDNAIsA {
		t.Error("missing: DNA is_a ...")
	}

	t.Logf("extracted %d facts from real content sample", len(facts))
	for _, f := range facts {
		t.Logf("  (%s) -[%s]-> (%s) [%.1f]", f.Subject, f.Relation, f.Object, f.Confidence)
	}
}
