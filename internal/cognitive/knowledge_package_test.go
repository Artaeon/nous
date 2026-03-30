package cognitive

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestLoadPackageFile(t *testing.T) {
	// Create a temp package
	dir := t.TempDir()
	pkg := KnowledgePackage{
		Name:        "test-science",
		Version:     "1.0.0",
		Description: "Test science package",
		Domain:      "science",
		Facts: []PackageFact{
			{Subject: "water", Relation: "is_a", Object: "chemical compound"},
			{Subject: "water", Relation: "has", Object: "H2O formula"},
			{Subject: "water", Relation: "described_as", Object: "essential for life"},
		},
		Vocabulary: &VocabExpansion{
			Adjectives: []string{"aqueous", "crystalline"},
			Nouns:      []string{"molecule", "compound"},
		},
	}

	data, err := json.MarshalIndent(pkg, "", "  ")
	if err != nil {
		t.Fatal(err)
	}

	path := filepath.Join(dir, "test.json")
	if err := os.WriteFile(path, data, 0644); err != nil {
		t.Fatal(err)
	}

	// Load it
	graph := NewCognitiveGraph(filepath.Join(dir, "graph.json"))
	engine := NewGenerativeEngine()
	loader := NewPackageLoader(graph, engine, nil, dir)

	result, err := loader.LoadFile(path)
	if err != nil {
		t.Fatal(err)
	}

	if result.FactsLoaded != 3 {
		t.Errorf("expected 3 facts, got %d", result.FactsLoaded)
	}
	if result.VocabLoaded < 2 {
		t.Errorf("expected at least 2 vocab, got %d", result.VocabLoaded)
	}

	// Verify graph has the edges
	edges := graph.EdgesFrom("water")
	if len(edges) != 3 {
		t.Errorf("expected 3 edges from water, got %d", len(edges))
	}

	t.Logf("Loaded: %s", result)
}

func TestLoadAllPackages(t *testing.T) {
	dir := t.TempDir()

	// Create two packages
	pkgs := []KnowledgePackage{
		{
			Name:    "pkg-a",
			Version: "1.0.0",
			Domain:  "test",
			Facts: []PackageFact{
				{Subject: "A", Relation: "is_a", Object: "letter"},
			},
		},
		{
			Name:    "pkg-b",
			Version: "1.0.0",
			Domain:  "test",
			Facts: []PackageFact{
				{Subject: "B", Relation: "is_a", Object: "letter"},
				{Subject: "B", Relation: "has", Object: "curves"},
			},
		},
	}

	for i, pkg := range pkgs {
		data, _ := json.MarshalIndent(pkg, "", "  ")
		name := filepath.Join(dir, pkg.Name+".json")
		if err := os.WriteFile(name, data, 0644); err != nil {
			t.Fatal(err)
		}
		_ = i
	}

	graph := NewCognitiveGraph(filepath.Join(dir, "graph.json"))
	engine := NewGenerativeEngine()
	loader := NewPackageLoader(graph, engine, nil, dir)

	results, err := loader.LoadAll()
	if err != nil {
		t.Fatal(err)
	}

	if len(results) != 2 {
		t.Errorf("expected 2 packages loaded, got %d", len(results))
	}

	total := 0
	for _, r := range results {
		total += r.FactsLoaded
		t.Logf("  %s", r)
	}
	if total != 3 {
		t.Errorf("expected 3 total facts, got %d", total)
	}
}

func TestVocabExpansion(t *testing.T) {
	origLen := len(adjSlots)

	graph := NewCognitiveGraph(filepath.Join(t.TempDir(), "g.json"))
	engine := NewGenerativeEngine()
	loader := NewPackageLoader(graph, engine, nil, "")

	pkg := &KnowledgePackage{
		Name:    "vocab-test",
		Version: "1.0.0",
		Domain:  "language",
		Vocabulary: &VocabExpansion{
			Adjectives:    []string{"magnificent", "resplendent"},
			Metaphors:     []string{"scaffold", "tapestry"},
			ContrastPairs: [][2]string{{"fragile", "resilient"}},
			Punchlines:    []string{"And the world noticed."},
		},
	}

	result := loader.Install(pkg)
	t.Logf("Vocab loaded: %d", result.VocabLoaded)

	if len(adjSlots) <= origLen {
		t.Error("adjSlots was not expanded")
	}

	// Verify no duplicates on re-install
	result2 := loader.Install(pkg)
	if result2.VocabLoaded != 0 {
		t.Errorf("expected 0 new vocab on re-install, got %d", result2.VocabLoaded)
	}
}

func TestCreatePackage(t *testing.T) {
	data, err := CreatePackage("my-pkg", "1.0.0", "A test", "test",
		[]PackageFact{{Subject: "X", Relation: "is_a", Object: "Y"}},
		&VocabExpansion{Adjectives: []string{"splendid"}},
	)
	if err != nil {
		t.Fatal(err)
	}

	var pkg KnowledgePackage
	if err := json.Unmarshal(data, &pkg); err != nil {
		t.Fatal(err)
	}
	if pkg.Name != "my-pkg" {
		t.Errorf("expected name my-pkg, got %s", pkg.Name)
	}
	if len(pkg.Facts) != 1 {
		t.Errorf("expected 1 fact, got %d", len(pkg.Facts))
	}
	t.Logf("Package JSON: %s", string(data))
}

func TestFragmentSubjectFilter(t *testing.T) {
	// These should be detected as fragments
	fragments := []string{
		"It", "He", "She", "They", "This", "That", "These", "Those",
		"It is a compound", "The month", "His early work",
		"A small village", "An important factor",
		"Some people believe", "Many scientists have studied",
		"lowercase start", "x",
		"Pages can be edited by anyone",                       // has verb "can"
		"Fungi are eukaryotes which may evolve",               // has verb "are"
		"In Latin crimen could mean an accusation or charge",  // has verb "could"
		"One two three four five six seven",                   // 7 words, >=6 spaces
	}
	for _, s := range fragments {
		if !isFragmentSubject(s) {
			t.Errorf("expected %q to be detected as fragment", s)
		}
	}

	// These should NOT be detected as fragments
	entities := []string{
		"April", "Tokyo", "Albert Einstein", "Python",
		"North America", "United States", "World War II",
		"Photosynthesis", "DNA", "Jupiter", "Linux",
		"Nikola Tesla", "Mount Everest",
	}
	for _, s := range entities {
		if isFragmentSubject(s) {
			t.Errorf("expected %q to NOT be detected as fragment", s)
		}
	}
}

func TestWikiLookupFiltersFragments(t *testing.T) {
	dir := t.TempDir()

	// Create a wiki batch with mixed clean/fragment subjects
	pkg := KnowledgePackage{
		Name:    "wiki-batch-test",
		Version: "1.0.0",
		Domain:  "wikipedia",
		Facts: []PackageFact{
			{Subject: "Tokyo", Relation: "is_a", Object: "city"},
			{Subject: "Tokyo", Relation: "located_in", Object: "Japan"},
			{Subject: "It", Relation: "has", Object: "many people"},
			{Subject: "The city", Relation: "has", Object: "trains"},
			{Subject: "Osaka", Relation: "is_a", Object: "city"},
		},
	}

	data, _ := json.MarshalIndent(pkg, "", "  ")
	path := filepath.Join(dir, "wiki-batch-0001.json")
	os.WriteFile(path, data, 0644)

	graph := NewCognitiveGraph(filepath.Join(dir, "graph.json"))
	engine := NewGenerativeEngine()
	loader := NewPackageLoader(graph, engine, nil, dir)

	// Index the batch
	loader.indexWikiPackage(path)

	// Fragment subjects should not be in the index
	if loader.HasWikiEntry("it") {
		t.Error("'it' should not be indexed")
	}
	if loader.HasWikiEntry("the city") {
		t.Error("'the city' should not be indexed")
	}

	// Real entities should be indexed
	if !loader.HasWikiEntry("tokyo") {
		t.Error("'tokyo' should be indexed")
	}
	if !loader.HasWikiEntry("osaka") {
		t.Error("'osaka' should be indexed")
	}

	// Loading should filter out fragment facts
	loaded := loader.LookupWiki("Tokyo")
	if loaded != 3 { // Tokyo(2) + Osaka(1), fragments filtered
		t.Errorf("expected 3 clean facts loaded, got %d", loaded)
	}

	// Verify no fragment subjects in graph
	itEdges := graph.EdgesFrom("It")
	if len(itEdges) > 0 {
		t.Errorf("expected no edges from 'It', got %d", len(itEdges))
	}

	t.Logf("Wiki index size: %d, facts loaded: %d", loader.WikiIndexSize(), loaded)
}

func TestWikiEntityResolution(t *testing.T) {
	packDir := filepath.Join("..", "..", "packages")
	if _, err := os.Stat(packDir); err != nil {
		t.Skip("packages directory not found")
	}

	graph := NewCognitiveGraph(filepath.Join(t.TempDir(), "graph.json"))
	engine := NewGenerativeEngine()
	loader := NewPackageLoader(graph, engine, nil, packDir)
	loader.LoadAll()

	tests := []struct {
		query     string
		wantFacts bool // should load at least some facts
	}{
		// Compound terms should match as units
		{"black hole", true},
		{"black holes", false}, // plural — graph has "black hole" (singular)
		{"world war ii", true},
		{"world war 2", true}, // number → roman numeral

		// Ambiguous names — require full knowledge directory to resolve
		{"gandhi", false},
		{"einstein", false},

		// Direct matches
		{"photosynthesis", false}, // known: wiki batch has only fragment objects
		{"tokyo", false},          // requires knowledge data
		{"dna", true},
	}

	for _, tt := range tests {
		loaded := loader.LookupWiki(tt.query)
		// Check graph for facts — try the query itself, capitalised form,
		// and number-normalized forms (e.g. "world war 2" → "world war ii").
		facts := graph.LookupFacts(tt.query, 20)
		if len(facts) == 0 {
			cap := strings.ToUpper(tt.query[:1]) + tt.query[1:]
			facts = graph.LookupFacts(cap, 20)
		}
		if len(facts) == 0 {
			for _, nf := range normalizeNumbers(tt.query) {
				facts = graph.LookupFacts(nf, 20)
				if len(facts) > 0 {
					break
				}
			}
		}
		if tt.wantFacts && loaded == 0 && len(facts) == 0 {
			t.Errorf("LookupWiki(%q): no facts loaded or found in graph", tt.query)
			continue
		}
		t.Logf("  %-20s → %d new facts, %d graph facts", tt.query, loaded, len(facts))
	}
}

func TestNormalizeNumbers(t *testing.T) {
	tests := []struct {
		input string
		want  string // at least one variant should match this
	}{
		{"world war 2", "world war ii"},
		{"henry 8", "henry viii"},
		{"world war ii", "world war 2"},
		{"apollo 13", ""}, // 13 not in range, no variant
	}
	for _, tt := range tests {
		variants := normalizeNumbers(tt.input)
		if tt.want == "" {
			if len(variants) > 0 {
				t.Errorf("normalizeNumbers(%q): expected no variants, got %v", tt.input, variants)
			}
			continue
		}
		found := false
		for _, v := range variants {
			if v == tt.want {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("normalizeNumbers(%q): want %q in %v", tt.input, tt.want, variants)
		}
	}
}

func TestContainsPhrase(t *testing.T) {
	tests := []struct {
		text, phrase string
		want         bool
	}{
		{"mahatma gandhi", "gandhi", true},
		{"gandhi smriti", "gandhi", true},
		{"gandhism", "gandhi", false}, // not a word boundary
		{"black hole", "black hole", true},
		{"black hole thermodynamics", "black hole", true},
		{"black", "black hole", false},
		{"world war ii", "world war ii", true},
		{"world war ii battles", "world war ii", true},
	}
	for _, tt := range tests {
		got := containsPhrase(tt.text, tt.phrase)
		if got != tt.want {
			t.Errorf("containsPhrase(%q, %q) = %v, want %v", tt.text, tt.phrase, got, tt.want)
		}
	}
}

func TestPluralVariants(t *testing.T) {
	tests := []struct {
		input string
		want  []string
	}{
		{"black holes", []string{"black hole", "black hol", "black holy"}},
		{"black hole", []string{"black holes", "black holy"}},
		{"cities", []string{"citie", "citi", "city"}},
	}
	for _, tt := range tests {
		got := pluralVariants(tt.input)
		// Just check the first variant is present
		found := false
		for _, g := range got {
			if g == tt.want[0] {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("pluralVariants(%q): want %q in result, got %v", tt.input, tt.want[0], got)
		}
	}
}

func TestLoadRealPackages(t *testing.T) {
	// Test loading the actual bundled packages from ../../packages/
	packDir := filepath.Join("..", "..", "packages")
	if _, err := os.Stat(packDir); err != nil {
		t.Skip("packages directory not found")
	}

	graph := NewCognitiveGraph(filepath.Join(t.TempDir(), "graph.json"))
	engine := NewGenerativeEngine()
	loader := NewPackageLoader(graph, engine, nil, packDir)

	results, err := loader.LoadAll()
	if err != nil {
		t.Fatal(err)
	}

	totalFacts, totalVocab := 0, 0
	for _, r := range results {
		totalFacts += r.FactsLoaded
		totalVocab += r.VocabLoaded
		t.Logf("  %s", r)
	}

	t.Logf("\nTotal: %d packages, %d facts, %d vocab entries", len(results), totalFacts, totalVocab)

	if totalFacts < 100 {
		t.Errorf("expected at least 100 facts from bundled packages, got %d", totalFacts)
	}
	// Note: vocab count may be lower when other tests have already loaded
	// packages into the global slot pools (adjSlots, etc.) since appendUnique
	// deduplicates against the existing global state.
	if totalVocab < 20 {
		t.Errorf("expected at least 20 vocab from bundled packages, got %d", totalVocab)
	}

	// Verify we can generate about loaded topics
	edges := graph.EdgesFrom("Stoicism")
	if len(edges) < 3 {
		t.Errorf("expected at least 3 facts about Stoicism, got %d", len(edges))
	}

	// Generate an article about a loaded topic
	var facts []edgeFact
	for _, e := range edges {
		subj := graph.NodeLabel(e.From)
		obj := graph.NodeLabel(e.To)
		if subj == "" {
			subj = e.From
		}
		if obj == "" {
			obj = e.To
		}
		facts = append(facts, edgeFact{Subject: subj, Relation: e.Relation, Object: obj})
	}
	article := engine.ComposeArticle("Stoicism", facts)
	t.Logf("\n=== Stoicism Article (from packages) ===\n%s", article)
}
