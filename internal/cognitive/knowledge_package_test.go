package cognitive

import (
	"encoding/json"
	"os"
	"path/filepath"
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
