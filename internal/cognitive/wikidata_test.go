package cognitive

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestWikidataPropMapping(t *testing.T) {
	tests := []struct {
		prop     string
		expected string
	}{
		{"instance of", "is_a"},
		{"subclass of", "is_a"},
		{"located in administrative territorial entity", "located_in"},
		{"country", "located_in"},
		{"part of", "part_of"},
		{"creator", "created_by"},
		{"discoverer or inventor", "created_by"},
		{"founded by", "founded_by"},
		{"inception", "founded_in"},
		{"date of birth", "founded_in"},
		{"date of death", "founded_in"},
		{"has effect", "causes"},
		{"has part", "has"},
		{"replaced by", "related_to"},
		{"replaces", "related_to"},
		{"occupation", "domain"},
		{"field of work", "domain"},
		{"place of birth", "located_in"},
		{"place of death", "located_in"},
		{"notable work", "created_by"},
		{"movement", "part_of"},
		{"something unknown", "related_to"},
		// case insensitivity
		{"Instance Of", "is_a"},
		{"COUNTRY", "located_in"},
	}

	for _, tt := range tests {
		got := wikidataPropToRelation(tt.prop)
		if got != tt.expected {
			t.Errorf("wikidataPropToRelation(%q) = %q, want %q", tt.prop, got, tt.expected)
		}
	}
}

func TestWikidataParseResults(t *testing.T) {
	mockJSON := `{
		"results": {
			"bindings": [
				{
					"itemLabel": {"value": "Albert Einstein", "type": "literal"},
					"propLabel": {"value": "instance of", "type": "literal"},
					"valueLabel": {"value": "physicist", "type": "literal"}
				},
				{
					"itemLabel": {"value": "Albert Einstein", "type": "literal"},
					"propLabel": {"value": "place of birth", "type": "literal"},
					"valueLabel": {"value": "Ulm", "type": "literal"}
				},
				{
					"itemLabel": {"value": "Albert Einstein", "type": "literal"},
					"propLabel": {"value": "notable work", "type": "literal"},
					"valueLabel": {"value": "general relativity", "type": "literal"}
				},
				{
					"itemLabel": {"value": "http://www.wikidata.org/entity/Q12345", "type": "uri"},
					"propLabel": {"value": "instance of", "type": "literal"},
					"valueLabel": {"value": "something", "type": "literal"}
				},
				{
					"itemLabel": {"value": "", "type": "literal"},
					"propLabel": {"value": "instance of", "type": "literal"},
					"valueLabel": {"value": "something", "type": "literal"}
				}
			]
		}
	}`

	wi := NewWikidataImporter()
	facts, memories, err := wi.parseResults([]byte(mockJSON), "science")
	if err != nil {
		t.Fatalf("parseResults error: %v", err)
	}

	// Should have 3 valid facts (skips Q-ID and empty entries)
	if len(facts) != 3 {
		t.Fatalf("expected 3 facts, got %d: %+v", len(facts), facts)
	}

	// First fact: Einstein is_a physicist
	if facts[0].Subject != "Albert Einstein" || facts[0].Relation != "is_a" || facts[0].Object != "physicist" {
		t.Errorf("unexpected first fact: %+v", facts[0])
	}

	// Second fact: Einstein located_in Ulm
	if facts[1].Subject != "Albert Einstein" || facts[1].Relation != "located_in" || facts[1].Object != "Ulm" {
		t.Errorf("unexpected second fact: %+v", facts[1])
	}

	// Third fact: notable work reversal — general relativity created_by Einstein
	if facts[2].Subject != "general relativity" || facts[2].Relation != "created_by" || facts[2].Object != "Albert Einstein" {
		t.Errorf("unexpected third fact (notable work reversal): %+v", facts[2])
	}

	// Should have at least 1 memory (from the is_a relation)
	if len(memories) < 1 {
		t.Fatalf("expected at least 1 memory, got %d", len(memories))
	}
	if memories[0].Category != "definition" {
		t.Errorf("expected memory category 'definition', got %q", memories[0].Category)
	}
}

func TestWikidataParseResultsDedup(t *testing.T) {
	mockJSON := `{
		"results": {
			"bindings": [
				{
					"itemLabel": {"value": "Go"},
					"propLabel": {"value": "instance of"},
					"valueLabel": {"value": "programming language"}
				},
				{
					"itemLabel": {"value": "Go"},
					"propLabel": {"value": "instance of"},
					"valueLabel": {"value": "programming language"}
				}
			]
		}
	}`

	wi := NewWikidataImporter()
	facts, _, err := wi.parseResults([]byte(mockJSON), "technology")
	if err != nil {
		t.Fatalf("parseResults error: %v", err)
	}

	if len(facts) != 1 {
		t.Errorf("expected 1 deduplicated fact, got %d", len(facts))
	}
}

func TestWikidataSparqlForDomain(t *testing.T) {
	wi := NewWikidataImporter()

	for domain := range domainQueries {
		query := wi.sparqlForDomain(domain, 50)

		if !strings.Contains(query, "SELECT") {
			t.Errorf("domain %q: query missing SELECT", domain)
		}
		if !strings.Contains(query, "LIMIT 50") {
			t.Errorf("domain %q: query missing LIMIT 50", domain)
		}
		if !strings.Contains(query, "wikibase:label") {
			t.Errorf("domain %q: query missing label service", domain)
		}
		if !strings.Contains(query, "wdt:P31") {
			t.Errorf("domain %q: query missing P31 (instance of)", domain)
		}
		if !strings.Contains(query, "itemLabel") {
			t.Errorf("domain %q: query missing itemLabel", domain)
		}

		// Check that all class QIDs are present
		info := domainQueries[domain]
		for _, qid := range info.classes {
			if !strings.Contains(query, "wd:"+qid) {
				t.Errorf("domain %q: query missing class QID %s", domain, qid)
			}
		}
	}
}

func TestWikidataSparqlForEntity(t *testing.T) {
	wi := NewWikidataImporter()
	query := wi.sparqlForEntity("Albert Einstein")

	if !strings.Contains(query, `"Albert Einstein"@en`) {
		t.Error("entity query missing label search for Albert Einstein")
	}
	if !strings.Contains(query, "SELECT") {
		t.Error("entity query missing SELECT")
	}
	if !strings.Contains(query, "LIMIT") {
		t.Error("entity query missing LIMIT")
	}
}

func TestWikidataSavePackage(t *testing.T) {
	wi := NewWikidataImporter()

	pkg := &KnowledgePackage{
		Name:        "wikidata-test",
		Version:     "1.0.0",
		Description: "Test package",
		Domain:      "test",
		Facts: []PackageFact{
			{Subject: "Go", Relation: "is_a", Object: "programming language"},
			{Subject: "Go", Relation: "created_by", Object: "Google"},
		},
		Memories: []PackageMemory{
			{Key: "Go", Value: "Go is a programming language", Category: "definition"},
		},
	}

	tmpDir := t.TempDir()
	err := wi.SavePackage(pkg, tmpDir)
	if err != nil {
		t.Fatalf("SavePackage error: %v", err)
	}

	outPath := filepath.Join(tmpDir, "wikidata_test.json")
	data, err := os.ReadFile(outPath)
	if err != nil {
		t.Fatalf("read output file: %v", err)
	}

	// Verify it's valid JSON and round-trips
	var loaded KnowledgePackage
	if err := json.Unmarshal(data, &loaded); err != nil {
		t.Fatalf("unmarshal output: %v", err)
	}

	if loaded.Name != "wikidata-test" {
		t.Errorf("expected name 'wikidata-test', got %q", loaded.Name)
	}
	if len(loaded.Facts) != 2 {
		t.Errorf("expected 2 facts, got %d", len(loaded.Facts))
	}
	if len(loaded.Memories) != 1 {
		t.Errorf("expected 1 memory, got %d", len(loaded.Memories))
	}

	// Verify pretty-printed (indented)
	if !strings.Contains(string(data), "  ") {
		t.Error("output does not appear to be pretty-printed")
	}
}

func TestWikidataNewImporter(t *testing.T) {
	wi := NewWikidataImporter()
	if wi.Endpoint != "https://query.wikidata.org/sparql" {
		t.Errorf("unexpected endpoint: %s", wi.Endpoint)
	}
	if wi.UserAgent != "Nous/1.0 (cognitive engine)" {
		t.Errorf("unexpected user agent: %s", wi.UserAgent)
	}
	if wi.client == nil {
		t.Error("http client is nil")
	}
}

func TestWikidataUnsupportedDomain(t *testing.T) {
	wi := NewWikidataImporter()
	_, err := wi.ImportDomain("nonexistent", 10)
	if err == nil {
		t.Fatal("expected error for unsupported domain")
	}
	if !strings.Contains(err.Error(), "unsupported domain") {
		t.Errorf("unexpected error message: %v", err)
	}
}

func TestWikidataEmptyEntity(t *testing.T) {
	wi := NewWikidataImporter()
	_, err := wi.ImportEntity("")
	if err == nil {
		t.Fatal("expected error for empty entity")
	}
}

func TestWikidataImportDomain(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping integration test in short mode")
	}
	if os.Getenv("CI") != "" {
		t.Skip("skipping network-dependent test in CI")
	}

	wi := NewWikidataImporter()
	pkg, err := wi.ImportDomain("science", 20)
	if err != nil {
		// Network failures are not test failures — skip gracefully.
		if strings.Contains(err.Error(), "timeout") || strings.Contains(err.Error(), "dial") ||
			strings.Contains(err.Error(), "connection refused") || strings.Contains(err.Error(), "no such host") {
			t.Skipf("skipping: network unavailable: %v", err)
		}
		t.Fatalf("ImportDomain error: %v", err)
	}

	if pkg.Name != "wikidata-science" {
		t.Errorf("unexpected package name: %s", pkg.Name)
	}
	if pkg.Domain != "science" {
		t.Errorf("unexpected domain: %s", pkg.Domain)
	}
	if len(pkg.Facts) == 0 {
		t.Error("expected at least some facts from Wikidata")
	}

	t.Logf("imported %d facts and %d memories for domain 'science'", len(pkg.Facts), len(pkg.Memories))
	for i, f := range pkg.Facts {
		if i >= 5 {
			break
		}
		t.Logf("  fact: %s %s %s", f.Subject, f.Relation, f.Object)
	}
}
