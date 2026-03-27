package cognitive

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestArticleToExemplars(t *testing.T) {
	text := `Albert Einstein was a brilliant theoretical physicist who changed science forever. ` +
		`Einstein is a theoretical physicist and Nobel Prize winner. ` +
		`General relativity was created by Albert Einstein in 1915. ` +
		`Physics is used for understanding the fundamental forces of nature and energy.`

	exemplars := ArticleToExemplars("Albert Einstein", text)

	if len(exemplars) == 0 {
		t.Fatal("ArticleToExemplars returned no exemplars")
	}

	t.Logf("Found %d exemplars:", len(exemplars))
	for _, ex := range exemplars {
		t.Logf("  [%s] %q  (subj=%q obj=%q)", ex.Relation, ex.Sentence, ex.Subject, ex.Object)
	}

	// Should have at least one is_a exemplar.
	hasIsA := false
	for _, ex := range exemplars {
		if ex.Relation == RelIsA {
			hasIsA = true
			break
		}
	}
	if !hasIsA {
		t.Error("expected at least one is_a exemplar")
	}
}

func TestSentenceCorpusSaveLoad(t *testing.T) {
	corpus := NewSentenceCorpus()

	corpus.Add(SentenceExemplar{
		Sentence: "Marie Curie was born in Warsaw, Poland in 1867.",
		Subject:  "Marie Curie",
		Object:   "Warsaw",
		Relation: RelLocatedIn,
	})
	corpus.Add(SentenceExemplar{
		Sentence: "Python is a programming language.",
		Subject:  "Python",
		Object:   "programming language",
		Relation: RelIsA,
	})

	if corpus.Size() != 2 {
		t.Fatalf("expected 2 exemplars, got %d", corpus.Size())
	}

	// Save and reload.
	path := filepath.Join(t.TempDir(), "test_corpus.json")
	if err := corpus.Save(path); err != nil {
		t.Fatal(err)
	}

	corpus2 := NewSentenceCorpus()
	if err := corpus2.Load(path); err != nil {
		t.Fatal(err)
	}

	if corpus2.Size() != 2 {
		t.Fatalf("reloaded corpus has %d exemplars, expected 2", corpus2.Size())
	}

	counts := corpus2.RelationCounts()
	if counts[RelLocatedIn] != 1 {
		t.Errorf("expected 1 located_in, got %d", counts[RelLocatedIn])
	}
	if counts[RelIsA] != 1 {
		t.Errorf("expected 1 is_a, got %d", counts[RelIsA])
	}
}

func TestAdaptSentence(t *testing.T) {
	tests := []struct {
		name       string
		ex         SentenceExemplar
		targetSubj string
		targetObj  string
		wantSubj   string // must appear in output
		wantObj    string // must appear in output
	}{
		{
			name: "simple entity swap",
			ex: SentenceExemplar{
				Sentence: "Python is a programming language.",
				Subject:  "Python",
				Object:   "programming language",
				Relation: RelIsA,
			},
			targetSubj: "Go",
			targetObj:  "compiled language",
			wantSubj:   "Go",
			wantObj:    "compiled language",
		},
		{
			name: "person swap",
			ex: SentenceExemplar{
				Sentence: "Marie Curie was born in Warsaw.",
				Subject:  "Marie Curie",
				Object:   "Warsaw",
				Relation: RelLocatedIn,
			},
			targetSubj: "Albert Einstein",
			targetObj:  "Ulm",
			wantSubj:   "Albert Einstein",
			wantObj:    "Ulm",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := adaptSentence(tt.ex, tt.targetSubj, tt.targetObj)
			if !strings.Contains(result, tt.wantSubj) {
				t.Errorf("adapted sentence missing subject %q: %q", tt.wantSubj, result)
			}
			if !strings.Contains(result, tt.wantObj) {
				t.Errorf("adapted sentence missing object %q: %q", tt.wantObj, result)
			}
			if !strings.HasSuffix(result, ".") {
				t.Errorf("adapted sentence doesn't end with period: %q", result)
			}
			t.Logf("  %s → %q", tt.name, result)
		})
	}
}

func TestRetrieveSentence(t *testing.T) {
	corpus := NewSentenceCorpus()

	// Add several exemplars for the same relation.
	corpus.Add(SentenceExemplar{
		Sentence: "Python is a programming language.",
		Subject:  "Python",
		Object:   "programming language",
		Relation: RelIsA,
	})
	corpus.Add(SentenceExemplar{
		Sentence: "Gold is a chemical element.",
		Subject:  "Gold",
		Object:   "chemical element",
		Relation: RelIsA,
	})
	corpus.Add(SentenceExemplar{
		Sentence: "Tokyo is a city in Japan.",
		Subject:  "Tokyo",
		Object:   "Japan",
		Relation: RelLocatedIn,
	})

	// Retrieve for a new entity — should NOT return the same entity.
	result := corpus.RetrieveSentence("Rust", RelIsA, "systems language")
	if result == "" {
		t.Fatal("RetrieveSentence returned empty")
	}
	if !strings.Contains(result, "Rust") {
		t.Errorf("retrieved sentence should contain target subject, got: %q", result)
	}
	t.Logf("Retrieved for (Rust, is_a, systems language): %q", result)

	// No match for unused relation.
	result = corpus.RetrieveSentence("X", RelContradicts, "Y")
	if result != "" {
		t.Errorf("expected empty for unused relation, got: %q", result)
	}
}

func TestCorpusFromRealWiki(t *testing.T) {
	packDir := filepath.Join("..", "..", "packages", "wiki")
	if _, err := os.Stat(packDir); err != nil {
		t.Skip("wiki packages not found")
	}

	// Process just the first batch file to check exemplar extraction.
	batchFile := filepath.Join(packDir, "wiki-batch-0001.json")
	data, err := os.ReadFile(batchFile)
	if err != nil {
		t.Skip("wiki-batch-0001.json not found")
	}

	// Parse the batch to get article data — we can't easily get
	// original text from the batch file, so just verify the corpus
	// file loads if it exists.
	corpusFile := filepath.Join(packDir, "sentence_corpus.json")
	if _, err := os.Stat(corpusFile); err != nil {
		t.Logf("No corpus file yet — run wikiimport to generate it")
		t.Logf("Batch file size: %d bytes", len(data))
		return
	}

	corpus := NewSentenceCorpus()
	if err := corpus.Load(corpusFile); err != nil {
		t.Fatalf("failed to load corpus: %v", err)
	}

	t.Logf("Corpus loaded: %d total exemplars", corpus.Size())
	for rel, count := range corpus.RelationCounts() {
		t.Logf("  %-14s %d sentences", rel, count)
	}

	// Try retrievals across different relation types.
	tests := []struct {
		subj string
		rel  RelType
		obj  string
	}{
		{"Beethoven", RelIsA, "composer"},
		{"Tokyo", RelLocatedIn, "Japan"},
		{"DNA", RelUsedFor, "genetic research"},
		{"Einstein", RelCreatedBy, "theory of relativity"},
		{"Rome", RelPartOf, "Italy"},
		{"Pollution", RelCauses, "health problems"},
		{"Apple", RelFoundedBy, "Steve Jobs"},
		{"Go", RelIsA, "programming language"},
		{"Mars", RelHas, "two moons"},
	}
	for _, tt := range tests {
		result := corpus.RetrieveVaried(tt.subj, tt.rel, tt.obj)
		if result != "" {
			t.Logf("  (%s, %s, %s) → %q", tt.subj, tt.rel, tt.obj, result)
		} else {
			t.Logf("  (%s, %s, %s) → (no match)", tt.subj, tt.rel, tt.obj)
		}
	}
}
