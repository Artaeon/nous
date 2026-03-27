package cognitive

import (
	"os"
	"path/filepath"
	"testing"
)

func TestExtractDiscourseSentences(t *testing.T) {
	text := `Albert Einstein was a theoretical physicist who developed the theory of relativity. ` +
		`Einstein is considered one of the most influential scientists of all time. ` +
		`Einstein developed general relativity because Newtonian gravity could not explain Mercury's orbit. ` +
		`For example, Einstein's famous equation E=mc2 shows that mass and energy are related. ` +
		`Unlike Newton, Einstein described gravity as curvature of spacetime. ` +
		`Einstein's work led to predictions about black holes and gravitational waves. ` +
		`Einstein's theory was first confirmed in 1919 during a solar eclipse.`

	sents := ExtractDiscourseSentences("Einstein", text)

	t.Logf("Extracted %d discourse sentences:", len(sents))
	for _, ds := range sents {
		t.Logf("  [%-12s q=%d] %s", ds.Function, ds.Quality, ds.Sentence)
	}

	if len(sents) < 3 {
		t.Errorf("expected at least 3 discourse sentences, got %d", len(sents))
	}

	// Check for specific functions.
	funcFound := make(map[DiscourseFunc]bool)
	for _, ds := range sents {
		funcFound[ds.Function] = true
	}

	if !funcFound[DFEvaluates] {
		t.Error("expected an evaluates sentence ('is considered one of the most')")
	}
	if !funcFound[DFExplainsWhy] {
		t.Error("expected an explains_why sentence ('because')")
	}
	if !funcFound[DFCompares] {
		t.Error("expected a compares sentence ('unlike Newton')")
	}
}

func TestDiscourseCorpusSaveLoad(t *testing.T) {
	dc := NewDiscourseCorpus()
	dc.Add(DiscourseSentence{
		Sentence: "Python is a programming language.",
		Topic:    "Python",
		Function: DFDefines,
		Quality:  2,
	})
	dc.Add(DiscourseSentence{
		Sentence: "Python is considered one of the easiest languages to learn.",
		Topic:    "Python",
		Function: DFEvaluates,
		Quality:  3,
	})

	path := filepath.Join(t.TempDir(), "test_dc.json")
	if err := dc.Save(path); err != nil {
		t.Fatal(err)
	}

	dc2 := NewDiscourseCorpus()
	if err := dc2.Load(path); err != nil {
		t.Fatal(err)
	}

	if dc2.Size() != 2 {
		t.Fatalf("expected 2, got %d", dc2.Size())
	}

	// Test retrieval.
	result := dc2.Retrieve("Python", DFEvaluates)
	if result == "" {
		t.Error("expected to find evaluates sentence for Python")
	}
	t.Logf("Retrieved: %s", result)
}

func TestDiscourseComposeResponse(t *testing.T) {
	dc := NewDiscourseCorpus()
	dc.Add(DiscourseSentence{Sentence: "Democracy is a system of government.", Topic: "Democracy", Function: DFDefines, Quality: 2})
	dc.Add(DiscourseSentence{Sentence: "Democracy is considered the most widely adopted form of government.", Topic: "Democracy", Function: DFEvaluates, Quality: 3})
	dc.Add(DiscourseSentence{Sentence: "Unlike autocracy, democracy distributes power among citizens.", Topic: "Democracy", Function: DFCompares, Quality: 3})

	response := dc.ComposeResponse("Democracy", "opinion")
	t.Logf("Opinion response: %s", response)
	if response == "" {
		t.Error("expected non-empty response")
	}

	response2 := dc.ComposeResponse("Democracy", "what_is")
	t.Logf("What_is response: %s", response2)
}

func TestDiscourseFromRealWiki(t *testing.T) {
	discPath := filepath.Join("..", "..", "packages", "wiki", "discourse_corpus.json")
	if _, err := os.Stat(discPath); err != nil {
		t.Skip("discourse corpus not found — run wikiimport first")
	}

	dc := NewDiscourseCorpus()
	if err := dc.Load(discPath); err != nil {
		t.Fatal(err)
	}

	t.Logf("Discourse corpus: %d sentences", dc.Size())
	for fn, count := range dc.FunctionCounts() {
		t.Logf("  %-14s %d sentences", fn, count)
	}

	// Test composition for various topics.
	topics := []struct {
		topic     string
		queryType string
	}{
		{"democracy", "opinion"},
		{"electricity", "why"},
		{"einstein", "what_is"},
		{"climate change", "why"},
		{"philosophy", "opinion"},
		{"artificial intelligence", "what_is"},
	}

	for _, tt := range topics {
		resp := dc.ComposeResponse(tt.topic, tt.queryType)
		if resp != "" {
			if len(resp) > 200 {
				resp = resp[:200] + "..."
			}
			t.Logf("  %-25s [%-7s] %s", tt.topic, tt.queryType, resp)
		} else {
			t.Logf("  %-25s [%-7s] (no match)", tt.topic, tt.queryType)
		}
	}
}
