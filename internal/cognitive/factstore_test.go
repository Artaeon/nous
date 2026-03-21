package cognitive

import (
	"path/filepath"
	"strings"
	"testing"
)

func TestPersistentFactStoreBasic(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "facts.json")

	// Create and populate
	store := NewPersistentFactStore(path)
	store.Add(Fact{Text: "Go is fast", Topic: "go", Source: "web", Score: 0.8})
	store.Add(Fact{Text: "Rust is safe", Topic: "rust", Source: "web", Score: 0.7})

	if store.Size() != 2 {
		t.Errorf("expected 2 facts, got %d", store.Size())
	}

	if err := store.Save(); err != nil {
		t.Fatalf("save error: %v", err)
	}

	// Load in a new instance — should have the same facts
	store2 := NewPersistentFactStore(path)
	if store2.Size() != 2 {
		t.Errorf("after reload: expected 2 facts, got %d", store2.Size())
	}

	goFacts := store2.FactsAbout("go")
	if len(goFacts) != 1 {
		t.Errorf("expected 1 Go fact, got %d", len(goFacts))
	}
}

func TestPersistentFactStoreDedup(t *testing.T) {
	dir := t.TempDir()
	store := NewPersistentFactStore(filepath.Join(dir, "facts.json"))

	store.Add(Fact{Text: "Go is fast", Topic: "go", Score: 0.8})
	store.Add(Fact{Text: "Go is fast", Topic: "go", Score: 0.9}) // duplicate

	if store.Size() != 1 {
		t.Errorf("duplicates should be skipped, got %d facts", store.Size())
	}
}

func TestPersistentFactStoreTopics(t *testing.T) {
	dir := t.TempDir()
	store := NewPersistentFactStore(filepath.Join(dir, "facts.json"))

	store.Add(Fact{Text: "Go is fast", Topic: "go", Score: 0.8})
	store.Add(Fact{Text: "Rust is safe", Topic: "rust", Score: 0.7})
	store.Add(Fact{Text: "Python is easy", Topic: "python", Score: 0.6})

	topics := store.Topics()
	if len(topics) != 3 {
		t.Errorf("expected 3 topics, got %d: %v", len(topics), topics)
	}
}

func TestPersistentFactStoreCrossSession(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "facts.json")

	// Session 1: user fetches stoicera.com
	tracker1 := NewConversationTrackerPersistent(path)
	tracker1.IngestContent(
		"Stoicera is a philosophy company based in Vienna. They create tools for modern stoics.",
		"https://stoicera.com", "Stoicera",
	)

	// Session 2: new tracker loads from disk
	tracker2 := NewConversationTrackerPersistent(path)

	// Should be able to answer from persisted facts
	answer := tracker2.AnswerQuestion("what is Stoicera")
	if answer == "" {
		t.Fatal("should answer from persistent facts loaded from disk")
	}
	if !strings.Contains(strings.ToLower(answer), "philosophy") {
		t.Errorf("answer should mention philosophy, got %q", answer)
	}
	t.Logf("Cross-session answer: %s", answer)
}

func TestQuestionTypeClassification(t *testing.T) {
	tests := []struct {
		question string
		want     QuestionType
	}{
		{"who founded Stoicera", QWho},
		{"what is Go", QWhat},
		{"when was it founded", QWhen},
		{"where is the company", QWhere},
		{"why use Go", QWhy},
		{"how does it work", QHow},
		{"what products do they offer", QList},
		{"list the features", QList},
		{"tell me about Rust", QGeneral},
	}

	for _, tt := range tests {
		got := classifyQuestion(tt.question)
		if got != tt.want {
			t.Errorf("classifyQuestion(%q) = %v, want %v", tt.question, got, tt.want)
		}
	}
}
