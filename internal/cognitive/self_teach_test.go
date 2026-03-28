package cognitive

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// -----------------------------------------------------------------------
// Self-Teach Tests
// -----------------------------------------------------------------------

// helper: create a temp knowledge directory with sample .txt files.
func setupKnowledgeDir(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()

	science := `Gravity is the fundamental force of attraction between all objects with mass. It was described by Isaac Newton as a universal force, and later reinterpreted by Albert Einstein in general relativity.

Quantum mechanics is the branch of physics describing the behavior of matter at atomic scales. It was developed in the early twentieth century by Planck, Heisenberg, and Schrodinger.

Thermodynamics is the study of heat, energy, work, and their transformations in physical systems.`

	tech := `Go is a compiled programming language created by Google. It is used for building web servers, command-line tools, and distributed systems.

Python is a high-level programming language. It was created by Guido van Rossum. Python is used for data science, automation, and web development.`

	os.WriteFile(filepath.Join(dir, "science.txt"), []byte(science), 0644)
	os.WriteFile(filepath.Join(dir, "tech.txt"), []byte(tech), 0644)
	return dir
}

func TestSelfTeach_SearchKnowledge(t *testing.T) {
	dir := setupKnowledgeDir(t)
	graph := NewCognitiveGraph("")
	st := NewSelfTeach(dir, graph)

	// Should find paragraphs about gravity.
	results := st.SearchKnowledge("gravity")
	if len(results) == 0 {
		t.Fatal("expected at least 1 paragraph about gravity")
	}
	if !strings.Contains(strings.ToLower(results[0]), "gravity") {
		t.Errorf("first result should mention gravity: %s", results[0])
	}

	// Should find paragraphs about Python.
	results = st.SearchKnowledge("Python")
	if len(results) == 0 {
		t.Fatal("expected at least 1 paragraph about Python")
	}

	// Non-existent topic should return nothing.
	results = st.SearchKnowledge("blockchain")
	if len(results) != 0 {
		t.Errorf("expected no results for 'blockchain', got %d", len(results))
	}
}

func TestSelfTeach_LearnAbout(t *testing.T) {
	dir := setupKnowledgeDir(t)
	graph := NewCognitiveGraph("")
	st := NewSelfTeach(dir, graph)

	n, err := st.LearnAbout("Python")
	if err != nil {
		t.Fatalf("LearnAbout error: %v", err)
	}
	if n == 0 {
		t.Fatal("expected at least 1 fact learned about Python")
	}

	// The graph should have nodes and edges related to Python.
	if graph.NodeCount() == 0 {
		t.Error("graph should have nodes after learning")
	}
	if graph.EdgeCount() == 0 {
		t.Error("graph should have edges after learning")
	}

	t.Logf("learned %d facts about Python, graph: %d nodes, %d edges",
		n, graph.NodeCount(), graph.EdgeCount())
}

func TestSelfTeach_HasLearned(t *testing.T) {
	dir := setupKnowledgeDir(t)
	graph := NewCognitiveGraph("")
	st := NewSelfTeach(dir, graph)

	if st.HasLearned("Python") {
		t.Error("should not be learned yet")
	}

	st.LearnAbout("Python")

	if !st.HasLearned("Python") {
		t.Error("should be learned after LearnAbout")
	}

	// Second call should be a no-op.
	n, _ := st.LearnAbout("Python")
	if n != 0 {
		t.Errorf("second LearnAbout should return 0, got %d", n)
	}
}

func TestSelfTeach_LearnAbout_EmptyTopic(t *testing.T) {
	dir := setupKnowledgeDir(t)
	graph := NewCognitiveGraph("")
	st := NewSelfTeach(dir, graph)

	n, err := st.LearnAbout("")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if n != 0 {
		t.Errorf("empty topic should learn 0 facts, got %d", n)
	}
}

func TestSelfTeach_SearchKnowledge_MaxParagraphs(t *testing.T) {
	// Create a knowledge dir with many matching paragraphs.
	dir := t.TempDir()
	var paras []string
	for i := 0; i < 10; i++ {
		paras = append(paras, "Gravity is important. This is paragraph about gravity.")
	}
	os.WriteFile(filepath.Join(dir, "many.txt"), []byte(strings.Join(paras, "\n\n")), 0644)

	graph := NewCognitiveGraph("")
	st := NewSelfTeach(dir, graph)

	results := st.SearchKnowledge("gravity")
	if len(results) > 3 {
		t.Errorf("expected at most 3 paragraphs, got %d", len(results))
	}
}

func TestExtractSimpleFacts(t *testing.T) {
	para := "Python is a high-level programming language. It was created by Guido van Rossum. Python is used for data science, automation, and web development."

	facts := extractSimpleFacts(para, "Python")
	if len(facts) == 0 {
		t.Fatal("expected at least 1 fact extracted from paragraph")
	}

	// Check that we got an IsA fact.
	foundIsA := false
	for _, f := range facts {
		if f.relation == RelIsA && strings.EqualFold(f.subject, "Python") {
			foundIsA = true
		}
		t.Logf("fact: %s -[%s]-> %s", f.subject, f.relation, f.object)
	}
	if !foundIsA {
		t.Error("expected an IsA fact for Python")
	}
}
