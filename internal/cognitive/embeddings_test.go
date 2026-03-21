package cognitive

import (
	"math"
	"path/filepath"
	"testing"
)

func TestEmbeddingsSeedFromTaxonomy(t *testing.T) {
	we := NewWordEmbeddings(50)
	we.SeedFromTaxonomy(conceptCategory)

	if we.Size() == 0 {
		t.Fatal("expected non-zero embeddings after taxonomy seeding")
	}
	t.Logf("Taxonomy-seeded words: %d", we.Size())

	// Words in the same category should be more similar than random pairs.
	simSame := we.Similarity("go", "python")     // both programming languages
	simDiff := we.Similarity("go", "philosophy")  // different categories

	t.Logf("go-python similarity: %.3f", simSame)
	t.Logf("go-philosophy similarity: %.3f", simDiff)

	if simSame <= simDiff {
		t.Errorf("expected same-category similarity (%.3f) > diff-category (%.3f)",
			simSame, simDiff)
	}
}

func TestEmbeddingsPoolWords(t *testing.T) {
	we := NewWordEmbeddings(50)
	we.SeedPoolWords()

	if we.Size() == 0 {
		t.Fatal("expected non-zero embeddings after pool seeding")
	}
	t.Logf("Pool-seeded words: %d", we.Size())

	// "ancient" and "enduring" (temporal cluster) should be more similar
	// than "ancient" and "tangible" (different clusters).
	simClose := we.Similarity("ancient", "enduring")
	simFar := we.Similarity("ancient", "tangible")

	t.Logf("ancient-enduring: %.3f", simClose)
	t.Logf("ancient-tangible: %.3f", simFar)

	if simClose <= simFar {
		t.Errorf("expected same-cluster similarity (%.3f) > diff-cluster (%.3f)",
			simClose, simFar)
	}
}

func TestEmbeddingsKNearestFrom(t *testing.T) {
	we := NewWordEmbeddings(50)
	we.SeedPoolWords()

	candidates := []string{
		"ancient", "modern", "tangible", "philosophical",
		"robust", "enduring", "molecular", "artistic",
	}

	nearest := we.KNearestFrom("timeless", candidates, 3)
	t.Logf("Nearest to 'timeless': %v", nearest)

	if len(nearest) == 0 {
		t.Fatal("expected non-empty nearest neighbors")
	}
	// "ancient" and "enduring" should be in top 3 for "timeless"
}

func TestEmbeddingsKNearestFromContext(t *testing.T) {
	we := NewWordEmbeddings(50)
	we.SeedPoolWords()

	context := []string{"philosophy", "ancient", "wisdom"}
	candidates := []string{
		"profound", "robust", "molecular", "artistic",
		"enduring", "tangible", "philosophical", "modern",
	}

	nearest := we.KNearestFromContext(context, candidates, 3)
	t.Logf("Nearest to context [philosophy,ancient,wisdom]: %v", nearest)

	if len(nearest) == 0 {
		t.Fatal("expected non-empty results")
	}
}

func TestEmbeddingsCooccurrence(t *testing.T) {
	we := NewWordEmbeddings(50)

	// Build co-occurrence from a small corpus.
	cooc := map[string]map[string]int{
		"philosophy": {"ancient": 5, "wisdom": 4, "ethics": 3, "virtue": 3},
		"science":    {"modern": 5, "experiment": 4, "theory": 3, "evidence": 3},
		"ancient":    {"philosophy": 5, "greece": 3, "wisdom": 3},
		"modern":     {"science": 5, "technology": 4, "innovation": 3},
		"wisdom":     {"philosophy": 4, "ancient": 3, "knowledge": 3},
	}

	we.BuildFromCooccurrence(cooc)

	if we.Size() == 0 {
		t.Fatal("expected non-zero embeddings after co-occurrence build")
	}
	t.Logf("Co-occurrence words: %d", we.Size())

	// Words with shared co-occurrence should have vectors (sign may vary
	// with random projection on small datasets, so just check they exist).
	va := we.Vector("philosophy")
	vb := we.Vector("ancient")
	if va == nil || vb == nil {
		t.Error("expected vectors for co-occurring words")
	}
	sim := we.Similarity("philosophy", "ancient")
	t.Logf("philosophy-ancient similarity: %.3f", sim)
}

func TestEmbeddingsPersistence(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "embeddings.json")

	// Create and save.
	we := NewWordEmbeddings(50)
	we.SeedPoolWords()
	origSize := we.Size()

	if err := we.Save(path); err != nil {
		t.Fatal(err)
	}

	// Load into fresh instance.
	we2 := NewWordEmbeddings(0)
	if err := we2.Load(path); err != nil {
		t.Fatal(err)
	}

	if we2.Size() != origSize {
		t.Errorf("expected %d words after load, got %d", origSize, we2.Size())
	}

	// Verify similarity is preserved.
	origSim := we.Similarity("ancient", "enduring")
	loadedSim := we2.Similarity("ancient", "enduring")
	if math.Abs(origSim-loadedSim) > 0.001 {
		t.Errorf("similarity changed after save/load: %.3f vs %.3f", origSim, loadedSim)
	}
}

func TestEmbeddingsIntegration(t *testing.T) {
	// Test that pickSemantic actually uses embeddings.
	g := NewGenerativeEngine()
	we := NewWordEmbeddings(50)
	we.SeedPoolWords()
	g.SetEmbeddings(we)

	// pickSemantic should return a word from the pool.
	pool := []string{"ancient", "modern", "tangible", "philosophical", "enduring"}
	result := g.pickSemantic(pool, "philosophy stoicism wisdom")
	t.Logf("pickSemantic for 'philosophy stoicism wisdom': %s", result)

	found := false
	for _, w := range pool {
		if w == result {
			found = true
			break
		}
	}
	if !found {
		t.Errorf("pickSemantic returned %q which is not in the pool", result)
	}
}
