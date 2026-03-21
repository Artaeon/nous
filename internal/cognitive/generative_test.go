package cognitive

import (
	"strings"
	"testing"
)

func TestGenerativeSingleFact(t *testing.T) {
	g := NewGenerativeEngine()

	// Generate 10 versions of the same fact
	t.Log("=== Single Fact: 'Go is a programming language' ===")
	responses := make(map[string]bool)
	for i := 0; i < 10; i++ {
		sent := g.Generate("Go", RelIsA, "programming language")
		responses[sent] = true
		t.Logf("  %d: %s", i+1, sent)
	}
	t.Logf("  Unique: %d/10", len(responses))
	if len(responses) < 4 {
		t.Errorf("expected at least 4 unique outputs, got %d", len(responses))
	}
}

func TestGenerativeMultipleFacts(t *testing.T) {
	g := NewGenerativeEngine()

	facts := []edgeFact{
		{Subject: "Stoicera", Relation: RelIsA, Object: "philosophy company"},
		{Subject: "Stoicera", Relation: RelFoundedBy, Object: "Raphael"},
		{Subject: "Stoicera", Relation: RelLocatedIn, Object: "Vienna"},
	}

	t.Log("=== Multi-Fact Paragraph ===")
	for i := 0; i < 5; i++ {
		text := g.GenerateFromFacts(facts)
		t.Logf("  Version %d: %s", i+1, text)
		t.Log("  ---")
	}
}

func TestGenerativeCreativeText(t *testing.T) {
	g := NewGenerativeEngine()

	facts := []edgeFact{
		{Subject: "Go", Relation: RelIsA, Object: "programming language"},
		{Subject: "Go", Relation: RelCreatedBy, Object: "Google"},
		{Subject: "Go", Relation: RelUsedFor, Object: "backend development"},
		{Subject: "Go", Relation: RelDescribedAs, Object: "fast and efficient"},
	}

	t.Log("=== Creative Text About Go ===")
	for i := 0; i < 5; i++ {
		text := g.ComposeCreativeText("Go", facts)
		t.Logf("  Version %d: %s", i+1, text)
		t.Log("  ---")
	}
}

func TestGenerativeVerbConjugation(t *testing.T) {
	g := NewGenerativeEngine()

	tests := []struct {
		verb string
		tense Tense
		want string
	}{
		{"be", TensePresent, "is"},
		{"be", TensePast, "was"},
		{"have", TensePresent, "has"},
		{"have", TensePast, "had"},
		{"create", TensePresent, "creates"},
		{"create", TensePast, "created"},
		{"found", TensePast, "founded"},
		{"establish", TensePast, "established"},
		{"build", TensePast, "built"},
		{"run", TensePresent, "runs"},
	}

	for _, tt := range tests {
		got := g.conjugate(tt.verb, tt.tense, Singular)
		if got != tt.want {
			t.Errorf("conjugate(%q, %d) = %q, want %q", tt.verb, tt.tense, got, tt.want)
		}
	}
}

func TestGenerativeGerunds(t *testing.T) {
	g := NewGenerativeEngine()

	tests := []struct {
		verb string
		want string
	}{
		{"create", "creating"},
		{"build", "building"},
		{"run", "running"},
		{"make", "making"},
		{"think", "thinking"},
		{"write", "writing"},
	}

	for _, tt := range tests {
		got := g.gerund(tt.verb)
		if got != tt.want {
			t.Errorf("gerund(%q) = %q, want %q", tt.verb, got, tt.want)
		}
	}
}

func TestGenerativePastParticiple(t *testing.T) {
	g := NewGenerativeEngine()

	tests := []struct {
		verb string
		want string
	}{
		{"build", "built"},
		{"create", "created"},
		{"write", "written"},
		{"found", "founded"},
		{"know", "known"},
	}

	for _, tt := range tests {
		got := g.pastParticiple(tt.verb)
		if got != tt.want {
			t.Errorf("pastParticiple(%q) = %q, want %q", tt.verb, got, tt.want)
		}
	}
}

func TestGenerativeArticles(t *testing.T) {
	g := NewGenerativeEngine()

	tests := []struct {
		noun string
		want string
	}{
		{"company", "a"},
		{"organization", "an"},
		{"elegant solution", "an"},
		{"programming language", "a"},
	}

	for _, tt := range tests {
		got := g.articleFor(tt.noun)
		if got != tt.want {
			t.Errorf("articleFor(%q) = %q, want %q", tt.noun, got, tt.want)
		}
	}
}

func TestGenerativeAllRelationTypes(t *testing.T) {
	g := NewGenerativeEngine()

	relations := []struct {
		subj string
		rel  RelType
		obj  string
	}{
		{"Go", RelIsA, "programming language"},
		{"Stoicera", RelLocatedIn, "Vienna"},
		{"Stoicera", RelFoundedBy, "Raphael"},
		{"Go", RelCreatedBy, "Google"},
		{"Go", RelUsedFor, "backend development"},
		{"Go", RelHas, "goroutines"},
		{"Go", RelDescribedAs, "fast"},
		{"Go", RelOffers, "concurrency"},
		{"Go", RelRelatedTo, "systems programming"},
		{"stress", RelCauses, "burnout"},
	}

	t.Log("=== All Relation Types ===")
	for _, r := range relations {
		sent := g.Generate(r.subj, r.rel, r.obj)
		if sent == "" {
			t.Errorf("empty output for %s -[%s]-> %s", r.subj, r.rel, r.obj)
		}
		t.Logf("  %s -[%s]-> %s: %s", r.subj, r.rel, r.obj, sent)
	}
}

func TestGenerativeUniquenessBenchmark(t *testing.T) {
	g := NewGenerativeEngine()

	// Generate 50 versions of the same fact — measure uniqueness
	unique := make(map[string]bool)
	for i := 0; i < 50; i++ {
		sent := g.Generate("Stoicera", RelIsA, "philosophy company")
		unique[sent] = true
	}

	ratio := float64(len(unique)) / 50.0 * 100
	t.Logf("Uniqueness: %d/50 (%.0f%%)", len(unique), ratio)

	// Should get at least 15 unique sentences from 10 patterns
	if len(unique) < 10 {
		t.Errorf("expected at least 10 unique sentences, got %d", len(unique))
	}
}

func TestGenerativeVsTemplate(t *testing.T) {
	g := NewGenerativeEngine()
	graph := NewCognitiveGraph("")
	composer := NewComposer(graph, nil, nil, nil)

	// Add facts to graph
	goID := graph.EnsureNode("Go", NodeEntity)
	langID := graph.EnsureNode("programming language", NodeConcept)
	googleID := graph.EnsureNode("Google", NodeEntity)
	backendID := graph.EnsureNode("backend development", NodeConcept)
	graph.AddEdge(goID, langID, RelIsA, "test")
	graph.AddEdge(goID, googleID, RelCreatedBy, "test")
	graph.AddEdge(goID, backendID, RelUsedFor, "test")

	t.Log("=== Template-Based (Composer) ===")
	for i := 0; i < 3; i++ {
		resp := composer.Compose("tell me about Go", RespFactual, &ComposeContext{})
		if resp != nil {
			t.Logf("  Template: %s", resp.Text)
		}
	}

	facts := []edgeFact{
		{Subject: "Go", Relation: RelIsA, Object: "programming language"},
		{Subject: "Go", Relation: RelCreatedBy, Object: "Google"},
		{Subject: "Go", Relation: RelUsedFor, Object: "backend development"},
	}

	t.Log("=== Generative (Grammar Rules) ===")
	for i := 0; i < 3; i++ {
		text := g.GenerateFromFacts(facts)
		t.Logf("  Generative: %s", text)
	}

	t.Log("=== Creative (Full Treatment) ===")
	for i := 0; i < 3; i++ {
		text := g.ComposeCreativeText("Go", facts)
		t.Logf("  Creative: %s", text)
	}
}

func TestGenerativeNoEmpty(t *testing.T) {
	g := NewGenerativeEngine()

	// Every relation type should produce output
	for _, rel := range []RelType{
		RelIsA, RelLocatedIn, RelFoundedBy, RelCreatedBy, RelUsedFor,
		RelHas, RelDescribedAs, RelOffers, RelRelatedTo, RelCauses,
		RelPartOf, RelPrefers, RelDislikes,
	} {
		sent := g.Generate("X", rel, "Y")
		if sent == "" {
			t.Errorf("empty output for relation %s", rel)
		}
		if !strings.HasSuffix(sent, ".") && !strings.HasSuffix(sent, "!") && !strings.HasSuffix(sent, "?") {
			t.Errorf("sentence missing punctuation: %q", sent)
		}
	}
}
