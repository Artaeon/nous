package cognitive

import (
	"path/filepath"
	"strings"
	"testing"
)

func TestPersonalGrowthCreation(t *testing.T) {
	g := NewPersonalGrowth("")
	if g == nil {
		t.Fatal("should not return nil")
	}
	stats := g.Stats()
	if stats.TotalInteractions != 0 {
		t.Error("new growth should have 0 interactions")
	}
}

func TestPersonalGrowthRecordInteraction(t *testing.T) {
	g := NewPersonalGrowth("")
	g.RecordInteraction("tell me about quantum physics")
	g.RecordInteraction("what is machine learning?")
	g.RecordInteraction("quantum entanglement explained")

	stats := g.Stats()
	if stats.TotalInteractions != 3 {
		t.Errorf("interactions = %d, want 3", stats.TotalInteractions)
	}
	if stats.TopicsTracked < 2 {
		t.Errorf("topics tracked = %d, want >= 2", stats.TopicsTracked)
	}
}

func TestPersonalGrowthTopInterests(t *testing.T) {
	g := NewPersonalGrowth("")

	// Mention physics multiple times
	for i := 0; i < 5; i++ {
		g.RecordInteraction("tell me about physics")
	}
	g.RecordInteraction("what is biology?")

	top := g.TopInterests(3)
	if len(top) < 1 {
		t.Fatal("should have at least 1 top interest")
	}
	if top[0].Name != "physics" {
		t.Errorf("top interest = %q, want physics", top[0].Name)
	}
}

func TestPersonalGrowthLearnFact(t *testing.T) {
	g := NewPersonalGrowth("")
	g.LearnFact("User is a data scientist", "work")
	g.LearnFact("User likes coffee", "preference")

	stats := g.Stats()
	if stats.FactsLearned != 2 {
		t.Errorf("facts learned = %d, want 2", stats.FactsLearned)
	}

	// Duplicate should be ignored
	g.LearnFact("User is a data scientist", "work")
	stats = g.Stats()
	if stats.FactsLearned != 2 {
		t.Errorf("after duplicate, facts = %d, want 2", stats.FactsLearned)
	}
}

func TestPersonalGrowthContextForQuery(t *testing.T) {
	g := NewPersonalGrowth("")
	g.LearnFact("User is a physicist", "work")

	// Query about physics should include the fact
	ctx := g.ContextForQuery("tell me about physics experiments")
	if !strings.Contains(ctx, "physicist") {
		t.Error("context should include relevant fact about user being a physicist")
	}

	// Unrelated query should not include it
	ctx = g.ContextForQuery("hello")
	if strings.Contains(ctx, "physicist") {
		t.Error("context should not include irrelevant facts")
	}
}

func TestPersonalGrowthContextIncludesInterests(t *testing.T) {
	g := NewPersonalGrowth("")
	for i := 0; i < 5; i++ {
		g.RecordInteraction("quantum physics is fascinating")
	}

	ctx := g.ContextForQuery("anything")
	if !strings.Contains(ctx, "interests") {
		t.Error("context should include user interests")
	}
}

func TestPersonalGrowthStyleDetection(t *testing.T) {
	g := NewPersonalGrowth("")
	g.RecordInteraction("show me an example of quantum physics")

	profile := g.Profile()
	if !profile.Style.PrefersExamples {
		t.Error("should detect example preference")
	}
}

func TestPersonalGrowthSaveLoad(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "growth.json")

	g := NewPersonalGrowth(path)
	g.RecordInteraction("quantum physics question")
	g.LearnFact("User likes physics", "interest")
	g.Save()

	// Load into new instance
	g2 := NewPersonalGrowth(path)
	stats := g2.Stats()
	if stats.TotalInteractions != 1 {
		t.Errorf("loaded interactions = %d, want 1", stats.TotalInteractions)
	}
	if stats.FactsLearned != 1 {
		t.Errorf("loaded facts = %d, want 1", stats.FactsLearned)
	}
}

func TestExtractTopics(t *testing.T) {
	topics := extractTopics("what is quantum physics and machine learning?")

	hasQuantum := false
	hasMachine := false
	for _, topic := range topics {
		if topic == "quantum" {
			hasQuantum = true
		}
		if topic == "machine" {
			hasMachine = true
		}
	}

	if !hasQuantum {
		t.Error("should extract 'quantum' as topic")
	}
	if !hasMachine {
		t.Error("should extract 'machine' as topic")
	}
}

func TestExtractTopicsBigrams(t *testing.T) {
	topics := extractTopics("explain quantum physics")

	hasBigram := false
	for _, topic := range topics {
		if topic == "quantum physics" {
			hasBigram = true
		}
	}

	if !hasBigram {
		t.Error("should extract 'quantum physics' as bigram topic")
	}
}

func TestCalculateWeight(t *testing.T) {
	// Recent, frequent topic
	w1 := calculateWeight(10, 0)
	// Old, rare topic
	w2 := calculateWeight(1, 30*24*3600e9) // 30 days in nanoseconds (but as Duration)

	if w1 <= w2 {
		t.Error("recent frequent topic should have higher weight than old rare topic")
	}
}

func TestGrowthTotalTokens(t *testing.T) {
	g := NewPersonalGrowth("")
	g.LearnFact("User is a physicist who specializes in quantum mechanics", "work")
	g.RecordInteraction("quantum physics")

	tokens := g.totalTokens()
	if tokens <= 0 {
		t.Error("should have positive token count")
	}
}

// --- Benchmark ---

func BenchmarkRecordInteraction(b *testing.B) {
	g := NewPersonalGrowth("")
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		g.RecordInteraction("tell me about quantum physics and machine learning")
	}
}

func BenchmarkContextForQuery(b *testing.B) {
	g := NewPersonalGrowth("")
	for i := 0; i < 100; i++ {
		g.RecordInteraction("quantum physics topic")
	}
	g.LearnFact("User is a physicist", "work")
	g.LearnFact("User likes coffee", "preference")

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		g.ContextForQuery("tell me about physics")
	}
}
