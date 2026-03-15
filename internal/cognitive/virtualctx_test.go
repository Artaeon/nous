package cognitive

import (
	"math"
	"strings"
	"sync"
	"testing"
)

func TestVirtualContextCreation(t *testing.T) {
	vc := NewVirtualContext(1500)
	if vc == nil {
		t.Fatal("should not return nil")
	}
	if vc.TotalSize() != 0 {
		t.Error("empty virtual context should have 0 size")
	}
}

func TestVirtualContextAddSource(t *testing.T) {
	vc := NewVirtualContext(1500)

	vc.AddSource(ContextSource{
		Name:     "test",
		Type:     SourceKnowledge,
		Size:     50000,
		Priority: 70,
		Query: func(query string, budget int) []ContextSlice {
			return []ContextSlice{{
				Source:    "test",
				Content:  "test content about " + query,
				Tokens:   10,
				Relevance: 0.9,
			}}
		},
	})

	if vc.TotalSize() != 50000 {
		t.Errorf("total size = %d, want 50000", vc.TotalSize())
	}
	if vc.SourceCount() != 1 {
		t.Errorf("source count = %d, want 1", vc.SourceCount())
	}
}

func TestVirtualContextWeaveEmpty(t *testing.T) {
	vc := NewVirtualContext(1500)
	assembly := vc.Weave("test query")
	if assembly == nil {
		t.Fatal("assembly should not be nil")
	}
	if len(assembly.Slices) != 0 {
		t.Error("empty context should produce no slices")
	}
}

func TestVirtualContextWeave(t *testing.T) {
	vc := NewVirtualContext(1500)

	vc.AddSource(ContextSource{
		Name:     "knowledge",
		Type:     SourceKnowledge,
		Size:     100000,
		Priority: 70,
		Query: func(query string, budget int) []ContextSlice {
			return []ContextSlice{
				{Source: "knowledge", Content: "Physics: quantum mechanics studies particles", Tokens: 10, Relevance: 0.9},
				{Source: "knowledge", Content: "History: Rome fell in 476 AD", Tokens: 8, Relevance: 0.5},
			}
		},
	})

	vc.AddSource(ContextSource{
		Name:     "personal",
		Type:     SourcePersonal,
		Size:     500,
		Priority: 80,
		Query: func(query string, budget int) []ContextSlice {
			return []ContextSlice{
				{Source: "personal", Content: "User is interested in physics", Tokens: 6, Relevance: 0.8},
			}
		},
	})

	assembly := vc.Weave("tell me about quantum physics")

	if len(assembly.Slices) < 2 {
		t.Errorf("should weave at least 2 slices, got %d", len(assembly.Slices))
	}

	// Highest relevance should be first
	if assembly.Slices[0].Relevance < assembly.Slices[1].Relevance {
		t.Error("slices should be sorted by relevance")
	}

	if assembly.SourcesUsed < 2 {
		t.Errorf("should use 2 sources, used %d", assembly.SourcesUsed)
	}

	if assembly.VirtualSize != 100500 {
		t.Errorf("virtual size = %d, want 100500", assembly.VirtualSize)
	}
}

func TestVirtualContextWeaveRespectsRelevanceThreshold(t *testing.T) {
	vc := NewVirtualContext(1500)

	vc.AddSource(ContextSource{
		Name: "low", Type: SourceKnowledge, Size: 1000, Priority: 50,
		Query: func(query string, budget int) []ContextSlice {
			return []ContextSlice{
				{Source: "low", Content: "irrelevant stuff", Tokens: 5, Relevance: 0.1},
			}
		},
	})

	assembly := vc.Weave("test")
	if len(assembly.Slices) != 0 {
		t.Error("should filter out low-relevance slices (< 0.2)")
	}
}

func TestVirtualContextFormatForPrompt(t *testing.T) {
	assembly := &ContextAssembly{
		Slices: []ContextSlice{
			{Source: "knowledge", Content: "Quantum physics studies particles."},
			{Source: "personal", Content: "User likes physics."},
		},
	}

	text := assembly.FormatForPrompt()
	if !strings.Contains(text, "Quantum") {
		t.Error("formatted prompt should contain knowledge content")
	}
	if !strings.Contains(text, "User likes") {
		t.Error("formatted prompt should contain personal content")
	}
}

func TestVirtualContextRecordSuccess(t *testing.T) {
	vc := NewVirtualContext(1500)
	vc.AddSource(ContextSource{Name: "test", Size: 100, Priority: 50})

	vc.RecordSuccess("test")
	vc.RecordSuccess("test")

	stats := vc.Stats()
	if len(stats.SourceDetails) != 1 {
		t.Fatal("should have 1 source detail")
	}
	if stats.SourceDetails[0].Successes != 2 {
		t.Errorf("successes = %d, want 2", stats.SourceDetails[0].Successes)
	}
}

func TestVirtualContextStats(t *testing.T) {
	vc := NewVirtualContext(1500)

	vc.AddSource(ContextSource{Name: "knowledge", Size: 500000, Type: SourceKnowledge})
	vc.AddSource(ContextSource{Name: "personal", Size: 2000, Type: SourcePersonal})

	stats := vc.Stats()
	if stats.VirtualTokens != 502000 {
		t.Errorf("virtual tokens = %d, want 502000", stats.VirtualTokens)
	}
	if stats.TotalSources != 2 {
		t.Errorf("total sources = %d, want 2", stats.TotalSources)
	}

	formatted := stats.FormatStats()
	if !strings.Contains(formatted, "502.0K") {
		t.Errorf("should format as 502.0K, got: %s", formatted)
	}
}

func TestFormatTokenCount(t *testing.T) {
	tests := []struct {
		input int
		want  string
	}{
		{500, "500"},
		{1500, "1.5K"},
		{50000, "50.0K"},
		{1500000, "1.5M"},
	}
	for _, tt := range tests {
		got := formatTokenCount(tt.input)
		if got != tt.want {
			t.Errorf("formatTokenCount(%d) = %q, want %q", tt.input, got, tt.want)
		}
	}
}

func TestKnowledgeSource(t *testing.T) {
	kv := NewKnowledgeVec(mockEmbed, "")
	kv.AddChunk("Quantum physics studies subatomic particles.", "physics")
	kv.AddChunk("Machine learning uses neural networks.", "cs")

	source := KnowledgeSource(kv)
	if source.Name != "knowledge" {
		t.Errorf("source name = %q, want knowledge", source.Name)
	}
	if source.Size != 200 { // 2 chunks * 100 tokens
		t.Errorf("source size = %d, want 200", source.Size)
	}

	slices := source.Query("quantum particles", 500)
	if len(slices) == 0 {
		t.Error("should return slices for matching query")
	}
}

func TestSortSlicesByRelevance(t *testing.T) {
	slices := []ContextSlice{
		{Content: "low", Relevance: 0.3},
		{Content: "high", Relevance: 0.9},
		{Content: "mid", Relevance: 0.6},
	}
	sortSlicesByRelevance(slices)

	if slices[0].Content != "high" {
		t.Error("highest relevance should be first")
	}
	if slices[2].Content != "low" {
		t.Error("lowest relevance should be last")
	}
}

func TestUpdateSourceSize(t *testing.T) {
	vc := NewVirtualContext(1500)
	vc.AddSource(ContextSource{Name: "test", Size: 100})

	vc.UpdateSourceSize("test", 5000)

	if vc.TotalSize() != 5000 {
		t.Errorf("after update, total size = %d, want 5000", vc.TotalSize())
	}
}

func TestVirtualContextRecordFailure(t *testing.T) {
	vc := NewVirtualContext(1500)
	vc.AddSource(ContextSource{Name: "bad", Size: 100, Priority: 50})

	vc.RecordFailure("bad")
	vc.RecordFailure("bad")

	stats := vc.Stats()
	if stats.SourceDetails[0].Failures != 2 {
		t.Errorf("failures = %d, want 2", stats.SourceDetails[0].Failures)
	}
}

func TestVirtualContextQualityEMA(t *testing.T) {
	vc := NewVirtualContext(1500)
	vc.AddSource(ContextSource{Name: "test", Size: 100, Priority: 50})

	// Initial success → quality should be 1.0
	vc.RecordSuccess("test")
	quality := vc.sourceQuality["test"]
	if quality != 1.0 {
		t.Errorf("initial quality = %f, want 1.0", quality)
	}

	// Then a failure → quality should drop but not to 0
	vc.RecordFailure("test")
	quality = vc.sourceQuality["test"]
	if quality >= 1.0 {
		t.Error("quality should drop after failure")
	}
	if quality <= 0 {
		t.Error("quality should not reach zero after one failure")
	}

	// Many failures → quality should approach 0
	for i := 0; i < 50; i++ {
		vc.RecordFailure("test")
	}
	quality = vc.sourceQuality["test"]
	if quality > 0.1 {
		t.Errorf("quality should be very low after many failures, got %f", quality)
	}
}

func TestVirtualContextQualityAffectsBudget(t *testing.T) {
	vc := NewVirtualContext(1500)

	vc.AddSource(ContextSource{Name: "good", Size: 10000, Priority: 50,
		Query: func(q string, b int) []ContextSlice {
			return []ContextSlice{{Source: "good", Content: "good stuff", Tokens: 10, Relevance: 0.8}}
		},
	})
	vc.AddSource(ContextSource{Name: "bad", Size: 10000, Priority: 50,
		Query: func(q string, b int) []ContextSlice {
			return []ContextSlice{{Source: "bad", Content: "bad stuff", Tokens: 10, Relevance: 0.8}}
		},
	})

	// Make "good" high quality and "bad" low quality
	for i := 0; i < 20; i++ {
		vc.RecordSuccess("good")
		vc.RecordFailure("bad")
	}

	// Check that budgets reflect quality
	sources := make([]ContextSource, len(vc.sources))
	copy(sources, vc.sources)
	budgets := vc.allocateBudgets(sources)

	if budgets[0] <= budgets[1] {
		t.Errorf("good source (budget=%d) should get more than bad source (budget=%d)", budgets[0], budgets[1])
	}
}

func TestSourceHealthReport(t *testing.T) {
	vc := NewVirtualContext(1500)
	vc.AddSource(ContextSource{Name: "knowledge", Size: 50000, Priority: 70})

	vc.RecordSuccess("knowledge")
	vc.RecordSuccess("knowledge")
	vc.RecordFailure("knowledge")

	report := vc.SourceHealthReport()
	if len(report) != 1 {
		t.Fatalf("expected 1 source in report, got %d", len(report))
	}
	if report[0].Successes != 2 {
		t.Errorf("successes = %d, want 2", report[0].Successes)
	}
	if report[0].Failures != 1 {
		t.Errorf("failures = %d, want 1", report[0].Failures)
	}
	if report[0].Quality <= 0 {
		t.Error("quality should be positive")
	}
}

func TestRecordQuality(t *testing.T) {
	vc := NewVirtualContext(1500)
	vc.RecordQuality("test", 0.8)
	if vc.sourceQuality["test"] != 0.8 {
		t.Errorf("initial quality = %f, want 0.8", vc.sourceQuality["test"])
	}
	vc.RecordQuality("test", 0.6)
	// EMA: 0.1*0.6 + 0.9*0.8 = 0.06 + 0.72 = 0.78
	expected := 0.78
	actual := vc.sourceQuality["test"]
	if actual < expected-0.01 || actual > expected+0.01 {
		t.Errorf("EMA quality = %f, want ~%f", actual, expected)
	}
}

// --- Benchmark ---

// --- Race Condition Tests ---

func TestVirtualContextConcurrentAccess(t *testing.T) {
	vc := NewVirtualContext(1500)
	vc.AddSource(ContextSource{
		Name: "test-src", Type: SourceKnowledge, Size: 50000, Priority: 70,
		Query: func(query string, budget int) []ContextSlice {
			return []ContextSlice{{Source: "test-src", Content: "content about " + query, Tokens: 10, Relevance: 0.8}}
		},
	})

	var wg sync.WaitGroup

	// RecordSuccess goroutines
	for g := 0; g < 5; g++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := 0; i < 50; i++ {
				vc.RecordSuccess("test-src")
			}
		}()
	}

	// RecordFailure goroutines
	for g := 0; g < 5; g++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := 0; i < 50; i++ {
				vc.RecordFailure("test-src")
			}
		}()
	}

	// RecordQuality goroutines
	for g := 0; g < 5; g++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := 0; i < 50; i++ {
				vc.RecordQuality("test-src", 0.8)
			}
		}()
	}

	// SourceHealthReport goroutines
	for g := 0; g < 5; g++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := 0; i < 50; i++ {
				_ = vc.SourceHealthReport()
			}
		}()
	}

	// Weave goroutines
	for g := 0; g < 5; g++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := 0; i < 20; i++ {
				a := vc.Weave("concurrent query")
				if a == nil {
					t.Error("Weave should never return nil")
				}
			}
		}()
	}

	wg.Wait()

	if vc.SourceCount() != 1 {
		t.Errorf("should still have 1 source, got %d", vc.SourceCount())
	}
}

// --- Formula Verification Tests ---

func TestQualityEMAFormula(t *testing.T) {
	vc := NewVirtualContext(1500)

	// EMA: new = alpha*value + (1-alpha)*old, alpha=0.1
	// First value sets quality directly
	vc.RecordQuality("src", 0.5)
	if math.Abs(vc.sourceQuality["src"]-0.5) > 1e-10 {
		t.Errorf("first quality should be 0.5, got %f", vc.sourceQuality["src"])
	}

	// Second: 0.1*0.9 + 0.9*0.5 = 0.54
	vc.RecordQuality("src", 0.9)
	expected := 0.1*0.9 + 0.9*0.5
	if math.Abs(vc.sourceQuality["src"]-expected) > 1e-10 {
		t.Errorf("second quality should be %f, got %f", expected, vc.sourceQuality["src"])
	}

	// Third: 0.1*0.2 + 0.9*0.54 = 0.506
	prev := expected
	vc.RecordQuality("src", 0.2)
	expected = 0.1*0.2 + 0.9*prev
	if math.Abs(vc.sourceQuality["src"]-expected) > 1e-10 {
		t.Errorf("third quality should be %f, got %f", expected, vc.sourceQuality["src"])
	}

	// Independent sources don't interfere
	vc.RecordQuality("other", 1.0)
	if math.Abs(vc.sourceQuality["other"]-1.0) > 1e-10 {
		t.Errorf("independent source should be 1.0, got %f", vc.sourceQuality["other"])
	}
	if math.Abs(vc.sourceQuality["src"]-expected) > 1e-10 {
		t.Errorf("original source should be unchanged at %f, got %f", expected, vc.sourceQuality["src"])
	}
}

func BenchmarkVirtualContextWeave(b *testing.B) {
	vc := NewVirtualContext(1500)
	for i := 0; i < 5; i++ {
		name := formatTokenCount(i)
		vc.AddSource(ContextSource{
			Name: name, Size: 100000, Priority: 50 + i*10,
			Query: func(query string, budget int) []ContextSlice {
				return []ContextSlice{{
					Source: name, Content: "content " + query,
					Tokens: 20, Relevance: 0.7,
				}}
			},
		})
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		vc.Weave("test query")
	}
}
