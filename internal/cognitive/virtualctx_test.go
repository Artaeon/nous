package cognitive

import (
	"strings"
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

// --- Benchmark ---

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
