package cognitive

import (
	"path/filepath"
	"strings"
	"testing"
)

// --- Neuroplastic Description Tests ---

func TestNeuroplasticRegisterDefault(t *testing.T) {
	nd := NewNeuroplasticDescriptions("qwen2.5:1.5b", "")
	nd.RegisterDefault("grep", "Search file contents for a regex pattern")

	desc := nd.GetDescription("grep")
	if desc != "Search file contents for a regex pattern" {
		t.Errorf("got %q, want default description", desc)
	}
}

func TestNeuroplasticRegisterDefaultNoDuplicate(t *testing.T) {
	nd := NewNeuroplasticDescriptions("qwen2.5:1.5b", "")
	nd.RegisterDefault("grep", "version 1")
	nd.RegisterDefault("grep", "version 2") // should not overwrite

	desc := nd.GetDescription("grep")
	if desc != "version 1" {
		t.Errorf("second register should not overwrite, got %q", desc)
	}
}

func TestNeuroplasticGetDescriptionMissing(t *testing.T) {
	nd := NewNeuroplasticDescriptions("qwen2.5:1.5b", "")
	desc := nd.GetDescription("nonexistent")
	if desc != "" {
		t.Errorf("missing tool should return empty, got %q", desc)
	}
}

func TestNeuroplasticRecordAttemptAndSuccess(t *testing.T) {
	nd := NewNeuroplasticDescriptions("qwen2.5:1.5b", "")
	nd.RegisterDefault("grep", "search files")

	nd.RecordAttempt("grep")
	nd.RecordAttempt("grep")
	nd.RecordSuccess("grep")

	stats := nd.Stats()
	gs, ok := stats["grep"]
	if !ok {
		t.Fatal("grep should be in stats")
	}
	if gs.Attempts != 2 {
		t.Errorf("attempts = %d, want 2", gs.Attempts)
	}
	if gs.Successes != 1 {
		t.Errorf("successes = %d, want 1", gs.Successes)
	}
	if gs.SuccessRate != 0.5 {
		t.Errorf("success rate = %f, want 0.5", gs.SuccessRate)
	}
}

func TestNeuroplasticAddVariant(t *testing.T) {
	nd := NewNeuroplasticDescriptions("qwen2.5:1.5b", "")
	nd.RegisterDefault("grep", "version 1")
	nd.AddVariant("grep", "version 2")

	stats := nd.Stats()
	if stats["grep"].Variants != 2 {
		t.Errorf("variants = %d, want 2", stats["grep"].Variants)
	}
}

func TestNeuroplasticAddVariantNoDuplicate(t *testing.T) {
	nd := NewNeuroplasticDescriptions("qwen2.5:1.5b", "")
	nd.RegisterDefault("grep", "version 1")
	nd.AddVariant("grep", "version 1") // duplicate

	stats := nd.Stats()
	if stats["grep"].Variants != 1 {
		t.Errorf("duplicate variant should not be added, got %d", stats["grep"].Variants)
	}
}

func TestNeuroplasticEvolve(t *testing.T) {
	nd := NewNeuroplasticDescriptions("qwen2.5:1.5b", "")
	nd.RegisterDefault("grep", "bad description")
	nd.AddVariant("grep", "good description")

	// Make the default bad (2/10 success)
	for i := 0; i < 10; i++ {
		nd.RecordAttempt("grep")
	}
	nd.RecordSuccess("grep")
	nd.RecordSuccess("grep")

	// Switch to variant and make it good (8/10 success)
	nd.mu.Lock()
	nd.entries["grep"].ActiveIdx = 1
	nd.mu.Unlock()

	for i := 0; i < 10; i++ {
		nd.RecordAttempt("grep")
	}
	for i := 0; i < 8; i++ {
		nd.RecordSuccess("grep")
	}

	// Switch back to bad to test evolution
	nd.mu.Lock()
	nd.entries["grep"].ActiveIdx = 0
	nd.mu.Unlock()

	changes := nd.Evolve()
	if len(changes) == 0 {
		t.Error("evolve should promote better variant")
	}

	desc := nd.GetDescription("grep")
	if desc != "good description" {
		t.Errorf("after evolve, active should be better variant, got %q", desc)
	}
}

func TestNeuroplasticEvolveNoChange(t *testing.T) {
	nd := NewNeuroplasticDescriptions("qwen2.5:1.5b", "")
	nd.RegisterDefault("grep", "only variant")

	changes := nd.Evolve()
	if len(changes) != 0 {
		t.Error("evolve with single variant should not change anything")
	}
}

func TestNeuroplasticGenerateVariants(t *testing.T) {
	nd := NewNeuroplasticDescriptions("qwen2.5:1.5b", "")
	nd.RegisterDefault("grep", "Search file contents for a regex pattern")

	// Simulate poor performance (3/15 success)
	for i := 0; i < 15; i++ {
		nd.RecordAttempt("grep")
	}
	for i := 0; i < 3; i++ {
		nd.RecordSuccess("grep")
	}

	evolved := nd.GenerateVariants()
	if len(evolved) == 0 {
		t.Error("should generate variants for underperforming tool")
	}

	stats := nd.Stats()
	if stats["grep"].Variants <= 1 {
		t.Error("should have more than 1 variant after generation")
	}
}

func TestNeuroplasticGenerateVariantsNotNeeded(t *testing.T) {
	nd := NewNeuroplasticDescriptions("qwen2.5:1.5b", "")
	nd.RegisterDefault("grep", "Search files")

	// Simulate good performance (9/10 success)
	for i := 0; i < 10; i++ {
		nd.RecordAttempt("grep")
		nd.RecordSuccess("grep")
	}
	nd.RecordAttempt("grep") // 1 failure

	evolved := nd.GenerateVariants()
	if len(evolved) != 0 {
		t.Error("should not generate variants for well-performing tool")
	}
}

func TestNeuroplasticPersistence(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "neuroplastic.json")

	nd1 := NewNeuroplasticDescriptions("qwen2.5:1.5b", path)
	nd1.RegisterDefault("grep", "search files")
	nd1.RecordAttempt("grep")
	nd1.RecordSuccess("grep")

	// Force save
	nd1.mu.Lock()
	nd1.save()
	nd1.mu.Unlock()

	nd2 := NewNeuroplasticDescriptions("qwen2.5:1.5b", path)
	desc := nd2.GetDescription("grep")
	if desc != "search files" {
		t.Errorf("loaded description = %q, want 'search files'", desc)
	}
}

// --- Description Enhancement Tests ---

func TestAddExampleToDesc(t *testing.T) {
	result := addExampleToDesc("grep", "Search file contents")
	if result == "" {
		t.Error("should add example to grep")
	}
	if result == "Search file contents" {
		t.Error("should modify the description")
	}
}

func TestAddExampleAlreadyHas(t *testing.T) {
	result := addExampleToDesc("grep", "Search files. Example: grep('TODO')")
	if result != "" {
		t.Error("should not add duplicate example")
	}
}

func TestMakeImperative(t *testing.T) {
	result := makeImperative("grep", "Search file contents")
	if !strings.HasPrefix(result, "USE THIS") {
		t.Errorf("imperative should start with 'USE THIS', got %q", result)
	}
}

func TestSimplifyDesc(t *testing.T) {
	long := "Search file contents for a regex pattern. Args: pattern (required), path (optional directory), glob (optional file filter like '*.go')."
	result := simplifyDesc("grep", long)
	if result == "" {
		t.Error("should simplify long description")
	}
	if len(result) >= len(long) {
		t.Error("simplified should be shorter")
	}
}

func TestTruncateDesc(t *testing.T) {
	if got := truncateDesc("short", 20); got != "short" {
		t.Errorf("short string should not be truncated, got %q", got)
	}
	if got := truncateDesc("this is a very long description", 10); len(got) > 10 {
		t.Errorf("long string should be truncated to %d, got len %d", 10, len(got))
	}
}

func TestSuccessRate(t *testing.T) {
	v := DescVariant{Attempts: 0}
	if v.SuccessRate() != 0.5 {
		t.Errorf("zero attempts should return 0.5 (neutral prior), got %f", v.SuccessRate())
	}

	v = DescVariant{Attempts: 10, Successes: 7}
	if v.SuccessRate() != 0.7 {
		t.Errorf("7/10 should be 0.7, got %f", v.SuccessRate())
	}
}

// --- Benchmark ---

func BenchmarkNeuroplasticGetDescription(b *testing.B) {
	nd := NewNeuroplasticDescriptions("qwen2.5:1.5b", "")
	for _, tool := range []string{"read", "write", "grep", "ls", "glob", "tree"} {
		nd.RegisterDefault(tool, "Description for "+tool)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		nd.GetDescription("grep")
	}
}
