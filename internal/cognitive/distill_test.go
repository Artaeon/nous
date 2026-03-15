package cognitive

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// --- Self-Distiller Tests ---

func TestDistillerCreation(t *testing.T) {
	sd := NewSelfDistiller("")
	if sd == nil {
		t.Fatal("NewSelfDistiller should not return nil")
	}
	if sd.Size() != 0 {
		t.Errorf("new distiller should have 0 pairs, got %d", sd.Size())
	}
	if sd.PatternCount() != 0 {
		t.Errorf("new distiller should have 0 patterns, got %d", sd.PatternCount())
	}
}

func TestRecordFailure(t *testing.T) {
	sd := NewSelfDistiller("")

	sd.RecordFailure(
		"find Pipeline in code",
		"system prompt",
		`{"tool":"grep","args":{"query":"Pipeline"}}`,
		"wrong_args",
		`{"tool":"grep","args":{"pattern":"Pipeline"}}`,
		"intent_compiler",
	)

	if sd.Size() != 1 {
		t.Errorf("size = %d, want 1", sd.Size())
	}
	if sd.PatternCount() != 1 {
		t.Errorf("patterns = %d, want 1", sd.PatternCount())
	}
}

func TestRecordToolMismatch(t *testing.T) {
	sd := NewSelfDistiller("")

	sd.RecordToolMismatch(
		"read main.go",
		"system prompt",
		"read",
		"grep",
		`{"tool":"read","args":{"path":"main.go"}}`,
	)

	if sd.Size() != 1 {
		t.Errorf("size = %d, want 1", sd.Size())
	}

	stats := sd.Stats()
	if stats.ByFailureType["wrong_tool"] != 1 {
		t.Error("should record wrong_tool failure type")
	}
}

func TestRecordArgError(t *testing.T) {
	sd := NewSelfDistiller("")

	sd.RecordArgError(
		"search for TODO",
		"grep",
		`{"query":"TODO"}`,
		`{"pattern":"TODO"}`,
	)

	if sd.Size() != 1 {
		t.Errorf("size = %d, want 1", sd.Size())
	}

	stats := sd.Stats()
	if stats.ByFailureType["wrong_args"] != 1 {
		t.Error("should record wrong_args failure type")
	}
}

func TestDistillerStats(t *testing.T) {
	sd := NewSelfDistiller("")

	// Record various failure types
	sd.RecordFailure("q1", "", "bad1", "wrong_args", "good1", "compiler")
	sd.RecordFailure("q2", "", "bad2", "wrong_args", "good2", "compiler")
	sd.RecordFailure("q3", "", "bad3", "invalid_json", "good3", "fallback")
	sd.RecordFailure("q4", "", "bad4", "wrong_tool", "", "")  // no correction

	stats := sd.Stats()

	if stats.TotalPairs != 4 {
		t.Errorf("total = %d, want 4", stats.TotalPairs)
	}
	if stats.ByFailureType["wrong_args"] != 2 {
		t.Errorf("wrong_args = %d, want 2", stats.ByFailureType["wrong_args"])
	}
	if stats.ByFailureType["invalid_json"] != 1 {
		t.Errorf("invalid_json = %d, want 1", stats.ByFailureType["invalid_json"])
	}
	if stats.CorrectionRate != 0.75 {
		t.Errorf("correction rate = %f, want 0.75", stats.CorrectionRate)
	}
}

func TestDistillerStatsEmpty(t *testing.T) {
	sd := NewSelfDistiller("")
	stats := sd.Stats()

	if stats.TotalPairs != 0 {
		t.Error("empty distiller should have 0 pairs")
	}
	if stats.CorrectionRate != 0 {
		t.Error("empty distiller should have 0 correction rate")
	}
}

func TestDistillerStatsTopPatterns(t *testing.T) {
	sd := NewSelfDistiller("")

	// Create a pattern that repeats 5 times
	for i := 0; i < 5; i++ {
		sd.RecordFailure("query", "", "bad", "wrong_args", "good", "compiler")
	}

	stats := sd.Stats()
	if len(stats.TopPatterns) != 1 {
		t.Errorf("top patterns = %d, want 1", len(stats.TopPatterns))
	}
	if stats.TopPatterns[0].Count != 5 {
		t.Errorf("top pattern count = %d, want 5", stats.TopPatterns[0].Count)
	}
}

func TestPatternTracking(t *testing.T) {
	sd := NewSelfDistiller("")

	// Same failure type with same detail should increment count
	sd.RecordFailure("q1", "", "bad", "wrong_args", "good", "compiler")
	sd.RecordFailure("q2", "", "bad", "wrong_args", "good", "compiler")
	sd.RecordFailure("q3", "", "bad", "wrong_args", "good", "compiler")

	if sd.PatternCount() != 1 {
		t.Errorf("same pattern should be counted once, got %d patterns", sd.PatternCount())
	}
}

func TestPatternTrackingDifferent(t *testing.T) {
	sd := NewSelfDistiller("")

	sd.RecordFailure("q1", "", "bad1", "wrong_args", "good1", "compiler")
	sd.RecordFailure("q2", "", "bad2", "wrong_tool", "good2", "compiler")
	sd.RecordFailure("q3", "", "bad3", "invalid_json", "good3", "compiler")

	if sd.PatternCount() != 3 {
		t.Errorf("3 different patterns, got %d", sd.PatternCount())
	}
}

func TestPatternExamplesLimited(t *testing.T) {
	sd := NewSelfDistiller("")

	// Record 10 failures — examples should be capped at 5
	for i := 0; i < 10; i++ {
		sd.RecordFailure("query", "", "bad", "wrong_args", "good", "compiler")
	}

	sd.mu.RLock()
	defer sd.mu.RUnlock()
	for _, p := range sd.patterns {
		if len(p.Examples) > 5 {
			t.Errorf("examples should be capped at 5, got %d", len(p.Examples))
		}
	}
}

func TestExportNegativeInstructions(t *testing.T) {
	sd := NewSelfDistiller("")

	// Need at least 3 occurrences for a pattern to be significant
	for i := 0; i < 3; i++ {
		sd.RecordFailure("query", "", `{"query":"x"}`, "wrong_args", `{"pattern":"x"}`, "compiler")
	}

	instructions := sd.ExportNegativeInstructions()
	if instructions == "" {
		t.Error("should generate negative instructions for recurring patterns")
	}
	if !strings.Contains(instructions, "IMPORTANT") {
		t.Error("instructions should start with IMPORTANT")
	}
	if !strings.Contains(instructions, "argument") {
		t.Error("instructions should mention argument names for wrong_args pattern")
	}
}

func TestExportNegativeInstructionsEmpty(t *testing.T) {
	sd := NewSelfDistiller("")
	if sd.ExportNegativeInstructions() != "" {
		t.Error("empty distiller should return empty instructions")
	}
}

func TestExportNegativeInstructionsBelowThreshold(t *testing.T) {
	sd := NewSelfDistiller("")

	// Only 2 occurrences — below threshold of 3
	sd.RecordFailure("q1", "", "bad", "wrong_args", "good", "compiler")
	sd.RecordFailure("q2", "", "bad", "wrong_args", "good", "compiler")

	if sd.ExportNegativeInstructions() != "" {
		t.Error("patterns below threshold should not generate instructions")
	}
}

func TestExportNegativeInstructionsCapped(t *testing.T) {
	sd := NewSelfDistiller("")

	// Create 10 different significant patterns
	for i := 0; i < 10; i++ {
		detail := strings.Repeat("x", i+1) // unique details
		for j := 0; j < 3; j++ {
			sd.RecordFailure("query", "", detail, "wrong_args", "good", "compiler")
		}
	}

	instructions := sd.ExportNegativeInstructions()
	// Should be capped at 5 instructions
	lines := strings.Split(strings.TrimSpace(instructions), "\n")
	// First line is "IMPORTANT...", then up to 5 numbered items
	numberedLines := 0
	for _, line := range lines {
		if len(line) > 0 && line[0] >= '1' && line[0] <= '9' {
			numberedLines++
		}
	}
	if numberedLines > 5 {
		t.Errorf("should cap at 5 instructions, got %d", numberedLines)
	}
}

func TestExportContrastive(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "contrastive.jsonl")

	sd := NewSelfDistiller("")
	sd.RecordFailure("find Pipeline", "", `{"tool":"grep","args":{"query":"Pipeline"}}`, "wrong_args",
		`{"tool":"grep","args":{"pattern":"Pipeline"}}`, "intent_compiler")
	sd.RecordFailure("uncorrected", "", "bad", "invalid_json", "", "") // no correction

	err := sd.ExportContrastive(path)
	if err != nil {
		t.Fatalf("export error: %v", err)
	}

	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read error: %v", err)
	}

	// Should only export pairs with corrections (1 out of 2)
	lines := strings.Split(strings.TrimSpace(string(data)), "\n")
	if len(lines) != 1 {
		t.Errorf("exported %d lines, want 1 (only pairs with corrections)", len(lines))
	}

	// Verify DPO format
	var entry map[string]any
	if err := json.Unmarshal([]byte(lines[0]), &entry); err != nil {
		t.Fatalf("invalid JSON in export: %v", err)
	}
	if _, ok := entry["chosen"]; !ok {
		t.Error("DPO format should have 'chosen' field")
	}
	if _, ok := entry["rejected"]; !ok {
		t.Error("DPO format should have 'rejected' field")
	}
	if _, ok := entry["prompt"]; !ok {
		t.Error("DPO format should have 'prompt' field")
	}
}

func TestExportContrastiveEmpty(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "contrastive.jsonl")

	sd := NewSelfDistiller("")
	err := sd.ExportContrastive(path)
	if err != nil {
		t.Fatalf("export error: %v", err)
	}

	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read error: %v", err)
	}
	if strings.TrimSpace(string(data)) != "" {
		t.Error("empty distiller should produce empty export")
	}
}

// --- Persistence Tests ---

func TestDistillerPersistence(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "distill.json")

	sd1 := NewSelfDistiller(path)
	sd1.RecordFailure("query1", "", "bad", "wrong_args", "good", "compiler")
	sd1.RecordFailure("query2", "", "bad", "wrong_tool", "good", "compiler")

	// Load into new instance
	sd2 := NewSelfDistiller(path)
	if sd2.Size() != 2 {
		t.Errorf("loaded size = %d, want 2", sd2.Size())
	}
	if sd2.PatternCount() != 2 {
		t.Errorf("loaded patterns = %d, want 2", sd2.PatternCount())
	}
}

func TestDistillerPersistenceNoPath(t *testing.T) {
	sd := NewSelfDistiller("")
	sd.RecordFailure("query", "", "bad", "wrong_args", "good", "compiler")

	// Should not panic without store path
	if sd.Size() != 1 {
		t.Error("should still work without persistence")
	}
}

// --- Utility Tests ---

func TestTruncateForStorage(t *testing.T) {
	short := "hello"
	if truncateForStorage(short, 100) != short {
		t.Error("short strings should not be truncated")
	}

	long := strings.Repeat("x", 200)
	result := truncateForStorage(long, 50)
	if len(result) > 50 {
		t.Errorf("truncated length = %d, want <= 50", len(result))
	}
	if !strings.HasSuffix(result, "...") {
		t.Error("truncated string should end with ...")
	}
}

func TestNormalizePatternKey(t *testing.T) {
	key := normalizePatternKey("UPPER CASE KEY")
	if key != "upper case key" {
		t.Errorf("key should be lowercased, got %q", key)
	}

	long := strings.Repeat("x", 100)
	key = normalizePatternKey(long)
	if len(key) > 50 {
		t.Errorf("key should be truncated to 50, got %d", len(key))
	}
}

// --- Benchmark ---

func BenchmarkRecordFailure(b *testing.B) {
	sd := NewSelfDistiller("")
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sd.RecordFailure("query", "", "bad", "wrong_args", "good", "compiler")
	}
}

func BenchmarkExportNegativeInstructions(b *testing.B) {
	sd := NewSelfDistiller("")
	for i := 0; i < 20; i++ {
		for j := 0; j < 5; j++ {
			sd.RecordFailure("query", "", strings.Repeat("x", i+1), "wrong_args", "good", "compiler")
		}
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sd.ExportNegativeInstructions()
	}
}
