package cognitive

import (
	"os"
	"path/filepath"
	"testing"
)

func TestDialoguePolicy_Recommend_Default(t *testing.T) {
	dp := NewDialoguePolicy()

	// With no data, should return sensible defaults.
	tests := []struct {
		queryType string
		want      string
	}{
		{"factual", "direct"},
		{"coaching", "socratic"},
		{"comparison", "compare"},
		{"teaching", "explain"},
		{"creative", "synthesis"},
		{"unknown_type", "synthesis"}, // unknown falls back to synthesis
	}

	for _, tt := range tests {
		dec := dp.Recommend("anything", tt.queryType)
		if dec == nil {
			t.Fatalf("Recommend(%q) returned nil", tt.queryType)
		}
		if dec.Strategy != tt.want {
			t.Errorf("Recommend(%q).Strategy = %q, want %q", tt.queryType, dec.Strategy, tt.want)
		}
		if dec.Reason != "default" {
			t.Errorf("Recommend(%q).Reason = %q, want 'default'", tt.queryType, dec.Reason)
		}
		if dec.Confidence != 0.3 {
			t.Errorf("Recommend(%q).Confidence = %.2f, want 0.3", tt.queryType, dec.Confidence)
		}
		if len(dec.Alternatives) == 0 {
			t.Errorf("Recommend(%q) should include alternatives", tt.queryType)
		}
		// The chosen strategy should NOT appear in alternatives.
		for _, alt := range dec.Alternatives {
			if alt == dec.Strategy {
				t.Errorf("Recommend(%q): strategy %q should not appear in alternatives", tt.queryType, dec.Strategy)
			}
		}
	}
}

func TestDialoguePolicy_Recommend_Learned(t *testing.T) {
	dp := NewDialoguePolicy()

	// Record that "direct" works well for science.
	for i := 0; i < 15; i++ {
		dp.RecordOutcome("science", "direct", "success")
	}
	for i := 0; i < 5; i++ {
		dp.RecordOutcome("science", "direct", "failure")
	}

	// Record that "socratic" works poorly for science.
	for i := 0; i < 3; i++ {
		dp.RecordOutcome("science", "socratic", "success")
	}
	for i := 0; i < 12; i++ {
		dp.RecordOutcome("science", "socratic", "failure")
	}

	dec := dp.Recommend("science", "factual")
	if dec == nil {
		t.Fatal("Recommend returned nil")
	}

	t.Logf("Learned recommendation: strategy=%s confidence=%.2f reason=%s alts=%v",
		dec.Strategy, dec.Confidence, dec.Reason, dec.Alternatives)

	if dec.Strategy != "direct" {
		t.Errorf("Strategy = %q, want 'direct' (higher success rate)", dec.Strategy)
	}
	if dec.Reason != "learned" {
		t.Errorf("Reason = %q, want 'learned'", dec.Reason)
	}
	if dec.Confidence <= 0.3 {
		t.Errorf("Confidence = %.2f, want > 0.3 for learned strategy", dec.Confidence)
	}

	// Alternatives should include socratic somewhere.
	found := false
	for _, alt := range dec.Alternatives {
		if alt == "socratic" {
			found = true
			break
		}
	}
	if !found {
		t.Error("alternatives should include 'socratic'")
	}
}

func TestDialoguePolicy_Recommend_Preliminary(t *testing.T) {
	dp := NewDialoguePolicy()

	// Only a few data points: should be "preliminary".
	dp.RecordOutcome("music", "explain", "success")
	dp.RecordOutcome("music", "explain", "success")

	dec := dp.Recommend("music", "teaching")
	if dec.Reason != "preliminary" {
		t.Errorf("Reason = %q, want 'preliminary' for small sample size", dec.Reason)
	}
}

func TestDialoguePolicy_RecordOutcome(t *testing.T) {
	dp := NewDialoguePolicy()

	dp.RecordOutcome("math", "direct", "success")
	dp.RecordOutcome("math", "direct", "success")
	dp.RecordOutcome("math", "direct", "failure")
	dp.RecordOutcome("math", "direct", "neutral")

	stats := dp.GetStats("math")
	if stats == nil {
		t.Fatal("GetStats returned nil")
	}

	rec, ok := stats["direct"]
	if !ok {
		t.Fatal("no record for 'direct' strategy")
	}

	if rec.Successes != 2 {
		t.Errorf("Successes = %d, want 2", rec.Successes)
	}
	if rec.Failures != 1 {
		t.Errorf("Failures = %d, want 1", rec.Failures)
	}
	if rec.Neutrals != 1 {
		t.Errorf("Neutrals = %d, want 1", rec.Neutrals)
	}

	// AvgQuality = (2*1.0 + 1*0.5) / 4 = 0.625
	expectedQ := 0.625
	if rec.AvgQuality < expectedQ-0.01 || rec.AvgQuality > expectedQ+0.01 {
		t.Errorf("AvgQuality = %.3f, want ~%.3f", rec.AvgQuality, expectedQ)
	}
}

func TestDialoguePolicy_GetStats_NoDomain(t *testing.T) {
	dp := NewDialoguePolicy()

	stats := dp.GetStats("nonexistent")
	if stats != nil {
		t.Errorf("expected nil for unknown domain, got %v", stats)
	}
}

func TestDialoguePolicy_SaveLoad(t *testing.T) {
	dp := NewDialoguePolicy()

	// Build some data.
	for i := 0; i < 10; i++ {
		dp.RecordOutcome("science", "direct", "success")
	}
	for i := 0; i < 5; i++ {
		dp.RecordOutcome("science", "socratic", "failure")
	}
	dp.RecordOutcome("history", "explain", "neutral")

	// Save to temp file.
	dir := t.TempDir()
	path := filepath.Join(dir, "dialogue_policy.json")

	if err := dp.Save(path); err != nil {
		t.Fatalf("Save failed: %v", err)
	}

	// Verify the file exists and is non-empty.
	info, err := os.Stat(path)
	if err != nil {
		t.Fatalf("stat after save: %v", err)
	}
	if info.Size() == 0 {
		t.Fatal("saved file is empty")
	}

	// Load into a fresh policy.
	dp2 := NewDialoguePolicy()
	if err := dp2.Load(path); err != nil {
		t.Fatalf("Load failed: %v", err)
	}

	// Verify the data survived the round-trip.
	stats := dp2.GetStats("science")
	if stats == nil {
		t.Fatal("GetStats returned nil after Load")
	}

	rec, ok := stats["direct"]
	if !ok {
		t.Fatal("no 'direct' record after Load")
	}
	if rec.Successes != 10 {
		t.Errorf("Successes after Load = %d, want 10", rec.Successes)
	}

	rec2, ok := stats["socratic"]
	if !ok {
		t.Fatal("no 'socratic' record after Load")
	}
	if rec2.Failures != 5 {
		t.Errorf("Failures after Load = %d, want 5", rec2.Failures)
	}

	// Verify recommendation still works after load.
	dec := dp2.Recommend("science", "factual")
	if dec.Strategy != "direct" {
		t.Errorf("Strategy after Load = %q, want 'direct'", dec.Strategy)
	}
}
