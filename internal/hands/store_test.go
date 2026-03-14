package hands

import (
	"testing"
	"time"
)

func TestStoreRoundTrip(t *testing.T) {
	dir := t.TempDir()
	store := NewStore(dir)

	h := &Hand{
		Name:        "test-hand",
		Description: "a test hand",
		Schedule:    "@daily",
		Enabled:     true,
		Config:      DefaultConfig(),
		Prompt:      "do the thing",
	}

	if err := store.SaveHand(h); err != nil {
		t.Fatalf("SaveHand() error: %v", err)
	}

	// Reload from disk
	reloaded := NewStore(dir)
	got, ok := reloaded.GetHand("test-hand")
	if !ok {
		t.Fatal("expected to find hand after reload")
	}
	if got.Name != "test-hand" {
		t.Errorf("Name = %q, want test-hand", got.Name)
	}
	if got.Description != "a test hand" {
		t.Errorf("Description = %q, want 'a test hand'", got.Description)
	}
	if got.Prompt != "do the thing" {
		t.Errorf("Prompt = %q, want 'do the thing'", got.Prompt)
	}
}

func TestStoreAllHands(t *testing.T) {
	store := NewStore(t.TempDir())

	for _, name := range []string{"alpha", "beta", "gamma"} {
		_ = store.SaveHand(&Hand{Name: name, Config: DefaultConfig()})
	}

	all := store.AllHands()
	if len(all) != 3 {
		t.Fatalf("AllHands() returned %d, want 3", len(all))
	}
}

func TestStoreRecordRunAndHistory(t *testing.T) {
	store := NewStore(t.TempDir())

	for i := 0; i < 5; i++ {
		_ = store.RecordRun(RunRecord{
			HandName:  "test",
			StartedAt: time.Now(),
			Duration:  int64(i * 1000),
			Success:   i%2 == 0,
			Output:    "output",
		})
	}

	history := store.History("test")
	if len(history) != 5 {
		t.Fatalf("History() returned %d records, want 5", len(history))
	}

	successes, failures, avgMs := store.Stats("test")
	if successes != 3 {
		t.Errorf("successes = %d, want 3", successes)
	}
	if failures != 2 {
		t.Errorf("failures = %d, want 2", failures)
	}
	// (0 + 1000 + 2000 + 3000 + 4000) / 5 = 2000
	if avgMs != 2000 {
		t.Errorf("avgMs = %d, want 2000", avgMs)
	}
}

func TestStoreMaxRuns(t *testing.T) {
	store := NewStore(t.TempDir())
	store.SetMaxRuns(3)

	for i := 0; i < 10; i++ {
		_ = store.RecordRun(RunRecord{
			HandName:  "trim",
			StartedAt: time.Now(),
			Duration:  int64(i),
			Success:   true,
			Output:    "ok",
		})
	}

	history := store.History("trim")
	if len(history) != 3 {
		t.Fatalf("History() returned %d records, want 3 (trimmed)", len(history))
	}
	// Should keep the last 3
	if history[0].Duration != 7 {
		t.Errorf("first retained record Duration = %d, want 7", history[0].Duration)
	}
}

func TestStoreDeleteHand(t *testing.T) {
	store := NewStore(t.TempDir())
	_ = store.SaveHand(&Hand{Name: "doomed", Config: DefaultConfig()})
	_ = store.RecordRun(RunRecord{HandName: "doomed", Success: true})

	if err := store.DeleteHand("doomed"); err != nil {
		t.Fatalf("DeleteHand() error: %v", err)
	}

	_, ok := store.GetHand("doomed")
	if ok {
		t.Error("hand should be deleted")
	}
	if len(store.History("doomed")) != 0 {
		t.Error("history should be deleted")
	}
}

func TestStoreGetHandNotFound(t *testing.T) {
	store := NewStore(t.TempDir())
	_, ok := store.GetHand("nonexistent")
	if ok {
		t.Error("expected not found for nonexistent hand")
	}
}

func TestStoreStatsEmpty(t *testing.T) {
	store := NewStore(t.TempDir())
	s, f, avg := store.Stats("missing")
	if s != 0 || f != 0 || avg != 0 {
		t.Errorf("Stats for missing hand should be all zeros, got %d %d %d", s, f, avg)
	}
}
