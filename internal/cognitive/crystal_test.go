package cognitive

import (
	"math"
	"path/filepath"
	"regexp"
	"sync"
	"testing"
	"time"
)

// --- CrystalBook Tests ---

func TestCrystalBookNew(t *testing.T) {
	cb := NewCrystalBook("")
	if cb.Size() != 0 {
		t.Errorf("new CrystalBook should be empty, got %d", cb.Size())
	}
}

func TestCrystallize(t *testing.T) {
	cb := NewCrystalBook("")

	pipe := NewPipeline("what Go version does this project use?")
	pipe.AddStep("read", "module github.com/artaeon/nous\n\ngo 1.22")

	cb.Crystallize("what Go version does this project use?", pipe, "This project uses Go version 1.22.")

	if cb.Size() != 1 {
		t.Fatalf("expected 1 crystal, got %d", cb.Size())
	}
}

func TestCrystalMatchBasic(t *testing.T) {
	cb := NewCrystalBook("")

	pipe := NewPipeline("what Go version does this project use?")
	pipe.AddStep("read", "module github.com/artaeon/nous\n\ngo 1.22")
	cb.Crystallize("what Go version does this project use?", pipe, "This project uses Go version 1.22.")

	// Exact match
	m := cb.Match("what Go version does this project use?")
	if m == nil {
		t.Fatal("expected match for exact query")
	}
	if m.Confidence < 0.7 {
		t.Errorf("confidence = %f, want >= 0.7", m.Confidence)
	}
}

func TestCrystalMatchSimilar(t *testing.T) {
	cb := NewCrystalBook("")

	pipe := NewPipeline("what Go version does this project use?")
	pipe.AddStep("read", "go 1.22")
	cb.Crystallize("what Go version does this project use?", pipe, "Go version 1.22")

	// Similar query (different wording, same keywords)
	m := cb.Match("which version of Go does this project use?")
	if m == nil {
		t.Fatal("expected match for similar query")
	}
}

func TestCrystalNoMatchDifferent(t *testing.T) {
	cb := NewCrystalBook("")

	pipe := NewPipeline("what Go version?")
	pipe.AddStep("read", "go 1.22")
	cb.Crystallize("what Go version?", pipe, "Go 1.22")

	m := cb.Match("explain quantum physics")
	if m != nil {
		t.Errorf("should not match unrelated query, got confidence %f", m.Confidence)
	}
}

func TestCrystalNoMatchEmpty(t *testing.T) {
	cb := NewCrystalBook("")
	if m := cb.Match(""); m != nil {
		t.Error("empty query should not match")
	}
	if m := cb.Match("hello"); m != nil {
		t.Error("query on empty book should not match")
	}
}

func TestCrystallizeDedup(t *testing.T) {
	cb := NewCrystalBook("")

	pipe := NewPipeline("read go.mod")
	pipe.AddStep("read", "go 1.22")

	cb.Crystallize("read go.mod", pipe, "Go version 1.22")
	cb.Crystallize("read go.mod", pipe, "Go version 1.22")

	if cb.Size() != 1 {
		t.Errorf("duplicate crystallization should not create new crystal, got %d", cb.Size())
	}
}

func TestCrystallizeEmpty(t *testing.T) {
	cb := NewCrystalBook("")

	// Nil pipeline
	cb.Crystallize("test", nil, "answer")
	if cb.Size() != 0 {
		t.Error("nil pipeline should not create crystal")
	}

	// Empty pipeline
	pipe := NewPipeline("test")
	cb.Crystallize("test", pipe, "answer")
	if cb.Size() != 0 {
		t.Error("empty pipeline should not create crystal")
	}

	// Short answer
	pipe.AddStep("read", "result")
	cb.Crystallize("test", pipe, "short")
	if cb.Size() != 0 {
		t.Error("short answer should not create crystal")
	}
}

func TestCrystalReportSuccess(t *testing.T) {
	cb := NewCrystalBook("")

	pipe := NewPipeline("test query with enough words")
	pipe.AddStep("read", "result data")
	cb.Crystallize("test query with enough words", pipe, "This is a long enough answer to crystallize.")

	id := cb.crystals[0].ID
	cb.ReportSuccess(id)

	cb.mu.RLock()
	if cb.crystals[0].Successes != 2 { // 1 from crystallize + 1 from report
		t.Errorf("successes = %d, want 2", cb.crystals[0].Successes)
	}
	if cb.crystals[0].Uses != 2 {
		t.Errorf("uses = %d, want 2", cb.crystals[0].Uses)
	}
	cb.mu.RUnlock()
}

func TestCrystalReportFailure(t *testing.T) {
	cb := NewCrystalBook("")

	pipe := NewPipeline("test query with enough words")
	pipe.AddStep("read", "result data")
	cb.Crystallize("test query with enough words", pipe, "This is a long enough answer to crystallize.")

	id := cb.crystals[0].ID
	cb.ReportFailure(id)

	cb.mu.RLock()
	if cb.crystals[0].Successes != 1 { // only from crystallize
		t.Errorf("successes = %d, want 1", cb.crystals[0].Successes)
	}
	if cb.crystals[0].Uses != 2 { // crystallize + failure
		t.Errorf("uses = %d, want 2", cb.crystals[0].Uses)
	}
	cb.mu.RUnlock()
}

func TestCrystalStats(t *testing.T) {
	cb := NewCrystalBook("")

	pipe := NewPipeline("what version")
	pipe.AddStep("read", "1.22")
	cb.Crystallize("what version", pipe, "This project uses version 1.22")

	pipe2 := NewPipeline("list all files")
	pipe2.AddStep("ls", "main.go\ngo.mod")
	cb.Crystallize("list all files", pipe2, "The project has main.go and go.mod files")

	stats := cb.Stats()
	if stats.Total != 2 {
		t.Errorf("total = %d, want 2", stats.Total)
	}
	if stats.TotalUses != 2 {
		t.Errorf("total uses = %d, want 2", stats.TotalUses)
	}
	if stats.TotalSuccesses != 2 {
		t.Errorf("total successes = %d, want 2", stats.TotalSuccesses)
	}
}

func TestCrystalPrune(t *testing.T) {
	cb := NewCrystalBook("")
	cb.maxSize = 3

	for i := 0; i < 5; i++ {
		pipe := NewPipeline("query with enough length and detail")
		pipe.AddStep("read", "result")
		query := "unique query with different words " + string(rune('a'+i))
		cb.Crystallize(query, pipe, "This is answer number enough to crystallize")
	}

	if cb.Size() > 3 {
		t.Errorf("pruning should keep size at %d, got %d", 3, cb.Size())
	}
}

func TestCrystalPersistence(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "crystals.json")

	// Create and save
	cb1 := NewCrystalBook(path)
	pipe := NewPipeline("test persistence query")
	pipe.AddStep("read", "result")
	cb1.Crystallize("test persistence query", pipe, "This is a persistent answer that should be saved")

	// Load from disk
	cb2 := NewCrystalBook(path)
	if cb2.Size() != 1 {
		t.Errorf("loaded crystal book should have 1 crystal, got %d", cb2.Size())
	}
}

// --- Trigger Building Tests ---

func TestBuildTrigger(t *testing.T) {
	trigger := buildTrigger("what Go version does this project use?")

	if len(trigger.Keywords) == 0 {
		t.Error("trigger should have keywords")
	}

	// Should contain significant words
	found := false
	for _, kw := range trigger.Keywords {
		if kw == "version" || kw == "project" {
			found = true
		}
	}
	if !found {
		t.Errorf("keywords should contain 'version' or 'project', got %v", trigger.Keywords)
	}

	if trigger.MinWords < 1 {
		t.Errorf("min words should be >= 1, got %d", trigger.MinWords)
	}
}

func TestFilterCrystalKeywords(t *testing.T) {
	words := []string{"what", "go", "version", "does", "this", "project", "use"}
	filtered := filterCrystalKeywords(words)

	for _, w := range filtered {
		if w == "what" || w == "does" || w == "this" {
			t.Errorf("stop word %q should be filtered", w)
		}
	}

	if len(filtered) == 0 {
		t.Error("should have non-empty keywords after filtering")
	}
}

func TestBuildLoosePattern(t *testing.T) {
	pattern := buildLoosePattern("what Go version?")
	if pattern == "" {
		t.Error("pattern should not be empty for meaningful query")
	}

	// Should be compilable
	_, err := regexp.Compile("(?i)" + pattern)
	if err != nil {
		t.Errorf("pattern should be valid regex, got error: %v", err)
	}
}

func TestBuildExtractionPattern(t *testing.T) {
	// Version number extraction
	re := buildExtractionPattern("go 1.22\nmodule foo", "This uses Go 1.22")
	if re == "" {
		t.Error("should extract version pattern")
	}
}

// --- Crystal Value Scoring ---

func TestCrystalValue(t *testing.T) {
	recent := Crystal{Uses: 10, Successes: 9, LastUsed: time.Now()}
	old := Crystal{Uses: 2, Successes: 1, LastUsed: time.Now().Add(-30 * 24 * time.Hour)}

	if crystalValue(&recent) <= crystalValue(&old) {
		t.Error("recent high-success crystal should have higher value")
	}
}

func TestCrystalValueZeroUses(t *testing.T) {
	c := Crystal{Uses: 0, Successes: 0}
	v := crystalValue(&c)
	if v < 0 {
		t.Errorf("crystal value should be non-negative, got %f", v)
	}
}

func TestCrystalTemporalDecay(t *testing.T) {
	now := time.Now()

	fresh := Crystal{
		Uses: 5, Successes: 5,
		LastUsed: now, CreatedAt: now,
	}
	stale := Crystal{
		Uses: 5, Successes: 5,
		LastUsed: now.Add(-30 * 24 * time.Hour), CreatedAt: now.Add(-60 * 24 * time.Hour),
	}

	freshVal := crystalValue(&fresh)
	staleVal := crystalValue(&stale)

	if staleVal >= freshVal {
		t.Errorf("stale crystal (val=%f) should be worth less than fresh (val=%f)", staleVal, freshVal)
	}
}

func TestCrystalAgePenalty(t *testing.T) {
	now := time.Now()

	// Old crystal with low usage — should be penalized
	oldLowUse := Crystal{
		Uses: 3, Successes: 3,
		LastUsed: now.Add(-7 * 24 * time.Hour),
		CreatedAt: now.Add(-100 * 24 * time.Hour), // >90 days, <5 uses
	}

	// Old crystal with high usage — should NOT be penalized
	oldHighUse := Crystal{
		Uses: 10, Successes: 9,
		LastUsed: now.Add(-7 * 24 * time.Hour),
		CreatedAt: now.Add(-100 * 24 * time.Hour), // >90 days, >=5 uses
	}

	lowVal := crystalValue(&oldLowUse)
	highVal := crystalValue(&oldHighUse)

	if lowVal >= highVal {
		t.Errorf("old low-use crystal (val=%f) should be worth less than old high-use (val=%f)", lowVal, highVal)
	}
}

// --- Benchmark ---

// --- Race Condition Tests ---

func TestCrystalConcurrentMatchCrystallize(t *testing.T) {
	cb := NewCrystalBook("")

	// Pre-populate
	for i := 0; i < 5; i++ {
		pipe := NewPipeline("initial query with keywords")
		pipe.AddStep("read", "initial result data")
		cb.Crystallize("initial query about component "+string(rune('a'+i)), pipe, "Initial answer that is long enough to be crystallized")
	}

	var wg sync.WaitGroup

	// 5 goroutines doing Match
	for g := 0; g < 5; g++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			for i := 0; i < 50; i++ {
				cb.Match("query about component " + string(rune('a'+id)))
			}
		}(g)
	}

	// 5 goroutines doing Crystallize
	for g := 0; g < 5; g++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			for i := 0; i < 20; i++ {
				pipe := NewPipeline("concurrent crystallize test")
				pipe.AddStep("grep", "match found in file.go")
				query := "concurrent query " + string(rune('a'+id)) + " iteration " + string(rune('0'+i%10))
				cb.Crystallize(query, pipe, "Concurrent answer that is long enough to crystallize properly")
			}
		}(g)
	}

	wg.Wait()

	if cb.Size() < 1 {
		t.Errorf("crystal book should have at least 1 crystal, got %d", cb.Size())
	}
}

// --- Formula Verification Tests ---

func TestCrystalTemporalDecayFormula(t *testing.T) {
	now := time.Now()

	// crystalValue uses: successRate*2.0 + uses*0.1, then *= recencyFactor,
	// then + recency bonus (1.0 for <1 day, 0.5 for <7 days, 0 otherwise)
	// recencyFactor = 1/(1 + daysSinceUse/14)

	recentCrystal := Crystal{Uses: 5, Successes: 5, LastUsed: now, CreatedAt: now}
	weekOldCrystal := Crystal{Uses: 5, Successes: 5, LastUsed: now.Add(-3 * 24 * time.Hour), CreatedAt: now}
	oldCrystal := Crystal{Uses: 5, Successes: 5, LastUsed: now.Add(-30 * 24 * time.Hour), CreatedAt: now}

	recentVal := crystalValue(&recentCrystal)
	weekVal := crystalValue(&weekOldCrystal)
	oldVal := crystalValue(&oldCrystal)

	// Recent > week-old > month-old
	if recentVal <= weekVal {
		t.Errorf("recent (%f) should exceed week-old (%f)", recentVal, weekVal)
	}
	if weekVal <= oldVal {
		t.Errorf("week-old (%f) should exceed month-old (%f)", weekVal, oldVal)
	}

	// Verify exact values:
	// Base: 5/5 * 2.0 + 5*0.1 = 2.5
	// Recent: recencyFactor ~1.0, bonus +1.0 → 2.5 * 1.0 + 1.0 = 3.5
	expectedRecent := 2.5*1.0 + 1.0
	if math.Abs(recentVal-expectedRecent) > 0.05 {
		t.Errorf("recent value = %f, want ~%f", recentVal, expectedRecent)
	}

	// 3-day-old: recencyFactor = 1/(1+3/14) ≈ 0.824, bonus +0.5
	expectedWeek := 2.5/(1.0+3.0/14.0) + 0.5
	if math.Abs(weekVal-expectedWeek) > 0.05 {
		t.Errorf("week value = %f, want ~%f", weekVal, expectedWeek)
	}

	// 30-day-old: recencyFactor = 1/(1+30/14) ≈ 0.318, no bonus
	expectedOld := 2.5 / (1.0 + 30.0/14.0)
	if math.Abs(oldVal-expectedOld) > 0.05 {
		t.Errorf("old value = %f, want ~%f", oldVal, expectedOld)
	}
}

func BenchmarkCrystalMatch(b *testing.B) {
	cb := NewCrystalBook("")
	for i := 0; i < 50; i++ {
		pipe := NewPipeline("query with some distinct keywords")
		pipe.AddStep("read", "result")
		query := "what is the version of component " + string(rune('a'+i%26))
		cb.Crystallize(query, pipe, "Component version is 1.0 for this specific case")
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		cb.Match("what is the version of component x")
	}
}
