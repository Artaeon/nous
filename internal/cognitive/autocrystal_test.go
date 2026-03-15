package cognitive

import (
	"testing"
	"time"

	"github.com/artaeon/nous/internal/memory"
)

func TestAutoCrystallizerNil(t *testing.T) {
	var ac *AutoCrystallizer
	if ac.Run() != 0 {
		t.Error("nil auto-crystallizer should return 0")
	}
}

func TestAutoCrystallizerEmpty(t *testing.T) {
	book := NewCrystalBook("")
	mem := memory.NewEpisodicMemory("", nil)
	ac := NewAutoCrystallizer(book, mem)

	created := ac.ForceRun()
	if created != 0 {
		t.Errorf("empty memory should create 0 crystals, got %d", created)
	}
}

func TestAutoCrystallizerCreates(t *testing.T) {
	book := NewCrystalBook("")
	mem := memory.NewEpisodicMemory("", nil)

	// Record the same successful pattern 5 times (above threshold of 3)
	for i := 0; i < 5; i++ {
		mem.Record(memory.Episode{
			Timestamp: time.Now().Add(time.Duration(i) * time.Second),
			Input:     "show me the reasoner code",
			ToolsUsed: []string{"grep", "read"},
			Success:   true,
			Duration:  100,
		})
	}

	ac := NewAutoCrystallizer(book, mem)
	created := ac.ForceRun()

	if created != 1 {
		t.Errorf("should create 1 crystal, got %d", created)
	}
	if book.Size() != 1 {
		t.Errorf("book should have 1 crystal, got %d", book.Size())
	}
}

func TestAutoCrystallizerDedup(t *testing.T) {
	book := NewCrystalBook("")
	mem := memory.NewEpisodicMemory("", nil)

	// Record pattern
	for i := 0; i < 5; i++ {
		mem.Record(memory.Episode{
			Timestamp: time.Now().Add(time.Duration(i) * time.Second),
			Input:     "list directory files",
			ToolsUsed: []string{"ls"},
			Success:   true,
			Duration:  50,
		})
	}

	ac := NewAutoCrystallizer(book, mem)

	// First run creates crystals
	first := ac.ForceRun()
	if first == 0 {
		t.Fatal("first run should create at least 1 crystal")
	}

	// Second run should not create duplicates
	second := ac.ForceRun()
	if second != 0 {
		t.Errorf("second run should not create duplicates, got %d", second)
	}
}

func TestAutoCrystallizerCooldown(t *testing.T) {
	book := NewCrystalBook("")
	mem := memory.NewEpisodicMemory("", nil)

	for i := 0; i < 5; i++ {
		mem.Record(memory.Episode{
			Timestamp: time.Now(),
			Input:     "test query pattern",
			ToolsUsed: []string{"read"},
			Success:   true,
		})
	}

	ac := NewAutoCrystallizer(book, mem)
	ac.cooldown = 1 * time.Hour

	// First run works
	ac.ForceRun()

	// Run() should respect cooldown
	created := ac.Run()
	if created != 0 {
		t.Error("should respect cooldown and return 0")
	}
}

func TestAutoCrystallizerBelowThreshold(t *testing.T) {
	book := NewCrystalBook("")
	mem := memory.NewEpisodicMemory("", nil)

	// Only 2 occurrences — below threshold of 3
	for i := 0; i < 2; i++ {
		mem.Record(memory.Episode{
			Timestamp: time.Now(),
			Input:     "rare query",
			ToolsUsed: []string{"grep"},
			Success:   true,
		})
	}

	ac := NewAutoCrystallizer(book, mem)
	created := ac.ForceRun()

	if created != 0 {
		t.Errorf("below-threshold patterns should not crystallize, got %d", created)
	}
}

func TestAutoCrystallizerMultiplePatterns(t *testing.T) {
	book := NewCrystalBook("")
	mem := memory.NewEpisodicMemory("", nil)

	// Pattern 1: grep + read (5 occurrences)
	for i := 0; i < 5; i++ {
		mem.Record(memory.Episode{
			Timestamp: time.Now().Add(time.Duration(i) * time.Millisecond),
			Input:     "find and read function code",
			ToolsUsed: []string{"grep", "read"},
			Success:   true,
			Duration:  200,
		})
	}

	// Pattern 2: ls (4 occurrences)
	for i := 0; i < 4; i++ {
		mem.Record(memory.Episode{
			Timestamp: time.Now().Add(time.Duration(10+i) * time.Millisecond),
			Input:     "list directory contents",
			ToolsUsed: []string{"ls"},
			Success:   true,
			Duration:  50,
		})
	}

	ac := NewAutoCrystallizer(book, mem)
	created := ac.ForceRun()

	if created < 2 {
		t.Errorf("should create at least 2 crystals from 2 patterns, got %d", created)
	}
}

func TestAutoCrystallizerCap(t *testing.T) {
	book := NewCrystalBook("")
	book.maxSize = 5
	mem := memory.NewEpisodicMemory("", nil)

	// Create many different patterns
	tools := [][]string{
		{"grep"}, {"read"}, {"ls"}, {"tree"}, {"glob"},
		{"git"}, {"write"}, {"shell"},
	}

	for _, toolSeq := range tools {
		for i := 0; i < 4; i++ {
			mem.Record(memory.Episode{
				Timestamp: time.Now().Add(time.Duration(i) * time.Millisecond),
				Input:     "query for " + toolSeq[0] + " action",
				ToolsUsed: toolSeq,
				Success:   true,
			})
		}
	}

	ac := NewAutoCrystallizer(book, mem)
	ac.ForceRun()

	if book.Size() > book.maxSize {
		t.Errorf("should respect maxSize %d, got %d", book.maxSize, book.Size())
	}
}

func TestBuildCrystalFromPattern(t *testing.T) {
	ac := &AutoCrystallizer{book: NewCrystalBook("")}

	pattern := memory.SuccessPattern{
		Tools:    []string{"grep", "read"},
		Count:    5,
		Keywords: []string{"find", "function", "code"},
	}

	crystal := ac.buildCrystalFromPattern(pattern)
	if crystal == nil {
		t.Fatal("should build crystal from valid pattern")
	}
	if len(crystal.Steps) != 2 {
		t.Errorf("crystal should have 2 steps, got %d", len(crystal.Steps))
	}
	if crystal.Steps[0].Tool != "grep" {
		t.Errorf("first step tool = %q, want grep", crystal.Steps[0].Tool)
	}
	if crystal.Steps[1].Tool != "read" {
		t.Errorf("second step tool = %q, want read", crystal.Steps[1].Tool)
	}
	if crystal.Uses != 5 {
		t.Errorf("uses = %d, want 5", crystal.Uses)
	}
	if len(crystal.Trigger.Keywords) != 3 {
		t.Errorf("trigger keywords = %d, want 3", len(crystal.Trigger.Keywords))
	}
}

func TestAutoCrystallizerMixedSuccessFailure(t *testing.T) {
	book := NewCrystalBook("")
	mem := memory.NewEpisodicMemory("", nil)

	// 4 successes with ["grep"] — above threshold
	for i := 0; i < 4; i++ {
		mem.Record(memory.Episode{
			Timestamp: time.Now().Add(time.Duration(i) * time.Second),
			Input:     "search for pattern",
			ToolsUsed: []string{"grep"},
			Success:   true,
			Duration:  100,
		})
	}
	// 5 failures with same tools — should NOT count toward crystallization
	for i := 0; i < 5; i++ {
		mem.Record(memory.Episode{
			Timestamp: time.Now().Add(time.Duration(10+i) * time.Second),
			Input:     "search for pattern",
			ToolsUsed: []string{"grep"},
			Success:   false,
			Duration:  200,
		})
	}

	ac := NewAutoCrystallizer(book, mem)
	created := ac.ForceRun()

	// Only successes should count: 4 >= threshold of 3
	if created < 1 {
		t.Errorf("4 successful episodes should crystallize, got %d created", created)
	}
}

func TestBuildCrystalFromPatternEmpty(t *testing.T) {
	ac := &AutoCrystallizer{book: NewCrystalBook("")}

	// No tools → nil
	if c := ac.buildCrystalFromPattern(memory.SuccessPattern{}); c != nil {
		t.Error("empty pattern should return nil")
	}

	// No keywords → nil
	if c := ac.buildCrystalFromPattern(memory.SuccessPattern{Tools: []string{"read"}}); c != nil {
		t.Error("pattern without keywords should return nil")
	}
}
