package cognitive

import (
	"strings"
	"time"

	"github.com/artaeon/nous/internal/memory"
)

// AutoCrystallizer mines episodic memory for recurring successful patterns
// and automatically generates crystals from them. This closes the loop
// between experience and instant recall — after enough interactions,
// common patterns crystallize into deterministic, zero-latency paths.
//
// Innovation: No AI agent automatically converts its interaction history
// into deterministic reasoning shortcuts. Most systems accumulate data
// but never USE it to build faster paths. AutoCrystallizer does exactly
// that — it's the system learning from its own autobiography.
//
// Flow:
//   1. Mine episodic memory for SuccessPatterns (tool sequences used ≥3 times)
//   2. For each pattern, extract common keywords from the matching episodes
//   3. Check if a similar crystal already exists (dedup)
//   4. Build trigger from keywords + tool sequence
//   5. Add crystal to the book
//
// Over 1000 interactions, this can generate 50-200 crystals automatically,
// making the system dramatically faster for the user's most common queries.
type AutoCrystallizer struct {
	book    *CrystalBook
	memory  *memory.EpisodicMemory
	lastRun time.Time
	cooldown time.Duration // minimum time between auto-crystallization runs
}

// NewAutoCrystallizer creates a new auto-crystallizer.
func NewAutoCrystallizer(book *CrystalBook, mem *memory.EpisodicMemory) *AutoCrystallizer {
	return &AutoCrystallizer{
		book:     book,
		memory:   mem,
		cooldown: 1 * time.Hour,
	}
}

// Run performs one auto-crystallization pass. Safe to call frequently —
// respects cooldown and does nothing if called too early.
func (ac *AutoCrystallizer) Run() int {
	if ac == nil || ac.book == nil || ac.memory == nil {
		return 0
	}

	if time.Since(ac.lastRun) < ac.cooldown {
		return 0
	}
	ac.lastRun = time.Now()

	return ac.crystallize()
}

// ForceRun performs auto-crystallization regardless of cooldown.
func (ac *AutoCrystallizer) ForceRun() int {
	if ac == nil || ac.book == nil || ac.memory == nil {
		return 0
	}
	ac.lastRun = time.Now()
	return ac.crystallize()
}

func (ac *AutoCrystallizer) crystallize() int {
	// Mine patterns that appeared at least 3 times successfully
	patterns := ac.memory.SuccessPatterns(3)
	if len(patterns) == 0 {
		return 0
	}

	created := 0
	for _, pattern := range patterns {
		if created >= 20 {
			break // cap per run to avoid flooding
		}

		// Skip patterns with no keywords (can't build a trigger)
		if len(pattern.Keywords) == 0 {
			continue
		}

		// Skip if a similar crystal already exists
		if ac.hasSimilarCrystal(pattern) {
			continue
		}

		// Build crystal from the pattern
		crystal := ac.buildCrystalFromPattern(pattern)
		if crystal == nil {
			continue
		}

		ac.book.mu.Lock()
		ac.book.crystals = append(ac.book.crystals, *crystal)
		ac.book.mu.Unlock()
		created++
	}

	// Save if we created any
	if created > 0 {
		ac.book.mu.Lock()
		// Prune if over capacity
		if len(ac.book.crystals) > ac.book.maxSize {
			ac.book.prune()
		}
		ac.book.save()
		ac.book.mu.Unlock()
	}

	return created
}

// hasSimilarCrystal checks if the crystal book already has a crystal
// covering this pattern's keywords.
func (ac *AutoCrystallizer) hasSimilarCrystal(pattern memory.SuccessPattern) bool {
	ac.book.mu.RLock()
	defer ac.book.mu.RUnlock()

	patternKeywords := make(map[string]bool)
	for _, kw := range pattern.Keywords {
		patternKeywords[kw] = true
	}

	for _, c := range ac.book.crystals {
		if len(c.Trigger.Keywords) == 0 {
			continue
		}
		// Check keyword overlap
		overlap := 0
		for _, kw := range c.Trigger.Keywords {
			if patternKeywords[kw] {
				overlap++
			}
		}
		// Same tool sequence
		sameTools := len(c.Steps) == len(pattern.Tools)
		if sameTools {
			for i := range c.Steps {
				if i >= len(pattern.Tools) || c.Steps[i].Tool != pattern.Tools[i] {
					sameTools = false
					break
				}
			}
		}
		// If >70% keyword overlap AND same tools, it's a duplicate
		ratio := float64(overlap) / float64(len(pattern.Keywords))
		if ratio >= 0.7 && sameTools {
			return true
		}
	}
	return false
}

// buildCrystalFromPattern converts a mined success pattern into a Crystal.
func (ac *AutoCrystallizer) buildCrystalFromPattern(pattern memory.SuccessPattern) *Crystal {
	if len(pattern.Tools) == 0 || len(pattern.Keywords) == 0 {
		return nil
	}

	// Build steps from tool sequence
	var steps []CrystalStep
	for _, tool := range pattern.Tools {
		steps = append(steps, CrystalStep{
			Tool:      tool,
			Args:      make(map[string]string),
			ResultVar: "result_" + tool,
		})
	}

	// Build trigger from keywords
	trigger := &CrystalTrigger{
		Keywords: pattern.Keywords,
		MinWords: 2,
		MaxWords: 20,
	}

	// Build a loose regex from keywords
	var parts []string
	for _, kw := range pattern.Keywords {
		if len(kw) >= 3 {
			parts = append(parts, kw)
		}
	}
	if len(parts) > 0 {
		trigger.Pattern = "(?i)" + strings.Join(parts, `.*`)
	}

	id := crystalID(strings.Join(pattern.Keywords, " ") + strings.Join(pattern.Tools, "+"))

	return &Crystal{
		ID:           id,
		Trigger:      trigger,
		Steps:        steps,
		ResponseTmpl: "", // auto-generated crystals don't have response templates
		Uses:         pattern.Count,
		Successes:    pattern.Count,
		CreatedAt:    time.Now(),
		LastUsed:     time.Now(),
	}
}
