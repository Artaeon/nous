package agent

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"
)

// -----------------------------------------------------------------------
// Agent Experience Memory — learn from past executions.
//
// Records which tool chains succeeded and failed for different goal types.
// Uses this history to refine future plans: boost tools that work, avoid
// tools that fail, adapt phase structures based on what actually helped.
//
// This makes agents smarter with every run — not just executing templates,
// but learning from their own experience.
// -----------------------------------------------------------------------

// ExperienceEntry records the outcome of a tool chain execution.
type ExperienceEntry struct {
	GoalType    string    `json:"goal_type"`    // "research", "writing", "analysis", etc.
	ToolChain   []string  `json:"tool_chain"`   // ordered tool names used
	Succeeded   bool      `json:"succeeded"`
	OutputWords int       `json:"output_words"` // rough quality proxy
	Duration    float64   `json:"duration_ms"`
	Timestamp   time.Time `json:"timestamp"`
	Goal        string    `json:"goal"`         // original goal text
}

// ToolScore tracks success rate and avg quality for a tool in a goal context.
type ToolScore struct {
	Tool        string  `json:"tool"`
	Uses        int     `json:"uses"`
	Successes   int     `json:"successes"`
	AvgWords    float64 `json:"avg_words"`
	SuccessRate float64 `json:"success_rate"`
}

// ExperienceMemory stores and retrieves agent execution history.
type ExperienceMemory struct {
	Entries []ExperienceEntry `json:"entries"`
	path    string
	mu      sync.Mutex
}

// NewExperienceMemory creates or loads experience memory.
func NewExperienceMemory(workspace string) *ExperienceMemory {
	p := filepath.Join(workspace, "experience.json")
	em := &ExperienceMemory{path: p}
	em.load()
	return em
}

// Record adds a new experience entry.
func (em *ExperienceMemory) Record(entry ExperienceEntry) {
	em.mu.Lock()
	defer em.mu.Unlock()

	entry.Timestamp = time.Now()
	em.Entries = append(em.Entries, entry)

	// Cap at 500 entries, remove oldest.
	if len(em.Entries) > 500 {
		em.Entries = em.Entries[len(em.Entries)-500:]
	}

	em.save()
}

// ToolScoresForGoal returns success rates for tools used in similar goals.
func (em *ExperienceMemory) ToolScoresForGoal(goalType string) []ToolScore {
	em.mu.Lock()
	defer em.mu.Unlock()

	scores := make(map[string]*ToolScore)

	for _, e := range em.Entries {
		if e.GoalType != goalType {
			continue
		}
		for _, tool := range e.ToolChain {
			s, ok := scores[tool]
			if !ok {
				s = &ToolScore{Tool: tool}
				scores[tool] = s
			}
			s.Uses++
			if e.Succeeded {
				s.Successes++
				s.AvgWords = (s.AvgWords*float64(s.Successes-1) + float64(e.OutputWords)) / float64(s.Successes)
			}
		}
	}

	var result []ToolScore
	for _, s := range scores {
		if s.Uses > 0 {
			s.SuccessRate = float64(s.Successes) / float64(s.Uses)
		}
		result = append(result, *s)
	}

	// Sort by success rate descending.
	for i := range result {
		for j := i + 1; j < len(result); j++ {
			if result[j].SuccessRate > result[i].SuccessRate {
				result[i], result[j] = result[j], result[i]
			}
		}
	}

	return result
}

// BestToolsForGoal returns the top N tools for a goal type based on experience.
func (em *ExperienceMemory) BestToolsForGoal(goalType string, n int) []string {
	scores := em.ToolScoresForGoal(goalType)
	var best []string
	for _, s := range scores {
		if s.SuccessRate >= 0.5 && s.Uses >= 2 {
			best = append(best, s.Tool)
			if len(best) >= n {
				break
			}
		}
	}
	return best
}

// FailedToolsForGoal returns tools that consistently fail for a goal type.
func (em *ExperienceMemory) FailedToolsForGoal(goalType string) []string {
	scores := em.ToolScoresForGoal(goalType)
	var failed []string
	for _, s := range scores {
		if s.Uses >= 3 && s.SuccessRate < 0.3 {
			failed = append(failed, s.Tool)
		}
	}
	return failed
}

// SimilarGoalOutcomes returns past entries for similar goals.
func (em *ExperienceMemory) SimilarGoalOutcomes(goal string) []ExperienceEntry {
	em.mu.Lock()
	defer em.mu.Unlock()

	lower := strings.ToLower(goal)
	words := strings.Fields(lower)

	var matches []ExperienceEntry
	for _, e := range em.Entries {
		eLower := strings.ToLower(e.Goal)
		overlap := 0
		for _, w := range words {
			if len(w) > 3 && strings.Contains(eLower, w) {
				overlap++
			}
		}
		if overlap >= 2 || (len(words) <= 2 && overlap >= 1) {
			matches = append(matches, e)
		}
	}

	return matches
}

// SuccessRate returns the overall agent success rate.
func (em *ExperienceMemory) SuccessRate() float64 {
	em.mu.Lock()
	defer em.mu.Unlock()

	if len(em.Entries) == 0 {
		return 0
	}

	successes := 0
	for _, e := range em.Entries {
		if e.Succeeded {
			successes++
		}
	}
	return float64(successes) / float64(len(em.Entries))
}

// Stats returns experience memory statistics.
func (em *ExperienceMemory) Stats() map[string]interface{} {
	em.mu.Lock()
	defer em.mu.Unlock()

	goalTypes := make(map[string]int)
	for _, e := range em.Entries {
		goalTypes[e.GoalType]++
	}

	return map[string]interface{}{
		"total_entries": len(em.Entries),
		"success_rate":  em.SuccessRate(),
		"goal_types":    goalTypes,
	}
}

func (em *ExperienceMemory) load() {
	data, err := os.ReadFile(em.path)
	if err != nil {
		return
	}
	json.Unmarshal(data, em)
}

func (em *ExperienceMemory) save() {
	data, err := json.MarshalIndent(em, "", "  ")
	if err != nil {
		return
	}
	os.WriteFile(em.path, data, 0o644)
}
