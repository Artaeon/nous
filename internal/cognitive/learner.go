package cognitive

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/artaeon/nous/internal/blackboard"
	"github.com/artaeon/nous/internal/ollama"
	"github.com/artaeon/nous/internal/safefile"
)

// Pattern represents a learned behavioral pattern extracted from experience.
type Pattern struct {
	Trigger    string    `json:"trigger"`
	ToolChain  []string  `json:"tool_chain"`
	Response   string    `json:"response"`
	Confidence float64   `json:"confidence"`
	Uses       int       `json:"uses"`
	Successes  int       `json:"successes"`
	LastUsed   time.Time `json:"last_used"`
}

// Learner observes successful interactions and extracts reusable patterns.
// It maintains a persistent pattern store that grows over time, allowing
// the system to improve without retraining the base model.
type Learner struct {
	Base
	patterns  []Pattern
	mu        sync.RWMutex
	storePath string
}

func NewLearner(board *blackboard.Blackboard, llm *ollama.Client, storePath string) *Learner {
	l := &Learner{
		Base:      Base{Board: board, LLM: llm},
		storePath: storePath,
	}
	l.loadPatterns()
	return l
}

func (l *Learner) Name() string { return "learner" }

func (l *Learner) Run(ctx context.Context) error {
	events := l.Board.Subscribe("goal_updated")

	for {
		select {
		case <-ctx.Done():
			return l.savePatterns()
		case ev := <-events:
			goalID, ok := ev.Payload.(string)
			if !ok {
				continue
			}
			l.learnFromGoal(goalID)
		}
	}
}

func (l *Learner) learnFromGoal(goalID string) {
	// Only learn from completed goals
	plan, ok := l.Board.PlanForGoal(goalID)
	if !ok || plan.Status != "completed" {
		return
	}

	// Extract the tool chain from completed steps
	var toolChain []string
	var stepDescriptions []string
	for _, s := range plan.Steps {
		if s.Status == "done" && s.Tool != "" {
			toolChain = append(toolChain, s.Tool)
			stepDescriptions = append(stepDescriptions, fmt.Sprintf("%s: %s", s.Tool, s.Description))
		}
	}

	if len(toolChain) == 0 {
		return
	}

	// Find the original percept for context
	trigger := ""
	if percept, ok := l.Board.LatestPercept(); ok {
		trigger = percept.Intent
		if trigger == "" {
			trigger = classifyTrigger(percept.Raw)
		}
	}

	// Check if we already have a pattern for this tool chain
	chainKey := strings.Join(toolChain, "→")
	l.mu.Lock()
	for i := range l.patterns {
		existing := strings.Join(l.patterns[i].ToolChain, "→")
		if existing == chainKey {
			// Reinforce existing pattern
			l.patterns[i].Uses++
			l.patterns[i].Successes++
			l.patterns[i].LastUsed = time.Now()
			l.patterns[i].Confidence = float64(l.patterns[i].Successes) / float64(l.patterns[i].Uses)
			l.mu.Unlock()
			_ = l.savePatterns()
			return
		}
	}

	// New pattern
	l.patterns = append(l.patterns, Pattern{
		Trigger:    trigger,
		ToolChain:  toolChain,
		Response:   fmt.Sprintf("Successful: %s", strings.Join(stepDescriptions, " → ")),
		Confidence: 0.5,
		Uses:       1,
		Successes:  1,
		LastUsed:   time.Now(),
	})

	// Prune to keep at most 100 patterns (remove lowest confidence)
	if len(l.patterns) > 100 {
		l.prunePatterns()
	}
	l.mu.Unlock()

	_ = l.savePatterns()
}

// classifyTrigger derives a simple trigger category from raw input.
func classifyTrigger(input string) string {
	lower := strings.ToLower(input)
	switch {
	case strings.Contains(lower, "read") || strings.Contains(lower, "show") || strings.Contains(lower, "what"):
		return "explore"
	case strings.Contains(lower, "fix") || strings.Contains(lower, "bug") || strings.Contains(lower, "error"):
		return "fix"
	case strings.Contains(lower, "write") || strings.Contains(lower, "create") || strings.Contains(lower, "add"):
		return "create"
	case strings.Contains(lower, "refactor") || strings.Contains(lower, "change") || strings.Contains(lower, "update"):
		return "modify"
	case strings.Contains(lower, "test") || strings.Contains(lower, "check") || strings.Contains(lower, "verify"):
		return "verify"
	default:
		return "general"
	}
}

// LearnFromTools records a successful tool sequence from direct reasoner interaction.
// This is the primary learning path — called after every successful multi-tool query.
func (l *Learner) LearnFromTools(trigger string, toolNames []string) {
	if len(toolNames) == 0 {
		return
	}

	intent := classifyTrigger(trigger)
	chainKey := strings.Join(toolNames, "→")

	l.mu.Lock()
	for i := range l.patterns {
		existing := strings.Join(l.patterns[i].ToolChain, "→")
		if existing == chainKey && l.patterns[i].Trigger == intent {
			l.patterns[i].Uses++
			l.patterns[i].Successes++
			l.patterns[i].LastUsed = time.Now()
			l.patterns[i].Confidence = float64(l.patterns[i].Successes) / float64(l.patterns[i].Uses)
			l.mu.Unlock()
			_ = l.savePatterns()
			return
		}
	}

	l.patterns = append(l.patterns, Pattern{
		Trigger:    intent,
		ToolChain:  toolNames,
		Response:   fmt.Sprintf("tools: %s", chainKey),
		Confidence: 0.5,
		Uses:       1,
		Successes:  1,
		LastUsed:   time.Now(),
	})
	if len(l.patterns) > 100 {
		l.prunePatterns()
	}
	l.mu.Unlock()
	_ = l.savePatterns()
}

// FindRelevantPatterns returns patterns matching the given intent, sorted by confidence.
func (l *Learner) FindRelevantPatterns(intent string) []Pattern {
	l.mu.RLock()
	defer l.mu.RUnlock()

	var relevant []Pattern
	for _, p := range l.patterns {
		if p.Trigger == intent || p.Confidence >= 0.8 {
			relevant = append(relevant, p)
		}
	}
	return relevant
}

// PatternCount returns the number of stored patterns.
func (l *Learner) PatternCount() int {
	l.mu.RLock()
	defer l.mu.RUnlock()
	return len(l.patterns)
}

// Patterns returns a copy of all patterns.
func (l *Learner) Patterns() []Pattern {
	l.mu.RLock()
	defer l.mu.RUnlock()
	out := make([]Pattern, len(l.patterns))
	copy(out, l.patterns)
	return out
}

func (l *Learner) prunePatterns() {
	// Remove lowest confidence patterns until under 100
	for len(l.patterns) > 100 {
		minIdx := 0
		minConf := l.patterns[0].Confidence
		for i, p := range l.patterns {
			if p.Confidence < minConf {
				minConf = p.Confidence
				minIdx = i
			}
		}
		l.patterns = append(l.patterns[:minIdx], l.patterns[minIdx+1:]...)
	}
}

func (l *Learner) loadPatterns() {
	if l.storePath == "" {
		return
	}

	path := filepath.Join(l.storePath, "patterns.json")
	data, err := os.ReadFile(path)
	if err != nil {
		return // No patterns file yet
	}

	l.mu.Lock()
	defer l.mu.Unlock()
	_ = json.Unmarshal(data, &l.patterns)
}

func (l *Learner) savePatterns() error {
	if l.storePath == "" {
		return nil
	}

	l.mu.RLock()
	data, err := json.MarshalIndent(l.patterns, "", "  ")
	l.mu.RUnlock()
	if err != nil {
		return err
	}

	return safefile.WriteAtomic(filepath.Join(l.storePath, "patterns.json"), data, 0644)
}
