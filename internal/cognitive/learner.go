package cognitive

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/artaeon/nous/internal/blackboard"
	"github.com/artaeon/nous/internal/ollama"
)

// Pattern represents a learned behavioral pattern extracted from experience.
type Pattern struct {
	Trigger    string    `json:"trigger"`
	Response   string    `json:"response"`
	Confidence float64   `json:"confidence"`
	Uses       int       `json:"uses"`
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

	// Find the original percept that triggered this goal
	percept, ok := l.Board.LatestPercept()
	if !ok {
		return
	}

	// Extract a pattern: what input led to what successful plan
	var steps []string
	for _, s := range plan.Steps {
		if s.Status == "done" {
			steps = append(steps, fmt.Sprintf("%s: %s", s.Tool, s.Description))
		}
	}

	if len(steps) == 0 {
		return
	}

	pattern := Pattern{
		Trigger:    percept.Intent,
		Response:   fmt.Sprintf("Successful plan: %v", steps),
		Confidence: 0.5,
		Uses:       1,
		LastUsed:   time.Now(),
	}

	l.mu.Lock()
	l.patterns = append(l.patterns, pattern)
	l.mu.Unlock()

	_ = l.savePatterns()
}

// FindRelevantPatterns returns patterns matching the given intent.
func (l *Learner) FindRelevantPatterns(intent string) []Pattern {
	l.mu.RLock()
	defer l.mu.RUnlock()

	var relevant []Pattern
	for _, p := range l.patterns {
		if p.Trigger == intent {
			relevant = append(relevant, p)
		}
	}
	return relevant
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

	if err := os.MkdirAll(l.storePath, 0755); err != nil {
		return err
	}

	return os.WriteFile(filepath.Join(l.storePath, "patterns.json"), data, 0644)
}
