package cognitive

import (
	"context"
	"fmt"
	"strings"

	"github.com/artaeon/nous/internal/blackboard"
	"github.com/artaeon/nous/internal/ollama"
)

// Reflector monitors the reasoning process and detects quality issues.
// It watches for completed actions and reasoning outputs, evaluates
// them for coherence and correctness, and triggers re-evaluation
// when problems are detected.
type Reflector struct {
	Base
}

func NewReflector(board *blackboard.Blackboard, llm *ollama.Client) *Reflector {
	return &Reflector{
		Base: Base{Board: board, LLM: llm},
	}
}

func (r *Reflector) Name() string { return "reflector" }

func (r *Reflector) Run(ctx context.Context) error {
	events := r.Board.Subscribe("action_recorded")

	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case ev := <-events:
			action, ok := ev.Payload.(blackboard.ActionRecord)
			if !ok {
				continue
			}
			r.reflect(action)
		}
	}
}

func (r *Reflector) reflect(action blackboard.ActionRecord) {
	// Skip reflection on quick, successful actions
	if action.Success && len(action.Output) < 100 {
		return
	}

	// Check for failures that need re-planning
	if !action.Success {
		r.Board.Set("reflection", fmt.Sprintf(
			"Action %s failed: %s. Consider an alternative approach.",
			action.Tool, action.Output,
		))
		r.Board.Set("needs_replan", action.StepID)
		return
	}

	// For complex outputs, use LLM to evaluate quality
	if len(action.Output) > 500 {
		r.evaluateOutput(action)
	}
}

func (r *Reflector) evaluateOutput(action blackboard.ActionRecord) {
	prompt := fmt.Sprintf(`Evaluate this action result. Is it correct and complete?
Tool: %s
Input: %s
Output (first 500 chars): %s

Respond with:
QUALITY: good|questionable|bad
ISSUE: <description if not good, otherwise "none">`, action.Tool, action.Input, truncate(action.Output, 500))

	resp, err := r.LLM.Chat([]ollama.Message{
		{Role: "system", Content: "You are a quality monitor. Evaluate action results briefly."},
		{Role: "user", Content: prompt},
	}, &ollama.ModelOptions{
		Temperature: 0.1,
		NumPredict:  100,
	})
	if err != nil {
		return // Reflection failure is non-critical
	}

	content := resp.Message.Content
	if strings.Contains(strings.ToLower(content), "bad") || strings.Contains(strings.ToLower(content), "questionable") {
		r.Board.Set("reflection", content)
		r.Board.Set("needs_replan", action.StepID)
	}
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "..."
}
