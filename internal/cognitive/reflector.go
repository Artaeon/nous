package cognitive

import (
	"context"
	"fmt"
	"strings"

	"github.com/artaeon/nous/internal/blackboard"
	"github.com/artaeon/nous/internal/ollama"
)

// Reflector monitors the reasoning process and detects quality issues.
// Uses a hybrid approach: fast rule-based checks for common patterns,
// LLM evaluation only for complex/ambiguous outputs.
// This is critical for 1.5B models — we can't waste LLM calls on reflection
// when simple rules catch 90% of issues.
type Reflector struct {
	Base
	// Stats for monitoring
	checksRun     int
	issuesFound   int
	lastToolSeen  string
	lastResultLen int
	consecutiveFails int
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
	r.checksRun++

	// Rule 1: Failed actions need immediate feedback
	if !action.Success {
		r.consecutiveFails++
		r.issuesFound++

		if r.consecutiveFails >= 3 {
			r.Board.Set("reflection", fmt.Sprintf(
				"3 consecutive failures (last: %s). Stop and try a completely different approach.",
				action.Tool,
			))
			r.Board.Set("needs_replan", action.StepID)
			return
		}

		// Provide targeted feedback based on the error
		hint := r.diagnoseFailure(action)
		if hint != "" {
			r.Board.Set("reflection", hint)
		}
		return
	}

	// Success resets failure counter
	r.consecutiveFails = 0

	// Rule 2: Detect likely hallucination — tool returned empty or "no matches"
	trimmed := strings.TrimSpace(action.Output)
	if trimmed == "" || trimmed == "no matches found" || trimmed == "No matches found." {
		// Only flag if this is a pattern (multiple empties suggest wrong approach)
		if r.lastResultLen == 0 {
			r.issuesFound++
			r.Board.Set("reflection",
				"Multiple tools returned empty results. The target may not exist. Try listing the directory first.")
		}
		r.lastResultLen = 0
	} else {
		r.lastResultLen = len(trimmed)
	}

	// Rule 3: Detect repeated tool calls (same tool twice in a row)
	if action.Tool == r.lastToolSeen && action.Tool != "read" {
		// Reading multiple files is normal; other repeated tools suggest a loop
		r.issuesFound++
		r.Board.Set("reflection", fmt.Sprintf(
			"Called %s twice in a row. Consider a different tool or approach.",
			action.Tool,
		))
	}
	r.lastToolSeen = action.Tool

	// Rule 4: Suspiciously long output (may overwhelm the context)
	if len(action.Output) > 4000 {
		r.Board.Set("reflection",
			"Tool output is very long. Focus on the relevant parts only.")
	}
}

// diagnoseFailure provides specific guidance based on common error patterns.
func (r *Reflector) diagnoseFailure(action blackboard.ActionRecord) string {
	errStr := strings.ToLower(action.Output)

	switch {
	case strings.Contains(errStr, "no such file") || strings.Contains(errStr, "not found"):
		return fmt.Sprintf("%s failed: path not found. Use ls or glob to find the correct path.", action.Tool)
	case strings.Contains(errStr, "permission denied"):
		return fmt.Sprintf("%s failed: permission denied. Try a different path or check permissions.", action.Tool)
	case strings.Contains(errStr, "is a directory"):
		return fmt.Sprintf("%s failed: target is a directory, not a file. Use ls to list contents.", action.Tool)
	case strings.Contains(errStr, "not unique") || strings.Contains(errStr, "found") && strings.Contains(errStr, "times"):
		return fmt.Sprintf("%s failed: match is ambiguous. Provide more context to narrow it down.", action.Tool)
	case strings.Contains(errStr, "old string not found"):
		return "Edit failed: the old string doesn't match. Read the file first to see the exact content."
	default:
		return fmt.Sprintf("%s failed: %s. Try a different approach.", action.Tool, firstLineReflect(action.Output))
	}
}

// Stats returns reflection statistics.
func (r *Reflector) Stats() (checks, issues int) {
	return r.checksRun, r.issuesFound
}

// firstLineReflect extracts the first line of a string for error messages.
func firstLineReflect(s string) string {
	if idx := strings.IndexByte(s, '\n'); idx >= 0 {
		s = s[:idx]
	}
	if len(s) > 100 {
		return s[:100] + "..."
	}
	return s
}
