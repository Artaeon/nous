package cognitive

import (
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/artaeon/nous/internal/blackboard"
)

// SuggestionType categorizes what triggered a proactive suggestion.
type SuggestionType string

const (
	SuggestIdle     SuggestionType = "idle"
	SuggestError    SuggestionType = "error_pattern"
	SuggestTime     SuggestionType = "time_based"
	SuggestContext  SuggestionType = "context"
	SuggestHand     SuggestionType = "hand_result"
)

// Suggestion is something nous proactively surfaces to the user.
type Suggestion struct {
	Type     SuggestionType
	Message  string
	Priority int    // 1 = low, 5 = critical
	Action   string // slash command the user can execute, e.g. "/checkin"
}

// ProactiveEngine monitors the blackboard and conversation context to
// generate non-intrusive suggestions. It respects a cooldown so the
// user is never spammed.
type ProactiveEngine struct {
	mu        sync.Mutex
	board     *blackboard.Blackboard
	lastCheck time.Time
	cooldown  time.Duration
	lastInput time.Time // track when the user last typed something
	suppressed bool     // true during focus mode
}

// NewProactiveEngine creates an engine that checks at most once per cooldown period.
func NewProactiveEngine(board *blackboard.Blackboard) *ProactiveEngine {
	return &ProactiveEngine{
		board:     board,
		cooldown:  5 * time.Minute,
		lastCheck: time.Now(),
		lastInput: time.Now(),
	}
}

// SetCooldown changes the minimum interval between suggestion rounds.
func (pe *ProactiveEngine) SetCooldown(d time.Duration) {
	pe.mu.Lock()
	defer pe.mu.Unlock()
	pe.cooldown = d
}

// RecordInput marks the current time as the last user activity.
func (pe *ProactiveEngine) RecordInput() {
	pe.mu.Lock()
	defer pe.mu.Unlock()
	pe.lastInput = time.Now()
}

// SetSuppressed toggles suppression (e.g. during focus mode).
func (pe *ProactiveEngine) SetSuppressed(v bool) {
	pe.mu.Lock()
	defer pe.mu.Unlock()
	pe.suppressed = v
}

// Check evaluates all proactive triggers and returns any applicable suggestions.
// Returns nil if the cooldown has not elapsed or if suppressed.
func (pe *ProactiveEngine) Check() []Suggestion {
	pe.mu.Lock()
	if pe.suppressed {
		pe.mu.Unlock()
		return nil
	}
	now := time.Now()
	if now.Sub(pe.lastCheck) < pe.cooldown {
		pe.mu.Unlock()
		return nil
	}
	pe.lastCheck = now
	lastInput := pe.lastInput
	pe.mu.Unlock()

	var suggestions []Suggestion

	// 1. Idle detection: if no input for 5+ minutes
	if now.Sub(lastInput) > 5*time.Minute {
		suggestions = append(suggestions, Suggestion{
			Type:     SuggestIdle,
			Message:  "You've been quiet for a while. Need a /checkin or want to review /tasks?",
			Priority: 1,
			Action:   "/tasks",
		})
	}

	// 2. Error patterns: if last 3 actions all failed
	recent := pe.board.RecentActions(3)
	if len(recent) >= 3 {
		allFailed := true
		for _, a := range recent {
			if a.Success {
				allFailed = false
				break
			}
		}
		if allFailed {
			suggestions = append(suggestions, Suggestion{
				Type:     SuggestError,
				Message:  "The last 3 tool calls failed. Consider trying a different approach or checking /status.",
				Priority: 3,
				Action:   "/status",
			})
		}
	}

	// 3. Time-based: morning greeting
	hour := now.Hour()
	if hour >= 6 && hour <= 9 {
		suggestions = append(suggestions, Suggestion{
			Type:     SuggestTime,
			Message:  "Good morning! Check your /briefing for today's agenda.",
			Priority: 2,
			Action:   "/briefing",
		})
	}
	// End-of-day summary
	if hour >= 17 && hour <= 19 {
		suggestions = append(suggestions, Suggestion{
			Type:     SuggestTime,
			Message:  "Wrapping up? Try /review for an end-of-day summary.",
			Priority: 2,
			Action:   "/review",
		})
	}

	// 4. Context-based: if recent actions involve file editing, suggest testing
	if len(recent) > 0 {
		hasFileEdit := false
		for _, a := range recent {
			tool := strings.ToLower(a.Tool)
			if tool == "write_file" || tool == "patch_file" || tool == "edit_file" {
				hasFileEdit = true
				break
			}
		}
		if hasFileEdit {
			// Check if any recently edited file looks like Go
			for _, a := range recent {
				if strings.HasSuffix(a.Input, ".go") {
					suggestions = append(suggestions, Suggestion{
						Type:     SuggestContext,
						Message:  "You've been editing Go files. Don't forget to run tests!",
						Priority: 2,
						Action:   "/plan run go tests",
					})
					break
				}
			}
		}
	}

	// 5. Hand completion: check if any hand recently completed
	if v, ok := pe.board.Get("hand_completed"); ok {
		if name, ok := v.(string); ok && name != "" {
			suggestions = append(suggestions, Suggestion{
				Type:     SuggestHand,
				Message:  fmt.Sprintf("Hand %q just completed. Check results with /hand status %s", name, name),
				Priority: 2,
				Action:   "/hand status " + name,
			})
		}
	}

	return suggestions
}

// FormatSuggestions renders suggestions as dim, non-intrusive terminal output.
func FormatSuggestions(suggestions []Suggestion) string {
	if len(suggestions) == 0 {
		return ""
	}
	var b strings.Builder
	b.WriteString("\n")
	for _, s := range suggestions {
		b.WriteString(fmt.Sprintf("  %s hint:%s %s", ColorDim, ColorReset, s.Message))
		if s.Action != "" {
			b.WriteString(fmt.Sprintf(" %s[%s]%s", ColorDim, s.Action, ColorReset))
		}
		b.WriteString("\n")
	}
	return b.String()
}
