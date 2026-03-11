package cognitive

import (
	"crypto/sha256"
	"fmt"
	"strings"

	"github.com/artaeon/nous/internal/ollama"
)

// ContextBudget tracks token consumption and prevents context overflow.
// Small models (1.5B) degrade severely when >70% of context is consumed.
type ContextBudget struct {
	MaxTokens     int
	CharsPerToken float64
}

// DefaultBudget returns a budget tuned for qwen2.5:1.5b (4096 context).
func DefaultBudget() *ContextBudget {
	return &ContextBudget{
		MaxTokens:     4096,
		CharsPerToken: 4.0,
	}
}

// EstimateTokens returns an approximate token count for the given text.
func (b *ContextBudget) EstimateTokens(text string) int {
	if b.CharsPerToken <= 0 {
		return len(text) / 4
	}
	return int(float64(len(text)) / b.CharsPerToken)
}

// EstimateMessages returns total token estimate across all messages.
func (b *ContextBudget) EstimateMessages(msgs []ollama.Message) int {
	total := 0
	for _, m := range msgs {
		// Account for role prefix overhead (~4 tokens per message)
		total += 4 + b.EstimateTokens(m.Content)
	}
	return total
}

// Remaining returns estimated tokens still available.
func (b *ContextBudget) Remaining(msgs []ollama.Message) int {
	used := b.EstimateMessages(msgs)
	if used >= b.MaxTokens {
		return 0
	}
	return b.MaxTokens - used
}

// UsagePercent returns what fraction of the budget is consumed (0.0-1.0).
func (b *ContextBudget) UsagePercent(msgs []ollama.Message) float64 {
	if b.MaxTokens <= 0 {
		return 1.0
	}
	return float64(b.EstimateMessages(msgs)) / float64(b.MaxTokens)
}

// ShouldCompress returns true when context usage exceeds 75%.
func (b *ContextBudget) ShouldCompress(msgs []ollama.Message) bool {
	return b.UsagePercent(msgs) > 0.75
}

// ShouldForceAnswer returns true when context usage exceeds 85%.
func (b *ContextBudget) ShouldForceAnswer(msgs []ollama.Message) bool {
	return b.UsagePercent(msgs) > 0.85
}

// SmartTruncate applies tool-specific truncation to keep results compact.
// For code files, preserves line numbers so the LLM can reference specific lines.
// This is critical for small models where every token matters.
func SmartTruncate(toolName, result string) string {
	lines := strings.Split(result, "\n")

	switch toolName {
	case "read":
		// Keep first 25 + landmark lines from middle + last 15
		// Include line markers so LLM can say "edit line 156"
		if len(lines) > 50 {
			head := strings.Join(lines[:25], "\n")
			tail := strings.Join(lines[len(lines)-15:], "\n")

			// Sample every 20th line from the middle for navigation
			var landmarks []string
			for i := 25; i < len(lines)-15; i += 20 {
				trimmed := strings.TrimSpace(lines[i])
				if trimmed != "" && trimmed != "{" && trimmed != "}" {
					landmarks = append(landmarks, fmt.Sprintf("  [line %d] %s", i+1, trimmed))
				}
			}

			omitted := len(lines) - 40
			midSection := fmt.Sprintf("[...%d lines omitted. Landmarks from middle:]", omitted)
			if len(landmarks) > 0 {
				midSection += "\n" + strings.Join(landmarks, "\n")
			}
			midSection += "\n[Use read with offset/limit to see specific sections]"

			return fmt.Sprintf("%s\n%s\n%s", head, midSection, tail)
		}
	case "grep", "glob":
		// Cap search results at 20 matches (increased from 15)
		if len(lines) > 20 {
			return strings.Join(lines[:20], "\n") + fmt.Sprintf("\n...and %d more", len(lines)-20)
		}
	case "tree", "ls":
		// Cap directory listings at 30 entries
		if len(lines) > 30 {
			return strings.Join(lines[:30], "\n") + fmt.Sprintf("\n...and %d more entries", len(lines)-30)
		}
	}

	// Universal hard limit
	if len(result) > 2048 {
		return result[:2048] + "\n... (truncated)"
	}
	return result
}

// ValidateToolResult checks a tool result for common issues and returns
// a cleaned result plus an optional warning hint for the model.
func ValidateToolResult(toolName, result string, err error) (string, string) {
	if err != nil {
		errStr := err.Error()
		var hint string
		switch {
		case strings.Contains(errStr, "no such file") || strings.Contains(errStr, "not found"):
			hint = "Path not found. Use ls or glob to find the correct path."
		case strings.Contains(errStr, "permission denied"):
			hint = "Permission denied. Try a different path."
		case strings.Contains(errStr, "is a directory"):
			hint = "That's a directory. Use ls to list its contents."
		default:
			hint = "Tool failed. Try a different approach."
		}
		return fmt.Sprintf("Error: %v", err), hint
	}

	// Check for suspiciously empty results from tools that should return content
	trimmed := strings.TrimSpace(result)
	switch toolName {
	case "read":
		if trimmed == "" {
			return result, "File appears empty. Verify the path is correct."
		}
	case "grep":
		if trimmed == "" {
			return "No matches found.", ""
		}
	case "glob":
		if trimmed == "" {
			return "No files matched the pattern.", ""
		}
	}

	return result, ""
}

// ReflectionGate performs synchronous rule-based validation after each tool call.
// Unlike the async Reflector stream, this runs inline in the reasoning loop
// and can inject corrective hints before the next LLM call.
type ReflectionGate struct {
	toolCallCount    int
	consecutiveEmpty int
	recentCalls      [4]string // circular buffer of "tool:argHash" for repetition detection
	recentIdx        int
}

// Reset clears the gate state for a new reasoning cycle.
func (g *ReflectionGate) Reset() {
	g.toolCallCount = 0
	g.consecutiveEmpty = 0
	g.recentCalls = [4]string{}
	g.recentIdx = 0
}

// CheckResult describes what the gate recommends.
type CheckResult struct {
	Hint       string // corrective hint to inject (empty = no hint)
	ForceStop  bool   // true = stop tool loop and force a final answer
}

// Check evaluates a tool result and returns a corrective hint (or empty CheckResult).
func (g *ReflectionGate) Check(toolName, result string, err error) CheckResult {
	g.toolCallCount++

	// Track this call for repetition detection
	callSig := toolName + ":" + shortHash(result)
	g.recentCalls[g.recentIdx%4] = callSig
	g.recentIdx++

	// Check for errors
	if err != nil {
		return CheckResult{} // ValidateToolResult already provides error hints
	}

	// Track empty results
	if strings.TrimSpace(result) == "" || strings.TrimSpace(result) == "No matches found." {
		g.consecutiveEmpty++
		if g.consecutiveEmpty >= 3 {
			return CheckResult{Hint: "Multiple tools returned empty. Stopping.", ForceStop: true}
		}
		if g.consecutiveEmpty >= 2 {
			return CheckResult{Hint: "Multiple tools returned empty results. Try a different approach."}
		}
	} else {
		g.consecutiveEmpty = 0
	}

	// Detect repetition: same tool call signature appearing twice in last 4
	if g.toolCallCount >= 2 {
		seen := make(map[string]int)
		for _, sig := range g.recentCalls {
			if sig != "" {
				seen[sig]++
			}
		}
		for _, count := range seen {
			if count >= 3 {
				return CheckResult{Hint: "Repeating the same call. Forcing answer.", ForceStop: true}
			}
			if count >= 2 {
				return CheckResult{Hint: "You already have this information. Answer now based on what you know. Do NOT call another tool."}
			}
		}
	}

	// Nudge if too many iterations (allow up to 6 before forcing)
	if g.toolCallCount >= 6 {
		return CheckResult{Hint: "Answer now with what you have. Do NOT call another tool.", ForceStop: true}
	}

	return CheckResult{}
}

func shortHash(s string) string {
	h := sha256.Sum256([]byte(s))
	return fmt.Sprintf("%x", h[:4])
}
