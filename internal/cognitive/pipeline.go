package cognitive

import (
	"fmt"
	"path/filepath"
	"strings"
	"time"

	"github.com/artaeon/nous/internal/ollama"
)

// StepResult holds the compressed outcome of one reasoning step.
type StepResult struct {
	StepNum   int
	ToolName  string
	Summary   string // one-line compressed summary of the tool result
	RawResult string // the actual result (kept for the current step only)
}

// Pipeline manages fresh-context reasoning across multiple steps.
// Instead of accumulating messages that fill the context window,
// each step gets a fresh LLM call with only:
// 1. A compact system prompt
// 2. The original user question
// 3. One-line summaries of all previous steps
// 4. The current tool result (if any)
type Pipeline struct {
	steps     []StepResult
	userQuery string
	distiller *ollama.Client // optional: fast model for thought distillation
}

// NewPipeline creates a pipeline for managing fresh-context reasoning.
func NewPipeline(query string) *Pipeline {
	return &Pipeline{
		userQuery: query,
	}
}

// SetDistiller configures an LLM client (typically the fast model) for
// thought distillation. When set, step summaries are produced by the LLM
// instead of rule-based compression, yielding much richer context for the
// reasoning model.
func (p *Pipeline) SetDistiller(llm *ollama.Client) {
	p.distiller = llm
}

// AddStep compresses and stores the result of a tool execution.
// Uses thought distillation (LLM-based) when a distiller is configured,
// falling back to rule-based compression otherwise.
func (p *Pipeline) AddStep(toolName, rawResult string) {
	summary := ""

	// Try thought distillation first
	if p.distiller != nil && len(rawResult) > 80 {
		if distilled := distillStep(p.distiller, p.userQuery, toolName, rawResult); distilled != "" {
			summary = distilled
		}
	}

	// Fall back to rule-based compression
	if summary == "" {
		summary = CompressStep(toolName, rawResult)
	}

	step := StepResult{
		StepNum:   len(p.steps) + 1,
		ToolName:  toolName,
		Summary:   summary,
		RawResult: rawResult,
	}
	// Clear RawResult from all previous steps to save memory
	for i := range p.steps {
		p.steps[i].RawResult = ""
	}
	p.steps = append(p.steps, step)
}

// distillStep uses the fast model to produce a semantically rich summary
// of a tool result. Returns empty string on failure (caller falls back to
// rule-based compression). Timeout: 3 seconds to avoid blocking.
func distillStep(llm *ollama.Client, userQuery, toolName, rawResult string) string {
	if llm == nil {
		return ""
	}

	// Truncate very large results before sending to distiller
	result := rawResult
	if len(result) > 1500 {
		result = result[:1500] + "\n... (truncated)"
	}

	prompt := fmt.Sprintf(`Tool "%s" returned this result. Summarize what it means for the task in one sentence (max 40 words). Focus on key content and findings, not metadata.

Task: %s

Result:
%s

Summary:`, toolName, userQuery, result)

	// Use a channel + goroutine for timeout control
	type resp struct {
		text string
		err  error
	}
	ch := make(chan resp, 1)
	go func() {
		r, err := llm.Chat([]ollama.Message{
			{Role: "user", Content: prompt},
		}, &ollama.ModelOptions{
			Temperature: 0.1,
			NumPredict:  60,
		})
		if err != nil {
			ch <- resp{err: err}
			return
		}
		ch <- resp{text: r.Message.Content}
	}()

	select {
	case r := <-ch:
		if r.err != nil {
			return ""
		}
		// Clean up: take first line, trim whitespace
		summary := strings.TrimSpace(r.text)
		if idx := strings.IndexByte(summary, '\n'); idx > 0 {
			summary = strings.TrimSpace(summary[:idx])
		}
		// Sanity: if distillation returned garbage or too long, reject
		if len(summary) < 5 || len(summary) > 200 {
			return ""
		}
		return fmt.Sprintf("[%s] %s", toolName, summary)
	case <-time.After(3 * time.Second):
		return "" // timeout — fall back to rule-based
	}
}

// CompressStep applies rule-based compression to produce a one-line summary.
// This is purely rule-based — no LLM call — to keep it fast and deterministic.
func CompressStep(toolName, rawResult string) string {
	// Handle errors first
	if strings.HasPrefix(rawResult, "Error:") || strings.HasPrefix(rawResult, "error:") {
		first := firstLine(rawResult)
		return "Error: " + first
	}

	lines := strings.Split(rawResult, "\n")

	switch toolName {
	case "read":
		// "Read FILE_PATH: FIRST_LINE... (N lines)"
		firstMeaningful := firstNonEmptyLine(lines)
		if len(firstMeaningful) > 60 {
			firstMeaningful = firstMeaningful[:60] + "..."
		}
		path := extractPath(rawResult)
		if path != "" {
			return fmt.Sprintf("Read %s: %s (%d lines)", path, firstMeaningful, len(lines))
		}
		return fmt.Sprintf("Read file: %s (%d lines)", firstMeaningful, len(lines))

	case "ls", "tree":
		// "Listed DIR: N entries including [first 3 names]..."
		entries := nonEmptyLines(lines)
		names := firstN(entries, 3)
		dir := extractDir(rawResult)
		if dir == "" {
			dir = "directory"
		}
		return fmt.Sprintf("Listed %s: %d entries including %s", dir, len(entries), strings.Join(names, ", "))

	case "grep":
		// "Searched PATTERN: N matches in [files]"
		matches := nonEmptyLines(lines)
		files := extractUniqueFiles(matches)
		fileStr := strings.Join(firstN(files, 3), ", ")
		if len(files) > 3 {
			fileStr += fmt.Sprintf(" (+%d more)", len(files)-3)
		}
		return fmt.Sprintf("Searched: %d matches in %s", len(matches), fileStr)

	case "glob":
		// "Found N files matching PATTERN"
		entries := nonEmptyLines(lines)
		return fmt.Sprintf("Found %d files matching pattern", len(entries))

	case "git":
		// "Git COMMAND: FIRST_LINE_OF_OUTPUT"
		first := firstNonEmptyLine(lines)
		if len(first) > 80 {
			first = first[:80] + "..."
		}
		return fmt.Sprintf("Git: %s", first)

	case "write", "edit", "patch", "find_replace", "replace_all":
		// "Modified FILE_PATH"
		path := extractPath(rawResult)
		if path != "" {
			return fmt.Sprintf("Modified %s", path)
		}
		return "Modified file"

	case "sysinfo":
		// "System: OS ARCH CPUs"
		first := firstNonEmptyLine(lines)
		if len(first) > 80 {
			first = first[:80] + "..."
		}
		return fmt.Sprintf("System: %s", first)

	default:
		// Default: first 80 chars of result
		trimmed := strings.TrimSpace(rawResult)
		if len(trimmed) <= 80 {
			return trimmed
		}
		return trimmed[:80] + "..."
	}
}

// BuildContext returns the accumulated step summaries as a formatted block.
// Returns empty string if no steps have been recorded.
func (p *Pipeline) BuildContext() string {
	if len(p.steps) == 0 {
		return ""
	}

	var sb strings.Builder
	sb.WriteString("[Previous steps]\n")
	for _, s := range p.steps {
		sb.WriteString(fmt.Sprintf("%d. %s\n", s.StepNum, s.Summary))
	}
	return strings.TrimRight(sb.String(), "\n")
}

// StepCount returns the number of steps completed so far.
func (p *Pipeline) StepCount() int {
	return len(p.steps)
}

// LastResult returns the raw result of the most recent step.
// Returns empty string if no steps exist.
func (p *Pipeline) LastResult() string {
	if len(p.steps) == 0 {
		return ""
	}
	return p.steps[len(p.steps)-1].RawResult
}

// --- helper functions ---

func firstLine(s string) string {
	if idx := strings.IndexByte(s, '\n'); idx >= 0 {
		return strings.TrimSpace(s[:idx])
	}
	return strings.TrimSpace(s)
}

func firstNonEmptyLine(lines []string) string {
	for _, l := range lines {
		trimmed := strings.TrimSpace(l)
		if trimmed != "" {
			return trimmed
		}
	}
	return ""
}

func nonEmptyLines(lines []string) []string {
	var result []string
	for _, l := range lines {
		if strings.TrimSpace(l) != "" {
			result = append(result, strings.TrimSpace(l))
		}
	}
	return result
}

func firstN(items []string, n int) []string {
	if len(items) <= n {
		return items
	}
	return items[:n]
}

// extractPath tries to find a file path in a tool result.
// Many tool results start with the path or contain "path:" references.
func extractPath(result string) string {
	lines := strings.Split(result, "\n")
	for _, line := range lines {
		trimmed := strings.TrimSpace(line)
		// Look for common path patterns
		if strings.Contains(trimmed, "/") || strings.Contains(trimmed, ".go") ||
			strings.Contains(trimmed, ".py") || strings.Contains(trimmed, ".js") ||
			strings.Contains(trimmed, ".ts") || strings.Contains(trimmed, ".md") {
			// Extract just the path-like portion
			words := strings.Fields(trimmed)
			for _, w := range words {
				if strings.Contains(w, "/") || strings.Contains(w, ".") {
					// Clean up common prefixes/suffixes
					w = strings.TrimRight(w, ":")
					if filepath.Ext(w) != "" || strings.Contains(w, "/") {
						return w
					}
				}
			}
		}
	}
	return ""
}

// extractDir tries to find a directory reference in a tool result.
func extractDir(result string) string {
	path := extractPath(result)
	if path != "" {
		if strings.HasSuffix(path, "/") {
			return path
		}
		dir := filepath.Dir(path)
		if dir != "." {
			return dir
		}
	}
	return ""
}

// extractUniqueFiles pulls file paths from grep-style output (path:line:content).
func extractUniqueFiles(lines []string) []string {
	seen := make(map[string]bool)
	var files []string
	for _, line := range lines {
		// grep output format: "file:line:content" or "file:content"
		if idx := strings.IndexByte(line, ':'); idx > 0 {
			candidate := line[:idx]
			if !seen[candidate] && (strings.Contains(candidate, "/") ||
				strings.Contains(candidate, ".")) {
				seen[candidate] = true
				files = append(files, candidate)
			}
		}
	}
	if len(files) == 0 {
		return []string{"(inline)"}
	}
	return files
}
