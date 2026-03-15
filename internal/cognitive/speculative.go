package cognitive

import (
	"strings"
	"sync"
	"time"

	"github.com/artaeon/nous/internal/tools"
)

// SpeculativeExecutor runs multiple plausible tools in parallel based on
// keyword analysis of the user's query. Instead of asking the LLM to pick
// ONE tool (unreliable with 1.5B models), it fires ALL plausible tools
// simultaneously and presents the results to the LLM for summarization.
//
// Innovation: This is "Retrieval Augmented Action" — not RAG (which retrieves
// documents) but RAA (which retrieves tool execution results). The LLM's job
// shifts from "generate a tool call" (hard) to "summarize results" (easy).
//
// Example: "find where ReflectionGate is defined" triggers:
//   - grep "ReflectionGate" --glob "*.go" → 4 matches
//   - codebase index lookup "ReflectionGate" → struct at grounding.go:45
//
// The LLM receives both results and just summarizes.
type SpeculativeExecutor struct {
	tools     *tools.Registry
	intent    *IntentCompiler
	timeout   time.Duration
	maxParallel int
}

// SpecResult holds a single speculative tool execution result.
type SpecResult struct {
	Tool     string
	Args     map[string]string
	Result   string
	Err      error
	Duration time.Duration
}

// SpecBundle holds all speculative results for a query.
type SpecBundle struct {
	Query   string
	Results []SpecResult
	Total   time.Duration
}

// NewSpeculativeExecutor creates a new speculative executor.
func NewSpeculativeExecutor(toolReg *tools.Registry, intent *IntentCompiler) *SpeculativeExecutor {
	return &SpeculativeExecutor{
		tools:       toolReg,
		intent:      intent,
		timeout:     5 * time.Second,
		maxParallel: 4,
	}
}

// Execute analyzes the query and runs all plausible read-only tools in parallel.
// Only read-only tools are executed (never write/edit/shell).
func (se *SpeculativeExecutor) Execute(query string) *SpecBundle {
	start := time.Now()
	candidates := se.analyzeCandidates(query)

	if len(candidates) == 0 {
		return nil
	}

	// Cap parallel executions
	if len(candidates) > se.maxParallel {
		candidates = candidates[:se.maxParallel]
	}

	// Execute all candidates in parallel
	var wg sync.WaitGroup
	results := make([]SpecResult, len(candidates))

	for i, cand := range candidates {
		wg.Add(1)
		go func(idx int, tc toolCall) {
			defer wg.Done()
			toolStart := time.Now()

			t, err := se.tools.Get(tc.Name)
			if err != nil {
				results[idx] = SpecResult{
					Tool: tc.Name, Args: tc.Args, Err: err,
					Duration: time.Since(toolStart),
				}
				return
			}

			result, execErr := t.Execute(tc.Args)
			results[idx] = SpecResult{
				Tool:     tc.Name,
				Args:     tc.Args,
				Result:   SmartTruncate(tc.Name, result),
				Err:      execErr,
				Duration: time.Since(toolStart),
			}
		}(i, cand)
	}

	wg.Wait()

	// Filter to successful, non-empty results
	var valid []SpecResult
	for _, r := range results {
		if r.Err == nil && strings.TrimSpace(r.Result) != "" {
			valid = append(valid, r)
		}
	}

	if len(valid) == 0 {
		return nil
	}

	return &SpecBundle{
		Query:   query,
		Results: valid,
		Total:   time.Since(start),
	}
}

// FormatEvidence formats speculative results as evidence for the LLM.
func (sb *SpecBundle) FormatEvidence() string {
	if sb == nil || len(sb.Results) == 0 {
		return ""
	}

	var b strings.Builder
	b.WriteString("[Speculative results — tools were pre-executed based on your query]\n")

	for i, r := range sb.Results {
		b.WriteString("\n--- ")
		b.WriteString(r.Tool)
		if args := formatArgs(r.Args); args != "" {
			b.WriteString(" (")
			b.WriteString(args)
			b.WriteString(")")
		}
		b.WriteString(" ---\n")

		result := r.Result
		// Truncate individual results to keep total manageable
		if len(result) > 1500 {
			result = result[:1500] + "\n... (truncated)"
		}
		b.WriteString(result)

		if i < len(sb.Results)-1 {
			b.WriteString("\n")
		}
	}

	return b.String()
}

// analyzeCandidates determines which tools to execute speculatively.
// Only read-only tools are considered (never write/edit/shell).
func (se *SpeculativeExecutor) analyzeCandidates(query string) []toolCall {
	var candidates []toolCall
	lower := strings.ToLower(query)
	words := strings.Fields(lower)

	// Strategy 1: Intent-compiled actions (highest confidence)
	if se.intent != nil {
		if actions := se.intent.Compile(query); len(actions) > 0 {
			for _, a := range actions {
				if isReadOnlyTool(a.Tool) {
					candidates = append(candidates, toolCall{Name: a.Tool, Args: a.Args})
				}
			}
		}
	}

	// Strategy 2: Search-related queries → grep + glob
	if containsAny(lower, "find", "search", "where", "look for", "grep") {
		// Extract the search term
		pattern := extractSearchTerm(query)
		if pattern != "" {
			// Already have grep from intent compiler? Skip
			if !hasTool(candidates, "grep") {
				args := map[string]string{"pattern": pattern}
				if containsAny(lower, ".go", "go file") {
					args["glob"] = "*.go"
				}
				candidates = append(candidates, toolCall{Name: "grep", Args: args})
			}
		}
	}

	// Strategy 3: File reference → read
	if path := extractFilePath(query); path != "" {
		if !hasTool(candidates, "read") {
			resolved := path
			if se.intent != nil {
				if r := se.intent.ResolvePath(path); r != "" {
					resolved = r
				}
			}
			candidates = append(candidates, toolCall{
				Name: "read",
				Args: map[string]string{"path": resolved},
			})
		}
	}

	// Strategy 4: Directory queries → ls
	if containsAny(lower, "directory", "folder", "files in", "list") {
		if !hasTool(candidates, "ls") {
			dir := extractDirPath(query, words)
			candidates = append(candidates, toolCall{
				Name: "ls",
				Args: map[string]string{"path": dir},
			})
		}
	}

	// Strategy 5: Git queries → git status
	if containsAny(lower, "git", "commit", "branch", "changes", "modified") {
		if !hasTool(candidates, "git") {
			cmd := "status"
			if containsAny(lower, "log", "history") {
				cmd = "log --oneline -10"
			} else if containsAny(lower, "diff") {
				cmd = "diff"
			}
			candidates = append(candidates, toolCall{
				Name: "git",
				Args: map[string]string{"command": cmd},
			})
		}
	}

	return candidates
}

// isReadOnlyTool returns true if the tool cannot modify state.
func isReadOnlyTool(name string) bool {
	switch name {
	case "read", "ls", "tree", "glob", "grep", "sysinfo", "diff", "git":
		return true
	}
	return false
}

func containsAny(s string, substrs ...string) bool {
	for _, sub := range substrs {
		if strings.Contains(s, sub) {
			return true
		}
	}
	return false
}

func hasTool(candidates []toolCall, name string) bool {
	for _, c := range candidates {
		if c.Name == name {
			return true
		}
	}
	return false
}

func extractSearchTerm(query string) string {
	// Try quoted terms first
	for _, delim := range []string{`"`, `'`, "`"} {
		start := strings.Index(query, delim)
		if start >= 0 {
			end := strings.Index(query[start+1:], delim)
			if end >= 0 {
				return query[start+1 : start+1+end]
			}
		}
	}

	// Extract CamelCase identifiers
	for _, word := range strings.Fields(query) {
		clean := strings.Trim(word, "?!.,;:'\"")
		if len(clean) >= 3 && isCamelCase(clean) {
			return clean
		}
	}

	return ""
}

func isCamelCase(s string) bool {
	hasUpper := false
	hasLower := false
	for _, r := range s {
		if r >= 'A' && r <= 'Z' {
			hasUpper = true
		}
		if r >= 'a' && r <= 'z' {
			hasLower = true
		}
	}
	return hasUpper && hasLower && len(s) >= 3
}

func extractFilePath(query string) string {
	for _, word := range strings.Fields(query) {
		clean := strings.Trim(word, "?!.,;:'\"")
		if strings.Contains(clean, ".") && !strings.HasPrefix(clean, ".") {
			ext := clean[strings.LastIndex(clean, ".")+1:]
			if len(ext) >= 1 && len(ext) <= 5 {
				return clean
			}
		}
		if strings.Contains(clean, "/") {
			return clean
		}
	}
	return ""
}

func extractDirPath(query string, words []string) string {
	// Look for path-like tokens after "in" or "of"
	for i, w := range words {
		if (w == "in" || w == "of") && i+1 < len(words) {
			next := strings.Trim(words[i+1], "?!.,;:'\"")
			if strings.Contains(next, "/") || next == "." {
				return next
			}
		}
	}
	return ""
}
