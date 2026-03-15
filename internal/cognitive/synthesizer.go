package cognitive

import (
	"fmt"
	"path/filepath"
	"strings"
)

// ResponseSynthesizer generates human-readable responses from tool results
// WITHOUT using the LLM. This is the key innovation: for tool-based queries,
// the response is assembled deterministically from facts, not hallucinated
// by a model that ignores its own evidence.
//
// Innovation: Every existing AI assistant sends tool results to the LLM
// and says "summarize this." A 1.5B model then hallucinates "no results
// found" while staring at 20 grep matches. The Synthesizer eliminates
// this entirely — it formats tool results directly into natural language
// using templates that are IMPOSSIBLE to get wrong.
//
// This is "Cognitive Exocortex" thinking: the response IS the data,
// formatted for human consumption. No interpretation step needed.
type ResponseSynthesizer struct{}

// NewResponseSynthesizer creates a new response synthesizer.
func NewResponseSynthesizer() *ResponseSynthesizer {
	return &ResponseSynthesizer{}
}

// SynthesizeResult holds a synthesized response.
type SynthesizeResult struct {
	Response string
	Tool     string
	Success  bool
}

// Synthesize generates a natural language response from tool execution results.
// Returns empty string if it can't synthesize (caller should fall back to LLM).
func (rs *ResponseSynthesizer) Synthesize(tool string, args map[string]string, result string, err error) string {
	if err != nil {
		return rs.synthesizeError(tool, args, err)
	}

	result = strings.TrimSpace(result)
	if result == "" {
		return rs.synthesizeEmpty(tool, args)
	}

	switch tool {
	case "grep":
		return rs.synthesizeGrep(args, result)
	case "read":
		return rs.synthesizeRead(args, result)
	case "ls":
		return rs.synthesizeLs(args, result)
	case "tree":
		return rs.synthesizeTree(args, result)
	case "glob":
		return rs.synthesizeGlob(args, result)
	case "git":
		return rs.synthesizeGit(args, result)
	case "write":
		return rs.synthesizeWrite(args, result)
	case "edit":
		return rs.synthesizeEdit(args, result)
	default:
		return rs.synthesizeGeneric(tool, result)
	}
}

// SynthesizeMulti generates a response from multiple tool results.
func (rs *ResponseSynthesizer) SynthesizeMulti(query string, steps []synthStep) string {
	if len(steps) == 0 {
		return ""
	}

	// Single step — use direct synthesis
	if len(steps) == 1 {
		return rs.Synthesize(steps[0].Tool, steps[0].Args, steps[0].Result, steps[0].Err)
	}

	// Multiple steps — combine results
	var b strings.Builder

	for i, step := range steps {
		if step.Err != nil {
			continue
		}
		result := strings.TrimSpace(step.Result)
		if result == "" {
			continue
		}

		single := rs.Synthesize(step.Tool, step.Args, step.Result, step.Err)
		if single != "" {
			if i > 0 {
				b.WriteString("\n\n")
			}
			b.WriteString(single)
		}
	}

	return b.String()
}

// synthStep holds one tool execution for multi-step synthesis.
type synthStep struct {
	Tool   string
	Args   map[string]string
	Result string
	Err    error
}

// --- Tool-Specific Synthesizers ---

func (rs *ResponseSynthesizer) synthesizeGrep(args map[string]string, result string) string {
	pattern := args["pattern"]
	glob := args["glob"]
	lines := strings.Split(result, "\n")

	// Count actual matches (non-empty lines)
	var matches []string
	for _, line := range lines {
		trimmed := strings.TrimSpace(line)
		if trimmed != "" && !strings.HasPrefix(trimmed, "...and ") {
			matches = append(matches, line)
		}
	}

	if len(matches) == 0 {
		scope := "the project"
		if glob != "" {
			scope = glob + " files"
		}
		return fmt.Sprintf("No matches found for `%s` in %s.", pattern, scope)
	}

	var b strings.Builder

	// Header
	scope := ""
	if glob != "" {
		scope = fmt.Sprintf(" in %s files", glob)
	}

	if len(matches) == 1 {
		b.WriteString(fmt.Sprintf("Found 1 match for `%s`%s:\n\n", pattern, scope))
	} else {
		b.WriteString(fmt.Sprintf("Found %d matches for `%s`%s:\n\n", len(matches), pattern, scope))
	}

	// Group matches by file
	fileMatches := groupByFile(matches)
	for _, fg := range fileMatches {
		b.WriteString(fmt.Sprintf("**%s**\n", fg.file))
		for _, m := range fg.matches {
			b.WriteString(fmt.Sprintf("  %s\n", m))
		}
		b.WriteString("\n")
	}

	// Overflow notice
	for _, line := range lines {
		if strings.HasPrefix(strings.TrimSpace(line), "...and ") {
			b.WriteString(strings.TrimSpace(line))
			b.WriteString("\n")
		}
	}

	return strings.TrimSpace(b.String())
}

func (rs *ResponseSynthesizer) synthesizeRead(args map[string]string, result string) string {
	path := args["path"]
	if path == "" {
		path = args["file"]
	}

	lines := strings.Split(result, "\n")
	lineCount := len(lines)

	// Short file — just show it
	if lineCount <= 30 {
		return fmt.Sprintf("**%s** (%d lines):\n\n```\n%s\n```", filepath.Base(path), lineCount, result)
	}

	// Long file — show preview with structure
	var b strings.Builder
	b.WriteString(fmt.Sprintf("**%s** (%d lines):\n\n", filepath.Base(path), lineCount))

	// Show first 20 lines
	preview := lines
	if len(preview) > 20 {
		preview = preview[:20]
	}
	b.WriteString("```\n")
	for _, l := range preview {
		b.WriteString(l)
		b.WriteString("\n")
	}
	b.WriteString("```\n")

	if lineCount > 20 {
		b.WriteString(fmt.Sprintf("\n... and %d more lines. Use `read %s` with offset to see more.", lineCount-20, path))
	}

	return strings.TrimSpace(b.String())
}

func (rs *ResponseSynthesizer) synthesizeLs(args map[string]string, result string) string {
	dir := args["path"]
	if dir == "" {
		dir = "."
	}

	lines := strings.Split(result, "\n")
	var entries []string
	for _, l := range lines {
		if strings.TrimSpace(l) != "" {
			entries = append(entries, strings.TrimSpace(l))
		}
	}

	if len(entries) == 0 {
		return fmt.Sprintf("Directory `%s` is empty.", dir)
	}

	var b strings.Builder
	b.WriteString(fmt.Sprintf("Contents of `%s` (%d entries):\n\n", dir, len(entries)))

	for _, entry := range entries {
		b.WriteString("  ")
		b.WriteString(entry)
		b.WriteString("\n")
	}

	return strings.TrimSpace(b.String())
}

func (rs *ResponseSynthesizer) synthesizeTree(args map[string]string, result string) string {
	dir := args["path"]
	if dir == "" {
		dir = "."
	}

	lines := strings.Split(result, "\n")
	nonEmpty := 0
	for _, l := range lines {
		if strings.TrimSpace(l) != "" {
			nonEmpty++
		}
	}

	return fmt.Sprintf("Project structure of `%s` (%d entries):\n\n```\n%s\n```", dir, nonEmpty, result)
}

func (rs *ResponseSynthesizer) synthesizeGlob(args map[string]string, result string) string {
	pattern := args["pattern"]
	lines := strings.Split(result, "\n")
	var files []string
	for _, l := range lines {
		if strings.TrimSpace(l) != "" {
			files = append(files, strings.TrimSpace(l))
		}
	}

	if len(files) == 0 {
		return fmt.Sprintf("No files match the pattern `%s`.", pattern)
	}

	var b strings.Builder
	if len(files) == 1 {
		b.WriteString(fmt.Sprintf("Found 1 file matching `%s`:\n\n", pattern))
	} else {
		b.WriteString(fmt.Sprintf("Found %d files matching `%s`:\n\n", len(files), pattern))
	}

	for _, f := range files {
		b.WriteString("  ")
		b.WriteString(f)
		b.WriteString("\n")
	}

	return strings.TrimSpace(b.String())
}

func (rs *ResponseSynthesizer) synthesizeGit(args map[string]string, result string) string {
	cmd := args["command"]
	if cmd == "" {
		cmd = "status"
	}

	switch {
	case strings.HasPrefix(cmd, "status"):
		return rs.synthesizeGitStatus(result)
	case strings.HasPrefix(cmd, "log"):
		return rs.synthesizeGitLog(result)
	case strings.HasPrefix(cmd, "diff"):
		return fmt.Sprintf("**Git diff:**\n\n```diff\n%s\n```", result)
	case strings.HasPrefix(cmd, "branch"):
		return fmt.Sprintf("**Git branches:**\n\n```\n%s\n```", result)
	default:
		return fmt.Sprintf("**git %s:**\n\n```\n%s\n```", cmd, result)
	}
}

func (rs *ResponseSynthesizer) synthesizeGitStatus(result string) string {
	lines := strings.Split(result, "\n")
	var modified, added, untracked []string

	for _, line := range lines {
		trimmed := strings.TrimSpace(line)
		switch {
		case strings.HasPrefix(trimmed, "modified:"):
			modified = append(modified, strings.TrimPrefix(trimmed, "modified:"))
		case strings.HasPrefix(trimmed, "new file:"):
			added = append(added, strings.TrimPrefix(trimmed, "new file:"))
		case strings.HasPrefix(trimmed, "??"):
			untracked = append(untracked, strings.TrimPrefix(trimmed, "?? "))
		}
	}

	// If no structured changes detected, show raw
	if len(modified)+len(added)+len(untracked) == 0 {
		if strings.Contains(result, "nothing to commit") {
			return "Working tree is clean — nothing to commit."
		}
		return fmt.Sprintf("**Git status:**\n\n```\n%s\n```", result)
	}

	var b strings.Builder
	b.WriteString("**Git status:**\n\n")

	if len(modified) > 0 {
		b.WriteString(fmt.Sprintf("Modified (%d):\n", len(modified)))
		for _, f := range modified {
			b.WriteString(fmt.Sprintf("  %s\n", strings.TrimSpace(f)))
		}
	}
	if len(added) > 0 {
		b.WriteString(fmt.Sprintf("New files (%d):\n", len(added)))
		for _, f := range added {
			b.WriteString(fmt.Sprintf("  %s\n", strings.TrimSpace(f)))
		}
	}
	if len(untracked) > 0 {
		b.WriteString(fmt.Sprintf("Untracked (%d):\n", len(untracked)))
		for _, f := range untracked {
			b.WriteString(fmt.Sprintf("  %s\n", strings.TrimSpace(f)))
		}
	}

	return strings.TrimSpace(b.String())
}

func (rs *ResponseSynthesizer) synthesizeGitLog(result string) string {
	lines := strings.Split(result, "\n")
	var commits []string
	for _, l := range lines {
		if strings.TrimSpace(l) != "" {
			commits = append(commits, strings.TrimSpace(l))
		}
	}

	if len(commits) == 0 {
		return "No commits found."
	}

	var b strings.Builder
	b.WriteString(fmt.Sprintf("**Recent commits** (%d):\n\n", len(commits)))
	for _, c := range commits {
		b.WriteString("  ")
		b.WriteString(c)
		b.WriteString("\n")
	}

	return strings.TrimSpace(b.String())
}

func (rs *ResponseSynthesizer) synthesizeWrite(args map[string]string, result string) string {
	path := args["path"]
	return fmt.Sprintf("File `%s` has been written successfully.", path)
}

func (rs *ResponseSynthesizer) synthesizeEdit(args map[string]string, result string) string {
	path := args["path"]
	return fmt.Sprintf("File `%s` has been edited successfully.", path)
}

func (rs *ResponseSynthesizer) synthesizeGeneric(tool, result string) string {
	if len(result) > 500 {
		return fmt.Sprintf("**%s result:**\n\n```\n%s\n```\n\n... and more.", tool, result[:500])
	}
	return fmt.Sprintf("**%s result:**\n\n```\n%s\n```", tool, result)
}

// --- Error and Empty Handling ---

func (rs *ResponseSynthesizer) synthesizeError(tool string, args map[string]string, err error) string {
	switch tool {
	case "read":
		path := args["path"]
		if path == "" {
			path = args["file"]
		}
		return fmt.Sprintf("Could not read `%s`: %v", path, err)
	case "grep":
		return fmt.Sprintf("Search failed: %v", err)
	default:
		return fmt.Sprintf("Tool `%s` failed: %v", tool, err)
	}
}

func (rs *ResponseSynthesizer) synthesizeEmpty(tool string, args map[string]string) string {
	switch tool {
	case "grep":
		pattern := args["pattern"]
		return fmt.Sprintf("No matches found for `%s`.", pattern)
	case "glob":
		pattern := args["pattern"]
		return fmt.Sprintf("No files match the pattern `%s`.", pattern)
	case "ls":
		dir := args["path"]
		if dir == "" {
			dir = "current directory"
		}
		return fmt.Sprintf("Directory `%s` is empty.", dir)
	default:
		return fmt.Sprintf("Tool `%s` returned no output.", tool)
	}
}

// --- Helpers ---

// fileGroup groups grep matches by file.
type fileGroup struct {
	file    string
	matches []string
}

// groupByFile groups grep output lines by file path.
func groupByFile(lines []string) []fileGroup {
	var groups []fileGroup
	seen := make(map[string]int)

	for _, line := range lines {
		// grep output format: path:line:content or path:content
		parts := strings.SplitN(line, ":", 3)
		if len(parts) < 2 {
			// Not a standard grep line, add to last group or create generic
			if len(groups) > 0 {
				idx := len(groups) - 1
				groups[idx].matches = append(groups[idx].matches, line)
			}
			continue
		}

		file := parts[0]
		rest := strings.Join(parts[1:], ":")

		if idx, ok := seen[file]; ok {
			groups[idx].matches = append(groups[idx].matches, strings.TrimSpace(rest))
		} else {
			seen[file] = len(groups)
			groups = append(groups, fileGroup{
				file:    file,
				matches: []string{strings.TrimSpace(rest)},
			})
		}
	}

	return groups
}

// CanSynthesize returns true if the synthesizer can handle this tool's output
// without needing the LLM.
func (rs *ResponseSynthesizer) CanSynthesize(tool string) bool {
	switch tool {
	case "grep", "read", "ls", "tree", "glob", "git", "write", "edit":
		return true
	}
	return false
}
