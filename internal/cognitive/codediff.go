package cognitive

import (
	"fmt"
	"strings"
	"unicode"
)

// DiffExplainer analyzes git diffs and produces natural language explanations
// of what changed and why. Works by parsing unified diff format, identifying
// the types of changes (additions, removals, modifications), and using
// pattern recognition to infer intent.
type DiffExplainer struct{}

// DiffResult holds the analysis of a diff.
type DiffResult struct {
	Files    []FileDiff `json:"files"`
	Summary  string     `json:"summary"`  // one-paragraph summary
	Intent   string     `json:"intent"`   // "feature", "bugfix", "refactor", "docs", "test", "perf"
	Risk     string     `json:"risk"`     // "low", "medium", "high"
	Breaking bool       `json:"breaking"` // potential breaking changes
}

// FileDiff describes changes within a single file.
type FileDiff struct {
	Path        string   `json:"path"`
	Added       int      `json:"added"`
	Removed     int      `json:"removed"`
	Description string   `json:"description"` // what changed in this file
	Changes     []Change `json:"changes"`
}

// Change describes a single semantic change within a file.
type Change struct {
	Type        string `json:"type"`        // "add_func", "modify_func", "remove_func", "add_field", etc.
	Name        string `json:"name"`        // entity name
	Description string `json:"description"` // what this change does
}

// ExplainDiff parses a unified diff string and returns a structured analysis
// including per-file changes, overall intent, risk level, and a natural
// language summary.
func (de *DiffExplainer) ExplainDiff(diff string) *DiffResult {
	if strings.TrimSpace(diff) == "" {
		return &DiffResult{
			Summary: "No changes detected.",
			Intent:  "",
			Risk:    "low",
		}
	}

	files := parseDiff(diff)
	for i := range files {
		classifyFileChanges(&files[i], diff)
	}

	result := &DiffResult{
		Files: files,
	}
	result.Intent = detectIntent(files, diff)
	result.Risk, result.Breaking = assessRisk(files)
	result.Summary = generateSummary(result)

	return result
}

// parsedHunk holds the added and removed lines of a single diff hunk,
// along with any context (unchanged) lines.
type parsedHunk struct {
	addedLines   []string
	removedLines []string
	contextLines []string
}

// parseDiff extracts file paths and line counts from a unified diff.
func parseDiff(diff string) []FileDiff {
	lines := strings.Split(diff, "\n")
	var files []FileDiff
	var current *FileDiff

	for _, line := range lines {
		if strings.HasPrefix(line, "+++ ") && !strings.HasPrefix(line, "+++ /dev/null") {
			path := line[4:] // strip "+++ "
			// Strip the single-char prefix (e.g. "b/", "w/").
			if len(path) > 2 && path[1] == '/' {
				path = path[2:]
			}
			files = append(files, FileDiff{Path: path})
			current = &files[len(files)-1]
			continue
		}
		if current == nil {
			continue
		}
		if strings.HasPrefix(line, "@@") {
			continue
		}
		if strings.HasPrefix(line, "+++ ") || strings.HasPrefix(line, "--- ") {
			continue
		}
		if strings.HasPrefix(line, "diff --git") {
			continue
		}
		if strings.HasPrefix(line, "index ") {
			continue
		}

		if strings.HasPrefix(line, "+") {
			current.Added++
		} else if strings.HasPrefix(line, "-") {
			current.Removed++
		}
	}
	return files
}

// classifyFileChanges detects semantic changes (new functions, removed
// functions, modified functions, struct fields, imports, etc.) for a
// single file within the diff.
func classifyFileChanges(fd *FileDiff, diff string) {
	// Extract the portion of the diff belonging to this file.
	fileSection := extractFileSection(diff, fd.Path)
	if fileSection == "" {
		return
	}

	lines := strings.Split(fileSection, "\n")

	addedFuncs := map[string]bool{}
	removedFuncs := map[string]bool{}
	addedFields := []string{}
	removedFields := []string{}
	hasImportChange := false
	hasCommentChange := false
	hasTestAdd := false

	for _, line := range lines {
		trimmed := strings.TrimSpace(line)

		if strings.HasPrefix(line, "+") && !strings.HasPrefix(line, "+++") {
			content := line[1:]
			if name := diffExtractFuncName(content); name != "" {
				addedFuncs[name] = true
			}
			if field := extractStructField(content); field != "" {
				addedFields = append(addedFields, field)
			}
			if isImportLine(content) {
				hasImportChange = true
			}
			if isCommentLine(content) {
				hasCommentChange = true
			}
			if strings.Contains(content, "func Test") {
				hasTestAdd = true
			}
		} else if strings.HasPrefix(line, "-") && !strings.HasPrefix(line, "---") {
			content := line[1:]
			if name := diffExtractFuncName(content); name != "" {
				removedFuncs[name] = true
			}
			if field := extractStructField(content); field != "" {
				removedFields = append(removedFields, field)
			}
			if isImportLine(content) {
				hasImportChange = true
			}
			if isCommentLine(content) {
				hasCommentChange = true
			}
		}
		_ = trimmed
	}

	// Classify: functions that appear in both added and removed are modifications.
	for name := range addedFuncs {
		if removedFuncs[name] {
			fd.Changes = append(fd.Changes, Change{
				Type:        "modify_func",
				Name:        name,
				Description: fmt.Sprintf("Modified function %s", name),
			})
			delete(removedFuncs, name)
		} else {
			fd.Changes = append(fd.Changes, Change{
				Type:        "add_func",
				Name:        name,
				Description: fmt.Sprintf("Added new function %s", name),
			})
		}
	}
	for name := range removedFuncs {
		fd.Changes = append(fd.Changes, Change{
			Type:        "remove_func",
			Name:        name,
			Description: fmt.Sprintf("Removed function %s", name),
		})
	}
	for _, field := range addedFields {
		fd.Changes = append(fd.Changes, Change{
			Type:        "add_field",
			Name:        field,
			Description: fmt.Sprintf("Added field %s", field),
		})
	}
	for _, field := range removedFields {
		fd.Changes = append(fd.Changes, Change{
			Type:        "remove_field",
			Name:        field,
			Description: fmt.Sprintf("Removed field %s", field),
		})
	}
	if hasImportChange {
		fd.Changes = append(fd.Changes, Change{
			Type:        "import_change",
			Name:        "",
			Description: "Modified imports",
		})
	}
	if hasCommentChange {
		fd.Changes = append(fd.Changes, Change{
			Type:        "comment_change",
			Name:        "",
			Description: "Modified comments",
		})
	}
	if hasTestAdd {
		fd.Changes = append(fd.Changes, Change{
			Type:        "add_test",
			Name:        "",
			Description: "Added test functions",
		})
	}

	// Build file description.
	fd.Description = describeFile(fd)
}

// extractFileSection returns the diff lines belonging to the given file path.
// It handles both standard "b/" prefixes and worktree-style "w/" prefixes.
func extractFileSection(diff, path string) string {
	// Try common prefixes: "b/", "w/", or no prefix.
	var idx int = -1
	var marker string
	for _, prefix := range []string{"+++ b/", "+++ w/", "+++ a/", "+++ "} {
		candidate := prefix + path
		idx = strings.Index(diff, candidate)
		if idx != -1 {
			marker = candidate
			break
		}
	}
	if idx == -1 {
		return ""
	}

	// Start from the marker line.
	section := diff[idx:]

	// Find the next file header (diff --git) after the current one.
	rest := section[len(marker):]
	nextDiff := strings.Index(rest, "\ndiff --git")
	if nextDiff != -1 {
		section = section[:len(marker)+nextDiff]
	}

	return section
}

// diffExtractFuncName returns the function name from a Go func declaration
// line, or "" if the line is not a func declaration. Uses the shared
// extractFuncName helper after verifying the line is a func declaration.
func diffExtractFuncName(line string) string {
	trimmed := strings.TrimSpace(line)
	if !strings.HasPrefix(trimmed, "func ") {
		return ""
	}
	return extractFuncName(trimmed)
}

// extractStructField detects lines that look like Go struct field declarations.
// Returns the field name or "".
func extractStructField(line string) string {
	trimmed := strings.TrimSpace(line)
	// A struct field line has an identifier followed by a type,
	// and does NOT start with keywords or punctuation.
	if trimmed == "" || trimmed == "{" || trimmed == "}" {
		return ""
	}

	// Filter out Go statements and keywords.
	keywords := []string{
		"func ", "if ", "for ", "return ", "//", "/*", "import ", "package ",
		"var ", "const ", "type ", "switch ", "case ", "default:", "select ",
		"go ", "defer ", "range ", "break", "continue", "fallthrough",
		"else ", "else{", "chan ", "map[", "fmt.", "log.", "os.", "err ",
	}
	for _, prefix := range keywords {
		if strings.HasPrefix(trimmed, prefix) {
			return ""
		}
	}

	// Must not end with '{', ')', or be an assignment.
	if strings.HasSuffix(trimmed, "{") || strings.HasSuffix(trimmed, ")") {
		return ""
	}
	if strings.Contains(trimmed, ":=") || strings.Contains(trimmed, "= ") {
		return ""
	}
	// Must not contain function calls.
	if strings.Contains(trimmed, "(") && strings.Contains(trimmed, ")") {
		return ""
	}

	// Must start with an uppercase or lowercase letter (identifier).
	if !unicode.IsLetter(rune(trimmed[0])) {
		return ""
	}

	// The line should have at least two space-separated tokens (name and type).
	parts := strings.Fields(trimmed)
	if len(parts) < 2 {
		return ""
	}

	// The first part should look like a Go identifier.
	name := parts[0]
	for _, r := range name {
		if !unicode.IsLetter(r) && !unicode.IsDigit(r) && r != '_' {
			return ""
		}
	}

	// The second token should look like a Go type: starts with uppercase,
	// is a builtin, or is a pointer/slice/map.
	typ := parts[1]
	goBuiltins := map[string]bool{
		"string": true, "int": true, "int8": true, "int16": true, "int32": true,
		"int64": true, "uint": true, "uint8": true, "uint16": true, "uint32": true,
		"uint64": true, "float32": true, "float64": true, "bool": true, "byte": true,
		"rune": true, "error": true, "any": true, "interface{}": true,
	}
	isType := goBuiltins[typ] ||
		unicode.IsUpper(rune(typ[0])) ||
		strings.HasPrefix(typ, "*") ||
		strings.HasPrefix(typ, "[]") ||
		strings.HasPrefix(typ, "map[")
	if !isType {
		return ""
	}

	return name
}

// extractCommentText collects all comment lines and commit message lines from
// the diff, lowercased, for intent keyword detection. This prevents false
// positives from identifiers like "prefix" matching "fix".
func extractCommentText(diff string) string {
	var sb strings.Builder
	for _, line := range strings.Split(diff, "\n") {
		trimmed := strings.TrimSpace(line)
		// Diff added/removed comment lines.
		if (strings.HasPrefix(line, "+") || strings.HasPrefix(line, "-")) &&
			!strings.HasPrefix(line, "+++") && !strings.HasPrefix(line, "---") {
			content := strings.TrimSpace(line[1:])
			if strings.HasPrefix(content, "//") || strings.HasPrefix(content, "/*") || strings.HasPrefix(content, "* ") {
				sb.WriteString(strings.ToLower(content))
				sb.WriteString(" ")
			}
		}
		// Git commit message lines (from git show output).
		if !strings.HasPrefix(trimmed, "+") && !strings.HasPrefix(trimmed, "-") &&
			!strings.HasPrefix(trimmed, "@@") && !strings.HasPrefix(trimmed, "diff") &&
			!strings.HasPrefix(trimmed, "index") && trimmed != "" {
			sb.WriteString(strings.ToLower(trimmed))
			sb.WriteString(" ")
		}
	}
	return sb.String()
}

// isImportLine returns true if the line appears to be inside a Go import block.
func isImportLine(line string) bool {
	trimmed := strings.TrimSpace(line)
	if strings.HasPrefix(trimmed, "import ") || trimmed == "import (" || trimmed == ")" {
		return true
	}
	// Quoted import path inside import block
	if strings.HasPrefix(trimmed, "\"") || strings.HasPrefix(trimmed, ". \"") {
		return true
	}
	return false
}

// isCommentLine returns true if the line is a Go comment.
func isCommentLine(line string) bool {
	trimmed := strings.TrimSpace(line)
	return strings.HasPrefix(trimmed, "//") || strings.HasPrefix(trimmed, "/*") || strings.HasPrefix(trimmed, "* ") || strings.HasPrefix(trimmed, "*/")
}

// describeFile produces a short human-readable description of what changed
// in a file.
func describeFile(fd *FileDiff) string {
	if len(fd.Changes) == 0 {
		if fd.Added > 0 && fd.Removed == 0 {
			return fmt.Sprintf("Added %d lines", fd.Added)
		}
		if fd.Removed > 0 && fd.Added == 0 {
			return fmt.Sprintf("Removed %d lines", fd.Removed)
		}
		if fd.Added > 0 && fd.Removed > 0 {
			return fmt.Sprintf("Modified (%d added, %d removed)", fd.Added, fd.Removed)
		}
		return "No significant changes"
	}

	var parts []string
	counts := map[string]int{}
	for _, c := range fd.Changes {
		counts[c.Type]++
	}
	if n := counts["add_func"]; n > 0 {
		parts = append(parts, fmt.Sprintf("added %d function(s)", n))
	}
	if n := counts["remove_func"]; n > 0 {
		parts = append(parts, fmt.Sprintf("removed %d function(s)", n))
	}
	if n := counts["modify_func"]; n > 0 {
		parts = append(parts, fmt.Sprintf("modified %d function(s)", n))
	}
	if n := counts["add_field"]; n > 0 {
		parts = append(parts, fmt.Sprintf("added %d field(s)", n))
	}
	if n := counts["remove_field"]; n > 0 {
		parts = append(parts, fmt.Sprintf("removed %d field(s)", n))
	}
	if counts["import_change"] > 0 {
		parts = append(parts, "updated imports")
	}
	if counts["add_test"] > 0 {
		parts = append(parts, "added tests")
	}
	if len(parts) == 0 {
		return fmt.Sprintf("Modified (%d added, %d removed)", fd.Added, fd.Removed)
	}
	return strings.Join(parts, ", ")
}

// detectIntent infers the overall intent of the diff from file paths and
// change patterns.
func detectIntent(files []FileDiff, diff string) string {
	hasTest := false
	hasDocs := false
	hasBugfix := false
	hasPerf := false
	hasNewFunc := false
	hasModifiedFunc := false

	lowerDiff := strings.ToLower(diff)

	for _, f := range files {
		if strings.HasSuffix(f.Path, "_test.go") || strings.Contains(f.Path, "test_") {
			hasTest = true
		}
		if strings.HasSuffix(f.Path, ".md") {
			hasDocs = true
		}
		for _, c := range f.Changes {
			if c.Type == "add_func" {
				hasNewFunc = true
			}
			if c.Type == "modify_func" {
				hasModifiedFunc = true
			}
			if c.Type == "add_test" {
				hasTest = true
			}
		}
	}

	// Check for bugfix indicators in comments and commit messages only,
	// not in arbitrary code (to avoid false positives from identifiers
	// like "prefix" containing "fix").
	commentText := extractCommentText(diff)
	for _, kw := range []string{"fix:", "fix ", " bug ", " bug.", "bugfix", "hotfix", " patch ", "error handling", "nil check", "nil pointer"} {
		if strings.Contains(commentText, kw) {
			hasBugfix = true
			break
		}
	}

	// Check for performance indicators.
	for _, kw := range []string{"benchmark", "optimize", "cache", "pool", "perf", "latency", "throughput"} {
		if strings.Contains(lowerDiff, kw) {
			hasPerf = true
			break
		}
	}

	// Priority-based intent classification.
	// Only test files changed → test.
	allTest := true
	allDocs := true
	for _, f := range files {
		if !strings.HasSuffix(f.Path, "_test.go") && !strings.Contains(f.Path, "test_") {
			allTest = false
		}
		if !strings.HasSuffix(f.Path, ".md") {
			allDocs = false
		}
	}

	if len(files) > 0 && allTest {
		return "test"
	}
	if len(files) > 0 && allDocs {
		return "docs"
	}
	if hasPerf && !hasNewFunc {
		return "perf"
	}
	if hasBugfix {
		return "bugfix"
	}
	if hasNewFunc && !hasModifiedFunc {
		return "feature"
	}
	if hasModifiedFunc && !hasNewFunc {
		return "refactor"
	}
	if hasTest && hasNewFunc {
		return "feature"
	}
	if hasNewFunc {
		return "feature"
	}
	if hasDocs {
		return "docs"
	}
	if hasModifiedFunc {
		return "refactor"
	}
	return "feature"
}

// assessRisk determines the risk level and whether there are breaking changes.
func assessRisk(files []FileDiff) (string, bool) {
	risk := "low"
	breaking := false

	for _, f := range files {
		// Database/SQL changes → high risk.
		lower := strings.ToLower(f.Path)
		if strings.Contains(lower, "sql") || strings.Contains(lower, "migration") || strings.Contains(lower, "database") || strings.Contains(lower, "schema") {
			risk = "high"
		}

		for _, c := range f.Changes {
			switch c.Type {
			case "remove_func":
				if isExported(c.Name) {
					risk = "high"
					breaking = true
				} else if risk != "high" {
					risk = "medium"
				}
			case "modify_func":
				if isExported(c.Name) && risk != "high" {
					risk = "medium"
				}
			case "remove_field":
				if isExported(c.Name) {
					risk = "high"
					breaking = true
				}
			case "add_func":
				// New functions: generally low risk unless it's init/main.
				if c.Name == "init" || c.Name == "main" {
					if risk != "high" {
						risk = "medium"
					}
				}
			}
		}
	}

	return risk, breaking
}

// isExported returns true if the name starts with an uppercase letter.
func isExported(name string) bool {
	if name == "" {
		return false
	}
	return unicode.IsUpper(rune(name[0]))
}

// generateSummary produces a natural language summary paragraph for the diff result.
func generateSummary(result *DiffResult) string {
	if len(result.Files) == 0 {
		return "No changes detected."
	}

	var sb strings.Builder

	// Count totals.
	totalAdded := 0
	totalRemoved := 0
	totalNewFuncs := 0
	totalRemovedFuncs := 0
	totalModifiedFuncs := 0
	totalNewTests := 0
	totalNewFields := 0

	for _, f := range result.Files {
		totalAdded += f.Added
		totalRemoved += f.Removed
		for _, c := range f.Changes {
			switch c.Type {
			case "add_func":
				totalNewFuncs++
			case "remove_func":
				totalRemovedFuncs++
			case "modify_func":
				totalModifiedFuncs++
			case "add_test":
				totalNewTests++
			case "add_field":
				totalNewFields++
			}
		}
	}

	// Build the opening sentence based on intent.
	intentLabel := result.Intent
	switch result.Intent {
	case "feature":
		intentLabel = "feature"
	case "bugfix":
		intentLabel = "bug fix"
	case "refactor":
		intentLabel = "refactoring"
	case "test":
		intentLabel = "test addition"
	case "docs":
		intentLabel = "documentation update"
	case "perf":
		intentLabel = "performance improvement"
	}

	sb.WriteString(fmt.Sprintf("This change is a %s affecting %d file(s)", intentLabel, len(result.Files)))

	// Details.
	details := []string{}
	if totalNewFuncs > 0 {
		details = append(details, fmt.Sprintf("%d new function(s)", totalNewFuncs))
	}
	if totalModifiedFuncs > 0 {
		details = append(details, fmt.Sprintf("%d modified function(s)", totalModifiedFuncs))
	}
	if totalRemovedFuncs > 0 {
		details = append(details, fmt.Sprintf("%d removed function(s)", totalRemovedFuncs))
	}
	if totalNewFields > 0 {
		details = append(details, fmt.Sprintf("%d new field(s)", totalNewFields))
	}
	if totalNewTests > 0 {
		details = append(details, "new tests")
	}

	if len(details) > 0 {
		sb.WriteString(": ")
		sb.WriteString(strings.Join(details, ", "))
	}
	sb.WriteString(fmt.Sprintf(". %d line(s) added, %d line(s) removed.", totalAdded, totalRemoved))

	// Risk sentence.
	switch result.Risk {
	case "low":
		sb.WriteString(" Risk: low (additive change, no existing behavior modified).")
	case "medium":
		sb.WriteString(" Risk: medium (existing exported functions modified).")
	case "high":
		if result.Breaking {
			sb.WriteString(" Risk: high (potential breaking changes — exported symbols removed).")
		} else {
			sb.WriteString(" Risk: high (sensitive area modified).")
		}
	}

	return sb.String()
}

// FormatDiffResult formats a DiffResult as a human-readable string for
// terminal display.
func FormatDiffResult(r *DiffResult) string {
	var sb strings.Builder

	sb.WriteString(r.Summary)
	sb.WriteString("\n")

	if len(r.Files) > 0 {
		sb.WriteString("\nFiles:\n")
		for _, f := range r.Files {
			sb.WriteString(fmt.Sprintf("  %s (+%d/-%d) — %s\n", f.Path, f.Added, f.Removed, f.Description))
			for _, c := range f.Changes {
				sb.WriteString(fmt.Sprintf("    [%s] %s\n", c.Type, c.Description))
			}
		}
	}

	if r.Intent != "" {
		sb.WriteString(fmt.Sprintf("\nIntent:   %s\n", r.Intent))
	}
	sb.WriteString(fmt.Sprintf("Risk:     %s\n", r.Risk))
	if r.Breaking {
		sb.WriteString("Breaking: yes\n")
	}

	return sb.String()
}
