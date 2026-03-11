package cognitive

import (
	"fmt"
	"strings"
)

// ANSI color codes for diff output.
const (
	colorRed   = "\033[31m"
	colorGreen = "\033[32m"
	colorCyan  = "\033[36m"
	colorReset = "\033[0m"
)

// DiffPreview generates a colored terminal diff between old and new content.
// Returns a string with ANSI color codes:
//   - Red (\033[31m) for removed lines (-)
//   - Green (\033[32m) for added lines (+)
//   - Cyan (\033[36m) for @@ headers
//   - Reset (\033[0m) after each colored line
//
// Uses a simple line-based diff algorithm (longest common subsequence).
func DiffPreview(oldContent, newContent, filename string) string {
	oldLines := splitLines(oldContent)
	newLines := splitLines(newContent)

	// Compute LCS-based edit script
	edits := lcsEditScript(oldLines, newLines)

	// Generate unified diff hunks with 3 lines of context
	hunks := buildHunks(edits, oldLines, newLines, 3)
	if len(hunks) == 0 {
		return ""
	}

	var sb strings.Builder
	sb.WriteString(fmt.Sprintf("%s--- a/%s%s\n", colorCyan, filename, colorReset))
	sb.WriteString(fmt.Sprintf("%s+++ b/%s%s\n", colorCyan, filename, colorReset))

	totalLines := 0
	for _, h := range hunks {
		if totalLines > 30 {
			sb.WriteString(fmt.Sprintf("%s... (diff truncated)%s\n", colorCyan, colorReset))
			break
		}
		sb.WriteString(fmt.Sprintf("%s@@ -%d,%d +%d,%d @@%s\n",
			colorCyan, h.oldStart+1, h.oldCount, h.newStart+1, h.newCount, colorReset))
		totalLines++

		for _, line := range h.lines {
			if totalLines > 30 {
				sb.WriteString(fmt.Sprintf("%s... (diff truncated)%s\n", colorCyan, colorReset))
				break
			}
			switch line.op {
			case opEqual:
				sb.WriteString(" " + line.text + "\n")
			case opDelete:
				sb.WriteString(fmt.Sprintf("%s-%s%s\n", colorRed, line.text, colorReset))
			case opInsert:
				sb.WriteString(fmt.Sprintf("%s+%s%s\n", colorGreen, line.text, colorReset))
			}
			totalLines++
		}
	}

	return sb.String()
}

// FormatWritePreview formats a preview for creating a new file.
func FormatWritePreview(path, content string) string {
	lines := splitLines(content)

	var sb strings.Builder
	sb.WriteString(fmt.Sprintf("%s+++ b/%s (new file)%s\n", colorCyan, path, colorReset))

	for i, line := range lines {
		if i >= 30 {
			sb.WriteString(fmt.Sprintf("%s... (+%d more lines)%s\n", colorCyan, len(lines)-i, colorReset))
			break
		}
		sb.WriteString(fmt.Sprintf("%s+%s%s\n", colorGreen, line, colorReset))
	}

	return sb.String()
}

// FormatEditPreview formats a preview for editing an existing file.
// Shows only the changed region with 3 lines of context.
func FormatEditPreview(path, oldContent, newContent string) string {
	return DiffPreview(oldContent, newContent, path)
}

// editOp represents a line-level edit operation.
type editOp int

const (
	opEqual  editOp = iota
	opDelete        // line exists in old but not new
	opInsert        // line exists in new but not old
)

// editEntry represents a single line in the edit script.
type editEntry struct {
	op       editOp
	text     string
	oldIndex int // line index in old (for delete/equal)
	newIndex int // line index in new (for insert/equal)
}

// hunkLine is a line within a diff hunk.
type hunkLine struct {
	op   editOp
	text string
}

// hunk represents a unified diff hunk.
type hunk struct {
	oldStart int
	oldCount int
	newStart int
	newCount int
	lines    []hunkLine
}

// splitLines splits content into lines, handling the trailing newline edge case.
func splitLines(s string) []string {
	if s == "" {
		return nil
	}
	lines := strings.Split(s, "\n")
	// Remove trailing empty string from final newline
	if len(lines) > 0 && lines[len(lines)-1] == "" {
		lines = lines[:len(lines)-1]
	}
	return lines
}

// lcsEditScript computes a line-level edit script using the LCS algorithm.
// Returns a sequence of equal/delete/insert operations.
func lcsEditScript(oldLines, newLines []string) []editEntry {
	m := len(oldLines)
	n := len(newLines)

	// Build LCS table
	dp := make([][]int, m+1)
	for i := range dp {
		dp[i] = make([]int, n+1)
	}
	for i := 1; i <= m; i++ {
		for j := 1; j <= n; j++ {
			if oldLines[i-1] == newLines[j-1] {
				dp[i][j] = dp[i-1][j-1] + 1
			} else if dp[i-1][j] >= dp[i][j-1] {
				dp[i][j] = dp[i-1][j]
			} else {
				dp[i][j] = dp[i][j-1]
			}
		}
	}

	// Backtrack to produce edit script
	var edits []editEntry
	i, j := m, n
	for i > 0 || j > 0 {
		if i > 0 && j > 0 && oldLines[i-1] == newLines[j-1] {
			edits = append(edits, editEntry{op: opEqual, text: oldLines[i-1], oldIndex: i - 1, newIndex: j - 1})
			i--
			j--
		} else if j > 0 && (i == 0 || dp[i][j-1] >= dp[i-1][j]) {
			edits = append(edits, editEntry{op: opInsert, text: newLines[j-1], newIndex: j - 1})
			j--
		} else {
			edits = append(edits, editEntry{op: opDelete, text: oldLines[i-1], oldIndex: i - 1})
			i--
		}
	}

	// Reverse to get forward order
	for left, right := 0, len(edits)-1; left < right; left, right = left+1, right-1 {
		edits[left], edits[right] = edits[right], edits[left]
	}

	return edits
}

// buildHunks groups edit entries into unified diff hunks with the given context lines.
func buildHunks(edits []editEntry, oldLines, newLines []string, contextLines int) []hunk {
	if len(edits) == 0 {
		return nil
	}

	// Find ranges of changes (non-equal entries)
	type changeRange struct {
		start, end int // indices into edits slice
	}
	var changes []changeRange

	inChange := false
	var cur changeRange
	for i, e := range edits {
		if e.op != opEqual {
			if !inChange {
				cur.start = i
				inChange = true
			}
			cur.end = i
		} else if inChange {
			changes = append(changes, cur)
			inChange = false
		}
	}
	if inChange {
		changes = append(changes, cur)
	}

	if len(changes) == 0 {
		return nil
	}

	// Merge nearby changes (those within 2*contextLines of each other)
	var merged []changeRange
	merged = append(merged, changes[0])
	for i := 1; i < len(changes); i++ {
		prev := &merged[len(merged)-1]
		if changes[i].start-prev.end <= 2*contextLines {
			prev.end = changes[i].end
		} else {
			merged = append(merged, changes[i])
		}
	}

	// Build hunks from merged change ranges
	var hunks []hunk
	for _, cr := range merged {
		// Expand to include context
		hunkStart := cr.start - contextLines
		if hunkStart < 0 {
			hunkStart = 0
		}
		hunkEnd := cr.end + contextLines
		if hunkEnd >= len(edits) {
			hunkEnd = len(edits) - 1
		}

		var h hunk
		h.oldStart = -1
		h.newStart = -1

		for i := hunkStart; i <= hunkEnd; i++ {
			e := edits[i]
			h.lines = append(h.lines, hunkLine{op: e.op, text: e.text})

			switch e.op {
			case opEqual:
				if h.oldStart == -1 {
					h.oldStart = e.oldIndex
				}
				if h.newStart == -1 {
					h.newStart = e.newIndex
				}
				h.oldCount++
				h.newCount++
			case opDelete:
				if h.oldStart == -1 {
					h.oldStart = e.oldIndex
				}
				if h.newStart == -1 {
					// Use the next new index we'll encounter
					h.newStart = e.oldIndex
				}
				h.oldCount++
			case opInsert:
				if h.newStart == -1 {
					h.newStart = e.newIndex
				}
				if h.oldStart == -1 {
					h.oldStart = e.newIndex
				}
				h.newCount++
			}
		}

		// Fix -1 starts for edge cases (all deletes or all inserts)
		if h.oldStart == -1 {
			h.oldStart = 0
		}
		if h.newStart == -1 {
			h.newStart = 0
		}

		hunks = append(hunks, h)
	}

	return hunks
}
