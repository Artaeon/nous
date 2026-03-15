package cognitive

import (
	"fmt"
	"strings"
	"testing"
)

func TestDiffPreviewIdenticalContent(t *testing.T) {
	content := "line one\nline two\nline three\n"
	result := DiffPreview(content, content, "test.go")
	if result != "" {
		t.Errorf("identical content should produce empty diff, got:\n%s", result)
	}
}

func TestDiffPreviewSingleLineChange(t *testing.T) {
	old := "line one\nline two\nline three\n"
	new := "line one\nline TWO\nline three\n"
	result := DiffPreview(old, new, "test.go")

	if !strings.Contains(result, colorRed+"-line two"+colorReset) {
		t.Error("diff should show removed line in red")
	}
	if !strings.Contains(result, colorGreen+"+line TWO"+colorReset) {
		t.Error("diff should show added line in green")
	}
	if !strings.Contains(result, "--- a/test.go") {
		t.Error("diff should include --- header with filename")
	}
	if !strings.Contains(result, "+++ b/test.go") {
		t.Error("diff should include +++ header with filename")
	}
}

func TestDiffPreviewMultiLineAdditions(t *testing.T) {
	old := "line one\nline three\n"
	new := "line one\nnew line A\nnew line B\nline three\n"
	result := DiffPreview(old, new, "file.txt")

	if !strings.Contains(result, colorGreen+"+new line A"+colorReset) {
		t.Error("diff should show first added line")
	}
	if !strings.Contains(result, colorGreen+"+new line B"+colorReset) {
		t.Error("diff should show second added line")
	}
	// The original lines should not appear as removed
	if strings.Contains(result, colorRed+"-line one") {
		t.Error("unchanged lines should not appear as removed")
	}
}

func TestDiffPreviewMultiLineRemovals(t *testing.T) {
	old := "line one\nremove A\nremove B\nline three\n"
	new := "line one\nline three\n"
	result := DiffPreview(old, new, "file.txt")

	if !strings.Contains(result, colorRed+"-remove A"+colorReset) {
		t.Error("diff should show first removed line")
	}
	if !strings.Contains(result, colorRed+"-remove B"+colorReset) {
		t.Error("diff should show second removed line")
	}
	// The remaining lines should not appear as added
	if strings.Contains(result, colorGreen+"+line three") {
		t.Error("unchanged lines should not appear as added")
	}
}

func TestDiffPreviewMixedChanges(t *testing.T) {
	old := "alpha\nbeta\ngamma\ndelta\n"
	new := "alpha\nBETA\ndelta\nepsilon\n"
	result := DiffPreview(old, new, "mixed.go")

	// beta removed, BETA added
	if !strings.Contains(result, colorRed+"-beta"+colorReset) {
		t.Error("diff should show removed 'beta'")
	}
	if !strings.Contains(result, colorGreen+"+BETA"+colorReset) {
		t.Error("diff should show added 'BETA'")
	}
	// gamma removed
	if !strings.Contains(result, colorRed+"-gamma"+colorReset) {
		t.Error("diff should show removed 'gamma'")
	}
	// epsilon added
	if !strings.Contains(result, colorGreen+"+epsilon"+colorReset) {
		t.Error("diff should show added 'epsilon'")
	}
}

func TestFormatWritePreviewNewFile(t *testing.T) {
	content := "line one\nline two\nline three\n"
	result := FormatWritePreview("newfile.go", content)

	if !strings.Contains(result, "(new file)") {
		t.Error("write preview should indicate new file")
	}
	if !strings.Contains(result, colorGreen+"+line one"+colorReset) {
		t.Error("all lines should be shown in green for new file")
	}
	if !strings.Contains(result, colorGreen+"+line two"+colorReset) {
		t.Error("all lines should be shown in green for new file")
	}
	if !strings.Contains(result, colorGreen+"+line three"+colorReset) {
		t.Error("all lines should be shown in green for new file")
	}
}

func TestFormatWritePreviewTruncation(t *testing.T) {
	var lines []string
	for i := 0; i < 50; i++ {
		lines = append(lines, fmt.Sprintf("line %d", i+1))
	}
	content := strings.Join(lines, "\n") + "\n"
	result := FormatWritePreview("big.go", content)

	if !strings.Contains(result, "more lines") {
		t.Error("long file preview should be truncated with 'more lines' indicator")
	}
	// Should show exactly 30 content lines before truncation
	if !strings.Contains(result, "+line 30") {
		t.Error("should show up to line 30")
	}
	if strings.Contains(result, "+line 31") {
		t.Error("should not show line 31 (truncated)")
	}
}

func TestFormatEditPreviewDelegatesToDiffPreview(t *testing.T) {
	old := "hello\n"
	new := "world\n"
	path := "edit.go"

	editResult := FormatEditPreview(path, old, new)
	diffResult := DiffPreview(old, new, path)

	if editResult != diffResult {
		t.Errorf("FormatEditPreview should delegate to DiffPreview\nedit: %q\ndiff: %q", editResult, diffResult)
	}
}

// --- LCS algorithm tests ---

func TestLCSEditScriptEmptyInputs(t *testing.T) {
	edits := lcsEditScript(nil, nil)
	if len(edits) != 0 {
		t.Errorf("empty inputs should produce empty edit script, got %d entries", len(edits))
	}
}

func TestLCSEditScriptAllInserts(t *testing.T) {
	edits := lcsEditScript(nil, []string{"a", "b", "c"})
	inserts := 0
	for _, e := range edits {
		if e.op == opInsert {
			inserts++
		}
	}
	if inserts != 3 {
		t.Errorf("expected 3 inserts from empty→3 lines, got %d", inserts)
	}
}

func TestLCSEditScriptAllDeletes(t *testing.T) {
	edits := lcsEditScript([]string{"a", "b", "c"}, nil)
	deletes := 0
	for _, e := range edits {
		if e.op == opDelete {
			deletes++
		}
	}
	if deletes != 3 {
		t.Errorf("expected 3 deletes from 3 lines→empty, got %d", deletes)
	}
}

func TestLCSEditScriptAllEqual(t *testing.T) {
	lines := []string{"alpha", "beta", "gamma"}
	edits := lcsEditScript(lines, lines)
	for i, e := range edits {
		if e.op != opEqual {
			t.Errorf("edit[%d] should be opEqual for identical input, got %v", i, e.op)
		}
	}
	if len(edits) != 3 {
		t.Errorf("expected 3 edits for 3 identical lines, got %d", len(edits))
	}
}

func TestLCSEditScriptSingleSubstitution(t *testing.T) {
	old := []string{"a", "b", "c"}
	new := []string{"a", "X", "c"}
	edits := lcsEditScript(old, new)

	var ops []editOp
	for _, e := range edits {
		ops = append(ops, e.op)
	}
	// Should have: equal(a), delete(b), insert(X), equal(c)
	hasDelete := false
	hasInsert := false
	equalCount := 0
	for _, e := range edits {
		switch e.op {
		case opEqual:
			equalCount++
		case opDelete:
			hasDelete = true
		case opInsert:
			hasInsert = true
		}
	}
	if !hasDelete || !hasInsert || equalCount != 2 {
		t.Errorf("expected 2 equals + 1 delete + 1 insert, got edits: %v", ops)
	}
}

func TestSplitLinesEdgeCases(t *testing.T) {
	tests := []struct {
		name  string
		input string
		want  int
	}{
		{"empty", "", 0},
		{"single newline", "\n", 1}, // splits to ["",""], removes trailing "" → [""]
		{"single line no newline", "hello", 1},
		{"single line with newline", "hello\n", 1},
		{"two lines", "a\nb\n", 2},
		{"two lines no trailing", "a\nb", 2},
		{"blank lines", "\n\n\n", 3}, // splits to ["", "", "", ""], removes trailing "" → 3
		{"mixed content", "a\n\nb\n", 3},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := len(splitLines(tt.input))
			if got != tt.want {
				t.Errorf("splitLines(%q) = %d lines, want %d", tt.input, got, tt.want)
			}
		})
	}
}

func TestBuildHunksEmpty(t *testing.T) {
	hunks := buildHunks(nil, nil, nil, 3)
	if len(hunks) != 0 {
		t.Errorf("empty edits should produce no hunks, got %d", len(hunks))
	}
}

func TestBuildHunksAllEqual(t *testing.T) {
	lines := []string{"a", "b", "c"}
	edits := lcsEditScript(lines, lines) // all equal
	hunks := buildHunks(edits, lines, lines, 3)
	if len(hunks) != 0 {
		t.Errorf("all-equal edits should produce no hunks, got %d", len(hunks))
	}
}

func TestBuildHunksIncludesContext(t *testing.T) {
	old := []string{"a", "b", "c", "d", "e", "f", "g", "h"}
	new := []string{"a", "b", "c", "X", "e", "f", "g", "h"}
	edits := lcsEditScript(old, new)
	hunks := buildHunks(edits, old, new, 3)

	if len(hunks) == 0 {
		t.Fatal("should produce at least one hunk")
	}

	// Count context lines in hunk
	contextCount := 0
	for _, line := range hunks[0].lines {
		if line.op == opEqual {
			contextCount++
		}
	}
	if contextCount == 0 {
		t.Error("hunk should include context lines around the change")
	}
}

func TestDiffPreviewEmptyToContent(t *testing.T) {
	result := DiffPreview("", "hello\nworld\n", "new.go")
	if !strings.Contains(result, "+hello") {
		t.Error("diff from empty should show all lines as additions")
	}
}

func TestDiffPreviewContentToEmpty(t *testing.T) {
	result := DiffPreview("hello\nworld\n", "", "deleted.go")
	if !strings.Contains(result, "-hello") {
		t.Error("diff to empty should show all lines as deletions")
	}
}

func TestDiffPreviewLargeTruncation(t *testing.T) {
	// Create a very large diff that exceeds 30-line limit
	var oldLines, newLines []string
	for i := 0; i < 50; i++ {
		oldLines = append(oldLines, fmt.Sprintf("old line %d", i))
		newLines = append(newLines, fmt.Sprintf("new line %d", i))
	}

	result := DiffPreview(
		strings.Join(oldLines, "\n")+"\n",
		strings.Join(newLines, "\n")+"\n",
		"huge.go",
	)

	if !strings.Contains(result, "truncated") {
		t.Error("very large diffs should be truncated")
	}
}

func TestVisibleLenStripsANSI(t *testing.T) {
	tests := []struct {
		name  string
		input string
		want  int
	}{
		{"plain text", "hello", 5},
		{"empty string", "", 0},
		{"single color", colorRed + "hi" + colorReset, 2},
		{"multiple colors", colorGreen + "a" + colorReset + colorCyan + "b" + colorReset, 2},
		{"bold and color", ColorBold + ColorCyan + "text" + ColorReset, 4},
		{"no visible chars", colorRed + colorReset, 0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := visibleLen(tt.input)
			if got != tt.want {
				t.Errorf("visibleLen(%q) = %d, want %d", tt.input, got, tt.want)
			}
		})
	}
}
