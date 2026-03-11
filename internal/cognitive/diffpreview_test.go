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
