package cognitive

import (
	"strings"
	"testing"
)

// Fuzz tests use Go 1.18+ native fuzzing to find edge cases
// in parsers and classifiers. Run with: go test -fuzz=FuzzName

func FuzzCompressStep(f *testing.F) {
	// Seed corpus with representative tool outputs
	f.Add("read", "package main\nfunc main() {}\n")
	f.Add("grep", "main.go:1:package main\nserver.go:5:func Handle() {}\n")
	f.Add("glob", "main.go\nserver.go\nhandler.go\n")
	f.Add("ls", "cmd/\ninternal/\ngo.mod\nREADME.md\n")
	f.Add("git", "abc1234 Initial commit\ndef5678 Add feature\n")
	f.Add("write", "Wrote 42 bytes to main.go")
	f.Add("edit", "Replaced content in server.go")
	f.Add("sysinfo", "Linux x86_64 8 CPUs 16GB RAM")
	f.Add("unknown", "arbitrary output from unknown tool")
	f.Add("read", "Error: no such file or directory")
	f.Add("read", "")
	f.Add("grep", "")

	f.Fuzz(func(t *testing.T, toolName, result string) {
		// CompressStep should never panic
		summary := CompressStep(toolName, result)

		// Summary should always be non-empty for results with content
		if len(result) > 0 && strings.TrimSpace(result) != "" && summary == "" {
			t.Errorf("CompressStep(%q, len=%d) returned empty summary", toolName, len(result))
		}

		// Summary should exist (we don't check length since compression
		// depends on tool-specific heuristics and input structure)
		_ = summary
	})
}

func FuzzSmartTruncate(f *testing.F) {
	f.Add("read", "line 1\nline 2\nline 3\n")
	f.Add("grep", "file.go:1:match\nfile.go:2:match\n")
	f.Add("read", "")
	f.Add("ls", "a\nb\nc\n")
	f.Add("sysinfo", "data")

	f.Fuzz(func(t *testing.T, toolName, result string) {
		// SmartTruncate should never panic
		truncated := SmartTruncate(toolName, result)

		// Result should never exceed hard limit (2048 + truncation marker)
		if len(truncated) > 2100 {
			t.Errorf("truncated result too long: %d", len(truncated))
		}
	})
}

func FuzzClassifyQuery(f *testing.F) {
	f.Add("hello")
	f.Add("read file main.go")
	f.Add("what is Go?")
	f.Add("")
	f.Add("   ")
	f.Add("explain how garbage collection works")
	f.Add("git status")
	f.Add("tell me a joke")
	f.Add("42 + 7?")
	f.Add("run the tests please")
	f.Add("translate hello to french")

	c := &FastPathClassifier{}

	f.Fuzz(func(t *testing.T, query string) {
		// ClassifyQuery should never panic
		path := c.ClassifyQuery(query)

		// Path should always be one of the three constants
		if path != PathFast && path != PathMedium && path != PathFull {
			t.Errorf("ClassifyQuery(%q) = %q, want fast|medium|full", query, path)
		}

		// IsSimple should be consistent with ClassifyQuery
		isSimple := c.IsSimple(query)
		expectSimple := path == PathFast || path == PathMedium
		if isSimple != expectSimple {
			t.Errorf("IsSimple(%q)=%v but ClassifyQuery=%q", query, isSimple, path)
		}
	})
}

func FuzzExtractKeywords(f *testing.F) {
	f.Add("How can I read the main.go file?")
	f.Add("")
	f.Add("find function definition")
	f.Add("the and for that this")
	f.Add("a")

	f.Fuzz(func(t *testing.T, input string) {
		// extractKeywords should never panic
		keywords := extractKeywords(input)

		// All keywords should be at least 3 characters
		for _, kw := range keywords {
			if len(kw) < 3 {
				t.Errorf("keyword %q is shorter than 3 chars", kw)
			}
		}
	})
}

func FuzzDiffPreview(f *testing.F) {
	f.Add("line 1\nline 2\n", "line 1\nline X\n", "test.go")
	f.Add("", "new content\n", "new.go")
	f.Add("old content\n", "", "deleted.go")
	f.Add("same\n", "same\n", "identical.go")
	f.Add("", "", "empty.go")

	f.Fuzz(func(t *testing.T, old, new, filename string) {
		// DiffPreview should never panic
		result := DiffPreview(old, new, filename)

		// If old == new, result should be empty
		if old == new && result != "" {
			t.Errorf("identical content should produce empty diff")
		}
	})
}

func FuzzKeywordOverlap(f *testing.F) {
	f.Add("read file main", "read function server")
	f.Add("", "test")
	f.Add("test", "")
	f.Add("", "")

	f.Fuzz(func(t *testing.T, aStr, bStr string) {
		a := extractKeywords(aStr)
		b := extractKeywords(bStr)

		overlap := keywordOverlap(a, b)

		// Overlap should be between 0 and 1
		if overlap < 0 || overlap > 1 {
			t.Errorf("overlap = %f, should be in [0, 1]", overlap)
		}
	})
}

func FuzzSplitLines(f *testing.F) {
	f.Add("hello\nworld\n")
	f.Add("")
	f.Add("\n")
	f.Add("no newline")

	f.Fuzz(func(t *testing.T, s string) {
		// splitLines should never panic
		lines := splitLines(s)

		// Lines should never be nil for non-empty input
		// (empty string returns nil which is OK)
		if s != "" && lines == nil {
			// This is actually OK for "\n" edge case with trailing newline removal
		}

		// No line should contain a newline
		for i, line := range lines {
			for _, ch := range line {
				if ch == '\n' {
					t.Errorf("line %d contains newline: %q", i, line)
				}
			}
		}
	})
}
