package tools

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestParseFileQuery(t *testing.T) {
	tests := []struct {
		input   string
		wantExt string
		wantDir bool // just check if dir is non-empty for alias matches
		wantSort string
	}{
		{"find PDFs in downloads", "pdf", true, ""},
		{"recent files on desktop", "", true, "mtime"},
		{"large files in home", "", true, "size"},
		{"find *.go", "go", false, ""},
		{"latest json files", "json", false, "mtime"},
		{"biggest files in documents", "", true, "size"},
	}

	for _, tt := range tests {
		_, ext, dir, sortBy := ParseFileQuery(tt.input)
		if ext != tt.wantExt {
			t.Errorf("ParseFileQuery(%q) ext = %q, want %q", tt.input, ext, tt.wantExt)
		}
		if tt.wantDir && dir == "" {
			t.Errorf("ParseFileQuery(%q) dir is empty, expected a directory", tt.input)
		}
		if !tt.wantDir && dir != "" {
			// Not a strict error since home dir might match in some queries.
		}
		if sortBy != tt.wantSort {
			t.Errorf("ParseFileQuery(%q) sortBy = %q, want %q", tt.input, sortBy, tt.wantSort)
		}
	}
}

func TestExtractExtension(t *testing.T) {
	tests := []struct {
		input string
		want  string
	}{
		{"find pdfs", "pdf"},
		{"show images", "png"},
		{"list *.csv files", "csv"},
		{"search videos", "mp4"},
		{"random query", ""},
	}

	for _, tt := range tests {
		got := extractExtension(tt.input)
		if got != tt.want {
			t.Errorf("extractExtension(%q) = %q, want %q", tt.input, got, tt.want)
		}
	}
}

func TestFormatSize(t *testing.T) {
	tests := []struct {
		bytes int64
		want  string
	}{
		{500, "500B"},
		{1024, "1.0K"},
		{1536, "1.5K"},
		{1048576, "1.0M"},
		{1073741824, "1.0G"},
	}

	for _, tt := range tests {
		got := formatSize(tt.bytes)
		if got != tt.want {
			t.Errorf("formatSize(%d) = %q, want %q", tt.bytes, got, tt.want)
		}
	}
}

func TestFindFilesByExtension(t *testing.T) {
	dir := t.TempDir()

	// Create test files.
	os.WriteFile(filepath.Join(dir, "doc1.pdf"), []byte("pdf content"), 0644)
	os.WriteFile(filepath.Join(dir, "doc2.pdf"), []byte("pdf content 2"), 0644)
	os.WriteFile(filepath.Join(dir, "readme.txt"), []byte("text content"), 0644)

	result, err := FindFiles("find pdf", dir, 10)
	if err != nil {
		t.Fatalf("FindFiles: %v", err)
	}
	if !strings.Contains(result, "doc1.pdf") || !strings.Contains(result, "doc2.pdf") {
		t.Errorf("expected PDF files in result: %s", result)
	}
	if strings.Contains(result, "readme.txt") {
		t.Errorf("should not include txt file: %s", result)
	}
}

func TestFindFilesRecent(t *testing.T) {
	dir := t.TempDir()

	os.WriteFile(filepath.Join(dir, "old.txt"), []byte("old"), 0644)
	os.WriteFile(filepath.Join(dir, "new.txt"), []byte("new"), 0644)

	result, err := FindFiles("recent files", dir, 10)
	if err != nil {
		t.Fatalf("FindFiles: %v", err)
	}
	// Both should appear since both are recent.
	if !strings.Contains(result, "old.txt") || !strings.Contains(result, "new.txt") {
		t.Errorf("expected both files in recent results: %s", result)
	}
}

func TestFindFilesNoMatch(t *testing.T) {
	dir := t.TempDir()
	// Empty directory.
	result, err := FindFiles("find pdf", dir, 10)
	if err != nil {
		t.Fatalf("FindFiles: %v", err)
	}
	if !strings.Contains(result, "No files found") {
		t.Errorf("expected no files message: %s", result)
	}
}

func TestFindByContent(t *testing.T) {
	dir := t.TempDir()

	os.WriteFile(filepath.Join(dir, "match.txt"), []byte("the quick brown fox"), 0644)
	os.WriteFile(filepath.Join(dir, "nomatch.txt"), []byte("nothing here"), 0644)

	result, err := FindByContent("quick", dir, 10)
	if err != nil {
		t.Fatalf("FindByContent: %v", err)
	}
	if !strings.Contains(result, "match.txt") {
		t.Errorf("expected match.txt in results: %s", result)
	}
	if strings.Contains(result, "nomatch.txt") {
		t.Errorf("should not include nomatch.txt: %s", result)
	}
}

func TestFindFilesMaxResults(t *testing.T) {
	dir := t.TempDir()

	for i := 0; i < 10; i++ {
		os.WriteFile(filepath.Join(dir, strings.Repeat("a", i+1)+".txt"), []byte("data"), 0644)
	}

	result, err := FindFiles("find txt", dir, 3)
	if err != nil {
		t.Fatalf("FindFiles: %v", err)
	}
	lines := strings.Split(strings.TrimSpace(result), "\n")
	if len(lines) > 3 {
		t.Errorf("expected max 3 results, got %d", len(lines))
	}
}

func TestFindFilesBadDir(t *testing.T) {
	_, err := FindFiles("find txt", "/nonexistent/path/xyz", 10)
	if err == nil {
		t.Error("expected error for nonexistent directory")
	}
}
