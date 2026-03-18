package tools

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestDetectArchiveFormat(t *testing.T) {
	tests := []struct {
		path string
		want string
	}{
		{"archive.tar.gz", "tar.gz"},
		{"archive.tgz", "tar.gz"},
		{"archive.tar.bz2", "tar.bz2"},
		{"archive.tbz2", "tar.bz2"},
		{"archive.tar", "tar"},
		{"archive.zip", "zip"},
		{"archive.txt", ""},
		{"ARCHIVE.TAR.GZ", "tar.gz"},
	}
	for _, tt := range tests {
		got := DetectArchiveFormat(tt.path)
		if got != tt.want {
			t.Errorf("DetectArchiveFormat(%q) = %q, want %q", tt.path, got, tt.want)
		}
	}
}

func TestFormatArchiveSize(t *testing.T) {
	tests := []struct {
		bytes int64
		want  string
	}{
		{500, "500B"},
		{1024, "1.0K"},
		{1048576, "1.0M"},
		{1073741824, "1.0G"},
	}
	for _, tt := range tests {
		got := FormatArchiveSize(tt.bytes)
		if got != tt.want {
			t.Errorf("FormatArchiveSize(%d) = %q, want %q", tt.bytes, got, tt.want)
		}
	}
}

func createTestFiles(t *testing.T, dir string) {
	t.Helper()
	os.MkdirAll(filepath.Join(dir, "subdir"), 0755)
	os.WriteFile(filepath.Join(dir, "file1.txt"), []byte("hello world"), 0644)
	os.WriteFile(filepath.Join(dir, "file2.txt"), []byte("second file content"), 0644)
	os.WriteFile(filepath.Join(dir, "subdir", "nested.txt"), []byte("nested content"), 0644)
}

func verifyExtractedFiles(t *testing.T, extractDir, srcDirName string) {
	t.Helper()
	base := filepath.Join(extractDir, srcDirName)

	content1, err := os.ReadFile(filepath.Join(base, "file1.txt"))
	if err != nil {
		t.Fatalf("missing file1.txt: %v", err)
	}
	if string(content1) != "hello world" {
		t.Errorf("file1.txt content = %q, want %q", string(content1), "hello world")
	}

	content2, err := os.ReadFile(filepath.Join(base, "file2.txt"))
	if err != nil {
		t.Fatalf("missing file2.txt: %v", err)
	}
	if string(content2) != "second file content" {
		t.Errorf("file2.txt content = %q, want %q", string(content2), "second file content")
	}

	nested, err := os.ReadFile(filepath.Join(base, "subdir", "nested.txt"))
	if err != nil {
		t.Fatalf("missing nested.txt: %v", err)
	}
	if string(nested) != "nested content" {
		t.Errorf("nested.txt content = %q, want %q", string(nested), "nested content")
	}
}

func TestCompressExtractTarGz(t *testing.T) {
	tmpDir := t.TempDir()
	srcDir := filepath.Join(tmpDir, "testdata")
	os.Mkdir(srcDir, 0755)
	createTestFiles(t, srcDir)

	archivePath := filepath.Join(tmpDir, "test.tar.gz")

	// Compress.
	result, err := ArchiveCompress(srcDir, archivePath, "tar.gz")
	if err != nil {
		t.Fatalf("ArchiveCompress: %v", err)
	}
	if !strings.Contains(result, "Created") {
		t.Errorf("expected 'Created' in result: %s", result)
	}

	if _, err := os.Stat(archivePath); err != nil {
		t.Fatalf("archive not created: %v", err)
	}

	// List.
	listResult, err := ArchiveList(archivePath)
	if err != nil {
		t.Fatalf("ArchiveList: %v", err)
	}
	if !strings.Contains(listResult, "file1.txt") {
		t.Errorf("list should contain file1.txt: %s", listResult)
	}
	if !strings.Contains(listResult, "nested.txt") {
		t.Errorf("list should contain nested.txt: %s", listResult)
	}

	// Extract.
	extractDir := filepath.Join(tmpDir, "extracted")
	os.Mkdir(extractDir, 0755)

	extractResult, err := ArchiveExtract(archivePath, extractDir)
	if err != nil {
		t.Fatalf("ArchiveExtract: %v", err)
	}
	if !strings.Contains(extractResult, "Extracted") {
		t.Errorf("expected 'Extracted' in result: %s", extractResult)
	}

	verifyExtractedFiles(t, extractDir, "testdata")
}

func TestCompressExtractZip(t *testing.T) {
	tmpDir := t.TempDir()
	srcDir := filepath.Join(tmpDir, "testdata")
	os.Mkdir(srcDir, 0755)
	createTestFiles(t, srcDir)

	archivePath := filepath.Join(tmpDir, "test.zip")

	// Compress.
	result, err := ArchiveCompress(srcDir, archivePath, "zip")
	if err != nil {
		t.Fatalf("ArchiveCompress: %v", err)
	}
	if !strings.Contains(result, "Created") {
		t.Errorf("expected 'Created' in result: %s", result)
	}

	// List.
	listResult, err := ArchiveList(archivePath)
	if err != nil {
		t.Fatalf("ArchiveList: %v", err)
	}
	if !strings.Contains(listResult, "file1.txt") {
		t.Errorf("list should contain file1.txt: %s", listResult)
	}

	// Extract.
	extractDir := filepath.Join(tmpDir, "extracted")
	os.Mkdir(extractDir, 0755)

	extractResult, err := ArchiveExtract(archivePath, extractDir)
	if err != nil {
		t.Fatalf("ArchiveExtract: %v", err)
	}
	if !strings.Contains(extractResult, "Extracted") {
		t.Errorf("expected 'Extracted' in result: %s", extractResult)
	}

	verifyExtractedFiles(t, extractDir, "testdata")
}

func TestArchiveCompressDefaultFormat(t *testing.T) {
	tmpDir := t.TempDir()
	testFile := filepath.Join(tmpDir, "data")
	os.Mkdir(testFile, 0755)
	os.WriteFile(filepath.Join(testFile, "a.txt"), []byte("content"), 0644)

	// No format specified, should default to tar.gz.
	result, err := ArchiveCompress(testFile, "", "")
	if err != nil {
		t.Fatalf("ArchiveCompress: %v", err)
	}
	if !strings.Contains(result, ".tar.gz") {
		t.Errorf("default format should be tar.gz: %s", result)
	}
}

func TestArchiveCompressBadSource(t *testing.T) {
	_, err := ArchiveCompress("/nonexistent/path/xyz", "", "tar.gz")
	if err == nil {
		t.Error("expected error for nonexistent source")
	}
}

func TestArchiveExtractBadFile(t *testing.T) {
	_, err := ArchiveExtract("/nonexistent/archive.tar.gz", "")
	if err == nil {
		t.Error("expected error for nonexistent archive")
	}
}

func TestArchiveExtractUnknownFormat(t *testing.T) {
	tmpDir := t.TempDir()
	badFile := filepath.Join(tmpDir, "data.unknown")
	os.WriteFile(badFile, []byte("not an archive"), 0644)

	_, err := ArchiveExtract(badFile, "")
	if err == nil {
		t.Error("expected error for unknown format")
	}
}

func TestArchiveToolRegistration(t *testing.T) {
	r := NewRegistry()
	RegisterArchiveTools(r)

	tool, err := r.Get("archive")
	if err != nil {
		t.Fatal("archive tool not registered")
	}
	if tool.Name != "archive" {
		t.Errorf("tool name = %q, want %q", tool.Name, "archive")
	}
}

func TestArchiveToolMissingPath(t *testing.T) {
	r := NewRegistry()
	RegisterArchiveTools(r)

	tool, _ := r.Get("archive")
	_, err := tool.Execute(map[string]string{"action": "list"})
	if err == nil {
		t.Error("expected error when path is missing")
	}
}

func TestArchiveToolBadAction(t *testing.T) {
	r := NewRegistry()
	RegisterArchiveTools(r)

	tool, _ := r.Get("archive")
	_, err := tool.Execute(map[string]string{"action": "explode", "path": "/tmp"})
	if err == nil {
		t.Error("expected error for unknown action")
	}
}
