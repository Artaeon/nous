package tools

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestFormatDiskSize(t *testing.T) {
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
		got := FormatDiskSize(tt.bytes)
		if got != tt.want {
			t.Errorf("FormatDiskSize(%d) = %q, want %q", tt.bytes, got, tt.want)
		}
	}
}

func TestGetDiskUsage(t *testing.T) {
	tmpDir := t.TempDir()

	// Create directory structure with known sizes.
	largeDir := filepath.Join(tmpDir, "large")
	smallDir := filepath.Join(tmpDir, "small")
	os.Mkdir(largeDir, 0755)
	os.Mkdir(smallDir, 0755)

	// Write files with known sizes.
	largeData := make([]byte, 10000)
	smallData := make([]byte, 100)

	os.WriteFile(filepath.Join(largeDir, "big1.bin"), largeData, 0644)
	os.WriteFile(filepath.Join(largeDir, "big2.bin"), largeData, 0644)
	os.WriteFile(filepath.Join(smallDir, "tiny.bin"), smallData, 0644)

	result, err := GetDiskUsage(tmpDir, 10, 3)
	if err != nil {
		t.Fatalf("GetDiskUsage: %v", err)
	}

	if !strings.Contains(result, "large") {
		t.Errorf("result should contain 'large' dir: %s", result)
	}
	if !strings.Contains(result, "small") {
		t.Errorf("result should contain 'small' dir: %s", result)
	}

	// "large" should appear before "small" (sorted by size desc).
	largeIdx := strings.Index(result, "large")
	smallIdx := strings.Index(result, "small")
	if largeIdx > smallIdx {
		t.Errorf("large dir should be listed before small dir:\n%s", result)
	}
}

func TestGetDiskUsageTopN(t *testing.T) {
	tmpDir := t.TempDir()

	// Create 5 directories.
	for i := 0; i < 5; i++ {
		dir := filepath.Join(tmpDir, string(rune('a'+i)))
		os.Mkdir(dir, 0755)
		data := make([]byte, (i+1)*1000)
		os.WriteFile(filepath.Join(dir, "file.bin"), data, 0644)
	}

	result, err := GetDiskUsage(tmpDir, 3, 3)
	if err != nil {
		t.Fatalf("GetDiskUsage: %v", err)
	}

	lines := strings.Split(strings.TrimSpace(result), "\n")
	if len(lines) > 3 {
		t.Errorf("expected max 3 results, got %d:\n%s", len(lines), result)
	}
}

func TestGetDiskUsageEmptyDir(t *testing.T) {
	tmpDir := t.TempDir()

	result, err := GetDiskUsage(tmpDir, 10, 3)
	if err != nil {
		t.Fatalf("GetDiskUsage: %v", err)
	}
	if !strings.Contains(result, "No directories found") {
		t.Errorf("expected no directories message: %s", result)
	}
}

func TestGetDiskUsageBadPath(t *testing.T) {
	_, err := GetDiskUsage("/nonexistent/path/xyz", 10, 3)
	if err == nil {
		t.Error("expected error for nonexistent path")
	}
}

func TestGetDiskUsageNotDir(t *testing.T) {
	tmpDir := t.TempDir()
	file := filepath.Join(tmpDir, "file.txt")
	os.WriteFile(file, []byte("data"), 0644)

	_, err := GetDiskUsage(file, 10, 3)
	if err == nil {
		t.Error("expected error for non-directory path")
	}
}

func TestGetDiskUsageSkipsHiddenDirs(t *testing.T) {
	tmpDir := t.TempDir()

	// Create a .git dir with large files.
	gitDir := filepath.Join(tmpDir, ".git")
	os.Mkdir(gitDir, 0755)
	os.WriteFile(filepath.Join(gitDir, "objects"), make([]byte, 50000), 0644)

	// Create a normal dir.
	srcDir := filepath.Join(tmpDir, "src")
	os.Mkdir(srcDir, 0755)
	os.WriteFile(filepath.Join(srcDir, "main.go"), []byte("package main"), 0644)

	result, err := GetDiskUsage(tmpDir, 10, 3)
	if err != nil {
		t.Fatalf("GetDiskUsage: %v", err)
	}

	// .git should be skipped.
	if strings.Contains(result, ".git") {
		t.Errorf(".git should be skipped: %s", result)
	}
	if !strings.Contains(result, "src") {
		t.Errorf("should contain src dir: %s", result)
	}
}

func TestFormatDiskUsage(t *testing.T) {
	entries := []DirSize{
		{Path: "/home/user/projects", Size: 1073741824},
		{Path: "/home/user/downloads", Size: 524288000},
	}
	result := FormatDiskUsage(entries, "/home/user")
	if !strings.Contains(result, "1.0G") {
		t.Errorf("should contain 1.0G: %s", result)
	}
	if !strings.Contains(result, "500.0M") {
		t.Errorf("should contain 500.0M: %s", result)
	}
}

func TestDiskUsageToolRegistration(t *testing.T) {
	r := NewRegistry()
	RegisterDiskUsageTools(r)

	tool, err := r.Get("diskusage")
	if err != nil {
		t.Fatal("diskusage tool not registered")
	}
	if tool.Name != "diskusage" {
		t.Errorf("tool name = %q, want %q", tool.Name, "diskusage")
	}
}
