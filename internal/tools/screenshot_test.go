package tools

import (
	"os"
	"strings"
	"testing"
)

func TestFindScreenshotTool(t *testing.T) {
	// This test verifies the function doesn't panic and returns a valid result.
	// On systems without any screenshot tool, it should return an error.
	name, args, err := FindScreenshotTool()
	if err != nil {
		// Expected on headless/CI systems
		if !strings.Contains(err.Error(), "no capture tool found") {
			t.Errorf("unexpected error: %v", err)
		}
		return
	}

	// If a tool was found, verify it's one of the expected ones
	validTools := map[string]bool{"grim": true, "scrot": true, "import": true}
	if !validTools[name] {
		t.Errorf("unexpected tool name %q", name)
	}
	if len(args) == 0 {
		t.Error("expected non-empty args")
	}
}

func TestScreenshotCommandOrder(t *testing.T) {
	// Verify the command priority order
	if len(screenshotCommands) != 3 {
		t.Fatalf("expected 3 screenshot commands, got %d", len(screenshotCommands))
	}
	if screenshotCommands[0].Name != "grim" {
		t.Errorf("expected first command to be 'grim', got %q", screenshotCommands[0].Name)
	}
	if screenshotCommands[1].Name != "scrot" {
		t.Errorf("expected second command to be 'scrot', got %q", screenshotCommands[1].Name)
	}
	if screenshotCommands[2].Name != "import" {
		t.Errorf("expected third command to be 'import', got %q", screenshotCommands[2].Name)
	}
}

func TestScreenshotPath(t *testing.T) {
	if screenshotPath != "/tmp/nous-screenshot.png" {
		t.Errorf("unexpected screenshot path: %s", screenshotPath)
	}
}

func TestDescribeScreenshot(t *testing.T) {
	// Create a temp file to describe
	tmpFile, err := os.CreateTemp("", "nous-test-*.png")
	if err != nil {
		t.Fatalf("create temp file: %v", err)
	}
	defer os.Remove(tmpFile.Name())

	// Write some dummy data
	tmpFile.Write([]byte("fake png data for testing"))
	tmpFile.Close()

	result, err := DescribeScreenshot(tmpFile.Name())
	if err != nil {
		t.Fatalf("DescribeScreenshot error: %v", err)
	}

	if !strings.Contains(result, "Screenshot saved") {
		t.Error("expected output to contain 'Screenshot saved'")
	}
	if !strings.Contains(result, "Size:") {
		t.Error("expected output to contain 'Size:'")
	}
	if !strings.Contains(result, "Vision model analysis not yet implemented") {
		t.Error("expected output to contain vision note")
	}
}

func TestDescribeScreenshotMissing(t *testing.T) {
	_, err := DescribeScreenshot("/tmp/nonexistent-nous-test.png")
	if err == nil {
		t.Error("expected error for missing file")
	}
}
