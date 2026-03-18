package tools

import (
	"strings"
	"testing"
)

func TestIsCommandAvailable(t *testing.T) {
	// "ls" should always be available
	if !IsCommandAvailable("ls") {
		t.Error("expected 'ls' to be available")
	}

	// A nonsense command should not be available
	if IsCommandAvailable("this_command_does_not_exist_abc123") {
		t.Error("expected nonexistent command to be unavailable")
	}
}

func TestQRGenerateMissingData(t *testing.T) {
	_, err := QRGenerate("", "")
	if err == nil {
		t.Error("expected error for empty data")
	}
	if !strings.Contains(err.Error(), "data is required") {
		t.Errorf("unexpected error: %v", err)
	}
}

func TestQRGenerateMissingTool(t *testing.T) {
	if IsCommandAvailable("qrencode") {
		t.Skip("qrencode is installed, cannot test missing tool detection")
	}
	_, err := QRGenerate("hello", "")
	if err == nil {
		t.Error("expected error when qrencode is not installed")
	}
	if !strings.Contains(err.Error(), "sudo pacman -S qrencode") {
		t.Errorf("expected install suggestion, got: %v", err)
	}
}

func TestQRReadMissingPath(t *testing.T) {
	_, err := QRRead("")
	if err == nil {
		t.Error("expected error for empty path")
	}
	if !strings.Contains(err.Error(), "path is required") {
		t.Errorf("unexpected error: %v", err)
	}
}

func TestQRReadMissingTool(t *testing.T) {
	if IsCommandAvailable("zbarimg") {
		t.Skip("zbarimg is installed, cannot test missing tool detection")
	}
	_, err := QRRead("/tmp/test.png")
	if err == nil {
		t.Error("expected error when zbarimg is not installed")
	}
	if !strings.Contains(err.Error(), "sudo pacman -S zbar") {
		t.Errorf("expected install suggestion, got: %v", err)
	}
}

func TestRegisterQRCodeTools(t *testing.T) {
	r := NewRegistry()
	RegisterQRCodeTools(r)

	tool, err := r.Get("qrcode")
	if err != nil {
		t.Fatalf("qrcode tool not registered: %v", err)
	}

	if tool.Name != "qrcode" {
		t.Errorf("tool name = %q, want %q", tool.Name, "qrcode")
	}

	// Missing data for generate should error
	_, err = tool.Execute(map[string]string{"action": "generate"})
	if err == nil {
		t.Error("expected error for missing data arg")
	}

	// Missing path for read should error
	_, err = tool.Execute(map[string]string{"action": "read"})
	if err == nil {
		t.Error("expected error for missing path arg")
	}

	// Unknown action should error
	_, err = tool.Execute(map[string]string{"action": "invalid"})
	if err == nil {
		t.Error("expected error for unknown action")
	}

	// Default action should be generate
	_, err = tool.Execute(map[string]string{"data": "hello"})
	// This may error due to missing qrencode, but should NOT error with "unknown action"
	if err != nil && strings.Contains(err.Error(), "unknown action") {
		t.Error("default action should be 'generate', not unknown")
	}
}
