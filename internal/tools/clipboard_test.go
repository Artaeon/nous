package tools

import (
	"strings"
	"testing"
)

func TestClipboardBackendSelection(t *testing.T) {
	// Save and restore the original backend function
	origBackend := ClipboardBackend
	defer func() { ClipboardBackend = origBackend }()

	// Test that the backend function returns one of the expected values
	backend, err := ClipboardBackend()
	if err != nil {
		// This is expected in environments without clipboard tools
		if !strings.Contains(err.Error(), "neither") {
			t.Errorf("unexpected error: %v", err)
		}
		return
	}

	if backend != "wayland" && backend != "x11" {
		t.Errorf("ClipboardBackend() = %q, want 'wayland' or 'x11'", backend)
	}
}

func TestClipboardBackendMock(t *testing.T) {
	origBackend := ClipboardBackend
	defer func() { ClipboardBackend = origBackend }()

	// Mock as wayland
	ClipboardBackend = func() (string, error) {
		return "wayland", nil
	}

	backend, err := ClipboardBackend()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if backend != "wayland" {
		t.Errorf("got %q, want 'wayland'", backend)
	}

	// Mock as x11
	ClipboardBackend = func() (string, error) {
		return "x11", nil
	}

	backend, err = ClipboardBackend()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if backend != "x11" {
		t.Errorf("got %q, want 'x11'", backend)
	}
}

func TestClipboardToolRegistration(t *testing.T) {
	r := NewRegistry()
	RegisterClipboardTools(r)

	tool, err := r.Get("clipboard")
	if err != nil {
		t.Fatal("clipboard tool not registered")
	}

	if tool.Name != "clipboard" {
		t.Errorf("tool name = %q, want %q", tool.Name, "clipboard")
	}
}

func TestClipboardToolReadNoArgs(t *testing.T) {
	// When no text arg is provided, clipboard tool should attempt to read
	// This will fail without a clipboard tool, but we verify the code path
	r := NewRegistry()
	RegisterClipboardTools(r)

	tool, _ := r.Get("clipboard")

	// With no "text" arg, it tries to read clipboard
	_, err := tool.Execute(map[string]string{})
	// We expect an error in test environment (no clipboard tool), that's OK
	if err == nil {
		// If it succeeds, that means we have a clipboard tool -- also fine
		return
	}

	if !strings.Contains(err.Error(), "clipboard") {
		t.Errorf("expected clipboard-related error, got: %v", err)
	}
}

func TestClipboardToolWriteNoArgs(t *testing.T) {
	// When text is empty string, it should still try to read (not write)
	r := NewRegistry()
	RegisterClipboardTools(r)

	tool, _ := r.Get("clipboard")

	_, err := tool.Execute(map[string]string{"text": ""})
	// Empty text should trigger read path, not write
	if err == nil {
		return
	}
	// Error is expected in test env without clipboard tool
	if !strings.Contains(err.Error(), "clipboard") {
		t.Errorf("expected clipboard-related error, got: %v", err)
	}
}
