package tools

import (
	"fmt"
	"os/exec"
	"strings"
)

// clipboardTool determines which clipboard backend to use.
// Returns "wayland" if wl-paste/wl-copy available, "x11" if xclip available, or error.
func clipboardBackend() (string, error) {
	if _, err := exec.LookPath("wl-paste"); err == nil {
		return "wayland", nil
	}
	if _, err := exec.LookPath("xclip"); err == nil {
		return "x11", nil
	}
	return "", fmt.Errorf("clipboard: neither wl-paste/wl-copy (Wayland) nor xclip (X11) is installed")
}

// ClipboardBackend is exported for testing. It returns "wayland", "x11", or error.
var ClipboardBackend = clipboardBackend

// ReadClipboard reads the current clipboard contents.
// Tries wl-paste (Wayland) first, falls back to xclip (X11).
func ReadClipboard() (string, error) {
	backend, err := ClipboardBackend()
	if err != nil {
		return "", err
	}

	var cmd *exec.Cmd
	switch backend {
	case "wayland":
		cmd = exec.Command("wl-paste")
	case "x11":
		cmd = exec.Command("xclip", "-selection", "clipboard", "-o")
	default:
		return "", fmt.Errorf("clipboard: unknown backend %q", backend)
	}

	out, err := cmd.Output()
	if err != nil {
		return "", fmt.Errorf("clipboard read: %w", err)
	}
	return string(out), nil
}

// WriteClipboard writes text to the system clipboard.
// Tries wl-copy (Wayland) first, falls back to xclip (X11).
func WriteClipboard(text string) error {
	backend, err := ClipboardBackend()
	if err != nil {
		return err
	}

	var cmd *exec.Cmd
	switch backend {
	case "wayland":
		cmd = exec.Command("wl-copy")
	case "x11":
		cmd = exec.Command("xclip", "-selection", "clipboard")
	default:
		return fmt.Errorf("clipboard: unknown backend %q", backend)
	}

	cmd.Stdin = strings.NewReader(text)
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("clipboard write: %w", err)
	}
	return nil
}

// RegisterClipboardTools adds the clipboard tool to the registry.
func RegisterClipboardTools(r *Registry) {
	r.Register(Tool{
		Name:        "clipboard",
		Description: "Read from or write to the system clipboard. Args: text (optional, if set writes to clipboard; if absent reads clipboard).",
		Execute: func(args map[string]string) (string, error) {
			if text, ok := args["text"]; ok && text != "" {
				if err := WriteClipboard(text); err != nil {
					return "", err
				}
				return fmt.Sprintf("wrote %d bytes to clipboard", len(text)), nil
			}
			return ReadClipboard()
		},
	})
}
