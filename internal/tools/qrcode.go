package tools

import (
	"fmt"
	"os/exec"
	"strings"
)

// QRGenerate creates a QR code from the given data.
// If path is provided, it writes a PNG file; otherwise it returns UTF8 terminal output.
// Requires the qrencode command to be installed.
func QRGenerate(data, path string) (string, error) {
	if data == "" {
		return "", fmt.Errorf("qrcode: data is required for generate")
	}

	if !IsCommandAvailable("qrencode") {
		return "", fmt.Errorf("qrencode is not installed. Install it with: sudo pacman -S qrencode")
	}

	if path != "" {
		cmd := exec.Command("qrencode", "-o", path, data)
		out, err := cmd.CombinedOutput()
		if err != nil {
			return "", fmt.Errorf("qrcode: qrencode failed: %s: %w", strings.TrimSpace(string(out)), err)
		}
		return fmt.Sprintf("QR code saved to %s", path), nil
	}

	cmd := exec.Command("qrencode", "-t", "UTF8", "-o", "-", data)
	out, err := cmd.CombinedOutput()
	if err != nil {
		return "", fmt.Errorf("qrcode: qrencode failed: %s: %w", strings.TrimSpace(string(out)), err)
	}
	return strings.TrimSpace(string(out)), nil
}

// QRRead decodes a QR code from an image file.
// Requires the zbarimg command to be installed.
func QRRead(path string) (string, error) {
	if path == "" {
		return "", fmt.Errorf("qrcode: path is required for read")
	}

	if !IsCommandAvailable("zbarimg") {
		return "", fmt.Errorf("zbarimg is not installed. Install it with: sudo pacman -S zbar")
	}

	cmd := exec.Command("zbarimg", "--quiet", "--raw", path)
	out, err := cmd.CombinedOutput()
	if err != nil {
		return "", fmt.Errorf("qrcode: zbarimg failed: %s: %w", strings.TrimSpace(string(out)), err)
	}
	return strings.TrimSpace(string(out)), nil
}

// IsCommandAvailable checks whether a command exists in the system PATH.
func IsCommandAvailable(name string) bool {
	_, err := exec.LookPath(name)
	return err == nil
}

// RegisterQRCodeTools adds the qrcode tool to the registry.
func RegisterQRCodeTools(r *Registry) {
	r.Register(Tool{
		Name:        "qrcode",
		Description: "Generate or read QR codes. Args: action (generate/read, default generate), data (text/URL for generate), path (image path for read, or output path for generate).",
		Execute: func(args map[string]string) (string, error) {
			action := strings.ToLower(strings.TrimSpace(args["action"]))
			if action == "" {
				action = "generate"
			}

			switch action {
			case "generate":
				data := args["data"]
				if data == "" {
					return "", fmt.Errorf("qrcode generate requires 'data' argument")
				}
				return QRGenerate(data, args["path"])
			case "read":
				path := args["path"]
				if path == "" {
					return "", fmt.Errorf("qrcode read requires 'path' argument")
				}
				return QRRead(path)
			default:
				return "", fmt.Errorf("qrcode: unknown action %q (use 'generate' or 'read')", action)
			}
		},
	})
}
