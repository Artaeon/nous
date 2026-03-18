package tools

import (
	"bytes"
	"fmt"
	"os"
	"os/exec"
	"strings"
)

const screenshotPath = "/tmp/nous-screenshot.png"

// screenshotCommands lists screenshot tools in priority order.
var screenshotCommands = []struct {
	Name string
	Args []string
}{
	{"grim", []string{screenshotPath}},
	{"scrot", []string{screenshotPath}},
	{"import", []string{"-window", "root", screenshotPath}},
}

// FindScreenshotTool returns the name and args of the first available screenshot tool.
func FindScreenshotTool() (string, []string, error) {
	for _, cmd := range screenshotCommands {
		if path, err := exec.LookPath(cmd.Name); err == nil && path != "" {
			return cmd.Name, cmd.Args, nil
		}
	}
	return "", nil, fmt.Errorf("screenshot: no capture tool found — install grim (Wayland), scrot (X11), or imagemagick (import)")
}

// TakeScreenshot captures a screenshot and returns the file path.
func TakeScreenshot() (string, error) {
	name, args, err := FindScreenshotTool()
	if err != nil {
		return "", err
	}

	cmd := exec.Command(name, args...)
	var stderr bytes.Buffer
	cmd.Stderr = &stderr

	if err := cmd.Run(); err != nil {
		return "", fmt.Errorf("screenshot: %s failed: %s (%w)", name, strings.TrimSpace(stderr.String()), err)
	}

	return screenshotPath, nil
}

// DescribeScreenshot returns metadata about a screenshot file.
func DescribeScreenshot(path string) (string, error) {
	info, err := os.Stat(path)
	if err != nil {
		return "", fmt.Errorf("screenshot: %w", err)
	}

	var sb strings.Builder
	fmt.Fprintf(&sb, "Screenshot saved: %s\n", path)
	fmt.Fprintf(&sb, "Size: %d bytes\n", info.Size())

	// Try to get dimensions via the file command
	cmd := exec.Command("file", path)
	if out, err := cmd.Output(); err == nil {
		outStr := strings.TrimSpace(string(out))
		// file output typically contains dimensions like "1920 x 1080"
		if strings.Contains(outStr, "PNG") || strings.Contains(outStr, "image") {
			fmt.Fprintf(&sb, "Info: %s\n", outStr)
		}
	}

	sb.WriteString("Note: Vision model analysis not yet implemented.")

	return sb.String(), nil
}

// RegisterScreenshotTools adds the screenshot tool to the registry.
func RegisterScreenshotTools(r *Registry) {
	r.Register(Tool{
		Name:        "screenshot",
		Description: "Capture a screenshot of the current screen. Args: none.",
		Execute: func(args map[string]string) (string, error) {
			path, err := TakeScreenshot()
			if err != nil {
				return "", err
			}
			return DescribeScreenshot(path)
		},
	})
}
