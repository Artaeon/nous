package tools

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
)

// CalculateBrightnessPercent computes the brightness percentage from current and max values.
func CalculateBrightnessPercent(current, max int) int {
	if max <= 0 {
		return 0
	}
	pct := (current * 100) / max
	if pct > 100 {
		pct = 100
	}
	if pct < 0 {
		pct = 0
	}
	return pct
}

// getBrightnessBrightnessctl gets the current brightness using brightnessctl.
func getBrightnessBrightnessctl() (string, error) {
	curCmd := exec.Command("brightnessctl", "get")
	curOut, err := curCmd.Output()
	if err != nil {
		return "", fmt.Errorf("brightness: brightnessctl get failed: %w", err)
	}

	maxCmd := exec.Command("brightnessctl", "max")
	maxOut, err := maxCmd.Output()
	if err != nil {
		return "", fmt.Errorf("brightness: brightnessctl max failed: %w", err)
	}

	current, err := strconv.Atoi(strings.TrimSpace(string(curOut)))
	if err != nil {
		return "", fmt.Errorf("brightness: invalid current value: %w", err)
	}

	max, err := strconv.Atoi(strings.TrimSpace(string(maxOut)))
	if err != nil {
		return "", fmt.Errorf("brightness: invalid max value: %w", err)
	}

	pct := CalculateBrightnessPercent(current, max)
	return fmt.Sprintf("Brightness: %d%%", pct), nil
}

// findBacklightDir finds the first backlight directory in /sys/class/backlight/.
func findBacklightDir() (string, error) {
	entries, err := os.ReadDir("/sys/class/backlight")
	if err != nil {
		return "", fmt.Errorf("brightness: cannot read /sys/class/backlight: %w", err)
	}
	if len(entries) == 0 {
		return "", fmt.Errorf("brightness: no backlight device found in /sys/class/backlight")
	}
	return filepath.Join("/sys/class/backlight", entries[0].Name()), nil
}

// getBrightnessSysfs gets brightness by reading /sys/class/backlight/*/brightness.
func getBrightnessSysfs() (string, error) {
	dir, err := findBacklightDir()
	if err != nil {
		return "", err
	}

	curData, err := os.ReadFile(filepath.Join(dir, "brightness"))
	if err != nil {
		return "", fmt.Errorf("brightness: cannot read brightness: %w", err)
	}

	maxData, err := os.ReadFile(filepath.Join(dir, "max_brightness"))
	if err != nil {
		return "", fmt.Errorf("brightness: cannot read max_brightness: %w", err)
	}

	current, err := strconv.Atoi(strings.TrimSpace(string(curData)))
	if err != nil {
		return "", fmt.Errorf("brightness: invalid current value: %w", err)
	}

	max, err := strconv.Atoi(strings.TrimSpace(string(maxData)))
	if err != nil {
		return "", fmt.Errorf("brightness: invalid max value: %w", err)
	}

	pct := CalculateBrightnessPercent(current, max)
	return fmt.Sprintf("Brightness: %d%%", pct), nil
}

// getBrightness tries brightnessctl first, then falls back to sysfs.
func getBrightness() (string, error) {
	if _, err := exec.LookPath("brightnessctl"); err == nil {
		return getBrightnessBrightnessctl()
	}
	return getBrightnessSysfs()
}

// setBrightness sets brightness to the given percentage using brightnessctl or sysfs.
func setBrightness(level int) (string, error) {
	if level < 0 || level > 100 {
		return "", fmt.Errorf("brightness: level must be between 0 and 100, got %d", level)
	}

	if _, err := exec.LookPath("brightnessctl"); err == nil {
		cmd := exec.Command("brightnessctl", "set", fmt.Sprintf("%d%%", level))
		if err := cmd.Run(); err != nil {
			return "", fmt.Errorf("brightness: brightnessctl set failed: %w", err)
		}
		return fmt.Sprintf("Brightness set to %d%%", level), nil
	}

	// Fallback to sysfs
	dir, err := findBacklightDir()
	if err != nil {
		return "", err
	}

	maxData, err := os.ReadFile(filepath.Join(dir, "max_brightness"))
	if err != nil {
		return "", fmt.Errorf("brightness: cannot read max_brightness: %w", err)
	}

	max, err := strconv.Atoi(strings.TrimSpace(string(maxData)))
	if err != nil {
		return "", fmt.Errorf("brightness: invalid max value: %w", err)
	}

	value := (level * max) / 100
	err = os.WriteFile(filepath.Join(dir, "brightness"), []byte(strconv.Itoa(value)), 0644)
	if err != nil {
		return "", fmt.Errorf("brightness: cannot write brightness: %w", err)
	}

	return fmt.Sprintf("Brightness set to %d%%", level), nil
}

// adjustBrightness changes brightness by a step amount using brightnessctl.
func adjustBrightness(direction string, step int) (string, error) {
	if _, err := exec.LookPath("brightnessctl"); err == nil {
		var arg string
		if direction == "up" {
			arg = fmt.Sprintf("+%d%%", step)
		} else {
			arg = fmt.Sprintf("%d%%-", step)
		}
		cmd := exec.Command("brightnessctl", "set", arg)
		if err := cmd.Run(); err != nil {
			return "", fmt.Errorf("brightness: brightnessctl set %s failed: %w", arg, err)
		}
		return fmt.Sprintf("Brightness %s by %d%%", direction, step), nil
	}

	// Fallback: get current, compute new, set
	dir, err := findBacklightDir()
	if err != nil {
		return "", err
	}

	curData, err := os.ReadFile(filepath.Join(dir, "brightness"))
	if err != nil {
		return "", fmt.Errorf("brightness: cannot read brightness: %w", err)
	}

	maxData, err := os.ReadFile(filepath.Join(dir, "max_brightness"))
	if err != nil {
		return "", fmt.Errorf("brightness: cannot read max_brightness: %w", err)
	}

	current, err := strconv.Atoi(strings.TrimSpace(string(curData)))
	if err != nil {
		return "", fmt.Errorf("brightness: invalid current value: %w", err)
	}

	max, err := strconv.Atoi(strings.TrimSpace(string(maxData)))
	if err != nil {
		return "", fmt.Errorf("brightness: invalid max value: %w", err)
	}

	currentPct := CalculateBrightnessPercent(current, max)
	var newPct int
	if direction == "up" {
		newPct = currentPct + step
	} else {
		newPct = currentPct - step
	}
	if newPct > 100 {
		newPct = 100
	}
	if newPct < 0 {
		newPct = 0
	}

	value := (newPct * max) / 100
	err = os.WriteFile(filepath.Join(dir, "brightness"), []byte(strconv.Itoa(value)), 0644)
	if err != nil {
		return "", fmt.Errorf("brightness: cannot write brightness: %w", err)
	}

	return fmt.Sprintf("Brightness %s by %d%%", direction, step), nil
}

// RegisterBrightnessTools adds the brightness tool to the registry.
func RegisterBrightnessTools(r *Registry) {
	r.Register(Tool{
		Name:        "brightness",
		Description: "Control screen brightness. Args: action (get/set/up/down), level (0-100 for set), step (default 10 for up/down). No args returns current brightness.",
		Execute: func(args map[string]string) (string, error) {
			action := strings.ToLower(strings.TrimSpace(args["action"]))

			switch action {
			case "", "get":
				return getBrightness()
			case "set":
				levelStr, ok := args["level"]
				if !ok || levelStr == "" {
					return "", fmt.Errorf("brightness: 'level' argument required for set action")
				}
				level, err := strconv.Atoi(strings.TrimSpace(levelStr))
				if err != nil {
					return "", fmt.Errorf("brightness: invalid level %q: %w", levelStr, err)
				}
				return setBrightness(level)
			case "up", "down":
				step := 10
				if stepStr, ok := args["step"]; ok && stepStr != "" {
					s, err := strconv.Atoi(strings.TrimSpace(stepStr))
					if err != nil {
						return "", fmt.Errorf("brightness: invalid step %q: %w", stepStr, err)
					}
					step = s
				}
				return adjustBrightness(action, step)
			default:
				return "", fmt.Errorf("brightness: unknown action %q (use get, set, up, down)", action)
			}
		},
	})
}
