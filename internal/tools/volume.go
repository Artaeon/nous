package tools

import (
	"fmt"
	"os/exec"
	"strconv"
	"strings"
)

// ParseVolumeOutput extracts the volume percentage from pactl get-sink-volume output.
// Example input: "Volume: front-left: 32768 /  50% / -18.06 dB,   front-right: 32768 /  50% / -18.06 dB"
func ParseVolumeOutput(output string) (int, error) {
	// Find the first percentage value
	parts := strings.Split(output, "/")
	for _, part := range parts {
		trimmed := strings.TrimSpace(part)
		if strings.HasSuffix(trimmed, "%") {
			numStr := strings.TrimSuffix(trimmed, "%")
			numStr = strings.TrimSpace(numStr)
			val, err := strconv.Atoi(numStr)
			if err != nil {
				continue
			}
			return val, nil
		}
	}
	return 0, fmt.Errorf("volume: could not parse percentage from output: %s", output)
}

// ParseMuteOutput extracts the mute status from pactl get-sink-mute output.
// Example input: "Mute: yes" or "Mute: no"
func ParseMuteOutput(output string) (bool, error) {
	lower := strings.ToLower(strings.TrimSpace(output))
	if strings.Contains(lower, "yes") {
		return true, nil
	}
	if strings.Contains(lower, "no") {
		return false, nil
	}
	return false, fmt.Errorf("volume: could not parse mute status from output: %s", output)
}

// getVolume returns the current volume percentage and mute status.
func getVolume() (string, error) {
	volCmd := exec.Command("pactl", "get-sink-volume", "@DEFAULT_SINK@")
	volOut, err := volCmd.Output()
	if err != nil {
		return "", fmt.Errorf("volume: pactl get-sink-volume failed: %w", err)
	}

	vol, err := ParseVolumeOutput(string(volOut))
	if err != nil {
		return "", err
	}

	muteCmd := exec.Command("pactl", "get-sink-mute", "@DEFAULT_SINK@")
	muteOut, err := muteCmd.Output()
	if err != nil {
		return "", fmt.Errorf("volume: pactl get-sink-mute failed: %w", err)
	}

	muted, err := ParseMuteOutput(string(muteOut))
	if err != nil {
		return "", err
	}

	muteStr := "unmuted"
	if muted {
		muteStr = "muted"
	}

	return fmt.Sprintf("Volume: %d%% (%s)", vol, muteStr), nil
}

// setVolume sets the volume to the given percentage (0-100).
func setVolume(level int) (string, error) {
	if level < 0 || level > 100 {
		return "", fmt.Errorf("volume: level must be between 0 and 100, got %d", level)
	}
	cmd := exec.Command("pactl", "set-sink-volume", "@DEFAULT_SINK@", fmt.Sprintf("%d%%", level))
	if err := cmd.Run(); err != nil {
		return "", fmt.Errorf("volume: pactl set-sink-volume failed: %w", err)
	}
	return fmt.Sprintf("Volume set to %d%%", level), nil
}

// muteVolume mutes the default sink.
func muteVolume() (string, error) {
	cmd := exec.Command("pactl", "set-sink-mute", "@DEFAULT_SINK@", "1")
	if err := cmd.Run(); err != nil {
		return "", fmt.Errorf("volume: pactl set-sink-mute failed: %w", err)
	}
	return "Audio muted", nil
}

// unmuteVolume unmutes the default sink.
func unmuteVolume() (string, error) {
	cmd := exec.Command("pactl", "set-sink-mute", "@DEFAULT_SINK@", "0")
	if err := cmd.Run(); err != nil {
		return "", fmt.Errorf("volume: pactl set-sink-mute failed: %w", err)
	}
	return "Audio unmuted", nil
}

// toggleMute toggles the mute state of the default sink.
func toggleMute() (string, error) {
	cmd := exec.Command("pactl", "set-sink-mute", "@DEFAULT_SINK@", "toggle")
	if err := cmd.Run(); err != nil {
		return "", fmt.Errorf("volume: pactl set-sink-mute toggle failed: %w", err)
	}
	return "Audio mute toggled", nil
}

// RegisterVolumeTools adds the volume tool to the registry.
func RegisterVolumeTools(r *Registry) {
	r.Register(Tool{
		Name:        "volume",
		Description: "Control audio volume. Args: action (get/set/mute/unmute/toggle), level (0-100 for set). No args returns current volume.",
		Execute: func(args map[string]string) (string, error) {
			action := strings.ToLower(strings.TrimSpace(args["action"]))

			switch action {
			case "", "get":
				return getVolume()
			case "set":
				levelStr, ok := args["level"]
				if !ok || levelStr == "" {
					return "", fmt.Errorf("volume: 'level' argument required for set action")
				}
				level, err := strconv.Atoi(strings.TrimSpace(levelStr))
				if err != nil {
					return "", fmt.Errorf("volume: invalid level %q: %w", levelStr, err)
				}
				return setVolume(level)
			case "mute":
				return muteVolume()
			case "unmute":
				return unmuteVolume()
			case "toggle":
				return toggleMute()
			default:
				return "", fmt.Errorf("volume: unknown action %q (use get, set, mute, unmute, toggle)", action)
			}
		},
	})
}
