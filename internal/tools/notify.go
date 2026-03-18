package tools

import (
	"fmt"
	"os/exec"
	"strings"
)

// BuildNotifyCommand constructs the notify-send command arguments from the given parameters.
func BuildNotifyCommand(title, body, urgency string) []string {
	args := []string{"-u", urgency}
	args = append(args, title)
	if body != "" {
		args = append(args, body)
	}
	return args
}

// ValidateNotifyUrgency checks that the urgency level is valid.
func ValidateNotifyUrgency(urgency string) error {
	switch urgency {
	case "low", "normal", "critical":
		return nil
	default:
		return fmt.Errorf("notify: invalid urgency %q (use low, normal, critical)", urgency)
	}
}

// sendNotification sends a desktop notification using notify-send.
func sendNotification(title, body, urgency string) (string, error) {
	if title == "" {
		return "", fmt.Errorf("notify: 'title' argument is required")
	}

	if urgency == "" {
		urgency = "normal"
	}

	if err := ValidateNotifyUrgency(urgency); err != nil {
		return "", err
	}

	cmdArgs := BuildNotifyCommand(title, body, urgency)
	cmd := exec.Command("notify-send", cmdArgs...)
	if err := cmd.Run(); err != nil {
		return "", fmt.Errorf("notify: notify-send failed: %w", err)
	}

	return fmt.Sprintf("Notification sent: %s", title), nil
}

// RegisterNotifyTools adds the notify tool to the registry.
func RegisterNotifyTools(r *Registry) {
	r.Register(Tool{
		Name:        "notify",
		Description: "Send a desktop notification. Args: title (required), body (optional), urgency (low/normal/critical, default normal).",
		Execute: func(args map[string]string) (string, error) {
			title := strings.TrimSpace(args["title"])
			body := strings.TrimSpace(args["body"])
			urgency := strings.ToLower(strings.TrimSpace(args["urgency"]))
			return sendNotification(title, body, urgency)
		},
	})
}
