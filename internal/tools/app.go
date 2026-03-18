package tools

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"syscall"
)

// DesktopEntry holds parsed fields from a .desktop file.
type DesktopEntry struct {
	Name string
	Exec string
	Icon string
}

// ParseDesktopFile extracts Name, Exec, and Icon from .desktop file content.
func ParseDesktopFile(content string) DesktopEntry {
	var entry DesktopEntry
	inDesktopSection := false

	for _, line := range strings.Split(content, "\n") {
		line = strings.TrimSpace(line)

		if line == "[Desktop Entry]" {
			inDesktopSection = true
			continue
		}
		if strings.HasPrefix(line, "[") && strings.HasSuffix(line, "]") {
			// Another section started — stop parsing.
			if inDesktopSection {
				break
			}
			continue
		}

		if !inDesktopSection {
			continue
		}

		parts := strings.SplitN(line, "=", 2)
		if len(parts) != 2 {
			continue
		}
		key := strings.TrimSpace(parts[0])
		val := strings.TrimSpace(parts[1])

		switch key {
		case "Name":
			if entry.Name == "" {
				entry.Name = val
			}
		case "Exec":
			if entry.Exec == "" {
				entry.Exec = val
			}
		case "Icon":
			if entry.Icon == "" {
				entry.Icon = val
			}
		}
	}

	return entry
}

// cleanExecLine strips desktop entry field codes (%f, %F, %u, %U, etc.) from an Exec line.
func cleanExecLine(execLine string) string {
	fields := strings.Fields(execLine)
	var cleaned []string
	for _, f := range fields {
		if strings.HasPrefix(f, "%") && len(f) == 2 {
			continue
		}
		cleaned = append(cleaned, f)
	}
	return strings.Join(cleaned, " ")
}

// appLaunch finds and launches a desktop application by name.
func appLaunch(name string) (string, error) {
	if name == "" {
		return "", fmt.Errorf("app launch requires 'name'")
	}

	entry, err := findDesktopEntry(name)
	if err != nil {
		return "", err
	}

	execLine := cleanExecLine(entry.Exec)
	if execLine == "" {
		return "", fmt.Errorf("app: no Exec line found for %q", name)
	}

	parts := strings.Fields(execLine)
	cmd := exec.Command(parts[0], parts[1:]...)
	cmd.SysProcAttr = &syscall.SysProcAttr{Setsid: true}
	cmd.Stdout = nil
	cmd.Stderr = nil
	cmd.Stdin = nil

	if err := cmd.Start(); err != nil {
		return "", fmt.Errorf("app: failed to launch %q: %w", entry.Name, err)
	}

	// Detach — don't wait for the process.
	go cmd.Wait()

	return fmt.Sprintf("Launched '%s' (pid %d)", entry.Name, cmd.Process.Pid), nil
}

// findDesktopEntry scans /usr/share/applications for a matching .desktop file.
func findDesktopEntry(name string) (DesktopEntry, error) {
	return findDesktopEntryInDir("/usr/share/applications", name)
}

// findDesktopEntryInDir scans a directory for a matching .desktop file (for testing).
func findDesktopEntryInDir(dir, name string) (DesktopEntry, error) {
	matches, err := filepath.Glob(filepath.Join(dir, "*.desktop"))
	if err != nil {
		return DesktopEntry{}, fmt.Errorf("app: cannot scan applications: %w", err)
	}

	lowerName := strings.ToLower(name)
	for _, path := range matches {
		data, err := os.ReadFile(path)
		if err != nil {
			continue
		}
		entry := ParseDesktopFile(string(data))
		if strings.Contains(strings.ToLower(entry.Name), lowerName) {
			return entry, nil
		}
	}

	return DesktopEntry{}, fmt.Errorf("app: no application found matching %q", name)
}

// appList returns a sorted list of installed applications from .desktop files.
func appList() (string, error) {
	return appListFromDir("/usr/share/applications")
}

// appListFromDir lists applications from a given directory (for testing).
func appListFromDir(dir string) (string, error) {
	matches, err := filepath.Glob(filepath.Join(dir, "*.desktop"))
	if err != nil {
		return "", fmt.Errorf("app: cannot scan applications: %w", err)
	}

	var names []string
	for _, path := range matches {
		data, err := os.ReadFile(path)
		if err != nil {
			continue
		}
		entry := ParseDesktopFile(string(data))
		if entry.Name != "" {
			names = append(names, entry.Name)
		}
	}

	sort.Strings(names)

	if len(names) == 0 {
		return "No applications found.", nil
	}

	var sb strings.Builder
	fmt.Fprintf(&sb, "%d application(s):\n", len(names))
	for _, name := range names {
		fmt.Fprintf(&sb, "- %s\n", name)
	}
	return sb.String(), nil
}

// appRunning finds running processes matching a name by reading /proc/*/comm.
func appRunning(name string) (string, error) {
	if name == "" {
		return "", fmt.Errorf("app running requires 'name'")
	}

	entries, err := os.ReadDir("/proc")
	if err != nil {
		return "", fmt.Errorf("app: cannot read /proc: %w", err)
	}

	lowerName := strings.ToLower(name)
	var matches []string

	for _, e := range entries {
		if !e.IsDir() {
			continue
		}
		// Only consider numeric directories (PIDs).
		pid, err := strconv.Atoi(e.Name())
		if err != nil {
			continue
		}

		commPath := filepath.Join("/proc", e.Name(), "comm")
		data, err := os.ReadFile(commPath)
		if err != nil {
			continue
		}
		comm := strings.TrimSpace(string(data))
		if strings.Contains(strings.ToLower(comm), lowerName) {
			matches = append(matches, fmt.Sprintf("- %s (pid %d)", comm, pid))
		}
	}

	if len(matches) == 0 {
		return fmt.Sprintf("No running processes matching %q.", name), nil
	}

	return fmt.Sprintf("%d process(es) matching %q:\n%s", len(matches), name, strings.Join(matches, "\n")), nil
}

// appKill finds a process by name and sends SIGTERM.
func appKill(name string) (string, error) {
	if name == "" {
		return "", fmt.Errorf("app kill requires 'name'")
	}

	entries, err := os.ReadDir("/proc")
	if err != nil {
		return "", fmt.Errorf("app: cannot read /proc: %w", err)
	}

	lowerName := strings.ToLower(name)

	for _, e := range entries {
		if !e.IsDir() {
			continue
		}
		pid, err := strconv.Atoi(e.Name())
		if err != nil {
			continue
		}

		commPath := filepath.Join("/proc", e.Name(), "comm")
		data, err := os.ReadFile(commPath)
		if err != nil {
			continue
		}
		comm := strings.TrimSpace(string(data))
		if strings.Contains(strings.ToLower(comm), lowerName) {
			proc, err := os.FindProcess(pid)
			if err != nil {
				return "", fmt.Errorf("app: cannot find process %d: %w", pid, err)
			}
			if err := proc.Signal(syscall.SIGTERM); err != nil {
				return "", fmt.Errorf("app: cannot kill process %d (%s): %w", pid, comm, err)
			}
			return fmt.Sprintf("Sent SIGTERM to '%s' (pid %d).", comm, pid), nil
		}
	}

	return "", fmt.Errorf("app: no running process matching %q", name)
}

// RegisterAppTools adds the app tool to the registry.
func RegisterAppTools(r *Registry) {
	r.Register(Tool{
		Name:        "app",
		Description: "Application launcher and process finder. Args: action (launch/list/kill/running), name (app name).",
		Execute: func(args map[string]string) (string, error) {
			return toolApp(args)
		},
	})
}

func toolApp(args map[string]string) (string, error) {
	action := strings.ToLower(strings.TrimSpace(args["action"]))

	switch action {
	case "launch":
		return appLaunch(args["name"])
	case "list":
		return appList()
	case "running":
		return appRunning(args["name"])
	case "kill":
		return appKill(args["name"])
	default:
		return "", fmt.Errorf("app: unknown action %q (use launch/list/kill/running)", action)
	}
}
