package tools

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestParseDesktopFile(t *testing.T) {
	content := `[Desktop Entry]
Name=Firefox Web Browser
Exec=firefox %u
Icon=firefox
Type=Application
Categories=Network;WebBrowser;

[Desktop Action new-window]
Name=Open a New Window
Exec=firefox --new-window
`

	entry := ParseDesktopFile(content)

	if entry.Name != "Firefox Web Browser" {
		t.Errorf("Name = %q, want 'Firefox Web Browser'", entry.Name)
	}
	if entry.Exec != "firefox %u" {
		t.Errorf("Exec = %q, want 'firefox %%u'", entry.Exec)
	}
	if entry.Icon != "firefox" {
		t.Errorf("Icon = %q, want 'firefox'", entry.Icon)
	}
}

func TestParseDesktopFileMinimal(t *testing.T) {
	content := `[Desktop Entry]
Name=Simple App
Exec=/usr/bin/simple
`

	entry := ParseDesktopFile(content)

	if entry.Name != "Simple App" {
		t.Errorf("Name = %q, want 'Simple App'", entry.Name)
	}
	if entry.Exec != "/usr/bin/simple" {
		t.Errorf("Exec = %q, want '/usr/bin/simple'", entry.Exec)
	}
	if entry.Icon != "" {
		t.Errorf("Icon = %q, want empty", entry.Icon)
	}
}

func TestParseDesktopFileEmpty(t *testing.T) {
	entry := ParseDesktopFile("")
	if entry.Name != "" || entry.Exec != "" || entry.Icon != "" {
		t.Errorf("expected empty entry for empty content, got: %+v", entry)
	}
}

func TestParseDesktopFileNoSection(t *testing.T) {
	content := `Name=Should Not Match
Exec=nope
`
	entry := ParseDesktopFile(content)
	if entry.Name != "" {
		t.Errorf("should not parse Name outside [Desktop Entry] section, got: %q", entry.Name)
	}
}

func TestCleanExecLine(t *testing.T) {
	tests := []struct {
		input, want string
	}{
		{"firefox %u", "firefox"},
		{"libreoffice --calc %F", "libreoffice --calc"},
		{"/usr/bin/app", "/usr/bin/app"},
		{"app %f %U", "app"},
		{"", ""},
	}

	for _, tt := range tests {
		got := cleanExecLine(tt.input)
		if got != tt.want {
			t.Errorf("cleanExecLine(%q) = %q, want %q", tt.input, got, tt.want)
		}
	}
}

func TestAppListFromDir(t *testing.T) {
	dir := t.TempDir()

	// Create mock .desktop files.
	writeDesktopFile(t, dir, "firefox.desktop", "Firefox", "firefox", "firefox")
	writeDesktopFile(t, dir, "terminal.desktop", "Terminal", "terminal", "terminal")
	writeDesktopFile(t, dir, "editor.desktop", "Code Editor", "code", "code")

	result, err := appListFromDir(dir)
	if err != nil {
		t.Fatalf("appListFromDir: %v", err)
	}

	if !strings.Contains(result, "3 application(s)") {
		t.Errorf("should show 3 apps, got: %s", result)
	}
	if !strings.Contains(result, "Firefox") {
		t.Errorf("should contain Firefox, got: %s", result)
	}
	if !strings.Contains(result, "Terminal") {
		t.Errorf("should contain Terminal, got: %s", result)
	}
	if !strings.Contains(result, "Code Editor") {
		t.Errorf("should contain Code Editor, got: %s", result)
	}

	// Verify sorted order.
	lines := strings.Split(result, "\n")
	var appNames []string
	for _, line := range lines {
		if strings.HasPrefix(line, "- ") {
			appNames = append(appNames, strings.TrimPrefix(line, "- "))
		}
	}
	if len(appNames) == 3 {
		if appNames[0] != "Code Editor" || appNames[1] != "Firefox" || appNames[2] != "Terminal" {
			t.Errorf("apps not sorted: %v", appNames)
		}
	}
}

func TestAppListFromDirEmpty(t *testing.T) {
	dir := t.TempDir()

	result, err := appListFromDir(dir)
	if err != nil {
		t.Fatalf("appListFromDir: %v", err)
	}
	if result != "No applications found." {
		t.Errorf("expected empty message, got: %s", result)
	}
}

func TestFindDesktopEntryInDir(t *testing.T) {
	dir := t.TempDir()

	writeDesktopFile(t, dir, "firefox.desktop", "Firefox Web Browser", "firefox %u", "firefox")
	writeDesktopFile(t, dir, "gimp.desktop", "GIMP Image Editor", "gimp %U", "gimp")

	// Exact match.
	entry, err := findDesktopEntryInDir(dir, "Firefox")
	if err != nil {
		t.Fatalf("findDesktopEntryInDir: %v", err)
	}
	if entry.Name != "Firefox Web Browser" {
		t.Errorf("Name = %q, want 'Firefox Web Browser'", entry.Name)
	}

	// Partial, case-insensitive match.
	entry, err = findDesktopEntryInDir(dir, "gimp")
	if err != nil {
		t.Fatalf("findDesktopEntryInDir: %v", err)
	}
	if entry.Name != "GIMP Image Editor" {
		t.Errorf("Name = %q, want 'GIMP Image Editor'", entry.Name)
	}

	// Not found.
	_, err = findDesktopEntryInDir(dir, "nonexistent")
	if err == nil {
		t.Error("expected error for nonexistent app")
	}
}

func TestAppToolRegistration(t *testing.T) {
	r := NewRegistry()
	RegisterAppTools(r)

	tool, err := r.Get("app")
	if err != nil {
		t.Fatal("app tool not registered")
	}
	if tool.Name != "app" {
		t.Errorf("tool name = %q, want %q", tool.Name, "app")
	}
}

func TestAppToolUnknownAction(t *testing.T) {
	_, err := toolApp(map[string]string{"action": "invalid"})
	if err == nil {
		t.Error("expected error for unknown action")
	}
}

func TestAppToolMissingName(t *testing.T) {
	_, err := toolApp(map[string]string{"action": "launch"})
	if err == nil {
		t.Error("expected error for launch without name")
	}

	_, err = toolApp(map[string]string{"action": "running"})
	if err == nil {
		t.Error("expected error for running without name")
	}

	_, err = toolApp(map[string]string{"action": "kill"})
	if err == nil {
		t.Error("expected error for kill without name")
	}
}

// writeDesktopFile is a test helper that creates a mock .desktop file.
func writeDesktopFile(t *testing.T, dir, filename, name, execLine, icon string) {
	t.Helper()
	content := "[Desktop Entry]\nName=" + name + "\nExec=" + execLine + "\nIcon=" + icon + "\nType=Application\n"
	err := os.WriteFile(filepath.Join(dir, filename), []byte(content), 0644)
	if err != nil {
		t.Fatalf("write desktop file: %v", err)
	}
}
