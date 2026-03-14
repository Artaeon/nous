package tools

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func writeScript(t *testing.T, dir, name, content string) string {
	t.Helper()
	path := filepath.Join(dir, name)
	if err := os.WriteFile(path, []byte(content), 0755); err != nil {
		t.Fatalf("failed to write script %s: %v", name, err)
	}
	return path
}

func TestLoadPluginsDescribeParsing(t *testing.T) {
	dir := t.TempDir()
	writeScript(t, dir, "greet.sh", `#!/bin/sh
if [ "$1" = "--describe" ]; then
	echo '{"name":"greet","description":"Greets someone","args":["name"]}'
	exit 0
fi
echo "Hello, $NOUS_ARG_name!"
`)

	registry := NewRegistry()
	loaded, err := LoadPlugins(dir, registry)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if loaded != 1 {
		t.Fatalf("expected 1 plugin loaded, got %d", loaded)
	}

	tool, err := registry.Get("greet")
	if err != nil {
		t.Fatalf("expected tool 'greet' to be registered: %v", err)
	}
	if tool.Description != "Greets someone" {
		t.Errorf("expected description 'Greets someone', got %q", tool.Description)
	}
}

func TestLoadPluginsExecution(t *testing.T) {
	dir := t.TempDir()
	writeScript(t, dir, "echo_args.sh", `#!/bin/sh
if [ "$1" = "--describe" ]; then
	echo '{"name":"echo_args","description":"Echoes args","args":["message","count"]}'
	exit 0
fi
echo "msg=$NOUS_ARG_message count=$NOUS_ARG_count"
`)

	registry := NewRegistry()
	_, err := LoadPlugins(dir, registry)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	tool, _ := registry.Get("echo_args")
	result, err := tool.Execute(map[string]string{
		"message": "hello",
		"count":   "3",
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.Contains(result, "msg=hello") {
		t.Errorf("expected 'msg=hello' in output, got %q", result)
	}
	if !strings.Contains(result, "count=3") {
		t.Errorf("expected 'count=3' in output, got %q", result)
	}
}

func TestLoadPluginsTimeout(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping timeout test in short mode")
	}

	dir := t.TempDir()
	// Use a script that traps signals and blocks, to verify our timeout works.
	// The script uses a subshell sleep to ensure the process group is killed.
	writeScript(t, dir, "slow.sh", `#!/bin/sh
if [ "$1" = "--describe" ]; then
	echo '{"name":"slow","description":"Slow plugin","args":[]}'
	exit 0
fi
# Loop with short sleeps so the process responds to SIGKILL promptly
i=0
while [ $i -lt 600 ]; do
	sleep 0.1
	i=$((i + 1))
done
echo "done"
`)

	registry := NewRegistry()
	_, err := LoadPlugins(dir, registry)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	tool, _ := registry.Get("slow")
	start := time.Now()
	_, err = tool.Execute(map[string]string{})
	elapsed := time.Since(start)

	if err == nil {
		t.Fatal("expected timeout error")
	}
	if !strings.Contains(err.Error(), "timed out") && !strings.Contains(err.Error(), "killed") && !strings.Contains(err.Error(), "failed") {
		t.Errorf("expected timeout-related error, got %q", err.Error())
	}
	// Should timeout around 30s, not 60s
	if elapsed > 35*time.Second {
		t.Errorf("expected timeout within ~30s, took %v", elapsed)
	}
}

func TestLoadPluginsEmptyDir(t *testing.T) {
	dir := t.TempDir()
	registry := NewRegistry()
	loaded, err := LoadPlugins(dir, registry)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if loaded != 0 {
		t.Errorf("expected 0 plugins, got %d", loaded)
	}
}

func TestLoadPluginsNonExistentDir(t *testing.T) {
	registry := NewRegistry()
	loaded, err := LoadPlugins("/nonexistent/path/nous/plugins", registry)
	if err != nil {
		t.Fatalf("unexpected error for nonexistent dir: %v", err)
	}
	if loaded != 0 {
		t.Errorf("expected 0 plugins, got %d", loaded)
	}
}

func TestLoadPluginsSkipsNonExecutable(t *testing.T) {
	dir := t.TempDir()
	// Write a non-executable file
	os.WriteFile(filepath.Join(dir, "notexec.sh"), []byte(`#!/bin/sh
echo '{"name":"notexec","description":"test","args":[]}'
`), 0644)

	registry := NewRegistry()
	loaded, err := LoadPlugins(dir, registry)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if loaded != 0 {
		t.Errorf("expected 0 plugins (non-executable skipped), got %d", loaded)
	}
}

func TestLoadPluginsSkipsBadDescribe(t *testing.T) {
	dir := t.TempDir()
	writeScript(t, dir, "bad.sh", `#!/bin/sh
if [ "$1" = "--describe" ]; then
	echo "not valid json"
	exit 0
fi
echo "hello"
`)

	registry := NewRegistry()
	loaded, err := LoadPlugins(dir, registry)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if loaded != 0 {
		t.Errorf("expected 0 plugins (bad describe skipped), got %d", loaded)
	}
}

func TestLoadPluginsUsesFilenameWhenNoName(t *testing.T) {
	dir := t.TempDir()
	writeScript(t, dir, "my-tool.sh", `#!/bin/sh
if [ "$1" = "--describe" ]; then
	echo '{"description":"A tool with no name field","args":[]}'
	exit 0
fi
echo "working"
`)

	registry := NewRegistry()
	loaded, err := LoadPlugins(dir, registry)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if loaded != 1 {
		t.Fatalf("expected 1 plugin, got %d", loaded)
	}

	// Should use filename without extension
	_, err = registry.Get("my-tool")
	if err != nil {
		t.Errorf("expected tool 'my-tool' to be registered: %v", err)
	}
}

func TestLoadPluginsScriptFailure(t *testing.T) {
	dir := t.TempDir()
	writeScript(t, dir, "failing.sh", `#!/bin/sh
if [ "$1" = "--describe" ]; then
	echo '{"name":"failing","description":"A failing plugin","args":[]}'
	exit 0
fi
echo "error message" >&2
exit 1
`)

	registry := NewRegistry()
	_, err := LoadPlugins(dir, registry)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	tool, _ := registry.Get("failing")
	_, err = tool.Execute(map[string]string{})
	if err == nil {
		t.Fatal("expected error from failing script")
	}
	if !strings.Contains(err.Error(), "error message") {
		t.Errorf("expected 'error message' in error, got %q", err.Error())
	}
}
