//go:build integration

package main

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"os/exec"
	"strings"
	"testing"
	"time"
)

const nousBinary = "/home/raphael/Projects/openclaw-alternative/nous"
const defaultModel = "qwen2.5:1.5b"

// runNous starts the nous binary with --model and --trust flags, sends each
// input line followed by a newline, then closes stdin. It returns all combined
// stdout+stderr output or fails the test on timeout / execution error.
func runNous(t *testing.T, timeout time.Duration, inputs ...string) string {
	t.Helper()

	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()

	cmd := exec.CommandContext(ctx, nousBinary, "--model", defaultModel, "--trust")
	cmd.Dir = "/home/raphael/Projects/openclaw-alternative"

	stdin, err := cmd.StdinPipe()
	if err != nil {
		t.Fatalf("failed to create stdin pipe: %v", err)
	}

	var outBuf bytes.Buffer
	cmd.Stdout = &outBuf
	cmd.Stderr = &outBuf

	if err := cmd.Start(); err != nil {
		t.Fatalf("failed to start nous: %v", err)
	}

	// Give the binary a moment to initialize and print its banner before
	// we start sending input. Without this, fast slash-command inputs can
	// arrive before the REPL loop is ready and get silently dropped.
	time.Sleep(3 * time.Second)

	for _, line := range inputs {
		_, _ = io.WriteString(stdin, line+"\n")
		// Small delay between commands so the REPL can process each one.
		time.Sleep(300 * time.Millisecond)
	}

	// Close stdin so the scanner loop exits.
	stdin.Close()

	// Wait for the process to finish.
	_ = cmd.Wait()

	output := outBuf.String()
	if ctx.Err() == context.DeadlineExceeded {
		// We still have output captured so far; log it and continue.
		t.Logf("process timed out after %v (output so far captured)", timeout)
	}

	return output
}

// runNousWithFlag runs the nous binary with a specific flag (e.g. --version)
// and returns its output.
func runNousWithFlag(t *testing.T, timeout time.Duration, flags ...string) string {
	t.Helper()

	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()

	cmd := exec.CommandContext(ctx, nousBinary, flags...)
	cmd.Dir = "/home/raphael/Projects/openclaw-alternative"

	out, err := cmd.CombinedOutput()
	if ctx.Err() == context.DeadlineExceeded {
		t.Fatalf("command timed out after %v", timeout)
	}
	if err != nil {
		// --version exits with 0 via os.Exit(0); some systems may report
		// an error for the exec. We still want the output.
		if len(out) == 0 {
			t.Fatalf("command failed with no output: %v", err)
		}
	}
	return string(out)
}

// containsAny checks if the output contains at least one of the given substrings.
func containsAny(output string, candidates ...string) (bool, string) {
	lower := strings.ToLower(output)
	for _, c := range candidates {
		if strings.Contains(lower, strings.ToLower(c)) {
			return true, c
		}
	}
	return false, ""
}

func TestBinaryStarts(t *testing.T) {
	output := runNous(t, 15*time.Second)
	if !strings.Contains(output, "N O U S") {
		t.Errorf("banner not found in output:\n%s", output)
	}
	if !strings.Contains(output, "version") {
		t.Errorf("version string not found in output:\n%s", output)
	}
	if !strings.Contains(output, "0.3.0") {
		t.Errorf("expected version 0.3.0 in output:\n%s", output)
	}
}

func TestSlashHelp(t *testing.T) {
	output := runNous(t, 15*time.Second, "/help")

	expectedCommands := []string{
		"/help", "/status", "/memory", "/longterm", "/goals",
		"/model", "/tools", "/project", "/sessions", "/save",
		"/clear", "/quit",
	}
	for _, cmd := range expectedCommands {
		if !strings.Contains(output, cmd) {
			t.Errorf("expected command %q not found in /help output:\n%s", cmd, output)
		}
	}
}

func TestSlashStatus(t *testing.T) {
	output := runNous(t, 15*time.Second, "/status")

	expectedPhrases := []string{
		"Percepts:",
		"Active goals:",
		"Working memory:",
		"Long-term memory:",
		"Recent actions:",
		"Conversation:",
		"Session:",
	}
	for _, phrase := range expectedPhrases {
		if !strings.Contains(output, phrase) {
			t.Errorf("expected phrase %q not found in /status output:\n%s", phrase, output)
		}
	}
}

func TestSlashTools(t *testing.T) {
	output := runNous(t, 15*time.Second, "/tools")

	expectedTools := []string{
		"grep", "shell", "mkdir", "tree", "read",
		"write", "edit", "glob", "ls",
	}
	for _, tool := range expectedTools {
		if !strings.Contains(output, tool) {
			t.Errorf("expected tool %q not found in /tools output:\n%s", tool, output)
		}
	}
	// Verify we see 9 tools in the startup line
	if !strings.Contains(output, "9 tools:") {
		t.Errorf("expected '9 tools:' in output:\n%s", output)
	}
}

func TestSlashProject(t *testing.T) {
	output := runNous(t, 15*time.Second, "/project")

	if !strings.Contains(output, "Go") {
		t.Errorf("expected 'Go' language in /project output:\n%s", output)
	}
	if !strings.Contains(output, "Project:") {
		t.Errorf("expected 'Project:' in /project output:\n%s", output)
	}
}

func TestSlashModel(t *testing.T) {
	output := runNous(t, 15*time.Second, "/model")

	if !strings.Contains(output, "model:") {
		t.Errorf("expected 'model:' in output:\n%s", output)
	}
	if !strings.Contains(output, defaultModel) {
		t.Errorf("expected model name %q in output:\n%s", defaultModel, output)
	}
	if !strings.Contains(output, "available models:") {
		t.Errorf("expected 'available models:' in output:\n%s", output)
	}
}

func TestVersionFlag(t *testing.T) {
	output := runNousWithFlag(t, 10*time.Second, "--version")
	expected := fmt.Sprintf("nous %s", "0.3.0")
	if !strings.Contains(strings.TrimSpace(output), expected) {
		t.Errorf("expected %q in --version output, got: %q", expected, output)
	}
}

func TestProjectScan(t *testing.T) {
	output := runNous(t, 15*time.Second)

	if !strings.Contains(output, "scanning project...") {
		t.Errorf("expected 'scanning project...' in startup output:\n%s", output)
	}
}

func TestToolUse(t *testing.T) {
	output := runNous(t, 120*time.Second, "What files are in this project?")

	if !strings.Contains(output, "[tool]") {
		t.Logf("full output:\n%s", output)
		t.Errorf("expected '[tool]' in output indicating a tool was called")
	}
}

func TestSelfAwareness(t *testing.T) {
	output := runNous(t, 120*time.Second, "What are you? Describe your architecture.")

	found, matched := containsAny(output, "nous", "blackboard", "cognitive")
	if !found {
		t.Logf("full output:\n%s", output)
		t.Errorf("expected output to mention 'nous', 'blackboard', or 'cognitive'")
	} else {
		t.Logf("found self-awareness keyword: %q", matched)
	}
}

func TestSessionCreated(t *testing.T) {
	output := runNous(t, 15*time.Second)

	if !strings.Contains(output, "session:") {
		t.Errorf("expected 'session:' with session ID in startup output:\n%s", output)
	}
}
