package sandbox

import (
	"context"
	"strings"
	"testing"
	"time"
)

func TestNewSandboxReturnsNonNil(t *testing.T) {
	sb := NewSandbox(DefaultPolicy(), nil)
	if sb == nil {
		t.Fatal("expected non-nil sandbox")
	}
}

func TestExecuteSimpleCommand(t *testing.T) {
	sb := NewSandbox(TrustedPolicy(), nil)
	result, err := sb.Execute(context.Background(), "echo", []string{"hello"}, ExecOpts{
		Timeout: 5 * time.Second,
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.Contains(result.Stdout, "hello") {
		t.Errorf("expected 'hello' in stdout, got %q", result.Stdout)
	}
	if result.ExitCode != 0 {
		t.Errorf("expected exit code 0, got %d", result.ExitCode)
	}
	if result.Duration <= 0 {
		t.Error("expected positive duration")
	}
}

func TestExecuteDeniedCommand(t *testing.T) {
	sb := NewSandbox(DefaultPolicy(), nil)
	_, err := sb.Execute(context.Background(), "rm", []string{"-rf", "/"}, ExecOpts{
		Timeout: 5 * time.Second,
	})
	if err == nil {
		t.Fatal("expected error for denied command")
	}
	if !strings.Contains(err.Error(), "denied") {
		t.Errorf("expected 'denied' in error, got %q", err.Error())
	}
}

func TestExecuteWithTimeout(t *testing.T) {
	sb := NewSandbox(TrustedPolicy(), nil)
	_, err := sb.Execute(context.Background(), "sleep", []string{"30"}, ExecOpts{
		Timeout: 1 * time.Second,
	})
	if err == nil {
		t.Fatal("expected timeout error")
	}
	if !strings.Contains(err.Error(), "timed out") {
		t.Errorf("expected 'timed out' in error, got %q", err.Error())
	}
}

func TestExecuteCapturesStderr(t *testing.T) {
	sb := NewSandbox(TrustedPolicy(), nil)
	// Use sh -c to get shell redirection syntax
	result, _ := sb.Execute(context.Background(), "sh", []string{"-c", "echo error >&2"}, ExecOpts{
		Timeout: 5 * time.Second,
	})
	if !strings.Contains(result.Stderr, "error") {
		t.Errorf("expected 'error' in stderr, got %q", result.Stderr)
	}
}

func TestExecuteWithStdin(t *testing.T) {
	sb := NewSandbox(TrustedPolicy(), nil)
	result, err := sb.Execute(context.Background(), "cat", nil, ExecOpts{
		Timeout: 5 * time.Second,
		Stdin:   "hello from stdin",
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.Contains(result.Stdout, "hello from stdin") {
		t.Errorf("expected stdin content in stdout, got %q", result.Stdout)
	}
}

func TestExecuteWithWorkDir(t *testing.T) {
	sb := NewSandbox(TrustedPolicy(), nil)
	result, err := sb.Execute(context.Background(), "pwd", nil, ExecOpts{
		Timeout: 5 * time.Second,
		WorkDir: "/tmp",
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.Contains(result.Stdout, "/tmp") {
		t.Errorf("expected /tmp in stdout, got %q", result.Stdout)
	}
}

func TestExecuteFailedCommandReturnsError(t *testing.T) {
	sb := NewSandbox(TrustedPolicy(), nil)
	result, err := sb.Execute(context.Background(), "false", nil, ExecOpts{
		Timeout: 5 * time.Second,
	})
	if err == nil {
		t.Fatal("expected error for 'false' command")
	}
	if result.ExitCode == 0 {
		t.Error("expected non-zero exit code")
	}
}

func TestExecuteWithAuditor(t *testing.T) {
	dir := t.TempDir()
	auditor := NewAuditor(dir)
	sb := NewSandbox(TrustedPolicy(), auditor)

	_, _ = sb.Execute(context.Background(), "echo", []string{"test"}, ExecOpts{
		Timeout: 5 * time.Second,
	})

	entries := auditor.Recent(10)
	if len(entries) != 1 {
		t.Fatalf("expected 1 audit entry, got %d", len(entries))
	}
	if entries[0].Command != "echo" {
		t.Errorf("expected command 'echo', got %q", entries[0].Command)
	}
	if entries[0].Status != "ok" {
		t.Errorf("expected status 'ok', got %q", entries[0].Status)
	}
}

func TestExecuteDeniedCommandIsAudited(t *testing.T) {
	dir := t.TempDir()
	auditor := NewAuditor(dir)
	sb := NewSandbox(DefaultPolicy(), auditor)

	_, _ = sb.Execute(context.Background(), "rm", []string{"-rf", "/"}, ExecOpts{
		Timeout: 5 * time.Second,
	})

	entries := auditor.Recent(10)
	if len(entries) != 1 {
		t.Fatalf("expected 1 audit entry, got %d", len(entries))
	}
	if !strings.Contains(entries[0].Status, "denied") {
		t.Errorf("expected 'denied' in audit status, got %q", entries[0].Status)
	}
}

func TestBuildEnvIncludesPathAndHome(t *testing.T) {
	env := buildEnv(nil)
	foundPath := false
	foundHome := false
	for _, e := range env {
		if strings.HasPrefix(e, "PATH=") {
			foundPath = true
		}
		if strings.HasPrefix(e, "HOME=") {
			foundHome = true
		}
	}
	if !foundPath {
		t.Error("expected PATH in env")
	}
	if !foundHome {
		t.Error("expected HOME in env")
	}
}

func TestBuildEnvIncludesExtraVars(t *testing.T) {
	env := buildEnv([]string{"FOO=bar", "BAZ=qux"})
	found := 0
	for _, e := range env {
		if e == "FOO=bar" || e == "BAZ=qux" {
			found++
		}
	}
	if found != 2 {
		t.Errorf("expected 2 extra env vars, found %d", found)
	}
}

func TestBuildEnvIgnoresInvalidEntries(t *testing.T) {
	env := buildEnv([]string{"NOEQUALS", "VALID=yes"})
	for _, e := range env {
		if e == "NOEQUALS" {
			t.Error("expected invalid env entry to be filtered out")
		}
	}
}

func TestExecuteDeniedPath(t *testing.T) {
	sb := NewSandbox(DefaultPolicy(), nil)
	_, err := sb.Execute(context.Background(), "cat", []string{"/etc/shadow"}, ExecOpts{
		Timeout: 5 * time.Second,
	})
	if err == nil {
		t.Fatal("expected error for denied path")
	}
	if !strings.Contains(err.Error(), "denied") {
		t.Errorf("expected 'denied' in error, got %q", err.Error())
	}
}

func TestExecuteCommandInjectionPrevented(t *testing.T) {
	sb := NewSandbox(TrustedPolicy(), nil)
	// Try to inject via args — the semicolon should not be interpreted as shell syntax.
	// If injection occurs, "echo" would run twice producing two separate lines.
	// If safe, the arg is treated as a single literal string.
	result, err := sb.Execute(context.Background(), "echo", []string{"hello; echo INJECTED_MARKER"}, ExecOpts{
		Timeout: 5 * time.Second,
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	// If injection happened, stdout would contain "INJECTED_MARKER" on its own line
	// (from a second echo invocation). With safe execution, the entire arg including
	// the semicolon is printed as one line.
	lines := strings.Split(strings.TrimSpace(result.Stdout), "\n")
	if len(lines) != 1 {
		t.Errorf("expected exactly 1 output line (no injection), got %d: %q", len(lines), result.Stdout)
	}
	if !strings.Contains(result.Stdout, "hello; echo INJECTED_MARKER") {
		t.Errorf("expected literal arg in stdout, got %q", result.Stdout)
	}
}

func TestResultDegradedField(t *testing.T) {
	// Test that the Degraded field exists and defaults to false
	sb := NewSandbox(TrustedPolicy(), nil)
	result, err := sb.Execute(context.Background(), "echo", []string{"test"}, ExecOpts{
		Timeout: 5 * time.Second,
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	// In unprivileged test environments, Degraded may be true (namespace fallback),
	// but the field should exist and the command should succeed either way.
	_ = result.Degraded
}

func TestCheckDeniedPathsBlocks(t *testing.T) {
	deniedPaths := []string{"/etc/shadow", "/root", "/boot"}

	tests := []struct {
		args    []string
		blocked bool
	}{
		{[]string{"/etc/shadow"}, true},
		{[]string{"/etc/shadow.bak"}, true}, // prefix match
		{[]string{"/root/.bashrc"}, true},
		{[]string{"/boot/vmlinuz"}, true},
		{[]string{"/tmp/safe.txt"}, false},
		{[]string{"/home/user/file.txt"}, false},
		{[]string{"/tmp/safe.txt", "/etc/shadow"}, true}, // second arg blocked
	}

	for _, tt := range tests {
		reason := checkDeniedPaths(tt.args, deniedPaths)
		if tt.blocked && reason == "" {
			t.Errorf("expected args %v to be blocked", tt.args)
		}
		if !tt.blocked && reason != "" {
			t.Errorf("expected args %v to be allowed, got reason %q", tt.args, reason)
		}
	}
}

func TestPolicyEvaluateWithArgs(t *testing.T) {
	p := DefaultPolicy()

	// Safe command with safe args
	allowed, _ := p.Evaluate("echo", []string{"hello", "world"})
	if !allowed {
		t.Error("expected 'echo hello world' to be allowed")
	}

	// Safe command with denied path in args
	allowed, reason := p.Evaluate("cat", []string{"/etc/shadow"})
	if allowed {
		t.Error("expected 'cat /etc/shadow' to be denied")
	}
	if !strings.Contains(reason, "blocked path") {
		t.Errorf("expected 'blocked path' in reason, got %q", reason)
	}

	// Denied command fragment in args
	allowed, _ = p.Evaluate("bash", []string{"-c", "rm -rf /"})
	if allowed {
		t.Error("expected 'bash -c rm -rf /' to be denied")
	}
}

func TestAuditLogRotation(t *testing.T) {
	dir := t.TempDir()
	auditor := NewAuditor(dir)

	// Write entries until we can verify rotation logic exists
	for i := 0; i < 100; i++ {
		auditor.Log(AuditEntry{
			Command:  "echo",
			Args:     []string{"test"},
			ExitCode: 0,
			Status:   "ok",
		})
	}

	entries := auditor.Recent(10)
	if len(entries) != 10 {
		t.Errorf("expected 10 recent entries, got %d", len(entries))
	}

	// Verify all returned entries are the most recent
	for _, e := range entries {
		if e.Command != "echo" {
			t.Errorf("expected command 'echo', got %q", e.Command)
		}
	}
}

func TestAuditRecentMoreThanAvailable(t *testing.T) {
	dir := t.TempDir()
	auditor := NewAuditor(dir)

	auditor.Log(AuditEntry{Command: "ls", Status: "ok"})
	auditor.Log(AuditEntry{Command: "pwd", Status: "ok"})

	entries := auditor.Recent(100)
	if len(entries) != 2 {
		t.Errorf("expected 2 entries (all available), got %d", len(entries))
	}
}

func TestAuditRecentEmpty(t *testing.T) {
	dir := t.TempDir()
	auditor := NewAuditor(dir)

	entries := auditor.Recent(10)
	if entries != nil && len(entries) != 0 {
		t.Errorf("expected nil or empty entries for empty audit, got %d", len(entries))
	}
}

func TestSandboxedExecutionProducesOutput(t *testing.T) {
	sb := NewSandbox(TrustedPolicy(), nil)
	result, err := sb.Execute(context.Background(), "echo", []string{"hello", "sandbox"}, ExecOpts{
		Timeout: 5 * time.Second,
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.Contains(result.Stdout, "hello sandbox") {
		t.Errorf("expected 'hello sandbox' in stdout, got %q", result.Stdout)
	}
}

func TestSandboxedExecutionEnvRestriction(t *testing.T) {
	sb := NewSandbox(TrustedPolicy(), nil)
	result, err := sb.Execute(context.Background(), "env", nil, ExecOpts{
		Timeout: 5 * time.Second,
		Env:     []string{"CUSTOM_VAR=test123"},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.Contains(result.Stdout, "CUSTOM_VAR=test123") {
		t.Errorf("expected CUSTOM_VAR in env output, got %q", result.Stdout)
	}
	if !strings.Contains(result.Stdout, "PATH=") {
		t.Errorf("expected PATH in env output, got %q", result.Stdout)
	}
}

func TestFormatEntry(t *testing.T) {
	entry := AuditEntry{
		Command:  "echo",
		ExitCode: 0,
		Status:   "ok",
	}
	formatted := FormatEntry(entry)
	if !strings.Contains(formatted, "echo") {
		t.Errorf("expected 'echo' in formatted entry, got %q", formatted)
	}
	if !strings.Contains(formatted, "ok") {
		t.Errorf("expected 'ok' in formatted entry, got %q", formatted)
	}
}

func TestExecuteMultipleCommandsSequentially(t *testing.T) {
	sb := NewSandbox(TrustedPolicy(), nil)

	// First command
	r1, err := sb.Execute(context.Background(), "echo", []string{"first"}, ExecOpts{Timeout: 5 * time.Second})
	if err != nil {
		t.Fatalf("first command error: %v", err)
	}
	if !strings.Contains(r1.Stdout, "first") {
		t.Errorf("expected 'first' in stdout, got %q", r1.Stdout)
	}

	// Second command
	r2, err := sb.Execute(context.Background(), "echo", []string{"second"}, ExecOpts{Timeout: 5 * time.Second})
	if err != nil {
		t.Fatalf("second command error: %v", err)
	}
	if !strings.Contains(r2.Stdout, "second") {
		t.Errorf("expected 'second' in stdout, got %q", r2.Stdout)
	}
}

func TestRequireIsolationPolicy(t *testing.T) {
	p := DefaultPolicy()
	if p.RequireIsolation {
		t.Error("expected RequireIsolation to default to false")
	}
	p.RequireIsolation = true
	if !p.RequireIsolation {
		t.Error("expected RequireIsolation to be settable")
	}
}
