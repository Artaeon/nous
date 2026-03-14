package sandbox

import (
	"strings"
	"testing"
)

func TestDefaultPolicyDeniesDestructiveCommands(t *testing.T) {
	p := DefaultPolicy()

	denied := []struct {
		cmd  string
		args []string
	}{
		{"rm", []string{"-rf", "/"}},
		{"rm -rf /", nil},
		{"rm", []string{"-rf", "--no-preserve-root", "/"}},
		{"mkfs.ext4", []string{"/dev/sda1"}},
		{"shutdown", []string{"-h", "now"}},
		{"reboot", nil},
		{"poweroff", nil},
		{"halt", nil},
		{":(){:|:&};:", nil},
		{"dd", []string{"if=/dev/zero", "of=/dev/sda"}},
		{"echo", []string{"bad", ">", "/dev/sda"}},
		{"chmod", []string{"-R", "777", "/etc"}},
		{"chown", []string{"-R", "nobody", "/etc"}},
		{"curl", []string{"http://evil.com"}},
		{"wget", []string{"http://evil.com"}},
		{"nc", []string{" -l", "8080"}},
		{"ssh", []string{" root@host"}},
		{"scp", []string{" file", "root@host:/tmp"}},
	}

	for _, tt := range denied {
		allowed, reason := p.Evaluate(tt.cmd, tt.args)
		full := tt.cmd
		if len(tt.args) > 0 {
			full = tt.cmd + " " + strings.Join(tt.args, " ")
		}
		if allowed {
			t.Errorf("expected %q to be denied, but was allowed", full)
		}
		if reason == "" {
			t.Errorf("expected denial reason for %q", full)
		}
	}
}

func TestDefaultPolicyAllowsSafeCommands(t *testing.T) {
	p := DefaultPolicy()

	safe := []struct {
		cmd  string
		args []string
	}{
		{"echo", []string{"hello"}},
		{"ls", []string{"-la"}},
		{"cat", []string{"README.md"}},
		{"go", []string{"build", "./..."}},
		{"go", []string{"test", "./..."}},
		{"python", []string{"script.py"}},
		{"git", []string{"status"}},
		{"grep", []string{"-r", "pattern", "."}},
		{"find", []string{".", "-name", "*.go"}},
	}

	for _, tt := range safe {
		allowed, reason := p.Evaluate(tt.cmd, tt.args)
		if !allowed {
			full := tt.cmd + " " + strings.Join(tt.args, " ")
			t.Errorf("expected %q to be allowed, but was denied: %s", full, reason)
		}
	}
}

func TestTrustedPolicyAllowsNetwork(t *testing.T) {
	p := TrustedPolicy()

	// Trusted mode allows curl/wget
	allowed, _ := p.Evaluate("curl", []string{"http://example.com"})
	if !allowed {
		t.Error("expected curl to be allowed in trusted mode")
	}

	allowed, _ = p.Evaluate("wget", []string{"http://example.com"})
	if !allowed {
		t.Error("expected wget to be allowed in trusted mode")
	}
}

func TestTrustedPolicyStillBlocksDestructive(t *testing.T) {
	p := TrustedPolicy()

	denied := []struct {
		cmd  string
		args []string
	}{
		{"rm", []string{"-rf", "/"}},
		{"mkfs.ext4", []string{"/dev/sda"}},
		{"shutdown", []string{"now"}},
		{":(){:|:&};:", nil},
	}

	for _, tt := range denied {
		allowed, _ := p.Evaluate(tt.cmd, tt.args)
		full := tt.cmd
		if len(tt.args) > 0 {
			full = tt.cmd + " " + strings.Join(tt.args, " ")
		}
		if allowed {
			t.Errorf("expected %q to be denied even in trusted mode", full)
		}
	}
}

func TestHandPolicyRestrictsToWhitelist(t *testing.T) {
	p := HandPolicy([]string{"echo", "cat", "ls"})

	allowed, _ := p.Evaluate("echo", []string{"hello"})
	if !allowed {
		t.Error("expected 'echo' to be allowed in hand policy")
	}

	allowed, _ = p.Evaluate("cat", []string{"file.txt"})
	if !allowed {
		t.Error("expected 'cat' to be allowed in hand policy")
	}

	allowed, reason := p.Evaluate("python", []string{"script.py"})
	if allowed {
		t.Error("expected 'python' to be denied in hand policy with whitelist")
	}
	if !strings.Contains(reason, "allowlist") {
		t.Errorf("expected 'allowlist' in reason, got %q", reason)
	}
}

func TestHandPolicyInheritsDefaultDenials(t *testing.T) {
	// Even if "rm" is whitelisted, destructive fragments are still blocked
	p := HandPolicy([]string{"rm", "echo"})

	allowed, _ := p.Evaluate("rm", []string{"-rf", "/"})
	if allowed {
		t.Error("expected 'rm -rf /' to be denied even when 'rm' is whitelisted")
	}
}

func TestPolicyEvaluateIsCaseInsensitive(t *testing.T) {
	p := DefaultPolicy()

	allowed, _ := p.Evaluate("SHUTDOWN", []string{"-h", "now"})
	if allowed {
		t.Error("expected case-insensitive denial of SHUTDOWN")
	}
}

func TestDefaultPolicyLimits(t *testing.T) {
	p := DefaultPolicy()

	if p.MaxCPUSeconds != 30 {
		t.Errorf("expected 30s CPU limit, got %d", p.MaxCPUSeconds)
	}
	if p.MaxMemoryMB != 256 {
		t.Errorf("expected 256MB memory limit, got %d", p.MaxMemoryMB)
	}
	if p.AllowNetwork {
		t.Error("expected network to be disabled in default policy")
	}
}

func TestTrustedPolicyLimits(t *testing.T) {
	p := TrustedPolicy()

	if p.MaxCPUSeconds != 300 {
		t.Errorf("expected 300s CPU limit, got %d", p.MaxCPUSeconds)
	}
	if p.MaxMemoryMB != 2048 {
		t.Errorf("expected 2048MB memory limit, got %d", p.MaxMemoryMB)
	}
	if !p.AllowNetwork {
		t.Error("expected network to be enabled in trusted policy")
	}
}

func TestEvaluateChecksDeniedPaths(t *testing.T) {
	p := DefaultPolicy()

	allowed, reason := p.Evaluate("cat", []string{"/etc/shadow"})
	if allowed {
		t.Error("expected denied path /etc/shadow to be blocked")
	}
	if !strings.Contains(reason, "blocked path") {
		t.Errorf("expected 'blocked path' in reason, got %q", reason)
	}

	allowed, reason = p.Evaluate("cat", []string{"/etc/shadow.bak"})
	if allowed {
		t.Error("expected denied path prefix /etc/shadow to block /etc/shadow.bak")
	}

	// Safe path should be allowed
	allowed, _ = p.Evaluate("cat", []string{"/tmp/test.txt"})
	if !allowed {
		t.Error("expected /tmp/test.txt to be allowed")
	}
}

func TestEvaluateChecksArgsForDeniedFragments(t *testing.T) {
	p := DefaultPolicy()

	// Trying to sneak a denied command fragment via args
	allowed, _ := p.Evaluate("echo", []string{"; rm -rf /"})
	if allowed {
		t.Error("expected denied fragment in args to be blocked")
	}
}
