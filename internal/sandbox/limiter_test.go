package sandbox

import (
	"strings"
	"testing"
	"time"
)

func TestWrapWithLimitsIncludesUlimitCommands(t *testing.T) {
	wrapped := WrapWithLimits("echo hello", ExecOpts{
		Timeout:     30 * time.Second,
		MaxMemoryMB: 256,
	}, nil)

	if !strings.Contains(wrapped, "ulimit") {
		t.Error("expected 'ulimit' in wrapped command")
	}
	if !strings.Contains(wrapped, "exec \"$@\"") {
		t.Error("expected 'exec \"$@\"' in wrapped command for safe arg passing")
	}
	if !strings.Contains(wrapped, "-t 30") {
		t.Error("expected CPU time limit in wrapped command")
	}
	if !strings.Contains(wrapped, "-v 262144") {
		t.Errorf("expected memory limit (256*1024=262144) in wrapped command, got %q", wrapped)
	}
	if !strings.Contains(wrapped, "-u 64") {
		t.Error("expected process limit in wrapped command")
	}
}

func TestWrapWithLimitsDefaultsOnZeroValues(t *testing.T) {
	wrapped := WrapWithLimits("echo hello", ExecOpts{}, nil)

	if !strings.Contains(wrapped, "-t 30") {
		t.Error("expected default 30s CPU limit")
	}
	if !strings.Contains(wrapped, "-v 262144") {
		t.Error("expected default 256MB memory limit")
	}
}

func TestWrapWithLimitsCustomValues(t *testing.T) {
	wrapped := WrapWithLimits("python train.py", ExecOpts{
		Timeout:     120 * time.Second,
		MaxMemoryMB: 1024,
	}, nil)

	if !strings.Contains(wrapped, "-t 120") {
		t.Errorf("expected CPU limit 120, got %q", wrapped)
	}
	if !strings.Contains(wrapped, "-v 1048576") {
		t.Errorf("expected memory limit 1048576 KB, got %q", wrapped)
	}
}

func TestWrapWithLimitsUsesPolicyValues(t *testing.T) {
	policy := &Policy{
		MaxFileSizeMB: 128,
		MaxProcesses:  32,
	}

	wrapped := WrapWithLimits("echo hello", ExecOpts{
		Timeout:     30 * time.Second,
		MaxMemoryMB: 256,
	}, policy)

	// 128MB = 128 * 2048 = 262144 blocks
	if !strings.Contains(wrapped, "-f 262144") {
		t.Errorf("expected file size limit from policy (128MB = 262144 blocks), got %q", wrapped)
	}
	if !strings.Contains(wrapped, "-u 32") {
		t.Errorf("expected process limit from policy (32), got %q", wrapped)
	}
}

func TestWrapWithLimitsDefaultFileSizeAndProcs(t *testing.T) {
	// When policy is nil, should use defaults (64MB, 64 procs)
	wrapped := WrapWithLimits("echo hello", ExecOpts{
		Timeout:     30 * time.Second,
		MaxMemoryMB: 256,
	}, nil)

	// 64MB = 64 * 2048 = 131072 blocks
	if !strings.Contains(wrapped, "-f 131072") {
		t.Errorf("expected default file size limit (64MB = 131072 blocks), got %q", wrapped)
	}
	if !strings.Contains(wrapped, "-u 64") {
		t.Errorf("expected default process limit (64), got %q", wrapped)
	}
}

func TestItoaPositive(t *testing.T) {
	tests := []struct {
		input int
		want  string
	}{
		{0, "0"},
		{1, "1"},
		{42, "42"},
		{12345, "12345"},
		{262144, "262144"},
	}
	for _, tt := range tests {
		got := itoa(tt.input)
		if got != tt.want {
			t.Errorf("itoa(%d) = %q, want %q", tt.input, got, tt.want)
		}
	}
}

func TestItoaNegative(t *testing.T) {
	got := itoa(-5)
	if got != "-5" {
		t.Errorf("itoa(-5) = %q, want \"-5\"", got)
	}
}

func TestCloneFlagsWithoutNetwork(t *testing.T) {
	flags := cloneFlags(ExecOpts{AllowNetwork: false})
	if flags == 0 {
		t.Error("expected non-zero clone flags")
	}
}

func TestCloneFlagsWithNetwork(t *testing.T) {
	flagsNo := cloneFlags(ExecOpts{AllowNetwork: false})
	flagsYes := cloneFlags(ExecOpts{AllowNetwork: true})

	if flagsYes >= flagsNo {
		t.Error("expected fewer flags when network is allowed")
	}
}
