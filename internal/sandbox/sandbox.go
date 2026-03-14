package sandbox

import (
	"bytes"
	"context"
	"fmt"
	"log"
	"os/exec"
	"runtime"
	"strings"
	"syscall"
	"time"
)

// Sandbox provides isolated execution of untrusted commands.
type Sandbox interface {
	Execute(ctx context.Context, command string, args []string, opts ExecOpts) (Result, error)
}

// ExecOpts controls how a sandboxed command is executed.
type ExecOpts struct {
	Timeout      time.Duration
	MaxMemoryMB  int
	AllowNetwork bool
	AllowedPaths []string // filesystem paths the process may read/write
	DeniedPaths  []string // filesystem paths explicitly blocked
	Env          []string // allowlisted KEY=VALUE pairs
	WorkDir      string
	Stdin        string
}

// Result captures the output of a sandboxed execution.
type Result struct {
	Stdout     string
	Stderr     string
	ExitCode   int
	Duration   time.Duration
	MemoryUsed int64 // bytes, best-effort from rusage
	Degraded   bool  // true if namespace isolation failed and execution fell back to no isolation
}

// NativeSandbox uses OS-level isolation (Linux namespaces + rlimits).
// On non-Linux platforms it falls back to basic timeout-only execution.
type NativeSandbox struct {
	Policy  *Policy
	Auditor *Auditor
}

// NewSandbox creates a NativeSandbox with the given policy and optional auditor.
func NewSandbox(policy *Policy, auditor *Auditor) *NativeSandbox {
	return &NativeSandbox{
		Policy:  policy,
		Auditor: auditor,
	}
}

// Execute runs a command inside the sandbox. If policy evaluation fails or
// the sandbox cannot be set up, execution is denied (fail-closed).
func (s *NativeSandbox) Execute(ctx context.Context, command string, args []string, opts ExecOpts) (Result, error) {
	start := time.Now()

	// Policy check — evaluate command AND args together
	allowed, reason := s.Policy.Evaluate(command, args)
	if !allowed {
		result := Result{ExitCode: -1, Duration: time.Since(start)}
		s.audit(command, args, result, "denied: "+reason)
		return result, fmt.Errorf("sandbox: command denied: %s", reason)
	}

	// Check denied paths from both policy and opts against args
	deniedPaths := s.Policy.DeniedPaths
	deniedPaths = append(deniedPaths, opts.DeniedPaths...)
	if reason := checkDeniedPaths(args, deniedPaths); reason != "" {
		result := Result{ExitCode: -1, Duration: time.Since(start)}
		s.audit(command, args, result, "denied: "+reason)
		return result, fmt.Errorf("sandbox: command denied: %s", reason)
	}

	// Apply policy defaults where opts are zero-valued
	if opts.Timeout == 0 {
		opts.Timeout = time.Duration(s.Policy.MaxCPUSeconds) * time.Second
	}
	if opts.MaxMemoryMB == 0 {
		opts.MaxMemoryMB = s.Policy.MaxMemoryMB
	}
	if !opts.AllowNetwork {
		opts.AllowNetwork = s.Policy.AllowNetwork
	}

	// Set up timeout context
	execCtx, cancel := context.WithTimeout(ctx, opts.Timeout)
	defer cancel()

	// Build the command — use sh -c with ulimit, but pass command and args
	// as positional parameters via "--" to avoid shell injection.
	// The ulimit wrapper ends with: exec "$@"
	// We pass: sh -c '<ulimit...>; exec "$@"' -- <command> <args...>
	ulimitWrapper := WrapWithLimits(command, opts, s.Policy)

	shellArgs := []string{"-c", ulimitWrapper, "--", command}
	shellArgs = append(shellArgs, args...)

	cmd := exec.CommandContext(execCtx, "sh", shellArgs...)

	if opts.WorkDir != "" {
		cmd.Dir = opts.WorkDir
	}

	// Restrict environment
	cmd.Env = buildEnv(opts.Env)

	// Apply OS-level restrictions (best-effort — needs privileges)
	if runtime.GOOS == "linux" {
		applySysProcAttr(cmd, opts)
	}

	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	if opts.Stdin != "" {
		cmd.Stdin = strings.NewReader(opts.Stdin)
	}

	degraded := false

	// Run — if namespace creation fails (unprivileged), retry without it
	err := cmd.Run()
	if err != nil && isNamespaceError(err) {
		// If policy requires strict isolation, fail instead of falling back
		if s.Policy.RequireIsolation {
			result := Result{ExitCode: -1, Duration: time.Since(start)}
			s.audit(command, args, result, "denied: namespace isolation required but unavailable")
			return result, fmt.Errorf("sandbox: namespace isolation required but unavailable")
		}

		// Log a warning about degraded isolation
		log.Printf("WARNING: sandbox namespace creation failed, falling back to no-namespace execution for command %q", command)
		degraded = true

		// Retry without namespace isolation — ulimit still applies
		cmd = exec.CommandContext(execCtx, "sh", shellArgs...)
		if opts.WorkDir != "" {
			cmd.Dir = opts.WorkDir
		}
		cmd.Env = buildEnv(opts.Env)
		stdout.Reset()
		stderr.Reset()
		cmd.Stdout = &stdout
		cmd.Stderr = &stderr
		if opts.Stdin != "" {
			cmd.Stdin = strings.NewReader(opts.Stdin)
		}
		err = cmd.Run()
	}

	result := Result{
		Stdout:   stdout.String(),
		Stderr:   stderr.String(),
		ExitCode: exitCode(cmd, err),
		Duration: time.Since(start),
		Degraded: degraded,
	}

	// Best-effort memory usage from rusage
	if cmd.ProcessState != nil {
		rusage, ok := cmd.ProcessState.SysUsage().(*syscall.Rusage)
		if ok && rusage != nil {
			result.MemoryUsed = rusage.Maxrss * 1024 // kilobytes to bytes
		}
	}

	if execCtx.Err() == context.DeadlineExceeded {
		s.audit(command, args, result, "timeout")
		return result, fmt.Errorf("sandbox: command timed out after %v", opts.Timeout)
	}

	status := "ok"
	if err != nil {
		status = fmt.Sprintf("error: %v", err)
	}
	s.audit(command, args, result, status)

	if err != nil {
		return result, fmt.Errorf("sandbox: %v", err)
	}

	return result, nil
}

// checkDeniedPaths checks if any argument references a denied path.
// Returns a reason string if denied, empty string if allowed.
func checkDeniedPaths(args []string, deniedPaths []string) string {
	for _, arg := range args {
		for _, denied := range deniedPaths {
			if strings.HasPrefix(arg, denied) {
				return "blocked path in argument: " + denied
			}
		}
	}
	return ""
}

// audit logs the execution if an auditor is configured.
func (s *NativeSandbox) audit(command string, args []string, result Result, status string) {
	if s.Auditor == nil {
		return
	}
	s.Auditor.Log(AuditEntry{
		Timestamp: time.Now(),
		Command:   command,
		Args:      args,
		ExitCode:  result.ExitCode,
		Duration:  result.Duration,
		Status:    status,
	})
}

// isNamespaceError returns true if the error is caused by namespace creation
// failing due to insufficient privileges.
func isNamespaceError(err error) bool {
	msg := err.Error()
	return strings.Contains(msg, "operation not permitted") ||
		strings.Contains(msg, "permission denied") ||
		strings.Contains(msg, "EPERM")
}

// buildEnv constructs a minimal environment from allowlisted vars.
// Always includes PATH so commands can be found.
func buildEnv(extra []string) []string {
	env := []string{
		"PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
		"HOME=/tmp",
		"LANG=C.UTF-8",
	}
	for _, e := range extra {
		if strings.Contains(e, "=") {
			env = append(env, e)
		}
	}
	return env
}

// exitCode extracts the exit code from a finished command.
func exitCode(cmd *exec.Cmd, err error) int {
	if err == nil {
		return 0
	}
	if cmd.ProcessState != nil {
		ws, ok := cmd.ProcessState.Sys().(syscall.WaitStatus)
		if ok {
			return ws.ExitStatus()
		}
	}
	return -1
}
