package sandbox

import (
	"os/exec"
	"runtime"
	"syscall"
)

// applySysProcAttr configures OS-level process isolation for Linux.
// On non-Linux this is a no-op.
func applySysProcAttr(cmd *exec.Cmd, opts ExecOpts) {
	if runtime.GOOS != "linux" {
		return
	}

	cmd.SysProcAttr = &syscall.SysProcAttr{
		// Create new PID namespace for process isolation.
		// Network namespace is added when networking is blocked.
		Cloneflags: cloneFlags(opts),
		// Kill the child if our process dies
		Pdeathsig: syscall.SIGKILL,
	}
}

// cloneFlags returns the namespace flags to use.
// We always try CLONE_NEWPID for PID isolation. Network isolation
// (CLONE_NEWNET) is added when AllowNetwork is false.
func cloneFlags(opts ExecOpts) uintptr {
	var flags uintptr

	// PID namespace — process only sees itself
	flags |= syscall.CLONE_NEWPID

	// Network namespace — isolate networking unless explicitly allowed
	if !opts.AllowNetwork {
		flags |= syscall.CLONE_NEWNET
	}

	return flags
}

// LimitConfig holds the resource limit parameters for WrapWithLimits.
type LimitConfig struct {
	CPUSeconds   int
	MemoryMB     int
	FileSizeMB   int
	MaxProcesses int
}

// WrapWithLimits prepends ulimit commands to a shell command string
// to enforce resource limits. This works as a defense-in-depth layer
// alongside namespace isolation, and is the primary enforcement
// mechanism for CPU, memory, file size, and process count limits.
//
// The command and args are passed as positional parameters to sh via
// exec "$@", avoiding shell metacharacter injection.
func WrapWithLimits(command string, opts ExecOpts, policy *Policy) string {
	cpuSeconds := int(opts.Timeout.Seconds())
	if cpuSeconds < 1 {
		cpuSeconds = 30
	}

	memKB := opts.MaxMemoryMB * 1024
	if memKB == 0 {
		memKB = 256 * 1024 // 256MB
	}

	fileSizeMB := 64
	if policy != nil && policy.MaxFileSizeMB > 0 {
		fileSizeMB = policy.MaxFileSizeMB
	}
	fileSizeBlocks := fileSizeMB * 2048 // convert MB to 512-byte blocks

	maxProcs := 64
	if policy != nil && policy.MaxProcesses > 0 {
		maxProcs = policy.MaxProcesses
	}

	// ulimit -t = CPU seconds, -v = virtual memory KB,
	// -f = file size (512-byte blocks), -u = max processes
	return "ulimit -t " + itoa(cpuSeconds) +
		" -v " + itoa(memKB) +
		" -f " + itoa(fileSizeBlocks) +
		" -u " + itoa(maxProcs) +
		" 2>/dev/null; exec \"$@\""
}

// itoa is a minimal int-to-string without importing strconv.
func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	neg := false
	if n < 0 {
		neg = true
		n = -n
	}
	var buf [20]byte
	i := len(buf)
	for n > 0 {
		i--
		buf[i] = byte('0' + n%10)
		n /= 10
	}
	if neg {
		i--
		buf[i] = '-'
	}
	return string(buf[i:])
}
