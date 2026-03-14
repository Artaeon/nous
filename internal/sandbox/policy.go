package sandbox

import (
	"path/filepath"
	"strings"
)

// Policy defines what a sandboxed process is allowed to do.
type Policy struct {
	AllowedCommands  []string // if non-empty, only these command prefixes are allowed
	DeniedCommands   []string // these command fragments are always blocked
	AllowedPaths     []string // filesystem paths the process may access
	DeniedPaths      []string // filesystem paths explicitly blocked
	MaxCPUSeconds    int      // CPU time limit
	MaxMemoryMB      int      // address space limit
	AllowNetwork     bool     // whether outbound networking is permitted
	MaxFileSizeMB    int      // max file size a process can create
	MaxProcesses     int      // max child processes
	RequireIsolation bool     // if true, fail instead of falling back when namespace creation fails
}

// DefaultPolicy returns a restrictive policy suitable for untrusted commands.
// No network access, limited FS, 30s CPU, 256MB memory.
func DefaultPolicy() *Policy {
	return &Policy{
		DeniedCommands: defaultDeniedCommands(),
		DeniedPaths:    defaultDeniedPaths(),
		MaxCPUSeconds:  30,
		MaxMemoryMB:    256,
		AllowNetwork:   false,
		MaxFileSizeMB:  64,
		MaxProcesses:   64,
	}
}

// TrustedPolicy returns a permissive policy for --trust mode.
// Network allowed, generous limits, but still blocks destructive commands.
func TrustedPolicy() *Policy {
	return &Policy{
		DeniedCommands: criticalDeniedCommands(),
		DeniedPaths:    criticalDeniedPaths(),
		MaxCPUSeconds:  300,
		MaxMemoryMB:    2048,
		AllowNetwork:   true,
		MaxFileSizeMB:  512,
		MaxProcesses:   256,
	}
}

// HandPolicy returns a policy for autonomous hand/planner execution.
// Allows specific tool commands but blocks shell escapes and destructive ops.
func HandPolicy(toolWhitelist []string) *Policy {
	p := DefaultPolicy()
	p.AllowedCommands = toolWhitelist
	p.MaxCPUSeconds = 60
	return p
}

// Evaluate checks whether a command with its arguments is allowed by this policy.
// Returns (allowed, reason).
func (p *Policy) Evaluate(command string, args []string) (bool, string) {
	lower := strings.ToLower(strings.TrimSpace(command))

	// Reconstruct the full command string for pattern matching
	fullCmd := lower
	if len(args) > 0 {
		fullCmd = lower + " " + strings.ToLower(strings.Join(args, " "))
	}

	// Check denied commands against both the command alone and the full command+args
	for _, denied := range p.DeniedCommands {
		deniedLower := strings.ToLower(denied)
		if strings.Contains(fullCmd, deniedLower) {
			return false, "blocked command fragment: " + denied
		}
	}

	// Also check each individual arg for denied command fragments
	for _, arg := range args {
		argLower := strings.ToLower(arg)
		for _, denied := range p.DeniedCommands {
			deniedLower := strings.ToLower(denied)
			if strings.Contains(argLower, deniedLower) {
				return false, "blocked command fragment in argument: " + denied
			}
		}
	}

	// Check denied paths against arguments
	for _, arg := range args {
		for _, deniedPath := range p.DeniedPaths {
			// Check if arg matches the denied path via prefix or glob
			if strings.HasPrefix(arg, deniedPath) {
				return false, "blocked path: " + deniedPath
			}
			if matched, _ := filepath.Match(deniedPath, arg); matched {
				return false, "blocked path: " + deniedPath
			}
		}
	}

	// Check allowed commands (if whitelist is set)
	if len(p.AllowedCommands) > 0 {
		found := false
		for _, allowed := range p.AllowedCommands {
			if strings.HasPrefix(lower, strings.ToLower(allowed)) {
				found = true
				break
			}
		}
		if !found {
			return false, "command not in allowlist"
		}
	}

	return true, ""
}

// defaultDeniedCommands returns command fragments that are always blocked
// in the default restrictive policy.
func defaultDeniedCommands() []string {
	return []string{
		"rm -rf /",
		"rm -rf --no-preserve-root",
		"mkfs",
		"shutdown",
		"reboot",
		"poweroff",
		"halt",
		":(){",       // fork bomb
		"dd if=",
		"> /dev/sd",
		"chmod -r 777",
		"chown -r",
		"curl",       // no network in default
		"wget",
		"nc ",
		"ncat",
		"socat",
		"ssh ",
		"scp ",
		"rsync",
		"nmap",
	}
}

// criticalDeniedCommands returns only the most dangerous commands,
// used by TrustedPolicy which allows network access.
func criticalDeniedCommands() []string {
	return []string{
		"rm -rf /",
		"rm -rf --no-preserve-root",
		"mkfs",
		"shutdown",
		"reboot",
		"poweroff",
		"halt",
		":(){",
		"dd if=/dev/zero",
		"dd if=/dev/urandom",
		"> /dev/sd",
		"chmod -r 777 /",
		"chown -r /",
	}
}

// defaultDeniedPaths returns filesystem paths blocked in the default policy.
func defaultDeniedPaths() []string {
	return []string{
		"/etc/shadow",
		"/etc/passwd",
		"/etc/sudoers",
		"/root",
		"/boot",
		"/dev",
		"/proc/sys",
		"/sys",
	}
}

// criticalDeniedPaths returns only the most sensitive paths.
func criticalDeniedPaths() []string {
	return []string{
		"/etc/shadow",
		"/etc/sudoers",
		"/boot",
	}
}
