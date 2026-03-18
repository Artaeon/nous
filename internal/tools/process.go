package tools

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"syscall"
	"time"
)

// ProcInfo holds parsed process information.
type ProcInfo struct {
	PID     int
	Command string
	CPUPct  float64
	MemRSS  int64 // in bytes
	State   string
}

// ProcStat holds raw fields from /proc/[pid]/stat.
type ProcStat struct {
	PID     int
	Comm    string
	State   string
	UTime   uint64
	STime   uint64
	RSS     int64 // in pages
}

// ParseProcStat parses the content of /proc/[pid]/stat.
func ParseProcStat(content string) (ProcStat, error) {
	// The comm field is in parens and may contain spaces, so find the last ')'.
	start := strings.Index(content, "(")
	end := strings.LastIndex(content, ")")
	if start < 0 || end < 0 || end <= start {
		return ProcStat{}, fmt.Errorf("process: invalid stat format")
	}

	pidStr := strings.TrimSpace(content[:start])
	pid, err := strconv.Atoi(pidStr)
	if err != nil {
		return ProcStat{}, fmt.Errorf("process: invalid pid: %w", err)
	}

	comm := content[start+1 : end]

	// Fields after the closing paren (space-separated).
	rest := strings.Fields(content[end+2:])
	if len(rest) < 22 {
		return ProcStat{}, fmt.Errorf("process: stat has too few fields")
	}

	state := rest[0]

	utime, _ := strconv.ParseUint(rest[11], 10, 64)
	stime, _ := strconv.ParseUint(rest[12], 10, 64)
	rss, _ := strconv.ParseInt(rest[21], 10, 64)

	return ProcStat{
		PID:   pid,
		Comm:  comm,
		State: state,
		UTime: utime,
		STime: stime,
		RSS:   rss,
	}, nil
}

// ParseProcCmdline parses the content of /proc/[pid]/cmdline.
func ParseProcCmdline(content string) string {
	// cmdline uses null bytes as separators.
	cmd := strings.ReplaceAll(content, "\x00", " ")
	cmd = strings.TrimSpace(cmd)
	if cmd == "" {
		return ""
	}
	// Truncate long commands.
	if len(cmd) > 120 {
		cmd = cmd[:120] + "..."
	}
	return cmd
}

// ReadProcessInfo reads process info from /proc for a given PID.
func ReadProcessInfo(pid int) (ProcInfo, error) {
	return readProcessInfoFrom("/proc", pid)
}

func readProcessInfoFrom(procRoot string, pid int) (ProcInfo, error) {
	pidStr := strconv.Itoa(pid)

	statData, err := os.ReadFile(filepath.Join(procRoot, pidStr, "stat"))
	if err != nil {
		return ProcInfo{}, fmt.Errorf("process: cannot read stat for pid %d: %w", pid, err)
	}

	stat, err := ParseProcStat(string(statData))
	if err != nil {
		return ProcInfo{}, err
	}

	cmdlineData, _ := os.ReadFile(filepath.Join(procRoot, pidStr, "cmdline"))
	cmdline := ParseProcCmdline(string(cmdlineData))
	if cmdline == "" {
		cmdline = "[" + stat.Comm + "]"
	}

	pageSize := int64(os.Getpagesize())
	memRSS := stat.RSS * pageSize

	return ProcInfo{
		PID:     pid,
		Command: cmdline,
		MemRSS:  memRSS,
		State:   stat.State,
	}, nil
}

// ListProcesses returns all running processes.
func ListProcesses() ([]ProcInfo, error) {
	return listProcessesFrom("/proc")
}

func listProcessesFrom(procRoot string) ([]ProcInfo, error) {
	entries, err := os.ReadDir(procRoot)
	if err != nil {
		return nil, fmt.Errorf("process: cannot read /proc: %w", err)
	}

	var procs []ProcInfo
	for _, entry := range entries {
		if !entry.IsDir() {
			continue
		}
		pid, err := strconv.Atoi(entry.Name())
		if err != nil {
			continue
		}

		info, err := readProcessInfoFrom(procRoot, pid)
		if err != nil {
			continue // process may have exited
		}
		procs = append(procs, info)
	}

	return procs, nil
}

// SearchProcesses finds processes matching a name substring.
func SearchProcesses(name string) ([]ProcInfo, error) {
	procs, err := ListProcesses()
	if err != nil {
		return nil, err
	}

	lowerName := strings.ToLower(name)
	var matches []ProcInfo
	for _, p := range procs {
		if strings.Contains(strings.ToLower(p.Command), lowerName) {
			matches = append(matches, p)
		}
	}
	return matches, nil
}

// TopProcesses returns the top N processes by memory usage.
func TopProcesses(count int) ([]ProcInfo, error) {
	if count <= 0 {
		count = 10
	}

	procs, err := ListProcesses()
	if err != nil {
		return nil, err
	}

	sort.Slice(procs, func(i, j int) bool {
		return procs[i].MemRSS > procs[j].MemRSS
	})

	if len(procs) > count {
		procs = procs[:count]
	}
	return procs, nil
}

// KillProcess sends a signal to a process.
func KillProcess(pid int, signal syscall.Signal) error {
	if pid <= 0 {
		return fmt.Errorf("process: invalid pid %d", pid)
	}
	err := syscall.Kill(pid, signal)
	if err != nil {
		return fmt.Errorf("process: kill(%d, %v) failed: %w", pid, signal, err)
	}
	return nil
}

// ParseSignal converts a signal name to a syscall.Signal.
func ParseSignal(name string) syscall.Signal {
	name = strings.ToUpper(strings.TrimSpace(name))
	name = strings.TrimPrefix(name, "SIG")

	signals := map[string]syscall.Signal{
		"TERM": syscall.SIGTERM,
		"KILL": syscall.SIGKILL,
		"HUP":  syscall.SIGHUP,
		"INT":  syscall.SIGINT,
		"QUIT": syscall.SIGQUIT,
		"USR1": syscall.SIGUSR1,
		"USR2": syscall.SIGUSR2,
		"STOP": syscall.SIGSTOP,
		"CONT": syscall.SIGCONT,
	}

	if sig, ok := signals[name]; ok {
		return sig
	}

	// Try numeric.
	if n, err := strconv.Atoi(name); err == nil {
		return syscall.Signal(n)
	}

	return syscall.SIGTERM
}

// FormatProcessList formats process info for display.
func FormatProcessList(procs []ProcInfo) string {
	if len(procs) == 0 {
		return "No processes found."
	}

	var sb strings.Builder
	fmt.Fprintf(&sb, "%-8s %-8s %s\n", "PID", "MEM", "COMMAND")
	for _, p := range procs {
		memStr := FormatProcessMem(p.MemRSS)
		fmt.Fprintf(&sb, "%-8d %-8s %s\n", p.PID, memStr, p.Command)
	}
	return strings.TrimRight(sb.String(), "\n")
}

// FormatProcessMem returns a human-readable memory size.
func FormatProcessMem(bytes int64) string {
	switch {
	case bytes >= 1<<30:
		return fmt.Sprintf("%.1fGB", float64(bytes)/(1<<30))
	case bytes >= 1<<20:
		return fmt.Sprintf("%.0fMB", float64(bytes)/(1<<20))
	case bytes >= 1<<10:
		return fmt.Sprintf("%.0fKB", float64(bytes)/(1<<10))
	default:
		return fmt.Sprintf("%dB", bytes)
	}
}

// ComputeCPUPercent reads CPU usage twice with a brief interval and computes percent.
func ComputeCPUPercent(pid int, interval time.Duration) (float64, error) {
	stat1, err := readStatTimes(pid)
	if err != nil {
		return 0, err
	}

	time.Sleep(interval)

	stat2, err := readStatTimes(pid)
	if err != nil {
		return 0, err
	}

	totalDelta := float64((stat2.UTime + stat2.STime) - (stat1.UTime + stat1.STime))
	// Convert clock ticks to seconds (typically 100 Hz).
	cpuPct := (totalDelta / float64(interval.Milliseconds()) * 1000.0) / 100.0 * 100.0
	return cpuPct, nil
}

func readStatTimes(pid int) (ProcStat, error) {
	data, err := os.ReadFile(fmt.Sprintf("/proc/%d/stat", pid))
	if err != nil {
		return ProcStat{}, err
	}
	return ParseProcStat(string(data))
}

// RegisterProcessTools adds the process tool to the registry.
func RegisterProcessTools(r *Registry) {
	r.Register(Tool{
		Name:        "process",
		Description: "Process manager. Args: action (list/search/kill/top), name (for search/kill), pid (for kill), signal (default SIGTERM), count (for top, default 10).",
		Execute: func(args map[string]string) (string, error) {
			action := strings.ToLower(strings.TrimSpace(args["action"]))

			switch action {
			case "list":
				procs, err := ListProcesses()
				if err != nil {
					return "", err
				}
				return FormatProcessList(procs), nil

			case "search":
				name := args["name"]
				if name == "" {
					return "", fmt.Errorf("process: 'name' argument required for search")
				}
				procs, err := SearchProcesses(name)
				if err != nil {
					return "", err
				}
				if len(procs) == 0 {
					return fmt.Sprintf("No processes matching %q found.", name), nil
				}
				return FormatProcessList(procs), nil

			case "kill":
				pidStr := args["pid"]
				if pidStr == "" {
					return "", fmt.Errorf("process: 'pid' argument required for kill")
				}
				pid, err := strconv.Atoi(pidStr)
				if err != nil {
					return "", fmt.Errorf("process: invalid pid %q", pidStr)
				}
				sig := ParseSignal(args["signal"])
				if err := KillProcess(pid, sig); err != nil {
					return "", err
				}
				return fmt.Sprintf("Sent %v to process %d.", sig, pid), nil

			case "top":
				count := 10
				if v, ok := args["count"]; ok {
					if n, err := strconv.Atoi(v); err == nil && n > 0 {
						count = n
					}
				}
				procs, err := TopProcesses(count)
				if err != nil {
					return "", err
				}
				return FormatProcessList(procs), nil

			default:
				return "", fmt.Errorf("process: unknown action %q (use list, search, kill, or top)", action)
			}
		},
	})
}
