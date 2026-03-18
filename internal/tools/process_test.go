package tools

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"syscall"
	"testing"
)

func TestParseProcStat(t *testing.T) {
	// Simulated /proc/[pid]/stat content.
	// Fields: pid (comm) state ppid pgrp session tty_nr tpgid flags
	//   minflt cminflt majflt cmajflt utime stime cutime cstime priority nice
	//   num_threads itrealvalue starttime vsize rss ...
	content := "12345 (firefox) S 1234 12345 12345 0 -1 4194304 " +
		"100 200 0 0 500 150 0 0 20 0 " +
		"10 0 123456789 1000000 3000 18446744073709551615"

	stat, err := ParseProcStat(content)
	if err != nil {
		t.Fatalf("ParseProcStat: %v", err)
	}

	if stat.PID != 12345 {
		t.Errorf("PID = %d, want 12345", stat.PID)
	}
	if stat.Comm != "firefox" {
		t.Errorf("Comm = %q, want %q", stat.Comm, "firefox")
	}
	if stat.State != "S" {
		t.Errorf("State = %q, want %q", stat.State, "S")
	}
	if stat.UTime != 500 {
		t.Errorf("UTime = %d, want 500", stat.UTime)
	}
	if stat.STime != 150 {
		t.Errorf("STime = %d, want 150", stat.STime)
	}
	if stat.RSS != 3000 {
		t.Errorf("RSS = %d, want 3000", stat.RSS)
	}
}

func TestParseProcStatParensInComm(t *testing.T) {
	// Some processes have parens in their comm name.
	content := "999 (Web Content (pid 998)) S 1 999 999 0 -1 4194304 " +
		"100 200 0 0 250 75 0 0 20 0 " +
		"5 0 123456789 500000 1500 18446744073709551615"

	stat, err := ParseProcStat(content)
	if err != nil {
		t.Fatalf("ParseProcStat: %v", err)
	}

	if stat.PID != 999 {
		t.Errorf("PID = %d, want 999", stat.PID)
	}
	if stat.Comm != "Web Content (pid 998)" {
		t.Errorf("Comm = %q, want %q", stat.Comm, "Web Content (pid 998)")
	}
}

func TestParseProcStatInvalid(t *testing.T) {
	_, err := ParseProcStat("not valid stat content")
	if err == nil {
		t.Error("expected error for invalid stat")
	}

	_, err = ParseProcStat("abc (test) S")
	if err == nil {
		t.Error("expected error for invalid pid")
	}
}

func TestParseProcCmdline(t *testing.T) {
	tests := []struct {
		input string
		want  string
	}{
		{"/usr/bin/firefox\x00--new-tab\x00https://example.com\x00", "/usr/bin/firefox --new-tab https://example.com"},
		{"\x00", ""},
		{"", ""},
		{"simple-command", "simple-command"},
	}

	for _, tt := range tests {
		got := ParseProcCmdline(tt.input)
		if got != tt.want {
			t.Errorf("ParseProcCmdline(%q) = %q, want %q", tt.input, got, tt.want)
		}
	}
}

func TestParseProcCmdlineTruncation(t *testing.T) {
	long := strings.Repeat("a", 200)
	got := ParseProcCmdline(long)
	if len(got) > 130 {
		t.Errorf("long cmdline should be truncated, got len %d", len(got))
	}
	if !strings.HasSuffix(got, "...") {
		t.Errorf("truncated cmdline should end with ..., got %q", got)
	}
}

func TestParseSignal(t *testing.T) {
	tests := []struct {
		input string
		want  syscall.Signal
	}{
		{"SIGTERM", syscall.SIGTERM},
		{"SIGKILL", syscall.SIGKILL},
		{"TERM", syscall.SIGTERM},
		{"KILL", syscall.SIGKILL},
		{"HUP", syscall.SIGHUP},
		{"sigint", syscall.SIGINT},
		{"9", syscall.Signal(9)},
		{"unknown", syscall.SIGTERM},
		{"", syscall.SIGTERM},
	}

	for _, tt := range tests {
		got := ParseSignal(tt.input)
		if got != tt.want {
			t.Errorf("ParseSignal(%q) = %v, want %v", tt.input, got, tt.want)
		}
	}
}

func TestFormatProcessMem(t *testing.T) {
	tests := []struct {
		bytes int64
		want  string
	}{
		{500, "500B"},
		{1024, "1KB"},
		{1048576, "1MB"},
		{157286400, "150MB"},
		{1073741824, "1.0GB"},
	}

	for _, tt := range tests {
		got := FormatProcessMem(tt.bytes)
		if got != tt.want {
			t.Errorf("FormatProcessMem(%d) = %q, want %q", tt.bytes, got, tt.want)
		}
	}
}

func TestFormatProcessList(t *testing.T) {
	procs := []ProcInfo{
		{PID: 1234, Command: "/usr/bin/firefox", MemRSS: 157286400},
		{PID: 5678, Command: "/usr/bin/code", MemRSS: 104857600},
	}

	result := FormatProcessList(procs)
	if !strings.Contains(result, "PID") {
		t.Errorf("should contain header: %s", result)
	}
	if !strings.Contains(result, "1234") {
		t.Errorf("should contain pid 1234: %s", result)
	}
	if !strings.Contains(result, "firefox") {
		t.Errorf("should contain firefox: %s", result)
	}
	if !strings.Contains(result, "150MB") {
		t.Errorf("should contain 150MB: %s", result)
	}
}

func TestFormatProcessListEmpty(t *testing.T) {
	result := FormatProcessList(nil)
	if !strings.Contains(result, "No processes found") {
		t.Errorf("expected no processes message: %s", result)
	}
}

func TestReadProcessInfoFromMock(t *testing.T) {
	// Create a mock /proc structure.
	procRoot := t.TempDir()
	pid := 12345
	pidDir := filepath.Join(procRoot, fmt.Sprintf("%d", pid))
	os.Mkdir(pidDir, 0755)

	statContent := "12345 (testproc) S 1 12345 12345 0 -1 4194304 " +
		"100 200 0 0 500 150 0 0 20 0 " +
		"10 0 123456789 1000000 3000 18446744073709551615"
	os.WriteFile(filepath.Join(pidDir, "stat"), []byte(statContent), 0644)

	cmdlineContent := "/usr/bin/testproc\x00--flag\x00value\x00"
	os.WriteFile(filepath.Join(pidDir, "cmdline"), []byte(cmdlineContent), 0644)

	info, err := readProcessInfoFrom(procRoot, pid)
	if err != nil {
		t.Fatalf("readProcessInfoFrom: %v", err)
	}

	if info.PID != 12345 {
		t.Errorf("PID = %d, want 12345", info.PID)
	}
	if !strings.Contains(info.Command, "testproc") {
		t.Errorf("Command should contain testproc: %q", info.Command)
	}
	if info.MemRSS <= 0 {
		t.Errorf("MemRSS should be positive: %d", info.MemRSS)
	}
}

func TestReadProcessInfoFromMockNoCmdline(t *testing.T) {
	procRoot := t.TempDir()
	pid := 99
	pidDir := filepath.Join(procRoot, fmt.Sprintf("%d", pid))
	os.Mkdir(pidDir, 0755)

	statContent := "99 (kworker) S 2 0 0 0 -1 69238880 " +
		"0 0 0 0 0 0 0 0 20 0 " +
		"1 0 100 0 0 18446744073709551615"
	os.WriteFile(filepath.Join(pidDir, "stat"), []byte(statContent), 0644)

	// No cmdline file - kernel thread.
	info, err := readProcessInfoFrom(procRoot, pid)
	if err != nil {
		t.Fatalf("readProcessInfoFrom: %v", err)
	}

	if info.Command != "[kworker]" {
		t.Errorf("Command = %q, want %q", info.Command, "[kworker]")
	}
}

func TestListProcessesFromMock(t *testing.T) {
	procRoot := t.TempDir()

	// Create two mock process dirs.
	for _, pid := range []int{100, 200} {
		pidDir := filepath.Join(procRoot, fmt.Sprintf("%d", pid))
		os.Mkdir(pidDir, 0755)

		statContent := fmt.Sprintf("%d (proc%d) S 1 %d %d 0 -1 4194304 "+
			"0 0 0 0 100 50 0 0 20 0 "+
			"1 0 123 500000 %d 18446744073709551615", pid, pid, pid, pid, pid*10)
		os.WriteFile(filepath.Join(pidDir, "stat"), []byte(statContent), 0644)
		os.WriteFile(filepath.Join(pidDir, "cmdline"), []byte(fmt.Sprintf("/bin/proc%d\x00", pid)), 0644)
	}

	// Also create a non-numeric dir (should be ignored).
	os.Mkdir(filepath.Join(procRoot, "self"), 0755)

	procs, err := listProcessesFrom(procRoot)
	if err != nil {
		t.Fatalf("listProcessesFrom: %v", err)
	}

	if len(procs) != 2 {
		t.Errorf("expected 2 processes, got %d", len(procs))
	}
}

func TestKillProcessInvalidPID(t *testing.T) {
	err := KillProcess(0, syscall.SIGTERM)
	if err == nil {
		t.Error("expected error for pid 0")
	}

	err = KillProcess(-1, syscall.SIGTERM)
	if err == nil {
		t.Error("expected error for pid -1")
	}
}

func TestProcessToolRegistration(t *testing.T) {
	r := NewRegistry()
	RegisterProcessTools(r)

	tool, err := r.Get("process")
	if err != nil {
		t.Fatal("process tool not registered")
	}
	if tool.Name != "process" {
		t.Errorf("tool name = %q, want %q", tool.Name, "process")
	}
}

func TestProcessToolBadAction(t *testing.T) {
	r := NewRegistry()
	RegisterProcessTools(r)

	tool, _ := r.Get("process")
	_, err := tool.Execute(map[string]string{"action": "explode"})
	if err == nil {
		t.Error("expected error for unknown action")
	}
}

func TestProcessToolSearchMissingName(t *testing.T) {
	r := NewRegistry()
	RegisterProcessTools(r)

	tool, _ := r.Get("process")
	_, err := tool.Execute(map[string]string{"action": "search"})
	if err == nil {
		t.Error("expected error when name is missing for search")
	}
}

func TestProcessToolKillMissingPID(t *testing.T) {
	r := NewRegistry()
	RegisterProcessTools(r)

	tool, _ := r.Get("process")
	_, err := tool.Execute(map[string]string{"action": "kill"})
	if err == nil {
		t.Error("expected error when pid is missing for kill")
	}
}

func TestProcessToolKillInvalidPID(t *testing.T) {
	r := NewRegistry()
	RegisterProcessTools(r)

	tool, _ := r.Get("process")
	_, err := tool.Execute(map[string]string{"action": "kill", "pid": "notanumber"})
	if err == nil {
		t.Error("expected error for non-numeric pid")
	}
}
