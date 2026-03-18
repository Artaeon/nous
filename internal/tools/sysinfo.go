package tools

import (
	"bytes"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"os/exec"
	"runtime"
	"strconv"
	"strings"
	"time"
)

// GetSystemInfo returns system information based on the query type.
// Supported queries: disk, storage, memory, ram, cpu, ip, uptime, hostname, os, system.
// An empty or unrecognized query returns all information combined.
func GetSystemInfo(query string) (string, error) {
	query = strings.ToLower(strings.TrimSpace(query))

	switch query {
	case "disk", "storage":
		return getSysDisk()
	case "memory", "ram":
		return getSysMemory()
	case "cpu":
		return getSysCPU()
	case "ip":
		return getSysIP()
	case "uptime":
		return getSysUptime()
	case "hostname":
		return getSysHostname()
	case "os", "system":
		return getSysOS()
	default:
		return getSysAll()
	}
}

func getSysDisk() (string, error) {
	cmd := exec.Command("df", "-h", "/")
	out, err := cmd.Output()
	if err != nil {
		return "", fmt.Errorf("sysinfo: df failed: %w", err)
	}
	return ParseDfOutput(string(out))
}

// ParseDfOutput extracts disk usage information from df -h output.
func ParseDfOutput(output string) (string, error) {
	lines := strings.Split(strings.TrimSpace(output), "\n")
	if len(lines) < 2 {
		return "", fmt.Errorf("sysinfo: unexpected df output")
	}
	fields := strings.Fields(lines[1])
	if len(fields) < 6 {
		return "", fmt.Errorf("sysinfo: unexpected df output format")
	}
	return fmt.Sprintf("Disk: %s total, %s used, %s available (%s used)", fields[1], fields[2], fields[3], fields[4]), nil
}

func getSysMemory() (string, error) {
	data, err := os.ReadFile("/proc/meminfo")
	if err != nil {
		return "", fmt.Errorf("sysinfo: cannot read /proc/meminfo: %w", err)
	}
	return ParseMeminfo(string(data))
}

// ParseMeminfo extracts memory information from /proc/meminfo content.
func ParseMeminfo(content string) (string, error) {
	fields := map[string]uint64{}
	for _, line := range strings.Split(content, "\n") {
		parts := strings.SplitN(line, ":", 2)
		if len(parts) != 2 {
			continue
		}
		key := strings.TrimSpace(parts[0])
		valStr := strings.TrimSpace(parts[1])
		valStr = strings.TrimSuffix(valStr, " kB")
		valStr = strings.TrimSpace(valStr)
		if v, err := strconv.ParseUint(valStr, 10, 64); err == nil {
			fields[key] = v
		}
	}

	total, okT := fields["MemTotal"]
	available, okA := fields["MemAvailable"]
	if !okT {
		return "", fmt.Errorf("sysinfo: MemTotal not found in meminfo")
	}
	if !okA {
		// Fallback: estimate available from Free + Buffers + Cached
		free := fields["MemFree"]
		buffers := fields["Buffers"]
		cached := fields["Cached"]
		available = free + buffers + cached
	}

	used := total - available
	totalMB := float64(total) / 1024
	usedMB := float64(used) / 1024
	availMB := float64(available) / 1024

	return fmt.Sprintf("Memory: %.0f MB total, %.0f MB used, %.0f MB available", totalMB, usedMB, availMB), nil
}

func getSysCPU() (string, error) {
	data, err := os.ReadFile("/proc/cpuinfo")
	if err != nil {
		// Fallback for non-Linux
		return fmt.Sprintf("CPU: %d cores (%s/%s)", runtime.NumCPU(), runtime.GOOS, runtime.GOARCH), nil
	}
	return ParseCPUInfo(string(data))
}

// ParseCPUInfo extracts CPU model and core count from /proc/cpuinfo content.
func ParseCPUInfo(content string) (string, error) {
	modelName := ""
	cores := 0
	for _, line := range strings.Split(content, "\n") {
		parts := strings.SplitN(line, ":", 2)
		if len(parts) != 2 {
			continue
		}
		key := strings.TrimSpace(parts[0])
		val := strings.TrimSpace(parts[1])
		if key == "model name" && modelName == "" {
			modelName = val
		}
		if key == "processor" {
			cores++
		}
	}

	if modelName == "" {
		modelName = "unknown"
	}
	if cores == 0 {
		cores = runtime.NumCPU()
	}

	return fmt.Sprintf("CPU: %s (%d cores)", modelName, cores), nil
}

func getSysIP() (string, error) {
	var sb strings.Builder

	// Local IP
	localIP := getLocalIP()
	fmt.Fprintf(&sb, "Local IP: %s", localIP)

	// External IP
	extIP, err := getExternalIP()
	if err == nil {
		fmt.Fprintf(&sb, "\nExternal IP: %s", extIP)
	}

	return sb.String(), nil
}

func getLocalIP() string {
	addrs, err := net.InterfaceAddrs()
	if err != nil {
		return "unknown"
	}
	for _, addr := range addrs {
		if ipnet, ok := addr.(*net.IPNet); ok && !ipnet.IP.IsLoopback() && ipnet.IP.To4() != nil {
			return ipnet.IP.String()
		}
	}
	return "unknown"
}

func getExternalIP() (string, error) {
	client := &http.Client{Timeout: 5 * time.Second}
	resp, err := client.Get("http://ifconfig.me")
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(io.LimitReader(resp.Body, 256))
	if err != nil {
		return "", err
	}
	ip := strings.TrimSpace(string(body))
	if strings.Contains(ip, "<") {
		return "", fmt.Errorf("got HTML instead of IP")
	}
	return ip, nil
}

func getSysUptime() (string, error) {
	data, err := os.ReadFile("/proc/uptime")
	if err != nil {
		return "", fmt.Errorf("sysinfo: cannot read /proc/uptime: %w", err)
	}
	return ParseUptime(string(data))
}

// ParseUptime extracts uptime from /proc/uptime content.
func ParseUptime(content string) (string, error) {
	fields := strings.Fields(strings.TrimSpace(content))
	if len(fields) < 1 {
		return "", fmt.Errorf("sysinfo: unexpected uptime format")
	}
	seconds, err := strconv.ParseFloat(fields[0], 64)
	if err != nil {
		return "", fmt.Errorf("sysinfo: invalid uptime value: %w", err)
	}

	days := int(seconds) / 86400
	hours := (int(seconds) % 86400) / 3600
	mins := (int(seconds) % 3600) / 60

	if days > 0 {
		return fmt.Sprintf("Uptime: %d days, %d hours, %d minutes", days, hours, mins), nil
	}
	if hours > 0 {
		return fmt.Sprintf("Uptime: %d hours, %d minutes", hours, mins), nil
	}
	return fmt.Sprintf("Uptime: %d minutes", mins), nil
}

func getSysHostname() (string, error) {
	hostname, err := os.Hostname()
	if err != nil {
		return "", fmt.Errorf("sysinfo: %w", err)
	}
	return fmt.Sprintf("Hostname: %s", hostname), nil
}

func getSysOS() (string, error) {
	hostname, _ := os.Hostname()
	kernel := "unknown"

	cmd := exec.Command("uname", "-r")
	if out, err := cmd.Output(); err == nil {
		kernel = strings.TrimSpace(string(out))
	}

	return fmt.Sprintf("Hostname: %s\nOS: %s/%s\nKernel: %s", hostname, runtime.GOOS, runtime.GOARCH, kernel), nil
}

func getSysAll() (string, error) {
	var parts []string

	if s, err := getSysHostname(); err == nil {
		parts = append(parts, s)
	}
	if s, err := getSysOS(); err == nil {
		// getSysOS includes hostname, so use only if hostname failed
		if len(parts) == 0 {
			parts = append(parts, s)
		} else {
			// Extract just OS and kernel lines
			for _, line := range strings.Split(s, "\n") {
				if strings.HasPrefix(line, "OS:") || strings.HasPrefix(line, "Kernel:") {
					parts = append(parts, line)
				}
			}
		}
	}
	if s, err := getSysCPU(); err == nil {
		parts = append(parts, s)
	}
	if s, err := getSysMemory(); err == nil {
		parts = append(parts, s)
	}
	if s, err := getSysDisk(); err == nil {
		parts = append(parts, s)
	}
	if s, err := getSysUptime(); err == nil {
		parts = append(parts, s)
	}
	if s, err := getSysIP(); err == nil {
		parts = append(parts, s)
	}

	if len(parts) == 0 {
		return "", fmt.Errorf("sysinfo: could not gather any system information")
	}

	return strings.Join(parts, "\n"), nil
}

// RegisterSysInfoTools adds the sysinfo tool to the registry.
func RegisterSysInfoTools(r *Registry) {
	r.Register(Tool{
		Name:        "sysinfo",
		Description: "Get system information. Args: query (optional: disk, memory, cpu, ip, uptime, hostname, os). No query returns all.",
		Execute: func(args map[string]string) (string, error) {
			return GetSystemInfo(args["query"])
		},
	})
}

// formatDfLine is used internally for testing; it formats a df output line.
func formatDfLine(filesystem, size, used, avail, usePct, mountedOn string) string {
	var buf bytes.Buffer
	fmt.Fprintf(&buf, "Filesystem      Size  Used Avail Use%% Mounted on\n")
	fmt.Fprintf(&buf, "%s  %s  %s  %s  %s  %s\n", filesystem, size, used, avail, usePct, mountedOn)
	return buf.String()
}
