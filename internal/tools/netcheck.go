package tools

import (
	"fmt"
	"io"
	"net"
	"net/http"
	"strings"
	"time"
)

// NetCheck performs network diagnostic operations.
// Supported actions: ping, dns, port, ip, connectivity.
// Default action is connectivity.
func NetCheck(action, target, port string) (string, error) {
	action = strings.ToLower(strings.TrimSpace(action))
	if action == "" {
		action = "connectivity"
	}

	switch action {
	case "ping":
		return netPing(target)
	case "dns":
		return netDNS(target)
	case "port":
		return netPort(target, port)
	case "ip":
		return netIP()
	case "connectivity":
		return netConnectivity()
	default:
		return "", fmt.Errorf("netcheck: unknown action %q — supported: ping, dns, port, ip, connectivity", action)
	}
}

func netPing(target string) (string, error) {
	target = strings.TrimSpace(target)
	if target == "" {
		return "", fmt.Errorf("netcheck: target is required for ping")
	}

	ports := []string{"443", "80"}
	var results []string

	for _, p := range ports {
		addr := net.JoinHostPort(target, p)
		start := time.Now()
		conn, err := net.DialTimeout("tcp", addr, 3*time.Second)
		elapsed := time.Since(start)

		if err != nil {
			results = append(results, fmt.Sprintf("  port %s: unreachable (%v)", p, err))
			continue
		}
		conn.Close()
		results = append(results, fmt.Sprintf("  port %s: reachable (%.1fms)", p, float64(elapsed.Microseconds())/1000.0))
	}

	return fmt.Sprintf("Ping %s:\n%s", target, strings.Join(results, "\n")), nil
}

func netDNS(target string) (string, error) {
	target = strings.TrimSpace(target)
	if target == "" {
		return "", fmt.Errorf("netcheck: target is required for dns")
	}

	ips, err := net.LookupHost(target)
	if err != nil {
		return "", fmt.Errorf("netcheck: DNS lookup failed for %s: %w", target, err)
	}

	return fmt.Sprintf("DNS %s:\n  %s", target, strings.Join(ips, "\n  ")), nil
}

func netPort(target, port string) (string, error) {
	target = strings.TrimSpace(target)
	port = strings.TrimSpace(port)
	if target == "" {
		return "", fmt.Errorf("netcheck: target is required for port check")
	}
	if port == "" {
		return "", fmt.Errorf("netcheck: port is required for port check")
	}

	addr := net.JoinHostPort(target, port)
	conn, err := net.DialTimeout("tcp", addr, 3*time.Second)
	if err != nil {
		return fmt.Sprintf("Port %s on %s: closed", port, target), nil
	}
	conn.Close()
	return fmt.Sprintf("Port %s on %s: open", port, target), nil
}

func netIP() (string, error) {
	var sb strings.Builder

	// Local IP
	localIP := getNetLocalIP()
	fmt.Fprintf(&sb, "Local IP: %s", localIP)

	// External IP
	client := &http.Client{Timeout: 5 * time.Second}
	resp, err := client.Get("http://ifconfig.me")
	if err == nil {
		defer resp.Body.Close()
		body, readErr := io.ReadAll(io.LimitReader(resp.Body, 256))
		if readErr == nil {
			extIP := strings.TrimSpace(string(body))
			if !strings.Contains(extIP, "<") {
				fmt.Fprintf(&sb, "\nExternal IP: %s", extIP)
			}
		}
	}

	return sb.String(), nil
}

func getNetLocalIP() string {
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

// ParseLocalIPs extracts non-loopback IPv4 addresses from interface addresses.
func ParseLocalIPs(addrs []net.Addr) []string {
	var ips []string
	for _, addr := range addrs {
		if ipnet, ok := addr.(*net.IPNet); ok && !ipnet.IP.IsLoopback() && ipnet.IP.To4() != nil {
			ips = append(ips, ipnet.IP.String())
		}
	}
	return ips
}

func netConnectivity() (string, error) {
	targets := []struct {
		name string
		addr string
	}{
		{"Cloudflare (1.1.1.1)", "1.1.1.1:443"},
		{"Google (8.8.8.8)", "8.8.8.8:443"},
	}

	var results []string
	online := false

	for _, t := range targets {
		conn, err := net.DialTimeout("tcp", t.addr, 3*time.Second)
		if err != nil {
			results = append(results, fmt.Sprintf("  %s: unreachable", t.name))
			continue
		}
		conn.Close()
		results = append(results, fmt.Sprintf("  %s: reachable", t.name))
		online = true
	}

	status := "offline"
	if online {
		status = "online"
	}

	return fmt.Sprintf("Connectivity: %s\n%s", status, strings.Join(results, "\n")), nil
}

// RegisterNetCheckTools adds the netcheck tool to the registry.
func RegisterNetCheckTools(r *Registry) {
	r.Register(Tool{
		Name:        "netcheck",
		Description: "Network diagnostics. Args: action (ping/dns/port/ip/connectivity, default connectivity), target (host for ping/dns/port), port (for port check).",
		Execute: func(args map[string]string) (string, error) {
			return NetCheck(args["action"], args["target"], args["port"])
		},
	})
}
