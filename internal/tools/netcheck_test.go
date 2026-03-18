package tools

import (
	"net"
	"strings"
	"testing"
)

func TestParseLocalIPs(t *testing.T) {
	// Create mock addresses
	_, ipnet1, _ := net.ParseCIDR("192.168.1.100/24")
	_, ipnet2, _ := net.ParseCIDR("127.0.0.1/8")
	_, ipnet3, _ := net.ParseCIDR("10.0.0.5/8")

	addrs := []net.Addr{
		&net.IPNet{IP: net.ParseIP("192.168.1.100"), Mask: ipnet1.Mask},
		&net.IPNet{IP: net.ParseIP("127.0.0.1"), Mask: ipnet2.Mask},
		&net.IPNet{IP: net.ParseIP("10.0.0.5"), Mask: ipnet3.Mask},
	}

	ips := ParseLocalIPs(addrs)

	// Should exclude loopback 127.0.0.1
	for _, ip := range ips {
		if ip == "127.0.0.1" {
			t.Error("should not include loopback address")
		}
	}

	// Should include non-loopback addresses
	found192 := false
	found10 := false
	for _, ip := range ips {
		if ip == "192.168.1.100" {
			found192 = true
		}
		if ip == "10.0.0.5" {
			found10 = true
		}
	}

	if !found192 {
		t.Errorf("should include 192.168.1.100, got %v", ips)
	}
	if !found10 {
		t.Errorf("should include 10.0.0.5, got %v", ips)
	}
}

func TestParseLocalIPsEmpty(t *testing.T) {
	ips := ParseLocalIPs(nil)
	if len(ips) != 0 {
		t.Errorf("expected empty, got %v", ips)
	}
}

func TestNetCheckPingMissingTarget(t *testing.T) {
	_, err := NetCheck("ping", "", "")
	if err == nil {
		t.Error("expected error for empty target on ping")
	}
}

func TestNetCheckDNSMissingTarget(t *testing.T) {
	_, err := NetCheck("dns", "", "")
	if err == nil {
		t.Error("expected error for empty target on dns")
	}
}

func TestNetCheckPortMissingArgs(t *testing.T) {
	_, err := NetCheck("port", "", "80")
	if err == nil {
		t.Error("expected error for empty target on port check")
	}

	_, err = NetCheck("port", "localhost", "")
	if err == nil {
		t.Error("expected error for empty port on port check")
	}
}

func TestNetCheckUnknownAction(t *testing.T) {
	_, err := NetCheck("traceroute", "example.com", "")
	if err == nil {
		t.Error("expected error for unknown action")
	}
}

func TestNetCheckDefaultAction(t *testing.T) {
	// Default action should be connectivity
	result, err := NetCheck("", "", "")
	if err != nil {
		t.Skipf("skipping connectivity test: %v (may not have network)", err)
	}
	if !strings.Contains(result, "Connectivity:") {
		t.Errorf("default action should return connectivity result, got %q", result)
	}
}

func TestNetCheckLocalhostPort(t *testing.T) {
	// Test port check against a port that's likely closed
	result, err := NetCheck("port", "localhost", "19999")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	// Port 19999 is very likely closed
	if !strings.Contains(result, "Port 19999 on localhost:") {
		t.Errorf("should contain port check result, got %q", result)
	}
}

func TestNetCheckDNSLocalhost(t *testing.T) {
	result, err := NetCheck("dns", "localhost", "")
	if err != nil {
		t.Skipf("skipping DNS test: %v", err)
	}
	if !strings.Contains(result, "DNS localhost:") {
		t.Errorf("should contain DNS result, got %q", result)
	}
	// localhost should resolve to 127.0.0.1 or ::1
	if !strings.Contains(result, "127.0.0.1") && !strings.Contains(result, "::1") {
		t.Errorf("localhost should resolve to loopback, got %q", result)
	}
}

func TestNetCheckToolRegistration(t *testing.T) {
	r := NewRegistry()
	RegisterNetCheckTools(r)

	tool, err := r.Get("netcheck")
	if err != nil {
		t.Fatal("netcheck tool not registered")
	}
	if tool.Name != "netcheck" {
		t.Errorf("tool name = %q, want %q", tool.Name, "netcheck")
	}
}
