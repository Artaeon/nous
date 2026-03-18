package tools

import (
	"strings"
	"testing"
)

func TestParseDfOutput(t *testing.T) {
	input := formatDfLine("/dev/sda1", "100G", "45G", "50G", "48%", "/")
	got, err := ParseDfOutput(input)
	if err != nil {
		t.Fatalf("ParseDfOutput error: %v", err)
	}

	if !strings.Contains(got, "100G total") {
		t.Errorf("should contain total, got %q", got)
	}
	if !strings.Contains(got, "45G used") {
		t.Errorf("should contain used, got %q", got)
	}
	if !strings.Contains(got, "50G available") {
		t.Errorf("should contain available, got %q", got)
	}
	if !strings.Contains(got, "48% used") {
		t.Errorf("should contain percentage, got %q", got)
	}
}

func TestParseDfOutputInvalid(t *testing.T) {
	_, err := ParseDfOutput("just one line")
	if err == nil {
		t.Error("expected error for single-line df output")
	}
}

func TestParseMeminfo(t *testing.T) {
	content := `MemTotal:       16384000 kB
MemFree:         4096000 kB
MemAvailable:    8192000 kB
Buffers:          512000 kB
Cached:          2048000 kB
`
	got, err := ParseMeminfo(content)
	if err != nil {
		t.Fatalf("ParseMeminfo error: %v", err)
	}

	if !strings.Contains(got, "Memory:") {
		t.Errorf("should start with 'Memory:', got %q", got)
	}
	if !strings.Contains(got, "16000 MB total") {
		t.Errorf("should contain total MB, got %q", got)
	}
	if !strings.Contains(got, "8000 MB available") {
		t.Errorf("should contain available MB, got %q", got)
	}
}

func TestParseMeminfoNoTotal(t *testing.T) {
	_, err := ParseMeminfo("MemFree: 1024 kB\n")
	if err == nil {
		t.Error("expected error when MemTotal is missing")
	}
}

func TestParseMeminfoFallback(t *testing.T) {
	// No MemAvailable, should use Free + Buffers + Cached
	content := `MemTotal:       8192000 kB
MemFree:         2048000 kB
Buffers:          512000 kB
Cached:          1024000 kB
`
	got, err := ParseMeminfo(content)
	if err != nil {
		t.Fatalf("ParseMeminfo error: %v", err)
	}

	if !strings.Contains(got, "Memory:") {
		t.Errorf("should contain 'Memory:', got %q", got)
	}
	// Available = Free(2048000) + Buffers(512000) + Cached(1024000) = 3584000 kB = 3500 MB
	if !strings.Contains(got, "3500 MB available") {
		t.Errorf("should compute available from fallback, got %q", got)
	}
}

func TestParseCPUInfo(t *testing.T) {
	content := `processor	: 0
vendor_id	: GenuineIntel
model name	: Intel(R) Core(TM) i7-9750H CPU @ 2.60GHz
cpu MHz		: 2600.000

processor	: 1
vendor_id	: GenuineIntel
model name	: Intel(R) Core(TM) i7-9750H CPU @ 2.60GHz
cpu MHz		: 2600.000

processor	: 2
vendor_id	: GenuineIntel
model name	: Intel(R) Core(TM) i7-9750H CPU @ 2.60GHz
cpu MHz		: 2600.000

processor	: 3
vendor_id	: GenuineIntel
model name	: Intel(R) Core(TM) i7-9750H CPU @ 2.60GHz
cpu MHz		: 2600.000
`
	got, err := ParseCPUInfo(content)
	if err != nil {
		t.Fatalf("ParseCPUInfo error: %v", err)
	}

	if !strings.Contains(got, "i7-9750H") {
		t.Errorf("should contain CPU model, got %q", got)
	}
	if !strings.Contains(got, "4 cores") {
		t.Errorf("should contain core count, got %q", got)
	}
}

func TestParseCPUInfoEmpty(t *testing.T) {
	got, err := ParseCPUInfo("")
	if err != nil {
		t.Fatalf("ParseCPUInfo error: %v", err)
	}
	if !strings.Contains(got, "unknown") {
		t.Errorf("should fall back to 'unknown' model, got %q", got)
	}
}

func TestParseUptime(t *testing.T) {
	tests := []struct {
		name    string
		content string
		want    string
	}{
		{
			name:    "days",
			content: "172800.50 345600.00",
			want:    "2 days",
		},
		{
			name:    "hours",
			content: "7200.00 14400.00",
			want:    "2 hours",
		},
		{
			name:    "minutes_only",
			content: "300.00 600.00",
			want:    "5 minutes",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := ParseUptime(tt.content)
			if err != nil {
				t.Fatalf("ParseUptime error: %v", err)
			}
			if !strings.Contains(got, tt.want) {
				t.Errorf("ParseUptime(%q) = %q, want to contain %q", tt.content, got, tt.want)
			}
		})
	}
}

func TestParseUptimeInvalid(t *testing.T) {
	_, err := ParseUptime("")
	if err == nil {
		t.Error("expected error for empty uptime")
	}

	_, err = ParseUptime("not_a_number 123.45")
	if err == nil {
		t.Error("expected error for non-numeric uptime")
	}
}

func TestGetSystemInfoValidQueries(t *testing.T) {
	// These queries should not return errors on a Linux system
	queries := []string{"hostname", "cpu"}
	for _, q := range queries {
		t.Run(q, func(t *testing.T) {
			got, err := GetSystemInfo(q)
			if err != nil {
				t.Skipf("skipping %q: %v (may not be available in test environment)", q, err)
			}
			if got == "" {
				t.Errorf("GetSystemInfo(%q) returned empty string", q)
			}
		})
	}
}

func TestSysInfoToolRegistration(t *testing.T) {
	r := NewRegistry()
	RegisterSysInfoTools(r)

	tool, err := r.Get("sysinfo")
	if err != nil {
		t.Fatal("sysinfo tool not registered")
	}

	if tool.Name != "sysinfo" {
		t.Errorf("tool name = %q, want %q", tool.Name, "sysinfo")
	}
}
