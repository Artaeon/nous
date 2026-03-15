package cognitive

import "testing"

func TestAutoApproveAlwaysTrue(t *testing.T) {
	if !AutoApprove("delete everything", "this will delete all files") {
		t.Error("AutoApprove should always return true")
	}
	if !AutoApprove("", "") {
		t.Error("AutoApprove should return true even with empty args")
	}
}

func TestIsDangerousKnownTools(t *testing.T) {
	dangerousTools := []struct {
		name       string
		wantDanger bool
	}{
		{"write", true},
		{"edit", true},
		{"patch", true},
		{"find_replace", true},
		{"shell", true},
		{"mkdir", true},
		{"read", false},
		{"ls", false},
		{"grep", false},
		{"glob", false},
		{"tree", false},
		{"git", false},
		{"sysinfo", false},
		{"diff", false},
		{"fetch", false},
		{"clipboard", false},
	}

	for _, tt := range dangerousTools {
		t.Run(tt.name, func(t *testing.T) {
			reason, isDangerous := IsDangerous(tt.name)
			if isDangerous != tt.wantDanger {
				t.Errorf("IsDangerous(%q) = %v, want %v", tt.name, isDangerous, tt.wantDanger)
			}
			if isDangerous && reason == "" {
				t.Errorf("IsDangerous(%q) returned empty reason for dangerous tool", tt.name)
			}
			if !isDangerous && reason != "" {
				t.Errorf("IsDangerous(%q) returned reason %q for safe tool", tt.name, reason)
			}
		})
	}
}

func TestDangerousToolsMapCompleteness(t *testing.T) {
	// Every entry in DangerousTools should have a non-empty reason
	for tool, reason := range DangerousTools {
		if reason == "" {
			t.Errorf("DangerousTools[%q] has empty reason", tool)
		}
		if tool == "" {
			t.Error("DangerousTools has entry with empty tool name")
		}
	}

	// Should have at least 6 dangerous tools
	if len(DangerousTools) < 6 {
		t.Errorf("expected at least 6 dangerous tools, got %d", len(DangerousTools))
	}
}

func TestIsDangerousUnknownTool(t *testing.T) {
	reason, isDangerous := IsDangerous("nonexistent_tool")
	if isDangerous {
		t.Error("unknown tools should not be dangerous")
	}
	if reason != "" {
		t.Errorf("unknown tools should have empty reason, got: %s", reason)
	}
}
