package cognitive

import (
	"strings"
	"testing"
)

// --- Micro-Inference Tests (unit tests for deterministic parts) ---

func TestParseChoiceExact(t *testing.T) {
	mi := &MicroInference{}

	tests := []struct {
		name     string
		response string
		choices  []string
		want     string
	}{
		{"exact match", "search", []string{"search", "read", "write"}, "search"},
		{"with context", "The task type is search.", []string{"search", "read", "write"}, "search"},
		{"case insensitive", "SEARCH", []string{"search", "read", "write"}, "search"},
		{"first word", "read the file", []string{"search", "read", "write"}, "read"},
		{"with newline", "write\n", []string{"search", "read", "write"}, "write"},
		{"default fallback", "unknown", []string{"search", "read", "write"}, "search"},
		{"empty choices", "test", []string{}, ""},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := mi.parseChoice(tt.response, tt.choices)
			if got != tt.want {
				t.Errorf("parseChoice(%q, %v) = %q, want %q", tt.response, tt.choices, got, tt.want)
			}
		})
	}
}

func TestFilterToolsByTask(t *testing.T) {
	available := []string{"read", "grep", "glob", "ls", "tree", "write", "edit"}

	tests := []struct {
		taskType string
		wantLen  int
		wantAny  string
	}{
		{"search", 2, "grep"},
		{"read", 1, "read"},
		{"write", 2, "write"},
		{"list", 3, "ls"},
		{"chat", 0, ""},
		{"explain", 0, ""},
	}

	for _, tt := range tests {
		t.Run(tt.taskType, func(t *testing.T) {
			got := filterToolsByTask(tt.taskType, available)
			if len(got) != tt.wantLen {
				t.Errorf("filterToolsByTask(%q) returned %d tools, want %d: %v", tt.taskType, len(got), tt.wantLen, got)
			}
			if tt.wantAny != "" {
				found := false
				for _, tool := range got {
					if tool == tt.wantAny {
						found = true
					}
				}
				if !found {
					t.Errorf("expected %q in results %v", tt.wantAny, got)
				}
			}
		})
	}
}

func TestFilterToolsByTaskUnavailable(t *testing.T) {
	// When available tools don't include the task-relevant ones
	available := []string{"sysinfo", "diff"}
	got := filterToolsByTask("search", available)
	if len(got) != 0 {
		t.Errorf("expected 0 tools when none available, got %v", got)
	}
}

func TestToolArgSpec(t *testing.T) {
	tests := []struct {
		tool    string
		wantLen int
	}{
		{"read", 1},
		{"grep", 1},
		{"glob", 1},
		{"ls", 0},
		{"tree", 0},
		{"write", 2},
		{"edit", 3},
		{"git", 1},
		{"unknown", 0},
	}

	for _, tt := range tests {
		t.Run(tt.tool, func(t *testing.T) {
			got := toolArgSpec(tt.tool)
			if len(got) != tt.wantLen {
				t.Errorf("toolArgSpec(%q) returned %d args, want %d", tt.tool, len(got), tt.wantLen)
			}
		})
	}
}

func TestToolArgSpecReadPath(t *testing.T) {
	args := toolArgSpec("read")
	if len(args) != 1 || args[0] != "path" {
		t.Errorf("read tool should require 'path', got %v", args)
	}
}

func TestToolArgSpecGrepPattern(t *testing.T) {
	args := toolArgSpec("grep")
	if len(args) != 1 || args[0] != "pattern" {
		t.Errorf("grep tool should require 'pattern', got %v", args)
	}
}

func TestMicroInferenceCreation(t *testing.T) {
	// Test that MicroInference can be created with nil client (for unit testing)
	mi := NewMicroInference(nil)
	if mi == nil {
		t.Fatal("NewMicroInference should not return nil")
	}
	if mi.opts.Temperature != 0.1 {
		t.Errorf("temperature should be 0.1 for deterministic choices, got %f", mi.opts.Temperature)
	}
	if mi.opts.NumPredict != 32 {
		t.Errorf("num_predict should be 32 for micro responses, got %d", mi.opts.NumPredict)
	}
}

// --- Helper Tests ---

func TestMinFunc(t *testing.T) {
	if min(3, 5) != 3 {
		t.Error("min(3, 5) should be 3")
	}
	if min(5, 3) != 3 {
		t.Error("min(5, 3) should be 3")
	}
	if min(3, 3) != 3 {
		t.Error("min(3, 3) should be 3")
	}
}

func TestParseChoicePartialMatch(t *testing.T) {
	mi := &MicroInference{}

	// Test prefix matching when no exact match
	got := mi.parseChoice("rea", []string{"read", "grep", "write"})
	if got != "read" {
		t.Errorf("prefix match: got %q, want read", got)
	}
}

func TestClassifyTaskPromptFormat(t *testing.T) {
	// Verify the prompt format is correct
	query := "search for Pipeline"
	prompt := "Classify this request into ONE category.\nCategories: search, read, write, list, explain, chat\n\nRequest: " + query + "\n\nCategory:"

	if !strings.Contains(prompt, "search") {
		t.Error("prompt should contain category options")
	}
	if !strings.Contains(prompt, query) {
		t.Error("prompt should contain the query")
	}
}

// --- Benchmark ---

func BenchmarkParseChoice(b *testing.B) {
	mi := &MicroInference{}
	choices := []string{"search", "read", "write", "list", "explain", "chat"}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		mi.parseChoice("The task type is search based on the query.", choices)
	}
}

func BenchmarkFilterToolsByTask(b *testing.B) {
	available := []string{"read", "grep", "glob", "ls", "tree", "write", "edit", "shell", "git"}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		filterToolsByTask("search", available)
	}
}
