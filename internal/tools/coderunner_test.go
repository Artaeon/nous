package tools

import (
	"os/exec"
	"strings"
	"testing"
	"time"
)

func TestDetectLanguage(t *testing.T) {
	tests := []struct {
		name string
		code string
		want string
	}{
		{"python import", "import os\nprint(os.getcwd())", "python"},
		{"python from import", "from sys import argv", "python"},
		{"python def", "def main():\n  pass", "python"},
		{"python class", "class Foo:\n  pass", "python"},
		{"python print", "x = 1\nprint(x)", "python"},
		{"bash shebang", "#!/bin/bash\necho hello", "bash"},
		{"node console.log", "const x = 1;\nconsole.log(x);", "node"},
		{"node const", "const fs = require('fs');", "node"},
		{"node let", "let x = 42;", "node"},
		{"node var", "var x = 42;", "node"},
		{"node arrow", "const f = () => { return 1; };", "node"},
		{"default bash", "echo hello", "bash"},
		{"empty", "", "bash"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := DetectLanguage(tt.code)
			if got != tt.want {
				t.Errorf("DetectLanguage(%q) = %q, want %q", tt.code, got, tt.want)
			}
		})
	}
}

func TestRunCode_Python(t *testing.T) {
	if _, err := exec.LookPath("python3"); err != nil {
		t.Skip("python3 not available")
	}

	result, err := RunCode("python", "print('hello from python')", 5*time.Second)
	if err != nil {
		t.Fatalf("RunCode(python) error: %v", err)
	}
	if !strings.Contains(result, "hello from python") {
		t.Errorf("expected python output, got %q", result)
	}
	if !strings.HasPrefix(result, "Output:\n") {
		t.Errorf("expected 'Output:' prefix, got %q", result)
	}
}

func TestRunCode_Bash(t *testing.T) {
	result, err := RunCode("bash", "echo 'hello from bash'", 5*time.Second)
	if err != nil {
		t.Fatalf("RunCode(bash) error: %v", err)
	}
	if !strings.Contains(result, "hello from bash") {
		t.Errorf("expected bash output, got %q", result)
	}
}

func TestRunCode_Node(t *testing.T) {
	if _, err := exec.LookPath("node"); err != nil {
		t.Skip("node not available")
	}

	result, err := RunCode("node", "console.log('hello from node')", 5*time.Second)
	if err != nil {
		t.Fatalf("RunCode(node) error: %v", err)
	}
	if !strings.Contains(result, "hello from node") {
		t.Errorf("expected node output, got %q", result)
	}
}

func TestRunCode_Stderr(t *testing.T) {
	result, err := RunCode("bash", "echo out; echo err >&2", 5*time.Second)
	if err != nil {
		t.Fatalf("RunCode error: %v", err)
	}
	if !strings.Contains(result, "Output:\n") {
		t.Error("expected Output section")
	}
	if !strings.Contains(result, "Errors:\n") {
		t.Error("expected Errors section")
	}
}

func TestRunCode_Timeout(t *testing.T) {
	_, err := RunCode("bash", "sleep 30", 100*time.Millisecond)
	if err == nil {
		t.Error("expected timeout error")
	}
	if !strings.Contains(err.Error(), "timed out") {
		t.Errorf("expected timeout message, got %v", err)
	}
}

func TestRunCode_EmptyCode(t *testing.T) {
	_, err := RunCode("bash", "", 5*time.Second)
	if err == nil {
		t.Error("expected error for empty code")
	}
}

func TestRunCode_UnsupportedLanguage(t *testing.T) {
	_, err := RunCode("cobol", "DISPLAY 'HELLO'", 5*time.Second)
	if err == nil {
		t.Error("expected error for unsupported language")
	}
	if !strings.Contains(err.Error(), "unsupported language") {
		t.Errorf("expected unsupported language error, got %v", err)
	}
}

func TestRunCode_OutputTruncation(t *testing.T) {
	// Generate output longer than maxCodeOutput
	code := "for i in $(seq 1 2000); do echo 'line number padded to be long enough for truncation testing abcdefghij'; done"
	result, err := RunCode("bash", code, 5*time.Second)
	if err != nil {
		t.Fatalf("RunCode error: %v", err)
	}
	if len(result) > maxCodeOutput+100 { // some slack for formatting
		t.Errorf("output should be truncated, got %d chars", len(result))
	}
}

func TestFormatCodeOutput(t *testing.T) {
	tests := []struct {
		name   string
		stdout string
		stderr string
		want   string
	}{
		{"stdout only", "hello", "", "Output:\nhello"},
		{"stderr only", "", "error", "Errors:\nerror"},
		{"both", "out", "err", "Output:\nout\nErrors:\nerr"},
		{"empty", "", "", "(no output)"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := formatCodeOutput(tt.stdout, tt.stderr)
			if got != tt.want {
				t.Errorf("formatCodeOutput(%q, %q) = %q, want %q", tt.stdout, tt.stderr, got, tt.want)
			}
		})
	}
}

func TestRegisterCodeRunnerTools(t *testing.T) {
	r := NewRegistry()
	RegisterCodeRunnerTools(r)

	tool, err := r.Get("coderunner")
	if err != nil {
		t.Fatalf("coderunner tool not registered: %v", err)
	}

	// Missing code should error
	_, err = tool.Execute(map[string]string{})
	if err == nil {
		t.Error("expected error for missing code")
	}

	// Basic execution with auto-detect
	result, err := tool.Execute(map[string]string{"code": "echo test123"})
	if err != nil {
		t.Fatalf("execute error: %v", err)
	}
	if !strings.Contains(result, "test123") {
		t.Errorf("expected output containing 'test123', got %q", result)
	}
}
