package tools

import (
	"strings"
	"testing"
)

func TestNewRegistry(t *testing.T) {
	r := NewRegistry()
	if r == nil {
		t.Fatal("NewRegistry returned nil")
	}
}

func TestRegisterAndGet(t *testing.T) {
	r := NewRegistry()

	r.Register(Tool{
		Name:        "test_tool",
		Description: "A test tool",
		Execute: func(args map[string]string) (string, error) {
			return "executed", nil
		},
	})

	tool, err := r.Get("test_tool")
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if tool.Name != "test_tool" {
		t.Errorf("expected name 'test_tool', got %q", tool.Name)
	}
	if tool.Description != "A test tool" {
		t.Errorf("expected description 'A test tool', got %q", tool.Description)
	}
}

func TestGetExecute(t *testing.T) {
	r := NewRegistry()
	r.Register(Tool{
		Name:        "echo",
		Description: "Echo back",
		Execute: func(args map[string]string) (string, error) {
			return args["msg"], nil
		},
	})

	tool, _ := r.Get("echo")
	result, err := tool.Execute(map[string]string{"msg": "hello"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result != "hello" {
		t.Errorf("expected 'hello', got %q", result)
	}
}

func TestGetNonExistent(t *testing.T) {
	r := NewRegistry()

	_, err := r.Get("nonexistent")
	if err == nil {
		t.Fatal("expected error for nonexistent tool")
	}
	if !strings.Contains(err.Error(), "unknown tool") {
		t.Errorf("expected 'unknown tool' in error, got %q", err.Error())
	}
}

func TestRegisterOverwrite(t *testing.T) {
	r := NewRegistry()

	r.Register(Tool{Name: "tool", Description: "v1"})
	r.Register(Tool{Name: "tool", Description: "v2"})

	tool, _ := r.Get("tool")
	if tool.Description != "v2" {
		t.Errorf("expected description 'v2' after re-register, got %q", tool.Description)
	}
}

func TestList(t *testing.T) {
	r := NewRegistry()
	r.Register(Tool{Name: "alpha", Description: "first"})
	r.Register(Tool{Name: "beta", Description: "second"})
	r.Register(Tool{Name: "gamma", Description: "third"})

	tools := r.List()
	if len(tools) != 3 {
		t.Fatalf("expected 3 tools, got %d", len(tools))
	}

	names := map[string]bool{}
	for _, tool := range tools {
		names[tool.Name] = true
	}
	for _, name := range []string{"alpha", "beta", "gamma"} {
		if !names[name] {
			t.Errorf("expected tool %q in list", name)
		}
	}
}

func TestListEmpty(t *testing.T) {
	r := NewRegistry()
	tools := r.List()
	if len(tools) != 0 {
		t.Errorf("expected 0 tools, got %d", len(tools))
	}
}

func TestDescribe(t *testing.T) {
	r := NewRegistry()
	r.Register(Tool{Name: "read", Description: "Read a file"})
	r.Register(Tool{Name: "write", Description: "Write a file"})

	desc := r.Describe()

	if !strings.Contains(desc, "read") {
		t.Error("expected description to contain 'read'")
	}
	if !strings.Contains(desc, "Read a file") {
		t.Error("expected description to contain 'Read a file'")
	}
	if !strings.Contains(desc, "write") {
		t.Error("expected description to contain 'write'")
	}
	if !strings.Contains(desc, "Write a file") {
		t.Error("expected description to contain 'Write a file'")
	}
}

func TestDescribeEmpty(t *testing.T) {
	r := NewRegistry()
	desc := r.Describe()
	if desc != "" {
		t.Errorf("expected empty description for empty registry, got %q", desc)
	}
}

func TestDescribeFormat(t *testing.T) {
	r := NewRegistry()
	r.Register(Tool{Name: "tool1", Description: "desc1"})

	desc := r.Describe()
	if !strings.Contains(desc, "- tool1: desc1") {
		t.Errorf("expected '- tool1: desc1' in description, got %q", desc)
	}
}
