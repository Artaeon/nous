package cli

import (
	"io"
	"strings"
	"testing"
)

func TestReadLineSingle(t *testing.T) {
	reader := strings.NewReader("hello world\n")
	h := NewHistory("/dev/null", 100)
	inp := NewInput(h, nil, reader)

	line, err := inp.ReadLine("")
	if err != nil {
		t.Fatalf("ReadLine: %v", err)
	}
	if line != "hello world" {
		t.Errorf("line = %q, want %q", line, "hello world")
	}
}

func TestReadLineMultiline(t *testing.T) {
	// Backslash continuation
	reader := strings.NewReader("first line \\\nsecond line \\\nthird line\n")
	h := NewHistory("/dev/null", 100)
	inp := NewInput(h, nil, reader)

	line, err := inp.ReadLine("")
	if err != nil {
		t.Fatalf("ReadLine: %v", err)
	}
	expected := "first line \nsecond line \nthird line"
	if line != expected {
		t.Errorf("line = %q, want %q", line, expected)
	}
}

func TestReadLineEOF(t *testing.T) {
	reader := strings.NewReader("")
	h := NewHistory("/dev/null", 100)
	inp := NewInput(h, nil, reader)

	_, err := inp.ReadLine("")
	if err != io.EOF {
		t.Errorf("err = %v, want io.EOF", err)
	}
}

func TestReadLineEOFDuringContinuation(t *testing.T) {
	// EOF after a continuation line should return what was accumulated
	reader := strings.NewReader("partial \\\nmore")
	h := NewHistory("/dev/null", 100)
	inp := NewInput(h, nil, reader)

	line, err := inp.ReadLine("")
	if err != nil {
		t.Fatalf("ReadLine: %v", err)
	}
	if line != "partial \nmore" {
		t.Errorf("line = %q, want %q", line, "partial \nmore")
	}
}

func TestSetCompletions(t *testing.T) {
	h := NewHistory("/dev/null", 100)
	inp := NewInput(h, []string{"/help", "/quit"}, strings.NewReader(""))

	if len(inp.Completions()) != 2 {
		t.Errorf("completions = %d, want 2", len(inp.Completions()))
	}

	inp.SetCompletions([]string{"/new"})
	if len(inp.Completions()) != 1 {
		t.Errorf("completions after set = %d, want 1", len(inp.Completions()))
	}
}

func TestMultipleReadLines(t *testing.T) {
	reader := strings.NewReader("line1\nline2\nline3\n")
	h := NewHistory("/dev/null", 100)
	inp := NewInput(h, nil, reader)

	for _, want := range []string{"line1", "line2", "line3"} {
		got, err := inp.ReadLine("")
		if err != nil {
			t.Fatalf("ReadLine: %v", err)
		}
		if got != want {
			t.Errorf("got = %q, want %q", got, want)
		}
	}

	// Next read should be EOF
	_, err := inp.ReadLine("")
	if err != io.EOF {
		t.Errorf("expected EOF, got %v", err)
	}
}
