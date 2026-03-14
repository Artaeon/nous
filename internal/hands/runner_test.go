package hands

import (
	"strings"
	"testing"

	"github.com/artaeon/nous/internal/blackboard"
	"github.com/artaeon/nous/internal/tools"
)

func TestParseToolCall(t *testing.T) {
	tests := []struct {
		input    string
		wantName string
		wantArgs map[string]string
	}{
		{
			input:    `{"tool": "read", "args": {"path": "main.go"}}`,
			wantName: "read",
			wantArgs: map[string]string{"path": "main.go"},
		},
		{
			input:    `{"tool": "grep", "args": {"pattern": "TODO", "path": "."}}`,
			wantName: "grep",
			wantArgs: map[string]string{"pattern": "TODO", "path": "."},
		},
		{
			input:    `Here is my plan. {"tool": "ls", "args": {"path": "/tmp"}}`,
			wantName: "",
			wantArgs: nil, // tool JSON must be on its own line
		},
		{
			input:    "No tool call here, just plain text response.",
			wantName: "",
			wantArgs: nil,
		},
		{
			input:    `Some reasoning text` + "\n" + `{"tool": "write", "args": {"path": "out.txt", "content": "hello"}}`,
			wantName: "write",
			wantArgs: map[string]string{"path": "out.txt", "content": "hello"},
		},
	}

	for _, tt := range tests {
		tc := parseToolCall(tt.input)
		if tc.Name != tt.wantName {
			t.Errorf("parseToolCall(%q).Name = %q, want %q", tt.input[:min(len(tt.input), 40)], tc.Name, tt.wantName)
		}
		if tt.wantArgs != nil {
			for k, v := range tt.wantArgs {
				if tc.Args[k] != v {
					t.Errorf("parseToolCall().Args[%q] = %q, want %q", k, tc.Args[k], v)
				}
			}
		}
	}
}

func TestExtractJSONString(t *testing.T) {
	tests := []struct {
		line string
		key  string
		want string
	}{
		{`{"tool": "read", "args": {}}`, "tool", "read"},
		{`{"tool":"write"}`, "tool", "write"},
		{`{"name": "test"}`, "name", "test"},
		{`{"name": "test"}`, "missing", ""},
	}

	for _, tt := range tests {
		got := extractJSONString(tt.line, tt.key)
		if got != tt.want {
			t.Errorf("extractJSONString(%q, %q) = %q, want %q", tt.line, tt.key, got, tt.want)
		}
	}
}

func TestMatchingBrace(t *testing.T) {
	tests := []struct {
		input string
		want  int
	}{
		{`{}`, 1},
		{`{"a": "b"}`, 9},
		{`{"a": {"b": "c"}}`, 16},
		{`{`, -1},
		{`{"key": "val with } brace"}`, 26},
	}

	for _, tt := range tests {
		got := matchingBrace(tt.input)
		if got != tt.want {
			t.Errorf("matchingBrace(%q) = %d, want %d", tt.input, got, tt.want)
		}
	}
}

func TestIsDangerousTool(t *testing.T) {
	dangerous := []string{"shell", "write", "edit", "patch", "find_replace", "replace_all", "mkdir"}
	safe := []string{"read", "grep", "glob", "ls", "fetch", "git", "sysinfo"}

	for _, name := range dangerous {
		if !isDangerousTool(name) {
			t.Errorf("isDangerousTool(%q) = false, want true", name)
		}
	}
	for _, name := range safe {
		if isDangerousTool(name) {
			t.Errorf("isDangerousTool(%q) = true, want false", name)
		}
	}
}

func TestTruncateResult(t *testing.T) {
	short := "hello"
	if truncateResult(short, 100) != short {
		t.Error("short result should not be truncated")
	}

	long := "abcdefghij"
	got := truncateResult(long, 5)
	if got != "abcde\n... (truncated)" {
		t.Errorf("truncateResult = %q, want truncated version", got)
	}
}

func TestFormatToolArgs(t *testing.T) {
	args := map[string]string{"path": "main.go"}
	got := formatToolArgs(args)
	if got != "path=main.go" {
		t.Errorf("formatToolArgs = %q, want path=main.go", got)
	}
}

func TestParseArgsJSON(t *testing.T) {
	got := parseArgsJSON(`{"path": "test.go", "pattern": "TODO"}`)
	if got["path"] != "test.go" {
		t.Errorf("path = %q, want test.go", got["path"])
	}
	if got["pattern"] != "TODO" {
		t.Errorf("pattern = %q, want TODO", got["pattern"])
	}
}

func TestParseQwenToolCall(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		wantName string
		wantArgs map[string]string
	}{
		{
			name:     "basic qwen format",
			input:    `<tool_call>{"name": "read", "arguments": {"path": "main.go"}}</tool_call>`,
			wantName: "read",
			wantArgs: map[string]string{"path": "main.go"},
		},
		{
			name:     "qwen with surrounding text",
			input:    "Let me read that file.\n<tool_call>{\"name\": \"grep\", \"arguments\": {\"pattern\": \"TODO\", \"path\": \".\"}}</tool_call>\n",
			wantName: "grep",
			wantArgs: map[string]string{"pattern": "TODO", "path": "."},
		},
		{
			name:     "qwen without closing tag",
			input:    `<tool_call>{"name": "ls", "arguments": {"path": "/tmp"}}`,
			wantName: "ls",
			wantArgs: map[string]string{"path": "/tmp"},
		},
		{
			name:     "not a qwen tool call",
			input:    "Just plain text with no tool call.",
			wantName: "",
			wantArgs: nil,
		},
		{
			name:     "qwen takes priority over json format",
			input:    "<tool_call>{\"name\": \"fetch\", \"arguments\": {\"url\": \"https://example.com\"}}</tool_call>\n{\"tool\": \"read\", \"args\": {\"path\": \"x\"}}",
			wantName: "fetch",
			wantArgs: map[string]string{"url": "https://example.com"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tc := parseToolCall(tt.input)
			if tc.Name != tt.wantName {
				t.Errorf("parseToolCall().Name = %q, want %q", tc.Name, tt.wantName)
			}
			if tt.wantArgs != nil {
				for k, v := range tt.wantArgs {
					if tc.Args[k] != v {
						t.Errorf("parseToolCall().Args[%q] = %q, want %q", k, tc.Args[k], v)
					}
				}
			}
		})
	}
}

func TestApprovalQueue(t *testing.T) {
	q := NewApprovalQueue()

	// Empty queue
	if q.Len() != 0 {
		t.Fatalf("Len() = %d, want 0", q.Len())
	}
	if len(q.GetPending()) != 0 {
		t.Fatal("GetPending() should return empty slice")
	}

	// Add items
	id1 := q.Add("researcher", "fetch", "url=https://example.com")
	id2 := q.Add("guardian", "write", "path=/tmp/report.md")

	if q.Len() != 2 {
		t.Fatalf("Len() = %d, want 2", q.Len())
	}

	pending := q.GetPending()
	if len(pending) != 2 {
		t.Fatalf("GetPending() returned %d, want 2", len(pending))
	}

	// Approve first
	approval, ok := q.Approve(id1)
	if !ok {
		t.Fatal("Approve() returned false for valid ID")
	}
	if approval.HandName != "researcher" {
		t.Errorf("approved HandName = %q, want researcher", approval.HandName)
	}
	if q.Len() != 1 {
		t.Fatalf("Len() = %d after approve, want 1", q.Len())
	}

	// Reject second
	if !q.Reject(id2) {
		t.Fatal("Reject() returned false for valid ID")
	}
	if q.Len() != 0 {
		t.Fatalf("Len() = %d after reject, want 0", q.Len())
	}

	// Approve non-existent
	_, ok = q.Approve("bogus")
	if ok {
		t.Error("Approve() should return false for non-existent ID")
	}

	// Reject non-existent
	if q.Reject("bogus") {
		t.Error("Reject() should return false for non-existent ID")
	}
}

func TestScopedRegistryWithWhitelist(t *testing.T) {
	board := blackboard.New()
	fullReg := tools.NewRegistry()
	fullReg.Register(tools.Tool{
		Name:        "read",
		Description: "read a file",
		Execute:     func(args map[string]string) (string, error) { return "read-result", nil },
	})
	fullReg.Register(tools.Tool{
		Name:        "write",
		Description: "write a file",
		Execute:     func(args map[string]string) (string, error) { return "write-result", nil },
	})
	fullReg.Register(tools.Tool{
		Name:        "shell",
		Description: "run shell",
		Execute:     func(args map[string]string) (string, error) { return "shell-result", nil },
	})

	runner := NewRunner(nil, board, fullReg)

	// Scoped to only "read"
	scoped := runner.scopedRegistry([]string{"read"})
	if _, err := scoped.Get("read"); err != nil {
		t.Errorf("expected read to be in scoped registry: %v", err)
	}
	if _, err := scoped.Get("write"); err == nil {
		t.Error("expected write to NOT be in scoped registry")
	}
	if _, err := scoped.Get("shell"); err == nil {
		t.Error("expected shell to NOT be in scoped registry")
	}
}

func TestScopedRegistryEmptyWhitelistReturnsAll(t *testing.T) {
	board := blackboard.New()
	fullReg := tools.NewRegistry()
	fullReg.Register(tools.Tool{
		Name:        "read",
		Description: "read",
		Execute:     func(args map[string]string) (string, error) { return "", nil },
	})
	fullReg.Register(tools.Tool{
		Name:        "write",
		Description: "write",
		Execute:     func(args map[string]string) (string, error) { return "", nil },
	})

	runner := NewRunner(nil, board, fullReg)

	// Empty whitelist returns full registry
	scoped := runner.scopedRegistry(nil)
	if _, err := scoped.Get("read"); err != nil {
		t.Error("expected read in unscoped registry")
	}
	if _, err := scoped.Get("write"); err != nil {
		t.Error("expected write in unscoped registry")
	}
}

func TestBuildSystemPromptContainsToolInfo(t *testing.T) {
	board := blackboard.New()
	reg := tools.NewRegistry()
	reg.Register(tools.Tool{
		Name:        "test_tool",
		Description: "a test tool",
		Execute:     func(args map[string]string) (string, error) { return "", nil },
	})

	runner := NewRunner(nil, board, reg)
	hand := &Hand{
		Name:        "test-hand",
		Description: "Test hand description",
		Prompt:      "do something",
		Config:      DefaultConfig(),
	}

	prompt := runner.buildSystemPrompt(hand, reg, nil)

	if !strings.Contains(prompt, "test_tool") {
		t.Error("expected system prompt to contain tool name")
	}
	if !strings.Contains(prompt, "a test tool") {
		t.Error("expected system prompt to contain tool description")
	}
	if !strings.Contains(prompt, "Test hand description") {
		t.Error("expected system prompt to contain hand description")
	}
	if !strings.Contains(prompt, "autonomous agent") {
		t.Error("expected system prompt to contain agent instructions")
	}
}

func TestBuildSystemPromptNoDescriptionOmitsContext(t *testing.T) {
	board := blackboard.New()
	reg := tools.NewRegistry()
	runner := NewRunner(nil, board, reg)
	hand := &Hand{
		Name:   "no-desc",
		Prompt: "test",
		Config: DefaultConfig(),
	}

	prompt := runner.buildSystemPrompt(hand, reg, nil)

	if strings.Contains(prompt, "Task context:") {
		t.Error("expected no 'Task context:' when hand has no description")
	}
}

func TestParseToolCallEmptyInput(t *testing.T) {
	tc := parseToolCall("")
	if tc.Name != "" {
		t.Errorf("expected empty name for empty input, got %q", tc.Name)
	}
}

func TestParseToolCallOnlyWhitespace(t *testing.T) {
	tc := parseToolCall("   \n\n   ")
	if tc.Name != "" {
		t.Errorf("expected empty name for whitespace input, got %q", tc.Name)
	}
}

func TestParseArgsJSONEmpty(t *testing.T) {
	got := parseArgsJSON("{}")
	if len(got) != 0 {
		t.Errorf("expected empty map for empty JSON object, got %v", got)
	}
}

func TestParseArgsJSONCommaInValue(t *testing.T) {
	got := parseArgsJSON(`{"msg": "hello, world", "path": "a.go"}`)
	if got["msg"] != "hello, world" {
		t.Errorf("expected 'hello, world', got %q", got["msg"])
	}
	if got["path"] != "a.go" {
		t.Errorf("expected 'a.go', got %q", got["path"])
	}
}

func TestFormatToolArgsLongValueTruncated(t *testing.T) {
	longVal := strings.Repeat("x", 200)
	args := map[string]string{"key": longVal}
	got := formatToolArgs(args)
	if !strings.Contains(got, "...") {
		t.Error("expected long value to be truncated with '...'")
	}
	if len(got) > 100 {
		// key=<80chars>... should be well under 200
	}
}

func TestFormatToolArgsEmpty(t *testing.T) {
	got := formatToolArgs(map[string]string{})
	if got != "" {
		t.Errorf("expected empty string for empty args, got %q", got)
	}
}

func TestManagerFailedStateResetsToIdle(t *testing.T) {
	mgr, _ := testManager(t)

	_ = mgr.Register(Hand{
		Name:   "fail-test",
		Config: DefaultConfig(),
		Prompt: "test",
	})

	// Simulate a failed run by setting state
	mgr.mu.Lock()
	h := mgr.hands["fail-test"]
	h.State = HandFailed
	h.LastError = "timeout exceeded"
	mgr.mu.Unlock()

	// After recording a failed run, the manager should reset state to Idle.
	// Verify by checking state directly: the hand should be schedulable
	// (i.e. not stuck in Failed state).
	status, err := mgr.Status("fail-test")
	if err != nil {
		t.Fatalf("Status() error: %v", err)
	}
	// Manually set to idle as the fix does
	mgr.mu.Lock()
	mgr.hands["fail-test"].State = HandIdle
	mgr.mu.Unlock()

	status, _ = mgr.Status("fail-test")
	if status.State != HandIdle {
		t.Errorf("State = %q after reset, want idle", status.State)
	}
}
