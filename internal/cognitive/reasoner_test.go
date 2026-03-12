package cognitive

import (
	"context"
	"strings"
	"testing"
	"time"

	"github.com/artaeon/nous/internal/blackboard"
	"github.com/artaeon/nous/internal/ollama"
	"github.com/artaeon/nous/internal/tools"
)

func TestPublishAnswerSetsDefaultAndRequestSpecificKeys(t *testing.T) {
	board := blackboard.New()
	board.Set("answer_key", "last_answer_42")

	r := &Reasoner{Base: Base{Board: board}}
	r.publishAnswer("done")

	if got, ok := board.Get("last_answer"); !ok || got.(string) != "done" {
		t.Fatalf("expected default answer key to be populated, got %v %t", got, ok)
	}
	if got, ok := board.Get("last_answer_42"); !ok || got.(string) != "done" {
		t.Fatalf("expected request-specific answer key to be populated, got %v %t", got, ok)
	}
}

func TestPublishAnswerWithoutRequestSpecificKey(t *testing.T) {
	board := blackboard.New()
	r := &Reasoner{Base: Base{Board: board}}

	r.publishAnswer("fallback only")

	if got, ok := board.Get("last_answer"); !ok || got.(string) != "fallback only" {
		t.Fatalf("expected default answer key, got %v %t", got, ok)
	}
}

func TestCompactSystemPromptIsAssistantFirst(t *testing.T) {
	r := &Reasoner{Tools: tools.NewRegistry()}
	prompt := r.compactSystemPrompt()
	checks := []string{"local personal assistant", "assistant memory", "NEVER guess what code does", "warm, natural", "next small step"}
	for _, check := range checks {
		if !strings.Contains(prompt, check) {
			t.Fatalf("compactSystemPrompt() should contain %q", check)
		}
	}
}

func TestRecentConversationContextKeepsLatestTurns(t *testing.T) {
	msgs := []ollama.Message{
		{Role: "system", Content: "sys"},
		{Role: "user", Content: "I want a calmer day today."},
		{Role: "assistant", Content: "Let's reduce things to one step first."},
		{Role: "user", Content: "okay\nwith extra detail"},
		{Role: "assistant", Content: "Start with the report before lunch."},
	}

	got := recentConversationContext(msgs, 3)
	checks := []string{"Assistant: Let's reduce things to one step first.", "User: okay", "Assistant: Start with the report before lunch."}
	for _, check := range checks {
		if !strings.Contains(got, check) {
			t.Fatalf("recentConversationContext() should contain %q, got %q", check, got)
		}
	}
	if strings.Contains(got, "with extra detail") {
		t.Fatalf("recentConversationContext() should compact multiline content, got %q", got)
	}
}

func TestDeterministicAssistantAnswersAreStoredInConversation(t *testing.T) {
	board := blackboard.New()
	r := NewReasoner(board, nil, tools.NewRegistry())
	r.AssistantAnswer = func(input string, recent string) (string, bool) {
		return "I remember the thread.", true
	}

	err := r.reason(context.Background(), blackboard.Percept{Raw: "tell me more", Timestamp: time.Now()})
	if err != nil {
		t.Fatalf("reason() error = %v", err)
	}
	msgs := r.Conv.Messages()
	if len(msgs) < 2 {
		t.Fatalf("expected conversation to contain deterministic turn, got %d messages", len(msgs))
	}
	if msgs[len(msgs)-2].Role != "user" || msgs[len(msgs)-2].Content != "tell me more" {
		t.Fatalf("expected stored user turn, got %+v", msgs[len(msgs)-2])
	}
	if msgs[len(msgs)-1].Role != "assistant" || msgs[len(msgs)-1].Content != "I remember the thread." {
		t.Fatalf("expected stored assistant turn, got %+v", msgs[len(msgs)-1])
	}
}

func TestCorrectArgNames(t *testing.T) {
	tests := []struct {
		name     string
		tool     string
		args     map[string]string
		expected map[string]string
	}{
		{
			name:     "file → path for read",
			tool:     "read",
			args:     map[string]string{"file": "/tmp/foo.go"},
			expected: map[string]string{"path": "/tmp/foo.go"},
		},
		{
			name:     "query → pattern for grep",
			tool:     "grep",
			args:     map[string]string{"query": "func main"},
			expected: map[string]string{"pattern": "func main"},
		},
		{
			name:     "old_text/new_text → old/new for edit",
			tool:     "edit",
			args:     map[string]string{"file": "x.go", "old_text": "a", "new_text": "b"},
			expected: map[string]string{"path": "x.go", "old": "a", "new": "b"},
		},
		{
			name:     "cmd → command for shell",
			tool:     "shell",
			args:     map[string]string{"cmd": "ls -la"},
			expected: map[string]string{"command": "ls -la"},
		},
		{
			name:     "already correct args unchanged",
			tool:     "read",
			args:     map[string]string{"path": "/tmp/foo.go"},
			expected: map[string]string{"path": "/tmp/foo.go"},
		},
		{
			name:     "unknown tool passes through",
			tool:     "custom_tool",
			args:     map[string]string{"file": "x.go"},
			expected: map[string]string{"file": "x.go"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := correctArgNames(tt.tool, tt.args)
			for k, v := range tt.expected {
				if got[k] != v {
					t.Errorf("expected args[%q]=%q, got %q (full: %v)", k, v, got[k], got)
				}
			}
			if len(got) != len(tt.expected) {
				t.Errorf("expected %d args, got %d: %v", len(tt.expected), len(got), got)
			}
		})
	}
}

func TestValidateRequiredArgs(t *testing.T) {
	tests := []struct {
		name    string
		tool    string
		args    map[string]string
		missing string
	}{
		{"read with path", "read", map[string]string{"path": "/tmp/x"}, ""},
		{"read missing path", "read", map[string]string{}, "path"},
		{"read empty path", "read", map[string]string{"path": "  "}, "path"},
		{"write with all", "write", map[string]string{"path": "x", "content": "y"}, ""},
		{"write missing content", "write", map[string]string{"path": "x"}, "content"},
		{"grep with pattern", "grep", map[string]string{"pattern": "foo"}, ""},
		{"grep missing pattern", "grep", map[string]string{"path": "."}, "pattern"},
		{"edit missing old", "edit", map[string]string{"path": "x", "new": "y"}, "old"},
		{"unknown tool always ok", "unknown", map[string]string{}, ""},
		{"sysinfo no args ok", "sysinfo", map[string]string{}, ""},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := validateRequiredArgs(tt.tool, tt.args)
			if got != tt.missing {
				t.Errorf("expected missing=%q, got %q", tt.missing, got)
			}
		})
	}
}

func TestInferRequestedPath(t *testing.T) {
	tests := []struct {
		name    string
		input   string
		content string
		want    string
	}{
		{"explicit filename", "create a file named bitcoin.md in this folder", "", "bitcoin.md"},
		{"markdown about topic", "create a markdown file about bitcoin in this folder", "", "bitcoin.md"},
		{"from heading", "create a markdown file in this folder", "# Bitcoin\n\nSummary", "bitcoin.md"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := inferRequestedPath(tt.input, tt.content); got != tt.want {
				t.Fatalf("inferRequestedPath() = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestInferMissingToolArgsForWrite(t *testing.T) {
	args := map[string]string{"content": "# Bitcoin\nSummary"}
	inferMissingToolArgs("write", args, "Research Bitcoin online and create a file named bitcoin.md in the current folder")
	if args["path"] != "bitcoin.md" {
		t.Fatalf("expected inferred path bitcoin.md, got %v", args)
	}
}

func TestNormalizeRequestedPathForCurrentFolder(t *testing.T) {
	args := map[string]string{"path": "/bitcoin.md"}
	inferMissingToolArgs("write", args, "Create bitcoin.md in the current folder")
	if args["path"] != "bitcoin.md" {
		t.Fatalf("expected normalized relative path, got %v", args)
	}
}

func TestTryResearchAndWrite(t *testing.T) {
	board := blackboard.New()
	reg := tools.NewRegistry()

	wrote := map[string]string{}
	reg.Register(tools.Tool{
		Name: "fetch",
		Execute: func(args map[string]string) (string, error) {
			return "Bitcoin is a decentralized digital currency. It runs on a blockchain. It is used as a peer-to-peer payment system.", nil
		},
	})
	reg.Register(tools.Tool{
		Name: "write",
		Execute: func(args map[string]string) (string, error) {
			wrote["path"] = args["path"]
			wrote["content"] = args["content"]
			return "ok", nil
		},
	})

	r := NewReasoner(board, nil, reg)
	r.Confirm = AutoApprove
	r.currentInput = "Research Bitcoin online and create a file named bitcoin.md in the current folder"

	answer, ok, err := r.tryResearchAndWrite(r.currentInput)
	if err != nil {
		t.Fatalf("tryResearchAndWrite() error = %v", err)
	}
	if !ok {
		t.Fatal("expected helper to handle request")
	}
	if !strings.Contains(answer, "bitcoin.md") {
		t.Fatalf("expected answer to mention created file, got %q", answer)
	}
	if wrote["path"] != "bitcoin.md" {
		t.Fatalf("expected file path bitcoin.md, got %v", wrote)
	}
	if !strings.Contains(wrote["content"], "# Bitcoin") || !strings.Contains(wrote["content"], "## Sources") {
		t.Fatalf("expected markdown content, got %q", wrote["content"])
	}
}

func TestIsCodeQuery(t *testing.T) {
	codeQueries := []string{
		"what does correctArgNames do",
		"explain the function renderBriefing",
		"how does semantic ranking work in working.go",
		"read file internal/cognitive/reasoner.go",
		"show me the struct definition for Pipeline",
		"where is SemanticSearch defined",
		"what is the implementation of preGroundCodeQuery",
	}
	for _, q := range codeQueries {
		if !isCodeQuery(q) {
			t.Errorf("isCodeQuery(%q) should be true", q)
		}
	}

	nonCodeQueries := []string{
		"what should I do now",
		"good morning",
		"set my preferred language to german",
		"what reminders do I have",
		"hello",
	}
	for _, q := range nonCodeQueries {
		if isCodeQuery(q) {
			t.Errorf("isCodeQuery(%q) should be false", q)
		}
	}
}

func TestPreGroundSkipsNonCodeQueries(t *testing.T) {
	r := &Reasoner{Tools: tools.NewRegistry()}
	pipe := NewPipeline("what should I do today")
	hint := r.preGroundCodeQuery("what should I do today", pipe)
	if pipe.StepCount() != 0 {
		t.Errorf("expected 0 steps for non-code query, got %d", pipe.StepCount())
	}
	if hint != "" {
		t.Errorf("expected empty hint for non-code query, got %q", hint)
	}
}

func TestLooksLikeFailedToolCall(t *testing.T) {
	tests := []struct {
		name     string
		response string
		expected bool
	}{
		{
			"malformed tool JSON",
			`I'll read the file for you. {"tool": "read", args: {path: "main.go"}}`,
			true,
		},
		{
			"has action key",
			`Let me help. {"action": "grep", "pattern": "main"}`,
			true,
		},
		{
			"plain text answer",
			"The project uses Go 1.22 and has 6 packages.",
			false,
		},
		{
			"braces but no tool hint",
			"The function signature is func main() {}",
			false,
		},
		{
			"function key in braces",
			`{"function": "read", "args": {}}`,
			true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := looksLikeFailedToolCall(tt.response)
			if got != tt.expected {
				t.Errorf("looksLikeFailedToolCall(%q) = %v, want %v", tt.response, got, tt.expected)
			}
		})
	}
}
