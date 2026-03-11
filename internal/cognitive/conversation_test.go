package cognitive

import (
	"strings"
	"testing"
)

func TestNewConversation(t *testing.T) {
	c := NewConversation(10)
	if c == nil {
		t.Fatal("NewConversation returned nil")
	}
	msgs := c.Messages()
	if len(msgs) != 0 {
		t.Errorf("expected 0 messages, got %d", len(msgs))
	}
}

func TestSystemMessage(t *testing.T) {
	c := NewConversation(10)
	c.System("You are a helpful assistant.")

	msgs := c.Messages()
	if len(msgs) != 1 {
		t.Fatalf("expected 1 message, got %d", len(msgs))
	}
	if msgs[0].Role != "system" {
		t.Errorf("expected role 'system', got %q", msgs[0].Role)
	}
	if msgs[0].Content != "You are a helpful assistant." {
		t.Errorf("expected system content, got %q", msgs[0].Content)
	}
}

func TestSystemMessageReplacement(t *testing.T) {
	c := NewConversation(10)
	c.System("Version 1")
	c.System("Version 2")

	msgs := c.Messages()
	if len(msgs) != 1 {
		t.Fatalf("expected 1 system message (replaced), got %d messages", len(msgs))
	}
	if msgs[0].Content != "Version 2" {
		t.Errorf("expected 'Version 2', got %q", msgs[0].Content)
	}
}

func TestUserMessage(t *testing.T) {
	c := NewConversation(10)
	c.User("Hello")

	msgs := c.Messages()
	if len(msgs) != 1 {
		t.Fatalf("expected 1 message, got %d", len(msgs))
	}
	if msgs[0].Role != "user" {
		t.Errorf("expected role 'user', got %q", msgs[0].Role)
	}
	if msgs[0].Content != "Hello" {
		t.Errorf("expected 'Hello', got %q", msgs[0].Content)
	}
}

func TestAssistantMessage(t *testing.T) {
	c := NewConversation(10)
	c.Assistant("I can help you.")

	msgs := c.Messages()
	if len(msgs) != 1 {
		t.Fatalf("expected 1 message, got %d", len(msgs))
	}
	if msgs[0].Role != "assistant" {
		t.Errorf("expected role 'assistant', got %q", msgs[0].Role)
	}
}

func TestToolResult(t *testing.T) {
	c := NewConversation(10)
	c.ToolResult("read", "file contents here")

	msgs := c.Messages()
	if len(msgs) != 1 {
		t.Fatalf("expected 1 message, got %d", len(msgs))
	}
	if msgs[0].Role != "user" {
		t.Errorf("expected role 'user' for tool result, got %q", msgs[0].Role)
	}
	if !strings.Contains(msgs[0].Content, "OBSERVE [read]:") {
		t.Error("expected 'OBSERVE [read]:' in tool result message")
	}
	if !strings.Contains(msgs[0].Content, "file contents here") {
		t.Error("expected tool output in message content")
	}
}

func TestMessageTracking(t *testing.T) {
	c := NewConversation(20)

	c.System("system prompt")
	c.User("question 1")
	c.Assistant("answer 1")
	c.User("question 2")
	c.Assistant("answer 2")

	msgs := c.Messages()
	if len(msgs) != 5 {
		t.Fatalf("expected 5 messages, got %d", len(msgs))
	}

	expected := []struct {
		role    string
		content string
	}{
		{"system", "system prompt"},
		{"user", "question 1"},
		{"assistant", "answer 1"},
		{"user", "question 2"},
		{"assistant", "answer 2"},
	}

	for i, exp := range expected {
		if msgs[i].Role != exp.role {
			t.Errorf("msg[%d]: expected role %q, got %q", i, exp.role, msgs[i].Role)
		}
		if msgs[i].Content != exp.content {
			t.Errorf("msg[%d]: expected content %q, got %q", i, exp.content, msgs[i].Content)
		}
	}
}

func TestTruncationWithSystemMessage(t *testing.T) {
	c := NewConversation(4) // max 4 non-system messages

	c.System("system prompt")
	c.User("q1")
	c.Assistant("a1")
	c.User("q2")
	c.Assistant("a2")
	c.User("q3")
	c.Assistant("a3")

	msgs := c.Messages()

	// System message must always be first
	if msgs[0].Role != "system" {
		t.Error("expected system message to be pinned at index 0")
	}
	if msgs[0].Content != "system prompt" {
		t.Errorf("expected system content preserved, got %q", msgs[0].Content)
	}

	// Should keep system + last 4 messages
	if len(msgs) != 5 { // 1 system + 4 kept
		t.Errorf("expected 5 messages after truncation, got %d", len(msgs))
	}

	// The oldest messages should be dropped
	for _, m := range msgs[1:] {
		if m.Content == "q1" || m.Content == "a1" {
			t.Error("expected oldest messages to be truncated")
		}
	}
}

func TestTruncationWithoutSystemMessage(t *testing.T) {
	c := NewConversation(3)

	c.User("q1")
	c.Assistant("a1")
	c.User("q2")
	c.Assistant("a2")
	c.User("q3")

	msgs := c.Messages()

	// Should keep last 3 messages
	if len(msgs) != 3 {
		t.Errorf("expected 3 messages after truncation, got %d", len(msgs))
	}

	// First message should be the 3rd-from-last: q2
	if msgs[0].Content != "q2" {
		t.Errorf("expected first kept message to be 'q2', got %q", msgs[0].Content)
	}
}

func TestNoTruncationWhenUnderLimit(t *testing.T) {
	c := NewConversation(10)

	c.System("sys")
	c.User("q1")
	c.Assistant("a1")

	msgs := c.Messages()
	if len(msgs) != 3 {
		t.Errorf("expected 3 messages (no truncation), got %d", len(msgs))
	}
}

func TestSummary(t *testing.T) {
	c := NewConversation(20)
	c.System("sys")
	c.User("q1")
	c.Assistant("a1")
	c.User("q2")

	summary := c.Summary()
	if !strings.Contains(summary, "4 messages") {
		t.Errorf("expected '4 messages' in summary, got %q", summary)
	}
	if !strings.Contains(summary, "2 user") {
		t.Errorf("expected '2 user' in summary, got %q", summary)
	}
	if !strings.Contains(summary, "1 assistant") {
		t.Errorf("expected '1 assistant' in summary, got %q", summary)
	}
}

func TestBuildContextBlock(t *testing.T) {
	result := BuildContextBlock(
		[]string{"Write tests", "Fix bug"},
		[]string{"Ran test suite", "Read file"},
		"Tests are failing",
	)

	if !strings.Contains(result, "[Cognitive Context]") {
		t.Error("expected '[Cognitive Context]' header")
	}
	if !strings.Contains(result, "Write tests") {
		t.Error("expected goals in context block")
	}
	if !strings.Contains(result, "Ran test suite") {
		t.Error("expected recent actions in context block")
	}
	if !strings.Contains(result, "Tests are failing") {
		t.Error("expected reflection in context block")
	}
}

func TestBuildContextBlockEmpty(t *testing.T) {
	result := BuildContextBlock(nil, nil, "")
	if result != "" {
		t.Errorf("expected empty string for empty context, got %q", result)
	}
}

func TestBuildContextBlockPartial(t *testing.T) {
	// Only goals, no actions or reflection
	result := BuildContextBlock([]string{"Goal 1"}, nil, "")
	if !strings.Contains(result, "Goal 1") {
		t.Error("expected goal in partial context block")
	}
	if strings.Contains(result, "Recent actions") {
		t.Error("expected no actions section when actions is nil")
	}
	if strings.Contains(result, "Reflection") {
		t.Error("expected no reflection section when reflection is empty")
	}
}
