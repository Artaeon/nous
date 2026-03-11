package cognitive

import (
	"fmt"
	"strings"

	"github.com/artaeon/nous/internal/ollama"
)

// Conversation tracks multi-turn message history with automatic truncation
// to keep the context window manageable for CPU inference.
type Conversation struct {
	messages    []ollama.Message
	maxMessages int
}

func NewConversation(maxMessages int) *Conversation {
	return &Conversation{
		maxMessages: maxMessages,
	}
}

// System adds a system message (only keeps the latest one).
func (c *Conversation) System(content string) {
	// Replace existing system message or prepend
	if len(c.messages) > 0 && c.messages[0].Role == "system" {
		c.messages[0].Content = content
	} else {
		c.messages = append([]ollama.Message{{Role: "system", Content: content}}, c.messages...)
	}
}

// User adds a user message.
func (c *Conversation) User(content string) {
	c.messages = append(c.messages, ollama.Message{Role: "user", Content: content})
	c.truncate()
}

// Assistant adds an assistant message.
func (c *Conversation) Assistant(content string) {
	c.messages = append(c.messages, ollama.Message{Role: "assistant", Content: content})
	c.truncate()
}

// AssistantToolCalls adds an assistant message that contains tool calls (native API).
func (c *Conversation) AssistantToolCalls(content string, toolCalls []ollama.ToolCall) {
	c.messages = append(c.messages, ollama.Message{
		Role:      "assistant",
		Content:   content,
		ToolCalls: toolCalls,
	})
	c.truncate()
}

// NativeToolResult adds a tool result message using the native tool API (role=tool).
func (c *Conversation) NativeToolResult(toolName, result string) {
	c.messages = append(c.messages, ollama.ToolResultMessage(toolName, result))
	c.truncate()
}

// ToolResult adds a tool execution result as a user message.
// Uses OBSERVE: prefix for the structured THINK/ACT/OBSERVE protocol.
func (c *Conversation) ToolResult(toolName, result string) {
	msg := fmt.Sprintf("OBSERVE [%s]: %s", toolName, result)
	c.messages = append(c.messages, ollama.Message{Role: "user", Content: msg})
	c.truncate()
}

// Messages returns the current message list for an LLM call.
func (c *Conversation) Messages() []ollama.Message {
	return c.messages
}

// MessageCount returns the number of messages in the conversation.
func (c *Conversation) MessageCount() int {
	return len(c.messages)
}

// TokenEstimate returns approximate token count across all messages.
func (c *Conversation) TokenEstimate(charsPerToken float64) int {
	total := 0
	for _, m := range c.messages {
		total += len(m.Content)
	}
	if charsPerToken <= 0 {
		charsPerToken = 4.0
	}
	return int(float64(total) / charsPerToken)
}

// CompressOldest replaces the oldest n non-system messages with a compressed summary.
func (c *Conversation) CompressOldest(n int, compressed string) {
	if len(c.messages) <= n+1 {
		return
	}
	// Keep system message (index 0), replace messages [1..n] with compressed
	hasSystem := len(c.messages) > 0 && c.messages[0].Role == "system"
	if !hasSystem {
		return
	}
	sys := c.messages[0]
	remaining := make([]ollama.Message, len(c.messages[n+1:]))
	copy(remaining, c.messages[n+1:])
	c.messages = append(
		[]ollama.Message{sys, {Role: "user", Content: "[Earlier context]\n" + compressed}},
		remaining...,
	)
}

// Summary returns a brief overview of the conversation state.
func (c *Conversation) Summary() string {
	userCount, assistantCount := 0, 0
	for _, m := range c.messages {
		switch m.Role {
		case "user":
			userCount++
		case "assistant":
			assistantCount++
		}
	}
	return fmt.Sprintf("%d messages (%d user, %d assistant)", len(c.messages), userCount, assistantCount)
}

func (c *Conversation) truncate() {
	// Keep system message + last N messages
	if len(c.messages) <= c.maxMessages+1 {
		return
	}

	// Always keep the system message (index 0)
	if len(c.messages) > 0 && c.messages[0].Role == "system" {
		keep := c.messages[len(c.messages)-c.maxMessages:]
		c.messages = append([]ollama.Message{c.messages[0]}, keep...)
	} else {
		c.messages = c.messages[len(c.messages)-c.maxMessages:]
	}
}

// BuildContextBlock formats blackboard state into a context block for the LLM.
func BuildContextBlock(goals []string, recentActions []string, reflection string) string {
	var parts []string

	if len(goals) > 0 {
		parts = append(parts, "Active goals:\n"+strings.Join(goals, "\n"))
	}

	if len(recentActions) > 0 {
		parts = append(parts, "Recent actions:\n"+strings.Join(recentActions, "\n"))
	}

	if reflection != "" {
		parts = append(parts, "Reflection: "+reflection)
	}

	if len(parts) == 0 {
		return ""
	}

	return "\n[Cognitive Context]\n" + strings.Join(parts, "\n\n")
}
