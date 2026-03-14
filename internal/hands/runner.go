package hands

import (
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/artaeon/nous/internal/blackboard"
	"github.com/artaeon/nous/internal/ollama"
	"github.com/artaeon/nous/internal/tools"
)

// Runner executes a hand by creating a fresh cognitive pipeline,
// injecting the hand's prompt, and running tool iterations.
type Runner struct {
	LLM        *ollama.Client
	Board      *blackboard.Blackboard
	Tools      *tools.Registry
	StateStore *HandStateStore
}

// NewRunner creates a hand execution engine.
func NewRunner(llm *ollama.Client, board *blackboard.Blackboard, toolReg *tools.Registry) *Runner {
	return &Runner{
		LLM:   llm,
		Board: board,
		Tools: toolReg,
	}
}

// NewRunnerWithState creates a hand execution engine with persistent state support.
func NewRunnerWithState(llm *ollama.Client, board *blackboard.Blackboard, toolReg *tools.Registry, stateStore *HandStateStore) *Runner {
	return &Runner{
		LLM:        llm,
		Board:      board,
		Tools:      toolReg,
		StateStore: stateStore,
	}
}

// Run executes a hand with the given configuration.
// It creates a scoped tool registry (respecting the whitelist),
// sends the hand's prompt through the LLM with tool calling,
// and collects the result.
func (r *Runner) Run(ctx context.Context, hand *Hand) HandResult {
	start := time.Now()

	// Apply timeout from config with a minimum floor to account for
	// model cold-start latency (first call can take ~100s to load).
	timeout := time.Duration(hand.Config.Timeout) * time.Second
	if timeout <= 0 {
		timeout = 180 * time.Second
	}
	const minTimeout = 300 * time.Second
	if timeout < minTimeout {
		timeout = minTimeout
	}
	ctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	maxSteps := hand.Config.MaxSteps
	if maxSteps <= 0 {
		maxSteps = 8
	}

	// Build scoped tool registry — only whitelisted tools
	scopedTools := r.scopedRegistry(hand.Config.Tools)

	// Load previous state if state store is available
	var previousState map[string]string
	if r.StateStore != nil {
		previousState = r.StateStore.GetAll(hand.Name)
	}

	// Build the system prompt with available tools and state
	sysPrompt := r.buildSystemPrompt(hand, scopedTools, previousState)

	// Run the autonomous tool loop
	var toolCallCount int
	var allOutput strings.Builder

	// Conversation messages for the LLM
	messages := []ollama.Message{
		{Role: "system", Content: sysPrompt},
		{Role: "user", Content: hand.Prompt},
	}

	for i := 0; i < maxSteps; i++ {
		select {
		case <-ctx.Done():
			return HandResult{
				Output:    allOutput.String(),
				Error:     "timeout exceeded",
				Duration:  time.Since(start),
				ToolCalls: toolCallCount,
			}
		default:
		}

		resp, err := r.LLM.ChatCtx(ctx, messages, &ollama.ModelOptions{
			Temperature: 0.3,
			NumPredict:  1024,
		})
		if err != nil {
			return HandResult{
				Output:    allOutput.String(),
				Error:     fmt.Sprintf("llm error: %v", err),
				Duration:  time.Since(start),
				ToolCalls: toolCallCount,
			}
		}

		content := strings.TrimSpace(resp.Message.Content)
		messages = append(messages, ollama.Message{Role: "assistant", Content: content})

		// Parse tool calls from response
		tc := parseToolCall(content)
		if tc.Name == "" {
			// No tool call — this is the final answer
			allOutput.WriteString(content)
			break
		}

		// Execute tool
		tool, toolErr := scopedTools.Get(tc.Name)
		if toolErr != nil {
			toolResult := fmt.Sprintf("Error: tool %q not available", tc.Name)
			messages = append(messages, ollama.Message{Role: "user", Content: "Tool result: " + toolResult})
			continue
		}

		// Approval gate — if the hand requires approval and uses dangerous tools,
		// we record the intent but do not execute
		if hand.Config.RequiresApproval && isDangerousTool(tc.Name) {
			result := fmt.Sprintf("[APPROVAL REQUIRED] Hand %q wants to execute %s with args: %v", hand.Name, tc.Name, tc.Args)
			allOutput.WriteString(result + "\n")
			messages = append(messages, ollama.Message{Role: "user", Content: "Tool result: " + result})
			continue
		}

		result, execErr := tool.Execute(tc.Args)
		toolCallCount++

		if execErr != nil {
			result = fmt.Sprintf("Error: %v", execErr)
		}

		// Record action on blackboard
		r.Board.RecordAction(blackboard.ActionRecord{
			StepID:    fmt.Sprintf("hand-%s-%d", hand.Name, i),
			Tool:      tc.Name,
			Input:     formatToolArgs(tc.Args),
			Output:    truncateResult(result, 500),
			Success:   execErr == nil,
			Duration:  time.Since(start),
			Timestamp: time.Now(),
		})

		// Add tool result to conversation
		observation := fmt.Sprintf("Tool %s returned:\n%s", tc.Name, truncateResult(result, 2000))
		messages = append(messages, ollama.Message{Role: "user", Content: observation})
	}

	output := strings.TrimSpace(allOutput.String())

	// Extract and save state from output
	if r.StateStore != nil {
		if stateEntries := ExtractState(output); len(stateEntries) > 0 {
			for k, v := range stateEntries {
				_ = r.StateStore.Set(hand.Name, k, v)
			}
		}
	}

	return HandResult{
		Output:    output,
		Duration:  time.Since(start),
		ToolCalls: toolCallCount,
	}
}

// scopedRegistry creates a Registry containing only whitelisted tools.
// If the whitelist is empty, all tools are available.
func (r *Runner) scopedRegistry(whitelist []string) *tools.Registry {
	if len(whitelist) == 0 {
		return r.Tools
	}
	scoped := tools.NewRegistry()
	for _, name := range whitelist {
		if tool, err := r.Tools.Get(name); err == nil {
			scoped.Register(tool)
		}
	}
	return scoped
}

// buildSystemPrompt creates the system prompt for a hand run.
func (r *Runner) buildSystemPrompt(hand *Hand, reg *tools.Registry, previousState map[string]string) string {
	var sb strings.Builder
	sb.WriteString("You are an autonomous agent executing a task. ")
	sb.WriteString("You have access to the following tools:\n\n")
	sb.WriteString(reg.Describe())
	sb.WriteString("\nTo use a tool, respond with JSON on a single line:\n")
	sb.WriteString(`{"tool": "TOOL_NAME", "args": {"key": "value"}}`)
	sb.WriteString("\n\nAfter receiving a tool result, continue reasoning or give your final answer as plain text (no JSON).\n")
	sb.WriteString("When you have completed the task, respond with your final summary as plain text.\n")
	sb.WriteString("To persist state between runs, include [STATE key=value] in your output.\n")
	if hand.Description != "" {
		sb.WriteString("\nTask context: ")
		sb.WriteString(hand.Description)
		sb.WriteString("\n")
	}
	if statePrompt := FormatStatePrompt(previousState); statePrompt != "" {
		sb.WriteString("\n")
		sb.WriteString(statePrompt)
	}
	return sb.String()
}

// toolCallParsed holds a parsed tool invocation.
type toolCallParsed struct {
	Name string
	Args map[string]string
}

// parseToolCall extracts a tool call from LLM output.
// Supports two formats:
//  1. JSON like {"tool": "name", "args": {...}}
//  2. Qwen3 format: <tool_call>{"name": "tool", "arguments": {...}}</tool_call>
func parseToolCall(content string) toolCallParsed {
	// Check for Qwen3-style <tool_call> tags first
	if tc := parseQwenToolCall(content); tc.Name != "" {
		return tc
	}

	// Look for JSON object in the response
	for _, line := range strings.Split(content, "\n") {
		line = strings.TrimSpace(line)
		if !strings.HasPrefix(line, "{") {
			continue
		}

		// Simple JSON extraction — find tool and args fields
		if !strings.Contains(line, `"tool"`) {
			continue
		}

		// Parse manually to avoid encoding/json import in hot path
		// (keeping consistent with the project's approach in reasoner)
		name := extractJSONString(line, "tool")
		if name == "" {
			continue
		}

		args := make(map[string]string)
		// Extract the args object
		argsIdx := strings.Index(line, `"args"`)
		if argsIdx >= 0 {
			rest := line[argsIdx:]
			braceStart := strings.IndexByte(rest, '{')
			if braceStart >= 0 {
				braceEnd := matchingBrace(rest[braceStart:])
				if braceEnd > 0 {
					argsJSON := rest[braceStart : braceStart+braceEnd+1]
					args = parseArgsJSON(argsJSON)
				}
			}
		}

		return toolCallParsed{Name: name, Args: args}
	}
	return toolCallParsed{}
}

// parseQwenToolCall handles the Qwen3 tool call format:
// <tool_call>{"name": "tool_name", "arguments": {"key": "value"}}</tool_call>
func parseQwenToolCall(content string) toolCallParsed {
	const openTag = "<tool_call>"
	const closeTag = "</tool_call>"

	start := strings.Index(content, openTag)
	if start < 0 {
		return toolCallParsed{}
	}
	rest := content[start+len(openTag):]
	end := strings.Index(rest, closeTag)
	if end < 0 {
		// No closing tag — try to parse what we have
		end = len(rest)
	}
	jsonStr := strings.TrimSpace(rest[:end])
	if len(jsonStr) == 0 || jsonStr[0] != '{' {
		return toolCallParsed{}
	}

	name := extractJSONString(jsonStr, "name")
	if name == "" {
		return toolCallParsed{}
	}

	args := make(map[string]string)
	argsIdx := strings.Index(jsonStr, `"arguments"`)
	if argsIdx >= 0 {
		argRest := jsonStr[argsIdx:]
		braceStart := strings.IndexByte(argRest, '{')
		if braceStart >= 0 {
			braceEnd := matchingBrace(argRest[braceStart:])
			if braceEnd > 0 {
				argsJSON := argRest[braceStart : braceStart+braceEnd+1]
				args = parseArgsJSON(argsJSON)
			}
		}
	}

	return toolCallParsed{Name: name, Args: args}
}

// extractJSONString extracts a string value for a key from a JSON-like line.
func extractJSONString(line, key string) string {
	search := `"` + key + `"`
	idx := strings.Index(line, search)
	if idx < 0 {
		return ""
	}
	rest := line[idx+len(search):]
	// Skip colon and whitespace
	rest = strings.TrimLeft(rest, ": \t")
	if len(rest) == 0 || rest[0] != '"' {
		return ""
	}
	rest = rest[1:]
	end := strings.IndexByte(rest, '"')
	if end < 0 {
		return ""
	}
	return rest[:end]
}

// matchingBrace finds the index of the closing brace for an opening brace at position 0.
func matchingBrace(s string) int {
	depth := 0
	inString := false
	for i, c := range s {
		if inString {
			if c == '"' && (i == 0 || s[i-1] != '\\') {
				inString = false
			}
			continue
		}
		switch c {
		case '"':
			inString = true
		case '{':
			depth++
		case '}':
			depth--
			if depth == 0 {
				return i
			}
		}
	}
	return -1
}

// parseArgsJSON extracts string key-value pairs from a simple JSON object.
// It iterates character by character, respecting quoted strings so that
// commas inside values do not break parsing.
func parseArgsJSON(s string) map[string]string {
	args := make(map[string]string)
	s = strings.TrimSpace(s)
	if len(s) < 2 {
		return args
	}
	// Strip outer braces
	s = s[1 : len(s)-1]

	// Collect key-value parts by splitting on commas that are not inside quotes.
	var parts []string
	var buf strings.Builder
	inQuote := false
	escaped := false
	for i := 0; i < len(s); i++ {
		c := s[i]
		if escaped {
			buf.WriteByte(c)
			escaped = false
			continue
		}
		if c == '\\' && inQuote {
			buf.WriteByte(c)
			escaped = true
			continue
		}
		if c == '"' {
			inQuote = !inQuote
			buf.WriteByte(c)
			continue
		}
		if c == ',' && !inQuote {
			parts = append(parts, buf.String())
			buf.Reset()
			continue
		}
		buf.WriteByte(c)
	}
	if buf.Len() > 0 {
		parts = append(parts, buf.String())
	}

	for _, part := range parts {
		part = strings.TrimSpace(part)
		colonIdx := strings.IndexByte(part, ':')
		if colonIdx < 0 {
			continue
		}
		key := strings.Trim(strings.TrimSpace(part[:colonIdx]), `"`)
		val := strings.Trim(strings.TrimSpace(part[colonIdx+1:]), `"`)
		if key != "" {
			args[key] = val
		}
	}
	return args
}

// isDangerousTool returns true for tools that modify the filesystem or run commands.
func isDangerousTool(name string) bool {
	switch name {
	case "shell", "write", "edit", "patch", "find_replace", "replace_all", "mkdir":
		return true
	}
	return false
}

// formatToolArgs formats tool arguments for logging.
func formatToolArgs(args map[string]string) string {
	var parts []string
	for k, v := range args {
		if len(v) > 80 {
			v = v[:80] + "..."
		}
		parts = append(parts, k+"="+v)
	}
	return strings.Join(parts, ", ")
}

// truncateResult truncates a string to maxLen characters.
func truncateResult(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "\n... (truncated)"
}
