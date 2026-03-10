package cognitive

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/artaeon/nous/internal/blackboard"
	"github.com/artaeon/nous/internal/ollama"
	"github.com/artaeon/nous/internal/tools"
)

const maxToolIterations = 15

// Reasoner performs autonomous chain-of-thought inference with tool use.
// It listens for percepts, reasons about them, and can autonomously chain
// multiple tool calls to accomplish complex tasks — like an autonomous agent.
type Reasoner struct {
	Base
	Tools    *tools.Registry
	Conv     *Conversation
	OnToken  func(token string, done bool)
	OnStatus func(status string)
}

func NewReasoner(board *blackboard.Blackboard, llm *ollama.Client, toolReg *tools.Registry) *Reasoner {
	return &Reasoner{
		Base:  Base{Board: board, LLM: llm},
		Tools: toolReg,
		Conv:  NewConversation(20),
	}
}

func (r *Reasoner) Name() string { return "reasoner" }

func (r *Reasoner) Run(ctx context.Context) error {
	events := r.Board.Subscribe("percept")

	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case ev := <-events:
			percept, ok := ev.Payload.(blackboard.Percept)
			if !ok {
				continue
			}
			if err := r.reason(ctx, percept); err != nil {
				r.Board.Set("reasoner_error", err.Error())
			}
		}
	}
}

func (r *Reasoner) reason(ctx context.Context, percept blackboard.Percept) error {
	// Build system prompt with persona and tool descriptions
	systemPrompt := Persona + "\n\n" + r.toolPrompt()

	r.Conv.System(systemPrompt)
	r.Conv.User(percept.Raw)

	// Autonomous tool loop — keep going until the LLM gives a final answer
	for i := 0; i < maxToolIterations; i++ {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}

		// Call LLM
		var fullResponse string
		var err error

		if r.OnToken != nil {
			fullResponse, err = r.streamCall()
		} else {
			fullResponse, err = r.batchCall()
		}

		if err != nil {
			return fmt.Errorf("reasoner iteration %d: %w", i, err)
		}

		r.Conv.Assistant(fullResponse)

		// Parse the response for tool calls
		parsed := r.parseResponse(fullResponse)

		// If there are tool calls, execute them and continue the loop
		if len(parsed.ToolCalls) > 0 {
			for _, tc := range parsed.ToolCalls {
				r.emitStatus(fmt.Sprintf("  [tool] %s", tc.Name))

				result, toolErr := r.executeTool(tc)
				if toolErr != nil {
					result = fmt.Sprintf("Error: %v", toolErr)
				}

				// Truncate large results
				if len(result) > 4096 {
					result = result[:4096] + "\n... (truncated)"
				}

				r.Conv.ToolResult(tc.Name, result)
			}
			continue // Loop back for next LLM call with tool results
		}

		// No tool calls — this is the final answer
		if parsed.Answer != "" {
			r.Board.Set("last_answer", parsed.Answer)
		} else {
			r.Board.Set("last_answer", fullResponse)
		}

		if parsed.Think != "" {
			r.Board.Set("last_thought", parsed.Think)
		}

		return nil
	}

	r.Board.Set("last_answer", "(reached maximum tool iterations)")
	return nil
}

func (r *Reasoner) toolPrompt() string {
	var sb strings.Builder
	sb.WriteString("## Available Tools\n\n")
	sb.WriteString("To use a tool, include a JSON block in your response:\n\n")
	sb.WriteString("```tool\n")
	sb.WriteString("{\"tool\": \"tool_name\", \"args\": {\"key\": \"value\"}}\n")
	sb.WriteString("```\n\n")
	sb.WriteString("You can call multiple tools in one response. After each tool call,\n")
	sb.WriteString("you will receive the results and can continue reasoning.\n\n")
	sb.WriteString("When you have a final answer for the user, simply write your response\n")
	sb.WriteString("without any tool blocks.\n\n")
	sb.WriteString("Tools:\n")
	for _, t := range r.Tools.List() {
		sb.WriteString(fmt.Sprintf("- **%s**: %s\n", t.Name, t.Description))
	}
	return sb.String()
}

type toolCall struct {
	Name string            `json:"tool"`
	Args map[string]string `json:"args"`
}

type parsedResponse struct {
	Think     string
	Answer    string
	ToolCalls []toolCall
}

func (r *Reasoner) parseResponse(content string) parsedResponse {
	var parsed parsedResponse

	// Extract tool calls from ```tool blocks
	parts := strings.Split(content, "```")
	var nonToolParts []string

	for i, part := range parts {
		if i%2 == 1 { // Inside a code block
			trimmed := strings.TrimSpace(part)
			// Check if it starts with "tool" marker
			if strings.HasPrefix(trimmed, "tool\n") || strings.HasPrefix(trimmed, "tool\r\n") {
				jsonStr := strings.TrimPrefix(trimmed, "tool\n")
				jsonStr = strings.TrimPrefix(jsonStr, "tool\r\n")
				jsonStr = strings.TrimSpace(jsonStr)

				var tc toolCall
				if err := json.Unmarshal([]byte(jsonStr), &tc); err == nil && tc.Name != "" {
					parsed.ToolCalls = append(parsed.ToolCalls, tc)
					continue
				}
			}
			// Also try parsing bare JSON blocks
			if strings.HasPrefix(trimmed, "json\n") || strings.HasPrefix(trimmed, "json\r\n") {
				jsonStr := strings.TrimPrefix(trimmed, "json\n")
				jsonStr = strings.TrimPrefix(jsonStr, "json\r\n")
				jsonStr = strings.TrimSpace(jsonStr)

				var tc toolCall
				if err := json.Unmarshal([]byte(jsonStr), &tc); err == nil && tc.Name != "" {
					parsed.ToolCalls = append(parsed.ToolCalls, tc)
					continue
				}
			}
		}
		nonToolParts = append(nonToolParts, part)
	}

	// Also try to find inline JSON tool calls (for models that don't use code blocks)
	if len(parsed.ToolCalls) == 0 {
		parsed.ToolCalls = r.findInlineToolCalls(content)
	}

	// Extract THINK/ANSWER from non-tool text
	text := strings.Join(nonToolParts, "")
	for _, line := range strings.Split(text, "\n") {
		trimmed := strings.TrimSpace(line)
		upper := strings.ToUpper(trimmed)

		if strings.HasPrefix(upper, "THINK:") {
			parsed.Think = strings.TrimSpace(trimmed[6:])
		} else if strings.HasPrefix(upper, "ANSWER:") {
			parsed.Answer = strings.TrimSpace(trimmed[7:])
		}
	}

	// If no explicit ANSWER but also no tool calls, the whole text is the answer
	if parsed.Answer == "" && len(parsed.ToolCalls) == 0 {
		parsed.Answer = strings.TrimSpace(text)
	}

	return parsed
}

func (r *Reasoner) findInlineToolCalls(content string) []toolCall {
	var calls []toolCall

	// Look for {"tool": "..."} patterns in the text
	for i := 0; i < len(content); i++ {
		if content[i] != '{' {
			continue
		}

		// Find matching closing brace
		depth := 0
		for j := i; j < len(content); j++ {
			if content[j] == '{' {
				depth++
			} else if content[j] == '}' {
				depth--
				if depth == 0 {
					candidate := content[i : j+1]
					var tc toolCall
					if err := json.Unmarshal([]byte(candidate), &tc); err == nil && tc.Name != "" {
						calls = append(calls, tc)
					}
					i = j
					break
				}
			}
		}
	}

	return calls
}

func (r *Reasoner) executeTool(tc toolCall) (string, error) {
	tool, err := r.Tools.Get(tc.Name)
	if err != nil {
		return "", err
	}

	if tc.Args == nil {
		tc.Args = make(map[string]string)
	}

	return tool.Execute(tc.Args)
}

func (r *Reasoner) streamCall() (string, error) {
	var full strings.Builder
	resp, err := r.LLM.ChatStream(r.Conv.Messages(), &ollama.ModelOptions{
		Temperature: 0.7,
		NumPredict:  2048,
	}, func(token string, done bool) {
		full.WriteString(token)
		if r.OnToken != nil {
			r.OnToken(token, done)
		}
	})
	if err != nil {
		return "", err
	}
	_ = resp
	return full.String(), nil
}

func (r *Reasoner) batchCall() (string, error) {
	resp, err := r.LLM.Chat(r.Conv.Messages(), &ollama.ModelOptions{
		Temperature: 0.7,
		NumPredict:  2048,
	})
	if err != nil {
		return "", err
	}
	return resp.Message.Content, nil
}

func (r *Reasoner) emitStatus(msg string) {
	if r.OnStatus != nil {
		r.OnStatus(msg)
	}
}
