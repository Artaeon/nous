package cognitive

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/artaeon/nous/internal/blackboard"
	"github.com/artaeon/nous/internal/memory"
	"github.com/artaeon/nous/internal/ollama"
	"github.com/artaeon/nous/internal/tools"
)

const maxToolIterations = 15

// Reasoner performs autonomous chain-of-thought inference with tool use.
// It listens for percepts, reasons about them, and can autonomously chain
// multiple tool calls to accomplish complex tasks — like an autonomous agent.
type Reasoner struct {
	Base
	Tools      *tools.Registry
	Conv       *Conversation
	WorkingMem *memory.WorkingMemory
	LongTermMem *memory.LongTermMemory
	OnToken    func(token string, done bool)
	OnStatus   func(status string)
	Confirm    ConfirmFunc
}

// CurrentProject holds the scanned project info for the system prompt.
var CurrentProject *ProjectInfo

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
	// Build system prompt — compact version optimized for small models
	systemPrompt := r.compactSystemPrompt()

	r.Conv.System(systemPrompt)

	// Inject relevant memories into the user message
	userMsg := percept.Raw
	memoryCtx := r.recallMemories(percept.Raw)
	if memoryCtx != "" {
		userMsg = percept.Raw + "\n\n" + memoryCtx
	}

	r.Conv.User(userMsg)

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
		finalAnswer := fullResponse
		if parsed.Answer != "" {
			finalAnswer = parsed.Answer
		}

		r.Board.Set("last_answer", finalAnswer)

		if parsed.Think != "" {
			r.Board.Set("last_thought", parsed.Think)
		}

		// Store to memory for future recall
		r.storeToMemory(percept.Raw, finalAnswer)

		return nil
	}

	r.Board.Set("last_answer", "(reached maximum tool iterations)")
	return nil
}

func (r *Reasoner) toolPrompt() string {
	var sb strings.Builder
	sb.WriteString("Available tools:\n")
	for _, t := range r.Tools.List() {
		sb.WriteString(fmt.Sprintf("- %s: %s\n", t.Name, t.Description))
	}
	return sb.String()
}

// WorkDir is set by main to inform the system prompt.
var WorkDir string

// compactSystemPrompt returns a focused system prompt optimized for small models.
func (r *Reasoner) compactSystemPrompt() string {
	wd := WorkDir
	if wd == "" {
		wd = "."
	}

	// Self-knowledge: Nous knows what it is
	modelName := "unknown"
	if r.LLM != nil {
		modelName = r.LLM.Model()
	}
	toolCount := len(r.Tools.List())
	wmSize := 0
	if r.WorkingMem != nil {
		wmSize = r.WorkingMem.Size()
	}
	ltmSize := 0
	if r.LongTermMem != nil {
		ltmSize = r.LongTermMem.Size()
	}
	selfKnowledge := SelfKnowledge(modelName, 6, toolCount, wmSize, ltmSize)

	projectCtx := ""
	if CurrentProject != nil {
		projectCtx = "\n" + CurrentProject.ContextString() + "\n"
	}

	return fmt.Sprintf(`%s

Working directory: %s
%s
You MUST use tools to interact with the filesystem. NEVER guess file contents.

TOOL CALL FORMAT — output this JSON on its own line, nothing else:
{"tool": "TOOL_NAME", "args": {"key": "value"}}

EXAMPLES:
User: "List files in this project"
{"tool": "tree", "args": {}}

User: "Read the README"
{"tool": "read", "args": {"path": "README.md"}}

User: "Find all Go files"
{"tool": "glob", "args": {"pattern": "*.go"}}

User: "Search for function main"
{"tool": "grep", "args": {"pattern": "func main"}}

RULES:
1. ALWAYS call a tool first to gather information. Never guess or hallucinate.
2. Use relative paths (e.g. "README.md" not "/full/path/README.md").
3. After a tool result, you can call more tools or give your final answer.
4. Final answer: respond normally WITHOUT any JSON tool call.
5. Be direct and concise.

%s`, selfKnowledge, wd, projectCtx, r.toolPrompt())
}

type toolCallRaw struct {
	Name string                 `json:"tool"`
	Args map[string]interface{} `json:"args"`
}

type toolCall struct {
	Name string
	Args map[string]string
}

func (r toolCallRaw) normalize() toolCall {
	tc := toolCall{Name: r.Name, Args: make(map[string]string)}
	for k, v := range r.Args {
		tc.Args[k] = fmt.Sprintf("%v", v)
	}
	return tc
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

				var raw toolCallRaw
				if err := json.Unmarshal([]byte(jsonStr), &raw); err == nil && raw.Name != "" {
					parsed.ToolCalls = append(parsed.ToolCalls, raw.normalize())
					continue
				}
			}
			// Also try parsing bare JSON blocks
			if strings.HasPrefix(trimmed, "json\n") || strings.HasPrefix(trimmed, "json\r\n") {
				jsonStr := strings.TrimPrefix(trimmed, "json\n")
				jsonStr = strings.TrimPrefix(jsonStr, "json\r\n")
				jsonStr = strings.TrimSpace(jsonStr)

				var raw toolCallRaw
				if err := json.Unmarshal([]byte(jsonStr), &raw); err == nil && raw.Name != "" {
					parsed.ToolCalls = append(parsed.ToolCalls, raw.normalize())
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
					var raw toolCallRaw
					if err := json.Unmarshal([]byte(candidate), &raw); err == nil && raw.Name != "" {
						calls = append(calls, raw.normalize())
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

	// Check if this is a dangerous action requiring confirmation
	if r.Confirm != nil {
		if reason, dangerous := IsDangerous(tc.Name); dangerous {
			detail := fmt.Sprintf("%s: %s %v", reason, tc.Name, tc.Args)
			if !r.Confirm(tc.Name+" "+formatArgs(tc.Args), detail) {
				return "Action denied by user.", nil
			}
		}
	}

	return tool.Execute(tc.Args)
}

func formatArgs(args map[string]string) string {
	if path, ok := args["path"]; ok {
		return path
	}
	if cmd, ok := args["command"]; ok {
		return cmd
	}
	parts := make([]string, 0, len(args))
	for k, v := range args {
		parts = append(parts, k+"="+v)
	}
	return strings.Join(parts, " ")
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

// recallMemories retrieves relevant context from working and long-term memory.
func (r *Reasoner) recallMemories(input string) string {
	var parts []string

	// Recall from working memory (most relevant items)
	if r.WorkingMem != nil {
		items := r.WorkingMem.MostRelevant(3)
		if len(items) > 0 {
			var memLines []string
			for _, item := range items {
				memLines = append(memLines, fmt.Sprintf("- %s: %v", item.Key, item.Value))
			}
			parts = append(parts, "[Working Memory]\n"+strings.Join(memLines, "\n"))
		}
	}

	// Recall from long-term memory (keyword match)
	if r.LongTermMem != nil {
		words := strings.Fields(strings.ToLower(input))
		seen := make(map[string]bool)
		var ltmLines []string
		for _, word := range words {
			if len(word) < 3 {
				continue
			}
			entries := r.LongTermMem.Search(word)
			for _, e := range entries {
				if !seen[e.Key] {
					seen[e.Key] = true
					ltmLines = append(ltmLines, fmt.Sprintf("- %s: %s", e.Key, e.Value))
				}
			}
			if len(ltmLines) >= 5 {
				break
			}
		}
		if len(ltmLines) > 0 {
			parts = append(parts, "[Long-term Memory]\n"+strings.Join(ltmLines, "\n"))
		}
	}

	if len(parts) == 0 {
		return ""
	}
	return strings.Join(parts, "\n\n")
}

// storeToMemory saves the current interaction context to working memory.
func (r *Reasoner) storeToMemory(input, answer string) {
	if r.WorkingMem == nil {
		return
	}

	// Store the latest interaction in working memory
	// Use the first 100 chars of input as key, full answer as value
	key := input
	if len(key) > 80 {
		key = key[:80] + "..."
	}

	// Truncate answer for working memory storage
	value := answer
	if len(value) > 200 {
		value = value[:200] + "..."
	}

	r.WorkingMem.Store("last:"+key, value, 0.8)
}
