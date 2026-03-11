package cognitive

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/artaeon/nous/internal/blackboard"
	"github.com/artaeon/nous/internal/compress"
	"github.com/artaeon/nous/internal/index"
	"github.com/artaeon/nous/internal/memory"
	"github.com/artaeon/nous/internal/ollama"
	"github.com/artaeon/nous/internal/tools"
)

const maxToolIterations = 8

// Reasoner performs autonomous chain-of-thought inference with tool use.
// It listens for percepts, reasons about them, and can autonomously chain
// multiple tool calls to accomplish complex tasks — like an autonomous agent.
//
// The Cognitive Grounding system prevents hallucinations by:
// 1. Progressive tool disclosure — only showing relevant tools per intent
// 2. Structured THINK/ACT/OBSERVE protocol — forcing reasoning structure
// 3. Smart result truncation — keeping tool results within token budget
// 4. Context budget tracking — compressing when context fills up
// 5. Synchronous reflection gate — catching errors before the next LLM call
type Reasoner struct {
	Base
	Tools       *tools.Registry
	Conv        *Conversation
	WorkingMem  *memory.WorkingMemory
	LongTermMem *memory.LongTermMemory
	ProjectMem  *memory.ProjectMemory
	Compressor  *compress.Compressor
	CodeIndex   *index.CodebaseIndex
	Budget      *ContextBudget
	Gate        *ReflectionGate
	Recipes     *RecipeBook
	Predictor   *Predictor
	Learner     *Learner
	OnToken     func(token string, done bool)
	OnStatus    func(status string)
	Confirm     ConfirmFunc

	// Active tool subset for current reasoning cycle
	activeTools []tools.Tool
	activeCats  map[ToolCategory]bool
}

// CurrentProject holds the scanned project info for the system prompt.
var CurrentProject *ProjectInfo

func NewReasoner(board *blackboard.Blackboard, llm *ollama.Client, toolReg *tools.Registry) *Reasoner {
	return &Reasoner{
		Base:       Base{Board: board, LLM: llm},
		Tools:      toolReg,
		Conv:       NewConversation(20),
		Budget:     DefaultBudget(),
		Gate:       &ReflectionGate{},
		activeCats: make(map[ToolCategory]bool),
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
	// Reset reflection gate for this reasoning cycle
	r.Gate.Reset()

	// 1. Progressive Tool Disclosure — select relevant tools based on intent
	r.activeTools = SelectToolsForIntent(percept.Intent, percept.Entities, percept.Raw, r.Tools.List())
	r.activeCats = make(map[ToolCategory]bool)
	for _, t := range r.activeTools {
		if cat, ok := ToolCategoryMap[t.Name]; ok {
			r.activeCats[cat] = true
		}
	}

	// Create pipeline for this reasoning cycle.
	pipe := NewPipeline(percept.Raw)

	// Recall key facts from memory for system context (not user message).
	// Only inject if truly relevant — avoid dumping memory into the prompt.
	memoryFacts := r.recallKeyFacts(percept.Raw)

	// 2. Autonomous tool loop with fresh-context pipeline
	// nativeConv tracks the accumulated conversation when using native tool calling.
	// When the model uses native tool calls, we keep building on the same conversation
	// so the model sees its own tool_calls and the tool results. When using fallback
	// JSON-in-text parsing, we rebuild fresh context from the pipeline each iteration.
	var nativeConv *Conversation

	for i := 0; i < maxToolIterations; i++ {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}

		if nativeConv != nil {
			// Native API continuation: reuse the accumulated conversation
			r.Conv = nativeConv
		} else {
			// Build FRESH conversation each iteration — this is the key innovation.
			conv := NewConversation(10)

			// System prompt: tools + instructions only (no memory, no self-knowledge)
			sysPrompt := r.compactSystemPrompt()
			if memoryFacts != "" {
				sysPrompt += "\n" + memoryFacts
			}
			conv.System(sysPrompt)

			// User message: ONLY the raw query + pipeline context
			userMsg := percept.Raw
			if pipeCtx := pipe.BuildContext(); pipeCtx != "" {
				userMsg += "\n\nPrevious steps:\n" + pipeCtx
			}
			conv.User(userMsg)

			// If we have a raw result from the last step, include it as a tool observation
			if pipe.StepCount() > 0 && pipe.LastResult() != "" {
				lastStep := pipe.steps[len(pipe.steps)-1]
				conv.ToolResult(lastStep.ToolName, SmartTruncate(lastStep.ToolName, lastStep.RawResult))
			}

			r.Conv = conv
		}

		// Call LLM with native tool definitions
		llmRes, err := r.callLLMNative()
		if err != nil {
			return fmt.Errorf("reasoner iteration %d: %w", i, err)
		}

		fullResponse := llmRes.Content

		// Convert native tool calls to our internal format
		var toolCalls []toolCall
		usedNativeAPI := len(llmRes.ToolCalls) > 0

		if usedNativeAPI {
			// Native tool calling — model returned structured tool calls
			r.Conv.AssistantToolCalls(fullResponse, llmRes.ToolCalls)
			for _, ntc := range llmRes.ToolCalls {
				tc := toolCall{Name: ntc.Function.Name, Args: make(map[string]string)}
				for k, v := range ntc.Function.Arguments {
					tc.Args[k] = fmt.Sprintf("%v", v)
				}
				toolCalls = append(toolCalls, tc)
			}
		} else {
			// No native tool calls — either fallback text parsing or final answer
			r.Conv.Assistant(fullResponse)
			parsed := r.parseResponse(fullResponse)
			toolCalls = parsed.ToolCalls

			if len(toolCalls) == 0 {
				// No tool calls at all — this is the final answer
				finalAnswer := fullResponse
				if parsed.Answer != "" {
					finalAnswer = parsed.Answer
				}
				r.Board.Set("last_answer", finalAnswer)
				if parsed.Think != "" {
					r.Board.Set("last_thought", parsed.Think)
				}
				r.finishReasoning(percept, pipe, finalAnswer)
				return nil
			}
			// Using fallback text-parsed tool calls; reset native conv tracking
			nativeConv = nil
		}

		// Execute tool calls
		for _, tc := range toolCalls {
			// Handle request_tools meta-tool
			if tc.Name == "request_tools" {
				cat := ToolCategory(tc.Args["category"])
				newTools := ExpandCategory(cat, r.Tools.List(), r.activeTools)
				if len(newTools) > 0 {
					r.activeTools = append(r.activeTools, newTools...)
					r.activeCats[cat] = true
					names := CategoryNames(cat, r.Tools.List())
					r.emitStatus(fmt.Sprintf("  [tools] +%s", strings.Join(names, ", ")))
				}
				continue
			}

			r.emitStatus(fmt.Sprintf("→ %s %s", tc.Name, formatArgs(tc.Args)))

			start := time.Now()
			result, toolErr := r.executeTool(tc)
			duration := time.Since(start)

			// 3a. Smart truncation (tool-specific)
			if toolErr == nil {
				result = SmartTruncate(tc.Name, result)
			}

			// 3b. Result validation
			result, hint := ValidateToolResult(tc.Name, result, toolErr)

			// Emit action_recorded for Reflector stream
			r.Board.RecordAction(blackboard.ActionRecord{
				StepID:    fmt.Sprintf("reason-%d-%d", i, len(pipe.steps)),
				Tool:      tc.Name,
				Input:     formatArgs(tc.Args),
				Output:    result,
				Success:   toolErr == nil,
				Duration:  duration,
				Timestamp: time.Now(),
			})

			// 5. Synchronous reflection gate
			gateCheck := r.Gate.Check(tc.Name, result, toolErr)

			// Inject hints into result for pipeline compression
			if hint != "" {
				result = result + "\nHint: " + hint
			}
			if gateCheck.Hint != "" {
				r.emitStatus(fmt.Sprintf("⚠ %s", gateCheck.Hint))
				result = result + "\n[System: " + gateCheck.Hint + "]"
			}

			// Check if Reflector posted feedback about this action
			if reflection, ok := r.Board.Get("reflection"); ok {
				if msg, isStr := reflection.(string); isStr && msg != "" {
					r.emitStatus(fmt.Sprintf("⚠ %s", msg))
					result = result + "\n[Reflection: " + msg + "]"
					r.Board.Delete("reflection")
				}
			}

			// Send tool result back via native API
			if usedNativeAPI {
				r.Conv.NativeToolResult(tc.Name, result)
			}

			// Compress and add to pipeline (used by both paths)
			pipe.AddStep(tc.Name, result)

			// Force stop if gate says so
			if gateCheck.ForceStop {
				r.emitStatus("⚠ forcing final answer")
				finalConv := NewConversation(10)
				finalConv.System(r.compactSystemPrompt())
				forceMsg := percept.Raw
				if pipeCtx := pipe.BuildContext(); pipeCtx != "" {
					forceMsg += "\n\n" + pipeCtx
				}
				forceMsg += "\n\n[System: You MUST give your final answer now. No more tool calls.]"
				finalConv.User(forceMsg)
				r.Conv = finalConv
				resp, err := r.callLLM()
				if err == nil {
					r.Board.Set("last_answer", resp)
					r.storeToMemory(percept.Raw, resp)
				}
				return err
			}
		}

		// Track native conversation for continuation
		if usedNativeAPI {
			nativeConv = r.Conv
		} else {
			nativeConv = nil
		}
	}

	r.Board.Set("last_answer", "(reached maximum tool iterations)")
	return nil
}

// llmResult holds the response from an LLM call, including any native tool calls.
type llmResult struct {
	Content   string
	ToolCalls []ollama.ToolCall
}

// callLLM calls the model with streaming or batch mode, using native tool calling.
func (r *Reasoner) callLLM() (string, error) {
	res, err := r.callLLMNative()
	if err != nil {
		return "", err
	}
	return res.Content, err
}

// callLLMNative calls the model and returns both content and native tool calls.
func (r *Reasoner) callLLMNative() (*llmResult, error) {
	nativeTools := r.buildNativeTools()
	if r.OnToken != nil {
		return r.streamCallNative(nativeTools)
	}
	return r.batchCallNative(nativeTools)
}

// compressOldTurns compresses the oldest conversation turns to free context space.
func (r *Reasoner) compressOldTurns() {
	msgs := r.Conv.Messages()
	if len(msgs) < 5 {
		return
	}

	// Compress using LLM if compressor is available
	if r.Compressor != nil {
		// Gather oldest 4 non-system messages
		var toCompress []string
		count := 0
		for _, m := range msgs[1:] {
			if count >= 4 {
				break
			}
			toCompress = append(toCompress, m.Role+": "+m.Content)
			count++
		}
		combined := strings.Join(toCompress, "\n")
		atom, err := r.Compressor.Compress(combined, "")
		if err == nil && atom != nil {
			r.Conv.CompressOldest(4, atom.Content)
			r.emitStatus("  [budget] compressed old context")
			return
		}
	}

	// Fallback: rule-based compression (no LLM call)
	var summary []string
	count := 0
	for _, m := range msgs[1:] {
		if count >= 4 {
			break
		}
		// Keep first line of each message as summary
		first := strings.SplitN(m.Content, "\n", 2)[0]
		if len(first) > 100 {
			first = first[:100] + "..."
		}
		summary = append(summary, m.Role+": "+first)
		count++
	}
	r.Conv.CompressOldest(4, strings.Join(summary, "\n"))
	r.emitStatus("  [budget] compressed old context (rule-based)")
}

func (r *Reasoner) toolPrompt() string {
	var sb strings.Builder
	sb.WriteString("Available tools:\n")
	for _, t := range r.Tools.List() {
		sb.WriteString(fmt.Sprintf("- %s: %s\n", t.Name, t.Description))
	}
	return sb.String()
}

// toolParamSchema defines the parameters for each tool for the native tool calling API.
// Extracted from the Description strings in builtin.go.
var toolParamSchema = map[string]ollama.ToolFunctionParams{
	"read": {Type: "object", Properties: map[string]ollama.ToolProperty{
		"path":   {Type: "string", Description: "File path to read"},
		"offset": {Type: "string", Description: "Line number to start from"},
		"limit":  {Type: "string", Description: "Max lines to read"},
	}, Required: []string{"path"}},
	"write": {Type: "object", Properties: map[string]ollama.ToolProperty{
		"path":    {Type: "string", Description: "File path to create/overwrite"},
		"content": {Type: "string", Description: "File content to write"},
	}, Required: []string{"path", "content"}},
	"edit": {Type: "object", Properties: map[string]ollama.ToolProperty{
		"path": {Type: "string", Description: "File path to edit"},
		"old":  {Type: "string", Description: "Exact text to find"},
		"new":  {Type: "string", Description: "Replacement text"},
		"line": {Type: "string", Description: "Line number for context matching"},
	}, Required: []string{"path", "old", "new"}},
	"glob": {Type: "object", Properties: map[string]ollama.ToolProperty{
		"pattern": {Type: "string", Description: "Glob pattern (e.g. '**/*.go')"},
		"path":    {Type: "string", Description: "Base directory"},
	}, Required: []string{"pattern"}},
	"grep": {Type: "object", Properties: map[string]ollama.ToolProperty{
		"pattern": {Type: "string", Description: "Regex search pattern"},
		"path":    {Type: "string", Description: "Directory to search in"},
		"glob":    {Type: "string", Description: "File filter (e.g. '*.go')"},
	}, Required: []string{"pattern"}},
	"ls": {Type: "object", Properties: map[string]ollama.ToolProperty{
		"path": {Type: "string", Description: "Directory path"},
	}},
	"shell": {Type: "object", Properties: map[string]ollama.ToolProperty{
		"command": {Type: "string", Description: "Shell command to execute"},
	}, Required: []string{"command"}},
	"mkdir": {Type: "object", Properties: map[string]ollama.ToolProperty{
		"path": {Type: "string", Description: "Directory path to create"},
	}, Required: []string{"path"}},
	"tree": {Type: "object", Properties: map[string]ollama.ToolProperty{
		"path":  {Type: "string", Description: "Root directory"},
		"depth": {Type: "string", Description: "Max depth (default 3)"},
	}},
	"fetch": {Type: "object", Properties: map[string]ollama.ToolProperty{
		"url": {Type: "string", Description: "URL to fetch"},
	}, Required: []string{"url"}},
	"run": {Type: "object", Properties: map[string]ollama.ToolProperty{
		"command": {Type: "string", Description: "Command to execute"},
		"stdin":   {Type: "string", Description: "Standard input"},
	}, Required: []string{"command"}},
	"sysinfo": {Type: "object", Properties: map[string]ollama.ToolProperty{}},
	"find_replace": {Type: "object", Properties: map[string]ollama.ToolProperty{
		"path":        {Type: "string", Description: "File path"},
		"pattern":     {Type: "string", Description: "Regex pattern to find"},
		"replacement": {Type: "string", Description: "Replacement string"},
		"all":         {Type: "string", Description: "Replace all occurrences ('true')"},
	}, Required: []string{"path", "pattern", "replacement"}},
	"git": {Type: "object", Properties: map[string]ollama.ToolProperty{
		"command": {Type: "string", Description: "Git subcommand (e.g. 'status', 'diff', 'log --oneline -10')"},
	}, Required: []string{"command"}},
	"patch": {Type: "object", Properties: map[string]ollama.ToolProperty{
		"path":   {Type: "string", Description: "File path"},
		"before": {Type: "string", Description: "Exact multi-line text to find"},
		"after":  {Type: "string", Description: "Replacement text"},
	}, Required: []string{"path", "before", "after"}},
	"replace_all": {Type: "object", Properties: map[string]ollama.ToolProperty{
		"old":  {Type: "string", Description: "String to find"},
		"new":  {Type: "string", Description: "Replacement string"},
		"glob": {Type: "string", Description: "File pattern (e.g. '*.go')"},
	}, Required: []string{"old", "new"}},
	"diff": {Type: "object", Properties: map[string]ollama.ToolProperty{
		"path":   {Type: "string", Description: "File path"},
		"staged": {Type: "string", Description: "Show staged changes ('true')"},
	}},
	"clipboard": {Type: "object", Properties: map[string]ollama.ToolProperty{
		"action":  {Type: "string", Description: "Action to perform", Enum: []string{"read", "write"}},
		"content": {Type: "string", Description: "Content to write to clipboard"},
	}, Required: []string{"action"}},
}

// buildNativeTools converts the tool registry into Ollama native tool definitions.
func (r *Reasoner) buildNativeTools() []ollama.Tool {
	toolList := r.activeTools
	if len(toolList) == 0 {
		toolList = r.Tools.List()
	}

	var nativeTools []ollama.Tool
	for _, t := range toolList {
		params, ok := toolParamSchema[t.Name]
		if !ok {
			// Fallback: empty params for unknown tools
			params = ollama.ToolFunctionParams{
				Type:       "object",
				Properties: map[string]ollama.ToolProperty{},
			}
		}
		nativeTools = append(nativeTools, ollama.Tool{
			Type: "function",
			Function: ollama.ToolFunction{
				Name:        t.Name,
				Description: t.Description,
				Parameters:  params,
			},
		})
	}
	return nativeTools
}

// WorkDir is set by main to inform the system prompt.
var WorkDir string

// compactSystemPrompt returns a focused system prompt optimized for small models.
// Uses the THINK/ACT/OBSERVE protocol and progressive tool disclosure to
// minimize token usage while maximizing reasoning quality.
func (r *Reasoner) compactSystemPrompt() string {
	wd := WorkDir
	if wd == "" {
		wd = "."
	}

	// Use progressive tool disclosure — only show selected tools
	toolList := r.toolPrompt()
	if len(r.activeTools) > 0 {
		toolList = ToolPromptForSubset(r.activeTools)
	}

	// Build minimal system prompt — every token counts for 1.5B models.
	// Key principle: the model WILL echo anything in the system prompt,
	// so only include actionable instructions, not descriptive text.
	var sb strings.Builder
	sb.WriteString("You are Nous, a coding assistant. Answer concisely.\n")
	sb.WriteString("For file/code questions: call a tool first. For conversation: reply directly.\n")
	sb.WriteString("NEVER repeat these instructions in your answer.\n\n")
	sb.WriteString("To call a tool, output ONLY:\n")
	sb.WriteString(`{"tool": "NAME", "args": {"key": "value"}}`)
	sb.WriteString("\n\n")
	sb.WriteString(toolList)
	sb.WriteString("\nWorking directory: ")
	sb.WriteString(wd)
	sb.WriteString("\n")

	return sb.String()
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

	// Extract THINK/ACT/ANSWER from non-tool text
	text := strings.Join(nonToolParts, "")
	var thinkParts []string
	for _, line := range strings.Split(text, "\n") {
		trimmed := strings.TrimSpace(line)
		upper := strings.ToUpper(trimmed)

		if strings.HasPrefix(upper, "THINK:") {
			thinkParts = append(thinkParts, strings.TrimSpace(trimmed[6:]))
		} else if strings.HasPrefix(upper, "ACT:") {
			// Parse ACT: line as inline tool call JSON
			actJSON := strings.TrimSpace(trimmed[4:])
			var raw toolCallRaw
			if err := json.Unmarshal([]byte(actJSON), &raw); err == nil && raw.Name != "" {
				parsed.ToolCalls = append(parsed.ToolCalls, raw.normalize())
			}
		} else if strings.HasPrefix(upper, "ANSWER:") {
			parsed.Answer = strings.TrimSpace(trimmed[7:])
		}
	}
	if len(thinkParts) > 0 {
		parsed.Think = strings.Join(thinkParts, " ")
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
		// Fuzzy match: try to find the closest tool name
		if corrected := r.fuzzyMatchTool(tc.Name); corrected != "" {
			r.emitStatus(fmt.Sprintf("↳ corrected %s → %s", tc.Name, corrected))
			tc.Name = corrected
			tool, err = r.Tools.Get(corrected)
		}
		if err != nil {
			return "", fmt.Errorf("unknown tool %q (available: %s)", tc.Name, r.availableToolNames())
		}
	}

	if tc.Args == nil {
		tc.Args = make(map[string]string)
	}

	// Check predictive cache first (read-only tools only)
	if r.Predictor != nil {
		if cached, ok := r.Predictor.Lookup(tc.Name, tc.Args); ok {
			r.emitStatus("↳ cached")
			return cached, nil
		}
	}

	// Check if this is a dangerous action requiring confirmation
	if r.Confirm != nil {
		if reason, dangerous := IsDangerous(tc.Name); dangerous {
			action := fmt.Sprintf("%s — %s", reason, formatArgs(tc.Args))
			// Generate diff preview for file-modifying tools
			detail := r.buildDiffPreview(tc)
			if !r.Confirm(action, detail) {
				return "Action denied by user.", nil
			}
		}
	}

	result, execErr := tool.Execute(tc.Args)

	// Feed result to predictor for speculative pre-computation
	if r.Predictor != nil && execErr == nil {
		r.Predictor.Predict(tc.Name, tc.Args, result)
	}

	return result, execErr
}

// buildDiffPreview generates a colored diff preview for file-modifying tools.
// Returns an empty string if a preview cannot be generated.
func (r *Reasoner) buildDiffPreview(tc toolCall) string {
	wd := WorkDir
	if wd == "" {
		wd = "."
	}

	resolvePath := func(path string) string {
		if filepath.IsAbs(path) {
			return path
		}
		return filepath.Join(wd, path)
	}

	switch tc.Name {
	case "write":
		path := tc.Args["path"]
		content := tc.Args["content"]
		if path == "" || content == "" {
			return ""
		}
		resolved := resolvePath(path)
		oldData, err := os.ReadFile(resolved)
		if err != nil {
			// New file — show write preview
			return FormatWritePreview(path, content)
		}
		return FormatEditPreview(path, string(oldData), content)

	case "edit":
		path := tc.Args["path"]
		oldText := tc.Args["old"]
		newText := tc.Args["new"]
		if path == "" || oldText == "" {
			return ""
		}
		resolved := resolvePath(path)
		data, err := os.ReadFile(resolved)
		if err != nil {
			return ""
		}
		oldContent := string(data)
		if !strings.Contains(oldContent, oldText) {
			return ""
		}
		newContent := strings.Replace(oldContent, oldText, newText, 1)
		return FormatEditPreview(path, oldContent, newContent)

	case "patch":
		path := tc.Args["path"]
		before := tc.Args["before"]
		after := tc.Args["after"]
		if path == "" || before == "" {
			return ""
		}
		resolved := resolvePath(path)
		data, err := os.ReadFile(resolved)
		if err != nil {
			return ""
		}
		oldContent := string(data)
		if !strings.Contains(oldContent, before) {
			return ""
		}
		newContent := strings.Replace(oldContent, before, after, 1)
		return FormatEditPreview(path, oldContent, newContent)

	case "find_replace":
		// find_replace uses regex, so we can't easily preview without importing regexp.
		// Return empty — the confirmation still shows the action detail.
		return ""
	}

	return ""
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

// modelOpts returns tuned ModelOptions for small model reliability.
// RepeatPenalty prevents the repetition loops that plague 1.5B models.
// NumPredict is kept modest to encourage tool calls over long prose.
func (r *Reasoner) modelOpts() *ollama.ModelOptions {
	return &ollama.ModelOptions{
		Temperature:   0.5,
		NumPredict:    512,
		RepeatPenalty: 1.15,
		RepeatLastN:   64,
	}
}

func (r *Reasoner) streamCallNative(nativeTools []ollama.Tool) (*llmResult, error) {
	var full strings.Builder
	var allToolCalls []ollama.ToolCall
	opts := r.modelOpts()
	_, err := r.LLM.ChatStreamWithTools(r.Conv.Messages(), nativeTools, opts,
		func(token string, toolCalls []ollama.ToolCall, done bool) {
			if token != "" {
				full.WriteString(token)
				if r.OnToken != nil {
					r.OnToken(token, done)
				}
			}
			if len(toolCalls) > 0 {
				allToolCalls = append(allToolCalls, toolCalls...)
			}
		})
	if err != nil {
		return nil, err
	}
	return &llmResult{Content: full.String(), ToolCalls: allToolCalls}, nil
}

func (r *Reasoner) batchCallNative(nativeTools []ollama.Tool) (*llmResult, error) {
	resp, err := r.LLM.ChatWithTools(r.Conv.Messages(), nativeTools, r.modelOpts())
	if err != nil {
		return nil, err
	}
	return &llmResult{Content: resp.Message.Content, ToolCalls: resp.Message.ToolCalls}, nil
}

// Legacy methods kept for compatibility with non-tool-calling code paths.
func (r *Reasoner) streamCall() (string, error) {
	res, err := r.streamCallNative(nil)
	if err != nil {
		return "", err
	}
	return res.Content, nil
}

func (r *Reasoner) batchCall() (string, error) {
	res, err := r.batchCallNative(nil)
	if err != nil {
		return "", err
	}
	return res.Content, nil
}

func (r *Reasoner) emitStatus(msg string) {
	if r.OnStatus != nil {
		r.OnStatus(msg)
	}
}

// recallKeyFacts retrieves only essential facts for the system prompt.
// Unlike recallMemories, this is selective — it only returns facts the model
// NEEDS to know (like user identity), not full conversation history.
func (r *Reasoner) recallKeyFacts(input string) string {
	var facts []string

	// Get user identity if stored
	if r.WorkingMem != nil {
		items := r.WorkingMem.MostRelevant(5)
		for _, item := range items {
			if strings.HasPrefix(item.Key, "user_identity") {
				facts = append(facts, fmt.Sprintf("User info: %v", item.Value))
			}
		}
	}

	// Get relevant codebase context
	if r.CodeIndex != nil && r.CodeIndex.Size() > 0 {
		indexCtx := r.CodeIndex.RelevantContext(input, 3)
		if indexCtx != "" {
			facts = append(facts, indexCtx)
		}
	}

	if len(facts) == 0 {
		return ""
	}
	return strings.Join(facts, "\n")
}

// recallMemories retrieves relevant context from working and long-term memory.
func (r *Reasoner) recallMemories(input string) string {
	var parts []string

	// Recall from working memory (most relevant items)
	if r.WorkingMem != nil {
		items := r.WorkingMem.MostRelevant(5)
		if len(items) > 0 {
			var memLines []string
			for _, item := range items {
				memLines = append(memLines, fmt.Sprintf("- %s → %v", item.Key, item.Value))
			}
			parts = append(parts, "[Previous context — use this to answer]\n"+strings.Join(memLines, "\n"))
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

	// Recall from codebase index (structural context)
	if r.CodeIndex != nil && r.CodeIndex.Size() > 0 {
		indexCtx := r.CodeIndex.RelevantContext(input, 5)
		if indexCtx != "" {
			parts = append(parts, indexCtx)
		}
	}

	if len(parts) == 0 {
		return ""
	}
	return strings.Join(parts, "\n\n")
}

// consultPriorKnowledge checks learned patterns and recipes for relevant prior experience.
// This is the key feedback loop: what the Learner extracted gets injected back into reasoning.
func (r *Reasoner) consultPriorKnowledge(intent, rawInput string) string {
	var parts []string

	// Check learned behavioral patterns from the Learner
	if r.Learner != nil {
		patterns := r.Learner.FindRelevantPatterns(intent)
		if len(patterns) > 0 {
			var patternLines []string
			for _, p := range patterns {
				if len(patternLines) >= 3 {
					break
				}
				chain := ""
				for i, tool := range p.ToolChain {
					if i > 0 {
						chain += " → "
					}
					chain += tool
				}
				patternLines = append(patternLines, fmt.Sprintf("- %s (confidence: %.0f%%, used %d times)", chain, p.Confidence*100, p.Uses))
			}
			if len(patternLines) > 0 {
				parts = append(parts, "[Learned Patterns]\nFor similar tasks, these tool sequences worked before:\n"+strings.Join(patternLines, "\n"))
			}
		}
	}

	// Check recipe book for matching multi-step sequences
	if r.Recipes != nil {
		recipes := r.Recipes.Match(intent, rawInput)
		if len(recipes) > 0 {
			best := recipes[0]
			var stepNames []string
			for _, s := range best.Steps {
				stepNames = append(stepNames, s.Tool)
			}
			chain := strings.Join(stepNames, " → ")
			parts = append(parts, fmt.Sprintf("[Recipe: %s]\nTried approach: %s (used %d times, %.0f%% success)", best.Name, chain, best.Uses, best.Confidence()*100))
		}
	}

	if len(parts) == 0 {
		return ""
	}
	return strings.Join(parts, "\n\n")
}

// fuzzyMatchTool finds the closest tool name using simple substring/alias matching.
// Handles common LLM mistakes like "find_replace" → "find_replace", "search" → "grep", etc.
func (r *Reasoner) fuzzyMatchTool(name string) string {
	lower := strings.ToLower(name)

	// Common aliases that 1.5B models generate
	aliases := map[string]string{
		"search":        "grep",
		"find":          "grep",
		"cat":           "read",
		"view":          "read",
		"open":          "read",
		"list":          "ls",
		"dir":           "ls",
		"create":        "write",
		"touch":         "write",
		"exec":          "shell",
		"execute":       "shell",
		"cmd":           "shell",
		"command":       "shell",
		"bash":          "shell",
		"mv":            "shell",
		"cp":            "shell",
		"rm":            "shell",
		"replace":       "find_replace",
		"sed":           "find_replace",
		"status":        "git",
		"commit":        "git",
		"log":           "git",
		"browse":        "fetch",
		"curl":          "fetch",
		"wget":          "fetch",
		"info":          "sysinfo",
		"system":        "sysinfo",
	}

	if corrected, ok := aliases[lower]; ok {
		return corrected
	}

	// Substring match: if the name contains a tool name or vice versa
	for _, t := range r.Tools.List() {
		if strings.Contains(lower, t.Name) || strings.Contains(t.Name, lower) {
			return t.Name
		}
	}

	return ""
}

// availableToolNames returns a comma-separated list of tool names for error messages.
func (r *Reasoner) availableToolNames() string {
	var names []string
	for _, t := range r.Tools.List() {
		names = append(names, t.Name)
	}
	return strings.Join(names, ", ")
}

// finishReasoning records recipes, emits goal completion for the Learner, and stores to memory.
func (r *Reasoner) finishReasoning(percept blackboard.Percept, pipe *Pipeline, finalAnswer string) {
	// Record successful tool sequence as a recipe
	if r.Recipes != nil && pipe.StepCount() >= 2 {
		r.Recipes.Record(pipe, percept.Intent, percept.Raw)
	}

	// Emit goal completion so Learner can extract patterns
	if pipe.StepCount() >= 1 {
		goalID := fmt.Sprintf("reason-%d", time.Now().UnixMilli())
		r.Board.PushGoal(blackboard.Goal{
			ID:          goalID,
			Description: percept.Raw,
			Status:      "pending",
			CreatedAt:   time.Now(),
		})
		var steps []blackboard.Step
		for j, s := range pipe.steps {
			steps = append(steps, blackboard.Step{
				ID:          fmt.Sprintf("%s-step-%d", goalID, j),
				Description: s.Summary,
				Tool:        s.ToolName,
				Status:      "done",
				Result:      s.Summary,
			})
		}
		r.Board.SetPlan(blackboard.Plan{
			GoalID: goalID,
			Steps:  steps,
			Status: "completed",
		})
		r.Board.UpdateGoalStatus(goalID, "completed")
	}

	r.storeToMemory(percept.Raw, finalAnswer)
}

// storeToMemory saves the current interaction context to working memory.
func (r *Reasoner) storeToMemory(input, answer string) {
	if r.WorkingMem == nil {
		return
	}

	// Store as a conversational exchange so the model understands context
	key := input
	if len(key) > 60 {
		key = key[:60] + "..."
	}

	value := answer
	if len(value) > 150 {
		value = value[:150] + "..."
	}

	// Store the exchange with high strength for recent recall
	r.WorkingMem.Store("user said: "+key, "I replied: "+value, 0.9)

	// Extract and store key facts (names, preferences, etc.)
	lower := strings.ToLower(input)
	if strings.Contains(lower, "my name") || strings.Contains(lower, "i'm ") || strings.Contains(lower, "i am ") {
		// Store the user's self-introduction as a persistent fact
		r.WorkingMem.Store("user_identity", input, 1.0)
	}
}
