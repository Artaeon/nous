package agent

import (
	"fmt"
	"strings"
	"time"

	"github.com/artaeon/nous/internal/tools"
)

// Executor runs tool chains and manages result passing between steps.
type Executor struct {
	Tools        *tools.Registry
	Workspace    string
	MaxRetries   int           // max retries per step (default 3)
	StepTimeout  time.Duration // timeout per step (default 30s)
	MaxToolCalls int           // safety limit per task (default 20)
	Brain        *CognitiveBridge // cognitive systems for synthesis
}

// StepResult is the outcome of a single tool step.
type StepResult struct {
	Tool     string        `json:"tool"`
	Output   string        `json:"output"`
	Error    string        `json:"error,omitempty"`
	Duration time.Duration `json:"duration_ns"`
}

// ChainResult is the outcome of a full tool chain.
type ChainResult struct {
	Steps       []StepResult  `json:"steps"`
	FinalOutput string        `json:"final_output"`
	Duration    time.Duration `json:"duration_ns"`
	ToolCalls   int           `json:"tool_calls"`
}

// NewExecutor creates an executor with default settings.
func NewExecutor(toolReg *tools.Registry, workspace string) *Executor {
	return &Executor{
		Tools:        toolReg,
		Workspace:    workspace,
		MaxRetries:   3,
		StepTimeout:  30 * time.Second,
		MaxToolCalls: 20,
	}
}

// ExecuteChain runs a sequence of tool steps, passing results between them.
// The context map is pre-populated with any data the caller provides (e.g.
// results from previous tasks). Step outputs are stored under their OutputKey
// and available to subsequent steps via the DependsOn index.
func (e *Executor) ExecuteChain(chain []ToolStep, context map[string]string) (*ChainResult, error) {
	if len(chain) == 0 {
		return &ChainResult{}, nil
	}

	start := time.Now()
	result := &ChainResult{}
	stepOutputs := make([]string, len(chain))

	// Merge caller context into a working copy
	ctx := make(map[string]string, len(context))
	for k, v := range context {
		ctx[k] = v
	}

	for i, step := range chain {
		if result.ToolCalls >= e.MaxToolCalls {
			return result, fmt.Errorf("safety limit: exceeded %d tool calls", e.MaxToolCalls)
		}

		// If this step depends on a previous step, inject its output
		if step.DependsOn >= 0 && step.DependsOn < i {
			depOutput := stepOutputs[step.DependsOn]
			ctx["_prev"] = depOutput
		}

		// Always resolve ${var} placeholders in args against the context.
		// This handles both DependsOn references and cross-task references
		// like ${synthesis} from a previous task's output.
		step = e.resolveArgs(step, ctx)

		// Resolve workspace-relative paths
		step = e.resolveWorkspace(step)

		sr, err := e.executeWithRetry(step, ctx)
		result.Steps = append(result.Steps, sr)
		result.ToolCalls++

		if err != nil {
			result.Duration = time.Since(start)
			result.FinalOutput = sr.Output
			return result, fmt.Errorf("step %d (%s) failed: %w", i, step.Tool, err)
		}

		stepOutputs[i] = sr.Output

		// Store output in context for later steps
		if step.OutputKey != "" {
			ctx[step.OutputKey] = sr.Output
		}
	}

	// Final output is the last step's output
	if len(stepOutputs) > 0 {
		result.FinalOutput = stepOutputs[len(stepOutputs)-1]
	}
	result.Duration = time.Since(start)
	return result, nil
}

// ExecuteStep runs a single tool step.
// Steps whose tool name starts with "_" are cognitive operations handled
// by the CognitiveBridge (summarize, think, compose, generate_doc).
// All other steps are dispatched to the tool registry.
func (e *Executor) ExecuteStep(step ToolStep, context map[string]string) (string, error) {
	// Build args, substituting context references
	args := make(map[string]string, len(step.Args))
	for k, v := range step.Args {
		args[k] = e.substituteVars(v, context)
	}

	// Cognitive pseudo-tools — handled by the brain, not the tool registry
	if strings.HasPrefix(step.Tool, "_") {
		return e.executeCognitiveStep(step.Tool, args, context)
	}

	tool, err := e.Tools.Get(step.Tool)
	if err != nil {
		return "", fmt.Errorf("unknown tool %q: %w", step.Tool, err)
	}

	output, err := tool.Execute(args)
	if err != nil {
		return output, err
	}

	return output, nil
}

// executeCognitiveStep handles internal cognitive operations.
func (e *Executor) executeCognitiveStep(tool string, args, context map[string]string) (string, error) {
	if e.Brain == nil {
		return "", fmt.Errorf("cognitive tool %q requires a brain (CognitiveBridge not connected)", tool)
	}

	switch tool {
	case "_summarize":
		text := args["text"]
		if text == "" {
			text = context["_prev"]
		}
		if text == "" {
			return "", fmt.Errorf("_summarize: no text to summarize")
		}
		return e.Brain.Summarize(text, 5), nil

	case "_think":
		query := args["query"]
		if query == "" {
			query = args["topic"]
		}
		if query == "" {
			return "", fmt.Errorf("_think: no query provided")
		}
		// Use topic-filtered thinking to prevent knowledge graph contamination
		topic := args["topic"]
		if topic == "" {
			topic = query
		}
		result := e.Brain.ThinkAbout(topic, query)
		if result != "" {
			return result, nil
		}
		// Fallback to Compose
		result = e.Brain.Compose(query)
		if result != "" {
			return result, nil
		}
		// Both failed — return a stub so downstream steps get something
		return "No detailed analysis available for: " + query, nil

	case "_reason":
		question := args["question"]
		if question == "" {
			question = args["query"]
		}
		answer, _ := e.Brain.Reason(question)
		if answer == "" {
			return "", fmt.Errorf("_reason: could not reason about %q", question)
		}
		return answer, nil

	case "_generate_doc":
		topic := args["topic"]
		style := args["style"]
		if style == "" {
			style = "report"
		}
		doc := e.Brain.GenerateDocument(topic, style)
		if doc != "" {
			return doc, nil
		}
		// DocGen couldn't produce content (topic not in knowledge graph).
		// Fall back to synthesis from context, or a stub.
		synth := e.Brain.SynthesizeResults("write a "+style+" about "+topic, context)
		if synth != "" {
			return synth, nil
		}
		return "# " + topic + "\n\nDocument generation pending — topic not yet in knowledge base.", nil

	case "_synthesize":
		goal := args["goal"]
		if goal == "" {
			goal = "summarize the findings"
		}
		// Filter context to only include task results, not internal keys
		filtered := make(map[string]string)
		for k, v := range context {
			if k == "_prev" || k == "" || v == "" {
				continue
			}
			filtered[k] = v
		}
		result := e.Brain.SynthesizeResults(goal, filtered)
		if result == "" {
			return "Analysis pending — insufficient data gathered.", nil
		}
		return result, nil

	case "_compose":
		query := args["query"]
		if query == "" {
			query = context["_prev"]
		}
		result := e.Brain.Compose(query)
		if result == "" {
			return "", fmt.Errorf("_compose: could not compose response for %q", query)
		}
		return result, nil

	default:
		return "", fmt.Errorf("unknown cognitive tool: %q", tool)
	}
}

// executeWithRetry runs a step with retries on failure.
func (e *Executor) executeWithRetry(step ToolStep, context map[string]string) (StepResult, error) {
	maxRetries := e.MaxRetries
	if maxRetries <= 0 {
		maxRetries = 1
	}

	var lastErr error
	for attempt := 0; attempt < maxRetries; attempt++ {
		start := time.Now()
		output, err := e.executeWithTimeout(step, context)
		dur := time.Since(start)

		if err == nil {
			return StepResult{
				Tool:     step.Tool,
				Output:   output,
				Duration: dur,
			}, nil
		}

		lastErr = err
		// Brief backoff between retries
		if attempt < maxRetries-1 {
			time.Sleep(time.Duration(attempt+1) * 500 * time.Millisecond)
		}
	}

	return StepResult{
		Tool:  step.Tool,
		Error: lastErr.Error(),
	}, lastErr
}

// executeWithTimeout wraps ExecuteStep with a timeout.
// If StepTimeout is zero, runs without a timeout.
func (e *Executor) executeWithTimeout(step ToolStep, context map[string]string) (string, error) {
	if e.StepTimeout <= 0 {
		return e.ExecuteStep(step, context)
	}

	type result struct {
		output string
		err    error
	}
	ch := make(chan result, 1)
	go func() {
		out, err := e.ExecuteStep(step, context)
		ch <- result{out, err}
	}()

	select {
	case r := <-ch:
		return r.output, r.err
	case <-time.After(e.StepTimeout):
		return "", fmt.Errorf("step %q timed out after %v", step.Tool, e.StepTimeout)
	}
}

// resolveArgs replaces context variable references in step arguments.
// Variables use the form ${key}.
func (e *Executor) resolveArgs(step ToolStep, context map[string]string) ToolStep {
	resolved := ToolStep{
		Tool:      step.Tool,
		DependsOn: step.DependsOn,
		OutputKey: step.OutputKey,
		Args:      make(map[string]string, len(step.Args)),
	}
	for k, v := range step.Args {
		resolved.Args[k] = e.substituteVars(v, context)
	}
	return resolved
}

// resolveWorkspace prepends the workspace path to relative file paths.
func (e *Executor) resolveWorkspace(step ToolStep) ToolStep {
	if e.Workspace == "" {
		return step
	}

	// Only adjust path args for file tools
	switch step.Tool {
	case "write", "read", "edit", "glob", "grep", "ls", "tree", "mkdir":
		resolved := ToolStep{
			Tool:      step.Tool,
			DependsOn: step.DependsOn,
			OutputKey: step.OutputKey,
			Args:      make(map[string]string, len(step.Args)),
		}
		for k, v := range step.Args {
			if k == "path" || k == "file" || k == "dir" || k == "directory" {
				if v != "" && !strings.HasPrefix(v, "/") && !strings.HasPrefix(v, e.Workspace) {
					v = e.Workspace + "/" + v
				}
			}
			resolved.Args[k] = v
		}
		return resolved
	}
	return step
}

// substituteVars replaces ${key} references with context values.
func (e *Executor) substituteVars(s string, context map[string]string) string {
	for k, v := range context {
		s = strings.ReplaceAll(s, "${"+k+"}", v)
	}
	return s
}

// IsDangerousTool returns true for tools that modify the filesystem or
// execute commands. The agent must get human approval before running these
// outside the workspace directory.
func IsDangerousTool(name string) bool {
	switch name {
	case "shell", "run", "write", "edit", "patch", "find_replace",
		"replace_all", "mkdir", "git":
		return true
	}
	return false
}

// IsSafeTool returns true for read-only tools that can run without approval.
func IsSafeTool(name string) bool {
	switch name {
	case "websearch", "web_search", "fetch", "read", "glob", "grep", "ls", "tree",
		"calculator", "convert", "weather", "translate", "timer",
		"notes", "todos", "habits", "expenses", "system_info":
		return true
	}
	return false
}
