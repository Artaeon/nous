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
			// Also merge into args where placeholders exist
			step = e.resolveArgs(step, ctx)
		}

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
func (e *Executor) ExecuteStep(step ToolStep, context map[string]string) (string, error) {
	tool, err := e.Tools.Get(step.Tool)
	if err != nil {
		return "", fmt.Errorf("unknown tool %q: %w", step.Tool, err)
	}

	// Build args, substituting context references
	args := make(map[string]string, len(step.Args))
	for k, v := range step.Args {
		args[k] = e.substituteVars(v, context)
	}

	output, err := tool.Execute(args)
	if err != nil {
		return output, err
	}

	return output, nil
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
		output, err := e.ExecuteStep(step, context)
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
	case "web_search", "fetch", "read", "glob", "grep", "ls", "tree",
		"calculator", "convert", "weather", "translate", "timer",
		"notes", "todos", "habits", "expenses", "system_info":
		return true
	}
	return false
}
