package cognitive

import (
	"bytes"
	"context"
	"fmt"
	"os"
	"os/exec"
	"strings"
	"time"

	"github.com/artaeon/nous/internal/blackboard"
	"github.com/artaeon/nous/internal/ollama"
)

// Executor runs tools and actions in the real world.
// It watches the blackboard for pending actions and plan steps,
// executes them safely, and records the results.
type Executor struct {
	Base
	// WorkDir is the working directory for shell commands.
	WorkDir string
	// AllowShell controls whether shell execution is permitted.
	AllowShell bool
}

func NewExecutor(board *blackboard.Blackboard, llm *ollama.Client) *Executor {
	wd, _ := os.Getwd()
	return &Executor{
		Base:       Base{Board: board, LLM: llm},
		WorkDir:    wd,
		AllowShell: false, // Disabled by default for safety
	}
}

func (e *Executor) Name() string { return "executor" }

func (e *Executor) Run(ctx context.Context) error {
	events := e.Board.Subscribe("plan_set")

	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case ev := <-events:
			plan, ok := ev.Payload.(blackboard.Plan)
			if !ok {
				continue
			}
			e.executePlan(ctx, plan)
		}
	}
}

func (e *Executor) executePlan(ctx context.Context, plan blackboard.Plan) {
	plan.Status = "executing"
	e.Board.SetPlan(plan)

	for i := range plan.Steps {
		select {
		case <-ctx.Done():
			return
		default:
		}

		step := &plan.Steps[i]
		step.Status = "running"
		e.Board.SetPlan(plan)

		start := time.Now()
		result, err := e.executeStep(step)
		duration := time.Since(start)

		success := err == nil
		if err != nil {
			step.Status = "failed"
			step.Result = err.Error()
		} else {
			step.Status = "done"
			step.Result = result
		}

		e.Board.RecordAction(blackboard.ActionRecord{
			StepID:    step.ID,
			Tool:      step.Tool,
			Input:     step.Args["raw"],
			Output:    step.Result,
			Success:   success,
			Duration:  duration,
			Timestamp: time.Now(),
		})

		e.Board.SetPlan(plan)

		if !success {
			plan.Status = "failed"
			e.Board.SetPlan(plan)
			e.Board.UpdateGoalStatus(plan.GoalID, "failed")
			return
		}
	}

	plan.Status = "completed"
	e.Board.SetPlan(plan)
	e.Board.UpdateGoalStatus(plan.GoalID, "completed")
}

func (e *Executor) executeStep(step *blackboard.Step) (string, error) {
	switch strings.ToLower(step.Tool) {
	case "shell":
		return e.execShell(step.Args["raw"])
	case "read":
		return e.execRead(step.Args["raw"])
	case "write":
		return e.execWrite(step.Args)
	case "search":
		return e.execSearch(step.Args["raw"])
	default:
		return "", fmt.Errorf("unknown tool: %s", step.Tool)
	}
}

func (e *Executor) execShell(command string) (string, error) {
	if !e.AllowShell {
		return "", fmt.Errorf("shell execution disabled — start with --allow-shell to enable")
	}

	cmd := exec.Command("sh", "-c", command)
	cmd.Dir = e.WorkDir

	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	if err := cmd.Run(); err != nil {
		return "", fmt.Errorf("shell error: %s\nstderr: %s", err, stderr.String())
	}

	return stdout.String(), nil
}

func (e *Executor) execRead(path string) (string, error) {
	path = strings.TrimSpace(path)
	data, err := os.ReadFile(path)
	if err != nil {
		return "", fmt.Errorf("read %s: %w", path, err)
	}

	// Limit output size to avoid overwhelming the context
	content := string(data)
	if len(content) > 8192 {
		content = content[:8192] + "\n... (truncated)"
	}

	return content, nil
}

func (e *Executor) execWrite(args map[string]string) (string, error) {
	path := strings.TrimSpace(args["raw"])
	content, ok := args["content"]
	if !ok {
		return "", fmt.Errorf("write requires 'content' argument")
	}

	if err := os.WriteFile(path, []byte(content), 0644); err != nil {
		return "", fmt.Errorf("write %s: %w", path, err)
	}

	return fmt.Sprintf("wrote %d bytes to %s", len(content), path), nil
}

func (e *Executor) execSearch(query string) (string, error) {
	// Simple recursive grep through the working directory
	if !e.AllowShell {
		return "", fmt.Errorf("search requires shell access — start with --allow-shell to enable")
	}

	cmd := exec.Command("grep", "-rn", "--include=*.go", "--include=*.md", "--include=*.txt", query, e.WorkDir)
	var stdout bytes.Buffer
	cmd.Stdout = &stdout

	_ = cmd.Run() // grep returns exit 1 on no match

	result := stdout.String()
	if result == "" {
		return "no matches found", nil
	}

	if len(result) > 4096 {
		result = result[:4096] + "\n... (truncated)"
	}

	return result, nil
}
