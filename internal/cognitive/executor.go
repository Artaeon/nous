package cognitive

import (
	"context"
	"fmt"
	"time"

	"github.com/artaeon/nous/internal/blackboard"
	"github.com/artaeon/nous/internal/ollama"
	"github.com/artaeon/nous/internal/tools"
)

// Executor runs tools and actions in the real world.
// It watches the blackboard for plans created by the Planner,
// executes each step using the shared tool registry, and records results.
type Executor struct {
	Base
	Tools *tools.Registry
}

func NewExecutor(board *blackboard.Blackboard, llm *ollama.Client, toolReg *tools.Registry) *Executor {
	return &Executor{
		Base:  Base{Board: board, LLM: llm},
		Tools: toolReg,
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
			// Only execute plans in "draft" status (freshly created by Planner)
			if plan.Status != "draft" {
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

		// Check for Reflector feedback — if it says needs_replan, abort
		if replanID, ok := e.Board.Get("needs_replan"); ok {
			if id, isStr := replanID.(string); isStr && id == step.ID {
				plan.Status = "failed"
				e.Board.SetPlan(plan)
				e.Board.UpdateGoalStatus(plan.GoalID, "failed")
				e.Board.Delete("needs_replan")
				return
			}
		}

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
	toolName := step.Tool
	if toolName == "" {
		return "", fmt.Errorf("step has no tool specified")
	}

	tool, err := e.Tools.Get(toolName)
	if err != nil {
		return "", fmt.Errorf("unknown tool %q: %w", toolName, err)
	}

	// Build args from the step's Args map
	args := make(map[string]string)
	for k, v := range step.Args {
		args[k] = v
	}

	// If there's a "raw" arg but no specific args, try to map it to common tool parameters
	if raw, ok := args["raw"]; ok && len(args) == 1 {
		switch toolName {
		case "read", "ls", "tree", "mkdir":
			args["path"] = raw
		case "grep":
			args["pattern"] = raw
		case "glob":
			args["pattern"] = raw
		case "shell", "run":
			args["command"] = raw
		case "git":
			args["command"] = raw
		}
	}

	return tool.Execute(args)
}
