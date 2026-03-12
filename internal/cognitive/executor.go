package cognitive

import (
	"context"
	"fmt"
	"strings"
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
		if corrected := normalizeExecutorToolName(toolName); corrected != "" && corrected != toolName {
			toolName = corrected
			step.Tool = corrected
			tool, err = e.Tools.Get(toolName)
		}
		if err != nil {
			return "", fmt.Errorf("unknown tool %q: %w", toolName, err)
		}
	}

	// Build args from the step's Args map
	args := make(map[string]string)
	for k, v := range step.Args {
		args[k] = v
	}
	if raw, ok := args["raw"]; ok {
		for k, v := range parseStepArgs(raw) {
			if _, exists := args[k]; !exists {
				args[k] = v
			}
		}
	}
	args = correctArgNames(toolName, args)
	normalizeToolArgs(toolName, args)
	inferArgsFromStepDescription(toolName, step.Description, args)

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

func normalizeToolArgs(toolName string, args map[string]string) {
	switch toolName {
	case "write":
		if args["path"] == "" {
			if args["file"] != "" {
				args["path"] = args["file"]
			} else if args["filename"] != "" {
				args["path"] = args["filename"]
			}
		}
		if args["content"] == "" {
			if args["text"] != "" {
				args["content"] = args["text"]
			} else if args["body"] != "" {
				args["content"] = args["body"]
			} else if args["new"] != "" {
				args["content"] = args["new"]
			}
		}
	case "edit":
		if args["path"] == "" && args["file"] != "" {
			args["path"] = args["file"]
		}
		if args["old"] == "" && args["before"] != "" {
			args["old"] = args["before"]
		}
		if args["new"] == "" {
			if args["after"] != "" {
				args["new"] = args["after"]
			} else if args["content"] != "" {
				args["new"] = args["content"]
			}
		}
	case "fetch":
		if args["url"] == "" && args["path"] != "" {
			args["url"] = args["path"]
		}
		if args["url"] == "" {
			query := args["query"]
			if query == "" {
				query = args["pattern"]
			}
			if query == "" {
				query = args["raw"]
			}
			if slug := slugifyWords(query); slug != "" {
				args["url"] = wikipediaSummaryURL(slug)
			}
		}
	}
}

func normalizeExecutorToolName(name string) string {
	switch strings.ToLower(strings.TrimSpace(name)) {
	case "search", "web_search", "websearch", "lookup":
		return "fetch"
	case "find":
		return "grep"
	case "create", "touch":
		return "write"
	default:
		return name
	}
}

func inferArgsFromStepDescription(toolName string, description string, args map[string]string) {
	desc := strings.TrimSpace(description)
	if desc == "" {
		return
	}
	quoted := ""
	if first := strings.Index(desc, `"`); first >= 0 {
		if second := strings.Index(desc[first+1:], `"`); second >= 0 {
			quoted = desc[first+1 : first+1+second]
		}
	}
	switch toolName {
	case "grep":
		if strings.TrimSpace(args["pattern"]) == "" {
			if quoted != "" {
				args["pattern"] = quoted
			} else if term := inferSearchTerm(desc); term != "" {
				args["pattern"] = term
			}
		}
	case "fetch":
		if strings.TrimSpace(args["url"]) == "" && quoted == "" {
			if term := inferSearchTerm(desc); term != "" {
				args["url"] = wikipediaSummaryURL(term)
			}
		}
	case "write":
		if strings.TrimSpace(args["path"]) == "" && quoted != "" && strings.Contains(quoted, ".") {
			args["path"] = quoted
		}
	}
}

func inferSearchTerm(description string) string {
	lower := strings.ToLower(description)
	for _, marker := range []string{" for ", " about ", " on ", " regarding "} {
		if idx := strings.Index(lower, marker); idx >= 0 {
			term := strings.TrimSpace(description[idx+len(marker):])
			if term != "" {
				words := strings.Fields(term)
				if len(words) > 4 {
					words = words[:4]
				}
				return strings.Trim(strings.Join(words, " "), ".,;:")
			}
		}
	}
	words := strings.Fields(description)
	if len(words) == 0 {
		return ""
	}
	last := strings.Trim(words[len(words)-1], ".,;:")
	if len(last) > 2 {
		return last
	}
	return ""
}
