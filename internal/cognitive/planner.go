package cognitive

import (
	"context"
	"fmt"
	"strings"

	"github.com/artaeon/nous/internal/blackboard"
	"github.com/artaeon/nous/internal/ollama"
)

// Planner decomposes high-level goals into executable step sequences.
// Uses hierarchical task network (HTN) decomposition via the LLM
// to break complex requests into ordered sub-tasks.
type Planner struct {
	Base
}

func NewPlanner(board *blackboard.Blackboard, llm *ollama.Client) *Planner {
	return &Planner{
		Base: Base{Board: board, LLM: llm},
	}
}

func (p *Planner) Name() string { return "planner" }

func (p *Planner) Run(ctx context.Context) error {
	events := p.Board.Subscribe("goal_pushed")

	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case ev := <-events:
			goal, ok := ev.Payload.(blackboard.Goal)
			if !ok {
				continue
			}
			if err := p.plan(goal); err != nil {
				p.Board.Set("planner_error", err.Error())
			}
		}
	}
}

func (p *Planner) plan(goal blackboard.Goal) error {
	prompt := fmt.Sprintf(`Decompose this goal into concrete executable steps.
Each step should be a single action. Use this format:

STEP: <description> | TOOL: <tool_name> | ARGS: <arguments>

Available tools:
- read (read a file), write (create/overwrite a file), edit (modify part of a file)
- grep (search file contents), glob (find files by pattern), ls (list directory), tree (directory tree)
- shell (run shell command), run (run a project command like build/test)
- fetch (HTTP GET a URL), git (run git commands)
- find_replace (regex find and replace), patch (multi-line edit), diff (compare files)
- mkdir (create directory)

Goal: %s`, goal.Description)

	resp, err := p.LLM.Chat([]ollama.Message{
		{Role: "system", Content: PlanPrompt},
		{Role: "user", Content: prompt},
	}, &ollama.ModelOptions{
		Temperature: 0.3,
		NumPredict:  512,
	})
	if err != nil {
		return fmt.Errorf("planner llm: %w", err)
	}

	plan := p.parsePlan(goal.ID, resp.Message.Content)
	p.Board.SetPlan(plan)
	p.Board.UpdateGoalStatus(goal.ID, "active")

	return nil
}

func (p *Planner) parsePlan(goalID, response string) blackboard.Plan {
	var steps []blackboard.Step
	stepNum := 0

	for _, line := range strings.Split(response, "\n") {
		line = strings.TrimSpace(line)
		if !strings.HasPrefix(strings.ToUpper(line), "STEP:") {
			continue
		}

		stepNum++
		step := blackboard.Step{
			ID:     fmt.Sprintf("%s-step-%d", goalID, stepNum),
			Status: "pending",
			Args:   make(map[string]string),
		}

		parts := strings.Split(line, "|")
		for _, part := range parts {
			part = strings.TrimSpace(part)
			upper := strings.ToUpper(part)

			if strings.HasPrefix(upper, "STEP:") {
				step.Description = strings.TrimSpace(part[5:])
			} else if strings.HasPrefix(upper, "TOOL:") {
				step.Tool = strings.TrimSpace(part[5:])
			} else if strings.HasPrefix(upper, "ARGS:") {
				step.Args["raw"] = strings.TrimSpace(part[5:])
			}
		}

		steps = append(steps, step)
	}

	return blackboard.Plan{
		GoalID: goalID,
		Steps:  steps,
		Status: "draft",
	}
}
