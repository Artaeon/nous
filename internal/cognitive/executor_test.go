package cognitive

import (
	"context"
	"fmt"
	"testing"
	"time"

	"github.com/artaeon/nous/internal/blackboard"
	"github.com/artaeon/nous/internal/tools"
)

func newTestRegistry() *tools.Registry {
	r := tools.NewRegistry()
	r.Register(tools.Tool{
		Name:        "read",
		Description: "Read a file",
		Execute: func(args map[string]string) (string, error) {
			return "file content for " + args["path"], nil
		},
	})
	r.Register(tools.Tool{
		Name:        "grep",
		Description: "Search for pattern",
		Execute: func(args map[string]string) (string, error) {
			return "match: " + args["pattern"], nil
		},
	})
	return r
}

func TestExecutorRunsPlan(t *testing.T) {
	board := blackboard.New()
	reg := newTestRegistry()
	exec := NewExecutor(board, nil, reg)

	plan := blackboard.Plan{
		GoalID: "goal-1",
		Steps: []blackboard.Step{
			{
				ID:          "s1",
				Tool:        "read",
				Description: "read main.go",
				Args:        map[string]string{"path": "main.go"},
				Status:      "pending",
			},
		},
		Status: "draft",
	}

	board.PushGoal(blackboard.Goal{
		ID:     "goal-1",
		Status: "pending",
	})

	exec.executePlan(context.Background(), plan)

	resultPlan, ok := board.PlanForGoal("goal-1")
	if !ok {
		t.Fatal("plan should exist after execution")
	}
	if resultPlan.Status != "completed" {
		t.Errorf("plan should be completed, got %q", resultPlan.Status)
	}
	if resultPlan.Steps[0].Status != "done" {
		t.Errorf("step should be done, got %q", resultPlan.Steps[0].Status)
	}
	if resultPlan.Steps[0].Result == "" {
		t.Error("step should have a result")
	}
}

func TestExecutorRecordsActions(t *testing.T) {
	board := blackboard.New()
	reg := newTestRegistry()
	exec := NewExecutor(board, nil, reg)

	plan := blackboard.Plan{
		GoalID: "goal-2",
		Steps: []blackboard.Step{
			{ID: "s1", Tool: "read", Args: map[string]string{"path": "a.go"}, Status: "pending"},
			{ID: "s2", Tool: "grep", Args: map[string]string{"pattern": "func"}, Status: "pending"},
		},
		Status: "draft",
	}
	board.PushGoal(blackboard.Goal{ID: "goal-2", Status: "pending"})

	exec.executePlan(context.Background(), plan)

	actions := board.RecentActions(10)
	if len(actions) != 2 {
		t.Fatalf("expected 2 recorded actions, got %d", len(actions))
	}
	if actions[0].Tool != "read" {
		t.Errorf("first action tool should be 'read', got %q", actions[0].Tool)
	}
	if !actions[0].Success {
		t.Error("first action should be successful")
	}
}

func TestExecutorHandlesFailure(t *testing.T) {
	board := blackboard.New()
	reg := tools.NewRegistry()
	reg.Register(tools.Tool{
		Name: "fail",
		Execute: func(args map[string]string) (string, error) {
			return "", fmt.Errorf("tool broke")
		},
	})
	exec := NewExecutor(board, nil, reg)

	plan := blackboard.Plan{
		GoalID: "goal-3",
		Steps: []blackboard.Step{
			{ID: "s1", Tool: "fail", Args: map[string]string{}, Status: "pending"},
			{ID: "s2", Tool: "read", Args: map[string]string{"path": "x"}, Status: "pending"},
		},
		Status: "draft",
	}
	board.PushGoal(blackboard.Goal{ID: "goal-3", Status: "pending"})

	exec.executePlan(context.Background(), plan)

	resultPlan, _ := board.PlanForGoal("goal-3")
	if resultPlan.Status != "failed" {
		t.Errorf("plan should be failed, got %q", resultPlan.Status)
	}
	if resultPlan.Steps[0].Status != "failed" {
		t.Errorf("first step should be failed, got %q", resultPlan.Steps[0].Status)
	}
	if resultPlan.Steps[1].Status != "pending" {
		t.Errorf("second step should still be pending (not executed), got %q", resultPlan.Steps[1].Status)
	}
}

func TestExecutorSkipsNonDraftPlans(t *testing.T) {
	board := blackboard.New()
	reg := newTestRegistry()
	exec := NewExecutor(board, nil, reg)

	ctx, cancel := context.WithCancel(context.Background())
	go exec.Run(ctx)
	time.Sleep(50 * time.Millisecond)

	// Post a plan that's already "completed" — executor should skip it
	board.SetPlan(blackboard.Plan{
		GoalID: "goal-skip",
		Steps:  []blackboard.Step{{ID: "s1", Tool: "read", Status: "done"}},
		Status: "completed",
	})

	time.Sleep(100 * time.Millisecond)
	cancel()

	// No actions should have been recorded
	actions := board.RecentActions(10)
	if len(actions) != 0 {
		t.Errorf("executor should skip non-draft plans, but recorded %d actions", len(actions))
	}
}

func TestExecutorMapsRawArgsToToolParams(t *testing.T) {
	board := blackboard.New()
	reg := tools.NewRegistry()

	var receivedArgs map[string]string
	reg.Register(tools.Tool{
		Name: "read",
		Execute: func(args map[string]string) (string, error) {
			receivedArgs = args
			return "ok", nil
		},
	})
	exec := NewExecutor(board, nil, reg)

	plan := blackboard.Plan{
		GoalID: "goal-map",
		Steps: []blackboard.Step{
			{
				ID:   "s1",
				Tool: "read",
				Args: map[string]string{"raw": "/etc/hostname"},
			},
		},
		Status: "draft",
	}
	board.PushGoal(blackboard.Goal{ID: "goal-map", Status: "pending"})

	exec.executePlan(context.Background(), plan)

	if receivedArgs["path"] != "/etc/hostname" {
		t.Errorf("raw arg should be mapped to 'path' for read tool, got args: %v", receivedArgs)
	}
}
