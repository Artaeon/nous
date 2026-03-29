package agent

import (
	"fmt"
	"strings"
	"time"
)

// Reporter generates progress reports and manages human communication.
type Reporter struct {
	state *AgentState
}

// NewReporter creates a reporter for the given agent state.
func NewReporter(state *AgentState) *Reporter {
	return &Reporter{state: state}
}

// ProgressReport generates a human-readable status update.
func (r *Reporter) ProgressReport() string {
	snap := r.state.Snapshot()
	if snap.Plan == nil {
		if snap.CurrentGoal == "" {
			return "[IDLE] No active goal."
		}
		return fmt.Sprintf("[PLANNING] Decomposing goal: %s", snap.CurrentGoal)
	}

	plan := snap.Plan
	totalPhases := len(plan.Phases)
	currentPhase := snap.Phase
	if currentPhase >= totalPhases {
		currentPhase = totalPhases - 1
	}

	phase := plan.Phases[currentPhase]
	totalTasks := len(phase.Tasks)
	currentTask := snap.Task
	completedTasks := 0
	for _, t := range phase.Tasks {
		if t.Status == TaskCompleted {
			completedTasks++
		}
	}

	var b strings.Builder
	fmt.Fprintf(&b, "[PROGRESS] Phase %d/%d: %s\n", currentPhase+1, totalPhases, phase.Name)
	fmt.Fprintf(&b, "  Completed: %d/%d tasks\n", completedTasks, totalTasks)

	if currentTask < totalTasks {
		fmt.Fprintf(&b, "  Current: %s\n", phase.Tasks[currentTask].Description)
	}

	// Next task
	if currentTask+1 < totalTasks {
		fmt.Fprintf(&b, "  Next: %s\n", phase.Tasks[currentTask+1].Description)
	} else if currentPhase+1 < totalPhases {
		nextPhase := plan.Phases[currentPhase+1]
		if len(nextPhase.Tasks) > 0 {
			fmt.Fprintf(&b, "  Next: %s (Phase %d)\n", nextPhase.Tasks[0].Description, currentPhase+2)
		}
	}

	fmt.Fprintf(&b, "  Tool calls: %d total\n", snap.TotalToolCalls)
	fmt.Fprintf(&b, "  Duration: %s\n", formatDuration(time.Since(snap.StartedAt)))

	// Check if waiting for human
	paused := false
	if currentTask < totalTasks && phase.Tasks[currentTask].Status == TaskNeedsHuman {
		paused = true
	}
	if paused {
		fmt.Fprintf(&b, "  Needs input: Yes\n")
	} else {
		fmt.Fprintf(&b, "  Needs input: No\n")
	}

	return b.String()
}

// TaskReport generates a report for a completed task.
func (r *Reporter) TaskReport(task *Task) string {
	var b strings.Builder
	switch task.Status {
	case TaskCompleted:
		fmt.Fprintf(&b, "[COMPLETED] %s\n", task.Description)
		if task.Result != "" {
			summary := truncateString(task.Result, 200)
			fmt.Fprintf(&b, "  Result: %s\n", summary)
		}
	case TaskFailed:
		fmt.Fprintf(&b, "[FAILED] %s\n", task.Description)
		if task.Error != "" {
			fmt.Fprintf(&b, "  Error: %s\n", task.Error)
		}
		fmt.Fprintf(&b, "  Retries: %d\n", task.Retries)
	case TaskNeedsHuman:
		fmt.Fprintf(&b, "[NEEDS INPUT] %s\n", task.Description)
		if task.HumanPrompt != "" {
			fmt.Fprintf(&b, "  %s\n", task.HumanPrompt)
		}
	default:
		fmt.Fprintf(&b, "[%s] %s\n", task.Status.String(), task.Description)
	}
	return b.String()
}

// NeedsInputReport generates a prompt for the human when input is needed.
func (r *Reporter) NeedsInputReport(task *Task) string {
	var b strings.Builder
	fmt.Fprintf(&b, "[NEEDS INPUT] %s\n", task.Description)
	if task.HumanPrompt != "" {
		fmt.Fprintf(&b, "\n  %s\n", task.HumanPrompt)
	}
	fmt.Fprintf(&b, "\n  Reply with your input, or type 'skip' to continue without input.\n")
	return b.String()
}

// PhaseReport generates a report for a completed phase.
func (r *Reporter) PhaseReport(phase *Phase) string {
	completed := 0
	failed := 0
	for _, t := range phase.Tasks {
		switch t.Status {
		case TaskCompleted:
			completed++
		case TaskFailed:
			failed++
		}
	}
	total := len(phase.Tasks)

	var b strings.Builder
	fmt.Fprintf(&b, "[PHASE COMPLETE] %s\n", phase.Name)
	fmt.Fprintf(&b, "  Tasks: %d completed, %d failed, %d total\n", completed, failed, total)
	return b.String()
}

// FinalReport generates the end-of-goal summary.
func (r *Reporter) FinalReport() string {
	snap := r.state.Snapshot()
	if snap.Plan == nil {
		return "[COMPLETE] No plan was created."
	}

	plan := snap.Plan
	totalTasks := 0
	completedTasks := 0
	failedTasks := 0
	for _, phase := range plan.Phases {
		for _, task := range phase.Tasks {
			totalTasks++
			switch task.Status {
			case TaskCompleted:
				completedTasks++
			case TaskFailed:
				failedTasks++
			}
		}
	}

	var b strings.Builder
	fmt.Fprintf(&b, "[COMPLETE] %s\n", plan.Goal)
	fmt.Fprintf(&b, "  Phases: %d\n", len(plan.Phases))
	fmt.Fprintf(&b, "  Tasks: %d completed, %d failed, %d total\n", completedTasks, failedTasks, totalTasks)
	fmt.Fprintf(&b, "  Tool calls: %d\n", snap.TotalToolCalls)
	fmt.Fprintf(&b, "  Human inputs: %d\n", len(snap.HumanInputs))
	fmt.Fprintf(&b, "  Duration: %s\n", formatDuration(time.Since(snap.StartedAt)))

	// List output files
	var files []string
	for _, result := range snap.Results {
		if strings.Contains(result, "agent_workspace/") {
			files = append(files, result)
		}
	}
	if len(files) > 0 {
		fmt.Fprintf(&b, "  Files produced:\n")
		for _, f := range files {
			fmt.Fprintf(&b, "    - %s\n", f)
		}
	}

	return b.String()
}

// formatDuration formats a duration in a human-friendly way.
func formatDuration(d time.Duration) string {
	if d < time.Minute {
		return fmt.Sprintf("%d seconds", int(d.Seconds()))
	}
	if d < time.Hour {
		return fmt.Sprintf("%d minutes", int(d.Minutes()))
	}
	hours := int(d.Hours())
	mins := int(d.Minutes()) % 60
	if mins == 0 {
		return fmt.Sprintf("%d hours", hours)
	}
	return fmt.Sprintf("%dh %dm", hours, mins)
}

// truncateString shortens s to maxLen characters.
func truncateString(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
}
