package agent

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/artaeon/nous/internal/tools"
)

// AgentConfig controls the autonomous agent's behavior.
type AgentConfig struct {
	Workspace          string        // where to save agent files (default ~/.nous/agent/)
	MaxToolCalls       int           // safety limit per task (default 20)
	MaxRetries         int           // retries per task (default 3)
	PauseOnUncertainty bool          // pause when confidence is low
	ReportInterval     time.Duration // how often to report progress
	StepTimeout        time.Duration // timeout per tool step (default 30s)
}

// DefaultConfig returns sensible defaults.
func DefaultConfig() AgentConfig {
	home, _ := os.UserHomeDir()
	return AgentConfig{
		Workspace:          filepath.Join(home, ".nous", "agent"),
		MaxToolCalls:       20,
		MaxRetries:         3,
		PauseOnUncertainty: true,
		ReportInterval:     30 * time.Second,
		StepTimeout:        90 * time.Second,
	}
}

// AgentStatus is a snapshot of the agent's current state.
type AgentStatus struct {
	Running    bool   `json:"running"`
	Goal       string `json:"goal"`
	Phase      string `json:"phase"`
	Task       string `json:"task"`
	Progress   string `json:"progress"` // human-readable progress report
	PausedFor  string `json:"paused_for,omitempty"`
	ToolCalls  int    `json:"tool_calls"`
	StartedAt  string `json:"started_at,omitempty"`
	Duration   string `json:"duration,omitempty"`
}

// Agent is an autonomous executor that takes goals and works toward them
// independently, using Nous's tools and knowledge.
type Agent struct {
	Tools      *tools.Registry
	Planner    *Planner
	Executor   *Executor
	State      *AgentState
	Reporter   *Reporter
	Config     AgentConfig
	Brain      *CognitiveBridge // cognitive systems for thinking, not just tool running
	Experience *ExperienceMemory // learns from past executions

	mu        sync.Mutex
	running   bool
	pausedFor string // reason agent is paused (waiting for human input)
	stopCh    chan struct{}
	resumeCh  chan string // human input arrives here
	reportFn  func(string) // callback for progress reports
}

// NewAgent creates an autonomous agent.
func NewAgent(toolReg *tools.Registry, config AgentConfig) *Agent {
	defaults := DefaultConfig()
	if config.Workspace == "" {
		config.Workspace = defaults.Workspace
	}
	if config.MaxToolCalls <= 0 {
		config.MaxToolCalls = defaults.MaxToolCalls
	}
	if config.MaxRetries <= 0 {
		config.MaxRetries = defaults.MaxRetries
	}
	if config.StepTimeout <= 0 {
		config.StepTimeout = defaults.StepTimeout
	}
	if config.ReportInterval <= 0 {
		config.ReportInterval = defaults.ReportInterval
	}

	// Ensure workspace directory exists
	os.MkdirAll(config.Workspace, 0o755)

	statePath := filepath.Join(config.Workspace, "state.json")
	state := NewAgentState(statePath)

	// Collect tool names for the planner
	var toolNames []string
	for _, t := range toolReg.List() {
		toolNames = append(toolNames, t.Name)
	}

	executor := NewExecutor(toolReg, config.Workspace)
	executor.MaxRetries = config.MaxRetries
	executor.MaxToolCalls = config.MaxToolCalls
	executor.StepTimeout = config.StepTimeout

	exp := NewExperienceMemory(config.Workspace)
	planner := NewPlanner(toolNames)
	planner.Experience = exp

	return &Agent{
		Tools:      toolReg,
		Planner:    planner,
		Executor:   executor,
		State:      state,
		Reporter:   NewReporter(state),
		Config:     config,
		Experience: exp,
		resumeCh:   make(chan string, 1),
	}
}

// SetBrain connects the agent to Nous's cognitive systems.
// This transforms the agent from a checklist runner into a thinking worker.
func (a *Agent) SetBrain(brain *CognitiveBridge) {
	a.Brain = brain
	a.Executor.Brain = brain
}

// SetReportCallback sets a function that gets called with progress reports.
func (a *Agent) SetReportCallback(fn func(string)) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.reportFn = fn
}

// Start begins autonomous execution of a goal.
func (a *Agent) Start(goal string) error {
	a.mu.Lock()
	if a.running {
		a.mu.Unlock()
		return fmt.Errorf("agent is already running (goal: %s)", a.State.CurrentGoal)
	}

	a.running = true
	a.pausedFor = ""
	a.stopCh = make(chan struct{})
	a.mu.Unlock()

	a.State.Reset(goal)
	a.report(fmt.Sprintf("[STARTING] Goal: %s", goal))

	go a.run()
	return nil
}

// Stop halts execution. The agent saves state and can be resumed.
func (a *Agent) Stop() {
	a.mu.Lock()
	if !a.running {
		a.mu.Unlock()
		return
	}

	close(a.stopCh)
	a.running = false
	a.mu.Unlock()

	// Report and save outside the lock to avoid deadlock (report() also locks a.mu).
	a.report("[STOPPED] Agent stopped by user. State saved — can be resumed.")
	a.State.Save()
}

// Resume continues after the agent paused for human input.
func (a *Agent) Resume(humanInput string) {
	a.mu.Lock()
	paused := a.pausedFor
	a.mu.Unlock()

	if paused == "" {
		return
	}

	// Record the human input
	task := a.State.CurrentTask()
	taskID := ""
	if task != nil {
		taskID = task.ID
	}
	a.State.RecordHumanInput(taskID, paused, humanInput)

	a.mu.Lock()
	a.pausedFor = ""
	a.mu.Unlock()

	// Send input to the run loop
	select {
	case a.resumeCh <- humanInput:
	default:
	}
}

// Status returns the current agent status.
func (a *Agent) Status() *AgentStatus {
	a.mu.Lock()
	running := a.running
	paused := a.pausedFor
	a.mu.Unlock()

	snap := a.State.Snapshot()

	status := &AgentStatus{
		Running:   running,
		Goal:      snap.CurrentGoal,
		PausedFor: paused,
		ToolCalls: snap.TotalToolCalls,
	}

	if !snap.StartedAt.IsZero() {
		status.StartedAt = snap.StartedAt.Format(time.RFC3339)
		status.Duration = formatDuration(time.Since(snap.StartedAt))
	}

	if snap.Plan != nil && snap.Phase < len(snap.Plan.Phases) {
		phase := snap.Plan.Phases[snap.Phase]
		status.Phase = phase.Name
		if snap.Task < len(phase.Tasks) {
			status.Task = phase.Tasks[snap.Task].Description
		}
	}

	status.Progress = a.Reporter.ProgressReport()
	return status
}

// Report returns the final report if the agent has finished, or progress otherwise.
func (a *Agent) Report() string {
	snap := a.State.Snapshot()
	if snap.Finished {
		return a.Reporter.FinalReport()
	}
	return a.Reporter.ProgressReport()
}

// run is the main execution loop (runs in a goroutine).
func (a *Agent) run() {
	defer func() {
		a.mu.Lock()
		a.running = false
		a.mu.Unlock()
		a.State.Save()
	}()

	// Step 1: If no active plan, decompose goal
	if a.State.Plan == nil {
		a.report("[PLANNING] Decomposing goal...")
		plan, err := a.Planner.DecomposeGoal(a.State.CurrentGoal)
		if err != nil {
			a.report(fmt.Sprintf("[ERROR] Failed to decompose goal: %v", err))
			return
		}

		a.State.SetPlan(plan)
		a.State.Save()

		// Report the plan
		var planReport strings.Builder
		fmt.Fprintf(&planReport, "[PLANNING] Decomposed into %d phases:\n", len(plan.Phases))
		for i, phase := range plan.Phases {
			fmt.Fprintf(&planReport, "  Phase %d: %s (%d tasks)\n", i+1, phase.Name, len(phase.Tasks))
		}
		a.report(planReport.String())
	}

	// Step 2: Execute phases and tasks
	for !a.stopped() {
		phase := a.State.CurrentPhase()
		if phase == nil {
			break // all phases complete
		}

		task := a.State.CurrentTask()
		if task == nil {
			break
		}

		// Read current phase index under lock
		a.State.mu.RLock()
		phaseIdx := a.State.Phase
		a.State.mu.RUnlock()

		// Check phase dependencies
		if !a.phaseDepsComplete(phaseIdx) {
			a.report(fmt.Sprintf("[BLOCKED] Phase %q waiting for dependencies", phase.Name))
			break
		}

		// Mark phase as running
		a.State.mu.Lock()
		a.State.Plan.Phases[a.State.Phase].Status = PhaseRunning
		a.State.mu.Unlock()

		// Execute the task
		a.executeTask(task)

		// Check if we need to stop
		if a.stopped() {
			break
		}

		// If task needs human input, pause
		a.State.mu.RLock()
		needsHuman := task.Status == TaskNeedsHuman
		a.State.mu.RUnlock()

		if needsHuman {
			a.pause(task)
			if a.stopped() {
				break
			}
			continue // re-evaluate the same task after human input
		}

		// Advance to next task — snapshot the task for reporting
		a.State.mu.RLock()
		taskReport := a.Reporter.TaskReport(task)
		a.State.mu.RUnlock()
		a.report(taskReport)

		// Check if phase is complete
		a.State.mu.Lock()
		phaseComplete := true
		curPhase := &a.State.Plan.Phases[a.State.Phase]
		for _, t := range curPhase.Tasks {
			if t.Status != TaskCompleted && t.Status != TaskFailed {
				phaseComplete = false
				break
			}
		}
		if phaseComplete {
			curPhase.Status = PhaseCompleted
		}
		a.State.mu.Unlock()

		if phaseComplete {
			a.report(a.Reporter.PhaseReport(phase))

			// Adaptive replanning: evaluate the completed phase
			a.State.mu.RLock()
			completedPhaseIdx := a.State.Phase
			a.State.mu.RUnlock()

			eval := a.evaluatePhase(completedPhaseIdx)
			a.applyEvaluation(completedPhaseIdx, eval)

			// If the phase was retried, don't advance — re-execute it
			a.State.mu.RLock()
			wasRetried := a.State.Plan.Phases[completedPhaseIdx].Status == PhasePending
			a.State.mu.RUnlock()
			if wasRetried {
				continue // re-enter the loop to execute the retried phase
			}
		}

		if !a.State.Advance() {
			break // all done
		}

		a.State.Save()
	}

	// Step 3: Generate final report
	a.State.mu.Lock()
	a.State.Finished = true
	a.State.mu.Unlock()
	a.State.Save()

	a.report(a.Reporter.FinalReport())
}

// executeTask runs a single task's tool chain.
func (a *Agent) executeTask(task *Task) {
	a.State.mu.Lock()
	task.Status = TaskRunning
	task.StartedAt = time.Now()
	a.State.mu.Unlock()

	// If task needs human input, mark it and return
	if task.NeedsHuman && task.HumanPrompt != "" {
		a.State.mu.Lock()
		task.Status = TaskNeedsHuman
		a.State.mu.Unlock()
		return
	}

	if len(task.ToolChain) == 0 {
		// No tools to run — mark complete
		a.State.mu.Lock()
		task.Status = TaskCompleted
		task.CompletedAt = time.Now()
		a.State.mu.Unlock()
		return
	}

	// Check safety: dangerous tools outside workspace need approval
	for _, step := range task.ToolChain {
		if IsDangerousTool(step.Tool) && !a.isInWorkspace(step) {
			a.State.mu.Lock()
			task.NeedsHuman = true
			task.HumanPrompt = fmt.Sprintf(
				"Task wants to use %q outside the workspace. Allow? (yes/no)",
				step.Tool,
			)
			task.Status = TaskNeedsHuman
			a.State.mu.Unlock()
			return
		}
	}

	// Build context from previous results
	context := make(map[string]string)
	a.State.mu.RLock()
	for k, v := range a.State.Results {
		context[k] = v
	}
	a.State.mu.RUnlock()

	// Execute the tool chain
	result, err := a.Executor.ExecuteChain(task.ToolChain, context)

	a.State.mu.Lock()
	if err != nil {
		task.Status = TaskFailed
		task.Error = err.Error()
		task.Retries++
	} else {
		task.Status = TaskCompleted
		task.Result = result.FinalOutput
		task.CompletedAt = time.Now()
	}
	a.State.mu.Unlock()

	if result != nil {
		a.State.AddToolCalls(result.ToolCalls)
		a.State.RecordResult(task.ID, result.FinalOutput)

		// Record experience for learning.
		if a.Experience != nil {
			var toolNames []string
			for _, step := range task.ToolChain {
				toolNames = append(toolNames, step.Tool)
			}
			a.Experience.Record(ExperienceEntry{
				GoalType:    goalTypeString(classifyGoal(a.State.CurrentGoal)),
				ToolChain:   toolNames,
				Succeeded:   task.Status == TaskCompleted,
				OutputWords: len(strings.Fields(result.FinalOutput)),
				Duration:    float64(result.Duration.Milliseconds()),
				Goal:        a.State.CurrentGoal,
			})
		}
	}
}

// pause waits for human input on a task.
func (a *Agent) pause(task *Task) {
	a.mu.Lock()
	a.pausedFor = task.HumanPrompt
	a.mu.Unlock()

	a.report(a.Reporter.NeedsInputReport(task))

	// Wait for human input or stop
	select {
	case input := <-a.resumeCh:
		// Human provided input — mark task complete with the input as result
		a.State.mu.Lock()
		task.Status = TaskCompleted
		task.Result = input
		task.CompletedAt = time.Now()
		a.State.mu.Unlock()

		a.State.RecordResult(task.ID, input)
		a.report(fmt.Sprintf("[RESUMING] Got input: %s", truncateString(input, 100)))

	case <-a.stopCh:
		return
	}
}

// applyEvaluation acts on the results of a phase evaluation:
// retry with new queries, inject tasks into the next phase, or skip phases.
func (a *Agent) applyEvaluation(phaseIdx int, eval *PhaseEvaluation) {
	if eval == nil || eval.QualitySufficient {
		if eval != nil && eval.SkipNextPhase {
			a.State.mu.Lock()
			nextIdx := phaseIdx + 1
			if nextIdx < len(a.State.Plan.Phases) {
				a.State.Plan.Phases[nextIdx].Status = PhaseCompleted
				a.report(fmt.Sprintf("[ADAPTIVE] Skipping phase %q — %s",
					a.State.Plan.Phases[nextIdx].Name, eval.SuggestedAdjustments[0]))
			}
			a.State.mu.Unlock()
		}
		return
	}

	a.report(fmt.Sprintf("[ADAPTIVE] Phase evaluation: %s", eval.Reason))

	// Quality insufficient — check if we can retry
	a.State.mu.Lock()
	phase := &a.State.Plan.Phases[phaseIdx]
	canRetry := phase.Retried < 1
	a.State.mu.Unlock()

	if len(eval.NeedsMoreData) > 0 {
		if canRetry {
			// Retry: reset the phase, inject new search tasks, and re-execute
			a.State.mu.Lock()
			phase.Retried++
			phase.Status = PhasePending
			// Reset all task statuses so they can re-run
			for i := range phase.Tasks {
				phase.Tasks[i].Status = TaskPending
				phase.Tasks[i].Error = ""
				phase.Tasks[i].Result = ""
			}
			// Inject new search tasks at the front of the phase
			injectSearchTasks(phase, eval.NeedsMoreData, fmt.Sprintf("phase%d", phaseIdx))
			// Reset the task pointer to the start of this phase
			a.State.Task = 0
			a.State.mu.Unlock()

			a.report(fmt.Sprintf("[ADAPTIVE] Retrying phase %q with %d new search queries: %s",
				phase.Name, len(eval.NeedsMoreData), strings.Join(eval.NeedsMoreData, ", ")))
		} else {
			// Already retried once — inject the queries into the NEXT phase instead
			a.State.mu.Lock()
			nextIdx := phaseIdx + 1
			if nextIdx < len(a.State.Plan.Phases) {
				injectSearchTasks(&a.State.Plan.Phases[nextIdx], eval.NeedsMoreData,
					fmt.Sprintf("phase%d-sup", nextIdx))
				a.report(fmt.Sprintf("[ADAPTIVE] Injecting %d extra searches into phase %q",
					len(eval.NeedsMoreData), a.State.Plan.Phases[nextIdx].Name))
			}
			a.State.mu.Unlock()
		}
	}
}

// phaseDepsComplete returns true if all prerequisite phases are done.
func (a *Agent) phaseDepsComplete(phaseIdx int) bool {
	a.State.mu.RLock()
	defer a.State.mu.RUnlock()

	if a.State.Plan == nil || phaseIdx >= len(a.State.Plan.Phases) {
		return false
	}

	phase := a.State.Plan.Phases[phaseIdx]
	for _, dep := range phase.DependsOn {
		if dep >= len(a.State.Plan.Phases) {
			return false
		}
		if a.State.Plan.Phases[dep].Status != PhaseCompleted {
			return false
		}
	}
	return true
}

// isInWorkspace checks if a tool step's file path is within the workspace.
func (a *Agent) isInWorkspace(step ToolStep) bool {
	for _, key := range []string{"path", "file", "dir", "directory"} {
		if v, ok := step.Args[key]; ok {
			if strings.HasPrefix(v, a.Config.Workspace) ||
				strings.HasPrefix(v, "agent_workspace/") ||
				(!strings.HasPrefix(v, "/") && !strings.Contains(v, "..")) {
				return true
			}
			return false
		}
	}
	// No path arg — assume safe
	return true
}

// stopped checks if a stop has been requested.
func (a *Agent) stopped() bool {
	select {
	case <-a.stopCh:
		return true
	default:
		return false
	}
}

// report sends a progress report via the callback.
func (a *Agent) report(msg string) {
	a.mu.Lock()
	fn := a.reportFn
	a.mu.Unlock()

	if fn != nil {
		fn(msg)
	}
}

// LoadAndResume attempts to load saved state and resume execution.
func (a *Agent) LoadAndResume() error {
	if err := a.State.Load(); err != nil {
		return err
	}

	if a.State.CurrentGoal == "" || a.State.Finished {
		return fmt.Errorf("no resumable goal found")
	}

	a.mu.Lock()
	if a.running {
		a.mu.Unlock()
		return fmt.Errorf("agent is already running")
	}
	a.running = true
	a.stopCh = make(chan struct{})
	a.mu.Unlock()

	a.report(fmt.Sprintf("[RESUMING] Goal: %s (phase %d, task %d)",
		a.State.CurrentGoal, a.State.Phase+1, a.State.Task+1))

	go a.run()
	return nil
}
