package agent

import (
	"encoding/json"
	"os"
	"path/filepath"
	"sync"
	"time"
)

// TaskStatus represents the lifecycle state of a task.
type TaskStatus int

const (
	TaskPending    TaskStatus = iota
	TaskRunning
	TaskCompleted
	TaskFailed
	TaskBlocked    // waiting for dependency
	TaskNeedsHuman // paused for human input
)

// String returns a human-readable label for the status.
func (s TaskStatus) String() string {
	switch s {
	case TaskPending:
		return "pending"
	case TaskRunning:
		return "running"
	case TaskCompleted:
		return "completed"
	case TaskFailed:
		return "failed"
	case TaskBlocked:
		return "blocked"
	case TaskNeedsHuman:
		return "needs_human"
	default:
		return "unknown"
	}
}

// MarshalJSON encodes TaskStatus as a string.
func (s TaskStatus) MarshalJSON() ([]byte, error) {
	return json.Marshal(s.String())
}

// UnmarshalJSON decodes TaskStatus from a string.
func (s *TaskStatus) UnmarshalJSON(b []byte) error {
	var str string
	if err := json.Unmarshal(b, &str); err != nil {
		return err
	}
	switch str {
	case "pending":
		*s = TaskPending
	case "running":
		*s = TaskRunning
	case "completed":
		*s = TaskCompleted
	case "failed":
		*s = TaskFailed
	case "blocked":
		*s = TaskBlocked
	case "needs_human":
		*s = TaskNeedsHuman
	default:
		*s = TaskPending
	}
	return nil
}

// PhaseStatus represents the lifecycle state of a phase.
type PhaseStatus int

const (
	PhasePending   PhaseStatus = iota
	PhaseRunning
	PhaseCompleted
	PhaseFailed
)

// String returns a human-readable label.
func (s PhaseStatus) String() string {
	switch s {
	case PhasePending:
		return "pending"
	case PhaseRunning:
		return "running"
	case PhaseCompleted:
		return "completed"
	case PhaseFailed:
		return "failed"
	default:
		return "unknown"
	}
}

// MarshalJSON encodes PhaseStatus as a string.
func (s PhaseStatus) MarshalJSON() ([]byte, error) {
	return json.Marshal(s.String())
}

// UnmarshalJSON decodes PhaseStatus from a string.
func (s *PhaseStatus) UnmarshalJSON(b []byte) error {
	var str string
	if err := json.Unmarshal(b, &str); err != nil {
		return err
	}
	switch str {
	case "pending":
		*s = PhasePending
	case "running":
		*s = PhaseRunning
	case "completed":
		*s = PhaseCompleted
	case "failed":
		*s = PhaseFailed
	default:
		*s = PhasePending
	}
	return nil
}

// ToolStep is one step in a tool chain.
type ToolStep struct {
	Tool      string            `json:"tool"`
	Args      map[string]string `json:"args"`
	DependsOn int               `json:"depends_on"` // index of previous step whose output feeds in (-1 for none)
	OutputKey string            `json:"output_key"` // key to store result for later steps
}

// Task is one atomic unit of work.
type Task struct {
	ID          string     `json:"id"`
	Description string     `json:"description"`
	ToolChain   []ToolStep `json:"tool_chain"`
	Status      TaskStatus `json:"status"`
	Result      string     `json:"result"`
	Error       string     `json:"error,omitempty"`
	NeedsHuman  bool       `json:"needs_human"`
	HumanPrompt string     `json:"human_prompt,omitempty"`
	Retries     int        `json:"retries"`
	StartedAt   time.Time  `json:"started_at,omitempty"`
	CompletedAt time.Time  `json:"completed_at,omitempty"`
}

// Phase is a group of related tasks.
type Phase struct {
	Name        string      `json:"name"`
	Description string      `json:"description"`
	Tasks       []Task      `json:"tasks"`
	DependsOn   []int       `json:"depends_on,omitempty"` // phase indices that must complete first
	Status      PhaseStatus `json:"status"`
	Retried     int         `json:"retried,omitempty"` // how many times this phase was retried (max 1)
}

// Plan represents a decomposed goal with phases, tasks, and dependencies.
type Plan struct {
	Goal           string        `json:"goal"`
	Phases         []Phase       `json:"phases"`
	CreatedAt      time.Time     `json:"created_at"`
	EstDuration    time.Duration `json:"est_duration_ns"`
	ExperienceNote string        `json:"experience_note,omitempty"` // insight from past similar goals
}

// HumanInput records one human-in-the-loop interaction.
type HumanInput struct {
	Prompt   string    `json:"prompt"`
	Response string    `json:"response"`
	TaskID   string    `json:"task_id"`
	Time     time.Time `json:"time"`
}

// AgentState persists the agent's current state to disk.
type AgentState struct {
	mu sync.RWMutex

	CurrentGoal    string            `json:"current_goal"`
	Plan           *Plan             `json:"plan,omitempty"`
	Phase          int               `json:"phase"`           // current phase index
	Task           int               `json:"task"`            // current task index within phase
	Results        map[string]string `json:"results"`         // task ID -> result
	HumanInputs   []HumanInput      `json:"human_inputs"`
	StartedAt      time.Time         `json:"started_at"`
	LastActivity   time.Time         `json:"last_activity"`
	TotalToolCalls int               `json:"total_tool_calls"`
	Finished       bool              `json:"finished"`

	path string // file path for persistence
}

// NewAgentState creates a new state manager backed by the given file path.
func NewAgentState(path string) *AgentState {
	return &AgentState{
		Results: make(map[string]string),
		path:    path,
	}
}

// Save persists the state to disk as JSON.
func (s *AgentState) Save() error {
	s.mu.RLock()
	defer s.mu.RUnlock()

	data, err := json.MarshalIndent(s, "", "  ")
	if err != nil {
		return err
	}

	dir := filepath.Dir(s.path)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}

	// Write to temp file then rename for atomicity.
	tmp := s.path + ".tmp"
	if err := os.WriteFile(tmp, data, 0o644); err != nil {
		return err
	}
	return os.Rename(tmp, s.path)
}

// Load reads the state from disk.
func (s *AgentState) Load() error {
	s.mu.Lock()
	defer s.mu.Unlock()

	data, err := os.ReadFile(s.path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil // no state yet
		}
		return err
	}

	return json.Unmarshal(data, s)
}

// Reset clears the state for a new goal.
func (s *AgentState) Reset(goal string) {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.CurrentGoal = goal
	s.Plan = nil
	s.Phase = 0
	s.Task = 0
	s.Results = make(map[string]string)
	s.HumanInputs = nil
	s.StartedAt = time.Now()
	s.LastActivity = time.Now()
	s.TotalToolCalls = 0
	s.Finished = false
}

// RecordResult stores a task result and updates the activity timestamp.
func (s *AgentState) RecordResult(taskID, result string) {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.Results[taskID] = result
	s.LastActivity = time.Now()
}

// RecordHumanInput stores a human response.
func (s *AgentState) RecordHumanInput(taskID, prompt, response string) {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.HumanInputs = append(s.HumanInputs, HumanInput{
		Prompt:   prompt,
		Response: response,
		TaskID:   taskID,
		Time:     time.Now(),
	})
	s.LastActivity = time.Now()
}

// AddToolCalls increments the total tool call counter.
func (s *AgentState) AddToolCalls(n int) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.TotalToolCalls += n
	s.LastActivity = time.Now()
}

// SetPlan stores the decomposed plan.
func (s *AgentState) SetPlan(plan *Plan) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.Plan = plan
	s.LastActivity = time.Now()
}

// Advance moves the pointer to the next task, advancing phases as needed.
// Returns false when all phases are complete.
func (s *AgentState) Advance() bool {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.Plan == nil {
		return false
	}

	s.Task++
	if s.Phase < len(s.Plan.Phases) && s.Task >= len(s.Plan.Phases[s.Phase].Tasks) {
		s.Task = 0
		s.Phase++
	}

	if s.Phase >= len(s.Plan.Phases) {
		s.Finished = true
		return false
	}
	return true
}

// CurrentTask returns the current task, or nil if done.
func (s *AgentState) CurrentTask() *Task {
	s.mu.RLock()
	defer s.mu.RUnlock()

	if s.Plan == nil || s.Phase >= len(s.Plan.Phases) {
		return nil
	}
	phase := &s.Plan.Phases[s.Phase]
	if s.Task >= len(phase.Tasks) {
		return nil
	}
	return &phase.Tasks[s.Task]
}

// CurrentPhase returns the current phase, or nil if done.
func (s *AgentState) CurrentPhase() *Phase {
	s.mu.RLock()
	defer s.mu.RUnlock()

	if s.Plan == nil || s.Phase >= len(s.Plan.Phases) {
		return nil
	}
	return &s.Plan.Phases[s.Phase]
}

// Snapshot returns a deep copy of the state safe for reading without locks.
func (s *AgentState) Snapshot() AgentState {
	s.mu.RLock()
	defer s.mu.RUnlock()

	cp := *s
	cp.Results = make(map[string]string, len(s.Results))
	for k, v := range s.Results {
		cp.Results[k] = v
	}

	// Deep-copy the plan so the snapshot doesn't share mutable task/phase slices.
	if s.Plan != nil {
		planCopy := *s.Plan
		planCopy.Phases = make([]Phase, len(s.Plan.Phases))
		for i, ph := range s.Plan.Phases {
			phCopy := ph
			phCopy.Tasks = make([]Task, len(ph.Tasks))
			copy(phCopy.Tasks, ph.Tasks)
			if ph.DependsOn != nil {
				phCopy.DependsOn = make([]int, len(ph.DependsOn))
				copy(phCopy.DependsOn, ph.DependsOn)
			}
			planCopy.Phases[i] = phCopy
		}
		cp.Plan = &planCopy
	}

	// Deep-copy HumanInputs
	if s.HumanInputs != nil {
		cp.HumanInputs = make([]HumanInput, len(s.HumanInputs))
		copy(cp.HumanInputs, s.HumanInputs)
	}

	return cp
}
