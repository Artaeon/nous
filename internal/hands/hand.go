package hands

import (
	"time"
)

// HandState represents the current state of a hand.
type HandState string

const (
	HandIdle      HandState = "idle"
	HandRunning   HandState = "running"
	HandPaused    HandState = "paused"
	HandFailed    HandState = "failed"
	HandCompleted HandState = "completed"
)

// HandConfig controls execution behavior of a hand.
type HandConfig struct {
	MaxSteps         int      `json:"max_steps"`          // max reasoning iterations (default 8)
	Timeout          int      `json:"timeout_seconds"`    // max wall-clock seconds per run (default 120)
	Model            string   `json:"model,omitempty"`    // override default model
	Tools            []string `json:"tools"`              // whitelist of allowed tool names
	RequiresApproval bool     `json:"requires_approval"`  // queue results for user review
}

// DefaultConfig returns a HandConfig with sensible defaults.
func DefaultConfig() HandConfig {
	return HandConfig{
		MaxSteps: 8,
		Timeout:  120,
	}
}

// HandResult is the outcome of a single hand run.
type HandResult struct {
	Output    string        `json:"output"`
	Artifacts []string      `json:"artifacts,omitempty"` // file paths produced
	Error     string        `json:"error,omitempty"`
	Duration  time.Duration `json:"duration_ns"`
	ToolCalls int           `json:"tool_calls"`
}

// Hand defines an autonomous agent that runs on a schedule.
type Hand struct {
	Name        string     `json:"name"`
	Description string     `json:"description"`
	Schedule    string     `json:"schedule"`     // cron expression or shortcut
	Enabled     bool       `json:"enabled"`
	Config      HandConfig `json:"config"`
	Prompt      string     `json:"prompt"`       // goal/instruction for the cognitive pipeline

	// Runtime state (not persisted in config, tracked by manager)
	State     HandState  `json:"state"`
	LastRun   time.Time  `json:"last_run,omitempty"`
	LastError string     `json:"last_error,omitempty"`
}

// RunRecord is a historical entry for a single hand execution.
type RunRecord struct {
	HandName  string     `json:"hand_name"`
	StartedAt time.Time  `json:"started_at"`
	Duration  int64      `json:"duration_ms"`
	Success   bool       `json:"success"`
	Output    string     `json:"output"`
	Error     string     `json:"error,omitempty"`
	ToolCalls int        `json:"tool_calls"`
}
