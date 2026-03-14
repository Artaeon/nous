package cognitive

import (
	"crypto/rand"
	"encoding/hex"
	"sync"
	"time"
)

// TraceStepType identifies the kind of reasoning step.
type TraceStepType string

const (
	TraceThink   TraceStepType = "think"
	TraceAct     TraceStepType = "act"
	TraceObserve TraceStepType = "observe"
	TraceReflect TraceStepType = "reflect"
)

// TraceStep records a single step in the reasoning process.
type TraceStep struct {
	Type      TraceStepType `json:"type"`
	Content   string        `json:"content"`
	Timestamp time.Time     `json:"timestamp"`
	ToolName  string        `json:"tool_name,omitempty"` // populated when Type == TraceAct
	Duration  time.Duration `json:"duration_ns"`
}

// ReasoningTrace captures the full reasoning process for a single query.
type ReasoningTrace struct {
	ID          string      `json:"id"`
	StartTime   time.Time   `json:"start_time"`
	EndTime     time.Time   `json:"end_time,omitempty"`
	Steps       []TraceStep `json:"steps"`
	Query       string      `json:"query"`
	FinalAnswer string      `json:"final_answer,omitempty"`

	mu        sync.Mutex
	stepStart time.Time
}

// NewTrace creates a new reasoning trace for the given query.
func NewTrace(query string) *ReasoningTrace {
	return &ReasoningTrace{
		ID:        generateTraceID(),
		StartTime: time.Now(),
		Query:     query,
		Steps:     make([]TraceStep, 0),
	}
}

// AddStep records a new step in the trace. Duration is measured from the
// last AddStep call (or trace creation if this is the first step).
func (t *ReasoningTrace) AddStep(stepType TraceStepType, content string) {
	t.AddStepWithTool(stepType, content, "")
}

// AddStepWithTool records a step that involves a tool invocation.
func (t *ReasoningTrace) AddStepWithTool(stepType TraceStepType, content, toolName string) {
	t.mu.Lock()
	defer t.mu.Unlock()

	now := time.Now()
	dur := time.Duration(0)
	if !t.stepStart.IsZero() {
		dur = now.Sub(t.stepStart)
	} else {
		dur = now.Sub(t.StartTime)
	}
	t.stepStart = now

	t.Steps = append(t.Steps, TraceStep{
		Type:      stepType,
		Content:   content,
		Timestamp: now,
		ToolName:  toolName,
		Duration:  dur,
	})
}

// Complete marks the trace as finished with the given answer.
func (t *ReasoningTrace) Complete(answer string) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.EndTime = time.Now()
	t.FinalAnswer = answer
}

// StepCount returns the number of steps in the trace.
func (t *ReasoningTrace) StepCount() int {
	t.mu.Lock()
	defer t.mu.Unlock()
	return len(t.Steps)
}

// TraceStore keeps the most recent reasoning traces in memory.
// It is safe for concurrent use.
type TraceStore struct {
	mu     sync.RWMutex
	traces []*ReasoningTrace
	byID   map[string]*ReasoningTrace
	cap    int
}

// NewTraceStore creates a store that retains the last maxTraces traces.
func NewTraceStore(maxTraces int) *TraceStore {
	if maxTraces <= 0 {
		maxTraces = 50
	}
	return &TraceStore{
		traces: make([]*ReasoningTrace, 0, maxTraces),
		byID:   make(map[string]*ReasoningTrace, maxTraces),
		cap:    maxTraces,
	}
}

// Save adds a completed trace to the store. If the store is full, the
// oldest trace is evicted.
func (s *TraceStore) Save(trace *ReasoningTrace) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if len(s.traces) >= s.cap {
		evicted := s.traces[0]
		delete(s.byID, evicted.ID)
		s.traces = s.traces[1:]
	}
	s.traces = append(s.traces, trace)
	s.byID[trace.ID] = trace
}

// Get returns a trace by ID, or nil if not found.
func (s *TraceStore) Get(id string) *ReasoningTrace {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.byID[id]
}

// Recent returns up to n of the most recent traces, newest first.
func (s *TraceStore) Recent(n int) []*ReasoningTrace {
	s.mu.RLock()
	defer s.mu.RUnlock()

	total := len(s.traces)
	if n > total {
		n = total
	}

	result := make([]*ReasoningTrace, n)
	for i := 0; i < n; i++ {
		result[i] = s.traces[total-1-i]
	}
	return result
}

// Len returns the number of stored traces.
func (s *TraceStore) Len() int {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return len(s.traces)
}

func generateTraceID() string {
	b := make([]byte, 8)
	_, _ = rand.Read(b)
	return hex.EncodeToString(b)
}
