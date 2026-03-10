package blackboard

import (
	"sync"
	"time"
)

// Event represents a change on the blackboard that streams can react to.
type Event struct {
	Type      string
	Source    string
	Payload   interface{}
	Timestamp time.Time
}

// Percept represents a parsed unit of input.
type Percept struct {
	Raw       string
	Intent    string
	Entities  map[string]string
	Timestamp time.Time
}

// Goal represents a desired outcome on the goal stack.
type Goal struct {
	ID          string
	Description string
	Priority    int
	Status      string // "pending", "active", "completed", "failed"
	SubGoals    []string
	CreatedAt   time.Time
}

// Plan represents a sequence of steps to achieve a goal.
type Plan struct {
	GoalID string
	Steps  []Step
	Status string // "draft", "executing", "completed", "failed"
}

// Step is a single action within a plan.
type Step struct {
	ID          string
	Description string
	Tool        string
	Args        map[string]string
	Status      string // "pending", "running", "done", "failed"
	Result      string
}

// ActionRecord logs an executed action for episodic memory.
type ActionRecord struct {
	StepID    string
	Tool      string
	Input     string
	Output    string
	Success   bool
	Duration  time.Duration
	Timestamp time.Time
}

// Blackboard is the shared cognitive workspace.
// All cognitive streams read from and write to this structure.
type Blackboard struct {
	mu sync.RWMutex

	// Current perception
	percepts []Percept

	// Working memory — short-lived context for the current interaction
	workingMemory map[string]interface{}

	// Goal stack — what the system is trying to achieve
	goals []Goal

	// Active plans
	plans []Plan

	// Action history — episodic memory of what was done
	actions []ActionRecord

	// Event bus — streams subscribe to blackboard changes
	subscribers map[string][]chan Event
}

// New creates a fresh blackboard.
func New() *Blackboard {
	return &Blackboard{
		workingMemory: make(map[string]interface{}),
		subscribers:   make(map[string][]chan Event),
	}
}

// --- Percepts ---

func (b *Blackboard) PostPercept(p Percept) {
	b.mu.Lock()
	b.percepts = append(b.percepts, p)
	b.mu.Unlock()
	b.emit(Event{Type: "percept", Source: "blackboard", Payload: p, Timestamp: time.Now()})
}

func (b *Blackboard) LatestPercept() (Percept, bool) {
	b.mu.RLock()
	defer b.mu.RUnlock()
	if len(b.percepts) == 0 {
		return Percept{}, false
	}
	return b.percepts[len(b.percepts)-1], true
}

func (b *Blackboard) Percepts() []Percept {
	b.mu.RLock()
	defer b.mu.RUnlock()
	out := make([]Percept, len(b.percepts))
	copy(out, b.percepts)
	return out
}

// --- Working Memory ---

func (b *Blackboard) Set(key string, value interface{}) {
	b.mu.Lock()
	b.workingMemory[key] = value
	b.mu.Unlock()
	b.emit(Event{Type: "memory_set", Source: "blackboard", Payload: key, Timestamp: time.Now()})
}

func (b *Blackboard) Get(key string) (interface{}, bool) {
	b.mu.RLock()
	defer b.mu.RUnlock()
	v, ok := b.workingMemory[key]
	return v, ok
}

func (b *Blackboard) Delete(key string) {
	b.mu.Lock()
	delete(b.workingMemory, key)
	b.mu.Unlock()
}

// --- Goals ---

func (b *Blackboard) PushGoal(g Goal) {
	b.mu.Lock()
	b.goals = append(b.goals, g)
	b.mu.Unlock()
	b.emit(Event{Type: "goal_pushed", Source: "blackboard", Payload: g, Timestamp: time.Now()})
}

func (b *Blackboard) ActiveGoals() []Goal {
	b.mu.RLock()
	defer b.mu.RUnlock()
	var active []Goal
	for _, g := range b.goals {
		if g.Status == "pending" || g.Status == "active" {
			active = append(active, g)
		}
	}
	return active
}

func (b *Blackboard) UpdateGoalStatus(id, status string) {
	b.mu.Lock()
	for i := range b.goals {
		if b.goals[i].ID == id {
			b.goals[i].Status = status
			break
		}
	}
	b.mu.Unlock()
	b.emit(Event{Type: "goal_updated", Source: "blackboard", Payload: id, Timestamp: time.Now()})
}

// --- Plans ---

func (b *Blackboard) SetPlan(p Plan) {
	b.mu.Lock()
	// Replace existing plan for the same goal, or append
	found := false
	for i := range b.plans {
		if b.plans[i].GoalID == p.GoalID {
			b.plans[i] = p
			found = true
			break
		}
	}
	if !found {
		b.plans = append(b.plans, p)
	}
	b.mu.Unlock()
	b.emit(Event{Type: "plan_set", Source: "blackboard", Payload: p, Timestamp: time.Now()})
}

func (b *Blackboard) PlanForGoal(goalID string) (Plan, bool) {
	b.mu.RLock()
	defer b.mu.RUnlock()
	for _, p := range b.plans {
		if p.GoalID == goalID {
			return p, true
		}
	}
	return Plan{}, false
}

// --- Actions ---

func (b *Blackboard) RecordAction(a ActionRecord) {
	b.mu.Lock()
	b.actions = append(b.actions, a)
	b.mu.Unlock()
	b.emit(Event{Type: "action_recorded", Source: "blackboard", Payload: a, Timestamp: time.Now()})
}

func (b *Blackboard) RecentActions(n int) []ActionRecord {
	b.mu.RLock()
	defer b.mu.RUnlock()
	if n > len(b.actions) {
		n = len(b.actions)
	}
	out := make([]ActionRecord, n)
	copy(out, b.actions[len(b.actions)-n:])
	return out
}

// --- Event Bus ---

// Subscribe returns a channel that receives events of the given type.
// Use "*" to receive all events.
func (b *Blackboard) Subscribe(eventType string) chan Event {
	ch := make(chan Event, 64)
	b.mu.Lock()
	b.subscribers[eventType] = append(b.subscribers[eventType], ch)
	b.mu.Unlock()
	return ch
}

func (b *Blackboard) emit(e Event) {
	b.mu.RLock()
	defer b.mu.RUnlock()

	// Send to type-specific subscribers
	for _, ch := range b.subscribers[e.Type] {
		select {
		case ch <- e:
		default: // non-blocking, drop if full
		}
	}

	// Send to wildcard subscribers
	for _, ch := range b.subscribers["*"] {
		select {
		case ch <- e:
		default:
		}
	}
}
