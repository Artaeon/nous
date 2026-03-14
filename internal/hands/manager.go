package hands

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/artaeon/nous/internal/blackboard"
)

const defaultMaxConcurrent = 2

// Manager owns all registered hands and runs the background scheduler.
type Manager struct {
	mu            sync.RWMutex
	wg            sync.WaitGroup
	store         *Store
	runner        *Runner
	board         *blackboard.Blackboard
	maxConcurrent int
	running       int
	hands         map[string]*Hand
	schedules     map[string]Schedule
	nextRuns      map[string]time.Time
	Approvals     *ApprovalQueue
}

// NewManager creates a hand lifecycle manager.
func NewManager(store *Store, runner *Runner, board *blackboard.Blackboard) *Manager {
	m := &Manager{
		store:         store,
		runner:        runner,
		board:         board,
		maxConcurrent: defaultMaxConcurrent,
		hands:         make(map[string]*Hand),
		schedules:     make(map[string]Schedule),
		nextRuns:      make(map[string]time.Time),
		Approvals:     NewApprovalQueue(),
	}

	// Load persisted hands
	for _, h := range store.AllHands() {
		m.hands[h.Name] = h
		if h.Enabled && h.Schedule != "" {
			if sched, err := ParseSchedule(h.Schedule); err == nil {
				m.schedules[h.Name] = sched
				m.nextRuns[h.Name] = NextRun(sched, time.Now())
			}
		}
	}

	return m
}

// SetMaxConcurrent configures the maximum number of hands running simultaneously.
func (m *Manager) SetMaxConcurrent(n int) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if n < 1 {
		n = 1
	}
	m.maxConcurrent = n
}

// Register adds a hand to the manager. If a hand with the same name exists,
// it is replaced.
func (m *Manager) Register(hand Hand) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	h := hand
	h.State = HandIdle
	m.hands[h.Name] = &h

	if h.Schedule != "" {
		sched, err := ParseSchedule(h.Schedule)
		if err != nil {
			return fmt.Errorf("invalid schedule for hand %q: %w", h.Name, err)
		}
		m.schedules[h.Name] = sched
		if h.Enabled {
			m.nextRuns[h.Name] = NextRun(sched, time.Now())
		}
	}

	return m.store.SaveHand(&h)
}

// Activate enables a hand for scheduled execution.
func (m *Manager) Activate(name string) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	h, ok := m.hands[name]
	if !ok {
		return fmt.Errorf("hand %q not found", name)
	}
	h.Enabled = true

	if sched, ok := m.schedules[name]; ok {
		m.nextRuns[name] = NextRun(sched, time.Now())
	}

	return m.store.SaveHand(h)
}

// Deactivate disables a hand from scheduled execution.
func (m *Manager) Deactivate(name string) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	h, ok := m.hands[name]
	if !ok {
		return fmt.Errorf("hand %q not found", name)
	}
	h.Enabled = false
	delete(m.nextRuns, name)

	return m.store.SaveHand(h)
}

// List returns all registered hands.
func (m *Manager) List() []*Hand {
	m.mu.RLock()
	defer m.mu.RUnlock()
	out := make([]*Hand, 0, len(m.hands))
	for _, h := range m.hands {
		handCopy := *h
		out = append(out, &handCopy)
	}
	return out
}

// Status returns the current state of a named hand.
func (m *Manager) Status(name string) (*Hand, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	h, ok := m.hands[name]
	if !ok {
		return nil, fmt.Errorf("hand %q not found", name)
	}
	handCopy := *h
	return &handCopy, nil
}

// History returns run history for a hand.
func (m *Manager) History(name string) []RunRecord {
	return m.store.History(name)
}

// RunNow triggers an immediate execution of a hand, ignoring its schedule.
// This is non-blocking — the hand runs in a goroutine.
func (m *Manager) RunNow(ctx context.Context, name string) error {
	m.mu.Lock()
	h, ok := m.hands[name]
	if !ok {
		m.mu.Unlock()
		return fmt.Errorf("hand %q not found", name)
	}
	if h.State == HandRunning {
		m.mu.Unlock()
		return fmt.Errorf("hand %q is already running", name)
	}
	if m.running >= m.maxConcurrent {
		m.mu.Unlock()
		return fmt.Errorf("concurrency limit reached (%d running)", m.running)
	}
	h.State = HandRunning
	m.running++
	m.wg.Add(1)
	m.mu.Unlock()

	go m.executeHand(ctx, name)
	return nil
}

// Run starts the background scheduler loop. It checks every 30 seconds
// for hands that are due to run.
func (m *Manager) Run(ctx context.Context) error {
	// Handle @startup hands
	m.mu.Lock()
	for name, sched := range m.schedules {
		if sched.AtStartup {
			h := m.hands[name]
			if h != nil && h.Enabled {
				h.State = HandRunning
				m.running++
				m.wg.Add(1)
				go m.executeHand(ctx, name)
			}
		}
	}
	m.mu.Unlock()

	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			m.wg.Wait()
			return ctx.Err()
		case now := <-ticker.C:
			m.checkSchedules(ctx, now)
		}
	}
}

// checkSchedules fires any hands that are due.
func (m *Manager) checkSchedules(ctx context.Context, now time.Time) {
	m.mu.Lock()
	defer m.mu.Unlock()

	for name, nextRun := range m.nextRuns {
		if nextRun.IsZero() || now.Before(nextRun) {
			continue
		}

		h, ok := m.hands[name]
		if !ok || !h.Enabled || h.State == HandRunning {
			continue
		}
		if m.running >= m.maxConcurrent {
			break // respect concurrency limit
		}

		h.State = HandRunning
		m.running++
		m.wg.Add(1)

		// Compute next run time
		if sched, ok := m.schedules[name]; ok {
			m.nextRuns[name] = NextRun(sched, now)
		}

		go m.executeHand(ctx, name)
	}
}

// executeHand runs a hand and records the result.
func (m *Manager) executeHand(ctx context.Context, name string) {
	defer m.wg.Done()

	m.mu.RLock()
	h, ok := m.hands[name]
	if !ok {
		m.mu.RUnlock()
		return
	}
	// Make a copy for the runner
	handCopy := *h
	m.mu.RUnlock()

	// Emit started event
	m.board.Set("hand_active:"+name, name)
	m.board.RecordAction(blackboard.ActionRecord{
		StepID:    fmt.Sprintf("hand-start-%s", name),
		Tool:      "hand",
		Input:     name,
		Output:    "started",
		Success:   true,
		Timestamp: time.Now(),
	})

	// Execute
	result := m.runner.Run(ctx, &handCopy)
	startedAt := time.Now().Add(-result.Duration)

	// Record run
	success := result.Error == ""
	rec := RunRecord{
		HandName:  name,
		StartedAt: startedAt,
		Duration:  result.Duration.Milliseconds(),
		Success:   success,
		Output:    truncateResult(result.Output, 2000),
		Error:     result.Error,
		ToolCalls: result.ToolCalls,
	}
	_ = m.store.RecordRun(rec)

	// Update hand state — always reset to Idle so the scheduler can
	// trigger the hand again on the next schedule tick. Leaving the
	// state as Failed would permanently prevent re-scheduling.
	m.mu.Lock()
	if h, ok := m.hands[name]; ok {
		h.LastRun = time.Now()
		if success {
			h.LastError = ""
		} else {
			h.LastError = result.Error
		}
		h.State = HandIdle
		_ = m.store.SaveHand(h)
	}
	m.running--
	m.mu.Unlock()

	// Emit completion event
	eventType := "hand_completed"
	if !success {
		eventType = "hand_failed"
	}
	m.board.Set(eventType, name)
	m.board.Delete("hand_active:" + name)
}
