package hands

import (
	"fmt"
	"sync"
	"time"
)

// PendingApproval represents a hand action awaiting user review.
type PendingApproval struct {
	ID        string    `json:"id"`
	HandName  string    `json:"hand_name"`
	ToolName  string    `json:"tool_name"`
	ToolArgs  string    `json:"tool_args"`
	Timestamp time.Time `json:"timestamp"`
}

// ApprovalQueue manages pending hand approvals.
type ApprovalQueue struct {
	mu      sync.RWMutex
	pending []PendingApproval
	nextID  int
}

// NewApprovalQueue creates an empty approval queue.
func NewApprovalQueue() *ApprovalQueue {
	return &ApprovalQueue{}
}

// Add queues a new pending approval and returns its ID.
func (q *ApprovalQueue) Add(handName, toolName, toolArgs string) string {
	q.mu.Lock()
	defer q.mu.Unlock()

	q.nextID++
	id := fmt.Sprintf("approval-%d", q.nextID)
	q.pending = append(q.pending, PendingApproval{
		ID:        id,
		HandName:  handName,
		ToolName:  toolName,
		ToolArgs:  toolArgs,
		Timestamp: time.Now(),
	})
	return id
}

// GetPending returns all pending approvals.
func (q *ApprovalQueue) GetPending() []PendingApproval {
	q.mu.RLock()
	defer q.mu.RUnlock()

	out := make([]PendingApproval, len(q.pending))
	copy(out, q.pending)
	return out
}

// Approve removes and returns the approval with the given ID.
// Returns the approval and true if found, or zero value and false if not.
func (q *ApprovalQueue) Approve(id string) (PendingApproval, bool) {
	q.mu.Lock()
	defer q.mu.Unlock()

	for i, a := range q.pending {
		if a.ID == id {
			q.pending = append(q.pending[:i], q.pending[i+1:]...)
			return a, true
		}
	}
	return PendingApproval{}, false
}

// Reject removes the approval with the given ID without executing it.
// Returns true if the approval was found and removed.
func (q *ApprovalQueue) Reject(id string) bool {
	q.mu.Lock()
	defer q.mu.Unlock()

	for i, a := range q.pending {
		if a.ID == id {
			q.pending = append(q.pending[:i], q.pending[i+1:]...)
			return true
		}
	}
	return false
}

// Len returns the number of pending approvals.
func (q *ApprovalQueue) Len() int {
	q.mu.RLock()
	defer q.mu.RUnlock()
	return len(q.pending)
}
