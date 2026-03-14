package cognitive

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/artaeon/nous/internal/ollama"
	"github.com/artaeon/nous/internal/safefile"
)

// Branch represents a fork of a conversation, allowing users to explore
// alternative approaches without losing the original thread.
type Branch struct {
	ID        string           `json:"id"`
	ParentID  string           `json:"parent_id,omitempty"` // empty for root
	Name      string           `json:"name"`
	Messages  []ollama.Message `json:"messages"`
	CreatedAt time.Time        `json:"created_at"`
}

// BranchManager tracks conversation branches with persistence.
type BranchManager struct {
	mu       sync.RWMutex
	branches map[string]*Branch
	current  string
	file     string
	counter  int64
}

// NewBranchManager creates a manager that persists branches to the given path.
func NewBranchManager(path string) *BranchManager {
	bm := &BranchManager{
		branches: make(map[string]*Branch),
		file:     path,
	}
	_ = bm.Load()
	return bm
}

// Fork creates a new branch from the current conversation state.
// It copies the provided messages into the new branch and returns the branch ID.
func (bm *BranchManager) Fork(name string, messages []ollama.Message) string {
	bm.mu.Lock()
	defer bm.mu.Unlock()

	bm.counter++
	id := fmt.Sprintf("br-%d-%d", time.Now().UnixMilli(), bm.counter)

	copied := make([]ollama.Message, len(messages))
	copy(copied, messages)

	branch := &Branch{
		ID:        id,
		ParentID:  bm.current,
		Name:      name,
		Messages:  copied,
		CreatedAt: time.Now(),
	}
	bm.branches[id] = branch

	// If this is the first branch, also store current as "main"
	if bm.current == "" {
		main := &Branch{
			ID:        "main",
			Name:      "main",
			Messages:  copied,
			CreatedAt: time.Now(),
		}
		bm.branches["main"] = main
		bm.current = "main"
	}

	bm.current = id
	return id
}

// Switch changes the active branch. Returns the branch so the caller
// can restore its messages into the conversation.
func (bm *BranchManager) Switch(id string) (*Branch, error) {
	bm.mu.Lock()
	defer bm.mu.Unlock()

	// Allow lookup by name
	target := bm.findLocked(id)
	if target == nil {
		return nil, fmt.Errorf("branch %q not found", id)
	}
	bm.current = target.ID
	return target, nil
}

// List returns all branches sorted by creation time.
func (bm *BranchManager) List() []*Branch {
	bm.mu.RLock()
	defer bm.mu.RUnlock()

	out := make([]*Branch, 0, len(bm.branches))
	for _, b := range bm.branches {
		out = append(out, b)
	}
	return out
}

// Current returns the active branch, or nil if no branches exist.
func (bm *BranchManager) Current() *Branch {
	bm.mu.RLock()
	defer bm.mu.RUnlock()
	return bm.branches[bm.current]
}

// CurrentID returns the current branch ID.
func (bm *BranchManager) CurrentID() string {
	bm.mu.RLock()
	defer bm.mu.RUnlock()
	return bm.current
}

// UpdateMessages updates the messages on the current branch (called after each exchange).
func (bm *BranchManager) UpdateMessages(messages []ollama.Message) {
	bm.mu.Lock()
	defer bm.mu.Unlock()
	if b, ok := bm.branches[bm.current]; ok {
		copied := make([]ollama.Message, len(messages))
		copy(copied, messages)
		b.Messages = copied
	}
}

// Merge appends all messages from the given branch to the current branch.
func (bm *BranchManager) Merge(id string) error {
	bm.mu.Lock()
	defer bm.mu.Unlock()

	source := bm.findLocked(id)
	if source == nil {
		return fmt.Errorf("branch %q not found", id)
	}
	cur, ok := bm.branches[bm.current]
	if !ok {
		return fmt.Errorf("no current branch")
	}
	if source.ID == cur.ID {
		return fmt.Errorf("cannot merge branch into itself")
	}

	// Append non-system messages from source that differ from current
	existingSet := make(map[string]bool)
	for _, m := range cur.Messages {
		existingSet[m.Role+":"+m.Content] = true
	}
	for _, m := range source.Messages {
		if m.Role == "system" {
			continue
		}
		key := m.Role + ":" + m.Content
		if !existingSet[key] {
			cur.Messages = append(cur.Messages, m)
		}
	}
	return nil
}

// Delete removes a branch. Cannot delete the current branch.
func (bm *BranchManager) Delete(id string) error {
	bm.mu.Lock()
	defer bm.mu.Unlock()

	target := bm.findLocked(id)
	if target == nil {
		return fmt.Errorf("branch %q not found", id)
	}
	if target.ID == bm.current {
		return fmt.Errorf("cannot delete the current branch (switch first)")
	}
	delete(bm.branches, target.ID)
	return nil
}

// Diff returns messages unique to each branch when comparing two branches.
func (bm *BranchManager) Diff(id string) (onlyCurrent, onlyOther []ollama.Message, err error) {
	bm.mu.RLock()
	defer bm.mu.RUnlock()

	other := bm.findLocked(id)
	if other == nil {
		return nil, nil, fmt.Errorf("branch %q not found", id)
	}
	cur, ok := bm.branches[bm.current]
	if !ok {
		return nil, nil, fmt.Errorf("no current branch")
	}

	otherSet := make(map[string]bool)
	for _, m := range other.Messages {
		otherSet[m.Role+":"+m.Content] = true
	}
	curSet := make(map[string]bool)
	for _, m := range cur.Messages {
		curSet[m.Role+":"+m.Content] = true
	}

	for _, m := range cur.Messages {
		if !otherSet[m.Role+":"+m.Content] {
			onlyCurrent = append(onlyCurrent, m)
		}
	}
	for _, m := range other.Messages {
		if !curSet[m.Role+":"+m.Content] {
			onlyOther = append(onlyOther, m)
		}
	}
	return
}

// Save persists branches to disk.
func (bm *BranchManager) Save() error {
	bm.mu.RLock()
	defer bm.mu.RUnlock()

	state := struct {
		Current  string             `json:"current"`
		Branches map[string]*Branch `json:"branches"`
	}{
		Current:  bm.current,
		Branches: bm.branches,
	}

	data, err := json.MarshalIndent(state, "", "  ")
	if err != nil {
		return err
	}
	return safefile.WriteAtomic(bm.file, data, 0644)
}

// Load restores branches from disk.
func (bm *BranchManager) Load() error {
	data, err := os.ReadFile(bm.file)
	if err != nil {
		return err
	}

	var state struct {
		Current  string             `json:"current"`
		Branches map[string]*Branch `json:"branches"`
	}
	if err := json.Unmarshal(data, &state); err != nil {
		return err
	}

	bm.mu.Lock()
	defer bm.mu.Unlock()
	if state.Branches != nil {
		bm.branches = state.Branches
	}
	bm.current = state.Current
	return nil
}

// findLocked searches by ID or name. Must be called with lock held.
func (bm *BranchManager) findLocked(idOrName string) *Branch {
	if b, ok := bm.branches[idOrName]; ok {
		return b
	}
	for _, b := range bm.branches {
		if b.Name == idOrName {
			return b
		}
	}
	return nil
}

// BranchDir returns the default directory for branch persistence.
func BranchDir(basePath string) string {
	return filepath.Join(basePath, "branches.json")
}
