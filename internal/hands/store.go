package hands

import (
	"encoding/json"
	"os"
	"path/filepath"
	"sync"

	"github.com/artaeon/nous/internal/safefile"
)

// Store persists hand configurations and run history to disk.
type Store struct {
	mu       sync.RWMutex
	baseDir  string
	hands    map[string]*Hand
	history  map[string][]RunRecord
	maxRuns  int // max history entries per hand (default 50)
}

// storeData is the on-disk format.
type storeData struct {
	Hands   map[string]*Hand        `json:"hands"`
	History map[string][]RunRecord  `json:"history"`
}

// NewStore creates or loads a hand store at the given base directory.
func NewStore(baseDir string) *Store {
	s := &Store{
		baseDir: baseDir,
		hands:   make(map[string]*Hand),
		history: make(map[string][]RunRecord),
		maxRuns: 50,
	}
	s.load()
	return s
}

// SetMaxRuns configures how many run records to retain per hand.
func (s *Store) SetMaxRuns(n int) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if n < 1 {
		n = 1
	}
	s.maxRuns = n
}

// SaveHand persists a hand configuration.
func (s *Store) SaveHand(h *Hand) error {
	s.mu.Lock()
	s.hands[h.Name] = h
	err := s.saveLocked()
	s.mu.Unlock()
	return err
}

// GetHand retrieves a hand by name.
func (s *Store) GetHand(name string) (*Hand, bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	h, ok := s.hands[name]
	if !ok {
		return nil, false
	}
	// Return a copy
	handCopy := *h
	return &handCopy, true
}

// AllHands returns all registered hands.
func (s *Store) AllHands() []*Hand {
	s.mu.RLock()
	defer s.mu.RUnlock()
	out := make([]*Hand, 0, len(s.hands))
	for _, h := range s.hands {
		handCopy := *h
		out = append(out, &handCopy)
	}
	return out
}

// RecordRun appends a run record and trims to maxRuns.
func (s *Store) RecordRun(rec RunRecord) error {
	s.mu.Lock()
	runs := s.history[rec.HandName]
	runs = append(runs, rec)
	if len(runs) > s.maxRuns {
		runs = runs[len(runs)-s.maxRuns:]
	}
	s.history[rec.HandName] = runs
	err := s.saveLocked()
	s.mu.Unlock()
	return err
}

// History returns run records for a hand, most recent last.
func (s *Store) History(name string) []RunRecord {
	s.mu.RLock()
	defer s.mu.RUnlock()
	runs := s.history[name]
	out := make([]RunRecord, len(runs))
	copy(out, runs)
	return out
}

// Stats returns success/fail counts and average duration for a hand.
func (s *Store) Stats(name string) (successes, failures int, avgDurationMs int64) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	runs := s.history[name]
	if len(runs) == 0 {
		return 0, 0, 0
	}
	var totalMs int64
	for _, r := range runs {
		if r.Success {
			successes++
		} else {
			failures++
		}
		totalMs += r.Duration
	}
	avgDurationMs = totalMs / int64(len(runs))
	return
}

// DeleteHand removes a hand and its history.
func (s *Store) DeleteHand(name string) error {
	s.mu.Lock()
	delete(s.hands, name)
	delete(s.history, name)
	err := s.saveLocked()
	s.mu.Unlock()
	return err
}

func (s *Store) load() {
	path := filepath.Join(s.baseDir, "hands.json")
	data, err := os.ReadFile(path)
	if err != nil {
		return // file doesn't exist yet
	}
	var sd storeData
	if err := json.Unmarshal(data, &sd); err != nil {
		return
	}
	if sd.Hands != nil {
		s.hands = sd.Hands
	}
	if sd.History != nil {
		s.history = sd.History
	}
}

// saveLocked persists the store to disk. The caller must hold s.mu.
func (s *Store) saveLocked() error {
	sd := storeData{
		Hands:   s.hands,
		History: s.history,
	}
	data, err := json.MarshalIndent(sd, "", "  ")
	if err != nil {
		return err
	}
	return safefile.WriteAtomic(filepath.Join(s.baseDir, "hands.json"), data, 0644)
}
