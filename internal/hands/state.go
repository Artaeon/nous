package hands

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"sync"
)

// HandStateStore provides persistent key-value state for each hand between runs.
// State is stored as JSON files in the configured directory, one per hand.
type HandStateStore struct {
	mu  sync.RWMutex
	dir string
	// In-memory cache: handName -> key -> value
	data map[string]map[string]string
}

// NewHandStateStore creates a state store backed by the given directory.
func NewHandStateStore(dir string) (*HandStateStore, error) {
	if err := os.MkdirAll(dir, 0755); err != nil {
		return nil, fmt.Errorf("create state dir: %w", err)
	}
	s := &HandStateStore{
		dir:  dir,
		data: make(map[string]map[string]string),
	}
	if err := s.loadAll(); err != nil {
		return nil, err
	}
	return s, nil
}

// Get returns the value for a key in the given hand's state.
// Returns empty string if key or hand doesn't exist.
func (s *HandStateStore) Get(handName, key string) string {
	s.mu.RLock()
	defer s.mu.RUnlock()
	if m, ok := s.data[handName]; ok {
		return m[key]
	}
	return ""
}

// Set stores a key-value pair for a hand and persists to disk.
func (s *HandStateStore) Set(handName, key, value string) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.data[handName] == nil {
		s.data[handName] = make(map[string]string)
	}
	s.data[handName][key] = value

	return s.persist(handName)
}

// GetAll returns a copy of all key-value pairs for a hand.
func (s *HandStateStore) GetAll(handName string) map[string]string {
	s.mu.RLock()
	defer s.mu.RUnlock()

	m, ok := s.data[handName]
	if !ok {
		return nil
	}
	out := make(map[string]string, len(m))
	for k, v := range m {
		out[k] = v
	}
	return out
}

// persist writes the state for a hand to its JSON file.
func (s *HandStateStore) persist(handName string) error {
	m := s.data[handName]
	if m == nil {
		return nil
	}

	data, err := json.MarshalIndent(m, "", "  ")
	if err != nil {
		return fmt.Errorf("marshal state for %s: %w", handName, err)
	}

	path := filepath.Join(s.dir, handName+".json")
	if err := os.WriteFile(path, data, 0644); err != nil {
		return fmt.Errorf("write state for %s: %w", handName, err)
	}
	return nil
}

// loadAll reads all existing state files from the directory into memory.
func (s *HandStateStore) loadAll() error {
	entries, err := os.ReadDir(s.dir)
	if err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		return fmt.Errorf("read state dir: %w", err)
	}

	for _, entry := range entries {
		if entry.IsDir() || filepath.Ext(entry.Name()) != ".json" {
			continue
		}
		handName := entry.Name()[:len(entry.Name())-5] // strip .json

		data, err := os.ReadFile(filepath.Join(s.dir, entry.Name()))
		if err != nil {
			continue
		}

		var m map[string]string
		if err := json.Unmarshal(data, &m); err != nil {
			continue
		}
		s.data[handName] = m
	}
	return nil
}

// statePattern matches [STATE key=value] in hand output.
var statePattern = regexp.MustCompile(`\[STATE\s+(\S+?)=(.+?)\]`)

// ExtractState parses [STATE key=value] patterns from hand output.
func ExtractState(output string) map[string]string {
	matches := statePattern.FindAllStringSubmatch(output, -1)
	if len(matches) == 0 {
		return nil
	}
	result := make(map[string]string, len(matches))
	for _, match := range matches {
		result[match[1]] = match[2]
	}
	return result
}

// FormatStatePrompt builds a system prompt section from persisted state.
func FormatStatePrompt(state map[string]string) string {
	if len(state) == 0 {
		return ""
	}
	out := "Previous run context:\n"
	for k, v := range state {
		out += fmt.Sprintf("  %s = %s\n", k, v)
	}
	return out
}
