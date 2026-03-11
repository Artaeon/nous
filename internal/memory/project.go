package memory

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"
)

// ProjectFact represents a single piece of project-specific knowledge.
type ProjectFact struct {
	Key        string    `json:"key"`
	Value      string    `json:"value"`
	Source     string    `json:"source"`     // how this was learned: "user", "inferred", "code"
	Confidence float64  `json:"confidence"` // 0.0–1.0
	CreatedAt  time.Time `json:"created_at"`
	UpdatedAt  time.Time `json:"updated_at"`
}

// ProjectMemory persists knowledge specific to the current project.
// It stores facts in the project's .nous/ directory, keeping project
// conventions, architecture decisions, and key patterns across sessions.
type ProjectMemory struct {
	mu    sync.RWMutex
	facts map[string]ProjectFact
	path  string
}

// NewProjectMemory creates a project memory stored at <projectDir>/.nous/project_memory.json.
func NewProjectMemory(projectDir string) *ProjectMemory {
	pm := &ProjectMemory{
		facts: make(map[string]ProjectFact),
		path:  filepath.Join(projectDir, ".nous", "project_memory.json"),
	}
	pm.load()
	return pm
}

// Remember stores a project fact. If the key already exists, it updates the value.
func (pm *ProjectMemory) Remember(key, value, source string, confidence float64) {
	pm.mu.Lock()
	defer pm.mu.Unlock()

	now := time.Now()
	existing, exists := pm.facts[key]

	fact := ProjectFact{
		Key:        key,
		Value:      value,
		Source:     source,
		Confidence: confidence,
		CreatedAt:  now,
		UpdatedAt:  now,
	}

	if exists {
		fact.CreatedAt = existing.CreatedAt
	}

	pm.facts[key] = fact
}

// Recall performs an exact key lookup.
func (pm *ProjectMemory) Recall(key string) (ProjectFact, bool) {
	pm.mu.RLock()
	defer pm.mu.RUnlock()
	fact, ok := pm.facts[key]
	return fact, ok
}

// Search performs a case-insensitive keyword search across keys and values.
func (pm *ProjectMemory) Search(query string) []ProjectFact {
	pm.mu.RLock()
	defer pm.mu.RUnlock()

	words := strings.Fields(strings.ToLower(query))
	var results []ProjectFact

	for _, fact := range pm.facts {
		keyLower := strings.ToLower(fact.Key)
		valueLower := strings.ToLower(fact.Value)

		for _, word := range words {
			if strings.Contains(keyLower, word) || strings.Contains(valueLower, word) {
				results = append(results, fact)
				break
			}
		}
	}
	return results
}

// Forget removes a fact by key.
func (pm *ProjectMemory) Forget(key string) bool {
	pm.mu.Lock()
	defer pm.mu.Unlock()
	_, existed := pm.facts[key]
	delete(pm.facts, key)
	return existed
}

// All returns every stored fact.
func (pm *ProjectMemory) All() []ProjectFact {
	pm.mu.RLock()
	defer pm.mu.RUnlock()

	results := make([]ProjectFact, 0, len(pm.facts))
	for _, f := range pm.facts {
		results = append(results, f)
	}
	return results
}

// Size returns the number of stored facts.
func (pm *ProjectMemory) Size() int {
	pm.mu.RLock()
	defer pm.mu.RUnlock()
	return len(pm.facts)
}

// Flush writes the current facts to disk.
func (pm *ProjectMemory) Flush() error {
	pm.mu.RLock()
	defer pm.mu.RUnlock()

	if err := os.MkdirAll(filepath.Dir(pm.path), 0755); err != nil {
		return err
	}

	data, err := json.MarshalIndent(pm.facts, "", "  ")
	if err != nil {
		return err
	}

	return os.WriteFile(pm.path, data, 0644)
}

func (pm *ProjectMemory) load() {
	data, err := os.ReadFile(pm.path)
	if err != nil {
		return
	}
	_ = json.Unmarshal(data, &pm.facts)
}
