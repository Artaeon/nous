package cognitive

import (
	"encoding/json"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"
)

// PersistentFactStore extends FactStore with disk persistence.
// Facts are saved as JSON and loaded on startup, giving Nous
// knowledge that survives across sessions.
type PersistentFactStore struct {
	mu       sync.RWMutex
	facts    []StoredFact
	byTopic  map[string][]int
	bySource map[string][]int
	path     string
	dirty    bool
}

// StoredFact is a Fact with persistence metadata.
type StoredFact struct {
	Text         string    `json:"text"`
	Source       string    `json:"source"`
	Topic        string    `json:"topic"`
	Score        float64   `json:"score"`
	IsDefinition bool     `json:"is_definition,omitempty"`
	StoredAt     time.Time `json:"stored_at"`
	AccessCount  int       `json:"access_count"`
}

// NewPersistentFactStore creates or loads a persistent fact store.
func NewPersistentFactStore(path string) *PersistentFactStore {
	pfs := &PersistentFactStore{
		byTopic:  make(map[string][]int),
		bySource: make(map[string][]int),
		path:     path,
	}
	pfs.load()
	return pfs
}

// Add stores a fact and marks for save.
func (pfs *PersistentFactStore) Add(f Fact) {
	pfs.mu.Lock()
	defer pfs.mu.Unlock()

	// Deduplicate by text
	for _, existing := range pfs.facts {
		if existing.Text == f.Text {
			return
		}
	}

	idx := len(pfs.facts)
	sf := StoredFact{
		Text:         f.Text,
		Source:       f.Source,
		Topic:        f.Topic,
		Score:        f.Score,
		IsDefinition: f.IsDefinition,
		StoredAt:     time.Now(),
	}
	pfs.facts = append(pfs.facts, sf)

	key := strings.ToLower(f.Topic)
	if key != "" {
		pfs.byTopic[key] = append(pfs.byTopic[key], idx)
	}
	if f.Source != "" {
		pfs.bySource[f.Source] = append(pfs.bySource[f.Source], idx)
	}
	pfs.dirty = true
}

// AddMany adds multiple facts efficiently.
func (pfs *PersistentFactStore) AddMany(facts []Fact) {
	for _, f := range facts {
		pfs.Add(f)
	}
}

// Size returns the number of stored facts.
func (pfs *PersistentFactStore) Size() int {
	pfs.mu.RLock()
	defer pfs.mu.RUnlock()
	return len(pfs.facts)
}

// FactsAbout returns facts related to a topic.
func (pfs *PersistentFactStore) FactsAbout(topic string) []Fact {
	pfs.mu.RLock()
	defer pfs.mu.RUnlock()

	key := strings.ToLower(topic)
	var results []Fact

	// Direct topic match
	if indices, ok := pfs.byTopic[key]; ok {
		for _, i := range indices {
			pfs.facts[i].AccessCount++
			results = append(results, pfs.toFact(i))
		}
	}

	// Partial match
	for t, indices := range pfs.byTopic {
		if t == key {
			continue
		}
		if strings.Contains(t, key) || strings.Contains(key, t) {
			for _, i := range indices {
				results = append(results, pfs.toFact(i))
			}
		}
	}

	return results
}

// FactsFromSource returns all facts from a given source.
func (pfs *PersistentFactStore) FactsFromSource(source string) []Fact {
	pfs.mu.RLock()
	defer pfs.mu.RUnlock()

	var results []Fact
	if indices, ok := pfs.bySource[source]; ok {
		for _, i := range indices {
			results = append(results, pfs.toFact(i))
		}
	}
	return results
}

// AllFacts returns all stored facts.
func (pfs *PersistentFactStore) AllFacts() []Fact {
	pfs.mu.RLock()
	defer pfs.mu.RUnlock()

	facts := make([]Fact, len(pfs.facts))
	for i := range pfs.facts {
		facts[i] = pfs.toFact(i)
	}
	return facts
}

// Topics returns all known topics.
func (pfs *PersistentFactStore) Topics() []string {
	pfs.mu.RLock()
	defer pfs.mu.RUnlock()

	var topics []string
	for t := range pfs.byTopic {
		topics = append(topics, t)
	}
	sort.Strings(topics)
	return topics
}

// Save persists facts to disk.
func (pfs *PersistentFactStore) Save() error {
	pfs.mu.Lock()
	defer pfs.mu.Unlock()

	if !pfs.dirty {
		return nil
	}

	// Prune old low-score facts to keep the store manageable
	if len(pfs.facts) > 500 {
		pfs.prune()
	}

	dir := filepath.Dir(pfs.path)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}

	data, err := json.MarshalIndent(pfs.facts, "", "  ")
	if err != nil {
		return err
	}

	if err := os.WriteFile(pfs.path, data, 0o644); err != nil {
		return err
	}

	pfs.dirty = false
	return nil
}

// load reads facts from disk.
func (pfs *PersistentFactStore) load() {
	data, err := os.ReadFile(pfs.path)
	if err != nil {
		return // file doesn't exist yet
	}

	var facts []StoredFact
	if err := json.Unmarshal(data, &facts); err != nil {
		return
	}

	pfs.facts = facts
	pfs.rebuildIndex()
}

// rebuildIndex rebuilds the topic and source indexes.
func (pfs *PersistentFactStore) rebuildIndex() {
	pfs.byTopic = make(map[string][]int)
	pfs.bySource = make(map[string][]int)

	for i, f := range pfs.facts {
		if f.Topic != "" {
			key := strings.ToLower(f.Topic)
			pfs.byTopic[key] = append(pfs.byTopic[key], i)
		}
		if f.Source != "" {
			pfs.bySource[f.Source] = append(pfs.bySource[f.Source], i)
		}
	}
}

// prune removes the lowest-scored facts to stay under limit.
func (pfs *PersistentFactStore) prune() {
	// Score = base score * recency * access boost
	type scored struct {
		idx   int
		score float64
	}

	now := time.Now()
	var items []scored
	for i, f := range pfs.facts {
		age := now.Sub(f.StoredAt).Hours() / 24 // days
		recency := 1.0 / (1.0 + age/30.0)       // half-life ~30 days
		accessBoost := 1.0 + float64(f.AccessCount)*0.1
		items = append(items, scored{i, f.Score * recency * accessBoost})
	}

	// Sort by score ascending (worst first)
	sort.Slice(items, func(i, j int) bool {
		return items[i].score < items[j].score
	})

	// Remove bottom 20%
	removeCount := len(items) / 5
	removeSet := make(map[int]bool)
	for i := 0; i < removeCount; i++ {
		removeSet[items[i].idx] = true
	}

	var kept []StoredFact
	for i, f := range pfs.facts {
		if !removeSet[i] {
			kept = append(kept, f)
		}
	}
	pfs.facts = kept
	pfs.rebuildIndex()
}

// toFact converts a StoredFact to a Fact.
func (pfs *PersistentFactStore) toFact(i int) Fact {
	sf := pfs.facts[i]
	return Fact{
		Text:         sf.Text,
		Source:       sf.Source,
		Topic:        sf.Topic,
		Score:        sf.Score,
		IsDefinition: sf.IsDefinition,
	}
}
