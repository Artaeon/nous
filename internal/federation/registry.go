package federation

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
)

// RegistryStats summarises the contents of a crystal registry.
type RegistryStats struct {
	TotalCrystals int            `json:"total_crystals"`
	TotalBundles  int            `json:"total_bundles"`
	AvgQuality    float64        `json:"avg_quality"`
	TopIntents    map[string]int `json:"top_intents"` // intent -> count
}

// Registry manages the shared crystal registry.
// Uses a simple file-based approach (directory of JSON bundles).
// Can optionally sync with a git remote for distribution.
type Registry struct {
	Path     string                    // local registry directory
	Crystals map[string]*SharedCrystal // ID -> crystal (merged from all bundles)
	bundles  int                       // number of loaded bundles
	mu       sync.RWMutex
}

// NewRegistry creates or opens a registry at the given directory path.
// If the directory does not exist it is created. Existing bundles are loaded.
func NewRegistry(path string) (*Registry, error) {
	if err := os.MkdirAll(path, 0o755); err != nil {
		return nil, fmt.Errorf("create registry dir: %w", err)
	}

	r := &Registry{
		Path:     path,
		Crystals: make(map[string]*SharedCrystal),
	}

	if err := r.Load(); err != nil {
		return nil, err
	}
	return r, nil
}

// Publish adds a bundle to the registry directory and merges its crystals
// into the in-memory index.
func (r *Registry) Publish(bundle *CrystalBundle) error {
	if err := bundle.Validate(); err != nil {
		return fmt.Errorf("invalid bundle: %w", err)
	}

	filename := fmt.Sprintf("bundle-%s-%d.json", bundle.Instance, bundle.Exported.Unix())
	path := filepath.Join(r.Path, filename)

	if err := bundle.Export(path); err != nil {
		return fmt.Errorf("write bundle: %w", err)
	}

	r.mu.Lock()
	defer r.mu.Unlock()

	for _, c := range bundle.Crystals {
		r.mergeLocked(c)
	}
	r.bundles++
	return nil
}

// Search returns crystals whose pattern or intent contain any of the
// whitespace-delimited keywords in query. Results are ordered by quality
// descending and capped at limit.
func (r *Registry) Search(query string, limit int) []SharedCrystal {
	keywords := strings.Fields(strings.ToLower(query))
	if len(keywords) == 0 {
		return nil
	}

	r.mu.RLock()
	defer r.mu.RUnlock()

	var results []SharedCrystal
	for _, c := range r.Crystals {
		text := strings.ToLower(c.Pattern + " " + c.Intent)
		match := false
		for _, kw := range keywords {
			if strings.Contains(text, kw) {
				match = true
				break
			}
		}
		if match {
			results = append(results, *c)
		}
	}

	sort.Slice(results, func(i, j int) bool {
		return results[i].Quality > results[j].Quality
	})

	if limit > 0 && len(results) > limit {
		results = results[:limit]
	}
	return results
}

// TopCrystals returns the n highest-quality crystals in the registry.
func (r *Registry) TopCrystals(n int) []SharedCrystal {
	r.mu.RLock()
	defer r.mu.RUnlock()

	all := make([]SharedCrystal, 0, len(r.Crystals))
	for _, c := range r.Crystals {
		all = append(all, *c)
	}

	sort.Slice(all, func(i, j int) bool {
		return all[i].Quality > all[j].Quality
	})

	if n > 0 && len(all) > n {
		all = all[:n]
	}
	return all
}

// Merge merges a single crystal into the registry. If a crystal with the
// same ID exists its votes are incremented and quality averaged; otherwise
// the crystal is added as-is.
func (r *Registry) Merge(crystal SharedCrystal) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.mergeLocked(crystal)
}

// mergeLocked performs the merge without acquiring the lock (caller must hold it).
func (r *Registry) mergeLocked(crystal SharedCrystal) {
	existing, ok := r.Crystals[crystal.ID]
	if !ok {
		c := crystal
		r.Crystals[crystal.ID] = &c
		return
	}

	// Weighted average quality, increment votes.
	total := existing.Votes + crystal.Votes
	existing.Quality = (existing.Quality*float64(existing.Votes) + crystal.Quality*float64(crystal.Votes)) / float64(total)
	existing.Votes = total

	if crystal.LastVoted.After(existing.LastVoted) {
		existing.LastVoted = crystal.LastVoted
	}
}

// Export writes all crystals with quality >= minQuality to a bundle file.
func (r *Registry) Export(path string, minQuality float64) error {
	r.mu.RLock()
	var crystals []SharedCrystal
	for _, c := range r.Crystals {
		if c.Quality >= minQuality {
			crystals = append(crystals, *c)
		}
	}
	r.mu.RUnlock()

	sort.Slice(crystals, func(i, j int) bool {
		return crystals[i].Quality > crystals[j].Quality
	})

	bundle := NewCrystalBundle("export", crystals)
	return bundle.Export(path)
}

// Load reads all JSON bundle files from the registry directory and merges
// their crystals into the in-memory index. It replaces the current state.
func (r *Registry) Load() error {
	entries, err := os.ReadDir(r.Path)
	if err != nil {
		return fmt.Errorf("read registry dir: %w", err)
	}

	r.mu.Lock()
	defer r.mu.Unlock()

	r.Crystals = make(map[string]*SharedCrystal)
	r.bundles = 0

	for _, entry := range entries {
		if entry.IsDir() || !strings.HasSuffix(entry.Name(), ".json") {
			continue
		}

		bundle, err := ImportBundle(filepath.Join(r.Path, entry.Name()))
		if err != nil {
			continue // skip corrupt bundles
		}
		if bundle.Validate() != nil {
			continue
		}

		for _, c := range bundle.Crystals {
			r.mergeLocked(c)
		}
		r.bundles++
	}
	return nil
}

// Stats returns aggregate statistics about the registry.
func (r *Registry) Stats() RegistryStats {
	r.mu.RLock()
	defer r.mu.RUnlock()

	stats := RegistryStats{
		TotalCrystals: len(r.Crystals),
		TotalBundles:  r.bundles,
		TopIntents:    make(map[string]int),
	}

	var totalQuality float64
	for _, c := range r.Crystals {
		totalQuality += c.Quality
		if c.Intent != "" {
			stats.TopIntents[c.Intent]++
		}
	}

	if stats.TotalCrystals > 0 {
		stats.AvgQuality = totalQuality / float64(stats.TotalCrystals)
	}
	return stats
}
