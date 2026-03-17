package cognitive

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/artaeon/nous/internal/safefile"
)

// NeuroplasticDescriptions evolves tool descriptions based on which phrasings
// lead to successful tool calls. Different models respond to different wording.
// By tracking success rates per description variant, the system automatically
// adapts to what works best for the specific model being used.
//
// Innovation: This is "prompt evolution" — a genetic algorithm applied to tool
// descriptions. Fitness is measured by actual tool call success rate. Over time,
// descriptions are optimized for THIS specific model's understanding.
//
// Example evolution:
//   Gen 1: "Search file contents for a regex pattern"  → 40% success
//   Gen 2: "USE THIS to find words inside files"       → 55% success
//   Gen 3: "Find text in code. Example: grep 'TODO'"   → 72% success
type NeuroplasticDescriptions struct {
	entries   map[string]*DescEntry // keyed by tool name
	model     string                // which model these are optimized for
	storePath string
	mu        sync.RWMutex
}

// DescEntry tracks description variants and their performance.
type DescEntry struct {
	ToolName    string        `json:"tool_name"`
	Variants    []DescVariant `json:"variants"`
	ActiveIdx   int           `json:"active_idx"`    // which variant is currently active
	Model       string        `json:"model"`
}

// DescVariant is one possible description for a tool.
type DescVariant struct {
	Text       string    `json:"text"`
	Attempts   int       `json:"attempts"`
	Successes  int       `json:"successes"`
	CreatedAt  time.Time `json:"created_at"`
	LastUsed   time.Time `json:"last_used"`
}

// SuccessRate returns the success rate of a variant.
func (v *DescVariant) SuccessRate() float64 {
	if v.Attempts == 0 {
		return 0.5 // neutral prior
	}
	return float64(v.Successes) / float64(v.Attempts)
}

// NewNeuroplasticDescriptions creates a new neuroplastic description system.
func NewNeuroplasticDescriptions(model, storePath string) *NeuroplasticDescriptions {
	nd := &NeuroplasticDescriptions{
		entries:   make(map[string]*DescEntry),
		model:     model,
		storePath: storePath,
	}
	nd.load()
	return nd
}

// RegisterDefault registers the default description for a tool.
// Only adds if no entry exists yet.
func (nd *NeuroplasticDescriptions) RegisterDefault(toolName, description string) {
	nd.mu.Lock()
	defer nd.mu.Unlock()

	if _, exists := nd.entries[toolName]; exists {
		return
	}

	nd.entries[toolName] = &DescEntry{
		ToolName: toolName,
		Model:    nd.model,
		Variants: []DescVariant{{
			Text:      description,
			CreatedAt: time.Now(),
		}},
		ActiveIdx: 0,
	}
}

// GetDescription returns the current best description for a tool.
func (nd *NeuroplasticDescriptions) GetDescription(toolName string) string {
	nd.mu.RLock()
	defer nd.mu.RUnlock()

	entry, ok := nd.entries[toolName]
	if !ok || len(entry.Variants) == 0 {
		return ""
	}

	return entry.Variants[entry.ActiveIdx].Text
}

// RecordAttempt records that a tool was called (attempt).
func (nd *NeuroplasticDescriptions) RecordAttempt(toolName string) {
	nd.mu.Lock()
	defer nd.mu.Unlock()

	entry, ok := nd.entries[toolName]
	if !ok || len(entry.Variants) == 0 {
		return
	}

	entry.Variants[entry.ActiveIdx].Attempts++
	entry.Variants[entry.ActiveIdx].LastUsed = time.Now()
}

// RecordSuccess records that a tool call was successful.
func (nd *NeuroplasticDescriptions) RecordSuccess(toolName string) {
	nd.mu.Lock()
	defer nd.mu.Unlock()

	entry, ok := nd.entries[toolName]
	if !ok || len(entry.Variants) == 0 {
		return
	}

	entry.Variants[entry.ActiveIdx].Successes++
	nd.save()
}

// AddVariant adds a new description variant for A/B testing.
func (nd *NeuroplasticDescriptions) AddVariant(toolName, description string) {
	nd.mu.Lock()
	defer nd.mu.Unlock()

	entry, ok := nd.entries[toolName]
	if !ok {
		return
	}

	// Don't add duplicates
	for _, v := range entry.Variants {
		if v.Text == description {
			return
		}
	}

	entry.Variants = append(entry.Variants, DescVariant{
		Text:      description,
		CreatedAt: time.Now(),
	})
	nd.save()
}

// Evolve checks if a better variant should be promoted to active.
// Call this periodically (e.g., every 100 interactions).
func (nd *NeuroplasticDescriptions) Evolve() map[string]string {
	nd.mu.Lock()
	defer nd.mu.Unlock()

	changes := make(map[string]string)

	for name, entry := range nd.entries {
		if len(entry.Variants) <= 1 {
			continue
		}

		// Find the variant with the highest success rate (with min 5 attempts)
		bestIdx := entry.ActiveIdx
		bestRate := entry.Variants[bestIdx].SuccessRate()

		for i, v := range entry.Variants {
			if v.Attempts >= 5 && v.SuccessRate() > bestRate {
				bestRate = v.SuccessRate()
				bestIdx = i
			}
		}

		if bestIdx != entry.ActiveIdx {
			oldDesc := entry.Variants[entry.ActiveIdx].Text
			entry.ActiveIdx = bestIdx
			changes[name] = fmt.Sprintf("%s → %s (%.0f%% → %.0f%%)",
				truncateDesc(oldDesc, 30),
				truncateDesc(entry.Variants[bestIdx].Text, 30),
				entry.Variants[entry.ActiveIdx].SuccessRate()*100,
				bestRate*100)
		}
	}

	if len(changes) > 0 {
		nd.save()
	}

	return changes
}

// GenerateVariants creates new description variants for underperforming tools.
// Returns tool names that got new variants.
func (nd *NeuroplasticDescriptions) GenerateVariants() []string {
	nd.mu.Lock()
	defer nd.mu.Unlock()

	var evolved []string

	for name, entry := range nd.entries {
		active := entry.Variants[entry.ActiveIdx]
		if active.Attempts < 10 {
			continue // not enough data
		}

		if active.SuccessRate() >= 0.7 {
			continue // performing well
		}

		// Generate enhanced variants based on common patterns
		newVariants := generateEnhancedDescriptions(name, active.Text)
		for _, nv := range newVariants {
			dupe := false
			for _, existing := range entry.Variants {
				if existing.Text == nv {
					dupe = true
					break
				}
			}
			if !dupe {
				entry.Variants = append(entry.Variants, DescVariant{
					Text:      nv,
					CreatedAt: time.Now(),
				})
			}
		}

		if len(entry.Variants) > len(newVariants) {
			evolved = append(evolved, name)
		}
	}

	if len(evolved) > 0 {
		nd.save()
	}

	return evolved
}

// Stats returns performance statistics for all tools.
func (nd *NeuroplasticDescriptions) Stats() map[string]DescStats {
	nd.mu.RLock()
	defer nd.mu.RUnlock()

	stats := make(map[string]DescStats)
	for name, entry := range nd.entries {
		active := entry.Variants[entry.ActiveIdx]
		stats[name] = DescStats{
			ActiveDesc:  truncateDesc(active.Text, 50),
			Attempts:    active.Attempts,
			Successes:   active.Successes,
			SuccessRate: active.SuccessRate(),
			Variants:    len(entry.Variants),
		}
	}
	return stats
}

// DescStats holds statistics for one tool's descriptions.
type DescStats struct {
	ActiveDesc  string
	Attempts    int
	Successes   int
	SuccessRate float64
	Variants    int
}

// --- Enhanced Description Generation ---

func generateEnhancedDescriptions(toolName, current string) []string {
	// Strategy: generate variants with different communication styles
	enhancers := map[string]func(string, string) string{
		"example":    addExampleToDesc,
		"imperative": makeImperative,
		"simplified": simplifyDesc,
	}

	var variants []string
	for _, fn := range enhancers {
		variant := fn(toolName, current)
		if variant != "" && variant != current {
			variants = append(variants, variant)
		}
	}
	return variants
}

func addExampleToDesc(toolName, current string) string {
	examples := map[string]string{
		"grep": ". Example: grep(pattern='TODO', glob='*.go') finds all TODOs in Go files",
		"read": ". Example: read(path='main.go') shows file contents",
		"ls":   ". Example: ls(path='src') lists files in src directory",
		"glob": ". Example: glob(pattern='**/*.go') finds all Go files",
		"tree": ". Example: tree(path='.') shows project structure",
		"git":  ". Example: git(command='status') shows repo status",
	}

	if ex, ok := examples[toolName]; ok {
		if !strings.Contains(current, "Example") {
			return current + ex
		}
	}
	return ""
}

func makeImperative(toolName, current string) string {
	imperative := map[string]string{
		"grep": "USE THIS to find words or patterns inside files. Needs: pattern (the search text). Optional: glob (file filter like '*.go')",
		"read": "USE THIS to see what is inside a file. Needs: path (the file to read)",
		"ls":   "USE THIS to see what files are in a directory. Optional: path (directory to list)",
		"glob": "USE THIS to find files by name pattern. Needs: pattern (like '**/*.go')",
		"tree": "USE THIS to see the directory structure. Optional: path (root directory)",
		"write": "USE THIS to create or overwrite a file. Needs: path and content",
	}

	if imp, ok := imperative[toolName]; ok {
		return imp
	}
	return ""
}

func simplifyDesc(toolName, current string) string {
	if len(current) > 80 {
		// Truncate to core description
		parts := strings.SplitN(current, ".", 2)
		if len(parts) >= 1 {
			return strings.TrimSpace(parts[0])
		}
	}
	return ""
}

func truncateDesc(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen-3] + "..."
}

// --- Prompt Evolution ---

// PromptEvolver extends neuroplastic evolution to any named system prompt.
// Unlike tool descriptions (which evolve per-tool), prompts evolve per-purpose.
type PromptEvolver struct {
	mu      sync.RWMutex
	prompts map[string]*PromptEntry // keyed by prompt name
	path    string
}

// PromptEntry tracks prompt variants and their performance.
type PromptEntry struct {
	Name      string          `json:"name"`
	Variants  []PromptVariant `json:"variants"`
	ActiveIdx int             `json:"active_idx"`
}

// PromptVariant is one version of a prompt with performance tracking.
type PromptVariant struct {
	Text      string    `json:"text"`
	Attempts  int       `json:"attempts"`
	Successes int       `json:"successes"`
	AvgScore  float64   `json:"avg_score"` // EMA of quality scores
	CreatedAt time.Time `json:"created_at"`
}

// SuccessRate returns the success rate of a prompt variant.
func (pv *PromptVariant) SuccessRate() float64 {
	if pv.Attempts == 0 {
		return 0.5
	}
	return float64(pv.Successes) / float64(pv.Attempts)
}

// NewPromptEvolver creates a prompt evolution system.
func NewPromptEvolver(path string) *PromptEvolver {
	pe := &PromptEvolver{
		prompts: make(map[string]*PromptEntry),
		path:    path,
	}
	pe.load()
	return pe
}

// Register registers a named prompt with its default text.
func (pe *PromptEvolver) Register(name, text string) {
	pe.mu.Lock()
	defer pe.mu.Unlock()

	if _, exists := pe.prompts[name]; exists {
		return
	}
	pe.prompts[name] = &PromptEntry{
		Name: name,
		Variants: []PromptVariant{{
			Text:      text,
			CreatedAt: time.Now(),
		}},
		ActiveIdx: 0,
	}
}

// Get returns the current best prompt for a given name.
func (pe *PromptEvolver) Get(name string) string {
	pe.mu.RLock()
	defer pe.mu.RUnlock()

	entry, ok := pe.prompts[name]
	if !ok || len(entry.Variants) == 0 {
		return ""
	}
	return entry.Variants[entry.ActiveIdx].Text
}

// RecordOutcome records a quality score (0.0-1.0) for the active prompt variant.
func (pe *PromptEvolver) RecordOutcome(name string, score float64, success bool) {
	pe.mu.Lock()
	defer pe.mu.Unlock()

	entry, ok := pe.prompts[name]
	if !ok || len(entry.Variants) == 0 {
		return
	}

	v := &entry.Variants[entry.ActiveIdx]
	v.Attempts++
	if success {
		v.Successes++
	}
	// EMA update for score
	const alpha = 0.2
	if v.Attempts == 1 {
		v.AvgScore = score
	} else {
		v.AvgScore = alpha*score + (1-alpha)*v.AvgScore
	}
}

// AddVariant adds a new prompt variant for testing.
func (pe *PromptEvolver) AddVariant(name, text string) {
	pe.mu.Lock()
	defer pe.mu.Unlock()

	entry, ok := pe.prompts[name]
	if !ok {
		return
	}

	for _, v := range entry.Variants {
		if v.Text == text {
			return // duplicate
		}
	}

	entry.Variants = append(entry.Variants, PromptVariant{
		Text:      text,
		CreatedAt: time.Now(),
	})
}

// Evolve promotes the best-performing variant for each prompt.
// Returns a map of prompt names that changed.
func (pe *PromptEvolver) Evolve() map[string]string {
	pe.mu.Lock()
	defer pe.mu.Unlock()

	changes := make(map[string]string)

	for name, entry := range pe.prompts {
		if len(entry.Variants) <= 1 {
			continue
		}

		bestIdx := entry.ActiveIdx
		bestScore := entry.Variants[bestIdx].AvgScore
		if entry.Variants[bestIdx].Attempts < 5 {
			bestScore = entry.Variants[bestIdx].SuccessRate()
		}

		for i, v := range entry.Variants {
			if v.Attempts < 5 {
				continue
			}
			score := v.AvgScore
			if score > bestScore {
				bestScore = score
				bestIdx = i
			}
		}

		if bestIdx != entry.ActiveIdx {
			changes[name] = fmt.Sprintf("promoted variant %d (score: %.2f → %.2f)",
				bestIdx, entry.Variants[entry.ActiveIdx].AvgScore, bestScore)
			entry.ActiveIdx = bestIdx
		}
	}

	if len(changes) > 0 {
		pe.save()
	}

	return changes
}

// Stats returns the current state of all prompt entries.
func (pe *PromptEvolver) Stats() map[string]PromptStats {
	pe.mu.RLock()
	defer pe.mu.RUnlock()

	stats := make(map[string]PromptStats)
	for name, entry := range pe.prompts {
		if len(entry.Variants) == 0 {
			continue
		}
		active := entry.Variants[entry.ActiveIdx]
		stats[name] = PromptStats{
			ActiveText:  truncateDesc(active.Text, 60),
			Attempts:    active.Attempts,
			AvgScore:    active.AvgScore,
			SuccessRate: active.SuccessRate(),
			VariantCount: len(entry.Variants),
		}
	}
	return stats
}

// PromptStats holds performance statistics for a named prompt.
type PromptStats struct {
	ActiveText   string
	Attempts     int
	AvgScore     float64
	SuccessRate  float64
	VariantCount int
}

func (pe *PromptEvolver) load() {
	if pe.path == "" {
		return
	}
	data, err := os.ReadFile(pe.path)
	if err != nil {
		return
	}
	var prompts map[string]*PromptEntry
	if err := json.Unmarshal(data, &prompts); err == nil && prompts != nil {
		pe.prompts = prompts
	}
}

func (pe *PromptEvolver) save() {
	if pe.path == "" {
		return
	}
	data, err := json.MarshalIndent(pe.prompts, "", "  ")
	if err != nil {
		return
	}
	safefile.WriteAtomic(pe.path, data, 0o644)
}

// --- Persistence ---

func (nd *NeuroplasticDescriptions) load() {
	if nd.storePath == "" {
		return
	}
	data, err := os.ReadFile(nd.storePath)
	if err != nil {
		return
	}
	var entries map[string]*DescEntry
	if err := json.Unmarshal(data, &entries); err == nil {
		nd.entries = entries
	}
}

func (nd *NeuroplasticDescriptions) save() {
	if nd.storePath == "" {
		return
	}
	data, err := json.MarshalIndent(nd.entries, "", "  ")
	if err != nil {
		return
	}
	safefile.WriteAtomic(nd.storePath, data, 0o644)
}
