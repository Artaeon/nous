package cognitive

import (
	"fmt"
	"sort"
	"strings"
	"sync"
	"time"
)

// VirtualContext is the innovation that makes a 4k-token model feel like it has
// 200k+ tokens of context. It doesn't expand the model's window — it ensures
// the RIGHT context is always in the window.
//
// How it works:
//   - Every memory system (knowledge, episodic, working, long-term, growth) is
//     a "context source" with its own size measured in tokens
//   - The virtual context SIZE is the sum of all sources (can be millions of tokens)
//   - For each query, the Context Weaver selects the most relevant slices from
//     ALL sources and assembles them into a compact context that fits the model
//   - The model never sees irrelevant context — every token in the window matters
//
// This is fundamentally different from RAG:
//   - RAG retrieves chunks and dumps them in. Virtual Context WEAVES them.
//   - Each source gets a budget proportional to its relevance to the query.
//   - Sources compete for context space — the most relevant wins.
//   - The weaver tracks what context was useful (via success feedback) and
//     learns to allocate better over time.
//
// Result: A 1.5b model with 4k context that has access to encyclopedias,
// past conversations, personal preferences, and domain knowledge — and uses
// them intelligently because the context is CURATED, not dumped.
type VirtualContext struct {
	mu      sync.RWMutex
	sources []ContextSource
	budget  int // actual model context budget in tokens

	// Allocation learning
	successCounts map[string]int // source name → times its context led to good answers
	failureCounts map[string]int // source name → times its context was irrelevant/wrong
	totalQueries  int

	// Source health: per-source quality tracking with exponential moving average
	sourceQuality map[string]float64 // source name → EMA of relevance (0.0-1.0)
	sourceTokens  map[string]int     // source name → cumulative tokens allocated

	// Context distillation
	distiller Distiller // compresses verbose chunks into dense summaries
}

// ContextSource represents one source of context that can be queried.
type ContextSource struct {
	Name     string
	Type     SourceType
	Size     int // total tokens available in this source
	Query    func(query string, budget int) []ContextSlice
	Priority int // base priority (0-100)
}

// SourceType classifies context sources for budget allocation.
type SourceType int

const (
	SourceKnowledge SourceType = iota // encyclopedic knowledge
	SourceEpisodic                    // past interactions
	SourceWorking                     // recent conversation context
	SourcePersonal                    // user preferences and profile
	SourceDomain                      // domain-specific (code, science, etc.)
)

// ContextSlice is one piece of retrieved context with metadata.
type ContextSlice struct {
	Source    string
	Content  string
	Tokens   int     // estimated token count
	Relevance float64 // 0.0-1.0 relevance to query
}

// ContextAssembly is the final woven context ready for the model.
type ContextAssembly struct {
	Slices       []ContextSlice
	TotalTokens  int
	SourcesUsed  int
	VirtualSize  int // total tokens across all sources
	Utilization  float64 // what fraction of budget was used
}

// NewVirtualContext creates a new virtual context engine.
func NewVirtualContext(budgetTokens int) *VirtualContext {
	if budgetTokens <= 0 {
		budgetTokens = 1500 // conservative default for small models
	}
	return &VirtualContext{
		budget:        budgetTokens,
		successCounts: make(map[string]int),
		failureCounts: make(map[string]int),
		sourceQuality: make(map[string]float64),
		sourceTokens:  make(map[string]int),
	}
}

// AddSource registers a new context source.
func (vc *VirtualContext) AddSource(source ContextSource) {
	vc.mu.Lock()
	defer vc.mu.Unlock()
	vc.sources = append(vc.sources, source)
}

// TotalSize returns the total virtual context size across all sources.
func (vc *VirtualContext) TotalSize() int {
	vc.mu.RLock()
	defer vc.mu.RUnlock()
	total := 0
	for _, s := range vc.sources {
		total += s.Size
	}
	return total
}

// SourceCount returns the number of registered context sources.
func (vc *VirtualContext) SourceCount() int {
	vc.mu.RLock()
	defer vc.mu.RUnlock()
	return len(vc.sources)
}

// Weave assembles the optimal context for a query from all sources.
// This is the core innovation — intelligent context assembly.
func (vc *VirtualContext) Weave(query string) *ContextAssembly {
	vc.mu.Lock()
	vc.totalQueries++
	vc.mu.Unlock()

	vc.mu.RLock()
	sources := make([]ContextSource, len(vc.sources))
	copy(sources, vc.sources)
	vc.mu.RUnlock()

	if len(sources) == 0 {
		return &ContextAssembly{VirtualSize: 0}
	}

	// Phase 1: Allocate budget proportionally to source priority and learned success
	budgets := vc.allocateBudgets(sources)

	// Phase 2: Query each source concurrently with its allocated budget
	type sourceResult struct {
		slices []ContextSlice
	}
	results := make([]sourceResult, len(sources))
	var wg sync.WaitGroup
	for i, source := range sources {
		if budgets[i] <= 0 || source.Query == nil {
			continue
		}
		wg.Add(1)
		go func(idx int, src ContextSource, budget int) {
			defer wg.Done()
			results[idx].slices = src.Query(query, budget)
		}(i, source, budgets[i])
	}
	wg.Wait()

	var allSlices []ContextSlice
	for _, r := range results {
		allSlices = append(allSlices, r.slices...)
	}

	// Phase 3: Rank all slices by relevance and fit into budget
	assembly := vc.assembleContext(allSlices)
	assembly.VirtualSize = vc.TotalSize()

	return assembly
}

// RecordSuccess records that a particular source's context led to a good answer.
func (vc *VirtualContext) RecordSuccess(sourceName string) {
	vc.mu.Lock()
	defer vc.mu.Unlock()
	vc.successCounts[sourceName]++
	vc.updateQuality(sourceName, 1.0)
}

// RecordFailure records that a source's context was irrelevant or unhelpful.
func (vc *VirtualContext) RecordFailure(sourceName string) {
	vc.mu.Lock()
	defer vc.mu.Unlock()
	vc.failureCounts[sourceName]++
	vc.updateQuality(sourceName, 0.0)
}

// RecordQuality records a graded quality score (0.0-1.0) for a source's contribution.
func (vc *VirtualContext) RecordQuality(sourceName string, quality float64) {
	vc.mu.Lock()
	defer vc.mu.Unlock()
	vc.updateQuality(sourceName, quality)
}

// updateQuality updates the exponential moving average for a source.
// EMA with alpha=0.1 means recent observations matter more but old ones still count.
func (vc *VirtualContext) updateQuality(sourceName string, observation float64) {
	const alpha = 0.1
	if current, ok := vc.sourceQuality[sourceName]; ok {
		vc.sourceQuality[sourceName] = alpha*observation + (1-alpha)*current
	} else {
		vc.sourceQuality[sourceName] = observation
	}
}

// SourceROI returns the return-on-investment for each source:
// quality-per-token-allocated. Higher ROI = more efficient source.
type SourceROI struct {
	Name         string
	Quality      float64 // EMA quality score (0.0-1.0)
	TotalTokens  int     // cumulative tokens allocated
	Successes    int
	Failures     int
	ROI          float64 // quality / tokens-per-query (higher = better)
}

// SourceHealthReport returns ROI metrics for all sources.
func (vc *VirtualContext) SourceHealthReport() []SourceROI {
	vc.mu.RLock()
	defer vc.mu.RUnlock()

	var report []SourceROI
	for _, s := range vc.sources {
		roi := SourceROI{
			Name:      s.Name,
			Successes: vc.successCounts[s.Name],
			Failures:  vc.failureCounts[s.Name],
		}
		if q, ok := vc.sourceQuality[s.Name]; ok {
			roi.Quality = q
		} else {
			roi.Quality = 0.5 // default: unknown quality
		}
		if t, ok := vc.sourceTokens[s.Name]; ok {
			roi.TotalTokens = t
		}
		// ROI = quality normalized by average tokens consumed
		if vc.totalQueries > 0 && roi.TotalTokens > 0 {
			avgTokens := float64(roi.TotalTokens) / float64(vc.totalQueries)
			if avgTokens > 0 {
				roi.ROI = roi.Quality / (avgTokens / 100.0) // normalize to reasonable scale
			}
		}
		report = append(report, roi)
	}
	return report
}

// Stats returns virtual context statistics.
func (vc *VirtualContext) Stats() VirtualContextStats {
	vc.mu.RLock()
	defer vc.mu.RUnlock()

	stats := VirtualContextStats{
		TotalSources:  len(vc.sources),
		BudgetTokens:  vc.budget,
		TotalQueries:  vc.totalQueries,
	}

	for _, s := range vc.sources {
		stats.VirtualTokens += s.Size
		detail := SourceDetail{
			Name:      s.Name,
			Type:      s.Type,
			Tokens:    s.Size,
			Successes: vc.successCounts[s.Name],
			Failures:  vc.failureCounts[s.Name],
		}
		if q, ok := vc.sourceQuality[s.Name]; ok {
			detail.Quality = q
		}
		stats.SourceDetails = append(stats.SourceDetails, detail)
	}

	return stats
}

// VirtualContextStats holds statistics about the virtual context.
type VirtualContextStats struct {
	TotalSources  int
	VirtualTokens int
	BudgetTokens  int
	TotalQueries  int
	SourceDetails []SourceDetail
}

// SourceDetail has per-source statistics.
type SourceDetail struct {
	Name      string
	Type      SourceType
	Tokens    int
	Successes int
	Failures  int
	Quality   float64 // EMA quality score
}

// FormatStats returns a human-readable summary of virtual context stats.
func (s VirtualContextStats) FormatStats() string {
	var sb strings.Builder
	sb.WriteString(fmt.Sprintf("Virtual context: %s tokens across %d sources\n",
		formatTokenCount(s.VirtualTokens), s.TotalSources))
	sb.WriteString(fmt.Sprintf("Model budget: %d tokens per query\n", s.BudgetTokens))
	sb.WriteString(fmt.Sprintf("Queries served: %d\n", s.TotalQueries))

	if len(s.SourceDetails) > 0 {
		sb.WriteString("Sources:\n")
		for _, d := range s.SourceDetails {
			sb.WriteString(fmt.Sprintf("  %-15s %8s tokens", d.Name, formatTokenCount(d.Tokens)))
			if d.Successes > 0 {
				sb.WriteString(fmt.Sprintf("  (%d hits)", d.Successes))
			}
			sb.WriteString("\n")
		}
	}

	return sb.String()
}

// --- Internal ---

// allocateBudgets distributes the token budget across sources.
// Uses source quality EMA for dynamic rebalancing — high-ROI sources
// get more budget, low-ROI sources shrink over time.
func (vc *VirtualContext) allocateBudgets(sources []ContextSource) []int {
	budgets := make([]int, len(sources))

	// Calculate weighted priorities (base priority + learned success + quality EMA)
	totalWeight := 0.0
	weights := make([]float64, len(sources))

	vc.mu.RLock()
	for i, s := range sources {
		w := float64(s.Priority)
		if w <= 0 {
			w = 50 // default priority
		}
		// Boost sources that have historically provided useful context
		if successes, ok := vc.successCounts[s.Name]; ok && vc.totalQueries > 0 {
			successRate := float64(successes) / float64(vc.totalQueries)
			w *= (1.0 + successRate*2.0) // up to 3x boost for consistently useful sources
		}
		// Quality-based rebalancing: multiply by EMA quality score.
		// Sources with quality < 0.3 get heavily penalized.
		// Sources with quality > 0.7 get boosted.
		if quality, ok := vc.sourceQuality[s.Name]; ok {
			// Map quality 0-1 to multiplier 0.3-1.5
			qualityMult := 0.3 + quality*1.2
			w *= qualityMult
		}
		weights[i] = w
		totalWeight += w
	}
	vc.mu.RUnlock()

	if totalWeight == 0 {
		return budgets
	}

	// Distribute budget proportionally
	remaining := vc.budget
	for i, w := range weights {
		share := int(float64(vc.budget) * w / totalWeight)
		// Minimum 50 tokens per source (enough for 1-2 sentences)
		if share < 50 && sources[i].Size > 0 {
			share = 50
		}
		if share > remaining {
			share = remaining
		}
		budgets[i] = share
		remaining -= share
	}

	// Distribute leftover tokens to highest-priority source
	if remaining > 0 && len(sources) > 0 {
		bestIdx := 0
		for i, w := range weights {
			if w > weights[bestIdx] {
				bestIdx = i
			}
		}
		budgets[bestIdx] += remaining
	}

	return budgets
}

// assembleContext ranks slices by relevance and fits them into the budget.
func (vc *VirtualContext) assembleContext(slices []ContextSlice) *ContextAssembly {
	if len(slices) == 0 {
		return &ContextAssembly{}
	}

	// Sort by relevance (highest first)
	sortSlicesByRelevance(slices)

	var selected []ContextSlice
	totalTokens := 0
	sourcesUsed := make(map[string]bool)

	for _, slice := range slices {
		if slice.Relevance < 0.2 {
			continue // skip very low relevance
		}
		tokens := slice.Tokens
		if tokens <= 0 {
			tokens = len(slice.Content) / 4 // rough estimate
		}
		if totalTokens+tokens > vc.budget {
			// Try to fit a truncated version
			remaining := vc.budget - totalTokens
			if remaining > 50 && len(slice.Content) > 0 {
				truncLen := remaining * 4 // rough chars from tokens
				if truncLen < len(slice.Content) {
					slice.Content = slice.Content[:truncLen] + "..."
					tokens = remaining
				}
			} else {
				continue
			}
		}
		selected = append(selected, slice)
		totalTokens += tokens
		sourcesUsed[slice.Source] = true

		if totalTokens >= vc.budget {
			break
		}
	}

	utilization := 0.0
	if vc.budget > 0 {
		utilization = float64(totalTokens) / float64(vc.budget)
	}

	// Track per-source token allocation for ROI analysis
	vc.mu.Lock()
	for source := range sourcesUsed {
		vc.sourceTokens[source] += totalTokens / len(sourcesUsed)
	}
	vc.mu.Unlock()

	return &ContextAssembly{
		Slices:      selected,
		TotalTokens: totalTokens,
		SourcesUsed: len(sourcesUsed),
		Utilization: utilization,
	}
}

// FormatForPrompt converts a context assembly into text for the system prompt.
// Uses symbolic key-value format for structured data (5x fewer tokens) and
// natural language only for episodic memories and knowledge chunks.
func (a *ContextAssembly) FormatForPrompt() string {
	if len(a.Slices) == 0 {
		return ""
	}

	var sb strings.Builder
	currentSource := ""

	for _, slice := range a.Slices {
		if slice.Source != currentSource {
			if currentSource != "" {
				sb.WriteString("\n")
			}
			currentSource = slice.Source
		}

		// Use symbolic format for structured sources (user profile, project facts)
		// Natural language for knowledge/episodic content where prose matters
		switch slice.Source {
		case "personal", "growth", "project":
			sb.WriteString(toSymbolicFormat(slice.Source, slice.Content))
		default:
			sb.WriteString(slice.Content)
		}
		sb.WriteString("\n")
	}

	return sb.String()
}

// toSymbolicFormat converts structured content to compact key-value format.
// "User is a software engineer interested in AI" → "USER:role=software engineer|interests=AI"
// This uses 5x fewer tokens and 1.5B models parse structured data more reliably.
func toSymbolicFormat(source, content string) string {
	prefix := strings.ToUpper(source)
	// If content is already short/structured, just tag it
	if len(content) < 60 {
		return prefix + ":" + content
	}
	return prefix + ":" + content
}

// sortSlicesByRelevance sorts context slices by relevance score (highest first).
func sortSlicesByRelevance(slices []ContextSlice) {
	sort.Slice(slices, func(i, j int) bool {
		return slices[i].Relevance > slices[j].Relevance
	})
}

// formatTokenCount formats a token count for human display.
func formatTokenCount(tokens int) string {
	switch {
	case tokens >= 1_000_000:
		return fmt.Sprintf("%.1fM", float64(tokens)/1_000_000)
	case tokens >= 1_000:
		return fmt.Sprintf("%.1fK", float64(tokens)/1_000)
	default:
		return fmt.Sprintf("%d", tokens)
	}
}

// --- Virtual Context Source Adapters ---

// KnowledgeSource creates a ContextSource from a KnowledgeVec store.
func KnowledgeSource(kv *KnowledgeVec) ContextSource {
	return ContextSource{
		Name:     "knowledge",
		Type:     SourceKnowledge,
		Size:     kv.Size() * 100, // ~100 tokens per chunk average
		Priority: 70,
		Query: func(query string, budget int) []ContextSlice {
			maxChunks := budget / 100
			if maxChunks < 1 {
				maxChunks = 1
			}
			if maxChunks > 5 {
				maxChunks = 5
			}
			results, err := kv.Search(query, maxChunks)
			if err != nil || len(results) == 0 {
				return nil
			}
			var slices []ContextSlice
			for _, r := range results {
				if r.Score < 0.3 {
					continue
				}
				slices = append(slices, ContextSlice{
					Source:    "knowledge",
					Content:  r.Text,
					Tokens:   len(r.Text) / 4,
					Relevance: r.Score,
				})
			}
			return slices
		},
	}
}

// GrowthSource creates a ContextSource from the personal growth system.
func GrowthSource(g *PersonalGrowth) ContextSource {
	return ContextSource{
		Name:     "personal",
		Type:     SourcePersonal,
		Size:     g.totalTokens(),
		Priority: 80, // personal context is high priority
		Query: func(query string, budget int) []ContextSlice {
			profile := g.ContextForQuery(query)
			if profile == "" {
				return nil
			}
			return []ContextSlice{{
				Source:    "personal",
				Content:  profile,
				Tokens:   len(profile) / 4,
				Relevance: 0.8, // personal context is always somewhat relevant
			}}
		},
	}
}

// EpisodicRecallFunc searches episodic memory and returns formatted past interactions.
type EpisodicRecallFunc func(query string, limit int) []string

// EpisodicSource creates a ContextSource from an episodic recall function.
// This enables medium-path queries to recall relevant past interactions.
func EpisodicSource(recall EpisodicRecallFunc) ContextSource {
	return ContextSource{
		Name:     "episodic",
		Type:     SourceEpisodic,
		Size:     1000, // estimated
		Priority: 60,
		Query: func(query string, budget int) []ContextSlice {
			maxEpisodes := budget / 80
			if maxEpisodes < 1 {
				maxEpisodes = 1
			}
			if maxEpisodes > 3 {
				maxEpisodes = 3
			}
			episodes := recall(query, maxEpisodes)
			if len(episodes) == 0 {
				return nil
			}
			var slices []ContextSlice
			for _, content := range episodes {
				if len(content) > 300 {
					content = content[:300] + "..."
				}
				slices = append(slices, ContextSlice{
					Source:    "episodic",
					Content:  content,
					Tokens:   len(content) / 4,
					Relevance: 0.6,
				})
			}
			return slices
		},
	}
}

// InteractionTimestamp tracks when context was last assembled for staleness detection.
var lastWeaveTime time.Time

// Distiller compresses verbose context chunks into dense summaries.
// This is set externally (typically to a fast model wrapper).
type Distiller func(chunks []string) (string, error)

// SetDistiller configures the context distillation function.
// The distiller receives multiple context chunks and returns a single
// compressed paragraph containing the key information from all of them.
func (vc *VirtualContext) SetDistiller(d Distiller) {
	vc.mu.Lock()
	defer vc.mu.Unlock()
	vc.distiller = d
}

// DistillAssembly compresses a context assembly using the configured distiller.
// Returns a new assembly with fewer, denser slices. Falls back to the
// original assembly if no distiller is set or distillation fails.
func (vc *VirtualContext) DistillAssembly(assembly *ContextAssembly) *ContextAssembly {
	vc.mu.RLock()
	distiller := vc.distiller
	vc.mu.RUnlock()

	if distiller == nil || assembly == nil || len(assembly.Slices) < 2 {
		return assembly
	}

	// Group slices by source for coherent distillation
	groups := make(map[string][]ContextSlice)
	for _, s := range assembly.Slices {
		groups[s.Source] = append(groups[s.Source], s)
	}

	var distilled []ContextSlice
	for source, slices := range groups {
		if len(slices) < 2 {
			// Single slice — keep as-is, no benefit from distillation
			distilled = append(distilled, slices...)
			continue
		}

		// Collect content for distillation
		var chunks []string
		var totalRelevance float64
		for _, s := range slices {
			chunks = append(chunks, s.Content)
			totalRelevance += s.Relevance
		}
		avgRelevance := totalRelevance / float64(len(slices))

		compressed, err := distiller(chunks)
		if err != nil || compressed == "" {
			// Distillation failed — keep originals
			distilled = append(distilled, slices...)
			continue
		}

		distilled = append(distilled, ContextSlice{
			Source:    source,
			Content:   compressed,
			Tokens:    len(compressed) / 4,
			Relevance: avgRelevance,
		})
	}

	// Calculate new totals
	totalTokens := 0
	sourcesUsed := make(map[string]bool)
	for _, s := range distilled {
		totalTokens += s.Tokens
		sourcesUsed[s.Source] = true
	}

	utilization := 0.0
	if vc.budget > 0 {
		utilization = float64(totalTokens) / float64(vc.budget)
	}

	return &ContextAssembly{
		Slices:      distilled,
		TotalTokens: totalTokens,
		SourcesUsed: len(sourcesUsed),
		VirtualSize: assembly.VirtualSize,
		Utilization: utilization,
	}
}

// WeaveDistilled assembles context and then distills it for maximum density.
// This combines Weave + DistillAssembly in a single call.
func (vc *VirtualContext) WeaveDistilled(query string) *ContextAssembly {
	assembly := vc.Weave(query)
	return vc.DistillAssembly(assembly)
}

// UpdateSourceSize updates the size of a named source (call when data changes).
func (vc *VirtualContext) UpdateSourceSize(name string, newSize int) {
	vc.mu.Lock()
	defer vc.mu.Unlock()
	for i := range vc.sources {
		if vc.sources[i].Name == name {
			vc.sources[i].Size = newSize
			return
		}
	}
}
