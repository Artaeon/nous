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

// SelfDistiller learns from the model's own failures by creating contrastive
// training pairs. When a tool call fails, it records what the model tried
// (the negative example) and what actually worked (the positive example).
// Over time, this builds a dataset that teaches the model to avoid its
// specific failure patterns.
//
// Innovation: Standard fine-tuning uses only positive examples ("do this").
// Self-distillation uses contrastive pairs ("don't do X, do Y instead"),
// which is 3-5x more effective per training sample for correcting specific
// failure modes. The model learns from its OWN mistakes, not generic data.
//
// Example contrastive pair:
//   Failed:    {"tool": "grep", "args": {"query": "Pipeline"}}     ← wrong arg name
//   Corrected: {"tool": "grep", "args": {"pattern": "Pipeline"}}   ← right arg name
//
// The distiller also tracks failure patterns to identify systematic weaknesses:
//   - Wrong argument names (query vs pattern): 23 occurrences
//   - Missing required args: 15 occurrences
//   - Wrong tool selection (read vs grep): 8 occurrences
type SelfDistiller struct {
	pairs     []ContrastivePair
	patterns  map[string]*FailurePattern
	storePath string
	mu        sync.RWMutex
}

// ContrastivePair holds a failed attempt alongside its correction.
type ContrastivePair struct {
	// Context
	Query     string `json:"query"`      // what the user asked
	System    string `json:"system"`     // system prompt used
	Timestamp string `json:"timestamp"`

	// Negative example (what the model produced)
	FailedOutput string `json:"failed_output"`
	FailureType  string `json:"failure_type"` // "wrong_args", "wrong_tool", "invalid_json", "hallucination"

	// Positive example (what should have been produced)
	CorrectedOutput string `json:"corrected_output"`
	CorrectionSource string `json:"correction_source"` // "intent_compiler", "reflection_gate", "user_retry", "fallback"

	// Metadata
	ToolExpected string `json:"tool_expected,omitempty"`
	ToolGot      string `json:"tool_got,omitempty"`
}

// FailurePattern tracks a recurring failure mode.
type FailurePattern struct {
	Type        string    `json:"type"`
	Description string    `json:"description"`
	Count       int       `json:"count"`
	Examples    []string  `json:"examples"` // up to 5 example queries
	FirstSeen   time.Time `json:"first_seen"`
	LastSeen    time.Time `json:"last_seen"`
}

// DistillStats holds distillation statistics.
type DistillStats struct {
	TotalPairs     int
	ByFailureType  map[string]int
	TopPatterns    []PatternSummary
	CorrectionRate float64 // what % of failures had corrections
}

// PatternSummary is a summary of a failure pattern for display.
type PatternSummary struct {
	Type        string
	Description string
	Count       int
}

// NewSelfDistiller creates a new self-distillation system.
func NewSelfDistiller(storePath string) *SelfDistiller {
	sd := &SelfDistiller{
		patterns:  make(map[string]*FailurePattern),
		storePath: storePath,
	}
	sd.load()
	return sd
}

// RecordFailure records a tool call failure with its correction.
func (sd *SelfDistiller) RecordFailure(query, system, failedOutput, failureType, correctedOutput, correctionSource string) {
	sd.mu.Lock()
	defer sd.mu.Unlock()

	pair := ContrastivePair{
		Query:            query,
		System:           truncateForStorage(system, 500),
		Timestamp:        time.Now().Format(time.RFC3339),
		FailedOutput:     failedOutput,
		FailureType:      failureType,
		CorrectedOutput:  correctedOutput,
		CorrectionSource: correctionSource,
	}

	sd.pairs = append(sd.pairs, pair)
	sd.trackPattern(failureType, failedOutput, query)
	sd.save()
}

// RecordToolMismatch records when the model picked the wrong tool.
func (sd *SelfDistiller) RecordToolMismatch(query, system, expected, got, correctedOutput string) {
	sd.mu.Lock()
	defer sd.mu.Unlock()

	pair := ContrastivePair{
		Query:            query,
		System:           truncateForStorage(system, 500),
		Timestamp:        time.Now().Format(time.RFC3339),
		FailedOutput:     fmt.Sprintf(`{"tool":"%s"}`, got),
		FailureType:      "wrong_tool",
		CorrectedOutput:  correctedOutput,
		CorrectionSource: "intent_compiler",
		ToolExpected:     expected,
		ToolGot:          got,
	}

	sd.pairs = append(sd.pairs, pair)
	sd.trackPattern("wrong_tool", fmt.Sprintf("%s→%s", got, expected), query)
	sd.save()
}

// RecordArgError records when the model used wrong argument names or values.
func (sd *SelfDistiller) RecordArgError(query, tool, failedArgs, correctedArgs string) {
	sd.mu.Lock()
	defer sd.mu.Unlock()

	pair := ContrastivePair{
		Query:            query,
		Timestamp:        time.Now().Format(time.RFC3339),
		FailedOutput:     fmt.Sprintf(`{"tool":"%s","args":%s}`, tool, failedArgs),
		FailureType:      "wrong_args",
		CorrectedOutput:  fmt.Sprintf(`{"tool":"%s","args":%s}`, tool, correctedArgs),
		CorrectionSource: "intent_compiler",
		ToolExpected:     tool,
		ToolGot:          tool,
	}

	sd.pairs = append(sd.pairs, pair)
	sd.trackPattern("wrong_args", failedArgs+"→"+correctedArgs, query)
	sd.save()
}

// trackPattern updates failure pattern tracking.
// Caller must hold sd.mu.
func (sd *SelfDistiller) trackPattern(failureType, detail, query string) {
	key := failureType + ":" + normalizePatternKey(detail)

	p, exists := sd.patterns[key]
	if !exists {
		p = &FailurePattern{
			Type:      failureType,
			FirstSeen: time.Now(),
		}
		sd.patterns[key] = p
	}

	p.Count++
	p.LastSeen = time.Now()
	p.Description = detail

	// Keep up to 5 example queries
	if len(p.Examples) < 5 {
		p.Examples = append(p.Examples, truncateForStorage(query, 100))
	}
}

// normalizePatternKey creates a stable key from failure details.
func normalizePatternKey(detail string) string {
	// Keep first 50 chars, lowercase
	key := strings.ToLower(detail)
	if len(key) > 50 {
		key = key[:50]
	}
	return key
}

// Stats returns distillation statistics.
func (sd *SelfDistiller) Stats() DistillStats {
	sd.mu.RLock()
	defer sd.mu.RUnlock()

	stats := DistillStats{
		TotalPairs:    len(sd.pairs),
		ByFailureType: make(map[string]int),
	}

	corrected := 0
	for _, p := range sd.pairs {
		stats.ByFailureType[p.FailureType]++
		if p.CorrectedOutput != "" {
			corrected++
		}
	}

	if stats.TotalPairs > 0 {
		stats.CorrectionRate = float64(corrected) / float64(stats.TotalPairs)
	}

	// Top patterns by count
	for _, p := range sd.patterns {
		stats.TopPatterns = append(stats.TopPatterns, PatternSummary{
			Type:        p.Type,
			Description: p.Description,
			Count:       p.Count,
		})
	}

	// Sort by count descending (simple insertion sort for small lists)
	for i := 1; i < len(stats.TopPatterns); i++ {
		for j := i; j > 0 && stats.TopPatterns[j].Count > stats.TopPatterns[j-1].Count; j-- {
			stats.TopPatterns[j], stats.TopPatterns[j-1] = stats.TopPatterns[j-1], stats.TopPatterns[j]
		}
	}

	// Keep top 10
	if len(stats.TopPatterns) > 10 {
		stats.TopPatterns = stats.TopPatterns[:10]
	}

	return stats
}

// ExportContrastive exports contrastive training pairs in ChatML format.
// Each pair has the failed output as a negative and corrected as positive.
// Format designed for DPO (Direct Preference Optimization) fine-tuning.
func (sd *SelfDistiller) ExportContrastive(outputPath string) error {
	sd.mu.RLock()
	pairs := make([]ContrastivePair, len(sd.pairs))
	copy(pairs, sd.pairs)
	sd.mu.RUnlock()

	return safefile.WriteAtomicFunc(outputPath, 0o644, func(f *os.File) error {
		encoder := json.NewEncoder(f)
		for _, pair := range pairs {
			if pair.CorrectedOutput == "" {
				continue // skip pairs without corrections
			}

			// DPO format: chosen (correct) vs rejected (failed)
			entry := map[string]any{
				"prompt": pair.Query,
				"chosen": []map[string]string{
					{"role": "user", "content": pair.Query},
					{"role": "assistant", "content": pair.CorrectedOutput},
				},
				"rejected": []map[string]string{
					{"role": "user", "content": pair.Query},
					{"role": "assistant", "content": pair.FailedOutput},
				},
			}
			if err := encoder.Encode(entry); err != nil {
				return err
			}
		}
		return nil
	})
}

// ExportNegativeInstructions generates system prompt additions based on
// the most common failure patterns. These are "negative instructions" that
// tell the model what NOT to do.
func (sd *SelfDistiller) ExportNegativeInstructions() string {
	sd.mu.RLock()
	defer sd.mu.RUnlock()

	if len(sd.patterns) == 0 {
		return ""
	}

	// Collect patterns with count >= 3 (recurring issues)
	var significant []FailurePattern
	for _, p := range sd.patterns {
		if p.Count >= 3 {
			significant = append(significant, *p)
		}
	}

	if len(significant) == 0 {
		return ""
	}

	// Sort by count descending
	for i := 1; i < len(significant); i++ {
		for j := i; j > 0 && significant[j].Count > significant[j-1].Count; j-- {
			significant[j], significant[j-1] = significant[j-1], significant[j]
		}
	}

	// Cap at 5 to avoid prompt bloat
	if len(significant) > 5 {
		significant = significant[:5]
	}

	var b strings.Builder
	b.WriteString("IMPORTANT — avoid these common mistakes:\n")
	for i, p := range significant {
		b.WriteString(fmt.Sprintf("%d. ", i+1))
		switch p.Type {
		case "wrong_args":
			b.WriteString(fmt.Sprintf("Use correct argument names: %s\n", p.Description))
		case "wrong_tool":
			b.WriteString(fmt.Sprintf("Use the right tool: %s\n", p.Description))
		case "invalid_json":
			b.WriteString("Always output valid JSON for tool calls\n")
		case "hallucination":
			b.WriteString(fmt.Sprintf("Do not hallucinate: %s\n", p.Description))
		default:
			b.WriteString(fmt.Sprintf("Avoid: %s\n", p.Description))
		}
	}

	return b.String()
}

// Size returns the number of contrastive pairs collected.
func (sd *SelfDistiller) Size() int {
	sd.mu.RLock()
	defer sd.mu.RUnlock()
	return len(sd.pairs)
}

// PatternCount returns the number of distinct failure patterns tracked.
func (sd *SelfDistiller) PatternCount() int {
	sd.mu.RLock()
	defer sd.mu.RUnlock()
	return len(sd.patterns)
}

// --- Persistence ---

type distillData struct {
	Pairs    []ContrastivePair         `json:"pairs"`
	Patterns map[string]*FailurePattern `json:"patterns"`
}

func (sd *SelfDistiller) save() {
	if sd.storePath == "" {
		return
	}
	data, err := json.MarshalIndent(distillData{
		Pairs:    sd.pairs,
		Patterns: sd.patterns,
	}, "", "  ")
	if err != nil {
		return
	}
	safefile.WriteAtomic(sd.storePath, data, 0o644)
}

func (sd *SelfDistiller) load() {
	if sd.storePath == "" {
		return
	}
	data, err := os.ReadFile(sd.storePath)
	if err != nil {
		return
	}
	var loaded distillData
	if err := json.Unmarshal(data, &loaded); err != nil {
		return
	}
	sd.pairs = loaded.Pairs
	if loaded.Patterns != nil {
		sd.patterns = loaded.Patterns
	}
}

// truncateForStorage truncates a string for storage efficiency.
func truncateForStorage(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen-3] + "..."
}
