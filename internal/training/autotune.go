package training

import (
	"fmt"
	"strings"
	"sync"
	"time"
)

// ModelCreator is the interface for creating Ollama models from Modelfiles.
type ModelCreator interface {
	CreateModel(name, modelfile string) error
}

// AutoTuner monitors training data accumulation and triggers fine-tuning
// when enough high-quality pairs are available.
type AutoTuner struct {
	collector    *Collector
	modelName    string          // base model (e.g. "qwen2.5:1.5b")
	tunedName    string          // fine-tuned model name (e.g. "nous-qwen2.5:1.5b")
	minPairs     int             // minimum pairs before considering tuning (default: 50)
	qualityFloor float64         // minimum average quality (default: 0.7)
	lastTuneAt   time.Time       // when we last triggered tuning
	lastAttemptAt time.Time      // when we last attempted tuning (success or failure)
	cooldown     time.Duration   // minimum time between tuning attempts (default: 1 hour)
	onTune       func(msg string) // callback for status updates
	creator      ModelCreator    // Ollama client for creating models

	// A/B testing: compare base vs tuned model performance
	abEnabled     bool
	abBaseWins    int
	abTunedWins   int
	abTotalTrials int

	// Adaptive cooldown: lengthen cooldown on failures, shorten on success
	adaptiveCooldown bool
	consecutiveFails int
	baseCooldown     time.Duration // original cooldown value

	mu           sync.Mutex
}

// AutoTuneStats holds current auto-tuning status.
type AutoTuneStats struct {
	PairCount        int
	AvgQuality       float64
	MinPairs         int
	QualityFloor     float64
	LastTuneAt       time.Time
	NextTuneAfter    time.Time
	TunedName        string
	Ready            bool // whether conditions are met
	ABBaseWins       int
	ABTunedWins      int
	ABTotalTrials    int
	ConsecutiveFails int
	EffectiveCooldown time.Duration
}

// NewAutoTuner creates an AutoTuner that monitors the collector and triggers
// Modelfile-based tuning via Ollama's CreateModel API.
func NewAutoTuner(collector *Collector, modelName string) *AutoTuner {
	// Derive tuned model name: "qwen2.5:1.5b" -> "nous-qwen2.5:1.5b"
	tunedName := "nous-" + modelName
	if strings.Contains(modelName, "/") {
		// Handle org/model format
		parts := strings.SplitN(modelName, "/", 2)
		tunedName = "nous-" + parts[len(parts)-1]
	}

	return &AutoTuner{
		collector:    collector,
		modelName:    modelName,
		tunedName:    tunedName,
		minPairs:     50,
		qualityFloor: 0.7,
		cooldown:     1 * time.Hour,
		onTune:       func(string) {}, // no-op default
	}
}

// WithMinPairs sets the minimum number of pairs before tuning triggers.
func (a *AutoTuner) WithMinPairs(n int) *AutoTuner {
	a.minPairs = n
	return a
}

// WithCooldown sets the minimum time between tuning attempts.
func (a *AutoTuner) WithCooldown(d time.Duration) *AutoTuner {
	a.cooldown = d
	return a
}

// WithCallback sets the status update callback.
func (a *AutoTuner) WithCallback(fn func(string)) *AutoTuner {
	a.onTune = fn
	return a
}

// WithCreator sets the Ollama model creator used to build the tuned model.
func (a *AutoTuner) WithCreator(c ModelCreator) *AutoTuner {
	a.creator = c
	return a
}

// WithABTesting enables A/B testing between base and tuned models.
func (a *AutoTuner) WithABTesting(enabled bool) *AutoTuner {
	a.abEnabled = enabled
	return a
}

// WithAdaptiveCooldown enables adaptive cooldown that increases on failures
// and decreases on success.
func (a *AutoTuner) WithAdaptiveCooldown(enabled bool) *AutoTuner {
	a.adaptiveCooldown = enabled
	a.baseCooldown = a.cooldown
	return a
}

// Check evaluates whether auto-tuning should trigger.
// Returns true if tuning was triggered.
// Call this after each interaction (from main.go's REPL loop).
func (a *AutoTuner) Check() bool {
	return a.check(false)
}

// CheckQuiet evaluates whether auto-tuning should trigger, but suppresses
// routine status chatter. This is intended for background checks during normal
// conversations so the assistant UI stays clean.
func (a *AutoTuner) CheckQuiet() bool {
	return a.check(true)
}

func (a *AutoTuner) check(quiet bool) bool {
	a.mu.Lock()
	defer a.mu.Unlock()

	if !a.shouldTuneLocked() {
		return false
	}

	if a.creator == nil {
		a.lastAttemptAt = time.Now()
		if !quiet {
			a.onTune("auto-tune skipped: no Ollama client configured")
		}
		return false
	}

	a.lastAttemptAt = time.Now()
	if !quiet {
		a.onTune("Starting auto fine-tune...")
	}

	// Build a system prompt that embeds learned patterns from training data
	systemPrompt := a.buildEnhancedSystemPrompt()

	// Generate the Modelfile
	cfg := DefaultModelfileConfig(a.modelName)
	cfg.Name = a.tunedName
	cfg.System = systemPrompt
	modelfile := GenerateModelfile(cfg)

	// Create the model via Ollama API
	if err := a.creator.CreateModel(a.tunedName, modelfile); err != nil {
		if !quiet {
			a.onTune(fmt.Sprintf("auto-tune failed: %v", err))
		}
		// Adaptive cooldown: increase cooldown on failure
		if a.adaptiveCooldown {
			a.consecutiveFails++
			a.cooldown = a.adaptiveCooldownValue()
		}
		return false
	}

	a.lastTuneAt = time.Now()
	// Adaptive cooldown: reset on success
	if a.adaptiveCooldown {
		a.consecutiveFails = 0
		a.cooldown = a.adaptiveCooldownValue()
	}
	a.onTune(fmt.Sprintf("Fine-tune complete: %s (%d pairs, avg quality %.2f)",
		a.tunedName, a.collector.Size(), a.collector.AverageQuality()))

	return true
}

// ShouldTune returns true if conditions are met for fine-tuning.
func (a *AutoTuner) ShouldTune() bool {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.shouldTuneLocked()
}

// shouldTuneLocked checks conditions without locking (caller must hold mu).
func (a *AutoTuner) shouldTuneLocked() bool {
	// Check cooldown
	lastAttempt := a.lastAttemptAt
	if lastAttempt.IsZero() || a.lastTuneAt.After(lastAttempt) {
		lastAttempt = a.lastTuneAt
	}
	if !lastAttempt.IsZero() && time.Since(lastAttempt) < a.cooldown {
		return false
	}

	// Check minimum pairs
	if a.collector.Size() < a.minPairs {
		return false
	}

	// Check average quality
	if a.collector.AverageQuality() < a.qualityFloor {
		return false
	}

	return true
}

// TunedModelName returns the name of the fine-tuned model.
func (a *AutoTuner) TunedModelName() string {
	return a.tunedName
}

// Stats returns current auto-tuning status.
func (a *AutoTuner) Stats() AutoTuneStats {
	a.mu.Lock()
	defer a.mu.Unlock()

	var nextTune time.Time
	if !a.lastTuneAt.IsZero() {
		nextTune = a.lastTuneAt.Add(a.cooldown)
	}

	return AutoTuneStats{
		PairCount:         a.collector.Size(),
		AvgQuality:        a.collector.AverageQuality(),
		MinPairs:          a.minPairs,
		QualityFloor:      a.qualityFloor,
		LastTuneAt:        a.lastTuneAt,
		NextTuneAfter:     nextTune,
		TunedName:         a.tunedName,
		Ready:             a.shouldTuneLocked(),
		ABBaseWins:        a.abBaseWins,
		ABTunedWins:       a.abTunedWins,
		ABTotalTrials:     a.abTotalTrials,
		ConsecutiveFails:  a.consecutiveFails,
		EffectiveCooldown: a.cooldown,
	}
}

// ForceCheck triggers tuning regardless of cooldown (but still requires
// minimum pairs and quality). Returns true if tuning was triggered.
func (a *AutoTuner) ForceCheck() bool {
	a.mu.Lock()
	// Temporarily zero out lastTuneAt to bypass cooldown
	saved := a.lastTuneAt
	savedAttempt := a.lastAttemptAt
	a.lastTuneAt = time.Time{}
	a.lastAttemptAt = time.Time{}
	a.mu.Unlock()

	triggered := a.Check()

	if !triggered {
		// Restore if we didn't actually tune
		a.mu.Lock()
		a.lastTuneAt = saved
		a.lastAttemptAt = savedAttempt
		a.mu.Unlock()
	}

	return triggered
}

// RecordABResult records the outcome of an A/B test between base and tuned models.
// winner should be "base" or "tuned".
func (a *AutoTuner) RecordABResult(winner string) {
	a.mu.Lock()
	defer a.mu.Unlock()

	a.abTotalTrials++
	switch winner {
	case "base":
		a.abBaseWins++
	case "tuned":
		a.abTunedWins++
	}
}

// ABWinRate returns the tuned model's win rate (0.0-1.0).
// Returns 0.5 if no trials have been run.
func (a *AutoTuner) ABWinRate() float64 {
	a.mu.Lock()
	defer a.mu.Unlock()

	if a.abTotalTrials == 0 {
		return 0.5
	}
	return float64(a.abTunedWins) / float64(a.abTotalTrials)
}

// ShouldUseTuned returns true if the tuned model should be used based on A/B results.
// Uses a simple threshold: tuned model must win >50% of trials with at least 10 trials.
func (a *AutoTuner) ShouldUseTuned() bool {
	a.mu.Lock()
	defer a.mu.Unlock()

	if !a.abEnabled || a.abTotalTrials < 10 {
		return true // default to tuned when not enough data
	}
	return float64(a.abTunedWins) > float64(a.abTotalTrials)/2
}

// adaptiveCooldownValue calculates the cooldown based on consecutive failures.
// Success: cooldown = base (minimum 30 minutes)
// Each failure doubles the cooldown, capped at 24 hours.
func (a *AutoTuner) adaptiveCooldownValue() time.Duration {
	if a.baseCooldown == 0 {
		a.baseCooldown = 1 * time.Hour
	}
	if a.consecutiveFails == 0 {
		// Success path: shrink toward base, minimum 30 min
		cd := a.baseCooldown
		if cd < 30*time.Minute {
			cd = 30 * time.Minute
		}
		return cd
	}
	// Failure path: exponential backoff, capped at 24h
	cd := a.baseCooldown
	for i := 0; i < a.consecutiveFails && i < 5; i++ {
		cd *= 2
	}
	if cd > 24*time.Hour {
		cd = 24 * time.Hour
	}
	return cd
}

// buildEnhancedSystemPrompt creates a system prompt that embeds learned patterns
// from high-quality training data into the model's identity.
func (a *AutoTuner) buildEnhancedSystemPrompt() string {
	base := NousSystemPrompt()

	// Get high-quality pairs to extract patterns
	pairs := a.collector.HighQualityPairs(0.8)
	if len(pairs) == 0 {
		return base
	}

	// Extract common tool usage patterns
	toolFreq := make(map[string]int)
	for _, p := range pairs {
		for _, t := range p.ToolCalls {
			toolFreq[t]++
		}
	}

	var patterns []string
	for tool, count := range toolFreq {
		if count >= 3 {
			patterns = append(patterns, fmt.Sprintf("- Use '%s' tool frequently (%d successful uses)", tool, count))
		}
	}

	if len(patterns) == 0 {
		return base
	}

	var sb strings.Builder
	sb.WriteString(base)
	sb.WriteString("\n\nLearned patterns from successful interactions:\n")
	for _, p := range patterns {
		sb.WriteString(p)
		sb.WriteString("\n")
	}

	return sb.String()
}
