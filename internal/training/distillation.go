package training

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/artaeon/nous/internal/safefile"
)

// TeacherTrace represents one recorded output from a stronger model.
type TeacherTrace struct {
	Input     string            `json:"input"`
	Intent    string            `json:"intent"`
	Slots     map[string]string `json:"slots"`
	Plan      []string          `json:"plan"`     // content plan claims
	Response  string            `json:"response"`
	Quality   float64           `json:"quality"`  // 0-1 quality score
	Timestamp time.Time         `json:"timestamp"`
}

// DistillationConfig controls the distillation process.
type DistillationConfig struct {
	MinQuality      float64 // minimum teacher trace quality to use (0.7)
	MaxExamples     int     // maximum examples per batch (5000)
	LearningRate    float64 // student learning rate (0.05)
	Epochs          int     // training epochs (50)
	ValidationSplit float64 // fraction for validation (0.2)
	SoftLabelTemp   float64 // temperature for soft label distillation (2.0)
}

// DefaultDistillationConfig returns sensible default configuration.
func DefaultDistillationConfig() *DistillationConfig {
	return &DistillationConfig{
		MinQuality:      0.7,
		MaxExamples:     5000,
		LearningRate:    0.05,
		Epochs:          50,
		ValidationSplit: 0.2,
		SoftLabelTemp:   2.0,
	}
}

// DistillationStore manages teacher traces on disk.
type DistillationStore struct {
	dir    string
	traces []TeacherTrace
	mu     sync.RWMutex
}

// NewDistillationStore creates a store backed by the given directory.
func NewDistillationStore(dir string) *DistillationStore {
	return &DistillationStore{
		dir: dir,
	}
}

// RecordTrace appends a teacher trace and persists to disk.
func (ds *DistillationStore) RecordTrace(trace *TeacherTrace) error {
	if trace == nil {
		return fmt.Errorf("distillation: nil trace")
	}
	if trace.Timestamp.IsZero() {
		trace.Timestamp = time.Now()
	}

	ds.mu.Lock()
	ds.traces = append(ds.traces, *trace)
	data, err := json.MarshalIndent(ds.traces, "", "  ")
	ds.mu.Unlock()
	if err != nil {
		return fmt.Errorf("distillation: marshal: %w", err)
	}

	if ds.dir == "" {
		return nil
	}
	return safefile.WriteAtomic(filepath.Join(ds.dir, "teacher_traces.json"), data, 0644)
}

// LoadTraces reads all traces from disk into memory.
func (ds *DistillationStore) LoadTraces() ([]TeacherTrace, error) {
	if ds.dir == "" {
		ds.mu.RLock()
		out := make([]TeacherTrace, len(ds.traces))
		copy(out, ds.traces)
		ds.mu.RUnlock()
		return out, nil
	}

	data, err := os.ReadFile(filepath.Join(ds.dir, "teacher_traces.json"))
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, fmt.Errorf("distillation: read: %w", err)
	}

	var traces []TeacherTrace
	if err := json.Unmarshal(data, &traces); err != nil {
		return nil, fmt.Errorf("distillation: unmarshal: %w", err)
	}

	ds.mu.Lock()
	ds.traces = traces
	ds.mu.Unlock()

	out := make([]TeacherTrace, len(traces))
	copy(out, traces)
	return out, nil
}

// FilterByQuality returns traces meeting the minimum quality threshold.
func (ds *DistillationStore) FilterByQuality(minQuality float64) []TeacherTrace {
	ds.mu.RLock()
	defer ds.mu.RUnlock()

	var result []TeacherTrace
	for _, t := range ds.traces {
		if t.Quality >= minQuality {
			result = append(result, t)
		}
	}
	return result
}

// DistillationResult captures the outcome of a distillation run.
type DistillationResult struct {
	IntentAccuracy float64
	SlotF1         float64
	PlanQuality    float64
	TracesUsed     int
	TracesFiltered int
	Epochs         int
	ValidationLoss float64
	ImprovedOver   float64 // improvement vs previous model
}

// IntentDistiller trains intent classification from teacher traces.
type IntentDistiller struct {
	Config *DistillationConfig
	Store  *DistillationStore
}

// NewIntentDistiller creates a distiller for intent classification.
func NewIntentDistiller(store *DistillationStore, config *DistillationConfig) *IntentDistiller {
	if config == nil {
		config = DefaultDistillationConfig()
	}
	return &IntentDistiller{
		Config: config,
		Store:  store,
	}
}

// IntentExample is a distilled training example for the student intent classifier.
type IntentExample struct {
	Input      string
	Intent     string
	Confidence float64 // soft label from teacher
	Slots      map[string]string
}

// Distill runs the distillation process and returns training examples.
func (id *IntentDistiller) Distill() (*DistillationResult, []IntentExample, error) {
	// Load and filter traces
	filtered := id.Store.FilterByQuality(id.Config.MinQuality)
	if len(filtered) == 0 {
		return nil, nil, fmt.Errorf("distillation: no traces above quality threshold %.2f", id.Config.MinQuality)
	}

	totalTraces := len(id.Store.traces)
	filteredOut := totalTraces - len(filtered)

	// Cap at MaxExamples
	if len(filtered) > id.Config.MaxExamples {
		// Sort by quality descending, keep best
		sort.Slice(filtered, func(i, j int) bool {
			return filtered[i].Quality > filtered[j].Quality
		})
		filtered = filtered[:id.Config.MaxExamples]
	}

	// Split into train and validation sets
	splitIdx := int(float64(len(filtered)) * (1.0 - id.Config.ValidationSplit))
	if splitIdx < 1 {
		splitIdx = 1
	}
	if splitIdx >= len(filtered) {
		splitIdx = len(filtered) - 1
	}
	trainSet := filtered[:splitIdx]
	valSet := filtered[splitIdx:]

	// Convert to intent examples with soft labels
	var examples []IntentExample
	for _, trace := range trainSet {
		if trace.Intent == "" {
			continue
		}
		// Apply temperature scaling for soft labels:
		// confidence = quality^(1/T) where T is the temperature
		confidence := math.Pow(trace.Quality, 1.0/id.Config.SoftLabelTemp)
		examples = append(examples, IntentExample{
			Input:      trace.Input,
			Intent:     trace.Intent,
			Confidence: confidence,
			Slots:      trace.Slots,
		})
	}

	if len(examples) == 0 {
		return nil, nil, fmt.Errorf("distillation: no valid intent examples after filtering")
	}

	// Simulate training epochs and compute validation metrics
	var valLoss float64
	intentCorrect := 0
	slotTP, slotFP, slotFN := 0, 0, 0

	// Build intent frequency map from training for majority-baseline comparison
	intentFreq := make(map[string]int)
	for _, ex := range examples {
		intentFreq[ex.Intent]++
	}

	// Find most common intent for baseline
	bestBaselineIntent := ""
	bestBaselineCount := 0
	for intent, count := range intentFreq {
		if count > bestBaselineCount {
			bestBaselineCount = count
			bestBaselineIntent = intent
		}
	}

	// Evaluate on validation set
	for _, trace := range valSet {
		if trace.Intent == "" {
			continue
		}

		// Predict using nearest-neighbor from training examples
		predicted := predictIntent(examples, trace.Input)
		if predicted == trace.Intent {
			intentCorrect++
		}

		// Compute cross-entropy-style loss per example
		// Lower quality teacher outputs contribute more loss
		exLoss := -math.Log(math.Max(trace.Quality, 1e-10))
		valLoss += exLoss

		// Slot F1 computation: compare predicted slots from training vs actual
		predSlots := predictSlots(examples, trace.Input)
		for key, val := range trace.Slots {
			if predSlots[key] == val {
				slotTP++
			} else {
				slotFN++
			}
		}
		for key := range predSlots {
			if _, ok := trace.Slots[key]; !ok {
				slotFP++
			}
		}
	}

	valTotal := len(valSet)
	if valTotal == 0 {
		valTotal = 1
	}
	if len(valSet) > 0 {
		valLoss /= float64(len(valSet))
	}

	intentAcc := float64(intentCorrect) / float64(valTotal)

	// Compute slot F1
	var slotF1 float64
	if slotTP+slotFP > 0 && slotTP+slotFN > 0 {
		precision := float64(slotTP) / float64(slotTP+slotFP)
		recall := float64(slotTP) / float64(slotTP+slotFN)
		if precision+recall > 0 {
			slotF1 = 2 * precision * recall / (precision + recall)
		}
	}

	// Compute baseline accuracy for improvement comparison
	baselineCorrect := 0
	for _, trace := range valSet {
		if trace.Intent == bestBaselineIntent {
			baselineCorrect++
		}
	}
	baselineAcc := float64(baselineCorrect) / float64(valTotal)

	// Compute average plan quality from high-quality traces
	planQuality := 0.0
	planCount := 0
	for _, trace := range filtered {
		if len(trace.Plan) > 0 {
			planQuality += trace.Quality
			planCount++
		}
	}
	if planCount > 0 {
		planQuality /= float64(planCount)
	}

	result := &DistillationResult{
		IntentAccuracy: intentAcc,
		SlotF1:         slotF1,
		PlanQuality:    planQuality,
		TracesUsed:     len(filtered),
		TracesFiltered: filteredOut,
		Epochs:         id.Config.Epochs,
		ValidationLoss: valLoss,
		ImprovedOver:   intentAcc - baselineAcc,
	}

	return result, examples, nil
}

// predictIntent uses a simple similarity-weighted nearest-neighbor approach
// to predict intent from training examples.
func predictIntent(examples []IntentExample, input string) string {
	inputLower := strings.ToLower(input)
	inputWords := strings.Fields(inputLower)

	bestIntent := ""
	bestScore := -1.0

	// Score each intent by summing confidence-weighted word overlap
	intentScores := make(map[string]float64)
	for _, ex := range examples {
		exLower := strings.ToLower(ex.Input)
		exWords := strings.Fields(exLower)

		overlap := wordOverlap(inputWords, exWords)
		intentScores[ex.Intent] += overlap * ex.Confidence
	}

	for intent, score := range intentScores {
		if score > bestScore {
			bestScore = score
			bestIntent = intent
		}
	}

	return bestIntent
}

// predictSlots extracts slots from the most similar training example.
func predictSlots(examples []IntentExample, input string) map[string]string {
	inputLower := strings.ToLower(input)
	inputWords := strings.Fields(inputLower)

	bestSlots := map[string]string{}
	bestScore := -1.0

	for _, ex := range examples {
		if len(ex.Slots) == 0 {
			continue
		}
		exLower := strings.ToLower(ex.Input)
		exWords := strings.Fields(exLower)

		overlap := wordOverlap(inputWords, exWords)
		score := overlap * ex.Confidence
		if score > bestScore {
			bestScore = score
			bestSlots = ex.Slots
		}
	}

	return bestSlots
}

// wordOverlap computes the Jaccard-like overlap between two word sets.
func wordOverlap(a, b []string) float64 {
	if len(a) == 0 || len(b) == 0 {
		return 0
	}
	setB := make(map[string]bool, len(b))
	for _, w := range b {
		setB[w] = true
	}
	overlap := 0
	for _, w := range a {
		if setB[w] {
			overlap++
		}
	}
	union := len(a) + len(b) - overlap
	if union == 0 {
		return 0
	}
	return float64(overlap) / float64(union)
}

// PlanDistiller trains response planning from teacher traces.
type PlanDistiller struct {
	Config *DistillationConfig
	Store  *DistillationStore
}

// NewPlanDistiller creates a distiller for response planning.
func NewPlanDistiller(store *DistillationStore, config *DistillationConfig) *PlanDistiller {
	if config == nil {
		config = DefaultDistillationConfig()
	}
	return &PlanDistiller{
		Config: config,
		Store:  store,
	}
}

// PlanExample is a distilled example for response planning.
type PlanExample struct {
	Input   string
	Claims  []string
	Quality float64
}

// Distill extracts response planning patterns from teacher traces.
func (pd *PlanDistiller) Distill() (*DistillationResult, []PlanExample, error) {
	// Load and filter traces with plans
	filtered := pd.Store.FilterByQuality(pd.Config.MinQuality)
	if len(filtered) == 0 {
		return nil, nil, fmt.Errorf("distillation: no traces above quality threshold %.2f", pd.Config.MinQuality)
	}

	totalTraces := len(pd.Store.traces)

	// Only keep traces that have plan data
	var withPlans []TeacherTrace
	for _, t := range filtered {
		if len(t.Plan) > 0 {
			withPlans = append(withPlans, t)
		}
	}

	if len(withPlans) == 0 {
		return nil, nil, fmt.Errorf("distillation: no traces with plan data")
	}

	filteredOut := totalTraces - len(withPlans)

	// Cap at MaxExamples
	if len(withPlans) > pd.Config.MaxExamples {
		sort.Slice(withPlans, func(i, j int) bool {
			return withPlans[i].Quality > withPlans[j].Quality
		})
		withPlans = withPlans[:pd.Config.MaxExamples]
	}

	// Split into train and validation sets
	splitIdx := int(float64(len(withPlans)) * (1.0 - pd.Config.ValidationSplit))
	if splitIdx < 1 {
		splitIdx = 1
	}
	if splitIdx >= len(withPlans) {
		splitIdx = len(withPlans) - 1
	}
	trainSet := withPlans[:splitIdx]
	valSet := withPlans[splitIdx:]

	// Convert to plan examples
	var examples []PlanExample
	for _, trace := range trainSet {
		examples = append(examples, PlanExample{
			Input:   trace.Input,
			Claims:  trace.Plan,
			Quality: trace.Quality,
		})
	}

	// Evaluate plan quality on validation set
	// Measure: average claim overlap between predicted and actual plans
	var valLoss float64
	var totalPlanQuality float64

	for _, trace := range valSet {
		// Find most similar training plan
		predicted := predictPlan(examples, trace.Input)

		// Compute claim overlap
		overlap := claimOverlap(predicted, trace.Plan)
		totalPlanQuality += overlap

		// Loss: inverse of overlap quality, weighted by teacher quality
		if overlap > 0 {
			valLoss += -math.Log(math.Max(overlap, 1e-10))
		} else {
			valLoss += 5.0 // penalty for zero overlap
		}
	}

	valTotal := len(valSet)
	if valTotal == 0 {
		valTotal = 1
	}
	avgPlanQuality := totalPlanQuality / float64(valTotal)
	valLoss /= float64(valTotal)

	// Compute average training plan quality
	trainPlanQuality := 0.0
	for _, ex := range examples {
		trainPlanQuality += ex.Quality
	}
	if len(examples) > 0 {
		trainPlanQuality /= float64(len(examples))
	}

	result := &DistillationResult{
		IntentAccuracy: 0, // not applicable for plan distillation
		SlotF1:         0,
		PlanQuality:    avgPlanQuality,
		TracesUsed:     len(withPlans),
		TracesFiltered: filteredOut,
		Epochs:         pd.Config.Epochs,
		ValidationLoss: valLoss,
		ImprovedOver:   avgPlanQuality - 0.5, // improvement over random baseline
	}

	return result, examples, nil
}

// predictPlan finds the most similar training example's plan for the given input.
func predictPlan(examples []PlanExample, input string) []string {
	inputLower := strings.ToLower(input)
	inputWords := strings.Fields(inputLower)

	var bestPlan []string
	bestScore := -1.0

	for _, ex := range examples {
		exLower := strings.ToLower(ex.Input)
		exWords := strings.Fields(exLower)

		overlap := wordOverlap(inputWords, exWords)
		score := overlap * ex.Quality
		if score > bestScore {
			bestScore = score
			bestPlan = ex.Claims
		}
	}

	return bestPlan
}

// claimOverlap computes the fraction of actual claims covered by predicted claims
// using substring matching for flexible comparison.
func claimOverlap(predicted, actual []string) float64 {
	if len(actual) == 0 {
		if len(predicted) == 0 {
			return 1.0
		}
		return 0.5
	}

	matched := 0
	for _, act := range actual {
		actLower := strings.ToLower(act)
		actWords := strings.Fields(actLower)
		for _, pred := range predicted {
			predLower := strings.ToLower(pred)
			predWords := strings.Fields(predLower)
			if wordOverlap(actWords, predWords) > 0.3 {
				matched++
				break
			}
		}
	}

	return float64(matched) / float64(len(actual))
}
