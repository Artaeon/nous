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

// PreferencePair represents a chosen/rejected pair for preference learning.
type PreferencePair struct {
	Input     string    `json:"input"`
	Chosen    string    `json:"chosen"`    // preferred response
	Rejected  string    `json:"rejected"`  // dispreferred response
	Margin    float64   `json:"margin"`    // how much better chosen is (0-1)
	Source    string    `json:"source"`    // "human", "auto_quality", "correction"
	Timestamp time.Time `json:"timestamp"`
}

// PreferenceConfig controls the optimization process.
type PreferenceConfig struct {
	Beta         float64 // DPO temperature parameter (0.1)
	LearningRate float64 // optimizer learning rate (0.01)
	BatchSize    int     // pairs per batch (32)
	MaxPairs     int     // maximum pairs to use (10000)
	MinMargin    float64 // minimum quality margin to use pair (0.1)
	RegWeight    float64 // regularization weight (0.01)
}

// DefaultPreferenceConfig returns sensible default configuration.
func DefaultPreferenceConfig() *PreferenceConfig {
	return &PreferenceConfig{
		Beta:         0.1,
		LearningRate: 0.01,
		BatchSize:    32,
		MaxPairs:     10000,
		MinMargin:    0.1,
		RegWeight:    0.01,
	}
}

// PreferenceStore manages preference pairs on disk.
type PreferenceStore struct {
	dir   string
	pairs []PreferencePair
	mu    sync.RWMutex
}

// NewPreferenceStore creates a store backed by the given directory.
func NewPreferenceStore(dir string) *PreferenceStore {
	return &PreferenceStore{
		dir: dir,
	}
}

// RecordPair appends a preference pair and persists to disk.
func (ps *PreferenceStore) RecordPair(pair *PreferencePair) error {
	if pair == nil {
		return fmt.Errorf("preference: nil pair")
	}
	if pair.Timestamp.IsZero() {
		pair.Timestamp = time.Now()
	}

	ps.mu.Lock()
	ps.pairs = append(ps.pairs, *pair)
	data, err := json.MarshalIndent(ps.pairs, "", "  ")
	ps.mu.Unlock()
	if err != nil {
		return fmt.Errorf("preference: marshal: %w", err)
	}

	if ps.dir == "" {
		return nil
	}
	return safefile.WriteAtomic(filepath.Join(ps.dir, "preference_pairs.json"), data, 0644)
}

// LoadPairs reads all pairs from disk into memory.
func (ps *PreferenceStore) LoadPairs() ([]PreferencePair, error) {
	if ps.dir == "" {
		ps.mu.RLock()
		out := make([]PreferencePair, len(ps.pairs))
		copy(out, ps.pairs)
		ps.mu.RUnlock()
		return out, nil
	}

	data, err := os.ReadFile(filepath.Join(ps.dir, "preference_pairs.json"))
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, fmt.Errorf("preference: read: %w", err)
	}

	var pairs []PreferencePair
	if err := json.Unmarshal(data, &pairs); err != nil {
		return nil, fmt.Errorf("preference: unmarshal: %w", err)
	}

	ps.mu.Lock()
	ps.pairs = pairs
	ps.mu.Unlock()

	out := make([]PreferencePair, len(pairs))
	copy(out, pairs)
	return out, nil
}

// FilterByMargin returns pairs with margin >= minMargin.
func (ps *PreferenceStore) FilterByMargin(minMargin float64) []PreferencePair {
	ps.mu.RLock()
	defer ps.mu.RUnlock()

	var result []PreferencePair
	for _, p := range ps.pairs {
		if p.Margin >= minMargin {
			result = append(result, p)
		}
	}
	return result
}

// PreferenceStats provides summary statistics about stored pairs.
type PreferenceStats struct {
	TotalPairs    int
	BySource      map[string]int
	AverageMargin float64
	MinMargin     float64
	MaxMargin     float64
}

// Stats computes summary statistics for stored pairs.
func (ps *PreferenceStore) Stats() PreferenceStats {
	ps.mu.RLock()
	defer ps.mu.RUnlock()

	stats := PreferenceStats{
		TotalPairs: len(ps.pairs),
		BySource:   make(map[string]int),
		MinMargin:  math.MaxFloat64,
		MaxMargin:  -math.MaxFloat64,
	}

	if len(ps.pairs) == 0 {
		stats.MinMargin = 0
		stats.MaxMargin = 0
		return stats
	}

	var sumMargin float64
	for _, p := range ps.pairs {
		stats.BySource[p.Source]++
		sumMargin += p.Margin
		if p.Margin < stats.MinMargin {
			stats.MinMargin = p.Margin
		}
		if p.Margin > stats.MaxMargin {
			stats.MaxMargin = p.Margin
		}
	}
	stats.AverageMargin = sumMargin / float64(len(ps.pairs))

	return stats
}

// PreferenceOptimizer implements offline DPO-style optimization.
type PreferenceOptimizer struct {
	Config *PreferenceConfig
	Store  *PreferenceStore
}

// NewPreferenceOptimizer creates a preference optimizer.
func NewPreferenceOptimizer(store *PreferenceStore, config *PreferenceConfig) *PreferenceOptimizer {
	if config == nil {
		config = DefaultPreferenceConfig()
	}
	return &PreferenceOptimizer{
		Config: config,
		Store:  store,
	}
}

// OptimizationResult captures the outcome of a preference optimization run.
type OptimizationResult struct {
	PairsUsed       int
	AverageReward   float64
	ChosenWinRate   float64 // fraction where chosen scored higher after optimization
	LossImprovement float64
	Iterations      int
}

// featurePair holds extracted feature vectors for a chosen/rejected pair.
type featurePair struct {
	chosenFeats   [6]float64
	rejectedFeats [6]float64
	margin        float64
}

// RewardWeights captures learned preferences for response qualities.
type RewardWeights struct {
	Correctness float64 `json:"correctness"`
	Helpfulness float64 `json:"helpfulness"`
	Conciseness float64 `json:"conciseness"`
	Specificity float64 `json:"specificity"`
	Safety      float64 `json:"safety"`
	Coherence   float64 `json:"coherence"`
}

// Optimize runs the preference optimization and returns reward weights.
//
// DPO loss: -log(sigmoid(beta * (r(chosen) - r(rejected))))
// where r(x) = w . features(x) is the implicit reward.
//
// Since we don't have a full language model, we use quality feature vectors
// extracted from text and optimize weights to maximize margin between
// chosen and rejected responses.
func (po *PreferenceOptimizer) Optimize() (*OptimizationResult, *RewardWeights, error) {
	pairs := po.Store.FilterByMargin(po.Config.MinMargin)
	if len(pairs) == 0 {
		return nil, nil, fmt.Errorf("preference: no pairs above margin threshold %.2f", po.Config.MinMargin)
	}

	// Cap at MaxPairs, keeping highest-margin pairs first
	if len(pairs) > po.Config.MaxPairs {
		sort.Slice(pairs, func(i, j int) bool {
			return pairs[i].Margin > pairs[j].Margin
		})
		pairs = pairs[:po.Config.MaxPairs]
	}

	// Initialize reward weights uniformly
	weights := &RewardWeights{
		Correctness: 1.0,
		Helpfulness: 1.0,
		Conciseness: 1.0,
		Specificity: 1.0,
		Safety:      1.0,
		Coherence:   1.0,
	}

	// Extract feature vectors for all chosen/rejected pairs
	var fpairs []featurePair
	for _, p := range pairs {
		cf := extractFeatures(p.Chosen, p.Input)
		rf := extractFeatures(p.Rejected, p.Input)
		fpairs = append(fpairs, featurePair{
			chosenFeats:  cf,
			rejectedFeats: rf,
			margin:       p.Margin,
		})
	}

	// Gradient descent on DPO loss
	beta := po.Config.Beta
	lr := po.Config.LearningRate
	reg := po.Config.RegWeight
	batchSize := po.Config.BatchSize
	if batchSize <= 0 {
		batchSize = len(fpairs)
	}

	w := [6]float64{weights.Correctness, weights.Helpfulness, weights.Conciseness,
		weights.Specificity, weights.Safety, weights.Coherence}

	initialLoss := computeDPOLoss(fpairs, w, beta)
	iterations := 0
	maxIter := len(fpairs) / batchSize
	if maxIter < 50 {
		maxIter = 50
	}
	if maxIter > 500 {
		maxIter = 500
	}

	for iter := 0; iter < maxIter; iter++ {
		iterations++

		// Compute gradients over mini-batch
		var grad [6]float64
		batchEnd := batchSize
		batchStart := (iter * batchSize) % len(fpairs)
		if batchStart+batchEnd > len(fpairs) {
			batchEnd = len(fpairs) - batchStart
		}
		if batchEnd <= 0 {
			batchEnd = len(fpairs)
			batchStart = 0
		}

		batch := fpairs[batchStart : batchStart+batchEnd]

		for _, fp := range batch {
			// r(chosen) - r(rejected) = w . (chosen_features - rejected_features)
			var diff [6]float64
			for d := 0; d < 6; d++ {
				diff[d] = fp.chosenFeats[d] - fp.rejectedFeats[d]
			}

			rewardDiff := dotProduct6(w, diff)
			// sigmoid(beta * reward_diff)
			sig := sigmoid(beta * rewardDiff)

			// Gradient of -log(sigmoid(beta * r_diff)) w.r.t. w:
			// = -beta * (1 - sigmoid(beta * r_diff)) * diff
			for d := 0; d < 6; d++ {
				grad[d] += -beta * (1 - sig) * diff[d]
			}
		}

		// Average gradient and apply regularization
		batchN := float64(len(batch))
		for d := 0; d < 6; d++ {
			grad[d] = grad[d]/batchN + reg*w[d]
		}

		// Update weights (gradient descent, so subtract gradient)
		for d := 0; d < 6; d++ {
			w[d] -= lr * grad[d]
			// Keep weights non-negative
			if w[d] < 0 {
				w[d] = 0
			}
		}
	}

	// Normalize weights so they sum to 6 (same as initial)
	var wSum float64
	for d := 0; d < 6; d++ {
		wSum += w[d]
	}
	if wSum > 0 {
		scale := 6.0 / wSum
		for d := 0; d < 6; d++ {
			w[d] *= scale
		}
	}

	weights.Correctness = w[0]
	weights.Helpfulness = w[1]
	weights.Conciseness = w[2]
	weights.Specificity = w[3]
	weights.Safety = w[4]
	weights.Coherence = w[5]

	// Compute final metrics
	finalLoss := computeDPOLoss(fpairs, w, beta)
	chosenWins := 0
	var totalReward float64
	for _, fp := range fpairs {
		chosenR := dotProduct6(w, fp.chosenFeats)
		rejectedR := dotProduct6(w, fp.rejectedFeats)
		if chosenR > rejectedR {
			chosenWins++
		}
		totalReward += chosenR - rejectedR
	}

	result := &OptimizationResult{
		PairsUsed:       len(pairs),
		AverageReward:   totalReward / float64(len(fpairs)),
		ChosenWinRate:   float64(chosenWins) / float64(len(fpairs)),
		LossImprovement: initialLoss - finalLoss,
		Iterations:      iterations,
	}

	return result, weights, nil
}

// computeDPOLoss computes the average DPO loss over all pairs.
func computeDPOLoss(fpairs []featurePair, w [6]float64, beta float64) float64 {
	var totalLoss float64
	for _, fp := range fpairs {
		var diff [6]float64
		for d := 0; d < 6; d++ {
			diff[d] = fp.chosenFeats[d] - fp.rejectedFeats[d]
		}
		rewardDiff := dotProduct6(w, diff)
		// DPO loss: -log(sigmoid(beta * reward_diff))
		totalLoss += -math.Log(math.Max(sigmoid(beta*rewardDiff), 1e-10))
	}
	return totalLoss / float64(len(fpairs))
}

// sigmoid computes the logistic sigmoid function.
func sigmoid(x float64) float64 {
	if x > 500 {
		return 1.0
	}
	if x < -500 {
		return 0.0
	}
	return 1.0 / (1.0 + math.Exp(-x))
}

// dotProduct6 computes dot product of two 6-element arrays.
func dotProduct6(a, b [6]float64) float64 {
	var sum float64
	for i := 0; i < 6; i++ {
		sum += a[i] * b[i]
	}
	return sum
}

// extractFeatures computes a 6-dimensional quality feature vector for a response.
// Features: [correctness, helpfulness, conciseness, specificity, safety, coherence]
func extractFeatures(response, query string) [6]float64 {
	var f [6]float64

	respLower := strings.ToLower(response)
	queryLower := strings.ToLower(query)
	respWords := strings.Fields(respLower)
	queryWords := strings.Fields(queryLower)
	respLen := len(respWords)

	// Feature 0: Correctness - approximated by query term coverage
	// Higher coverage of query terms suggests more on-topic response
	if len(queryWords) > 0 {
		covered := 0
		for _, qw := range queryWords {
			if strings.Contains(respLower, qw) {
				covered++
			}
		}
		f[0] = float64(covered) / float64(len(queryWords))
	}

	// Feature 1: Helpfulness - measured by information density
	// Presence of specific helpful indicators: numbers, examples, explanations
	helpSignals := 0
	for _, w := range respWords {
		if len(w) > 0 && w[0] >= '0' && w[0] <= '9' {
			helpSignals++ // contains numbers
		}
	}
	if strings.Contains(respLower, "for example") || strings.Contains(respLower, "such as") {
		helpSignals += 2
	}
	if strings.Contains(respLower, "because") || strings.Contains(respLower, "therefore") {
		helpSignals += 2
	}
	f[1] = math.Min(float64(helpSignals)/10.0, 1.0)

	// Feature 2: Conciseness - penalize very long or very short responses
	// Optimal length is roughly 20-100 words for most queries
	if respLen > 0 {
		if respLen <= 100 {
			f[2] = float64(respLen) / 100.0
		} else {
			f[2] = math.Max(0, 1.0-float64(respLen-100)/500.0)
		}
	}

	// Feature 3: Specificity - ratio of unique content words (>3 chars)
	uniqueWords := make(map[string]bool)
	contentWords := 0
	for _, w := range respWords {
		if len(w) > 3 {
			uniqueWords[w] = true
			contentWords++
		}
	}
	if contentWords > 0 {
		f[3] = float64(len(uniqueWords)) / float64(contentWords)
	}

	// Feature 4: Safety - absence of harmful patterns
	f[4] = 1.0
	unsafePatterns := []string{"i cannot", "i can't help", "as an ai", "i don't have opinions"}
	for _, pat := range unsafePatterns {
		if strings.Contains(respLower, pat) {
			f[4] -= 0.25
		}
	}
	if f[4] < 0 {
		f[4] = 0
	}

	// Feature 5: Coherence - measured by sentence structure quality
	sentences := splitSentences(response)
	if len(sentences) > 0 {
		wellFormed := 0
		for _, s := range sentences {
			s = strings.TrimSpace(s)
			if len(s) > 10 && (s[len(s)-1] == '.' || s[len(s)-1] == '!' || s[len(s)-1] == '?') {
				wellFormed++
			}
		}
		f[5] = float64(wellFormed) / float64(len(sentences))
	}

	return f
}

// splitSentences splits text into sentences using basic punctuation rules.
func splitSentences(text string) []string {
	var sentences []string
	var current strings.Builder
	for _, r := range text {
		current.WriteRune(r)
		if r == '.' || r == '!' || r == '?' {
			s := strings.TrimSpace(current.String())
			if len(s) > 0 {
				sentences = append(sentences, s)
			}
			current.Reset()
		}
	}
	if s := strings.TrimSpace(current.String()); len(s) > 0 {
		sentences = append(sentences, s)
	}
	return sentences
}

// ScoreResponse uses reward weights to score a candidate response.
func (rw *RewardWeights) ScoreResponse(response, query string) float64 {
	features := extractFeatures(response, query)
	w := [6]float64{rw.Correctness, rw.Helpfulness, rw.Conciseness,
		rw.Specificity, rw.Safety, rw.Coherence}

	score := dotProduct6(w, features)

	// Normalize to 0-1 range (max possible score is 6.0 with unit weights)
	var wSum float64
	for _, v := range w {
		wSum += v
	}
	if wSum > 0 {
		score /= wSum
	}

	return math.Max(0, math.Min(1, score))
}

// ScoredOutput pairs a response with its quality score for pair generation.
type ScoredOutput struct {
	Input    string
	Response string
	Quality  float64
}

// AutoGeneratePairs creates preference pairs from quality-scored outputs.
// For each input that has multiple responses, it pairs higher-quality responses
// (chosen) with lower-quality responses (rejected).
func AutoGeneratePairs(outputs []ScoredOutput) []PreferencePair {
	// Group outputs by input
	byInput := make(map[string][]ScoredOutput)
	for _, o := range outputs {
		byInput[o.Input] = append(byInput[o.Input], o)
	}

	var pairs []PreferencePair
	now := time.Now()

	for input, group := range byInput {
		if len(group) < 2 {
			continue
		}

		// Sort by quality descending
		sort.Slice(group, func(i, j int) bool {
			return group[i].Quality > group[j].Quality
		})

		// Generate pairs: each higher-quality output vs each lower-quality output
		for i := 0; i < len(group)-1; i++ {
			for j := i + 1; j < len(group); j++ {
				margin := group[i].Quality - group[j].Quality
				if margin < 0.05 {
					continue // skip pairs with negligible margin
				}
				pairs = append(pairs, PreferencePair{
					Input:     input,
					Chosen:    group[i].Response,
					Rejected:  group[j].Response,
					Margin:    margin,
					Source:    "auto_quality",
					Timestamp: now,
				})
			}
		}
	}

	return pairs
}
