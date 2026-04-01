package federation

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sync"
	"time"
)

// TrustScorer evaluates the trustworthiness of imported crystals.
type TrustScorer struct {
	MinVotes     int                `json:"min_votes"`     // minimum votes to trust (default: 3)
	MinQuality   float64            `json:"min_quality"`   // minimum quality threshold (default: 0.6)
	DecayDays    int                `json:"decay_days"`    // days after which trust decays (default: 90)
	BundleScores map[string]float64 `json:"bundle_scores"` // instance ID -> trust score
	mu           sync.RWMutex
}

// NewTrustScorer creates a TrustScorer with sensible defaults.
func NewTrustScorer() *TrustScorer {
	return &TrustScorer{
		MinVotes:     3,
		MinQuality:   0.6,
		DecayDays:    90,
		BundleScores: make(map[string]float64),
	}
}

// Score computes a trust score for a crystal in the range [0.0, 1.0].
//
// The score is a weighted combination of:
//   - Base quality (40%): the crystal's own quality rating
//   - Vote confidence (30%): saturates at MinVotes
//   - Recency (20%): exponential decay based on time since LastVoted
//   - Bundle trust (10%): if the source instance is known and trusted
func (ts *TrustScorer) Score(crystal SharedCrystal) float64 {
	ts.mu.RLock()
	defer ts.mu.RUnlock()

	// Base quality component (40%).
	base := crystal.Quality * 0.4

	// Vote confidence component (30%).
	voteRatio := float64(crystal.Votes) / float64(ts.MinVotes)
	if voteRatio > 1.0 {
		voteRatio = 1.0
	}
	votes := voteRatio * 0.3

	// Recency component (20%).
	daysSinceVote := time.Since(crystal.LastVoted).Hours() / 24.0
	decay := math.Exp(-daysSinceVote / float64(ts.DecayDays))
	recency := decay * 0.2

	// Bundle trust component (10%).
	bundleTrust := 0.5 // neutral default
	if score, ok := ts.BundleScores[crystal.Source]; ok {
		bundleTrust = score
	}
	bundle := bundleTrust * 0.1

	total := base + votes + recency + bundle
	if total < 0.0 {
		return 0.0
	}
	if total > 1.0 {
		return 1.0
	}
	return total
}

// ShouldImport returns true if the crystal's trust score exceeds 0.5.
func (ts *TrustScorer) ShouldImport(crystal SharedCrystal) bool {
	return ts.Score(crystal) > 0.5
}

// ObserveBundle updates the trust score for a specific instance based on the
// acceptance rate of its crystals. The score is an exponential moving average
// biased toward recent observations.
func (ts *TrustScorer) ObserveBundle(instanceID string, accepted, rejected int) {
	total := accepted + rejected
	if total == 0 {
		return
	}

	rate := float64(accepted) / float64(total)

	ts.mu.Lock()
	defer ts.mu.Unlock()

	prev, ok := ts.BundleScores[instanceID]
	if !ok {
		ts.BundleScores[instanceID] = rate
		return
	}

	// Exponential moving average (alpha = 0.3 for recent bias).
	const alpha = 0.3
	ts.BundleScores[instanceID] = alpha*rate + (1-alpha)*prev
}

// Save persists the trust scorer state to a JSON file.
func (ts *TrustScorer) Save(path string) error {
	ts.mu.RLock()
	defer ts.mu.RUnlock()

	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return fmt.Errorf("create directory: %w", err)
	}

	data, err := json.MarshalIndent(ts, "", "  ")
	if err != nil {
		return fmt.Errorf("marshal trust scorer: %w", err)
	}

	if err := os.WriteFile(path, data, 0o644); err != nil {
		return fmt.Errorf("write trust scorer: %w", err)
	}
	return nil
}

// Load restores the trust scorer state from a JSON file.
func (ts *TrustScorer) Load(path string) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return fmt.Errorf("read trust scorer: %w", err)
	}

	ts.mu.Lock()
	defer ts.mu.Unlock()

	if err := json.Unmarshal(data, ts); err != nil {
		return fmt.Errorf("unmarshal trust scorer: %w", err)
	}

	if ts.BundleScores == nil {
		ts.BundleScores = make(map[string]float64)
	}
	return nil
}
