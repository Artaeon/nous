package eval

import (
	"encoding/json"
	"fmt"
	"os"
	"sort"
	"time"
)

// PRGateConfig defines the quality thresholds for PR acceptance.
type PRGateConfig struct {
	// Quality benchmarks
	MinIntentAccuracy    float64 // minimum intent classification accuracy (0.92)
	MinNLGQuality        float64 // minimum NLG quality score (0.70)
	MinScorecardPass     float64 // minimum fraction of scorecards passing (0.875 = 7/8)
	MaxHallucinationRate float64 // maximum hallucination rate (0.05)
	MaxRegressionRate    float64 // maximum regression rate vs baseline (0.02)

	// Latency budgets (milliseconds)
	MaxP50Latency int64 // 50th percentile (50ms)
	MaxP95Latency int64 // 95th percentile (200ms)
	MaxP99Latency int64 // 99th percentile (500ms)

	// Crash/hang tests
	MaxCrashes    int   // maximum crashes in test suite (0)
	MaxHangs      int   // maximum hangs in test suite (0)
	HangTimeoutMs int64 // what counts as a hang (5000ms)

	// Regression
	BaselinePath string // path to baseline metrics file
}

// DefaultPRGateConfig returns a PRGateConfig with production defaults.
func DefaultPRGateConfig() *PRGateConfig {
	return &PRGateConfig{
		MinIntentAccuracy:    0.92,
		MinNLGQuality:        0.70,
		MinScorecardPass:     0.875,
		MaxHallucinationRate: 0.05,
		MaxRegressionRate:    0.02,

		MaxP50Latency: 50,
		MaxP95Latency: 200,
		MaxP99Latency: 500,

		MaxCrashes:    0,
		MaxHangs:      0,
		HangTimeoutMs: 5000,

		BaselinePath: "baseline.json",
	}
}

// PRGateResult captures the full PR validation result.
type PRGateResult struct {
	Pass         bool
	Failures     []PRGateFailure
	Warnings     []string
	QualityScore float64
	LatencyP50   int64
	LatencyP95   int64
	LatencyP99   int64
	CrashCount   int
	HangCount    int
	Duration     time.Duration
	Timestamp    time.Time
}

// PRGateFailure describes a single gate check failure.
type PRGateFailure struct {
	Gate     string
	Expected string
	Actual   string
	Severity string // "blocking", "warning"
}

// BenchmarkResults contains the test/benchmark data to validate.
type BenchmarkResults struct {
	IntentAccuracy    float64
	NLGQuality        float64
	ScorecardResults  []ScorecardResult
	HallucinationRate float64
	Latencies         []int64 // individual latency measurements in ms
	Crashes           int
	Hangs             int
	RegressionResults []RegressionResult
}

// RegressionResult captures the comparison of one metric against its baseline.
type RegressionResult struct {
	TestName  string
	Baseline  float64
	Current   float64
	Regressed bool
	Delta     float64
}

// PRGate runs the full suite of PR quality checks.
type PRGate struct {
	Config *PRGateConfig
}

// NewPRGate creates a new PRGate with the given config.
func NewPRGate(config *PRGateConfig) *PRGate {
	if config == nil {
		config = DefaultPRGateConfig()
	}
	return &PRGate{Config: config}
}

// RunAll executes all PR gates and returns the result.
func (pg *PRGate) RunAll(results *BenchmarkResults) *PRGateResult {
	start := time.Now()

	var allFailures []PRGateFailure
	var warnings []string

	// Quality benchmarks
	qualityFailures := pg.CheckQualityBenchmarks(results)
	allFailures = append(allFailures, qualityFailures...)

	// Latency budgets
	latencyFailures := pg.CheckLatencyBudgets(results.Latencies)
	allFailures = append(allFailures, latencyFailures...)

	// Crash/hang
	crashFailures := pg.CheckCrashHang(results.Crashes, results.Hangs)
	allFailures = append(allFailures, crashFailures...)

	// Regressions
	regressionFailures := pg.CheckRegressions(results.RegressionResults)
	allFailures = append(allFailures, regressionFailures...)

	// Compute latency percentiles
	p50, p95, p99 := ComputeLatencyPercentiles(results.Latencies)

	// Compute overall quality score as weighted average of key metrics
	qualityScore := computeQualityScore(results)

	// Separate blocking failures from warnings
	var blocking []PRGateFailure
	for _, f := range allFailures {
		if f.Severity == "warning" {
			warnings = append(warnings, fmt.Sprintf("%s: expected %s, got %s", f.Gate, f.Expected, f.Actual))
		} else {
			blocking = append(blocking, f)
		}
	}

	pass := len(blocking) == 0

	return &PRGateResult{
		Pass:         pass,
		Failures:     allFailures,
		Warnings:     warnings,
		QualityScore: qualityScore,
		LatencyP50:   p50,
		LatencyP95:   p95,
		LatencyP99:   p99,
		CrashCount:   results.Crashes,
		HangCount:    results.Hangs,
		Duration:     time.Since(start),
		Timestamp:    time.Now(),
	}
}

// computeQualityScore calculates a weighted overall quality metric.
func computeQualityScore(results *BenchmarkResults) float64 {
	// Weight: intent accuracy 30%, NLG quality 20%, scorecard pass rate 30%,
	// hallucination penalty 20%
	scorecardPassRate := 0.0
	if len(results.ScorecardResults) > 0 {
		passCount := 0
		for _, sr := range results.ScorecardResults {
			if sr.Pass {
				passCount++
			}
		}
		scorecardPassRate = float64(passCount) / float64(len(results.ScorecardResults))
	}

	hallucinationScore := 1.0 - results.HallucinationRate
	if hallucinationScore < 0 {
		hallucinationScore = 0
	}

	score := results.IntentAccuracy*0.30 +
		results.NLGQuality*0.20 +
		scorecardPassRate*0.30 +
		hallucinationScore*0.20

	return score
}

// CheckQualityBenchmarks validates quality metrics against configured thresholds.
func (pg *PRGate) CheckQualityBenchmarks(results *BenchmarkResults) []PRGateFailure {
	var failures []PRGateFailure

	if results.IntentAccuracy < pg.Config.MinIntentAccuracy {
		failures = append(failures, PRGateFailure{
			Gate:     "intent_accuracy",
			Expected: fmt.Sprintf(">= %.4f", pg.Config.MinIntentAccuracy),
			Actual:   fmt.Sprintf("%.4f", results.IntentAccuracy),
			Severity: "blocking",
		})
	}

	if results.NLGQuality < pg.Config.MinNLGQuality {
		failures = append(failures, PRGateFailure{
			Gate:     "nlg_quality",
			Expected: fmt.Sprintf(">= %.4f", pg.Config.MinNLGQuality),
			Actual:   fmt.Sprintf("%.4f", results.NLGQuality),
			Severity: "blocking",
		})
	}

	// Scorecard pass rate
	if len(results.ScorecardResults) > 0 {
		passCount := 0
		for _, sr := range results.ScorecardResults {
			if sr.Pass {
				passCount++
			}
		}
		passRate := float64(passCount) / float64(len(results.ScorecardResults))
		if passRate < pg.Config.MinScorecardPass {
			failures = append(failures, PRGateFailure{
				Gate:     "scorecard_pass_rate",
				Expected: fmt.Sprintf(">= %.4f", pg.Config.MinScorecardPass),
				Actual:   fmt.Sprintf("%.4f (%d/%d)", passRate, passCount, len(results.ScorecardResults)),
				Severity: "blocking",
			})
		}
	}

	if results.HallucinationRate > pg.Config.MaxHallucinationRate {
		failures = append(failures, PRGateFailure{
			Gate:     "hallucination_rate",
			Expected: fmt.Sprintf("<= %.4f", pg.Config.MaxHallucinationRate),
			Actual:   fmt.Sprintf("%.4f", results.HallucinationRate),
			Severity: "blocking",
		})
	}

	return failures
}

// CheckLatencyBudgets validates latency percentiles against configured budgets.
func (pg *PRGate) CheckLatencyBudgets(latencies []int64) []PRGateFailure {
	var failures []PRGateFailure

	if len(latencies) == 0 {
		return failures
	}

	p50, p95, p99 := ComputeLatencyPercentiles(latencies)

	if p50 > pg.Config.MaxP50Latency {
		failures = append(failures, PRGateFailure{
			Gate:     "latency_p50",
			Expected: fmt.Sprintf("<= %dms", pg.Config.MaxP50Latency),
			Actual:   fmt.Sprintf("%dms", p50),
			Severity: "blocking",
		})
	}

	if p95 > pg.Config.MaxP95Latency {
		failures = append(failures, PRGateFailure{
			Gate:     "latency_p95",
			Expected: fmt.Sprintf("<= %dms", pg.Config.MaxP95Latency),
			Actual:   fmt.Sprintf("%dms", p95),
			Severity: "blocking",
		})
	}

	if p99 > pg.Config.MaxP99Latency {
		failures = append(failures, PRGateFailure{
			Gate:     "latency_p99",
			Expected: fmt.Sprintf("<= %dms", pg.Config.MaxP99Latency),
			Actual:   fmt.Sprintf("%dms", p99),
			Severity: "blocking",
		})
	}

	return failures
}

// CheckCrashHang validates that no crashes or hangs occurred.
func (pg *PRGate) CheckCrashHang(crashes, hangs int) []PRGateFailure {
	var failures []PRGateFailure

	if crashes > pg.Config.MaxCrashes {
		failures = append(failures, PRGateFailure{
			Gate:     "crashes",
			Expected: fmt.Sprintf("<= %d", pg.Config.MaxCrashes),
			Actual:   fmt.Sprintf("%d", crashes),
			Severity: "blocking",
		})
	}

	if hangs > pg.Config.MaxHangs {
		failures = append(failures, PRGateFailure{
			Gate:     "hangs",
			Expected: fmt.Sprintf("<= %d", pg.Config.MaxHangs),
			Actual:   fmt.Sprintf("%d", hangs),
			Severity: "blocking",
		})
	}

	return failures
}

// CheckRegressions validates no metric regressions beyond the allowed threshold.
func (pg *PRGate) CheckRegressions(results []RegressionResult) []PRGateFailure {
	var failures []PRGateFailure

	for _, r := range results {
		if !r.Regressed {
			continue
		}

		// Calculate regression magnitude
		var regressionRate float64
		if r.Baseline > 0 {
			regressionRate = (r.Baseline - r.Current) / r.Baseline
		} else {
			regressionRate = r.Delta
		}

		if regressionRate > pg.Config.MaxRegressionRate {
			failures = append(failures, PRGateFailure{
				Gate:     fmt.Sprintf("regression/%s", r.TestName),
				Expected: fmt.Sprintf("regression <= %.4f", pg.Config.MaxRegressionRate),
				Actual:   fmt.Sprintf("%.4f (baseline=%.4f, current=%.4f)", regressionRate, r.Baseline, r.Current),
				Severity: "blocking",
			})
		} else {
			// Small regression, just a warning
			failures = append(failures, PRGateFailure{
				Gate:     fmt.Sprintf("regression/%s", r.TestName),
				Expected: fmt.Sprintf("regression <= %.4f", pg.Config.MaxRegressionRate),
				Actual:   fmt.Sprintf("%.4f (baseline=%.4f, current=%.4f)", regressionRate, r.Baseline, r.Current),
				Severity: "warning",
			})
		}
	}

	return failures
}

// baselineData is the serializable representation of BenchmarkResults for JSON.
type baselineData struct {
	IntentAccuracy    float64            `json:"intent_accuracy"`
	NLGQuality        float64            `json:"nlg_quality"`
	HallucinationRate float64            `json:"hallucination_rate"`
	Latencies         []int64            `json:"latencies"`
	Crashes           int                `json:"crashes"`
	Hangs             int                `json:"hangs"`
	RegressionResults []RegressionResult `json:"regression_results"`
	ScorecardPassing  int                `json:"scorecard_passing"`
	ScorecardTotal    int                `json:"scorecard_total"`
	Timestamp         time.Time          `json:"timestamp"`
}

// SaveBaseline saves current metrics as the baseline for future comparisons.
func SaveBaseline(results *BenchmarkResults, path string) error {
	passCount := 0
	for _, sr := range results.ScorecardResults {
		if sr.Pass {
			passCount++
		}
	}

	data := baselineData{
		IntentAccuracy:    results.IntentAccuracy,
		NLGQuality:        results.NLGQuality,
		HallucinationRate: results.HallucinationRate,
		Latencies:         results.Latencies,
		Crashes:           results.Crashes,
		Hangs:             results.Hangs,
		RegressionResults: results.RegressionResults,
		ScorecardPassing:  passCount,
		ScorecardTotal:    len(results.ScorecardResults),
		Timestamp:         time.Now(),
	}

	b, err := json.MarshalIndent(data, "", "  ")
	if err != nil {
		return fmt.Errorf("marshal baseline: %w", err)
	}

	return os.WriteFile(path, b, 0644)
}

// LoadBaseline loads previously saved baseline metrics.
func LoadBaseline(path string) (*BenchmarkResults, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read baseline: %w", err)
	}

	var data baselineData
	if err := json.Unmarshal(b, &data); err != nil {
		return nil, fmt.Errorf("unmarshal baseline: %w", err)
	}

	// Reconstruct scorecard results with pass/fail based on saved counts
	scorecardResults := make([]ScorecardResult, data.ScorecardTotal)
	for i := range scorecardResults {
		scorecardResults[i] = ScorecardResult{
			Capability: fmt.Sprintf("baseline_%d", i),
			Pass:       i < data.ScorecardPassing,
			Scores:     make(map[string]float64),
		}
	}

	return &BenchmarkResults{
		IntentAccuracy:    data.IntentAccuracy,
		NLGQuality:        data.NLGQuality,
		ScorecardResults:  scorecardResults,
		HallucinationRate: data.HallucinationRate,
		Latencies:         data.Latencies,
		Crashes:           data.Crashes,
		Hangs:             data.Hangs,
		RegressionResults: data.RegressionResults,
	}, nil
}

// ComputeLatencyPercentiles calculates p50, p95, p99 from raw latencies.
// Returns zeros if the input slice is empty.
func ComputeLatencyPercentiles(latencies []int64) (p50, p95, p99 int64) {
	n := len(latencies)
	if n == 0 {
		return 0, 0, 0
	}

	// Sort a copy to avoid mutating the input
	sorted := make([]int64, n)
	copy(sorted, latencies)
	sort.Slice(sorted, func(i, j int) bool { return sorted[i] < sorted[j] })

	p50 = percentile(sorted, 0.50)
	p95 = percentile(sorted, 0.95)
	p99 = percentile(sorted, 0.99)
	return
}

// percentile returns the value at the given percentile from a sorted slice.
func percentile(sorted []int64, p float64) int64 {
	n := len(sorted)
	if n == 0 {
		return 0
	}
	if n == 1 {
		return sorted[0]
	}

	// Use nearest-rank method
	rank := p * float64(n-1)
	lower := int(rank)
	if lower >= n-1 {
		return sorted[n-1]
	}

	// Linear interpolation between adjacent ranks
	frac := rank - float64(lower)
	return sorted[lower] + int64(frac*float64(sorted[lower+1]-sorted[lower]))
}
