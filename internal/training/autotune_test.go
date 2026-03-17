package training

import (
	"math"
	"testing"
	"time"
)

// mockCreator implements ModelCreator for testing.
type mockCreator struct {
	called    bool
	name      string
	modelfile string
	err       error
}

func (m *mockCreator) CreateModel(name, modelfile string) error {
	m.called = true
	m.name = name
	m.modelfile = modelfile
	return m.err
}

func newTestCollector(pairs int, quality float64) *Collector {
	c := NewCollector("")
	for i := 0; i < pairs; i++ {
		c.Collect("system", "input", "output", []string{"read"}, quality)
	}
	return c
}

func TestNewAutoTunerDerivesTunedName(t *testing.T) {
	tests := []struct {
		modelName string
		wantTuned string
	}{
		{"qwen2.5:1.5b", "nous-qwen2.5:1.5b"},
		{"llama3.2", "nous-llama3.2"},
		{"org/mymodel", "nous-mymodel"},
		{"deepseek-r1:8b", "nous-deepseek-r1:8b"},
	}

	for _, tt := range tests {
		t.Run(tt.modelName, func(t *testing.T) {
			c := NewCollector("")
			at := NewAutoTuner(c, tt.modelName)
			if at.TunedModelName() != tt.wantTuned {
				t.Errorf("TunedModelName() = %q, want %q", at.TunedModelName(), tt.wantTuned)
			}
		})
	}
}

func TestShouldTuneReturnsFalseWithTooFewPairs(t *testing.T) {
	c := newTestCollector(10, 0.9) // 10 pairs, well below default minPairs of 50
	at := NewAutoTuner(c, "qwen2.5:1.5b")

	if at.ShouldTune() {
		t.Error("ShouldTune should return false with too few pairs")
	}
}

func TestShouldTuneReturnsFalseWithLowQuality(t *testing.T) {
	// Create collector with many low quality pairs
	c := NewCollector("")
	// minQuality on collector is 0.6, so use 0.65 to pass collector filter
	// but fail the autotuner qualityFloor of 0.7
	for i := 0; i < 60; i++ {
		c.Collect("sys", "in", "out", nil, 0.65)
	}
	at := NewAutoTuner(c, "qwen2.5:1.5b")

	if at.ShouldTune() {
		t.Error("ShouldTune should return false with low average quality")
	}
}

func TestShouldTuneReturnsTrueWhenConditionsMet(t *testing.T) {
	c := newTestCollector(60, 0.9) // 60 pairs, quality 0.9 — both above defaults
	at := NewAutoTuner(c, "qwen2.5:1.5b")

	if !at.ShouldTune() {
		t.Error("ShouldTune should return true when all conditions met")
	}
}

func TestCheckReturnsFalseWithoutCreator(t *testing.T) {
	c := newTestCollector(60, 0.9)
	at := NewAutoTuner(c, "qwen2.5:1.5b")
	// No creator set

	if at.Check() {
		t.Error("Check should return false without a creator")
	}
}

func TestCheckReturnsTrueWithMockCreator(t *testing.T) {
	c := newTestCollector(60, 0.9)
	mock := &mockCreator{}
	at := NewAutoTuner(c, "qwen2.5:1.5b").WithCreator(mock)

	result := at.Check()

	if !result {
		t.Error("Check should return true when conditions met and creator available")
	}
	if !mock.called {
		t.Error("creator.CreateModel should have been called")
	}
	if mock.name != "nous-qwen2.5:1.5b" {
		t.Errorf("creator received name %q, want %q", mock.name, "nous-qwen2.5:1.5b")
	}
	if mock.modelfile == "" {
		t.Error("creator should receive a non-empty modelfile")
	}
}

func TestCheckReturnsFalseOnCreatorError(t *testing.T) {
	c := newTestCollector(60, 0.9)
	mock := &mockCreator{err: errMock}
	at := NewAutoTuner(c, "qwen2.5:1.5b").WithCreator(mock)

	result := at.Check()

	if result {
		t.Error("Check should return false when creator returns error")
	}
}

func TestFailedAttemptStillEntersCooldown(t *testing.T) {
	c := newTestCollector(60, 0.9)
	mock := &mockCreator{err: errMock}
	at := NewAutoTuner(c, "qwen2.5:1.5b").
		WithCreator(mock).
		WithCooldown(1 * time.Hour)

	if at.Check() {
		t.Fatal("first Check should fail when creator errors")
	}

	mock.called = false
	if at.Check() {
		t.Fatal("second Check should be blocked by cooldown after a failed attempt")
	}
	if mock.called {
		t.Fatal("creator should not be called again during cooldown after failure")
	}
}

var errMock = &mockError{}

type mockError struct{}

func (e *mockError) Error() string { return "mock error" }

func TestStatsReturnsCorrectValues(t *testing.T) {
	c := newTestCollector(60, 0.9)
	at := NewAutoTuner(c, "qwen2.5:1.5b")

	stats := at.Stats()

	if stats.PairCount != 60 {
		t.Errorf("PairCount = %d, want 60", stats.PairCount)
	}
	if math.Abs(stats.AvgQuality-0.9) > 1e-9 {
		t.Errorf("AvgQuality = %f, want 0.9", stats.AvgQuality)
	}
	if stats.MinPairs != 50 {
		t.Errorf("MinPairs = %d, want 50", stats.MinPairs)
	}
	if stats.QualityFloor != 0.7 {
		t.Errorf("QualityFloor = %f, want 0.7", stats.QualityFloor)
	}
	if stats.TunedName != "nous-qwen2.5:1.5b" {
		t.Errorf("TunedName = %q, want %q", stats.TunedName, "nous-qwen2.5:1.5b")
	}
	if !stats.Ready {
		t.Error("Ready should be true when conditions are met")
	}
	if !stats.LastTuneAt.IsZero() {
		t.Error("LastTuneAt should be zero when never tuned")
	}
	if !stats.NextTuneAfter.IsZero() {
		t.Error("NextTuneAfter should be zero when never tuned")
	}
}

func TestCooldownPreventsImmediateReTuning(t *testing.T) {
	c := newTestCollector(60, 0.9)
	mock := &mockCreator{}
	at := NewAutoTuner(c, "qwen2.5:1.5b").
		WithCreator(mock).
		WithCooldown(1 * time.Hour)

	// First check should trigger tuning
	if !at.Check() {
		t.Fatal("first Check should trigger tuning")
	}

	// Reset mock
	mock.called = false

	// Second check should be blocked by cooldown
	if at.Check() {
		t.Error("Check should return false during cooldown period")
	}
	if mock.called {
		t.Error("creator should not be called during cooldown")
	}
}

func TestForceCheckBypassesCooldown(t *testing.T) {
	c := newTestCollector(60, 0.9)
	mock := &mockCreator{}
	at := NewAutoTuner(c, "qwen2.5:1.5b").
		WithCreator(mock).
		WithCooldown(1 * time.Hour)

	// First check triggers tuning
	if !at.Check() {
		t.Fatal("first Check should trigger tuning")
	}

	// Reset mock
	mock.called = false

	// ForceCheck should bypass cooldown
	if !at.ForceCheck() {
		t.Error("ForceCheck should bypass cooldown and trigger tuning")
	}
	if !mock.called {
		t.Error("creator should be called on ForceCheck")
	}
}

func TestWithMinPairsSetter(t *testing.T) {
	c := newTestCollector(20, 0.9)
	at := NewAutoTuner(c, "qwen2.5:1.5b").WithMinPairs(10)

	// With minPairs=10 and 20 pairs available, should be ready
	if !at.ShouldTune() {
		t.Error("ShouldTune should return true with custom minPairs=10 and 20 pairs")
	}
}

func TestWithCooldownSetter(t *testing.T) {
	c := newTestCollector(60, 0.9)
	mock := &mockCreator{}
	at := NewAutoTuner(c, "qwen2.5:1.5b").
		WithCreator(mock).
		WithCooldown(1 * time.Millisecond) // very short cooldown

	at.Check()

	// Wait for cooldown to expire
	time.Sleep(5 * time.Millisecond)

	mock.called = false
	if !at.Check() {
		t.Error("Check should succeed after short cooldown expires")
	}
}

func TestWithCallbackSetter(t *testing.T) {
	c := newTestCollector(60, 0.9)
	mock := &mockCreator{}

	var messages []string
	callback := func(msg string) {
		messages = append(messages, msg)
	}

	at := NewAutoTuner(c, "qwen2.5:1.5b").
		WithCreator(mock).
		WithCallback(callback)

	at.Check()

	if len(messages) == 0 {
		t.Error("callback should have been called during Check")
	}

	// Should have both "Starting" and "complete" messages
	foundStart := false
	foundComplete := false
	for _, msg := range messages {
		if msg == "Starting auto fine-tune..." {
			foundStart = true
		}
		if len(msg) > 0 && msg[:10] == "Fine-tune " {
			foundComplete = true
		}
	}
	if !foundStart {
		t.Error("callback should receive 'Starting auto fine-tune...' message")
	}
	if !foundComplete {
		t.Error("callback should receive completion message")
	}
}

func TestCheckQuietSuppressesFailureNoise(t *testing.T) {
	c := newTestCollector(60, 0.9)
	mock := &mockCreator{err: errMock}

	var messages []string
	at := NewAutoTuner(c, "qwen2.5:1.5b").
		WithCreator(mock).
		WithCallback(func(msg string) { messages = append(messages, msg) })

	if at.CheckQuiet() {
		t.Fatal("CheckQuiet should still return false when creator errors")
	}
	if len(messages) != 0 {
		t.Fatalf("CheckQuiet should suppress failure chatter, got %v", messages)
	}
}

func TestABTestingRecordAndWinRate(t *testing.T) {
	c := newTestCollector(60, 0.9)
	at := NewAutoTuner(c, "qwen2.5:1.5b").WithABTesting(true)

	// Initial win rate should be 0.5 (no data)
	if wr := at.ABWinRate(); wr != 0.5 {
		t.Errorf("initial win rate = %f, want 0.5", wr)
	}

	// Record some results
	at.RecordABResult("tuned")
	at.RecordABResult("tuned")
	at.RecordABResult("base")

	if wr := at.ABWinRate(); math.Abs(wr-2.0/3.0) > 0.01 {
		t.Errorf("win rate = %f, want ~0.667", wr)
	}

	stats := at.Stats()
	if stats.ABTotalTrials != 3 {
		t.Errorf("ABTotalTrials = %d, want 3", stats.ABTotalTrials)
	}
	if stats.ABTunedWins != 2 {
		t.Errorf("ABTunedWins = %d, want 2", stats.ABTunedWins)
	}
	if stats.ABBaseWins != 1 {
		t.Errorf("ABBaseWins = %d, want 1", stats.ABBaseWins)
	}
}

func TestShouldUseTunedDefaultsTrue(t *testing.T) {
	c := newTestCollector(60, 0.9)
	at := NewAutoTuner(c, "qwen2.5:1.5b").WithABTesting(true)

	// With fewer than 10 trials, should default to tuned
	for i := 0; i < 9; i++ {
		at.RecordABResult("base")
	}
	if !at.ShouldUseTuned() {
		t.Error("ShouldUseTuned should return true with <10 trials")
	}
}

func TestShouldUseTunedRejectsAfterEnoughTrials(t *testing.T) {
	c := newTestCollector(60, 0.9)
	at := NewAutoTuner(c, "qwen2.5:1.5b").WithABTesting(true)

	// Tuned wins less than 50% after 10+ trials
	for i := 0; i < 7; i++ {
		at.RecordABResult("base")
	}
	for i := 0; i < 3; i++ {
		at.RecordABResult("tuned")
	}
	// 10 trials, tuned wins 3/10 = 30%
	if at.ShouldUseTuned() {
		t.Error("ShouldUseTuned should return false when tuned wins <50%")
	}
}

func TestAdaptiveCooldownIncreasesOnFailure(t *testing.T) {
	c := newTestCollector(60, 0.9)
	mock := &mockCreator{err: errMock}
	at := NewAutoTuner(c, "qwen2.5:1.5b").
		WithCreator(mock).
		WithCooldown(1 * time.Millisecond).
		WithAdaptiveCooldown(true)

	// First failure
	at.Check()
	stats := at.Stats()
	if stats.ConsecutiveFails != 1 {
		t.Errorf("consecutive fails = %d, want 1", stats.ConsecutiveFails)
	}
	// Cooldown should be doubled from base
	if stats.EffectiveCooldown < 2*time.Millisecond {
		t.Errorf("cooldown should increase after failure, got %v", stats.EffectiveCooldown)
	}
}

func TestAdaptiveCooldownResetsOnSuccess(t *testing.T) {
	c := newTestCollector(60, 0.9)
	failMock := &mockCreator{err: errMock}
	at := NewAutoTuner(c, "qwen2.5:1.5b").
		WithCreator(failMock).
		WithCooldown(1 * time.Millisecond).
		WithAdaptiveCooldown(true)

	// Cause 2 failures
	at.Check()
	time.Sleep(5 * time.Millisecond)
	at.Check()

	stats := at.Stats()
	if stats.ConsecutiveFails != 2 {
		t.Errorf("consecutive fails = %d, want 2", stats.ConsecutiveFails)
	}

	// Now succeed
	successMock := &mockCreator{}
	at.WithCreator(successMock)
	// Wait for cooldown to expire before checking
	time.Sleep(time.Duration(stats.EffectiveCooldown) + 10*time.Millisecond)
	at.Check()

	stats = at.Stats()
	if stats.ConsecutiveFails != 0 {
		t.Errorf("consecutive fails should reset to 0 on success, got %d", stats.ConsecutiveFails)
	}
}

func TestAdaptiveCooldownCapsAt24Hours(t *testing.T) {
	c := newTestCollector(60, 0.9)
	at := NewAutoTuner(c, "qwen2.5:1.5b").
		WithCooldown(1 * time.Hour).
		WithAdaptiveCooldown(true)

	// Simulate many failures
	at.mu.Lock()
	at.consecutiveFails = 10
	at.mu.Unlock()

	cd := at.adaptiveCooldownValue()
	if cd > 24*time.Hour {
		t.Errorf("cooldown should cap at 24h, got %v", cd)
	}
	if cd != 24*time.Hour {
		t.Errorf("cooldown should be exactly 24h with many failures, got %v", cd)
	}
}
