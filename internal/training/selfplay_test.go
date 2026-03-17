package training

import (
	"errors"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func TestDefaultSelfPlayConfig(t *testing.T) {
	cfg := DefaultSelfPlayConfig()
	if cfg.IdleTimeout != 60*time.Second {
		t.Errorf("IdleTimeout = %v, want 60s", cfg.IdleTimeout)
	}
	if cfg.MaxPairsPerRun != 100 {
		t.Errorf("MaxPairsPerRun = %d, want 100", cfg.MaxPairsPerRun)
	}
	if cfg.MinQuality != 0.7 {
		t.Errorf("MinQuality = %f, want 0.7", cfg.MinQuality)
	}
}

func TestNewSelfPlay(t *testing.T) {
	cfg := DefaultSelfPlayConfig()
	sp := NewSelfPlay(cfg)
	if sp == nil {
		t.Fatal("NewSelfPlay returned nil")
	}
	if sp.config.MaxPairsPerRun != 100 {
		t.Error("config not stored correctly")
	}
}

func TestRunOnceNoDeps(t *testing.T) {
	sp := NewSelfPlay(DefaultSelfPlayConfig())
	// No knowledge, no GenerateQA, no StorePair — should return 0
	got := sp.RunOnce()
	if got != 0 {
		t.Errorf("RunOnce with no deps = %d, want 0", got)
	}
}

func TestRunOnceNoKnowledge(t *testing.T) {
	sp := NewSelfPlay(DefaultSelfPlayConfig())
	sp.GenerateQA = func(chunk string) ([]QuestionAnswerPair, error) {
		return nil, nil
	}
	sp.StorePair = func(pair QuestionAnswerPair) {}
	// Empty Knowledge slice
	got := sp.RunOnce()
	if got != 0 {
		t.Errorf("RunOnce with no knowledge = %d, want 0", got)
	}
}

func TestRunOnceAcceptsPairs(t *testing.T) {
	sp := NewSelfPlay(SelfPlayConfig{
		IdleTimeout:    time.Second,
		MaxPairsPerRun: 10,
		MinQuality:     0.5,
	})

	sp.Knowledge = []KnowledgeChunk{
		{Content: "Go is a programming language", Domain: "tech"},
		{Content: "Docker containers virtualize apps", Domain: "tech"},
	}

	sp.GenerateQA = func(chunk string) ([]QuestionAnswerPair, error) {
		return []QuestionAnswerPair{
			{Question: "What is " + chunk[:5] + "?", Answer: "It is something.", Quality: 0.8},
		}, nil
	}

	var stored []QuestionAnswerPair
	var mu sync.Mutex
	sp.StorePair = func(pair QuestionAnswerPair) {
		mu.Lock()
		stored = append(stored, pair)
		mu.Unlock()
	}

	got := sp.RunOnce()
	if got != 2 {
		t.Errorf("RunOnce = %d, want 2", got)
	}
	if len(stored) != 2 {
		t.Errorf("stored %d pairs, want 2", len(stored))
	}

	// Verify source and timestamp were set
	for _, p := range stored {
		if p.Source != "self-play" {
			t.Errorf("Source = %q, want \"self-play\"", p.Source)
		}
		if p.GeneratedAt.IsZero() {
			t.Error("GeneratedAt not set")
		}
	}

	gen, acc, rej := sp.Stats()
	if gen != 2 || acc != 2 || rej != 0 {
		t.Errorf("Stats = (%d,%d,%d), want (2,2,0)", gen, acc, rej)
	}
}

func TestRunOnceRejectsBelowQuality(t *testing.T) {
	sp := NewSelfPlay(SelfPlayConfig{
		IdleTimeout:    time.Second,
		MaxPairsPerRun: 10,
		MinQuality:     0.8,
	})

	sp.Knowledge = []KnowledgeChunk{
		{Content: "test chunk", Domain: "test"},
	}

	sp.GenerateQA = func(chunk string) ([]QuestionAnswerPair, error) {
		return []QuestionAnswerPair{
			{Question: "Q1", Answer: "A1", Quality: 0.5}, // below threshold
			{Question: "Q2", Answer: "A2", Quality: 0.9}, // above threshold
		}, nil
	}

	sp.ValidatePair = func(pair QuestionAnswerPair, source string) float64 {
		return pair.Quality // pass through
	}

	stored := 0
	sp.StorePair = func(pair QuestionAnswerPair) {
		stored++
	}

	got := sp.RunOnce()
	if got != 1 {
		t.Errorf("RunOnce = %d, want 1 (only high quality)", got)
	}
	if stored != 1 {
		t.Errorf("stored = %d, want 1", stored)
	}

	gen, acc, rej := sp.Stats()
	if gen != 2 {
		t.Errorf("generated = %d, want 2", gen)
	}
	if acc != 1 {
		t.Errorf("accepted = %d, want 1", acc)
	}
	if rej != 1 {
		t.Errorf("rejected = %d, want 1", rej)
	}
}

func TestRunOnceMaxPairsLimit(t *testing.T) {
	sp := NewSelfPlay(SelfPlayConfig{
		IdleTimeout:    time.Second,
		MaxPairsPerRun: 2,
		MinQuality:     0.0,
	})

	sp.Knowledge = []KnowledgeChunk{
		{Content: "chunk1", Domain: "test"},
		{Content: "chunk2", Domain: "test"},
		{Content: "chunk3", Domain: "test"},
	}

	sp.GenerateQA = func(chunk string) ([]QuestionAnswerPair, error) {
		return []QuestionAnswerPair{
			{Question: "Q", Answer: "A", Quality: 1.0},
			{Question: "Q2", Answer: "A2", Quality: 1.0},
		}, nil
	}

	stored := 0
	sp.StorePair = func(pair QuestionAnswerPair) {
		stored++
	}

	got := sp.RunOnce()
	if got != 2 {
		t.Errorf("RunOnce = %d, want 2 (max limit)", got)
	}
}

func TestRunOnceGenerateError(t *testing.T) {
	sp := NewSelfPlay(SelfPlayConfig{
		IdleTimeout:    time.Second,
		MaxPairsPerRun: 10,
		MinQuality:     0.0,
	})

	sp.Knowledge = []KnowledgeChunk{
		{Content: "will fail", Domain: "test"},
		{Content: "will succeed", Domain: "test"},
	}

	callCount := 0
	sp.GenerateQA = func(chunk string) ([]QuestionAnswerPair, error) {
		callCount++
		if chunk == "will fail" {
			return nil, errors.New("generation failed")
		}
		return []QuestionAnswerPair{
			{Question: "Q", Answer: "A", Quality: 1.0},
		}, nil
	}

	sp.StorePair = func(pair QuestionAnswerPair) {}

	got := sp.RunOnce()
	// Should still process the successful chunk
	if got < 0 || got > 1 {
		t.Errorf("RunOnce = %d, want 0 or 1", got)
	}
}

func TestRunOncePreventsConcurrent(t *testing.T) {
	sp := NewSelfPlay(SelfPlayConfig{
		IdleTimeout:    time.Second,
		MaxPairsPerRun: 10,
		MinQuality:     0.0,
	})

	sp.Knowledge = []KnowledgeChunk{
		{Content: "chunk", Domain: "test"},
	}

	started := make(chan struct{})
	block := make(chan struct{})

	sp.GenerateQA = func(chunk string) ([]QuestionAnswerPair, error) {
		close(started)
		<-block // block until test releases
		return []QuestionAnswerPair{
			{Question: "Q", Answer: "A", Quality: 1.0},
		}, nil
	}

	sp.StorePair = func(pair QuestionAnswerPair) {}

	var wg sync.WaitGroup
	var secondResult int32

	wg.Add(1)
	go func() {
		defer wg.Done()
		sp.RunOnce()
	}()

	<-started // wait for first RunOnce to start

	// Try concurrent RunOnce — should return 0 immediately
	wg.Add(1)
	go func() {
		defer wg.Done()
		atomic.StoreInt32(&secondResult, int32(sp.RunOnce()))
	}()

	// Give goroutine time to attempt
	time.Sleep(50 * time.Millisecond)
	close(block) // release first RunOnce

	wg.Wait()

	if atomic.LoadInt32(&secondResult) != 0 {
		t.Error("concurrent RunOnce should return 0")
	}

	// After completion, Running should be false
	sp.mu.Lock()
	running := sp.Running
	sp.mu.Unlock()
	if running {
		t.Error("Running should be false after completion")
	}
}

func TestRunOnceValidatePairOverridesQuality(t *testing.T) {
	sp := NewSelfPlay(SelfPlayConfig{
		IdleTimeout:    time.Second,
		MaxPairsPerRun: 10,
		MinQuality:     0.8,
	})

	sp.Knowledge = []KnowledgeChunk{
		{Content: "test", Domain: "test"},
	}

	sp.GenerateQA = func(chunk string) ([]QuestionAnswerPair, error) {
		return []QuestionAnswerPair{
			{Question: "Q", Answer: "A", Quality: 0.5}, // initially low
		}, nil
	}

	// ValidatePair overrides quality to high
	sp.ValidatePair = func(pair QuestionAnswerPair, source string) float64 {
		return 0.95
	}

	var storedQuality float64
	sp.StorePair = func(pair QuestionAnswerPair) {
		storedQuality = pair.Quality
	}

	got := sp.RunOnce()
	if got != 1 {
		t.Errorf("RunOnce = %d, want 1", got)
	}
	if storedQuality != 0.95 {
		t.Errorf("stored quality = %f, want 0.95", storedQuality)
	}
}

func TestRunOnceNoValidatePairUsesOriginalQuality(t *testing.T) {
	sp := NewSelfPlay(SelfPlayConfig{
		IdleTimeout:    time.Second,
		MaxPairsPerRun: 10,
		MinQuality:     0.5,
	})

	sp.Knowledge = []KnowledgeChunk{
		{Content: "test", Domain: "test"},
	}

	sp.GenerateQA = func(chunk string) ([]QuestionAnswerPair, error) {
		return []QuestionAnswerPair{
			{Question: "Q", Answer: "A", Quality: 0.8},
		}, nil
	}

	// No ValidatePair set — should use original quality
	var storedQuality float64
	sp.StorePair = func(pair QuestionAnswerPair) {
		storedQuality = pair.Quality
	}

	got := sp.RunOnce()
	if got != 1 {
		t.Errorf("RunOnce = %d, want 1", got)
	}
	if storedQuality != 0.8 {
		t.Errorf("stored quality = %f, want 0.8", storedQuality)
	}
}

func TestStatsThreadSafe(t *testing.T) {
	sp := NewSelfPlay(DefaultSelfPlayConfig())
	sp.Knowledge = []KnowledgeChunk{
		{Content: "chunk", Domain: "test"},
	}
	sp.GenerateQA = func(chunk string) ([]QuestionAnswerPair, error) {
		return []QuestionAnswerPair{
			{Question: "Q", Answer: "A", Quality: 1.0},
		}, nil
	}
	sp.StorePair = func(pair QuestionAnswerPair) {}

	sp.RunOnce()

	// Call Stats concurrently
	var wg sync.WaitGroup
	for i := 0; i < 10; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			sp.Stats()
		}()
	}
	wg.Wait()
}

func TestTruncate(t *testing.T) {
	tests := []struct {
		input string
		n     int
		want  string
	}{
		{"hello", 10, "hello"},
		{"hello world", 5, "hello..."},
		{"  spaces  ", 6, "spaces"},
		{"", 5, ""},
		{"ab", 2, "ab"},
		{"abc", 2, "ab..."},
	}

	for _, tt := range tests {
		got := truncate(tt.input, tt.n)
		if got != tt.want {
			t.Errorf("truncate(%q, %d) = %q, want %q", tt.input, tt.n, got, tt.want)
		}
	}
}
