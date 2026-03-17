package cognitive

import (
	"sync"
	"testing"
	"time"
)

func TestNewPredictiveWarmer(t *testing.T) {
	pw := NewPredictiveWarmer(nil, nil)
	if pw == nil {
		t.Fatal("NewPredictiveWarmer returned nil")
	}
	if pw.maxHistory != 50 {
		t.Errorf("maxHistory = %d, want 50", pw.maxHistory)
	}
}

func TestPredictFileFollowUp(t *testing.T) {
	pw := NewPredictiveWarmer(nil, nil)
	predictions := pw.predict("read main.go")

	if len(predictions) == 0 {
		t.Fatal("expected predictions for file query")
	}

	found := false
	for _, p := range predictions {
		if p == "main.go test" || p == "explain main.go" {
			found = true
			break
		}
	}
	if !found {
		t.Errorf("expected file-related prediction, got %v", predictions)
	}
}

func TestPredictWhatIsFollowUp(t *testing.T) {
	pw := NewPredictiveWarmer(nil, nil)
	predictions := pw.predict("what is kubernetes?")

	found := false
	for _, p := range predictions {
		if p == "how does kubernetes work" {
			found = true
			break
		}
	}
	if !found {
		t.Errorf("expected 'how does kubernetes work', got %v", predictions)
	}
}

func TestPredictHowToFollowUp(t *testing.T) {
	pw := NewPredictiveWarmer(nil, nil)
	predictions := pw.predict("how to deploy containers")

	found := false
	for _, p := range predictions {
		if p == "deploy containers example" {
			found = true
			break
		}
	}
	if !found {
		t.Errorf("expected 'deploy containers example', got %v", predictions)
	}
}

func TestPredictRepeatedTopics(t *testing.T) {
	pw := NewPredictiveWarmer(nil, nil)

	// Add history with repeated topic
	pw.mu.Lock()
	pw.history = []queryRecord{
		{query: "what is docker", at: time.Now()},
		{query: "how does docker work", at: time.Now()},
		{query: "docker compose example", at: time.Now()},
	}
	pw.mu.Unlock()

	topics := pw.extractRepeatedTopics()
	found := false
	for _, t := range topics {
		if t == "docker" {
			found = true
			break
		}
	}
	if !found {
		t.Errorf("expected 'docker' in repeated topics, got %v", topics)
	}
}

func TestAfterQueryRecordsHistory(t *testing.T) {
	pw := NewPredictiveWarmer(nil, nil)
	pw.AfterQuery("hello world")

	pw.mu.Lock()
	defer pw.mu.Unlock()
	if len(pw.history) != 1 {
		t.Errorf("history length = %d, want 1", len(pw.history))
	}
	if pw.history[0].query != "hello world" {
		t.Errorf("query = %q, want 'hello world'", pw.history[0].query)
	}
}

func TestAfterQueryTrimsHistory(t *testing.T) {
	pw := NewPredictiveWarmer(nil, nil)
	pw.maxHistory = 3

	pw.AfterQuery("query1")
	pw.AfterQuery("query2")
	pw.AfterQuery("query3")
	pw.AfterQuery("query4")

	pw.mu.Lock()
	defer pw.mu.Unlock()
	if len(pw.history) != 3 {
		t.Errorf("history length = %d, want 3", len(pw.history))
	}
	if pw.history[0].query != "query2" {
		t.Errorf("oldest query = %q, want 'query2'", pw.history[0].query)
	}
}

func TestCheckHit(t *testing.T) {
	pw := NewPredictiveWarmer(nil, nil)
	pw.mu.Lock()
	pw.predictions = []string{"how does kubernetes work", "kubernetes example"}
	pw.mu.Unlock()

	if !pw.CheckHit("how does kubernetes work") {
		t.Error("should match exact prediction")
	}

	total, hits := pw.Stats()
	if hits != 1 {
		t.Errorf("hits = %d, want 1", hits)
	}
	_ = total
}

func TestCheckHitMiss(t *testing.T) {
	pw := NewPredictiveWarmer(nil, nil)
	pw.mu.Lock()
	pw.predictions = []string{"how does kubernetes work"}
	pw.mu.Unlock()

	if pw.CheckHit("tell me a joke") {
		t.Error("should not match unrelated query")
	}
}

func TestWarmPredictionsWithEmbedCache(t *testing.T) {
	cache := NewEmbedCache(100, 5*time.Minute)
	pw := NewPredictiveWarmer(nil, cache)

	called := 0
	var mu sync.Mutex
	pw.SetEmbedFunc(func(text string) []float64 {
		mu.Lock()
		called++
		mu.Unlock()
		return []float64{0.1, 0.2, 0.3}
	})

	pw.warmPredictions([]string{"test query 1", "test query 2"})

	mu.Lock()
	c := called
	mu.Unlock()
	if c != 2 {
		t.Errorf("embedFunc called %d times, want 2", c)
	}

	// Check cache was populated
	vec := cache.Get("test query 1")
	if vec == nil {
		t.Error("cache should have 'test query 1'")
	}
}

func TestPredictionsLimitedTo5(t *testing.T) {
	pw := NewPredictiveWarmer(nil, nil)

	// Add lots of history to generate many predictions
	pw.mu.Lock()
	for i := 0; i < 20; i++ {
		pw.history = append(pw.history, queryRecord{query: "topic alpha beta", at: time.Now()})
	}
	pw.mu.Unlock()

	predictions := pw.predict("what is alpha.go?")
	if len(predictions) > 5 {
		t.Errorf("predictions should be capped at 5, got %d", len(predictions))
	}
}

func TestContainsFileRef(t *testing.T) {
	tests := []struct {
		input string
		want  bool
	}{
		{"read main.go", true},
		{"check config.yaml", true},
		{"hello world", false},
		{"app.js is broken", true},
		{"Dockerfile", false},
	}
	for _, tt := range tests {
		if got := containsFileRef(tt.input); got != tt.want {
			t.Errorf("containsFileRef(%q) = %v, want %v", tt.input, got, tt.want)
		}
	}
}

func TestExtractFileRef(t *testing.T) {
	if got := extractFileRef("read main.go please"); got != "main.go" {
		t.Errorf("extractFileRef = %q, want 'main.go'", got)
	}
	if got := extractFileRef("no file here"); got != "" {
		t.Errorf("extractFileRef = %q, want empty", got)
	}
}

func TestIsStopWord(t *testing.T) {
	if !isStopWord("about") {
		t.Error("'about' should be a stop word")
	}
	if isStopWord("kubernetes") {
		t.Error("'kubernetes' should not be a stop word")
	}
}

func TestStatsConcurrency(t *testing.T) {
	pw := NewPredictiveWarmer(nil, nil)
	var wg sync.WaitGroup
	for i := 0; i < 10; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			pw.AfterQuery("concurrent query")
			pw.Stats()
		}()
	}
	wg.Wait()
}
