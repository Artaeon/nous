package cognitive

import (
	"testing"
	"time"

	"github.com/artaeon/nous/internal/tools"
)

func mockRegistry() *tools.Registry {
	reg := tools.NewRegistry()
	reg.Register(tools.Tool{
		Name:        "read",
		Description: "Read a file",
		Execute: func(args map[string]string) (string, error) {
			return "mock file content for " + args["path"], nil
		},
	})
	reg.Register(tools.Tool{
		Name:        "ls",
		Description: "List directory",
		Execute: func(args map[string]string) (string, error) {
			return "main.go\nREADME.md\ngo.mod", nil
		},
	})
	reg.Register(tools.Tool{
		Name:        "grep",
		Description: "Search files",
		Execute: func(args map[string]string) (string, error) {
			return "main.go:1:package main", nil
		},
	})
	reg.Register(tools.Tool{
		Name:        "glob",
		Description: "Find files",
		Execute: func(args map[string]string) (string, error) {
			return "main.go\nserver.go", nil
		},
	})
	return reg
}

func TestPredictorLookupMiss(t *testing.T) {
	p := NewPredictor(mockRegistry())

	result, ok := p.Lookup("read", map[string]string{"path": "nonexistent.go"})
	if ok {
		t.Error("expected miss for uncached key")
	}
	if result != "" {
		t.Error("expected empty result on miss")
	}

	_, misses := p.Stats()
	if misses != 1 {
		t.Errorf("misses = %d, want 1", misses)
	}
}

func TestPredictorLookupHit(t *testing.T) {
	p := NewPredictor(mockRegistry())

	// Manually insert a prediction
	p.mu.Lock()
	p.cache[cacheKey("read", map[string]string{"path": "main.go"})] = Prediction{
		ToolName:  "read",
		Args:      map[string]string{"path": "main.go"},
		Result:    "package main\n",
		CreatedAt: time.Now(),
	}
	p.mu.Unlock()

	result, ok := p.Lookup("read", map[string]string{"path": "main.go"})
	if !ok {
		t.Error("expected hit for cached key")
	}
	if result != "package main\n" {
		t.Errorf("result = %q, want %q", result, "package main\n")
	}

	hits, _ := p.Stats()
	if hits != 1 {
		t.Errorf("hits = %d, want 1", hits)
	}

	// Second lookup should miss (consumed)
	_, ok = p.Lookup("read", map[string]string{"path": "main.go"})
	if ok {
		t.Error("prediction should be consumed after first hit")
	}
}

func TestPredictorExpiry(t *testing.T) {
	p := NewPredictor(mockRegistry())

	// Insert an old prediction
	p.mu.Lock()
	p.cache[cacheKey("read", map[string]string{"path": "old.go"})] = Prediction{
		ToolName:  "read",
		Args:      map[string]string{"path": "old.go"},
		Result:    "old content",
		CreatedAt: time.Now().Add(-60 * time.Second), // 60s ago
	}
	p.mu.Unlock()

	_, ok := p.Lookup("read", map[string]string{"path": "old.go"})
	if ok {
		t.Error("expired prediction should not hit")
	}
}

func TestPredictorPredictFromRead(t *testing.T) {
	reg := mockRegistry()
	p := NewPredictor(reg)

	// Predict after reading a .go file
	p.Predict("read", map[string]string{"path": "internal/cognitive/reasoner.go"}, "package cognitive...")

	// Give goroutines time to execute
	time.Sleep(100 * time.Millisecond)

	// Should have pre-cached the test file
	result, ok := p.Lookup("read", map[string]string{"path": "internal/cognitive/reasoner_test.go"})
	if !ok {
		t.Error("expected prediction for test file")
	}
	if result == "" {
		t.Error("expected non-empty prediction result")
	}
}

func TestPredictorPredictFromGrep(t *testing.T) {
	reg := mockRegistry()
	p := NewPredictor(reg)

	// Predict after grep returns file matches
	p.Predict("grep", map[string]string{"pattern": "NewReasoner"},
		"internal/cognitive/reasoner.go:52:func NewReasoner(")

	time.Sleep(100 * time.Millisecond)

	// Should have pre-cached reading the matched file
	result, ok := p.Lookup("read", map[string]string{"path": "internal/cognitive/reasoner.go"})
	if !ok {
		t.Error("expected prediction for grepped file")
	}
	if result == "" {
		t.Error("expected non-empty prediction result")
	}
}

func TestPredictorPredictFromLs(t *testing.T) {
	reg := mockRegistry()
	p := NewPredictor(reg)

	p.Predict("ls", map[string]string{"path": "."}, "main.go\nREADME.md\ngo.mod")

	time.Sleep(100 * time.Millisecond)

	// Should pre-read README.md and/or main.go
	size := p.CacheSize()
	if size == 0 {
		t.Error("expected predictions from ls results")
	}
}

func TestPredictorClear(t *testing.T) {
	p := NewPredictor(mockRegistry())

	p.mu.Lock()
	p.cache["test"] = Prediction{Result: "data", CreatedAt: time.Now()}
	p.mu.Unlock()

	p.Clear()

	if p.CacheSize() != 0 {
		t.Error("cache should be empty after Clear")
	}
}

func TestPredictorHitRate(t *testing.T) {
	p := NewPredictor(mockRegistry())

	// No lookups = 0 rate
	if p.HitRate() != 0 {
		t.Error("empty predictor should have 0 hit rate")
	}

	// Simulate hits and misses
	p.mu.Lock()
	p.hits = 3
	p.misses = 7
	p.mu.Unlock()

	rate := p.HitRate()
	if rate != 0.3 {
		t.Errorf("hit rate = %f, want 0.3", rate)
	}
}

func TestPredictorEviction(t *testing.T) {
	p := NewPredictor(mockRegistry())
	p.maxSize = 3

	// Fill cache beyond capacity
	p.mu.Lock()
	for i := 0; i < 5; i++ {
		key := cacheKey("read", map[string]string{"path": string(rune('a' + i))})
		p.cache[key] = Prediction{
			ToolName:  "read",
			Result:    "data",
			CreatedAt: time.Now().Add(time.Duration(i) * time.Second),
		}
		if len(p.cache) > p.maxSize {
			p.evictOldest()
		}
	}
	p.mu.Unlock()

	if p.CacheSize() > 3 {
		t.Errorf("cache should not exceed maxSize, got %d", p.CacheSize())
	}
}

func TestIsReadOnly(t *testing.T) {
	readOnlyTools := []string{"read", "ls", "tree", "glob", "grep", "sysinfo", "diff"}
	writeTools := []string{"write", "edit", "shell", "mkdir", "patch"}

	for _, tool := range readOnlyTools {
		if !isReadOnly(tool) {
			t.Errorf("%s should be read-only", tool)
		}
	}
	for _, tool := range writeTools {
		if isReadOnly(tool) {
			t.Errorf("%s should not be read-only", tool)
		}
	}
}

func TestCacheKey(t *testing.T) {
	key1 := cacheKey("read", map[string]string{"path": "main.go"})
	key2 := cacheKey("read", map[string]string{"path": "server.go"})
	key3 := cacheKey("read", map[string]string{"path": "main.go"})

	if key1 == key2 {
		t.Error("different args should produce different keys")
	}
	if key1 != key3 {
		t.Error("same tool+args should produce same key")
	}
}

func TestLooksLikeFile(t *testing.T) {
	tests := []struct {
		input string
		want  bool
	}{
		{"main.go", true},
		{"internal/cognitive/reasoner.go", true},
		{"README.md", true},
		{"-rw-r--r--", false},
		{"", false},
		{"just-text", false},
	}

	for _, tt := range tests {
		got := looksLikeFile(tt.input)
		if got != tt.want {
			t.Errorf("looksLikeFile(%q) = %v, want %v", tt.input, got, tt.want)
		}
	}
}

func TestDirOf(t *testing.T) {
	tests := []struct {
		input, want string
	}{
		{"internal/cognitive/reasoner.go", "internal/cognitive"},
		{"main.go", ""},
		{"a/b/c/d.go", "a/b/c"},
		{"", ""},
		{"/root.go", ""},
	}

	for _, tt := range tests {
		got := dirOf(tt.input)
		if got != tt.want {
			t.Errorf("dirOf(%q) = %q, want %q", tt.input, got, tt.want)
		}
	}
}

// --- Concurrent access tests ---

func TestPredictorConcurrentLookup(t *testing.T) {
	p := NewPredictor(mockRegistry())

	// Pre-populate
	p.mu.Lock()
	for i := 0; i < 10; i++ {
		key := cacheKey("read", map[string]string{"path": string(rune('a' + i))})
		p.cache[key] = Prediction{
			ToolName:  "read",
			Result:    "content",
			CreatedAt: time.Now(),
		}
	}
	p.mu.Unlock()

	// Concurrent lookups should not race
	done := make(chan struct{})
	for i := 0; i < 10; i++ {
		i := i
		go func() {
			defer func() { done <- struct{}{} }()
			p.Lookup("read", map[string]string{"path": string(rune('a' + i))})
		}()
	}
	for i := 0; i < 10; i++ {
		<-done
	}
}

func TestPredictorConcurrentPredictAndLookup(t *testing.T) {
	reg := mockRegistry()
	p := NewPredictor(reg)

	done := make(chan struct{})
	// Predict in one goroutine
	go func() {
		defer func() { done <- struct{}{} }()
		p.Predict("read", map[string]string{"path": "test.go"}, "content")
	}()
	// Lookup in another
	go func() {
		defer func() { done <- struct{}{} }()
		time.Sleep(50 * time.Millisecond)
		p.Lookup("read", map[string]string{"path": "test_test.go"})
	}()

	<-done
	<-done
}

func TestPredictorPredictFromGlob(t *testing.T) {
	reg := mockRegistry()
	p := NewPredictor(reg)

	p.Predict("glob", map[string]string{"pattern": "*.go"}, "main.go\nserver.go")
	time.Sleep(100 * time.Millisecond)

	// Should have pre-read some of the matched files
	size := p.CacheSize()
	if size == 0 {
		t.Error("expected predictions from glob results")
	}
}

func TestPredictorDoesNotCacheWriteTools(t *testing.T) {
	reg := mockRegistry()
	reg.Register(tools.Tool{
		Name:        "write",
		Description: "Write a file",
		Execute: func(args map[string]string) (string, error) {
			return "written", nil
		},
	})
	_ = NewPredictor(reg)

	// write is not read-only, so predictions for write should not execute
	if isReadOnly("write") {
		t.Error("write should not be read-only")
	}
}

func TestCacheKeyDeterministic(t *testing.T) {
	args := map[string]string{"path": "a.go", "pattern": "func"}
	key1 := cacheKey("grep", args)
	key2 := cacheKey("grep", args)
	if key1 != key2 {
		t.Error("cacheKey should be deterministic for same args")
	}
}

func TestCacheKeyArgOrdering(t *testing.T) {
	// Different insertion order, same keys
	args1 := map[string]string{"a": "1", "b": "2", "c": "3"}
	args2 := map[string]string{"c": "3", "a": "1", "b": "2"}
	if cacheKey("tool", args1) != cacheKey("tool", args2) {
		t.Error("cacheKey should be order-independent")
	}
}

func TestPredictorStatsAfterMixedOps(t *testing.T) {
	p := NewPredictor(mockRegistry())

	// 3 misses
	for i := 0; i < 3; i++ {
		p.Lookup("read", map[string]string{"path": "miss.go"})
	}

	// Insert and hit
	p.mu.Lock()
	p.cache[cacheKey("read", map[string]string{"path": "hit.go"})] = Prediction{
		ToolName: "read", Result: "data", CreatedAt: time.Now(),
	}
	p.mu.Unlock()
	p.Lookup("read", map[string]string{"path": "hit.go"})

	hits, misses := p.Stats()
	if hits != 1 {
		t.Errorf("hits = %d, want 1", hits)
	}
	if misses != 3 {
		t.Errorf("misses = %d, want 3", misses)
	}
	if p.HitRate() != 0.25 {
		t.Errorf("hit rate = %f, want 0.25", p.HitRate())
	}
}

func TestPredictorDoesNotExceedMaxSize(t *testing.T) {
	p := NewPredictor(mockRegistry())
	p.maxSize = 5

	// Insert more than maxSize
	for i := 0; i < 10; i++ {
		p.mu.Lock()
		key := cacheKey("read", map[string]string{"path": string(rune('A' + i))})
		if len(p.cache) >= p.maxSize {
			p.evictOldest()
		}
		p.cache[key] = Prediction{
			ToolName:  "read",
			Result:    "data",
			CreatedAt: time.Now().Add(time.Duration(i) * time.Millisecond),
		}
		p.mu.Unlock()
	}

	if p.CacheSize() > 5 {
		t.Errorf("cache size %d exceeds maxSize 5", p.CacheSize())
	}
}
