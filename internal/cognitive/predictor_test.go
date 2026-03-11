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
	}

	for _, tt := range tests {
		got := dirOf(tt.input)
		if got != tt.want {
			t.Errorf("dirOf(%q) = %q, want %q", tt.input, got, tt.want)
		}
	}
}
