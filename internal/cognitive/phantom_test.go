package cognitive

import (
	"strings"
	"testing"
)

// --- Phantom Reasoning Tests ---

func TestPhantomReasonerCreation(t *testing.T) {
	pr := NewPhantomReasoner()
	if pr == nil {
		t.Fatal("NewPhantomReasoner should not return nil")
	}
}

func TestPhantomChainEmpty(t *testing.T) {
	pr := NewPhantomReasoner()
	chain := pr.BuildChain("test", nil)
	if chain == nil {
		t.Fatal("should return empty chain, not nil")
	}
	if len(chain.Steps) != 0 {
		t.Error("empty steps should produce empty chain")
	}
}

func TestPhantomChainGrep(t *testing.T) {
	pr := NewPhantomReasoner()
	steps := []synthStep{
		{
			Tool: "grep",
			Args: map[string]string{"pattern": "NewReasoner"},
			Result: "reasoner.go:74:func NewReasoner(board *blackboard.Blackboard...\n" +
				"reasoner.go:85:func (r *Reasoner) Name() string...\n" +
				"main.go:100:r := cognitive.NewReasoner(board, llm, tools)",
		},
	}

	chain := pr.BuildChain("find NewReasoner", steps)
	if len(chain.Steps) != 1 {
		t.Fatalf("should have 1 step, got %d", len(chain.Steps))
	}
	if !strings.Contains(chain.Steps[0].Action, "NewReasoner") {
		t.Error("step should mention the search pattern")
	}
	if !strings.Contains(chain.Steps[0].Observation, "3 matches") {
		t.Errorf("should count 3 matches, got: %s", chain.Steps[0].Observation)
	}
	if !strings.Contains(chain.Steps[0].Fact, "3 matches") {
		t.Errorf("fact should state 3 matches, got: %s", chain.Steps[0].Fact)
	}
	if !strings.Contains(chain.FullContext, "Step 1") {
		t.Error("full context should include step numbering")
	}
	if !strings.Contains(chain.FullContext, "Therefore") {
		t.Error("full context should end with conclusion prompt")
	}
}

func TestPhantomChainGlob(t *testing.T) {
	pr := NewPhantomReasoner()
	files := "a.go\nb.go\nc.go\nd.go\ne.go\nf.go\ng.go"
	steps := []synthStep{
		{
			Tool:   "glob",
			Args:   map[string]string{"pattern": "*.go"},
			Result: files,
		},
	}

	chain := pr.BuildChain("how many Go files", steps)
	if !strings.Contains(chain.Steps[0].Fact, "7 files") {
		t.Errorf("fact should count 7 files, got: %s", chain.Steps[0].Fact)
	}
}

func TestPhantomChainBypassQuantitative(t *testing.T) {
	pr := NewPhantomReasoner()
	steps := []synthStep{
		{
			Tool:   "glob",
			Args:   map[string]string{"pattern": "*.go"},
			Result: "a.go\nb.go\nc.go",
		},
	}

	chain := pr.BuildChain("how many Go files are there", steps)
	if !chain.CanBypass {
		t.Error("quantitative question with complete fact should enable bypass")
	}
	if chain.DirectAnswer == "" {
		t.Error("bypass should have a direct answer")
	}
	if !strings.Contains(chain.DirectAnswer, "3 files") {
		t.Errorf("direct answer should contain file count, got: %s", chain.DirectAnswer)
	}
}

func TestPhantomChainNoBypassQualitative(t *testing.T) {
	pr := NewPhantomReasoner()
	steps := []synthStep{
		{
			Tool:   "read",
			Args:   map[string]string{"path": "main.go"},
			Result: "package main\n\nfunc main() {\n\tfmt.Println(\"hello\")\n}",
		},
	}

	chain := pr.BuildChain("what does main.go do", steps)
	if chain.CanBypass {
		t.Error("qualitative question should NOT bypass")
	}
}

func TestPhantomChainRead(t *testing.T) {
	pr := NewPhantomReasoner()
	steps := []synthStep{
		{
			Tool: "read",
			Args: map[string]string{"path": "reasoner.go"},
			Result: "package cognitive\n\n" +
				"type Reasoner struct {\n\tBoard *blackboard.Blackboard\n}\n\n" +
				"func NewReasoner() *Reasoner {\n\treturn &Reasoner{}\n}\n\n" +
				"func (r *Reasoner) Run() error {\n\treturn nil\n}\n",
		},
	}

	chain := pr.BuildChain("show me reasoner.go", steps)
	step := chain.Steps[0]
	if !strings.Contains(step.Action, "reasoner.go") {
		t.Error("action should mention file name")
	}
	if !strings.Contains(step.Fact, "reasoner.go") {
		t.Error("fact should mention file name")
	}
	if !strings.Contains(step.Fact, "lines") {
		t.Error("fact should mention line count")
	}
}

func TestPhantomChainLs(t *testing.T) {
	pr := NewPhantomReasoner()
	steps := []synthStep{
		{
			Tool:   "ls",
			Args:   map[string]string{"path": "internal/"},
			Result: "cognitive/\ntools/\nmemory/\nserver/",
		},
	}

	chain := pr.BuildChain("list internal directory", steps)
	step := chain.Steps[0]
	if !strings.Contains(step.Action, "internal/") {
		t.Error("action should mention directory")
	}
	if !strings.Contains(step.Observation, "4 entries") {
		t.Errorf("observation should count 4 entries, got: %s", step.Observation)
	}
}

func TestPhantomChainGitStatus(t *testing.T) {
	pr := NewPhantomReasoner()
	steps := []synthStep{
		{
			Tool:   "git",
			Args:   map[string]string{"command": "status"},
			Result: "On branch main\nnothing to commit, working tree clean",
		},
	}

	chain := pr.BuildChain("git status", steps)
	step := chain.Steps[0]
	if !strings.Contains(step.Fact, "clean") {
		t.Errorf("clean status fact should mention clean, got: %s", step.Fact)
	}
	if !chain.CanBypass {
		t.Error("status question with clean result should bypass")
	}
}

func TestPhantomChainGitLog(t *testing.T) {
	pr := NewPhantomReasoner()
	steps := []synthStep{
		{
			Tool:   "git",
			Args:   map[string]string{"command": "log --oneline -5"},
			Result: "abc1234 First commit\ndef5678 Second commit\nghi9012 Third commit",
		},
	}

	chain := pr.BuildChain("show recent commits", steps)
	step := chain.Steps[0]
	if !strings.Contains(step.Observation, "3 entries") {
		t.Errorf("should count 3 log entries, got: %s", step.Observation)
	}
}

func TestPhantomChainWrite(t *testing.T) {
	pr := NewPhantomReasoner()
	steps := []synthStep{
		{
			Tool:   "write",
			Args:   map[string]string{"path": "/tmp/test.txt"},
			Result: "ok",
		},
	}

	chain := pr.BuildChain("create test file", steps)
	step := chain.Steps[0]
	if !strings.Contains(step.Fact, "written successfully") {
		t.Errorf("write fact should confirm success, got: %s", step.Fact)
	}
}

func TestPhantomChainMultiStep(t *testing.T) {
	pr := NewPhantomReasoner()
	steps := []synthStep{
		{
			Tool:   "grep",
			Args:   map[string]string{"pattern": "Pipeline"},
			Result: "pipeline.go:5:type Pipeline struct {",
		},
		{
			Tool:   "read",
			Args:   map[string]string{"path": "pipeline.go"},
			Result: "package cognitive\n\ntype Pipeline struct {\n\tsteps []PipeStep\n}\n",
		},
	}

	chain := pr.BuildChain("find and read Pipeline", steps)
	if len(chain.Steps) != 2 {
		t.Fatalf("should have 2 steps, got %d", len(chain.Steps))
	}
	if !strings.Contains(chain.FullContext, "Step 1") {
		t.Error("should have Step 1")
	}
	if !strings.Contains(chain.FullContext, "Step 2") {
		t.Error("should have Step 2")
	}
}

func TestPhantomChainSkipsErrors(t *testing.T) {
	pr := NewPhantomReasoner()
	steps := []synthStep{
		{
			Tool: "grep",
			Args: map[string]string{"pattern": "test"},
			Result: "a.go:1:test",
		},
		{
			Tool: "read",
			Err:  errNotFound,
		},
	}

	chain := pr.BuildChain("test", steps)
	if len(chain.Steps) != 1 {
		t.Fatalf("should skip error step, got %d steps", len(chain.Steps))
	}
}

func TestPhantomChainSkipsEmptyResults(t *testing.T) {
	pr := NewPhantomReasoner()
	steps := []synthStep{
		{
			Tool:   "grep",
			Args:   map[string]string{"pattern": "xyz"},
			Result: "",
		},
	}

	chain := pr.BuildChain("test", steps)
	if len(chain.Steps) != 0 {
		t.Error("should skip empty result steps")
	}
}

// --- Helper Tests ---

func TestExtractFuncName(t *testing.T) {
	tests := []struct {
		line string
		want string
	}{
		{"func main() {", "main"},
		{"func NewReasoner(board *blackboard.Blackboard) *Reasoner {", "NewReasoner"},
		{"func (r *Reasoner) Run() error {", "Run"},
		{"func (r *Reasoner) reason(ctx context.Context) error {", "reason"},
		{"not a func", ""},
	}

	for _, tt := range tests {
		got := extractFuncName(tt.line)
		if got != tt.want {
			t.Errorf("extractFuncName(%q) = %q, want %q", tt.line, got, tt.want)
		}
	}
}

func TestCountNonEmptyLines(t *testing.T) {
	lines := []string{"a", "", "b", "", "", "c"}
	if countNonEmptyLines(lines) != 3 {
		t.Errorf("should count 3, got %d", countNonEmptyLines(lines))
	}
}

func TestTruncatePhantom(t *testing.T) {
	short := "hello"
	if truncatePhantom(short, 100) != "hello" {
		t.Error("short string should not be truncated")
	}

	long := strings.Repeat("x", 200)
	result := truncatePhantom(long, 50)
	if len(result) != 53 { // 50 + "..."
		t.Errorf("should truncate to 53, got %d", len(result))
	}
}

// error helper for tests
var errNotFound = &testError{"not found"}

type testError struct{ msg string }
func (e *testError) Error() string { return e.msg }

// --- Cache Tests ---

func TestPhantomCacheHit(t *testing.T) {
	pr := NewPhantomReasoner()
	steps := []synthStep{
		{Tool: "glob", Args: map[string]string{"pattern": "*.go"}, Result: "a.go\nb.go\nc.go"},
	}

	// First call — cache miss, builds chain
	chain1 := pr.BuildChainCached("how many Go files", steps)
	if chain1 == nil || len(chain1.Steps) == 0 {
		t.Fatal("first call should build chain")
	}
	size1, _ := pr.CacheStats()
	if size1 != 1 {
		t.Errorf("cache should have 1 entry, got %d", size1)
	}

	// Second call — should be cache hit (same pointer)
	chain2 := pr.BuildChainCached("how many Go files", steps)
	if chain2 != chain1 {
		t.Error("second call should return cached chain (same pointer)")
	}
}

func TestPhantomCacheMiss(t *testing.T) {
	pr := NewPhantomReasoner()

	steps1 := []synthStep{{Tool: "glob", Args: map[string]string{"pattern": "*.go"}, Result: "a.go"}}
	steps2 := []synthStep{{Tool: "glob", Args: map[string]string{"pattern": "*.go"}, Result: "a.go\nb.go"}}

	chain1 := pr.BuildChainCached("files", steps1)
	chain2 := pr.BuildChainCached("files", steps2) // different results → different key

	if chain1 == chain2 {
		t.Error("different tool results should produce different chains")
	}
	size, _ := pr.CacheStats()
	if size != 2 {
		t.Errorf("cache should have 2 entries, got %d", size)
	}
}

func TestPhantomCacheEviction(t *testing.T) {
	pr := NewPhantomReasoner()
	pr.maxCache = 3

	for i := 0; i < 5; i++ {
		steps := []synthStep{{Tool: "glob", Result: strings.Repeat("x", i+1)}}
		pr.BuildChainCached("query", steps)
	}

	size, max := pr.CacheStats()
	if size > max {
		t.Errorf("cache size %d should not exceed max %d", size, max)
	}
}

func TestPhantomCacheInvalidate(t *testing.T) {
	pr := NewPhantomReasoner()
	steps := []synthStep{{Tool: "glob", Result: "a.go"}}
	pr.BuildChainCached("test", steps)

	if size, _ := pr.CacheStats(); size != 1 {
		t.Fatal("should have 1 cached entry")
	}

	pr.InvalidateCache()
	size, _ := pr.CacheStats()
	if size != 0 {
		t.Errorf("after invalidation, cache size = %d, want 0", size)
	}
}

// --- Benchmark ---

func BenchmarkPhantomChainCached(b *testing.B) {
	pr := NewPhantomReasoner()
	steps := []synthStep{
		{Tool: "grep", Args: map[string]string{"pattern": "Pipeline"}, Result: "a.go:5:type Pipeline\nb.go:10:NewPipeline"},
	}
	// Warm cache
	pr.BuildChainCached("find Pipeline", steps)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		pr.BuildChainCached("find Pipeline", steps)
	}
}

func BenchmarkPhantomChainBuild(b *testing.B) {
	pr := NewPhantomReasoner()
	steps := []synthStep{
		{Tool: "grep", Args: map[string]string{"pattern": "Pipeline"}, Result: "a.go:5:type Pipeline\nb.go:10:NewPipeline"},
		{Tool: "read", Args: map[string]string{"path": "a.go"}, Result: "package cognitive\n\ntype Pipeline struct{}"},
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		pr.BuildChain("find Pipeline", steps)
	}
}
