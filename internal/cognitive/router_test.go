package cognitive

import (
	"strings"
	"testing"
)

func TestNewModelRouter(t *testing.T) {
	r := NewModelRouter("http://localhost:11434", "qwen2.5:1.5b")
	if r == nil {
		t.Fatal("NewModelRouter returned nil")
	}
	if r.host != "http://localhost:11434" {
		t.Errorf("expected host 'http://localhost:11434', got %q", r.host)
	}
	if r.defaultModel != "qwen2.5:1.5b" {
		t.Errorf("expected default model 'qwen2.5:1.5b', got %q", r.defaultModel)
	}
	// Should have a client for the default model
	if _, ok := r.clients["qwen2.5:1.5b"]; !ok {
		t.Error("expected pre-created client for default model")
	}
}

func TestRouteDefault(t *testing.T) {
	r := NewModelRouter("http://localhost:11434", "qwen2.5:1.5b")

	// Without discovery, all tasks should route to the default model
	tasks := []TaskType{TaskPerception, TaskReasoning, TaskCompression, TaskReflection}
	for _, task := range tasks {
		model := r.Route(task)
		if model != "qwen2.5:1.5b" {
			t.Errorf("Route(%s) = %q, want 'qwen2.5:1.5b'", task, model)
		}
	}
}

func TestClientForDefault(t *testing.T) {
	r := NewModelRouter("http://localhost:11434", "qwen2.5:1.5b")

	client := r.ClientFor(TaskReasoning)
	if client == nil {
		t.Fatal("ClientFor(TaskReasoning) returned nil")
	}
	if client.Model() != "qwen2.5:1.5b" {
		t.Errorf("expected client model 'qwen2.5:1.5b', got %q", client.Model())
	}
}

func TestClassifyModel(t *testing.T) {
	tests := []struct {
		name     string
		expected string
	}{
		{"tinyllama:latest", "tinyllama"},
		{"tinyllama", "tinyllama"},
		{"qwen2.5:1.5b", "qwen"},
		{"qwen2.5:7b", "qwen"},
		{"llama3.2:latest", "llama"},
		{"llama3.1:8b", "llama"},
		{"deepseek-r1:8b", "deepseek"},
		{"mistral:7b", "mistral"},
		{"some-unknown-model:latest", "unknown"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := ClassifyModel(tt.name)
			if got != tt.expected {
				t.Errorf("ClassifyModel(%q) = %q, want %q", tt.name, got, tt.expected)
			}
		})
	}
}

func TestDiscoverRoutes(t *testing.T) {
	r := NewModelRouter("http://localhost:11434", "qwen2.5:1.5b")

	// Simulate discovering tinyllama + qwen
	r.DiscoverFromList([]ModelProfile{
		{Name: "tinyllama:latest", SizeBytes: 637_000_000, Family: "tinyllama", Available: true},
		{Name: "qwen2.5:1.5b", SizeBytes: 986_000_000, Family: "qwen", Available: true},
	})

	// Perception should go to tinyllama (fast model)
	if got := r.Route(TaskPerception); got != "tinyllama:latest" {
		t.Errorf("Route(perception) = %q, want 'tinyllama:latest'", got)
	}

	// Reasoning should go to qwen
	if got := r.Route(TaskReasoning); got != "qwen2.5:1.5b" {
		t.Errorf("Route(reasoning) = %q, want 'qwen2.5:1.5b'", got)
	}

	// Compression should go to tinyllama (fast model)
	if got := r.Route(TaskCompression); got != "tinyllama:latest" {
		t.Errorf("Route(compression) = %q, want 'tinyllama:latest'", got)
	}

	// Reflection should go to tinyllama (fast model)
	if got := r.Route(TaskReflection); got != "tinyllama:latest" {
		t.Errorf("Route(reflection) = %q, want 'tinyllama:latest'", got)
	}
}

func TestDiscoverRoutesWithLargeModel(t *testing.T) {
	r := NewModelRouter("http://localhost:11434", "qwen2.5:1.5b")

	// Simulate discovering tinyllama + qwen + large llama
	r.DiscoverFromList([]ModelProfile{
		{Name: "tinyllama:latest", SizeBytes: 637_000_000, Family: "tinyllama", Available: true},
		{Name: "qwen2.5:1.5b", SizeBytes: 986_000_000, Family: "qwen", Available: true},
		{Name: "llama3.2:latest", SizeBytes: 3_500_000_000, Family: "llama", Available: true},
	})

	// Reasoning should still prefer qwen over large llama
	if got := r.Route(TaskReasoning); got != "qwen2.5:1.5b" {
		t.Errorf("Route(reasoning) = %q, want 'qwen2.5:1.5b'", got)
	}

	// Perception should go to tinyllama
	if got := r.Route(TaskPerception); got != "tinyllama:latest" {
		t.Errorf("Route(perception) = %q, want 'tinyllama:latest'", got)
	}
}

func TestDiscoverLlamaOnlyAsReasoner(t *testing.T) {
	r := NewModelRouter("http://localhost:11434", "default-model")

	// Only a large llama available, no qwen
	r.DiscoverFromList([]ModelProfile{
		{Name: "tinyllama:latest", SizeBytes: 637_000_000, Family: "tinyllama", Available: true},
		{Name: "llama3.2:latest", SizeBytes: 3_500_000_000, Family: "llama", Available: true},
	})

	// Large llama should become the reasoning model
	if got := r.Route(TaskReasoning); got != "llama3.2:latest" {
		t.Errorf("Route(reasoning) = %q, want 'llama3.2:latest'", got)
	}

	// Perception should go to tinyllama
	if got := r.Route(TaskPerception); got != "tinyllama:latest" {
		t.Errorf("Route(perception) = %q, want 'tinyllama:latest'", got)
	}
}

func TestSingleModelFallback(t *testing.T) {
	r := NewModelRouter("http://localhost:11434", "qwen2.5:1.5b")

	// Only one model available — it should be used for everything
	r.DiscoverFromList([]ModelProfile{
		{Name: "qwen2.5:1.5b", SizeBytes: 986_000_000, Family: "qwen", Available: true},
	})

	tasks := []TaskType{TaskPerception, TaskReasoning, TaskCompression, TaskReflection}
	for _, task := range tasks {
		model := r.Route(task)
		if model != "qwen2.5:1.5b" {
			t.Errorf("Route(%s) = %q, want 'qwen2.5:1.5b' (single model fallback)", task, model)
		}
	}
}

func TestStatus(t *testing.T) {
	r := NewModelRouter("http://localhost:11434", "qwen2.5:1.5b")

	// Default: all routes to the same model
	status := r.Status()
	if !strings.HasPrefix(status, "routing:") {
		t.Errorf("Status() should start with 'routing:', got %q", status)
	}
	if !strings.Contains(status, "all->qwen2.5:1.5b") {
		t.Errorf("Status() with single model should show 'all->qwen2.5:1.5b', got %q", status)
	}

	// After discovery with multiple models
	r.DiscoverFromList([]ModelProfile{
		{Name: "tinyllama:latest", SizeBytes: 637_000_000, Family: "tinyllama", Available: true},
		{Name: "qwen2.5:1.5b", SizeBytes: 986_000_000, Family: "qwen", Available: true},
	})

	status = r.Status()
	if !strings.Contains(status, "perception->tinyllama") {
		t.Errorf("Status() should contain 'perception->tinyllama', got %q", status)
	}
	if !strings.Contains(status, "reasoning->qwen2.5:1.5b") {
		t.Errorf("Status() should contain 'reasoning->qwen2.5:1.5b', got %q", status)
	}
}

func TestStatusShortensModelNames(t *testing.T) {
	r := NewModelRouter("http://localhost:11434", "tinyllama:latest")

	status := r.Status()
	// ":latest" should be stripped
	if strings.Contains(status, ":latest") {
		t.Errorf("Status() should strip ':latest' suffix, got %q", status)
	}
	if !strings.Contains(status, "tinyllama") {
		t.Errorf("Status() should contain 'tinyllama', got %q", status)
	}
}

func TestModelsReturnsDiscoveredProfiles(t *testing.T) {
	r := NewModelRouter("http://localhost:11434", "qwen2.5:1.5b")

	// Before discovery
	models := r.Models()
	if len(models) != 0 {
		t.Errorf("Models() before discovery should be empty, got %d", len(models))
	}

	// After discovery
	r.DiscoverFromList([]ModelProfile{
		{Name: "tinyllama:latest", SizeBytes: 637_000_000, Family: "tinyllama", Available: true},
		{Name: "qwen2.5:1.5b", SizeBytes: 986_000_000, Family: "qwen", Available: true},
	})

	models = r.Models()
	if len(models) != 2 {
		t.Fatalf("Models() after discovery should have 2, got %d", len(models))
	}
}

// --- ClientForQuery tests ---

func TestClientForQueryRouting(t *testing.T) {
	r := NewModelRouter("http://localhost:11434", "qwen2.5:1.5b")

	r.DiscoverFromList([]ModelProfile{
		{Name: "tinyllama:latest", SizeBytes: 637_000_000, Family: "tinyllama", Available: true},
		{Name: "qwen2.5:1.5b", SizeBytes: 986_000_000, Family: "qwen", Available: true},
	})

	tests := []struct {
		query     string
		wantModel string
		desc      string
	}{
		// Fast queries → smallest model (perception/tinyllama)
		{"hello", "tinyllama:latest", "greeting is fast"},
		{"thanks", "tinyllama:latest", "thanks is fast"},
		{"hi there", "tinyllama:latest", "short greeting is fast"},

		// Medium queries → reasoning model (knowledge queries need quality)
		{"explain why the sky is blue", "qwen2.5:1.5b", "explanation is medium"},
		{"what is quantum entanglement", "qwen2.5:1.5b", "definitional question is medium"},

		// Full queries → reasoning model (qwen)
		{"read the file go.mod and tell me the version", "qwen2.5:1.5b", "file read is full"},
		{"search for TODO comments in all go files", "qwen2.5:1.5b", "code search is full"},
		{"run the tests and show me failures", "qwen2.5:1.5b", "test execution is full"},
	}

	for _, tt := range tests {
		t.Run(tt.desc, func(t *testing.T) {
			client := r.ClientForQuery(tt.query)
			if client == nil {
				t.Fatal("ClientForQuery returned nil")
			}
			if client.Model() != tt.wantModel {
				t.Errorf("ClientForQuery(%q) model = %q, want %q", tt.query, client.Model(), tt.wantModel)
			}
		})
	}
}

func TestQueryRoute(t *testing.T) {
	r := NewModelRouter("http://localhost:11434", "qwen2.5:1.5b")

	r.DiscoverFromList([]ModelProfile{
		{Name: "tinyllama:latest", SizeBytes: 637_000_000, Family: "tinyllama", Available: true},
		{Name: "qwen2.5:1.5b", SizeBytes: 986_000_000, Family: "qwen", Available: true},
		{Name: "llama3.1:latest", SizeBytes: 4_900_000_000, Family: "llama", Available: true},
	})

	// Simple query should route to fast model
	fast := r.QueryRoute("hello")
	if fast != "tinyllama:latest" {
		t.Errorf("QueryRoute(hello) = %q, want tinyllama", fast)
	}

	// Complex query should route to reasoning model
	full := r.QueryRoute("search for all error handling patterns in the codebase")
	if full != "qwen2.5:1.5b" {
		t.Errorf("QueryRoute(complex) = %q, want qwen2.5:1.5b", full)
	}
}

func TestClientForQuerySingleModel(t *testing.T) {
	r := NewModelRouter("http://localhost:11434", "qwen2.5:1.5b")

	// Single model — everything routes to it regardless of query
	r.DiscoverFromList([]ModelProfile{
		{Name: "qwen2.5:1.5b", SizeBytes: 986_000_000, Family: "qwen", Available: true},
	})

	queries := []string{"hello", "explain quantum physics", "read go.mod"}
	for _, q := range queries {
		client := r.ClientForQuery(q)
		if client.Model() != "qwen2.5:1.5b" {
			t.Errorf("ClientForQuery(%q) = %q with single model, want qwen2.5:1.5b", q, client.Model())
		}
	}
}

func TestClientForCreatesClients(t *testing.T) {
	r := NewModelRouter("http://localhost:11434", "qwen2.5:1.5b")

	r.DiscoverFromList([]ModelProfile{
		{Name: "tinyllama:latest", SizeBytes: 637_000_000, Family: "tinyllama", Available: true},
		{Name: "qwen2.5:1.5b", SizeBytes: 986_000_000, Family: "qwen", Available: true},
	})

	// ClientFor perception should return a tinyllama client
	client := r.ClientFor(TaskPerception)
	if client == nil {
		t.Fatal("ClientFor(perception) returned nil")
	}
	if client.Model() != "tinyllama:latest" {
		t.Errorf("ClientFor(perception) model = %q, want 'tinyllama:latest'", client.Model())
	}

	// ClientFor reasoning should return a qwen client
	client = r.ClientFor(TaskReasoning)
	if client == nil {
		t.Fatal("ClientFor(reasoning) returned nil")
	}
	if client.Model() != "qwen2.5:1.5b" {
		t.Errorf("ClientFor(reasoning) model = %q, want 'qwen2.5:1.5b'", client.Model())
	}
}
