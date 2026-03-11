package cognitive

import (
	"context"
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/artaeon/nous/internal/ollama"
)

// TaskType classifies what kind of inference is needed.
type TaskType string

const (
	TaskPerception  TaskType = "perception"  // intent extraction, entity parsing
	TaskReasoning   TaskType = "reasoning"   // main chain-of-thought
	TaskCompression TaskType = "compression" // summarizing results
	TaskReflection  TaskType = "reflection"  // quality checking
)

// ModelProfile describes a locally available model's capabilities.
type ModelProfile struct {
	Name      string
	SizeBytes int64
	Family    string        // "qwen", "tinyllama", "llama", etc.
	Speed     time.Duration // measured probe latency
	Available bool
}

// ModelRouter dynamically routes cognitive tasks to the best available local model.
type ModelRouter struct {
	mu           sync.RWMutex
	host         string
	defaultModel string
	models       []ModelProfile
	routes       map[TaskType]string        // task type -> model name
	clients      map[string]*ollama.Client   // model name -> client
}

// NewModelRouter creates a router with a default model used for all task types
// until Discover is called to find and assign specialized models.
func NewModelRouter(host, defaultModel string) *ModelRouter {
	r := &ModelRouter{
		host:         host,
		defaultModel: defaultModel,
		routes: map[TaskType]string{
			TaskPerception:  defaultModel,
			TaskReasoning:   defaultModel,
			TaskCompression: defaultModel,
			TaskReflection:  defaultModel,
		},
		clients: make(map[string]*ollama.Client),
	}
	// Pre-create client for the default model
	r.clients[defaultModel] = ollama.New(
		ollama.WithHost(host),
		ollama.WithModel(defaultModel),
	)
	return r
}

// ClassifyModel determines the family of a model based on its name.
func ClassifyModel(name string) string {
	lower := strings.ToLower(name)
	switch {
	case strings.Contains(lower, "tinyllama"):
		return "tinyllama"
	case strings.Contains(lower, "qwen"):
		return "qwen"
	case strings.Contains(lower, "llama"):
		return "llama"
	case strings.Contains(lower, "deepseek"):
		return "deepseek"
	case strings.Contains(lower, "mistral"):
		return "mistral"
	default:
		return "unknown"
	}
}

// Discover queries Ollama for available models and builds the routing table.
// It classifies each model by name pattern and assigns them to task types:
//   - tinyllama -> fast tasks: perception, compression, reflection
//   - qwen -> reasoning (chain-of-thought)
//   - llama (>2GB) -> mid-tier reasoning fallback
//
// If only one model is available, it is used for everything.
func (r *ModelRouter) Discover(ctx context.Context) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	// Use a temporary client to list models
	c := ollama.New(
		ollama.WithHost(r.host),
		ollama.WithModel(r.defaultModel),
	)
	modelInfos, err := c.ListModels()
	if err != nil {
		return fmt.Errorf("discover models: %w", err)
	}

	r.models = make([]ModelProfile, 0, len(modelInfos))
	for _, m := range modelInfos {
		profile := ModelProfile{
			Name:      m.Name,
			SizeBytes: m.Size,
			Family:    ClassifyModel(m.Name),
			Available: true,
		}
		r.models = append(r.models, profile)
	}

	// If only one model, use it for everything
	if len(r.models) == 1 {
		only := r.models[0].Name
		r.routes[TaskPerception] = only
		r.routes[TaskReasoning] = only
		r.routes[TaskCompression] = only
		r.routes[TaskReflection] = only
		r.ensureClient(only)
		return nil
	}

	// Build route table from available models
	r.buildRoutes()

	return nil
}

// DiscoverFromList builds the routing table from a pre-fetched model list.
// This is useful for testing or when the model list is already known.
func (r *ModelRouter) DiscoverFromList(models []ModelProfile) {
	r.mu.Lock()
	defer r.mu.Unlock()

	r.models = models

	if len(r.models) == 1 {
		only := r.models[0].Name
		r.routes[TaskPerception] = only
		r.routes[TaskReasoning] = only
		r.routes[TaskCompression] = only
		r.routes[TaskReflection] = only
		r.ensureClient(only)
		return
	}

	r.buildRoutes()
}

// buildRoutes assigns models to task types based on their classification.
func (r *ModelRouter) buildRoutes() {
	var fastModel string
	var reasonModel string

	// Find the best model for each role
	for _, m := range r.models {
		switch m.Family {
		case "tinyllama":
			// TinyLlama is fastest — good for perception, compression, reflection
			fastModel = m.Name
		case "qwen":
			// Prefer the largest qwen for reasoning
			if reasonModel == "" || m.SizeBytes > r.sizeOf(reasonModel) {
				reasonModel = m.Name
			}
		case "llama":
			// Large llama models are mid-tier reasoning fallbacks
			if m.SizeBytes > 2*1024*1024*1024 { // > 2GB
				if reasonModel == "" {
					reasonModel = m.Name
				}
			}
			// Small llama models can serve as fast models
			if fastModel == "" {
				fastModel = m.Name
			}
		}
	}

	// Apply routes — fall back to default if no specialized model found
	if fastModel != "" {
		r.routes[TaskPerception] = fastModel
		r.routes[TaskCompression] = fastModel
		r.routes[TaskReflection] = fastModel
		r.ensureClient(fastModel)
	}
	if reasonModel != "" {
		r.routes[TaskReasoning] = reasonModel
		r.ensureClient(reasonModel)
	}
}

// sizeOf returns the size of a model by name, or 0 if not found.
func (r *ModelRouter) sizeOf(name string) int64 {
	for _, m := range r.models {
		if m.Name == name {
			return m.SizeBytes
		}
	}
	return 0
}

// ensureClient creates an ollama.Client for the given model if one doesn't exist.
func (r *ModelRouter) ensureClient(model string) {
	if _, exists := r.clients[model]; !exists {
		r.clients[model] = ollama.New(
			ollama.WithHost(r.host),
			ollama.WithModel(model),
		)
	}
}

// ClientFor returns the best ollama.Client for the given task type.
func (r *ModelRouter) ClientFor(task TaskType) *ollama.Client {
	r.mu.RLock()
	defer r.mu.RUnlock()

	model := r.routes[task]
	if model == "" {
		model = r.defaultModel
	}

	if client, ok := r.clients[model]; ok {
		return client
	}

	// Shouldn't happen, but create on demand
	r.mu.RUnlock()
	r.mu.Lock()
	r.ensureClient(model)
	client := r.clients[model]
	r.mu.Unlock()
	r.mu.RLock()
	return client
}

// Route returns the model name assigned to a task type.
func (r *ModelRouter) Route(task TaskType) string {
	r.mu.RLock()
	defer r.mu.RUnlock()

	if model, ok := r.routes[task]; ok {
		return model
	}
	return r.defaultModel
}

// Models returns the discovered model profiles.
func (r *ModelRouter) Models() []ModelProfile {
	r.mu.RLock()
	defer r.mu.RUnlock()

	out := make([]ModelProfile, len(r.models))
	copy(out, r.models)
	return out
}

// Status returns a formatted routing table for display.
func (r *ModelRouter) Status() string {
	r.mu.RLock()
	defer r.mu.RUnlock()

	// Collect unique routes in a stable order
	tasks := []TaskType{TaskPerception, TaskReasoning, TaskCompression, TaskReflection}
	var parts []string

	seen := make(map[string][]string) // model -> list of task names
	for _, task := range tasks {
		model := r.routes[task]
		if model == "" {
			model = r.defaultModel
		}
		// Shorten model name for display (remove tag if redundant)
		short := shortModelName(model)
		seen[short] = append(seen[short], string(task))
	}

	// If all routes go to the same model, show a compact form
	if len(seen) == 1 {
		for model := range seen {
			return fmt.Sprintf("routing: all->%s", model)
		}
	}

	// Show each unique route
	for _, task := range tasks {
		model := r.routes[task]
		if model == "" {
			model = r.defaultModel
		}
		parts = append(parts, fmt.Sprintf("%s->%s", string(task), shortModelName(model)))
	}

	return "routing: " + strings.Join(parts, ", ")
}

// shortModelName trims the model name for compact display.
func shortModelName(name string) string {
	// Remove ":latest" suffix if present
	name = strings.TrimSuffix(name, ":latest")
	return name
}

// Probe sends a minimal prompt to a model and measures the response latency.
func (r *ModelRouter) Probe(ctx context.Context, model string) (time.Duration, error) {
	c := ollama.New(
		ollama.WithHost(r.host),
		ollama.WithModel(model),
		ollama.WithTimeout(30*time.Second),
	)

	start := time.Now()
	_, err := c.Chat([]ollama.Message{
		{Role: "user", Content: "hi"},
	}, &ollama.ModelOptions{
		Temperature: 0,
		NumPredict:  1,
	})
	elapsed := time.Since(start)

	if err != nil {
		return 0, fmt.Errorf("probe %s: %w", model, err)
	}

	return elapsed, nil
}
