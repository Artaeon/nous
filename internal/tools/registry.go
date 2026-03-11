package tools

import "fmt"

// Tool represents an executable capability that the cognitive system can invoke.
type Tool struct {
	Name        string
	Description string
	Execute     func(args map[string]string) (string, error)
}

// Registry holds all available tools.
type Registry struct {
	tools map[string]Tool
}

func NewRegistry() *Registry {
	return &Registry{
		tools: make(map[string]Tool),
	}
}

// Register adds a tool to the registry.
func (r *Registry) Register(tool Tool) {
	r.tools[tool.Name] = tool
}

// Get retrieves a tool by name.
func (r *Registry) Get(name string) (Tool, error) {
	t, ok := r.tools[name]
	if !ok {
		return Tool{}, fmt.Errorf("unknown tool: %s", name)
	}
	return t, nil
}

// List returns all registered tool names and descriptions.
func (r *Registry) List() []Tool {
	out := make([]Tool, 0, len(r.tools))
	for _, t := range r.tools {
		out = append(out, t)
	}
	return out
}

// ListByNames returns tools matching the given names, preserving order.
func (r *Registry) ListByNames(names []string) []Tool {
	var out []Tool
	for _, name := range names {
		if t, ok := r.tools[name]; ok {
			out = append(out, t)
		}
	}
	return out
}

// Describe returns a formatted string listing all tools for LLM prompts.
func (r *Registry) Describe() string {
	var desc string
	for _, t := range r.tools {
		desc += fmt.Sprintf("- %s: %s\n", t.Name, t.Description)
	}
	return desc
}
