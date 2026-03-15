package cognitive

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/artaeon/nous/internal/ollama"
)

// GrammarDecoder uses Ollama's structured output format parameter to constrain
// the model's output to match a dynamic JSON schema. Instead of hoping the
// model produces valid JSON in the right structure, the decoding layer itself
// enforces the schema — the model literally cannot produce invalid output.
//
// Innovation: Existing approaches use format:"json" (any valid JSON) or
// static tool schemas. This dynamically generates the MINIMUM schema needed
// for each specific micro-decision, eliminating unnecessary fields that
// confuse small models. The schema changes per turn based on context.
//
// Example: When we know the user wants grep, instead of exposing all 18 tools:
//   Static:  {"tool": "...", "args": {"any": "thing"}}     → model hallucinates
//   Dynamic: {"pattern": "string", "glob": "string?"}      → model fills blanks
type GrammarDecoder struct {
	llm  *ollama.Client
	opts *ollama.ModelOptions
}

// GrammarResult holds the parsed result from grammar-constrained decoding.
type GrammarResult struct {
	Tool   string
	Args   map[string]string
	Raw    string // raw JSON response
	Schema any    // schema that was used
}

// NewGrammarDecoder creates a new grammar-constrained decoder.
func NewGrammarDecoder(llm *ollama.Client) *GrammarDecoder {
	return &GrammarDecoder{
		llm: llm,
		opts: &ollama.ModelOptions{
			Temperature:   0.1, // deterministic output
			NumPredict:    64,  // schemas are small
			RepeatPenalty: 1.0,
		},
	}
}

// --- Schema Types ---

// JSONSchema represents a JSON Schema for Ollama's format parameter.
type JSONSchema struct {
	Type       string                `json:"type"`
	Properties map[string]SchemaProperty `json:"properties"`
	Required   []string              `json:"required,omitempty"`
}

// SchemaProperty describes one property in a JSON Schema.
type SchemaProperty struct {
	Type        string   `json:"type"`
	Description string   `json:"description,omitempty"`
	Enum        []string `json:"enum,omitempty"`
}

// --- Dynamic Schema Builders ---

// ChoiceSchema builds a schema that constrains the model to pick one choice.
// Output: {"choice": "option_a"} where choice is enum-constrained.
func ChoiceSchema(fieldName string, options []string) JSONSchema {
	return JSONSchema{
		Type: "object",
		Properties: map[string]SchemaProperty{
			fieldName: {
				Type: "string",
				Enum: options,
			},
		},
		Required: []string{fieldName},
	}
}

// ToolCallSchema builds a schema for a specific tool's arguments.
// Only includes the fields this tool actually needs — nothing extra.
func ToolCallSchema(toolName string) JSONSchema {
	specs := toolSchemaSpec(toolName)
	props := make(map[string]SchemaProperty)
	var required []string

	for _, s := range specs {
		props[s.Name] = SchemaProperty{
			Type:        s.Type,
			Description: s.Description,
		}
		if s.Required {
			required = append(required, s.Name)
		}
	}

	return JSONSchema{
		Type:       "object",
		Properties: props,
		Required:   required,
	}
}

// ToolSelectSchema builds a schema for selecting a tool from available options.
// Output: {"tool": "grep"} where tool is enum-constrained.
func ToolSelectSchema(tools []string) JSONSchema {
	return ChoiceSchema("tool", tools)
}

// TaskClassifySchema builds a schema for classifying a task type.
// Output: {"task": "search"} where task is enum-constrained.
func TaskClassifySchema() JSONSchema {
	return ChoiceSchema("task", []string{
		"search", "read", "write", "list", "explain", "chat",
	})
}

// ExtractValueSchema builds a schema for extracting a single value.
// Output: {"value": "ReflectionGate"} — freeform string.
func ExtractValueSchema(fieldName, description string) JSONSchema {
	return JSONSchema{
		Type: "object",
		Properties: map[string]SchemaProperty{
			fieldName: {
				Type:        "string",
				Description: description,
			},
		},
		Required: []string{fieldName},
	}
}

// MultiFieldSchema builds a schema for extracting multiple named values.
// Each field maps to a freeform string.
func MultiFieldSchema(fields []FieldSpec) JSONSchema {
	props := make(map[string]SchemaProperty)
	var required []string

	for _, f := range fields {
		props[f.Name] = SchemaProperty{
			Type:        "string",
			Description: f.Description,
		}
		if f.Required {
			required = append(required, f.Name)
		}
	}

	return JSONSchema{
		Type:       "object",
		Properties: props,
		Required:   required,
	}
}

// FieldSpec describes a field for schema generation.
type FieldSpec struct {
	Name        string
	Description string
	Type        string
	Required    bool
}

// --- Constrained Decoding Methods ---

// ClassifyTask uses schema-constrained decoding to classify a query.
func (gd *GrammarDecoder) ClassifyTask(query string) (string, error) {
	schema := TaskClassifySchema()
	prompt := fmt.Sprintf("Classify this request. Reply with JSON.\n\nRequest: %s", query)

	result, err := gd.decode(prompt, schema)
	if err != nil {
		return "", err
	}

	var parsed struct {
		Task string `json:"task"`
	}
	if err := json.Unmarshal([]byte(result), &parsed); err != nil {
		return "", fmt.Errorf("parse task classification: %w", err)
	}

	return parsed.Task, nil
}

// SelectTool uses schema-constrained decoding to pick a tool.
func (gd *GrammarDecoder) SelectTool(query string, tools []string) (string, error) {
	if len(tools) == 0 {
		return "", fmt.Errorf("no tools to select from")
	}
	if len(tools) == 1 {
		return tools[0], nil
	}

	schema := ToolSelectSchema(tools)
	prompt := fmt.Sprintf("Pick the best tool for this request. Reply with JSON.\n\nAvailable: %s\nRequest: %s",
		strings.Join(tools, ", "), query)

	result, err := gd.decode(prompt, schema)
	if err != nil {
		return "", err
	}

	var parsed struct {
		Tool string `json:"tool"`
	}
	if err := json.Unmarshal([]byte(result), &parsed); err != nil {
		return "", fmt.Errorf("parse tool selection: %w", err)
	}

	return parsed.Tool, nil
}

// ExtractArgs uses schema-constrained decoding to extract tool arguments.
func (gd *GrammarDecoder) ExtractArgs(query, toolName string) (map[string]string, error) {
	schema := ToolCallSchema(toolName)
	if len(schema.Properties) == 0 {
		return map[string]string{}, nil
	}

	// Build a focused prompt listing only the fields needed
	var fields []string
	for name, prop := range schema.Properties {
		desc := name
		if prop.Description != "" {
			desc = name + " (" + prop.Description + ")"
		}
		fields = append(fields, desc)
	}

	prompt := fmt.Sprintf("Extract these fields from the request. Reply with JSON.\n\nFields: %s\nRequest: %s",
		strings.Join(fields, ", "), query)

	result, err := gd.decode(prompt, schema)
	if err != nil {
		return nil, err
	}

	var parsed map[string]string
	if err := json.Unmarshal([]byte(result), &parsed); err != nil {
		// Try map[string]any and convert
		var loose map[string]any
		if err2 := json.Unmarshal([]byte(result), &loose); err2 != nil {
			return nil, fmt.Errorf("parse extracted args: %w", err)
		}
		parsed = make(map[string]string)
		for k, v := range loose {
			parsed[k] = fmt.Sprintf("%v", v)
		}
	}

	// Remove empty/none values
	for k, v := range parsed {
		if v == "" || strings.ToLower(v) == "none" {
			delete(parsed, k)
		}
	}

	return parsed, nil
}

// Resolve performs full grammar-constrained resolution: classify → select → extract.
func (gd *GrammarDecoder) Resolve(query string, availableTools []string) (*GrammarResult, error) {
	// Step 1: Classify task type
	taskType, err := gd.ClassifyTask(query)
	if err != nil {
		return nil, fmt.Errorf("classify: %w", err)
	}

	if taskType == "chat" || taskType == "explain" {
		return &GrammarResult{Tool: taskType}, nil
	}

	// Step 2: Filter tools by task type
	relevant := filterToolsByTask(taskType, availableTools)
	if len(relevant) == 0 {
		return nil, fmt.Errorf("no tools for task %q", taskType)
	}

	// Step 3: Select tool (skip LLM if only one option)
	tool, err := gd.SelectTool(query, relevant)
	if err != nil {
		return nil, fmt.Errorf("select: %w", err)
	}

	// Step 4: Extract args with tool-specific schema
	args, err := gd.ExtractArgs(query, tool)
	if err != nil {
		return nil, fmt.Errorf("extract: %w", err)
	}

	return &GrammarResult{
		Tool: tool,
		Args: args,
	}, nil
}

// decode sends a prompt with schema constraint and returns the raw JSON string.
func (gd *GrammarDecoder) decode(prompt string, schema JSONSchema) (string, error) {
	resp, err := gd.llm.ChatWithSchema(
		[]ollama.Message{{Role: "user", Content: prompt}},
		schema,
		gd.opts,
	)
	if err != nil {
		return "", err
	}
	return strings.TrimSpace(resp.Message.Content), nil
}

// --- Tool Schema Specifications ---

// toolSchemaField describes one field in a tool's schema.
type toolSchemaField struct {
	Name        string
	Type        string
	Description string
	Required    bool
}

// toolSchemaSpec returns the schema fields for a specific tool.
// This is the authoritative source for what each tool accepts.
func toolSchemaSpec(tool string) []toolSchemaField {
	switch tool {
	case "read":
		return []toolSchemaField{
			{Name: "path", Type: "string", Description: "file path to read", Required: true},
		}
	case "grep":
		return []toolSchemaField{
			{Name: "pattern", Type: "string", Description: "regex search pattern", Required: true},
			{Name: "glob", Type: "string", Description: "file filter like *.go", Required: false},
		}
	case "glob":
		return []toolSchemaField{
			{Name: "pattern", Type: "string", Description: "glob pattern like **/*.go", Required: true},
		}
	case "ls":
		return []toolSchemaField{
			{Name: "path", Type: "string", Description: "directory to list", Required: false},
		}
	case "tree":
		return []toolSchemaField{
			{Name: "path", Type: "string", Description: "root directory", Required: false},
		}
	case "write":
		return []toolSchemaField{
			{Name: "path", Type: "string", Description: "file path to write", Required: true},
			{Name: "content", Type: "string", Description: "file content", Required: true},
		}
	case "edit":
		return []toolSchemaField{
			{Name: "path", Type: "string", Description: "file to edit", Required: true},
			{Name: "old", Type: "string", Description: "text to replace", Required: true},
			{Name: "new", Type: "string", Description: "replacement text", Required: true},
		}
	case "git":
		return []toolSchemaField{
			{Name: "command", Type: "string", Description: "git subcommand (status, log, diff, etc.)", Required: true},
		}
	default:
		return nil
	}
}

