package cognitive

import (
	"encoding/json"
	"testing"
)

// --- Schema Builder Tests ---

func TestChoiceSchema(t *testing.T) {
	schema := ChoiceSchema("task", []string{"search", "read", "write"})

	if schema.Type != "object" {
		t.Errorf("type = %q, want object", schema.Type)
	}
	if len(schema.Required) != 1 || schema.Required[0] != "task" {
		t.Errorf("required = %v, want [task]", schema.Required)
	}

	prop, ok := schema.Properties["task"]
	if !ok {
		t.Fatal("missing task property")
	}
	if prop.Type != "string" {
		t.Errorf("task type = %q, want string", prop.Type)
	}
	if len(prop.Enum) != 3 {
		t.Errorf("enum count = %d, want 3", len(prop.Enum))
	}
}

func TestChoiceSchemaJSON(t *testing.T) {
	schema := ChoiceSchema("tool", []string{"grep", "glob"})
	data, err := json.Marshal(schema)
	if err != nil {
		t.Fatalf("marshal error: %v", err)
	}

	// Verify it produces valid JSON Schema
	var parsed map[string]any
	if err := json.Unmarshal(data, &parsed); err != nil {
		t.Fatalf("invalid JSON: %v", err)
	}

	if parsed["type"] != "object" {
		t.Error("schema type should be object")
	}
	props := parsed["properties"].(map[string]any)
	toolProp := props["tool"].(map[string]any)
	enum := toolProp["enum"].([]any)
	if len(enum) != 2 {
		t.Errorf("enum should have 2 items, got %d", len(enum))
	}
}

func TestToolCallSchemaGrep(t *testing.T) {
	schema := ToolCallSchema("grep")

	if len(schema.Properties) != 2 {
		t.Errorf("grep schema should have 2 properties, got %d", len(schema.Properties))
	}

	pattern, ok := schema.Properties["pattern"]
	if !ok {
		t.Fatal("missing pattern property")
	}
	if pattern.Type != "string" {
		t.Errorf("pattern type = %q, want string", pattern.Type)
	}

	glob, ok := schema.Properties["glob"]
	if !ok {
		t.Fatal("missing glob property")
	}
	if glob.Type != "string" {
		t.Errorf("glob type = %q, want string", glob.Type)
	}

	// pattern is required, glob is not
	if len(schema.Required) != 1 || schema.Required[0] != "pattern" {
		t.Errorf("required = %v, want [pattern]", schema.Required)
	}
}

func TestToolCallSchemaRead(t *testing.T) {
	schema := ToolCallSchema("read")

	if len(schema.Properties) != 1 {
		t.Errorf("read schema should have 1 property, got %d", len(schema.Properties))
	}
	if len(schema.Required) != 1 || schema.Required[0] != "path" {
		t.Errorf("required = %v, want [path]", schema.Required)
	}
}

func TestToolCallSchemaWrite(t *testing.T) {
	schema := ToolCallSchema("write")

	if len(schema.Properties) != 2 {
		t.Errorf("write schema should have 2 properties, got %d", len(schema.Properties))
	}
	if len(schema.Required) != 2 {
		t.Errorf("write should have 2 required fields, got %d", len(schema.Required))
	}
}

func TestToolCallSchemaEdit(t *testing.T) {
	schema := ToolCallSchema("edit")

	if len(schema.Properties) != 3 {
		t.Errorf("edit schema should have 3 properties, got %d", len(schema.Properties))
	}
	if len(schema.Required) != 3 {
		t.Errorf("edit should have 3 required fields, got %d", len(schema.Required))
	}
}

func TestToolCallSchemaLs(t *testing.T) {
	schema := ToolCallSchema("ls")

	if len(schema.Properties) != 1 {
		t.Errorf("ls schema should have 1 property, got %d", len(schema.Properties))
	}
	if len(schema.Required) != 0 {
		t.Errorf("ls should have 0 required fields, got %d", len(schema.Required))
	}
}

func TestToolCallSchemaUnknown(t *testing.T) {
	schema := ToolCallSchema("nonexistent")

	if len(schema.Properties) != 0 {
		t.Errorf("unknown tool should have 0 properties, got %d", len(schema.Properties))
	}
}

func TestToolSelectSchema(t *testing.T) {
	schema := ToolSelectSchema([]string{"grep", "glob", "read"})

	prop := schema.Properties["tool"]
	if len(prop.Enum) != 3 {
		t.Errorf("tool enum should have 3 options, got %d", len(prop.Enum))
	}
}

func TestTaskClassifySchema(t *testing.T) {
	schema := TaskClassifySchema()

	prop := schema.Properties["task"]
	if len(prop.Enum) != 6 {
		t.Errorf("task enum should have 6 categories, got %d", len(prop.Enum))
	}

	// Verify all categories are present
	expected := map[string]bool{
		"search": true, "read": true, "write": true,
		"list": true, "explain": true, "chat": true,
	}
	for _, e := range prop.Enum {
		if !expected[e] {
			t.Errorf("unexpected category: %q", e)
		}
	}
}

func TestExtractValueSchema(t *testing.T) {
	schema := ExtractValueSchema("pattern", "the search term")

	prop, ok := schema.Properties["pattern"]
	if !ok {
		t.Fatal("missing pattern property")
	}
	if prop.Description != "the search term" {
		t.Errorf("description = %q", prop.Description)
	}
	if len(schema.Required) != 1 || schema.Required[0] != "pattern" {
		t.Errorf("required = %v, want [pattern]", schema.Required)
	}
}

func TestMultiFieldSchema(t *testing.T) {
	schema := MultiFieldSchema([]FieldSpec{
		{Name: "path", Description: "file path", Required: true},
		{Name: "content", Description: "file content", Required: true},
		{Name: "mode", Description: "file mode", Required: false},
	})

	if len(schema.Properties) != 3 {
		t.Errorf("should have 3 properties, got %d", len(schema.Properties))
	}
	if len(schema.Required) != 2 {
		t.Errorf("should have 2 required fields, got %d", len(schema.Required))
	}
}

// --- Schema Serialization Tests ---

func TestSchemaSerializesToValidJSONSchema(t *testing.T) {
	schemas := []struct {
		name   string
		schema JSONSchema
	}{
		{"choice", ChoiceSchema("answer", []string{"yes", "no"})},
		{"tool_call", ToolCallSchema("grep")},
		{"task_classify", TaskClassifySchema()},
		{"extract", ExtractValueSchema("value", "extracted value")},
		{"multi", MultiFieldSchema([]FieldSpec{
			{Name: "a", Required: true},
			{Name: "b", Required: false},
		})},
	}

	for _, tt := range schemas {
		t.Run(tt.name, func(t *testing.T) {
			data, err := json.Marshal(tt.schema)
			if err != nil {
				t.Fatalf("marshal error: %v", err)
			}

			// Must be valid JSON
			var parsed map[string]any
			if err := json.Unmarshal(data, &parsed); err != nil {
				t.Fatalf("produced invalid JSON: %v", err)
			}

			// Must have type: "object"
			if parsed["type"] != "object" {
				t.Error("type should be object")
			}

			// Must have properties
			if _, ok := parsed["properties"]; !ok {
				t.Error("should have properties")
			}
		})
	}
}

// --- Tool Schema Spec Tests ---

func TestToolSchemaSpecCoverage(t *testing.T) {
	tools := []struct {
		name      string
		wantCount int
		wantReq   int
	}{
		{"read", 1, 1},
		{"grep", 2, 1},
		{"glob", 1, 1},
		{"ls", 1, 0},
		{"tree", 1, 0},
		{"write", 2, 2},
		{"edit", 3, 3},
		{"git", 1, 1},
		{"unknown", 0, 0},
	}

	for _, tt := range tools {
		t.Run(tt.name, func(t *testing.T) {
			spec := toolSchemaSpec(tt.name)
			if len(spec) != tt.wantCount {
				t.Errorf("field count = %d, want %d", len(spec), tt.wantCount)
			}

			reqCount := 0
			for _, s := range spec {
				if s.Required {
					reqCount++
				}
			}
			if reqCount != tt.wantReq {
				t.Errorf("required count = %d, want %d", reqCount, tt.wantReq)
			}
		})
	}
}

func TestToolSchemaSpecFieldTypes(t *testing.T) {
	for _, tool := range []string{"read", "grep", "glob", "ls", "tree", "write", "edit", "git"} {
		spec := toolSchemaSpec(tool)
		for _, field := range spec {
			if field.Type != "string" {
				t.Errorf("%s.%s: type = %q, expected string", tool, field.Name, field.Type)
			}
			if field.Name == "" {
				t.Errorf("%s: field has empty name", tool)
			}
		}
	}
}

// --- GrammarDecoder Creation ---

func TestGrammarDecoderCreation(t *testing.T) {
	gd := NewGrammarDecoder(nil)
	if gd == nil {
		t.Fatal("NewGrammarDecoder should not return nil")
	}
	if gd.opts.Temperature != 0.1 {
		t.Errorf("temperature = %f, want 0.1", gd.opts.Temperature)
	}
	if gd.opts.NumPredict != 64 {
		t.Errorf("num_predict = %d, want 64", gd.opts.NumPredict)
	}
}

// --- Schema Compatibility with Ollama API ---

func TestSchemaMatchesOllamaFormat(t *testing.T) {
	// Verify the schema structure matches what Ollama expects
	// Ollama wants: {"type": "object", "properties": {...}, "required": [...]}
	schema := ToolCallSchema("grep")
	data, err := json.Marshal(schema)
	if err != nil {
		t.Fatal(err)
	}

	var raw map[string]any
	json.Unmarshal(data, &raw)

	// Must have these exact top-level keys
	if _, ok := raw["type"]; !ok {
		t.Error("missing 'type' key")
	}
	if _, ok := raw["properties"]; !ok {
		t.Error("missing 'properties' key")
	}

	// Properties must have type field
	props := raw["properties"].(map[string]any)
	for name, prop := range props {
		p := prop.(map[string]any)
		if _, ok := p["type"]; !ok {
			t.Errorf("property %q missing 'type'", name)
		}
	}
}

// --- Benchmark ---

func BenchmarkToolCallSchema(b *testing.B) {
	for i := 0; i < b.N; i++ {
		ToolCallSchema("grep")
	}
}

func BenchmarkSchemaSerialize(b *testing.B) {
	schema := ToolCallSchema("grep")
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		json.Marshal(schema)
	}
}

func BenchmarkTaskClassifySchema(b *testing.B) {
	for i := 0; i < b.N; i++ {
		TaskClassifySchema()
	}
}
