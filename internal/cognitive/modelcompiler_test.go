package cognitive

import (
	"strings"
	"testing"
)

// --- Model Compiler Tests ---

func TestModelCompilerCreation(t *testing.T) {
	mc := NewModelCompiler("qwen2.5:1.5b", nil, nil)
	if mc == nil {
		t.Fatal("should not return nil")
	}
}

func TestModelCompilerCompileBasic(t *testing.T) {
	mc := NewModelCompiler("qwen2.5:1.5b", nil, nil)
	prompt := mc.Compile()

	if prompt == "" {
		t.Fatal("compiled prompt should not be empty")
	}
	if !strings.Contains(prompt, "Nous") {
		t.Error("compiled prompt should include identity")
	}
	if !strings.Contains(prompt, "try-catch") {
		t.Error("compiled prompt should include language rules")
	}
}

func TestModelCompilerCompileWithDistiller(t *testing.T) {
	distiller := NewSelfDistiller("")
	// Need >= 3 failures of same type for ExportNegativeInstructions to include it
	for i := 0; i < 5; i++ {
		distiller.RecordFailure("test", "", "try-catch", "language_impossible", "if err != nil", "firewall")
	}

	mc := NewModelCompiler("qwen2.5:1.5b", distiller, nil)
	prompt := mc.Compile()

	if !strings.Contains(prompt, "AVOID") {
		t.Error("should include anti-hallucination rules from distiller")
	}
}

func TestModelCompilerCompileWithCortex(t *testing.T) {
	cortex := NewNeuralCortex(4, 3, []string{"grep", "read"}, "")
	// Train enough to show stats
	for i := 0; i < 100; i++ {
		cortex.Train([]float64{1, 0, 0, 0}, "grep")
	}

	mc := NewModelCompiler("qwen2.5:1.5b", nil, nil)
	mc.SetCortex(cortex)
	prompt := mc.Compile()

	if !strings.Contains(prompt, "100") {
		t.Error("should include cortex training stats")
	}
}

func TestModelCompilerCompileTruncation(t *testing.T) {
	mc := NewModelCompiler("qwen2.5:1.5b", nil, nil)
	prompt := mc.Compile()

	// Should respect token budget
	if len(prompt) > 2500 {
		t.Errorf("compiled prompt too long: %d chars", len(prompt))
	}
}

func TestModelCompilerGenerateModelfile(t *testing.T) {
	mc := NewModelCompiler("qwen2.5:1.5b", nil, nil)
	modelfile := mc.GenerateModelfile()

	if !strings.Contains(modelfile, "FROM qwen2.5:1.5b") {
		t.Error("modelfile should reference base model")
	}
	if !strings.Contains(modelfile, "SYSTEM") {
		t.Error("modelfile should include SYSTEM directive")
	}
	if !strings.Contains(modelfile, "PARAMETER temperature") {
		t.Error("modelfile should set temperature")
	}
	if !strings.Contains(modelfile, "Nous") {
		t.Error("modelfile should include identity in system prompt")
	}
}

func TestModelCompilerModelName(t *testing.T) {
	mc := NewModelCompiler("qwen2.5:1.5b", nil, nil)

	if mc.ModelName() != "nous-v1" {
		t.Errorf("first model should be nous-v1, got %s", mc.ModelName())
	}

	// Generate a modelfile (increments version)
	mc.GenerateModelfile()

	if mc.ModelName() != "nous-v2" {
		t.Errorf("second model should be nous-v2, got %s", mc.ModelName())
	}
}

func TestModelCompilerVersionHistory(t *testing.T) {
	mc := NewModelCompiler("qwen2.5:1.5b", nil, nil)

	mc.GenerateModelfile()
	mc.GenerateModelfile()

	if len(mc.Versions) != 2 {
		t.Errorf("should have 2 versions, got %d", len(mc.Versions))
	}
	if mc.Versions[0].Version != 1 {
		t.Error("first version should be 1")
	}
	if mc.Versions[1].Version != 2 {
		t.Error("second version should be 2")
	}
	if mc.Versions[0].ModelName != "nous-v1" {
		t.Errorf("first model name = %s, want nous-v1", mc.Versions[0].ModelName)
	}
}

func TestModelCompilerLanguageRules(t *testing.T) {
	mc := NewModelCompiler("qwen2.5:1.5b", nil, nil)
	prompt := mc.Compile()

	goRules := []string{"try-catch", "struct", "nil", "goroutine"}
	for _, rule := range goRules {
		if !strings.Contains(prompt, rule) {
			t.Errorf("compiled prompt should mention %s", rule)
		}
	}
}

func TestModelCompilerIdentityFirst(t *testing.T) {
	mc := NewModelCompiler("qwen2.5:1.5b", nil, nil)
	prompt := mc.Compile()

	// Identity should be at the very beginning
	if !strings.HasPrefix(prompt, "You are Nous") {
		t.Error("identity should be first in compiled prompt (small models attend to beginning)")
	}
}

// --- Benchmark ---

func BenchmarkModelCompile(b *testing.B) {
	distiller := NewSelfDistiller("")
	distiller.RecordFailure("test", "", "bad", "type", "good", "src")
	mc := NewModelCompiler("qwen2.5:1.5b", distiller, nil)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		mc.Compile()
	}
}
