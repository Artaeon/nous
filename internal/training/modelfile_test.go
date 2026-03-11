package training

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestDefaultModelfileConfig(t *testing.T) {
	cfg := DefaultModelfileConfig("qwen2.5:1.5b")

	if cfg.BaseModel != "qwen2.5:1.5b" {
		t.Errorf("BaseModel = %q, want %q", cfg.BaseModel, "qwen2.5:1.5b")
	}
	if cfg.Name != "nous-custom" {
		t.Errorf("Name = %q, want %q", cfg.Name, "nous-custom")
	}
	if cfg.NumCtx != 4096 {
		t.Errorf("NumCtx = %d, want 4096", cfg.NumCtx)
	}
}

func TestGenerateModelfile(t *testing.T) {
	cfg := ModelfileConfig{
		BaseModel:   "qwen2.5:1.5b",
		Temperature: 0.7,
		TopP:        0.9,
		NumCtx:      4096,
		NumPredict:  2048,
		System:      "You are Nous.",
		Stop:        []string{"<|im_end|>"},
	}

	mf := GenerateModelfile(cfg)

	if !strings.Contains(mf, "FROM qwen2.5:1.5b") {
		t.Error("should contain FROM directive")
	}
	if !strings.Contains(mf, "SYSTEM") {
		t.Error("should contain SYSTEM directive")
	}
	if !strings.Contains(mf, "You are Nous.") {
		t.Error("should contain system prompt")
	}
	if !strings.Contains(mf, "PARAMETER temperature 0.7") {
		t.Error("should contain temperature parameter")
	}
	if !strings.Contains(mf, "PARAMETER num_ctx 4096") {
		t.Error("should contain num_ctx parameter")
	}
	if !strings.Contains(mf, `PARAMETER stop "<|im_end|>"`) {
		t.Error("should contain stop parameter")
	}
}

func TestGenerateModelfileWithAdapter(t *testing.T) {
	cfg := ModelfileConfig{
		BaseModel:   "qwen2.5:1.5b",
		AdapterPath: "/path/to/lora/adapter",
		Temperature: 0.7,
	}

	mf := GenerateModelfile(cfg)

	if !strings.Contains(mf, "ADAPTER /path/to/lora/adapter") {
		t.Error("should contain ADAPTER directive")
	}
}

func TestGenerateModelfileMinimal(t *testing.T) {
	cfg := ModelfileConfig{
		BaseModel: "llama3.2",
	}

	mf := GenerateModelfile(cfg)

	if !strings.Contains(mf, "FROM llama3.2") {
		t.Error("should contain FROM directive")
	}
	// Should not contain SYSTEM if empty
	if strings.Contains(mf, "SYSTEM") {
		t.Error("should not contain SYSTEM if not set")
	}
}

func TestWriteModelfile(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "Modelfile")

	cfg := DefaultModelfileConfig("qwen2.5:1.5b")
	cfg.System = NousSystemPrompt()

	if err := WriteModelfile(path, cfg); err != nil {
		t.Fatalf("WriteModelfile: %v", err)
	}

	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}

	content := string(data)
	if !strings.Contains(content, "FROM qwen2.5:1.5b") {
		t.Error("file should contain FROM directive")
	}
	if !strings.Contains(content, "Nous") {
		t.Error("file should contain Nous system prompt")
	}
}

func TestNousSystemPrompt(t *testing.T) {
	prompt := NousSystemPrompt()

	if !strings.Contains(prompt, "Nous") {
		t.Error("should mention Nous")
	}
	if !strings.Contains(prompt, "unrestricted") {
		t.Error("should be unrestricted")
	}
	if !strings.Contains(prompt, "Bible") {
		t.Error("should reference the Bible")
	}
	if !strings.Contains(prompt, `{"tool"`) {
		t.Error("should include tool call format")
	}
}

func TestFineTuneScript(t *testing.T) {
	script := FineTuneScript("/data/training.jsonl", "/output", "qwen2.5:1.5b")

	if !strings.Contains(script, "unsloth") {
		t.Error("should use unsloth")
	}
	if !strings.Contains(script, "qwen2.5:1.5b") {
		t.Error("should reference base model")
	}
	if !strings.Contains(script, "LoRA") {
		t.Error("should mention LoRA")
	}
	if !strings.Contains(script, "q4_k_m") {
		t.Error("should export as GGUF")
	}
	if !strings.Contains(script, "ollama create") {
		t.Error("should show ollama create command")
	}
}
