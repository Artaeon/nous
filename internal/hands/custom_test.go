package hands

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func init() {
	// Set up known tools for validation in tests.
	SetKnownTools([]string{"fetch", "read", "write", "grep", "glob", "run"})
}

func validConfig() CustomHandConfig {
	return CustomHandConfig{
		Name:        "test-hand",
		Description: "A test hand",
		Schedule:    "@daily",
		Enabled:     true,
		Prompt:      "Do something useful.",
		Config: HandConfig{
			MaxSteps: 8,
			Timeout:  120,
			Tools:    []string{"fetch", "read"},
		},
	}
}

func TestParseValidJSON(t *testing.T) {
	cfg := validConfig()
	data, err := json.Marshal(cfg)
	if err != nil {
		t.Fatal(err)
	}

	var parsed CustomHandConfig
	if err := json.Unmarshal(data, &parsed); err != nil {
		t.Fatal(err)
	}
	if parsed.Name != "test-hand" {
		t.Errorf("expected name 'test-hand', got %q", parsed.Name)
	}
	if parsed.Config.MaxSteps != 8 {
		t.Errorf("expected max_steps 8, got %d", parsed.Config.MaxSteps)
	}
	if err := ValidateCustomHand(parsed); err != nil {
		t.Errorf("expected valid config, got: %v", err)
	}
}

func TestRejectInvalidSchedule(t *testing.T) {
	cfg := validConfig()
	cfg.Schedule = "not a cron"
	if err := ValidateCustomHand(cfg); err == nil {
		t.Error("expected error for invalid schedule")
	}
}

func TestRejectInvalidTools(t *testing.T) {
	cfg := validConfig()
	cfg.Config.Tools = []string{"fetch", "nonexistent_tool"}
	if err := ValidateCustomHand(cfg); err == nil {
		t.Error("expected error for unknown tool")
	}
}

func TestRejectMissingName(t *testing.T) {
	cfg := validConfig()
	cfg.Name = ""
	if err := ValidateCustomHand(cfg); err == nil {
		t.Error("expected error for missing name")
	}
}

func TestRejectMaxStepsOutOfRange(t *testing.T) {
	cfg := validConfig()

	cfg.Config.MaxSteps = 0
	if err := ValidateCustomHand(cfg); err == nil {
		t.Error("expected error for max_steps=0")
	}

	cfg.Config.MaxSteps = 21
	if err := ValidateCustomHand(cfg); err == nil {
		t.Error("expected error for max_steps=21")
	}

	// Boundary: valid values
	cfg.Config.MaxSteps = 1
	if err := ValidateCustomHand(cfg); err != nil {
		t.Errorf("max_steps=1 should be valid: %v", err)
	}

	cfg.Config.MaxSteps = 20
	if err := ValidateCustomHand(cfg); err != nil {
		t.Errorf("max_steps=20 should be valid: %v", err)
	}
}

func TestRejectTimeoutOutOfRange(t *testing.T) {
	cfg := validConfig()

	cfg.Config.Timeout = 5
	if err := ValidateCustomHand(cfg); err == nil {
		t.Error("expected error for timeout=5")
	}

	cfg.Config.Timeout = 700
	if err := ValidateCustomHand(cfg); err == nil {
		t.Error("expected error for timeout=700")
	}

	// Boundary: valid values
	cfg.Config.Timeout = 10
	if err := ValidateCustomHand(cfg); err != nil {
		t.Errorf("timeout=10 should be valid: %v", err)
	}

	cfg.Config.Timeout = 600
	if err := ValidateCustomHand(cfg); err != nil {
		t.Errorf("timeout=600 should be valid: %v", err)
	}
}

func TestLoadFromDirectory(t *testing.T) {
	dir := t.TempDir()

	// Write two valid configs
	for _, name := range []string{"alpha", "beta"} {
		cfg := validConfig()
		cfg.Name = name
		data, _ := json.MarshalIndent(cfg, "", "  ")
		if err := os.WriteFile(filepath.Join(dir, name+".json"), data, 0644); err != nil {
			t.Fatal(err)
		}
	}

	hands, err := LoadCustomHands(dir)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(hands) != 2 {
		t.Errorf("expected 2 hands, got %d", len(hands))
	}
}

func TestSkipNonJSONFiles(t *testing.T) {
	dir := t.TempDir()

	// Write a valid JSON hand
	cfg := validConfig()
	data, _ := json.MarshalIndent(cfg, "", "  ")
	if err := os.WriteFile(filepath.Join(dir, "valid.json"), data, 0644); err != nil {
		t.Fatal(err)
	}

	// Write a non-JSON file
	if err := os.WriteFile(filepath.Join(dir, "notes.txt"), []byte("ignore me"), 0644); err != nil {
		t.Fatal(err)
	}

	// Write a YAML file (should be skipped)
	if err := os.WriteFile(filepath.Join(dir, "config.yaml"), []byte("name: skip"), 0644); err != nil {
		t.Fatal(err)
	}

	hands, err := LoadCustomHands(dir)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(hands) != 1 {
		t.Errorf("expected 1 hand (JSON only), got %d", len(hands))
	}
}

func TestHandleEmptyDirectory(t *testing.T) {
	dir := t.TempDir()

	hands, err := LoadCustomHands(dir)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(hands) != 0 {
		t.Errorf("expected 0 hands, got %d", len(hands))
	}
}

func TestHandleNonexistentDirectory(t *testing.T) {
	hands, err := LoadCustomHands("/tmp/nonexistent-custom-hands-dir-test")
	if err != nil {
		t.Fatalf("nonexistent dir should return nil error, got: %v", err)
	}
	if hands != nil {
		t.Errorf("expected nil hands for nonexistent dir, got %d", len(hands))
	}
}

func TestRejectDuplicateNames(t *testing.T) {
	dir := t.TempDir()

	// Write two files with the same hand name
	cfg := validConfig()
	cfg.Name = "dupe"
	data, _ := json.MarshalIndent(cfg, "", "  ")
	os.WriteFile(filepath.Join(dir, "a.json"), data, 0644)
	os.WriteFile(filepath.Join(dir, "b.json"), data, 0644)

	hands, err := LoadCustomHands(dir)
	if err == nil {
		t.Error("expected error for duplicate names")
	}
	// Should still load the first one
	if len(hands) != 1 {
		t.Errorf("expected 1 hand (first of duplicates), got %d", len(hands))
	}
}

func TestCustomHandTemplate(t *testing.T) {
	data := CustomHandTemplate("my-hand")
	var cfg CustomHandConfig
	if err := json.Unmarshal(data, &cfg); err != nil {
		t.Fatalf("template should be valid JSON: %v", err)
	}
	if cfg.Name != "my-hand" {
		t.Errorf("expected name 'my-hand', got %q", cfg.Name)
	}
	if cfg.Config.MaxSteps != 8 {
		t.Errorf("expected default max_steps 8, got %d", cfg.Config.MaxSteps)
	}
}

func TestCreateCustomHandFile(t *testing.T) {
	dir := t.TempDir()
	customDir := filepath.Join(dir, "custom")

	path, err := CreateCustomHandFile(customDir, "new-hand")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if filepath.Base(path) != "new-hand.json" {
		t.Errorf("expected new-hand.json, got %s", filepath.Base(path))
	}

	// File should exist and be valid JSON
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var cfg CustomHandConfig
	if err := json.Unmarshal(data, &cfg); err != nil {
		t.Fatalf("file should contain valid JSON: %v", err)
	}

	// Creating again should fail
	_, err = CreateCustomHandFile(customDir, "new-hand")
	if err == nil {
		t.Error("expected error for duplicate file")
	}
}
