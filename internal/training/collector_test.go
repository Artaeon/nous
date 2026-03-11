package training

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestCollectorCollect(t *testing.T) {
	c := NewCollector("")

	c.Collect("system", "input", "output", []string{"read"}, 0.8)

	if c.Size() != 1 {
		t.Errorf("expected 1 pair, got %d", c.Size())
	}
}

func TestCollectorQualityFilter(t *testing.T) {
	c := NewCollector("")

	c.Collect("system", "low quality", "output", nil, 0.3)

	if c.Size() != 0 {
		t.Error("low quality pair should be filtered")
	}
}

func TestCollectorPersistence(t *testing.T) {
	dir := t.TempDir()

	c1 := NewCollector(dir)
	c1.Collect("sys", "in", "out", []string{"grep"}, 0.9)
	c1.Save()

	c2 := NewCollector(dir)
	if c2.Size() != 1 {
		t.Errorf("expected 1 pair after reload, got %d", c2.Size())
	}
}

func TestCollectorExportJSONL(t *testing.T) {
	dir := t.TempDir()
	c := NewCollector("")

	c.Collect("sys", "input1", "output1", nil, 0.8)
	c.Collect("sys", "input2", "output2", nil, 0.9)

	outputPath := filepath.Join(dir, "training.jsonl")
	if err := c.ExportJSONL(outputPath); err != nil {
		t.Fatalf("ExportJSONL: %v", err)
	}

	data, err := os.ReadFile(outputPath)
	if err != nil {
		t.Fatal(err)
	}

	lines := strings.Split(strings.TrimSpace(string(data)), "\n")
	if len(lines) != 2 {
		t.Errorf("expected 2 JSONL lines, got %d", len(lines))
	}

	// Verify valid JSON
	var pair TrainingPair
	if err := json.Unmarshal([]byte(lines[0]), &pair); err != nil {
		t.Errorf("invalid JSONL: %v", err)
	}
	if pair.Input != "input1" {
		t.Errorf("input = %q, want %q", pair.Input, "input1")
	}
}

func TestCollectorExportAlpaca(t *testing.T) {
	dir := t.TempDir()
	c := NewCollector("")

	c.Collect("instruction", "my input", "my output", nil, 0.8)

	outputPath := filepath.Join(dir, "alpaca.json")
	if err := c.ExportAlpaca(outputPath); err != nil {
		t.Fatalf("ExportAlpaca: %v", err)
	}

	data, err := os.ReadFile(outputPath)
	if err != nil {
		t.Fatal(err)
	}

	var pairs []AlpacaPair
	if err := json.Unmarshal(data, &pairs); err != nil {
		t.Fatalf("invalid Alpaca JSON: %v", err)
	}
	if len(pairs) != 1 {
		t.Errorf("expected 1 pair, got %d", len(pairs))
	}
	if pairs[0].Output != "my output" {
		t.Errorf("output = %q, want %q", pairs[0].Output, "my output")
	}
}

func TestCollectorExportChatML(t *testing.T) {
	dir := t.TempDir()
	c := NewCollector("")

	c.Collect("system prompt", "user input", "assistant output", nil, 0.8)

	outputPath := filepath.Join(dir, "chatml.jsonl")
	if err := c.ExportChatML(outputPath); err != nil {
		t.Fatalf("ExportChatML: %v", err)
	}

	data, err := os.ReadFile(outputPath)
	if err != nil {
		t.Fatal(err)
	}

	var entry map[string]interface{}
	if err := json.Unmarshal([]byte(strings.TrimSpace(string(data))), &entry); err != nil {
		t.Fatalf("invalid ChatML JSON: %v", err)
	}

	messages, ok := entry["messages"].([]interface{})
	if !ok || len(messages) != 3 {
		t.Errorf("expected 3 messages (system, user, assistant), got: %v", entry)
	}
}

func TestCollectorQualityDistribution(t *testing.T) {
	c := NewCollector("")

	c.Collect("s", "i", "o", nil, 0.6)
	c.Collect("s", "i", "o", nil, 0.7)
	c.Collect("s", "i", "o", nil, 0.7)
	c.Collect("s", "i", "o", nil, 0.9)

	dist := c.QualityDistribution()
	if dist["0.6"] != 1 {
		t.Errorf("0.6 bucket = %d, want 1", dist["0.6"])
	}
	if dist["0.7"] != 2 {
		t.Errorf("0.7 bucket = %d, want 2", dist["0.7"])
	}
}

func TestCollectorPurgeBelow(t *testing.T) {
	c := NewCollector("")

	c.Collect("s", "i", "o", nil, 0.6)
	c.Collect("s", "i", "o", nil, 0.7)
	c.Collect("s", "i", "o", nil, 0.9)

	removed := c.PurgeBelow(0.8)
	if removed != 2 {
		t.Errorf("removed = %d, want 2", removed)
	}
	if c.Size() != 1 {
		t.Errorf("remaining = %d, want 1", c.Size())
	}
}
