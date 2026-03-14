package hands

import (
	"encoding/json"
	"path/filepath"
	"testing"
)

func TestExportHand(t *testing.T) {
	hand := &Hand{
		Name:        "test-hand",
		Description: "a test hand",
		Schedule:    "@daily",
		Enabled:     true,
		Config:      DefaultConfig(),
		Prompt:      "do something",
	}

	pkg, err := ExportHand(hand, "tester")
	if err != nil {
		t.Fatalf("ExportHand failed: %v", err)
	}
	if pkg.Name != "test-hand" {
		t.Errorf("expected name 'test-hand', got %q", pkg.Name)
	}
	if pkg.Author != "tester" {
		t.Errorf("expected author 'tester', got %q", pkg.Author)
	}
	if pkg.Version != "1" {
		t.Errorf("expected version '1', got %q", pkg.Version)
	}
}

func TestExportHand_Nil(t *testing.T) {
	_, err := ExportHand(nil, "tester")
	if err == nil {
		t.Error("expected error for nil hand")
	}
}

func TestImportHand(t *testing.T) {
	pkg := HandPackage{
		Version: "1",
		Name:    "imported",
		Hand: Hand{
			Name:        "imported",
			Description: "imported hand",
			Schedule:    "@hourly",
			Enabled:     true,
			Config:      DefaultConfig(),
			Prompt:      "check things",
			State:       HandRunning, // should be reset
			LastError:   "old error", // should be reset
		},
	}

	data, err := json.Marshal(pkg)
	if err != nil {
		t.Fatal(err)
	}

	hand, err := ImportHand(data)
	if err != nil {
		t.Fatalf("ImportHand failed: %v", err)
	}
	if hand.Name != "imported" {
		t.Errorf("expected name 'imported', got %q", hand.Name)
	}
	// Runtime state should be reset
	if hand.State != HandIdle {
		t.Errorf("expected idle state, got %q", hand.State)
	}
	if hand.LastError != "" {
		t.Error("expected empty last error")
	}
}

func TestImportHand_InvalidJSON(t *testing.T) {
	_, err := ImportHand([]byte("not json"))
	if err == nil {
		t.Error("expected error for invalid JSON")
	}
}

func TestImportHand_NoName(t *testing.T) {
	data, _ := json.Marshal(HandPackage{Version: "1", Hand: Hand{}})
	_, err := ImportHand(data)
	if err == nil {
		t.Error("expected error for hand with no name")
	}
}

func TestExportToFileAndImportFromFile(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "hand.json")

	hand := &Hand{
		Name:        "roundtrip",
		Description: "test roundtrip",
		Schedule:    "@startup",
		Enabled:     true,
		Config:      DefaultConfig(),
		Prompt:      "do stuff",
	}

	pkg, err := ExportHand(hand, "author")
	if err != nil {
		t.Fatal(err)
	}

	if err := ExportToFile(pkg, path); err != nil {
		t.Fatalf("ExportToFile failed: %v", err)
	}

	imported, err := ImportFromFile(path)
	if err != nil {
		t.Fatalf("ImportFromFile failed: %v", err)
	}
	if imported.Name != "roundtrip" {
		t.Errorf("expected name 'roundtrip', got %q", imported.Name)
	}
	if imported.Prompt != "do stuff" {
		t.Errorf("expected prompt 'do stuff', got %q", imported.Prompt)
	}
}

func TestImportFromFile_NotFound(t *testing.T) {
	_, err := ImportFromFile("/nonexistent/hand.json")
	if err == nil {
		t.Error("expected error for missing file")
	}
}
