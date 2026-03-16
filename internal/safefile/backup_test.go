package safefile

import (
	"os"
	"path/filepath"
	"testing"
)

func TestWriteAtomicWithBackup(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "data.json")

	// Write initial data.
	if err := WriteAtomic(path, []byte(`{"v":1}`), 0644); err != nil {
		t.Fatal(err)
	}

	// Write again with backup.
	if err := WriteAtomicWithBackup(path, []byte(`{"v":2}`), 0644, 3); err != nil {
		t.Fatal(err)
	}

	// New file should have v2.
	data, _ := os.ReadFile(path)
	if string(data) != `{"v":2}` {
		t.Errorf("data = %s, want {\"v\":2}", data)
	}

	// Backup directory should exist with one backup.
	backups, _ := filepath.Glob(filepath.Join(dir, "backups", "data-*.json"))
	if len(backups) != 1 {
		t.Errorf("expected 1 backup, got %d", len(backups))
	}

	// Backup should contain v1.
	if len(backups) > 0 {
		bdata, _ := os.ReadFile(backups[0])
		if string(bdata) != `{"v":1}` {
			t.Errorf("backup = %s, want {\"v\":1}", bdata)
		}
	}
}

func TestBackupPruning(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "mem.json")

	// Write 6 versions — should keep only last 3 backups.
	for i := 0; i < 6; i++ {
		data := []byte(`{"v":` + string(rune('0'+i)) + `}`)
		if err := WriteAtomicWithBackup(path, data, 0644, 3); err != nil {
			t.Fatal(err)
		}
	}

	backups, _ := filepath.Glob(filepath.Join(dir, "backups", "mem-*.json"))
	if len(backups) > 3 {
		t.Errorf("expected at most 3 backups after pruning, got %d", len(backups))
	}
}

func TestBackupNoExistingFile(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "new.json")

	// First write — no existing file to back up.
	if err := WriteAtomicWithBackup(path, []byte(`{"v":1}`), 0644, 3); err != nil {
		t.Fatal(err)
	}

	// No backups should exist.
	backups, _ := filepath.Glob(filepath.Join(dir, "backups", "new-*.json"))
	if len(backups) != 0 {
		t.Errorf("expected 0 backups for first write, got %d", len(backups))
	}
}
