package safefile

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func TestWriteAtomic(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "test.json")

	data := []byte(`{"key": "value"}`)
	if err := WriteAtomic(path, data, 0644); err != nil {
		t.Fatalf("WriteAtomic: %v", err)
	}

	got, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("ReadFile: %v", err)
	}
	if string(got) != string(data) {
		t.Fatalf("content mismatch: got %q, want %q", got, data)
	}

	// Verify permissions.
	info, _ := os.Stat(path)
	if perm := info.Mode().Perm(); perm != 0644 {
		t.Fatalf("perm = %o, want 0644", perm)
	}
}

func TestWriteAtomicOverwrite(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "test.json")

	_ = os.WriteFile(path, []byte("old"), 0644)

	if err := WriteAtomic(path, []byte("new"), 0644); err != nil {
		t.Fatalf("WriteAtomic: %v", err)
	}

	got, _ := os.ReadFile(path)
	if string(got) != "new" {
		t.Fatalf("content = %q, want %q", got, "new")
	}
}

func TestWriteAtomicCreatesDir(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "sub", "deep", "file.json")

	if err := WriteAtomic(path, []byte("data"), 0644); err != nil {
		t.Fatalf("WriteAtomic: %v", err)
	}

	got, _ := os.ReadFile(path)
	if string(got) != "data" {
		t.Fatalf("content = %q, want %q", got, "data")
	}
}

func TestWriteAtomicNoTempLeftOnError(t *testing.T) {
	// Writing to a non-existent dir that can't be created (permission denied)
	// is hard to test portably, but we can at least verify no temp files remain
	// after a successful write.
	dir := t.TempDir()
	path := filepath.Join(dir, "test.json")

	_ = WriteAtomic(path, []byte("ok"), 0644)

	entries, _ := os.ReadDir(dir)
	for _, e := range entries {
		if e.Name() != "test.json" {
			t.Errorf("unexpected file left behind: %s", e.Name())
		}
	}
}

func TestWriteAtomicFunc(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "streamed.jsonl")

	items := []map[string]string{{"a": "1"}, {"b": "2"}}

	err := WriteAtomicFunc(path, 0644, func(f *os.File) error {
		enc := json.NewEncoder(f)
		for _, item := range items {
			if err := enc.Encode(item); err != nil {
				return err
			}
		}
		return nil
	})
	if err != nil {
		t.Fatalf("WriteAtomicFunc: %v", err)
	}

	got, _ := os.ReadFile(path)
	if len(got) == 0 {
		t.Fatal("empty file")
	}
}
