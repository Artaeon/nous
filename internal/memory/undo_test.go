package memory

import (
	"os"
	"path/filepath"
	"testing"
)

func TestUndoStackPushAndPop(t *testing.T) {
	u := NewUndoStack(10)

	u.Push(UndoEntry{Path: "/a", Action: "write"})
	u.Push(UndoEntry{Path: "/b", Action: "edit"})

	if u.Size() != 2 {
		t.Errorf("expected size 2, got %d", u.Size())
	}

	entry, ok := u.Pop()
	if !ok {
		t.Fatal("expected to pop entry")
	}
	if entry.Path != "/b" {
		t.Errorf("expected /b, got %q", entry.Path)
	}

	entry, ok = u.Pop()
	if !ok {
		t.Fatal("expected to pop entry")
	}
	if entry.Path != "/a" {
		t.Errorf("expected /a, got %q", entry.Path)
	}

	_, ok = u.Pop()
	if ok {
		t.Error("expected Pop to return false on empty stack")
	}
}

func TestUndoStackCapacity(t *testing.T) {
	u := NewUndoStack(3)

	u.Push(UndoEntry{Path: "/a", Action: "write"})
	u.Push(UndoEntry{Path: "/b", Action: "write"})
	u.Push(UndoEntry{Path: "/c", Action: "write"})
	u.Push(UndoEntry{Path: "/d", Action: "write"})

	if u.Size() != 3 {
		t.Errorf("expected size 3 (capped), got %d", u.Size())
	}

	// Oldest (/a) should be evicted
	entry, _ := u.Pop()
	if entry.Path != "/d" {
		t.Errorf("expected /d (newest), got %q", entry.Path)
	}
	entry, _ = u.Pop()
	if entry.Path != "/c" {
		t.Errorf("expected /c, got %q", entry.Path)
	}
	entry, _ = u.Pop()
	if entry.Path != "/b" {
		t.Errorf("expected /b (oldest kept), got %q", entry.Path)
	}
}

func TestUndoStackPeek(t *testing.T) {
	u := NewUndoStack(10)

	_, ok := u.Peek()
	if ok {
		t.Error("expected Peek to return false on empty stack")
	}

	u.Push(UndoEntry{Path: "/a", Action: "write"})
	entry, ok := u.Peek()
	if !ok {
		t.Fatal("expected Peek to succeed")
	}
	if entry.Path != "/a" {
		t.Errorf("expected /a, got %q", entry.Path)
	}

	// Peek should not remove the entry
	if u.Size() != 1 {
		t.Error("Peek should not modify the stack")
	}
}

func TestUndoStackList(t *testing.T) {
	u := NewUndoStack(10)

	u.Push(UndoEntry{Path: "/a", Action: "write"})
	u.Push(UndoEntry{Path: "/b", Action: "edit"})
	u.Push(UndoEntry{Path: "/c", Action: "write"})

	list := u.List()
	if len(list) != 3 {
		t.Fatalf("expected 3 entries, got %d", len(list))
	}

	// Newest first
	if list[0].Path != "/c" {
		t.Errorf("expected /c first, got %q", list[0].Path)
	}
	if list[2].Path != "/a" {
		t.Errorf("expected /a last, got %q", list[2].Path)
	}
}

func TestUndoStackClear(t *testing.T) {
	u := NewUndoStack(10)
	u.Push(UndoEntry{Path: "/a", Action: "write"})
	u.Push(UndoEntry{Path: "/b", Action: "write"})
	u.Clear()

	if u.Size() != 0 {
		t.Errorf("expected size 0 after clear, got %d", u.Size())
	}
}

func TestUndoRevertNewFile(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "new_file.txt")

	// Create the file
	if err := os.WriteFile(path, []byte("content"), 0644); err != nil {
		t.Fatal(err)
	}

	u := NewUndoStack(10)
	u.Push(UndoEntry{Path: path, Action: "write", WasNew: true})

	msg, err := u.Undo()
	if err != nil {
		t.Fatalf("undo error: %v", err)
	}
	if msg == "" {
		t.Error("expected non-empty undo message")
	}

	// File should be deleted
	if _, err := os.Stat(path); !os.IsNotExist(err) {
		t.Error("expected file to be removed after undo")
	}
}

func TestUndoRevertExistingFile(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "existing.txt")

	// Write the modified version
	if err := os.WriteFile(path, []byte("modified"), 0644); err != nil {
		t.Fatal(err)
	}

	u := NewUndoStack(10)
	u.Push(UndoEntry{
		Path:   path,
		Action: "edit",
		Before: "original",
		After:  "modified",
		WasNew: false,
	})

	msg, err := u.Undo()
	if err != nil {
		t.Fatalf("undo error: %v", err)
	}
	if msg == "" {
		t.Error("expected non-empty undo message")
	}

	// File should be restored to original
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if string(data) != "original" {
		t.Errorf("expected 'original', got %q", string(data))
	}
}

func TestUndoEmpty(t *testing.T) {
	u := NewUndoStack(10)
	_, err := u.Undo()
	if err == nil {
		t.Error("expected error undoing empty stack")
	}
}
