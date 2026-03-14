package cognitive

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/artaeon/nous/internal/ollama"
)

func TestBranchManager_ForkAndSwitch(t *testing.T) {
	dir := t.TempDir()
	bm := NewBranchManager(filepath.Join(dir, "branches.json"))

	msgs := []ollama.Message{
		{Role: "system", Content: "you are helpful"},
		{Role: "user", Content: "hello"},
		{Role: "assistant", Content: "hi there"},
	}

	id := bm.Fork("test-branch", msgs)
	if id == "" {
		t.Fatal("Fork returned empty ID")
	}

	cur := bm.Current()
	if cur == nil {
		t.Fatal("Current() returned nil after Fork")
	}
	if cur.Name != "test-branch" {
		t.Errorf("expected name 'test-branch', got %q", cur.Name)
	}
	if len(cur.Messages) != 3 {
		t.Errorf("expected 3 messages, got %d", len(cur.Messages))
	}

	// Fork again
	id2 := bm.Fork("second-branch", msgs[:2])
	if id2 == id {
		t.Fatal("second fork should have different ID")
	}

	// Switch back
	br, err := bm.Switch(id)
	if err != nil {
		t.Fatalf("Switch failed: %v", err)
	}
	if br.ID != id {
		t.Errorf("expected switched branch ID %s, got %s", id, br.ID)
	}
}

func TestBranchManager_List(t *testing.T) {
	dir := t.TempDir()
	bm := NewBranchManager(filepath.Join(dir, "branches.json"))

	msgs := []ollama.Message{{Role: "user", Content: "test"}}
	bm.Fork("a", msgs)
	bm.Fork("b", msgs)

	list := bm.List()
	// Should have "main", "a", and "b"
	if len(list) < 2 {
		t.Errorf("expected at least 2 branches, got %d", len(list))
	}
}

func TestBranchManager_Merge(t *testing.T) {
	dir := t.TempDir()
	bm := NewBranchManager(filepath.Join(dir, "branches.json"))

	msgs1 := []ollama.Message{
		{Role: "user", Content: "hello"},
	}
	msgs2 := []ollama.Message{
		{Role: "user", Content: "world"},
	}

	id1 := bm.Fork("first", msgs1)
	id2 := bm.Fork("second", msgs2)

	// Switch back to first
	_, err := bm.Switch(id1)
	if err != nil {
		t.Fatal(err)
	}

	// Merge second into first
	if err := bm.Merge(id2); err != nil {
		t.Fatalf("Merge failed: %v", err)
	}

	cur := bm.Current()
	if cur == nil {
		t.Fatal("current is nil after merge")
	}
	// Should now have messages from both branches
	if len(cur.Messages) < 2 {
		t.Errorf("expected at least 2 messages after merge, got %d", len(cur.Messages))
	}
}

func TestBranchManager_Delete(t *testing.T) {
	dir := t.TempDir()
	bm := NewBranchManager(filepath.Join(dir, "branches.json"))

	msgs := []ollama.Message{{Role: "user", Content: "test"}}
	id := bm.Fork("to-delete", msgs)
	id2 := bm.Fork("keep", msgs)

	// Cannot delete current
	if err := bm.Delete(id2); err == nil {
		t.Error("expected error deleting current branch")
	}

	// Can delete non-current
	if err := bm.Delete(id); err != nil {
		t.Fatalf("Delete failed: %v", err)
	}
}

func TestBranchManager_Diff(t *testing.T) {
	dir := t.TempDir()
	bm := NewBranchManager(filepath.Join(dir, "branches.json"))

	msgs1 := []ollama.Message{
		{Role: "user", Content: "shared"},
		{Role: "user", Content: "only in first"},
	}
	msgs2 := []ollama.Message{
		{Role: "user", Content: "shared"},
		{Role: "user", Content: "only in second"},
	}

	id1 := bm.Fork("first", msgs1)
	bm.Fork("second", msgs2)

	onlyCur, onlyOther, err := bm.Diff(id1)
	if err != nil {
		t.Fatal(err)
	}
	if len(onlyCur) == 0 {
		t.Error("expected messages only in current")
	}
	if len(onlyOther) == 0 {
		t.Error("expected messages only in other")
	}
}

func TestBranchManager_Persistence(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "branches.json")
	bm := NewBranchManager(path)

	msgs := []ollama.Message{{Role: "user", Content: "persistent"}}
	id := bm.Fork("saved", msgs)
	if err := bm.Save(); err != nil {
		t.Fatalf("Save failed: %v", err)
	}

	// Verify file exists
	if _, err := os.Stat(path); err != nil {
		t.Fatalf("branch file not created: %v", err)
	}

	// Load into new manager
	bm2 := NewBranchManager(path)
	cur := bm2.Current()
	if cur == nil {
		t.Fatal("loaded manager has no current branch")
	}
	if cur.ID != id {
		t.Errorf("expected current branch %s, got %s", id, cur.ID)
	}
}

func TestBranchManager_SwitchByName(t *testing.T) {
	dir := t.TempDir()
	bm := NewBranchManager(filepath.Join(dir, "branches.json"))

	msgs := []ollama.Message{{Role: "user", Content: "test"}}
	bm.Fork("alpha", msgs)
	bm.Fork("beta", msgs)

	br, err := bm.Switch("alpha")
	if err != nil {
		t.Fatalf("Switch by name failed: %v", err)
	}
	if br.Name != "alpha" {
		t.Errorf("expected branch name 'alpha', got %q", br.Name)
	}
}
