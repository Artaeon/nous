package cognitive

import (
	"testing"

	"github.com/artaeon/nous/internal/memory"
)

func TestFactExtractName(t *testing.T) {
	ltm := memory.NewLongTermMemory(t.TempDir())
	fe := &FactExtractor{LTM: ltm}

	n := fe.Extract("hey! my name is Raphael and I'm a software engineer")
	if n == 0 {
		t.Fatal("expected at least 1 fact extracted")
	}

	val, ok := ltm.Retrieve("user.name")
	if !ok {
		t.Fatal("user.name not stored in LTM")
	}
	if val != "Raphael" {
		t.Errorf("user.name = %q, want Raphael", val)
	}
}

func TestFactExtractInterests(t *testing.T) {
	ltm := memory.NewLongTermMemory(t.TempDir())
	fe := &FactExtractor{LTM: ltm}

	fe.Extract("I love cosmology and reading about black holes")

	val, ok := ltm.Retrieve("user.interests")
	if !ok {
		t.Fatal("user.interests not stored in LTM")
	}
	if val == "" {
		t.Error("user.interests is empty")
	}
}

func TestFactExtractWork(t *testing.T) {
	ltm := memory.NewLongTermMemory(t.TempDir())
	fe := &FactExtractor{LTM: ltm}

	fe.Extract("I'm working on adding episodic memory to my AI project")

	val, ok := ltm.Retrieve("user.current_work")
	if !ok {
		t.Fatal("user.current_work not stored in LTM")
	}
	if val == "" {
		t.Error("user.current_work is empty")
	}
}

func TestFactExtractNoMatch(t *testing.T) {
	ltm := memory.NewLongTermMemory(t.TempDir())
	fe := &FactExtractor{LTM: ltm}

	n := fe.Extract("read go.mod")
	if n != 0 {
		t.Errorf("expected 0 facts from tool query, got %d", n)
	}
}

func TestFactExtractWorkingMemory(t *testing.T) {
	ltm := memory.NewLongTermMemory(t.TempDir())
	wm := memory.NewWorkingMemory(10)
	fe := &FactExtractor{LTM: ltm, WorkingMem: wm}

	fe.Extract("my name is Alice")

	val, ok := wm.Retrieve("user.name")
	if !ok {
		t.Fatal("user.name not in working memory")
	}
	if val != "Alice" {
		t.Errorf("working mem user.name = %v, want Alice", val)
	}
}
