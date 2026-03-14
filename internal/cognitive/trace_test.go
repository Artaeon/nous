package cognitive

import (
	"sync"
	"testing"
	"time"
)

func TestNewTraceAssignsIDAndQuery(t *testing.T) {
	tr := NewTrace("what is Go?")
	if tr.ID == "" {
		t.Fatal("expected non-empty trace ID")
	}
	if tr.Query != "what is Go?" {
		t.Fatalf("query = %q, want %q", tr.Query, "what is Go?")
	}
	if tr.StartTime.IsZero() {
		t.Fatal("expected non-zero start time")
	}
}

func TestTraceAddStepAndComplete(t *testing.T) {
	tr := NewTrace("hello")
	tr.AddStep(TraceThink, "thinking about hello")
	tr.AddStepWithTool(TraceAct, "searching files", "grep")
	tr.AddStep(TraceObserve, "found 3 results")
	tr.AddStep(TraceReflect, "the results look relevant")

	if tr.StepCount() != 4 {
		t.Fatalf("step count = %d, want 4", tr.StepCount())
	}

	tr.Complete("Hello! I found the answer.")
	if tr.FinalAnswer != "Hello! I found the answer." {
		t.Fatalf("final answer = %q", tr.FinalAnswer)
	}
	if tr.EndTime.IsZero() {
		t.Fatal("expected end time to be set after Complete")
	}

	// Check tool name on the act step
	if tr.Steps[1].ToolName != "grep" {
		t.Fatalf("step[1].ToolName = %q, want grep", tr.Steps[1].ToolName)
	}
	if tr.Steps[0].ToolName != "" {
		t.Fatalf("step[0].ToolName should be empty, got %q", tr.Steps[0].ToolName)
	}
}

func TestTraceStepDuration(t *testing.T) {
	tr := NewTrace("test duration")
	time.Sleep(5 * time.Millisecond)
	tr.AddStep(TraceThink, "step 1")

	if tr.Steps[0].Duration < time.Millisecond {
		t.Fatalf("expected duration >= 1ms, got %v", tr.Steps[0].Duration)
	}
}

func TestTraceStoreSaveAndGet(t *testing.T) {
	store := NewTraceStore(50)

	tr := NewTrace("query 1")
	tr.Complete("answer 1")
	store.Save(tr)

	got := store.Get(tr.ID)
	if got == nil {
		t.Fatal("expected to find trace by ID")
	}
	if got.Query != "query 1" {
		t.Fatalf("query = %q, want %q", got.Query, "query 1")
	}
}

func TestTraceStoreGetNotFound(t *testing.T) {
	store := NewTraceStore(50)
	if store.Get("nonexistent") != nil {
		t.Fatal("expected nil for nonexistent trace")
	}
}

func TestTraceStoreRecent(t *testing.T) {
	store := NewTraceStore(50)

	for i := 0; i < 5; i++ {
		tr := NewTrace("query")
		tr.Complete("answer")
		store.Save(tr)
	}

	recent := store.Recent(3)
	if len(recent) != 3 {
		t.Fatalf("recent count = %d, want 3", len(recent))
	}

	// Should be newest first
	if recent[0].StartTime.Before(recent[2].StartTime) {
		t.Fatal("expected newest trace first")
	}
}

func TestTraceStoreRecentMoreThanAvailable(t *testing.T) {
	store := NewTraceStore(50)
	tr := NewTrace("only one")
	tr.Complete("done")
	store.Save(tr)

	recent := store.Recent(100)
	if len(recent) != 1 {
		t.Fatalf("recent count = %d, want 1", len(recent))
	}
}

func TestTraceStoreEvictsOldest(t *testing.T) {
	store := NewTraceStore(3)

	ids := make([]string, 5)
	for i := 0; i < 5; i++ {
		tr := NewTrace("query")
		tr.Complete("answer")
		store.Save(tr)
		ids[i] = tr.ID
	}

	if store.Len() != 3 {
		t.Fatalf("store len = %d, want 3", store.Len())
	}

	// First two should be evicted
	if store.Get(ids[0]) != nil {
		t.Fatal("expected oldest trace to be evicted")
	}
	if store.Get(ids[1]) != nil {
		t.Fatal("expected second-oldest trace to be evicted")
	}
	// Last three should remain
	for _, id := range ids[2:] {
		if store.Get(id) == nil {
			t.Fatalf("expected trace %s to be present", id)
		}
	}
}

func TestTraceStoreConcurrentAccess(t *testing.T) {
	store := NewTraceStore(50)
	var wg sync.WaitGroup

	for i := 0; i < 20; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			tr := NewTrace("concurrent")
			tr.AddStep(TraceThink, "thinking")
			tr.Complete("done")
			store.Save(tr)
		}()
	}

	for i := 0; i < 20; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			_ = store.Recent(5)
		}()
	}

	wg.Wait()

	if store.Len() != 20 {
		t.Fatalf("store len = %d, want 20", store.Len())
	}
}

func TestTraceStoreDefaultCap(t *testing.T) {
	store := NewTraceStore(0)
	if store.cap != 50 {
		t.Fatalf("default cap = %d, want 50", store.cap)
	}
}
