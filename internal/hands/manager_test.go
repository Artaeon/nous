package hands

import (
	"context"
	"testing"
	"time"

	"github.com/artaeon/nous/internal/blackboard"
)

func testManager(t *testing.T) (*Manager, *blackboard.Blackboard) {
	t.Helper()
	dir := t.TempDir()
	store := NewStore(dir)
	board := blackboard.New()
	// Runner with nil LLM — we won't execute actual LLM calls in unit tests
	runner := &Runner{Board: board}
	return NewManager(store, runner, board), board
}

func TestManagerRegisterAndList(t *testing.T) {
	mgr, _ := testManager(t)

	err := mgr.Register(Hand{
		Name:        "test",
		Description: "test hand",
		Schedule:    "@daily",
		Enabled:     false,
		Config:      DefaultConfig(),
		Prompt:      "do stuff",
	})
	if err != nil {
		t.Fatalf("Register() error: %v", err)
	}

	list := mgr.List()
	if len(list) != 1 {
		t.Fatalf("List() returned %d, want 1", len(list))
	}
	if list[0].Name != "test" {
		t.Errorf("Name = %q, want test", list[0].Name)
	}
	if list[0].State != HandIdle {
		t.Errorf("State = %q, want idle", list[0].State)
	}
}

func TestManagerActivateDeactivate(t *testing.T) {
	mgr, _ := testManager(t)

	_ = mgr.Register(Hand{
		Name:     "toggle",
		Schedule: "@hourly",
		Config:   DefaultConfig(),
		Prompt:   "tick",
	})

	if err := mgr.Activate("toggle"); err != nil {
		t.Fatalf("Activate() error: %v", err)
	}
	h, _ := mgr.Status("toggle")
	if !h.Enabled {
		t.Error("hand should be enabled after Activate")
	}

	if err := mgr.Deactivate("toggle"); err != nil {
		t.Fatalf("Deactivate() error: %v", err)
	}
	h, _ = mgr.Status("toggle")
	if h.Enabled {
		t.Error("hand should be disabled after Deactivate")
	}
}

func TestManagerActivateNotFound(t *testing.T) {
	mgr, _ := testManager(t)
	if err := mgr.Activate("ghost"); err == nil {
		t.Error("Activate() should fail for nonexistent hand")
	}
}

func TestManagerDeactivateNotFound(t *testing.T) {
	mgr, _ := testManager(t)
	if err := mgr.Deactivate("ghost"); err == nil {
		t.Error("Deactivate() should fail for nonexistent hand")
	}
}

func TestManagerStatusNotFound(t *testing.T) {
	mgr, _ := testManager(t)
	_, err := mgr.Status("ghost")
	if err == nil {
		t.Error("Status() should fail for nonexistent hand")
	}
}

func TestManagerRunNowNotFound(t *testing.T) {
	mgr, _ := testManager(t)
	err := mgr.RunNow(context.Background(), "ghost")
	if err == nil {
		t.Error("RunNow() should fail for nonexistent hand")
	}
}

func TestManagerRunNowConcurrencyLimit(t *testing.T) {
	mgr, _ := testManager(t)
	mgr.SetMaxConcurrent(1)

	// Register and put one hand in running state
	_ = mgr.Register(Hand{Name: "a", Config: DefaultConfig(), Prompt: "x"})
	_ = mgr.Register(Hand{Name: "b", Config: DefaultConfig(), Prompt: "y"})

	mgr.mu.Lock()
	mgr.hands["a"].State = HandRunning
	mgr.running = 1
	mgr.mu.Unlock()

	err := mgr.RunNow(context.Background(), "b")
	if err == nil {
		t.Error("RunNow() should fail when concurrency limit is reached")
	}
}

func TestManagerRegisterInvalidSchedule(t *testing.T) {
	mgr, _ := testManager(t)
	err := mgr.Register(Hand{
		Name:     "bad-sched",
		Schedule: "not a cron expression",
		Config:   DefaultConfig(),
		Prompt:   "x",
	})
	if err == nil {
		t.Error("Register() should fail with invalid schedule")
	}
}

func TestManagerHistory(t *testing.T) {
	mgr, _ := testManager(t)
	_ = mgr.Register(Hand{Name: "hist", Config: DefaultConfig(), Prompt: "x"})

	// Record directly via store
	_ = mgr.store.RecordRun(RunRecord{
		HandName:  "hist",
		StartedAt: time.Now(),
		Duration:  100,
		Success:   true,
		Output:    "done",
	})

	history := mgr.History("hist")
	if len(history) != 1 {
		t.Fatalf("History() returned %d, want 1", len(history))
	}
}

func TestManagerSetMaxConcurrent(t *testing.T) {
	mgr, _ := testManager(t)
	mgr.SetMaxConcurrent(5)
	if mgr.maxConcurrent != 5 {
		t.Errorf("maxConcurrent = %d, want 5", mgr.maxConcurrent)
	}

	// Minimum is 1
	mgr.SetMaxConcurrent(0)
	if mgr.maxConcurrent != 1 {
		t.Errorf("maxConcurrent = %d, want 1 (minimum)", mgr.maxConcurrent)
	}
}

func TestManagerRunSchedulerCancellation(t *testing.T) {
	mgr, _ := testManager(t)
	ctx, cancel := context.WithCancel(context.Background())

	done := make(chan error, 1)
	go func() {
		done <- mgr.Run(ctx)
	}()

	// Cancel immediately
	cancel()

	select {
	case err := <-done:
		if err != context.Canceled {
			t.Errorf("Run() error = %v, want context.Canceled", err)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("Run() did not return after context cancellation")
	}
}
