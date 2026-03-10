package blackboard

import (
	"sync"
	"testing"
	"time"
)

func TestNew(t *testing.T) {
	bb := New()
	if bb == nil {
		t.Fatal("New() returned nil")
	}
	if bb.workingMemory == nil {
		t.Fatal("workingMemory map not initialized")
	}
	if bb.subscribers == nil {
		t.Fatal("subscribers map not initialized")
	}
}

// --- Percept tests ---

func TestPostAndLatestPercept(t *testing.T) {
	bb := New()

	// Empty blackboard should return false
	_, ok := bb.LatestPercept()
	if ok {
		t.Fatal("expected no percept on empty blackboard")
	}

	p1 := Percept{Raw: "hello", Intent: "greeting", Timestamp: time.Now()}
	bb.PostPercept(p1)

	latest, ok := bb.LatestPercept()
	if !ok {
		t.Fatal("expected percept after posting")
	}
	if latest.Raw != "hello" {
		t.Errorf("expected Raw='hello', got %q", latest.Raw)
	}
	if latest.Intent != "greeting" {
		t.Errorf("expected Intent='greeting', got %q", latest.Intent)
	}
}

func TestPercepts(t *testing.T) {
	bb := New()

	bb.PostPercept(Percept{Raw: "first"})
	bb.PostPercept(Percept{Raw: "second"})
	bb.PostPercept(Percept{Raw: "third"})

	all := bb.Percepts()
	if len(all) != 3 {
		t.Fatalf("expected 3 percepts, got %d", len(all))
	}
	if all[0].Raw != "first" {
		t.Errorf("expected first percept Raw='first', got %q", all[0].Raw)
	}
	if all[2].Raw != "third" {
		t.Errorf("expected third percept Raw='third', got %q", all[2].Raw)
	}
}

func TestPerceptsReturnsCopy(t *testing.T) {
	bb := New()
	bb.PostPercept(Percept{Raw: "original"})

	all := bb.Percepts()
	all[0].Raw = "modified"

	// The original should be unchanged
	original := bb.Percepts()
	if original[0].Raw != "original" {
		t.Error("Percepts() did not return a copy; original was modified")
	}
}

func TestPerceptEntities(t *testing.T) {
	bb := New()
	p := Percept{
		Raw:      "read file.go",
		Intent:   "read_file",
		Entities: map[string]string{"path": "file.go"},
	}
	bb.PostPercept(p)

	latest, _ := bb.LatestPercept()
	if latest.Entities["path"] != "file.go" {
		t.Errorf("expected entity path='file.go', got %q", latest.Entities["path"])
	}
}

// --- Working Memory tests ---

func TestSetAndGet(t *testing.T) {
	bb := New()

	bb.Set("language", "Go")
	val, ok := bb.Get("language")
	if !ok {
		t.Fatal("expected key 'language' to exist")
	}
	if val != "Go" {
		t.Errorf("expected 'Go', got %v", val)
	}
}

func TestGetMissing(t *testing.T) {
	bb := New()
	_, ok := bb.Get("nonexistent")
	if ok {
		t.Error("expected ok=false for nonexistent key")
	}
}

func TestDelete(t *testing.T) {
	bb := New()
	bb.Set("temp", 42)
	bb.Delete("temp")

	_, ok := bb.Get("temp")
	if ok {
		t.Error("expected key to be deleted")
	}
}

func TestSetOverwrite(t *testing.T) {
	bb := New()
	bb.Set("key", "v1")
	bb.Set("key", "v2")

	val, _ := bb.Get("key")
	if val != "v2" {
		t.Errorf("expected 'v2', got %v", val)
	}
}

// --- Goal tests ---

func TestPushAndActiveGoals(t *testing.T) {
	bb := New()

	bb.PushGoal(Goal{ID: "g1", Description: "Write tests", Priority: 1, Status: "active"})
	bb.PushGoal(Goal{ID: "g2", Description: "Review code", Priority: 2, Status: "pending"})
	bb.PushGoal(Goal{ID: "g3", Description: "Old task", Priority: 3, Status: "completed"})

	active := bb.ActiveGoals()
	if len(active) != 2 {
		t.Fatalf("expected 2 active goals, got %d", len(active))
	}

	ids := map[string]bool{}
	for _, g := range active {
		ids[g.ID] = true
	}
	if !ids["g1"] || !ids["g2"] {
		t.Error("expected g1 and g2 in active goals")
	}
	if ids["g3"] {
		t.Error("completed goal g3 should not be active")
	}
}

func TestUpdateGoalStatus(t *testing.T) {
	bb := New()
	bb.PushGoal(Goal{ID: "g1", Status: "active"})

	bb.UpdateGoalStatus("g1", "completed")

	active := bb.ActiveGoals()
	if len(active) != 0 {
		t.Errorf("expected 0 active goals after completing g1, got %d", len(active))
	}
}

func TestUpdateGoalStatusNonExistent(t *testing.T) {
	bb := New()
	// Should not panic
	bb.UpdateGoalStatus("nonexistent", "completed")
}

func TestActiveGoalsFiltersCorrectly(t *testing.T) {
	bb := New()
	bb.PushGoal(Goal{ID: "g1", Status: "failed"})
	bb.PushGoal(Goal{ID: "g2", Status: "completed"})

	active := bb.ActiveGoals()
	if len(active) != 0 {
		t.Errorf("expected 0 active goals, got %d", len(active))
	}
}

// --- Plan tests ---

func TestSetAndGetPlan(t *testing.T) {
	bb := New()

	plan := Plan{
		GoalID: "g1",
		Steps: []Step{
			{ID: "s1", Description: "Step one", Tool: "read", Status: "pending"},
			{ID: "s2", Description: "Step two", Tool: "write", Status: "pending"},
		},
		Status: "draft",
	}

	bb.SetPlan(plan)

	got, ok := bb.PlanForGoal("g1")
	if !ok {
		t.Fatal("expected plan for goal g1")
	}
	if len(got.Steps) != 2 {
		t.Errorf("expected 2 steps, got %d", len(got.Steps))
	}
	if got.Status != "draft" {
		t.Errorf("expected status 'draft', got %q", got.Status)
	}
}

func TestSetPlanReplacesExisting(t *testing.T) {
	bb := New()

	bb.SetPlan(Plan{GoalID: "g1", Status: "draft"})
	bb.SetPlan(Plan{GoalID: "g1", Status: "executing"})

	got, ok := bb.PlanForGoal("g1")
	if !ok {
		t.Fatal("expected plan for goal g1")
	}
	if got.Status != "executing" {
		t.Errorf("expected plan to be replaced, got status %q", got.Status)
	}
}

func TestPlanForGoalNotFound(t *testing.T) {
	bb := New()
	_, ok := bb.PlanForGoal("nonexistent")
	if ok {
		t.Error("expected no plan for nonexistent goal")
	}
}

func TestMultiplePlansForDifferentGoals(t *testing.T) {
	bb := New()
	bb.SetPlan(Plan{GoalID: "g1", Status: "draft"})
	bb.SetPlan(Plan{GoalID: "g2", Status: "executing"})

	p1, ok1 := bb.PlanForGoal("g1")
	p2, ok2 := bb.PlanForGoal("g2")

	if !ok1 || !ok2 {
		t.Fatal("expected both plans to exist")
	}
	if p1.Status != "draft" {
		t.Errorf("expected g1 plan status 'draft', got %q", p1.Status)
	}
	if p2.Status != "executing" {
		t.Errorf("expected g2 plan status 'executing', got %q", p2.Status)
	}
}

// --- Action tests ---

func TestRecordAndRecentActions(t *testing.T) {
	bb := New()

	for i := 0; i < 5; i++ {
		bb.RecordAction(ActionRecord{
			StepID:  "s" + string(rune('0'+i)),
			Tool:    "read",
			Success: true,
		})
	}

	recent := bb.RecentActions(3)
	if len(recent) != 3 {
		t.Fatalf("expected 3 recent actions, got %d", len(recent))
	}
}

func TestRecentActionsMoreThanAvailable(t *testing.T) {
	bb := New()
	bb.RecordAction(ActionRecord{StepID: "s1"})

	recent := bb.RecentActions(10)
	if len(recent) != 1 {
		t.Fatalf("expected 1 action (only 1 exists), got %d", len(recent))
	}
}

func TestRecentActionsEmpty(t *testing.T) {
	bb := New()
	recent := bb.RecentActions(5)
	if len(recent) != 0 {
		t.Fatalf("expected 0 actions, got %d", len(recent))
	}
}

func TestRecentActionsReturnsLatest(t *testing.T) {
	bb := New()
	bb.RecordAction(ActionRecord{StepID: "first", Tool: "read"})
	bb.RecordAction(ActionRecord{StepID: "second", Tool: "write"})
	bb.RecordAction(ActionRecord{StepID: "third", Tool: "edit"})

	recent := bb.RecentActions(2)
	if recent[0].StepID != "second" {
		t.Errorf("expected 'second', got %q", recent[0].StepID)
	}
	if recent[1].StepID != "third" {
		t.Errorf("expected 'third', got %q", recent[1].StepID)
	}
}

// --- Event Bus tests ---

func TestSubscribeReceivesEvents(t *testing.T) {
	bb := New()
	ch := bb.Subscribe("percept")

	bb.PostPercept(Percept{Raw: "test"})

	select {
	case e := <-ch:
		if e.Type != "percept" {
			t.Errorf("expected event type 'percept', got %q", e.Type)
		}
	case <-time.After(time.Second):
		t.Fatal("timed out waiting for event")
	}
}

func TestWildcardSubscription(t *testing.T) {
	bb := New()
	ch := bb.Subscribe("*")

	bb.Set("key", "value")

	select {
	case e := <-ch:
		if e.Type != "memory_set" {
			t.Errorf("expected event type 'memory_set', got %q", e.Type)
		}
	case <-time.After(time.Second):
		t.Fatal("timed out waiting for wildcard event")
	}
}

func TestSubscribeDoesNotReceiveOtherTypes(t *testing.T) {
	bb := New()
	ch := bb.Subscribe("goal_pushed")

	// This emits a "percept" event, not "goal_pushed"
	bb.PostPercept(Percept{Raw: "test"})

	select {
	case e := <-ch:
		t.Errorf("should not have received event, got type %q", e.Type)
	case <-time.After(50 * time.Millisecond):
		// Good — no event received
	}
}

func TestMultipleSubscribers(t *testing.T) {
	bb := New()
	ch1 := bb.Subscribe("percept")
	ch2 := bb.Subscribe("percept")

	bb.PostPercept(Percept{Raw: "test"})

	for _, ch := range []chan Event{ch1, ch2} {
		select {
		case e := <-ch:
			if e.Type != "percept" {
				t.Errorf("expected 'percept', got %q", e.Type)
			}
		case <-time.After(time.Second):
			t.Fatal("timed out waiting for event on subscriber")
		}
	}
}

func TestEventBusEmitsForAllOperations(t *testing.T) {
	bb := New()
	ch := bb.Subscribe("*")

	expectedTypes := map[string]bool{
		"percept":         false,
		"memory_set":      false,
		"goal_pushed":     false,
		"goal_updated":    false,
		"plan_set":        false,
		"action_recorded": false,
	}

	bb.PostPercept(Percept{Raw: "test"})
	bb.Set("k", "v")
	bb.PushGoal(Goal{ID: "g1", Status: "active"})
	bb.UpdateGoalStatus("g1", "completed")
	bb.SetPlan(Plan{GoalID: "g1"})
	bb.RecordAction(ActionRecord{StepID: "s1"})

	for i := 0; i < len(expectedTypes); i++ {
		select {
		case e := <-ch:
			expectedTypes[e.Type] = true
		case <-time.After(time.Second):
			t.Fatal("timed out waiting for events")
		}
	}

	for typ, received := range expectedTypes {
		if !received {
			t.Errorf("did not receive event type %q", typ)
		}
	}
}

// --- Concurrency test ---

func TestConcurrentAccess(t *testing.T) {
	bb := New()
	var wg sync.WaitGroup

	// Concurrent writes
	for i := 0; i < 100; i++ {
		wg.Add(1)
		go func(n int) {
			defer wg.Done()
			bb.PostPercept(Percept{Raw: "concurrent"})
			bb.Set("key", n)
			bb.PushGoal(Goal{ID: "g", Status: "active"})
			bb.RecordAction(ActionRecord{StepID: "s"})
		}(i)
	}

	// Concurrent reads
	for i := 0; i < 100; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			bb.LatestPercept()
			bb.Percepts()
			bb.Get("key")
			bb.ActiveGoals()
			bb.RecentActions(5)
		}()
	}

	wg.Wait()
}
