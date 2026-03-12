package assistant

import (
	"context"
	"testing"
	"time"

	"github.com/artaeon/nous/internal/blackboard"
)

func TestStoreAddTaskAndPersist(t *testing.T) {
	dir := t.TempDir()
	store := NewStore(dir)
	due := time.Now().Add(2 * time.Hour).Round(time.Second)
	task, err := store.AddTask("Call mom", due, "")
	if err != nil {
		t.Fatalf("AddTask() error = %v", err)
	}
	if task.Status != TaskPending {
		t.Fatalf("task status = %q, want pending", task.Status)
	}

	reloaded := NewStore(dir)
	if len(reloaded.PendingTasks()) != 1 {
		t.Fatalf("expected persisted task, got %d", len(reloaded.PendingTasks()))
	}
}

func TestStoreReloadsTasksAndPreferences(t *testing.T) {
	dir := t.TempDir()
	store := NewStore(dir)
	due := time.Now().Add(time.Hour)
	if _, err := store.AddTask("Dentist", due, "daily"); err != nil {
		t.Fatalf("AddTask() error = %v", err)
	}
	if err := store.SetPreference("language", "de"); err != nil {
		t.Fatalf("SetPreference() error = %v", err)
	}

	reloaded := NewStore(dir)
	if len(reloaded.PendingTasks()) != 1 {
		t.Fatalf("expected 1 pending task, got %d", len(reloaded.PendingTasks()))
	}
	prefs := reloaded.Preferences()
	if len(prefs) != 1 || prefs[0].Value != "de" {
		t.Fatalf("expected persisted preference, got %+v", prefs)
	}
}

func TestStoreTriggerDueCreatesNotification(t *testing.T) {
	store := NewStore(t.TempDir())
	due := time.Now().Add(-time.Minute)
	task, err := store.AddTask("Stretch", due, "")
	if err != nil {
		t.Fatalf("AddTask() error = %v", err)
	}

	notes, err := store.TriggerDue(time.Now())
	if err != nil {
		t.Fatalf("TriggerDue() error = %v", err)
	}
	if len(notes) != 1 {
		t.Fatalf("expected 1 notification, got %d", len(notes))
	}
	if notes[0].TaskID != task.ID {
		t.Fatalf("notification task id = %q, want %q", notes[0].TaskID, task.ID)
	}
	if len(store.UnreadNotifications()) != 1 {
		t.Fatalf("expected unread notification to be stored")
	}

	notes, err = store.TriggerDue(time.Now().Add(time.Minute))
	if err != nil {
		t.Fatalf("TriggerDue() second call error = %v", err)
	}
	if len(notes) != 0 {
		t.Fatalf("expected no duplicate notifications, got %d", len(notes))
	}
}

func TestStoreRecurringDailyTaskAdvancesDueDate(t *testing.T) {
	store := NewStore(t.TempDir())
	due := time.Date(2026, 3, 11, 8, 0, 0, 0, time.UTC)
	task, err := store.AddTask("Morning briefing", due, "daily")
	if err != nil {
		t.Fatalf("AddTask() error = %v", err)
	}

	_, err = store.TriggerDue(due.Add(time.Minute))
	if err != nil {
		t.Fatalf("TriggerDue() error = %v", err)
	}
	updated := store.PendingTasks()[0]
	if updated.ID != task.ID {
		t.Fatalf("unexpected task id %q", updated.ID)
	}
	if !updated.DueAt.Equal(due.Add(24 * time.Hour)) {
		t.Fatalf("due date = %v, want %v", updated.DueAt, due.Add(24*time.Hour))
	}
}

func TestStoreMarkDone(t *testing.T) {
	store := NewStore(t.TempDir())
	task, err := store.AddTask("Buy milk", time.Now().Add(time.Hour), "")
	if err != nil {
		t.Fatalf("AddTask() error = %v", err)
	}
	updated, err := store.MarkDone(task.ID)
	if err != nil {
		t.Fatalf("MarkDone() error = %v", err)
	}
	if updated.Status != TaskDone {
		t.Fatalf("status = %q, want done", updated.Status)
	}
}

func TestStoreTriggerRoutinesGeneratesDailyAndWeekdayTasksOncePerDay(t *testing.T) {
	store := NewStore(t.TempDir())
	if _, err := store.AddRoutine("Morning review", "daily", "08:00"); err != nil {
		t.Fatalf("AddRoutine daily error = %v", err)
	}
	if _, err := store.AddRoutine("Inbox zero", "weekdays", "09:00"); err != nil {
		t.Fatalf("AddRoutine weekdays error = %v", err)
	}

	now := time.Date(2026, 3, 11, 9, 30, 0, 0, time.UTC)
	generated, err := store.TriggerRoutines(now)
	if err != nil {
		t.Fatalf("TriggerRoutines() error = %v", err)
	}
	if len(generated) != 2 {
		t.Fatalf("generated = %d, want 2", len(generated))
	}

	again, err := store.TriggerRoutines(now.Add(time.Hour))
	if err != nil {
		t.Fatalf("TriggerRoutines() second error = %v", err)
	}
	if len(again) != 0 {
		t.Fatalf("expected routines to generate once per day, got %d", len(again))
	}
}

func TestStoreTriggerRoutinesSkipsWeekdaysOnWeekend(t *testing.T) {
	store := NewStore(t.TempDir())
	if _, err := store.AddRoutine("Commute prep", "weekdays", "08:00"); err != nil {
		t.Fatalf("AddRoutine() error = %v", err)
	}

	now := time.Date(2026, 3, 14, 9, 0, 0, 0, time.UTC)
	generated, err := store.TriggerRoutines(now)
	if err != nil {
		t.Fatalf("TriggerRoutines() error = %v", err)
	}
	if len(generated) != 0 {
		t.Fatalf("expected no weekend routine tasks, got %d", len(generated))
	}
}

func TestOverdue(t *testing.T) {
	store := NewStore(t.TempDir())
	now := time.Date(2026, 3, 11, 14, 0, 0, 0, time.Local)

	// Past task — overdue
	store.AddTask("Overdue task", now.Add(-2*time.Hour), "")
	// Future task — not overdue
	store.AddTask("Future task", now.Add(2*time.Hour), "")

	overdue := store.Overdue(now)
	if len(overdue) != 1 {
		t.Fatalf("Overdue() = %d, want 1", len(overdue))
	}
	if overdue[0].Title != "Overdue task" {
		t.Fatalf("got %q, want %q", overdue[0].Title, "Overdue task")
	}
}

func TestCompletedToday(t *testing.T) {
	store := NewStore(t.TempDir())
	now := time.Now()

	store.AddTask("Task A", now.Add(-time.Hour), "")
	tasks := store.PendingTasks()
	store.MarkDone(tasks[0].ID)

	done := store.CompletedToday(now)
	if len(done) != 1 {
		t.Fatalf("CompletedToday() = %d, want 1", len(done))
	}
	if done[0].Title != "Task A" {
		t.Fatalf("got %q, want %q", done[0].Title, "Task A")
	}
}

func TestActiveRoutinesForDay(t *testing.T) {
	store := NewStore(t.TempDir())

	store.AddRoutine("Daily standup", "daily", "09:00")
	store.AddRoutine("Weekday review", "weekdays", "17:00")

	// Wednesday — both should fire
	wed := time.Date(2026, 3, 11, 10, 0, 0, 0, time.Local) // Wednesday
	routines := store.ActiveRoutinesForDay(wed)
	if len(routines) != 2 {
		t.Fatalf("ActiveRoutinesForDay(Wed) = %d, want 2", len(routines))
	}

	// Saturday — only daily should fire
	sat := time.Date(2026, 3, 14, 10, 0, 0, 0, time.Local) // Saturday
	routines = store.ActiveRoutinesForDay(sat)
	if len(routines) != 1 {
		t.Fatalf("ActiveRoutinesForDay(Sat) = %d, want 1", len(routines))
	}
	if routines[0].Title != "Daily standup" {
		t.Fatalf("got %q, want %q", routines[0].Title, "Daily standup")
	}
}

func TestSchedulerPublishesNotificationsToBlackboard(t *testing.T) {
	store := NewStore(t.TempDir())
	_, err := store.AddTask("Call family", time.Now().Add(-time.Second), "")
	if err != nil {
		t.Fatalf("AddTask() error = %v", err)
	}
	board := blackboard.New()
	s := NewScheduler(store, board)
	s.Interval = 10 * time.Millisecond
	notified := make(chan Notification, 1)
	s.OnNotify = func(n Notification) { notified <- n }

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go s.Run(ctx)

	select {
	case <-notified:
	case <-time.After(500 * time.Millisecond):
		t.Fatal("scheduler did not emit notification in time")
	}

	if got, ok := board.Get("assistant_notification"); !ok || got == "" {
		t.Fatal("expected assistant_notification on blackboard")
	}
}
