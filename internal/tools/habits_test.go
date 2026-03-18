package tools

import (
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestHabitCreateCheckStatus(t *testing.T) {
	path := filepath.Join(t.TempDir(), "habits.json")
	store := newHabitStoreAt(path)

	// Create a habit.
	err := store.Create("exercise", "daily")
	if err != nil {
		t.Fatalf("Create: %v", err)
	}

	// Check it off.
	err = store.Check("exercise")
	if err != nil {
		t.Fatalf("Check: %v", err)
	}

	// Get status.
	status, err := store.Status("exercise")
	if err != nil {
		t.Fatalf("Status: %v", err)
	}
	if !strings.Contains(status, "exercise") {
		t.Errorf("status missing name: %s", status)
	}
	if !strings.Contains(status, "Streak: 1") {
		t.Errorf("expected streak of 1: %s", status)
	}
	if !strings.Contains(status, "Total completions: 1") {
		t.Errorf("expected 1 completion: %s", status)
	}
}

func TestHabitStreakConsecutive(t *testing.T) {
	path := filepath.Join(t.TempDir(), "habits.json")
	store := newHabitStoreAt(path)

	store.Create("meditate", "daily")

	// Check for today and previous 4 days.
	now := time.Now()
	for i := 0; i < 5; i++ {
		date := now.AddDate(0, 0, -i).Format("2006-01-02")
		store.CheckDate("meditate", date)
	}

	status, err := store.Status("meditate")
	if err != nil {
		t.Fatalf("Status: %v", err)
	}
	if !strings.Contains(status, "Streak: 5") {
		t.Errorf("expected streak of 5: %s", status)
	}
}

func TestHabitStreakBroken(t *testing.T) {
	path := filepath.Join(t.TempDir(), "habits.json")
	store := newHabitStoreAt(path)

	store.Create("read", "daily")

	now := time.Now()
	// Check today and yesterday.
	store.CheckDate("read", now.Format("2006-01-02"))
	store.CheckDate("read", now.AddDate(0, 0, -1).Format("2006-01-02"))
	// Skip day -2, check day -3.
	store.CheckDate("read", now.AddDate(0, 0, -3).Format("2006-01-02"))

	status, err := store.Status("read")
	if err != nil {
		t.Fatalf("Status: %v", err)
	}
	// Streak should be 2 (today + yesterday), not 3.
	if !strings.Contains(status, "Streak: 2") {
		t.Errorf("expected streak of 2 (broken): %s", status)
	}
	if !strings.Contains(status, "Total completions: 3") {
		t.Errorf("expected 3 total: %s", status)
	}
}

func TestHabitListTodayStatus(t *testing.T) {
	path := filepath.Join(t.TempDir(), "habits.json")
	store := newHabitStoreAt(path)

	store.Create("exercise", "daily")
	store.Create("read", "daily")
	store.Check("exercise") // Check exercise for today.

	list, err := store.List()
	if err != nil {
		t.Fatalf("List: %v", err)
	}
	if !strings.Contains(list, "[x] exercise") {
		t.Errorf("exercise should be checked: %s", list)
	}
	if !strings.Contains(list, "[ ] read") {
		t.Errorf("read should be unchecked: %s", list)
	}
}

func TestHabitIdempotentCheck(t *testing.T) {
	path := filepath.Join(t.TempDir(), "habits.json")
	store := newHabitStoreAt(path)

	store.Create("meditate", "daily")
	store.Check("meditate")
	store.Check("meditate") // Second check same day.
	store.Check("meditate") // Third check same day.

	status, err := store.Status("meditate")
	if err != nil {
		t.Fatalf("Status: %v", err)
	}
	if !strings.Contains(status, "Total completions: 1") {
		t.Errorf("idempotent check failed, expected 1 completion: %s", status)
	}
}

func TestHabitDelete(t *testing.T) {
	path := filepath.Join(t.TempDir(), "habits.json")
	store := newHabitStoreAt(path)

	store.Create("exercise", "daily")
	store.Create("read", "daily")

	err := store.Delete("exercise")
	if err != nil {
		t.Fatalf("Delete: %v", err)
	}

	list, err := store.List()
	if err != nil {
		t.Fatalf("List: %v", err)
	}
	if strings.Contains(list, "exercise") {
		t.Errorf("deleted habit should not appear: %s", list)
	}
	if !strings.Contains(list, "read") {
		t.Errorf("remaining habit missing: %s", list)
	}
}

func TestHabitDeleteNotFound(t *testing.T) {
	path := filepath.Join(t.TempDir(), "habits.json")
	store := newHabitStoreAt(path)

	err := store.Delete("nonexistent")
	if err == nil {
		t.Error("expected error deleting nonexistent habit")
	}
}

func TestHabitCreateDuplicate(t *testing.T) {
	path := filepath.Join(t.TempDir(), "habits.json")
	store := newHabitStoreAt(path)

	store.Create("exercise", "daily")
	err := store.Create("exercise", "daily")
	if err == nil {
		t.Error("expected error creating duplicate habit")
	}
}

func TestHabitDefaultFrequency(t *testing.T) {
	path := filepath.Join(t.TempDir(), "habits.json")
	store := newHabitStoreAt(path)

	store.Create("exercise", "")

	status, err := store.Status("exercise")
	if err != nil {
		t.Fatalf("Status: %v", err)
	}
	if !strings.Contains(status, "daily") {
		t.Errorf("expected default frequency 'daily': %s", status)
	}
}

func TestHabitPersistence(t *testing.T) {
	path := filepath.Join(t.TempDir(), "habits.json")

	store1 := newHabitStoreAt(path)
	store1.Create("exercise", "daily")
	store1.Check("exercise")

	store2 := newHabitStoreAt(path)
	status, err := store2.Status("exercise")
	if err != nil {
		t.Fatalf("Status: %v", err)
	}
	if !strings.Contains(status, "Total completions: 1") {
		t.Errorf("not persisted: %s", status)
	}
}

func TestHabitEmptyList(t *testing.T) {
	path := filepath.Join(t.TempDir(), "habits.json")
	store := newHabitStoreAt(path)

	list, err := store.List()
	if err != nil {
		t.Fatalf("List: %v", err)
	}
	if list != "No habits tracked." {
		t.Errorf("expected empty message: %s", list)
	}
}
