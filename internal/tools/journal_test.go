package tools

import (
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestJournalWriteAndRead(t *testing.T) {
	path := filepath.Join(t.TempDir(), "journal.json")
	store := newJournalStoreAt(path)

	entry, err := store.Write("Had a great day at the park", 4, []string{"outdoors", "fun"})
	if err != nil {
		t.Fatalf("Write: %v", err)
	}
	if entry.Text != "Had a great day at the park" {
		t.Errorf("text = %q", entry.Text)
	}
	if entry.Mood != 4 {
		t.Errorf("mood = %d, want 4", entry.Mood)
	}
	if len(entry.Tags) != 2 {
		t.Errorf("tags = %v, want 2 tags", entry.Tags)
	}

	// Verify it appears in today's entries.
	result, err := store.Today()
	if err != nil {
		t.Fatalf("Today: %v", err)
	}
	if !strings.Contains(result, "great day at the park") {
		t.Errorf("today missing entry: %s", result)
	}
	if !strings.Contains(result, "mood:4/5") {
		t.Errorf("today missing mood: %s", result)
	}
	if !strings.Contains(result, "#outdoors") {
		t.Errorf("today missing tag: %s", result)
	}
}

func TestJournalTodayFilter(t *testing.T) {
	path := filepath.Join(t.TempDir(), "journal.json")
	store := newJournalStoreAt(path)

	// Write an entry for today.
	store.Write("Today's entry", 3, nil)

	// Manually inject an old entry.
	old := JournalEntry{
		Timestamp: time.Now().AddDate(0, 0, -5),
		Text:      "Old entry from 5 days ago",
		Mood:      2,
	}
	store.entries = append(store.entries, old)
	store.save()

	result, err := store.Today()
	if err != nil {
		t.Fatalf("Today: %v", err)
	}
	if !strings.Contains(result, "Today's entry") {
		t.Errorf("today should contain today's entry: %s", result)
	}
	if strings.Contains(result, "Old entry") {
		t.Errorf("today should not contain old entry: %s", result)
	}
}

func TestJournalSearch(t *testing.T) {
	path := filepath.Join(t.TempDir(), "journal.json")
	store := newJournalStoreAt(path)

	store.Write("Went running in the morning", 4, []string{"exercise"})
	store.Write("Read a book about history", 3, []string{"reading"})
	store.Write("Morning yoga session", 5, []string{"exercise"})

	result, err := store.Search("morning")
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if !strings.Contains(result, "2 match") {
		t.Errorf("expected 2 matches: %s", result)
	}
	if !strings.Contains(result, "running") {
		t.Errorf("search missing running entry: %s", result)
	}
	if !strings.Contains(result, "yoga") {
		t.Errorf("search missing yoga entry: %s", result)
	}

	// Search with no results.
	result, err = store.Search("nonexistent")
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if !strings.Contains(result, "No journal entries") {
		t.Errorf("expected no matches: %s", result)
	}
}

func TestJournalMoodTracking(t *testing.T) {
	path := filepath.Join(t.TempDir(), "journal.json")
	store := newJournalStoreAt(path)

	// Valid moods.
	_, err := store.Write("Bad day", 1, nil)
	if err != nil {
		t.Fatalf("Write mood 1: %v", err)
	}
	_, err = store.Write("Great day", 5, nil)
	if err != nil {
		t.Fatalf("Write mood 5: %v", err)
	}

	// No mood (0 = unset).
	_, err = store.Write("No mood entry", 0, nil)
	if err != nil {
		t.Fatalf("Write mood 0: %v", err)
	}

	// Invalid mood.
	_, err = store.Write("Invalid", 6, nil)
	if err == nil {
		t.Error("expected error for mood > 5")
	}
	_, err = store.Write("Invalid", -1, nil)
	if err == nil {
		t.Error("expected error for mood < 0")
	}
}

func TestJournalWeekSummary(t *testing.T) {
	path := filepath.Join(t.TempDir(), "journal.json")
	store := newJournalStoreAt(path)

	// Add entries within the last 7 days.
	store.entries = []JournalEntry{
		{Timestamp: time.Now(), Text: "Entry 1", Mood: 4, Tags: []string{"work"}},
		{Timestamp: time.Now().AddDate(0, 0, -1), Text: "Entry 2", Mood: 2, Tags: []string{"personal"}},
		{Timestamp: time.Now().AddDate(0, 0, -3), Text: "Entry 3", Mood: 5, Tags: []string{"work", "fun"}},
		{Timestamp: time.Now().AddDate(0, 0, -10), Text: "Old entry", Mood: 1, Tags: []string{"old"}},
	}
	store.save()

	result, err := store.Week()
	if err != nil {
		t.Fatalf("Week: %v", err)
	}
	if !strings.Contains(result, "Entries: 3") {
		t.Errorf("expected 3 entries in week: %s", result)
	}
	// Average mood: (4+2+5)/3 = 3.7
	if !strings.Contains(result, "3.7") {
		t.Errorf("expected average mood 3.7: %s", result)
	}
	if !strings.Contains(result, "fun") || !strings.Contains(result, "work") || !strings.Contains(result, "personal") {
		t.Errorf("missing tags in week summary: %s", result)
	}
	if strings.Contains(result, "old") {
		t.Errorf("week summary should not include old entries: %s", result)
	}
}

func TestJournalTagFiltering(t *testing.T) {
	path := filepath.Join(t.TempDir(), "journal.json")
	store := newJournalStoreAt(path)

	store.Write("Work meeting", 3, []string{"work"})
	store.Write("Gym session", 4, []string{"exercise"})
	store.Write("Work project", 2, []string{"work"})

	result, err := store.List("", "", "work")
	if err != nil {
		t.Fatalf("List by tag: %v", err)
	}
	if !strings.Contains(result, "Work meeting") || !strings.Contains(result, "Work project") {
		t.Errorf("tag filter missing work entries: %s", result)
	}
	if strings.Contains(result, "Gym") {
		t.Errorf("tag filter should not include exercise entry: %s", result)
	}
}

func TestJournalPersistence(t *testing.T) {
	path := filepath.Join(t.TempDir(), "journal.json")

	store1 := newJournalStoreAt(path)
	store1.Write("Persistent entry", 3, []string{"test"})

	store2 := newJournalStoreAt(path)
	result, err := store2.Today()
	if err != nil {
		t.Fatalf("Today: %v", err)
	}
	if !strings.Contains(result, "Persistent entry") {
		t.Errorf("entry not persisted: %s", result)
	}
}

func TestJournalEmptyEntry(t *testing.T) {
	path := filepath.Join(t.TempDir(), "journal.json")
	store := newJournalStoreAt(path)

	_, err := store.Write("", 3, nil)
	if err == nil {
		t.Error("expected error for empty entry text")
	}
}

func TestJournalListDateRange(t *testing.T) {
	path := filepath.Join(t.TempDir(), "journal.json")
	store := newJournalStoreAt(path)

	store.entries = []JournalEntry{
		{Timestamp: time.Date(2025, 1, 15, 10, 0, 0, 0, time.Local), Text: "Jan entry"},
		{Timestamp: time.Date(2025, 2, 15, 10, 0, 0, 0, time.Local), Text: "Feb entry"},
		{Timestamp: time.Date(2025, 3, 15, 10, 0, 0, 0, time.Local), Text: "Mar entry"},
	}
	store.save()

	result, err := store.List("2025-02-01", "2025-02-28", "")
	if err != nil {
		t.Fatalf("List: %v", err)
	}
	if !strings.Contains(result, "Feb entry") {
		t.Errorf("date range should include Feb entry: %s", result)
	}
	if strings.Contains(result, "Jan entry") || strings.Contains(result, "Mar entry") {
		t.Errorf("date range should exclude Jan/Mar: %s", result)
	}
}
