package tools

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

// Habit represents a tracked habit.
type Habit struct {
	Frequency   string   `json:"frequency"`
	Completions []string `json:"completions"`
	CreatedAt   string   `json:"created_at"`
}

// HabitStore manages habits persisted in a JSON file.
type HabitStore struct {
	filePath string
	habits   map[string]*Habit
}

// NewHabitStore creates a new HabitStore at the default location.
func NewHabitStore() *HabitStore {
	home, _ := os.UserHomeDir()
	dir := filepath.Join(home, ".nous")
	os.MkdirAll(dir, 0755)
	return newHabitStoreAt(filepath.Join(dir, "habits.json"))
}

// newHabitStoreAt creates a HabitStore at a specific path (for testing).
func newHabitStoreAt(path string) *HabitStore {
	hs := &HabitStore{filePath: path}
	hs.load()
	return hs
}

func (hs *HabitStore) load() {
	data, err := os.ReadFile(hs.filePath)
	if err != nil {
		hs.habits = make(map[string]*Habit)
		return
	}
	hs.habits = make(map[string]*Habit)
	if err := json.Unmarshal(data, &hs.habits); err != nil {
		hs.habits = make(map[string]*Habit)
	}
}

func (hs *HabitStore) save() error {
	data, err := json.MarshalIndent(hs.habits, "", "  ")
	if err != nil {
		return fmt.Errorf("marshal habits: %w", err)
	}
	dir := filepath.Dir(hs.filePath)
	tmp, err := os.CreateTemp(dir, ".habits-*.tmp")
	if err != nil {
		return fmt.Errorf("create temp: %w", err)
	}
	tmpPath := tmp.Name()
	if _, err := tmp.Write(data); err != nil {
		tmp.Close()
		os.Remove(tmpPath)
		return fmt.Errorf("write temp: %w", err)
	}
	if err := tmp.Close(); err != nil {
		os.Remove(tmpPath)
		return fmt.Errorf("close temp: %w", err)
	}
	if err := os.Rename(tmpPath, hs.filePath); err != nil {
		os.Remove(tmpPath)
		return fmt.Errorf("rename: %w", err)
	}
	return nil
}

// Create adds a new habit.
func (hs *HabitStore) Create(name, frequency string) error {
	if name == "" {
		return fmt.Errorf("habit name is required")
	}
	if _, exists := hs.habits[name]; exists {
		return fmt.Errorf("habit %q already exists", name)
	}
	if frequency == "" {
		frequency = "daily"
	}
	if frequency != "daily" && frequency != "weekly" {
		return fmt.Errorf("frequency must be 'daily' or 'weekly'")
	}
	hs.habits[name] = &Habit{
		Frequency:   frequency,
		Completions: []string{},
		CreatedAt:   time.Now().Format("2006-01-02"),
	}
	return hs.save()
}

// Check marks a habit as done for today (idempotent).
func (hs *HabitStore) Check(name string) error {
	h, ok := hs.habits[name]
	if !ok {
		return fmt.Errorf("habit %q not found", name)
	}
	today := time.Now().Format("2006-01-02")
	for _, d := range h.Completions {
		if d == today {
			return hs.save() // Already checked, no-op but not an error.
		}
	}
	h.Completions = append(h.Completions, today)
	return hs.save()
}

// CheckDate marks a habit as done for a specific date (for testing).
func (hs *HabitStore) CheckDate(name, date string) error {
	h, ok := hs.habits[name]
	if !ok {
		return fmt.Errorf("habit %q not found", name)
	}
	for _, d := range h.Completions {
		if d == date {
			return nil
		}
	}
	h.Completions = append(h.Completions, date)
	sort.Strings(h.Completions)
	return hs.save()
}

// List shows all habits with today's status.
func (hs *HabitStore) List() (string, error) {
	if len(hs.habits) == 0 {
		return "No habits tracked.", nil
	}

	today := time.Now().Format("2006-01-02")
	var names []string
	for name := range hs.habits {
		names = append(names, name)
	}
	sort.Strings(names)

	var sb strings.Builder
	for _, name := range names {
		h := hs.habits[name]
		done := false
		for _, d := range h.Completions {
			if d == today {
				done = true
				break
			}
		}
		status := "[ ]"
		if done {
			status = "[x]"
		}
		fmt.Fprintf(&sb, "%s %s (%s)\n", status, name, h.Frequency)
	}

	return fmt.Sprintf("%d habit(s):\n%s", len(names), sb.String()), nil
}

// Status shows detailed stats for a specific habit.
func (hs *HabitStore) Status(name string) (string, error) {
	h, ok := hs.habits[name]
	if !ok {
		return "", fmt.Errorf("habit %q not found", name)
	}

	streak := hs.calculateStreak(name)
	total := len(h.Completions)
	rate := hs.completionRate(name, 30)

	var sb strings.Builder
	fmt.Fprintf(&sb, "Habit: %s (%s)\n", name, h.Frequency)
	fmt.Fprintf(&sb, "  Streak: %d day(s)\n", streak)
	fmt.Fprintf(&sb, "  Total completions: %d\n", total)
	fmt.Fprintf(&sb, "  Completion rate (30 days): %.0f%%\n", rate)
	fmt.Fprintf(&sb, "  Tracking since: %s\n", h.CreatedAt)

	return sb.String(), nil
}

// Delete removes a habit.
func (hs *HabitStore) Delete(name string) error {
	if _, ok := hs.habits[name]; !ok {
		return fmt.Errorf("habit %q not found", name)
	}
	delete(hs.habits, name)
	return hs.save()
}

// calculateStreak counts consecutive days backward from today.
func (hs *HabitStore) calculateStreak(name string) int {
	h := hs.habits[name]
	if len(h.Completions) == 0 {
		return 0
	}

	// Build a set of completion dates.
	dateSet := make(map[string]bool)
	for _, d := range h.Completions {
		dateSet[d] = true
	}

	streak := 0
	day := time.Now()
	for {
		dateStr := day.Format("2006-01-02")
		if !dateSet[dateStr] {
			break
		}
		streak++
		day = day.AddDate(0, 0, -1)
	}

	return streak
}

// completionRate calculates the percentage of days completed in the last N days.
func (hs *HabitStore) completionRate(name string, days int) float64 {
	h := hs.habits[name]
	dateSet := make(map[string]bool)
	for _, d := range h.Completions {
		dateSet[d] = true
	}

	completed := 0
	now := time.Now()
	for i := 0; i < days; i++ {
		dateStr := now.AddDate(0, 0, -i).Format("2006-01-02")
		if dateSet[dateStr] {
			completed++
		}
	}

	return float64(completed) / float64(days) * 100
}

// RegisterHabitTools adds the habits tool to the registry.
func RegisterHabitTools(r *Registry) {
	store := NewHabitStore()
	r.Register(Tool{
		Name:        "habits",
		Description: "Track daily habits. Args: action (create/check/list/status/delete), name, frequency (daily/weekly).",
		Execute: func(args map[string]string) (string, error) {
			return toolHabits(store, args)
		},
	})
}

func toolHabits(store *HabitStore, args map[string]string) (string, error) {
	action := args["action"]
	switch action {
	case "create":
		name := args["name"]
		if name == "" {
			return "", fmt.Errorf("habits create requires 'name'")
		}
		frequency := args["frequency"]
		if err := store.Create(name, frequency); err != nil {
			return "", err
		}
		return fmt.Sprintf("created habit %q", name), nil

	case "check":
		name := args["name"]
		if name == "" {
			return "", fmt.Errorf("habits check requires 'name'")
		}
		if err := store.Check(name); err != nil {
			return "", err
		}
		return fmt.Sprintf("checked off %q for today", name), nil

	case "list":
		return store.List()

	case "status":
		name := args["name"]
		if name == "" {
			return "", fmt.Errorf("habits status requires 'name'")
		}
		return store.Status(name)

	case "delete":
		name := args["name"]
		if name == "" {
			return "", fmt.Errorf("habits delete requires 'name'")
		}
		if err := store.Delete(name); err != nil {
			return "", err
		}
		return fmt.Sprintf("deleted habit %q", name), nil

	default:
		return "", fmt.Errorf("habits: unknown action %q (use create/check/list/status/delete)", action)
	}
}
