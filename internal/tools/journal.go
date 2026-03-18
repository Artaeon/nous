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

// JournalEntry represents a single journal entry.
type JournalEntry struct {
	Timestamp time.Time `json:"timestamp"`
	Text      string    `json:"text"`
	Mood      int       `json:"mood,omitempty"`
	Tags      []string  `json:"tags,omitempty"`
}

// JournalStore manages journal entries persisted in a JSON file.
type JournalStore struct {
	filePath string
	entries  []JournalEntry
}

// NewJournalStore creates a new JournalStore at the default location.
func NewJournalStore() *JournalStore {
	home, _ := os.UserHomeDir()
	dir := filepath.Join(home, ".nous")
	os.MkdirAll(dir, 0755)
	return newJournalStoreAt(filepath.Join(dir, "journal.json"))
}

// newJournalStoreAt creates a JournalStore at a specific path (for testing).
func newJournalStoreAt(path string) *JournalStore {
	js := &JournalStore{filePath: path}
	js.load()
	return js
}

func (js *JournalStore) load() {
	data, err := os.ReadFile(js.filePath)
	if err != nil {
		js.entries = nil
		return
	}
	if err := json.Unmarshal(data, &js.entries); err != nil {
		js.entries = nil
	}
}

func (js *JournalStore) save() error {
	data, err := json.MarshalIndent(js.entries, "", "  ")
	if err != nil {
		return fmt.Errorf("marshal journal: %w", err)
	}
	dir := filepath.Dir(js.filePath)
	tmp, err := os.CreateTemp(dir, ".journal-*.tmp")
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
	if err := os.Rename(tmpPath, js.filePath); err != nil {
		os.Remove(tmpPath)
		return fmt.Errorf("rename: %w", err)
	}
	return nil
}

// Write adds a new journal entry.
func (js *JournalStore) Write(text string, mood int, tags []string) (*JournalEntry, error) {
	if text == "" {
		return nil, fmt.Errorf("journal entry text is required")
	}
	if mood < 0 || mood > 5 {
		return nil, fmt.Errorf("mood must be between 1 and 5 (or 0 for unset)")
	}
	entry := JournalEntry{
		Timestamp: time.Now(),
		Text:      text,
		Mood:      mood,
		Tags:      tags,
	}
	js.entries = append(js.entries, entry)
	if err := js.save(); err != nil {
		return nil, err
	}
	return &entry, nil
}

// Today returns all entries from today.
func (js *JournalStore) Today() (string, error) {
	today := time.Now().Format("2006-01-02")
	return js.entriesForDateRange(today, today)
}

// List returns entries optionally filtered by date range and/or tag.
func (js *JournalStore) List(from, to, tag string) (string, error) {
	var filtered []JournalEntry

	for _, e := range js.entries {
		dateStr := e.Timestamp.Format("2006-01-02")
		if from != "" && dateStr < from {
			continue
		}
		if to != "" && dateStr > to {
			continue
		}
		if tag != "" && !hasTag(e.Tags, tag) {
			continue
		}
		filtered = append(filtered, e)
	}

	if len(filtered) == 0 {
		return "No journal entries found.", nil
	}

	return formatEntries(filtered), nil
}

// Search performs full-text search across all entries.
func (js *JournalStore) Search(query string) (string, error) {
	if query == "" {
		return "", fmt.Errorf("search query is required")
	}
	lowerQuery := strings.ToLower(query)
	var matches []JournalEntry

	for _, e := range js.entries {
		if strings.Contains(strings.ToLower(e.Text), lowerQuery) {
			matches = append(matches, e)
		}
	}

	if len(matches) == 0 {
		return fmt.Sprintf("No journal entries matching %q.", query), nil
	}

	return fmt.Sprintf("%d match(es):\n%s", len(matches), formatEntries(matches)), nil
}

// Week returns a summary of the past 7 days.
func (js *JournalStore) Week() (string, error) {
	now := time.Now()
	weekAgo := now.AddDate(0, 0, -7).Format("2006-01-02")
	today := now.Format("2006-01-02")

	var weekEntries []JournalEntry
	for _, e := range js.entries {
		dateStr := e.Timestamp.Format("2006-01-02")
		if dateStr >= weekAgo && dateStr <= today {
			weekEntries = append(weekEntries, e)
		}
	}

	if len(weekEntries) == 0 {
		return "No journal entries in the past 7 days.", nil
	}

	// Count entries.
	count := len(weekEntries)

	// Average mood (only counting entries with mood set).
	moodSum := 0
	moodCount := 0
	for _, e := range weekEntries {
		if e.Mood > 0 {
			moodSum += e.Mood
			moodCount++
		}
	}

	// Collect unique tags.
	tagSet := make(map[string]bool)
	for _, e := range weekEntries {
		for _, t := range e.Tags {
			tagSet[t] = true
		}
	}
	var tags []string
	for t := range tagSet {
		tags = append(tags, t)
	}
	sort.Strings(tags)

	var sb strings.Builder
	fmt.Fprintf(&sb, "Week summary (%s to %s):\n", weekAgo, today)
	fmt.Fprintf(&sb, "  Entries: %d\n", count)
	if moodCount > 0 {
		avg := float64(moodSum) / float64(moodCount)
		fmt.Fprintf(&sb, "  Average mood: %.1f/5\n", avg)
	} else {
		sb.WriteString("  Average mood: N/A\n")
	}
	if len(tags) > 0 {
		fmt.Fprintf(&sb, "  Tags used: %s\n", strings.Join(tags, ", "))
	} else {
		sb.WriteString("  Tags used: none\n")
	}

	return sb.String(), nil
}

func (js *JournalStore) entriesForDateRange(from, to string) (string, error) {
	var filtered []JournalEntry
	for _, e := range js.entries {
		dateStr := e.Timestamp.Format("2006-01-02")
		if dateStr >= from && dateStr <= to {
			filtered = append(filtered, e)
		}
	}
	if len(filtered) == 0 {
		return "No journal entries found.", nil
	}
	return formatEntries(filtered), nil
}

func formatEntries(entries []JournalEntry) string {
	var sb strings.Builder
	for i, e := range entries {
		if i > 0 {
			sb.WriteString("\n")
		}
		ts := e.Timestamp.Format("2006-01-02 15:04")
		fmt.Fprintf(&sb, "[%s]", ts)
		if e.Mood > 0 {
			fmt.Fprintf(&sb, " mood:%d/5", e.Mood)
		}
		if len(e.Tags) > 0 {
			fmt.Fprintf(&sb, " #%s", strings.Join(e.Tags, " #"))
		}
		fmt.Fprintf(&sb, "\n%s\n", e.Text)
	}
	return sb.String()
}

func hasTag(tags []string, target string) bool {
	lower := strings.ToLower(target)
	for _, t := range tags {
		if strings.ToLower(t) == lower {
			return true
		}
	}
	return false
}

// RegisterJournalTools adds the journal tool to the registry.
func RegisterJournalTools(r *Registry) {
	store := NewJournalStore()
	r.Register(Tool{
		Name:        "journal",
		Description: "Personal journal. Args: action (write/today/list/search/week), entry, mood (1-5), tag, from, to, query.",
		Execute: func(args map[string]string) (string, error) {
			return toolJournal(store, args)
		},
	})
}

func toolJournal(store *JournalStore, args map[string]string) (string, error) {
	action := args["action"]
	switch action {
	case "write":
		text := args["entry"]
		if text == "" {
			return "", fmt.Errorf("journal write requires 'entry'")
		}
		mood := 0
		if m, ok := args["mood"]; ok && m != "" {
			fmt.Sscanf(m, "%d", &mood)
		}
		var tags []string
		if t, ok := args["tag"]; ok && t != "" {
			tags = strings.Split(t, ",")
			for i := range tags {
				tags[i] = strings.TrimSpace(tags[i])
			}
		}
		entry, err := store.Write(text, mood, tags)
		if err != nil {
			return "", err
		}
		ts := entry.Timestamp.Format("2006-01-02 15:04")
		return fmt.Sprintf("journal entry saved at %s", ts), nil

	case "today":
		return store.Today()

	case "list":
		from := args["from"]
		to := args["to"]
		tag := args["tag"]
		return store.List(from, to, tag)

	case "search":
		query := args["query"]
		if query == "" {
			query = args["entry"]
		}
		return store.Search(query)

	case "week":
		return store.Week()

	default:
		return "", fmt.Errorf("journal: unknown action %q (use write/today/list/search/week)", action)
	}
}
