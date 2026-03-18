package tools

import (
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"time"
)

// NoteStore manages markdown notes on disk.
type NoteStore struct {
	basePath string
}

// NewNoteStore creates a new NoteStore and ensures the notes directory exists.
func NewNoteStore() *NoteStore {
	home, _ := os.UserHomeDir()
	base := filepath.Join(home, ".nous", "notes")
	os.MkdirAll(base, 0755)
	return &NoteStore{basePath: base}
}

// newNoteStoreAt creates a NoteStore rooted at a custom path (for testing).
func newNoteStoreAt(base string) *NoteStore {
	os.MkdirAll(base, 0755)
	return &NoteStore{basePath: base}
}

var noteSanitizeRe = regexp.MustCompile(`[^a-z0-9_-]+`)

// sanitizeTitle converts a title into a safe filename stem.
func sanitizeTitle(title string) string {
	s := strings.ToLower(strings.TrimSpace(title))
	s = noteSanitizeRe.ReplaceAllString(s, "-")
	s = strings.Trim(s, "-")
	if s == "" {
		s = "untitled"
	}
	return s
}

// SaveNote writes a markdown note with YAML frontmatter. Returns the file path.
func (ns *NoteStore) SaveNote(title, content string) (string, error) {
	if title == "" {
		return "", fmt.Errorf("note title is required")
	}

	filename := sanitizeTitle(title) + ".md"
	path := filepath.Join(ns.basePath, filename)

	now := time.Now().Format(time.RFC3339)
	created := now

	// If file already exists, preserve the original created timestamp.
	if existing, err := os.ReadFile(path); err == nil {
		if ts := extractFrontmatterField(string(existing), "created"); ts != "" {
			created = ts
		}
	}

	var sb strings.Builder
	sb.WriteString("---\n")
	fmt.Fprintf(&sb, "title: %s\n", title)
	fmt.Fprintf(&sb, "created: %s\n", created)
	fmt.Fprintf(&sb, "modified: %s\n", now)
	sb.WriteString("---\n\n")
	sb.WriteString(content)

	if err := os.WriteFile(path, []byte(sb.String()), 0644); err != nil {
		return "", fmt.Errorf("save note: %w", err)
	}

	return path, nil
}

// GetNote reads a note by title using fuzzy matching (case-insensitive, partial).
func (ns *NoteStore) GetNote(title string) (string, error) {
	path, err := ns.findNote(title)
	if err != nil {
		return "", err
	}

	data, err := os.ReadFile(path)
	if err != nil {
		return "", fmt.Errorf("read note: %w", err)
	}

	return string(data), nil
}

// ListNotes returns a formatted list of all notes.
func (ns *NoteStore) ListNotes() (string, error) {
	entries, err := os.ReadDir(ns.basePath)
	if err != nil {
		return "", fmt.Errorf("list notes: %w", err)
	}

	var sb strings.Builder
	count := 0
	for _, e := range entries {
		if e.IsDir() || !strings.HasSuffix(e.Name(), ".md") {
			continue
		}
		data, err := os.ReadFile(filepath.Join(ns.basePath, e.Name()))
		if err != nil {
			continue
		}
		content := string(data)
		noteTitle := extractFrontmatterField(content, "title")
		if noteTitle == "" {
			noteTitle = strings.TrimSuffix(e.Name(), ".md")
		}
		modified := extractFrontmatterField(content, "modified")
		if modified == "" {
			info, _ := e.Info()
			if info != nil {
				modified = info.ModTime().Format("2006-01-02 15:04")
			}
		} else if t, err := time.Parse(time.RFC3339, modified); err == nil {
			modified = t.Format("2006-01-02 15:04")
		}
		count++
		fmt.Fprintf(&sb, "- %s  (%s)\n", noteTitle, modified)
	}

	if count == 0 {
		return "No notes found.", nil
	}

	return fmt.Sprintf("%d note(s):\n%s", count, sb.String()), nil
}

// SearchNotes searches all notes for a keyword and returns matching snippets.
func (ns *NoteStore) SearchNotes(query string) (string, error) {
	if query == "" {
		return "", fmt.Errorf("search query is required")
	}

	entries, err := os.ReadDir(ns.basePath)
	if err != nil {
		return "", fmt.Errorf("search notes: %w", err)
	}

	lowerQuery := strings.ToLower(query)
	var sb strings.Builder
	matches := 0

	for _, e := range entries {
		if e.IsDir() || !strings.HasSuffix(e.Name(), ".md") {
			continue
		}
		data, err := os.ReadFile(filepath.Join(ns.basePath, e.Name()))
		if err != nil {
			continue
		}
		content := string(data)
		if !strings.Contains(strings.ToLower(content), lowerQuery) {
			continue
		}

		noteTitle := extractFrontmatterField(content, "title")
		if noteTitle == "" {
			noteTitle = strings.TrimSuffix(e.Name(), ".md")
		}

		// Find matching lines for snippets.
		lines := strings.Split(content, "\n")
		for _, line := range lines {
			if strings.Contains(strings.ToLower(line), lowerQuery) {
				trimmed := strings.TrimSpace(line)
				if trimmed != "" && !strings.HasPrefix(trimmed, "---") {
					matches++
					fmt.Fprintf(&sb, "[%s] %s\n", noteTitle, trimmed)
				}
			}
		}
	}

	if matches == 0 {
		return fmt.Sprintf("No matches found for %q.", query), nil
	}

	return fmt.Sprintf("%d match(es):\n%s", matches, sb.String()), nil
}

// DeleteNote removes a note by title.
func (ns *NoteStore) DeleteNote(title string) error {
	path, err := ns.findNote(title)
	if err != nil {
		return err
	}
	return os.Remove(path)
}

// AppendToNote appends content to an existing note and updates the modified timestamp.
func (ns *NoteStore) AppendToNote(title, content string) error {
	path, err := ns.findNote(title)
	if err != nil {
		return err
	}

	data, err := os.ReadFile(path)
	if err != nil {
		return fmt.Errorf("read note for append: %w", err)
	}

	existing := string(data)

	// Update the modified timestamp in frontmatter.
	now := time.Now().Format(time.RFC3339)
	if idx := strings.Index(existing, "modified:"); idx >= 0 {
		end := strings.Index(existing[idx:], "\n")
		if end >= 0 {
			existing = existing[:idx] + "modified: " + now + existing[idx+end:]
		}
	}

	updated := existing + "\n" + content
	return os.WriteFile(path, []byte(updated), 0644)
}

// findNote locates a note file by fuzzy title matching.
func (ns *NoteStore) findNote(title string) (string, error) {
	entries, err := os.ReadDir(ns.basePath)
	if err != nil {
		return "", fmt.Errorf("read notes dir: %w", err)
	}

	lowerTitle := strings.ToLower(title)

	// First pass: exact sanitized match.
	target := sanitizeTitle(title) + ".md"
	for _, e := range entries {
		if e.Name() == target {
			return filepath.Join(ns.basePath, e.Name()), nil
		}
	}

	// Second pass: case-insensitive partial match on filename or frontmatter title.
	for _, e := range entries {
		if e.IsDir() || !strings.HasSuffix(e.Name(), ".md") {
			continue
		}
		name := strings.TrimSuffix(e.Name(), ".md")
		if strings.Contains(strings.ToLower(name), lowerTitle) {
			return filepath.Join(ns.basePath, e.Name()), nil
		}
		// Check frontmatter title.
		data, err := os.ReadFile(filepath.Join(ns.basePath, e.Name()))
		if err != nil {
			continue
		}
		ft := extractFrontmatterField(string(data), "title")
		if strings.Contains(strings.ToLower(ft), lowerTitle) {
			return filepath.Join(ns.basePath, e.Name()), nil
		}
	}

	return "", fmt.Errorf("note %q not found", title)
}

// extractFrontmatterField extracts a simple key: value from YAML frontmatter.
func extractFrontmatterField(content, key string) string {
	if !strings.HasPrefix(content, "---\n") {
		return ""
	}
	end := strings.Index(content[4:], "\n---")
	if end < 0 {
		return ""
	}
	fm := content[4 : 4+end]
	for _, line := range strings.Split(fm, "\n") {
		parts := strings.SplitN(line, ":", 2)
		if len(parts) == 2 && strings.TrimSpace(parts[0]) == key {
			return strings.TrimSpace(parts[1])
		}
	}
	return ""
}

// RegisterNoteTools adds the notes tool to the registry.
func RegisterNoteTools(r *Registry) {
	store := NewNoteStore()
	r.Register(Tool{
		Name:        "notes",
		Description: "Manage markdown notes. Args: action (save/get/list/search/delete/append), title, content, query.",
		Execute: func(args map[string]string) (string, error) {
			return toolNotes(store, args)
		},
	})
}

func toolNotes(store *NoteStore, args map[string]string) (string, error) {
	action := args["action"]
	switch action {
	case "save":
		title := args["title"]
		content := args["content"]
		if title == "" {
			return "", fmt.Errorf("notes save requires 'title'")
		}
		path, err := store.SaveNote(title, content)
		if err != nil {
			return "", err
		}
		return fmt.Sprintf("saved note to %s", path), nil

	case "get":
		title := args["title"]
		if title == "" {
			return "", fmt.Errorf("notes get requires 'title'")
		}
		return store.GetNote(title)

	case "list":
		return store.ListNotes()

	case "search":
		query := args["query"]
		if query == "" {
			query = args["content"]
		}
		return store.SearchNotes(query)

	case "delete":
		title := args["title"]
		if title == "" {
			return "", fmt.Errorf("notes delete requires 'title'")
		}
		if err := store.DeleteNote(title); err != nil {
			return "", err
		}
		return fmt.Sprintf("deleted note %q", title), nil

	case "append":
		title := args["title"]
		content := args["content"]
		if title == "" {
			return "", fmt.Errorf("notes append requires 'title'")
		}
		if err := store.AppendToNote(title, content); err != nil {
			return "", err
		}
		return fmt.Sprintf("appended to note %q", title), nil

	default:
		return "", fmt.Errorf("notes: unknown action %q (use save/get/list/search/delete/append)", action)
	}
}
