package tools

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestSanitizeTitle(t *testing.T) {
	tests := []struct {
		input, want string
	}{
		{"My First Note", "my-first-note"},
		{"hello world!", "hello-world"},
		{"   spaces   ", "spaces"},
		{"ALLCAPS", "allcaps"},
		{"foo/bar:baz", "foo-bar-baz"},
		{"", "untitled"},
	}
	for _, tt := range tests {
		got := sanitizeTitle(tt.input)
		if got != tt.want {
			t.Errorf("sanitizeTitle(%q) = %q, want %q", tt.input, got, tt.want)
		}
	}
}

func TestNoteSaveAndGet(t *testing.T) {
	dir := t.TempDir()
	store := newNoteStoreAt(dir)

	path, err := store.SaveNote("Test Note", "Hello world")
	if err != nil {
		t.Fatalf("SaveNote: %v", err)
	}
	if !strings.HasSuffix(path, "test-note.md") {
		t.Errorf("unexpected path: %s", path)
	}

	// Verify file exists.
	if _, err := os.Stat(path); os.IsNotExist(err) {
		t.Fatal("note file not created")
	}

	// Get by exact title.
	content, err := store.GetNote("Test Note")
	if err != nil {
		t.Fatalf("GetNote: %v", err)
	}
	if !strings.Contains(content, "Hello world") {
		t.Error("content missing body text")
	}
	if !strings.Contains(content, "title: Test Note") {
		t.Error("content missing frontmatter title")
	}
}

func TestNoteGetFuzzy(t *testing.T) {
	dir := t.TempDir()
	store := newNoteStoreAt(dir)

	store.SaveNote("Meeting Notes January", "Discussed Q1 plans")

	// Partial, case-insensitive match.
	content, err := store.GetNote("meeting")
	if err != nil {
		t.Fatalf("GetNote fuzzy: %v", err)
	}
	if !strings.Contains(content, "Discussed Q1 plans") {
		t.Error("fuzzy match did not find correct note")
	}
}

func TestNoteList(t *testing.T) {
	dir := t.TempDir()
	store := newNoteStoreAt(dir)

	store.SaveNote("Alpha", "first")
	store.SaveNote("Beta", "second")

	list, err := store.ListNotes()
	if err != nil {
		t.Fatalf("ListNotes: %v", err)
	}
	if !strings.Contains(list, "Alpha") || !strings.Contains(list, "Beta") {
		t.Errorf("list missing notes: %s", list)
	}
	if !strings.Contains(list, "2 note(s)") {
		t.Errorf("unexpected count in list: %s", list)
	}
}

func TestNoteListEmpty(t *testing.T) {
	dir := t.TempDir()
	store := newNoteStoreAt(dir)

	list, err := store.ListNotes()
	if err != nil {
		t.Fatalf("ListNotes: %v", err)
	}
	if list != "No notes found." {
		t.Errorf("expected empty message, got: %s", list)
	}
}

func TestNoteSearch(t *testing.T) {
	dir := t.TempDir()
	store := newNoteStoreAt(dir)

	store.SaveNote("Groceries", "Buy milk and eggs")
	store.SaveNote("Work", "Finish the report")

	result, err := store.SearchNotes("milk")
	if err != nil {
		t.Fatalf("SearchNotes: %v", err)
	}
	if !strings.Contains(result, "Groceries") {
		t.Errorf("search should find Groceries note: %s", result)
	}
	if strings.Contains(result, "Work") {
		t.Errorf("search should not match Work note: %s", result)
	}
}

func TestNoteSearchNoMatch(t *testing.T) {
	dir := t.TempDir()
	store := newNoteStoreAt(dir)

	store.SaveNote("Test", "something")

	result, err := store.SearchNotes("nonexistent")
	if err != nil {
		t.Fatalf("SearchNotes: %v", err)
	}
	if !strings.Contains(result, "No matches") {
		t.Errorf("expected no matches message: %s", result)
	}
}

func TestNoteDelete(t *testing.T) {
	dir := t.TempDir()
	store := newNoteStoreAt(dir)

	store.SaveNote("To Delete", "bye")

	err := store.DeleteNote("To Delete")
	if err != nil {
		t.Fatalf("DeleteNote: %v", err)
	}

	// Should no longer exist.
	_, err = store.GetNote("To Delete")
	if err == nil {
		t.Error("expected error getting deleted note")
	}
}

func TestNoteAppend(t *testing.T) {
	dir := t.TempDir()
	store := newNoteStoreAt(dir)

	store.SaveNote("Append Test", "Line 1")

	err := store.AppendToNote("Append Test", "Line 2")
	if err != nil {
		t.Fatalf("AppendToNote: %v", err)
	}

	content, _ := store.GetNote("Append Test")
	if !strings.Contains(content, "Line 1") || !strings.Contains(content, "Line 2") {
		t.Errorf("appended content missing: %s", content)
	}
}

func TestExtractFrontmatterField(t *testing.T) {
	content := "---\ntitle: My Note\ncreated: 2024-01-01T00:00:00Z\nmodified: 2024-01-02T00:00:00Z\n---\n\nBody"

	if got := extractFrontmatterField(content, "title"); got != "My Note" {
		t.Errorf("title = %q, want 'My Note'", got)
	}
	if got := extractFrontmatterField(content, "created"); got != "2024-01-01T00:00:00Z" {
		t.Errorf("created = %q", got)
	}
	if got := extractFrontmatterField(content, "nonexistent"); got != "" {
		t.Errorf("nonexistent = %q, want empty", got)
	}
	if got := extractFrontmatterField("no frontmatter", "title"); got != "" {
		t.Errorf("no frontmatter = %q, want empty", got)
	}
}

func TestNoteSavePreservesCreated(t *testing.T) {
	dir := t.TempDir()
	store := newNoteStoreAt(dir)

	store.SaveNote("Preserve", "v1")
	content1, _ := store.GetNote("Preserve")
	created1 := extractFrontmatterField(content1, "created")

	// Re-save same note.
	store.SaveNote("Preserve", "v2")
	content2, _ := store.GetNote("Preserve")
	created2 := extractFrontmatterField(content2, "created")

	if created1 != created2 {
		t.Errorf("created timestamp changed: %s -> %s", created1, created2)
	}
	if !strings.Contains(content2, "v2") {
		t.Error("content not updated")
	}
}

func TestNoteNotFound(t *testing.T) {
	dir := t.TempDir()
	store := newNoteStoreAt(dir)

	_, err := store.GetNote("nonexistent")
	if err == nil {
		t.Error("expected error for nonexistent note")
	}

	err = store.DeleteNote("nonexistent")
	if err == nil {
		t.Error("expected error for deleting nonexistent note")
	}
}

func TestNoteStoreDirectoryCreated(t *testing.T) {
	dir := filepath.Join(t.TempDir(), "sub", "notes")
	store := newNoteStoreAt(dir)
	_ = store // just verify no panic

	if _, err := os.Stat(dir); os.IsNotExist(err) {
		t.Error("notes directory was not created")
	}
}
