package tools

import (
	"path/filepath"
	"strings"
	"testing"
)

func TestBookmarkSaveAndList(t *testing.T) {
	dir := t.TempDir()
	store := newBookmarkStoreAt(filepath.Join(dir, "bookmarks.json"))

	// Save
	result, err := toolBookmarks(store, map[string]string{
		"action": "save",
		"url":    "https://example.com",
		"title":  "Example",
		"tags":   "test, demo",
	})
	if err != nil {
		t.Fatalf("save error: %v", err)
	}
	if !strings.Contains(result, "Saved") {
		t.Errorf("expected 'Saved', got %q", result)
	}

	// List
	result, err = toolBookmarks(store, map[string]string{"action": "list"})
	if err != nil {
		t.Fatalf("list error: %v", err)
	}
	if !strings.Contains(result, "Example") {
		t.Errorf("expected 'Example' in list, got %q", result)
	}
	if !strings.Contains(result, "example.com") {
		t.Errorf("expected URL in list, got %q", result)
	}
}

func TestBookmarkSearch(t *testing.T) {
	dir := t.TempDir()
	store := newBookmarkStoreAt(filepath.Join(dir, "bookmarks.json"))

	toolBookmarks(store, map[string]string{
		"action": "save", "url": "https://golang.org", "title": "Go Language", "tags": "programming",
	})
	toolBookmarks(store, map[string]string{
		"action": "save", "url": "https://rust-lang.org", "title": "Rust Language", "tags": "programming",
	})
	toolBookmarks(store, map[string]string{
		"action": "save", "url": "https://news.ycombinator.com", "title": "Hacker News", "tags": "news",
	})

	// Search by title
	result, _ := toolBookmarks(store, map[string]string{"action": "search", "query": "go language"})
	if !strings.Contains(result, "golang.org") {
		t.Errorf("search by title failed: %q", result)
	}

	// Search by URL
	result, _ = toolBookmarks(store, map[string]string{"action": "search", "query": "rust-lang"})
	if !strings.Contains(result, "Rust") {
		t.Errorf("search by URL failed: %q", result)
	}

	// Search by tag
	result, _ = toolBookmarks(store, map[string]string{"action": "search", "query": "programming"})
	if !strings.Contains(result, "2 result") {
		t.Errorf("search by tag should find 2 results: %q", result)
	}
}

func TestBookmarkDelete(t *testing.T) {
	dir := t.TempDir()
	store := newBookmarkStoreAt(filepath.Join(dir, "bookmarks.json"))

	toolBookmarks(store, map[string]string{
		"action": "save", "url": "https://example.com", "title": "Example",
	})

	result, err := toolBookmarks(store, map[string]string{
		"action": "delete", "url": "https://example.com",
	})
	if err != nil {
		t.Fatalf("delete error: %v", err)
	}
	if !strings.Contains(result, "Deleted") {
		t.Errorf("expected 'Deleted', got %q", result)
	}

	// Verify gone
	result, _ = toolBookmarks(store, map[string]string{"action": "list"})
	if !strings.Contains(result, "No bookmarks") {
		t.Errorf("expected empty list after delete: %q", result)
	}
}

func TestBookmarkDuplicateUpdates(t *testing.T) {
	dir := t.TempDir()
	store := newBookmarkStoreAt(filepath.Join(dir, "bookmarks.json"))

	toolBookmarks(store, map[string]string{
		"action": "save", "url": "https://example.com", "title": "Original",
	})
	result, _ := toolBookmarks(store, map[string]string{
		"action": "save", "url": "https://example.com", "title": "Updated",
	})
	if !strings.Contains(result, "Updated") {
		t.Errorf("expected 'Updated', got %q", result)
	}

	// Should only have 1 bookmark
	list, _ := toolBookmarks(store, map[string]string{"action": "list"})
	if !strings.Contains(list, "1 bookmark") {
		t.Errorf("expected 1 bookmark after update: %q", list)
	}
}

func TestBookmarkTagFilter(t *testing.T) {
	dir := t.TempDir()
	store := newBookmarkStoreAt(filepath.Join(dir, "bookmarks.json"))

	toolBookmarks(store, map[string]string{
		"action": "save", "url": "https://a.com", "title": "A", "tags": "work",
	})
	toolBookmarks(store, map[string]string{
		"action": "save", "url": "https://b.com", "title": "B", "tags": "personal",
	})

	result, _ := toolBookmarks(store, map[string]string{"action": "list", "tag": "work"})
	if !strings.Contains(result, "1 bookmark") {
		t.Errorf("expected 1 bookmark with tag 'work': %q", result)
	}
	if strings.Contains(result, "b.com") {
		t.Errorf("should not include personal bookmark: %q", result)
	}
}

func TestBookmarkToolRegistration(t *testing.T) {
	r := NewRegistry()
	RegisterBookmarkTools(r)
	tool, err := r.Get("bookmarks")
	if err != nil {
		t.Fatalf("tool not registered: %v", err)
	}
	if tool.Name != "bookmarks" {
		t.Errorf("expected name 'bookmarks', got %q", tool.Name)
	}
}

func TestBookmarkEmptyList(t *testing.T) {
	dir := t.TempDir()
	store := newBookmarkStoreAt(filepath.Join(dir, "bookmarks.json"))
	result, _ := toolBookmarks(store, map[string]string{"action": "list"})
	if !strings.Contains(result, "No bookmarks") {
		t.Errorf("expected 'No bookmarks', got %q", result)
	}
}

func TestBookmarkDeleteNotFound(t *testing.T) {
	dir := t.TempDir()
	store := newBookmarkStoreAt(filepath.Join(dir, "bookmarks.json"))
	_, err := toolBookmarks(store, map[string]string{
		"action": "delete", "url": "https://nonexistent.com",
	})
	if err == nil {
		t.Error("expected error deleting nonexistent bookmark")
	}
}
