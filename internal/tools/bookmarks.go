package tools

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// Bookmark represents a saved URL with metadata.
type Bookmark struct {
	URL       string   `json:"url"`
	Title     string   `json:"title"`
	Tags      []string `json:"tags,omitempty"`
	Notes     string   `json:"notes,omitempty"`
	CreatedAt string   `json:"created_at"`
}

// BookmarkStore manages bookmarks on disk.
type BookmarkStore struct {
	path string
}

// NewBookmarkStore creates a store at the default location.
func NewBookmarkStore() *BookmarkStore {
	home, _ := os.UserHomeDir()
	return &BookmarkStore{path: filepath.Join(home, ".nous", "bookmarks.json")}
}

func newBookmarkStoreAt(path string) *BookmarkStore {
	return &BookmarkStore{path: path}
}

func (bs *BookmarkStore) load() ([]Bookmark, error) {
	data, err := os.ReadFile(bs.path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, err
	}
	var bookmarks []Bookmark
	if err := json.Unmarshal(data, &bookmarks); err != nil {
		return nil, err
	}
	return bookmarks, nil
}

func (bs *BookmarkStore) save(bookmarks []Bookmark) error {
	if err := os.MkdirAll(filepath.Dir(bs.path), 0755); err != nil {
		return err
	}
	data, err := json.MarshalIndent(bookmarks, "", "  ")
	if err != nil {
		return err
	}
	tmp := bs.path + ".tmp"
	if err := os.WriteFile(tmp, data, 0644); err != nil {
		return err
	}
	return os.Rename(tmp, bs.path)
}

// RegisterBookmarkTools registers the bookmarks tool.
func RegisterBookmarkTools(r *Registry) {
	store := NewBookmarkStore()
	r.Register(Tool{
		Name:        "bookmarks",
		Description: "Save, list, search, and delete bookmarks. Args: action (save/list/search/delete), url (for save/delete), title (optional), tags (optional, comma-separated), notes (optional), query (for search), tag (for list filter).",
		Execute: func(args map[string]string) (string, error) {
			return toolBookmarks(store, args)
		},
	})
}

func registerBookmarkToolsAt(r *Registry, path string) {
	store := newBookmarkStoreAt(path)
	r.Register(Tool{
		Name:        "bookmarks",
		Description: "Bookmarks tool.",
		Execute: func(args map[string]string) (string, error) {
			return toolBookmarks(store, args)
		},
	})
}

func toolBookmarks(store *BookmarkStore, args map[string]string) (string, error) {
	action := args["action"]
	if action == "" {
		action = "list"
	}

	switch action {
	case "save":
		return bookmarkSave(store, args)
	case "list":
		return bookmarkList(store, args)
	case "search":
		return bookmarkSearch(store, args)
	case "delete":
		return bookmarkDelete(store, args)
	default:
		return "", fmt.Errorf("unknown action %q — use save, list, search, or delete", action)
	}
}

func bookmarkSave(store *BookmarkStore, args map[string]string) (string, error) {
	url := args["url"]
	if url == "" {
		return "", fmt.Errorf("save requires 'url' argument")
	}

	bookmarks, err := store.load()
	if err != nil {
		return "", err
	}

	title := args["title"]
	if title == "" {
		title = url
	}
	notes := args["notes"]

	var tags []string
	if t := args["tags"]; t != "" {
		for _, tag := range strings.Split(t, ",") {
			tag = strings.TrimSpace(tag)
			if tag != "" {
				tags = append(tags, tag)
			}
		}
	}

	// Update existing or append new
	found := false
	for i, bm := range bookmarks {
		if bm.URL == url {
			bookmarks[i].Title = title
			bookmarks[i].Tags = tags
			bookmarks[i].Notes = notes
			found = true
			break
		}
	}

	if !found {
		bookmarks = append(bookmarks, Bookmark{
			URL:       url,
			Title:     title,
			Tags:      tags,
			Notes:     notes,
			CreatedAt: time.Now().Format("2006-01-02 15:04"),
		})
	}

	if err := store.save(bookmarks); err != nil {
		return "", err
	}

	if found {
		return fmt.Sprintf("Updated bookmark: %s", title), nil
	}
	return fmt.Sprintf("Saved bookmark: %s", title), nil
}

func bookmarkList(store *BookmarkStore, args map[string]string) (string, error) {
	bookmarks, err := store.load()
	if err != nil {
		return "", err
	}
	if len(bookmarks) == 0 {
		return "No bookmarks saved.", nil
	}

	tagFilter := strings.ToLower(args["tag"])

	var out strings.Builder
	count := 0
	for _, bm := range bookmarks {
		if tagFilter != "" {
			found := false
			for _, t := range bm.Tags {
				if strings.ToLower(t) == tagFilter {
					found = true
					break
				}
			}
			if !found {
				continue
			}
		}
		count++
		fmt.Fprintf(&out, "• %s\n  %s\n", bm.Title, bm.URL)
		if len(bm.Tags) > 0 {
			fmt.Fprintf(&out, "  tags: %s\n", strings.Join(bm.Tags, ", "))
		}
		if bm.Notes != "" {
			fmt.Fprintf(&out, "  notes: %s\n", bm.Notes)
		}
	}

	if count == 0 {
		return fmt.Sprintf("No bookmarks with tag %q.", tagFilter), nil
	}
	return fmt.Sprintf("%d bookmark(s):\n%s", count, out.String()), nil
}

func bookmarkSearch(store *BookmarkStore, args map[string]string) (string, error) {
	query := strings.ToLower(args["query"])
	if query == "" {
		return "", fmt.Errorf("search requires 'query' argument")
	}

	bookmarks, err := store.load()
	if err != nil {
		return "", err
	}

	var out strings.Builder
	count := 0
	for _, bm := range bookmarks {
		match := strings.Contains(strings.ToLower(bm.Title), query) ||
			strings.Contains(strings.ToLower(bm.URL), query) ||
			strings.Contains(strings.ToLower(bm.Notes), query)
		if !match {
			for _, t := range bm.Tags {
				if strings.Contains(strings.ToLower(t), query) {
					match = true
					break
				}
			}
		}
		if match {
			count++
			fmt.Fprintf(&out, "• %s\n  %s\n", bm.Title, bm.URL)
			if len(bm.Tags) > 0 {
				fmt.Fprintf(&out, "  tags: %s\n", strings.Join(bm.Tags, ", "))
			}
		}
	}

	if count == 0 {
		return fmt.Sprintf("No bookmarks matching %q.", query), nil
	}
	return fmt.Sprintf("%d result(s):\n%s", count, out.String()), nil
}

func bookmarkDelete(store *BookmarkStore, args map[string]string) (string, error) {
	url := args["url"]
	if url == "" {
		return "", fmt.Errorf("delete requires 'url' argument")
	}

	bookmarks, err := store.load()
	if err != nil {
		return "", err
	}

	newBookmarks := make([]Bookmark, 0, len(bookmarks))
	found := false
	for _, bm := range bookmarks {
		if bm.URL == url {
			found = true
			continue
		}
		newBookmarks = append(newBookmarks, bm)
	}

	if !found {
		return "", fmt.Errorf("no bookmark with URL %q", url)
	}

	if err := store.save(newBookmarks); err != nil {
		return "", err
	}
	return fmt.Sprintf("Deleted bookmark: %s", url), nil
}
