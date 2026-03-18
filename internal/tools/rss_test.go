package tools

import (
	"strings"
	"testing"
)

var testRSSXML = []byte(`<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>Test Feed</title>
    <item>
      <title>First Article</title>
      <link>https://example.com/1</link>
      <description>Description of first article.</description>
      <pubDate>Mon, 01 Jan 2024 00:00:00 GMT</pubDate>
    </item>
    <item>
      <title>Second Article</title>
      <link>https://example.com/2</link>
      <description>Description of second article.</description>
      <pubDate>Tue, 02 Jan 2024 00:00:00 GMT</pubDate>
    </item>
    <item>
      <title>Third Article</title>
      <link>https://example.com/3</link>
      <description>Description of third article.</description>
    </item>
  </channel>
</rss>`)

var testAtomXML = []byte(`<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>Atom Test Feed</title>
  <entry>
    <title>Atom Entry One</title>
    <link href="https://example.com/atom/1"/>
    <summary>Summary of entry one.</summary>
    <updated>2024-01-01T00:00:00Z</updated>
  </entry>
  <entry>
    <title>Atom Entry Two</title>
    <link href="https://example.com/atom/2"/>
    <content>Content of entry two.</content>
    <updated>2024-01-02T00:00:00Z</updated>
  </entry>
</feed>`)

func TestParseFeedXML_RSS(t *testing.T) {
	feed, err := ParseFeedXML(testRSSXML)
	if err != nil {
		t.Fatalf("ParseFeedXML(RSS) error: %v", err)
	}

	if feed.Title != "Test Feed" {
		t.Errorf("feed title = %q, want %q", feed.Title, "Test Feed")
	}
	if len(feed.Items) != 3 {
		t.Fatalf("expected 3 items, got %d", len(feed.Items))
	}
	if feed.Items[0].Title != "First Article" {
		t.Errorf("item[0].Title = %q, want %q", feed.Items[0].Title, "First Article")
	}
	if feed.Items[0].Link != "https://example.com/1" {
		t.Errorf("item[0].Link = %q, want %q", feed.Items[0].Link, "https://example.com/1")
	}
	if feed.Items[1].Description != "Description of second article." {
		t.Errorf("item[1].Description = %q", feed.Items[1].Description)
	}
}

func TestParseFeedXML_Atom(t *testing.T) {
	feed, err := ParseFeedXML(testAtomXML)
	if err != nil {
		t.Fatalf("ParseFeedXML(Atom) error: %v", err)
	}

	if feed.Title != "Atom Test Feed" {
		t.Errorf("feed title = %q, want %q", feed.Title, "Atom Test Feed")
	}
	if len(feed.Items) != 2 {
		t.Fatalf("expected 2 items, got %d", len(feed.Items))
	}
	if feed.Items[0].Title != "Atom Entry One" {
		t.Errorf("item[0].Title = %q", feed.Items[0].Title)
	}
	if feed.Items[0].Link != "https://example.com/atom/1" {
		t.Errorf("item[0].Link = %q", feed.Items[0].Link)
	}
	if feed.Items[0].Description != "Summary of entry one." {
		t.Errorf("item[0].Description = %q", feed.Items[0].Description)
	}
	// Entry two has no summary, should fall back to content
	if feed.Items[1].Description != "Content of entry two." {
		t.Errorf("item[1].Description = %q, want content fallback", feed.Items[1].Description)
	}
}

func TestParseFeedXML_Invalid(t *testing.T) {
	_, err := ParseFeedXML([]byte("not xml at all"))
	if err == nil {
		t.Error("expected error for invalid XML")
	}
}

func TestFormatFeed(t *testing.T) {
	feed := Feed{
		Title: "My Feed",
		Items: []FeedItem{
			{Title: "Item A", Link: "https://a.com", Description: "Desc A"},
			{Title: "Item B", Link: "https://b.com", Description: "Desc B"},
			{Title: "Item C", Link: "https://c.com", Description: "Desc C"},
		},
	}

	result := FormatFeed(feed, 2)
	if !strings.Contains(result, "# My Feed") {
		t.Error("expected feed title in output")
	}
	if !strings.Contains(result, "1. Item A") {
		t.Error("expected item 1")
	}
	if !strings.Contains(result, "2. Item B") {
		t.Error("expected item 2")
	}
	if strings.Contains(result, "3. Item C") {
		t.Error("item 3 should be excluded with maxItems=2")
	}
}

func TestFormatFeed_LongDescription(t *testing.T) {
	longDesc := strings.Repeat("x", 200)
	feed := Feed{
		Items: []FeedItem{{Title: "Long", Description: longDesc}},
	}

	result := FormatFeed(feed, 5)
	if strings.Contains(result, longDesc) {
		t.Error("description should be truncated")
	}
	if !strings.Contains(result, "...") {
		t.Error("truncated description should end with ...")
	}
}

func TestFeedShortcuts(t *testing.T) {
	shortcuts := []string{"tech", "technology", "world", "news", "science", "linux", "golang", "go"}
	for _, name := range shortcuts {
		if _, ok := feedShortcuts[name]; !ok {
			t.Errorf("missing feed shortcut: %s", name)
		}
	}
}

func TestListFeeds(t *testing.T) {
	result := ListFeeds()
	if !strings.Contains(result, "Available feed shortcuts:") {
		t.Error("ListFeeds should contain header")
	}
}

func TestRegisterRSSTools(t *testing.T) {
	r := NewRegistry()
	RegisterRSSTools(r)

	tool, err := r.Get("rss")
	if err != nil {
		t.Fatalf("rss tool not registered: %v", err)
	}

	// Missing feed should error
	_, err = tool.Execute(map[string]string{})
	if err == nil {
		t.Error("expected error for missing feed arg")
	}

	// "list" should return feed list
	result, err := tool.Execute(map[string]string{"feed": "list"})
	if err != nil {
		t.Fatalf("list feeds error: %v", err)
	}
	if !strings.Contains(result, "Available feed shortcuts:") {
		t.Error("expected feed list output")
	}
}
