package tools

import (
	"encoding/xml"
	"fmt"
	"io"
	"net/http"
	"strconv"
	"strings"
	"time"
)

// FeedItem represents a single entry from an RSS or Atom feed.
type FeedItem struct {
	Title       string
	Link        string
	Description string
	PubDate     string
}

// Feed represents a parsed RSS or Atom feed.
type Feed struct {
	Title string
	Items []FeedItem
}

// feedShortcuts maps friendly names to feed URLs.
var feedShortcuts = map[string]string{
	"tech":       "https://news.ycombinator.com/rss",
	"technology": "https://news.ycombinator.com/rss",
	"world":      "https://feeds.bbci.co.uk/news/world/rss.xml",
	"news":       "https://feeds.bbci.co.uk/news/world/rss.xml",
	"science":    "https://feeds.bbci.co.uk/news/science_and_environment/rss.xml",
	"linux":      "https://www.phoronix.com/rss.php",
	"golang":     "https://www.reddit.com/r/golang/.rss",
	"go":         "https://www.reddit.com/r/golang/.rss",
}

// RSS 2.0 XML structures
type rssXML struct {
	XMLName xml.Name `xml:"rss"`
	Channel struct {
		Title string       `xml:"title"`
		Items []rssItemXML `xml:"item"`
	} `xml:"channel"`
}

type rssItemXML struct {
	Title       string `xml:"title"`
	Link        string `xml:"link"`
	Description string `xml:"description"`
	PubDate     string `xml:"pubDate"`
}

// Atom XML structures
type atomXML struct {
	XMLName xml.Name       `xml:"feed"`
	Title   string         `xml:"title"`
	Entries []atomEntryXML `xml:"entry"`
}

type atomEntryXML struct {
	Title   string      `xml:"title"`
	Link    atomLinkXML `xml:"link"`
	Summary string      `xml:"summary"`
	Content string      `xml:"content"`
	Updated string      `xml:"updated"`
}

type atomLinkXML struct {
	Href string `xml:"href,attr"`
}

// ParseFeedXML parses raw XML bytes into a Feed, supporting both RSS 2.0 and Atom.
func ParseFeedXML(data []byte) (Feed, error) {
	// Try RSS first
	var rss rssXML
	if err := xml.Unmarshal(data, &rss); err == nil && rss.Channel.Title != "" {
		feed := Feed{Title: rss.Channel.Title}
		for _, item := range rss.Channel.Items {
			feed.Items = append(feed.Items, FeedItem{
				Title:       item.Title,
				Link:        item.Link,
				Description: item.Description,
				PubDate:     item.PubDate,
			})
		}
		return feed, nil
	}

	// Try Atom
	var atom atomXML
	if err := xml.Unmarshal(data, &atom); err == nil && atom.Title != "" {
		feed := Feed{Title: atom.Title}
		for _, entry := range atom.Entries {
			desc := entry.Summary
			if desc == "" {
				desc = entry.Content
			}
			feed.Items = append(feed.Items, FeedItem{
				Title:       entry.Title,
				Link:        entry.Link.Href,
				Description: desc,
				PubDate:     entry.Updated,
			})
		}
		return feed, nil
	}

	return Feed{}, fmt.Errorf("rss: could not parse feed as RSS or Atom")
}

// FetchFeed fetches and parses an RSS or Atom feed, returning formatted text.
func FetchFeed(url string, maxItems int) (string, error) {
	if url == "" {
		return "", fmt.Errorf("rss: feed URL is required")
	}
	if maxItems <= 0 {
		maxItems = 5
	}

	client := &http.Client{Timeout: 15 * time.Second}
	req, err := http.NewRequest("GET", url, nil)
	if err != nil {
		return "", fmt.Errorf("rss: invalid URL: %w", err)
	}
	req.Header.Set("User-Agent", "Mozilla/5.0 (compatible; Nous/1.0; +https://github.com/artaeon/nous)")
	req.Header.Set("Accept", "application/rss+xml, application/atom+xml, application/xml, text/xml")

	resp, err := client.Do(req)
	if err != nil {
		return "", fmt.Errorf("rss: fetch %s: %w", url, err)
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 400 {
		return "", fmt.Errorf("rss: HTTP %d for %s", resp.StatusCode, url)
	}

	body, err := io.ReadAll(io.LimitReader(resp.Body, 2<<20))
	if err != nil {
		return "", fmt.Errorf("rss: reading body: %w", err)
	}

	feed, err := ParseFeedXML(body)
	if err != nil {
		return "", err
	}

	return FormatFeed(feed, maxItems), nil
}

// FormatFeed formats a Feed as a numbered list.
func FormatFeed(feed Feed, maxItems int) string {
	var sb strings.Builder
	if feed.Title != "" {
		fmt.Fprintf(&sb, "# %s\n\n", feed.Title)
	}

	count := len(feed.Items)
	if count > maxItems {
		count = maxItems
	}

	for i := 0; i < count; i++ {
		item := feed.Items[i]
		fmt.Fprintf(&sb, "%d. %s\n", i+1, item.Title)
		if item.Link != "" {
			fmt.Fprintf(&sb, "   %s\n", item.Link)
		}
		if item.Description != "" {
			desc := StripTags(item.Description)
			if len(desc) > 150 {
				desc = desc[:150] + "..."
			}
			fmt.Fprintf(&sb, "   %s\n", desc)
		}
		sb.WriteString("\n")
	}

	return strings.TrimRight(sb.String(), "\n")
}

// ListFeeds returns a formatted list of available feed shortcuts.
func ListFeeds() string {
	// Deduplicate: multiple names can point to the same URL
	seen := make(map[string][]string)
	for name, url := range feedShortcuts {
		seen[url] = append(seen[url], name)
	}

	var sb strings.Builder
	sb.WriteString("Available feed shortcuts:\n")
	for url, names := range seen {
		fmt.Fprintf(&sb, "  %s -> %s\n", strings.Join(names, " / "), url)
	}
	return sb.String()
}

// RegisterRSSTools adds the rss tool to the registry.
func RegisterRSSTools(r *Registry) {
	r.Register(Tool{
		Name:        "rss",
		Description: "Read an RSS/Atom feed. Args: feed (name or URL, required), count (optional, default 5). Use feed='list' to see shortcuts.",
		Execute: func(args map[string]string) (string, error) {
			feed := args["feed"]
			if feed == "" {
				return "", fmt.Errorf("rss requires 'feed' argument (name or URL)")
			}

			if feed == "list" {
				return ListFeeds(), nil
			}

			maxItems := 5
			if v, ok := args["count"]; ok {
				if n, err := strconv.Atoi(v); err == nil && n > 0 {
					maxItems = n
				}
			}

			// Check for shortcut
			if url, ok := feedShortcuts[strings.ToLower(feed)]; ok {
				return FetchFeed(url, maxItems)
			}

			// Treat as direct URL
			return FetchFeed(feed, maxItems)
		},
	})
}
