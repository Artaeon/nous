package tools

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"regexp"
	"strconv"
	"strings"
	"time"
)

// SearchResult represents a single web search result.
type SearchResult struct {
	Title   string
	URL     string
	Snippet string
}

// Compiled regexes for DuckDuckGo HTML parsing.
var (
	reResult        = regexp.MustCompile(`(?is)<div[^>]*class\s*=\s*["'][^"']*\bresult\b[^"']*["'][^>]*>(.*?)</div>\s*(?:<div|$)`)
	reResultBlock   = regexp.MustCompile(`(?is)<div[^>]*class\s*=\s*["'][^"']*\bresult\s[^"']*["'][^>]*>`)
	reResultTitle   = regexp.MustCompile(`(?is)<a[^>]*class\s*=\s*["'][^"']*\bresult__a\b[^"']*["'][^>]*>(.*?)</a>`)
	reResultURL     = regexp.MustCompile(`(?is)<a[^>]*class\s*=\s*["'][^"']*\bresult__url\b[^"']*["'][^>]*>(.*?)</a>`)
	reResultSnippet = regexp.MustCompile(`(?is)<a[^>]*class\s*=\s*["'][^"']*\bresult__snippet\b[^"']*["'][^>]*>(.*?)</a>`)
	reResultHref    = regexp.MustCompile(`(?i)href\s*=\s*["']([^"']*?)["']`)
	reDDGRedirect   = regexp.MustCompile(`[?&]uddg=([^&]+)`)
)

// WebSearch searches DuckDuckGo and returns results.
// No API key needed -- uses the HTML search interface.
func WebSearch(query string, maxResults int) ([]SearchResult, error) {
	if query == "" {
		return nil, fmt.Errorf("search query is required")
	}
	if maxResults <= 0 {
		maxResults = 5
	}

	searchURL := "https://html.duckduckgo.com/html/?q=" + url.QueryEscape(query)

	// Rate limit
	waitForRateLimit("html.duckduckgo.com")

	client := &http.Client{
		Timeout: 10 * time.Second,
		CheckRedirect: func(req *http.Request, via []*http.Request) error {
			if len(via) >= 5 {
				return fmt.Errorf("too many redirects")
			}
			return nil
		},
	}

	req, err := http.NewRequest("GET", searchURL, nil)
	if err != nil {
		return nil, fmt.Errorf("websearch: invalid request: %w", err)
	}
	req.Header.Set("User-Agent", "Mozilla/5.0 (compatible; Nous/1.0; +https://github.com/artaeon/nous)")
	req.Header.Set("Accept", "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8")
	req.Header.Set("Accept-Language", "en-US,en;q=0.5")

	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("websearch: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 400 {
		return nil, fmt.Errorf("websearch: HTTP %d", resp.StatusCode)
	}

	body, err := io.ReadAll(io.LimitReader(resp.Body, 2<<20))
	if err != nil {
		return nil, fmt.Errorf("websearch: reading response: %w", err)
	}

	return ParseDuckDuckGoResults(string(body), maxResults), nil
}

// ParseDuckDuckGoResults extracts search results from DuckDuckGo HTML.
func ParseDuckDuckGoResults(html string, maxResults int) []SearchResult {
	var results []SearchResult

	// Find all result blocks by locating opening tags and extracting content between them.
	locs := reResultBlock.FindAllStringIndex(html, -1)
	if len(locs) == 0 {
		return results
	}

	for i, loc := range locs {
		if len(results) >= maxResults {
			break
		}

		// Extract the block from this result div to the next one (or end of doc).
		end := len(html)
		if i+1 < len(locs) {
			end = locs[i+1][0]
		}
		block := html[loc[0]:end]

		var result SearchResult

		// Extract title from result__a link
		if m := reResultTitle.FindStringSubmatch(block); len(m) > 1 {
			result.Title = strings.TrimSpace(StripTags(m[1]))
		}

		// Extract URL -- DuckDuckGo wraps URLs through a redirect, the actual URL
		// is in the uddg= parameter of the href on the result__a link.
		if m := reResultTitle.FindString(block); m != "" {
			if href := reResultHref.FindStringSubmatch(m); len(href) > 1 {
				result.URL = extractDDGURL(href[1])
			}
		}

		// Fallback: try result__url element for display URL
		if result.URL == "" {
			if m := reResultURL.FindStringSubmatch(block); len(m) > 1 {
				u := strings.TrimSpace(StripTags(m[1]))
				if !strings.HasPrefix(u, "http") {
					u = "https://" + u
				}
				result.URL = u
			}
		}

		// Extract snippet
		if m := reResultSnippet.FindStringSubmatch(block); len(m) > 1 {
			result.Snippet = strings.TrimSpace(StripTags(m[1]))
		}

		// Only include if we got at least a title or URL
		if result.Title != "" || result.URL != "" {
			results = append(results, result)
		}
	}

	return results
}

// extractDDGURL extracts the real URL from a DuckDuckGo redirect link.
// DuckDuckGo href looks like: //duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com&...
func extractDDGURL(href string) string {
	if m := reDDGRedirect.FindStringSubmatch(href); len(m) > 1 {
		if decoded, err := url.QueryUnescape(m[1]); err == nil {
			return decoded
		}
	}
	// If no redirect wrapper, use href directly
	href = strings.TrimSpace(href)
	if strings.HasPrefix(href, "//") {
		href = "https:" + href
	}
	if strings.HasPrefix(href, "http") {
		return href
	}
	return ""
}

// FormatSearchResults formats results for display/injection into prompts.
func FormatSearchResults(results []SearchResult) string {
	if len(results) == 0 {
		return "No results found."
	}

	var sb strings.Builder
	for i, r := range results {
		fmt.Fprintf(&sb, "%d. %s\n", i+1, r.Title)
		if r.URL != "" {
			fmt.Fprintf(&sb, "   %s\n", r.URL)
		}
		if r.Snippet != "" {
			fmt.Fprintf(&sb, "   %s\n", r.Snippet)
		}
		sb.WriteString("\n")
	}
	return strings.TrimRight(sb.String(), "\n")
}

// BuildSearchURL constructs a DuckDuckGo HTML search URL from a query string.
func BuildSearchURL(query string) string {
	return "https://html.duckduckgo.com/html/?q=" + url.QueryEscape(query)
}

// WikiSummary fetches the summary of a Wikipedia article.
// Uses the Wikipedia REST API (no key needed).
func WikiSummary(topic string) (string, error) {
	if topic == "" {
		return "", fmt.Errorf("wikipedia topic is required")
	}

	// Normalize: replace spaces with underscores for the URL
	normalized := strings.ReplaceAll(strings.TrimSpace(topic), " ", "_")
	apiURL := "https://en.wikipedia.org/api/rest_v1/page/summary/" + url.PathEscape(normalized)

	// SSRF protection
	if err := validateHost("en.wikipedia.org"); err != nil {
		return "", err
	}

	waitForRateLimit("en.wikipedia.org")

	client := &http.Client{Timeout: 10 * time.Second}
	req, err := http.NewRequest("GET", apiURL, nil)
	if err != nil {
		return "", fmt.Errorf("wikipedia: invalid request: %w", err)
	}
	req.Header.Set("User-Agent", "Mozilla/5.0 (compatible; Nous/1.0; +https://github.com/artaeon/nous)")
	req.Header.Set("Accept", "application/json")

	resp, err := client.Do(req)
	if err != nil {
		return "", fmt.Errorf("wikipedia: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode == 404 {
		return "", fmt.Errorf("wikipedia: no article found for %q", topic)
	}
	if resp.StatusCode >= 400 {
		return "", fmt.Errorf("wikipedia: HTTP %d", resp.StatusCode)
	}

	body, err := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if err != nil {
		return "", fmt.Errorf("wikipedia: reading response: %w", err)
	}

	return ParseWikiSummary(body)
}

// ParseWikiSummary extracts the summary text from Wikipedia API JSON response.
func ParseWikiSummary(data []byte) (string, error) {
	var result struct {
		Title       string `json:"title"`
		DisplayName string `json:"displaytitle"`
		Extract     string `json:"extract"`
		Description string `json:"description"`
		ContentURLs struct {
			Desktop struct {
				Page string `json:"page"`
			} `json:"desktop"`
		} `json:"content_urls"`
	}

	if err := json.Unmarshal(data, &result); err != nil {
		return "", fmt.Errorf("wikipedia: invalid JSON: %w", err)
	}

	if result.Extract == "" {
		return "", fmt.Errorf("wikipedia: no summary available")
	}

	var sb strings.Builder
	fmt.Fprintf(&sb, "# %s\n\n", result.Title)
	if result.Description != "" {
		fmt.Fprintf(&sb, "*%s*\n\n", result.Description)
	}
	sb.WriteString(result.Extract)
	if result.ContentURLs.Desktop.Page != "" {
		fmt.Fprintf(&sb, "\n\nSource: %s", result.ContentURLs.Desktop.Page)
	}

	return sb.String(), nil
}

// RegisterSearchTools adds web search tools to the registry.
func RegisterSearchTools(r *Registry) {
	r.Register(Tool{
		Name:        "websearch",
		Description: "Search the web using DuckDuckGo. Args: query (required), max_results (optional, default 5).",
		Execute: func(args map[string]string) (string, error) {
			return toolWebSearch(args)
		},
	})

	r.Register(Tool{
		Name:        "wikipedia",
		Description: "Look up a topic on Wikipedia. Args: topic (required).",
		Execute: func(args map[string]string) (string, error) {
			return toolWikipedia(args)
		},
	})
}

func toolWebSearch(args map[string]string) (string, error) {
	query := args["query"]
	if query == "" {
		return "", fmt.Errorf("websearch requires 'query' argument")
	}

	maxResults := 5
	if v, ok := args["max_results"]; ok {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			maxResults = n
		}
	}

	results, err := WebSearch(query, maxResults)
	if err != nil {
		return "", err
	}

	if len(results) == 0 {
		return "No results found for: " + query, nil
	}

	return FormatSearchResults(results), nil
}

func toolWikipedia(args map[string]string) (string, error) {
	topic := args["topic"]
	if topic == "" {
		return "", fmt.Errorf("wikipedia requires 'topic' argument")
	}

	summary, err := WikiSummary(topic)
	if err != nil {
		return "", err
	}

	return summary, nil
}
