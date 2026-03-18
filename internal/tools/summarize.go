package tools

import (
	"fmt"
	"io"
	"net/http"
	"regexp"
	"strings"
	"time"
)

var (
	reSummarizeMain    = regexp.MustCompile(`(?is)<(?:main|article)[^>]*>(.*?)</(?:main|article)>`)
	reSummarizeSidebar = regexp.MustCompile(`(?is)<(?:aside|sidebar)[^>]*>.*?</(?:aside|sidebar)>`)
)

const maxSummarizeChars = 3000

// ExtractTitle extracts the <title> tag content from HTML.
func ExtractTitle(html string) string {
	if m := reTitle.FindStringSubmatch(html); len(m) > 1 {
		return strings.TrimSpace(StripTags(m[1]))
	}
	return ""
}

// ExtractReadableText strips nav, header, footer, sidebar elements and keeps
// main/article content. Falls back to the full body if no main/article found.
func ExtractReadableText(html string) string {
	// Try to extract main or article content first
	if m := reSummarizeMain.FindStringSubmatch(html); len(m) > 1 {
		text := m[1]
		text = reScript.ReplaceAllString(text, "")
		text = reStyle.ReplaceAllString(text, "")
		text = reSummarizeSidebar.ReplaceAllString(text, "")
		return StripTags(text)
	}

	// Fallback: strip boilerplate elements from full HTML
	text := html
	text = reScript.ReplaceAllString(text, "")
	text = reStyle.ReplaceAllString(text, "")
	text = reNav.ReplaceAllString(text, "")
	text = reHeader.ReplaceAllString(text, "")
	text = reFooter.ReplaceAllString(text, "")
	text = reSummarizeSidebar.ReplaceAllString(text, "")
	text = reComment.ReplaceAllString(text, "")
	text = reSVG.ReplaceAllString(text, "")
	return StripTags(text)
}

// FetchAndExtract fetches a URL, strips HTML, and extracts the main readable
// content. The result is truncated to 3000 characters for LLM context.
func FetchAndExtract(url string) (string, error) {
	if url == "" {
		return "", fmt.Errorf("summarize: URL is required")
	}

	client := &http.Client{Timeout: 15 * time.Second}
	req, err := http.NewRequest("GET", url, nil)
	if err != nil {
		return "", fmt.Errorf("summarize: invalid URL: %w", err)
	}
	req.Header.Set("User-Agent", "Mozilla/5.0 (compatible; Nous/1.0; +https://github.com/artaeon/nous)")
	req.Header.Set("Accept", "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8")

	resp, err := client.Do(req)
	if err != nil {
		return "", fmt.Errorf("summarize: fetch %s: %w", url, err)
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 400 {
		return "", fmt.Errorf("summarize: HTTP %d for %s", resp.StatusCode, url)
	}

	body, err := io.ReadAll(io.LimitReader(resp.Body, 2<<20))
	if err != nil {
		return "", fmt.Errorf("summarize: reading body: %w", err)
	}

	html := string(body)
	title := ExtractTitle(html)
	text := ExtractReadableText(html)

	// Truncate to max chars
	if len(text) > maxSummarizeChars {
		text = text[:maxSummarizeChars] + "..."
	}

	var sb strings.Builder
	if title != "" {
		fmt.Fprintf(&sb, "# %s\n\n", title)
	}
	sb.WriteString(text)

	return sb.String(), nil
}

// RegisterSummarizeTools adds the summarize tool to the registry.
func RegisterSummarizeTools(r *Registry) {
	r.Register(Tool{
		Name:        "summarize",
		Description: "Fetch a URL and extract readable text for summarization. Args: url (required).",
		Execute: func(args map[string]string) (string, error) {
			u := args["url"]
			if u == "" {
				return "", fmt.Errorf("summarize requires 'url' argument")
			}
			return FetchAndExtract(u)
		},
	})
}
