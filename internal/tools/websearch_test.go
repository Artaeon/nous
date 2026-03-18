package tools

import (
	"strings"
	"testing"
)

// Sample DuckDuckGo HTML for testing (simplified but structurally accurate).
const sampleDDGHTML = `
<html>
<body>
<div id="links" class="results">

<div class="result results_links results_links_deep web-result">
  <div class="links_main links_deep result__body">
    <h2 class="result__title">
      <a rel="nofollow" class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fen.wikipedia.org%2Fwiki%2FGo_%28programming_language%29&amp;rut=abc123">Go (programming language) - Wikipedia</a>
    </h2>
    <a class="result__url" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fen.wikipedia.org%2Fwiki%2FGo_%28programming_language%29&amp;rut=abc123">en.wikipedia.org/wiki/Go_(programming_language)</a>
    <a class="result__snippet" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fen.wikipedia.org%2Fwiki%2FGo_%28programming_language%29&amp;rut=abc123">Go is a statically typed, compiled high-level programming language designed at Google.</a>
  </div>
</div>

<div class="result results_links results_links_deep web-result">
  <div class="links_main links_deep result__body">
    <h2 class="result__title">
      <a rel="nofollow" class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fgo.dev%2F&amp;rut=def456">The Go Programming Language</a>
    </h2>
    <a class="result__url" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fgo.dev%2F&amp;rut=def456">go.dev</a>
    <a class="result__snippet" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fgo.dev%2F&amp;rut=def456">Build simple, secure, scalable systems with Go. An open-source programming language supported by Google.</a>
  </div>
</div>

<div class="result results_links results_links_deep web-result">
  <div class="links_main links_deep result__body">
    <h2 class="result__title">
      <a rel="nofollow" class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fgo.dev%2Fdoc%2F&amp;rut=ghi789">Documentation - The Go Programming Language</a>
    </h2>
    <a class="result__url" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fgo.dev%2Fdoc%2F&amp;rut=ghi789">go.dev/doc/</a>
    <a class="result__snippet" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fgo.dev%2Fdoc%2F&amp;rut=ghi789">The Go programming language is an open source project to make programmers more productive.</a>
  </div>
</div>

</div>
</body>
</html>
`

func TestWebSearchParseResults(t *testing.T) {
	results := ParseDuckDuckGoResults(sampleDDGHTML, 10)

	if len(results) != 3 {
		t.Fatalf("expected 3 results, got %d", len(results))
	}

	// Check first result
	r := results[0]
	if r.Title != "Go (programming language) - Wikipedia" {
		t.Errorf("result 0 title = %q, want %q", r.Title, "Go (programming language) - Wikipedia")
	}
	if r.URL != "https://en.wikipedia.org/wiki/Go_(programming_language)" {
		t.Errorf("result 0 URL = %q, want %q", r.URL, "https://en.wikipedia.org/wiki/Go_(programming_language)")
	}
	if !strings.Contains(r.Snippet, "statically typed") {
		t.Errorf("result 0 snippet should contain 'statically typed', got %q", r.Snippet)
	}

	// Check second result
	r = results[1]
	if r.Title != "The Go Programming Language" {
		t.Errorf("result 1 title = %q, want %q", r.Title, "The Go Programming Language")
	}
	if r.URL != "https://go.dev/" {
		t.Errorf("result 1 URL = %q, want %q", r.URL, "https://go.dev/")
	}

	// Check third result
	r = results[2]
	if r.URL != "https://go.dev/doc/" {
		t.Errorf("result 2 URL = %q, want %q", r.URL, "https://go.dev/doc/")
	}
}

func TestWebSearchParseMaxResults(t *testing.T) {
	results := ParseDuckDuckGoResults(sampleDDGHTML, 1)
	if len(results) != 1 {
		t.Fatalf("expected 1 result (max_results=1), got %d", len(results))
	}
	if results[0].Title != "Go (programming language) - Wikipedia" {
		t.Errorf("expected first result, got %q", results[0].Title)
	}
}

func TestWebSearchParseEmpty(t *testing.T) {
	results := ParseDuckDuckGoResults("<html><body>No results</body></html>", 5)
	if len(results) != 0 {
		t.Fatalf("expected 0 results from empty HTML, got %d", len(results))
	}
}

func TestWebSearchBuildURL(t *testing.T) {
	got := BuildSearchURL("golang tutorial")
	want := "https://html.duckduckgo.com/html/?q=golang+tutorial"
	if got != want {
		t.Errorf("BuildSearchURL = %q, want %q", got, want)
	}
}

func TestWebSearchBuildURLSpecialChars(t *testing.T) {
	got := BuildSearchURL("what is 2+2?")
	if !strings.Contains(got, "html.duckduckgo.com") {
		t.Errorf("URL should contain duckduckgo domain, got %q", got)
	}
	if !strings.Contains(got, "q=") {
		t.Errorf("URL should contain query parameter, got %q", got)
	}
}

func TestWebSearchExtractDDGURL(t *testing.T) {
	tests := []struct {
		name string
		href string
		want string
	}{
		{
			name: "standard redirect",
			href: "//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fpage&rut=abc",
			want: "https://example.com/page",
		},
		{
			name: "direct https URL",
			href: "https://example.com/direct",
			want: "https://example.com/direct",
		},
		{
			name: "protocol-relative",
			href: "//example.com/page",
			want: "https://example.com/page",
		},
		{
			name: "empty",
			href: "",
			want: "",
		},
		{
			name: "relative path",
			href: "/some/path",
			want: "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := extractDDGURL(tt.href)
			if got != tt.want {
				t.Errorf("extractDDGURL(%q) = %q, want %q", tt.href, got, tt.want)
			}
		})
	}
}

func TestWebSearchFormatResults(t *testing.T) {
	results := []SearchResult{
		{Title: "Example", URL: "https://example.com", Snippet: "An example site."},
		{Title: "Test", URL: "https://test.com", Snippet: "A test site."},
	}

	formatted := FormatSearchResults(results)
	if !strings.Contains(formatted, "1. Example") {
		t.Error("formatted results should contain numbered title")
	}
	if !strings.Contains(formatted, "https://example.com") {
		t.Error("formatted results should contain URL")
	}
	if !strings.Contains(formatted, "An example site.") {
		t.Error("formatted results should contain snippet")
	}
	if !strings.Contains(formatted, "2. Test") {
		t.Error("formatted results should contain second result")
	}
}

func TestWebSearchFormatEmpty(t *testing.T) {
	formatted := FormatSearchResults(nil)
	if formatted != "No results found." {
		t.Errorf("FormatSearchResults(nil) = %q, want %q", formatted, "No results found.")
	}
}

// Sample Wikipedia API JSON for testing.
const sampleWikiJSON = `{
  "title": "Photosynthesis",
  "displaytitle": "Photosynthesis",
  "description": "Biological process to convert light into chemical energy",
  "extract": "Photosynthesis is a process used by plants and other organisms to convert light energy, normally from the Sun, into chemical energy that can be later released to fuel the organism's activities.",
  "content_urls": {
    "desktop": {
      "page": "https://en.wikipedia.org/wiki/Photosynthesis"
    }
  }
}`

func TestWikiParseSummary(t *testing.T) {
	result, err := ParseWikiSummary([]byte(sampleWikiJSON))
	if err != nil {
		t.Fatalf("ParseWikiSummary error: %v", err)
	}

	if !strings.Contains(result, "# Photosynthesis") {
		t.Error("should contain title heading")
	}
	if !strings.Contains(result, "Biological process") {
		t.Error("should contain description")
	}
	if !strings.Contains(result, "convert light energy") {
		t.Error("should contain extract text")
	}
	if !strings.Contains(result, "https://en.wikipedia.org/wiki/Photosynthesis") {
		t.Error("should contain source URL")
	}
}

func TestWikiParseSummaryEmpty(t *testing.T) {
	_, err := ParseWikiSummary([]byte(`{"title":"Test","extract":""}`))
	if err == nil {
		t.Error("expected error for empty extract")
	}
}

func TestWikiParseSummaryInvalidJSON(t *testing.T) {
	_, err := ParseWikiSummary([]byte(`not json`))
	if err == nil {
		t.Error("expected error for invalid JSON")
	}
}

func TestWebSearchToolRegistration(t *testing.T) {
	r := NewRegistry()
	RegisterSearchTools(r)

	tools := r.List()
	found := map[string]bool{}
	for _, tool := range tools {
		found[tool.Name] = true
	}

	if !found["websearch"] {
		t.Error("websearch tool not registered")
	}
	if !found["wikipedia"] {
		t.Error("wikipedia tool not registered")
	}
}

func TestWebSearchToolMissingQuery(t *testing.T) {
	r := NewRegistry()
	RegisterSearchTools(r)

	tool, err := r.Get("websearch")
	if err != nil {
		t.Fatal("websearch tool not found")
	}

	_, err = tool.Execute(map[string]string{})
	if err == nil {
		t.Error("expected error when query is missing")
	}
}

func TestWikipediaToolMissingTopic(t *testing.T) {
	r := NewRegistry()
	RegisterSearchTools(r)

	tool, err := r.Get("wikipedia")
	if err != nil {
		t.Fatal("wikipedia tool not found")
	}

	_, err = tool.Execute(map[string]string{})
	if err == nil {
		t.Error("expected error when topic is missing")
	}
}
