package tools

import (
	"net"
	"strings"
	"testing"
)

// --- HTMLToMarkdown ---

func TestHTMLToMarkdownHeadings(t *testing.T) {
	html := `<h1>Title</h1><h2>Subtitle</h2><h3>Section</h3>`
	result := HTMLToMarkdown(html)

	if !strings.Contains(result, "# Title") {
		t.Errorf("expected '# Title' in result, got %q", result)
	}
	if !strings.Contains(result, "## Subtitle") {
		t.Errorf("expected '## Subtitle' in result, got %q", result)
	}
	if !strings.Contains(result, "### Section") {
		t.Errorf("expected '### Section' in result, got %q", result)
	}
}

func TestHTMLToMarkdownLinks(t *testing.T) {
	html := `<p>Visit <a href="https://example.com">Example</a> for more.</p>`
	result := HTMLToMarkdown(html)

	if !strings.Contains(result, "[Example](https://example.com)") {
		t.Errorf("expected markdown link in result, got %q", result)
	}
}

func TestHTMLToMarkdownStripsScripts(t *testing.T) {
	html := `<p>Hello</p><script>alert('xss')</script><p>World</p>`
	result := HTMLToMarkdown(html)

	if strings.Contains(result, "alert") {
		t.Errorf("expected scripts to be stripped, got %q", result)
	}
	if !strings.Contains(result, "Hello") {
		t.Errorf("expected 'Hello' in result, got %q", result)
	}
	if !strings.Contains(result, "World") {
		t.Errorf("expected 'World' in result, got %q", result)
	}
}

func TestHTMLToMarkdownStripsStyles(t *testing.T) {
	html := `<p>Content</p><style>body { color: red; }</style>`
	result := HTMLToMarkdown(html)

	if strings.Contains(result, "color: red") {
		t.Errorf("expected styles to be stripped, got %q", result)
	}
	if !strings.Contains(result, "Content") {
		t.Errorf("expected 'Content' in result, got %q", result)
	}
}

func TestHTMLToMarkdownStripsNav(t *testing.T) {
	html := `<nav><a href="/">Home</a><a href="/about">About</a></nav><main><p>Main content</p></main>`
	result := HTMLToMarkdown(html)

	if strings.Contains(result, "Home") {
		t.Errorf("expected nav content to be stripped, got %q", result)
	}
	if !strings.Contains(result, "Main content") {
		t.Errorf("expected 'Main content' in result, got %q", result)
	}
}

func TestHTMLToMarkdownBoldItalicCode(t *testing.T) {
	html := `<p>This is <strong>bold</strong> and <em>italic</em> and <code>code</code>.</p>`
	result := HTMLToMarkdown(html)

	if !strings.Contains(result, "**bold**") {
		t.Errorf("expected **bold** in result, got %q", result)
	}
	if !strings.Contains(result, "*italic*") {
		t.Errorf("expected *italic* in result, got %q", result)
	}
	if !strings.Contains(result, "`code`") {
		t.Errorf("expected `code` in result, got %q", result)
	}
}

func TestHTMLToMarkdownPre(t *testing.T) {
	html := `<pre>func main() {
    fmt.Println("hello")
}</pre>`
	result := HTMLToMarkdown(html)

	if !strings.Contains(result, "```") {
		t.Errorf("expected code fence in result, got %q", result)
	}
	if !strings.Contains(result, "func main()") {
		t.Errorf("expected code content in result, got %q", result)
	}
}

func TestHTMLToMarkdownLists(t *testing.T) {
	html := `<ul><li>First</li><li>Second</li><li>Third</li></ul>`
	result := HTMLToMarkdown(html)

	if !strings.Contains(result, "- First") {
		t.Errorf("expected '- First' in result, got %q", result)
	}
	if !strings.Contains(result, "- Second") {
		t.Errorf("expected '- Second' in result, got %q", result)
	}
}

func TestHTMLToMarkdownOrderedLists(t *testing.T) {
	html := `<ol><li>Alpha</li><li>Beta</li></ol>`
	result := HTMLToMarkdown(html)

	if !strings.Contains(result, "1. Alpha") {
		t.Errorf("expected '1. Alpha' in result, got %q", result)
	}
	if !strings.Contains(result, "2. Beta") {
		t.Errorf("expected '2. Beta' in result, got %q", result)
	}
}

func TestHTMLToMarkdownEntities(t *testing.T) {
	html := `<p>Tom &amp; Jerry &lt;3 &quot;cartoons&quot;</p>`
	result := HTMLToMarkdown(html)

	if !strings.Contains(result, "Tom & Jerry") {
		t.Errorf("expected decoded &amp;, got %q", result)
	}
	if !strings.Contains(result, `<3`) {
		t.Errorf("expected decoded &lt;, got %q", result)
	}
	if !strings.Contains(result, `"cartoons"`) {
		t.Errorf("expected decoded &quot;, got %q", result)
	}
}

// --- StripTags ---

func TestStripTags(t *testing.T) {
	html := `<p>Hello <b>World</b></p><script>evil()</script>`
	result := StripTags(html)

	if strings.Contains(result, "<") {
		t.Errorf("expected no HTML tags in result, got %q", result)
	}
	if !strings.Contains(result, "Hello") {
		t.Errorf("expected 'Hello' in result, got %q", result)
	}
	if !strings.Contains(result, "World") {
		t.Errorf("expected 'World' in result, got %q", result)
	}
	if strings.Contains(result, "evil") {
		t.Errorf("expected script content to be stripped, got %q", result)
	}
}

// --- ExtractBySelector ---

func TestExtractBySelectorID(t *testing.T) {
	html := `<div id="header">Header</div><div id="main">Main Content</div><div id="footer">Footer</div>`
	result := ExtractBySelector(html, "#main")

	if !strings.Contains(result, "Main Content") {
		t.Errorf("expected 'Main Content' for #main selector, got %q", result)
	}
	if strings.Contains(result, "Header") {
		t.Errorf("expected no 'Header' for #main selector, got %q", result)
	}
}

func TestExtractBySelectorClass(t *testing.T) {
	html := `<div class="sidebar">Side</div><div class="content main-area">Article text here</div>`
	result := ExtractBySelector(html, ".content")

	if !strings.Contains(result, "Article text here") {
		t.Errorf("expected 'Article text here' for .content selector, got %q", result)
	}
}

func TestExtractBySelectorTag(t *testing.T) {
	html := `<header>Nav stuff</header><article><p>The real content</p></article><footer>Foot</footer>`
	result := ExtractBySelector(html, "article")

	if !strings.Contains(result, "The real content") {
		t.Errorf("expected 'The real content' for article selector, got %q", result)
	}
}

func TestExtractBySelectorTagWithClass(t *testing.T) {
	html := `<div class="other">Not this</div><div class="target">This one</div>`
	result := ExtractBySelector(html, "div.target")

	if !strings.Contains(result, "This one") {
		t.Errorf("expected 'This one' for div.target selector, got %q", result)
	}
}

func TestExtractBySelectorNoMatch(t *testing.T) {
	html := `<div id="content">Hello</div>`
	result := ExtractBySelector(html, "#nonexistent")

	if result != "" {
		t.Errorf("expected empty result for non-matching selector, got %q", result)
	}
}

// --- ExtractLinks ---

func TestExtractLinks(t *testing.T) {
	html := `
		<a href="/about">About Us</a>
		<a href="https://external.com/page">External</a>
		<a href="/contact">Contact</a>
		<a href="https://external.com/page">Duplicate External</a>
		<a href="javascript:void(0)">Skip</a>
		<a href="#">Skip hash</a>
	`
	links := ExtractLinks(html, "https://example.com")

	if len(links) != 3 {
		t.Fatalf("expected 3 links (deduped, no js/hash), got %d", len(links))
	}

	// Check internal vs external
	internalCount := 0
	externalCount := 0
	for _, l := range links {
		if l.Internal {
			internalCount++
		} else {
			externalCount++
		}
	}

	if internalCount != 2 {
		t.Errorf("expected 2 internal links, got %d", internalCount)
	}
	if externalCount != 1 {
		t.Errorf("expected 1 external link, got %d", externalCount)
	}
}

func TestExtractLinksResolvesRelative(t *testing.T) {
	html := `<a href="/path/page">Link</a>`
	links := ExtractLinks(html, "https://example.com/base/")

	if len(links) != 1 {
		t.Fatalf("expected 1 link, got %d", len(links))
	}
	if links[0].URL != "https://example.com/path/page" {
		t.Errorf("expected resolved URL, got %q", links[0].URL)
	}
}

func TestExtractLinksDeduplicates(t *testing.T) {
	html := `
		<a href="https://example.com/page">First</a>
		<a href="https://example.com/page">Second</a>
	`
	links := ExtractLinks(html, "https://example.com")

	if len(links) != 1 {
		t.Errorf("expected 1 deduplicated link, got %d", len(links))
	}
}

// --- ExtractTables ---

func TestExtractTables(t *testing.T) {
	html := `
		<table>
			<tr><th>Name</th><th>Age</th></tr>
			<tr><td>Alice</td><td>30</td></tr>
			<tr><td>Bob</td><td>25</td></tr>
		</table>
	`
	tables := ExtractTables(html)

	if len(tables) != 1 {
		t.Fatalf("expected 1 table, got %d", len(tables))
	}

	table := tables[0]
	if len(table) != 3 {
		t.Fatalf("expected 3 rows, got %d", len(table))
	}

	if table[0][0] != "Name" || table[0][1] != "Age" {
		t.Errorf("expected header [Name, Age], got %v", table[0])
	}
	if table[1][0] != "Alice" || table[1][1] != "30" {
		t.Errorf("expected row [Alice, 30], got %v", table[1])
	}
	if table[2][0] != "Bob" || table[2][1] != "25" {
		t.Errorf("expected row [Bob, 25], got %v", table[2])
	}
}

func TestExtractTablesMultiple(t *testing.T) {
	html := `
		<table><tr><td>A</td></tr></table>
		<table><tr><td>B</td></tr></table>
	`
	tables := ExtractTables(html)

	if len(tables) != 2 {
		t.Errorf("expected 2 tables, got %d", len(tables))
	}
}

func TestExtractTablesNone(t *testing.T) {
	html := `<p>No tables here</p>`
	tables := ExtractTables(html)

	if len(tables) != 0 {
		t.Errorf("expected 0 tables, got %d", len(tables))
	}
}

// --- ExtractMeta ---

func TestExtractMeta(t *testing.T) {
	html := `
		<html><head>
			<title>Test Page</title>
			<meta name="description" content="A test page">
			<meta name="keywords" content="test, page">
			<meta property="og:title" content="OG Title">
		</head><body></body></html>
	`
	meta := ExtractMeta(html)

	if meta["title"] != "Test Page" {
		t.Errorf("expected title 'Test Page', got %q", meta["title"])
	}
	if meta["description"] != "A test page" {
		t.Errorf("expected description 'A test page', got %q", meta["description"])
	}
	if meta["keywords"] != "test, page" {
		t.Errorf("expected keywords, got %q", meta["keywords"])
	}
	if meta["og:title"] != "OG Title" {
		t.Errorf("expected og:title 'OG Title', got %q", meta["og:title"])
	}
}

// --- ExtractOpenGraph ---

func TestExtractOpenGraph(t *testing.T) {
	html := `
		<meta property="og:title" content="My Page">
		<meta property="og:description" content="Page description">
		<meta property="og:image" content="https://example.com/img.png">
		<meta name="author" content="John">
	`
	og := ExtractOpenGraph(html)

	if len(og) != 3 {
		t.Errorf("expected 3 OG tags, got %d", len(og))
	}
	if og["og:title"] != "My Page" {
		t.Errorf("expected og:title 'My Page', got %q", og["og:title"])
	}
	if og["og:image"] != "https://example.com/img.png" {
		t.Errorf("expected og:image URL, got %q", og["og:image"])
	}
}

// --- ExtractLists ---

func TestExtractLists(t *testing.T) {
	html := `
		<ul>
			<li>Apple</li>
			<li>Banana</li>
		</ul>
		<ol>
			<li>First</li>
			<li>Second</li>
		</ol>
	`
	lists := ExtractLists(html)

	if len(lists) != 2 {
		t.Fatalf("expected 2 lists, got %d", len(lists))
	}

	if lists[0][0] != "Apple" || lists[0][1] != "Banana" {
		t.Errorf("expected [Apple, Banana], got %v", lists[0])
	}
	if lists[1][0] != "First" || lists[1][1] != "Second" {
		t.Errorf("expected [First, Second], got %v", lists[1])
	}
}

// --- decodeEntities ---

func TestDecodeEntities(t *testing.T) {
	tests := []struct {
		input    string
		expected string
	}{
		{"&amp;", "&"},
		{"&lt;", "<"},
		{"&gt;", ">"},
		{"&quot;", "\""},
		{"&#39;", "'"},
		{"&nbsp;", " "},
		{"&#65;", "A"},
		{"&#x41;", "A"},
		{"no entities here", "no entities here"},
	}

	for _, tt := range tests {
		got := decodeEntities(tt.input)
		if got != tt.expected {
			t.Errorf("decodeEntities(%q) = %q, want %q", tt.input, got, tt.expected)
		}
	}
}

// --- Tool registration ---

func TestBrowserToolsRegistered(t *testing.T) {
	r := NewRegistry()
	RegisterBrowserTools(r)

	expectedTools := []string{"browse", "screenshot", "links", "scrape"}
	for _, name := range expectedTools {
		if _, err := r.Get(name); err != nil {
			t.Errorf("expected tool %q to be registered: %v", name, err)
		}
	}
}

// --- Tool argument validation ---

func TestToolBrowseMissingURL(t *testing.T) {
	r := NewRegistry()
	RegisterBrowserTools(r)

	tool, _ := r.Get("browse")
	_, err := tool.Execute(map[string]string{})
	if err == nil {
		t.Fatal("expected error for missing url")
	}
	if !strings.Contains(err.Error(), "url") {
		t.Errorf("expected error about url, got %q", err.Error())
	}
}

func TestToolScreenshotMissingURL(t *testing.T) {
	r := NewRegistry()
	RegisterBrowserTools(r)

	tool, _ := r.Get("screenshot")
	_, err := tool.Execute(map[string]string{})
	if err == nil {
		t.Fatal("expected error for missing url")
	}
}

func TestToolLinksMissingURL(t *testing.T) {
	r := NewRegistry()
	RegisterBrowserTools(r)

	tool, _ := r.Get("links")
	_, err := tool.Execute(map[string]string{})
	if err == nil {
		t.Fatal("expected error for missing url")
	}
}

func TestToolScrapeMissingArgs(t *testing.T) {
	r := NewRegistry()
	RegisterBrowserTools(r)

	tool, _ := r.Get("scrape")

	_, err := tool.Execute(map[string]string{})
	if err == nil {
		t.Fatal("expected error for missing url")
	}

	_, err = tool.Execute(map[string]string{"url": "http://127.0.0.1:1"})
	if err == nil {
		t.Fatal("expected error for missing type")
	}
}

// --- Complex HTML conversion ---

func TestHTMLToMarkdownFullPage(t *testing.T) {
	html := `
		<!DOCTYPE html>
		<html>
		<head>
			<title>Test Page</title>
			<style>body { font-size: 14px; }</style>
			<script>var x = 1;</script>
		</head>
		<body>
			<nav><a href="/">Home</a></nav>
			<h1>Welcome</h1>
			<p>This is a <strong>test</strong> page with a <a href="https://example.com">link</a>.</p>
			<ul>
				<li>Item one</li>
				<li>Item two</li>
			</ul>
			<footer>Copyright 2024</footer>
		</body>
		</html>
	`
	result := HTMLToMarkdown(html)

	// Should have the heading
	if !strings.Contains(result, "# Welcome") {
		t.Errorf("expected '# Welcome', got %q", result)
	}

	// Should have bold text
	if !strings.Contains(result, "**test**") {
		t.Errorf("expected '**test**', got %q", result)
	}

	// Should have the link
	if !strings.Contains(result, "[link](https://example.com)") {
		t.Errorf("expected markdown link, got %q", result)
	}

	// Should have list items
	if !strings.Contains(result, "- Item one") {
		t.Errorf("expected list items, got %q", result)
	}

	// Should NOT have script content
	if strings.Contains(result, "var x") {
		t.Errorf("expected script to be stripped, got %q", result)
	}

	// Should NOT have style content
	if strings.Contains(result, "font-size") {
		t.Errorf("expected style to be stripped, got %q", result)
	}

	// Should NOT have nav content
	if strings.Contains(result, "Home") {
		t.Errorf("expected nav to be stripped, got %q", result)
	}

	// Should NOT have footer content
	if strings.Contains(result, "Copyright") {
		t.Errorf("expected footer to be stripped, got %q", result)
	}
}

func TestHTMLToMarkdownBlockquote(t *testing.T) {
	html := `<blockquote>A wise quote</blockquote>`
	result := HTMLToMarkdown(html)

	if !strings.Contains(result, "> A wise quote") {
		t.Errorf("expected blockquote formatting, got %q", result)
	}
}

// --- Nested selector extraction ---

func TestExtractBySelectorNested(t *testing.T) {
	html := `<div id="outer"><div id="inner"><p>Deep content</p></div></div><div id="other">Other</div>`
	result := ExtractBySelector(html, "#outer")

	if !strings.Contains(result, "Deep content") {
		t.Errorf("expected nested content for #outer, got %q", result)
	}
	if !strings.Contains(result, "inner") {
		t.Errorf("expected nested div in #outer, got %q", result)
	}
}

// --- SSRF protection ---

func TestIsPrivateIPBlocked(t *testing.T) {
	privateIPs := []string{
		"127.0.0.1",
		"127.0.0.2",
		"10.0.0.1",
		"10.255.255.255",
		"172.16.0.1",
		"172.31.255.255",
		"192.168.0.1",
		"192.168.1.100",
		"169.254.1.1",
		"::1",
	}

	for _, ipStr := range privateIPs {
		ip := net.ParseIP(ipStr)
		if ip == nil {
			t.Errorf("failed to parse IP %q", ipStr)
			continue
		}
		if !isPrivateIP(ip) {
			t.Errorf("expected %s to be classified as private IP", ipStr)
		}
	}
}

func TestIsPrivateIPAllowsPublic(t *testing.T) {
	publicIPs := []string{
		"8.8.8.8",
		"1.1.1.1",
		"203.0.113.1",
		"198.51.100.1",
	}

	for _, ipStr := range publicIPs {
		ip := net.ParseIP(ipStr)
		if ip == nil {
			t.Errorf("failed to parse IP %q", ipStr)
			continue
		}
		if isPrivateIP(ip) {
			t.Errorf("expected %s to NOT be classified as private IP", ipStr)
		}
	}
}

// --- HTML to markdown edge cases ---

func TestHTMLToMarkdownEmptyInput(t *testing.T) {
	result := HTMLToMarkdown("")
	if result != "" {
		t.Errorf("expected empty string for empty input, got %q", result)
	}
}

func TestHTMLToMarkdownOnlyScripts(t *testing.T) {
	html := `<script>alert('evil')</script><script>console.log('more')</script>`
	result := HTMLToMarkdown(html)
	if strings.Contains(result, "alert") || strings.Contains(result, "console") {
		t.Errorf("expected all scripts stripped, got %q", result)
	}
}

func TestHTMLToMarkdownNestedFormatting(t *testing.T) {
	html := `<p>This has <strong><em>bold italic</em></strong> text.</p>`
	result := HTMLToMarkdown(html)
	if !strings.Contains(result, "bold italic") {
		t.Errorf("expected 'bold italic' in result, got %q", result)
	}
}

func TestHTMLToMarkdownMultipleParagraphs(t *testing.T) {
	html := `<p>First paragraph.</p><p>Second paragraph.</p><p>Third paragraph.</p>`
	result := HTMLToMarkdown(html)
	if !strings.Contains(result, "First paragraph") {
		t.Errorf("expected first paragraph, got %q", result)
	}
	if !strings.Contains(result, "Third paragraph") {
		t.Errorf("expected third paragraph, got %q", result)
	}
}

func TestHTMLToMarkdownImages(t *testing.T) {
	html := `<p>Look: <img src="photo.jpg" alt="A photo"></p>`
	result := HTMLToMarkdown(html)
	// Image tags should be stripped
	if strings.Contains(result, "<img") {
		t.Errorf("expected img tag stripped, got %q", result)
	}
}

// --- Link extraction edge cases ---

func TestExtractLinksRelativeURLs(t *testing.T) {
	html := `
		<a href="/docs/api">API Docs</a>
		<a href="../parent/page">Parent</a>
		<a href="sibling.html">Sibling</a>
	`
	links := ExtractLinks(html, "https://example.com/path/index.html")

	for _, l := range links {
		if !strings.HasPrefix(l.URL, "https://example.com") {
			t.Errorf("expected resolved URL, got %q", l.URL)
		}
	}
}

func TestExtractLinksSkipsJavaScript(t *testing.T) {
	html := `
		<a href="javascript:void(0)">JS Link</a>
		<a href="mailto:test@example.com">Email</a>
		<a href="tel:+1234567890">Phone</a>
		<a href="data:text/plain,foo">Data</a>
		<a href="https://example.com/real">Real</a>
	`
	links := ExtractLinks(html, "https://example.com")

	if len(links) != 1 {
		t.Fatalf("expected 1 link (only real one), got %d", len(links))
	}
	if links[0].URL != "https://example.com/real" {
		t.Errorf("expected real link, got %q", links[0].URL)
	}
}

func TestExtractLinksEmpty(t *testing.T) {
	html := `<p>No links here</p>`
	links := ExtractLinks(html, "https://example.com")

	if len(links) != 0 {
		t.Errorf("expected 0 links, got %d", len(links))
	}
}

// --- Scrape with malformed HTML ---

func TestHTMLToMarkdownMalformedHTML(t *testing.T) {
	html := `<p>Unclosed paragraph
	<div>Nested without close
	<strong>Bold without end
	<a href="url">Link`
	result := HTMLToMarkdown(html)
	// Should not panic, should extract some text
	if !strings.Contains(result, "Unclosed paragraph") {
		t.Errorf("expected some text from malformed HTML, got %q", result)
	}
}

func TestExtractTablesMalformedHTML(t *testing.T) {
	html := `<table><tr><td>Cell 1<td>Cell 2</tr></table>`
	tables := ExtractTables(html)
	// Should handle gracefully without panicking
	_ = tables
}

func TestExtractMetaEmpty(t *testing.T) {
	html := `<html><body>No meta</body></html>`
	meta := ExtractMeta(html)
	if len(meta) != 0 {
		t.Errorf("expected 0 meta tags, got %d", len(meta))
	}
}

// --- truncateUTF8 ---

func TestTruncateUTF8Short(t *testing.T) {
	s := "hello"
	result := truncateUTF8(s, 100)
	if result != "hello" {
		t.Errorf("expected 'hello', got %q", result)
	}
}

func TestTruncateUTF8ExactBoundary(t *testing.T) {
	s := "hello"
	result := truncateUTF8(s, 5)
	if result != "hello" {
		t.Errorf("expected 'hello', got %q", result)
	}
}

func TestTruncateUTF8MultibyteChar(t *testing.T) {
	// UTF-8 encoded string with multi-byte characters
	s := "hello\xc3\xa9world" // e with accent is 2 bytes
	result := truncateUTF8(s, 6)
	// Should not split the multi-byte character
	if len(result) > 6 {
		t.Errorf("expected result <= 6 bytes, got %d bytes", len(result))
	}
}

// --- StripTags edge cases ---

func TestStripTagsEmpty(t *testing.T) {
	result := StripTags("")
	if result != "" {
		t.Errorf("expected empty string, got %q", result)
	}
}

func TestStripTagsPreservesPlainText(t *testing.T) {
	text := "Just plain text with no HTML"
	result := StripTags(text)
	if result != text {
		t.Errorf("expected plain text preserved, got %q", result)
	}
}

func TestStripTagsSVG(t *testing.T) {
	html := `<p>Before</p><svg><circle r="10"/></svg><p>After</p>`
	result := StripTags(html)
	if strings.Contains(result, "circle") {
		t.Errorf("expected SVG stripped, got %q", result)
	}
	if !strings.Contains(result, "Before") || !strings.Contains(result, "After") {
		t.Errorf("expected surrounding text preserved, got %q", result)
	}
}
