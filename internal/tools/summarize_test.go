package tools

import (
	"strings"
	"testing"
)

func TestExtractTitle(t *testing.T) {
	tests := []struct {
		name string
		html string
		want string
	}{
		{"basic title", "<html><head><title>Hello World</title></head></html>", "Hello World"},
		{"no title", "<html><head></head></html>", ""},
		{"title with whitespace", "<title>  Spaced Title  </title>", "Spaced Title"},
		{"title with tags", "<title><b>Bold</b> Title</title>", "Bold Title"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := ExtractTitle(tt.html)
			if got != tt.want {
				t.Errorf("ExtractTitle() = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestExtractReadableText(t *testing.T) {
	tests := []struct {
		name     string
		html     string
		contains string
		excludes string
	}{
		{
			"extracts article content",
			`<html><nav>Menu</nav><article>Main article content here.</article><footer>Footer</footer></html>`,
			"Main article content here.",
			"Menu",
		},
		{
			"extracts main content",
			`<html><header>Header</header><main>Important content.</main><aside>Sidebar</aside></html>`,
			"Important content.",
			"",
		},
		{
			"strips scripts and styles",
			`<html><script>var x = 1;</script><style>.foo{color:red}</style><p>Real content.</p></html>`,
			"Real content.",
			"var x",
		},
		{
			"fallback strips nav/header/footer",
			`<html><nav>Nav stuff</nav><div>Body text here</div><footer>Foot</footer></html>`,
			"Body text here",
			"Nav stuff",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := ExtractReadableText(tt.html)
			if !strings.Contains(got, tt.contains) {
				t.Errorf("ExtractReadableText() should contain %q, got %q", tt.contains, got)
			}
			if tt.excludes != "" && strings.Contains(got, tt.excludes) {
				t.Errorf("ExtractReadableText() should not contain %q, got %q", tt.excludes, got)
			}
		})
	}
}

func TestExtractReadableText_EntityDecoding(t *testing.T) {
	html := `<article>Tom &amp; Jerry &lt;forever&gt;</article>`
	got := ExtractReadableText(html)
	if !strings.Contains(got, "Tom & Jerry") {
		t.Errorf("expected decoded entities, got %q", got)
	}
}

func TestFetchAndExtract_Truncation(t *testing.T) {
	// Build HTML with content longer than maxSummarizeChars
	longText := strings.Repeat("word ", 1000) // 5000 chars
	html := "<article>" + longText + "</article>"

	text := ExtractReadableText(html)
	// Direct truncation test: simulate what FetchAndExtract does
	if len(text) > maxSummarizeChars {
		text = text[:maxSummarizeChars] + "..."
	}

	if len(text) > maxSummarizeChars+10 {
		t.Errorf("text should be truncated to ~%d chars, got %d", maxSummarizeChars, len(text))
	}
	if !strings.HasSuffix(text, "...") {
		t.Error("truncated text should end with '...'")
	}
}

func TestRegisterSummarizeTools(t *testing.T) {
	r := NewRegistry()
	RegisterSummarizeTools(r)

	tool, err := r.Get("summarize")
	if err != nil {
		t.Fatalf("summarize tool not registered: %v", err)
	}

	// Missing URL should error
	_, err = tool.Execute(map[string]string{})
	if err == nil {
		t.Error("expected error for missing url")
	}
}
