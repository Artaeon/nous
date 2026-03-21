package cognitive

import (
	"bytes"
	"encoding/xml"
	"strings"
	"testing"
)

func TestStripWikitext(t *testing.T) {
	tests := []struct {
		name  string
		input string
		want  string // substring that must appear
		notWant string // substring that must NOT appear
	}{
		{
			name:  "plain link",
			input: "[[Albert Einstein]] was a physicist.",
			want:  "Albert Einstein was a physicist.",
		},
		{
			name:  "piped link",
			input: "He lived in [[United States|the US]].",
			want:  "the US",
			notWant: "United States|",
		},
		{
			name:  "template removal",
			input: "Born in 1879{{cite web|url=http://example.com}} in Germany.",
			want:  "Born in 1879 in Germany.",
			notWant: "cite web",
		},
		{
			name:  "nested templates",
			input: "Text{{outer|{{inner|deep}}}} after.",
			want:  "Text after.",
			notWant: "outer",
		},
		{
			name:    "ref paired",
			input:   "Physics is fundamental<ref>Smith 2020</ref> to science.",
			want:    "Physics is fundamental to science.",
			notWant: "Smith",
		},
		{
			name:    "ref self-closing",
			input:   "Water is essential<ref name=\"source1\" /> for life.",
			want:    "Water is essential for life.",
			notWant: "source1",
		},
		{
			name:    "table removal",
			input:   "Before table.\n{| class=\"wikitable\"\n|-\n| cell1 || cell2\n|}\nAfter table.",
			want:    "After table.",
			notWant: "cell1",
		},
		{
			name:  "section headers",
			input: "Intro text.\n== Early life ==\nHe was born.",
			want:  "Early life",
		},
		{
			name:  "bold markup",
			input: "'''Albert Einstein''' was born.",
			want:  "Albert Einstein was born.",
			notWant: "'''",
		},
		{
			name:  "italic markup",
			input: "The ''theory of relativity'' changed physics.",
			want:  "theory of relativity",
			notWant: "''",
		},
		{
			name:  "bullet list",
			input: "* First item\n* Second item",
			want:  "First item",
			notWant: "* ",
		},
		{
			name:    "comment removal",
			input:   "Visible text<!-- hidden comment --> more text.",
			want:    "Visible text more text.",
			notWant: "hidden",
		},
		{
			name:    "category removal",
			input:   "Article text.\n[[Category:Physics]]",
			want:    "Article text.",
			notWant: "Category",
		},
		{
			name:  "external link with text",
			input: "See [http://example.com Example Site] for details.",
			want:  "Example Site",
			notWant: "http://",
		},
		{
			name:  "nowiki tags",
			input: "Use <nowiki>[[not a link]]</nowiki> syntax.",
			want:  "[[not a link]]",
			notWant: "<nowiki>",
		},
		{
			name:    "file link removal",
			input:   "Text before [[File:Example.png|thumb|Caption]] text after.",
			want:    "text after",
			notWant: "Example.png",
		},
		{
			name:    "magic words",
			input:   "__NOTOC__ Article text here.",
			want:    "Article text here.",
			notWant: "__NOTOC__",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := StripWikitext(tt.input)
			if tt.want != "" && !strings.Contains(got, tt.want) {
				t.Errorf("StripWikitext() = %q, want it to contain %q", got, tt.want)
			}
			if tt.notWant != "" && strings.Contains(got, tt.notWant) {
				t.Errorf("StripWikitext() = %q, should NOT contain %q", got, tt.notWant)
			}
		})
	}
}

func TestArticleToFacts(t *testing.T) {
	text := `Albert Einstein was a theoretical physicist. He was born in Ulm, Germany. ` +
		`Einstein is a theoretical physicist. ` +
		`General relativity was created by Albert Einstein.`

	facts := ArticleToFacts("Albert Einstein", text)

	if len(facts) == 0 {
		t.Fatal("ArticleToFacts returned no facts")
	}

	// Should have a described_as fact from first sentence
	hasDescribed := false
	hasIsA := false
	hasCreatedBy := false

	for _, f := range facts {
		if f.Relation == "described_as" && f.Subject == "Albert Einstein" {
			hasDescribed = true
		}
		if f.Relation == "is_a" && f.Subject == "Einstein" && strings.Contains(f.Object, "physicist") {
			hasIsA = true
		}
		if f.Relation == "created_by" && f.Subject == "General relativity" {
			hasCreatedBy = true
		}
	}

	if !hasDescribed {
		t.Error("missing described_as fact for Albert Einstein")
	}
	if !hasIsA {
		t.Error("missing is_a fact for Einstein")
	}
	if !hasCreatedBy {
		t.Error("missing created_by fact for General relativity")
	}
}

func TestArticleToFactsEmpty(t *testing.T) {
	facts := ArticleToFacts("Test", "")
	if facts != nil {
		t.Errorf("expected nil for empty text, got %d facts", len(facts))
	}
}

func TestBatchToPackage(t *testing.T) {
	facts := []PackageFact{
		{Subject: "water", Relation: "is_a", Object: "liquid"},
		{Subject: "water", Relation: "has", Object: "hydrogen"},
	}

	pkg := BatchToPackage(1, "science", facts)

	if pkg.Name != "wiki-science-batch-0001" {
		t.Errorf("unexpected name: %s", pkg.Name)
	}
	if pkg.Domain != "science" {
		t.Errorf("unexpected domain: %s", pkg.Domain)
	}
	if pkg.Version != "1.0.0" {
		t.Errorf("unexpected version: %s", pkg.Version)
	}
	if len(pkg.Facts) != 2 {
		t.Errorf("expected 2 facts, got %d", len(pkg.Facts))
	}
	if pkg.Author != "wikiimport" {
		t.Errorf("unexpected author: %s", pkg.Author)
	}
}

func TestStripWikitextRealArticle(t *testing.T) {
	// Realistic multi-paragraph article with mixed markup
	wikitext := `'''Water''' is a [[chemical substance]] with the [[chemical formula]]
H{{sub|2}}O. It is a transparent, tasteless, odorless,<ref name="WHO">World Health
Organization report, 2019</ref> and nearly colorless [[chemical compound]].

== Properties ==
Water is the most abundant substance on [[Earth]]'s surface.<ref>{{cite journal
|title=Water on Earth|year=2020}}</ref> It covers about 71% of the planet.

{| class="wikitable"
|-
! Property !! Value
|-
| Boiling point || 100 °C
|-
| Melting point || 0 °C
|}

=== Phase transitions ===
* Water freezes at 0 degrees [[Celsius]]
* Water boils at 100 degrees Celsius
* Water has a high [[specific heat capacity]]

<!-- TODO: add more phase info -->

Water is used for drinking, [[agriculture]], and [[industry]].
It is similar to [[heavy water|deuterium oxide]] in many ways.

[[Category:Chemical substances]]
[[Category:Water]]`

	got := StripWikitext(wikitext)

	// Should contain clean text
	mustContain := []string{
		"Water",
		"chemical substance",
		"transparent",
		"most abundant",
		"71%",
		"freezes at 0 degrees",
		"Celsius",
		"drinking",
		"deuterium oxide",
	}
	for _, want := range mustContain {
		if !strings.Contains(got, want) {
			t.Errorf("result should contain %q but doesn't.\nGot: %s", want, got)
		}
	}

	// Should NOT contain markup artifacts
	mustNotContain := []string{
		"'''",
		"[[",
		"]]",
		"{{",
		"}}",
		"<ref",
		"</ref>",
		"cite journal",
		"{|",
		"|}",
		"wikitable",
		"<!--",
		"-->",
		"Category:",
		"TODO:",
	}
	for _, bad := range mustNotContain {
		if strings.Contains(got, bad) {
			t.Errorf("result should NOT contain %q.\nGot: %s", bad, got)
		}
	}
}

func TestParsWikiDump(t *testing.T) {
	// Build a minimal MediaWiki XML dump in memory
	xmlData := `<mediawiki>
  <page>
    <title>Test Article</title>
    <ns>0</ns>
    <id>1</id>
    <revision>
      <text>'''Test Article''' is a thing. It has properties.</text>
    </revision>
  </page>
  <page>
    <title>Redirect Page</title>
    <ns>0</ns>
    <id>2</id>
    <revision>
      <text>#REDIRECT [[Test Article]]</text>
    </revision>
  </page>
  <page>
    <title>Talk:Test Article</title>
    <ns>1</ns>
    <id>3</id>
    <revision>
      <text>Discussion about the article.</text>
    </revision>
  </page>
  <page>
    <title>Another Article</title>
    <ns>0</ns>
    <id>4</id>
    <revision>
      <text>'''Another Article''' is a [[concept]] in [[science]].</text>
    </revision>
  </page>
</mediawiki>`

	reader := bytes.NewReader([]byte(xmlData))

	var articles []WikiArticle
	err := ParseWikiDump(reader, func(a WikiArticle) error {
		articles = append(articles, a)
		return nil
	})
	if err != nil {
		t.Fatalf("ParseWikiDump error: %v", err)
	}

	// Should get 2 articles (skip redirect and non-ns0)
	if len(articles) != 2 {
		t.Fatalf("expected 2 articles, got %d", len(articles))
	}

	if articles[0].Title != "Test Article" {
		t.Errorf("first article title = %q, want %q", articles[0].Title, "Test Article")
	}
	if articles[0].ID != 1 {
		t.Errorf("first article ID = %d, want 1", articles[0].ID)
	}
	// Text should be stripped of markup
	if strings.Contains(articles[0].Text, "'''") {
		t.Errorf("article text still has bold markup: %q", articles[0].Text)
	}

	if articles[1].Title != "Another Article" {
		t.Errorf("second article title = %q, want %q", articles[1].Title, "Another Article")
	}
	if strings.Contains(articles[1].Text, "[[") {
		t.Errorf("article text still has wiki links: %q", articles[1].Text)
	}
}

// TestXMLPageDecode verifies our XML struct mapping works.
func TestXMLPageDecode(t *testing.T) {
	xmlStr := `<page>
		<title>Test</title>
		<ns>0</ns>
		<id>42</id>
		<revision>
			<text>Hello world</text>
		</revision>
	</page>`

	var page xmlPage
	if err := xml.Unmarshal([]byte(xmlStr), &page); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if page.Title != "Test" {
		t.Errorf("title = %q", page.Title)
	}
	if page.NS != 0 {
		t.Errorf("ns = %d", page.NS)
	}
	if page.ID != 42 {
		t.Errorf("id = %d", page.ID)
	}
	if page.Revision.Text != "Hello world" {
		t.Errorf("text = %q", page.Revision.Text)
	}
}
