package cognitive

import (
	"encoding/xml"
	"fmt"
	"io"
	"regexp"
	"strings"
	"time"
)

// -----------------------------------------------------------------------
// Wikipedia Dump Parser — streams bzip2'd XML dumps into KnowledgePackages.
//
// Converts MediaWiki markup to plain text, extracts triples using the
// existing ExtractTriples patterns, and batches articles into packages
// that can be loaded by the PackageLoader.
//
// Designed for Simple English Wikipedia but works with any MediaWiki dump.
// -----------------------------------------------------------------------

// WikiArticle is a single parsed Wikipedia article.
type WikiArticle struct {
	Title string
	Text  string // plain text after markup stripping
	ID    int
}

// xmlPage mirrors the <page> element in a MediaWiki XML dump.
type xmlPage struct {
	Title    string      `xml:"title"`
	NS       int         `xml:"ns"`
	ID       int         `xml:"id"`
	Revision xmlRevision `xml:"revision"`
}

// xmlRevision mirrors the <revision> element.
type xmlRevision struct {
	Text string `xml:"text"`
}

// ParseWikiDump streams articles from a MediaWiki XML dump reader.
// The reader should already be decompressed (e.g. passed through compress/bzip2).
// Calls handler for each article in namespace 0, skipping redirects.
func ParseWikiDump(reader io.Reader, handler func(WikiArticle) error) error {
	decoder := xml.NewDecoder(reader)

	for {
		tok, err := decoder.Token()
		if err == io.EOF {
			break
		}
		if err != nil {
			return fmt.Errorf("xml decode: %w", err)
		}

		se, ok := tok.(xml.StartElement)
		if !ok || se.Name.Local != "page" {
			continue
		}

		var page xmlPage
		if err := decoder.DecodeElement(&page, &se); err != nil {
			continue // skip malformed pages
		}

		// Only main namespace
		if page.NS != 0 {
			continue
		}

		// Skip redirects
		text := strings.TrimSpace(page.Revision.Text)
		if strings.HasPrefix(strings.ToUpper(text), "#REDIRECT") {
			continue
		}

		plainText := StripWikitext(text)
		plainText = strings.TrimSpace(plainText)
		if plainText == "" {
			continue
		}

		article := WikiArticle{
			Title: page.Title,
			Text:  plainText,
			ID:    page.ID,
		}

		if err := handler(article); err != nil {
			return fmt.Errorf("handler error on %q: %w", page.Title, err)
		}
	}

	return nil
}

// -----------------------------------------------------------------------
// Wikitext stripping — convert MediaWiki markup to clean plain text.
// -----------------------------------------------------------------------

// Precompiled regexps for wikitext stripping.
var (
	// Comments: <!-- ... -->
	wikiCommentRe = regexp.MustCompile(`<!--[\s\S]*?-->`)

	// <ref> tags: both paired <ref>...</ref> and self-closing <ref ... />
	wikiRefPairedRe = regexp.MustCompile(`(?i)<ref[^>]*>[\s\S]*?</ref>`)
	wikiRefSelfRe   = regexp.MustCompile(`(?i)<ref[^/]*/\s*>`)

	// <nowiki>content</nowiki> — remove tags, keep content
	wikiNowikiRe = regexp.MustCompile(`(?i)<nowiki>([\s\S]*?)</nowiki>`)

	// All remaining HTML tags — remove tags, keep content
	wikiHTMLTagRe = regexp.MustCompile(`<[^>]+>`)

	// Tables: {| ... |}
	wikiTableRe = regexp.MustCompile(`\{\|[\s\S]*?\|\}`)

	// Category links: [[Category:...]]
	wikiCategoryRe = regexp.MustCompile(`(?i)\[\[Category:[^\]]*\]\]`)

	// File/Image links: [[File:...]] or [[Image:...]]
	wikiFileRe = regexp.MustCompile(`(?i)\[\[(?:File|Image):[^\]]*\]\]`)

	// Section headers: == Title == → ". Title"
	wikiSectionRe = regexp.MustCompile(`(?m)^=+\s*(.+?)\s*=+\s*$`)

	// Bold and italic
	wikiBoldItalicRe = regexp.MustCompile(`'{2,5}`)

	// Bullet and numbered lists: * item, # item, : item, ; item
	wikiListRe = regexp.MustCompile(`(?m)^[*#:;]+\s*`)

	// External links: [http://... display text] → display text
	wikiExtLinkWithTextRe = regexp.MustCompile(`\[https?://\S+\s+([^\]]+)\]`)
	// External links without display text: [http://...] → remove
	wikiExtLinkBareRe = regexp.MustCompile(`\[https?://\S+\]`)

	// Bare URLs
	wikiBareURLRe = regexp.MustCompile(`https?://\S+`)

	// Multiple spaces / blank lines
	wikiMultiSpaceRe   = regexp.MustCompile(`[ \t]{2,}`)
	wikiMultiNewlineRe = regexp.MustCompile(`\n{3,}`)

	// Magic words / behavior switches
	wikiMagicWordRe = regexp.MustCompile(`(?i)__[A-Z]+__`)
)

// StripWikitext converts MediaWiki markup to plain text.
func StripWikitext(wikitext string) string {
	s := wikitext

	// 1. Comments
	s = wikiCommentRe.ReplaceAllString(s, "")

	// 2. <nowiki> — protect content from further processing using placeholders
	var nowikiBlocks []string
	s = wikiNowikiRe.ReplaceAllStringFunc(s, func(match string) string {
		sub := wikiNowikiRe.FindStringSubmatch(match)
		if len(sub) < 2 {
			return match
		}
		idx := len(nowikiBlocks)
		nowikiBlocks = append(nowikiBlocks, sub[1])
		return fmt.Sprintf("\x00NOWIKI%d\x00", idx)
	})

	// 3. <ref> tags (must be before general HTML removal)
	s = wikiRefPairedRe.ReplaceAllString(s, "")
	s = wikiRefSelfRe.ReplaceAllString(s, "")

	// 4. Templates: {{...}} — handle nesting by counting braces
	s = stripNestedBraces(s, '{', '}')

	// 5. Tables: {| ... |} (in case nested stripping missed them)
	s = wikiTableRe.ReplaceAllString(s, "")

	// 6. Category and File/Image links
	s = wikiCategoryRe.ReplaceAllString(s, "")
	s = wikiFileRe.ReplaceAllString(s, "")

	// 7. Wikilinks: [[target|display]] → display, [[target]] → target
	s = stripWikilinks(s)

	// 8. Section headers → sentence boundary + text
	s = wikiSectionRe.ReplaceAllString(s, ". $1. ")

	// 9. Bold/italic markup
	s = wikiBoldItalicRe.ReplaceAllString(s, "")

	// 10. List items → sentence-like text
	s = wikiListRe.ReplaceAllString(s, "")

	// 11. External links
	s = wikiExtLinkWithTextRe.ReplaceAllString(s, "$1")
	s = wikiExtLinkBareRe.ReplaceAllString(s, "")
	s = wikiBareURLRe.ReplaceAllString(s, "")

	// 12. HTML tags (keep content)
	s = wikiHTMLTagRe.ReplaceAllString(s, "")

	// 13. Magic words
	s = wikiMagicWordRe.ReplaceAllString(s, "")

	// 14. Restore <nowiki> content
	for i, block := range nowikiBlocks {
		s = strings.ReplaceAll(s, fmt.Sprintf("\x00NOWIKI%d\x00", i), block)
	}

	// 15. Cleanup
	s = wikiMultiSpaceRe.ReplaceAllString(s, " ")
	s = wikiMultiNewlineRe.ReplaceAllString(s, "\n\n")

	// Clean up artifacts: ". ." patterns, leading/trailing dots
	s = strings.ReplaceAll(s, ". .", ".")
	s = strings.ReplaceAll(s, "..", ".")

	lines := strings.Split(s, "\n")
	var cleaned []string
	for _, line := range lines {
		line = strings.TrimSpace(line)
		if line != "" {
			cleaned = append(cleaned, line)
		}
	}

	return strings.Join(cleaned, "\n")
}

// stripNestedBraces removes content between matching brace pairs,
// handling arbitrary nesting depth. Used for {{ templates }}.
func stripNestedBraces(s string, open, close byte) string {
	var result []byte
	depth := 0
	i := 0
	for i < len(s) {
		if i+1 < len(s) && s[i] == open && s[i+1] == open {
			depth++
			i += 2
			continue
		}
		if i+1 < len(s) && s[i] == close && s[i+1] == close {
			if depth > 0 {
				depth--
			}
			i += 2
			continue
		}
		if depth == 0 {
			result = append(result, s[i])
		}
		i++
	}
	return string(result)
}

// stripWikilinks handles [[target|display]] → display and [[target]] → target.
// Also handles nested links by tracking bracket depth.
func stripWikilinks(s string) string {
	var result strings.Builder
	result.Grow(len(s))

	i := 0
	for i < len(s) {
		if i+1 < len(s) && s[i] == '[' && s[i+1] == '[' {
			// Find the matching ]]
			j := i + 2
			depth := 1
			for j < len(s)-1 && depth > 0 {
				if s[j] == '[' && s[j+1] == '[' {
					depth++
					j += 2
					continue
				}
				if s[j] == ']' && s[j+1] == ']' {
					depth--
					if depth == 0 {
						break
					}
					j += 2
					continue
				}
				j++
			}

			if depth == 0 {
				// Extract content between [[ and ]]
				content := s[i+2 : j]
				// Use display text (after |) if present
				if pipeIdx := strings.LastIndex(content, "|"); pipeIdx >= 0 {
					result.WriteString(content[pipeIdx+1:])
				} else {
					result.WriteString(content)
				}
				i = j + 2
			} else {
				// Unmatched — keep as-is
				result.WriteByte(s[i])
				i++
			}
		} else {
			result.WriteByte(s[i])
			i++
		}
	}

	return result.String()
}

// -----------------------------------------------------------------------
// Fact extraction from articles
// -----------------------------------------------------------------------

// ArticleToFacts extracts PackageFacts from a plain-text article.
// Uses ExtractTriples patterns. Also creates a "described_as" fact
// from the first sentence (the Wikipedia definition pattern).
func ArticleToFacts(title, plainText string) []PackageFact {
	if plainText == "" {
		return nil
	}

	var facts []PackageFact
	seen := make(map[string]bool)

	addFact := func(s, r, o string) {
		key := s + "|" + r + "|" + o
		if !seen[key] {
			seen[key] = true
			facts = append(facts, PackageFact{Subject: s, Relation: r, Object: o})
		}
	}

	// First sentence extraction: Wikipedia articles typically begin with
	// "Title is a ..." — extract as a described_as fact.
	sentences := splitSentences(plainText)
	if len(sentences) > 0 {
		first := strings.TrimSpace(sentences[0])
		if first != "" && len(first) > 10 {
			// Use the first sentence as a description
			addFact(title, "described_as", first)
		}
	}

	// Extract triples from all sentences
	for _, sent := range sentences {
		sent = strings.TrimSpace(sent)
		if sent == "" || len(sent) < 10 {
			continue
		}
		if isBoilerplate(sent) {
			continue
		}

		triples := ExtractTriples(sent)
		for _, t := range triples {
			rel := relTypeToString(t.Relation)
			addFact(t.Subject, rel, t.Object)
		}
	}

	return facts
}

// relTypeToString converts a RelType back to its string representation.
func relTypeToString(r RelType) string {
	switch r {
	case RelIsA:
		return "is_a"
	case RelLocatedIn:
		return "located_in"
	case RelPartOf:
		return "part_of"
	case RelCreatedBy:
		return "created_by"
	case RelFoundedBy:
		return "founded_by"
	case RelFoundedIn:
		return "founded_in"
	case RelHas:
		return "has"
	case RelOffers:
		return "offers"
	case RelUsedFor:
		return "used_for"
	case RelRelatedTo:
		return "related_to"
	case RelSimilarTo:
		return "similar_to"
	case RelCauses:
		return "causes"
	case RelContradicts:
		return "contradicts"
	case RelFollows:
		return "follows"
	case RelPrefers:
		return "prefers"
	case RelDislikes:
		return "dislikes"
	case RelDomain:
		return "domain"
	case RelDescribedAs:
		return "described_as"
	default:
		return "related_to"
	}
}

// BatchToPackage creates a KnowledgePackage from a batch of extracted facts.
func BatchToPackage(batchNum int, domain string, facts []PackageFact) *KnowledgePackage {
	return &KnowledgePackage{
		Name:        fmt.Sprintf("wiki-%s-batch-%04d", domain, batchNum),
		Version:     "1.0.0",
		Description: fmt.Sprintf("Wikipedia knowledge batch %d — auto-extracted from %s dump", batchNum, domain),
		Author:      "wikiimport",
		Domain:      domain,
		Facts:       facts,
	}
}

// PackageToJSON serializes a KnowledgePackage to pretty-printed JSON.
func PackageToJSON(pkg *KnowledgePackage) ([]byte, error) {
	return CreatePackage(pkg.Name, pkg.Version, pkg.Description, pkg.Domain, pkg.Facts, pkg.Vocabulary)
}

// WikiImportStats tracks progress during a wiki import.
type WikiImportStats struct {
	ArticlesProcessed int
	ArticlesSkipped   int
	FactsExtracted    int
	PackagesWritten   int
	StartTime         time.Time
}

func (s *WikiImportStats) String() string {
	elapsed := time.Since(s.StartTime)
	rate := float64(s.ArticlesProcessed) / elapsed.Seconds()
	return fmt.Sprintf("articles=%d skipped=%d facts=%d packages=%d elapsed=%s rate=%.0f/s",
		s.ArticlesProcessed, s.ArticlesSkipped, s.FactsExtracted,
		s.PackagesWritten, elapsed.Round(time.Millisecond), rate)
}
